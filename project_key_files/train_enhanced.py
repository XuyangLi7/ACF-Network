#!/usr/bin/env python3
"""
Enhanced training script for collaborative_framework_project555
- Aligns with UNIFIED_CONFIG and UniversalMultiModalDataset
- Mixed precision (AMP) + optional DataParallel
- Composite loss: CE (label smoothing, class weights) + Focal + Dice + IoU + Boundary + Aux branch
- Validation each epoch on val split (Vaihingen test IDs)
- Saves best checkpoint as best_enhanced_model_epoch_{E}.pth and last checkpoint as last_model.pth
"""

import os
import sys
import json
import math
import time
import argparse
import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from unified_config import UNIFIED_CONFIG
from enhanced_multimodal_framework import EnhancedMultimodalFramework
from universal_dataset import UniversalMultiModalDataset

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------------------
# Loss functions
# -------------------------------

def make_class_weights(weights_list, device):
    w = torch.tensor(weights_list, dtype=torch.float32, device=device)
    return w

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class weights tensor on device
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: (B,C,H,W), targets: (B,H,W) long
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        # gather per-pixel log-prob and prob of target class
        tgt = targets.unsqueeze(1)  # (B,1,H,W)
        log_pt = log_probs.gather(1, tgt).squeeze(1)  # (B,H,W)
        pt = probs.gather(1, tgt).squeeze(1)  # (B,H,W)
        # focal factor
        focal_factor = (1 - pt).clamp(min=1e-6) ** self.gamma
        loss = -focal_factor * log_pt  # (B,H,W)
        if self.alpha is not None:
            alpha_factor = self.alpha[tgt.squeeze(1)]  # (B,H,W)
            loss = loss * alpha_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # multi-class soft dice
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        # one-hot targets
        with torch.no_grad():
            target_onehot = F.one_hot(targets.clamp(min=0, max=num_classes-1), num_classes=num_classes)  # (B,H,W,C)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # (B,C,H,W)
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        union = probs.sum(dims) + target_onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()
        return loss

class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        with torch.no_grad():
            target_onehot = F.one_hot(targets.clamp(min=0, max=num_classes-1), num_classes=num_classes)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        intersection = (probs * target_onehot).sum(dims)
        total = (probs + target_onehot).sum(dims)
        union = total - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - iou.mean()
        return loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernels
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))

    def edges(self, x: torch.Tensor):
        # x: (B,C,H,W)
        # depthwise conv per channel
        B, C, H, W = x.shape
        # match device and dtype (important for AMP half precision)
        kx = self.kx.to(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)
        ky = self.ky.to(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)
        gx = F.conv2d(x, kx, padding=1, groups=C)
        gy = F.conv2d(x, ky, padding=1, groups=C)
        g = torch.sqrt(gx * gx + gy * gy + 1e-8)
        # normalize per-sample
        g = g / (g.amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return g

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # compute on class probabilities vs one-hot labels
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        with torch.no_grad():
            target_onehot = F.one_hot(targets.clamp(min=0, max=num_classes-1), num_classes=num_classes)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        g_pred = self.edges(probs)
        g_true = self.edges(target_onehot)
        return F.l1_loss(g_pred, g_true)

# -------------------------------
# Metrics
# -------------------------------

def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    mask = (gt >= 0) & (gt < num_classes)
    hist = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    cm += hist
    return cm

def compute_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    # per-class stats
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    tn = cm.sum() - (tp + fp + fn)

    # per-class IoU/precision/recall/f1
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # overall accuracy
    total = cm.sum().astype(np.float64) + 1e-8
    oa = tp.sum() / total

    # weighted F1 (by support)
    support = cm.sum(axis=1).astype(np.float64)
    f1_weighted = (f1 * support).sum() / (support.sum() + 1e-8)

    # Cohen's Kappa
    row_marginals = cm.sum(axis=1).astype(np.float64)
    col_marginals = cm.sum(axis=0).astype(np.float64)
    pe = (row_marginals * col_marginals).sum() / (total * total)
    po = oa
    kappa = (po - pe) / (1.0 - pe + 1e-8)

    # top-5 classes mIoU (Vaihingen standard: first 5)
    miou_top5 = iou[:5].mean()

    return {
        'oa': float(oa),
        'miou': float(miou_top5),
        'aa': float(precision.mean()),
        'f1': float(f1.mean()),  # macro F1
        'f1_weighted': float(f1_weighted),
        'kappa': float(kappa),
        'per_class_iou': [float(x) for x in iou.tolist()],
    }

# -------------------------------
# Trainer
# -------------------------------

class Trainer:
    def __init__(self, args):
        self.args = args
        
        # DDP初始化
        self.is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
        if self.is_ddp:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.global_rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            logger.info(f"DDP initialized: rank {self.global_rank}/{self.world_size}, device {self.device}")
        else:
            self.local_rank = 0
            self.global_rank = 0
            self.world_size = 1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
        
        self.cfg = UNIFIED_CONFIG

        # Seed
        self._set_seed(args.seed)

        # Model
        self.model = EnhancedMultimodalFramework(
            rgb_channels=self.cfg['model']['rgb_channels'],
            dsm_channels=self.cfg['model']['dsm_channels'],
            num_classes=self.cfg['model']['num_classes'],
            embed_dim=self.cfg['model']['embed_dim'],
            enable_remote_sensing_innovations=self.cfg['model']['enable_remote_sensing_innovations'],
            pretrained=self.cfg['model'].get('pretrained', False),
            use_multi_scale_aggregator=self.cfg['model'].get('use_multi_scale_aggregator', False),
            use_simple_mode=self.cfg['model'].get('use_simple_mode', False)
        ).to(self.device)
        
        # 使用DDP或DataParallel
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            logger.info(f"Using DistributedDataParallel with {self.world_size} GPUs")
        elif torch.cuda.device_count() > 1 and not args.no_dp:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")

        # Datasets
        win_h, win_w = self.cfg['data']['window_size']
        train_stride = self.cfg['data']['train_stride'] if args.train_stride is None else args.train_stride
        # Use DATA_CONFIG['eval_stride'] for training-time validation; EVAL_CONFIG['stride'] is for final evaluation
        eval_stride = self.cfg['data']['eval_stride'] if args.val_stride is None else args.val_stride

        self.train_set = UniversalMultiModalDataset(
            data_path=args.data_path, dataset_name=args.dataset_name,
            split='train', window_size=(win_h, win_w), stride=train_stride, cache=True
        )
        # Use 'val' to map to test IDs but without augmentation in dataset
        self.val_set = UniversalMultiModalDataset(
            data_path=args.data_path, dataset_name=args.dataset_name,
            split='val', window_size=(win_h, win_w), stride=eval_stride, cache=False
        )

        # DDP Sampler
        if self.is_ddp:
            self.train_sampler = DistributedSampler(self.train_set, num_replicas=self.world_size, rank=self.global_rank, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_set, num_replicas=self.world_size, rank=self.global_rank, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
        
        self.train_loader = DataLoader(
            self.train_set, batch_size=args.batch_size, 
            sampler=self.train_sampler, shuffle=(self.train_sampler is None),
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=args.val_batch_size,
            sampler=self.val_sampler, shuffle=False,
            num_workers=max(1, args.num_workers // 2), pin_memory=True, drop_last=False
        )

        # Optimizer & Scheduler
        tcfg = self.cfg['train']
        params = [p for p in self.model.parameters() if p.requires_grad]
        if tcfg['optimizer'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                params, lr=tcfg['initial_lr'], momentum=tcfg['momentum'], weight_decay=tcfg['weight_decay']
            )
        else:
            self.optimizer = torch.optim.AdamW(params, lr=tcfg['initial_lr'], weight_decay=tcfg['weight_decay'])
        # Scheduler
        if tcfg.get('scheduler', 'MultiStepLR') == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=tcfg.get('cosine_t_max', tcfg['epochs']),
                eta_min=tcfg.get('cosine_eta_min', 0.0001)
            )
            logger.info(f"Using CosineAnnealingLR scheduler")
        else:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=tcfg['scheduler_milestones'], gamma=tcfg['scheduler_gamma']
            )
            logger.info(f"Using MultiStepLR scheduler")
        
        # Warmup config
        self.use_warmup = tcfg.get('use_warmup', False)
        self.warmup_epochs = tcfg.get('warmup_epochs', 3)
        self.warmup_start_lr = tcfg.get('warmup_start_lr', tcfg['initial_lr'] * 0.1)
        self.warmup_target_lr = tcfg['initial_lr']

        # Losses
        num_classes = self.cfg['model']['num_classes']
        class_weights = make_class_weights(tcfg['class_weights'], self.device)
        label_smoothing = tcfg.get('label_smoothing', 0.0) if tcfg.get('use_label_smoothing', False) else 0.0
        try:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        except TypeError:
            # Fallback without label_smoothing
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(gamma=tcfg['focal_loss_gamma'], alpha=class_weights)
        self.dice_loss = DiceLoss(smooth=1.0)
        self.iou_loss = IoULoss(smooth=1.0)
        self.boundary_loss = BoundaryLoss()
        self.loss_weights = tcfg['loss_weights']
        self.aux_weight = tcfg['aux_loss_weight']
        self.max_grad_norm = tcfg['max_grad_norm']

        # AMP
        self.use_amp = args.amp and torch.cuda.is_available()
        # Use torch.amp to avoid deprecation warnings
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Checkpoints
        os.makedirs(args.output_dir, exist_ok=True)
        self.best_metric = -1.0
        self.start_epoch = 1
        if args.resume and os.path.isfile(args.resume):
            self._load_checkpoint(args.resume)

        # History
        self.history_path = os.path.join(args.output_dir, 'training_history.json')
        self.history = []

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def _step_losses(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses = {}
        loss_ce = self.ce_loss(logits, targets)
        losses['ce'] = float(loss_ce.detach().item())
        loss_focal = self.focal_loss(logits, targets)
        losses['focal'] = float(loss_focal.detach().item())
        loss_dice = self.dice_loss(logits, targets)
        losses['dice'] = float(loss_dice.detach().item())
        loss_iou = self.iou_loss(logits, targets)
        losses['iou'] = float(loss_iou.detach().item())
        loss_boundary = self.boundary_loss(logits, targets)
        losses['boundary'] = float(loss_boundary.detach().item())

        total = (
            self.loss_weights['ce'] * loss_ce +
            self.loss_weights['focal'] * loss_focal +
            self.loss_weights['dice'] * loss_dice +
            self.loss_weights['iou'] * loss_iou +
            self.loss_weights['boundary'] * loss_boundary
        )
        return total, losses

    def train(self):
        args = self.args
        epochs = args.epochs
        logger.info(f"Start training for {epochs} epochs. Train iters: {len(self.train_loader)}, Val iters: {len(self.val_loader)}")

        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            
            # DDP: 设置epoch用于sampler的shuffle
            if self.is_ddp and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            # Warmup learning rate adjustment
            if self.use_warmup and epoch <= self.warmup_epochs:
                warmup_lr = self.warmup_start_lr + (self.warmup_target_lr - self.warmup_start_lr) * (epoch - 1) / max(1, self.warmup_epochs - 1)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.info(f"Warmup epoch {epoch}/{self.warmup_epochs}, LR: {warmup_lr:.6f}")
            
            self.model.train()
            # reset CUDA peak memory per epoch (if available)
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats(self.device)
                except Exception:
                    pass

            running = {'total': 0.0, 'ce': 0.0, 'focal': 0.0, 'dice': 0.0, 'iou': 0.0, 'boundary': 0.0}

            for it, batch in enumerate(self.train_loader, 1):
                if it == 1:
                    logger.info(f"Epoch {epoch}: Loading first batch...")
                inputs, targets = batch
                if it == 1:
                    logger.info(f"Epoch {epoch}: First batch loaded, transferring to GPU...")
                rgb = inputs['rgb'].to(self.device, non_blocking=True)
                dsm = inputs['dsm'].to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if it == 1:
                    logger.info(f"Epoch {epoch}: Data on GPU, starting forward pass...")

                self.optimizer.zero_grad(set_to_none=True)
                # autocast via torch.amp
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    outputs = self.model(rgb, dsm)
                if it == 1:
                    logger.info(f"Epoch {epoch}: Forward pass done, computing loss...")
                if isinstance(outputs, tuple):
                    main_logits, aux_logits = outputs
                else:
                    main_logits, aux_logits = outputs, None
                main_loss, main_parts = self._step_losses(main_logits, targets)
                if aux_logits is not None:
                    aux_loss, _ = self._step_losses(aux_logits, targets)
                    total_loss = main_loss + self.aux_weight * aux_loss
                else:
                    total_loss = main_loss

                if it == 1:
                    logger.info(f"Epoch {epoch}: Loss computed, starting backward...")
                self.scaler.scale(total_loss).backward()
                if it == 1:
                    logger.info(f"Epoch {epoch}: Backward done, clipping gradients...")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if it == 1:
                    logger.info(f"Epoch {epoch}: First iteration complete!")

                running['total'] += float(total_loss.detach().item())
                # accumulate unweighted component losses from main branch for reporting
                running['ce'] += main_parts['ce']
                running['focal'] += main_parts['focal']
                running['dice'] += main_parts['dice']
                running['iou'] += main_parts['iou']
                running['boundary'] += main_parts['boundary']
                # 每20个batch打印一次，或每5%打印一次（取较小值）
                print_interval = min(20, max(1, len(self.train_loader)//20))
                if it % print_interval == 0:
                    avg_loss = running['total']/it
                    logger.info(f"Epoch {epoch} [{it}/{len(self.train_loader)}] loss={avg_loss:.4f} "
                               f"(ce={running['ce']/it:.4f}, dice={running['dice']/it:.4f}, "
                               f"iou={running['iou']/it:.4f})")

            # Scheduler step
            if not self.use_warmup or epoch > self.warmup_epochs:
                self.scheduler.step()

            # Validation
            val_metrics = self.validate(max_batches=args.max_val_batches)

            # Save history
            iters = max(1, len(self.train_loader))
            train_avg_total = running['total']/iters
            rec = {
                'epoch': epoch,
                'train_loss': train_avg_total,
                'train_loss_ce': running['ce']/iters,
                'train_loss_focal': running['focal']/iters,
                'train_loss_dice': running['dice']/iters,
                'train_loss_iou': running['iou']/iters,
                'train_loss_boundary': running['boundary']/iters,
                'val_oa': val_metrics['oa'],
                'val_miou': val_metrics['miou'],
                'val_aa': val_metrics['aa'],
                'val_f1': val_metrics['f1'],
                'val_f1_weighted': val_metrics.get('f1_weighted', None),
                'val_kappa': val_metrics.get('kappa', None),
                'val_loss': val_metrics.get('val_loss', None),
                'lr': self.optimizer.param_groups[0]['lr'],
                'time_sec': time.time() - t0,
            }
            self.history.append(rec)
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f, indent=2)

            # Save checkpoints
            is_best = val_metrics['miou'] > self.best_metric
            if is_best:
                self.best_metric = val_metrics['miou']
                self._save_checkpoint(best=True, epoch=epoch)
            self._save_checkpoint(best=False, epoch=epoch)

            # Detailed per-epoch logging
            logger.info(f"Epoch {epoch} summary:")
            logger.info(
                f"  训练损失: {train_avg_total:.4f} (CE: {rec['train_loss_ce']:.4f}, "
                f"Focal: {rec['train_loss_focal']:.4f}, Dice: {rec['train_loss_dice']:.4f}, "
                f"IoU: {rec['train_loss_iou']:.4f}, Boundary: {rec['train_loss_boundary']:.4f})"
            )
            if val_metrics.get('val_loss', None) is not None:
                logger.info(f"  验证损失: {val_metrics['val_loss']:.4f}")
            logger.info("  验证指标:")
            logger.info(f"    OA: {val_metrics['oa']:.4f}")
            logger.info(f"    mIoU (前5类): {val_metrics['miou']:.4f}")
            logger.info(f"    F1-macro: {val_metrics['f1']:.4f}")
            if 'f1_weighted' in val_metrics:
                logger.info(f"    F1-weighted: {val_metrics['f1_weighted']:.4f}")
            if 'kappa' in val_metrics:
                logger.info(f"    Kappa: {val_metrics['kappa']:.4f}")
            if 'per_class_iou' in val_metrics:
                cls_iou_str = [f"{x:.3f}" for x in val_metrics['per_class_iou']]
                logger.info(f"    各类别IoU: {cls_iou_str}")
            # Time and GPU memory
            elapsed = time.time() - t0
            if torch.cuda.is_available():
                try:
                    cur = torch.cuda.memory_allocated(self.device) / (1024**3)
                    peak = torch.cuda.max_memory_allocated(self.device) / (1024**3)
                    logger.info(f"  时间: {elapsed:.2f}s")
                    logger.info(f"  GPU内存: {cur:.2f}GB (峰值: {peak:.2f}GB)")
                except Exception:
                    logger.info(f"  时间: {elapsed:.2f}s")
            else:
                logger.info(f"  时间: {elapsed:.2f}s")
            logger.info(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"  Best mIoU: {self.best_metric:.4f}")

        logger.info("Training finished.")

    @torch.no_grad()
    def validate(self, max_batches: int = 256) -> Dict[str, float]:
        self.model.eval()
        num_classes = self.cfg['model']['num_classes']
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        total_loss = 0.0
        count = 0
        for it, batch in enumerate(self.val_loader, 1):
            inputs, targets = batch
            rgb = inputs['rgb'].to(self.device, non_blocking=True)
            dsm = inputs['dsm'].to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                outputs = self.model(rgb, dsm)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                pred = logits.argmax(dim=1)
                # compute validation total loss
                loss, _ = self._step_losses(logits, targets)
                total_loss += float(loss.detach().item())

            pred_np = pred.detach().cpu().numpy()
            tgt_np = targets.detach().cpu().numpy()
            for b in range(pred_np.shape[0]):
                cm += compute_confusion_matrix(pred_np[b], tgt_np[b], num_classes)

            count += 1
            if max_batches > 0 and it >= max_batches:
                break

        metrics = compute_metrics_from_cm(cm)
        if count > 0:
            metrics['val_loss'] = total_loss / count
        return metrics

    def _save_checkpoint(self, best: bool, epoch: int):
        # DDP: 只在rank 0保存
        if self.is_ddp and self.global_rank != 0:
            return
        
        # Support DP/DDP save
        if isinstance(self.model, (nn.DataParallel, DDP)):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
            
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'history': self.history,
        }
        last_path = os.path.join(self.args.output_dir, 'last_model.pth')
        torch.save(ckpt, last_path)
        if best:
            best_path = os.path.join(self.args.output_dir, f'best_enhanced_model_epoch_{epoch}.pth')
            torch.save(ckpt['model_state_dict'], best_path)
            logger.info(f"Saved BEST model to: {best_path}")

    def _load_checkpoint(self, path: str):
        logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get('model_state_dict', ckpt)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(state_dict, strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.best_metric = ckpt.get('best_metric', -1.0)
        self.start_epoch = ckpt.get('epoch', 1) + 1


def parse_args():
    parser = argparse.ArgumentParser(description='Train Enhanced Multimodal Framework (555)')
    parser.add_argument('--data_path', type=str, default=UNIFIED_CONFIG['paths']['data_path'])
    parser.add_argument('--dataset_name', type=str, default='vaihingen')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')

    parser.add_argument('--epochs', type=int, default=UNIFIED_CONFIG['train']['epochs'])
    parser.add_argument('--batch_size', type=int, default=UNIFIED_CONFIG['train']['batch_size'])
    parser.add_argument('--num_workers', type=int, default=UNIFIED_CONFIG['train']['num_workers'])

    parser.add_argument('--train_stride', type=int, default=None, help='override train stride')
    parser.add_argument('--val_stride', type=int, default=None, help='override val stride')
    parser.add_argument('--val_batch_size', type=int, default=4)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true', help='use mixed precision (AMP)')
    parser.add_argument('--no-dp', dest='no_dp', action='store_true', help='disable DataParallel even if multiple GPUs')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume')
    parser.add_argument('--max_val_batches', type=int, default=256, help='limit validation batches per epoch for speed')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'train_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    trainer = Trainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
