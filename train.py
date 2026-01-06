#!/usr/bin/env python3
"""
ACF Network Training Script - 极致精度版本
目标: mIoU > 85%

核心优化:
- 使用ACF Network (5个创新模块)
- embed_dim=768 (匹配ViT-Base)
- 窗口512×512 (更大上下文)
- 3层CMA (深度跨模态交互)
- 预训练权重 (ImageNet21k)
- 极致类别平衡
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

# 导入配置和模块
from acf.config import UNIFIED_CONFIG
from acf.network import create_acf_model
from acf.training_hooks import TrainingInterpreter, GradientAttributionTracker
from universal_dataset import UniversalMultiModalDataset

# 可视化模块（可选）
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("Matplotlib not available, visualization disabled")

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

        # Model - 使用ACF Network
        logger.info("创建ACF Network模型...")
        logger.info(f"  embed_dim: {self.cfg['model']['embed_dim']}")
        logger.info(f"  num_classes: {self.cfg['model']['num_classes']}")
        logger.info(f"  预训练: {self.cfg['model'].get('pretrained', False)}")
        
        self.model = create_acf_model(
            dataset=args.dataset_name,
            num_classes=self.cfg['model']['num_classes'],
            embed_dim=self.cfg['model']['embed_dim'],
            num_heads=12,  # 768 / 64 = 12
            patch_size=16,  # 标准ViT配置
            num_cma_layers=3,  # 3层CMA深度交互
            use_consistency_loss=True,  # ⭐ 启用一致性损失
            consistency_loss_weight=0.01  # ⭐ 使用小权重以保持训练稳定
        ).to(self.device)
        
        logger.info(f"ACF Network创建成功!")
        
        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        
        # 使用DDP或DataParallel
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            logger.info(f"Using DistributedDataParallel with {self.world_size} GPUs")
        elif torch.cuda.device_count() > 1 and not args.no_dp:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        else:
            logger.info(f"Using single GPU: {self.device}")

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
            # 不使用ignore_index - 与参考项目对齐
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
        
        # ==========================================
        # Interpretability System (训练可解释性系统)
        # ==========================================
        if args.enable_interpretability:
            logger.info("初始化训练可解释性系统...")
            
            # 获取实际模型（处理DDP/DataParallel包装）
            actual_model = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model
            
            # 定义要监控的模块
            target_modules = [
                'scaf',  # SCAF模块
                'multi_granularity_fusion',  # MGCF模块
                'modality_balancing',  # AMB模块
                'cross_modal_attention',  # CMA模块
            ]
            
            # 创建interpretability目录
            interp_dir = os.path.join(args.output_dir, 'interpretability')
            
            # 初始化TrainingInterpreter
            self.interpreter = TrainingInterpreter(
                model=actual_model,
                target_modules=target_modules,
                save_dir=os.path.join(interp_dir, 'module_analysis'),
                log_interval=100,  # 每100步记录一次
                vis_interval=5000  # 每5000步生成可视化
            )
            
            # 初始化GradientAttributionTracker
            self.attribution_tracker = GradientAttributionTracker(
                save_dir=os.path.join(interp_dir, 'attribution')
            )
            
            logger.info(f"✓ 训练可解释性系统已启用")
            logger.info(f"  监控模块: {target_modules}")
            logger.info(f"  保存目录: {interp_dir}")
        else:
            self.interpreter = None
            self.attribution_tracker = None
            logger.info("训练可解释性系统未启用 (使用 --enable_interpretability 启用)")

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
                    # ACF Network需要字典输入
                    model_output = self.model({'rgb': rgb, 'dsm': dsm})
                if it == 1:
                    logger.info(f"Epoch {epoch}: Forward pass done, computing loss...")
                
                # 处理模型输出（可能包含一致性损失）
                if isinstance(model_output, tuple) and len(model_output) == 2:
                    # 检查是否是 (output, aux_losses) 格式
                    if isinstance(model_output[1], dict):
                        # 新格式：(output, {'consistency_loss': ...})
                        outputs = model_output[0]
                        aux_losses_dict = model_output[1]
                        consistency_loss = aux_losses_dict.get('consistency_loss', 0.0)
                    else:
                        # 旧格式：(main_logits, aux_logits)
                        outputs = model_output
                        consistency_loss = 0.0
                else:
                    outputs = model_output
                    consistency_loss = 0.0
                
                # 处理分割输出
                if isinstance(outputs, tuple):
                    main_logits, aux_logits = outputs
                else:
                    main_logits, aux_logits = outputs, None
                
                # 计算分割损失
                main_loss, main_parts = self._step_losses(main_logits, targets)
                if aux_logits is not None:
                    aux_loss, _ = self._step_losses(aux_logits, targets)
                    seg_loss = main_loss + self.aux_weight * aux_loss
                else:
                    seg_loss = main_loss
                
                # 总损失 = 分割损失 + 一致性损失
                total_loss = seg_loss + consistency_loss

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
                # 记录一致性损失
                if isinstance(consistency_loss, torch.Tensor):
                    running['consistency'] = running.get('consistency', 0.0) + float(consistency_loss.detach().item())
                
                # ==========================================
                # Interpretability: 记录训练统计
                # ==========================================
                if self.interpreter is not None:
                    self.interpreter.step(total_loss, epoch, it)
                
                # ==========================================
                # Interpretability: 梯度归因分析 (每1000步)
                # ==========================================
                if self.attribution_tracker is not None and it % 1000 == 0:
                    try:
                        # 准备输入字典
                        inputs_dict = {'rgb': rgb, 'dsm': dsm}
                        self.attribution_tracker.compute_attribution(
                            model=self.model,
                            inputs=inputs_dict,
                            targets=targets,
                            num_classes=self.cfg['model']['num_classes']
                        )
                    except Exception as e:
                        logger.warning(f"Attribution tracking failed: {e}")
                
                # 每20个batch打印一次，或每5%打印一次（取较小值）
                print_interval = min(20, max(1, len(self.train_loader)//20))
                if it % print_interval == 0:
                    avg_loss = running['total']/it
                    cons_loss_str = f", cons={running.get('consistency', 0.0)/it:.6f}" if 'consistency' in running else ""
                    logger.info(f"Epoch {epoch} [{it}/{len(self.train_loader)}] loss={avg_loss:.4f} "
                               f"(ce={running['ce']/it:.4f}, dice={running['dice']/it:.4f}, "
                               f"iou={running['iou']/it:.4f}{cons_loss_str})")

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
            
            # 可视化（每10个epoch或最佳epoch）
            if is_best or (epoch % 10 == 0):
                vis_dir = os.path.join(self.args.output_dir, 'visualizations')
                self._visualize_predictions(epoch, vis_dir)

        # ==========================================
        # 训练结束：保存Interpretability数据
        # ==========================================
        if self.interpreter is not None:
            logger.info("\n" + "="*80)
            logger.info("保存训练可解释性数据...")
            logger.info("="*80)
            
            # 保存模块统计数据
            self.interpreter.save_statistics()
            
            # 分析模块有效性
            try:
                effectiveness = self.interpreter.analyze_module_effectiveness()
                effectiveness_path = os.path.join(
                    self.args.output_dir, 
                    'interpretability', 
                    'module_effectiveness.json'
                )
                with open(effectiveness_path, 'w') as f:
                    json.dump(effectiveness, f, indent=2)
                logger.info(f"✓ 模块有效性分析已保存: {effectiveness_path}")
                
                # 打印模块有效性
                logger.info("\n模块有效性排名:")
                for module_name, score in sorted(effectiveness.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {module_name}: {score:.4f}")
            except Exception as e:
                logger.warning(f"模块有效性分析失败: {e}")
            
            # 保存梯度归因数据
            if self.attribution_tracker is not None:
                try:
                    attr_path = os.path.join(
                        self.args.output_dir,
                        'interpretability',
                        'attribution',
                        f'gradient_attributions_final.json'
                    )
                    with open(attr_path, 'w') as f:
                        # Convert defaultdict to regular dict
                        attr_dict = {k: v for k, v in self.attribution_tracker.attributions.items()}
                        json.dump(attr_dict, f, indent=2)
                    logger.info(f"✓ 梯度归因数据已保存: {attr_path}")
                except Exception as e:
                    logger.warning(f"梯度归因数据保存失败: {e}")
            
            # 移除hooks
            self.interpreter.remove_hooks()
            logger.info("✓ 训练可解释性系统已清理")
            logger.info("="*80 + "\n")
        
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
                # ACF Network需要字典输入
                outputs = self.model({'rgb': rgb, 'dsm': dsm})
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
    
    def _visualize_predictions(self, epoch: int, save_dir: str):
        """简洁的预测可视化"""
        if not VISUALIZATION_AVAILABLE:
            return
        
        try:
            # 获取一个验证batch
            val_iter = iter(self.val_loader)
            inputs, labels = next(val_iter)
            
            rgb = inputs['rgb'][:1].to(self.device)  # 只取第一个样本
            dsm = inputs['dsm'][:1].to(self.device)
            label = labels[:1].to(self.device)
            
            # 预测
            self.model.eval()
            with torch.no_grad():
                output = self.model({'rgb': rgb, 'dsm': dsm})
                pred = output.argmax(dim=1)
            self.model.train()
            
            # 转换为numpy
            rgb_np = rgb[0].cpu().numpy().transpose(1, 2, 0)
            rgb_np = (rgb_np - rgb_np.min()) / (rgb_np.max() - rgb_np.min() + 1e-8)
            
            dsm_np = dsm[0, 0].cpu().numpy()
            dsm_np = (dsm_np - dsm_np.min()) / (dsm_np.max() - dsm_np.min() + 1e-8)
            
            pred_np = pred[0].cpu().numpy()
            label_np = label[0].cpu().numpy()
            
            # 创建颜色映射
            colors = [
                [128, 128, 128],  # 0: Impervious (灰色)
                [128, 0, 0],      # 1: Building (深红)
                [0, 255, 0],      # 2: Low veg (绿色)
                [0, 128, 0],      # 3: Tree (深绿)
                [255, 255, 0],    # 4: Car (黄色)
                [255, 0, 0],      # 5: Clutter (红色)
            ]
            cmap = ListedColormap(np.array(colors) / 255.0)
            
            # 绘图
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            axes[0, 0].imshow(rgb_np)
            axes[0, 0].set_title('RGB Input')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(dsm_np, cmap='terrain')
            axes[0, 1].set_title('DSM Input')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(label_np, cmap=cmap, vmin=0, vmax=5)
            axes[1, 0].set_title('Ground Truth')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(pred_np, cmap=cmap, vmin=0, vmax=5)
            axes[1, 1].set_title(f'Prediction (Epoch {epoch})')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # 保存
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'prediction_epoch_{epoch:03d}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  ✓ Visualization saved: {save_path}")
            
        except Exception as e:
            logger.warning(f"Visualization failed: {e}")
    

def parse_args():
    parser = argparse.ArgumentParser(description='Train ACF Network - 目标mIoU>85%')
    parser.add_argument('--data_path', type=str, default=UNIFIED_CONFIG['paths']['data_path'])
    parser.add_argument('--dataset', type=str, default='vaihingen', dest='dataset_name')
    parser.add_argument('--output_dir', type=str, default='./outputs/vaihingen_ultimate')

    parser.add_argument('--epochs', type=int, default=UNIFIED_CONFIG['train']['epochs'])
    parser.add_argument('--batch_size', type=int, default=UNIFIED_CONFIG['train']['batch_size'])
    parser.add_argument('--num_workers', type=int, default=UNIFIED_CONFIG['train']['num_workers'])

    parser.add_argument('--train_stride', type=int, default=None, help='override train stride')
    parser.add_argument('--val_stride', type=int, default=None, help='override val stride')
    parser.add_argument('--val_batch_size', type=int, default=2)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true', help='use mixed precision (AMP)')
    parser.add_argument('--no-dp', dest='no_dp', action='store_true', help='disable DataParallel even if multiple GPUs')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume')
    parser.add_argument('--max_val_batches', type=int, default=256, help='limit validation batches per epoch for speed')
    
    # 保留参数兼容性，但不使用
    parser.add_argument('--enable_visualization', type=str, default='false', help='deprecated, kept for compatibility')
    
    # Interpretability data collection
    parser.add_argument('--enable_interpretability', action='store_true', help='Enable training interpretability data collection')

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
