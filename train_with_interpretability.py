#!/usr/bin/env python3
"""
Training script with integrated interpretability analysis

This script trains the ACF network while simultaneously:
1. Monitoring module effectiveness through gradient flow
2. Extracting features for t-SNE visualization
3. Computing gradient attribution for multi-modal inputs
4. Generating real-time interpretability visualizations

The interpretability analysis happens DURING training, not just at evaluation time,
providing authentic insights into module contributions.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add ACF module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'acf'))

from acf import create_acf_model
from acf.dataset import create_universal_dataloader
from acf.training_hooks import (
    TrainingInterpreter,
    FeatureExtractor,
    GradientAttributionTracker
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ACF with Interpretability')
    
    # Basic training arguments
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['vaihingen', 'augsburg', 'muufl', 'trento'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Interpretability arguments
    parser.add_argument('--enable_interpretability', action='store_true',
                        help='Enable training-time interpretability analysis')
    parser.add_argument('--target_modules', type=str, nargs='+',
                        default=['cma', 'amb', 'scaf', 'mgcf', 'staf'],
                        help='Modules to monitor')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Steps between logging module statistics')
    parser.add_argument('--vis_interval', type=int, default=1000,
                        help='Steps between generating visualizations')
    parser.add_argument('--tsne_interval', type=int, default=5,
                        help='Epochs between t-SNE visualization')
    parser.add_argument('--attribution_interval', type=int, default=10,
                        help='Epochs between gradient attribution analysis')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    
    return parser.parse_args()


class InterpretableTrainer:
    """
    Trainer with integrated interpretability analysis
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        self.output_dir = args.output_dir
        self.interp_dir = os.path.join(args.output_dir, 'interpretability')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.interp_dir, exist_ok=True)
        
        # Model
        print("Creating model...")
        self.model = create_acf_model(
            dataset=args.dataset,
            num_classes=6,  # Adjust based on dataset
            pretrained=True
        ).to(self.device)
        
        # Data loaders
        print("Creating data loaders...")
        self.train_loader = create_universal_dataloader(
            data_path=os.path.join(args.data_root, args.dataset.capitalize()),
            dataset_name=args.dataset,
            split='train',
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            window_size=(256, 256),
            stride=128
        )
        
        self.val_loader = create_universal_dataloader(
            data_path=os.path.join(args.data_root, args.dataset.capitalize()),
            dataset_name=args.dataset,
            split='val',
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            window_size=(256, 256),
            stride=256
        )
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=1e-6
        )
        
        # Mixed precision
        self.use_amp = args.amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Interpretability components
        if args.enable_interpretability:
            print("\nInitializing interpretability system...")
            
            # Training interpreter for module effectiveness
            self.interpreter = TrainingInterpreter(
                model=self.model,
                target_modules=args.target_modules,
                save_dir=os.path.join(self.interp_dir, 'module_analysis'),
                log_interval=args.log_interval,
                vis_interval=args.vis_interval
            )
            
            # Feature extractor for t-SNE
            self.feature_extractor = FeatureExtractor(
                model=self.model,
                layer_name='scaf'  # Extract from SCAF module
            )
            self.feature_extractor.register()
            
            # Gradient attribution tracker
            self.attribution_tracker = GradientAttributionTracker(
                save_dir=os.path.join(self.interp_dir, 'attribution')
            )
            
            print("✓ Interpretability system initialized")
        else:
            self.interpreter = None
            self.feature_extractor = None
            self.attribution_tracker = None
            
        self.best_miou = 0.0
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.epochs}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(**inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Interpretability analysis
            if self.interpreter is not None:
                self.interpreter.step(loss, epoch, batch_idx)
                
            # Feature extraction for t-SNE
            if self.feature_extractor is not None:
                self.feature_extractor.add_labels(targets)
                
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
        
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For mIoU calculation
        num_classes = 6  # Adjust based on dataset
        intersection = torch.zeros(num_classes)
        union = torch.zeros(num_classes)
        
        for inputs, targets in tqdm(self.val_loader, desc='Validating'):
            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(**inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            loss = self.criterion(outputs, targets)
            total_loss += loss.item()
            
            # Predictions
            preds = outputs.argmax(dim=1)
            
            # Accuracy
            correct += (preds == targets).sum().item()
            total += targets.numel()
            
            # IoU per class
            for cls in range(num_classes):
                pred_mask = (preds == cls)
                target_mask = (targets == cls)
                intersection[cls] += (pred_mask & target_mask).sum().item()
                union[cls] += (pred_mask | target_mask).sum().item()
                
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # Compute mIoU
        iou_per_class = intersection / (union + 1e-8)
        miou = iou_per_class.mean().item()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'miou': miou * 100,
            'iou_per_class': iou_per_class.tolist()
        }
        
    def generate_tsne_visualization(self, epoch: int):
        """Generate t-SNE visualization from collected features"""
        if self.feature_extractor is None:
            return
            
        print(f"\nGenerating t-SNE visualization for epoch {epoch}...")
        
        features, labels = self.feature_extractor.get_features_and_labels()
        
        if len(features) == 0:
            print("No features collected, skipping t-SNE")
            return
            
        # Subsample if too many
        max_samples = 5000
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
            
        # Apply t-SNE
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        class_names = ['Impervious', 'Building', 'Low veg', 'Tree', 'Car', 'Clutter']
        colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
        
        for i, class_name in enumerate(class_names):
            mask = labels == i
            if mask.sum() > 0:
                plt.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=[colors[i]],
                    label=class_name,
                    alpha=0.6,
                    s=20
                )
                
        plt.legend(loc='best', fontsize=10)
        plt.title(f't-SNE Feature Clustering - Epoch {epoch}', fontsize=14)
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.interp_dir, f'tsne_epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved t-SNE visualization to: {save_path}")
        
        # Clear features for next epoch
        self.feature_extractor.clear()
        
    def compute_gradient_attribution(self, epoch: int):
        """Compute gradient attribution for multi-modal inputs"""
        if self.attribution_tracker is None:
            return
            
        print(f"\nComputing gradient attribution for epoch {epoch}...")
        
        # Sample a few batches
        num_batches = 10
        for batch_idx, (inputs, targets) in enumerate(self.val_loader):
            if batch_idx >= num_batches:
                break
                
            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            targets = targets.to(self.device)
            
            # Compute attribution
            self.attribution_tracker.compute_attribution(
                model=self.model,
                inputs=inputs,
                targets=targets,
                num_classes=6
            )
            
        # Save results
        self.attribution_tracker.save_attributions(
            filename=f'gradient_attributions_epoch_{epoch:03d}.json'
        )
        
    def train(self):
        """Main training loop"""
        print("\n" + "="*80)
        print("Starting Training with Interpretability Analysis")
        print("="*80)
        
        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log results
            print(f"\nEpoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.2f}%")
            print(f"  Val mIoU: {val_metrics['miou']:.2f}%")
            
            # Save best model
            if val_metrics['miou'] > self.best_miou:
                self.best_miou = val_metrics['miou']
                save_path = os.path.join(self.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'miou': self.best_miou
                }, save_path)
                print(f"  ✓ Saved best model (mIoU: {self.best_miou:.2f}%)")
                
            # Periodic interpretability analysis
            if self.args.enable_interpretability:
                # t-SNE visualization
                if epoch % self.args.tsne_interval == 0:
                    self.generate_tsne_visualization(epoch)
                    
                # Gradient attribution
                if epoch % self.args.attribution_interval == 0:
                    self.compute_gradient_attribution(epoch)
                    
        # Final interpretability analysis
        if self.args.enable_interpretability:
            print("\n" + "="*80)
            print("Generating Final Interpretability Analysis")
            print("="*80)
            
            # Save module effectiveness analysis
            if self.interpreter is not None:
                self.interpreter.save_statistics()
                effectiveness = self.interpreter.analyze_module_effectiveness()
                
                print("\nModule Effectiveness Scores:")
                for module_name, score in sorted(effectiveness.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {module_name}: {score:.6f}")
                    
                # Save effectiveness scores
                import json
                with open(os.path.join(self.interp_dir, 'module_effectiveness.json'), 'w') as f:
                    json.dump(effectiveness, f, indent=2)
                    
            # Final t-SNE
            self.generate_tsne_visualization(self.args.epochs)
            
            # Final gradient attribution
            self.compute_gradient_attribution(self.args.epochs)
            
        print("\n" + "="*80)
        print("Training Completed!")
        print(f"Best mIoU: {self.best_miou:.2f}%")
        print(f"Results saved to: {self.output_dir}")
        if self.args.enable_interpretability:
            print(f"Interpretability analysis saved to: {self.interp_dir}")
        print("="*80)


def main():
    args = parse_args()
    
    # Create trainer
    trainer = InterpretableTrainer(args)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()
