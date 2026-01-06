#!/usr/bin/env python3
"""
Analyze Training-time Interpretability Results

This script analyzes the interpretability data collected during training to:
1. Visualize module effectiveness over time
2. Compare gradient flow across modules
3. Analyze multi-modal contribution patterns
4. Generate comprehensive interpretability figures for papers

Usage:
    python analyze_training_interpretability.py --interp_dir checkpoints/interpretability
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Training Interpretability')
    parser.add_argument('--interp_dir', type=str, required=True,
                        help='Directory containing interpretability results')
    parser.add_argument('--output_dir', type=str, default='./interpretability_analysis',
                        help='Directory to save analysis figures')
    return parser.parse_args()


class InterpretabilityAnalyzer:
    """Analyze and visualize training-time interpretability data"""
    
    def __init__(self, interp_dir: str, output_dir: str):
        self.interp_dir = Path(interp_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.module_stats = self._load_module_statistics()
        self.effectiveness = self._load_module_effectiveness()
        self.attributions = self._load_gradient_attributions()
        
    def _load_module_statistics(self) -> Dict:
        """Load module statistics from training"""
        stats_file = self.interp_dir / 'module_analysis' / 'training_interpretability.json'
        
        if not stats_file.exists():
            print(f"Warning: {stats_file} not found")
            return {}
            
        with open(stats_file, 'r') as f:
            return json.load(f)
            
    def _load_module_effectiveness(self) -> Dict:
        """Load module effectiveness scores"""
        eff_file = self.interp_dir / 'module_effectiveness.json'
        
        if not eff_file.exists():
            print(f"Warning: {eff_file} not found")
            return {}
            
        with open(eff_file, 'r') as f:
            return json.load(f)
            
    def _load_gradient_attributions(self) -> Dict:
        """Load gradient attribution data"""
        attr_dir = self.interp_dir / 'attribution'
        
        if not attr_dir.exists():
            return {}
            
        # Load all attribution files
        attributions = {}
        for attr_file in attr_dir.glob('gradient_attributions_epoch_*.json'):
            epoch = int(attr_file.stem.split('_')[-1])
            with open(attr_file, 'r') as f:
                attributions[epoch] = json.load(f)
                
        return attributions
        
    def plot_module_effectiveness(self):
        """Plot module effectiveness comparison"""
        if not self.effectiveness:
            print("No effectiveness data available")
            return
            
        print("\nGenerating module effectiveness plot...")
        
        # Sort by effectiveness
        modules = list(self.effectiveness.keys())
        scores = [self.effectiveness[m] for m in modules]
        
        # Sort
        sorted_indices = np.argsort(scores)[::-1]
        modules = [modules[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(modules)))
        bars = ax.bar(range(len(modules)), scores, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Module', fontsize=14, fontweight='bold')
        ax.set_ylabel('Effectiveness Score (Avg. Gradient Flow)', fontsize=14, fontweight='bold')
        ax.set_title('Module Effectiveness During Training', fontsize=16, fontweight='bold')
        ax.set_xticks(range(len(modules)))
        ax.set_xticklabels(modules, rotation=45, ha='right', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / 'module_effectiveness.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved to: {save_path}")
        
    def plot_gradient_flow_over_time(self):
        """Plot gradient flow for each module over training"""
        if not self.module_stats:
            print("No module statistics available")
            return
            
        print("\nGenerating gradient flow over time plot...")
        
        # Extract gradient flow data
        module_names = set()
        for epoch_data in self.module_stats.values():
            for batch_data in epoch_data.values():
                module_names.update(batch_data['modules'].keys())
                
        module_names = sorted(list(module_names))
        
        # Collect data
        data = {name: {'steps': [], 'grad_flow': []} for name in module_names}
        
        for epoch, epoch_data in self.module_stats.items():
            for batch, batch_data in epoch_data.items():
                step = batch_data['step']
                for module_name in module_names:
                    if module_name in batch_data['modules']:
                        mstats = batch_data['modules'][module_name]
                        if 'gradient_flow' in mstats:
                            data[module_name]['steps'].append(step)
                            data[module_name]['grad_flow'].append(mstats['gradient_flow'])
                            
        # Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(module_names)))
        
        for i, module_name in enumerate(module_names):
            if data[module_name]['steps']:
                ax.plot(data[module_name]['steps'],
                       data[module_name]['grad_flow'],
                       label=module_name,
                       color=colors[i],
                       linewidth=2,
                       alpha=0.8)
                       
        ax.set_xlabel('Training Step', fontsize=14, fontweight='bold')
        ax.set_ylabel('Gradient Flow', fontsize=14, fontweight='bold')
        ax.set_title('Module Gradient Flow During Training', fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'gradient_flow_over_time.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved to: {save_path}")
        
    def plot_activation_statistics(self):
        """Plot activation statistics (mean, std, sparsity) over time"""
        if not self.module_stats:
            print("No module statistics available")
            return
            
        print("\nGenerating activation statistics plot...")
        
        # Extract data
        module_names = set()
        for epoch_data in self.module_stats.values():
            for batch_data in epoch_data.values():
                module_names.update(batch_data['modules'].keys())
                
        module_names = sorted(list(module_names))
        
        # Collect activation statistics
        data = {name: {
            'steps': [],
            'act_mean': [],
            'act_std': [],
            'sparsity': []
        } for name in module_names}
        
        for epoch, epoch_data in self.module_stats.items():
            for batch, batch_data in epoch_data.items():
                step = batch_data['step']
                for module_name in module_names:
                    if module_name in batch_data['modules']:
                        mstats = batch_data['modules'][module_name]
                        if 'activation_mean' in mstats:
                            data[module_name]['steps'].append(step)
                            data[module_name]['act_mean'].append(mstats['activation_mean'])
                            data[module_name]['act_std'].append(mstats.get('activation_std', 0))
                            data[module_name]['sparsity'].append(mstats.get('sparsity', 0))
                            
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(module_names)))
        
        # Activation mean
        for i, module_name in enumerate(module_names):
            if data[module_name]['steps']:
                axes[0].plot(data[module_name]['steps'],
                           data[module_name]['act_mean'],
                           label=module_name,
                           color=colors[i],
                           linewidth=2,
                           alpha=0.8)
                           
        axes[0].set_ylabel('Activation Mean', fontsize=12, fontweight='bold')
        axes[0].set_title('Module Activation Statistics During Training', fontsize=14, fontweight='bold')
        axes[0].legend(loc='best', fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        # Activation std
        for i, module_name in enumerate(module_names):
            if data[module_name]['steps']:
                axes[1].plot(data[module_name]['steps'],
                           data[module_name]['act_std'],
                           label=module_name,
                           color=colors[i],
                           linewidth=2,
                           alpha=0.8)
                           
        axes[1].set_ylabel('Activation Std', fontsize=12, fontweight='bold')
        axes[1].legend(loc='best', fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        # Sparsity
        for i, module_name in enumerate(module_names):
            if data[module_name]['steps']:
                axes[2].plot(data[module_name]['steps'],
                           data[module_name]['sparsity'],
                           label=module_name,
                           color=colors[i],
                           linewidth=2,
                           alpha=0.8)
                           
        axes[2].set_xlabel('Training Step', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Sparsity', fontsize=12, fontweight='bold')
        axes[2].legend(loc='best', fontsize=9)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'activation_statistics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved to: {save_path}")
        
    def plot_multimodal_attribution(self):
        """Plot multi-modal gradient attribution"""
        if not self.attributions:
            print("No attribution data available")
            return
            
        print("\nGenerating multi-modal attribution plot...")
        
        # Extract modality names and classes
        first_epoch = list(self.attributions.values())[0]
        keys = list(first_epoch.keys())
        
        modalities = set()
        classes = set()
        for key in keys:
            parts = key.rsplit('_class', 1)
            if len(parts) == 2:
                modalities.add(parts[0])
                classes.add(int(parts[1]))
                
        modalities = sorted(list(modalities))
        classes = sorted(list(classes))
        
        # Aggregate attribution across epochs
        attribution_matrix = np.zeros((len(modalities), len(classes)))
        
        for epoch_data in self.attributions.values():
            for i, modality in enumerate(modalities):
                for j, cls in enumerate(classes):
                    key = f'{modality}_class{cls}'
                    if key in epoch_data:
                        attribution_matrix[i, j] += epoch_data[key]['mean']
                        
        # Average
        attribution_matrix /= len(self.attributions)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_names = ['Impervious', 'Building', 'Low veg', 'Tree', 'Car', 'Clutter']
        
        sns.heatmap(attribution_matrix,
                   annot=True,
                   fmt='.4f',
                   cmap='YlOrRd',
                   xticklabels=class_names[:len(classes)],
                   yticklabels=modalities,
                   cbar_kws={'label': 'Attribution Score'},
                   ax=ax)
                   
        ax.set_title('Multi-Modal Gradient Attribution by Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Modality', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / 'multimodal_attribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved to: {save_path}")
        
    def generate_summary_report(self):
        """Generate a summary report"""
        print("\nGenerating summary report...")
        
        report_path = self.output_dir / 'interpretability_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Training-time Interpretability Analysis Report\n")
            f.write("="*80 + "\n\n")
            
            # Module effectiveness
            if self.effectiveness:
                f.write("Module Effectiveness Scores:\n")
                f.write("-" * 40 + "\n")
                for module_name, score in sorted(self.effectiveness.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  {module_name:30s}: {score:.6f}\n")
                f.write("\n")
                
            # Attribution summary
            if self.attributions:
                f.write("Multi-Modal Attribution Summary:\n")
                f.write("-" * 40 + "\n")
                
                # Average across all epochs
                all_attrs = {}
                for epoch_data in self.attributions.values():
                    for key, stats in epoch_data.items():
                        if key not in all_attrs:
                            all_attrs[key] = []
                        all_attrs[key].append(stats['mean'])
                        
                for key, values in sorted(all_attrs.items()):
                    avg = np.mean(values)
                    std = np.std(values)
                    f.write(f"  {key:40s}: {avg:.6f} ± {std:.6f}\n")
                f.write("\n")
                
            f.write("="*80 + "\n")
            f.write("Analysis completed successfully!\n")
            f.write(f"Figures saved to: {self.output_dir}\n")
            f.write("="*80 + "\n")
            
        print(f"✓ Saved report to: {report_path}")
        
    def analyze_all(self):
        """Run all analyses"""
        print("\n" + "="*80)
        print("Training-time Interpretability Analysis")
        print("="*80)
        
        self.plot_module_effectiveness()
        self.plot_gradient_flow_over_time()
        self.plot_activation_statistics()
        self.plot_multimodal_attribution()
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("✓ Analysis completed!")
        print(f"✓ All figures saved to: {self.output_dir}")
        print("="*80)


def main():
    args = parse_args()
    
    analyzer = InterpretabilityAnalyzer(args.interp_dir, args.output_dir)
    analyzer.analyze_all()


if __name__ == '__main__':
    main()
