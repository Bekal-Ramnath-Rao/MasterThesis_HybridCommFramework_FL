#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pruning Results Comparison Tool
Compares FL experiment results with and without pruning
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import argparse
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class PruningComparator:
    """Compare FL experiments with and without pruning"""
    
    def __init__(self, baseline_dir: str, pruned_dir: str, output_dir: str = None):
        self.baseline_dir = Path(baseline_dir)
        self.pruned_dir = Path(pruned_dir)
        
        # Create descriptive output folder name
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Extract experiment names from input folders
            baseline_name = self.baseline_dir.name
            pruned_name = self.pruned_dir.name
            
            # Create comparison folder name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            folder_name = f"pruning_comparison_{baseline_name}_vs_{pruned_name}_{timestamp}"
            
            # Place in comparison_results directory
            script_dir = Path(__file__).parent.absolute()
            project_root = script_dir.parent
            self.output_dir = project_root / "comparison_results" / folder_name
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Comparison results will be saved to: {self.output_dir}")
        
        self.results = []
        
    def extract_metrics_from_logs(self, log_file: Path) -> Dict:
        """Extract performance metrics from server log file"""
        metrics = {
            'final_loss': None,
            'final_mse': None,
            'final_mae': None,
            'final_mape': None,
            'avg_sparsity': None,
            'avg_compression_ratio': None,
            'avg_training_time': None,
            'convergence_round': None,
            'model_size_reduction_pct': None,
            'communication_savings_pct': None
        }
        
        if not log_file.exists():
            return metrics
            
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract final round metrics
            loss_matches = re.findall(r'Round \d+.*?Loss:\s*([\d.]+)', content, re.IGNORECASE)
            if loss_matches:
                metrics['final_loss'] = float(loss_matches[-1])
                
            mse_matches = re.findall(r'MSE:\s*([\d.]+)', content, re.IGNORECASE)
            if mse_matches:
                metrics['final_mse'] = float(mse_matches[-1])
                
            mae_matches = re.findall(r'MAE:\s*([\d.]+)', content, re.IGNORECASE)
            if mae_matches:
                metrics['final_mae'] = float(mae_matches[-1])
                
            mape_matches = re.findall(r'MAPE:\s*([\d.]+)', content, re.IGNORECASE)
            if mape_matches:
                metrics['final_mape'] = float(mape_matches[-1])
                
            # Extract pruning-specific metrics
            # Pattern: [Server Pruning] Overall Sparsity: 50.0%
            sparsity_matches = re.findall(r'Overall Sparsity:\s*([\d.]+)%', content, re.IGNORECASE)
            if sparsity_matches:
                sparsities = [float(s) for s in sparsity_matches]
                metrics['avg_sparsity'] = sum(sparsities) / len(sparsities)
            
            # Pattern: Compression Ratio: 2.00x
            compression_matches = re.findall(r'Compression\s+Ratio:\s*([\d.]+)x?', content, re.IGNORECASE)
            if compression_matches:
                ratios = [float(r) for r in compression_matches]
                metrics['avg_compression_ratio'] = sum(ratios) / len(ratios)
            
            # Pattern: Size reduction: 61.5%
            size_reduction_matches = re.findall(r'Size\s+reduction.*?:\s*([\d.]+)%', content, re.IGNORECASE)
            if size_reduction_matches:
                reductions = [float(r) for r in size_reduction_matches]
                metrics['model_size_reduction_pct'] = sum(reductions) / len(reductions)
            
            # Pattern: Comm. reduction: 45.0%
            comm_reduction_matches = re.findall(r'Comm.*?reduction.*?:\s*([\d.]+)%', content, re.IGNORECASE)
            if comm_reduction_matches:
                reductions = [float(r) for r in comm_reduction_matches]
                metrics['communication_savings_pct'] = sum(reductions) / len(reductions)
                
            # Extract training times
            time_matches = re.findall(r'Training time:\s*([\d.]+)', content, re.IGNORECASE)
            if time_matches:
                times = [float(t) for t in time_matches]
                metrics['avg_training_time'] = sum(times) / len(times)
                
            # Detect convergence (loss stabilization)
            if loss_matches and len(loss_matches) > 5:
                losses = [float(l) for l in loss_matches]
                for i in range(5, len(losses)):
                    window = losses[i-5:i]
                    if max(window) - min(window) < 0.01:  # Stable within 0.01
                        metrics['convergence_round'] = i
                        break
                        
        except Exception as e:
            print(f"[WARNING] Error parsing {log_file}: {e}")
            
        return metrics
    
    def compare_experiments(self) -> pd.DataFrame:
        """Compare all experiments between baseline and pruned runs"""
        print("\n[INFO] Scanning experiment directories...")
        
        # Find all experiment subdirectories
        baseline_experiments = {d.name: d for d in self.baseline_dir.iterdir() if d.is_dir()}
        pruned_experiments = {d.name: d for d in self.pruned_dir.iterdir() if d.is_dir()}
        
        # Find common experiments
        common_experiments = set(baseline_experiments.keys()) & set(pruned_experiments.keys())
        print(f"[INFO] Found {len(common_experiments)} matching experiments")
        
        for exp_name in sorted(common_experiments):
            print(f"[INFO] Processing: {exp_name}")
            
            # Parse experiment name (protocol_scenario or protocol_scenario_congestion)
            parts = exp_name.split('_')
            if len(parts) >= 2:
                protocol = parts[0]
                if len(parts) == 2:
                    scenario = parts[1]
                    congestion = "none"
                else:
                    scenario = parts[1]
                    congestion = parts[2] if len(parts) > 2 else "none"
            else:
                continue
                
            # Extract metrics from server logs
            baseline_log = baseline_experiments[exp_name] / "server.log"
            pruned_log = pruned_experiments[exp_name] / "server.log"
            
            baseline_metrics = self.extract_metrics_from_logs(baseline_log)
            pruned_metrics = self.extract_metrics_from_logs(pruned_log)
            
            # Calculate improvements
            result = {
                'protocol': protocol,
                'scenario': scenario,
                'congestion': congestion,
                'baseline_loss': baseline_metrics['final_loss'],
                'pruned_loss': pruned_metrics['final_loss'],
                'baseline_mse': baseline_metrics['final_mse'],
                'pruned_mse': pruned_metrics['final_mse'],
                'baseline_mae': baseline_metrics['final_mae'],
                'pruned_mae': pruned_metrics['final_mae'],
                'baseline_mape': baseline_metrics['final_mape'],
                'pruned_mape': pruned_metrics['final_mape'],
                'baseline_training_time': baseline_metrics['avg_training_time'],
                'pruned_training_time': pruned_metrics['avg_training_time'],
                'baseline_convergence': baseline_metrics['convergence_round'],
                'pruned_convergence': pruned_metrics['convergence_round'],
                'sparsity': pruned_metrics['avg_sparsity'],
                'compression_ratio': pruned_metrics['avg_compression_ratio'],
                'size_reduction_pct': pruned_metrics['model_size_reduction_pct'],
                'communication_savings_pct': pruned_metrics['communication_savings_pct']
            }
            
            # Calculate percentage improvements/changes
            if baseline_metrics['final_loss'] and pruned_metrics['final_loss']:
                result['loss_change_pct'] = ((baseline_metrics['final_loss'] - pruned_metrics['final_loss']) / 
                                            baseline_metrics['final_loss'] * 100)
            else:
                result['loss_change_pct'] = None
                
            if baseline_metrics['avg_training_time'] and pruned_metrics['avg_training_time']:
                result['time_change_pct'] = ((baseline_metrics['avg_training_time'] - pruned_metrics['avg_training_time']) / 
                                            baseline_metrics['avg_training_time'] * 100)
            else:
                result['time_change_pct'] = None
                
            self.results.append(result)
            
        df = pd.DataFrame(self.results)
        print(f"\n[OK] Processed {len(df)} experiments")
        return df
    
    def generate_comparison_report(self, df: pd.DataFrame) -> str:
        """Generate text report of comparison results"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("PRUNING COMPARISON REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Baseline Directory: {self.baseline_dir}")
        report_lines.append(f"Pruned Directory: {self.pruned_dir}")
        report_lines.append(f"Total Experiments: {len(df)}")
        report_lines.append("="*80)
        
        # Overall statistics
        report_lines.append("\nOVERALL STATISTICS:")
        report_lines.append("-"*80)
        
        # Pruning effectiveness
        valid_sparsity = df['sparsity'].dropna()
        if len(valid_sparsity) > 0:
            report_lines.append(f"Average Model Sparsity: {valid_sparsity.mean():.2f}%")
            report_lines.append(f"Average Compression Ratio: {df['compression_ratio'].dropna().mean():.2f}x")
        
        valid_size_reduction = df['size_reduction_pct'].dropna()
        if len(valid_size_reduction) > 0:
            report_lines.append(f"Average Model Size Reduction: {valid_size_reduction.mean():.2f}%")
        
        valid_comm_savings = df['communication_savings_pct'].dropna()
        if len(valid_comm_savings) > 0:
            report_lines.append(f"Average Communication Savings: {valid_comm_savings.mean():.2f}%")
        
        # Accuracy impact
        valid_loss_changes = df['loss_change_pct'].dropna()
        if len(valid_loss_changes) > 0:
            report_lines.append(f"\nAverage Loss Change: {valid_loss_changes.mean():.2f}%")
            report_lines.append(f"Best Loss Improvement: {valid_loss_changes.max():.2f}%")
            report_lines.append(f"Worst Loss Degradation: {valid_loss_changes.min():.2f}%")
            
            # Count improvements vs degradations
            improvements = (valid_loss_changes > 0).sum()
            degradations = (valid_loss_changes < 0).sum()
            neutral = (valid_loss_changes == 0).sum()
            report_lines.append(f"Experiments with Improved Loss: {improvements}/{len(valid_loss_changes)}")
            report_lines.append(f"Experiments with Degraded Loss: {degradations}/{len(valid_loss_changes)}")
            report_lines.append(f"Experiments with Same Loss: {neutral}/{len(valid_loss_changes)}")
        
        valid_time_changes = df['time_change_pct'].dropna()
        if len(valid_time_changes) > 0:
            report_lines.append(f"\nAverage Training Time Change: {valid_time_changes.mean():.2f}%")
            report_lines.append(f"Best Time Improvement: {valid_time_changes.max():.2f}%")
        
        # Per-protocol analysis
        report_lines.append("\n" + "="*80)
        report_lines.append("PER-PROTOCOL ANALYSIS:")
        report_lines.append("="*80)
        
        for protocol in sorted(df['protocol'].unique()):
            protocol_df = df[df['protocol'] == protocol]
            report_lines.append(f"\n{protocol.upper()}:")
            report_lines.append("-"*80)
            
            report_lines.append(f"  Experiments: {len(protocol_df)}")
            
            valid_sparsity = protocol_df['sparsity'].dropna()
            if len(valid_sparsity) > 0:
                report_lines.append(f"  Average Sparsity: {valid_sparsity.mean():.2f}%")
                report_lines.append(f"  Average Compression: {protocol_df['compression_ratio'].dropna().mean():.2f}x")
            
            valid_changes = protocol_df['loss_change_pct'].dropna()
            if len(valid_changes) > 0:
                report_lines.append(f"  Average Loss Change: {valid_changes.mean():.2f}%")
                
                # Best and worst scenarios
                best_idx = protocol_df['loss_change_pct'].idxmax()
                worst_idx = protocol_df['loss_change_pct'].idxmin()
                
                if pd.notna(best_idx):
                    best = protocol_df.loc[best_idx]
                    report_lines.append(f"  Best: {best['scenario']} ({best['congestion']}) - {best['loss_change_pct']:.2f}% improvement")
                    
                if pd.notna(worst_idx):
                    worst = protocol_df.loc[worst_idx]
                    report_lines.append(f"  Worst: {worst['scenario']} ({worst['congestion']}) - {worst['loss_change_pct']:.2f}% change")
        
        # Per-scenario analysis
        report_lines.append("\n" + "="*80)
        report_lines.append("PER-SCENARIO ANALYSIS:")
        report_lines.append("="*80)
        
        for scenario in sorted(df['scenario'].unique()):
            scenario_df = df[df['scenario'] == scenario]
            report_lines.append(f"\n{scenario.upper()}:")
            report_lines.append("-"*80)
            
            valid_changes = scenario_df['loss_change_pct'].dropna()
            valid_sparsity = scenario_df['sparsity'].dropna()
            
            if len(valid_changes) > 0:
                report_lines.append(f"  Average Loss Change: {valid_changes.mean():.2f}%")
            if len(valid_sparsity) > 0:
                report_lines.append(f"  Average Sparsity: {valid_sparsity.mean():.2f}%")
                report_lines.append(f"  Average Compression: {scenario_df['compression_ratio'].dropna().mean():.2f}x")
        
        return "\n".join(report_lines)
    
    def plot_comparison_charts(self, df: pd.DataFrame):
        """Generate comparison visualization charts"""
        print("\n[INFO] Generating comparison charts...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 10)
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Pruning Impact Analysis', fontsize=16, fontweight='bold')
        
        # 1. Loss comparison by protocol
        ax1 = axes[0, 0]
        protocols = sorted(df['protocol'].unique())
        baseline_losses = [df[df['protocol'] == p]['baseline_loss'].mean() for p in protocols]
        pruned_losses = [df[df['protocol'] == p]['pruned_loss'].mean() for p in protocols]
        
        x = range(len(protocols))
        width = 0.35
        ax1.bar([i - width/2 for i in x], baseline_losses, width, label='Baseline', alpha=0.8)
        ax1.bar([i + width/2 for i in x], pruned_losses, width, label='Pruned', alpha=0.8)
        ax1.set_xlabel('Protocol', fontweight='bold')
        ax1.set_ylabel('Final Loss', fontweight='bold')
        ax1.set_title('Final Loss Comparison by Protocol')
        ax1.set_xticks(x)
        ax1.set_xticklabels([p.upper() for p in protocols])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Sparsity and compression ratio
        ax2 = axes[0, 1]
        avg_sparsity = [df[df['protocol'] == p]['sparsity'].mean() for p in protocols]
        avg_compression = [df[df['protocol'] == p]['compression_ratio'].mean() for p in protocols]
        
        ax2_twin = ax2.twinx()
        bars1 = ax2.bar([i - width/2 for i in x], avg_sparsity, width, label='Sparsity (%)', alpha=0.8, color='steelblue')
        bars2 = ax2_twin.bar([i + width/2 for i in x], avg_compression, width, label='Compression Ratio', alpha=0.8, color='coral')
        
        ax2.set_xlabel('Protocol', fontweight='bold')
        ax2.set_ylabel('Sparsity (%)', fontweight='bold', color='steelblue')
        ax2_twin.set_ylabel('Compression Ratio', fontweight='bold', color='coral')
        ax2.set_title('Pruning Effectiveness by Protocol')
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.upper() for p in protocols])
        ax2.tick_params(axis='y', labelcolor='steelblue')
        ax2_twin.tick_params(axis='y', labelcolor='coral')
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 3. Communication savings
        ax3 = axes[1, 0]
        comm_savings = [df[df['protocol'] == p]['communication_savings_pct'].mean() for p in protocols]
        bars = ax3.bar(x, comm_savings, alpha=0.8, color='forestgreen')
        ax3.set_xlabel('Protocol', fontweight='bold')
        ax3.set_ylabel('Communication Savings (%)', fontweight='bold')
        ax3.set_title('Communication Efficiency Improvement')
        ax3.set_xticks(x)
        ax3.set_xticklabels([p.upper() for p in protocols])
        ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, comm_savings)):
            if pd.notna(val):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Loss change by scenario
        ax4 = axes[1, 1]
        scenarios = sorted(df['scenario'].unique())
        loss_changes = [df[df['scenario'] == s]['loss_change_pct'].mean() for s in scenarios]
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in loss_changes]
        
        bars = ax4.barh(range(len(scenarios)), loss_changes, alpha=0.8, color=colors)
        ax4.set_ylabel('Network Scenario', fontweight='bold')
        ax4.set_xlabel('Loss Change (%)', fontweight='bold')
        ax4.set_title('Accuracy Impact by Scenario (positive = improvement)')
        ax4.set_yticks(range(len(scenarios)))
        ax4.set_yticklabels([s.replace('_', ' ').title() for s in scenarios])
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, loss_changes)):
            if pd.notna(val):
                label_x = val + (2 if val > 0 else -2)
                ha = 'left' if val > 0 else 'right'
                ax4.text(label_x, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', ha=ha, va='center', fontweight='bold')
        
        plt.tight_layout()
        chart_path = self.output_dir / "pruning_comparison_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Charts saved to: {chart_path}")
        plt.close()
        
    def save_results(self, df: pd.DataFrame):
        """Save comparison results to CSV and generate report"""
        # Save CSV
        csv_path = self.output_dir / "pruning_comparison_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] CSV results saved to: {csv_path}")
        
        # Generate and save report
        report = self.generate_comparison_report(df)
        report_path = self.output_dir / "pruning_comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[OK] Text report saved to: {report_path}")
        
        # Print report to console
        print("\n" + report)


def main():
    parser = argparse.ArgumentParser(
        description="Compare FL experiments with and without pruning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Compare baseline vs pruned experiments
  python compare_pruning_results.py \\
    --baseline experiment_results/temperature_baseline_20260111_120000 \\
    --pruned experiment_results/temperature_pruned_50pct_20260111_130000
  
  # Specify output directory
  python compare_pruning_results.py \\
    --baseline experiment_results/temperature_baseline \\
    --pruned experiment_results/temperature_pruned \\
    --output comparison_results/my_pruning_comparison
        """
    )
    
    parser.add_argument("--baseline", "-b", required=True,
                       help="Path to baseline (non-pruned) experiment directory")
    parser.add_argument("--pruned", "-p", required=True,
                       help="Path to pruned experiment directory")
    parser.add_argument("--output", "-o", default=None,
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # Create comparator and run analysis
    comparator = PruningComparator(args.baseline, args.pruned, args.output)
    
    print("\n[INFO] Starting pruning comparison analysis...")
    df = comparator.compare_experiments()
    
    if len(df) == 0:
        print("\n[ERROR] No matching experiments found to compare!")
        return 1
    
    # Generate outputs
    comparator.save_results(df)
    comparator.plot_comparison_charts(df)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"Results saved to: {comparator.output_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
