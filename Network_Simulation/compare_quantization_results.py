#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantization Results Comparison Tool
Compares FL experiment results with and without quantization
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


class QuantizationComparator:
    """Compare FL experiments with and without quantization"""
    
    def __init__(self, baseline_dir: str, quantized_dir: str, output_dir: str = None):
        self.baseline_dir = Path(baseline_dir)
        self.quantized_dir = Path(quantized_dir)
        
        # Create descriptive output folder name
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Extract experiment names from input folders
            baseline_name = self.baseline_dir.name
            quantized_name = self.quantized_dir.name
            
            # Create comparison folder name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            folder_name = f"comparison_{baseline_name}_vs_{quantized_name}_{timestamp}"
            
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
            'avg_compression_ratio': None,
            'avg_training_time': None,
            'convergence_round': None
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
                
            # Extract compression ratios
            compression_matches = re.findall(r'Compression ratio:\s*([\d.]+)', content, re.IGNORECASE)
            if compression_matches:
                ratios = [float(r) for r in compression_matches]
                metrics['avg_compression_ratio'] = sum(ratios) / len(ratios)
                
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
        """Compare all experiments between baseline and quantized runs"""
        print("\n[INFO] Scanning experiment directories...")
        
        # Find all experiment subdirectories
        baseline_experiments = {d.name: d for d in self.baseline_dir.iterdir() if d.is_dir()}
        quantized_experiments = {d.name: d for d in self.quantized_dir.iterdir() if d.is_dir()}
        
        # Find common experiments
        common_experiments = set(baseline_experiments.keys()) & set(quantized_experiments.keys())
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
            quantized_log = quantized_experiments[exp_name] / "server.log"
            
            baseline_metrics = self.extract_metrics_from_logs(baseline_log)
            quantized_metrics = self.extract_metrics_from_logs(quantized_log)
            
            # Calculate improvements
            result = {
                'protocol': protocol,
                'scenario': scenario,
                'congestion': congestion,
                'baseline_loss': baseline_metrics['final_loss'],
                'quantized_loss': quantized_metrics['final_loss'],
                'baseline_mse': baseline_metrics['final_mse'],
                'quantized_mse': quantized_metrics['final_mse'],
                'baseline_mae': baseline_metrics['final_mae'],
                'quantized_mae': quantized_metrics['final_mae'],
                'baseline_mape': baseline_metrics['final_mape'],
                'quantized_mape': quantized_metrics['final_mape'],
                'baseline_compression': baseline_metrics['avg_compression_ratio'],
                'quantized_compression': quantized_metrics['avg_compression_ratio'],
                'baseline_training_time': baseline_metrics['avg_training_time'],
                'quantized_training_time': quantized_metrics['avg_training_time'],
                'baseline_convergence': baseline_metrics['convergence_round'],
                'quantized_convergence': quantized_metrics['convergence_round']
            }
            
            # Calculate percentage improvements
            if baseline_metrics['final_loss'] and quantized_metrics['final_loss']:
                result['loss_improvement_pct'] = ((baseline_metrics['final_loss'] - quantized_metrics['final_loss']) / 
                                                 baseline_metrics['final_loss'] * 100)
            else:
                result['loss_improvement_pct'] = None
                
            if baseline_metrics['avg_training_time'] and quantized_metrics['avg_training_time']:
                result['time_improvement_pct'] = ((baseline_metrics['avg_training_time'] - quantized_metrics['avg_training_time']) / 
                                                 baseline_metrics['avg_training_time'] * 100)
            else:
                result['time_improvement_pct'] = None
                
            self.results.append(result)
            
        df = pd.DataFrame(self.results)
        print(f"\n[OK] Processed {len(df)} experiments")
        return df
    
    def generate_comparison_report(self, df: pd.DataFrame) -> str:
        """Generate text report of comparison results"""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("QUANTIZATION COMPARISON REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Baseline Directory: {self.baseline_dir}")
        report_lines.append(f"Quantized Directory: {self.quantized_dir}")
        report_lines.append(f"Total Experiments: {len(df)}")
        report_lines.append("="*80)
        
        # Overall statistics
        report_lines.append("\nOVERALL STATISTICS:")
        report_lines.append("-"*80)
        
        valid_loss_improvements = df['loss_improvement_pct'].dropna()
        if len(valid_loss_improvements) > 0:
            report_lines.append(f"Average Loss Improvement: {valid_loss_improvements.mean():.2f}%")
            report_lines.append(f"Best Loss Improvement: {valid_loss_improvements.max():.2f}%")
            report_lines.append(f"Worst Loss Degradation: {valid_loss_improvements.min():.2f}%")
            
            # Count improvements vs degradations
            improvements = (valid_loss_improvements > 0).sum()
            degradations = (valid_loss_improvements < 0).sum()
            report_lines.append(f"Experiments with Improved Loss: {improvements}/{len(valid_loss_improvements)}")
            report_lines.append(f"Experiments with Degraded Loss: {degradations}/{len(valid_loss_improvements)}")
        
        valid_time_improvements = df['time_improvement_pct'].dropna()
        if len(valid_time_improvements) > 0:
            report_lines.append(f"\nAverage Training Time Improvement: {valid_time_improvements.mean():.2f}%")
            report_lines.append(f"Best Time Improvement: {valid_time_improvements.max():.2f}%")
        
        # Per-protocol analysis
        report_lines.append("\n" + "="*80)
        report_lines.append("PER-PROTOCOL ANALYSIS:")
        report_lines.append("="*80)
        
        for protocol in sorted(df['protocol'].unique()):
            protocol_df = df[df['protocol'] == protocol]
            report_lines.append(f"\n{protocol.upper()}:")
            report_lines.append("-"*80)
            
            valid_improvements = protocol_df['loss_improvement_pct'].dropna()
            if len(valid_improvements) > 0:
                report_lines.append(f"  Average Loss Improvement: {valid_improvements.mean():.2f}%")
                report_lines.append(f"  Experiments: {len(protocol_df)}")
                
                # Best and worst scenarios
                best_idx = protocol_df['loss_improvement_pct'].idxmax()
                worst_idx = protocol_df['loss_improvement_pct'].idxmin()
                
                if pd.notna(best_idx):
                    best = protocol_df.loc[best_idx]
                    report_lines.append(f"  Best: {best['scenario']} ({best['congestion']}) - {best['loss_improvement_pct']:.2f}% improvement")
                    
                if pd.notna(worst_idx):
                    worst = protocol_df.loc[worst_idx]
                    report_lines.append(f"  Worst: {worst['scenario']} ({worst['congestion']}) - {worst['loss_improvement_pct']:.2f}% change")
        
        # Per-scenario analysis
        report_lines.append("\n" + "="*80)
        report_lines.append("PER-SCENARIO ANALYSIS:")
        report_lines.append("="*80)
        
        for scenario in sorted(df['scenario'].unique()):
            scenario_df = df[df['scenario'] == scenario]
            report_lines.append(f"\n{scenario.upper()}:")
            report_lines.append("-"*80)
            
            valid_improvements = scenario_df['loss_improvement_pct'].dropna()
            if len(valid_improvements) > 0:
                report_lines.append(f"  Average Loss Improvement: {valid_improvements.mean():.2f}%")
                report_lines.append(f"  Experiments: {len(scenario_df)}")
        
        # Top improvements and degradations
        report_lines.append("\n" + "="*80)
        report_lines.append("TOP 10 IMPROVEMENTS:")
        report_lines.append("="*80)
        
        top_improvements = df.nlargest(10, 'loss_improvement_pct')
        for idx, row in top_improvements.iterrows():
            if pd.notna(row['loss_improvement_pct']):
                report_lines.append(f"{row['protocol']:6} | {row['scenario']:12} | {row['congestion']:8} | {row['loss_improvement_pct']:+7.2f}%")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("TOP 10 DEGRADATIONS:")
        report_lines.append("="*80)
        
        top_degradations = df.nsmallest(10, 'loss_improvement_pct')
        for idx, row in top_degradations.iterrows():
            if pd.notna(row['loss_improvement_pct']):
                report_lines.append(f"{row['protocol']:6} | {row['scenario']:12} | {row['congestion']:8} | {row['loss_improvement_pct']:+7.2f}%")
        
        report_lines.append("\n" + "="*80)
        
        return "\n".join(report_lines)
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualization plots"""
        print("\n[INFO] Generating visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 10)
        
        # 1. Loss improvement by protocol and scenario
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Heatmap of loss improvements
        pivot_data = df.pivot_table(
            values='loss_improvement_pct',
            index='scenario',
            columns='protocol',
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   ax=axes[0, 0], cbar_kws={'label': 'Loss Improvement %'})
        axes[0, 0].set_title('Average Loss Improvement by Protocol and Scenario')
        axes[0, 0].set_xlabel('Protocol')
        axes[0, 0].set_ylabel('Network Scenario')
        
        # Box plot of loss improvements by protocol
        df_valid = df.dropna(subset=['loss_improvement_pct'])
        if len(df_valid) > 0:
            sns.boxplot(data=df_valid, x='protocol', y='loss_improvement_pct', ax=axes[0, 1])
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Loss Improvement Distribution by Protocol')
            axes[0, 1].set_xlabel('Protocol')
            axes[0, 1].set_ylabel('Loss Improvement %')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Bar plot: average improvement by scenario
        scenario_avg = df.groupby('scenario')['loss_improvement_pct'].mean().sort_values()
        colors = ['green' if x > 0 else 'red' for x in scenario_avg.values]
        scenario_avg.plot(kind='barh', ax=axes[1, 0], color=colors)
        axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].set_title('Average Loss Improvement by Network Scenario')
        axes[1, 0].set_xlabel('Loss Improvement %')
        axes[1, 0].set_ylabel('Network Scenario')
        
        # Scatter: baseline vs quantized loss
        df_scatter = df.dropna(subset=['baseline_loss', 'quantized_loss'])
        if len(df_scatter) > 0:
            axes[1, 1].scatter(df_scatter['baseline_loss'], df_scatter['quantized_loss'], 
                             alpha=0.6, c=df_scatter['loss_improvement_pct'], 
                             cmap='RdYlGn', s=100)
            
            # Add diagonal line (y=x)
            max_val = max(df_scatter['baseline_loss'].max(), df_scatter['quantized_loss'].max())
            axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='No change')
            
            axes[1, 1].set_title('Baseline vs Quantized Loss')
            axes[1, 1].set_xlabel('Baseline Loss')
            axes[1, 1].set_ylabel('Quantized Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.output_dir / 'loss_comparison.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved: {plot_file}")
        plt.close()
        
        # 2. Training time comparison
        if df['time_improvement_pct'].notna().any():
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Box plot by protocol
            df_time = df.dropna(subset=['time_improvement_pct'])
            if len(df_time) > 0:
                sns.boxplot(data=df_time, x='protocol', y='time_improvement_pct', ax=axes[0])
                axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                axes[0].set_title('Training Time Improvement by Protocol')
                axes[0].set_xlabel('Protocol')
                axes[0].set_ylabel('Time Improvement %')
                axes[0].tick_params(axis='x', rotation=45)
            
            # Scatter: baseline vs quantized time
            df_time_scatter = df.dropna(subset=['baseline_training_time', 'quantized_training_time'])
            if len(df_time_scatter) > 0:
                axes[1].scatter(df_time_scatter['baseline_training_time'], 
                              df_time_scatter['quantized_training_time'],
                              alpha=0.6, s=100)
                
                max_time = max(df_time_scatter['baseline_training_time'].max(),
                             df_time_scatter['quantized_training_time'].max())
                axes[1].plot([0, max_time], [0, max_time], 'r--', alpha=0.5, label='No change')
                axes[1].set_title('Baseline vs Quantized Training Time')
                axes[1].set_xlabel('Baseline Time (s)')
                axes[1].set_ylabel('Quantized Time (s)')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            time_plot_file = self.output_dir / 'training_time_comparison.png'
            plt.savefig(time_plot_file, dpi=150, bbox_inches='tight')
            print(f"[OK] Saved: {time_plot_file}")
            plt.close()
        
        # 3. Convergence comparison
        if df['baseline_convergence'].notna().any() or df['quantized_convergence'].notna().any():
            fig, ax = plt.subplots(figsize=(12, 6))
            
            df_conv = df.dropna(subset=['baseline_convergence', 'quantized_convergence'])
            if len(df_conv) > 0:
                x_pos = range(len(df_conv))
                width = 0.35
                
                ax.bar([x - width/2 for x in x_pos], df_conv['baseline_convergence'], 
                      width, label='Baseline', alpha=0.8)
                ax.bar([x + width/2 for x in x_pos], df_conv['quantized_convergence'],
                      width, label='Quantized', alpha=0.8)
                
                ax.set_xlabel('Experiment')
                ax.set_ylabel('Convergence Round')
                ax.set_title('Convergence Speed: Baseline vs Quantized')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                plt.tight_layout()
                conv_plot_file = self.output_dir / 'convergence_comparison.png'
                plt.savefig(conv_plot_file, dpi=150, bbox_inches='tight')
                print(f"[OK] Saved: {conv_plot_file}")
                plt.close()
        
        print("[OK] All visualizations generated")
    
    def export_csv(self, df: pd.DataFrame):
        """Export detailed results to CSV"""
        csv_file = self.output_dir / 'detailed_comparison.csv'
        df.to_csv(csv_file, index=False)
        print(f"[OK] Detailed results exported to: {csv_file}")
    
    def run_comparison(self):
        """Run complete comparison analysis"""
        print("\n" + "="*80)
        print("STARTING QUANTIZATION COMPARISON ANALYSIS")
        print("="*80)
        
        # Compare experiments
        df = self.compare_experiments()
        
        if df.empty:
            print("[ERROR] No matching experiments found!")
            return
        
        # Generate report
        print("\n[INFO] Generating comparison report...")
        report = self.generate_comparison_report(df)
        
        # Save report
        report_file = self.output_dir / 'comparison_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[OK] Report saved to: {report_file}")
        
        # Print report to console
        print("\n" + report)
        
        # Create visualizations
        self.create_visualizations(df)
        
        # Export CSV
        self.export_csv(df)
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print(f"All results saved to: {self.output_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Compare FL experiments with and without quantization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two experiment result folders
  python compare_quantization_results.py \\
      --baseline experiment_results/temperature_20260108_112455 \\
      --quantized experiment_results/temperature_quantized_8bit_20260108_184411
  
  # Specify custom output directory
  python compare_quantization_results.py \\
      --baseline experiment_results/emotion_20260104_151749 \\
      --quantized experiment_results/emotion_quantized_20260104_152121 \\
      --output comparison_results/emotion_custom_analysis
        """
    )
    
    parser.add_argument('--baseline', required=True,
                       help='Path to baseline (non-quantized) experiment results directory')
    parser.add_argument('--quantized', required=True,
                       help='Path to quantized experiment results directory')
    parser.add_argument('--output', 
                       help='Output directory for comparison results (auto-generated if not specified)')
    
    args = parser.parse_args()
    
    # Validate input directories
    baseline_path = Path(args.baseline)
    quantized_path = Path(args.quantized)
    
    if not baseline_path.exists():
        print(f"[ERROR] Baseline directory not found: {baseline_path}")
        sys.exit(1)
    
    if not quantized_path.exists():
        print(f"[ERROR] Quantized directory not found: {quantized_path}")
        sys.exit(1)
    
    # Run comparison
    comparator = QuantizationComparator(
        baseline_dir=str(baseline_path),
        quantized_dir=str(quantized_path),
        output_dir=args.output
    )
    comparator.run_comparison()


if __name__ == "__main__":
    main()
