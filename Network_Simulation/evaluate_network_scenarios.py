#!/usr/bin/env python3
"""
Comprehensive Network Scenario Evaluation Script
Evaluates and compares FL protocol performance across all network conditions
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse
import pandas as pd
from matplotlib.patches import Rectangle


class NetworkScenarioEvaluator:
    """Evaluate FL protocol performance across different network scenarios"""
    
    def __init__(self, use_case="temperature"):
        self.use_case = use_case
        self.protocols = ["mqtt", "amqp", "grpc", "quic", "dds"]
        self.scenarios = ["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                         "congested_light", "congested_moderate", "congested_heavy"]
        self.results = {}  # {scenario: {protocol: data}}
        
        # Network scenario descriptions for visualization
        self.scenario_descriptions = {
            "excellent": "Excellent (LAN)\n2ms latency",
            "good": "Good (Broadband)\n10ms latency",
            "moderate": "Moderate (4G/LTE)\n50ms latency",
            "poor": "Poor (3G)\n100ms latency",
            "very_poor": "Very Poor (2G)\n300ms latency",
            "satellite": "Satellite\n600ms latency",
            "congested_light": "Light Congestion\n30ms latency",
            "congested_moderate": "Moderate Congestion\n75ms latency",
            "congested_heavy": "Heavy Congestion\n150ms latency"
        }
        
    def load_all_results(self):
        """Load results for all scenarios and protocols"""
        results_dir = Path(f"Server/{self.use_case.title()}_Regulation/results")
        
        if not results_dir.exists():
            print(f"[ERROR] Results directory not found: {results_dir}")
            return False
        
        loaded_count = 0
        
        # Load standard scenarios
        for scenario in self.scenarios:
            self.results[scenario] = {}
            for protocol in self.protocols:
                result_file = results_dir / f"{protocol}_{scenario}_training_results.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        self.results[scenario][protocol] = json.load(f)
                        loaded_count += 1
                        print(f"✓ Loaded {protocol.upper()} - {scenario}")
        
        # Load congestion variant results
        congestion_levels = ["none", "light", "moderate", "heavy", "extreme"]
        for base_scenario in ["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                             "congested_light", "congested_moderate", "congested_heavy"]:
            for congestion in congestion_levels:
                if congestion == "none":
                    continue
                    
                scenario_key = f"{base_scenario}_congestion_{congestion}"
                
                # Check if any results exist for this combination
                has_results = False
                for protocol in self.protocols:
                    result_file = results_dir / f"{protocol}_{base_scenario}_congestion_{congestion}_training_results.json"
                    if result_file.exists():
                        if scenario_key not in self.results:
                            self.results[scenario_key] = {}
                        with open(result_file, 'r') as f:
                            self.results[scenario_key][protocol] = json.load(f)
                            loaded_count += 1
                            has_results = True
                            print(f"✓ Loaded {protocol.upper()} - {base_scenario} + {congestion} congestion")
        
        print(f"\n✓ Loaded {loaded_count} result files\n")
        return loaded_count > 0
    
    def plot_protocol_across_scenarios(self, protocol, save_path=None):
        """Plot a single protocol's performance across all network scenarios"""
        # Collect data for this protocol across scenarios
        scenarios_data = []
        times = []
        losses = []
        
        for scenario in self.scenarios:
            if scenario in self.results and protocol in self.results[scenario]:
                data = self.results[scenario][protocol]
                scenarios_data.append(scenario)
                times.append(data.get('convergence_time_seconds', 0))
                loss_values = data.get('loss', [])
                losses.append(loss_values[-1] if loss_values else 0)
        
        if not scenarios_data:
            print(f"No data available for {protocol.upper()}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#9B59B6', '#34495E']
        
        # Plot 1: Training Time
        axes[0].bar(range(len(scenarios_data)), times, color=colors[:len(scenarios_data)], alpha=0.8)
        axes[0].set_xticks(range(len(scenarios_data)))
        axes[0].set_xticklabels([s.title() for s in scenarios_data], rotation=45, ha='right')
        axes[0].set_ylabel('Training Time (seconds)', fontweight='bold')
        axes[0].set_title(f'{protocol.upper()} - Time to Convergence', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, t in enumerate(times):
            axes[0].text(i, t, f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Final Loss
        axes[1].bar(range(len(scenarios_data)), losses, color=colors[:len(scenarios_data)], alpha=0.8)
        axes[1].set_xticks(range(len(scenarios_data)))
        axes[1].set_xticklabels([s.title() for s in scenarios_data], rotation=45, ha='right')
        axes[1].set_ylabel('Final Loss', fontweight='bold')
        axes[1].set_title(f'{protocol.upper()} - Final Loss', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, l in enumerate(losses):
            axes[1].text(i, l, f'{l:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.suptitle(f'{protocol.upper()} Performance Across Network Scenarios - {self.use_case.title()}',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
    
    def plot_scenario_heatmap(self, metric='time', save_path=None):
        """Create a heatmap showing protocol performance across scenarios"""
        # Collect data
        data_matrix = []
        available_scenarios = []
        
        for scenario in self.scenarios:
            if scenario not in self.results or not self.results[scenario]:
                continue
            
            row = []
            for protocol in self.protocols:
                if protocol in self.results[scenario]:
                    data = self.results[scenario][protocol]
                    if metric == 'time':
                        value = data.get('convergence_time_seconds', 0)
                    elif metric == 'loss':
                        loss_values = data.get('loss', [])
                        value = loss_values[-1] if loss_values else 0
                    elif metric == 'rounds':
                        value = data.get('total_rounds', 0)
                    else:
                        value = 0
                    row.append(value)
                else:
                    row.append(0)
            
            if any(row):  # Only add if there's data
                data_matrix.append(row)
                available_scenarios.append(scenario)
        
        if not data_matrix:
            print(f"No data available for heatmap ({metric})")
            return
        
        data_matrix = np.array(data_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(self.protocols)))
        ax.set_yticks(np.arange(len(available_scenarios)))
        ax.set_xticklabels([p.upper() for p in self.protocols])
        ax.set_yticklabels([s.title() for s in available_scenarios])
        
        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add values to cells
        for i in range(len(available_scenarios)):
            for j in range(len(self.protocols)):
                if metric == 'time':
                    text = ax.text(j, i, f'{data_matrix[i, j]:.1f}',
                                 ha="center", va="center", color="black", fontweight='bold')
                elif metric == 'loss':
                    text = ax.text(j, i, f'{data_matrix[i, j]:.4f}',
                                 ha="center", va="center", color="black", fontweight='bold', fontsize=9)
                else:
                    text = ax.text(j, i, f'{int(data_matrix[i, j])}',
                                 ha="center", va="center", color="black", fontweight='bold')
        
        # Labels
        metric_labels = {'time': 'Training Time (seconds)', 'loss': 'Final Loss', 'rounds': 'Rounds'}
        ax.set_xlabel('Protocol', fontweight='bold', fontsize=12)
        ax.set_ylabel('Network Scenario', fontweight='bold', fontsize=12)
        ax.set_title(f'{metric_labels.get(metric, metric)} Heatmap - {self.use_case.title()}',
                    fontweight='bold', fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_labels.get(metric, metric), rotation=270, labelpad=20, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
    
    def plot_cross_scenario_comparison(self, save_path=None):
        """Create grouped bar chart comparing all protocols across scenarios"""
        # Collect data
        scenario_data = {}
        
        for scenario in self.scenarios:
            if scenario not in self.results or not self.results[scenario]:
                continue
            
            times = []
            for protocol in self.protocols:
                if protocol in self.results[scenario]:
                    time = self.results[scenario][protocol].get('convergence_time_seconds', 0)
                    times.append(time)
                else:
                    times.append(0)
            
            if any(times):
                scenario_data[scenario] = times
        
        if not scenario_data:
            print("No data available for cross-scenario comparison")
            return
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(self.protocols))
        width = 0.12
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C', '#9B59B6', '#34495E']
        
        # Plot bars for each scenario
        for i, (scenario, times) in enumerate(scenario_data.items()):
            offset = width * (i - len(scenario_data)/2 + 0.5)
            bars = ax.bar(x + offset, times, width, label=scenario.title(), 
                         color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Protocol', fontweight='bold', fontsize=12)
        ax.set_ylabel('Training Time (seconds)', fontweight='bold', fontsize=12)
        ax.set_title(f'Protocol Performance Across Network Scenarios - {self.use_case.title()}',
                    fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([p.upper() for p in self.protocols])
        ax.legend(title='Network Scenario', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        else:
            plt.show()
    
    def generate_comprehensive_table(self):
        """Generate a comprehensive comparison table"""
        print(f"\n{'='*120}")
        print(f"Comprehensive Network Scenario Evaluation - {self.use_case.title()}")
        print(f"{'='*120}\n")
        
        for scenario in self.scenarios:
            if scenario not in self.results or not self.results[scenario]:
                print(f"{scenario.upper()}: No data available\n")
                continue
            
            print(f"{'─'*120}")
            print(f"{scenario.upper()} Network Scenario")
            print(f"{'─'*120}")
            print(f"{'Protocol':<10} {'Rounds':<8} {'Time (s)':<12} {'Time (min)':<12} {'Final Loss':<12} {'Final MSE':<12}")
            print(f"{'-'*120}")
            
            for protocol in self.protocols:
                if protocol in self.results[scenario]:
                    data = self.results[scenario][protocol]
                    rounds = data.get('total_rounds', 0)
                    time_s = data.get('convergence_time_seconds', 0)
                    time_m = data.get('convergence_time_minutes', 0)
                    loss_values = data.get('loss', [])
                    mse_values = data.get('mse', [])
                    final_loss = loss_values[-1] if loss_values else 0
                    final_mse = mse_values[-1] if mse_values else 0
                    
                    print(f"{protocol.upper():<10} {rounds:<8} {time_s:<12.2f} {time_m:<12.2f} "
                          f"{final_loss:<12.6f} {final_mse:<12.6f}")
                else:
                    print(f"{protocol.upper():<10} {'N/A':<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
            print()
        
        print(f"{'='*120}\n")
    
    def export_to_csv(self, output_dir="comparison_results"):
        """Export all results to CSV files for further analysis"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a comprehensive CSV
        rows = []
        for scenario in self.results.keys():  # Use all loaded scenarios
            for protocol in self.protocols:
                if protocol in self.results[scenario]:
                    data = self.results[scenario][protocol]
                    
                    # Parse scenario and congestion level
                    base_scenario = scenario
                    congestion_level = "none"
                    if "_congestion_" in scenario:
                        parts = scenario.split("_congestion_")
                        base_scenario = parts[0]
                        congestion_level = parts[1] if len(parts) > 1 else "none"
                    
                    rows.append({
                        'use_case': self.use_case,
                        'base_scenario': base_scenario,
                        'congestion_level': congestion_level,
                        'scenario_full': scenario,
                        'protocol': protocol,
                        'rounds': data.get('total_rounds', 0),
                        'time_seconds': data.get('convergence_time_seconds', 0),
                        'time_minutes': data.get('convergence_time_minutes', 0),
                        'final_loss': data.get('loss', [])[-1] if data.get('loss') else 0,
                        'final_mse': data.get('mse', [])[-1] if data.get('mse') else 0,
                        'final_mae': data.get('mae', [])[-1] if data.get('mae') else 0,
                        'final_mape': data.get('mape', [])[-1] if data.get('mape') else 0,
                    })
        
        if rows:
            df = pd.DataFrame(rows)
            output_file = f"{output_dir}/{self.use_case}_all_scenarios_comparison.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ Exported comprehensive results to: {output_file}")
            
            # Also create separate CSV for congestion analysis if congestion data exists
            congestion_rows = [r for r in rows if r['congestion_level'] != 'none']
            if congestion_rows:
                df_congestion = pd.DataFrame(congestion_rows)
                output_file_congestion = f"{output_dir}/{self.use_case}_congestion_comparison.csv"
                df_congestion.to_csv(output_file_congestion, index=False)
                print(f"✓ Exported congestion results to: {output_file_congestion}")
        else:
            print("No data to export")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Network Scenario Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--use-case", "-u",
                       choices=["emotion", "mentalstate", "temperature"],
                       default="temperature",
                       help="Use case to evaluate")
    parser.add_argument("--output-dir", "-o",
                       default="comparison_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--export-csv", action="store_true",
                       help="Export results to CSV")
    parser.add_argument("--experiment-folder",
                       help="Name of experiment folder (for naming output files)")
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = NetworkScenarioEvaluator(use_case=args.use_case)
    
    # Load all results
    print(f"\nLoading all network scenario results for {args.use_case.title()}...\n")
    if not evaluator.load_all_results():
        print("[ERROR] No results loaded. Run consolidate_results.py first!")
        return
    
    # Generate comprehensive table
    evaluator.generate_comprehensive_table()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating comprehensive visualizations...\n")
    
    # 1. Heatmaps
    print("Creating heatmaps...")
    evaluator.plot_scenario_heatmap(
        metric='time',
        save_path=f"{args.output_dir}/{args.use_case}_heatmap_time.png"
    )
    evaluator.plot_scenario_heatmap(
        metric='loss',
        save_path=f"{args.output_dir}/{args.use_case}_heatmap_loss.png"
    )
    evaluator.plot_scenario_heatmap(
        metric='rounds',
        save_path=f"{args.output_dir}/{args.use_case}_heatmap_rounds.png"
    )
    
    # 2. Cross-scenario comparison
    print("\nCreating cross-scenario comparison...")
    evaluator.plot_cross_scenario_comparison(
        save_path=f"{args.output_dir}/{args.use_case}_cross_scenario_comparison.png"
    )
    
    # 3. Individual protocol analysis
    print("\nCreating individual protocol analyses...")
    for protocol in evaluator.protocols:
        evaluator.plot_protocol_across_scenarios(
            protocol,
            save_path=f"{args.output_dir}/{args.use_case}_{protocol}_across_scenarios.png"
        )
    
    # 4. Export to CSV
    if args.export_csv:
        print("\nExporting to CSV...")
        evaluator.export_to_csv(args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"✓ Evaluation complete! All results saved to: {args.output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
