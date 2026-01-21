#!/usr/bin/env python3
"""
Visualize Federated Learning Experiment Results
Creates graphs and charts from experiment data
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import seaborn as sns
from typing import Dict, List

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsVisualizer:
    """Visualize FL experiment results with various charts and graphs"""
    
    def __init__(self, use_case: str = "emotion", experiment_folder: str = None):
        self.use_case = use_case
        
        # Determine results directory
        if experiment_folder:
            self.exp_dir = Path("experiment_results") / experiment_folder
        else:
            # Use latest experiment folder for this use case
            exp_base = Path("experiment_results")
            exp_dirs = sorted(exp_base.glob(f"{use_case}_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            self.exp_dir = exp_dirs[0] if exp_dirs else exp_base
        
        # Output directory for plots
        self.output_dir = self.exp_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Results Directory: {self.exp_dir}")
        print(f"Output Directory: {self.output_dir}")
        
        self.protocols = ["mqtt", "amqp", "grpc", "quic", "dds"]
        self.scenarios = ["excellent", "good", "moderate", "poor", "very_poor", 
                         "satellite", "congested_light", "congested_moderate", "congested_heavy"]
        
        # Color mapping for protocols
        self.protocol_colors = {
            "mqtt": "#FF6B6B",
            "amqp": "#4ECDC4",
            "grpc": "#45B7D1",
            "quic": "#FFA07A",
            "dds": "#98D8C8"
        }
    
    def load_results(self) -> Dict:
        """Load all experiment results"""
        all_results = {}
        
        for scenario in self.scenarios:
            all_results[scenario] = {}
            
            for protocol in self.protocols:
                # Try both standard and congestion formats
                patterns = [
                    f"{protocol}_{scenario}",
                    f"{protocol}_{scenario}_congestion_*"
                ]
                
                for pattern in patterns:
                    for proto_dir in self.exp_dir.glob(pattern):
                        # Load results from JSON if available
                        json_file = proto_dir / f"{protocol}_{scenario}_training_results.json"
                        
                        # Also check for congestion variants
                        if not json_file.exists():
                            json_files = list(proto_dir.glob(f"{protocol}_*.json"))
                            json_file = json_files[0] if json_files else None
                        
                        if json_file and json_file.exists():
                            with open(json_file, 'r') as f:
                                all_results[scenario][protocol] = json.load(f)
                            break
        
        # Filter out empty scenarios
        all_results = {k: v for k, v in all_results.items() if v}
        
        print(f"\nLoaded results for {len(all_results)} scenarios")
        for scenario, protocols in all_results.items():
            print(f"  {scenario}: {list(protocols.keys())}")
        
        return all_results
    
    def plot_convergence_time_comparison(self, results: Dict):
        """Plot convergence time comparison across protocols and scenarios"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios_with_data = sorted(results.keys())
        x = np.arange(len(scenarios_with_data))
        width = 0.15
        
        for i, protocol in enumerate(self.protocols):
            times = []
            for scenario in scenarios_with_data:
                if protocol in results[scenario]:
                    times.append(results[scenario][protocol].get('convergence_time_seconds', 0))
                else:
                    times.append(0)
            
            offset = (i - 2) * width
            bars = ax.bar(x + offset, times, width, label=protocol.upper(), 
                         color=self.protocol_colors.get(protocol, 'gray'))
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}s',
                           ha='center', va='bottom', fontsize=8, rotation=0)
        
        ax.set_xlabel('Network Scenario', fontsize=12, fontweight='bold')
        ax.set_ylabel('Convergence Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'Convergence Time Comparison Across Protocols and Network Scenarios\n{self.use_case.title()} Use Case', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios_with_data], rotation=45, ha='right')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "convergence_time_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_training_rounds_comparison(self, results: Dict):
        """Plot total training rounds comparison"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios_with_data = sorted(results.keys())
        x = np.arange(len(scenarios_with_data))
        width = 0.15
        
        for i, protocol in enumerate(self.protocols):
            rounds = []
            for scenario in scenarios_with_data:
                if protocol in results[scenario]:
                    rounds.append(results[scenario][protocol].get('total_rounds', 0))
                else:
                    rounds.append(0)
            
            offset = (i - 2) * width
            bars = ax.bar(x + offset, rounds, width, label=protocol.upper(),
                         color=self.protocol_colors.get(protocol, 'gray'))
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Network Scenario', fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Rounds', fontsize=12, fontweight='bold')
        ax.set_title(f'Training Rounds Until Convergence\n{self.use_case.title()} Use Case',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios_with_data], rotation=45, ha='right')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "training_rounds_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_loss_curves(self, results: Dict, scenario: str = "moderate"):
        """Plot loss curves for all protocols in a specific scenario"""
        if scenario not in results or not results[scenario]:
            print(f"No data for scenario: {scenario}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for protocol in self.protocols:
            if protocol in results[scenario]:
                data = results[scenario][protocol]
                if 'rounds' in data and 'loss' in data:
                    ax.plot(data['rounds'], data['loss'], 
                           marker='o', label=protocol.upper(), linewidth=2,
                           color=self.protocol_colors.get(protocol, 'gray'))
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'Training Loss Curves - {scenario.replace("_", " ").title()} Network\n{self.use_case.title()} Use Case',
                    fontsize=14, fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f"loss_curves_{scenario}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_accuracy_curves(self, results: Dict, scenario: str = "moderate"):
        """Plot accuracy curves for all protocols in a specific scenario"""
        if scenario not in results or not results[scenario]:
            print(f"No data for scenario: {scenario}")
            return
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        has_data = False
        for protocol in self.protocols:
            if protocol in results[scenario]:
                data = results[scenario][protocol]
                if 'rounds' in data and 'accuracy' in data:
                    has_data = True
                    # Convert to percentage if needed
                    acc = [a * 100 if a <= 1 else a for a in data['accuracy']]
                    ax.plot(data['rounds'], acc,
                           marker='s', label=protocol.upper(), linewidth=2,
                           color=self.protocol_colors.get(protocol, 'gray'))
        
        if not has_data:
            print(f"No accuracy data for scenario: {scenario}")
            plt.close()
            return
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Training Accuracy - {scenario.replace("_", " ").title()} Network\n{self.use_case.title()} Use Case',
                    fontsize=14, fontweight='bold')
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / f"accuracy_curves_{scenario}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def plot_protocol_performance_heatmap(self, results: Dict):
        """Create heatmap of convergence times across protocols and scenarios"""
        scenarios_with_data = sorted(results.keys())
        
        # Create matrix for heatmap
        data_matrix = []
        for protocol in self.protocols:
            row = []
            for scenario in scenarios_with_data:
                if protocol in results[scenario]:
                    row.append(results[scenario][protocol].get('convergence_time_seconds', 0))
                else:
                    row.append(0)
            data_matrix.append(row)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(scenarios_with_data)))
        ax.set_yticks(np.arange(len(self.protocols)))
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios_with_data], rotation=45, ha='right')
        ax.set_yticklabels([p.upper() for p in self.protocols])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Convergence Time (seconds)', fontsize=11, fontweight='bold')
        
        # Add text annotations
        for i in range(len(self.protocols)):
            for j in range(len(scenarios_with_data)):
                text = ax.text(j, i, f'{data_matrix[i][j]:.1f}',
                             ha="center", va="center", color="black", fontsize=9, fontweight='bold')
        
        ax.set_title(f'Protocol Performance Heatmap - Convergence Time\n{self.use_case.title()} Use Case',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_file = self.output_dir / "performance_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualization plots"""
        print(f"\n{'='*70}")
        print(f"Generating Visualizations for {self.use_case.title()} Use Case")
        print(f"{'='*70}\n")
        
        results = self.load_results()
        
        if not results:
            print("❌ No results found to visualize!")
            return
        
        print("\nGenerating plots...")
        
        # Overall comparisons
        self.plot_convergence_time_comparison(results)
        self.plot_training_rounds_comparison(results)
        self.plot_protocol_performance_heatmap(results)
        
        # Per-scenario plots
        for scenario in results.keys():
            if results[scenario]:
                self.plot_loss_curves(results, scenario)
                self.plot_accuracy_curves(results, scenario)
        
        print(f"\n{'='*70}")
        print(f"✓ All visualizations saved to: {self.output_dir}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize FL Experiment Results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize specific experiment folder
  python visualize_results.py --use-case emotion --experiment-folder emotion_20260120_153149
  
  # Visualize latest experiment for a use case
  python visualize_results.py --use-case emotion
        """
    )
    
    parser.add_argument("--use-case", "-u",
                       choices=["emotion", "mentalstate", "temperature"],
                       default="emotion",
                       help="Use case to visualize")
    parser.add_argument("--experiment-folder", "-e",
                       help="Specific experiment folder name (e.g., emotion_20260120_153149)")
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(
        use_case=args.use_case,
        experiment_folder=args.experiment_folder
    )
    
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()
