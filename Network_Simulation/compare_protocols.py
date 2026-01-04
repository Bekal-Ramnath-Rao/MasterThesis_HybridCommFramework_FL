#!/usr/bin/env python3
"""
Protocol Comparison Visualization Script
Compares FL protocol performance across different metrics
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import argparse


class ProtocolComparator:
    """Compare FL protocol performance across different network scenarios"""
    
    def __init__(self, use_case="temperature", scenarios=None):
        self.use_case = use_case
        self.scenarios = scenarios or ["excellent"]  # Default to excellent if not specified
        self.protocols = ["mqtt", "amqp", "grpc", "quic", "dds"]
        self.results = {}  # {scenario: {protocol: data}}
        
    def load_results(self, scenario=None):
        """Load results from Server results directory for specified scenario(s)"""
        results_dir = f"Server/{self.use_case.title()}_Regulation/results"
        
        scenarios_to_load = [scenario] if scenario else self.scenarios
        
        for scen in scenarios_to_load:
            if scen not in self.results:
                self.results[scen] = {}
            
            for protocol in self.protocols:
                result_file = f"{results_dir}/{protocol}_{scen}_training_results.json"
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        self.results[scen][protocol] = json.load(f)
                        print(f"✓ Loaded {protocol.upper()} results for {scen} scenario")
                else:
                    print(f"✗ No results found for {protocol.upper()} in {scen} scenario: {result_file}")
    
    def load_all_scenarios(self):
        """Load results for all available scenarios"""
        all_scenarios = ["excellent", "good", "moderate", "poor", "very_poor", "satellite"]
        self.scenarios = all_scenarios
        self.load_results()
    
    def plot_convergence_comparison(self, scenario=None, save_path=None):
        """Plot rounds vs time for all protocols in a specific scenario"""
        scenario = scenario or self.scenarios[0]  # Use first scenario if not specified
        
        if scenario not in self.results or not self.results[scenario]:
            print(f"No results loaded for scenario: {scenario}")
            return
        
        results = self.results[scenario]
        
        plt.figure(figsize=(12, 6))
        
        for protocol, data in results.items():
            rounds = data.get('rounds', [])
            total_time = data.get('convergence_time_seconds', 0)
            
            # Calculate cumulative time (assuming equal time per round)
            if rounds and total_time > 0:
                time_per_round = total_time / len(rounds)
                cumulative_time = [time_per_round * (i + 1) for i in range(len(rounds))]
                
                plt.plot(rounds, cumulative_time, marker='o', linewidth=2, 
                        markersize=6, label=protocol.upper())
        
        plt.xlabel('Training Rounds', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative Training Time (seconds)', fontsize=12, fontweight='bold')
        plt.title(f'FL Protocol Comparison - {self.use_case.title()} ({scenario.title()} Network)', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_loss_comparison(self, scenario=None, save_path=None):
        """Plot loss convergence for all protocols in a specific scenario"""
        scenario = scenario or self.scenarios[0]
        
        if scenario not in self.results or not self.results[scenario]:
            print(f"No results loaded for scenario: {scenario}")
            return
        
        results = self.results[scenario]
        
        plt.figure(figsize=(12, 6))
        
        for protocol, data in results.items():
            rounds = data.get('rounds', [])
            loss = data.get('loss', [])
            
            if rounds and loss:
                plt.plot(rounds, loss, marker='o', linewidth=2, 
                        markersize=6, label=protocol.upper())
        
        plt.xlabel('Training Rounds', fontsize=12, fontweight='bold')
        plt.ylabel('Loss', fontsize=12, fontweight='bold')
        plt.title(f'Loss Convergence Comparison - {self.use_case.title()} ({scenario.title()} Network)', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_summary_bars(self, scenario=None, save_path=None):
        """Create bar charts comparing total time and rounds for a specific scenario"""
        scenario = scenario or self.scenarios[0]
        
        if scenario not in self.results or not self.results[scenario]:
            print(f"No results loaded for scenario: {scenario}")
            return
        
        results = self.results[scenario]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        protocols = []
        times = []
        rounds = []
        final_loss = []
        
        for protocol, data in sorted(results.items()):
            protocols.append(protocol.upper())
            times.append(data.get('convergence_time_seconds', 0))
            rounds.append(data.get('total_rounds', 0))
            loss_values = data.get('loss', [])
            final_loss.append(loss_values[-1] if loss_values else 0)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        
        # Plot 1: Training Time
        axes[0].bar(protocols, times, color=colors[:len(protocols)], alpha=0.8)
        axes[0].set_ylabel('Training Time (seconds)', fontweight='bold')
        axes[0].set_title('Time to Convergence', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, (p, t) in enumerate(zip(protocols, times)):
            axes[0].text(i, t, f'{t:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Rounds
        axes[1].bar(protocols, rounds, color=colors[:len(protocols)], alpha=0.8)
        axes[1].set_ylabel('Rounds', fontweight='bold')
        axes[1].set_title('Rounds to Convergence', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        for i, (p, r) in enumerate(zip(protocols, rounds)):
            axes[1].text(i, r, f'{r}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Final Loss
        axes[2].bar(protocols, final_loss, color=colors[:len(protocols)], alpha=0.8)
        axes[2].set_ylabel('Final Loss', fontweight='bold')
        axes[2].set_title('Final Loss Value', fontweight='bold')
        axes[2].grid(axis='y', alpha=0.3)
        for i, (p, l) in enumerate(zip(protocols, final_loss)):
            axes[2].text(i, l, f'{l:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle(f'Protocol Performance Comparison - {self.use_case.title()} ({scenario.title()} Network)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")
        else:
            plt.show()
    
    def generate_summary_table(self, scenario=None):
        """Print a summary table of all protocols for a specific scenario"""
        scenario = scenario or self.scenarios[0]
        
        if scenario not in self.results or not self.results[scenario]:
            print(f"No results loaded for scenario: {scenario}")
            return
        
        results = self.results[scenario]
        
        print(f"\n{'='*90}")
        print(f"Protocol Performance Summary - {self.use_case.title()} ({scenario.title()} Network)")
        print(f"{'='*90}")
        print(f"{'Protocol':<10} {'Rounds':<8} {'Time (s)':<12} {'Time (min)':<12} {'Final Loss':<12} {'Final MSE':<12}")
        print(f"{'-'*90}")
        
        for protocol in sorted(results.keys()):
            data = results[protocol]
            rounds = data.get('total_rounds', 0)
            time_s = data.get('convergence_time_seconds', 0)
            time_m = data.get('convergence_time_minutes', 0)
            loss_values = data.get('loss', [])
            mse_values = data.get('mse', [])
            final_loss = loss_values[-1] if loss_values else 0
            final_mse = mse_values[-1] if mse_values else 0
            
            print(f"{protocol.upper():<10} {rounds:<8} {time_s:<12.2f} {time_m:<12.2f} {final_loss:<12.6f} {final_mse:<12.6f}")
        
        print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare FL Protocol Performance")
    parser.add_argument("--use-case", "-u", 
                       choices=["emotion", "mentalstate", "temperature"],
                       default="temperature",
                       help="Use case to compare")
    parser.add_argument("--scenarios", "-s",
                       nargs="+",
                       choices=["excellent", "good", "moderate", "poor", "very_poor", "satellite"],
                       help="Network scenarios to compare (default: all available)")
    parser.add_argument("--output-dir", "-o",
                       default="comparison_results",
                       help="Directory to save comparison plots")
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = ProtocolComparator(use_case=args.use_case, scenarios=args.scenarios)
    
    # Load results
    print(f"\nLoading results for {args.use_case.title()}...\n")
    comparator.load_results()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate comparison for each loaded scenario
    for scenario in comparator.scenarios:
        if scenario in comparator.results and comparator.results[scenario]:
            print(f"\n{'='*80}")
            print(f"Generating comparison for {scenario.upper()} network scenario")
            print(f"{'='*80}\n")
            
            # Print summary table
            comparator.generate_summary_table(scenario)
            
            # Generate plots
            print(f"Generating comparison plots for {scenario}...\n")
            
            comparator.plot_convergence_comparison(
                scenario=scenario,
                save_path=f"{args.output_dir}/{args.use_case}_{scenario}_convergence_time.png"
            )
            
            comparator.plot_loss_comparison(
                scenario=scenario,
                save_path=f"{args.output_dir}/{args.use_case}_{scenario}_loss_comparison.png"
            )
            
            comparator.plot_summary_bars(
                scenario=scenario,
                save_path=f"{args.output_dir}/{args.use_case}_{scenario}_summary_comparison.png"
            )
            
            print(f"\n✓ All comparison plots for {scenario} saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
