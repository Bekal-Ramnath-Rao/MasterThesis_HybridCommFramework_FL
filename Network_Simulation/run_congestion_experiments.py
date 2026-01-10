#!/usr/bin/env python3
"""
Congestion Experiment Runner
Runs FL experiments with active network congestion from traffic generators
Combines network_simulator.py (tc-based simulation) with congestion_manager.py (traffic generators)
"""

import subprocess
import time
import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List
import sys

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from Network_Simulation.congestion_manager import CongestionManager
from Network_Simulation.run_network_experiments import ExperimentRunner


class CongestionExperimentRunner(ExperimentRunner):
    """Extended experiment runner with congestion support"""
    
    def __init__(self, use_case: str = "temperature", num_rounds: int = 10):
        super().__init__(use_case, num_rounds)
        self.congestion_manager = CongestionManager(use_case=use_case, verbose=True)
        
        # Define congestion test scenarios
        # Each scenario combines network conditions + traffic load
        self.congestion_scenarios = {
            "baseline": {
                "network": "good",
                "congestion": "none",
                "description": "Baseline - Good network, no congestion"
            },
            "light_load": {
                "network": "good",
                "congestion": "light",
                "description": "Good network with light background traffic"
            },
            "moderate_load": {
                "network": "moderate",
                "congestion": "moderate",
                "description": "Moderate network with moderate background traffic"
            },
            "heavy_load": {
                "network": "moderate",
                "congestion": "heavy",
                "description": "Moderate network with heavy background traffic"
            },
            "extreme_load": {
                "network": "poor",
                "congestion": "extreme",
                "description": "Poor network with extreme background traffic"
            },
            "congestion_only": {
                "network": "excellent",
                "congestion": "heavy",
                "description": "Excellent network with heavy congestion (isolates congestion effect)"
            },
            "degraded_congested": {
                "network": "congested_moderate",
                "congestion": "moderate",
                "description": "Using congested_moderate scenario + traffic generators"
            }
        }
    
    def run_congestion_experiment(self, protocol: str, scenario_name: str):
        """Run experiment with both network simulation and traffic congestion"""
        if scenario_name not in self.congestion_scenarios:
            print(f"[ERROR] Unknown congestion scenario: {scenario_name}")
            return False
        
        scenario = self.congestion_scenarios[scenario_name]
        network_condition = scenario["network"]
        congestion_level = scenario["congestion"]
        
        print(f"\n{'#'*70}")
        print(f"# CONGESTION EXPERIMENT: {protocol.upper()} - {scenario_name.upper()}")
        print(f"{'#'*70}")
        print(f"# Use Case: {self.use_case.title()}")
        print(f"# Network Condition: {network_condition}")
        print(f"# Congestion Level: {congestion_level}")
        print(f"# Description: {scenario['description']}")
        print(f"# Rounds: {self.num_rounds}")
        print(f"{'#'*70}\n")
        
        try:
            # 1. Start FL containers
            print("\n[Step 1/5] Starting FL containers...")
            if not self.start_containers(protocol, network_condition):
                print(f"[ERROR] Failed to start containers for {protocol}")
                return False
            
            # 2. Apply network conditions
            print(f"\n[Step 2/5] Applying network conditions ({network_condition})...")
            if not self.apply_network_scenario(network_condition, protocol):
                print(f"[WARNING] Failed to apply network scenario, continuing...")
            
            # 3. Start traffic generators
            print(f"\n[Step 3/5] Starting traffic generators ({congestion_level})...")
            if congestion_level != "none":
                if not self.congestion_manager.start_traffic_generators(congestion_level):
                    print("[WARNING] Failed to start traffic generators, continuing...")
                time.sleep(5)  # Let traffic generators stabilize
            else:
                print("No traffic generators needed for this scenario")
            
            # 4. Wait for FL training to complete
            print("\n[Step 4/5] Running FL training...")
            if not self.wait_for_completion(protocol, timeout=3600):
                print(f"[WARNING] Experiment may not have completed")
            
            # 5. Collect results
            print("\n[Step 5/5] Collecting results...")
            self.collect_results(protocol, f"{scenario_name}_{network_condition}")
            
            print(f"\n✓ Congestion experiment completed: {protocol.upper()} - {scenario_name.upper()}\n")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {e}\n")
            return False
            
        finally:
            # Cleanup
            print("\nCleaning up...")
            self.congestion_manager.stop_traffic_generators()
            self.stop_containers(protocol)
            time.sleep(3)
    
    def run_all_congestion_experiments(self, protocols: List[str] = None, 
                                     scenarios: List[str] = None):
        """Run congestion experiments for all protocol and scenario combinations"""
        protocols = protocols or self.protocols
        scenarios = scenarios or list(self.congestion_scenarios.keys())
        
        total_experiments = len(protocols) * len(scenarios)
        current = 0
        
        print(f"\n{'='*70}")
        print(f"STARTING CONGESTION EXPERIMENTS")
        print(f"{'='*70}")
        print(f"Use Case: {self.use_case.title()}")
        print(f"Protocols: {', '.join(protocols)}")
        print(f"Congestion Scenarios: {', '.join(scenarios)}")
        print(f"Total Experiments: {total_experiments}")
        print(f"Results Directory: {self.results_dir}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        for protocol in protocols:
            for scenario in scenarios:
                current += 1
                print(f"\n{'*'*70}")
                print(f"Progress: {current}/{total_experiments}")
                print(f"{'*'*70}")
                
                if self.run_congestion_experiment(protocol, scenario):
                    successful += 1
                else:
                    failed += 1
                
                # Brief pause between experiments
                time.sleep(10)
        
        # Final cleanup
        print("\nFinal cleanup...")
        self.congestion_manager.stop_traffic_generators()
        
        # Summary
        duration = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"CONGESTION EXPERIMENTS COMPLETED")
        print(f"{'='*70}")
        print(f"Total: {total_experiments}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration/60:.2f} minutes")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*70}\n")
    
    def show_scenarios(self):
        """Display all available congestion scenarios"""
        print("\n" + "="*70)
        print("Available Congestion Test Scenarios")
        print("="*70 + "\n")
        
        for name, config in self.congestion_scenarios.items():
            print(f"Scenario: {name}")
            print(f"  Network: {config['network']}")
            print(f"  Congestion: {config['congestion']}")
            print(f"  Description: {config['description']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Congestion Experiment Runner for FL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all protocols under moderate load
  python run_congestion_experiments.py --use-case temperature --scenario moderate_load

  # Run specific protocol with extreme load
  python run_congestion_experiments.py --use-case temperature --protocols mqtt --scenario extreme_load

  # Run all congestion scenarios for specific protocols
  python run_congestion_experiments.py --use-case temperature --protocols mqtt grpc --all-scenarios

  # List available scenarios
  python run_congestion_experiments.py --list
        """
    )
    
    parser.add_argument("--use-case", "-u",
                       choices=["emotion", "mentalstate", "temperature"],
                       default="temperature",
                       help="Use case to run experiments for")
    parser.add_argument("--protocols", "-p",
                       nargs="+",
                       choices=["mqtt", "amqp", "grpc", "quic", "dds"],
                       help="Specific protocols to test (default: all)")
    parser.add_argument("--scenario", "-s",
                       help="Specific congestion scenario to test")
    parser.add_argument("--all-scenarios", action="store_true",
                       help="Run all congestion scenarios")
    parser.add_argument("--rounds", "-r",
                       type=int,
                       default=10,
                       help="Number of FL rounds (default: 10)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available congestion scenarios")
    
    args = parser.parse_args()
    
    runner = CongestionExperimentRunner(use_case=args.use_case, num_rounds=args.rounds)
    
    if args.list:
        runner.show_scenarios()
        runner.congestion_manager.show_levels()
    elif args.all_scenarios:
        runner.run_all_congestion_experiments(protocols=args.protocols)
    elif args.scenario:
        protocols = args.protocols or runner.protocols
        for protocol in protocols:
            runner.run_congestion_experiment(protocol, args.scenario)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
