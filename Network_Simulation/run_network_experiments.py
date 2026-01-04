#!/usr/bin/env python3
"""
Automated Network Experiment Runner
Runs FL experiments across all protocols and network conditions
"""

import subprocess
import time
import json
import os
from datetime import datetime
from typing import List, Dict
import argparse


class ExperimentRunner:
    """Automates running FL experiments across different network conditions"""
    
    def __init__(self, use_case: str = "emotion", num_rounds: int = 10):
        self.use_case = use_case
        self.num_rounds = num_rounds
        self.results_dir = f"experiment_results/{use_case}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define protocols to test
        self.protocols = ["mqtt", "amqp", "grpc", "quic", "dds"]
        
        # Define network scenarios to test
        self.network_scenarios = [
            "excellent",
            "good", 
            "moderate",
            "poor",
            "very_poor",
            "satellite"
        ]
        
        # Docker compose file mapping
        self.compose_files = {
            "emotion": "docker-compose-emotion.yml",
            "mentalstate": "docker-compose-mentalstate.yml",
            "temperature": "docker-compose-temperature.yml"
        }
        
        # Service name patterns
        self.service_patterns = {
            "emotion": {
                "mqtt": ["mqtt-broker", "fl-server-mqtt-emotion", "fl-client-mqtt-emotion-1", "fl-client-mqtt-emotion-2"],
                "amqp": ["rabbitmq", "fl-server-amqp-emotion", "fl-client-amqp-emotion-1", "fl-client-amqp-emotion-2"],
                "grpc": ["fl-server-grpc-emotion", "fl-client-grpc-emotion-1", "fl-client-grpc-emotion-2"],
                "quic": ["fl-server-quic-emotion", "fl-client-quic-emotion-1", "fl-client-quic-emotion-2"],
                "dds": ["fl-server-dds-emotion", "fl-client-dds-emotion-1", "fl-client-dds-emotion-2"]
            },
            "mentalstate": {
                "mqtt": ["mqtt-broker-mental", "fl-server-mqtt-mental", "fl-client-mqtt-mental-1", "fl-client-mqtt-mental-2"],
                "amqp": ["rabbitmq-mental", "fl-server-amqp-mental", "fl-client-amqp-mental-1", "fl-client-amqp-mental-2"],
                "grpc": ["fl-server-grpc-mental", "fl-client-grpc-mental-1", "fl-client-grpc-mental-2"],
                "quic": ["fl-server-quic-mental", "fl-client-quic-mental-1", "fl-client-quic-mental-2"],
                "dds": ["fl-server-dds-mental", "fl-client-dds-mental-1", "fl-client-dds-mental-2"]
            },
            "temperature": {
                "mqtt": ["mqtt-broker-temp", "fl-server-mqtt-temp", "fl-client-mqtt-temp-1", "fl-client-mqtt-temp-2"],
                "amqp": ["rabbitmq-temp", "fl-server-amqp-temp", "fl-client-amqp-temp-1", "fl-client-amqp-temp-2"],
                "grpc": ["fl-server-grpc-temp", "fl-client-grpc-temp-1", "fl-client-grpc-temp-2"],
                "quic": ["fl-server-quic-temp", "fl-client-quic-temp-1", "fl-client-quic-temp-2"],
                "dds": ["fl-server-dds-temp", "fl-client-dds-temp-1", "fl-client-dds-temp-2"]
            }
        }
    
    def run_command(self, command: List[str], check=True, env_vars: Dict[str, str] = None) -> subprocess.CompletedProcess:
        """Execute a shell command with optional environment variables"""
        print(f"[CMD] {' '.join(command)}")
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)
            print(f"[ENV] {env_vars}")
        return subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', check=check, env=env)
    
    def start_containers(self, protocol: str, scenario: str = "excellent"):
        """Start Docker containers for a specific protocol with staged startup"""
        compose_file = self.compose_files[self.use_case]
        services = self.service_patterns[self.use_case][protocol]
        
        print(f"\n{'='*70}")
        print(f"Starting containers for {protocol.upper()} protocol...")
        print(f"Network Scenario: {scenario.upper()}")
        print(f"{'='*70}")
        
        # Separate services into broker, server, and clients
        broker = None
        server = None
        clients = []
        
        for service in services:
            if 'broker' in service.lower() or 'rabbitmq' in service.lower():
                broker = service
            elif 'server' in service.lower():
                server = service
            elif 'client' in service.lower():
                clients.append(service)
        
        # Stage 1: Start broker first (if exists)
        if broker:
            print(f"\n[Stage 1/3] Starting broker: {broker}")
            cmd_broker = ["docker-compose", "-f", compose_file, "up", "-d", broker]
            result = self.run_command(cmd_broker)
            if result.returncode != 0:
                print(f"[ERROR] Failed to start broker")
                return False
            
            print(f"Waiting 10 seconds for broker to initialize...")
            time.sleep(10)
        
        # Stage 2: Start server
        if server:
            print(f"\n[Stage 2/3] Starting server: {server}")
            cmd_server = ["docker-compose", "-f", compose_file, "up", "-d", server]
            result = self.run_command(cmd_server)
            if result.returncode != 0:
                print(f"[ERROR] Failed to start server")
                return False
            
            print(f"Waiting 8 seconds for server to initialize...")
            time.sleep(8)
        
        # Stage 3: Start clients
        if clients:
            print(f"\n[Stage 3/3] Starting clients: {', '.join(clients)}")
            cmd_clients = ["docker-compose", "-f", compose_file, "up", "-d"] + clients
            result = self.run_command(cmd_clients)
            if result.returncode != 0:
                print(f"[ERROR] Failed to start clients")
                return False
            
            print(f"Waiting 5 seconds for clients to connect...")
            time.sleep(5)
        
        print(f"\n✓ All containers started successfully with staged delays")
        return True
    
    def stop_containers(self, protocol: str):
        """Stop Docker containers for a specific protocol"""
        compose_file = self.compose_files[self.use_case]
        services = self.service_patterns[self.use_case][protocol]
        
        print(f"\nStopping containers for {protocol.upper()} protocol...")
        
        cmd = ["docker-compose", "-f", compose_file, "down"]
        self.run_command(cmd, check=False)
        
        # Give time for containers to fully stop
        time.sleep(5)
    
    def apply_network_scenario(self, scenario: str, protocol: str):
        """Apply network conditions using network_simulator.py"""
        print(f"\n{'='*70}")
        print(f"Applying network scenario: {scenario.upper()}")
        print(f"{'='*70}")
        
        # Get actual container names for this protocol
        services = self.service_patterns[self.use_case][protocol]
        
        # Apply network conditions to each container individually
        from network_simulator import NetworkSimulator
        sim = NetworkSimulator(verbose=True)
        
        if scenario not in sim.NETWORK_SCENARIOS:
            print(f"[WARNING] Unknown scenario: {scenario}, skipping network simulation")
            return True  # Don't fail the experiment
        
        conditions = sim.NETWORK_SCENARIOS[scenario]
        
        # Apply to all containers in this protocol
        success_count = 0
        for container in services:
            try:
                if sim.apply_network_conditions(container, conditions):
                    success_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to apply conditions to {container}: {e}")
        
        print(f"\nApplied network conditions to {success_count}/{len(services)} containers")
        time.sleep(2)
        return success_count > 0  # At least one container should succeed
    
    def wait_for_completion(self, protocol: str, timeout: int = 3600):
        """Wait for FL training to complete"""
        print(f"\nWaiting for {protocol.upper()} training to complete (timeout: {timeout}s)...")
        
        # Get server container name
        services = self.service_patterns[self.use_case][protocol]
        server_container = [s for s in services if "server" in s][0]
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if server container is still running
            result = self.run_command([
                "docker", "ps", "--filter", f"name={server_container}", "--format", "{{.Names}}"
            ], check=False)
            
            if server_container not in result.stdout:
                print(f"Server container stopped. Training complete!")
                return True
            
            # Check logs for completion indicators
            logs = self.run_command([
                "docker", "logs", "--tail", "50", server_container
            ], check=False)
            
            if "Training completed" in logs.stdout or "All rounds completed" in logs.stdout:
                print(f"Training completed successfully!")
                time.sleep(5)  # Give time for final results to be written
                return True
            
            time.sleep(10)  # Check every 10 seconds
        
        print(f"[WARNING] Training timed out after {timeout}s")
        return False
    
    def collect_results(self, protocol: str, scenario: str):
        """Collect and save experiment results"""
        print(f"\nCollecting results for {protocol.upper()} - {scenario}...")
        
        # Create directory for this experiment
        exp_dir = os.path.join(self.results_dir, f"{protocol}_{scenario}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Get server container logs
        services = self.service_patterns[self.use_case][protocol]
        server_container = [s for s in services if "server" in s][0]
        
        logs = self.run_command([
            "docker", "logs", server_container
        ], check=False)
        
        # Save logs
        with open(os.path.join(exp_dir, "server_logs.txt"), "w", encoding='utf-8', errors='replace') as f:
            f.write(logs.stdout or "")
            f.write("\n\n=== STDERR ===\n\n")
            f.write(logs.stderr or "")
        
        # Copy result files if they exist
        # Try to copy from container
        result_files = [
            f"{protocol}_training_results.json",
            f"{protocol}_training_results.csv",
            "fl_results.json"
        ]
        
        for result_file in result_files:
            try:
                # Try to copy from server container
                self.run_command([
                    "docker", "cp",
                    f"{server_container}:/app/Server/{self.use_case.title()}_Recognition/results/{result_file}",
                    os.path.join(exp_dir, result_file)
                ], check=False)
            except:
                pass
        
        # Save experiment metadata
        metadata = {
            "protocol": protocol,
            "scenario": scenario,
            "use_case": self.use_case,
            "num_rounds": self.num_rounds,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results saved to: {exp_dir}")
    
    def run_single_experiment(self, protocol: str, scenario: str):
        """Run a single experiment with specific protocol and network scenario"""
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {protocol.upper()} - {scenario.upper()}")
        print(f"# Use Case: {self.use_case.title()}")
        print(f"# Rounds: {self.num_rounds}")
        print(f"{'#'*70}\n")
        
        try:
            # 1. Start containers
            if not self.start_containers(protocol,scenario):
                print(f"[ERROR] Failed to start containers for {protocol}")
                return False
            
            # 2. Apply network scenario
            if not self.apply_network_scenario(scenario, protocol):
                print(f"[WARNING] Failed to apply network scenario {scenario}, continuing anyway...")
            
            # 3. Wait for completion
            if not self.wait_for_completion(protocol, timeout=3600):
                print(f"[WARNING] Experiment may not have completed")
            
            # 4. Collect results
            self.collect_results(protocol, scenario)
            
            # 5. Stop containers
            self.stop_containers(protocol)
            
            print(f"\n✓ Experiment completed: {protocol.upper()} - {scenario.upper()}\n")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {e}\n")
            self.stop_containers(protocol)
            return False
    
    def run_all_experiments(self, protocols: List[str] = None, scenarios: List[str] = None):
        """Run experiments for all protocol and network combinations"""
        protocols = protocols or self.protocols
        scenarios = scenarios or self.network_scenarios
        
        total_experiments = len(protocols) * len(scenarios)
        current = 0
        
        print(f"\n{'='*70}")
        print(f"STARTING AUTOMATED EXPERIMENTS")
        print(f"{'='*70}")
        print(f"Use Case: {self.use_case.title()}")
        print(f"Protocols: {', '.join(protocols)}")
        print(f"Network Scenarios: {', '.join(scenarios)}")
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
                
                if self.run_single_experiment(protocol, scenario):
                    successful += 1
                else:
                    failed += 1
                
                # Brief pause between experiments
                time.sleep(5)
        
        # Summary
        duration = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"EXPERIMENTS COMPLETED")
        print(f"{'='*70}")
        print(f"Total: {total_experiments}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration/60:.2f} minutes")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automated Network Experiment Runner for FL",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--use-case", "-u", 
                       choices=["emotion", "mentalstate", "temperature"],
                       default="emotion",
                       help="Use case to run experiments for")
    parser.add_argument("--protocols", "-p", 
                       nargs="+",
                       choices=["mqtt", "amqp", "grpc", "quic", "dds"],
                       help="Specific protocols to test (default: all)")
    parser.add_argument("--scenarios", "-s",
                       nargs="+",
                       choices=["excellent", "good", "moderate", "poor", "very_poor", "satellite"],
                       help="Specific network scenarios to test (default: all)")
    parser.add_argument("--rounds", "-r",
                       type=int,
                       default=10,
                       help="Number of FL rounds (default: 10)")
    parser.add_argument("--single", action="store_true",
                       help="Run single experiment (requires --protocol and --scenario)")
    parser.add_argument("--protocol",
                       choices=["mqtt", "amqp", "grpc", "quic", "dds"],
                       help="Protocol for single experiment")
    parser.add_argument("--scenario",
                       choices=["excellent", "good", "moderate", "poor", "very_poor", "satellite"],
                       help="Network scenario for single experiment")
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(use_case=args.use_case, num_rounds=args.rounds)
    
    if args.single:
        if not args.protocol or not args.scenario:
            print("[ERROR] --single requires both --protocol and --scenario")
            parser.print_help()
            return
        runner.run_single_experiment(args.protocol, args.scenario)
    else:
        runner.run_all_experiments(
            protocols=args.protocols,
            scenarios=args.scenarios
        )


if __name__ == "__main__":
    main()
