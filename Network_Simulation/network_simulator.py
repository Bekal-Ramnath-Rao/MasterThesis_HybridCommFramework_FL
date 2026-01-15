#!/usr/bin/env python3
"""
Network Simulation Tool for Docker Containers
Applies network conditions (latency, bandwidth, packet loss, jitter) to running containers
"""

import subprocess
import argparse
import json
import time
from typing import Dict, List

class NetworkSimulator:
    """Simulates various network conditions on Docker containers using tc (traffic control)"""
    
    # Predefined network scenarios
    NETWORK_SCENARIOS = {
        "excellent": {
            "name": "Excellent Network (LAN)",
            "latency": "2ms",
            "jitter": "0.5ms",
            "bandwidth": "1000mbit",
            "loss": "0.01%"
        },
        "good": {
            "name": "Good Network (Broadband)",
            "latency": "10ms",
            "jitter": "2ms",
            "bandwidth": "100mbit",
            "loss": "0.1%"
        },
        "moderate": {
            "name": "Moderate Network (4G/LTE)",
            "latency": "50ms",
            "jitter": "10ms",
            "bandwidth": "20mbit",
            "loss": "1%"
        },
        "poor": {
            "name": "Poor Network (3G)",
            "latency": "100ms",
            "jitter": "30ms",
            "bandwidth": "2mbit",
            "loss": "3%"
        },
        "very_poor": {
            "name": "Very Poor Network (Edge/2G)",
            "latency": "300ms",
            "jitter": "100ms",
            "bandwidth": "384kbit",
            "loss": "5%"
        },
        "satellite": {
            "name": "Satellite Network",
            "latency": "600ms",
            "jitter": "50ms",
            "bandwidth": "5mbit",
            "loss": "2%"
        },
        "congested_light": {
            "name": "Light Congestion (Shared Network)",
            "latency": "30ms",
            "jitter": "15ms",
            "bandwidth": "10mbit",
            "loss": "1.5%"
        },
        "congested_moderate": {
            "name": "Moderate Congestion (Peak Hours)",
            "latency": "75ms",
            "jitter": "35ms",
            "bandwidth": "5mbit",
            "loss": "3.5%"
        },
        "congested_heavy": {
            "name": "Heavy Congestion (Network Overload)",
            "latency": "150ms",
            "jitter": "60ms",
            "bandwidth": "2mbit",
            "loss": "6%"
        }
    }
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def log(self, message):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def run_command(self, command: List[str], check=True) -> subprocess.CompletedProcess:
        """Execute a shell command"""
        self.log(f"Running: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=check)
        if result.returncode != 0 and check:
            print(f"[ERROR] Command failed: {' '.join(command)}")
            print(f"[ERROR] Output: {result.stderr}")
        return result
    
    def get_container_pid(self, container_name: str) -> str:
        """Get the PID of a running container"""
        result = self.run_command([
            "docker", "inspect", "-f", "{{.State.Pid}}", container_name
        ])
        return result.stdout.strip()
    
    def get_container_interface(self, container_name: str) -> str:
        """Get the network interface inside a container"""
        # Most containers use eth0 as the default interface
        return "eth0"
    
    def check_tc_available(self, container_name: str) -> bool:
        """Check if tc command is available in the container"""
        try:
            result = self.run_command([
                "docker", "exec", container_name,
                "sh", "-c", "command -v tc"
            ], check=False)
            return result.returncode == 0
        except Exception:
            return False
    
    def apply_network_conditions(self, container_name: str, conditions: Dict[str, str]):
        """Apply network conditions to a specific container"""
        try:
            print(f"\n{'='*60}")
            print(f"Applying network conditions to: {container_name}")
            print(f"{'='*60}")
            
            # Check if tc command is available
            if not self.check_tc_available(container_name):
                print(f"[WARNING] Container {container_name} does not have 'tc' command (iproute2 package)")
                print(f"[WARNING] Skipping network conditions for {container_name}")
                print(f"[INFO] To enable network simulation on this container, install iproute2 package")
                print(f"{'='*60}\n")
                return True  # Return True to not count as failure, just skipped
            
            # First, reset any existing tc rules
            self.reset_container_network(container_name)
            
            # Get container interface
            interface = self.get_container_interface(container_name)
            
            # Build tc-netem command
            netem_params = []
            
            if conditions.get("latency"):
                netem_params.append(f"delay {conditions['latency']}")
                if conditions.get("jitter"):
                    netem_params.append(conditions["jitter"])
            
            if conditions.get("loss"):
                netem_params.append(f"loss {conditions['loss']}")
            
            # Apply netem rules
            if netem_params:
                netem_cmd = " ".join(netem_params)
                self.run_command([
                    "docker", "exec", container_name,
                    "sh", "-c", f"tc qdisc add dev {interface} root netem {netem_cmd}"
                ])
                print(f"✓ Applied: {netem_cmd}")
            
            # Apply bandwidth limit if specified
            if conditions.get("bandwidth"):
                self.run_command([
                    "docker", "exec", container_name,
                    "sh", "-c", 
                    f"tc qdisc add dev {interface} root tbf rate {conditions['bandwidth']} "
                    f"burst 32kbit latency 400ms"
                ])
                print(f"✓ Applied bandwidth limit: {conditions['bandwidth']}")
            
            print(f"{'='*60}\n")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to apply network conditions to {container_name}: {e}")
            return False
    
    def reset_container_network(self, container_name: str):
        """Reset network conditions on a container"""
        try:
            interface = self.get_container_interface(container_name)
            # Delete existing tc rules (ignore errors if none exist)
            self.run_command([
                "docker", "exec", container_name,
                "sh", "-c", f"tc qdisc del dev {interface} root"
            ], check=False)
            self.log(f"Reset network conditions for {container_name}")
        except Exception as e:
            self.log(f"Note: Could not reset {container_name} (may not have existing rules): {e}")
    
    def apply_scenario_to_containers(self, scenario_name: str, container_pattern: str = None):
        """Apply a predefined scenario to all matching containers"""
        if scenario_name not in self.NETWORK_SCENARIOS:
            print(f"[ERROR] Unknown scenario: {scenario_name}")
            print(f"Available scenarios: {', '.join(self.NETWORK_SCENARIOS.keys())}")
            return
        
        scenario = self.NETWORK_SCENARIOS[scenario_name]
        print(f"\n{'#'*60}")
        print(f"# Applying Scenario: {scenario['name']}")
        print(f"#{'#'*59}")
        print(f"# Latency: {scenario.get('latency', 'N/A')}")
        print(f"# Jitter: {scenario.get('jitter', 'N/A')}")
        print(f"# Bandwidth: {scenario.get('bandwidth', 'N/A')}")
        print(f"# Packet Loss: {scenario.get('loss', 'N/A')}")
        print(f"{'#'*60}\n")
        
        # Get list of running containers
        result = self.run_command(["docker", "ps", "--format", "{{.Names}}"])
        containers = result.stdout.strip().split('\n')
        
        # Filter containers if pattern is provided
        if container_pattern:
            containers = [c for c in containers if container_pattern in c]
        
        if not containers:
            print("[WARNING] No matching containers found!")
            return
        
        print(f"Found {len(containers)} container(s) to configure:\n")
        
        # Apply conditions to each container
        success_count = 0
        for container in containers:
            if container:  # Skip empty lines
                if self.apply_network_conditions(container, scenario):
                    success_count += 1
                time.sleep(0.5)  # Small delay between containers
        
        print(f"\n{'='*60}")
        print(f"Successfully configured {success_count}/{len(containers)} containers")
        print(f"{'='*60}\n")
    
    def reset_all_containers(self, container_pattern: str = None):
        """Reset network conditions on all containers"""
        result = self.run_command(["docker", "ps", "--format", "{{.Names}}"])
        containers = result.stdout.strip().split('\n')
        
        if container_pattern:
            containers = [c for c in containers if container_pattern in c]
        
        print(f"\nResetting network conditions for {len(containers)} container(s)...")
        for container in containers:
            if container:
                self.reset_container_network(container)
        print("Done!\n")
    
    def show_scenarios(self):
        """Display all available network scenarios"""
        print("\n" + "="*70)
        print("Available Network Scenarios")
        print("="*70 + "\n")
        
        for key, scenario in self.NETWORK_SCENARIOS.items():
            print(f"Scenario: {key}")
            print(f"  Name: {scenario['name']}")
            print(f"  Latency: {scenario.get('latency', 'N/A')}")
            print(f"  Jitter: {scenario.get('jitter', 'N/A')}")
            print(f"  Bandwidth: {scenario.get('bandwidth', 'N/A')}")
            print(f"  Packet Loss: {scenario.get('loss', 'N/A')}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Network Simulation Tool for Docker Containers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply 'poor' network conditions to all FL containers
  python network_simulator.py --scenario poor --pattern fl-

  # Apply 'moderate' network to MQTT containers only
  python network_simulator.py --scenario moderate --pattern mqtt

  # Reset all containers
  python network_simulator.py --reset

  # Show available scenarios
  python network_simulator.py --list

  # Apply custom conditions
  python network_simulator.py --custom --latency 50ms --jitter 10ms --loss 2% --bandwidth 10mbit --pattern fl-client
        """
    )
    
    parser.add_argument("--scenario", "-s", 
                       help="Predefined network scenario to apply")
    parser.add_argument("--pattern", "-p", 
                       help="Container name pattern to match (e.g., 'fl-client', 'mqtt')")
    parser.add_argument("--reset", "-r", action="store_true",
                       help="Reset network conditions on containers")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List all available scenarios")
    parser.add_argument("--custom", "-c", action="store_true",
                       help="Apply custom network conditions")
    parser.add_argument("--latency", help="Custom latency (e.g., 100ms)")
    parser.add_argument("--jitter", help="Custom jitter (e.g., 20ms)")
    parser.add_argument("--bandwidth", help="Custom bandwidth (e.g., 10mbit)")
    parser.add_argument("--loss", help="Custom packet loss (e.g., 5%%)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    sim = NetworkSimulator(verbose=args.verbose)
    
    if args.list:
        sim.show_scenarios()
    elif args.reset:
        sim.reset_all_containers(args.pattern)
    elif args.custom:
        conditions = {}
        if args.latency:
            conditions["latency"] = args.latency
        if args.jitter:
            conditions["jitter"] = args.jitter
        if args.bandwidth:
            conditions["bandwidth"] = args.bandwidth
        if args.loss:
            conditions["loss"] = args.loss
        
        if not conditions:
            print("[ERROR] No custom conditions specified!")
            parser.print_help()
            return
        
        # Get containers
        result = sim.run_command(["docker", "ps", "--format", "{{.Names}}"])
        containers = result.stdout.strip().split('\n')
        
        if args.pattern:
            containers = [c for c in containers if args.pattern in c]
        
        print(f"\nApplying custom conditions to {len(containers)} container(s)...")
        for container in containers:
            if container:
                sim.apply_network_conditions(container, conditions)
    elif args.scenario:
        sim.apply_scenario_to_containers(args.scenario, args.pattern)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
