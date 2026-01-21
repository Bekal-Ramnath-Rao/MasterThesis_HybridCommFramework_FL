"""
Dynamic Network Controller for Runtime Network Condition Changes

This module allows changing network conditions (latency, bandwidth, packet loss)
dynamically while Docker containers are running, simulating mobility scenarios.
"""

import subprocess
import time
import os
from typing import Dict, Optional
import json


class DynamicNetworkController:
    """
    Controller for dynamically changing network conditions in Docker containers
    Simulates mobility by varying network parameters at runtime
    """
    
    # Network scenario presets
    NETWORK_SCENARIOS = {
        'excellent': {
            'latency_ms': 5,
            'jitter_ms': 1,
            'bandwidth_mbps': 100,
            'packet_loss': 0.0
        },
        'good': {
            'latency_ms': 20,
            'jitter_ms': 5,
            'bandwidth_mbps': 50,
            'packet_loss': 0.1
        },
        'moderate': {
            'latency_ms': 50,
            'jitter_ms': 10,
            'bandwidth_mbps': 20,
            'packet_loss': 0.5
        },
        'poor': {
            'latency_ms': 150,
            'jitter_ms': 30,
            'bandwidth_mbps': 5,
            'packet_loss': 1.0
        },
        'very_poor': {
            'latency_ms': 300,
            'jitter_ms': 50,
            'bandwidth_mbps': 1,
            'packet_loss': 3.0
        },
        'satellite': {
            'latency_ms': 500,
            'jitter_ms': 100,
            'bandwidth_mbps': 10,
            'packet_loss': 0.5
        }
    }
    
    # Mobility patterns
    MOBILITY_PATTERNS = {
        'static': ['excellent', 'excellent', 'excellent'],
        'low': ['excellent', 'good', 'excellent'],
        'medium': ['good', 'moderate', 'good'],
        'high': ['good', 'poor', 'moderate', 'excellent']
    }
    
    def __init__(self, container_prefix: str = "fl-client"):
        """
        Initialize Dynamic Network Controller
        
        Args:
            container_prefix: Prefix for Docker container names to apply changes
        """
        self.container_prefix = container_prefix
        self.current_scenario = 'moderate'
        self.mobility_pattern = None
        self.mobility_index = 0
        
    def get_container_names(self) -> list:
        """Get list of container names matching prefix"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                check=True
            )
            containers = [
                name for name in result.stdout.strip().split('\n')
                if self.container_prefix in name
            ]
            return containers
        except subprocess.CalledProcessError as e:
            print(f"[Network Controller] Error getting container names: {e}")
            return []
    
    def apply_network_condition(
        self,
        container_name: str,
        latency_ms: float,
        jitter_ms: float,
        bandwidth_mbps: float,
        packet_loss: float
    ) -> bool:
        """
        Apply network conditions to a specific container using tc (traffic control)
        
        Args:
            container_name: Name of Docker container
            latency_ms: Network latency in milliseconds
            jitter_ms: Jitter in milliseconds
            bandwidth_mbps: Bandwidth limit in Mbps
            packet_loss: Packet loss percentage
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear existing tc rules
            clear_cmd = [
                'docker', 'exec', container_name,
                'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'
            ]
            subprocess.run(clear_cmd, capture_output=True, stderr=subprocess.DEVNULL)
            
            # Convert bandwidth to kbps
            bandwidth_kbps = int(bandwidth_mbps * 1000)
            
            # Apply new tc rules with tbf (Token Bucket Filter) for rate limiting
            # and netem for latency, jitter, and packet loss
            tc_cmd = [
                'docker', 'exec', container_name,
                'tc', 'qdisc', 'add', 'dev', 'eth0', 'root', 'handle', '1:',
                'tbf', 'rate', f'{bandwidth_kbps}kbit',
                'burst', '32kbit',
                'latency', '400ms'
            ]
            subprocess.run(tc_cmd, capture_output=True, text=True, check=True)
            
            # Add netem for latency, jitter, and packet loss
            netem_cmd = [
                'docker', 'exec', container_name,
                'tc', 'qdisc', 'add', 'dev', 'eth0', 'parent', '1:1', 'handle', '10:',
                'netem',
                'delay', f'{latency_ms}ms', f'{jitter_ms}ms',
                'loss', f'{packet_loss}%'
            ]
            subprocess.run(netem_cmd, capture_output=True, text=True, check=True)
            
            print(f"[Network Controller] Applied to {container_name}: "
                  f"latency={latency_ms}ms, jitter={jitter_ms}ms, "
                  f"bandwidth={bandwidth_mbps}Mbps, loss={packet_loss}%")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[Network Controller] Error applying conditions to {container_name}: {e}")
            return False
    
    def apply_scenario(self, scenario: str, containers: Optional[list] = None) -> bool:
        """
        Apply a predefined network scenario to containers
        
        Args:
            scenario: Name of network scenario (e.g., 'excellent', 'poor')
            containers: List of container names (None = apply to all matching prefix)
            
        Returns:
            True if successful, False otherwise
        """
        if scenario not in self.NETWORK_SCENARIOS:
            print(f"[Network Controller] Unknown scenario: {scenario}")
            return False
        
        params = self.NETWORK_SCENARIOS[scenario]
        
        # Get containers to apply to
        if containers is None:
            containers = self.get_container_names()
        
        if not containers:
            print("[Network Controller] No containers found")
            return False
        
        print(f"\n[Network Controller] Applying '{scenario}' scenario to {len(containers)} containers")
        
        success = True
        for container in containers:
            result = self.apply_network_condition(
                container,
                params['latency_ms'],
                params['jitter_ms'],
                params['bandwidth_mbps'],
                params['packet_loss']
            )
            success = success and result
        
        if success:
            self.current_scenario = scenario
        
        return success
    
    def clear_network_conditions(self, containers: Optional[list] = None) -> bool:
        """
        Clear all network conditions (restore to normal)
        
        Args:
            containers: List of container names (None = apply to all matching prefix)
            
        Returns:
            True if successful, False otherwise
        """
        if containers is None:
            containers = self.get_container_names()
        
        if not containers:
            print("[Network Controller] No containers found")
            return False
        
        print(f"\n[Network Controller] Clearing network conditions from {len(containers)} containers")
        
        success = True
        for container in containers:
            try:
                clear_cmd = [
                    'docker', 'exec', container,
                    'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'
                ]
                subprocess.run(clear_cmd, capture_output=True, check=True)
                print(f"[Network Controller] Cleared conditions from {container}")
            except subprocess.CalledProcessError:
                # Already cleared or no rules
                pass
        
        self.current_scenario = 'excellent'
        return success
    
    def simulate_mobility(
        self,
        pattern: str,
        interval_seconds: int = 30,
        containers: Optional[list] = None
    ):
        """
        Simulate mobility by changing network conditions over time
        
        Args:
            pattern: Mobility pattern ('static', 'low', 'medium', 'high')
            interval_seconds: Time between condition changes
            containers: List of container names (None = apply to all matching prefix)
        """
        if pattern not in self.MOBILITY_PATTERNS:
            print(f"[Network Controller] Unknown mobility pattern: {pattern}")
            return
        
        self.mobility_pattern = self.MOBILITY_PATTERNS[pattern]
        self.mobility_index = 0
        
        print(f"\n[Network Controller] Starting mobility simulation: {pattern}")
        print(f"[Network Controller] Pattern: {self.mobility_pattern}")
        print(f"[Network Controller] Interval: {interval_seconds}s")
        
        if containers is None:
            containers = self.get_container_names()
        
        try:
            while True:
                scenario = self.mobility_pattern[self.mobility_index]
                print(f"\n[Mobility] Step {self.mobility_index + 1}: Switching to '{scenario}'")
                
                self.apply_scenario(scenario, containers)
                
                # Move to next scenario in pattern
                self.mobility_index = (self.mobility_index + 1) % len(self.mobility_pattern)
                
                # Wait before next change
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n[Network Controller] Mobility simulation stopped")
            self.clear_network_conditions(containers)
    
    def get_current_scenario(self) -> str:
        """Get current network scenario"""
        return self.current_scenario
    
    def get_scenario_params(self, scenario: str) -> Dict:
        """Get parameters for a specific scenario"""
        return self.NETWORK_SCENARIOS.get(scenario, {})
    
    def custom_condition(
        self,
        latency_ms: float = 50,
        jitter_ms: float = 10,
        bandwidth_mbps: float = 20,
        packet_loss: float = 0.5,
        containers: Optional[list] = None
    ) -> bool:
        """
        Apply custom network conditions
        
        Args:
            latency_ms: Network latency in milliseconds
            jitter_ms: Jitter in milliseconds
            bandwidth_mbps: Bandwidth limit in Mbps
            packet_loss: Packet loss percentage
            containers: List of container names (None = apply to all matching prefix)
            
        Returns:
            True if successful, False otherwise
        """
        if containers is None:
            containers = self.get_container_names()
        
        if not containers:
            print("[Network Controller] No containers found")
            return False
        
        print(f"\n[Network Controller] Applying custom conditions to {len(containers)} containers")
        
        success = True
        for container in containers:
            result = self.apply_network_condition(
                container, latency_ms, jitter_ms, bandwidth_mbps, packet_loss
            )
            success = success and result
        
        return success


def main():
    """Command-line interface for network controller"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dynamic Network Controller for FL Containers")
    parser.add_argument('--prefix', default='fl-client', help='Container name prefix')
    parser.add_argument('--scenario', choices=list(DynamicNetworkController.NETWORK_SCENARIOS.keys()),
                       help='Apply network scenario')
    parser.add_argument('--clear', action='store_true', help='Clear network conditions')
    parser.add_argument('--mobility', choices=list(DynamicNetworkController.MOBILITY_PATTERNS.keys()),
                       help='Simulate mobility pattern')
    parser.add_argument('--interval', type=int, default=30,
                       help='Interval for mobility changes (seconds)')
    parser.add_argument('--custom', action='store_true', help='Apply custom conditions')
    parser.add_argument('--latency', type=float, default=50, help='Latency in ms')
    parser.add_argument('--jitter', type=float, default=10, help='Jitter in ms')
    parser.add_argument('--bandwidth', type=float, default=20, help='Bandwidth in Mbps')
    parser.add_argument('--loss', type=float, default=0.5, help='Packet loss percentage')
    
    args = parser.parse_args()
    
    controller = DynamicNetworkController(container_prefix=args.prefix)
    
    if args.clear:
        controller.clear_network_conditions()
    elif args.scenario:
        controller.apply_scenario(args.scenario)
    elif args.mobility:
        controller.simulate_mobility(args.mobility, args.interval)
    elif args.custom:
        controller.custom_condition(
            args.latency, args.jitter, args.bandwidth, args.loss
        )
    else:
        print("Please specify --scenario, --clear, --mobility, or --custom")
        parser.print_help()


if __name__ == "__main__":
    main()
