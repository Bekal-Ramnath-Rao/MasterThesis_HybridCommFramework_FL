#!/usr/bin/env python3
"""
Network Condition Manager for Federated Learning
Applies network conditions for both Docker and local execution
Supports explicit network condition configuration via environment variables
"""

import os
import subprocess
import socket
import sys
from typing import Dict, Optional
from enum import Enum


class ExecutionMode(Enum):
    """Execution environment"""
    DOCKER = "docker"
    LOCAL = "local"


class NetworkCondition(Enum):
    """Predefined network conditions"""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    POOR = "poor"
    VERY_POOR = "very_poor"
    SATELLITE = "satellite"
    NONE = "none"


class NetworkConditionManager:
    """
    Manages network conditions for FL experiments
    Works for both Docker containers and local execution
    """
    
    # Network condition parameters
    CONDITIONS = {
        "excellent": {
            "name": "Excellent Network (LAN)",
            "latency_ms": 2,
            "jitter_ms": 0.5,
            "bandwidth_mbps": 1000,
            "loss_percent": 0.01
        },
        "good": {
            "name": "Good Network (Broadband)",
            "latency_ms": 10,
            "jitter_ms": 2,
            "bandwidth_mbps": 100,
            "loss_percent": 0.1
        },
        "moderate": {
            "name": "Moderate Network (4G/LTE)",
            "latency_ms": 50,
            "jitter_ms": 10,
            "bandwidth_mbps": 20,
            "loss_percent": 1.0
        },
        "poor": {
            "name": "Poor Network (3G)",
            "latency_ms": 100,
            "jitter_ms": 30,
            "bandwidth_mbps": 2,
            "loss_percent": 3.0
        },
        "very_poor": {
            "name": "Very Poor Network (Edge/2G)",
            "latency_ms": 300,
            "jitter_ms": 100,
            "bandwidth_mbps": 0.384,  # 384 kbps
            "loss_percent": 5.0
        },
        "satellite": {
            "name": "Satellite Network",
            "latency_ms": 600,
            "jitter_ms": 50,
            "bandwidth_mbps": 5,
            "loss_percent": 2.0
        },
        "none": {
            "name": "No Network Conditions (Default)",
            "latency_ms": 0,
            "jitter_ms": 0,
            "bandwidth_mbps": 0,  # 0 means unlimited
            "loss_percent": 0
        }
    }
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.execution_mode = self._detect_execution_mode()
        self.interface = self._detect_network_interface()
        self.applied = False
        
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled"""
        if self.verbose or level == "ERROR":
            print(f"[{level}] {message}")
    
    def _detect_execution_mode(self) -> ExecutionMode:
        """Detect if running in Docker or local"""
        # Check if running in Docker container
        if os.path.exists('/.dockerenv'):
            return ExecutionMode.DOCKER
        
        # Check cgroup for docker
        try:
            with open('/proc/1/cgroup', 'rt') as f:
                if 'docker' in f.read():
                    return ExecutionMode.DOCKER
        except Exception:
            pass
        
        return ExecutionMode.LOCAL
    
    def _detect_network_interface(self) -> str:
        """Detect the network interface to apply conditions to"""
        if self.execution_mode == ExecutionMode.DOCKER:
            return "eth0"  # Default Docker interface
        else:
            # For local execution, use loopback interface
            return "lo"
    
    def _check_tc_available(self) -> bool:
        """Check if tc (traffic control) command is available"""
        try:
            result = subprocess.run(
                ["which", "tc"],
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _run_tc_command(self, command: list, check: bool = True) -> bool:
        """Run tc command with proper permissions"""
        try:
            full_cmd = ["sudo", "tc"] + command
            if self.verbose:
                self.log(f"Running: {' '.join(full_cmd)}")
            
            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.returncode != 0 and check:
                self.log(f"Command failed: {result.stderr}", "ERROR")
                return False
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"TC command failed: {e}", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error running tc: {e}", "ERROR")
            return False
    
    def reset_network_conditions(self) -> bool:
        """Reset any applied network conditions"""
        if not self._check_tc_available():
            self.log("tc command not available. Install iproute2 package.", "ERROR")
            return False
        
        self.log(f"Resetting network conditions on {self.interface}...")
        
        # Delete existing qdisc (ignore errors if none exist)
        self._run_tc_command(
            ["qdisc", "del", "dev", self.interface, "root"],
            check=False
        )
        
        self.applied = False
        self.log("Network conditions reset successfully")
        return True
    
    def apply_network_condition(self, condition_name: str) -> bool:
        """
        Apply predefined network condition
        
        Args:
            condition_name: Name of the condition (excellent, good, moderate, poor, very_poor, satellite, none)
        
        Returns:
            bool: True if applied successfully, False otherwise
        """
        if condition_name.lower() == "none":
            return self.reset_network_conditions()
        
        if condition_name not in self.CONDITIONS:
            self.log(f"Unknown condition: {condition_name}", "ERROR")
            self.log(f"Available: {', '.join(self.CONDITIONS.keys())}", "INFO")
            return False
        
        if not self._check_tc_available():
            self.log("tc command not available. Install: sudo apt-get install iproute2", "ERROR")
            return False
        
        # Reset any existing conditions first
        self.reset_network_conditions()
        
        condition = self.CONDITIONS[condition_name]
        
        print(f"\n{'='*60}")
        print(f"Applying Network Condition: {condition['name']}")
        print(f"{'='*60}")
        print(f"  Mode: {self.execution_mode.value}")
        print(f"  Interface: {self.interface}")
        print(f"  Latency: {condition['latency_ms']}ms")
        print(f"  Jitter: {condition['jitter_ms']}ms")
        print(f"  Bandwidth: {condition['bandwidth_mbps']}Mbps")
        print(f"  Packet Loss: {condition['loss_percent']}%")
        print(f"{'='*60}\n")
        
        # Build netem parameters
        netem_params = []
        
        if condition['latency_ms'] > 0:
            netem_params.extend(["delay", f"{condition['latency_ms']}ms"])
            if condition['jitter_ms'] > 0:
                netem_params.append(f"{condition['jitter_ms']}ms")
        
        if condition['loss_percent'] > 0:
            netem_params.extend(["loss", f"{condition['loss_percent']}%"])
        
        # Apply netem rules
        if netem_params:
            cmd = ["qdisc", "add", "dev", self.interface, "root", "netem"] + netem_params
            if not self._run_tc_command(cmd):
                return False
            self.log(f"✓ Applied latency and loss parameters")
        
        # Apply bandwidth limit if specified
        if condition['bandwidth_mbps'] > 0:
            # For very low bandwidth (< 1 Mbps), use kbps
            if condition['bandwidth_mbps'] < 1:
                rate = f"{int(condition['bandwidth_mbps'] * 1024)}kbit"
            else:
                rate = f"{int(condition['bandwidth_mbps'])}mbit"
            
            # If netem was applied, add tbf as a child qdisc
            if netem_params:
                # Need to replace root qdisc with tbf, then add netem as child
                self.reset_network_conditions()
                
                # Add tbf as root
                cmd = ["qdisc", "add", "dev", self.interface, "root", "handle", "1:", "tbf",
                       "rate", rate, "burst", "32kbit", "latency", "400ms"]
                if not self._run_tc_command(cmd):
                    return False
                
                # Add netem as child
                cmd = ["qdisc", "add", "dev", self.interface, "parent", "1:1", "handle", "10:", "netem"] + netem_params
                if not self._run_tc_command(cmd):
                    return False
            else:
                # Just add tbf
                cmd = ["qdisc", "add", "dev", self.interface, "root", "tbf",
                       "rate", rate, "burst", "32kbit", "latency", "400ms"]
                if not self._run_tc_command(cmd):
                    return False
            
            self.log(f"✓ Applied bandwidth limit: {rate}")
        
        self.applied = True
        print(f"✓ Network conditions applied successfully\n")
        return True
    
    def apply_custom_conditions(self, latency_ms: float = 0, jitter_ms: float = 0,
                               bandwidth_mbps: float = 0, loss_percent: float = 0) -> bool:
        """Apply custom network conditions"""
        # Create a custom condition
        custom_condition = {
            "name": "Custom Network Condition",
            "latency_ms": latency_ms,
            "jitter_ms": jitter_ms,
            "bandwidth_mbps": bandwidth_mbps,
            "loss_percent": loss_percent
        }
        
        # Temporarily add to conditions dict
        self.CONDITIONS["_custom"] = custom_condition
        result = self.apply_network_condition("_custom")
        del self.CONDITIONS["_custom"]
        
        return result
    
    @staticmethod
    def from_environment(verbose: bool = False) -> Optional['NetworkConditionManager']:
        """
        Create and configure NetworkConditionManager from environment variables
        
        Environment Variables:
            NETWORK_CONDITION: Name of predefined condition (excellent, good, moderate, poor, very_poor, satellite, none)
            APPLY_NETWORK_CONDITION: Set to "true" to enable (default: false)
        
        Returns:
            NetworkConditionManager instance if enabled, None otherwise
        """
        apply_condition = os.getenv("APPLY_NETWORK_CONDITION", "false").lower() in ("true", "1", "yes", "y")
        
        if not apply_condition:
            return None
        
        manager = NetworkConditionManager(verbose=verbose)
        
        condition_name = os.getenv("NETWORK_CONDITION", "none").lower()
        
        if condition_name != "none":
            success = manager.apply_network_condition(condition_name)
            if not success:
                print(f"[WARNING] Failed to apply network condition: {condition_name}")
                print(f"[WARNING] Continuing without network conditions")
                return None
        
        return manager


def main():
    """CLI interface for network condition management"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Network Condition Manager for FL Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply poor network conditions
  python network_condition_manager.py apply --condition poor

  # Apply custom conditions
  python network_condition_manager.py apply --latency 100 --jitter 30 --bandwidth 2 --loss 3

  # Reset network conditions
  python network_condition_manager.py reset

  # List available conditions
  python network_condition_manager.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Apply command
    apply_parser = subparsers.add_parser('apply', help='Apply network conditions')
    apply_parser.add_argument('--condition', '-c', help='Predefined condition name')
    apply_parser.add_argument('--latency', type=float, help='Custom latency in ms')
    apply_parser.add_argument('--jitter', type=float, help='Custom jitter in ms')
    apply_parser.add_argument('--bandwidth', type=float, help='Custom bandwidth in Mbps')
    apply_parser.add_argument('--loss', type=float, help='Custom packet loss in percent')
    apply_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Reset command
    reset_parser = subparsers.add_parser('reset', help='Reset network conditions')
    reset_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available conditions')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        print("\n" + "="*70)
        print("Available Network Conditions")
        print("="*70 + "\n")
        
        for key, condition in NetworkConditionManager.CONDITIONS.items():
            print(f"Condition: {key}")
            print(f"  Name: {condition['name']}")
            print(f"  Latency: {condition['latency_ms']}ms")
            print(f"  Jitter: {condition['jitter_ms']}ms")
            print(f"  Bandwidth: {condition['bandwidth_mbps']}Mbps")
            print(f"  Packet Loss: {condition['loss_percent']}%")
            print()
    
    elif args.command == 'apply':
        manager = NetworkConditionManager(verbose=args.verbose)
        
        if args.condition:
            manager.apply_network_condition(args.condition)
        elif any([args.latency, args.jitter, args.bandwidth, args.loss]):
            manager.apply_custom_conditions(
                latency_ms=args.latency or 0,
                jitter_ms=args.jitter or 0,
                bandwidth_mbps=args.bandwidth or 0,
                loss_percent=args.loss or 0
            )
        else:
            print("[ERROR] Specify either --condition or custom parameters")
            apply_parser.print_help()
    
    elif args.command == 'reset':
        manager = NetworkConditionManager(verbose=args.verbose)
        manager.reset_network_conditions()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
