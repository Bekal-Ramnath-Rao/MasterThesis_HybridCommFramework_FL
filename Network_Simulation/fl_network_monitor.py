#!/usr/bin/env python3
"""
FL Network Monitor and Controller
Real-time network condition control and FL training monitoring
Applies network conditions from the host side using veth interfaces

Features:
1. Select specific clients
2. Control packet loss, jitter, latency, and bandwidth
3. Apply changes in parallel during training
4. Monitor FL setup in real-time
5. Host-level tc control using veth interfaces

Usage:
    # Interactive mode with monitoring
    python fl_network_monitor.py --monitor
    
    # Apply conditions to specific client
    python fl_network_monitor.py --client-id 1 --latency 200ms --loss 2
    
    # Quick change during training
    python fl_network_monitor.py --client-id 2 --bandwidth 1mbit
"""

import subprocess
import argparse
import time
import threading
import sys
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


class FLNetworkMonitor:
    """Monitor and control FL network conditions from host level"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.monitoring = False
        self.client_containers = []
        self.container_veth_map = {}
        self.refresh_containers()
        
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {
                "INFO": "ℹ",
                "SUCCESS": "✓",
                "ERROR": "✗",
                "WARNING": "⚠"
            }.get(level, "•")
            print(f"[{timestamp}] {prefix} {message}")
    
    def run_command(self, command: List[str], check: bool = False) -> Tuple[bool, str, str]:
        """Execute command and return (success, stdout, stderr)"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=check
            )
            return (result.returncode == 0, result.stdout, result.stderr)
        except subprocess.CalledProcessError as e:
            return (False, e.stdout if hasattr(e, 'stdout') else '', 
                   e.stderr if hasattr(e, 'stderr') else str(e))
        except Exception as e:
            return (False, '', str(e))
    
    def refresh_containers(self) -> bool:
        """Get list of running client containers and their veth interfaces"""
        try:
            # Get running containers
            success, stdout, _ = self.run_command(
                ['docker', 'ps', '--format', '{{.Names}}']
            )
            
            if not success:
                return False
            
            all_containers = [name.strip() for name in stdout.split('\n') if name.strip()]
            
            # Filter client containers
            self.client_containers = [
                c for c in all_containers 
                if 'client' in c.lower() and 'server' not in c.lower()
            ]
            
            # Map each container to its veth interface
            self.container_veth_map = {}
            for container in self.client_containers:
                veth = self.get_container_veth(container)
                if veth:
                    self.container_veth_map[container] = veth
            
            self.log(f"Found {len(self.client_containers)} client container(s)", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error refreshing containers: {e}", "ERROR")
            return False
    
    def get_container_veth(self, container_name: str) -> Optional[str]:
        """Get the host veth interface for a container"""
        try:
            # Get container ID
            success, container_id, _ = self.run_command(
                ['docker', 'ps', '-qf', f'name={container_name}']
            )
            
            if not success or not container_id.strip():
                return None
            
            container_id = container_id.strip()
            
            # Get container's eth0 iflink index
            success, iflink, _ = self.run_command(
                ['docker', 'exec', container_id, 'cat', '/sys/class/net/eth0/iflink']
            )
            
            if not success or not iflink.strip():
                return None
            
            ifindex = iflink.strip()
            
            # Find the veth interface on host with this ifindex
            success, ip_output, _ = self.run_command(['ip', 'link'])
            
            if not success:
                return None
            
            # Parse ip link output to find interface with matching index
            for line in ip_output.split('\n'):
                if line.startswith(f"{ifindex}:"):
                    # Extract interface name
                    match = re.search(r'^\d+:\s+([^:@]+)', line)
                    if match:
                        veth_name = match.group(1).strip()
                        return veth_name
            
            return None
            
        except Exception as e:
            self.log(f"Error getting veth for {container_name}: {e}", "ERROR")
            return None
    
    def apply_network_conditions(self, container: str, latency: str = None, 
                                 bandwidth: str = None, loss: float = None, 
                                 jitter: str = None) -> bool:
        """Apply network conditions to container using host-level tc on veth interface"""
        
        veth = self.container_veth_map.get(container)
        if not veth:
            self.log(f"No veth interface found for {container}", "ERROR")
            return False
        
        self.log(f"Applying network conditions to {container} via {veth}")
        
        # Clear existing rules first
        self.run_command(['sudo', 'tc', 'qdisc', 'del', 'dev', veth, 'root'])
        time.sleep(0.5)  # Give kernel time to process
        
        # Build tc command based on parameters
        has_bandwidth = bandwidth is not None
        has_netem = any([latency, loss is not None, jitter])
        
        try:
            if has_bandwidth and has_netem:
                # Apply bandwidth control with tbf
                cmd_tbf = ['sudo', 'tc', 'qdisc', 'add', 'dev', veth, 'root',
                          'handle', '1:', 'tbf',
                          'rate', bandwidth,
                          'burst', '32kbit',
                          'latency', '400ms']
                
                success, _, stderr = self.run_command(cmd_tbf)
                if not success:
                    self.log(f"Failed to set bandwidth: {stderr}", "ERROR")
                    return False
                
                # Add netem as child qdisc
                cmd_netem = ['sudo', 'tc', 'qdisc', 'add', 'dev', veth,
                            'parent', '1:1', 'handle', '10:', 'netem']
                
                if latency:
                    cmd_netem.extend(['delay', latency])
                    if jitter:
                        cmd_netem.append(jitter)
                
                if loss is not None:
                    cmd_netem.extend(['loss', f'{loss}%'])
                
                success, _, stderr = self.run_command(cmd_netem)
                if not success:
                    self.log(f"Failed to set netem: {stderr}", "ERROR")
                    return False
                    
            elif has_netem:
                # Only netem parameters
                cmd = ['sudo', 'tc', 'qdisc', 'add', 'dev', veth, 'root', 'netem']
                
                if latency:
                    cmd.extend(['delay', latency])
                    if jitter:
                        cmd.append(jitter)
                
                if loss is not None:
                    cmd.extend(['loss', f'{loss}%'])
                
                success, _, stderr = self.run_command(cmd)
                if not success:
                    self.log(f"Failed to apply netem: {stderr}", "ERROR")
                    return False
                    
            elif has_bandwidth:
                # Only bandwidth
                cmd = ['sudo', 'tc', 'qdisc', 'add', 'dev', veth, 'root',
                      'tbf', 'rate', bandwidth,
                      'burst', '32kbit', 'latency', '400ms']
                
                success, _, stderr = self.run_command(cmd)
                if not success:
                    self.log(f"Failed to set bandwidth: {stderr}", "ERROR")
                    return False
            
            # Build params string for display
            params = []
            if latency:
                params.append(f"latency={latency}")
            if bandwidth:
                params.append(f"bandwidth={bandwidth}")
            if loss is not None:
                params.append(f"loss={loss}%")
            if jitter:
                params.append(f"jitter={jitter}")
            
            self.log(f"Applied to {container}: {', '.join(params)}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Error applying conditions: {e}", "ERROR")
            return False
    
    def clear_network_conditions(self, container: str) -> bool:
        """Clear network conditions from container"""
        veth = self.container_veth_map.get(container)
        if not veth:
            return False
        
        success, _, _ = self.run_command(['sudo', 'tc', 'qdisc', 'del', 'dev', veth, 'root'])
        if success:
            self.log(f"Cleared conditions from {container}", "SUCCESS")
        return success
    
    def show_current_conditions(self, container: str = None):
        """Show current network conditions"""
        containers = [container] if container else self.client_containers
        
        print("\n" + "="*100)
        print(f"{'Container':<35} {'Veth Interface':<20} {'Network Conditions':<45}")
        print("="*100)
        
        for cont in containers:
            veth = self.container_veth_map.get(cont, "N/A")
            
            if veth and veth != "N/A":
                success, stdout, _ = self.run_command(
                    ['sudo', 'tc', 'qdisc', 'show', 'dev', veth]
                )
                
                if success and stdout.strip():
                    # Parse tc output
                    conditions = stdout.strip().replace('\n', ' | ')
                else:
                    conditions = "No conditions"
            else:
                conditions = "N/A"
            
            print(f"{cont:<35} {veth:<20} {conditions:<45}")
        
        print("="*100 + "\n")
    
    def monitor_fl_training(self, interval: int = 5):
        """Monitor FL training progress"""
        print("\n" + "="*80)
        print("FL Training Monitor - Press Ctrl+C to stop")
        print("="*80 + "\n")
        
        self.monitoring = True
        
        try:
            while self.monitoring:
                # Check container status
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update:")
                print("-" * 80)
                
                for container in self.client_containers:
                    # Check if container is still running
                    success, _, _ = self.run_command(
                        ['docker', 'ps', '-qf', f'name={container}']
                    )
                    
                    status = "Running" if success else "Stopped"
                    veth = self.container_veth_map.get(container, "N/A")
                    
                    # Get basic stats
                    stats_cmd = ['docker', 'stats', container, '--no-stream', 
                                '--format', '{{.CPUPerc}}\t{{.MemUsage}}']
                    success, stats, _ = self.run_command(stats_cmd)
                    
                    if success:
                        stats_info = stats.strip()
                    else:
                        stats_info = "N/A"
                    
                    print(f"  {container:<35} | {status:<10} | veth: {veth:<15} | {stats_info}")
                
                print("-" * 80)
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            self.monitoring = False
    
    def interactive_mode(self):
        """Interactive mode for network control and monitoring"""
        print("\n" + "="*80)
        print("FL Network Monitor & Controller")
        print("Host-level network control using veth interfaces")
        print("="*80)
        
        while True:
            self.refresh_containers()
            
            print(f"\n{len(self.client_containers)} client container(s) found:")
            for idx, container in enumerate(self.client_containers, 1):
                veth = self.container_veth_map.get(container, "N/A")
                print(f"  {idx}. {container:<40} (veth: {veth})")
            
            print("\nOptions:")
            print("  1. Apply network conditions to specific client")
            print("  2. Apply conditions to all clients")
            print("  3. Change single parameter for specific client")
            print("  4. Show current network conditions")
            print("  5. Clear conditions from specific client")
            print("  6. Clear all conditions")
            print("  7. Start FL training monitor")
            print("  8. Refresh container/veth mapping")
            print("  9. Test veth interface connectivity")
            print("  0. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            
            elif choice == '1':
                # Apply to specific client
                try:
                    client_num = int(input("Enter client number: ").strip())
                    container = self.client_containers[client_num - 1]
                    
                    print(f"\nApplying conditions to: {container}")
                    print("Leave blank to skip a parameter\n")
                    
                    latency = input("  Latency (e.g., 200ms, 300ms): ").strip() or None
                    bandwidth = input("  Bandwidth (e.g., 1mbit, 500kbit): ").strip() or None
                    loss_str = input("  Packet loss % (e.g., 2, 5): ").strip()
                    loss = float(loss_str) if loss_str else None
                    jitter = input("  Jitter (e.g., 10ms, 30ms): ").strip() or None
                    
                    if any([latency, bandwidth, loss, jitter]):
                        self.apply_network_conditions(container, latency, bandwidth, loss, jitter)
                    else:
                        print("No parameters specified")
                        
                except (ValueError, IndexError) as e:
                    print(f"Invalid selection: {e}")
            
            elif choice == '2':
                # Apply to all clients
                print("\nApplying conditions to ALL clients")
                print("Leave blank to skip a parameter\n")
                
                latency = input("  Latency (e.g., 200ms, 300ms): ").strip() or None
                bandwidth = input("  Bandwidth (e.g., 1mbit, 500kbit): ").strip() or None
                loss_str = input("  Packet loss % (e.g., 2, 5): ").strip()
                loss = float(loss_str) if loss_str else None
                jitter = input("  Jitter (e.g., 10ms, 30ms): ").strip() or None
                
                if any([latency, bandwidth, loss, jitter]):
                    for container in self.client_containers:
                        self.apply_network_conditions(container, latency, bandwidth, loss, jitter)
                else:
                    print("No parameters specified")
            
            elif choice == '3':
                # Change single parameter
                try:
                    client_num = int(input("Enter client number: ").strip())
                    container = self.client_containers[client_num - 1]
                    
                    print("\nSelect parameter to modify:")
                    print("  1. Latency")
                    print("  2. Bandwidth")
                    print("  3. Packet Loss")
                    print("  4. Jitter")
                    
                    param_choice = input("Parameter: ").strip()
                    
                    if param_choice == '1':
                        value = input("New latency (e.g., 200ms): ").strip()
                        if value:
                            self.apply_network_conditions(container, latency=value)
                    elif param_choice == '2':
                        value = input("New bandwidth (e.g., 1mbit): ").strip()
                        if value:
                            self.apply_network_conditions(container, bandwidth=value)
                    elif param_choice == '3':
                        value = input("New packet loss % (e.g., 2): ").strip()
                        if value:
                            self.apply_network_conditions(container, loss=float(value))
                    elif param_choice == '4':
                        value = input("New jitter (e.g., 10ms): ").strip()
                        if value:
                            self.apply_network_conditions(container, jitter=value)
                    else:
                        print("Invalid parameter selection")
                        
                except (ValueError, IndexError) as e:
                    print(f"Invalid selection: {e}")
            
            elif choice == '4':
                self.show_current_conditions()
            
            elif choice == '5':
                # Clear specific client
                try:
                    client_num = int(input("Enter client number: ").strip())
                    container = self.client_containers[client_num - 1]
                    self.clear_network_conditions(container)
                except (ValueError, IndexError) as e:
                    print(f"Invalid selection: {e}")
            
            elif choice == '6':
                # Clear all
                confirm = input("Clear conditions from all clients? (y/n): ").strip().lower()
                if confirm == 'y':
                    for container in self.client_containers:
                        self.clear_network_conditions(container)
            
            elif choice == '7':
                # Start monitoring
                interval = input("Monitor interval in seconds (default 5): ").strip()
                interval = int(interval) if interval else 5
                
                print("\nStarting FL training monitor...")
                print("You can apply network changes in another terminal while monitoring")
                self.monitor_fl_training(interval)
            
            elif choice == '8':
                print("Refreshing container and veth mapping...")
                self.refresh_containers()
            
            elif choice == '9':
                # Test veth connectivity
                for container in self.client_containers:
                    veth = self.container_veth_map.get(container)
                    if veth:
                        success, stdout, _ = self.run_command(['ip', 'link', 'show', veth])
                        status = "UP" if "state UP" in stdout else "DOWN/ERROR"
                        print(f"  {container}: {veth} - {status}")
                    else:
                        print(f"  {container}: No veth found")
            
            else:
                print("Invalid option")
            
            input("\nPress Enter to continue...")


def main():
    parser = argparse.ArgumentParser(
        description='FL Network Monitor and Controller (Host-level veth control)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with monitoring
  python fl_network_monitor.py --monitor
  
  # Apply conditions to specific client by number
  python fl_network_monitor.py --client-id 1 --latency 200ms --loss 2
  
  # Quick bandwidth change during training
  python fl_network_monitor.py --client-id 2 --bandwidth 1mbit
  
  # Apply to all clients
  python fl_network_monitor.py --all --latency 300ms --jitter 50ms
  
  # Show current conditions
  python fl_network_monitor.py --show-status
  
  # Clear conditions
  python fl_network_monitor.py --client-id 1 --clear
        """
    )
    
    parser.add_argument('-m', '--monitor', action='store_true',
                       help='Run in interactive monitoring mode')
    parser.add_argument('--client-id', type=int,
                       help='Client number (1-based index)')
    parser.add_argument('--all', action='store_true',
                       help='Apply to all clients')
    parser.add_argument('--latency', type=str,
                       help='Latency (e.g., 200ms, 300ms)')
    parser.add_argument('--bandwidth', type=str,
                       help='Bandwidth (e.g., 1mbit, 500kbit)')
    parser.add_argument('--loss', type=float,
                       help='Packet loss percentage (e.g., 2, 5)')
    parser.add_argument('--jitter', type=str,
                       help='Jitter (e.g., 10ms, 30ms)')
    parser.add_argument('--show-status', action='store_true',
                       help='Show current network conditions')
    parser.add_argument('--clear', action='store_true',
                       help='Clear network conditions')
    parser.add_argument('--interval', type=int, default=5,
                       help='Monitoring interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    monitor = FLNetworkMonitor(verbose=True)
    
    if not monitor.client_containers:
        print("No client containers found. Make sure FL containers are running.")
        return 1
    
    # Interactive monitoring mode
    if args.monitor:
        monitor.interactive_mode()
        return 0
    
    # Show status
    if args.show_status:
        monitor.show_current_conditions()
        return 0
    
    # Determine target containers
    target_containers = []
    
    if args.all:
        target_containers = monitor.client_containers
    elif args.client_id:
        try:
            target_containers = [monitor.client_containers[args.client_id - 1]]
        except IndexError:
            print(f"Error: Client ID {args.client_id} not found")
            print(f"Available clients: 1-{len(monitor.client_containers)}")
            return 1
    
    # Clear conditions
    if args.clear:
        if not target_containers:
            print("Error: Specify --client-id or --all with --clear")
            return 1
        
        for container in target_containers:
            monitor.clear_network_conditions(container)
        return 0
    
    # Apply network conditions
    if target_containers and any([args.latency, args.bandwidth, args.loss, args.jitter]):
        for container in target_containers:
            monitor.apply_network_conditions(
                container,
                latency=args.latency,
                bandwidth=args.bandwidth,
                loss=args.loss,
                jitter=args.jitter
            )
        return 0
    
    # No action specified
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
