#!/usr/bin/env python3
"""
Dynamic Network Parameter Controller for FL Clients
Allows real-time modification of network conditions (latency, bandwidth, packet loss, jitter)
for client containers while FL training is running.

Supports ANY custom values - not limited to predefined scenarios:
- Latency: ANY milliseconds (e.g., 75ms, 125.5ms, 8ms)
- Bandwidth: ANY rate (e.g., 8.5mbit, 500kbit, 1.2mbit)
- Packet Loss: ANY percentage (e.g., 1.5%, 0.3%, 7.8%)
- Jitter: ANY milliseconds (e.g., 15.5ms, 3.2ms, 22ms)

Usage:
    # Interactive mode (recommended for dynamic changes)
    python dynamic_network_controller.py --interactive
    
    # Apply preset scenario to all clients
    python dynamic_network_controller.py --scenario poor
    
    # Set ANY custom values for all clients
    python dynamic_network_controller.py --latency 75ms --bandwidth 8.5mbit --loss 1.5 --jitter 12ms
    
    # Modify single parameter (keeps others unchanged)
    python dynamic_network_controller.py --latency 125.5ms
    
    # Apply to specific protocol
    python dynamic_network_controller.py --protocol mqtt --bandwidth 3.2mbit --jitter 18.5ms
    
    # Show current network status
    python dynamic_network_controller.py --show-status
"""

import subprocess
import argparse
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Predefined network scenarios
NETWORK_SCENARIOS = {
    'excellent': {
        'latency': '5ms',
        'bandwidth': '100mbit',
        'loss': '0.1',
        'jitter': '1ms'
    },
    'good': {
        'latency': '20ms',
        'bandwidth': '50mbit',
        'loss': '0.5',
        'jitter': '5ms'
    },
    'moderate': {
        'latency': '50ms',
        'bandwidth': '10mbit',
        'loss': '1',
        'jitter': '10ms'
    },
    'poor': {
        'latency': '100ms',
        'bandwidth': '5mbit',
        'loss': '2',
        'jitter': '20ms'
    },
    'very_poor': {
        'latency': '200ms',
        'bandwidth': '1mbit',
        'loss': '1',
        'jitter': '50ms'
    }
}


class NetworkController:
    """Controls network parameters for Docker containers"""
    
    def __init__(self):
        self.running_containers = []
        self.client_containers = []
        self.refresh_containers()
    
    def refresh_containers(self):
        """Get list of running Docker containers"""
        try:
            result = subprocess.run(
                ['docker', 'ps', '--format', '{{.Names}}'],
                capture_output=True,
                text=True,
                check=True
            )
            self.running_containers = [name.strip() for name in result.stdout.split('\n') if name.strip()]
            
            # Filter client containers (exclude server and broker containers)
            self.client_containers = [
                c for c in self.running_containers 
                if 'client' in c.lower() and 'server' not in c.lower()
            ]
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error getting container list: {e}")
            return False
    
    def get_containers_by_protocol(self, protocol: str) -> List[str]:
        """Get client containers for a specific protocol"""
        protocol_lower = protocol.lower()
        return [c for c in self.client_containers if protocol_lower in c.lower()]
    
    def get_containers_by_usecase(self, usecase: str) -> List[str]:
        """Get client containers for a specific use case"""
        usecase_lower = usecase.lower()
        # Support both full names and abbreviations (temperature -> temp, emotion -> emo, mental -> ment)
        usecase_map = {
            'temperature': ['temperature', 'temp'],
            'temp': ['temperature', 'temp'],
            'emotion': ['emotion', 'emo'],
            'emo': ['emotion', 'emo'],
            'mental': ['mental', 'ment'],
            'ment': ['mental', 'ment']
        }
        
        search_terms = usecase_map.get(usecase_lower, [usecase_lower])
        return [c for c in self.client_containers 
                if any(term in c.lower() for term in search_terms)]
    
    def clear_network_rules(self, container_name: str) -> bool:
        """Clear existing tc rules from a container"""
        try:
            # Try to delete root qdisc with retry logic
            max_attempts = 5
            for attempt in range(max_attempts):
                result = subprocess.run(
                    ['docker', 'exec', container_name, 'tc', 'qdisc', 'del', 'dev', 'eth0', 'root'],
                    capture_output=True,
                    stderr=subprocess.PIPE,
                    text=True
                )
                # If no error or "No such file or directory" (no rules), we're done
                if result.returncode == 0 or 'RTNETLINK answers: No such file or directory' in result.stderr:
                    # Give kernel time to fully release the interface (critical!)
                    time.sleep(1.0)
                    return True
                # Wait a bit before retrying
                time.sleep(0.1 * (attempt + 1))
            return True
        except Exception:
            # If there are no rules, this will fail - that's okay
            return True
    
    def apply_network_params(self, container_name: str, params: Dict) -> bool:
        """Apply network parameters to a container"""
        try:
            # First clear existing rules - this includes a wait period
            self.clear_network_rules(container_name)
            
            # Build tc command
            latency = params.get('latency', '0ms')
            bandwidth = params.get('bandwidth', '100mbit')
            loss = params.get('loss', '0')
            jitter = params.get('jitter', '0ms')
            
            # Determine if we need bandwidth control
            has_bandwidth = bandwidth != '100mbit'
            has_netem_params = latency != '0ms' or float(loss) > 0 or jitter != '0ms'
            
            if has_bandwidth and has_netem_params:
                # Need both tbf (bandwidth) and netem (latency/jitter/loss)
                # Use replace instead of add to handle existing rules
                cmd = [
                    'docker', 'exec', container_name,
                    'tc', 'qdisc', 'replace', 'dev', 'eth0', 'root',
                    'handle', '1:', 'tbf',
                    'rate', bandwidth,
                    'burst', '32kbit',
                    'latency', '400ms'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    # Check if it's just a warning about exclusivity but rules were still applied
                    if "Exclusivity flag" not in result.stderr:
                        print(f"Warning: Failed to set bandwidth for {container_name}: {result.stderr.strip()}")
                        return False
                
                # Add netem qdisc as child
                netem_cmd = [
                    'docker', 'exec', container_name,
                    'tc', 'qdisc', 'replace', 'dev', 'eth0', 'parent', '1:1',
                    'handle', '10:', 'netem'
                ]
                
                if latency != '0ms':
                    netem_cmd.extend(['delay', latency])
                    if jitter != '0ms':
                        netem_cmd.append(jitter)
                
                if float(loss) > 0:
                    netem_cmd.extend(['loss', f'{loss}%'])
                
                result = subprocess.run(netem_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    # Check if it's just a warning about exclusivity but rules were still applied
                    if "Exclusivity flag" not in result.stderr:
                        print(f"Error applying netem to {container_name}: {result.stderr.strip()}")
                        return False
                    
            elif has_netem_params:
                # Only netem parameters (no bandwidth control)
                netem_cmd = [
                    'docker', 'exec', container_name,
                    'tc', 'qdisc', 'replace', 'dev', 'eth0', 'root',
                    'netem'
                ]
                
                if latency != '0ms':
                    netem_cmd.extend(['delay', latency])
                    if jitter != '0ms':
                        netem_cmd.append(jitter)
                
                if float(loss) > 0:
                    netem_cmd.extend(['loss', f'{loss}%'])
                
                result = subprocess.run(netem_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error applying netem to {container_name}: {result.stderr.strip()}")
                    return False
                    
            elif has_bandwidth:
                # Only bandwidth control
                cmd = [
                    'docker', 'exec', container_name,
                    'tc', 'qdisc', 'replace', 'dev', 'eth0', 'root',
                    'tbf',
                    'rate', bandwidth,
                    'burst', '32kbit',
                    'latency', '400ms'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error setting bandwidth for {container_name}: {result.stderr.strip()}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error applying network params to {container_name}: {e}")
            return False
    
    def show_current_rules(self, container_name: str):
        """Display current tc rules for a container"""
        try:
            result = subprocess.run(
                ['docker', 'exec', container_name, 'tc', 'qdisc', 'show', 'dev', 'eth0'],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"\n{container_name}:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error showing rules for {container_name}: {e}")
    
    def apply_scenario(self, containers: List[str], scenario: str) -> Dict:
        """Apply a predefined scenario to containers"""
        if scenario not in NETWORK_SCENARIOS:
            print(f"Unknown scenario: {scenario}")
            print(f"Available scenarios: {', '.join(NETWORK_SCENARIOS.keys())}")
            return {}
        
        params = NETWORK_SCENARIOS[scenario]
        results = {}
        
        for container in containers:
            success = self.apply_network_params(container, params)
            results[container] = success
            status = "✓" if success else "✗"
            print(f"{status} {container}: {scenario} scenario applied")
        
        return results
    
    def apply_custom_params(self, containers: List[str], latency=None, bandwidth=None, 
                           loss=None, jitter=None) -> Dict:
        """Apply custom network parameters to containers
        
        Supports any custom values:
        - latency: any milliseconds (e.g., '75ms', '125ms', '8.5ms')
        - bandwidth: any rate (e.g., '8.5mbit', '500kbit', '1.2mbit')
        - loss: any percentage (e.g., '1.5', '0.3', '7.8')
        - jitter: any milliseconds (e.g., '15.5ms', '3.2ms')
        """
        params = {}
        
        if latency is not None:
            params['latency'] = latency if latency.endswith('ms') else f'{latency}ms'
        if bandwidth is not None:
            params['bandwidth'] = bandwidth if 'bit' in bandwidth else f'{bandwidth}mbit'
        if loss is not None:
            params['loss'] = str(loss)
        if jitter is not None:
            params['jitter'] = jitter if jitter.endswith('ms') else f'{jitter}ms'
        
        if not params:
            print("No parameters specified")
            return {}
        
        results = {}
        for container in containers:
            success = self.apply_network_params(container, params)
            results[container] = success
            status = "✓" if success else "✗"
            print(f"{status} {container}: Custom parameters applied - {params}")
        
        return results
    
    def get_current_params(self, container_name: str) -> Dict:
        """Extract current network parameters from tc rules"""
        try:
            result = subprocess.run(
                ['docker', 'exec', container_name, 'tc', 'qdisc', 'show', 'dev', 'eth0'],
                capture_output=True,
                text=True,
                check=True
            )
            
            params = {
                'latency': 'unknown',
                'bandwidth': 'unknown',
                'loss': 'unknown',
                'jitter': 'unknown'
            }
            
            lines = result.stdout.split('\n')
            for line in lines:
                # Parse bandwidth from tbf
                if 'tbf' in line and 'rate' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'rate' and i + 1 < len(parts):
                            params['bandwidth'] = parts[i + 1]
                
                # Parse latency, jitter, loss from netem
                if 'netem' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'delay' and i + 1 < len(parts):
                            params['latency'] = parts[i + 1]
                            if i + 2 < len(parts) and parts[i + 2] != 'loss':
                                params['jitter'] = parts[i + 2]
                        elif part == 'loss' and i + 1 < len(parts):
                            params['loss'] = parts[i + 1].rstrip('%')
            
            return params
        except Exception as e:
            print(f"Error getting current params for {container_name}: {e}")
            return {}
    
    def show_current_status(self, containers: List[str] = None):
        """Display current network parameters for containers in a table format"""
        if containers is None:
            containers = self.client_containers
        
        print("\n" + "="*100)
        print(f"{'Container':<30} {'Latency':<15} {'Bandwidth':<15} {'Loss':<10} {'Jitter':<15}")
        print("="*100)
        
        for container in containers:
            params = self.get_current_params(container)
            print(f"{container:<30} {params.get('latency', 'N/A'):<15} "
                  f"{params.get('bandwidth', 'N/A'):<15} {params.get('loss', 'N/A'):<10} "
                  f"{params.get('jitter', 'N/A'):<15}")
        
        print("="*100)
    
    def interactive_mode(self):
        """Interactive mode for dynamic parameter changes"""
        print("\n" + "="*70)
        print("Dynamic Network Parameter Controller - Interactive Mode")
        print("="*70)
        
        while True:
            self.refresh_containers()
            
            print(f"\n{len(self.client_containers)} client container(s) found:")
            for idx, container in enumerate(self.client_containers, 1):
                print(f"  {idx}. {container}")
            
            print("\nOptions:")
            print("  1. Apply scenario to all clients")
            print("  2. Apply scenario to specific protocol")
            print("  3. Apply ANY custom values to all clients")
            print("  4. Apply ANY custom values to specific containers")
            print("  5. Show current network status (parameters)")
            print("  6. Show detailed tc rules")
            print("  7. Modify single parameter for all clients")
            print("  8. Clear all network rules")
            print("  9. Refresh container list")
            print("  0. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '0':
                print("Exiting...")
                break
            
            elif choice == '1':
                print(f"\nAvailable scenarios: {', '.join(NETWORK_SCENARIOS.keys())}")
                scenario = input("Enter scenario name: ").strip()
                self.apply_scenario(self.client_containers, scenario)
            
            elif choice == '2':
                protocol = input("Enter protocol (mqtt/amqp/grpc/quic/dds): ").strip()
                containers = self.get_containers_by_protocol(protocol)
                if containers:
                    print(f"\nFound {len(containers)} container(s) for {protocol}")
                    print(f"Available scenarios: {', '.join(NETWORK_SCENARIOS.keys())}")
                    scenario = input("Enter scenario name: ").strip()
                    self.apply_scenario(containers, scenario)
                else:
                    print(f"No containers found for protocol: {protocol}")
            
            elif choice == '3':
                print("\n--- Set ANY Custom Values ---")
                print("Examples: latency=75ms, bandwidth=8.5mbit, loss=1.5, jitter=12.3ms")
                print("Leave blank to skip a parameter\n")
                latency = input("  Latency (e.g., 75ms, 125.5ms): ").strip() or None
                bandwidth = input("  Bandwidth (e.g., 8.5mbit, 500kbit): ").strip() or None
                loss = input("  Packet loss % (e.g., 1.5, 0.3): ").strip() or None
                jitter = input("  Jitter (e.g., 12.5ms, 8ms): ").strip() or None
                
                if any([latency, bandwidth, loss, jitter]):
                    self.apply_custom_params(
                        self.client_containers, 
                        latency=latency,
                        bandwidth=bandwidth,
                        loss=loss,
                        jitter=jitter
                    )
                else:
                    print("No parameters specified")
            
            elif choice == '4':
                print("\nSelect containers (comma-separated numbers):")
                selection = input("Container numbers: ").strip()
                try:
                    indices = [int(x.strip()) - 1 for x in selection.split(',')]
                    selected = [self.client_containers[i] for i in indices]
                    
                    print("\n--- Set ANY Custom Values ---")
                    print("Examples: latency=75ms, bandwidth=8.5mbit, loss=1.5, jitter=12.3ms")
                    print("Leave blank to skip a parameter\n")
                    latency = input("  Latency (e.g., 75ms, 125.5ms): ").strip() or None
                    bandwidth = input("  Bandwidth (e.g., 8.5mbit, 500kbit): ").strip() or None
                    loss = input("  Packet loss % (e.g., 1.5, 0.3): ").strip() or None
                    jitter = input("  Jitter (e.g., 12.5ms, 8ms): ").strip() or None
                    
                    if any([latency, bandwidth, loss, jitter]):
                        self.apply_custom_params(
                            selected,
                            latency=latency,
                            bandwidth=bandwidth,
                            loss=loss,
                            jitter=jitter
                        )
                    else:
                        print("No parameters specified")
                except (ValueError, IndexError) as e:
                    print(f"Invalid selection: {e}")
            
            elif choice == '5':
                self.show_current_status()
            
            elif choice == '6':
                for container in self.client_containers:
                    self.show_current_rules(container)
            
            elif choice == '7':
                print("\n--- Modify Single Parameter ---")
                print("This will change ONLY the selected parameter, keeping others unchanged")
                print("\nSelect parameter to modify:")
                print("  1. Latency")
                print("  2. Bandwidth")
                print("  3. Packet Loss")
                print("  4. Jitter")
                param_choice = input("\nParameter: ").strip()
                
                param_map = {
                    '1': ('latency', 'Enter new latency (e.g., 75ms, 125.5ms): '),
                    '2': ('bandwidth', 'Enter new bandwidth (e.g., 8.5mbit, 500kbit): '),
                    '3': ('loss', 'Enter new packet loss % (e.g., 1.5, 0.3): '),
                    '4': ('jitter', 'Enter new jitter (e.g., 12.5ms, 8ms): ')
                }
                
                if param_choice in param_map:
                    param_name, prompt = param_map[param_choice]
                    value = input(prompt).strip()
                    
                    if value:
                        kwargs = {param_name: value}
                        self.apply_custom_params(self.client_containers, **kwargs)
                    else:
                        print("No value entered")
                else:
                    print("Invalid parameter selection")
            
            elif choice == '8':
                confirm = input("Clear all network rules from all clients? (y/n): ").strip().lower()
                if confirm == 'y':
                    for container in self.client_containers:
                        self.clear_network_rules(container)
                        print(f"✓ Cleared rules from {container}")
            
            elif choice == '9':
                print("Refreshing container list...")
                self.refresh_containers()
            
            else:
                print("Invalid option")
            
            input("\nPress Enter to continue...")


def main():
    parser = argparse.ArgumentParser(
        description='Dynamic Network Parameter Controller for FL Clients',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended for real-time changes during FL training)
  python dynamic_network_controller.py --interactive
  
  # Apply preset scenario
  python dynamic_network_controller.py --scenario poor
  
  # Set ANY custom values (not limited to presets)
  python dynamic_network_controller.py --latency 75ms --bandwidth 8.5mbit --loss 1.5 --jitter 12.3ms
  
  # Change single parameter (e.g., only bandwidth, keep other params)
  python dynamic_network_controller.py --bandwidth 3.2mbit
  
  # Target specific protocol with custom values
  python dynamic_network_controller.py --protocol mqtt --latency 125.5ms --jitter 18ms
  
  # Target specific containers
  python dynamic_network_controller.py --clients client-temp-1 client-temp-2 --loss 0.75
  
  # Show current network status
  python dynamic_network_controller.py --show-status
  
  # Show detailed tc rules
  python dynamic_network_controller.py --show-rules
  
  # Clear all rules
  python dynamic_network_controller.py --clear
        """
    )
    
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('-s', '--scenario', type=str,
                       help=f'Apply predefined scenario: {", ".join(NETWORK_SCENARIOS.keys())}')
    parser.add_argument('-p', '--protocol', type=str,
                       help='Target specific protocol (mqtt, amqp, grpc, quic, dds)')
    parser.add_argument('-u', '--usecase', type=str,
                       help='Target specific use case (temperature, emotion, mental)')
    parser.add_argument('-c', '--clients', nargs='+',
                       help='Target specific client container names')
    parser.add_argument('--latency', type=str,
                       help='Set latency (e.g., 50ms, 100ms)')
    parser.add_argument('--bandwidth', type=str,
                       help='Set bandwidth (e.g., 10mbit, 1mbit)')
    parser.add_argument('--loss', type=str,
                       help='Set packet loss percentage (e.g., 1, 2.5)')
    parser.add_argument('--jitter', type=str,
                       help='Set jitter (e.g., 10ms, 20ms)')
    parser.add_argument('--show-rules', action='store_true',
                       help='Show current tc rules for all clients')
    parser.add_argument('--show-status', action='store_true',
                       help='Show current network parameters in table format')
    parser.add_argument('--clear', action='store_true',
                       help='Clear all network rules from all clients')
    parser.add_argument('--force-clear', action='store_true',
                       help='Force clear rules before applying (use if getting NLM_F_REPLACE errors)')
    
    args = parser.parse_args()
    
    controller = NetworkController()
    
    if not controller.client_containers:
        print("No client containers found. Make sure FL containers are running.")
        return 1
    
    # Interactive mode
    if args.interactive:
        controller.interactive_mode()
        return 0
    
    # Show status
    if args.show_status:
        controller.show_current_status()
        return 0
    
    # Show rules
    if args.show_rules:
        for container in controller.client_containers:
            controller.show_current_rules(container)
        return 0
    
    # Clear rules
    if args.clear:
        confirm = input(f"Clear network rules from {len(controller.client_containers)} client container(s)? (y/n): ")
        if confirm.lower() == 'y':
            for container in controller.client_containers:
                controller.clear_network_rules(container)
                print(f"✓ Cleared rules from {container}")
        return 0
    
    # Force clear before applying (useful if containers have existing rules)
    if args.force_clear:
        print("Force clearing existing rules...")
        for container in controller.client_containers:
            controller.clear_network_rules(container)
            print(f"✓ Cleared rules from {container}")
    
    # Determine target containers
    target_containers = controller.client_containers
    
    if args.clients:
        target_containers = args.clients
    elif args.protocol:
        target_containers = controller.get_containers_by_protocol(args.protocol)
        if not target_containers:
            print(f"No client containers found for protocol: {args.protocol}")
            return 1
    elif args.usecase:
        target_containers = controller.get_containers_by_usecase(args.usecase)
        if not target_containers:
            print(f"No client containers found for use case: {args.usecase}")
            return 1
    
    # Apply scenario or custom parameters
    if args.scenario:
        controller.apply_scenario(target_containers, args.scenario)
    elif any([args.latency, args.bandwidth, args.loss, args.jitter]):
        controller.apply_custom_params(
            target_containers,
            latency=args.latency,
            bandwidth=args.bandwidth,
            loss=args.loss,
            jitter=args.jitter
        )
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
