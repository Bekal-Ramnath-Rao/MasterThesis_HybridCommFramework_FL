#!/usr/bin/env python3
"""
FL Training Dashboard with Network Control
Real-time monitoring and network condition control during FL training

Features:
- Real-time FL training progress monitoring
- Live network condition display
- Quick network parameter changes
- Container resource monitoring
- Protocol performance comparison

Usage:
    # Start dashboard (runs in background or separate terminal)
    python fl_training_dashboard.py
    
    # With custom refresh rate
    python fl_training_dashboard.py --interval 3
"""

import subprocess
import time
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import threading
import argparse


class FLTrainingDashboard:
    """Real-time FL training monitoring dashboard with baseline comparison"""
    
    def __init__(self, interval: int = 5, use_case: str = "emotion"):
        self.interval = interval
        self.running = True
        self.use_case = use_case
        self.client_containers = []
        self.server_containers = []
        self.container_stats = {}
        self.network_conditions = {}
        self.baseline_data = {}
        # Real-time RTT tracking
        self.current_rtt_data = {}  # {protocol: {'round_times': [], 'current_round': 0, 'avg_rtt': 0}}
        self.last_round_time = {}  # {protocol: timestamp}
        self.load_baseline_data()
        
    def load_baseline_data(self):
        """Load baseline results for comparison"""
        baseline_dir = Path(__file__).parent.parent / "experiment_results_baseline" / self.use_case
        
        if not baseline_dir.exists():
            print(f"[INFO] No baseline data found for {self.use_case} at {baseline_dir}")
            return
        
        # Load baseline RTT data for all protocols
        for protocol in ["mqtt", "amqp", "grpc", "quic", "dds"]:
            # Look for baseline folder
            baseline_folder = baseline_dir / f"{protocol}_baseline"
            if baseline_folder.exists():
                rtt_file = baseline_folder / f"{protocol}_baseline_rtt.json"
                if rtt_file.exists():
                    try:
                        with open(rtt_file, 'r') as f:
                            self.baseline_data[protocol] = json.load(f)
                        print(f"[INFO] Loaded baseline data for {protocol}: Avg RTT = {self.baseline_data[protocol].get('avg_rtt_per_round', 'N/A'):.2f}s")
                    except Exception as e:
                        print(f"[WARNING] Failed to load baseline for {protocol}: {e}")
    
    def get_protocol_from_container(self, container_name: str) -> Optional[str]:
        """Extract protocol from container name"""
        for protocol in ["mqtt", "amqp", "grpc", "quic", "dds"]:
            if protocol in container_name.lower():
                return protocol
        return None
        
    def run_command(self, command: List[str]) -> Tuple[bool, str, str]:
        """Execute command and return result"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10
            )
            return (result.returncode == 0, result.stdout, result.stderr)
        except Exception as e:
            return (False, '', str(e))
    
    def get_containers(self):
        """Get list of FL containers"""
        success, stdout, _ = self.run_command(['docker', 'ps', '--format', '{{.Names}}'])
        
        if not success:
            return
        
        all_containers = [name.strip() for name in stdout.split('\n') if name.strip()]
        
        self.client_containers = [c for c in all_containers if 'client' in c.lower()]
        self.server_containers = [c for c in all_containers if 'server' in c.lower()]
        
        # Update RTT tracking from server logs
        for server in self.server_containers:
            protocol = self.get_protocol_from_container(server)
            if protocol:
                # Get more log lines for round detection
                cmd = ['docker', 'logs', '--tail', '50', server]
                success, logs, _ = self.run_command(cmd)
                if success:
                    current_round = self.extract_round_info(logs, protocol)
                    if current_round is not None:
                        self.update_rtt_tracking(protocol, current_round)
    
    def get_container_stats(self, container: str) -> Dict:
        """Get container resource usage stats"""
        cmd = ['docker', 'stats', container, '--no-stream',
               '--format', '{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}']
        
        success, stdout, _ = self.run_command(cmd)
        
        if not success:
            return {}
        
        try:
            parts = stdout.strip().split('\t')
            return {
                'cpu': parts[0] if len(parts) > 0 else 'N/A',
                'memory': parts[1] if len(parts) > 1 else 'N/A',
                'network': parts[2] if len(parts) > 2 else 'N/A',
                'disk': parts[3] if len(parts) > 3 else 'N/A'
            }
        except:
            return {}
    
    def get_veth_interface(self, container: str) -> str:
        """Get veth interface for container"""
        try:
            # Get container ID
            success, container_id, _ = self.run_command(
                ['docker', 'ps', '-qf', f'name={container}']
            )
            
            if not success or not container_id.strip():
                return "N/A"
            
            container_id = container_id.strip()
            
            # Get iflink
            success, iflink, _ = self.run_command(
                ['docker', 'exec', container_id, 'cat', '/sys/class/net/eth0/iflink']
            )
            
            if not success:
                return "N/A"
            
            ifindex = iflink.strip()
            
            # Find veth on host
            success, ip_output, _ = self.run_command(['ip', 'link'])
            
            if not success:
                return "N/A"
            
            for line in ip_output.split('\n'):
                if line.startswith(f"{ifindex}:"):
                    import re
                    match = re.search(r'^\d+:\s+([^:@]+)', line)
                    if match:
                        return match.group(1).strip()
            
            return "N/A"
        except:
            return "N/A"
    
    def get_network_conditions(self, container: str) -> str:
        """Get current network conditions for container"""
        veth = self.get_veth_interface(container)
        
        if veth == "N/A":
            return "No veth"
        
        success, stdout, _ = self.run_command(['sudo', 'tc', 'qdisc', 'show', 'dev', veth])
        
        if not success or not stdout.strip() or 'noqueue' in stdout:
            return "None"
        
        # Parse tc output for human-readable format
        conditions = []
        
        if 'delay' in stdout:
            import re
            delay_match = re.search(r'delay (\S+)', stdout)
            if delay_match:
                conditions.append(f"L:{delay_match.group(1)}")
        
        if 'rate' in stdout:
            import re
            rate_match = re.search(r'rate (\S+)', stdout)
            if rate_match:
                conditions.append(f"BW:{rate_match.group(1)}")
        
        if 'loss' in stdout:
            import re
            loss_match = re.search(r'loss (\S+)', stdout)
            if loss_match:
                conditions.append(f"Loss:{loss_match.group(1)}")
        
        return ' '.join(conditions) if conditions else "Active (check tc)"
    
    def get_container_logs_tail(self, container: str, lines: int = 3) -> List[str]:
        """Get last few lines of container logs"""
        cmd = ['docker', 'logs', '--tail', str(lines), container]
        success, stdout, _ = self.run_command(cmd)
        
        if not success:
            return []
        
        return [line.strip() for line in stdout.split('\n') if line.strip()][-lines:]
    
    def extract_round_info(self, logs: str, protocol: str) -> Optional[int]:
        """Extract current round number from server logs"""
        import re
        
        # Patterns to detect round completion
        patterns = [
            r'Round\s+(\d+)\s+completed',
            r'\[Round\s+(\d+)\]\s+completed',
            r'Completed\s+round\s+(\d+)',
            r'Round\s+(\d+)\s+finished',
            r'Starting\s+round\s+(\d+)',
            r'Round\s+(\d+)/',
        ]
        
        max_round = None
        for pattern in patterns:
            matches = re.findall(pattern, logs, re.IGNORECASE)
            if matches:
                rounds = [int(m) for m in matches]
                if rounds:
                    max_round = max(rounds) if max_round is None else max(max_round, max(rounds))
        
        return max_round
    
    def update_rtt_tracking(self, protocol: str, current_round: int):
        """Update RTT tracking for a protocol"""
        if protocol not in self.current_rtt_data:
            self.current_rtt_data[protocol] = {
                'round_times': [],
                'current_round': 0,
                'avg_rtt': 0.0,
                'last_rtt': 0.0
            }
        
        data = self.current_rtt_data[protocol]
        current_time = time.time()
        
        # Check if we've moved to a new round
        if current_round > data['current_round']:
            if protocol in self.last_round_time:
                # Calculate RTT for the completed round
                rtt = current_time - self.last_round_time[protocol]
                data['round_times'].append(rtt)
                data['last_rtt'] = rtt
                
                # Calculate running average
                if data['round_times']:
                    data['avg_rtt'] = sum(data['round_times']) / len(data['round_times'])
            
            # Update round tracking
            data['current_round'] = current_round
            self.last_round_time[protocol] = current_time
        elif current_round == data['current_round'] and protocol not in self.last_round_time:
            # Initialize timestamp for first round
            self.last_round_time[protocol] = current_time
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name != 'nt' else 'cls')
    
    def display_dashboard(self):
        """Display the main dashboard with baseline comparison"""
        self.clear_screen()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("=" * 140)
        print(f"{'FL TRAINING DASHBOARD - Use Case: ' + self.use_case.upper():^140}")
        print(f"{'Last Update: ' + timestamp:^140}")
        print("=" * 140)
        
        # Baseline Comparison Section (if baseline data available)
        if self.baseline_data and (self.server_containers or self.client_containers):
            print("\n📈 REAL-TIME BASELINE COMPARISON")
            print("-" * 140)
            print(f"{'Protocol':<12} {'Baseline RTT':<15} {'Current RTT':<15} {'Degradation':<15} {'Round':<10} {'Network Conditions':<35} {'Status':<38}")
            print("-" * 140)
            
            protocols_detected = set()
            for container in self.server_containers + self.client_containers:
                protocol = self.get_protocol_from_container(container)
                if protocol and protocol not in protocols_detected:
                    protocols_detected.add(protocol)
                    
                    # Baseline RTT
                    baseline_rtt = "N/A"
                    baseline_rtt_val = 0
                    if protocol in self.baseline_data:
                        baseline_rtt_val = self.baseline_data[protocol].get('avg_rtt_per_round', 0)
                        baseline_rtt = f"{baseline_rtt_val:.2f}s"
                    
                    # Current RTT
                    current_rtt = "Waiting..."
                    current_rtt_val = 0
                    current_round = 0
                    if protocol in self.current_rtt_data:
                        rtt_data = self.current_rtt_data[protocol]
                        current_round = rtt_data['current_round']
                        if rtt_data['avg_rtt'] > 0:
                            current_rtt_val = rtt_data['avg_rtt']
                            current_rtt = f"{current_rtt_val:.2f}s"
                    
                    # Degradation
                    degradation = "--"
                    if baseline_rtt_val > 0 and current_rtt_val > 0:
                        deg_pct = ((current_rtt_val - baseline_rtt_val) / baseline_rtt_val) * 100
                        degradation = f"{deg_pct:+.1f}%"
                    
                    # Round info
                    round_info = f"{current_round}" if current_round > 0 else "--"
                    
                    # Get current network conditions for this protocol
                    current_conditions = "None"
                    for client in self.client_containers:
                        if protocol in client.lower():
                            current_conditions = self.get_network_conditions(client)
                            break
                    
                    # Status indicator
                    status = "🟢 Ideal"
                    if current_conditions and current_conditions != "None":
                        if baseline_rtt_val > 0 and current_rtt_val > 0:
                            if current_rtt_val > baseline_rtt_val * 2.0:
                                status = "🔴 High Impact (>100% overhead)"
                            elif current_rtt_val > baseline_rtt_val * 1.5:
                                status = "🟠 Moderate Impact (>50% overhead)"
                            else:
                                status = "🟡 Low Impact (<50% overhead)"
                        else:
                            status = "🟡 Measuring..."
                    
                    print(f"{protocol.upper():<12} {baseline_rtt:<15} {current_rtt:<15} {degradation:<15} {round_info:<10} {current_conditions:<35} {status:<38}")
            
            print("-" * 140)
        
        # Server Status
        if self.server_containers:
            print("\n📊 SERVER STATUS")
            print("-" * 140)
            print(f"{'Container':<45} {'Protocol':<12} {'CPU':<10} {'Memory':<20} {'Network I/O':<25} {'Status':<10}")
            print("-" * 140)
            
            for server in self.server_containers:
                stats = self.get_container_stats(server)
                protocol = self.get_protocol_from_container(server)
                protocol_str = protocol.upper() if protocol else "N/A"
                print(f"{server:<45} {protocol_str:<12} {stats.get('cpu', 'N/A'):<10} {stats.get('memory', 'N/A'):<20} "
                      f"{stats.get('network', 'N/A'):<25} {'Running':<10}")
            
            print("-" * 140)
        
        # Client Status
        if self.client_containers:
            print("\n👥 CLIENT STATUS")
            print("-" * 140)
            print(f"{'Container':<40} {'Protocol':<12} {'CPU':<8} {'Memory':<18} {'Net I/O':<20} {'Network Conditions':<38}")
            print("-" * 140)
            
            for client in self.client_containers:
                stats = self.get_container_stats(client)
                net_cond = self.get_network_conditions(client)
                protocol = self.get_protocol_from_container(client)
                protocol_str = protocol.upper() if protocol else "N/A"
                
                print(f"{client:<40} {protocol_str:<12} {stats.get('cpu', 'N/A'):<8} {stats.get('memory', 'N/A'):<18} "
                      f"{stats.get('network', 'N/A'):<20} {net_cond:<38}")
            
            print("-" * 140)
        
        # Recent Activity
        print("\n📝 RECENT ACTIVITY (Last 2 lines per container)")
        print("-" * 140)
        
        for container in (self.server_containers + self.client_containers)[:5]:  # Show first 5 containers
            logs = self.get_container_logs_tail(container, 2)
            protocol = self.get_protocol_from_container(container)
            protocol_tag = f"[{protocol.upper()}] " if protocol else ""
            print(f"\n{protocol_tag}{container}:")
            for log in logs:
                # Truncate long logs
                log_display = log[:125] + "..." if len(log) > 125 else log
                print(f"  {log_display}")
        
        print("\n" + "-" * 140)
        print("\n💡 TIPS:")
        print("  • Baseline data loaded from: experiment_results_baseline/" + self.use_case)
        print("  • To change network conditions: python fl_network_monitor.py --client-id <NUM> --latency <VALUE>")
        print("  • To view detailed conditions: python fl_network_monitor.py --show-status")
        print("  • Run baseline first: python run_network_experiments.py --use-case " + self.use_case + " --baseline")
        print("  • Press Ctrl+C to stop monitoring")
        print("\n" + "=" * 140)
    
    def run(self):
        """Main monitoring loop"""
        print("Starting FL Training Dashboard...")
        print(f"Refresh interval: {self.interval} seconds")
        print("Press Ctrl+C to stop\n")
        
        time.sleep(2)
        
        try:
            while self.running:
                self.get_containers()
                self.display_dashboard()
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            print("\n\nDashboard stopped by user")
            self.running = False
        
        except Exception as e:
            print(f"\n\nError: {e}")
            self.running = False


def main():
    parser = argparse.ArgumentParser(
        description='FL Training Dashboard - Real-time monitoring with baseline comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--interval', type=int, default=5,
                       help='Refresh interval in seconds (default: 5)')
    parser.add_argument('-u', '--use-case',
                       choices=['emotion', 'mentalstate', 'temperature'],
                       default='emotion',
                       help='Use case for baseline comparison (default: emotion)')
    parser.add_argument('--minimal', action='store_true',
                       help='Minimal output mode (less detail)')
    
    args = parser.parse_args()
    
    dashboard = FLTrainingDashboard(interval=args.interval, use_case=args.use_case)
    dashboard.run()


if __name__ == '__main__':
    main()
