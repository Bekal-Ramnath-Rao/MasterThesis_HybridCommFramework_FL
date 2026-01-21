"""
Real-Time RL Protocol Selection Monitor
Connects to FL client output and visualizes protocol choices
"""

import re
import sys
import time
from collections import defaultdict, deque
from datetime import datetime

class RLMonitor:
    def __init__(self):
        self.protocol_counts = defaultdict(int)
        self.protocol_history = deque(maxlen=50)
        self.rewards = []
        self.episodes = []
        self.current_round = 0
        
    def parse_line(self, line):
        """Parse FL client output line"""
        # RL Selection
        if "[RL Selection] Selected Protocol:" in line:
            match = re.search(r'Selected Protocol: (\w+)', line)
            if match:
                protocol = match.group(1).lower()
                self.protocol_counts[protocol] += 1
                self.protocol_history.append(protocol)
                self.print_protocol_selection(protocol)
        
        # RL Reward
        elif "[RL] Reward:" in line:
            match = re.search(r'Reward: ([-\d.]+)', line)
            if match:
                reward = float(match.group(1))
                self.rewards.append(reward)
                self.print_reward(reward)
        
        # Round tracking
        elif "# ROUND" in line:
            match = re.search(r'# ROUND (\d+)/(\d+)', line)
            if match:
                self.current_round = int(match.group(1))
                print(f"\n{'='*80}")
                print(f"  ROUND {match.group(1)}/{match.group(2)}")
                print(f"{'='*80}")
        
        # State information
        elif "[RL Selection] State:" in line:
            match = re.search(r'State: \(([^)]+)\)', line)
            if match:
                state = match.group(1)
                print(f"  🎯 State: ({state})")
        
        # Training metrics
        elif "[Training]" in line and "MAE:" in line:
            match = re.search(r'Time: ([\d.]+)s, MAE: ([\d.]+)', line)
            if match:
                train_time = float(match.group(1))
                mae = float(match.group(2))
                print(f"  📊 Training: {train_time:.2f}s | MAE: {mae:.4f}")
    
    def print_protocol_selection(self, protocol):
        """Print protocol selection with emoji"""
        emoji = {
            'mqtt': '📨',
            'amqp': '🐰',
            'grpc': '⚡',
            'quic': '🚀',
            'dds': '📡'
        }
        print(f"\n  {emoji.get(protocol, '🔧')} PROTOCOL: {protocol.upper()}")
    
    def print_reward(self, reward):
        """Print reward with color indicator"""
        if reward > 15:
            indicator = "🟢 EXCELLENT"
        elif reward > 10:
            indicator = "🟡 GOOD"
        elif reward > 5:
            indicator = "🟠 FAIR"
        else:
            indicator = "🔴 POOR"
        print(f"  💰 Reward: {reward:+.2f} {indicator}")
    
    def print_statistics(self):
        """Print final statistics"""
        print(f"\n{'='*80}")
        print("RL PROTOCOL SELECTION STATISTICS")
        print(f"{'='*80}\n")
        
        if self.protocol_counts:
            total = sum(self.protocol_counts.values())
            print("Protocol Usage:")
            for protocol, count in sorted(self.protocol_counts.items(), key=lambda x: -x[1]):
                pct = (count / total) * 100
                bar = '█' * int(pct / 2)
                print(f"  {protocol.upper():6s}: {bar:50s} {count:3d} ({pct:5.1f}%)")
        
        if self.rewards:
            print(f"\nReward Statistics:")
            print(f"  Average: {sum(self.rewards)/len(self.rewards):+.2f}")
            print(f"  Maximum: {max(self.rewards):+.2f}")
            print(f"  Minimum: {min(self.rewards):+.2f}")
            print(f"  Total Rounds: {len(self.rewards)}")
        
        if len(self.protocol_history) > 0:
            recent = list(self.protocol_history)[-10:]
            print(f"\nLast 10 Selections: {' → '.join([p.upper() for p in recent])}")
        
        print(f"\n{'='*80}\n")


def main():
    """Monitor FL client output in real-time"""
    print("=" * 80)
    print("RL PROTOCOL SELECTION MONITOR")
    print("=" * 80)
    print("\nListening to FL client output...")
    print("(Pipe the FL client output to this script or run in parallel)\n")
    
    monitor = RLMonitor()
    
    try:
        for line in sys.stdin:
            line = line.strip()
            if line:
                monitor.parse_line(line)
                # Also print original line for full context
                if any(keyword in line for keyword in ['[Error]', '[Warning]', 'Failed']):
                    print(f"  ⚠️  {line}")
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    
    finally:
        monitor.print_statistics()


if __name__ == "__main__":
    main()
