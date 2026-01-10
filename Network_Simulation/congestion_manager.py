#!/usr/bin/env python3
"""
Network Congestion Manager
Manages traffic generator containers to create realistic network congestion
"""

import subprocess
import argparse
import time
from typing import List, Dict
from pathlib import Path
import yaml


class CongestionManager:
    """Manages Docker containers that generate network traffic for congestion testing"""
    
    # Congestion intensity levels with container name suffixes by use case
    CONGESTION_LEVELS = {
        "none": {
            "name": "No Congestion (Baseline)",
            "description": "No traffic generators running",
            "containers": []
        },
        "light": {
            "name": "Light Congestion",
            "description": "1-2 HTTP traffic generators",
            "containers": ["http-traffic-gen-1"]
        },
        "moderate": {
            "name": "Moderate Congestion", 
            "description": "HTTP + Bandwidth hog",
            "containers": ["http-traffic-gen-1", "http-traffic-gen-2", "bandwidth-hog"]
        },
        "heavy": {
            "name": "Heavy Congestion",
            "description": "All traffic generators + packet spammer",
            "containers": ["http-traffic-gen-1", "http-traffic-gen-2", "bandwidth-hog", "packet-spammer"]
        },
        "extreme": {
            "name": "Extreme Congestion",
            "description": "All traffic generators active",
            "containers": ["http-traffic-gen-1", "http-traffic-gen-2", "bandwidth-hog", 
                         "packet-spammer", "connection-flooder"]
        }
    }
    
    # Container name suffixes for each use case
    USE_CASE_SUFFIXES = {
        "temperature": "-temp",
        "emotion": "-emotion",
        "mentalstate": "-mental"
    }
    
    def __init__(self, use_case: str = "temperature", verbose: bool = False):
        self.use_case = use_case
        self.verbose = verbose
        self.suffix = self.USE_CASE_SUFFIXES.get(use_case, "-temp")
        
        # Get paths
        self.script_dir = Path(__file__).parent.absolute()
        self.project_root = self.script_dir.parent
        
        # Use the main compose file for the use case
        compose_file_map = {
            "temperature": "docker-compose-temperature.yml",
            "emotion": "docker-compose-emotion.yml",
            "mentalstate": "docker-compose-mentalstate.yml"
        }
        
        compose_filename = compose_file_map.get(use_case, "docker-compose-temperature.yml")
        self.compose_file = self.project_root / "Docker" / compose_filename
        
        if not self.compose_file.exists():
            raise FileNotFoundError(f"Compose file not found: {self.compose_file}")
    
    def log(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(f"[INFO] {message}")
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Execute a shell command"""
        self.log(f"Running: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    
    def start_traffic_generators(self, level: str = "moderate"):
        """Start traffic generator containers at specified congestion level"""
        if level not in self.CONGESTION_LEVELS:
            print(f"[ERROR] Unknown congestion level: {level}")
            print(f"Available levels: {', '.join(self.CONGESTION_LEVELS.keys())}")
            return False
        
        config = self.CONGESTION_LEVELS[level]
        containers = config["containers"]
        
        if not containers:
            print("No traffic generators needed for 'none' level")
            return True
        
        print(f"\n{'='*70}")
        print(f"Starting Traffic Generators")
        print(f"{'='*70}")
        print(f"Congestion Level: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Containers: {len(containers)}")
        print(f"{'='*70}\n")
        
        # Add suffix to container names for this use case
        containers_with_suffix = [f"{c}{self.suffix}" for c in containers]
        
        # Start containers using docker-compose with congestion profile
        print(f"Starting {len(containers)} traffic generator(s)...")
        for container_base, container in zip(containers, containers_with_suffix):
            print(f"  - {container}")
            result = self.run_command([
                "docker-compose", "-f", str(self.compose_file),
                "--profile", "congestion",
                "up", "-d", container_base + self.suffix
            ], check=False)
            
            if result.returncode != 0:
                print(f"[WARNING] Failed to start {container}")
            else:
                print(f"✓ Started: {container}")
            
            time.sleep(1)
        
        print(f"\n✓ Traffic generators started successfully")
        print(f"Congestion level: {config['name']}\n")
        return True
    
    def stop_traffic_generators(self, containers: List[str] = None):
        """Stop traffic generator containers"""
        print("\nStopping traffic generators...")
        
        if containers:
            # Add suffix if not already present
            containers_with_suffix = []
            for c in containers:
                if not c.endswith(self.suffix):
                    containers_with_suffix.append(f"{c}{self.suffix}")
                else:
                    containers_with_suffix.append(c)
            
            # Stop specific containers
            for container in containers_with_suffix:
                self.run_command([
                    "docker-compose", "-f", str(self.compose_file),
                    "stop", container
                ], check=False)
                self.run_command([
                    "docker-compose", "-f", str(self.compose_file),
                    "rm", "-f", container
                ], check=False)
        else:
            # Stop all traffic generators using profile
            self.run_command([
                "docker-compose", "-f", str(self.compose_file),
                "--profile", "congestion",
                "down"
            ], check=False)
        
        print("✓ Traffic generators stopped\n")
    
    def get_traffic_stats(self):
        """Get statistics from running traffic generators"""
        print("\n" + "="*70)
        print("Traffic Generator Status")
        print("="*70 + "\n")
        
        all_containers = []
        for level_config in self.CONGESTION_LEVELS.values():
            all_containers.extend(level_config["containers"])
        
        all_containers = list(set(all_containers))  # Remove duplicates
        
        # Add suffix to container names
        for container_base in all_containers:
            container = f"{container_base}{self.suffix}"
            result = self.run_command([
                "docker", "ps", "--filter", f"name={container}",
                "--format", "{{.Names}} - {{.Status}}"
            ], check=False)
            
            if result.stdout.strip():
                print(f"✓ {result.stdout.strip()}")
            else:
                print(f"✗ {container} - Not running")
        
        print("\n" + "="*70 + "\n")
    
    def show_levels(self):
        """Display all available congestion levels"""
        print("\n" + "="*70)
        print("Available Congestion Levels")
        print("="*70 + "\n")
        
        for level, config in self.CONGESTION_LEVELS.items():
            print(f"Level: {level}")
            print(f"  Name: {config['name']}")
            print(f"  Description: {config['description']}")
            print(f"  Containers: {len(config['containers'])}")
            if config['containers']:
                print(f"    - {', '.join(config['containers'])}")
            print()
    
    def scale_congestion(self, from_level: str, to_level: str):
        """Transition from one congestion level to another"""
        if from_level not in self.CONGESTION_LEVELS or to_level not in self.CONGESTION_LEVELS:
            print("[ERROR] Invalid congestion level")
            return False
        
        from_containers = set(self.CONGESTION_LEVELS[from_level]["containers"])
        to_containers = set(self.CONGESTION_LEVELS[to_level]["containers"])
        
        # Stop containers that are not needed
        to_stop = from_containers - to_containers
        if to_stop:
            print(f"Stopping: {', '.join(to_stop)}")
            self.stop_traffic_generators(list(to_stop))
        
        # Start new containers
        to_start = to_containers - from_containers
        if to_start:
            print(f"Starting: {', '.join(to_start)}")
            for container in to_start:
                self.run_command([
                    "docker-compose", "-f", str(self.compose_file),
                    "up", "-d", container
                ], check=False)
                time.sleep(1)
        
        print(f"\n✓ Scaled from {from_level} to {to_level}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Network Congestion Manager for FL Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start moderate congestion
  python congestion_manager.py --start --level moderate

  # Stop all traffic generators
  python congestion_manager.py --stop

  # Check status
  python congestion_manager.py --status

  # List available levels
  python congestion_manager.py --list

  # Scale from light to heavy congestion
  python congestion_manager.py --scale-from light --scale-to heavy
        """
    )
    
    parser.add_argument("--start", "-s", action="store_true",
                       help="Start traffic generators")
    parser.add_argument("--stop", action="store_true",
                       help="Stop traffic generators")
    parser.add_argument("--status", action="store_true",
                       help="Show traffic generator status")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available congestion levels")
    parser.add_argument("--level", "-L",
                       choices=["none", "light", "moderate", "heavy", "extreme"],
                       default="moderate",
                       help="Congestion level (default: moderate)")
    parser.add_argument("--use-case", "-u",
                       choices=["emotion", "mentalstate", "temperature"],
                       default="temperature",
                       help="Use case (default: temperature)")
    parser.add_argument("--scale-from",
                       choices=["none", "light", "moderate", "heavy", "extreme"],
                       help="Scale from this congestion level")
    parser.add_argument("--scale-to",
                       choices=["none", "light", "moderate", "heavy", "extreme"],
                       help="Scale to this congestion level")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    manager = CongestionManager(use_case=args.use_case, verbose=args.verbose)
    
    if args.list:
        manager.show_levels()
    elif args.status:
        manager.get_traffic_stats()
    elif args.stop:
        manager.stop_traffic_generators()
    elif args.scale_from and args.scale_to:
        manager.scale_congestion(args.scale_from, args.scale_to)
    elif args.start:
        manager.start_traffic_generators(args.level)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
