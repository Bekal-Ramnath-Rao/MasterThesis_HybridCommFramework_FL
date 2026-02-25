#!/usr/bin/env python3
"""
Network Simulation Tool for Docker Containers
Applies network conditions (latency, bandwidth, packet loss, jitter) to running containers
"""

import subprocess
import argparse
import json
import time
import os
from typing import Dict, List

# Optional: set FL_SUDO_PASSWORD in environment to run host 'tc' without typing password.
# Example (do NOT commit .env): echo 'FL_SUDO_PASSWORD=yourpassword' >> .env && source .env
# Or for one run: FL_SUDO_PASSWORD=yourpassword python3 run_network_experiments.py ...
#
# Macvlan (when using host-network mode for experiments/diagnostics): set FL_MACVLAN_PARENT
# to the host interface to use (e.g. eno2). If unset, the host's default route interface is used.

MACVLAN_NETWORK_NAME = "fl-macvlan"
MACVLAN_SUBNET = os.environ.get("FL_MACVLAN_SUBNET", "192.168.254.0/24")
MACVLAN_GATEWAY = os.environ.get("FL_MACVLAN_GATEWAY", "192.168.254.1")


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
        self._host_interface_used = None  # set when we apply tc on host (host-network mode)
        
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

    def is_host_network_mode(self, container_name: str) -> bool:
        """Return True if the container uses network_mode: host (no per-container eth0/tc)."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.HostConfig.NetworkMode}}", container_name],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip().lower() == "host":
                return True
        except Exception:
            pass
        return False

    def _run_host_tc(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run tc on the host; if permission denied, retry with sudo. Uses FL_SUDO_PASSWORD if set."""
        r = subprocess.run(
            ["tc"] + args,
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return r
        err = (r.stderr or r.stdout or "").lower()
        if "permission denied" in err or "rtnetlink" in err or "operation not permitted" in err:
            sudo_pw = os.environ.get("FL_SUDO_PASSWORD", "").strip()
            if sudo_pw:
                self.log("Retrying with sudo (using FL_SUDO_PASSWORD)...")
                return subprocess.run(
                    ["sudo", "-S", "tc"] + args,
                    input=sudo_pw + "\n",
                    capture_output=True, text=True, timeout=15
                )
            self.log("Retrying with sudo (you may be prompted for your password)...")
            return subprocess.run(
                ["sudo", "tc"] + args,
                capture_output=True, text=True, timeout=15
            )
        return r

    def get_host_default_interface(self) -> str:
        """Return the host's default route interface (e.g. eth0, ens33). Used for host-network tc."""
        try:
            r = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0 or not r.stdout:
                return "eth0"
            parts = r.stdout.strip().split()
            if "dev" in parts:
                idx = parts.index("dev")
                if idx + 1 < len(parts):
                    return parts[idx + 1]
            return "eth0"
        except Exception:
            return "eth0"

    def get_macvlan_parent_interface(self) -> str:
        """Return the host interface to use as macvlan parent (e.g. eno2). Uses FL_MACVLAN_PARENT if set."""
        return os.environ.get("FL_MACVLAN_PARENT", "").strip() or self.get_host_default_interface()

    def ensure_macvlan_network(
        self,
        parent_interface: str = None,
        network_name: str = None,
        subnet: str = None,
        gateway: str = None,
    ) -> bool:
        """Create Docker macvlan network if it does not exist. Used when running experiments/diagnostics
        in 'host network' mode so tc can be applied per container instead of on the host interface.
        Returns True if the network exists or was created, False on failure."""
        parent_interface = parent_interface or self.get_macvlan_parent_interface()
        network_name = network_name or MACVLAN_NETWORK_NAME
        subnet = subnet or MACVLAN_SUBNET
        gateway = gateway or MACVLAN_GATEWAY
        try:
            r = subprocess.run(
                ["docker", "network", "inspect", network_name],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                self.log(f"Macvlan network '{network_name}' already exists")
                return True
            self.log(f"Creating macvlan network '{network_name}' (parent={parent_interface}, subnet={subnet}, gateway={gateway})")
            r = subprocess.run(
                [
                    "docker", "network", "create", "-d", "macvlan",
                    "-o", f"parent={parent_interface}",
                    "--subnet", subnet,
                    "--gateway", gateway,
                    network_name,
                ],
                capture_output=True, text=True, timeout=15
            )
            if r.returncode != 0:
                print(f"[ERROR] Failed to create macvlan network: {r.stderr or r.stdout}")
                return False
            print(f"[INFO] Created macvlan network '{network_name}' on {parent_interface}")
            return True
        except Exception as e:
            print(f"[ERROR] ensure_macvlan_network: {e}")
            return False

    def remove_macvlan_network(self, network_name: str = None) -> bool:
        """Remove the Docker macvlan network. Optional cleanup after experiments."""
        network_name = network_name or MACVLAN_NETWORK_NAME
        try:
            r = subprocess.run(
                ["docker", "network", "rm", network_name],
                capture_output=True, text=True, timeout=10
            )
            if r.returncode == 0:
                self.log(f"Removed macvlan network '{network_name}'")
                return True
            if "no such network" in (r.stderr or "").lower():
                return True
            self.log(f"Could not remove network '{network_name}': {r.stderr}")
            return False
        except Exception as e:
            self.log(f"remove_macvlan_network: {e}")
            return False

    def apply_network_conditions_host(self, conditions: Dict[str, str]) -> bool:
        """Apply network scenario to the host's default interface (for network_mode: host).
        All host-network containers (server and all clients) share this interface, so the same
        delay/loss/jitter applies to every packet between server and clients. Requires tc on
        the host (often needs root). Returns True if applied, False otherwise.

        Uses HTB when both netem (delay/loss) and bandwidth are needed: netem is classless so
        it does not expose a 1:1 class; we use root handle 1: htb, class 1:1 with rate, then
        netem as child of 1:1."""
        interface = self.get_host_default_interface()
        self._host_interface_used = interface
        netem_params = []
        if conditions.get("latency"):
            netem_params.append(f"delay {conditions['latency']}")
            if conditions.get("jitter"):
                netem_params.append(conditions["jitter"])
        if conditions.get("loss"):
            netem_params.append(f"loss {conditions['loss']}")
        has_netem = len(netem_params) > 0
        has_bandwidth = bool(conditions.get("bandwidth"))
        try:
            # Reset any existing root qdisc on host interface (uses sudo if needed)
            self._run_host_tc(["qdisc", "del", "dev", interface, "root"])
            if has_netem and has_bandwidth:
                # HTB at root (classful), one class 1:1 with rate, netem as child (netem is classless, no 1:1)
                r = self._run_host_tc(
                    ["qdisc", "add", "dev", interface, "root", "handle", "1:", "htb", "default", "1"]
                )
                if r.returncode != 0:
                    raise RuntimeError(r.stderr or r.stdout or "tc htb root failed")
                r2 = self._run_host_tc([
                    "class", "add", "dev", interface, "parent", "1:", "classid", "1:1", "htb",
                    "rate", conditions["bandwidth"], "ceil", conditions["bandwidth"]
                ])
                if r2.returncode != 0:
                    raise RuntimeError(r2.stderr or r2.stdout or "tc htb class failed")
                netem_args = " ".join(netem_params).split()
                r3 = self._run_host_tc(
                    ["qdisc", "add", "dev", interface, "parent", "1:1", "handle", "10:", "netem", *netem_args]
                )
                if r3.returncode != 0:
                    raise RuntimeError(r3.stderr or r3.stdout or "tc netem failed")
            elif has_netem:
                netem_cmd = " ".join(netem_params)
                r = self._run_host_tc(
                    ["qdisc", "add", "dev", interface, "root", "netem", *netem_cmd.split()]
                )
                if r.returncode != 0:
                    raise RuntimeError(r.stderr or r.stdout or "tc failed")
            elif has_bandwidth:
                r = self._run_host_tc([
                    "qdisc", "add", "dev", interface, "root", "tbf",
                    "rate", conditions["bandwidth"], "burst", "32kbit", "latency", "400ms"
                ])
                if r.returncode != 0:
                    raise RuntimeError(r.stderr or r.stdout or "tc failed")
            else:
                return False
            if has_netem:
                print(f"  Host interface {interface}: netem {', '.join(netem_params)}")
            if has_bandwidth:
                print(f"  Host interface {interface}: rate {conditions['bandwidth']}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to apply host network conditions: {e}")
            print(f"[INFO] If prompted, enter your sudo password. Running from GUI? Use a terminal with sudo, or add to sudoers: youruser ALL=(ALL) NOPASSWD: /usr/sbin/tc")
            self._host_interface_used = None
            return False

    def reset_host_network(self) -> bool:
        """Remove tc qdisc from the host interface that was used in apply_network_conditions_host."""
        if not self._host_interface_used:
            return True
        try:
            r = self._run_host_tc(["qdisc", "del", "dev", self._host_interface_used, "root"])
            if r.returncode == 0:
                self.log(f"Reset host tc on {self._host_interface_used}")
            self._host_interface_used = None
            return True
        except Exception as e:
            print(f"[WARNING] Could not reset host tc: {e}")
            self._host_interface_used = None
            return False

    def show_host_tc(self, interface: str = None) -> str:
        """Show current tc qdisc (and class if HTB) on the host. Use to verify tc was applied.
        Returns the combined output; pass interface or uses the one we last used for host tc."""
        iface = interface or self._host_interface_used or self.get_host_default_interface()
        out = []
        r = self._run_host_tc(["qdisc", "show", "dev", iface])
        out.append(f"=== tc qdisc show dev {iface} ===\n" + (r.stdout or r.stderr or ""))
        if r.returncode == 0 and "htb" in (r.stdout or ""):
            r2 = self._run_host_tc(["class", "show", "dev", iface])
            out.append(f"\n=== tc class show dev {iface} ===\n" + (r2.stdout or r2.stderr or ""))
        return "\n".join(out).strip()

    def show_container_tc(self, container_name: str, interface: str = "eth0") -> str:
        """Show current tc qdisc on a container's interface. Use to verify tc was applied."""
        try:
            r = subprocess.run(
                ["docker", "exec", container_name, "tc", "qdisc", "show", "dev", interface],
                capture_output=True, text=True, timeout=5
            )
            out = (r.stdout or r.stderr or "").strip()
            if r.returncode == 0 and "htb" in out:
                r2 = subprocess.run(
                    ["docker", "exec", container_name, "tc", "class", "show", "dev", interface],
                    capture_output=True, text=True, timeout=5
                )
                out += "\n" + (r2.stdout or r2.stderr or "").strip()
            return f"=== tc (qdisc/class) show dev {interface} in {container_name} ===\n" + out
        except Exception as e:
            return f"[ERROR] Could not show tc for {container_name}: {e}"

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
        """Apply network conditions to a specific container.
        
        Uses a hierarchical qdisc setup when both netem (latency/loss/jitter) and
        bandwidth limiting are needed: netem is classless (no 1:1), so we use
          root -> htb (handle 1:) -> class 1:1 (rate) -> netem (handle 10:)
        This avoids invalid 'parent 1:1' under netem.
        """
        try:
            print(f"\n{'='*60}")
            print(f"Applying network conditions to: {container_name}")
            print(f"{'='*60}")
            
            # With network_mode: host, container shares host network; no per-container eth0/tc
            if self.is_host_network_mode(container_name):
                print(f"[INFO] Container {container_name} uses network_mode: host.")
                print(f"[INFO] Skipping tc (no per-container interface); scenario is nominal only.")
                print(f"{'='*60}\n")
                return True

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
            
            has_netem = len(netem_params) > 0
            has_bandwidth = bool(conditions.get("bandwidth"))
            
            if has_netem and has_bandwidth:
                # Hierarchical setup: htb root, class 1:1 with rate, netem as child (netem is classless)
                self.run_command([
                    "docker", "exec", container_name,
                    "sh", "-c",
                    f"tc qdisc add dev {interface} root handle 1: htb default 1"
                ])
                self.run_command([
                    "docker", "exec", container_name,
                    "sh", "-c",
                    f"tc class add dev {interface} parent 1: classid 1:1 htb "
                    f"rate {conditions['bandwidth']} ceil {conditions['bandwidth']}"
                ])
                netem_cmd = " ".join(netem_params)
                self.run_command([
                    "docker", "exec", container_name,
                    "sh", "-c",
                    f"tc qdisc add dev {interface} parent 1:1 handle 10: netem {netem_cmd}"
                ])
                print(f"  Applied htb+netem: rate {conditions['bandwidth']}, netem {netem_cmd}")
                
            elif has_netem:
                # Only netem (latency/loss/jitter), no bandwidth cap
                netem_cmd = " ".join(netem_params)
                self.run_command([
                    "docker", "exec", container_name,
                    "sh", "-c",
                    f"tc qdisc add dev {interface} root netem {netem_cmd}"
                ])
                print(f"  Applied netem: {netem_cmd}")
                
            elif has_bandwidth:
                # Only bandwidth limit, no netem
                self.run_command([
                    "docker", "exec", container_name,
                    "sh", "-c", 
                    f"tc qdisc add dev {interface} root tbf rate {conditions['bandwidth']} "
                    f"burst 32kbit latency 400ms"
                ])
                print(f"  Applied tbf: rate {conditions['bandwidth']}")
            
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

  # Verify tc is applied (host or containers)
  python network_simulator.py --show-tc --host
  python network_simulator.py --show-tc --pattern fl-client
        """
    )
    
    parser.add_argument("--show-tc", action="store_true",
                       help="Show current tc qdisc/class on interface (use --host for host, else containers matching --pattern)")
    parser.add_argument("--host", action="store_true",
                       help="With --show-tc: show tc on host default interface (for host-network mode)")
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
    
    if args.show_tc:
        if args.host:
            print(sim.show_host_tc())
        else:
            result = sim.run_command(["docker", "ps", "--format", "{{.Names}}"], check=False)
            containers = [c for c in result.stdout.strip().split("\n") if c and (not args.pattern or args.pattern in c)]
            if not containers:
                print("[WARNING] No matching containers. Use --pattern to match, or --host for host interface.")
            for c in containers:
                print(sim.show_container_tc(c))
    elif args.list:
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
