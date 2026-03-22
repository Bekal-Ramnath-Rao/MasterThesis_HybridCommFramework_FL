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
import random
from typing import Dict, List

from network_delay_model import (
    build_delay_models,
    sample_delay_and_jitter_ms,
    NetworkScenarioModel,
)

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

    # Predefined network scenarios. All four keys (latency, jitter, bandwidth, loss) are applied
    # verbatim to tc: netem for delay/jitter/loss, htb or tbf for bandwidth.
    NETWORK_SCENARIOS = {
        "excellent": {
            "name": "Excellent Network (LAN)",
            "latency": "2ms",
            "jitter": "1ms",
            "bandwidth": "100mbit",
            "loss": "0.01%"
        },
        "good": {
            "name": "Good Network (Broadband)",
            "latency": "20ms",
            "jitter": "5ms",
            "bandwidth": "100mbit",
            "loss": "0.1%"
        },
        "moderate": {
            "name": "Moderate Network (4G/LTE)",
            "latency": "30ms",
            "jitter": "10ms",
            "bandwidth": "10mbit",
            "loss": "0.3%"
        },
        "poor": {
            "name": "Poor Network (3G)",
            "latency": "120ms",
            "jitter": "30ms",
            "bandwidth": "1mbit",
            "loss": "0.5%"
        },
        "very_poor": {
            "name": "Very Poor Network (Edge/2G)",
            "latency": "400ms",
            "jitter": "100ms",
            "bandwidth": "75kbit",
            "loss": "1%"
        },
        "satellite": {
            "name": "Satellite Network",
            "latency": "600ms",
            "jitter": "50ms",
            "bandwidth": "5mbit",
            "loss": "1.5%"
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
        },
        # Dynamic scenario: randomly selects one of these base scenarios
        # at each application (e.g., per FL round) so that the effective
        # network alternates between excellent / moderate / poor / congested_light.
        "dynamic": {
            "name": "Dynamic Network (Random Excellent/Moderate/Poor/Congested-Light)",
            # Latency/jitter values here are placeholders and are not used
            # when Gaussian sampling + per-round resampling is enabled.
            # Bandwidth/loss will be taken from the randomly chosen base scenario.
            "latency": "30ms",
            "jitter": "10ms",
            "bandwidth": "10mbit",
            "loss": "0.3%"
        },
    }

    # Keys passed to tc (netem: delay/jitter/loss; htb/tbf: bandwidth)
    _TC_CONDITION_KEYS = ("latency", "jitter", "bandwidth", "loss")

    @classmethod
    def get_scenario_conditions(cls, scenario_name: str) -> Dict[str, str]:
        """
        Return the tc-relevant conditions (latency, jitter, bandwidth, loss) for a scenario.
        Ensures all four values from NETWORK_SCENARIOS are used when applying tc.
        """
        if scenario_name not in cls.NETWORK_SCENARIOS:
            raise KeyError(f"Unknown scenario: {scenario_name}")
        scenario = cls.NETWORK_SCENARIOS[scenario_name]
        conditions = {k: scenario[k] for k in cls._TC_CONDITION_KEYS if k in scenario}
        missing = [k for k in cls._TC_CONDITION_KEYS if k not in conditions]
        if missing:
            raise ValueError(
                f"Scenario '{scenario_name}' is missing tc parameters: {missing}. "
                "Each scenario must define latency, jitter, bandwidth, and loss."
            )
        return conditions

    @classmethod
    def get_scenario_conditions_sampled(
        cls,
        scenario_name: str,
        models: Dict[str, NetworkScenarioModel],
        use_extra_jitter: bool = False,
    ) -> Dict[str, str]:
        """
        Return tc-relevant conditions with latency and jitter sampled from Gaussian models.
        Delay and jitter follow normal distributions for realistic experiments.
        """
        if scenario_name not in cls.NETWORK_SCENARIOS:
            raise KeyError(f"Unknown scenario: {scenario_name}")

        # Special handling for "dynamic": randomly pick one of the base
        # scenarios and return sampled conditions for that scenario.
        if scenario_name == "dynamic":
            dynamic_bases = ["excellent", "moderate", "poor", "congested_light"]
            chosen = random.choice(dynamic_bases)
        else:
            chosen = scenario_name

        base_ms, jitter_ms = sample_delay_and_jitter_ms(models, chosen, use_extra_jitter=use_extra_jitter)
        scenario = cls.NETWORK_SCENARIOS[chosen]
        conditions = {k: scenario[k] for k in cls._TC_CONDITION_KEYS if k in scenario}
        conditions["latency"] = f"{base_ms:.2f}ms"
        conditions["jitter"] = f"{jitter_ms:.2f}ms"
        return conditions

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._host_interface_used = None  # set when we apply tc on host (host-network mode)
        self._host_ifb_used = None  # set when we apply ingress tc on host (ifb device name)
        
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

    def _run_host_ip(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run ip on the host; if permission denied, retry with sudo. Uses FL_SUDO_PASSWORD if set."""
        r = subprocess.run(
            ["ip"] + args,
            capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return r
        err = (r.stderr or r.stdout or "").lower()
        if "permission denied" in err or "rtnetlink" in err or "operation not permitted" in err:
            sudo_pw = os.environ.get("FL_SUDO_PASSWORD", "").strip()
            if sudo_pw:
                self.log("Retrying ip with sudo (using FL_SUDO_PASSWORD)...")
                return subprocess.run(
                    ["sudo", "-S", "ip"] + args,
                    input=sudo_pw + "\n",
                    capture_output=True, text=True, timeout=15
                )
            self.log("Retrying ip with sudo (you may be prompted for your password)...")
            return subprocess.run(
                ["sudo", "ip"] + args,
                capture_output=True, text=True, timeout=15
            )
        return r

    def _tc_egress_commands(self, device: str, conditions: Dict[str, str]) -> List[List[str]]:
        """Return list of tc command arg lists (without 'tc' prefix) for egress-style shaping on device.
        Used for root (egress) and for ifb (ingress mirror)."""
        netem_params: List[str] = []
        if conditions.get("latency"):
            netem_params.append("delay")
            netem_params.append(conditions["latency"])
            if conditions.get("jitter"):
                netem_params.append(conditions["jitter"])
        if conditions.get("loss"):
            netem_params.append("loss")
            netem_params.append(conditions["loss"])
        has_netem = len(netem_params) > 0
        has_bandwidth = bool(conditions.get("bandwidth"))
        if has_netem and has_bandwidth:
            return [
                ["qdisc", "add", "dev", device, "root", "handle", "1:", "htb", "default", "1"],
                ["class", "add", "dev", device, "parent", "1:", "classid", "1:1", "htb",
                 "rate", conditions["bandwidth"], "ceil", conditions["bandwidth"]],
                ["qdisc", "add", "dev", device, "parent", "1:1", "handle", "10:", "netem"] + netem_params,
            ]
        if has_netem:
            return [["qdisc", "add", "dev", device, "root", "netem"] + netem_params]
        if has_bandwidth:
            return [["qdisc", "add", "dev", device, "root", "tbf", "rate", conditions["bandwidth"],
                     "burst", "32kbit", "latency", "400ms"]]
        return []

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
        The same delay, jitter, loss, and bandwidth are applied to both egress and ingress
        on the host interface so that traffic in both directions sees identical shaping.
        All host-network containers share this interface. Requires tc on the host (often root).

        Egress: root qdisc on interface; ingress: same conditions via IFB mirror.
        Uses HTB when both netem and bandwidth are needed."""
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
                print(f"  Host interface {interface}: netem (egress) {', '.join(netem_params)}")
            if has_bandwidth:
                print(f"  Host interface {interface}: rate {conditions['bandwidth']}")

            # Ingress: same delay/jitter/loss/bandwidth on incoming traffic (via IFB)
            ifb_name = "ifb_fl"
            try:
                self._run_host_tc(["qdisc", "del", "dev", interface, "ingress"])
                self._run_host_tc(["qdisc", "del", "dev", ifb_name, "root"])
                self._run_host_ip(["link", "set", ifb_name, "down"])
                self._run_host_ip(["link", "del", ifb_name])
            except Exception:
                pass
            rip = self._run_host_ip(["link", "add", "name", ifb_name, "type", "ifb"])
            if rip.returncode != 0:
                self.log(f"Host ingress (ifb): ip link add failed: {rip.stderr or rip.stdout}; continuing with egress only.")
            else:
                self._run_host_ip(["link", "set", ifb_name, "up"])
                rinq = self._run_host_tc(["qdisc", "add", "dev", interface, "ingress"])
                if rinq.returncode != 0:
                    self.log(f"Host ingress: tc qdisc add ingress failed; continuing with egress only.")
                else:
                    rfil = self._run_host_tc([
                        "filter", "add", "dev", interface, "parent", "ffff:", "protocol", "all",
                        "u32", "match", "u32", "0", "0", "flowid", "1:1",
                        "action", "mirred", "egress", "redirect", "dev", ifb_name
                    ])
                    if rfil.returncode != 0:
                        self.log(f"Host ingress: tc filter failed; continuing with egress only.")
                    else:
                        for cmd in self._tc_egress_commands(ifb_name, conditions):
                            rr = self._run_host_tc(cmd)
                            if rr.returncode != 0:
                                self.log(f"Host ingress (ifb): tc {' '.join(cmd)} failed: {rr.stderr or rr.stdout}")
                                break
                        else:
                            self._host_ifb_used = ifb_name
                            print(f"  Host {interface}: same delay/jitter/loss on egress and ingress (ingress via {ifb_name})")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to apply host network conditions: {e}")
            print(f"[INFO] If prompted, enter your sudo password. Running from GUI? Use a terminal with sudo, or add to sudoers: youruser ALL=(ALL) NOPASSWD: /usr/sbin/tc")
            self._host_interface_used = None
            self._host_ifb_used = None
            return False

    def reset_host_network(self) -> bool:
        """Remove tc qdisc from the host interface (egress and ingress) that was used in apply_network_conditions_host."""
        iface = self._host_interface_used
        ifb_name = self._host_ifb_used
        try:
            if ifb_name and iface:
                self._run_host_tc(["qdisc", "del", "dev", iface, "ingress"])
                self._run_host_tc(["qdisc", "del", "dev", ifb_name, "root"])
                self._run_host_ip(["link", "set", ifb_name, "down"])
                self._run_host_ip(["link", "del", ifb_name])
                self.log(f"Reset host ingress tc ({ifb_name})")
            if iface:
                r = self._run_host_tc(["qdisc", "del", "dev", iface, "root"])
                if r.returncode == 0:
                    self.log(f"Reset host tc on {iface}")
            self._host_interface_used = None
            self._host_ifb_used = None
            return True
        except Exception as e:
            print(f"[WARNING] Could not reset host tc: {e}")
            self._host_interface_used = None
            self._host_ifb_used = None
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
        
        The same delay, jitter, loss, and bandwidth are applied to both egress and
        ingress on the container interface so that traffic in both directions sees
        identical shaping.
        
        Uses a hierarchical qdisc setup when both netem (latency/loss/jitter) and
        bandwidth limiting are needed: netem is classless (no 1:1), so we use
          root -> htb (handle 1:) -> class 1:1 (rate) -> netem (handle 10:)
        This avoids invalid 'parent 1:1' under netem.
        Egress: root qdisc on interface; ingress: same conditions via IFB mirror.
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
            
            # Egress: apply delay/jitter/loss (and optionally bandwidth) on the interface root
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

            # Ingress: apply the same delay/jitter/loss/bandwidth to incoming traffic (via IFB mirror)
            ifb_name = "ifb0"
            if has_netem or has_bandwidth:
                try:
                    self.run_command(
                        ["docker", "exec", container_name, "tc", "qdisc", "del", "dev", interface, "ingress"],
                        check=False,
                    )
                    self.run_command(
                        ["docker", "exec", container_name, "tc", "qdisc", "del", "dev", ifb_name, "root"],
                        check=False,
                    )
                    self.run_command(
                        ["docker", "exec", container_name, "ip", "link", "set", ifb_name, "down"],
                        check=False,
                    )
                    self.run_command(
                        ["docker", "exec", container_name, "ip", "link", "del", ifb_name],
                        check=False,
                    )
                except Exception:
                    pass
                try:
                    r = self.run_command(
                        ["docker", "exec", container_name, "ip", "link", "add", "name", ifb_name, "type", "ifb"],
                        check=False,
                    )
                    if r.returncode != 0:
                        self.log(f"Container {container_name}: ip link add ifb failed (ingress skipped): {getattr(r, 'stderr', '') or getattr(r, 'stdout', '')}")
                    else:
                        self.run_command(
                            ["docker", "exec", container_name, "ip", "link", "set", ifb_name, "up"],
                            check=False,
                        )
                        self.run_command(
                            ["docker", "exec", container_name, "tc", "qdisc", "add", "dev", interface, "ingress"],
                            check=False,
                        )
                        self.run_command(
                            [
                                "docker", "exec", container_name, "tc", "filter", "add", "dev", interface,
                                "parent", "ffff:", "protocol", "all", "u32", "match", "u32", "0", "0",
                                "flowid", "1:1", "action", "mirred", "egress", "redirect", "dev", ifb_name,
                            ],
                            check=False,
                        )
                        for cmd in self._tc_egress_commands(ifb_name, conditions):
                            r = self.run_command(
                                ["docker", "exec", container_name, "tc"] + cmd,
                                check=False,
                            )
                            if r.returncode != 0:
                                self.log(f"Container {container_name}: ingress tc cmd failed: {r.stderr or r.stdout}")
                        print(f"  Same delay/jitter/loss on egress and ingress (ingress via {ifb_name})")
                except Exception as e:
                    self.log(f"Container {container_name}: ingress tc failed: {e}; egress only.")
                    print(f"  [WARNING] Ingress tc failed; same delay applied to egress only.")
            
            print(f"{'='*60}\n")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to apply network conditions to {container_name}: {e}")
            return False
    
    def reset_container_network(self, container_name: str):
        """Reset network conditions on a container (egress and ingress)."""
        try:
            interface = self.get_container_interface(container_name)
            # Delete ingress and IFB first, then egress
            self.run_command(
                ["docker", "exec", container_name, "tc", "qdisc", "del", "dev", interface, "ingress"],
                check=False,
            )
            self.run_command(
                ["docker", "exec", container_name, "tc", "qdisc", "del", "dev", "ifb0", "root"],
                check=False,
            )
            self.run_command(
                ["docker", "exec", container_name, "ip", "link", "set", "ifb0", "down"],
                check=False,
            )
            self.run_command(
                ["docker", "exec", container_name, "ip", "link", "del", "ifb0"],
                check=False,
            )
            self.run_command([
                "docker", "exec", container_name,
                "sh", "-c", f"tc qdisc del dev {interface} root"
            ], check=False)
            self.log(f"Reset network conditions for {container_name}")
        except Exception as e:
            self.log(f"Note: Could not reset {container_name} (may not have existing rules): {e}")
    
    def apply_scenario_to_containers(self, scenario_name: str, container_pattern: str = None):
        """Apply a predefined scenario to all matching containers. Uses latency, jitter, bandwidth,
        and loss from NETWORK_SCENARIOS for all tc commands."""
        try:
            conditions = self.get_scenario_conditions(scenario_name)
        except KeyError as e:
            print(f"[ERROR] {e}")
            print(f"Available scenarios: {', '.join(self.NETWORK_SCENARIOS.keys())}")
            return
        except ValueError as e:
            print(f"[ERROR] {e}")
            return

        scenario = self.NETWORK_SCENARIOS[scenario_name]
        print(f"\n{'#'*60}")
        print(f"# Applying Scenario: {scenario['name']}")
        print(f"#{'#'*59}")
        print(f"# Latency: {conditions['latency']}")
        print(f"# Jitter: {conditions['jitter']}")
        print(f"# Bandwidth: {conditions['bandwidth']}")
        print(f"# Packet Loss: {conditions['loss']}")
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

        # Apply conditions (latency, jitter, bandwidth, loss) to each container via tc
        success_count = 0
        for container in containers:
            if container:  # Skip empty lines
                if self.apply_network_conditions(container, conditions):
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


class NamespaceEndpoint:
    """Represents one endpoint (server or client) in a Linux network namespace topology."""

    def __init__(self, name: str, ns_name: str, veth_ns: str, veth_br: str, ip: str):
        self.name = name
        self.ns_name = ns_name
        self.veth_ns = veth_ns
        self.veth_br = veth_br
        self.ip = ip


class NamespaceNetworkSimulator:
    """
    Network simulator for native Python processes using Linux network namespaces + veth pairs.

    This is designed to complement the Docker-based NetworkSimulator by providing a way to:
      - Run FL server and clients as normal Python scripts on the SAME host
      - Isolate them in separate network namespaces
      - Connect them through a bridge using veth pairs
      - Apply tc/netem independently on each namespace interface (per-direction shaping)

    Typical usage (from another script):
        sim = NamespaceNetworkSimulator(verbose=True)
        server_ep, client_eps = sim.setup_topology("grpc-emotion", num_clients=2)
        # Downstream (server -> clients)
        sim.apply_conditions(
            server_ep,
            client_eps,
            downstream_conditions=NetworkSimulator.NETWORK_SCENARIOS["poor"],
            upstream_conditions=NetworkSimulator.NETWORK_SCENARIOS["excellent"],
        )
        # Launch processes with ip netns exec server_ep.ns_name / client_ep.ns_name ...
        ...
        sim.cleanup([server_ep] + client_eps)
    """

    def __init__(
        self,
        bridge_name: str = "fl-br0",
        subnet: str = "10.200.0.0/24",
        gateway: str = "10.200.0.254",
        verbose: bool = False,
    ):
        self.bridge_name = bridge_name
        self.subnet = subnet
        self.gateway = gateway
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[NS] {msg}")

    def _run(self, cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a host command (ip, tc, etc.), retrying with sudo on permission errors."""
        self.log("Running: " + " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").lower()
            # Retry with sudo on typical permission errors, regardless of 'check' flag
            if "permission denied" in err or "operation not permitted" in err or "rtnetlink" in err:
                sudo_pw = os.environ.get("FL_SUDO_PASSWORD", "").strip()
                if sudo_pw:
                    self.log("Retrying with sudo (using FL_SUDO_PASSWORD)...")
                    result = subprocess.run(
                        ["sudo", "-S"] + cmd,
                        input=sudo_pw + "\n",
                        capture_output=True,
                        text=True,
                    )
                else:
                    self.log("Retrying with sudo (you may be prompted for your password)...")
                    result = subprocess.run(
                        ["sudo"] + cmd,
                        capture_output=True,
                        text=True,
                    )
        if check and result.returncode != 0:
            print(f"[ERROR] Command failed: {' '.join(cmd)}")
            print(f"[ERROR] Output: {result.stderr}")
        return result

    def _run_tc_in_ns(self, ns_name: str, args: List[str]) -> subprocess.CompletedProcess:
        """Run tc inside a network namespace, with sudo fallback similar to _run."""
        base_cmd = ["ip", "netns", "exec", ns_name, "tc"] + args
        return self._run(base_cmd, check=False)

    @staticmethod
    def _tc_egress_commands(device: str, conditions: Dict[str, str]) -> List[List[str]]:
        """Return list of tc command arg lists (without 'tc' prefix) for egress-style shaping on device.
        Used for root (egress) and for ifb (ingress mirror). Matches NetworkSimulator._tc_egress_commands."""
        netem_params: List[str] = []
        if conditions.get("latency"):
            netem_params.append("delay")
            netem_params.append(conditions["latency"])
            if conditions.get("jitter"):
                netem_params.append(conditions["jitter"])
        if conditions.get("loss"):
            netem_params.append("loss")
            netem_params.append(conditions["loss"])
        has_netem = len(netem_params) > 0
        has_bandwidth = bool(conditions.get("bandwidth"))
        if has_netem and has_bandwidth:
            return [
                ["qdisc", "add", "dev", device, "root", "handle", "1:", "htb", "default", "1"],
                ["class", "add", "dev", device, "parent", "1:", "classid", "1:1", "htb",
                 "rate", conditions["bandwidth"], "ceil", conditions["bandwidth"]],
                ["qdisc", "add", "dev", device, "parent", "1:1", "handle", "10:", "netem"] + netem_params,
            ]
        if has_netem:
            return [["qdisc", "add", "dev", device, "root", "netem"] + netem_params]
        if has_bandwidth:
            return [["qdisc", "add", "dev", device, "root", "tbf", "rate", conditions["bandwidth"],
                     "burst", "32kbit", "latency", "400ms"]]
        return []

    @staticmethod
    def _split_subnet(subnet: str) -> (str, int):
        cidr_parts = subnet.split("/")
        base = cidr_parts[0]
        prefix = int(cidr_parts[1]) if len(cidr_parts) == 2 else 24
        return base, prefix

    def _format_ip(self, last_octet: int) -> str:
        base, prefix = self._split_subnet(self.subnet)
        parts = base.split(".")
        if len(parts) != 4:
            raise ValueError(f"Unsupported subnet format: {self.subnet}")
        return f"{parts[0]}.{parts[1]}.{parts[2]}.{last_octet}/{prefix}"

    def ensure_bridge(self) -> None:
        """Ensure the Linux bridge for namespaces exists and is up."""
        self.log(f"Ensuring bridge {self.bridge_name} exists")
        # Create bridge if missing
        result = self._run(["ip", "link", "show", self.bridge_name], check=False)
        if result.returncode != 0:
            self._run(["ip", "link", "add", self.bridge_name, "type", "bridge"], check=False)
        # Assign gateway IP (with /24 so host is on same subnet as namespaces) and bring it up
        self._run(["ip", "addr", "flush", "dev", self.bridge_name], check=False)
        gw_addr = self.gateway if "/" in self.gateway else f"{self.gateway}/24"
        self._run(["ip", "addr", "add", gw_addr, "dev", self.bridge_name], check=False)
        self._run(["ip", "link", "set", self.bridge_name, "up"], check=False)
        self.log(f"Bridge {self.bridge_name} is up with gateway {self.gateway}")

    def create_namespace(self, ns_name: str) -> None:
        self.log(f"Creating namespace {ns_name}")
        self._run(["ip", "netns", "add", ns_name], check=False)

    def delete_namespace(self, ns_name: str) -> None:
        self.log(f"Deleting namespace {ns_name}")
        self._run(["ip", "netns", "del", ns_name], check=False)

    def _create_veth_pair(self, veth_ns: str, veth_br: str, ns_name: str) -> None:
        self.log(f"Creating veth pair {veth_ns} <-> {veth_br} for {ns_name}")
        # Delete if existing
        self._run(["ip", "link", "del", veth_br], check=False)
        # Create and move one end into namespace
        self._run(["ip", "link", "add", veth_ns, "type", "veth", "peer", "name", veth_br])
        self._run(["ip", "link", "set", veth_ns, "netns", ns_name])
        # Attach bridge-side to bridge
        self._run(["ip", "link", "set", veth_br, "master", self.bridge_name])
        self._run(["ip", "link", "set", veth_br, "up"])

    def _configure_ns_interface(self, ns_name: str, veth_ns: str, ip_cidr: str) -> None:
        self.log(f"Configuring {veth_ns} in {ns_name} with IP {ip_cidr}")
        # Bring loopback up
        self._run(["ip", "netns", "exec", ns_name, "ip", "link", "set", "lo", "up"], check=False)
        # Assign IP and bring interface up
        self._run(["ip", "netns", "exec", ns_name, "ip", "addr", "flush", "dev", veth_ns], check=False)
        self._run(["ip", "netns", "exec", ns_name, "ip", "addr", "add", ip_cidr, "dev", veth_ns], check=False)
        self._run(["ip", "netns", "exec", ns_name, "ip", "link", "set", veth_ns, "up"], check=False)
        # Default route via bridge gateway so namespaces can reach host (e.g. AMQP proxy at gateway:25673)
        gw = self.gateway.split("/")[0]
        self._run(["ip", "netns", "exec", ns_name, "ip", "route", "add", "default", "via", gw], check=False)

    def create_endpoint(self, name: str, ip_last_octet: int) -> NamespaceEndpoint:
        """
        Create one namespace + veth pair + IP for an endpoint.

        Returns a NamespaceEndpoint describing the created topology.
        """
        # Keep actual interface and namespace names short to satisfy Linux ifname limits (~15 chars)
        safe = name[:8]
        ns_name = f"fl-{safe}"
        veth_ns = f"veth-{safe}"
        veth_br = f"veth-{safe}-br"
        ip_cidr = self._format_ip(ip_last_octet)

        self.create_namespace(ns_name)
        self._create_veth_pair(veth_ns, veth_br, ns_name)
        self._configure_ns_interface(ns_name, veth_ns, ip_cidr)

        return NamespaceEndpoint(name=name, ns_name=ns_name, veth_ns=veth_ns, veth_br=veth_br, ip=ip_cidr.split("/")[0])

    def setup_topology(self, base_name: str, num_clients: int) -> (NamespaceEndpoint, List[NamespaceEndpoint]):
        """
        Create a simple star topology:
          - One server namespace
          - N client namespaces
          - All connected to a common bridge with per-namespace veth pairs

        IP layout (subnet 10.200.0.0/24 by default):
          - Server:  10.200.0.1
          - Clients: 10.200.0.10 + i
        """
        self.ensure_bridge()
        # Use short, fixed names so veth/ns names stay within kernel ifname limits
        server_ep = self.create_endpoint("srv", ip_last_octet=1)

        client_eps: List[NamespaceEndpoint] = []
        for i in range(num_clients):
            # Leave some room between server and clients
            ip_octet = 10 + i
            ep = self.create_endpoint(f"c{i+1}", ip_last_octet=ip_octet)
            client_eps.append(ep)

        self.log(
            f"Created topology: server={server_ep.ns_name} ({server_ep.ip}), "
            f"{len(client_eps)} client namespaces connected to {self.bridge_name}"
        )
        return server_ep, client_eps

    def reset_tc(self, endpoints: List[NamespaceEndpoint]) -> None:
        """Delete tc qdisc (egress and ingress) from all namespace interfaces."""
        for ep in endpoints:
            try:
                self._run_tc_in_ns(ep.ns_name, ["qdisc", "del", "dev", ep.veth_ns, "ingress"])
                self._run_tc_in_ns(ep.ns_name, ["qdisc", "del", "dev", "ifb0", "root"])
                self._run(["ip", "netns", "exec", ep.ns_name, "ip", "link", "set", "ifb0", "down"], check=False)
                self._run(["ip", "netns", "exec", ep.ns_name, "ip", "link", "del", "ifb0"], check=False)
                self._run_tc_in_ns(ep.ns_name, ["qdisc", "del", "dev", ep.veth_ns, "root"])
            except Exception as e:
                self.log(f"reset_tc: could not reset {ep.ns_name}:{ep.veth_ns}: {e}")

    def apply_tc(
        self,
        endpoint: NamespaceEndpoint,
        conditions: Dict[str, str],
    ) -> bool:
        """
        Apply netem/htb/tbf shaping on a namespace interface (egress and ingress).

        The same delay, jitter, loss, and bandwidth are applied to both egress and ingress
        on the endpoint's interface so that traffic in both directions sees identical shaping.
        Egress: root qdisc; ingress: same conditions via IFB mirror.
        """
        if not conditions:
            return False

        netem_params: List[str] = []
        if conditions.get("latency"):
            netem_params.append(f"delay {conditions['latency']}")
            if conditions.get("jitter"):
                netem_params.append(conditions["jitter"])
        if conditions.get("loss"):
            netem_params.append(f"loss {conditions['loss']}")

        has_netem = len(netem_params) > 0
        has_bandwidth = bool(conditions.get("bandwidth"))

        # First, reset any existing qdisc
        self._run_tc_in_ns(endpoint.ns_name, ["qdisc", "del", "dev", endpoint.veth_ns, "root"])

        try:
            if has_netem and has_bandwidth:
                # Root HTB class with rate, netem as child (same pattern as container mode)
                r1 = self._run_tc_in_ns(
                    endpoint.ns_name,
                    ["qdisc", "add", "dev", endpoint.veth_ns, "root", "handle", "1:", "htb", "default", "1"],
                )
                if r1.returncode != 0:
                    raise RuntimeError(r1.stderr or r1.stdout or "tc htb root failed")
                r2 = self._run_tc_in_ns(
                    endpoint.ns_name,
                    [
                        "class",
                        "add",
                        "dev",
                        endpoint.veth_ns,
                        "parent",
                        "1:",
                        "classid",
                        "1:1",
                        "htb",
                        "rate",
                        conditions["bandwidth"],
                        "ceil",
                        conditions["bandwidth"],
                    ],
                )
                if r2.returncode != 0:
                    raise RuntimeError(r2.stderr or r2.stdout or "tc htb class failed")
                netem_args = " ".join(netem_params).split()
                r3 = self._run_tc_in_ns(
                    endpoint.ns_name,
                    ["qdisc", "add", "dev", endpoint.veth_ns, "parent", "1:1", "handle", "10:", "netem", *netem_args],
                )
                if r3.returncode != 0:
                    raise RuntimeError(r3.stderr or r3.stdout or "tc netem failed")
                self.log(
                    f"{endpoint.ns_name}:{endpoint.veth_ns}: "
                    f"rate={conditions['bandwidth']}, netem={', '.join(netem_params)}"
                )
            elif has_netem:
                netem_args = " ".join(netem_params).split()
                r = self._run_tc_in_ns(
                    endpoint.ns_name,
                    ["qdisc", "add", "dev", endpoint.veth_ns, "root", "netem", *netem_args],
                )
                if r.returncode != 0:
                    raise RuntimeError(r.stderr or r.stdout or "tc netem failed")
                self.log(f"{endpoint.ns_name}:{endpoint.veth_ns}: netem={', '.join(netem_params)}")
            elif has_bandwidth:
                r = self._run_tc_in_ns(
                    endpoint.ns_name,
                    [
                        "qdisc",
                        "add",
                        "dev",
                        endpoint.veth_ns,
                        "root",
                        "tbf",
                        "rate",
                        conditions["bandwidth"],
                        "burst",
                        "32kbit",
                        "latency",
                        "400ms",
                    ],
                )
                if r.returncode != 0:
                    raise RuntimeError(r.stderr or r.stdout or "tc tbf failed")
                self.log(f"{endpoint.ns_name}:{endpoint.veth_ns}: rate={conditions['bandwidth']}")
            else:
                # Nothing to apply
                return False

            # Ingress: same delay/jitter/loss/bandwidth on incoming traffic (via IFB)
            ifb_name = "ifb0"
            try:
                self._run_tc_in_ns(endpoint.ns_name, ["qdisc", "del", "dev", endpoint.veth_ns, "ingress"])
                self._run_tc_in_ns(endpoint.ns_name, ["qdisc", "del", "dev", ifb_name, "root"])
                self._run(["ip", "netns", "exec", endpoint.ns_name, "ip", "link", "set", ifb_name, "down"], check=False)
                self._run(["ip", "netns", "exec", endpoint.ns_name, "ip", "link", "del", ifb_name], check=False)
            except Exception:
                pass
            try:
                rip = self._run(["ip", "netns", "exec", endpoint.ns_name, "ip", "link", "add", "name", ifb_name, "type", "ifb"], check=False)
                if rip.returncode != 0:
                    self.log(f"{endpoint.ns_name}: ip link add ifb failed (ingress skipped)")
                else:
                    self._run(["ip", "netns", "exec", endpoint.ns_name, "ip", "link", "set", ifb_name, "up"])
                    rinq = self._run_tc_in_ns(endpoint.ns_name, ["qdisc", "add", "dev", endpoint.veth_ns, "ingress"])
                    if rinq.returncode != 0:
                        self.log(f"{endpoint.ns_name}: tc ingress failed")
                    else:
                        self._run_tc_in_ns(
                            endpoint.ns_name,
                            [
                                "filter", "add", "dev", endpoint.veth_ns, "parent", "ffff:", "protocol", "all",
                                "u32", "match", "u32", "0", "0", "flowid", "1:1",
                                "action", "mirred", "egress", "redirect", "dev", ifb_name,
                            ],
                        )
                        for cmd in self._tc_egress_commands(ifb_name, conditions):
                            self._run_tc_in_ns(endpoint.ns_name, cmd)
                        self.log(f"{endpoint.ns_name}:{endpoint.veth_ns}: same delay/jitter/loss on egress and ingress (ingress via {ifb_name})")
            except Exception as e:
                self.log(f"{endpoint.ns_name}: ingress tc failed: {e}; egress only.")
                print(f"  [WARNING] Ingress tc failed for {endpoint.ns_name}; same delay applied to egress only.")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to apply tc in namespace {endpoint.ns_name} on {endpoint.veth_ns}: {e}")
            return False

    def apply_conditions(
        self,
        server_endpoint: NamespaceEndpoint,
        client_endpoints: List[NamespaceEndpoint],
        downstream_conditions: Dict[str, str],
        upstream_conditions: Dict[str, str],
    ) -> None:
        """
        Apply separate tc profiles for:
          - downstream: server -> clients (egress from server namespace)
          - upstream:   clients -> server (egress from each client namespace)

        This achieves independent delay/loss/bandwidth per direction.
        """
        if downstream_conditions:
            self.log("Applying downstream conditions (server -> clients)")
            self.apply_tc(server_endpoint, downstream_conditions)
        if upstream_conditions:
            self.log("Applying upstream conditions (clients -> server)")
            for ep in client_endpoints:
                self.apply_tc(ep, upstream_conditions)

    def cleanup(self, endpoints: List[NamespaceEndpoint]) -> None:
        """Best-effort cleanup of tc + namespaces. Bridge is left in place for reuse."""
        self.log("Cleaning up namespaces and tc rules")
        self.reset_tc(endpoints)
        for ep in endpoints:
            self.delete_namespace(ep.ns_name)


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
