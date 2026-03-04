#!/usr/bin/env python3
"""
Native FL experiment runner without Docker.

This script runs the FL server and multiple FL clients as normal Python
processes on the SAME host, using Linux network namespaces + veth pairs
for isolation and tc/netem for network emulation.

Key properties:
  - No Docker is required for the FL server/clients.
  - Each endpoint (server or client) runs in its own network namespace.
  - A shared bridge (default: fl-br0) connects all namespaces.
  - tc/netem rules are applied independently on:
        * server namespace interface (server -> clients direction)
        * each client namespace interface (clients -> server direction)

Currently this runner is intentionally conservative:
  - It supports running ONE protocol and ONE scenario at a time.
  - It is primarily tested with the gRPC protocol.
  - Other protocols can be wired in later using the same pattern.
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _wait_for_port(host: str, port: int, timeout_sec: float = 30, ns_name: Optional[str] = None) -> bool:
    """Wait until host:port is accepting connections. If ns_name is set, check from that network namespace (uses sudo)."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            if ns_name:
                # ip netns exec requires root; use sudo so the check matches what server/clients will see
                cmd = ["sudo", "-E", "ip", "netns", "exec", ns_name, "python3", "-c",
                       f"import socket; s=socket.socket(); s.settimeout(2); s.connect(('{host}', {port})); s.close()"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    timeout=10,
                    cwd=str(PROJECT_ROOT),
                )
                if result.returncode == 0:
                    return True
            else:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2)
                s.connect((host, port))
                s.close()
                return True
        except (socket.error, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        time.sleep(1)
    return False


# Use-case and protocol → script naming helpers
USE_CASE_DIR: Dict[str, str] = {
    "emotion": "Emotion_Recognition",
    "mentalstate": "MentalState_Recognition",
    "temperature": "Temperature_Regulation",
}

PROTOCOL_SUFFIX: Dict[str, str] = {
    "mqtt": "MQTT",
    "amqp": "AMQP",
    "grpc": "gRPC",
    "quic": "QUIC",
    "http3": "HTTP3",
    "dds": "DDS",
    "rl_unified": "Unified",
}


def _resolve_script_path(use_case: str, protocol: str, role: str) -> Path:
    """
    Resolve server/client script path for (use_case, protocol).

    role: "server" or "client"
    """
    if use_case not in USE_CASE_DIR:
        raise ValueError(f"Unknown use case '{use_case}'")
    if protocol not in PROTOCOL_SUFFIX:
        raise ValueError(f"Unknown protocol '{protocol}'")

    uc_dir = USE_CASE_DIR[use_case]
    proto_suffix = PROTOCOL_SUFFIX[protocol]

    if role == "server":
        rel = Path("Server") / uc_dir / f"FL_Server_{proto_suffix}.py"
    elif role == "client":
        rel = Path("Client") / uc_dir / f"FL_Client_{proto_suffix}.py"
    else:
        raise ValueError(f"Invalid role '{role}' (expected 'server' or 'client')")

    path = PROJECT_ROOT / rel
    if not path.is_file():
        raise FileNotFoundError(f"{role.title()} script not found for {use_case}/{protocol}: {path}")
    return path


class NativeExperimentRunner:
    """
    Orchestrates a single FL run for a given (use_case, protocol, scenario) without Docker.

    Responsibilities:
      - Set up network namespaces + veth topology via NamespaceNetworkSimulator.
      - Apply tc/netem separately for server->clients and clients->server.
      - Launch server/client Python scripts inside the appropriate namespaces.
      - Wait for completion and perform best-effort cleanup.
    """

    def __init__(
        self,
        use_case: str,
        protocol: str,
        scenario: str,
        num_rounds: int,
        num_clients: int,
        downstream_scenario: str = None,
        upstream_scenario: str = None,
        enable_gpu: bool = False,
        apply_tc_after_round_1: Optional[str] = None,
        use_pruning: bool = False,
        pruning_sparsity: Optional[float] = None,
        pruning_structured: bool = False,
    ) -> None:
        self.use_case = use_case
        self.protocol = protocol
        self.scenario = scenario
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.downstream_scenario = downstream_scenario or scenario
        self.upstream_scenario = upstream_scenario or scenario
        self.enable_gpu = enable_gpu
        # When set (e.g. "good"): start with no tc; after round 1 complete apply this scenario to clients
        self.apply_tc_after_round_1 = apply_tc_after_round_1
        # Pruning configuration (applied via environment variables)
        self.use_pruning = use_pruning
        self.pruning_sparsity = pruning_sparsity
        self.pruning_structured = pruning_structured

        # Resolve concrete script paths for this (use_case, protocol)
        self.server_script = _resolve_script_path(use_case, protocol, role="server")
        self.client_script = _resolve_script_path(use_case, protocol, role="client")

        self._processes: List[subprocess.Popen] = []
        self._broker_processes: List[subprocess.Popen] = []
        self._amqp_proxy_process: Optional[subprocess.Popen] = None  # TCP proxy gateway:port -> 127.0.0.1:5672
        self._endpoints = None
        self._ns_sim = None
        self._gateway_ip: str = ""
        self._broker_ip: str = ""  # MQTT broker / AMQP broker when run in server ns (server namespace IP)
        self._amqp_host: Optional[str] = None  # When set, AMQP uses this (e.g. gateway for host RabbitMQ fallback)
        self._amqp_port: Optional[int] = None  # When set (e.g. 25673), use proxy port so namespaces can reach host RabbitMQ
        self._log_files: list = []  # keep refs so files stay open until processes exit

    def _load_scenario(self, name: str) -> Dict[str, str]:
        from network_simulator import NetworkSimulator

        if not name:
            return {}
        if name not in NetworkSimulator.NETWORK_SCENARIOS:
            print(f"[WARNING] Unknown scenario '{name}', skipping network shaping for this direction.")
            return {}
        return dict(NetworkSimulator.NETWORK_SCENARIOS[name])

    def setup_network(self) -> None:
        from network_simulator import NamespaceNetworkSimulator

        base_name = f"{self.protocol}-{self.use_case}"
        sim = NamespaceNetworkSimulator(verbose=True)

        server_ep, client_eps = sim.setup_topology(base_name, self.num_clients)

        # Native: apply tc only to clients (upstream), not to the server. Server has no delay/loss/bandwidth shaping.
        # If apply_tc_after_round_1 is set, start with excellent (no tc); tc will be applied mid-run after round 1.
        downstream_conditions = {}  # No tc on server
        if self.apply_tc_after_round_1:
            upstream_conditions = self._load_scenario("excellent")
        else:
            upstream_conditions = self._load_scenario(self.upstream_scenario)
        sim.apply_conditions(server_ep, client_eps, downstream_conditions, upstream_conditions)

        # Store for later cleanup
        self._ns_sim = sim
        self._gateway_ip = sim.gateway
        self._endpoints = (server_ep, client_eps)

        print("\n=== Native Network Topology (Namespaces) ===")
        print(f"  Bridge: {sim.bridge_name} ({sim.gateway})")
        print(f"  Server: {server_ep.ns_name} @ {server_ep.ip} (iface={server_ep.veth_ns})")
        for ep in client_eps:
            print(f"  Client: {ep.ns_name} @ {ep.ip} (iface={ep.veth_ns})")
        print("===========================================\n")

    def _start_broker(self, log_dir: Path) -> None:
        """Start MQTT and/or AMQP broker so server and clients in namespaces can connect.
        Brokers run inside the server namespace (at server_ep.ip) so all participants can reach them via the bridge.
        """
        broker_log_dir = log_dir
        need_mqtt = self.protocol in ("mqtt", "rl_unified")
        need_amqp = self.protocol in ("amqp", "rl_unified")
        server_ep, _ = self._endpoints
        # Broker address for both MQTT and AMQP in native mode: server namespace IP (reachable by server and clients)
        self._broker_ip = server_ep.ip

        if need_mqtt:
            # Run Mosquitto inside the server namespace so it is at server_ep.ip (10.200.0.1);
            # server and clients then connect to that IP without needing host-bridge reachability.
            try:
                subprocess.run(
                    ["sudo", "ip", "netns", "exec", server_ep.ns_name, "pkill", "-x", "mosquitto"],
                    capture_output=True,
                    timeout=5,
                )
                time.sleep(0.5)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            mqtt_config = broker_log_dir / "mosquitto_native.conf"
            try:
                # Allow large FL model updates (~12MB+); 0 = no limit (Mosquitto 2.x)
                mqtt_config.write_text(
                    "listener 1883 0.0.0.0\nallow_anonymous true\nmessage_size_limit 0\n",
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[WARNING] Could not write mosquitto config: {e}")
            broker_log = (broker_log_dir / "broker_mqtt.log").open("w", encoding="utf-8", errors="replace")
            self._log_files.append(broker_log)
            base_cmd = ["ip", "netns", "exec", server_ep.ns_name, "mosquitto", "-c", str(mqtt_config), "-v"]
            sudo_pw = os.environ.get("FL_SUDO_PASSWORD", "").strip()
            if sudo_pw:
                cmd = ["sudo", "-E", "-S"] + base_cmd
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=broker_log,
                    stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                )
                if proc.stdin:
                    proc.stdin.write((sudo_pw + "\n").encode())
                    proc.stdin.close()
            else:
                cmd = ["sudo", "-E"] + base_cmd
                proc = subprocess.Popen(
                    cmd,
                    stdout=broker_log,
                    stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                )
            self._broker_processes.append(proc)
            print(f"[INFO] Started MQTT broker (mosquitto) in namespace {server_ep.ns_name} at {self._broker_ip}:1883")
            if not _wait_for_port("127.0.0.1", 1883, timeout_sec=15, ns_name=server_ep.ns_name):
                if proc.poll() is not None:
                    print("[ERROR] mosquitto exited immediately. Check broker_mqtt.log")
                    raise RuntimeError("MQTT broker (mosquitto) failed to start. Install: sudo apt install mosquitto")
                print("[WARNING] MQTT port 1883 not ready in time; continuing anyway.")
            else:
                print("[INFO] MQTT broker port 1883 is ready.")

        if need_amqp:
            # RabbitMQ must run on the HOST (not in a namespace) due to Erlang epmd. Server and clients
            # in namespaces connect to the host via the bridge gateway. Since RabbitMQ often listens only
            # on 127.0.0.1:5672, we start a TCP proxy on the host that listens on 0.0.0.0:25673 and
            # forwards to 127.0.0.1:5672, so namespaces can use gateway:25673 without changing RabbitMQ config.
            broker_log = (broker_log_dir / "broker_amqp.log").open("w", encoding="utf-8", errors="replace")
            self._log_files.append(broker_log)
            self._amqp_host = self._gateway_ip
            amqp_proxy_port = 25673

            # Check if RabbitMQ is already running (avoid "port 25672 in use" when starting a second node)
            rabbitmq_already_running = False
            try:
                r = subprocess.run(
                    ["systemctl", "is-active", "rabbitmq-server"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if r.returncode == 0 and (r.stdout or "").strip() == "active":
                    rabbitmq_already_running = True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            if not rabbitmq_already_running:
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(2)
                    s.connect(("127.0.0.1", 5672))
                    s.close()
                    rabbitmq_already_running = True
                except (socket.error, OSError):
                    pass

            if rabbitmq_already_running:
                print("[INFO] AMQP broker (RabbitMQ) already running on host; skipping start")
            else:
                for cmd_try in [
                    ["sudo", "systemctl", "start", "rabbitmq-server"],
                    ["rabbitmq-server", "-detached"],
                ]:
                    try:
                        r = subprocess.run(
                            cmd_try,
                            capture_output=True,
                            text=True,
                            timeout=15,
                            cwd=str(PROJECT_ROOT),
                        )
                        out = (r.stdout or "") + (r.stderr or "")
                        if out:
                            broker_log.write(out)
                            broker_log.flush()
                        if r.returncode == 0 or "already running" in out.lower():
                            print("[INFO] AMQP broker (RabbitMQ) started or already running on host")
                            break
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        continue
                else:
                    print("[WARNING] Could not start RabbitMQ via systemctl or rabbitmq-server -detached")

            # Wait for RabbitMQ on localhost (required for the proxy to work)
            if not _wait_for_port("127.0.0.1", 5672, timeout_sec=30, ns_name=None):
                print(
                    "[WARNING] RabbitMQ not reachable on 127.0.0.1:5672. "
                    "Start it with: sudo systemctl start rabbitmq-server"
                )
            else:
                # Start TCP proxy so namespaces can reach RabbitMQ via gateway:amqp_proxy_port
                # Bind proxy to gateway IP so the host accepts connections from the bridge
                proxy_script = PROJECT_ROOT / "Network_Simulation" / "amqp_proxy.py"
                if proxy_script.is_file():
                    try:
                        proc = subprocess.Popen(
                            [sys.executable, str(proxy_script), str(amqp_proxy_port), self._gateway_ip],
                            stdout=broker_log,
                            stderr=subprocess.STDOUT,
                            cwd=str(PROJECT_ROOT),
                        )
                        time.sleep(2)
                        if proc.poll() is not None:
                            print(f"[WARNING] AMQP proxy exited immediately (check broker_amqp.log). Using AMQP_PORT=5672; ensure RabbitMQ listens on 0.0.0.0.")
                            self._amqp_port = 5672
                        else:
                            self._amqp_proxy_process = proc
                            self._amqp_port = amqp_proxy_port
                            # Verify from host first, then from namespace
                            if _wait_for_port(self._gateway_ip, amqp_proxy_port, timeout_sec=5, ns_name=None):
                                if _wait_for_port(self._gateway_ip, amqp_proxy_port, timeout_sec=15, ns_name=server_ep.ns_name):
                                    print(f"[INFO] AMQP proxy running: {self._gateway_ip}:{amqp_proxy_port} -> 127.0.0.1:5672 (server/clients use gateway:{amqp_proxy_port})")
                                else:
                                    print(f"[INFO] AMQP proxy listening on host; if server/clients fail to connect, allow port {amqp_proxy_port} from 10.200.0.0/24 (e.g. iptables/nftables).")
                            else:
                                print(f"[INFO] AMQP proxy started; server/clients will use {self._gateway_ip}:{amqp_proxy_port}")
                    except Exception as e:
                        print(f"[WARNING] Could not start AMQP proxy: {e}. Set NODE_IP_ADDRESS=0.0.0.0 in /etc/rabbitmq/rabbitmq-env.conf and use AMQP_PORT=5672.")
                        self._amqp_port = 5672
                else:
                    print("[WARNING] amqp_proxy.py not found; server/clients may not reach RabbitMQ from namespaces.")
                    self._amqp_port = 5672
                    if not _wait_for_port(self._gateway_ip, 5672, timeout_sec=10, ns_name=server_ep.ns_name):
                        localhost_ok = _wait_for_port("127.0.0.1", 5672, timeout_sec=5, ns_name=None)
                        if localhost_ok:
                            print("  → Add NODE_IP_ADDRESS=0.0.0.0 to /etc/rabbitmq/rabbitmq-env.conf and restart RabbitMQ.")

    def _netns_exec_args(self, ns_name: str, executable: str, script: str) -> tuple:
        """
        Build command list to run a script inside a network namespace.
        Entering a namespace requires root, so we wrap with sudo -E (preserve env).
        Returns (cmd_list, stdin_payload or None). If stdin_payload is set, caller
        must pass stdin=subprocess.PIPE and write it then close stdin.
        """
        base = ["ip", "netns", "exec", ns_name, executable, script]
        sudo_pw = os.environ.get("FL_SUDO_PASSWORD", "").strip()
        if sudo_pw:
            cmd = ["sudo", "-E", "-S"] + base
            return cmd, (sudo_pw + "\n").encode()
        cmd = ["sudo", "-E"] + base
        return cmd, None

    def _spawn_server(self, log_file=None) -> subprocess.Popen:
        server_ep, client_eps = self._endpoints

        env = os.environ.copy()
        # Unbuffered stdout so FL_DIAG lines (and logs) are flushed immediately; critical for
        # diagnostic pipeline T_actual (round index 2) when we terminate clients after server exit.
        env["PYTHONUNBUFFERED"] = "1"
        # Common FL configuration
        env["NUM_ROUNDS"] = str(self.num_rounds)
        env["MIN_CLIENTS"] = str(self.num_clients)
        env["MAX_CLIENTS"] = str(self.num_clients)
        # Match Docker defaults for convergence behaviour unless explicitly overridden
        env.setdefault("CONVERGENCE_THRESHOLD", "0.001")
        env.setdefault("CONVERGENCE_PATIENCE", "2")
        env.setdefault("MIN_ROUNDS", "3")

        # Protocol-specific server-side configuration
        if self.protocol == "grpc":
            # Bind gRPC on all addresses within namespace
            env.setdefault("GRPC_HOST", "0.0.0.0")
        elif self.protocol == "mqtt":
            # MQTT broker runs in server namespace at _broker_ip (10.200.0.1)
            if self._broker_ip or self._gateway_ip:
                env["MQTT_BROKER"] = self._broker_ip or self._gateway_ip
        elif self.protocol == "amqp":
            # AMQP broker: gateway + proxy port (or direct 5672 if no proxy)
            amqp_host = getattr(self, "_amqp_host", None) or self._broker_ip
            if amqp_host:
                env["AMQP_HOST"] = amqp_host
            if getattr(self, "_amqp_port", None) is not None:
                env["AMQP_PORT"] = str(self._amqp_port)
        elif self.protocol == "quic":
            # QUIC server binds on all addresses in namespace
            env.setdefault("QUIC_HOST", "0.0.0.0")
        elif self.protocol == "http3":
            env.setdefault("HTTP3_HOST", "0.0.0.0")
        elif self.protocol == "dds":
            # DDS uses domain ID and multicast; namespace isolation + bridge is enough.
            pass
        elif self.protocol == "rl_unified":
            # Unified server uses multiple back-ends; MQTT broker in server ns, AMQP on host
            if self._broker_ip or self._gateway_ip:
                env["MQTT_BROKER"] = self._broker_ip or self._gateway_ip
            amqp_host = getattr(self, "_amqp_host", None) or self._broker_ip
            if amqp_host:
                env["AMQP_BROKER"] = amqp_host
            if getattr(self, "_amqp_port", None) is not None:
                env["AMQP_PORT"] = str(self._amqp_port)
            env.setdefault("QUIC_HOST", "0.0.0.0")
            env.setdefault("HTTP3_HOST", "0.0.0.0")

        # Pruning configuration (server side)
        if self.use_pruning:
            env["USE_PRUNING"] = "1"
            if self.pruning_sparsity is not None:
                env["PRUNING_SPARSITY"] = str(self.pruning_sparsity)
            # Default to "false" unless explicitly structured
            env["PRUNING_STRUCTURED"] = "true" if self.pruning_structured else env.get("PRUNING_STRUCTURED", "false")

        if self.enable_gpu:
            # In native mode we simply respect whatever CUDA_VISIBLE_DEVICES the user already set.
            print("[INFO] GPU mode enabled for native server (respecting existing CUDA_VISIBLE_DEVICES).")

        cmd, stdin_payload = self._netns_exec_args(
            server_ep.ns_name, sys.executable, str(self.server_script)
        )
        print(f"[CMD] {' '.join(cmd)}")
        kw = {"cwd": str(PROJECT_ROOT), "env": env}
        if stdin_payload is not None:
            kw["stdin"] = subprocess.PIPE
        if log_file is not None:
            kw["stdout"] = log_file
            kw["stderr"] = subprocess.STDOUT
        proc = subprocess.Popen(cmd, **kw)
        if stdin_payload is not None and proc.stdin:
            proc.stdin.write(stdin_payload)
            proc.stdin.close()
        self._processes.append(proc)
        return proc

    def _spawn_clients(self, client_log_files=None) -> List[subprocess.Popen]:
        server_ep, client_eps = self._endpoints
        procs: List[subprocess.Popen] = []

        for idx, ep in enumerate(client_eps, start=1):
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"  # Flush FL_DIAG immediately for diagnostic pipeline T_actual
            env["CLIENT_ID"] = str(idx)
            env["NUM_CLIENTS"] = str(self.num_clients)
            # Match Docker defaults for convergence behaviour unless explicitly overridden
            env.setdefault("CONVERGENCE_THRESHOLD", "0.001")
            env.setdefault("CONVERGENCE_PATIENCE", "2")
            env.setdefault("MIN_ROUNDS", "3")

            # Protocol-specific client-side configuration
            if self.protocol == "grpc":
                # Connect directly to gRPC server in server namespace
                env["GRPC_HOST"] = server_ep.ip
            elif self.protocol == "mqtt":
                # MQTT broker in server namespace at _broker_ip
                if self._broker_ip or self._gateway_ip:
                    env["MQTT_BROKER"] = self._broker_ip or self._gateway_ip
            elif self.protocol == "amqp":
                amqp_host = getattr(self, "_amqp_host", None) or self._broker_ip
                if amqp_host:
                    env["AMQP_HOST"] = amqp_host
                if getattr(self, "_amqp_port", None) is not None:
                    env["AMQP_PORT"] = str(self._amqp_port)
            elif self.protocol == "quic":
                # QUIC server in server namespace
                env["QUIC_HOST"] = server_ep.ip
            elif self.protocol == "http3":
                env["HTTP3_HOST"] = server_ep.ip
            elif self.protocol == "dds":
                # DDS uses domain ID; all participants share same bridge + domain.
                pass
            elif self.protocol == "rl_unified":
                # Unified client talks to brokers + unified server over several back-ends.
                if self._broker_ip or self._gateway_ip:
                    env["MQTT_BROKER"] = self._broker_ip or self._gateway_ip
                amqp_host = getattr(self, "_amqp_host", None) or self._broker_ip
                if amqp_host:
                    env["AMQP_HOST"] = amqp_host
                if getattr(self, "_amqp_port", None) is not None:
                    env["AMQP_PORT"] = str(self._amqp_port)
                env["GRPC_HOST"] = server_ep.ip
                env["QUIC_HOST"] = server_ep.ip
                env["HTTP3_HOST"] = server_ep.ip

            # Pruning configuration (client side)
            if self.use_pruning:
                env["USE_PRUNING"] = "1"
                if self.pruning_sparsity is not None:
                    env["PRUNING_SPARSITY"] = str(self.pruning_sparsity)
                env["PRUNING_STRUCTURED"] = "true" if self.pruning_structured else env.get("PRUNING_STRUCTURED", "false")

            if self.enable_gpu:
                # Ensure every client uses a GPU: pin to GPU 0 so both use the same GPU
                # (avoids one client falling back to CPU when only one GPU is present)
                env["CUDA_VISIBLE_DEVICES"] = "0"
                env.setdefault("GPU_DEVICE_ID", "0")

            cmd, stdin_payload = self._netns_exec_args(
                ep.ns_name, sys.executable, str(self.client_script)
            )
            print(f"[CMD] {' '.join(cmd)}")
            kw = {"cwd": str(PROJECT_ROOT), "env": env}
            if stdin_payload is not None:
                kw["stdin"] = subprocess.PIPE
            if client_log_files is not None and idx - 1 < len(client_log_files):
                kw["stdout"] = client_log_files[idx - 1]
                kw["stderr"] = subprocess.STDOUT
            proc = subprocess.Popen(cmd, **kw)
            if stdin_payload is not None and proc.stdin:
                proc.stdin.write(stdin_payload)
                proc.stdin.close()
            self._processes.append(proc)
            procs.append(proc)

        return procs

    def run(self) -> int:
        tc_note = (
            f"  Tc: round 1 no tc; after round 1 complete, apply scenario={self.apply_tc_after_round_1} to clients.\n"
            if self.apply_tc_after_round_1
            else f"  Tc: clients only (scenario={self.upstream_scenario}); server has no delay/loss/bandwidth.\n"
        )
        print(
            f"\n=== Starting NATIVE FL experiment ===\n"
            f"  Use case : {self.use_case}\n"
            f"  Protocol : {self.protocol}\n"
            f"  Scenario : {self.scenario}\n"
            f"  Rounds   : {self.num_rounds}\n"
            f"  Clients  : {self.num_clients}\n"
            f"{tc_note}"
            f"=====================================\n"
        )

        # Create log directory and marker first so the GUI can attach Server/Client log tabs immediately
        log_dir = PROJECT_ROOT / "Network_Simulation" / "native_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            (log_dir / "latest_run.txt").write_text(str(log_dir.resolve()), encoding="utf-8")
        except Exception:
            pass
        server_log = (log_dir / "server.log").open("w", encoding="utf-8", errors="replace")
        client_logs = [(log_dir / f"client_{i}.log").open("w", encoding="utf-8", errors="replace") for i in range(1, self.num_clients + 1)]
        self._log_files = [server_log] + client_logs
        print(f"[INFO] Server logs: {log_dir / 'server.log'}")
        for i in range(1, self.num_clients + 1):
            print(f"[INFO] Client {i} logs: {log_dir / f'client_{i}.log'}")

        self.setup_network()

        # Start broker first for MQTT/AMQP so server and clients can connect via bridge IP
        if self.protocol in ("mqtt", "amqp", "rl_unified"):
            self._start_broker(log_dir)

        if not os.environ.get("FL_SUDO_PASSWORD", "").strip():
            print("[INFO] Server and clients run inside namespaces via sudo; you may be prompted for your password (once per process). Set FL_SUDO_PASSWORD to avoid prompts.\n")

        exit_code = 0
        server_proc = None
        client_procs: List[subprocess.Popen] = []

        try:
            server_proc = self._spawn_server(log_file=server_log)
            # Small delay to let server bind its port
            time.sleep(5)
            client_procs = self._spawn_clients(client_log_files=client_logs)

            # Wait for server completion; when it exits, we stop clients as well
            print("[INFO] Do not interrupt (Ctrl+C/Stop) until the server finishes all rounds.")
            if self.apply_tc_after_round_1:
                # Phase 1: round 1 runs without tc. After round 1 complete, apply tc so round 2 send is affected.
                # No timeout: wait indefinitely (FL_DIAG_ROUND1_WAIT_SEC=0 or unset). Set to positive for a limit.
                client_log_paths = [log_dir / f"client_{i}.log" for i in range(1, self.num_clients + 1)]
                poll_interval = 2
                round1_wait_sec = int(os.environ.get("FL_DIAG_ROUND1_WAIT_SEC", "0"))
                deadline = (time.monotonic() + round1_wait_sec) if round1_wait_sec > 0 else None  # None = no limit
                if deadline is None:
                    print("[INFO] Waiting for round 1 completion (no timeout; set FL_DIAG_ROUND1_WAIT_SEC>0 for a limit).")
                while (deadline is None or time.monotonic() < deadline) and server_proc.poll() is None:
                    counts = []
                    for path in client_log_paths:
                        try:
                            text = path.read_text(encoding="utf-8", errors="replace")
                            n = sum(1 for line in text.splitlines() if "FL_DIAG" in line and "O_send=" in line)
                            counts.append(n)
                        except (IOError, OSError):
                            counts.append(0)
                    if counts and all(c >= 1 for c in counts):
                        print("[INFO] Round 1 complete (all clients sent). Applying tc for round 2...")
                        server_ep, client_eps = self._endpoints
                        conditions = self._load_scenario(self.apply_tc_after_round_1)
                        if conditions:
                            for ep in client_eps:
                                self._ns_sim.apply_tc(ep, conditions)
                            print(f"[INFO] Tc applied to clients: {self.apply_tc_after_round_1}")
                        break
                    time.sleep(poll_interval)
                if server_proc.poll() is not None:
                    print("[WARNING] Server exited before tc was applied; round 1 may have been the only round.")
            exit_code = server_proc.wait()
            # Give clients a short grace period to flush stdout (FL_DIAG)
            time.sleep(3)
            print(f"[INFO] Server exited with code {exit_code}, terminating clients...")
            if exit_code != 0:
                print(f"[WARNING] Server exited with non-zero code. Check {log_dir / 'server.log'} for errors.")
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Ctrl+C received, terminating all processes...")
            exit_code = 1
        finally:
            # Stop clients first
            for proc in client_procs:
                if proc.poll() is None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
            # Then server if still running
            if server_proc is not None and server_proc.poll() is None:
                try:
                    server_proc.terminate()
                except Exception:
                    pass

            # Give processes a brief grace period, then kill if needed
            time.sleep(5)
            for proc in self._processes:
                if proc.poll() is None:
                    try:
                        proc.kill()
                    except Exception:
                        pass

            # Stop broker processes (MQTT/AMQP) started for this run
            for proc in getattr(self, "_broker_processes", []):
                if proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            self._broker_processes = []
            # Stop AMQP TCP proxy if we started it
            proxy_proc = getattr(self, "_amqp_proxy_process", None)
            if proxy_proc is not None and proxy_proc.poll() is None:
                try:
                    proxy_proc.terminate()
                    proxy_proc.wait(timeout=5)
                except Exception:
                    try:
                        proxy_proc.kill()
                    except Exception:
                        pass
            self._amqp_proxy_process = None

            # Close log files so they are flushed (GUI can still read them)
            for f in getattr(self, "_log_files", []):
                try:
                    f.close()
                except Exception:
                    pass
            self._log_files = []

            # Best-effort cleanup of namespaces and tc
            try:
                if self._endpoints is not None:
                    server_ep, client_eps = self._endpoints
                    self._ns_sim.cleanup([server_ep] + client_eps)
            except Exception as e:
                print(f"[WARNING] Namespace cleanup failed: {e}")

        # If pruning was enabled, plot model memory vs round (only when pruning data is present)
        if getattr(self, "use_pruning", False) and log_dir is not None:
            try:
                sys.path.insert(0, str(PROJECT_ROOT / "Network_Simulation"))
                from pruning_memory_plot import plot_pruning_memory_from_experiment
                plot_path = plot_pruning_memory_from_experiment(
                    log_dir,
                    output_filename="pruning_memory_by_round.png",
                    server_log_name="server.log",
                )
                if plot_path:
                    print(f"[Pruning] Memory plot saved: {plot_path}")
            except Exception as e:
                print(f"[WARNING] Could not generate pruning memory plot: {e}")

        print(f"\n=== Native experiment finished with code {exit_code} ===\n")
        return exit_code


def _install_signal_handlers(runner: NativeExperimentRunner):
    def _handler(signum, frame):
        print(f"\n[INFO] Received signal {signum}, stopping native experiment...")
        # Just rely on runner.run() finally-block for cleanup; raising KeyboardInterrupt is enough.
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FL server + clients natively using Linux namespaces (no Docker).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--use-case",
        "-u",
        choices=["emotion", "mentalstate", "temperature"],
        required=True,
        help="Use case to run experiments for.",
    )
    parser.add_argument(
        "--protocol",
        "-p",
        choices=["mqtt", "amqp", "grpc", "quic", "http3", "dds", "rl_unified"],
        required=True,
        help="Communication protocol.",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        choices=[
            "excellent",
            "good",
            "moderate",
            "poor",
            "very_poor",
            "satellite",
            "congested_light",
            "congested_moderate",
            "congested_heavy",
        ],
        required=True,
        help="Base network scenario to apply.",
    )
    parser.add_argument(
        "--downstream-scenario",
        help="Optional scenario for server->clients (if omitted, uses --scenario).",
    )
    parser.add_argument(
        "--upstream-scenario",
        help="Optional scenario for clients->server (if omitted, uses --scenario).",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=10,
        help="Number of FL rounds (passed to server via NUM_ROUNDS env).",
    )
    parser.add_argument(
        "--num-clients",
        "-n",
        type=int,
        default=2,
        help="Number of FL clients to start.",
    )
    parser.add_argument(
        "--enable-gpu",
        action="store_true",
        help="Enable GPU usage for native processes (honors existing CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--use-pruning",
        action="store_true",
        help="Enable model pruning for server and clients (sets USE_PRUNING env).",
    )
    parser.add_argument(
        "--pruning-sparsity",
        type=float,
        help="Target sparsity for pruning as a fraction between 0.0 and 1.0 (PRUNING_SPARSITY).",
    )
    parser.add_argument(
        "--pruning-structured",
        action="store_true",
        help="Enable structured pruning (sets PRUNING_STRUCTURED=true).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runner = NativeExperimentRunner(
        use_case=args.use_case,
        protocol=args.protocol,
        scenario=args.scenario,
        num_rounds=args.rounds,
        num_clients=args.num_clients,
        downstream_scenario=args.downstream_scenario,
        upstream_scenario=args.upstream_scenario,
        enable_gpu=args.enable_gpu,
        use_pruning=args.use_pruning,
        pruning_sparsity=args.pruning_sparsity,
        pruning_structured=args.pruning_structured,
    )
    _install_signal_handlers(runner)
    code = runner.run()
    sys.exit(code)


if __name__ == "__main__":
    main()

