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

This runner supports native execution for one or more protocols and scenarios.
When multiple protocols/scenarios are provided, it runs each protocol × scenario
combination sequentially.
"""

import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NETWORK_SIM_DIR = PROJECT_ROOT / "Network_Simulation"


def _load_local_privileged_env() -> None:
    """Load local privileged env files into process env if values are missing."""
    placeholder_values = {
        "your_linux_username",
        "your_linux_password",
        "changeme",
    }

    def _is_placeholder(value: str) -> bool:
        return value.strip().lower() in placeholder_values

    candidates = [
        NETWORK_SIM_DIR / ".privileged_ops.env",
        NETWORK_SIM_DIR / "privileged_ops.env",
    ]

    for path in candidates:
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                        value = value[1:-1]
                    if _is_placeholder(value):
                        continue
                    # Preserve explicit shell exports; only fill unset keys.
                    if key and value and not os.environ.get(key):
                        os.environ[key] = value
        except OSError:
            # Keep native runner robust even if local file permissions are unusual.
            continue


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
        use_quantization: bool = False,
        quantization_bits: Optional[int] = None,
        quantization_strategy: Optional[str] = None,
        quantization_symmetric: bool = False,
        use_ql_convergence: bool = False,
        rl_reward_scenario: Optional[str] = None,
        use_communication_model_reward: bool = True,
        reset_epsilon: bool = True,
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
        # Quantization (pruning must run first when both are enabled)
        self.use_quantization = use_quantization
        self.quantization_bits = quantization_bits or 8
        self.quantization_strategy = quantization_strategy or "parameter_quantization"
        self.quantization_symmetric = quantization_symmetric
        self.use_ql_convergence = use_ql_convergence
        # When set (e.g. "good", "moderate", "poor"): run in excellent conditions but reward/state as if in that scenario (RL_REWARD_SCENARIO)
        self.rl_reward_scenario = (rl_reward_scenario or "").strip().lower() or None
        self.use_communication_model_reward = use_communication_model_reward
        self.reset_epsilon = reset_epsilon

        # Resolve concrete script paths for this (use_case, protocol)
        self.server_script = _resolve_script_path(use_case, protocol, role="server")
        self.client_script = _resolve_script_path(use_case, protocol, role="client")

        self._processes: List[subprocess.Popen] = []
        self._broker_processes: List[subprocess.Popen] = []
        self._broker_log_processes: List[subprocess.Popen] = []
        self._traffic_processes: List[subprocess.Popen] = []
        self._amqp_proxy_process: Optional[subprocess.Popen] = None  # TCP proxy gateway:port -> 127.0.0.1:5672
        self._endpoints = None
        self._ns_sim = None
        self._gateway_ip: str = ""
        self._broker_ip: str = ""  # MQTT broker / AMQP broker when run in server ns (server namespace IP)
        self._amqp_host: Optional[str] = None  # When set, AMQP uses this (e.g. gateway for host RabbitMQ fallback)
        self._amqp_port: Optional[int] = None  # When set (e.g. 25673), use proxy port so namespaces can reach host RabbitMQ
        self._log_files: list = []  # keep refs so files stay open until processes exit

    def _start_rabbitmq_log_capture(self, broker_log_dir: Path) -> None:
        """Capture host RabbitMQ service logs into the native log directory."""
        rabbit_log = (broker_log_dir / "broker_rabbitmq.log").open("w", encoding="utf-8", errors="replace")
        self._log_files.append(rabbit_log)

        capture_cmd = [
            "/bin/bash",
            "-lc",
            (
                "journalctl -u rabbitmq-server -f -n 0 -o short-iso --no-pager 2>/dev/null "
                "|| tail -n 0 -F /var/log/rabbitmq/*.log 2>/dev/null "
                "|| echo '[broker_log_capture] Unable to access RabbitMQ logs via journalctl or /var/log/rabbitmq'"
            ),
        ]

        sudo_pw = os.environ.get("FL_SUDO_PASSWORD", "").strip()
        if sudo_pw:
            cmd = ["sudo", "-E", "-S"] + capture_cmd
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=rabbit_log,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            if proc.stdin:
                proc.stdin.write((sudo_pw + "\n").encode())
                proc.stdin.close()
        else:
            proc = subprocess.Popen(
                capture_cmd,
                stdout=rabbit_log,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )

        self._broker_log_processes.append(proc)

    def _effective_num_rounds(self) -> Optional[int]:
        """
        Return the FL round cap to pass to the server.

        In RL Q-convergence training mode, the GUI rounds input must not control
        training length. We therefore do not forward NUM_ROUNDS unless the user
        explicitly provides RL_MAX_TRAINING_ROUNDS as a safety cap.
        """
        if self.protocol == "rl_unified" and self.use_ql_convergence:
            override = os.environ.get("RL_MAX_TRAINING_ROUNDS", "").strip()
            if not override:
                return None
            try:
                return int(override)
            except ValueError:
                print(
                    f"[WARNING] Ignoring invalid RL_MAX_TRAINING_ROUNDS={override!r}; "
                    "using server default round cap."
                )
                return None
        return self.num_rounds

    def _load_scenario(self, name: str) -> Dict[str, str]:
        from network_simulator import NetworkSimulator

        if not name:
            return {}
        if name not in NetworkSimulator.NETWORK_SCENARIOS:
            print(f"[WARNING] Unknown scenario '{name}', skipping network shaping for this direction.")
            return {}
        return dict(NetworkSimulator.NETWORK_SCENARIOS[name])

    def _get_scenario_conditions(self, name: str) -> Dict[str, str]:
        """Return scenario conditions; when USE_GAUSSIAN_DELAY=1, delay/jitter are sampled from normal distributions."""
        from network_simulator import NetworkSimulator

        if not name or name not in NetworkSimulator.NETWORK_SCENARIOS:
            return self._load_scenario(name)
        use_gaussian = os.environ.get("USE_GAUSSIAN_DELAY", "1").strip().lower() in ("1", "true", "yes")
        if not use_gaussian:
            return self._load_scenario(name)
        from network_delay_model import build_models
        sigma_factor = float(os.environ.get("GAUSSIAN_SIGMA_FACTOR", "0.05"))
        use_extra_jitter = os.environ.get("USE_EXTRA_JITTER", "0").strip().lower() in ("1", "true", "yes")
        models = build_models(NetworkSimulator.NETWORK_SCENARIOS, sigma_factor=sigma_factor)
        return NetworkSimulator.get_scenario_conditions_sampled(name, models, use_extra_jitter=use_extra_jitter)

    def _actual_upstream_scenario(self) -> str:
        """Actual tc scenario used on the wire during this native run."""
        return "excellent" if self.rl_reward_scenario else self.upstream_scenario

    def _native_congestion_level(self) -> Optional[str]:
        """Map tc congestion scenarios to active native background load levels."""
        mapping = {
            "congested_light": "light",
            "congested_moderate": "moderate",
            "congested_heavy": "heavy",
        }
        for scenario_name in (
            self.apply_tc_after_round_1,
            self.scenario,
            self.downstream_scenario,
            self.upstream_scenario,
        ):
            if scenario_name in mapping:
                return mapping[scenario_name]
        return None

    def _start_native_congestion(self, log_dir: Path) -> None:
        """Start best-effort active background load for native congested_* scenarios."""
        level = self._native_congestion_level()
        if not level or self._endpoints is None:
            return
        if subprocess.run(["which", "iperf3"], capture_output=True, text=True).returncode != 0:
            print("[WARNING] iperf3 not installed; skipping active native congestion load.")
            return

        server_ep, client_eps = self._endpoints
        if not client_eps:
            return
        traffic_dir = log_dir / "native_congestion"
        traffic_dir.mkdir(parents=True, exist_ok=True)

        jobs_by_level = {
            "light": [
                {"kind": "tcp", "port": 5301, "client_index": 0, "parallel": 1},
            ],
            "moderate": [
                {"kind": "tcp", "port": 5301, "client_index": 0, "parallel": 1},
                {"kind": "tcp", "port": 5302, "client_index": 1, "parallel": 1},
                {"kind": "udp", "port": 5303, "client_index": 0, "bandwidth": "4M"},
            ],
            "heavy": [
                {"kind": "tcp", "port": 5301, "client_index": 0, "parallel": 2},
                {"kind": "tcp", "port": 5302, "client_index": 1, "parallel": 2},
                {"kind": "udp", "port": 5303, "client_index": 0, "bandwidth": "8M"},
                {"kind": "udp", "port": 5304, "client_index": 1, "bandwidth": "12M"},
            ],
        }
        jobs = jobs_by_level.get(level, [])
        if not jobs:
            return

        print(f"[INFO] Starting native background congestion load: {level}")

        for port in sorted({job["port"] for job in jobs}):
            server_log = (traffic_dir / f"iperf3_server_{port}.log").open("w", encoding="utf-8", errors="replace")
            self._log_files.append(server_log)
            proc = subprocess.Popen(
                ["sudo", "-E", "ip", "netns", "exec", server_ep.ns_name, "iperf3", "-s", "-p", str(port)],
                stdout=server_log,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            self._traffic_processes.append(proc)

        time.sleep(1)

        for job_index, job in enumerate(jobs, start=1):
            client_ep = client_eps[job["client_index"] % len(client_eps)]
            port = int(job["port"])
            if not _wait_for_port("127.0.0.1", port, timeout_sec=10, ns_name=server_ep.ns_name):
                print(f"[WARNING] Native congestion iperf3 server on port {port} did not become ready.")
                continue

            client_log = (traffic_dir / f"iperf3_client_{job_index}_{job['kind']}_{port}.log").open(
                "w", encoding="utf-8", errors="replace"
            )
            self._log_files.append(client_log)
            client_cmd = [
                "sudo", "-E", "ip", "netns", "exec", client_ep.ns_name,
                "iperf3", "-c", server_ep.ip, "-p", str(port), "-t", "36000", "--forceflush",
            ]
            if job["kind"] == "udp":
                client_cmd.extend(["-u", "-b", job["bandwidth"], "-l", "1200"])
            else:
                client_cmd.extend(["-P", str(job.get("parallel", 1))])

            proc = subprocess.Popen(
                client_cmd,
                stdout=client_log,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
            self._traffic_processes.append(proc)
            print(f"[INFO] Native congestion flow started: {job['kind'].upper()} {client_ep.ns_name} -> {server_ep.ns_name}:{port}")

    def _stop_native_congestion(self) -> None:
        """Stop any native background load processes launched for congestion scenarios."""
        for proc in getattr(self, "_traffic_processes", []):
            if proc.poll() is not None:
                continue
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._traffic_processes = []

    def _get_latest_round_from_server_log(self, server_log_path: Path) -> int:
        """Parse server log for round completion; return the latest completed round number (0 if none)."""
        round_patterns = [
            re.compile(r"Round (\d+) - Aggregated Metrics", re.IGNORECASE),
            re.compile(r"ROUND (\d+)/\d+", re.IGNORECASE),
            re.compile(r"Completed round (\d+)", re.IGNORECASE),
        ]
        max_round = 0
        try:
            if not server_log_path.exists():
                return 0
            text = server_log_path.read_text(encoding="utf-8", errors="replace")
            for pattern in round_patterns:
                for m in pattern.finditer(text):
                    max_round = max(max_round, int(m.group(1)))
            # "Starting Round N/" means round N is starting, so round N-1 is completed
            for m in re.finditer(r"Starting Round (\d+)/", text, re.IGNORECASE):
                max_round = max(max_round, int(m.group(1)) - 1)
        except (OSError, ValueError):
            pass
        return max(0, max_round)

    def setup_network(self) -> None:
        from network_simulator import NamespaceNetworkSimulator

        base_name = f"{self.protocol}-{self.use_case}"
        sim = NamespaceNetworkSimulator(verbose=True)

        server_ep, client_eps = sim.setup_topology(base_name, self.num_clients)

        # Native: apply tc only to clients (upstream), not to the server; tc applied at both ingress and egress per client.
        # If apply_tc_after_round_1 is set, start with excellent (no tc); tc will be applied mid-run after round 1.
        downstream_conditions = {}  # No tc on server
        if self.apply_tc_after_round_1:
            upstream_conditions = self._load_scenario("excellent")
        else:
            upstream_conditions = self._get_scenario_conditions(self._actual_upstream_scenario())
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
                # Align native MQTT broker with the configured 128 KB payload cap.
                mqtt_config.write_text(
                    (
                        "listener 1883 0.0.0.0\n"
                        "allow_anonymous true\n"
                        "max_packet_size 131072\n"
                        "message_size_limit 131072\n"
                        "connection_messages true\n"
                        "log_timestamp true\n"
                        "log_type all\n"
                    ),
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
            broker_log.write("[INFO] broker_amqp.log captures RabbitMQ startup/proxy activity.\n")
            broker_log.write("[INFO] See broker_rabbitmq.log for host RabbitMQ service logs.\n")
            broker_log.flush()
            self._start_rabbitmq_log_capture(broker_log_dir)
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
        effective_num_rounds = self._effective_num_rounds()
        if effective_num_rounds is not None:
            env["NUM_ROUNDS"] = str(effective_num_rounds)
        else:
            env.pop("NUM_ROUNDS", None)
        if self.protocol == "rl_unified":
            if self.use_ql_convergence:
                # RL training: run with a single client until the learned table converges.
                env["MIN_CLIENTS"] = "1"
                env["MAX_CLIENTS"] = "1"
            else:
                env["MIN_CLIENTS"] = str(self.num_clients)
                env["MAX_CLIENTS"] = str(self.num_clients)
        else:
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
            env["PRUNING_STRUCTURED"] = "true" if self.pruning_structured else env.get("PRUNING_STRUCTURED", "false")
        # Quantization (server side)
        if self.use_quantization:
            env["USE_QUANTIZATION"] = "1"
            env["QUANTIZATION_BITS"] = str(self.quantization_bits)
            env["QUANTIZATION_STRATEGY"] = self.quantization_strategy or "parameter_quantization"
            env["QUANTIZATION_SYMMETRIC"] = "true" if self.quantization_symmetric else "false"

        if self.enable_gpu:
            # Pin server to GPU 0; put pip CUDA 12 ptxas on PATH
            env["CUDA_VISIBLE_DEVICES"] = "0"
            env["GPU_DEVICE_ID"] = "0"
            _nvcc_bin = os.path.join(
                sys.prefix, 'lib',
                f'python{sys.version_info[0]}.{sys.version_info[1]}',
                'site-packages', 'nvidia', 'cuda_nvcc', 'bin')
            if os.path.isdir(_nvcc_bin):
                env["PATH"] = _nvcc_bin + ":" + env.get("PATH", "")
            print("[INFO] GPU mode enabled for native server (GPU 0).")

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
        effective_num_rounds = self._effective_num_rounds()

        for idx, ep in enumerate(client_eps, start=1):
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"  # Flush FL_DIAG immediately for diagnostic pipeline T_actual
            env["CLIENT_ID"] = str(idx)
            env["NUM_CLIENTS"] = str(self.num_clients)
            if effective_num_rounds is not None:
                env["NUM_ROUNDS"] = str(effective_num_rounds)
            else:
                # RL Q-convergence mode: server decides when to stop; do not impose local round cap.
                env.pop("NUM_ROUNDS", None)
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
                env["USE_QL_CONVERGENCE"] = "true" if self.use_ql_convergence else "false"
                env["USE_COMMUNICATION_MODEL_REWARD"] = "true" if self.use_communication_model_reward else "false"
                if getattr(self, "rl_reward_scenario", None):
                    env["RL_REWARD_SCENARIO"] = self.rl_reward_scenario

            # Pruning configuration (client side)
            if self.use_pruning:
                env["USE_PRUNING"] = "1"
                if self.pruning_sparsity is not None:
                    env["PRUNING_SPARSITY"] = str(self.pruning_sparsity)
                env["PRUNING_STRUCTURED"] = "true" if self.pruning_structured else env.get("PRUNING_STRUCTURED", "false")
            # Quantization (client side; independent of pruning. If both enabled, pruning happens first in client flow.)
            if self.use_quantization:
                env["USE_QUANTIZATION"] = "1"
                env["QUANTIZATION_BITS"] = str(self.quantization_bits)
                env["QUANTIZATION_STRATEGY"] = self.quantization_strategy or "parameter_quantization"
                env["QUANTIZATION_SYMMETRIC"] = "true" if self.quantization_symmetric else "false"

            if self.enable_gpu:
                # Pin clients to GPU 1 if available, otherwise GPU 0
                env["CUDA_VISIBLE_DEVICES"] = "1"
                env["GPU_DEVICE_ID"] = "0"  # After CUDA_VISIBLE_DEVICES filtering, it's device 0
                _nvcc_bin = os.path.join(
                    sys.prefix, 'lib',
                    f'python{sys.version_info[0]}.{sys.version_info[1]}',
                    'site-packages', 'nvidia', 'cuda_nvcc', 'bin')
                if os.path.isdir(_nvcc_bin):
                    env["PATH"] = _nvcc_bin + ":" + env.get("PATH", "")

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
        use_gaussian = os.environ.get("USE_GAUSSIAN_DELAY", "1").strip().lower() in ("1", "true", "yes")
        actual_upstream_scenario = self._actual_upstream_scenario()
        effective_num_rounds = self._effective_num_rounds()
        scenario_for_tc_note = self.apply_tc_after_round_1 or actual_upstream_scenario
        tc_note = (
            f"  Tc: round 1 no tc; after round 1 complete, apply scenario={self.apply_tc_after_round_1} to clients.\n"
            if self.apply_tc_after_round_1
            else f"  Tc: clients only (scenario={actual_upstream_scenario}); server has no delay/loss/bandwidth.\n"
        )
        if scenario_for_tc_note:
            tc_note += (
                f"  Per-round tc: delay/jitter re-sampled from Gaussian model after each FL round (USE_GAUSSIAN_DELAY={use_gaussian}).\n"
            )
        rl_reward_note = (
            f"  RL reward scenario : {self.rl_reward_scenario} (train with actual network={actual_upstream_scenario}, reward as if in {self.rl_reward_scenario})\n"
            if getattr(self, "rl_reward_scenario", None) else ""
        )
        rounds_note = (
            "  Rounds   : auto (RL Q-convergence mode; GUI rounds ignored"
            + (
                f", safety cap={effective_num_rounds}"
                if effective_num_rounds is not None
                else ", server default safety cap"
            )
            + ")\n"
            if self.protocol == "rl_unified" and self.use_ql_convergence
            else f"  Rounds   : {self.num_rounds}\n"
        )
        print(
            f"\n=== Starting NATIVE FL experiment ===\n"
            f"  Use case : {self.use_case}\n"
            f"  Protocol : {self.protocol}\n"
            f"  Scenario : {self.scenario}\n"
            f"{rl_reward_note}"
            f"{rounds_note}"
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
        self._start_native_congestion(log_dir)

        if not os.environ.get("FL_SUDO_PASSWORD", "").strip():
            print("[INFO] Server and clients run inside namespaces via sudo; you may be prompted for your password (once per process). Set FL_SUDO_PASSWORD to avoid prompts.\n")

        # Create reset_epsilon flag file for RL-unified protocol
        if self.protocol == "rl_unified":
            print(f"\n{'='*70}")
            if self.reset_epsilon:
                print(f"[Q-Learning] Preparing epsilon reset for new experiment: {self.scenario}")
            else:
                print(f"[Q-Learning] Continuing with previous epsilon (resume mode): {self.scenario}")
            print(f"{'='*70}")
            
            # Set environment variable to signal epsilon reset
            os.environ["RESET_EPSILON"] = "true" if self.reset_epsilon else "false"
            
            # Create a flag file in shared_data for clients to check
            shared_data_path = PROJECT_ROOT / "shared_data"
            shared_data_path.mkdir(exist_ok=True)
            
            # Create flag file with scenario identifier, unique experiment ID, and timestamp
            reset_flag_file = shared_data_path / "reset_epsilon_flag.txt"
            try:
                import uuid
                timestamp = time.time()
                experiment_id = str(uuid.uuid4())[:8]  # Short unique ID for this experiment
                with open(reset_flag_file, 'w') as f:
                    f.write(f"experiment_id={experiment_id}\n")
                    f.write(f"scenario={self.scenario}\n")
                    f.write(f"timestamp={timestamp}\n")
                    f.write(f"reset_epsilon={1.0 if self.reset_epsilon else 0.0}\n")  # 1.0 = reset, 0.0 = continue
                print(f"[Q-Learning] ✓ Created flag file: {reset_flag_file}")
                print(f"[Q-Learning]   Experiment ID: {experiment_id}")
                print(f"[Q-Learning]   Scenario: {self.scenario}")
                if self.reset_epsilon:
                    print(f"[Q-Learning]   All clients will reset epsilon to 1.0 on initialization")
                    print(f"[Q-Learning]   Note: Q-table will persist across scenarios for multi-scenario training")
                else:
                    print(f"[Q-Learning]   Clients will continue with previous epsilon value (resume mode)")
                    print(f"[Q-Learning]   Note: Q-table, rewards, and learning progress will continue from previous state")
            except Exception as e:
                print(f"[Q-Learning] ✗ Warning: Could not create reset flag file: {e}")
                print(f"[Q-Learning]   Epsilon behavior may not work as expected")
            
            print(f"{'='*70}\n")

        exit_code = 0
        server_proc = None
        client_procs: List[subprocess.Popen] = []

        try:
            server_proc = self._spawn_server(log_file=server_log)
            # Small delay to let server bind its port
            time.sleep(5)
            client_procs = self._spawn_clients(client_log_files=client_logs)

            # Wait for server completion; when it exits, we stop clients as well.
            # Per-round tc: (re-)apply delay/jitter from Gaussian model after each FL round (ingress + egress).
            print("[INFO] Do not interrupt (Ctrl+C/Stop) until the server finishes all rounds.")
            client_log_paths = [log_dir / f"client_{i}.log" for i in range(1, self.num_clients + 1)]
            server_log_path = log_dir / "server.log"
            poll_interval = 2
            round1_wait_sec = int(os.environ.get("FL_DIAG_ROUND1_WAIT_SEC", "0"))
            deadline = (time.monotonic() + round1_wait_sec) if round1_wait_sec > 0 else None
            if self.apply_tc_after_round_1 and deadline is None:
                print("[INFO] Waiting for round 1 completion (no timeout; set FL_DIAG_ROUND1_WAIT_SEC>0 for a limit).")
            scenario_for_tc = self.apply_tc_after_round_1 or actual_upstream_scenario
            last_round_tc_applied = 0  # after each round completion we apply tc for the *next* round's send

            while (deadline is None or time.monotonic() < deadline) and server_proc.poll() is None:
                # Round completion: from client FL_DIAG O_send= count (if FL_DIAGNOSTIC_PIPELINE=1) and/or server log
                counts = []
                for path in client_log_paths:
                    try:
                        text = path.read_text(encoding="utf-8", errors="replace")
                        n = sum(1 for line in text.splitlines() if "FL_DIAG" in line and "O_send=" in line)
                        counts.append(n)
                    except (IOError, OSError):
                        counts.append(0)
                min_sends = min(counts) if counts else 0
                round_from_server = self._get_latest_round_from_server_log(server_log_path)
                current_completed_rounds = max(min_sends, round_from_server)

                if current_completed_rounds > last_round_tc_applied and scenario_for_tc:
                    server_ep, client_eps = self._endpoints
                    conditions = self._get_scenario_conditions(scenario_for_tc)
                    if conditions:
                        for ep in client_eps:
                            self._ns_sim.apply_tc(ep, conditions)  # ingress + egress per client
                        next_round = last_round_tc_applied + 1
                        print(f"[INFO] Tc (ingress+egress) re-sampled for round {next_round + 1} send (Gaussian delay/jitter): {conditions.get('latency', '')} {conditions.get('jitter', '')}")
                    last_round_tc_applied = current_completed_rounds
                time.sleep(poll_interval)

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
            for proc in getattr(self, "_broker_log_processes", []):
                if proc.poll() is None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            self._broker_log_processes = []
            self._stop_native_congestion()
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

    protocol_choices = ["mqtt", "amqp", "grpc", "quic", "http3", "dds", "rl_unified"]
    scenario_choices = [
        "excellent",
        "good",
        "moderate",
        "poor",
        "very_poor",
        "satellite",
        "congested_light",
        "congested_moderate",
        "congested_heavy",
    ]

    parser.add_argument(
        "--use-case",
        "-u",
        choices=["emotion", "mentalstate", "temperature"],
        required=True,
        help="Use case to run experiments for.",
    )
    protocol_group = parser.add_mutually_exclusive_group(required=True)
    protocol_group.add_argument(
        "--protocol",
        "-p",
        choices=protocol_choices,
        help="Single communication protocol.",
    )
    protocol_group.add_argument(
        "--protocols",
        nargs="+",
        choices=protocol_choices,
        help="One or more communication protocols; runs each protocol × scenario combination.",
    )

    scenario_group = parser.add_mutually_exclusive_group(required=True)
    scenario_group.add_argument(
        "--scenario",
        "-s",
        choices=scenario_choices,
        help="Single base network scenario to apply.",
    )
    scenario_group.add_argument(
        "--scenarios",
        nargs="+",
        choices=scenario_choices,
        help="One or more network scenarios; runs each protocol × scenario combination.",
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
        "--termination-mode",
        choices=["client_convergence", "fixed_rounds"],
        default="client_convergence",
        help="End condition: default ends on client convergence (may stop early), fixed_rounds runs selected rounds.",
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
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        help="Enable model quantization (sets USE_QUANTIZATION). If pruning is also enabled, flow is prune then quantize.",
    )
    parser.add_argument(
        "--quantization-bits",
        type=int,
        default=8,
        choices=[4, 8, 16, 32],
        help="Quantization bit width (default: 8). 4-bit uses nibble packing for 2x compression.",
    )
    parser.add_argument(
        "--quantization-strategy",
        type=str,
        default="parameter_quantization",
        help="Quantization strategy (e.g. parameter_quantization, full_quantization).",
    )
    parser.add_argument(
        "--quantization-symmetric",
        action="store_true",
        help="Use symmetric quantization.",
    )
    parser.add_argument(
        "--dds-impl",
        choices=["cyclonedds", "fastdds"],
        default="cyclonedds",
        help="DDS implementation vendor to use when protocol='dds'.",
    )
    parser.add_argument(
        "--rl-reward-scenario",
        metavar="SCENARIO",
        help="For protocol=rl_unified: train in excellent conditions but use reward/state for SCENARIO (e.g. good, moderate, poor). Sets RL_REWARD_SCENARIO for clients.",
    )
    parser.add_argument(
        "--use-ql-convergence",
        action="store_true",
        help="For protocol=rl_unified: run RL training mode and stop when the learned Q-table converges.",
    )
    parser.add_argument(
        "--disable-communication-model-reward",
        action="store_true",
        help="For protocol=rl_unified: do not let communication-model T_calc affect RL rewards.",
    )
    parser.add_argument(
        "--no-reset-epsilon",
        action="store_true",
        help="For protocol=rl_unified training: do NOT reset epsilon to 1.0 on start. Continue with previous epsilon value (useful for resuming interrupted training).",
    )

    return parser.parse_args()


def main() -> None:
    _load_local_privileged_env()
    args = parse_args()
    
    # Propagate selected termination mode into native process env.
    os.environ["STOP_ON_CLIENT_CONVERGENCE"] = "false" if args.termination_mode == "fixed_rounds" else "true"

    protocols = list(dict.fromkeys((args.protocols or [args.protocol]) if hasattr(args, "protocols") else [args.protocol]))
    scenarios = list(dict.fromkeys((args.scenarios or [args.scenario]) if hasattr(args, "scenarios") else [args.scenario]))

    # Propagate DDS implementation choice so native DDS server/clients can switch vendor based on DDS_IMPL
    if getattr(args, "dds_impl", None):
        os.environ["DDS_IMPL"] = args.dds_impl

    use_quant = getattr(args, "use_quantization", False)
    run_results: List[Tuple[str, str, int]] = []

    for protocol in protocols:
        for scenario in scenarios:
            run_num_clients = args.num_clients
            if protocol == "rl_unified" and args.use_ql_convergence and run_num_clients != 1:
                print(f"[INFO] RL unified: forcing num_clients=1 for training (was {run_num_clients}) for protocol={protocol}, scenario={scenario}.")
                run_num_clients = 1

            print(
                f"\n[RUN] Starting native experiment {len(run_results) + 1}/{len(protocols) * len(scenarios)}"
                f" -> protocol={protocol}, scenario={scenario}\n"
            )

            runner = NativeExperimentRunner(
                use_case=args.use_case,
                protocol=protocol,
                scenario=scenario,
                num_rounds=args.rounds,
                num_clients=run_num_clients,
                downstream_scenario=args.downstream_scenario,
                upstream_scenario=args.upstream_scenario,
                enable_gpu=args.enable_gpu,
                use_pruning=args.use_pruning,
                pruning_sparsity=args.pruning_sparsity,
                pruning_structured=args.pruning_structured,
                use_quantization=use_quant,
                quantization_bits=getattr(args, "quantization_bits", 8),
                quantization_strategy=getattr(args, "quantization_strategy", "parameter_quantization"),
                quantization_symmetric=getattr(args, "quantization_symmetric", False),
                use_ql_convergence=getattr(args, "use_ql_convergence", False),
                rl_reward_scenario=getattr(args, "rl_reward_scenario", None),
                use_communication_model_reward=not getattr(args, "disable_communication_model_reward", False),
                reset_epsilon=not getattr(args, "no_reset_epsilon", False),  # Default True (reset), --no-reset-epsilon makes it False
            )
            _install_signal_handlers(runner)
            code = runner.run()
            run_results.append((protocol, scenario, code))

    if len(run_results) > 1:
        print("\n=== Native experiment summary ===")
        for protocol, scenario, code in run_results:
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"  - protocol={protocol}, scenario={scenario}: {status}")

    failed = [item for item in run_results if item[2] != 0]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

