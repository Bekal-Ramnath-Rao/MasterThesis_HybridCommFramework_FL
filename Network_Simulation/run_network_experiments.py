#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Network Experiment Runner
Runs FL experiments across all protocols and network conditions
"""

import subprocess
import time
import json
import os
import sys
import re
import shutil
from datetime import datetime
from typing import List, Dict, Optional
import argparse
from pathlib import Path
import tempfile
import glob

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Load .env from this directory so FL_SUDO_PASSWORD is set (for host tc). .env is gitignored.
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    try:
        with open(_env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line and line.split("=", 1)[0].strip() == "FL_SUDO_PASSWORD":
                    key, _, val = line.partition("=")
                    os.environ[key.strip()] = val.strip().strip("'\"").strip()
                    break
    except Exception:
        pass

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    from Client.battery_model import (
        BATTERY_CAP_J,
        PROTOCOL_CPU_BETA,
        PROTOCOL_ENERGY_ALPHA,
        E_fixed,
        P_CPU_MAX,
        k_tx,
        k_rx,
    )
except ImportError:
    BATTERY_CAP_J = 60.0 * 3600.0
    PROTOCOL_CPU_BETA = {"mqtt": 1.0, "amqp": 1.05, "grpc": 1.1, "quic": 1.05, "http3": 1.15, "dds": 1.0}
    PROTOCOL_ENERGY_ALPHA = {"mqtt": 1.0, "amqp": 1.1, "grpc": 1.2, "quic": 1.1, "http3": 1.25, "dds": 1.1}
    E_fixed = 0.1
    P_CPU_MAX = 10.0
    k_tx = 1e-8
    k_rx = 1e-8


def parse_fl_dataset_for_client(entries: Optional[List[str]]) -> Optional[Dict[int, int]]:
    """Parse --fl-dataset-for-client CLIENT=SHARD into a map of CLIENT_ID -> shard index (1-based)."""
    if not entries:
        return None
    out: Dict[int, int] = {}
    for raw in entries:
        item = str(raw).strip()
        if "=" not in item:
            continue
        left, _, right = item.partition("=")
        out[int(left.strip())] = int(right.strip())
    return out or None


class ExperimentRunner:
    """Automates running FL experiments across different network conditions"""
    
    def __init__(
        self,
        use_case: str = "emotion",
        num_rounds: int = 10,
        enable_congestion: bool = False,
        use_quantization: bool = False,
        quantization_params: Dict[str, str] = None,
        use_pruning: bool = False,
        pruning_params: Dict[str, str] = None,
        enable_gpu: bool = False,
        network_mode: str = "gpu",
        baseline_mode: bool = False,
        use_ql_convergence: bool = False,
        rl_inference_only: bool = False,
        local_clients: int = 2,
        min_clients: Optional[int] = None,
        use_communication_model_reward: bool = True,
        reset_epsilon: bool = True,
        dataset_client_map: Optional[Dict[int, int]] = None,
    ):
        self.use_case = use_case
        self.num_rounds = num_rounds
        self.enable_congestion = enable_congestion
        self.use_quantization = use_quantization
        self.use_pruning = use_pruning
        self.enable_gpu = enable_gpu
        self.network_mode = (network_mode or "gpu").lower().strip()
        if self.network_mode not in ("gpu", "host", "host_macvlan"):
            self.network_mode = "gpu"
        self.baseline_mode = baseline_mode
        self.use_ql_convergence = use_ql_convergence
        self.rl_inference_only = bool(rl_inference_only)
        self.use_communication_model_reward = use_communication_model_reward
        self.reset_epsilon = reset_epsilon
        # Number of client containers started from this runner on the central machine (0 = server/brokers only).
        self.local_clients = max(0, int(local_clients))
        # Total participants the server should wait for (local + remote); None => same as local_clients
        self.min_clients = int(min_clients) if min_clients is not None else None
        # quantization_params expected to be a dict of simple string values
        self.quantization_params = quantization_params or {}
        # pruning_params expected to be a dict of simple string values
        self.pruning_params = pruning_params or {}
        self._runtime_compose_files: Dict[str, str] = {}
        self._runtime_compose_files_stop: Dict[str, str] = {}
        self._runtime_compose_files_num_clients: Dict[str, str] = {}
        self._runtime_compose_files_cpu: Dict[str, str] = {}
        self.dataset_client_map: Dict[int, int] = dict(dataset_client_map or {})
        self._runtime_compose_files_dataset: Dict[tuple, str] = {}
        
        # Map use_case values to actual Server directory names
        self.use_case_dir_map = {
            "emotion": "Emotion_Recognition",
            "mentalstate": "MentalState_Recognition",
            "temperature": "Temperature_Regulation"
        }
        self.use_case_dir = self.use_case_dir_map.get(use_case, f"{use_case.title()}_Recognition")

        # Print compression (quantization + pruning) status
        print(f"\n{'='*70}")
        if self.use_quantization:
            print("QUANTIZATION ENABLED")
            print(f"Quantization params: {self.quantization_params}")
        else:
            print("QUANTIZATION DISABLED")
        if self.use_pruning:
            print("PRUNING ENABLED")
            print(f"Pruning params: {self.pruning_params}")
        else:
            print("PRUNING DISABLED")
        if self.dataset_client_map:
            print(f"DATASET_CLIENT_ID map (by CLIENT_ID): {self.dataset_client_map}")
        print(f"{'='*70}\n")
        
        # Get the script's directory and project root
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        # Used when compose is copied to /tmp: build context and volumes must be absolute
        self.project_root = project_root.resolve()
        self.docker_dir = (project_root / "Docker").resolve()

        # Build descriptive folder name
        if baseline_mode:
            # Baseline results go to dedicated baseline folder
            _uc_b = (use_case or "emotion").lower()
            if _uc_b == "mentalstate":
                _uc_b = "mental_state"
            folder_name = _uc_b
            self.results_dir = project_root / "experiment_results_baseline" / folder_name
        else:
            # Regular experiments
            # Canonical use_case token in folder names (matches server ``get_experiment_results_dir``).
            _uc_folder = (use_case or "emotion").lower()
            if _uc_folder == "mentalstate":
                _uc_folder = "mental_state"
            folder_parts = [_uc_folder]
            if use_quantization:
                folder_parts.append("quantized")
                if quantization_params.get('QUANTIZATION_BITS'):
                    folder_parts.append(f"{quantization_params['QUANTIZATION_BITS']}bit")
            if use_pruning:
                folder_parts.append("pruned")
                # Try to include pruning sparsity percentage when available
                sparsity_str = (pruning_params or {}).get("PRUNING_SPARSITY")
                if sparsity_str is not None:
                    try:
                        sparsity_val = float(sparsity_str)
                        # Interpret values > 1.0 as already-percentage
                        if sparsity_val <= 1.0:
                            sparsity_pct = int(round(sparsity_val * 100))
                        else:
                            sparsity_pct = int(round(sparsity_val))
                        folder_parts.append(f"{sparsity_pct}pct")
                    except (ValueError, TypeError):
                        pass
            if enable_congestion:
                folder_parts.append("congestion")
            folder_parts.append(datetime.now().strftime('%Y%m%d_%H%M%S'))
            folder_name = "_".join(folder_parts)
            self.results_dir = project_root / "experiment_results" / folder_name
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = str(self.results_dir)
        # Align in-container writes (``get_experiment_results_dir``) with this run's host folder.
        if "experiment_results_baseline" not in self.results_dir.replace("\\", "/"):
            os.environ.setdefault(
                "EXPERIMENT_RESULTS_RUN_SEGMENT",
                os.path.basename(self.results_dir),
            )

        # Initialize congestion manager if enabled
        self.congestion_manager = None
        if enable_congestion:
            # Import congestion_manager module
            if str(script_dir) not in sys.path:
                sys.path.insert(0, str(script_dir))
            try:
                from congestion_manager import CongestionManager
                self.congestion_manager = CongestionManager(use_case=use_case, verbose=False)
                print(f"[INFO] Congestion manager initialized")
            except ImportError as e:
                print(f"[WARNING] Could not import CongestionManager: {e}")
                print(f"[WARNING] Congestion features will be disabled")
                self.enable_congestion = False
        
        # Define protocols to test
        self.protocols = ["mqtt", "amqp", "grpc", "quic", "http3", "dds", "rl_unified"]
        
        # Define network scenarios to test
        self.network_scenarios = [
            "excellent",
            "good",
            "moderate",
            "poor",
            "very_poor",
            "satellite",
            "congested_light",
            "congested_moderate",
            "congested_heavy",
            # Dynamic: draw_dynamic_base_scenario() (shuffle-bag by default)
            "dynamic",
        ]
        
        # Docker compose file mapping (relative to project root)
        docker_dir = project_root / "Docker"
        
        # Three network modes: gpu (Docker bridge), host (tc on host), host_macvlan (macvlan, per-container tc)
        if self.network_mode == "host":
            # Host network: *.host-network.yml (network_mode: host); tc applied on host interface
            self.compose_files = {
                "emotion": str(docker_dir / "docker-compose-emotion.host-network.yml"),
                "mentalstate": str(docker_dir / "docker-compose-mentalstate.host-network.yml"),
                "temperature": str(docker_dir / "docker-compose-temperature.host-network.yml")
            }
            self.unified_compose_files = {
                "emotion": str(docker_dir / "docker-compose-unified-emotion.yml"),
                "mentalstate": str(docker_dir / "docker-compose-unified-mentalstate.host-network.yml"),
                "temperature": str(docker_dir / "docker-compose-unified-temperature.host-network.yml")
            }
            self.gpu_overlay_files = {}
            self.macvlan_overlay_files = {}
            self.macvlan_overlay_unified = None
        elif self.network_mode == "host_macvlan":
            # Host + macvlan: *.macvlan.yml (external fl-macvlan); tc applied per container
            self.compose_files = {
                "emotion": str(docker_dir / "docker-compose-emotion.macvlan.yml"),
                "mentalstate": str(docker_dir / "docker-compose-mentalstate.macvlan.yml"),
                "temperature": str(docker_dir / "docker-compose-temperature.macvlan.yml")
            }
            self.unified_compose_files = {
                "emotion": str(docker_dir / "docker-compose-unified-emotion.macvlan.yml"),
                "mentalstate": str(docker_dir / "docker-compose-unified-mentalstate.macvlan.yml"),
                "temperature": str(docker_dir / "docker-compose-unified-temperature.macvlan.yml")
            }
            self.gpu_overlay_files = {}
            self.macvlan_overlay_files = {}
            self.macvlan_overlay_unified = None
        elif enable_gpu:
            # GPU + Docker bridge: use gpu-isolated compose (bridge network, no host mode)
            self.compose_files = {
                "emotion": str(docker_dir / "docker-compose-emotion.gpu-isolated.yml"),
                "mentalstate": str(docker_dir / "docker-compose-mentalstate.gpu-isolated.yml"),
                "temperature": str(docker_dir / "docker-compose-temperature.gpu-isolated.yml")
            }
            # Unified uses host-network compose so CycloneDDS RTPS (ephemeral UDP ports) can
            # reach all participants.  The bridge variant only publishes SPDP ports and breaks
            # DDS uplink/downlink; single-protocol DDS in gpu-isolated already uses host_mode.
            self.unified_compose_files = {
                "emotion": str(docker_dir / "docker-compose-unified-emotion.yml"),
                "mentalstate": str(docker_dir / "docker-compose-unified-mentalstate.yml"),
                "temperature": str(docker_dir / "docker-compose-unified-temperature.yml")
            }
            self.gpu_overlay_files = {}
            self.macvlan_overlay_files = {}
            self.macvlan_overlay_unified = None
        else:
            # Non-GPU + Docker bridge: standard compose files
            self.compose_files = {
                "emotion": str(docker_dir / "docker-compose-emotion.yml"),
                "mentalstate": str(docker_dir / "docker-compose-mentalstate.yml"),
                "temperature": str(docker_dir / "docker-compose-temperature.yml")
            }
            # Same as GPU path above: unified must use host-network so DDS works.
            # docker-compose-unified-emotion.bridge.yml is kept for non-DDS protocol-only
            # deployments; the rl_unified experiment includes DDS so host mode is required.
            self.unified_compose_files = {
                "emotion": str(docker_dir / "docker-compose-unified-emotion.yml"),
                "mentalstate": str(docker_dir / "docker-compose-unified-mentalstate.yml"),
                "temperature": str(docker_dir / "docker-compose-unified-temperature.yml")
            }
            self.gpu_overlay_files = {
                "emotion": str(docker_dir / "docker-compose-emotion.gpu.yml"),
                "mentalstate": str(docker_dir / "docker-compose-mentalstate.gpu.yml"),
                "temperature": str(docker_dir / "docker-compose-temperature.gpu.yml")
            }
            self.macvlan_overlay_files = {}
            self.macvlan_overlay_unified = None
        
        # Service name patterns (broker names differ: host/gpu-isolated use amqp-broker, standard bridge uses rabbitmq)
        broker_mqtt = "mqtt-broker"
        broker_amqp = "amqp-broker" if (self.network_mode in ("host", "host_macvlan") or enable_gpu) else "rabbitmq"

        # gpu-isolated and macvlan compose files use -mentalstate suffix with plain broker names;
        # host-network and standard bridge compose files use -mental suffix with namespaced broker names.
        _ms_new_naming = enable_gpu or self.network_mode == "host_macvlan"
        _ms = "mentalstate" if _ms_new_naming else "mental"
        _ms_mqtt_broker = "mqtt-broker" if _ms_new_naming else "mqtt-broker-mental"
        _ms_amqp_broker = "amqp-broker" if _ms_new_naming else "rabbitmq-mental"

        # gpu-isolated and macvlan compose files use -temperature suffix with plain broker names;
        # host-network compose uses -temp suffix with plain broker names;
        # standard bridge compose uses -temp suffix with fl- prefixed broker names.
        _temp_new_naming = enable_gpu or self.network_mode == "host_macvlan"
        _temp = "temperature" if _temp_new_naming else "temp"
        if _temp_new_naming:
            _temp_mqtt_broker = "mqtt-broker-temperature"
            _temp_amqp_broker = "amqp-broker-temperature"
        elif self.network_mode == "host":
            _temp_mqtt_broker = "mqtt-broker-temp"
            _temp_amqp_broker = "rabbitmq-temp"
        else:
            _temp_mqtt_broker = "fl-mqtt-broker-temp"
            _temp_amqp_broker = "fl-rabbitmq-temp"

        self.service_patterns = {
            "emotion": {
                "mqtt": [broker_mqtt, "fl-server-mqtt-emotion", "fl-client-mqtt-emotion-1", "fl-client-mqtt-emotion-2", "fl-client-mqtt-emotion-3"],
                "amqp": [broker_amqp, "fl-server-amqp-emotion", "fl-client-amqp-emotion-1", "fl-client-amqp-emotion-2", "fl-client-amqp-emotion-3"],
                "grpc": ["fl-server-grpc-emotion", "fl-client-grpc-emotion-1", "fl-client-grpc-emotion-2", "fl-client-grpc-emotion-3"],
                "quic": ["fl-server-quic-emotion", "fl-client-quic-emotion-1", "fl-client-quic-emotion-2", "fl-client-quic-emotion-3"],
                "http3": ["fl-server-http3-emotion", "fl-client-http3-emotion-1", "fl-client-http3-emotion-2", "fl-client-http3-emotion-3"],
                "dds": ["fl-server-dds-emotion", "fl-client-dds-emotion-1", "fl-client-dds-emotion-2", "fl-client-dds-emotion-3"],
                "rl_unified": ["fl-server-unified-emotion", "fl-client-unified-emotion-1", "fl-client-unified-emotion-2", "fl-client-unified-emotion-3"]
            },
            "mentalstate": {
                "mqtt": [_ms_mqtt_broker, f"fl-server-mqtt-{_ms}", f"fl-client-mqtt-{_ms}-1", f"fl-client-mqtt-{_ms}-2"],
                "amqp": [_ms_amqp_broker, f"fl-server-amqp-{_ms}", f"fl-client-amqp-{_ms}-1", f"fl-client-amqp-{_ms}-2"],
                "grpc": [f"fl-server-grpc-{_ms}", f"fl-client-grpc-{_ms}-1", f"fl-client-grpc-{_ms}-2"],
                "quic": [f"fl-server-quic-{_ms}", f"fl-client-quic-{_ms}-1", f"fl-client-quic-{_ms}-2"],
                "http3": [f"fl-server-http3-{_ms}", f"fl-client-http3-{_ms}-1", f"fl-client-http3-{_ms}-2"],
                "dds": [f"fl-server-dds-{_ms}", f"fl-client-dds-{_ms}-1", f"fl-client-dds-{_ms}-2"],
                "rl_unified": ["fl-server-unified-mentalstate", "fl-client-unified-mentalstate-1", "fl-client-unified-mentalstate-2"]
            },
            "temperature": {
                "mqtt": [_temp_mqtt_broker, f"fl-server-mqtt-{_temp}", f"fl-client-mqtt-{_temp}-1", f"fl-client-mqtt-{_temp}-2"],
                "amqp": [_temp_amqp_broker, f"fl-server-amqp-{_temp}", f"fl-client-amqp-{_temp}-1", f"fl-client-amqp-{_temp}-2"],
                "grpc": [f"fl-server-grpc-{_temp}", f"fl-client-grpc-{_temp}-1", f"fl-client-grpc-{_temp}-2"],
                "quic": [f"fl-server-quic-{_temp}", f"fl-client-quic-{_temp}-1", f"fl-client-quic-{_temp}-2"],
                "http3": [f"fl-server-http3-{_temp}", f"fl-client-http3-{_temp}-1", f"fl-client-http3-{_temp}-2"],
                "dds": [f"fl-server-dds-{_temp}", f"fl-client-dds-{_temp}-1", f"fl-client-dds-{_temp}-2"],
                "rl_unified": [f"fl-server-unified-{_temp}", f"fl-client-unified-{_temp}-1", f"fl-client-unified-{_temp}-2"]
            }
        }
    
    def run_command(self, command: List[str], check=True, env_vars: Dict[str, str] = None, cwd: str = None) -> subprocess.CompletedProcess:
        """Execute a shell command with optional environment variables"""
        print(f"[CMD] {' '.join(command)}")
        env = os.environ.copy()
        # Inject quantization-related environment variables for subprocesses when enabled
        if getattr(self, 'use_quantization', False):
            env.update({"USE_QUANTIZATION": "1"})
            # Add any explicit quantization params passed to the runner
            for k, v in self.quantization_params.items():
                # Ensure keys are prefixed consistently (caller should pass keys like 'QUANTIZATION_BITS')
                env[str(k)] = str(v)
            # Print quantization env vars for debugging
            quant_env = {k: v for k, v in env.items() if 'QUANTIZATION' in k or k == 'USE_QUANTIZATION'}
            if quant_env:
                print(f"[QUANTIZATION ENV] {quant_env}")
        # Inject pruning-related environment variables for subprocesses when enabled
        if getattr(self, 'use_pruning', False):
            env.update({"USE_PRUNING": "1"})
            for k, v in self.pruning_params.items():
                env[str(k)] = str(v)
            prune_env = {k: v for k, v in env.items() if 'PRUNING' in k or k == 'USE_PRUNING'}
            if prune_env:
                print(f"[PRUNING ENV] {prune_env}")
        if env_vars:
            env.update(env_vars)
            print(f"[ENV] {env_vars}")
        # Use project root as working directory if not specified
        if cwd is None:
            cwd = str(Path(__file__).parent.parent)
        return subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', check=check, env=env, cwd=cwd)

    def _experiment_use_case_results_name(self) -> str:
        """Folder name under experiment_results mount: emotion | mental_state | temperature."""
        u = (self.use_case or "emotion").lower()
        if u == "mentalstate":
            return "mental_state"
        return u

    def _rl_unified_training_results_paths_in_container(self, scenario: Optional[str]) -> List[str]:
        """In-container paths for unified server snapshot JSON (stable filename for the experiment runner)."""
        uc = self._experiment_use_case_results_name()
        ucl = (self.use_case or "emotion").lower()
        run_seg = os.environ.get("EXPERIMENT_RESULTS_RUN_SEGMENT", "").strip()
        scen = (scenario or "default").strip() or "default"
        paths: List[str] = []
        for base in ("/app/results", "/app/experiment_results"):
            if run_seg:
                paths.append(f"{base}/{run_seg}/unified/{scen}/rl_unified_training_results.json")
                paths.append(f"{base}/{run_seg}/unified/default/rl_unified_training_results.json")
            paths.append(f"{base}/{uc}/unified/{scen}/rl_unified_training_results.json")
            paths.append(f"{base}/{uc}/unified/default/rl_unified_training_results.json")
            if ucl != uc:
                paths.append(f"{base}/{ucl}/unified/{scen}/rl_unified_training_results.json")
                paths.append(f"{base}/{ucl}/unified/default/rl_unified_training_results.json")
        return paths

    def _running_docker_container_names(self) -> set:
        """Names of currently running containers (``docker ps``)."""
        result = self.run_command(["docker", "ps", "--format", "{{.Names}}"], check=False)
        if result.returncode != 0 or not (result.stdout or "").strip():
            return set()
        return {n.strip() for n in result.stdout.strip().splitlines() if n.strip()}

    def _docker_container_exists(self, name: str) -> bool:
        """True if a container exists (running or stopped) with this exact name."""
        r = self.run_command(["docker", "inspect", "-f", "{{.Id}}", name], check=False)
        return r.returncode == 0 and bool((r.stdout or "").strip())

    def _delete_stale_results_files(self, protocol: str, scenario: str) -> None:
        """Delete stale results files from volume-mounted host paths before starting containers.

        For protocols whose server container bind-mounts a host directory as /app/results
        (notably rl_unified via ../experiment_results:/app/results), any JSON results file
        left over from a previous run is immediately visible in the new container, causing
        wait_for_completion to declare false success on the very first poll.

        For other protocols, the results directory is NOT volume-mounted (results live only
        inside the container and are lost on docker compose down), so no cleanup is needed
        beyond the .dockerignore fix that prevents stale files from being baked into images.
        """
        use_case_lc = self.use_case.lower()
        use_case_res = self._experiment_use_case_results_name()
        scenario_norm = (scenario or "default").strip() or "default"
        run_seg = os.environ.get("EXPERIMENT_RESULTS_RUN_SEGMENT", "").strip()

        if protocol == "rl_unified":
            # rl_unified server mounts ../experiment_results → /app/results in container.
            # Results are written under .../unified/{network_scenario}/rl_unified_training_results.json.
            # When the GUI runs multiple scenarios in one batch, a finished scenario (e.g. excellent)
            # must not leave a snapshot that wait_for_completion mistakes for the *next* scenario.
            # So remove runner snapshots under every unified/* subdir for this use case's run segments.
            base = self.project_root / "experiment_results"
            filenames = ["rl_unified_training_results.json", "rl_unified_training_results.csv"]
            import subprocess as _sp

            def _rm_stale(stale: Path) -> None:
                if not stale.exists():
                    return
                try:
                    stale.unlink()
                    print(f"[INFO] Removed stale results file: {stale}")
                except PermissionError:
                    try:
                        _sp.run(["sudo", "rm", "-f", str(stale)], check=True, timeout=10)
                        print(f"[INFO] Removed stale results file (sudo): {stale}")
                    except Exception as e2:
                        print(f"[WARN] Could not remove stale file {stale}: {e2}")
                except OSError as e:
                    print(f"[WARN] Could not remove stale file {stale}: {e}")

            segs: List[str] = []
            if run_seg:
                segs.append(run_seg)
            segs.append(use_case_res)
            if use_case_lc != use_case_res:
                segs.append(use_case_lc)
            seen_seg: set = set()
            for seg in segs:
                if not seg or seg in seen_seg:
                    continue
                seen_seg.add(seg)
                unified_root = base / seg / "unified"
                if not unified_root.is_dir():
                    continue
                try:
                    for child in unified_root.iterdir():
                        if not child.is_dir():
                            continue
                        for fname in filenames:
                            _rm_stale(child / fname)
                except OSError as e:
                    print(f"[WARN] Could not scan {unified_root}: {e}")

    def _running_fl_client_containers(self, protocol: str) -> List[str]:
        """Client service names from the protocol list that are actually running (e.g. RL single-client)."""
        running = self._running_docker_container_names()
        services = self.service_patterns[self.use_case][protocol]
        return [
            s
            for s in services
            if "client" in s
            and "broker" not in s.lower()
            and "rabbitmq" not in s.lower()
            and s in running
        ]

    def _total_expected_federation_clients(self) -> int:
        """Clients the server waits for (local + remote). At least the number started on this host."""
        local = self.local_clients
        if self.min_clients is None:
            return local
        return max(local, int(self.min_clients))

    def _compose_project_name(self) -> str:
        """
        Stable Docker Compose project name. Runtime-patched compose files live under /tmp; without an
        explicit -p, Compose would use the /tmp directory name (e.g. fl_runtime_compose) and try to
        recreate containers that already exist with fixed container_name from the Docker/ stack.
        """
        return self.docker_dir.name.lower()

    def _docker_compose_base(self, compose_file: str) -> List[str]:
        """``docker compose -p <project> -f <file>`` prefix for all compose invocations."""
        return ["docker", "compose", "-p", self._compose_project_name(), "-f", compose_file]

    def _compose_requires_runtime_patch(self) -> bool:
        """Whether compose files should be patched at runtime to expose pruning env vars."""
        return bool(getattr(self, "use_pruning", False))

    def _absolutize_compose_paths_for_runtime_copy(self, content: str) -> str:
        """
        Runtime-patched compose files live under /tmp; Docker resolves ``context: ..`` and
        ``../...`` volumes relative to that file, which breaks (e.g. /tmp/Server).
        Rewrite to absolute paths anchored at the real project and Docker/ directory.
        """
        pr = str(self.project_root)
        dd = str(self.docker_dir)
        out_lines: List[str] = []
        vol_re = re.compile(r"^(\s*-\s*)((?:\.\./|\./)[^:]+)(:.+)?\s*$")
        for line in content.splitlines():
            if re.match(r"^\s*context:\s*\.\.\s*$", line):
                ind = re.match(r"^(\s*)", line).group(1)
                line = f"{ind}context: {pr}"
            elif re.match(r"^\s*context:\s*\.\s*$", line):
                ind = re.match(r"^(\s*)", line).group(1)
                line = f"{ind}context: {dd}"
            else:
                m = vol_re.match(line)
                if m:
                    host = m.group(2)
                    abs_host = str((self.docker_dir / host).resolve())
                    line = m.group(1) + abs_host + (m.group(3) or "")
            out_lines.append(line)
        result = "\n".join(out_lines)
        if content.endswith("\n"):
            result += "\n"
        return result

    def _write_runtime_compose_patch(self, patched_path: Path, content: str) -> None:
        """Write a compose file copy under /tmp with absolute build/volume paths."""
        patched_path.write_text(
            self._absolutize_compose_paths_for_runtime_copy(content),
            encoding="utf-8",
        )

    def _patch_compose_pruning_env(self, compose_file: str) -> str:
        """Create a runtime compose file that injects pruning env placeholders next to quantization vars."""
        if compose_file in self._runtime_compose_files:
            return self._runtime_compose_files[compose_file]

        src_path = Path(compose_file)
        if not src_path.exists():
            return compose_file

        try:
            original = src_path.read_text(encoding="utf-8")
        except Exception:
            return compose_file

        lines = original.splitlines()
        patched_lines: List[str] = []

        for idx, line in enumerate(lines):
            patched_lines.append(line)
            if "USE_QUANTIZATION=${USE_QUANTIZATION:-false}" in line:
                window = "\n".join(lines[idx + 1: idx + 8])
                indent = line.split("-", 1)[0]
                if "USE_PRUNING=${USE_PRUNING:-false}" not in window:
                    patched_lines.append(f"{indent}- USE_PRUNING=${{USE_PRUNING:-false}}")
                if "PRUNING_SPARSITY=${PRUNING_SPARSITY:-0.5}" not in window:
                    patched_lines.append(f"{indent}- PRUNING_SPARSITY=${{PRUNING_SPARSITY:-0.5}}")
                if "PRUNING_STRUCTURED=${PRUNING_STRUCTURED:-false}" not in window:
                    patched_lines.append(f"{indent}- PRUNING_STRUCTURED=${{PRUNING_STRUCTURED:-false}}")

        patched = "\n".join(patched_lines) + ("\n" if original.endswith("\n") else "")
        if patched == original:
            self._runtime_compose_files[compose_file] = compose_file
            return compose_file

        tmp_dir = Path(tempfile.gettempdir()) / "fl_runtime_compose"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patched_path = tmp_dir / f"{src_path.stem}.runtime-pruning{src_path.suffix}"
        self._write_runtime_compose_patch(patched_path, patched)
        self._runtime_compose_files[compose_file] = str(patched_path)
        print(f"[INFO] Using runtime patched compose file: {patched_path}")
        return str(patched_path)

    def _get_runtime_compose_file(self, compose_file: str) -> str:
        """Return compose file path used at runtime (possibly patched)."""
        patched_path = compose_file
        if self._compose_requires_runtime_patch():
            patched_path = self._patch_compose_pruning_env(patched_path)
        patched_path = self._patch_compose_stop_env(patched_path)
        patched_path = self._patch_compose_num_clients_substitution(patched_path)
        patched_path = self._patch_compose_dataset_env(patched_path)
        if not self.enable_gpu:
            patched_path = self._patch_compose_disable_gpu_reservations(patched_path)
        return patched_path

    def _patch_compose_disable_gpu_reservations(self, compose_file: str) -> str:
        """Create a runtime compose file without NVIDIA device reservations for CPU runs."""
        if compose_file in self._runtime_compose_files_cpu:
            return self._runtime_compose_files_cpu[compose_file]

        src_path = Path(compose_file)
        if not src_path.exists():
            return compose_file

        try:
            original = src_path.read_text(encoding="utf-8")
        except Exception:
            return compose_file

        lines = original.splitlines()
        patched_lines: List[str] = []
        changed = False
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            if line.strip() != "deploy:":
                patched_lines.append(line)
                i += 1
                continue

            indent = len(line) - len(line.lstrip())
            block_lines = [line]
            j = i + 1
            while j < n:
                next_line = lines[j]
                if next_line.strip() == "":
                    block_lines.append(next_line)
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= indent:
                    break
                block_lines.append(next_line)
                j += 1

            block_text = "\n".join(block_lines)
            if "driver: nvidia" in block_text or "capabilities: [gpu]" in block_text:
                changed = True
            else:
                patched_lines.extend(block_lines)
            i = j

        if not changed:
            self._runtime_compose_files_cpu[compose_file] = compose_file
            return compose_file

        patched = "\n".join(patched_lines) + ("\n" if original.endswith("\n") else "")
        tmp_dir = Path(tempfile.gettempdir()) / "fl_runtime_compose"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patched_path = tmp_dir / f"{src_path.stem}.runtime-cpu{src_path.suffix}"
        self._write_runtime_compose_patch(patched_path, patched)
        self._runtime_compose_files_cpu[compose_file] = str(patched_path)
        print(f"[INFO] Using CPU-safe compose file: {patched_path}")
        return str(patched_path)

    def _patch_compose_dataset_env(self, compose_file: str) -> str:
        """Inject DATASET_CLIENT_ID next to matching CLIENT_ID lines for per-container data shards."""
        if not self.dataset_client_map:
            return compose_file

        cache_key = (compose_file, frozenset(self.dataset_client_map.items()))
        if cache_key in self._runtime_compose_files_dataset:
            return self._runtime_compose_files_dataset[cache_key]

        src_path = Path(compose_file)
        if not src_path.exists():
            self._runtime_compose_files_dataset[cache_key] = compose_file
            return compose_file

        try:
            original = src_path.read_text(encoding="utf-8")
        except Exception:
            self._runtime_compose_files_dataset[cache_key] = compose_file
            return compose_file

        lines = original.splitlines()
        new_lines: List[str] = []
        i = 0
        n = len(lines)
        changed = False
        while i < n:
            line = lines[i]
            m = re.match(r"^(\s*)-\s*CLIENT_ID=(\d+)\s*$", line)
            if not m:
                new_lines.append(line)
                i += 1
                continue
            indent, cid = m.group(1), int(m.group(2))
            new_lines.append(line)
            i += 1
            v = self.dataset_client_map.get(cid)
            if v is None:
                continue
            changed = True
            if i < n and re.match(r"^\s*-\s*DATASET_CLIENT_ID=\s*\S+", lines[i]):
                i += 1
            new_lines.append(f"{indent}- DATASET_CLIENT_ID={v}")

        if not changed:
            self._runtime_compose_files_dataset[cache_key] = compose_file
            return compose_file

        patched = "\n".join(new_lines) + ("\n" if original.endswith("\n") else "")
        tmp_dir = Path(tempfile.gettempdir()) / "fl_runtime_compose"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patched_path = tmp_dir / f"{src_path.stem}.runtime-dataset{src_path.suffix}"
        self._write_runtime_compose_patch(patched_path, patched)
        self._runtime_compose_files_dataset[cache_key] = str(patched_path)
        print(f"[INFO] Using runtime compose (DATASET_CLIENT_ID): {patched_path}")
        return str(patched_path)

    def _patch_compose_stop_env(self, compose_file: str) -> str:
        """Create a runtime compose file that injects STOP env into fl-server/fl-client services."""
        if compose_file in self._runtime_compose_files_stop:
            return self._runtime_compose_files_stop[compose_file]

        src_path = Path(compose_file)
        if not src_path.exists():
            return compose_file

        try:
            original = src_path.read_text(encoding="utf-8")
        except Exception:
            return compose_file

        # If already present anywhere, assume it was patched already.
        if "STOP_ON_CLIENT_CONVERGENCE" in original:
            self._runtime_compose_files_stop[compose_file] = compose_file
            return compose_file

        lines = original.splitlines()
        patched_lines: List[str] = []

        current_service: Optional[str] = None
        service_line_prefix = " " * 2

        for idx, line in enumerate(lines):
            stripped = line.strip()
            patched_lines.append(line)

            # Track the current service we're inside by the service definition line.
            # Compose files in this repo use 2-space indentation for service keys under `services:`.
            if line.startswith(service_line_prefix) and stripped.endswith(":") and not stripped.startswith("#"):
                candidate = stripped[:-1]
                if candidate.startswith("fl-server") or candidate.startswith("fl-client"):
                    current_service = candidate
                else:
                    current_service = None

            # Inject into environment list blocks for fl-server/fl-client services.
            if current_service and stripped == "environment:":
                # If any of the next few lines already contain STOP, avoid duplicating.
                window = "\n".join(lines[idx + 1 : idx + 12])
                if "STOP_ON_CLIENT_CONVERGENCE" in window:
                    continue

                env_indent = line[: len(line) - len(line.lstrip(" "))]
                item_indent = env_indent + "  "
                patched_lines.append(
                    f"{item_indent}- STOP_ON_CLIENT_CONVERGENCE=${{STOP_ON_CLIENT_CONVERGENCE:-true}}"
                )

        patched = "\n".join(patched_lines) + ("\n" if original.endswith("\n") else "")
        if patched == original:
            self._runtime_compose_files_stop[compose_file] = compose_file
            return compose_file

        tmp_dir = Path(tempfile.gettempdir()) / "fl_runtime_compose"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patched_path = tmp_dir / f"{src_path.stem}.runtime-stop{src_path.suffix}"
        self._write_runtime_compose_patch(patched_path, patched)

        self._runtime_compose_files_stop[compose_file] = str(patched_path)
        print(f"[INFO] Using runtime patched compose file (STOP env): {patched_path}")
        return str(patched_path)

    def _patch_compose_num_clients_substitution(self, compose_file: str) -> str:
        """Replace hardcoded NUM_CLIENTS=N with compose substitution so host env can set federation size."""
        if compose_file in self._runtime_compose_files_num_clients:
            return self._runtime_compose_files_num_clients[compose_file]

        src_path = Path(compose_file)
        if not src_path.exists():
            return compose_file

        try:
            original = src_path.read_text(encoding="utf-8")
        except Exception:
            return compose_file

        if "NUM_CLIENTS=${NUM_CLIENTS" in original:
            self._runtime_compose_files_num_clients[compose_file] = compose_file
            return compose_file

        lines = original.splitlines()
        changed = False
        new_lines: List[str] = []
        num_re = re.compile(r"^(\s*)-\s*NUM_CLIENTS=\d+\s*$")
        for line in lines:
            m = num_re.match(line)
            if m:
                indent = m.group(1)
                new_lines.append(f"{indent}- NUM_CLIENTS=${{NUM_CLIENTS:-2}}")
                changed = True
            else:
                new_lines.append(line)

        if not changed:
            self._runtime_compose_files_num_clients[compose_file] = compose_file
            return compose_file

        patched = "\n".join(new_lines) + ("\n" if original.endswith("\n") else "")
        tmp_dir = Path(tempfile.gettempdir()) / "fl_runtime_compose"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        patched_path = tmp_dir / f"{src_path.stem}.runtime-numclients{src_path.suffix}"
        self._write_runtime_compose_patch(patched_path, patched)
        self._runtime_compose_files_num_clients[compose_file] = str(patched_path)
        print(f"[INFO] Using runtime patched compose file (NUM_CLIENTS env): {patched_path}")
        return str(patched_path)

    def start_containers(self, protocol: str, scenario: str = "excellent", congestion_level: str = "none"):
        """Start Docker containers for a specific protocol with staged startup"""
        
        # Check if rl_unified is selected - use unified compose file
        if protocol == "rl_unified":
            print("\n" + "="*70)
            print("STARTING RL-UNIFIED MODE")
            print("="*70)
            print("Using unified docker-compose file with all protocol brokers")
            print("Server will handle: MQTT, AMQP, gRPC, QUIC, HTTP/3, DDS")
            print("Clients will use RL-based Q-Learning to select protocol")
            if self.use_ql_convergence:
                print("End condition: Q-learning convergence (multiple episodes)")
            else:
                print("End condition: accuracy convergence (current behavior)")
            if self.use_ql_convergence:
                print("RL exploration: epsilon-greedy (USE_RL_EXPLORATION=true)")
            elif self.rl_inference_only:
                print("RL exploration: OFF (greedy / inference-only)")
            else:
                print("RL exploration: epsilon-greedy on multi-client FL (USE_RL_EXPLORATION=true)")
            print("="*70 + "\n")
            
            compose_file = self._get_runtime_compose_file(self.unified_compose_files[self.use_case])
            if self.use_ql_convergence:
                os.environ["USE_QL_CONVERGENCE"] = "true"
                os.environ["USE_RL_EXPLORATION"] = "true"
                # Emotion unified client: collect→boundaries→RL training (override via env if needed)
                os.environ.setdefault("RL_BOUNDARY_PIPELINE", "true")
                os.environ.setdefault("RL_PHASE0_ROUNDS", "20")
                if int(self.local_clients) <= 0:
                    # Server/brokers only: federation size from --min-clients (same as non-QL unified).
                    total_fed = self._total_expected_federation_clients()
                    os.environ["MIN_CLIENTS"] = str(total_fed)
                    os.environ["NUM_CLIENTS"] = str(total_fed)
                    os.environ["MAX_CLIENTS"] = "100"
                    print(
                        "RL training mode: no local client containers; "
                        "server uses federation MIN_CLIENTS/NUM_CLIENTS until remote clients attach."
                    )
                else:
                    # RL training: one local client; server waits for 1 client; run until Q converges and exit
                    os.environ["MIN_CLIENTS"] = "1"
                    os.environ["NUM_CLIENTS"] = "1"
                    print("RL training mode: starting only 1 client (converge and exit)")
            else:
                os.environ["USE_QL_CONVERGENCE"] = "false"
                # Multi-client FL: explore (train Q) unless explicit inference-only from GUI/CLI.
                os.environ["USE_RL_EXPLORATION"] = (
                    "false" if self.rl_inference_only else "true"
                )
            os.environ["USE_COMMUNICATION_MODEL_REWARD"] = "true" if self.use_communication_model_reward else "false"
            
            # host_macvlan: ensure fl-macvlan network exists before up
            if self.network_mode == "host_macvlan":
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from network_simulator import NetworkSimulator
                sim = NetworkSimulator(verbose=True)
                if not sim.ensure_macvlan_network():
                    raise RuntimeError("Failed to create macvlan network for host_macvlan mode")
            # When QL convergence: start only brokers + server + first client (single client for training)
            if self.use_ql_convergence:
                uc = self.use_case  # emotion, mentalstate, temperature
                if int(self.local_clients) <= 0:
                    services_ql = [
                        "mqtt-broker-unified",
                        "amqp-broker-unified",
                        f"fl-server-unified-{uc}",
                    ]
                    compose_cmd = self._docker_compose_base(compose_file) + ["up", "-d"] + services_ql
                else:
                    services_single_client = [
                        "mqtt-broker-unified",
                        "amqp-broker-unified",
                        f"fl-server-unified-{uc}",
                        f"fl-client-unified-{uc}-1",
                    ]
                    compose_cmd = self._docker_compose_base(compose_file) + ["up", "-d"] + services_single_client
            else:
                # Unified compose defines up to three client services; a bare `up -d` starts all and ignores
                # --local-clients. Start only brokers + server + the first N client services (N=0..3).
                uc = self.use_case
                max_unified_clients = 3
                num_local = max(0, min(int(self.local_clients), max_unified_clients))
                total_fed = self._total_expected_federation_clients()
                os.environ["MIN_CLIENTS"] = str(total_fed)
                os.environ["NUM_CLIENTS"] = str(total_fed)
                os.environ["MAX_CLIENTS"] = "100"
                services_multi = [
                    "mqtt-broker-unified",
                    "amqp-broker-unified",
                    f"fl-server-unified-{uc}",
                ]
                for i in range(1, num_local + 1):
                    services_multi.append(f"fl-client-unified-{uc}-{i}")
                compose_cmd = self._docker_compose_base(compose_file) + ["up", "-d"] + services_multi
            
            if self.use_ql_convergence:
                _n_loc = 0 if int(self.local_clients) <= 0 else 1
                _n_tot = self._total_expected_federation_clients()
            else:
                _n_loc = num_local
                _n_tot = self._total_expected_federation_clients()
            print(
                f"Starting unified FL system for {self.use_case}... "
                f"(local client containers: {_n_loc}; server expects total participants: {_n_tot})"
            )
            result = self.run_command(compose_cmd, check=False)
            
            if result.returncode != 0:
                print(f"[ERROR] Failed to start unified containers:")
                print(result.stderr)
                raise RuntimeError("Failed to start unified containers")
            
            print("[SUCCESS] All unified containers started")
            print("  ✓ MQTT Broker")
            print("  ✓ AMQP Broker")
            print("  ✓ Unified FL Server")
            if _n_loc > 0:
                print("  ✓ Unified FL Clients")
            else:
                print("  — Unified FL Clients: none started locally (server-only mode)")
            
            # Wait for services to initialize
            print("\nWaiting for services to initialize (15 seconds)...")
            time.sleep(15)
            
            return True
        
        # Regular protocol handling (existing code)
        compose_file = self._get_runtime_compose_file(self.compose_files[self.use_case])
        total_fed = self._total_expected_federation_clients()
        os.environ["MIN_CLIENTS"] = str(total_fed)
        os.environ["NUM_CLIENTS"] = str(total_fed)
        print(
            f"[INFO] Federation size for Docker compose (MIN_CLIENTS/NUM_CLIENTS): {total_fed} "
            f"(local client containers this host: {self.local_clients})"
        )

        # host_macvlan: ensure fl-macvlan network exists before up
        if self.network_mode == "host_macvlan":
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from network_simulator import NetworkSimulator
            sim = NetworkSimulator(verbose=True)
            if not sim.ensure_macvlan_network():
                raise RuntimeError("Failed to create macvlan network for host_macvlan mode")
        compose_cmd_base = self._docker_compose_base(compose_file)
        
        services = self.service_patterns[self.use_case][protocol]
        
        mode_label = {"gpu": "GPU (Docker bridge)", "host": "Host network (tc on host)", "host_macvlan": "Host + macvlan (per-container tc)"}.get(self.network_mode, "GPU (Docker bridge)")
        print(f"\n{'='*70}")
        print(f"Starting containers for {protocol.upper()} protocol...")
        print(f"Network Scenario: {scenario.upper()}")
        print(f"Network mode: {mode_label}")
        if self.enable_gpu:
            print(f"GPU Acceleration: ENABLED")
        if self.enable_congestion and congestion_level != "none":
            print(f"Congestion Level: {congestion_level.upper()}")
        print(f"{'='*70}")
        
        # Separate services into broker, server, and clients
        broker = None
        server = None
        clients = []
        
        for service in services:
            if 'broker' in service.lower() or 'rabbitmq' in service.lower():
                broker = service
            elif 'server' in service.lower():
                server = service
            elif 'client' in service.lower():
                clients.append(service)
        
        # Stage 1: Start broker first (if exists)
        if broker:
            print(f"\n[Stage 1/4] Starting broker: {broker}")
            cmd_broker = compose_cmd_base + ["up", "-d", broker]
            result = self.run_command(cmd_broker, check=False)
            if result.returncode != 0:
                print(f"[ERROR] Failed to start broker")
                print(f"[ERROR] stdout: {result.stdout}")
                print(f"[ERROR] stderr: {result.stderr}")
                return False
            
            print(f"Waiting 5 seconds for broker to initialize...")
            time.sleep(5)
        
        # Stage 2: Start server
        if server:
            stage_num = "[Stage 2/4]" if broker else "[Stage 1/4]"
            print(f"\n{stage_num} Starting server: {server}")
            cmd_server = compose_cmd_base + ["up", "-d", server]
            result = self.run_command(cmd_server, check=False)
            if result.returncode != 0:
                print(f"[ERROR] Failed to start server")
                print(f"[ERROR] stdout: {result.stdout}")
                print(f"[ERROR] stderr: {result.stderr}")
                return False
            
            print(f"Waiting 5 seconds for server to initialize...")
            time.sleep(5)
        
        # Stage 3: Start traffic generators (if congestion enabled)
        if self.enable_congestion and congestion_level != "none" and self.congestion_manager:
            stage_num = "[Stage 3/4]"
            print(f"\n{stage_num} Starting traffic generators (Congestion Level: {congestion_level.upper()})")
            if not self.congestion_manager.start_traffic_generators(congestion_level):
                print(f"[WARNING] Failed to start traffic generators, continuing without congestion...")
            else:
                print(f"Waiting 5 seconds for traffic to stabilize...")
                time.sleep(5)
        
        # Stage 4: Start clients
        if clients:
            # Respect configured number of local clients to start on this machine.
            num_local = max(0, min(self.local_clients, len(clients)))
            clients_to_start = clients[:num_local]
            if clients_to_start:
                stage_num = "[Stage 4/4]" if (broker and self.enable_congestion and congestion_level != "none") else "[Stage 3/4]"
                print(f"\n{stage_num} Starting clients ({num_local} local): {', '.join(clients_to_start)}")
                cmd_clients = compose_cmd_base + ["up", "-d"] + clients_to_start
                result = self.run_command(cmd_clients, check=False)
                if result.returncode != 0:
                    print(f"[ERROR] Failed to start clients")
                    print(f"[ERROR] stdout: {result.stdout}")
                    print(f"[ERROR] stderr: {result.stderr}")
                    return False

                print(f"Waiting 5 seconds for clients to connect...")
                time.sleep(5)
            else:
                print("\n[Stage 4/4] Skipping local clients (0 requested; server-only mode).")
        
        print(f"\n[OK] All containers started successfully with staged delays")
        return True
    
    def stop_containers(self, protocol: str):
        """Stop Docker containers for a specific protocol"""
        
        # Reset host tc if we applied it for host-network mode
        if getattr(self, "_host_network_sim", None) is not None:
            try:
                self._host_network_sim.reset_host_network()
            except Exception as e:
                print(f"[WARNING] Could not reset host network conditions: {e}")
            self._host_network_sim = None

        # Use unified compose file for rl_unified
        if protocol == "rl_unified":
            compose_file = self._get_runtime_compose_file(self.unified_compose_files[self.use_case])
        else:
            compose_file = self._get_runtime_compose_file(self.compose_files[self.use_case])
        services = self.service_patterns[self.use_case][protocol]
        
        print(f"\nStopping containers for {protocol.upper()} protocol...")
        
        cmd = self._docker_compose_base(compose_file) + ["down"]
        
        self.run_command(cmd, check=False)
        
        # Give time for containers to fully stop
        time.sleep(5)
    
    def apply_network_scenario(self, scenario: str, protocol: str):
        """Apply network conditions at ingress and egress: host mode = tc on host interface; gpu/host_macvlan = per-container tc."""
        import sys
        from pathlib import Path
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from network_simulator import (
            NetworkSimulator,
            draw_dynamic_base_scenario,
            reset_dynamic_base_scenario_draw,
        )

        print(f"\n{'='*70}")
        print(f"Applying network scenario: {scenario.upper()}")
        print(f"{'='*70}")
        
        sim = NetworkSimulator(verbose=True)

        # Dynamic: pick base via draw_dynamic_base_scenario() (shuffle-bag by default so
        # excellent/good/moderate are not starved in short runs). Reset bag at experiment start.
        if scenario == "dynamic":
            reset_dynamic_base_scenario_draw()
            base_scenario = draw_dynamic_base_scenario()
            print(f"[DYNAMIC] Initial dynamic base scenario selected: {base_scenario}")
        else:
            base_scenario = scenario

        try:
            _sd = Path(__file__).parent.parent / "shared_data"
            _sd.mkdir(exist_ok=True)
            _tag = (base_scenario or "default").strip().lower() or "default"
            (_sd / "current_rl_network_scenario.txt").write_text(f"scenario={_tag}\n", encoding="utf-8")
        except Exception as _e:
            print(f"[RL] Warning: could not write current_rl_network_scenario.txt ({_e})")

        if base_scenario not in sim.NETWORK_SCENARIOS:
            print(f"[WARNING] Unknown scenario: {base_scenario}, skipping network simulation")
            return True
        use_gaussian = os.environ.get("USE_GAUSSIAN_DELAY", "1").strip().lower() in ("1", "true", "yes")
        sigma_factor = float(os.environ.get("GAUSSIAN_SIGMA_FACTOR", "0.05"))
        use_extra_jitter = os.environ.get("USE_EXTRA_JITTER", "0").strip().lower() in ("1", "true", "yes")
        try:
            if use_gaussian:
                from network_simulator import build_delay_models
                models = build_delay_models(sim.NETWORK_SCENARIOS, sigma_factor=sigma_factor)
                conditions = sim.get_scenario_conditions_sampled(base_scenario, models, use_extra_jitter=use_extra_jitter)
            else:
                conditions = sim.get_scenario_conditions(base_scenario)
        except (ValueError, KeyError) as e:
            print(f"[WARNING] {e}, skipping network simulation")
            return True
        services = self.service_patterns[self.use_case][protocol]

        if self.network_mode == "host":
            # Host network: apply tc on host default interface (affects all containers)
            if sim.apply_network_conditions_host(conditions):
                print(f"[INFO] Applied scenario '{scenario}' to host interface (all containers share it).")
                self._host_network_sim = sim
            else:
                print(f"[WARNING] Could not apply host tc (sudo may be required). Experiment continues without shaping.")
                self._host_network_sim = None
            print(f"{'='*70}\n")
            return True

        # gpu or host_macvlan: per-container tc (only running client containers; RL training may start one client)
        running = self._running_docker_container_names()
        success_count = 0
        client_targets = 0
        for container in services:
            if 'broker' in container.lower() or 'rabbitmq' in container.lower() or 'server' in container.lower():
                print(f"[INFO] Skipping network conditions for broker/server: {container} (infrastructure)")
                continue
            if container not in running:
                print(f"[INFO] Skipping network conditions for {container}: container not running (e.g. single-client RL mode)")
                continue
            client_targets += 1
            try:
                if sim.apply_network_conditions(container, conditions):
                    success_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to apply conditions to {container}: {e}")
        if client_targets == 0:
            print(f"\n[INFO] No running FL client containers in compose list; network scenario not applied to clients")
            print(f"{'='*70}\n")
            return True
        print(f"\nApplied network conditions to {success_count}/{client_targets} running client container(s)")
        time.sleep(2)
        return success_count > 0
    
    def wait_for_completion(self, protocol: str, timeout: Optional[int] = 3600, scenario: Optional[str] = None):
        """Wait for FL training to complete and track round trip times.
        When timeout is None, wait indefinitely (used for rl_unified + Q-learning convergence).
        When scenario is set and USE_GAUSSIAN_DELAY=1, delay/jitter are re-sampled from normal distribution
        before each round's client->server send (per-round tc update)."""
        if timeout is None:
            print(f"\nWaiting for {protocol.upper()} training to complete (no time limit - until Q-learning converges)...")
        else:
            print(f"\nWaiting for {protocol.upper()} training to complete (timeout: {timeout}s)...")
        
        # Get server container name and client containers (for per-round tc re-apply)
        services = self.service_patterns[self.use_case][protocol]
        server_container = [s for s in services if "server" in s][0]
        use_gaussian = os.environ.get("USE_GAUSSIAN_DELAY", "1").strip().lower() in ("1", "true", "yes")
        resample_tc_per_round = scenario and use_gaussian
        last_round_tc_applied = 0  # re-apply tc before each round's send when round completes
        
        # Track round trip times (time from global model sent to next round start)
        round_trip_times = []
        last_round_complete_time = None
        current_round = 0
        
        # Common completion markers that servers print when training ends
        completion_markers = [
            "TRAINING COMPLETE",
            "Training Complete",
            "Training completed",
            "All rounds completed",
            "Converged",
            "CONVERGENCE ACHIEVED",  # temperature / mental unified servers
            "COMPLETED",  # e.g. "COMPLETED N ROUNDS"
            "Results saved to",
            "Experiment runner snapshot",  # temperature unified save_results()
            "Experiment finished",
        ]

        start_time = time.time()
        while timeout is None or (time.time() - start_time < timeout):
            elapsed = time.time() - start_time
            # Check if server container is still running
            result = self.run_command([
                "docker", "ps", "--filter", f"name={server_container}", "--format", "{{.Names}}"
            ], check=False)
            
            if server_container not in result.stdout:
                print(f"Server container stopped. Training complete!")
                return True, round_trip_times
            
            # Check logs for completion indicators and RTT tracking
            logs = self.run_command([
                "docker", "logs", "--tail", "100", server_container
            ], check=False)
            
            # Track round trip time by looking for round completion markers
            log_content = logs.stdout or ""
            
            # Look for round completion patterns
            import re
            round_patterns = [
                r'Round (\d+)/\d+ completed',
                r'\[Round (\d+)\] completed',
                r'Completed round (\d+)',
                r'Round (\d+) finished',
                r'Starting Round (\d+)/',  # QUIC format
                r'Round (\d+) - Aggregated Metrics:',  # QUIC aggregation
                r'Aggregated global model from round (\d+)'  # QUIC model distribution
            ]
            
            for pattern in round_patterns:
                matches = re.findall(pattern, log_content)
                if matches:
                    latest_round = max([int(m) for m in matches])
                    if latest_round > current_round:
                        current_time = time.time()
                        if last_round_complete_time is not None:
                            rtt = current_time - last_round_complete_time
                            round_trip_times.append(rtt)
                            print(f"  Round {latest_round} RTT: {rtt:.2f}s")
                        last_round_complete_time = current_time
                        current_round = latest_round
                        # Re-apply tc with new delay/jitter sample before next round's client->server send
                        if resample_tc_per_round and current_round > last_round_tc_applied:
                            script_dir = Path(__file__).resolve().parent
                            if str(script_dir) not in sys.path:
                                sys.path.insert(0, str(script_dir))
                            try:
                                from network_simulator import (
                                    NetworkSimulator,
                                    build_delay_models,
                                    draw_dynamic_base_scenario,
                                )
                                sim = NetworkSimulator(verbose=False)
                                sigma_factor = float(os.environ.get("GAUSSIAN_SIGMA_FACTOR", "0.05"))
                                use_extra_jitter = os.environ.get("USE_EXTRA_JITTER", "0").strip().lower() in ("1", "true", "yes")
                                models = build_delay_models(sim.NETWORK_SCENARIOS, sigma_factor=sigma_factor)

                                # Dynamic: next base from same shuffle-bag / mode as initial apply (see draw_dynamic_base_scenario).
                                if scenario == "dynamic":
                                    base_scenario = draw_dynamic_base_scenario()
                                    print(f"  [DYNAMIC] Per-round base scenario (client->server send): {base_scenario}")
                                else:
                                    base_scenario = scenario

                                conditions = sim.get_scenario_conditions_sampled(base_scenario, models, use_extra_jitter=use_extra_jitter)
                                if self.network_mode == "host" and getattr(self, "_host_network_sim", None) is not None:
                                    self._host_network_sim.apply_network_conditions_host(conditions)
                                else:
                                    for container in self._running_fl_client_containers(protocol):
                                        try:
                                            sim.apply_network_conditions(container, conditions)
                                        except Exception as e:
                                            print(f"[WARNING] Per-round tc failed for {container}: {e}")
                                print(f"  Tc (delay/jitter) re-sampled for round {current_round + 1} send: {conditions.get('latency', '')} {conditions.get('jitter', '')}")
                            except Exception as e:
                                print(f"[WARNING] Per-round tc re-apply failed: {e}")
                            last_round_tc_applied = current_round
            
            # If any known completion marker appears, treat as complete
            if any(marker in log_content for marker in completion_markers):
                print(f"Training completed successfully (marker detected)!")
                time.sleep(5)  # Give time for final results to be written
                return True, round_trip_times

            # Check if training has reached the target number of rounds
            # by examining the results file content (not just existence)
            #
            # Servers write results via get_experiment_results_dir() which resolves to:
            #   /app/results/{use_case}/{protocol}/{scenario}/  (if /app/results is mounted)
            #   /app/experiment_results/{use_case}/{protocol}/{scenario}/  (fallback)
            # NETWORK_SCENARIO may not be injected into the container → defaults to "default".
            # The legacy path /app/Server/{use_case_dir}/results/ can contain STALE files baked
            # into the Docker image from a previous run (COPY Server/ ./Server/ in Dockerfile).
            # Always check the actively-written new paths FIRST to avoid stale-file false positives.
            expected_json = f"{protocol}_training_results.json"
            use_case_lc = self.use_case.lower()  # e.g. "emotion" or "mentalstate"
            use_case_res = self._experiment_use_case_results_name()
            scenario_norm = (scenario or "default").lower()
            run_seg = os.environ.get("EXPERIMENT_RESULTS_RUN_SEGMENT", "").strip()
            new_paths = []
            for base in ("/app/results", "/app/experiment_results"):
                for scen in (scenario_norm, "default"):
                    if run_seg:
                        p = f"{base}/{run_seg}/{protocol}/{scen}/{expected_json}"
                        if p not in new_paths:
                            new_paths.append(p)
                    p = f"{base}/{use_case_res}/{protocol}/{scen}/{expected_json}"
                    if p not in new_paths:
                        new_paths.append(p)
                    if use_case_lc != use_case_res:
                        p = f"{base}/{use_case_lc}/{protocol}/{scen}/{expected_json}"
                        if p not in new_paths:
                            new_paths.append(p)
            legacy_path = f"/app/Server/{self.use_case_dir}/results/{expected_json}"
            cat_paths = new_paths + [legacy_path]
            if protocol == "rl_unified":
                cat_paths = self._rl_unified_training_results_paths_in_container(scenario) + cat_paths

            read_results = None
            matched_path = None
            for cat_path in cat_paths:
                r = self.run_command(
                    ["docker", "exec", server_container, "cat", cat_path],
                    check=False,
                )
                if r.returncode == 0 and (r.stdout or "").strip():
                    # Guard against stale results files from a previous run.
                    # This applies to ALL paths because:
                    #   - legacy path: file may be baked into the Docker image
                    #   - rl_unified paths: file persists on the bind-mounted host filesystem
                    # A file is considered stale when it reports more rounds than what the
                    # current run is supposed to do AND the elapsed time is too short for
                    # training to have actually completed.  We use a minimum elapsed-time
                    # heuristic (60 s) because convergence-based runs may legitimately stop
                    # early (total_rounds < num_rounds), so we cannot rely on an exact match.
                    try:
                        _d = json.loads(r.stdout)
                        _total = int(_d.get("total_rounds", 0) or len(_d.get("rounds") or []))
                        if _total and _total >= self.num_rounds and elapsed < 60:
                            print(f"[WARN] Stale results at {cat_path} ({_total} rounds in {elapsed:.0f}s < 60s) – ignoring")
                            continue
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                    read_results = r
                    matched_path = cat_path
                    break
            else:
                read_results = type("R", (), {"returncode": 1, "stdout": ""})()

            if read_results.returncode == 0 and read_results.stdout:
                try:
                    results_data = json.loads(read_results.stdout)
                    # Check if we have results for all expected rounds
                    if isinstance(results_data, dict):
                        rounds_completed = int(results_data.get("rounds_completed", 0) or 0)
                        if not rounds_completed:
                            rlist = results_data.get("rounds") or []
                            if isinstance(rlist, list):
                                rounds_completed = len(rlist)
                        if rounds_completed >= self.num_rounds:
                            print(f"Training completed successfully ({rounds_completed}/{self.num_rounds} rounds)!")
                            time.sleep(3)
                            return True, round_trip_times
                        else:
                            print(f"Progress: {rounds_completed}/{self.num_rounds} rounds completed...")
                    elif isinstance(results_data, list) and len(results_data) >= self.num_rounds:
                        print(f"Training completed successfully ({len(results_data)}/{self.num_rounds} rounds)!")
                        time.sleep(3)
                        return True, round_trip_times
                except json.JSONDecodeError:
                    # File exists but not valid JSON yet (still being written)
                    pass
            
            time.sleep(10)  # Check every 10 seconds
        
        if timeout is not None:
            print(f"[WARNING] Training timed out after {timeout}s")
        return False, round_trip_times
    
    def collect_results(self, protocol: str, scenario: str, round_trip_times: List[float] = None):
        """Collect and save experiment results including RTT data"""
        print(f"\nCollecting results for {protocol.upper()} - {scenario}...")
        
        # Create directory for this experiment
        exp_dir = os.path.join(self.results_dir, f"{protocol}_{scenario}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # Get server container logs
        services = self.service_patterns[self.use_case][protocol]
        server_container = [s for s in services if "server" in s][0]
        
        logs = self.run_command([
            "docker", "logs", server_container
        ], check=False)
        
        # Save logs
        with open(os.path.join(exp_dir, "server_logs.txt"), "w", encoding='utf-8', errors='replace') as f:
            f.write(logs.stdout or "")
            f.write("\n\n=== STDERR ===\n\n")
            f.write(logs.stderr or "")

        # Save broker logs (if broker exists for this protocol)
        broker_containers = [s for s in services if "broker" in s.lower() or "rabbitmq" in s.lower()]
        if broker_containers:
            print(f"Collecting logs for broker: {', '.join(broker_containers)}")
            for broker in broker_containers:
                try:
                    b_logs = self.run_command(["docker", "logs", broker], check=False)
                    out_path = os.path.join(exp_dir, "broker_logs.txt")
                    with open(out_path, "w", encoding='utf-8', errors='replace') as bf:
                        bf.write(b_logs.stdout or "")
                        bf.write("\n\n=== STDERR ===\n\n")
                        bf.write(b_logs.stderr or "")
                except Exception as e:
                    print(f"[WARNING] Failed to collect logs for {broker}: {e}")

        # Save client logs as well for debugging client-side behavior (only containers that exist this run)
        client_containers = [
            s for s in services if "client" in s and self._docker_container_exists(s)
        ]
        if client_containers:
            print(f"Collecting logs for clients: {', '.join(client_containers)}")
            for client in client_containers:
                try:
                    c_logs = self.run_command(["docker", "logs", client], check=False)
                    out_path = os.path.join(exp_dir, f"{client}_logs.txt")
                    with open(out_path, "w", encoding='utf-8', errors='replace') as cf:
                        cf.write(c_logs.stdout or "")
                        cf.write("\n\n=== STDERR ===\n\n")
                        cf.write(c_logs.stderr or "")
                except Exception as e:
                    print(f"[WARNING] Failed to collect logs for {client}: {e}")
        
        # Copy result files if they exist
        # Try to copy from container
        result_files = [
            f"{protocol}_training_results.json",
            f"{protocol}_training_results.csv",
            "fl_results.json"
        ]
        if getattr(self, "use_pruning", False):
            result_files.append("pruning_metrics.json")

        use_case_lc = self.use_case.lower()
        use_case_res = self._experiment_use_case_results_name()
        scenario_norm = (scenario or "default").lower()
        run_seg = os.environ.get("EXPERIMENT_RESULTS_RUN_SEGMENT", "").strip()

        for result_file in result_files:
            dest = os.path.join(exp_dir, result_file)
            copied = False
            # Try new paths first (where servers actually write via get_experiment_results_dir),
            # then fall back to the legacy path. This avoids collecting stale baked-in files.
            for base in ("/app/results", "/app/experiment_results"):
                for scen in (scenario_norm, "default"):
                    srcs = []
                    if run_seg:
                        srcs.append(f"{server_container}:{base}/{run_seg}/{protocol}/{scen}/{result_file}")
                    srcs.append(f"{server_container}:{base}/{use_case_res}/{protocol}/{scen}/{result_file}")
                    if use_case_lc != use_case_res:
                        srcs.append(f"{server_container}:{base}/{use_case_lc}/{protocol}/{scen}/{result_file}")
                    for src in srcs:
                        r = self.run_command(["docker", "cp", src, dest], check=False)
                        if r.returncode == 0 and os.path.isfile(dest) and os.path.getsize(dest) > 0:
                            copied = True
                            break
                    if copied:
                        break
                if copied:
                    break
            if not copied:
                # Legacy fallback
                try:
                    self.run_command([
                        "docker", "cp",
                        f"{server_container}:/app/Server/{self.use_case_dir}/results/{result_file}",
                        dest
                    ], check=False)
                except Exception:
                    pass

        if protocol == "rl_unified":
            dst = os.path.join(exp_dir, "rl_unified_training_results.json")
            for cp in self._rl_unified_training_results_paths_in_container(scenario):
                r = self.run_command(
                    ["docker", "cp", f"{server_container}:{cp}", dst],
                    check=False,
                )
                if r.returncode == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0:
                    print(f"  Copied RL-unified training results from container:{cp}")
                    break

        # Save experiment metadata
        metadata = {
            "protocol": protocol,
            "scenario": scenario,
            "use_case": self.use_case,
            "num_rounds": self.num_rounds,
            "timestamp": datetime.now().isoformat(),
            "baseline_mode": self.baseline_mode,
            "network_conditions_applied": not self.baseline_mode
        }
        
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save RTT data if available
        if round_trip_times and len(round_trip_times) > 0:
            avg_rtt = sum(round_trip_times) / len(round_trip_times)
            rtt_data = {
                "protocol": protocol,
                "scenario": scenario,
                "use_case": self.use_case,
                "num_rounds": self.num_rounds,
                "rtt_per_round": round_trip_times,
                "avg_rtt_per_round": avg_rtt,
                "min_rtt": min(round_trip_times),
                "max_rtt": max(round_trip_times),
                "total_rtt": sum(round_trip_times),
                "timestamp": datetime.now().isoformat(),
                "baseline_mode": self.baseline_mode
            }
            
            rtt_filename = f"{protocol}_baseline_rtt.json" if self.baseline_mode else f"{protocol}_rtt.json"
            with open(os.path.join(exp_dir, rtt_filename), "w") as f:
                json.dump(rtt_data, f, indent=2)
            
            print(f"  RTT Stats: Avg={avg_rtt:.2f}s, Min={min(round_trip_times):.2f}s, Max={max(round_trip_times):.2f}s")
        
        # If pruning was enabled, plot model memory vs round (only when pruning data is present)
        if getattr(self, "use_pruning", False):
            try:
                script_dir = Path(__file__).resolve().parent
                if str(script_dir) not in sys.path:
                    sys.path.insert(0, str(script_dir))
                from pruning_memory_plot import plot_pruning_memory_from_experiment
                exp_dir_path = Path(exp_dir)
                plot_path = plot_pruning_memory_from_experiment(
                    exp_dir_path,
                    output_filename="pruning_memory_by_round.png",
                    server_log_name="server_logs.txt",
                )
                if plot_path:
                    print(f"  Pruning memory plot saved: {plot_path}")
            except Exception as e:
                print(f"  [WARNING] Could not generate pruning memory plot: {e}")

        # Finalize plots/artifacts for this experiment folder
        try:
            self._finalize_experiment_artifacts(
                server_container=server_container,
                protocol=protocol,
                exp_dir=exp_dir,
                scenario=scenario,
            )
        except Exception as e:
            print(f"  [WARNING] Could not finalize experiment artifacts: {e}")
        
        print(f"Results saved to: {exp_dir}")

    def _read_training_results(self, exp_dir: str, protocol: str) -> Optional[Dict]:
        """Load training results JSON if available."""
        candidates = [
            os.path.join(exp_dir, f"{protocol}_training_results.json"),
            os.path.join(exp_dir, "fl_results.json"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
        return None

    def _parse_training_results_from_server_logs(self, exp_dir: str) -> Optional[Dict]:
        """Parse rounds/loss/accuracy/convergence from server_logs.txt.

        Used as fallback when copied `*_training_results.json` is missing/stale.
        """
        server_log_path = os.path.join(exp_dir, "server_logs.txt")
        if not os.path.exists(server_log_path):
            return None

        try:
            with open(server_log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return None

        # Parse multiple protocol-specific summary formats
        rounds = []
        loss = []
        accuracy = []
        mse = []
        mae = []
        mape = []
        metrics_by_round: Dict[int, Dict[str, float]] = {}

        # Pattern 1: Round X - Aggregated Metrics:
        # Pattern 2: Round X Aggregated Metrics:
        # followed by two lines: Loss / Accuracy
        aggregated_patterns = [
            re.compile(
                r"Round\s+(\d+)\s*-\s*Aggregated\s+Metrics:\s*"
                r"(?:\n|\r\n)+\s*Loss:\s*([\d.]+)\s*"
                r"(?:\n|\r\n)+\s*Accuracy:\s*([\d.]+)",
                re.IGNORECASE,
            ),
            re.compile(
                r"Round\s+(\d+)\s+Aggregated\s+Metrics:\s*"
                r"(?:\n|\r\n)+\s*Loss:\s*([\d.]+)\s*"
                r"(?:\n|\r\n)+\s*Accuracy:\s*([\d.]+)",
                re.IGNORECASE,
            ),
        ]

        for pattern in aggregated_patterns:
            for round_str, loss_str, acc_str in pattern.findall(content):
                round_num = int(round_str)
                metrics_by_round[round_num] = {
                    "loss": float(loss_str),
                    "accuracy": float(acc_str),
                }

        # Pattern 3 (gRPC):
        # ROUND X SUMMARY ... Average Loss: Y ... Average Accuracy: Z
        grpc_summary_pattern = re.compile(
            r"ROUND\s+(\d+)\s+SUMMARY\s*"
            r"(?:.|\n|\r\n)*?Average\s+Loss:\s*([\d.]+)\s*"
            r"(?:.|\n|\r\n)*?Average\s+Accuracy:\s*([\d.]+)",
            re.IGNORECASE,
        )
        for round_str, loss_str, acc_str in grpc_summary_pattern.findall(content):
            round_num = int(round_str)
            metrics_by_round[round_num] = {
                "loss": float(loss_str),
                "accuracy": float(acc_str),
            }

        # Pattern 4 (unified / RL-unified emotion server):
        # Round X Results:
        #   Avg Loss: ...
        #   Avg Accuracy: ...
        unified_round_pattern = re.compile(
            r"Round\s+(\d+)\s+Results:\s*"
            r"(?:\n|\r\n)+\s*Avg\s+Loss:\s*([\d.]+)\s*"
            r"(?:\n|\r\n)+\s*Avg\s+Accuracy:\s*([\d.]+)",
            re.IGNORECASE,
        )
        for round_str, loss_str, acc_str in unified_round_pattern.findall(content):
            round_num = int(round_str)
            metrics_by_round[round_num] = {
                "loss": float(loss_str),
                "accuracy": float(acc_str),
            }

        # Pattern 5 (temperature servers):
        # Round X - Aggregated Metrics:
        #   Loss: ...
        #   MSE:  ...
        #   MAE:  ...
        #   MAPE: ...
        temperature_round_pattern = re.compile(
            r"Round\s+(\d+)\s*-?\s*Aggregated\s+Metrics:\s*"
            r"(?:\n|\r\n)+\s*Loss:\s*([\d.]+)\s*"
            r"(?:\n|\r\n)+\s*MSE:\s*([\d.]+)\s*"
            r"(?:\n|\r\n)+\s*MAE:\s*([\d.]+)\s*"
            r"(?:\n|\r\n)+\s*MAPE:\s*([\d.]+)",
            re.IGNORECASE,
        )
        for round_str, loss_str, mse_str, mae_str, mape_str in temperature_round_pattern.findall(content):
            round_num = int(round_str)
            mse_val = float(mse_str)
            metrics_by_round[round_num] = {
                "loss": float(loss_str),
                "mse": mse_val,
                "mae": float(mae_str),
                "mape": float(mape_str),
                # Match temperature server JSON behavior: use a bounded proxy accuracy.
                "accuracy": max(0.0, 1.0 - mse_val),
            }

        # Pattern 6 (temperature RL-unified server):
        # ROUND X/200
        # Avg Accuracy (proxy): ...
        # Avg Loss: ...
        unified_temperature_round_pattern = re.compile(
            r"ROUND\s+(\d+)\s*/\s*\d+"
            r"(?:.|\n|\r\n)*?Avg\s+Accuracy\s*\(proxy\)\s*:\s*([\d.]+)\s*"
            r"(?:.|\n|\r\n)*?Avg\s+Loss\s*:\s*([\d.]+)",
            re.IGNORECASE,
        )
        for round_str, acc_str, loss_str in unified_temperature_round_pattern.findall(content):
            round_num = int(round_str)
            metrics_by_round[round_num] = {
                "loss": float(loss_str),
                "accuracy": float(acc_str),
            }

        if not metrics_by_round:
            return None

        for round_num in sorted(metrics_by_round.keys()):
            round_metrics = metrics_by_round[round_num]
            loss_val = float(round_metrics.get("loss", 0.0))
            acc_val = float(round_metrics.get("accuracy", 0.0))
            rounds.append(round_num)
            loss.append(loss_val)
            accuracy.append(acc_val / 100.0 if acc_val > 1.0 else acc_val)
            if "mse" in round_metrics:
                mse.append(float(round_metrics["mse"]))
            if "mae" in round_metrics:
                mae.append(float(round_metrics["mae"]))
            if "mape" in round_metrics:
                mape.append(float(round_metrics["mape"]))

        if not rounds:
            return None

        convergence_time = None
        conv_patterns = [
            re.compile(r"Convergence\s*time\s*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
            re.compile(r"Total\s+Training\s+Time\s*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
            re.compile(r"Total\s+time\s*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
            re.compile(r"Time\s+to\s+Convergence\s*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
        ]
        for pattern in conv_patterns:
            m = pattern.search(content)
            if m:
                convergence_time = float(m.group(1))
                break

        if convergence_time is None:
            # Estimate from recorded RTTs if available
            try:
                for name in os.listdir(exp_dir):
                    if name.endswith("_rtt.json"):
                        with open(os.path.join(exp_dir, name), "r", encoding="utf-8") as f:
                            rtt_data = json.load(f)
                        if isinstance(rtt_data, dict) and rtt_data.get("total_rtt"):
                            convergence_time = float(rtt_data.get("total_rtt"))
                            break
            except Exception:
                pass

        results = {
            "rounds": rounds,
            "loss": loss,
            "accuracy": accuracy,
            "convergence_time_seconds": convergence_time,
            "convergence_time_minutes": (convergence_time / 60.0) if convergence_time is not None else None,
            "total_rounds": len(rounds),
            "final_accuracy": accuracy[-1] if accuracy else None,
            "final_loss": loss[-1] if loss else None,
        }
        if len(mse) == len(rounds):
            results["mse"] = mse
        if len(mae) == len(rounds):
            results["mae"] = mae
        if len(mape) == len(rounds):
            results["mape"] = mape
        return results

    @staticmethod
    def _normalize_training_results(data: Dict) -> Dict:
        """Ensure convergence_time_seconds is always available at the top level.

        Some protocols (e.g. AMQP, DDS) nest it inside a ``summary`` sub-dict when
        fixed-round mode is used instead of convergence-based early stopping.  The
        plot generator and cross-run comparison scripts only read the top-level key,
        so promote it here once and fall back to summing round_times_seconds when
        both locations are absent.
        """
        if data.get("convergence_time_seconds") is None:
            summary = data.get("summary") or {}
            ct = summary.get("convergence_time_seconds")
            if ct is None:
                # Last resort: sum the per-round wall times recorded by the server
                rts = data.get("round_times_seconds") or []
                ct = sum(float(t) for t in rts) if rts else None
            if ct is not None:
                data = dict(data)
                data["convergence_time_seconds"] = float(ct)
                data["convergence_time_minutes"] = float(ct) / 60.0
                # Promote other useful summary fields when missing at top level
                for key in ("total_rounds", "num_clients", "final_accuracy", "final_loss", "converged"):
                    if data.get(key) is None and summary.get(key) is not None:
                        data[key] = summary[key]
        return data

    def _resolve_training_results(self, exp_dir: str, protocol: str) -> Optional[Dict]:
        """Resolve best available training results, preferring data consistent with server logs."""
        json_results = self._read_training_results(exp_dir, protocol)
        log_results = self._parse_training_results_from_server_logs(exp_dir)

        if json_results is None and log_results is None:
            return None
        if json_results is None:
            return self._normalize_training_results(log_results)
        if log_results is None:
            return self._normalize_training_results(json_results)

        # If JSON rounds are stale/incomplete, trust server log parse
        json_rounds = json_results.get("rounds", []) or []
        log_rounds = log_results.get("rounds", []) or []
        if len(log_rounds) > len(json_rounds):
            merged = dict(log_results)
            # preserve richer fields from json when present
            if json_results.get("num_clients") is not None:
                merged["num_clients"] = json_results.get("num_clients")
            # merge battery-related series if JSON has them
            for key in (
                "battery_consumption",
                "battery_model_consumption",
                "battery_model_consumption_source",
                "battery_soc",
                "avg_battery_soc",
                "round_times_seconds",
            ):
                if json_results.get(key):
                    merged[key] = json_results.get(key)
            return self._normalize_training_results(merged)

        return self._normalize_training_results(json_results)

    def _persist_training_results(self, exp_dir: str, protocol: str, training: Optional[Dict]):
        """Persist corrected training results to protocol JSON in experiment folder."""
        if not training:
            return
        path = os.path.join(exp_dir, f"{protocol}_training_results.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(training, f, indent=2)
        except Exception:
            pass

    def _extract_model_size_series_mb(self, exp_dir: str) -> List[float]:
        """Extract per-round model serialized size (MB) from client logs."""
        series_by_client = []
        size_pattern_mb = re.compile(r"serialized\s+update\s+size\s*:\s*([\d.]+)\s*MB", re.IGNORECASE)
        size_pattern_bytes = re.compile(
            r"chunking\s+.*?model\s+update\s*:\s*(\d+)\s+bytes\s+total",
            re.IGNORECASE,
        )
        size_pattern_bytes_alt = re.compile(
            r"model\s+update.*?\((\d+)\s+bytes(?:\s+total)?\)",
            re.IGNORECASE,
        )
        size_pattern_bytes_sent = re.compile(
            r"sent\s+update\s+in\s+\d+\s+chunks\s*\((\d+)\s+bytes(?:\s+total)?\)",
            re.IGNORECASE,
        )

        for name in os.listdir(exp_dir):
            if not name.endswith("_logs.txt") or "client" not in name.lower():
                continue
            path = os.path.join(exp_dir, name)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                sizes = [float(m) for m in size_pattern_mb.findall(content)]
                if not sizes:
                    sizes = [int(m) / (1024.0 * 1024.0) for m in size_pattern_bytes.findall(content)]
                if not sizes:
                    sizes = [int(m) / (1024.0 * 1024.0) for m in size_pattern_bytes_alt.findall(content)]
                if not sizes:
                    sizes = [int(m) / (1024.0 * 1024.0) for m in size_pattern_bytes_sent.findall(content)]
                if sizes:
                    series_by_client.append(sizes)
            except Exception:
                continue

        if not series_by_client:
            return []

        max_len = max(len(s) for s in series_by_client)
        avg_series = []
        for idx in range(max_len):
            vals = [s[idx] for s in series_by_client if idx < len(s)]
            if vals:
                avg_series.append(sum(vals) / len(vals))
        return avg_series

    def _estimate_battery_series_from_logs(self, exp_dir: str, protocol: str = "") -> List[float]:
        """Estimate cumulative battery consumption per round from saved client logs.

        Fallback for historical experiments where the server generated a battery plot
        but the timestamped experiment folder did not receive either the plot or the
        `battery_consumption` series in copied JSON.

        Applies the same PROTOCOL_ENERGY_ALPHA / PROTOCOL_CPU_BETA multipliers used by
        BatteryModel so that the fallback estimate is consistent with _estimate_battery_model_series_from_logs
        and the live server-computed battery_consumption values.
        """
        # Shared battery model constants used by clients
        k_tx = 1e-8
        k_rx = 1e-8
        E_fixed = 0.1
        P_CPU_MAX = 10.0
        BATTERY_CAP_J = 60.0 * 3600.0
        cpu_util_fraction = 0.5  # conservative fallback when exact runtime CPU is unavailable

        p = (protocol or "").strip().lower()
        # For rl_unified/unified the protocol changes per round; use neutral baseline (alpha=beta=1.0)
        # to avoid biasing the estimate, consistent with _estimate_battery_model_series_from_logs.
        if p in ("rl_unified", "unified"):
            alpha = 1.0
            beta = 1.0
        else:
            alpha = float(PROTOCOL_ENERGY_ALPHA.get(p, 1.0))
            beta = float(PROTOCOL_CPU_BETA.get(p, 1.0))

        round_start_pattern = re.compile(r"starting training for round\s+(\d+)", re.IGNORECASE)
        send_size_pattern = re.compile(
            r"chunking\s+.*?model\s+update:\s*(\d+)\s+bytes\s+total",
            re.IGNORECASE,
        )
        send_size_pattern_alt = re.compile(
            r"model\s+update.*?\((\d+)\s+bytes(?:\s+total)?\)",
            re.IGNORECASE,
        )
        send_size_pattern_sent = re.compile(
            r"sent\s+update\s+in\s+\d+\s+chunks\s*\((\d+)\s+bytes(?:\s+total)?\)",
            re.IGNORECASE,
        )
        epoch_time_pattern = re.compile(r"-\s*([\d.]+)(ms|s)/epoch", re.IGNORECASE)

        per_client_series = []
        for name in os.listdir(exp_dir):
            lower = name.lower()
            is_named_client_logs = name.endswith("_logs.txt") and "client" in lower
            is_native_style = lower.startswith("client_") and lower.endswith(".log")
            if not (is_named_client_logs or is_native_style):
                continue

            path = os.path.join(exp_dir, name)
            current_round = None
            training_times = {}
            bytes_sent = {}
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        m_round = round_start_pattern.search(line)
                        if m_round:
                            current_round = int(m_round.group(1))
                            training_times.setdefault(current_round, 0.0)
                            continue

                        if current_round is not None:
                            m_epoch = epoch_time_pattern.search(line)
                            if m_epoch:
                                val = float(m_epoch.group(1))
                                unit = m_epoch.group(2).lower()
                                training_times[current_round] = training_times.get(current_round, 0.0) + (val / 1000.0 if unit == "ms" else val)

                            m_send = send_size_pattern.search(line)
                            if not m_send:
                                m_send = send_size_pattern_alt.search(line)
                            if not m_send:
                                m_send = send_size_pattern_sent.search(line)
                            if m_send:
                                bytes_sent[current_round] = int(m_send.group(1))

                rounds = sorted(set(training_times.keys()) & set(bytes_sent.keys()))
                if not rounds:
                    continue

                cumulative = 0.0
                series = []
                for rnd in rounds:
                    bits_tx = bytes_sent[rnd] * 8
                    # bits_rx unknown from logs; use 0 as approximation (same as _estimate_battery_model_series_from_logs)
                    bits_rx = 0
                    e_radio_baseline = k_tx * bits_tx + k_rx * bits_rx + E_fixed
                    e_radio = alpha * e_radio_baseline
                    e_cpu = P_CPU_MAX * cpu_util_fraction * training_times.get(rnd, 0.0) * beta
                    e_total = e_radio + e_cpu
                    cumulative += e_total / BATTERY_CAP_J
                    series.append(cumulative)

                if series:
                    per_client_series.append(series)
            except Exception:
                continue

        if not per_client_series:
            return []

        max_len = max(len(s) for s in per_client_series)
        avg_series = []
        for idx in range(max_len):
            vals = [s[idx] for s in per_client_series if idx < len(s)]
            if vals:
                avg_series.append(sum(vals) / len(vals))
        return avg_series

    def _battery_model_series_from_jsonl(self, exp_dir: str) -> List[float]:
        """Cumulative drain fraction per round from client_fl_metrics *.jsonl (BatteryModel on clients).

        Only reads JSONL files whose filename matches the use case inferred from *exp_dir* so that
        emotion / mental_state / temperature experiments never cross-contaminate each other's
        battery drain series (all three write to the same shared_data directory).
        """
        # Infer use case from experiment directory path so we only read the matching JSONL files.
        path_lower = exp_dir.lower().replace(os.sep, "/")
        if "mentalstate" in path_lower or "mental_state" in path_lower:
            use_case_filter = "mental_state"
        elif "temperature" in path_lower:
            use_case_filter = "temperature"
        elif "emotion" in path_lower:
            use_case_filter = "emotion"
        else:
            use_case_filter = None  # unknown use case — accept all files

        roots = [
            exp_dir,
            os.path.join(os.path.dirname(exp_dir), "shared_data"),
            str(_REPO_ROOT / "shared_data"),
        ]
        files: List[str] = []
        for r in roots:
            if r and os.path.isdir(r):
                files.extend(glob.glob(os.path.join(r, "client_fl_metrics_*_client*.jsonl")))
        if not files:
            return []

        # Filter to only JSONL files that belong to the current experiment's use case.
        if use_case_filter:
            files = [
                fp for fp in files
                if os.path.basename(fp).startswith(f"client_fl_metrics_{use_case_filter}_")
            ]
        if not files:
            return []

        by_round: Dict[int, List[float]] = {}
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        rnd = int(rec.get("round") or 0)
                        if rnd <= 0:
                            continue
                        cum = rec.get("cumulative_battery_energy_joules")
                        if cum is None:
                            continue
                        by_round.setdefault(rnd, []).append(float(cum) / BATTERY_CAP_J)
            except Exception:
                continue
        if not by_round:
            return []
        max_r = max(by_round.keys())
        out: List[float] = []
        for r in range(1, max_r + 1):
            vals = by_round.get(r)
            if vals:
                out.append(sum(vals) / len(vals))
        return out

    def _cpu_memory_series_from_jsonl(self, exp_dir: str) -> Dict[str, List[float]]:
        """Read per-round average CPU% and memory% from client_fl_metrics JSONL files.

        Returns a dict with keys ``cpu_percent`` and ``memory_percent``, each a list indexed
        by round (1-based, same convention as battery / accuracy series).  Returns empty lists
        when no usable data is found.
        """
        path_lower = exp_dir.lower().replace(os.sep, "/")
        if "mentalstate" in path_lower or "mental_state" in path_lower:
            use_case_filter = "mental_state"
        elif "temperature" in path_lower:
            use_case_filter = "temperature"
        elif "emotion" in path_lower:
            use_case_filter = "emotion"
        else:
            use_case_filter = None

        roots = [
            exp_dir,
            os.path.join(os.path.dirname(exp_dir), "shared_data"),
            str(_REPO_ROOT / "shared_data"),
        ]
        files: List[str] = []
        for r in roots:
            if r and os.path.isdir(r):
                files.extend(glob.glob(os.path.join(r, "client_fl_metrics_*_client*.jsonl")))
        if use_case_filter:
            files = [
                fp for fp in files
                if os.path.basename(fp).startswith(f"client_fl_metrics_{use_case_filter}_")
            ]
        if not files:
            return {"cpu_percent": [], "memory_percent": []}

        cpu_by_round: Dict[int, List[float]] = {}
        mem_by_round: Dict[int, List[float]] = {}
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        rnd = int(rec.get("round") or 0)
                        if rnd <= 0:
                            continue
                        cpu = rec.get("cpu_percent")
                        mem = rec.get("memory_percent")
                        if cpu is not None:
                            try:
                                cpu_by_round.setdefault(rnd, []).append(float(cpu))
                            except (TypeError, ValueError):
                                pass
                        if mem is not None:
                            try:
                                mem_by_round.setdefault(rnd, []).append(float(mem))
                            except (TypeError, ValueError):
                                pass
            except Exception:
                continue

        def _avg_series(by_round: Dict[int, List[float]]) -> List[float]:
            if not by_round:
                return []
            max_r = max(by_round.keys())
            out: List[float] = []
            for r in range(1, max_r + 1):
                vals = by_round.get(r)
                if vals:
                    out.append(sum(vals) / len(vals))
            return out

        return {
            "cpu_percent": _avg_series(cpu_by_round),
            "memory_percent": _avg_series(mem_by_round),
        }

    def _estimate_battery_model_series_from_logs(self, exp_dir: str, protocol: str) -> List[float]:
        """Log-based fallback for battery_model_consumption, used when JSONL data is unavailable.

        Applies PROTOCOL_ENERGY_ALPHA / PROTOCOL_CPU_BETA — consistent with _estimate_battery_series_from_logs."""
        p = (protocol or "mqtt").strip().lower()
        # For rl_unified/unified the protocol is dynamically selected each round, so we use
        # the neutral baseline multipliers (alpha=1.0, beta=1.0) to avoid biasing the estimate
        # toward any single protocol's overhead coefficients.
        if p in ("rl_unified", "unified"):
            alpha = 1.0
            beta = 1.0
        else:
            alpha = float(PROTOCOL_ENERGY_ALPHA.get(p, 1.0))
            beta = float(PROTOCOL_CPU_BETA.get(p, 1.0))

        k_tx_l = k_tx
        k_rx_l = k_rx
        E_fixed_l = E_fixed
        P_CPU_MAX_l = P_CPU_MAX
        BATTERY_CAP_J_l = BATTERY_CAP_J
        cpu_util_fraction = 0.5

        round_start_pattern = re.compile(r"starting training for round\s+(\d+)", re.IGNORECASE)
        send_size_pattern = re.compile(
            r"chunking\s+.*?model\s+update:\s*(\d+)\s+bytes\s+total",
            re.IGNORECASE,
        )
        send_size_pattern_alt = re.compile(
            r"model\s+update.*?\((\d+)\s+bytes(?:\s+total)?\)",
            re.IGNORECASE,
        )
        send_size_pattern_sent = re.compile(
            r"sent\s+update\s+in\s+\d+\s+chunks\s*\((\d+)\s+bytes(?:\s+total)?\)",
            re.IGNORECASE,
        )
        epoch_time_pattern = re.compile(r"-\s*([\d.]+)(ms|s)/epoch", re.IGNORECASE)

        per_client_series: List[List[float]] = []
        for name in os.listdir(exp_dir):
            lower = name.lower()
            is_named_client_logs = name.endswith("_logs.txt") and "client" in lower
            is_native_style = lower.startswith("client_") and lower.endswith(".log")
            if not (is_named_client_logs or is_native_style):
                continue

            path = os.path.join(exp_dir, name)
            current_round = None
            training_times: Dict[int, float] = {}
            bytes_sent: Dict[int, int] = {}
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        m_round = round_start_pattern.search(line)
                        if m_round:
                            current_round = int(m_round.group(1))
                            training_times.setdefault(current_round, 0.0)
                            continue

                        if current_round is not None:
                            m_epoch = epoch_time_pattern.search(line)
                            if m_epoch:
                                val = float(m_epoch.group(1))
                                unit = m_epoch.group(2).lower()
                                training_times[current_round] = training_times.get(current_round, 0.0) + (
                                    val / 1000.0 if unit == "ms" else val
                                )

                            m_send = send_size_pattern.search(line)
                            if not m_send:
                                m_send = send_size_pattern_alt.search(line)
                            if not m_send:
                                m_send = send_size_pattern_sent.search(line)
                            if m_send:
                                bytes_sent[current_round] = int(m_send.group(1))

                rounds = sorted(set(training_times.keys()) & set(bytes_sent.keys()))
                if not rounds:
                    continue

                cumulative = 0.0
                series: List[float] = []
                for rnd in rounds:
                    bits_tx = bytes_sent[rnd] * 8
                    bits_rx = 0
                    e_radio_baseline = k_tx_l * bits_tx + k_rx_l * bits_rx + E_fixed_l
                    e_radio = alpha * e_radio_baseline
                    e_cpu = P_CPU_MAX_l * cpu_util_fraction * training_times.get(rnd, 0.0) * beta
                    e_total = e_radio + e_cpu
                    cumulative += e_total / BATTERY_CAP_J_l
                    series.append(cumulative)

                if series:
                    per_client_series.append(series)
            except Exception:
                continue

        if not per_client_series:
            return []

        max_len = max(len(s) for s in per_client_series)
        avg_series: List[float] = []
        for idx in range(max_len):
            vals = [s[idx] for s in per_client_series if idx < len(s)]
            if vals:
                avg_series.append(sum(vals) / len(vals))
        return avg_series

    def _copy_server_plots_to_experiment_folder(
        self, server_container: str, protocol: str, exp_dir: str, scenario: Optional[str] = None
    ):
        """Copy server-side generated plots into this experiment folder when available."""
        protocol_alias = "unified" if protocol == "rl_unified" else protocol
        uc_fs = self._experiment_use_case_results_name()
        use_case_aliases = list({self.use_case.lower(), uc_fs})
        scen = (scenario or os.getenv("NETWORK_SCENARIO") or "default").strip() or "default"

        candidate_dirs = []
        for uc in use_case_aliases:
            candidate_dirs.extend([
                f"/app/results/{uc}/{protocol_alias}/{scen}",
                f"/app/results/{uc}/{protocol_alias}/default",
                f"/app/results/{uc}/{protocol_alias}",
                f"/app/experiment_results/{uc}/{protocol_alias}/{scen}",
                f"/app/experiment_results/{uc}/{protocol_alias}/default",
                f"/app/experiment_results/{uc}/{protocol_alias}",
            ])

        cmd_parts = [f'if [ -d "{d}" ]; then find "{d}" -maxdepth 1 -type f \\( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \\); fi' for d in candidate_dirs]
        list_cmd = " ; ".join(cmd_parts)
        found = self.run_command([
            "docker", "exec", server_container, "sh", "-lc", list_cmd
        ], check=False)

        if found.returncode != 0:
            return

        copied_any = False
        for line in (found.stdout or "").splitlines():
            src = line.strip()
            if not src:
                continue
            dst = os.path.join(exp_dir, os.path.basename(src))
            self.run_command(["docker", "cp", f"{server_container}:{src}", dst], check=False)
            copied_any = True

        if copied_any:
            print("  Copied server-generated plots into experiment folder")

    def _generate_standard_plots(self, protocol: str, exp_dir: str, training: Optional[Dict]):
        """Generate standard plots in each experiment folder."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"  [WARNING] matplotlib unavailable for local plot generation: {e}")
            return

        if not training:
            return

        rounds = training.get("rounds", []) or []
        loss = training.get("loss", []) or []
        accuracy = training.get("accuracy", []) or []
        total_rounds = int(training.get("total_rounds", len(rounds) if rounds else 0) or 0)
        convergence_time_sec = float(
            training.get("convergence_time_seconds", training.get("convergence_time", 0.0)) or 0.0
        )

        if rounds and loss and accuracy and len(rounds) == len(loss) == len(accuracy):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
            ax1.plot(rounds, loss, marker="o", linewidth=2)
            ax1.set_title("Loss per Round")
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Loss")
            ax1.grid(True, alpha=0.3)

            acc_plot = [a * 100.0 if a <= 1.0 else a for a in accuracy]
            ax2.plot(rounds, acc_plot, marker="s", linewidth=2)
            ax2.set_title("Accuracy per Round")
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Accuracy (%)")
            ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, "accuracy_loss_per_round.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

        # Training rounds + convergence time summary plot
        if total_rounds > 0 or convergence_time_sec > 0:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            labels = ["Training Rounds", "Convergence Time (s)"]
            values = [max(total_rounds, 0), max(convergence_time_sec, 0.0)]
            bars = ax.bar(labels, values)
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2.0, h, f"{h:.2f}", ha="center", va="bottom")
            ax.set_title("FL Completion Summary")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, "training_rounds_and_convergence_time.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

        # Model size (memory) per round from client logs
        model_sizes_mb = self._extract_model_size_series_mb(exp_dir)
        if model_sizes_mb:
            rounds_model = list(range(1, len(model_sizes_mb) + 1))
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(rounds_model, model_sizes_mb, marker="o", linewidth=2)
            ax.set_title("Model Size per Round")
            ax.set_xlabel("Round")
            ax.set_ylabel("Serialized Model Size (MB)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, "model_size_per_round.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

        # Optional battery plot if battery data exists in training JSON
        battery_series = training.get("battery_consumption", []) or training.get("battery_soc", []) or training.get("avg_battery_soc", [])
        battery_title = "Battery Consumption / SoC per Round"
        if not battery_series:
            battery_series = self._estimate_battery_series_from_logs(exp_dir, protocol=protocol)
            if battery_series:
                battery_title = "Estimated Battery Consumption per Round"
        if rounds and battery_series and len(battery_series) != len(rounds):
            if len(battery_series) > len(rounds):
                battery_series = battery_series[:len(rounds)]
            else:
                battery_series = list(battery_series) + [battery_series[-1]] * (len(rounds) - len(battery_series))
        if rounds and battery_series and len(rounds) == len(battery_series):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            bat_vals = [b * 100.0 if b <= 1.0 else b for b in battery_series]
            ax.plot(rounds, bat_vals, marker="o", linewidth=2)
            ax.set_title(battery_title)
            ax.set_xlabel("Round")
            ax.set_ylabel("Battery (%)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, "battery_consumption_per_round.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

        # CPU utilisation and memory consumption per round from client JSONL
        cpu_mem = self._cpu_memory_series_from_jsonl(exp_dir)
        cpu_series = (
            training.get("avg_cpu_percent", []) or cpu_mem.get("cpu_percent", [])
        )
        mem_series = (
            training.get("avg_memory_percent", []) or cpu_mem.get("memory_percent", [])
        )

        def _align(series: List[float], ref: List) -> List[float]:
            if not series or not ref:
                return list(series)
            n = len(ref)
            if len(series) > n:
                return series[:n]
            if len(series) < n:
                return list(series) + [series[-1]] * (n - len(series))
            return list(series)

        cpu_series = _align(cpu_series, rounds)
        mem_series = _align(mem_series, rounds)

        if rounds and (cpu_series or mem_series):
            n_plots = int(bool(cpu_series)) + int(bool(mem_series))
            fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 4.5))
            if n_plots == 1:
                axes = [axes]
            ax_idx = 0
            if cpu_series and len(cpu_series) == len(rounds):
                axes[ax_idx].plot(rounds, cpu_series, marker="o", linewidth=2, color="tab:orange")
                axes[ax_idx].set_title("Avg CPU Utilization per Round")
                axes[ax_idx].set_xlabel("Round")
                axes[ax_idx].set_ylabel("CPU (%)")
                axes[ax_idx].set_ylim(0, 100)
                axes[ax_idx].grid(True, alpha=0.3)
                ax_idx += 1
            if mem_series and len(mem_series) == len(rounds):
                axes[ax_idx].plot(rounds, mem_series, marker="s", linewidth=2, color="tab:purple")
                axes[ax_idx].set_title("Avg Memory Usage per Round")
                axes[ax_idx].set_xlabel("Round")
                axes[ax_idx].set_ylabel("Memory (%)")
                axes[ax_idx].set_ylim(0, 100)
                axes[ax_idx].grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(exp_dir, "cpu_memory_per_round.png"), dpi=180, bbox_inches="tight")
            plt.close(fig)

    def _ensure_battery_plot_alias(self, exp_dir: str):
        """Ensure a canonical battery plot exists when server plot was copied with protocol-specific name."""
        canonical = os.path.join(exp_dir, "battery_consumption_per_round.png")
        if os.path.exists(canonical):
            return
        for name in os.listdir(exp_dir):
            lower = name.lower()
            if ("battery" in lower) and (lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg")):
                try:
                    shutil.copyfile(os.path.join(exp_dir, name), canonical)
                    return
                except Exception:
                    return

    @classmethod
    def for_artifact_finalization(cls, use_case: str) -> "ExperimentRunner":
        """Build a minimal runner instance for post-run JSON/plot finalization (e.g. native GUI, no Docker)."""
        obj = cls.__new__(cls)
        obj.use_case = use_case
        obj.use_case_dir_map = {
            "emotion": "Emotion_Recognition",
            "mentalstate": "MentalState_Recognition",
            "temperature": "Temperature_Regulation",
        }
        obj.use_case_dir = obj.use_case_dir_map.get(
            use_case, f"{(use_case or 'emotion').title()}_Recognition"
        )
        return obj

    def _ensure_training_includes_battery_series(self, exp_dir: str, training: Optional[Dict], protocol: str = "") -> Optional[Dict]:
        """If training dict has rounds but no battery series, estimate from client logs and attach."""
        if not training or not isinstance(training, dict):
            return training
        rounds = training.get("rounds", []) or []
        if not rounds:
            return training
        has_battery = False
        for key in ("battery_consumption", "battery_soc", "avg_battery_soc"):
            series = training.get(key)
            if isinstance(series, (list, tuple)) and len(series) > 0:
                has_battery = True
                break
        if has_battery:
            return training
        est = self._estimate_battery_series_from_logs(exp_dir, protocol=protocol)
        if not est:
            return training
        n = len(rounds)
        if len(est) > n:
            est = est[:n]
        elif len(est) < n and est:
            est = list(est) + [est[-1]] * (n - len(est))
        out = dict(training)
        out["battery_consumption"] = est
        out["battery_consumption_source"] = "estimated_from_client_logs"
        return out

    def _ensure_training_includes_battery_model_series(
        self, exp_dir: str, protocol: str, training: Optional[Dict]
    ) -> Optional[Dict]:
        """Attach battery_model_consumption (BatteryModel drain) without changing battery_consumption."""
        if not training or not isinstance(training, dict):
            return training
        rounds = training.get("rounds", []) or []
        if not rounds:
            return training
        existing = training.get("battery_model_consumption")
        if isinstance(existing, (list, tuple)) and len(existing) == len(rounds):
            return training

        est = self._battery_model_series_from_jsonl(exp_dir)
        source = "client_metrics_jsonl"
        if not est:
            est = self._estimate_battery_model_series_from_logs(exp_dir, protocol)
            source = "battery_model_estimated_from_client_logs"
        if not est:
            return training

        n = len(rounds)
        if len(est) > n:
            est = est[:n]
        elif len(est) < n and est:
            est = list(est) + [est[-1]] * (n - len(est))
        out = dict(training)
        out["battery_model_consumption"] = est
        out["battery_model_consumption_source"] = source
        return out

    def _ensure_training_includes_cpu_memory_series(
        self, exp_dir: str, training: Optional[Dict]
    ) -> Optional[Dict]:
        """Attach avg_cpu_percent and avg_memory_percent series from client JSONL when not already present."""
        if not training or not isinstance(training, dict):
            return training
        rounds = training.get("rounds", []) or []
        if not rounds:
            return training
        n = len(rounds)
        needs_cpu = not isinstance(training.get("avg_cpu_percent"), (list, tuple)) or len(training.get("avg_cpu_percent", [])) == 0
        needs_mem = not isinstance(training.get("avg_memory_percent"), (list, tuple)) or len(training.get("avg_memory_percent", [])) == 0
        if not needs_cpu and not needs_mem:
            return training

        cpu_mem = self._cpu_memory_series_from_jsonl(exp_dir)
        out = dict(training)
        if needs_cpu and cpu_mem.get("cpu_percent"):
            s = cpu_mem["cpu_percent"]
            s = s[:n] if len(s) > n else (list(s) + [s[-1]] * (n - len(s)) if len(s) < n else s)
            out["avg_cpu_percent"] = s
            out["avg_cpu_percent_source"] = "client_metrics_jsonl"
        if needs_mem and cpu_mem.get("memory_percent"):
            s = cpu_mem["memory_percent"]
            s = s[:n] if len(s) > n else (list(s) + [s[-1]] * (n - len(s)) if len(s) < n else s)
            out["avg_memory_percent"] = s
            out["avg_memory_percent_source"] = "client_metrics_jsonl"
        return out

    def _finalize_experiment_artifacts(
        self,
        server_container: Optional[str],
        protocol: str,
        exp_dir: str,
        scenario: Optional[str] = None,
    ):
        """Copy/generate all expected artifacts for an experiment folder."""
        if server_container:
            self._copy_server_plots_to_experiment_folder(server_container, protocol, exp_dir, scenario=scenario)
        training = self._resolve_training_results(exp_dir, protocol)
        training = self._ensure_training_includes_battery_series(exp_dir, training, protocol=protocol)
        training = self._ensure_training_includes_battery_model_series(exp_dir, protocol, training)
        training = self._ensure_training_includes_cpu_memory_series(exp_dir, training)
        self._persist_training_results(exp_dir, protocol, training)
        self._generate_standard_plots(protocol, exp_dir, training)
        self._ensure_battery_plot_alias(exp_dir)
    
    def run_single_experiment(self, protocol: str, scenario: str, congestion_level: str = "none"):
        """Run a single experiment with specific protocol and network scenario"""
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {protocol.upper()} - {scenario.upper()}")
        print(f"# Use Case: {self.use_case.title()}")
        print(f"# Rounds: {self.num_rounds}")
        if self.baseline_mode:
            print(f"# Mode: BASELINE (no network conditions)")
        if self.enable_congestion and congestion_level != "none":
            print(f"# Congestion Level: {congestion_level.upper()}")
        print(f"{'#'*70}\n")
        
        # Adaptive timeout based on network scenario
        # For rl_unified + Q-learning convergence: no time limit - run until Q-values converge
        # Poor networks need much more time due to retransmissions and delays
        if protocol == "rl_unified" and self.use_ql_convergence:
            timeout = None  # No limit; completion determined by Q-learning convergence
        else:
            timeout_map = {
                "excellent": 3600,          # 1 hour
                "good": 3600,               # 1 hour
                "moderate": 5400,           # 1.5 hours
                "poor": 14400,              # 4 hours
                "very_poor": 21600,         # 6 hours (300ms latency + 5% loss = very slow)
                "satellite": 9000,          # 2.5 hours (600ms latency)
                "congested_light": 5400,    # 1.5 hours
                "congested_moderate": 7200, # 2 hours
                "congested_heavy": 9000,    # 2.5 hours
                # Dynamic alternates between excellent/moderate/poor/congested_light,
                # so use a mid-range timeout similar to moderate / light congestion.
                "dynamic": 7200,
            }
            timeout = timeout_map.get(scenario, 3600)
        
        try:
            # 0. Reset epsilon for Q-learning when starting a new experiment/scenario
            # This ensures re-exploration for each new network scenario
            if protocol == "rl_unified":
                print(f"\n{'='*70}")
                if self.reset_epsilon:
                    print(f"[Q-Learning] Preparing epsilon reset for new experiment: {scenario}")
                else:
                    print(f"[Q-Learning] Continuing with previous epsilon (resume mode): {scenario}")
                print(f"{'='*70}")
                
                # Set environment variable to signal epsilon reset
                os.environ["RESET_EPSILON"] = "true" if self.reset_epsilon else "false"
                
                # Create a flag file in shared_data for containers to check
                # This ensures the reset signal persists even if env vars aren't passed through
                # Use absolute path to ensure we're writing to the correct location
                project_root = Path(__file__).parent.parent
                shared_data_path = project_root / "shared_data"
                shared_data_path.mkdir(exist_ok=True)
                
                # Create flag file with scenario identifier, unique experiment ID, and timestamp
                # Using experiment_id ensures epsilon resets for EACH new GUI experiment
                reset_flag_file = shared_data_path / "reset_epsilon_flag.txt"
                try:
                    import time
                    import uuid
                    timestamp = time.time()
                    experiment_id = str(uuid.uuid4())[:8]  # Short unique ID for this experiment
                    with open(reset_flag_file, 'w') as f:
                        f.write(f"experiment_id={experiment_id}\n")
                        f.write(f"scenario={scenario}\n")
                        f.write(f"timestamp={timestamp}\n")
                        f.write(f"reset_epsilon={1.0 if self.reset_epsilon else 0.0}\n")  # 1.0 = reset, 0.0 = continue
                    print(f"[Q-Learning] ✓ Created flag file: {reset_flag_file}")
                    print(f"[Q-Learning]   Experiment ID: {experiment_id}")
                    print(f"[Q-Learning]   Scenario: {scenario}")
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
            
            # Propagate NETWORK_SCENARIO so server containers write results to the
            # correct subdirectory via get_experiment_results_dir().
            os.environ["NETWORK_SCENARIO"] = scenario

            # Pre-run cleanup: remove stale results files from volume-mounted paths so
            # the completion detector cannot be fooled by data from previous runs.
            # This is critical for rl_unified where ../experiment_results is bind-mounted
            # as /app/results inside the server container, making leftover JSON files
            # from a prior run immediately visible to the new container.
            self._delete_stale_results_files(protocol, scenario)

            # 1. Start containers (includes traffic generators if congestion enabled)
            if not self.start_containers(protocol, scenario, congestion_level):
                print(f"[ERROR] Failed to start containers for {protocol}")
                return False
            
            # 2. Apply network scenario (skip if baseline mode)
            if not self.baseline_mode:
                if not self.apply_network_scenario(scenario, protocol):
                    print(f"[WARNING] Failed to apply network scenario {scenario}, continuing anyway...")
            else:
                print(f"[BASELINE] Skipping network conditions - running with ideal network")
            
            # 3. Wait for completion with adaptive timeout and RTT tracking
            if timeout is None:
                print(f"[INFO] No time limit - waiting until Q-learning converges for {scenario} network")
            else:
                print(f"[INFO] Using timeout: {timeout}s ({timeout/3600:.1f} hours) for {scenario} network")
            success, round_trip_times = self.wait_for_completion(protocol, timeout=timeout, scenario=scenario)
            
            if not success:
                print(f"[WARNING] Experiment may not have completed")
            
            # 4. Collect results including RTT data
            result_suffix = f"{scenario}_congestion_{congestion_level}" if congestion_level != "none" else scenario
            if self.baseline_mode:
                result_suffix = "baseline"
            self.collect_results(protocol, result_suffix, round_trip_times)
            
            # 5. Stop traffic generators (if running)
            if self.enable_congestion and self.congestion_manager and congestion_level != "none":
                print(f"\n[Congestion] Stopping traffic generators...")
                self.congestion_manager.stop_traffic_generators()
                time.sleep(2)
            
            # 6. Stop containers
            self.stop_containers(protocol)
            
            # 7. Clean up epsilon reset flag file after experiment completes
            # (This ensures next experiment creates a fresh flag)
            if protocol == "rl_unified":
                project_root = Path(__file__).parent.parent
                shared_data_path = project_root / "shared_data"
                reset_flag_file = shared_data_path / "reset_epsilon_flag.txt"
                try:
                    if reset_flag_file.exists():
                        reset_flag_file.unlink()
                        print(f"[Q-Learning] Cleaned up reset flag file after experiment completion")
                except Exception as e:
                    print(f"[Q-Learning] Warning: Could not clean up reset flag file: {e}")
            
            print(f"\n[OK] Experiment completed: {protocol.upper()} - {scenario.upper()}\n")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {e}\n")
            self.stop_containers(protocol)
            return False
    
    def run_all_experiments(self, protocols: List[str] = None, scenarios: List[str] = None, 
                          congestion_levels: List[str] = None):
        """Run experiments for all protocol and network combinations"""
        protocols = protocols or self.protocols
        scenarios = scenarios or self.network_scenarios
        congestion_levels = congestion_levels or ["none"]
        
        # If congestion is not enabled, force congestion_levels to ["none"]
        if not self.enable_congestion:
            congestion_levels = ["none"]
        
        total_experiments = len(protocols) * len(scenarios) * len(congestion_levels)
        current = 0
        
        print(f"\n{'='*70}")
        print(f"STARTING AUTOMATED EXPERIMENTS")
        print(f"{'='*70}")
        print(f"Use Case: {self.use_case.title()}")
        print(f"Protocols: {', '.join(protocols)}")
        print(f"Network Scenarios: {', '.join(scenarios)}")
        if self.enable_congestion and len(congestion_levels) > 1 or congestion_levels[0] != "none":
            print(f"Congestion Levels: {', '.join(congestion_levels)}")
        print(f"Total Experiments: {total_experiments}")
        print(f"Results Directory: {self.results_dir}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        for protocol in protocols:
            for scenario in scenarios:
                for congestion_level in congestion_levels:
                    current += 1
                    print(f"\n{'*'*70}")
                    print(f"Progress: {current}/{total_experiments}")
                    print(f"{'*'*70}")
                    
                    if self.run_single_experiment(protocol, scenario, congestion_level):
                        successful += 1
                    else:
                        failed += 1
                    
                    # Brief pause between experiments
                    time.sleep(5)
        
        # Final cleanup
        if self.enable_congestion and self.congestion_manager:
            print("\n[Congestion] Final cleanup - stopping all traffic generators...")
            self.congestion_manager.stop_traffic_generators()
        
        # Summary
        duration = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"EXPERIMENTS COMPLETED")
        print(f"{'='*70}")
        print(f"Total: {total_experiments}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration/60:.2f} minutes")
        print(f"Results saved in: {self.results_dir}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Automated Network Experiment Runner for FL",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--use-case", "-u", 
                       choices=["emotion", "mentalstate", "temperature"],
                       default="emotion",
                       help="Use case to run experiments for")
    parser.add_argument("--protocols", "-p",
                       nargs="+",
                       choices=["mqtt", "amqp", "grpc", "quic", "http3", "dds", "rl_unified"],
                       help="Specific protocols to test (default: all). Use 'rl_unified' for RL-based dynamic protocol selection")
    parser.add_argument("--scenarios", "-s",
                       nargs="+",
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
                           "dynamic",
                       ],
                       help="Specific network scenarios to test (default: all)")
    parser.add_argument("--rounds", "-r",
                       type=int,
                       default=10,
                       help="Number of FL rounds (default: 10)")
    parser.add_argument(
        "--termination-mode",
        choices=["client_convergence", "fixed_rounds"],
        default="client_convergence",
        help="End condition: default ends on client convergence (may stop early), fixed_rounds runs selected rounds."
    )
    parser.add_argument("--enable-congestion", action="store_true",
                       help="Enable network congestion using traffic generators")
    parser.add_argument("--congestion-level", "-c",
                       choices=["none", "light", "moderate", "heavy", "extreme"],
                       help="Congestion level for experiments (requires --enable-congestion)")
    parser.add_argument("--congestion-levels",
                       nargs="+",
                       choices=["none", "light", "moderate", "heavy", "extreme"],
                       help="Multiple congestion levels to test (requires --enable-congestion)")
    parser.add_argument("--single", action="store_true",
                       help="Run single experiment (requires --protocol and --scenario)")
    parser.add_argument("--protocol",
                       choices=["mqtt", "amqp", "grpc", "quic", "http3", "dds", "rl_unified"],
                       help="Protocol for single experiment (use 'rl_unified' for RL-based selection)")
    parser.add_argument("--scenario",
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
                           "dynamic",
                       ],
                       help="Network scenario for single experiment")
    parser.add_argument("--use-quantization", action="store_true",
                help="Enable quantization for clients and servers (sets USE_QUANTIZATION env var)")
    parser.add_argument("--quantization-strategy",
                choices=["qat", "ptq", "parameter_quantization"],
                help="Quantization strategy to set in environment (QUANTIZATION_STRATEGY)")
    parser.add_argument("--quantization-bits", type=int,
                choices=[4, 8, 16, 32],
                help="Bit width for quantization (QUANTIZATION_BITS). 4-bit uses nibble packing.")
    parser.add_argument("--quantization-symmetric", action="store_true",
                help="Set QUANTIZATION_SYMMETRIC=1 if symmetric quantization should be used")
    parser.add_argument("--quantization-per-channel", action="store_true",
                help="Set QUANTIZATION_PER_CHANNEL=1 if per-channel quantization should be used")
    parser.add_argument("--use-pruning", action="store_true",
                help="Enable model pruning for clients and server (sets USE_PRUNING env var)")
    parser.add_argument("--pruning-sparsity", type=float,
                help="Target sparsity for pruning as a fraction between 0.0 and 1.0 (PRUNING_SPARSITY)")
    parser.add_argument("--pruning-structured", action="store_true",
                help="Enable structured pruning (sets PRUNING_STRUCTURED=true)")
    parser.add_argument("--enable-gpu", "-g", action="store_true",
                help="Enable GPU acceleration using NVIDIA runtime (requires nvidia-docker)")
    parser.add_argument("--network-mode", choices=["gpu", "host", "host_macvlan"], default="gpu",
                help="Network mode: gpu=Docker bridge (*.gpu-isolated.yml, per-container tc), host=host network (*.host-network.yml, tc on host), host_macvlan=macvlan (*.macvlan.yml, per-container tc)")
    parser.add_argument("--baseline", "-b", action="store_true",
                help="Baseline mode: Run without network conditions and save to baseline folder")
    parser.add_argument("--use-ql-convergence", action="store_true",
                help="Unified only: End training when Q-learning value converges (multiple episodes); else end on accuracy convergence")
    parser.add_argument(
        "--rl-inference-only",
        action="store_true",
        help="Unified only (with --use-ql-convergence off): greedy protocol selection, no epsilon exploration. GUI Inference mode sets this.",
    )
    parser.add_argument(
        "--disable-communication-model-reward",
        action="store_true",
        help="Unified only: do not let communication-model T_calc affect RL rewards.",
    )
    parser.add_argument(
        "--no-reset-epsilon",
        action="store_true",
        help="Unified training only: Do NOT reset epsilon to 1.0 on experiment start. Continue with previous epsilon value and accumulated learning (useful for resuming interrupted training).",
    )
    parser.add_argument(
        "--local-clients",
        type=int,
        default=2,
        help="Number of client containers to start from this runner on the central machine (default: 2). Use 0 for server/brokers only.",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Total participants the FL server waits for (local + remote). "
            "Defaults to --local-clients when omitted. Set to match experiment GUI 'Min Clients (server)'."
        ),
    )
    parser.add_argument(
        "--dds-impl",
        choices=["cyclonedds", "fastdds"],
        default="cyclonedds",
        help="DDS implementation vendor to use when running DDS experiments.",
    )
    parser.add_argument(
        "--fl-dataset-for-client",
        action="append",
        metavar="CLIENT=SHARD",
        default=None,
        help=(
            "Map container CLIENT_ID to data shard (1-based): emotion uses Dataset/client_SHARD; "
            "mental state uses non-IID partition SHARD (converted to 0-based internally). "
            "Repeatable, e.g. --fl-dataset-for-client 1=3 --fl-dataset-for-client 2=1"
        ),
    )
    
    args = parser.parse_args()
    
    # Propagate selected termination mode and round cap into Docker compose variable substitution.
    os.environ["NUM_ROUNDS"] = str(args.rounds)
    os.environ["STOP_ON_CLIENT_CONVERGENCE"] = "false" if args.termination_mode == "fixed_rounds" else "true"
    os.environ["TRAINING_TERMINATION_MODE"] = (
        "fixed_rounds" if args.termination_mode == "fixed_rounds" else "client_convergence"
    )
    
    # Propagate DDS implementation choice to environment so compose scripts and containers can read it
    if args.dds_impl:
        os.environ["DDS_IMPL"] = args.dds_impl

    # Determine congestion level(s)
    congestion_levels = None
    if args.enable_congestion:
        if args.congestion_levels:
            congestion_levels = args.congestion_levels
        elif args.congestion_level:
            congestion_levels = [args.congestion_level]
        else:
            # Default to moderate congestion if enabled but no level specified
            congestion_levels = ["moderate"]
    
    # Build quantization params dict to pass into runner
    quant_params = {}
    if args.use_quantization:
        if args.quantization_strategy:
            quant_params["QUANTIZATION_STRATEGY"] = args.quantization_strategy
        if args.quantization_bits:
            quant_params["QUANTIZATION_BITS"] = str(args.quantization_bits)
        if args.quantization_symmetric:
            quant_params["QUANTIZATION_SYMMETRIC"] = "1"
        if args.quantization_per_channel:
            quant_params["QUANTIZATION_PER_CHANNEL"] = "1"
    # Build pruning params dict to pass into runner
    pruning_params: Dict[str, str] = {}
    if args.use_pruning:
        if args.pruning_sparsity is not None:
            pruning_params["PRUNING_SPARSITY"] = str(args.pruning_sparsity)
        if args.pruning_structured:
            pruning_params["PRUNING_STRUCTURED"] = "true"

    dataset_client_map = parse_fl_dataset_for_client(getattr(args, "fl_dataset_for_client", None))

    runner = ExperimentRunner(
        use_case=args.use_case,
        num_rounds=args.rounds,
        enable_congestion=args.enable_congestion,
        use_quantization=args.use_quantization,
        quantization_params=quant_params,
        use_pruning=args.use_pruning,
        pruning_params=pruning_params,
        enable_gpu=args.enable_gpu,
        network_mode=args.network_mode,
        baseline_mode=args.baseline,
        use_ql_convergence=args.use_ql_convergence,
        rl_inference_only=getattr(args, "rl_inference_only", False),
        local_clients=args.local_clients,
        min_clients=args.min_clients,
        use_communication_model_reward=not args.disable_communication_model_reward,
        reset_epsilon=not args.no_reset_epsilon,  # Default True (reset), --no-reset-epsilon makes it False
        dataset_client_map=dataset_client_map,
    )
    
    # In baseline mode, run all protocols with excellent scenario (no network conditions)
    if args.baseline:
        print("\n" + "="*70)
        print("BASELINE MODE: Running all protocols without network conditions")
        print("="*70 + "\n")
        protocols = args.protocols or runner.protocols
        for protocol in protocols:
            runner.run_single_experiment(protocol, "excellent", "none")
        return
    
    if args.single:
        if not args.protocol or not args.scenario:
            print("[ERROR] --single requires both --protocol and --scenario")
            parser.print_help()
            return
        congestion = args.congestion_level if args.enable_congestion and args.congestion_level else "none"
        runner.run_single_experiment(args.protocol, args.scenario, congestion)
    else:
        runner.run_all_experiments(
            protocols=args.protocols,
            scenarios=args.scenarios,
            congestion_levels=congestion_levels
        )


if __name__ == "__main__":
    main()
