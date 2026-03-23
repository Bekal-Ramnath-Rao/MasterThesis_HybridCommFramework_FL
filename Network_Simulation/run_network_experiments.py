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
import random

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
        # Number of client containers started from this runner on the central machine
        self.local_clients = max(1, int(local_clients or 1))
        # quantization_params expected to be a dict of simple string values
        self.quantization_params = quantization_params or {}
        # pruning_params expected to be a dict of simple string values
        self.pruning_params = pruning_params or {}
        self._runtime_compose_files: Dict[str, str] = {}
        self._runtime_compose_files_stop: Dict[str, str] = {}
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
            folder_name = use_case
            self.results_dir = project_root / "experiment_results_baseline" / folder_name
        else:
            # Regular experiments
            folder_parts = [use_case]
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
            # Dynamic: per-round random choice between a subset of scenarios
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
                "emotion": str(docker_dir / "docker-compose-unified-emotion.host-network.yml"),
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
        
        self.service_patterns = {
            "emotion": {
                "mqtt": [broker_mqtt, "fl-server-mqtt-emotion", "fl-client-mqtt-emotion-1", "fl-client-mqtt-emotion-2"],
                "amqp": [broker_amqp, "fl-server-amqp-emotion", "fl-client-amqp-emotion-1", "fl-client-amqp-emotion-2"],
                "grpc": ["fl-server-grpc-emotion", "fl-client-grpc-emotion-1", "fl-client-grpc-emotion-2"],
                "quic": ["fl-server-quic-emotion", "fl-client-quic-emotion-1", "fl-client-quic-emotion-2"],
                "http3": ["fl-server-http3-emotion", "fl-client-http3-emotion-1", "fl-client-http3-emotion-2"],
                "dds": ["fl-server-dds-emotion", "fl-client-dds-emotion-1", "fl-client-dds-emotion-2"],
                "rl_unified": ["fl-server-unified-emotion", "fl-client-unified-emotion-1", "fl-client-unified-emotion-2"]
            },
            "mentalstate": {
                "mqtt": ["mqtt-broker-mental", "fl-server-mqtt-mental", "fl-client-mqtt-mental-1", "fl-client-mqtt-mental-2"],
                "amqp": ["rabbitmq-mental", "fl-server-amqp-mental", "fl-client-amqp-mental-1", "fl-client-amqp-mental-2"],
                "grpc": ["fl-server-grpc-mental", "fl-client-grpc-mental-1", "fl-client-grpc-mental-2"],
                "quic": ["fl-server-quic-mental", "fl-client-quic-mental-1", "fl-client-quic-mental-2"],
                "http3": ["fl-server-http3-mental", "fl-client-http3-mental-1", "fl-client-http3-mental-2"],
                "dds": ["fl-server-dds-mental", "fl-client-dds-mental-1", "fl-client-dds-mental-2"],
                "rl_unified": ["fl-server-unified-mental", "fl-client-unified-mental-1", "fl-client-unified-mental-2"]
            },
            "temperature": {
                "mqtt": ["mqtt-broker-temp", "fl-server-mqtt-temp", "fl-client-mqtt-temp-1", "fl-client-mqtt-temp-2"],
                "amqp": ["rabbitmq-temp", "fl-server-amqp-temp", "fl-client-amqp-temp-1", "fl-client-amqp-temp-2"],
                "grpc": ["fl-server-grpc-temp", "fl-client-grpc-temp-1", "fl-client-grpc-temp-2"],
                "quic": ["fl-server-quic-temp", "fl-client-quic-temp-1", "fl-client-quic-temp-2"],
                "http3": ["fl-server-http3-temp", "fl-client-http3-temp-1", "fl-client-http3-temp-2"],
                "dds": ["fl-server-dds-temp", "fl-client-dds-temp-1", "fl-client-dds-temp-2"],
                "rl_unified": ["fl-server-unified-temp", "fl-client-unified-temp-1", "fl-client-unified-temp-2"]
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
        return self._patch_compose_dataset_env(patched_path)

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
                # RL training: only one client runs; server waits for 1 client; run until Q converges and exit
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
                services_single_client = [
                    "mqtt-broker-unified",
                    "amqp-broker-unified",
                    f"fl-server-unified-{uc}",
                    f"fl-client-unified-{uc}-1",
                ]
                compose_cmd = ["docker", "compose", "-f", compose_file, "up", "-d"] + services_single_client
            else:
                compose_cmd = ["docker", "compose", "-f", compose_file, "up", "-d"]
            
            print(f"Starting unified FL system for {self.use_case}...")
            result = self.run_command(compose_cmd, check=False)
            
            if result.returncode != 0:
                print(f"[ERROR] Failed to start unified containers:")
                print(result.stderr)
                raise RuntimeError("Failed to start unified containers")
            
            print("[SUCCESS] All unified containers started")
            print("  ✓ MQTT Broker")
            print("  ✓ AMQP Broker")
            print("  ✓ Unified FL Server")
            print("  ✓ Unified FL Clients")
            
            # Wait for services to initialize
            print("\nWaiting for services to initialize (15 seconds)...")
            time.sleep(15)
            
            return True
        
        # Regular protocol handling (existing code)
        compose_file = self._get_runtime_compose_file(self.compose_files[self.use_case])
        
        # host_macvlan: ensure fl-macvlan network exists before up
        if self.network_mode == "host_macvlan":
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from network_simulator import NetworkSimulator
            sim = NetworkSimulator(verbose=True)
            if not sim.ensure_macvlan_network():
                raise RuntimeError("Failed to create macvlan network for host_macvlan mode")
        compose_cmd_base = ["docker", "compose", "-f", compose_file]
        
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
            num_local = max(1, min(self.local_clients, len(clients)))
            clients_to_start = clients[:num_local]
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
        
        cmd = ["docker", "compose", "-f", compose_file, "down"]
        
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
        from network_simulator import NetworkSimulator

        print(f"\n{'='*70}")
        print(f"Applying network scenario: {scenario.upper()}")
        print(f"{'='*70}")
        
        sim = NetworkSimulator(verbose=True)

        # For the special "dynamic" scenario, we do not apply a fixed profile
        # here. Instead, we pick a base scenario at apply-time so that the
        # network can change over time and between experiments.
        dynamic_bases = ["excellent", "moderate", "poor", "congested_light"]
        if scenario == "dynamic":
            base_scenario = random.choice(dynamic_bases)
            print(f"[DYNAMIC] Initial dynamic base scenario selected: {base_scenario}")
        else:
            base_scenario = scenario

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

        # gpu or host_macvlan: per-container tc
        success_count = 0
        for container in services:
            if 'broker' in container.lower() or 'rabbitmq' in container.lower() or 'server' in container.lower():
                print(f"[INFO] Skipping network conditions for broker/server: {container} (infrastructure)")
                continue
            try:
                if sim.apply_network_conditions(container, conditions):
                    success_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to apply conditions to {container}: {e}")
        print(f"\nApplied network conditions to {success_count}/{len(services)} containers")
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
        client_containers = [s for s in services if "client" in s and ("broker" not in s.lower() and "rabbitmq" not in s.lower())]
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
            "Results saved to",
            "Experiment finished",
        ]

        start_time = time.time()
        while timeout is None or (time.time() - start_time < timeout):
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
                                from network_simulator import NetworkSimulator, build_delay_models
                                sim = NetworkSimulator(verbose=False)
                                sigma_factor = float(os.environ.get("GAUSSIAN_SIGMA_FACTOR", "0.05"))
                                use_extra_jitter = os.environ.get("USE_EXTRA_JITTER", "0").strip().lower() in ("1", "true", "yes")
                                models = build_delay_models(sim.NETWORK_SCENARIOS, sigma_factor=sigma_factor)

                                # For dynamic scenario, pick a fresh base scenario
                                # every time we resample, to mimic a time-varying
                                # network that switches between the four options
                                # before client->server send.
                                if scenario == "dynamic":
                                    dynamic_bases = ["excellent", "moderate", "poor", "congested_light"]
                                    base_scenario = random.choice(dynamic_bases)
                                    print(f"  [DYNAMIC] Per-round base scenario (client->server send): {base_scenario}")
                                else:
                                    base_scenario = scenario

                                conditions = sim.get_scenario_conditions_sampled(base_scenario, models, use_extra_jitter=use_extra_jitter)
                                if self.network_mode == "host" and getattr(self, "_host_network_sim", None) is not None:
                                    self._host_network_sim.apply_network_conditions_host(conditions)
                                else:
                                    for container in client_containers:
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
            results_dir = f"/app/Server/{self.use_case_dir}/results"
            expected_json = f"{protocol}_training_results.json"
            
            # Read the results file content to check if expected rounds are complete
            read_results = self.run_command([
                "docker", "exec", server_container, "cat",
                f"{results_dir}/{expected_json}"
            ], check=False)
            
            if read_results.returncode == 0 and read_results.stdout:
                try:
                    results_data = json.loads(read_results.stdout)
                    # Check if we have results for all expected rounds
                    if isinstance(results_data, dict):
                        rounds_completed = results_data.get("rounds_completed", 0)
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

        # Save client logs as well for debugging client-side behavior
        client_containers = [s for s in services if "client" in s]
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
        
        for result_file in result_files:
            try:
                # Try to copy from server container
                self.run_command([
                    "docker", "cp",
                    f"{server_container}:/app/Server/{self.use_case_dir}/results/{result_file}",
                    os.path.join(exp_dir, result_file)
                ], check=False)
            except:
                pass
        
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
        metrics_by_round = {}

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
                metrics_by_round[round_num] = (float(loss_str), float(acc_str))

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
            metrics_by_round[round_num] = (float(loss_str), float(acc_str))

        if not metrics_by_round:
            return None

        for round_num in sorted(metrics_by_round.keys()):
            loss_val, acc_val = metrics_by_round[round_num]
            rounds.append(round_num)
            loss.append(loss_val)
            accuracy.append(acc_val / 100.0 if acc_val > 1.0 else acc_val)

        if not rounds:
            return None

        convergence_time = None
        conv_patterns = [
            re.compile(r"Convergence\s*time\s*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
            re.compile(r"Total\s+Training\s+Time\s*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
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

        return {
            "rounds": rounds,
            "loss": loss,
            "accuracy": accuracy,
            "convergence_time_seconds": convergence_time,
            "convergence_time_minutes": (convergence_time / 60.0) if convergence_time is not None else None,
            "total_rounds": len(rounds),
            "final_accuracy": accuracy[-1] if accuracy else None,
            "final_loss": loss[-1] if loss else None,
        }

    def _resolve_training_results(self, exp_dir: str, protocol: str) -> Optional[Dict]:
        """Resolve best available training results, preferring data consistent with server logs."""
        json_results = self._read_training_results(exp_dir, protocol)
        log_results = self._parse_training_results_from_server_logs(exp_dir)

        if json_results is None and log_results is None:
            return None
        if json_results is None:
            return log_results
        if log_results is None:
            return json_results

        # If JSON rounds are stale/incomplete, trust server log parse
        json_rounds = json_results.get("rounds", []) or []
        log_rounds = log_results.get("rounds", []) or []
        if len(log_rounds) > len(json_rounds):
            merged = dict(log_results)
            # preserve richer fields from json when present
            if json_results.get("num_clients") is not None:
                merged["num_clients"] = json_results.get("num_clients")
            # merge battery-related series if JSON has them
            for key in ("battery_consumption", "battery_soc", "avg_battery_soc", "round_times_seconds"):
                if json_results.get(key):
                    merged[key] = json_results.get(key)
            return merged

        return json_results

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

    def _estimate_battery_series_from_logs(self, exp_dir: str) -> List[float]:
        """Estimate cumulative battery consumption per round from saved client logs.

        Fallback for historical experiments where the server generated a battery plot
        but the timestamped experiment folder did not receive either the plot or the
        `battery_consumption` series in copied JSON.
        """
        # Shared battery model constants used by clients
        k_tx = 1e-8
        E_fixed = 0.1
        P_CPU_MAX = 10.0
        BATTERY_CAP_J = 60.0 * 3600.0
        cpu_util_fraction = 0.5  # conservative fallback when exact runtime CPU is unavailable

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
            if not name.endswith("_logs.txt") or "client" not in name.lower():
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
                    e_radio = (k_tx * bits_tx) + E_fixed
                    e_cpu = P_CPU_MAX * cpu_util_fraction * training_times.get(rnd, 0.0)
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

    def _copy_server_plots_to_experiment_folder(self, server_container: str, protocol: str, exp_dir: str):
        """Copy server-side generated plots into this experiment folder when available."""
        protocol_alias = "unified" if protocol == "rl_unified" else protocol
        use_case_aliases = [self.use_case.lower()]
        if self.use_case.lower() == "mentalstate":
            use_case_aliases.append("mental_state")

        candidate_dirs = []
        for uc in use_case_aliases:
            candidate_dirs.extend([
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
        convergence_time_sec = float(training.get("convergence_time_seconds", 0.0) or 0.0)

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
            battery_series = self._estimate_battery_series_from_logs(exp_dir)
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

    def _finalize_experiment_artifacts(self, server_container: str, protocol: str, exp_dir: str):
        """Copy/generate all expected artifacts for an experiment folder."""
        self._copy_server_plots_to_experiment_folder(server_container, protocol, exp_dir)
        training = self._resolve_training_results(exp_dir, protocol)
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
        help="Number of client containers to start from this runner on the central machine (default: 2).",
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
