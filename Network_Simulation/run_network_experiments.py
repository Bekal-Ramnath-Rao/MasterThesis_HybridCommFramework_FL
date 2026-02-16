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
from datetime import datetime
from typing import List, Dict, Optional
import argparse
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class ExperimentRunner:
    """Automates running FL experiments across different network conditions"""
    
    def __init__(self, use_case: str = "emotion", num_rounds: int = 10, enable_congestion: bool = False,
                 use_quantization: bool = False, quantization_params: Dict[str, str] = None, enable_gpu: bool = False,
                 baseline_mode: bool = False, use_ql_convergence: bool = False):
        self.use_case = use_case
        self.num_rounds = num_rounds
        self.enable_congestion = enable_congestion
        self.use_quantization = use_quantization
        self.enable_gpu = enable_gpu
        self.baseline_mode = baseline_mode
        self.use_ql_convergence = use_ql_convergence
        # quantization_params expected to be a dict of simple string values
        self.quantization_params = quantization_params or {}
        
        # Print quantization status
        if self.use_quantization:
            print(f"\n{'='*70}")
            print("QUANTIZATION ENABLED")
            print(f"Parameters: {self.quantization_params}")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print("QUANTIZATION DISABLED")
            print(f"{'='*70}\n")
        
        # Get the script's directory and project root
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        
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
        self.protocols = ["mqtt", "amqp", "grpc", "quic", "dds", "rl_unified"]
        
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
            "congested_heavy"
        ]
        
        # Docker compose file mapping (relative to project root)
        docker_dir = project_root / "Docker"
        
        # Use GPU-isolated files when GPU is enabled, otherwise use standard files
        if enable_gpu:
            self.compose_files = {
                "emotion": str(docker_dir / "docker-compose-emotion.gpu-isolated.yml"),
                "mentalstate": str(docker_dir / "docker-compose-mentalstate.gpu-isolated.yml"),
                "temperature": str(docker_dir / "docker-compose-temperature.gpu-isolated.yml")
            }
            # Unified compose files for RL-based protocol selection
            self.unified_compose_files = {
                "emotion": str(docker_dir / "docker-compose-unified-emotion.yml"),
                "mentalstate": str(docker_dir / "docker-compose-unified-mentalstate.yml"),
                "temperature": str(docker_dir / "docker-compose-unified-temperature.yml")
            }
            # No overlay files needed with GPU-isolated (they are standalone)
            self.gpu_overlay_files = {}
        else:
            self.compose_files = {
                "emotion": str(docker_dir / "docker-compose-emotion.yml"),
                "mentalstate": str(docker_dir / "docker-compose-mentalstate.yml"),
                "temperature": str(docker_dir / "docker-compose-temperature.yml")
            }
            # Unified compose files for RL-based protocol selection (non-GPU)
            self.unified_compose_files = {
                "emotion": str(docker_dir / "docker-compose-unified-emotion.yml"),
                "mentalstate": str(docker_dir / "docker-compose-unified-mentalstate.yml"),
                "temperature": str(docker_dir / "docker-compose-unified-temperature.yml")
            }
            # GPU overlay files (not used when enable_gpu=False)
            self.gpu_overlay_files = {
                "emotion": str(docker_dir / "docker-compose-emotion.gpu.yml"),
                "mentalstate": str(docker_dir / "docker-compose-mentalstate.gpu.yml"),
                "temperature": str(docker_dir / "docker-compose-temperature.gpu.yml")
            }
        
        # Service name patterns
        # Note: Use different broker names for GPU-isolated vs standard compose files
        broker_mqtt = "mqtt-broker" if enable_gpu else "mqtt-broker"
        broker_amqp = "amqp-broker" if enable_gpu else "rabbitmq"
        
        self.service_patterns = {
            "emotion": {
                "mqtt": [broker_mqtt, "fl-server-mqtt-emotion", "fl-client-mqtt-emotion-1", "fl-client-mqtt-emotion-2"],
                "amqp": [broker_amqp, "fl-server-amqp-emotion", "fl-client-amqp-emotion-1", "fl-client-amqp-emotion-2"],
                "grpc": ["fl-server-grpc-emotion", "fl-client-grpc-emotion-1", "fl-client-grpc-emotion-2"],
                "quic": ["fl-server-quic-emotion", "fl-client-quic-emotion-1", "fl-client-quic-emotion-2"],
                "dds": ["fl-server-dds-emotion", "fl-client-dds-emotion-1", "fl-client-dds-emotion-2"],
                "rl_unified": ["fl-server-unified-emotion", "fl-client-unified-emotion-1", "fl-client-unified-emotion-2"]
            },
            "mentalstate": {
                "mqtt": ["mqtt-broker-mental", "fl-server-mqtt-mental", "fl-client-mqtt-mental-1", "fl-client-mqtt-mental-2"],
                "amqp": ["rabbitmq-mental", "fl-server-amqp-mental", "fl-client-amqp-mental-1", "fl-client-amqp-mental-2"],
                "grpc": ["fl-server-grpc-mental", "fl-client-grpc-mental-1", "fl-client-grpc-mental-2"],
                "quic": ["fl-server-quic-mental", "fl-client-quic-mental-1", "fl-client-quic-mental-2"],
                "dds": ["fl-server-dds-mental", "fl-client-dds-mental-1", "fl-client-dds-mental-2"],
                "rl_unified": ["fl-server-unified-mental", "fl-client-unified-mental-1", "fl-client-unified-mental-2"]
            },
            "temperature": {
                "mqtt": ["mqtt-broker-temp", "fl-server-mqtt-temp", "fl-client-mqtt-temp-1", "fl-client-mqtt-temp-2"],
                "amqp": ["rabbitmq-temp", "fl-server-amqp-temp", "fl-client-amqp-temp-1", "fl-client-amqp-temp-2"],
                "grpc": ["fl-server-grpc-temp", "fl-client-grpc-temp-1", "fl-client-grpc-temp-2"],
                "quic": ["fl-server-quic-temp", "fl-client-quic-temp-1", "fl-client-quic-temp-2"],
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
        if env_vars:
            env.update(env_vars)
            print(f"[ENV] {env_vars}")
        # Use project root as working directory if not specified
        if cwd is None:
            cwd = str(Path(__file__).parent.parent)
        return subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', check=check, env=env, cwd=cwd)
    
    def start_containers(self, protocol: str, scenario: str = "excellent", congestion_level: str = "none"):
        """Start Docker containers for a specific protocol with staged startup"""
        
        # Check if rl_unified is selected - use unified compose file
        if protocol == "rl_unified":
            print("\n" + "="*70)
            print("STARTING RL-UNIFIED MODE")
            print("="*70)
            print("Using unified docker-compose file with all protocol brokers")
            print("Server will handle: MQTT, AMQP, gRPC, QUIC, DDS")
            print("Clients will use RL-based Q-Learning to select protocol")
            if self.use_ql_convergence:
                print("End condition: Q-learning convergence (multiple episodes)")
            else:
                print("End condition: accuracy convergence (current behavior)")
            print("="*70 + "\n")
            
            compose_file = self.unified_compose_files[self.use_case]
            if self.use_ql_convergence:
                os.environ["USE_QL_CONVERGENCE"] = "true"
            else:
                os.environ["USE_QL_CONVERGENCE"] = "false"
            
            # For unified mode, start all services together
            compose_cmd = ["docker", "compose", "-f", compose_file, "up", "-d"]
            
            print(f"Starting unified FL system for {self.use_case}...")
            result = self.run_command(compose_cmd)
            
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
        compose_file = self.compose_files[self.use_case]
        
        # Build compose command (no overlay needed with GPU-isolated files)
        compose_cmd_base = ["docker", "compose", "-f", compose_file]
        # Note: GPU-isolated files are standalone, no overlay needed
        
        services = self.service_patterns[self.use_case][protocol]
        
        print(f"\n{'='*70}")
        print(f"Starting containers for {protocol.upper()} protocol...")
        print(f"Network Scenario: {scenario.upper()}")
        if self.enable_gpu:
            print(f"GPU Acceleration: ENABLED (2x RTX 3080)")
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
            result = self.run_command(cmd_broker)
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
            result = self.run_command(cmd_server)
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
            stage_num = "[Stage 4/4]" if (broker and self.enable_congestion and congestion_level != "none") else "[Stage 3/4]"
            print(f"\n{stage_num} Starting clients: {', '.join(clients)}")
            cmd_clients = compose_cmd_base + ["up", "-d"] + clients
            result = self.run_command(cmd_clients)
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
        
        # Use unified compose file for rl_unified
        if protocol == "rl_unified":
            compose_file = self.unified_compose_files[self.use_case]
        else:
            compose_file = self.compose_files[self.use_case]
        services = self.service_patterns[self.use_case][protocol]
        
        print(f"\nStopping containers for {protocol.upper()} protocol...")
        
        # Build compose command (no overlay needed with GPU-isolated files)
        cmd = ["docker", "compose", "-f", compose_file, "down"]
        # Note: GPU-isolated files are standalone, no overlay needed
        
        self.run_command(cmd, check=False)
        
        # Give time for containers to fully stop
        time.sleep(5)
    
    def apply_network_scenario(self, scenario: str, protocol: str):
        """Apply network conditions using network_simulator.py"""
        print(f"\n{'='*70}")
        print(f"Applying network scenario: {scenario.upper()}")
        print(f"{'='*70}")
        
        # Get actual container names for this protocol
        services = self.service_patterns[self.use_case][protocol]
        
        # Apply network conditions to each container individually
        import sys
        from pathlib import Path
        # Add parent directory to path to import network_simulator
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from network_simulator import NetworkSimulator
        sim = NetworkSimulator(verbose=True)
        
        if scenario not in sim.NETWORK_SCENARIOS:
            print(f"[WARNING] Unknown scenario: {scenario}, skipping network simulation")
            return True  # Don't fail the experiment
        
        conditions = sim.NETWORK_SCENARIOS[scenario]
        
        # Apply to all containers EXCEPT brokers (brokers should be reliable infrastructure)
        # Only apply to FL clients and servers
        success_count = 0
        for container in services:
            # Skip brokers - they represent reliable infrastructure
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
        return success_count > 0  # At least one container should succeed
    
    def wait_for_completion(self, protocol: str, timeout: Optional[int] = 3600):
        """Wait for FL training to complete and track round trip times.
        When timeout is None, wait indefinitely (used for rl_unified + Q-learning convergence)."""
        if timeout is None:
            print(f"\nWaiting for {protocol.upper()} training to complete (no time limit - until Q-learning converges)...")
        else:
            print(f"\nWaiting for {protocol.upper()} training to complete (timeout: {timeout}s)...")
        
        # Get server container name
        services = self.service_patterns[self.use_case][protocol]
        server_container = [s for s in services if "server" in s][0]
        
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
            
            # If any known completion marker appears, treat as complete
            if any(marker in log_content for marker in completion_markers):
                print(f"Training completed successfully (marker detected)!")
                time.sleep(5)  # Give time for final results to be written
                return True, round_trip_times

            # Check if training has reached the target number of rounds
            # by examining the results file content (not just existence)
            results_dir = f"/app/Server/{self.use_case.title()}_Recognition/results"
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
        
        for result_file in result_files:
            try:
                # Try to copy from server container
                self.run_command([
                    "docker", "cp",
                    f"{server_container}:/app/Server/{self.use_case.title()}_Recognition/results/{result_file}",
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
        
        print(f"Results saved to: {exp_dir}")
    
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
                "excellent": 3600,      # 1 hour
                "good": 3600,           # 1 hour
                "moderate": 5400,       # 1.5 hours
                "poor": 14400,           # 4 hours
                "very_poor": 21600,     # 6 hours (300ms latency + 5% loss = very slow)
                "satellite": 9000,      # 2.5 hours (600ms latency)
                "congested_light": 5400,   # 1.5 hours
                "congested_moderate": 7200, # 2 hours
                "congested_heavy": 9000     # 2.5 hours
            }
            timeout = timeout_map.get(scenario, 3600)
        
        try:
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
            success, round_trip_times = self.wait_for_completion(protocol, timeout=timeout)
            
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
                       choices=["mqtt", "amqp", "grpc", "quic", "dds", "rl_unified"],
                       help="Specific protocols to test (default: all). Use 'rl_unified' for RL-based dynamic protocol selection")
    parser.add_argument("--scenarios", "-s",
                       nargs="+",
                       choices=["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                               "congested_light", "congested_moderate", "congested_heavy"],
                       help="Specific network scenarios to test (default: all)")
    parser.add_argument("--rounds", "-r",
                       type=int,
                       default=10,
                       help="Number of FL rounds (default: 10)")
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
                       choices=["mqtt", "amqp", "grpc", "quic", "dds", "rl_unified"],
                       help="Protocol for single experiment (use 'rl_unified' for RL-based selection)")
    parser.add_argument("--scenario",
                       choices=["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                               "congested_light", "congested_moderate", "congested_heavy"],
                       help="Network scenario for single experiment")
    parser.add_argument("--use-quantization", action="store_true",
                help="Enable quantization for clients and servers (sets USE_QUANTIZATION env var)")
    parser.add_argument("--quantization-strategy",
                choices=["qat", "ptq", "parameter_quantization"],
                help="Quantization strategy to set in environment (QUANTIZATION_STRATEGY)")
    parser.add_argument("--quantization-bits", type=int,
                choices=[8, 16, 32],
                help="Bit width for quantization (QUANTIZATION_BITS)")
    parser.add_argument("--quantization-symmetric", action="store_true",
                help="Set QUANTIZATION_SYMMETRIC=1 if symmetric quantization should be used")
    parser.add_argument("--quantization-per-channel", action="store_true",
                help="Set QUANTIZATION_PER_CHANNEL=1 if per-channel quantization should be used")
    parser.add_argument("--enable-gpu", "-g", action="store_true",
                help="Enable GPU acceleration using NVIDIA runtime (requires nvidia-docker)")
    parser.add_argument("--baseline", "-b", action="store_true",
                help="Baseline mode: Run without network conditions and save to baseline folder")
    parser.add_argument("--use-ql-convergence", action="store_true",
                help="Unified only: End training when Q-learning value converges (multiple episodes); else end on accuracy convergence")
    
    args = parser.parse_args()
    
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

    runner = ExperimentRunner(use_case=args.use_case, num_rounds=args.rounds, 
                            enable_congestion=args.enable_congestion,
                            use_quantization=args.use_quantization,
                            quantization_params=quant_params,
                            enable_gpu=args.enable_gpu,
                            baseline_mode=args.baseline,
                            use_ql_convergence=args.use_ql_convergence)
    
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
