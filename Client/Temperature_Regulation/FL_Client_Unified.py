"""
Unified Federated Learning Client for Temperature Regulation
with RL-based Protocol Selection

Supports: MQTT, AMQP, gRPC, QUIC, DDS
Uses Q-Learning to dynamically select the best protocol
"""

import os
import sys
import time
import socket
import numpy as np
import pandas as pd
_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
if _xla_flags:
    sanitized_flags = [f for f in _xla_flags.split() if f != "--xla_gpu_enable_command_buffer="]
    if sanitized_flags:
        os.environ["XLA_FLAGS"] = " ".join(sanitized_flags)
    else:
        os.environ.pop("XLA_FLAGS", None)
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Optional, List
import json
import pickle
import base64
import logging

# Import protocol-specific modules
import paho.mqtt.client as mqtt_client
import pika  # AMQP
import grpc  # gRPC
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))
    import federated_learning_pb2
    import federated_learning_pb2_grpc
    GRPC_PROTO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    federated_learning_pb2 = None
    federated_learning_pb2_grpc = None
    GRPC_PROTO_AVAILABLE = False
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_q_learning_selector import QLearningProtocolSelector, EnvironmentStateManager
try:
    from rl_q_learning_selector import finalize_rl_boundary_collection_and_start_training
except ImportError:
    finalize_rl_boundary_collection_and_start_training = None  # type: ignore[misc,assignment]
try:
    from rl_q_learning_selector import normalize_coarse_network_scenario
except ImportError:
    normalize_coarse_network_scenario = None  # type: ignore[misc,assignment]
from dynamic_network_controller import DynamicNetworkController

# q_learning_logger lives in scripts/utilities (Docker: /app/scripts/utilities, local: project_root/scripts/utilities)
if os.path.exists('/app'):
    _project_root = '/app'
else:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
_utilities_path = os.path.join(_project_root, 'scripts', 'utilities')
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)
try:
    from q_learning_logger import init_db as init_qlearning_db, log_q_step, rl_state_network_kwargs
except ImportError:
    init_qlearning_db = None
    log_q_step = None

    def rl_state_network_kwargs(_state=None):
        return {}
from client_fl_metrics_log import append_client_fl_metrics_record, use_case_from_env

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# Environment variables
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
USE_RL_SELECTION = os.getenv("USE_RL_SELECTION", "true").lower() == "true"
USE_QL_CONVERGENCE = os.getenv("USE_QL_CONVERGENCE", "false").lower() == "true"
_ue = os.getenv("USE_RL_EXPLORATION", "").strip().lower()
if _ue in ("true", "1", "yes"):
    USE_RL_EXPLORATION = True
elif _ue in ("false", "0", "no"):
    USE_RL_EXPLORATION = False
else:
    USE_RL_EXPLORATION = USE_QL_CONVERGENCE
USE_COMMUNICATION_MODEL_REWARD = os.getenv("USE_COMMUNICATION_MODEL_REWARD", "true").lower() == "true"

Q_CONVERGENCE_THRESHOLD = float(os.getenv("Q_CONVERGENCE_THRESHOLD", "0.01"))
Q_CONVERGENCE_PATIENCE = int(os.getenv("Q_CONVERGENCE_PATIENCE", "5"))

# --- Goal-aware convergence (mirrored from Emotion Recognition) ---
# Loss-based local convergence (fallback when USE_QL_CONVERGENCE is off)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
# Enable loss-based local convergence stop (goal: low loss = good model quality)
ENABLE_LOCAL_CONVERGENCE_STOP = os.getenv("ENABLE_LOCAL_CONVERGENCE_STOP", "false").lower() == "true"


def stop_on_client_convergence() -> bool:
    """
    Controls whether the client should stop on local/RL convergence.
    Goals: low comm time, low battery consumption, high resource availability.
    TRAINING_TERMINATION_MODE (fixed_rounds | client_convergence) overrides
    STOP_ON_CLIENT_CONVERGENCE when set.
    """
    mode = (os.getenv("TRAINING_TERMINATION_MODE") or "").strip().lower()
    if mode == "fixed_rounds":
        return False
    if mode == "client_convergence":
        return True
    v = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").strip().lower()
    return v in ("1", "true", "yes")


RL_PHASE0_ROUNDS = int(os.getenv("RL_PHASE0_ROUNDS", "20"))
RL_BOUNDARY_PIPELINE = os.getenv("RL_BOUNDARY_PIPELINE", "true").lower() in ("1", "true", "yes")

_num_rounds_env_rl = os.getenv("NUM_ROUNDS", "").strip()
if _num_rounds_env_rl and RL_BOUNDARY_PIPELINE and RL_PHASE0_ROUNDS > 0:
    try:
        _nr = int(_num_rounds_env_rl)
        if _nr < RL_PHASE0_ROUNDS:
            print(
                f"[RL] Warning: NUM_ROUNDS={_nr} < RL_PHASE0_ROUNDS={RL_PHASE0_ROUNDS}. "
                f"Phase 1 (boundary data collection) will not complete before FL stops; "
                f"raise NUM_ROUNDS to at least {RL_PHASE0_ROUNDS} (plus rounds for Phase 3), "
                f"or lower RL_PHASE0_ROUNDS."
            )
    except ValueError:
        pass


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y")


RL_INFERENCE_ONLY = _env_truthy("RL_INFERENCE_ONLY")
_RL_PROTOCOL_SELECTION_RECORD_LEARNING = not RL_INFERENCE_ONLY

TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))
GRPC_CHUNK_SIZE = max(262144, GRPC_MAX_MESSAGE_BYTES - 4096)


def _amqp_use_case_tag() -> str:
    raw = (os.getenv("USE_CASE") or os.getenv("CLIENT_USE_CASE") or "temperature").strip().lower()
    return raw.replace(" ", "_").replace("-", "_") if raw else "temperature"


_AMQP_UC = _amqp_use_case_tag()
AMQP_MODEL_UPDATE_ROUTING_KEY = os.getenv(
    "AMQP_MODEL_UPDATE_ROUTING_KEY", f"client.update.{_AMQP_UC}"
)
AMQP_MODEL_METRICS_ROUTING_KEY = os.getenv(
    "AMQP_MODEL_METRICS_ROUTING_KEY", f"client.metrics.{_AMQP_UC}"
)


def _resolve_unified_grpc_host(default_host: str = "fl-server-unified-temperature") -> str:
    """
    Unified compose files set GRPC_SERVER to the real service name (e.g. fl-server-unified-temperature).
    Client/Dockerfile also sets GRPC_HOST=fl-server, which breaks DNS if compose does not override it.
    Prefer GRPC_SERVER so compose wins over the image default.
    """
    gs = (os.getenv("GRPC_SERVER") or "").strip()
    if gs:
        return gs.split(":", 1)[0].strip()
    gh = (os.getenv("GRPC_HOST") or "").strip()
    if gh:
        return gh.split(":", 1)[0].strip()
    uf = (os.getenv("UNIFIED_FL_SERVER_HOST") or "").strip()
    if uf:
        return uf.split(":", 1)[0].strip()
    return default_host


def _coarse_network_bucket_for_scenario(name: str) -> str:
    """Map simulator-style scenario names onto RL ``network`` buckets (env manager / logs)."""
    if normalize_coarse_network_scenario is not None:
        return normalize_coarse_network_scenario(name)
    n = (name or "").strip().lower()
    if n in ("excellent", "good"):
        return "excellent"
    if n in ("moderate", "congested_light", "dynamic"):
        return "moderate"
    if n in ("poor", "very_poor", "satellite", "congested_moderate", "congested_heavy"):
        return "poor"
    return "moderate"


def _effective_rl_network_scenario_label_from_env() -> Optional[str]:
    """
    Explicit label for Q-table ``network_scenario`` / SQLite state_network_scenario columns.
    Priority: RL_REWARD_SCENARIO, RL_STATE_NETWORK_SCENARIO, NETWORK_SCENARIO, SIMULATOR_SCENARIO.
    """
    rs = os.environ.get("RL_REWARD_SCENARIO", "").strip().lower()
    if rs:
        return rs
    for key in ("RL_STATE_NETWORK_SCENARIO", "NETWORK_SCENARIO", "SIMULATOR_SCENARIO"):
        v = os.environ.get(key, "").strip().lower()
        if v:
            return v
    return None


def _read_scenario_from_reset_epsilon_flag_file() -> Optional[str]:
    """Orchestrator / GUI: ``scenario=`` line in ``reset_epsilon_flag.txt`` (when hint file absent)."""
    flag_paths = (
        "/shared_data/reset_epsilon_flag.txt",
        "./shared_data/reset_epsilon_flag.txt",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "shared_data",
            "reset_epsilon_flag.txt",
        ),
    )
    for check_path in flag_paths:
        if not check_path or not os.path.isfile(check_path):
            continue
        try:
            with open(check_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("scenario="):
                        return line.split("=", 1)[1].strip().lower()
        except OSError:
            continue
    return None


class UnifiedFLClient_Temperature:
    """
    Unified Federated Learning Client for Temperature Regulation
    Integrates all 5 protocols with RL-based selection
    """
    
    def __init__(self, client_id: int, num_clients: int, dataframe: pd.DataFrame):
        """
        Initialize Unified FL Client
        
        Args:
            client_id: Unique client identifier
            num_clients: Total number of clients in FL
            dataframe: Temperature regulation dataset
        """
        self.client_id = client_id
        self.num_clients = num_clients
        
        # Process data
        self.dataframe = dataframe
        self.X_train, self.y_train = self._prepare_data()
        
        # Model
        self.model = None
        self.local_epochs = 20
        self.batch_size = 16

        # RL boundary pipeline Phase 1 state (aligned with emotion unified client)
        self._rl_boundary_collection_phase = False
        self._rl_boundary_comm_samples: List[float] = []
        self._rl_boundary_downlink_comm_samples: List[float] = []
        self._rl_boundary_res_samples: List[float] = []
        self._rl_boundary_batt_samples: List[float] = []
        self._rl_boundary_res_samples_dl: List[float] = []
        self._rl_boundary_batt_samples_dl: List[float] = []
        self._last_downlink_comm_wall_s: float = 0.0
        self._last_uplink_rl_state: Optional[Dict] = None
        self._temperature_rl_converged = False
        # Loss-based local convergence tracking (goal: low loss → good model quality)
        self.best_loss: float = float("inf")
        self.rounds_without_improvement: int = 0
        self.has_converged: bool = False
        self._rl_tracked_network_scenario_key: Optional[str] = None
        # Rolling latency samples for mobility (same idea as emotion unified client)
        self.latency_history: List[float] = []
        
        # RL Components: two SEPARATE agents for UPLINK and DOWNLINK, each with own Q-table
        if USE_RL_SELECTION:
            # --- Uplink agent ---
            if os.path.exists("/shared_data"):
                save_path_uplink = "/shared_data/q_table_temperature_uplink_trained.pkl"
            else:
                save_path_uplink = f"q_table_temperature_uplink_client_{client_id}.pkl"
            initial_load_path_uplink = None
            if os.path.exists("/shared_data"):
                for _ul_cand in ("/shared_data/q_table_temperature_uplink_trained.pkl",
                                 "/shared_data/q_table_temperature_trained.pkl"):
                    if os.path.exists(_ul_cand):
                        initial_load_path_uplink = _ul_cand
                        break
            if initial_load_path_uplink is None:
                pretrained_dir = os.getenv("PRETRAINED_Q_TABLE_DIR")
                if pretrained_dir:
                    for candidate in (
                        os.path.join(pretrained_dir, "q_table_temperature_uplink_trained.pkl"),
                        os.path.join(pretrained_dir, "q_table_temperature_trained.pkl"),
                        os.path.join(pretrained_dir, f"q_table_temperature_uplink_client_{client_id}.pkl"),
                        os.path.join(pretrained_dir, f"q_table_temperature_client_{client_id}.pkl"),
                    ):
                        if os.path.exists(candidate):
                            initial_load_path_uplink = candidate
                            break
            self.rl_selector_uplink = QLearningProtocolSelector(
                save_path=save_path_uplink,
                initial_load_path=initial_load_path_uplink,
                use_communication_model_reward=USE_COMMUNICATION_MODEL_REWARD,
                link_role="uplink",
            )
            # --- Downlink agent ---
            if os.path.exists("/shared_data"):
                save_path_downlink = "/shared_data/q_table_temperature_downlink_trained.pkl"
            else:
                save_path_downlink = f"q_table_temperature_downlink_client_{client_id}.pkl"
            initial_load_path_downlink = None
            if os.path.exists("/shared_data") and os.path.exists("/shared_data/q_table_temperature_downlink_trained.pkl"):
                initial_load_path_downlink = "/shared_data/q_table_temperature_downlink_trained.pkl"
            if initial_load_path_downlink is None:
                pretrained_dir = os.getenv("PRETRAINED_Q_TABLE_DIR")
                if pretrained_dir:
                    for candidate in (
                        os.path.join(pretrained_dir, "q_table_temperature_downlink_trained.pkl"),
                        os.path.join(pretrained_dir, f"q_table_temperature_downlink_client_{client_id}.pkl"),
                    ):
                        if os.path.exists(candidate):
                            initial_load_path_downlink = candidate
                            break
            self.rl_selector_downlink = QLearningProtocolSelector(
                save_path=save_path_downlink,
                initial_load_path=initial_load_path_downlink,
                use_communication_model_reward=USE_COMMUNICATION_MODEL_REWARD,
                link_role="downlink",
            )
            # Backward-compat alias
            self.rl_selector = self.rl_selector_uplink

            # reset_epsilon flag / fresh training (aligned with emotion unified client)
            should_reset = False
            reset_flag_file = None
            scenario_info = None
            experiment_id = None
            reset_epsilon_value = None
            if os.getenv("RESET_EPSILON", "false").lower() == "true":
                should_reset = True
                print(f"[Client {client_id}] RESET_EPSILON environment variable detected")
                os.environ["RESET_EPSILON"] = "false"
            flag_paths = [
                "/shared_data/reset_epsilon_flag.txt",
                "./shared_data/reset_epsilon_flag.txt",
                os.path.join(os.path.dirname(__file__), "..", "..", "shared_data", "reset_epsilon_flag.txt"),
            ]
            for check_path in flag_paths:
                if os.path.exists(check_path):
                    reset_flag_file = check_path
                    try:
                        with open(check_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        for line in content.split("\n"):
                            if line.startswith("scenario="):
                                scenario_info = line.split("=", 1)[1].strip()
                            elif line.startswith("experiment_id="):
                                experiment_id = line.split("=", 1)[1].strip()
                            elif line.startswith("reset_epsilon="):
                                reset_epsilon_value = line.split("=", 1)[1].strip()
                        if reset_epsilon_value and (
                            reset_epsilon_value == "1.0" or reset_epsilon_value.lower() == "true"
                        ):
                            should_reset = True
                        else:
                            print(
                                f"[Client {client_id}] Flag file exists but reset_epsilon={reset_epsilon_value} "
                                f"(resume mode)"
                            )
                            should_reset = False
                    except OSError as e:
                        print(f"[Client {client_id}] Warning: Could not read reset flag file: {e}")
                    break
            if should_reset:
                last_experiment_id = getattr(self.rl_selector_uplink, "last_experiment_id", None)
                if experiment_id and experiment_id == last_experiment_id:
                    print(
                        f"[Client {client_id}] Already processed experiment {experiment_id!r}, skipping reset"
                    )
                    print(f"[Client {client_id}] Scenario: {scenario_info}")
                    should_reset = False
                else:
                    print(f"\n{'='*70}")
                    print(
                        f"[Client {client_id}] Fresh RL training: archive / zero scenario slice, "
                        f"epsilon=1.0 (uplink + downlink)"
                    )
                    if experiment_id:
                        print(f"[Client {client_id}]   New experiment ID: {experiment_id}")
                        if last_experiment_id:
                            print(f"[Client {client_id}]   Previous experiment ID: {last_experiment_id}")
                    if scenario_info:
                        print(f"[Client {client_id}]   Training scenario: {scenario_info}")
                    print(f"[Client {client_id}]   Uplink epsilon before reset: {self.rl_selector_uplink.epsilon:.4f}")
                    print(
                        f"[Client {client_id}]   Downlink epsilon before reset: "
                        f"{self.rl_selector_downlink.epsilon:.4f}"
                    )
                    uplink_arch = self.rl_selector_uplink.begin_fresh_training_for_scenario(
                        scenario_info, experiment_id=experiment_id
                    )
                    downlink_arch = self.rl_selector_downlink.begin_fresh_training_for_scenario(
                        scenario_info, experiment_id=experiment_id
                    )
                    print(f"[Client {client_id}]   Uplink epsilon: {self.rl_selector_uplink.epsilon:.4f}")
                    print(f"[Client {client_id}]   Downlink epsilon: {self.rl_selector_downlink.epsilon:.4f}")
                    if uplink_arch:
                        print(f"[Client {client_id}]   Uplink archive: {uplink_arch}")
                    if downlink_arch:
                        print(f"[Client {client_id}]   Downlink archive: {downlink_arch}")
                    print(f"{'='*70}\n")
                    try:
                        self.rl_selector_uplink.save_q_table()
                    except OSError as e:
                        print(f"[Client {client_id}] Warning: Could not save uplink Q-table after reset: {e}")
                    try:
                        self.rl_selector_downlink.save_q_table()
                    except OSError as e:
                        print(f"[Client {client_id}] Warning: Could not save downlink Q-table after reset: {e}")
            elif reset_flag_file and experiment_id:
                last_experiment_id = getattr(self.rl_selector_uplink, "last_experiment_id", None)
                if experiment_id != last_experiment_id:
                    print(f"\n{'='*70}")
                    print(f"[Client {client_id}] Continuing with previous epsilon (resume mode)")
                    if experiment_id:
                        print(f"[Client {client_id}]   Experiment ID: {experiment_id}")
                    if scenario_info:
                        print(f"[Client {client_id}]   Training scenario: {scenario_info}")
                    print(f"[Client {client_id}]   Uplink epsilon (preserved): {self.rl_selector_uplink.epsilon:.4f}")
                    print(
                        f"[Client {client_id}]   Downlink epsilon (preserved): "
                        f"{self.rl_selector_downlink.epsilon:.4f}"
                    )
                    print(f"{'='*70}\n")
                    self.rl_selector_uplink.last_experiment_id = experiment_id
                    self.rl_selector_downlink.last_experiment_id = experiment_id
                    if scenario_info:
                        _sn = scenario_info.strip().lower()
                        self.rl_selector_uplink.ensure_scenario(_sn)
                        self.rl_selector_downlink.ensure_scenario(_sn)
                    try:
                        self.rl_selector_uplink.save_q_table()
                    except OSError as e:
                        print(f"[Client {client_id}] Warning: Could not save uplink Q-table: {e}")
                    try:
                        self.rl_selector_downlink.save_q_table()
                    except OSError as e:
                        print(f"[Client {client_id}] Warning: Could not save downlink Q-table: {e}")

            self.env_manager = EnvironmentStateManager()
            self.env_manager.update_model_size("small")  # Temperature (small model)
            self.env_manager.sync_comm_thresholds_from_selector(self.rl_selector_uplink)
            if init_qlearning_db is not None:
                init_qlearning_db()
                try:
                    from q_learning_logger import get_db_path as _qldb_path
                    print(f"[Q-Learning] SQLite steps (uplink+downlink): {_qldb_path()}")
                except Exception:
                    pass
            self._rl_boundary_collection_phase = bool(
                RL_BOUNDARY_PIPELINE
                and RL_PHASE0_ROUNDS > 0
                and USE_RL_EXPLORATION
                and finalize_rl_boundary_collection_and_start_training is not None
            )
            if self._rl_boundary_collection_phase:
                self.rl_selector_uplink.epsilon = 1.0
                self.rl_selector_downlink.epsilon = 1.0
                print(
                    f"[Client {client_id}] RL pipeline: Phase 1 = first {RL_PHASE0_ROUNDS} FL rounds "
                    f"(uplink+downlink wall comm time, resource load, battery SoC; epsilon=1; no Q-updates). "
                    f"Then Phase 2 = boundaries, Phase 3 = Q-learning training."
                )
            # Hint file / reset flag / env — single place for SQLite ``state_*_network_scenario`` columns.
            self._sync_rl_network_scenario_round_start()
        else:
            self.rl_selector_uplink = None
            self.rl_selector_downlink = None
            self.rl_selector = None
            self.env_manager = None
        
        # Protocol handlers
        self.protocol_handlers = {
            'mqtt': self._handle_mqtt,
            'amqp': self._handle_amqp,
            'grpc': self._handle_grpc,
            'quic': self._handle_quic,
            'dds': self._handle_dds
        }
        
        # Metrics tracking
        self.round_metrics = {
            'communication_time': 0.0,
            'convergence_time': 0.0,
            'accuracy': 0.0,
            'success': False
        }
        self.current_round = 0
        self.selected_protocol = None
        self.selected_downlink_protocol = 'grpc'
        self.initial_global_model_downloaded = False
        self.last_protocol_query_key = None
        # Downlink RL tracking
        self._last_downlink_rl_state = None
        self._downlink_select_time = None
        self.grpc_host = _resolve_unified_grpc_host()
        self.grpc_port = int(os.getenv("GRPC_PORT", "50051"))
        self.grpc_registered = False
        
        print(f"\n{'='*70}")
        print(f"UNIFIED FL CLIENT - TEMPERATURE REGULATION")
        print(f"{'='*70}")
        print(f"Client ID: {self.client_id}/{self.num_clients}")
        print(f"gRPC (control / protocol negotiation): {self.grpc_host}:{self.grpc_port}")
        print(f"Training Samples: {len(self.y_train)}")
        print(f"RL Protocol Selection: {'ENABLED' if USE_RL_SELECTION else 'DISABLED'}")
        if USE_RL_SELECTION:
            print(f"RL Exploration (epsilon-greedy): {'ENABLED' if USE_RL_EXPLORATION else 'DISABLED (greedy)'}")
            print(f"Q-Convergence Stop: {'ENABLED' if USE_QL_CONVERGENCE else 'DISABLED'}")
        print(
            f"Local Convergence Stop (ENABLE_LOCAL_CONVERGENCE_STOP): "
            f"{'ENABLED' if ENABLE_LOCAL_CONVERGENCE_STOP else 'DISABLED'}"
        )
        print(
            f"Training Termination Mode: "
            f"{os.getenv('TRAINING_TERMINATION_MODE', 'not set (defaults to STOP_ON_CLIENT_CONVERGENCE)')}"
        )
        if USE_RL_SELECTION:
            print(f"Communication Model in Reward: {'ENABLED' if USE_COMMUNICATION_MODEL_REWARD else 'DISABLED'}")
        if USE_RL_SELECTION and USE_RL_EXPLORATION and self.rl_selector_uplink is not None:
            print(
                "RL: separate uplink/downlink Q-tables + pickles; uplink ε decays after each uplink Q-update; "
                "downlink ε decays after each downlink Q-update (server ProtocolQuery+gRPC model fetch, "
                "or gRPC CheckTrainingStatus fallback when the server does not send a query)."
            )
        if USE_RL_SELECTION and getattr(self, "_rl_boundary_collection_phase", False):
            print(
                f"RL boundary pipeline: ENABLED (Phase 1: {RL_PHASE0_ROUNDS} FL rounds collect metrics → "
                f"Phase 2: min/max tercile boundaries → Phase 3: Q-learning). "
                f"RL_BOUNDARY_PIPELINE={RL_BOUNDARY_PIPELINE}"
            )
        print(f"{'='*70}\n")
    
    def _rl_discrete_state_for_selector(self, selector, comm_time: float, cpu: float, memory: float):
        """Map measurements to Q-state, or a fixed placeholder during Phase 1 (no provisional thresholds)."""
        if getattr(self, "_rl_boundary_collection_phase", False) and self.env_manager:
            return self.env_manager.neutral_rl_state_before_boundaries()
        return self.env_manager.state_for_rl_selector(selector, comm_time, cpu, memory)

    def _rl_q_logging_allowed(self) -> bool:
        """False during Phase 1 boundary data collection — no q_learning_log writes for Q-steps."""
        return not getattr(self, "_rl_boundary_collection_phase", False)

    def _update_client_convergence_and_maybe_stop(self, loss: float) -> bool:
        """
        Track loss-based local convergence (goal: model quality converging).
        Returns True when convergence is detected and the caller should stop the FL loop.

        Goals encoded here mirror Emotion Recognition:
          - low communication time   → optimised by RL reward
          - low battery consumption  → optimised by RL reward
          - high resource availability → optimised by RL reward
          - stable model loss         → handled here (loss convergence)
        """
        if self.current_round < MIN_ROUNDS:
            self.best_loss = min(self.best_loss, float(loss))
            return False

        if self.best_loss - float(loss) > CONVERGENCE_THRESHOLD:
            self.best_loss = float(loss)
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

        if self.rounds_without_improvement >= CONVERGENCE_PATIENCE and not self.has_converged:
            self.has_converged = True
            print(
                f"[Client {self.client_id}] Local loss-convergence reached at round {self.current_round} "
                f"(goal: model quality stable; {CONVERGENCE_PATIENCE} rounds without >{CONVERGENCE_THRESHOLD:.4f} improvement)"
            )
            if stop_on_client_convergence():
                self._notify_rl_converged_grpc()
                return True
        return False

    def _read_current_rl_network_scenario_from_shared(self) -> Optional[str]:
        """GUI / orchestrator hint: ``shared_data/current_rl_network_scenario.txt`` (``scenario=name``)."""
        seen = set()
        for base in ("/shared_data", self._shared_data_dir()):
            if not base or base in seen or not os.path.isdir(base):
                continue
            seen.add(base)
            path = os.path.join(base, "current_rl_network_scenario.txt")
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.lower().startswith("scenario="):
                            return line.split("=", 1)[1].strip().lower()
                        return line.strip().lower()
            except OSError:
                continue
        return None

    def _resolved_rl_network_scenario_label(self) -> str:
        """
        Prefer ``current_rl_network_scenario.txt`` (GUI dynamic ticks / fine-grained label),
        else ``reset_epsilon_flag.txt`` scenario= (orchestrator), else process env.
        """
        fp = self._read_current_rl_network_scenario_from_shared()
        if fp:
            return fp.strip().lower() or "default"
        flag_scen = _read_scenario_from_reset_epsilon_flag_file()
        if flag_scen:
            return flag_scen.strip().lower() or "default"
        eff = _effective_rl_network_scenario_label_from_env()
        if eff:
            return eff.strip().lower() or "default"
        return "default"

    def _sync_rl_network_scenario_round_start(self) -> None:
        """
        Keep env + RL selectors aligned with the active network scenario; on label change,
        re-enter RL Phase 1 (``RL_PHASE0_ROUNDS``) like a fresh segment (emotion-style).
        """
        if not USE_RL_SELECTION or self.env_manager is None:
            return
        new_label = self._resolved_rl_network_scenario_label()
        prev = getattr(self, "_rl_tracked_network_scenario_key", None)
        if prev == new_label:
            return
        self.env_manager.update_network_scenario(
            new_label,
            rl_uplink=self.rl_selector_uplink,
            rl_downlink=self.rl_selector_downlink,
        )
        self.env_manager.update_network_condition(_coarse_network_bucket_for_scenario(new_label))
        if self.rl_selector_uplink is not None:
            self.rl_selector_uplink.ensure_scenario(new_label)
        if self.rl_selector_downlink is not None:
            self.rl_selector_downlink.ensure_scenario(new_label)
        if (
            prev is not None
            and prev != new_label
            and RL_BOUNDARY_PIPELINE
            and RL_PHASE0_ROUNDS > 0
            and USE_RL_EXPLORATION
            and finalize_rl_boundary_collection_and_start_training is not None
        ):
            if getattr(self, "_rl_boundary_collection_phase", False):
                # Already mid-Phase 1 — don't discard collected samples; just update
                # the scenario label so boundaries will be tagged with the final scenario.
                # Phase 1 will finalize normally after RL_PHASE0_ROUNDS uplink samples.
                print(
                    f"[Client {self.client_id}] Network scenario changed {prev!r} → {new_label!r} "
                    f"during Phase 1 (sample {len(self._rl_boundary_comm_samples)}/{RL_PHASE0_ROUNDS}): "
                    f"continuing Phase 1 uninterrupted (no reset — epsilon stays 1, no Q-updates)."
                )
            else:
                # Phase 3 is active — restart Phase 1 for the new scenario.
                self._rl_boundary_collection_phase = True
                self._rl_boundary_comm_samples = []
                self._rl_boundary_downlink_comm_samples = []
                self._rl_boundary_res_samples = []
                self._rl_boundary_batt_samples = []
                self._rl_boundary_res_samples_dl = []
                self._rl_boundary_batt_samples_dl = []
                self.rl_selector_uplink.epsilon = 1.0
                self.rl_selector_downlink.epsilon = 1.0
                self._temperature_rl_converged = False
                print(
                    f"[Client {self.client_id}] Network scenario changed {prev!r} → {new_label!r}: "
                    f"re-entering RL Phase 1 ({RL_PHASE0_ROUNDS} rounds, epsilon=1, no Q-updates until "
                    f"boundaries recomputed for this scenario)."
                )
        self._rl_tracked_network_scenario_key = new_label

    def _has_explicit_rl_network_scenario_config(self) -> bool:
        """True when experiment scenario is set via env, hint file, or reset flag (do not overwrite with RTT bucket)."""
        if _effective_rl_network_scenario_label_from_env():
            return True
        if self._read_current_rl_network_scenario_from_shared():
            return True
        if _read_scenario_from_reset_epsilon_flag_file():
            return True
        return False

    def measure_network_condition(self) -> None:
        """
        Client-measured network bucket (TCP RTT to MQTT broker → latency / bandwidth estimate).
        Feeds ``detected_network_scenario`` on env + RL selectors — same path as emotion
        ``measure_network_condition`` / ``state_detected_network_scenario`` in q_learning_log.
        """
        if not self.env_manager:
            return
        try:
            target_host = os.getenv("MQTT_BROKER", "mqtt-broker")
            latencies: List[float] = []
            port = int(os.getenv("MQTT_PORT", "1883"))
            for _ in range(3):
                start = time.time()
                try:
                    with socket.create_connection((target_host, port), timeout=2):
                        pass
                    latencies.append((time.time() - start) * 1000.0)
                except OSError:
                    latencies.append(500.0)

            latency_ms = sum(latencies) / len(latencies) if latencies else 300.0

            if latency_ms < 20:
                bandwidth_mbps = 100.0
            elif latency_ms < 50:
                bandwidth_mbps = 20.0
            elif latency_ms < 150:
                bandwidth_mbps = 5.0
            elif latency_ms < 400:
                bandwidth_mbps = 1.0
            else:
                bandwidth_mbps = 0.5

            condition = self.env_manager.detect_network_condition(latency_ms, bandwidth_mbps)
            self.env_manager.update_network_condition(condition)
            if USE_RL_SELECTION and self.rl_selector_uplink is not None:
                explicit_cfg = self._has_explicit_rl_network_scenario_config()
                if explicit_cfg:
                    # Inside Docker, container-to-container RTT is always < 20 ms regardless of
                    # tc netem / network scenario applied to the container's external interface.
                    # This means the RTT probe always classifies as "excellent", overriding the
                    # configured NETWORK_SCENARIO in training mode (USE_RL_EXPLORATION=true uses
                    # detected_network_scenario preferentially via RL_Q_USE_DETECTED_NETWORK).
                    # Fix: mirror the configured label into detected_network_scenario so that
                    # both training and inference paths use the correct Q-table scenario slice.
                    configured_label = self._resolved_rl_network_scenario_label()
                    self.env_manager.update_detected_network_scenario(
                        configured_label,
                        rl_uplink=self.rl_selector_uplink,
                        rl_downlink=self.rl_selector_downlink,
                    )
                    print(
                        f"[Network] Explicit NETWORK_SCENARIO configured → "
                        f"detected_network_scenario overridden to '{configured_label}' "
                        f"(RTT={latency_ms:.1f} ms raw bucket='{condition}' ignored for Q-state)"
                    )
                else:
                    # No explicit config — use RTT-derived bucket for detected scenario.
                    self.env_manager.update_detected_network_scenario(
                        condition,
                        rl_uplink=self.rl_selector_uplink,
                        rl_downlink=self.rl_selector_downlink,
                    )
                    self.env_manager.update_network_scenario(
                        condition,
                        rl_uplink=self.rl_selector_uplink,
                        rl_downlink=self.rl_selector_downlink,
                    )
            try:
                self.latency_history.append(latency_ms)
                if len(self.latency_history) > 20:
                    self.latency_history.pop(0)
                mobility = "static"
                if len(self.latency_history) >= 5:
                    avg = sum(self.latency_history) / len(self.latency_history)
                    variance = sum((x - avg) ** 2 for x in self.latency_history) / len(self.latency_history)
                    stddev = variance ** 0.5
                    mobility = "static" if stddev < 5 else "mobile"
                self.env_manager.update_mobility(mobility)
            except Exception as e:
                print(f"[Mobility] Failed to update mobility level: {e}")

            print(
                f"[Network] latency={latency_ms:.1f} ms, "
                f"est_bandwidth={bandwidth_mbps:.1f} Mbps -> RTT-condition={condition}"
            )
        except Exception as e:
            print(f"[Network] Failed to measure network condition: {e}")

    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare temperature data for training (same partitioning as FL_Client_MQTT: 1-based CLIENT_ID)."""
        total_samples = len(self.dataframe)
        # CLIENT_ID is 1-based in Docker; map to 0-based partition index (matches other temperature clients).
        client_index = (self.client_id - 1) if self.client_id >= 1 else self.client_id
        client_index = int(client_index) % max(1, self.num_clients)
        samples_per_client = total_samples // self.num_clients
        start_idx = client_index * samples_per_client
        if client_index >= self.num_clients - 1:
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_client

        client_data = self.dataframe.iloc[start_idx:end_idx]

        _feat = ("Ambient_Temp", "Cabin_Temp", "Relative_Humidity", "Solar_Load")
        _tgt = "Set_temp"
        if all(c in client_data.columns for c in _feat) and _tgt in client_data.columns:
            X = client_data[list(_feat)].values
            y = client_data[_tgt].values
        elif "target_temperature" in client_data.columns:
            # Synthetic fallback from load_temperature_data()
            drop_cols = [c for c in ("target_temperature",) if c in client_data.columns]
            X = client_data.drop(columns=drop_cols, errors="ignore").values
            y = client_data["target_temperature"].values
        else:
            X = client_data.iloc[:, :-1].values
            y = client_data.iloc[:, -1].values

        print(f"[Data Preparation] Client {self.client_id}")
        print(f"  Total dataset: {total_samples} samples")
        print(f"  Client subset: {len(y)} samples (indices {start_idx} to {end_idx}, partition {client_index}/{self.num_clients})")
        print(f"  Features shape: {X.shape}")

        return X.astype(np.float32), y.astype(np.float32)
    
    def build_model(self, input_dim) -> keras.Model:
        """Build temperature regulation model (Dense network for regression)"""
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_dim=input_dim),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1)  # Regression output
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def select_protocol(self) -> str:
        """Select UPLINK protocol using the dedicated UPLINK RL agent (CPU-only Q-learning)"""
        if USE_RL_SELECTION and self.rl_selector_uplink and self.env_manager:
            try:
                # RL logic runs on CPU only (no GPU); keeps Q-table updates off GPU
                with tf.device('/CPU:0'):
                    import psutil
                    cpu = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory().percent

                    resource_level = self.env_manager.detect_resource_level(cpu, memory)
                    self.env_manager.update_resource_level(resource_level)

                    rm = self.round_metrics or {}
                    t_comm = float(
                        (rm.get("communication_time") or 0)
                        or (
                            float(rm.get("uplink_model_comm_time", 0) or 0)
                            + float(rm.get("uplink_metrics_comm_time", 0) or 0)
                        )
                    )
                    state = self._rl_discrete_state_for_selector(
                        self.rl_selector_uplink, t_comm, cpu, memory
                    )
                    self._last_uplink_rl_state = state
                    training_mode = USE_RL_EXPLORATION
                    if getattr(self, "_rl_boundary_collection_phase", False):
                        self.rl_selector_uplink.epsilon = 1.0
                    protocol = self.rl_selector_uplink.select_protocol(
                        state,
                        training=training_mode,
                        record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING,
                    )

                print(f"\n[Uplink RL Protocol Selection]")
                print(f"  CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
                if getattr(self, "_rl_boundary_collection_phase", False):
                    soc = float(self.env_manager.battery_soc) if self.env_manager else float("nan")
                    print(
                        f"  Phase 1 (boundary collection): raw uplink wall comm ≈ {t_comm:.3f}s, "
                        f"SoC={soc:.3f} — discrete levels are assigned only after all "
                        f"{RL_PHASE0_ROUNDS} rounds (min/max → thresholds)."
                    )
                    print(f"  Placeholder state for random exploration only: {state}")
                else:
                    print(f"  State: {state}")
                print(f"  Selected Protocol: {protocol.upper()}")
                print(f"  Round: {self.current_round}")
                print(f"  Uplink epsilon: {self.rl_selector_uplink.epsilon:.4f}\n")

                self.selected_protocol = protocol
                return protocol
            except Exception as e:
                print(f"[Uplink RL Selection] Error: {e}, falling back to MQTT")
                return 'mqtt'
        else:
            return os.getenv("DEFAULT_PROTOCOL", "mqtt").lower()

    def _shared_data_dir(self):
        if os.path.exists("/shared_data"):
            return "/shared_data"
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "shared_data")

    def _get_t_calc_for_reward(self, protocol: str, payload_bytes: int):
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Network_Simulation"))
            from communication_model import get_t_calc_for_reward
            from pathlib import Path
            path = Path(self._shared_data_dir()) / "iperf3_network_params.json"
            return get_t_calc_for_reward(protocol, payload_bytes, json_path=path)
        except Exception:
            return None

    def _update_downlink_rl_after_reception(self, round_num: int = 0):
        """Compute and log downlink Q-learning reward after global model is received."""
        if (not USE_RL_SELECTION
                or self.rl_selector_downlink is None
                or self._downlink_select_time is None
                or self._last_downlink_rl_state is None):
            return
        if not USE_RL_EXPLORATION:
            self._downlink_select_time = None
            self._last_downlink_rl_state = None
            return
        try:
            comm_time = time.time() - self._downlink_select_time
            self._downlink_select_time = None
            self._last_downlink_comm_wall_s = float(comm_time)

            if getattr(self, "_rl_boundary_collection_phase", False):
                import psutil
                cpu_m = psutil.cpu_percent(interval=0.0)
                mem_m = psutil.virtual_memory().percent
                resource_load_dl = (cpu_m + mem_m) / 2.0
                batt_dl = float(self.env_manager.battery_soc) if self.env_manager else 1.0
                self._rl_boundary_downlink_comm_samples.append(comm_time)
                self._rl_boundary_res_samples_dl.append(resource_load_dl)
                self._rl_boundary_batt_samples_dl.append(batt_dl)
                nd = len(self._rl_boundary_downlink_comm_samples)
                print(
                    f"[Client {self.client_id}] RL Phase 1 (downlink data collection): sample {nd} "
                    f"(wall comm={comm_time:.3f}s, load={resource_load_dl:.1f}%, SoC={batt_dl:.3f})"
                )
                self._last_downlink_rl_state = None
                return

            downlink_state = self._last_downlink_rl_state
            self._last_downlink_rl_state = None
            protocol = self.selected_downlink_protocol or 'grpc'
            # Pass protocol + payload bytes so resource penalty reflects actual protocol cost
            # (goal: favour protocols with low resource consumption)
            resources = (
                self.env_manager.get_resource_consumption(
                    protocol=protocol,
                    uplink_payload_bytes=int(self.round_metrics.get("payload_bytes") or 0) or None,
                    downlink_payload_bytes=getattr(self, "_downlink_payload_bytes", None),
                )
                if self.env_manager
                else {}
            )
            payload_bytes = int(self.round_metrics.get("payload_bytes") or 0) or (12 * 1024 * 1024)
            t_calc = self._get_t_calc_for_reward(protocol, payload_bytes) if USE_COMMUNICATION_MODEL_REWARD else None
            reward = self.rl_selector_downlink.calculate_reward(
                communication_time=comm_time,
                success=True,
                resource_consumption=resources,
                t_calc=t_calc,
            )
            # Propagate next state for proper Q-bootstrapping so the agent learns
            # long-horizon goal alignment (low comm time, low battery, high resource).
            if self.env_manager:
                self.env_manager.update_comm_level_from_time(
                    comm_time,
                    self.rl_selector_downlink.comm_t_low,
                    self.rl_selector_downlink.comm_t_high,
                )
                self.env_manager.sync_battery_level_from_soc(None)
                next_state_dl = self.env_manager.get_current_state()
            else:
                next_state_dl = None
            self.rl_selector_downlink.update_q_value(
                reward,
                next_state=next_state_dl,
                done=False if next_state_dl is not None else True,
            )
            q_delta = self.rl_selector_downlink.get_last_q_delta()
            q_value = self.rl_selector_downlink.get_last_q_value()
            avg_reward = (
                float(np.mean(self.rl_selector_downlink.total_rewards[-100:]))
                if self.rl_selector_downlink.total_rewards else 0.0
            )
            self.rl_selector_downlink.end_episode()
            q_converged = self.rl_selector_downlink.check_q_converged(
                threshold=Q_CONVERGENCE_THRESHOLD,
                patience=Q_CONVERGENCE_PATIENCE,
                state=downlink_state,
            )
            print(f"[Downlink RL] round={round_num} | protocol={protocol.upper()} | "
                  f"comm_time={comm_time:.3f}s | reward={reward:.2f} | "
                  f"epsilon={self.rl_selector_downlink.epsilon:.4f}")
            if log_q_step is not None and self._rl_q_logging_allowed():
                reward_details = self.rl_selector_downlink.get_last_reward_breakdown()
                log_q_step(
                    client_id=self.client_id,
                    round_num=round_num or int(getattr(self, 'current_round', 0)),
                    episode=self.rl_selector_downlink.episode_count - 1,
                    state_comm_level=downlink_state.get('comm_level', ''),
                    state_resource=downlink_state.get('resource', ''),
                    state_battery_level=downlink_state.get('battery_level', ''),
                    **rl_state_network_kwargs(downlink_state),
                    action=protocol,
                    reward=reward,
                    q_delta=q_delta,
                    epsilon=self.rl_selector_downlink.epsilon,
                    q_value=q_value,
                    avg_reward_last_100=avg_reward,
                    converged=q_converged,
                    metric_communication_time=reward_details.get('communication_time'),
                    metric_success=reward_details.get('success'),
                    metric_cpu_usage=reward_details.get('cpu_usage'),
                    metric_memory_usage=reward_details.get('memory_usage'),
                    metric_bandwidth_usage=reward_details.get('bandwidth_usage'),
                    metric_battery_level=reward_details.get('battery_level'),
                    metric_energy_usage=reward_details.get('energy_usage'),
                    reward_base=reward_details.get('reward_base'),
                    reward_communication_time=reward_details.get('reward_communication_time'),
                    reward_resource_penalty=reward_details.get('reward_resource_penalty'),
                    reward_battery_penalty=reward_details.get('reward_battery_penalty'),
                    reward_total=reward_details.get('reward_total'),
                    link_direction='downlink',
                )
        except Exception as e:
            print(f"[Downlink RL] reward update error: {e}")

    def _downlink_rl_grpc_control_fallback(self, round_num: int) -> None:
        """
        When the server never sends a ProtocolQuery, the downlink agent is never armed — no DB rows.
        Emotion always has a gRPC listener; here we run one downlink RL step on gRPC control RTT
        (CheckTrainingStatus) so the downlink Q-table and q_learning_log get updates.
        """
        if not USE_RL_SELECTION or not USE_RL_EXPLORATION:
            return
        if self.rl_selector_downlink is None or self.env_manager is None:
            return
        if not GRPC_PROTO_AVAILABLE:
            print("[Downlink RL] gRPC-control fallback skipped: protobuf stubs unavailable")
            return
        try:
            with tf.device('/CPU:0'):
                import psutil
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory().percent
                t_dl = float(getattr(self, "_last_downlink_comm_wall_s", 0.0) or 0.0)
                if t_dl <= 0.0:
                    rm = self.round_metrics or {}
                    t_dl = float(rm.get("communication_time", 0) or 0.0)
                st = self._rl_discrete_state_for_selector(
                    self.rl_selector_downlink, t_dl, cpu, memory
                )
                if getattr(self, "_rl_boundary_collection_phase", False):
                    self.rl_selector_downlink.epsilon = 1.0
                proto = self.rl_selector_downlink.select_protocol(
                    st,
                    training=USE_RL_EXPLORATION,
                    record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING,
                )
            self.selected_downlink_protocol = proto
            t0 = time.time()
            ok = False
            channel, stub = self._open_grpc_stub()
            if stub is not None:
                try:
                    stub.CheckTrainingStatus(
                        federated_learning_pb2.StatusRequest(
                            client_id=self.client_id, current_round=round_num
                        )
                    )
                    ok = True
                except Exception:
                    ok = False
                finally:
                    channel.close()
            comm = time.time() - t0
            self._last_downlink_comm_wall_s = float(comm)

            if getattr(self, "_rl_boundary_collection_phase", False):
                import psutil
                cpu_m = psutil.cpu_percent(interval=0.0)
                mem_m = psutil.virtual_memory().percent
                resource_load_dl = (cpu_m + mem_m) / 2.0
                batt_dl = float(self.env_manager.battery_soc) if self.env_manager else 1.0
                self._rl_boundary_downlink_comm_samples.append(comm)
                self._rl_boundary_res_samples_dl.append(resource_load_dl)
                self._rl_boundary_batt_samples_dl.append(batt_dl)
                nd = len(self._rl_boundary_downlink_comm_samples)
                print(
                    f"[Client {self.client_id}] RL Phase 1 (downlink gRPC fallback): sample {nd} "
                    f"(wall comm={comm:.3f}s, load={resource_load_dl:.1f}%, SoC={batt_dl:.3f})"
                )
                return

            pb = int(self.round_metrics.get("payload_bytes", 0) or 0) or (12 * 1024 * 1024)
            # Pass protocol + payload so resource penalty reflects actual protocol cost
            resources = (
                self.env_manager.get_resource_consumption(
                    protocol=proto,
                    uplink_payload_bytes=pb,
                    downlink_payload_bytes=getattr(self, "_downlink_payload_bytes", None),
                )
                if self.env_manager
                else {}
            )
            t_calc = self._get_t_calc_for_reward(proto, pb) if USE_COMMUNICATION_MODEL_REWARD else None
            reward = self.rl_selector_downlink.calculate_reward(
                communication_time=comm,
                success=ok,
                resource_consumption=resources,
                t_calc=t_calc,
            )
            # Propagate next state for proper Q-bootstrapping (goal: long-horizon optimisation)
            if self.env_manager:
                self.env_manager.update_comm_level_from_time(
                    comm,
                    self.rl_selector_downlink.comm_t_low,
                    self.rl_selector_downlink.comm_t_high,
                )
                self.env_manager.sync_battery_level_from_soc(None)
                _next_st_grpc = self.env_manager.get_current_state()
            else:
                _next_st_grpc = None
            self.rl_selector_downlink.update_q_value(
                reward,
                next_state=_next_st_grpc,
                done=False if _next_st_grpc is not None else True,
            )
            q_delta = self.rl_selector_downlink.get_last_q_delta()
            q_value = self.rl_selector_downlink.get_last_q_value()
            avg_reward = (
                float(np.mean(self.rl_selector_downlink.total_rewards[-100:]))
                if self.rl_selector_downlink.total_rewards
                else 0.0
            )
            self.rl_selector_downlink.end_episode()
            q_converged = self.rl_selector_downlink.check_q_converged(
                threshold=Q_CONVERGENCE_THRESHOLD,
                patience=Q_CONVERGENCE_PATIENCE,
                state=st,
            )
            print(
                f"[Downlink RL] gRPC-control fallback round={round_num} | protocol={proto.upper()} | "
                f"comm={comm:.4f}s | ok={ok} | reward={reward:.2f} | ε={self.rl_selector_downlink.epsilon:.4f}"
            )
            if log_q_step is not None and self._rl_q_logging_allowed():
                reward_details = self.rl_selector_downlink.get_last_reward_breakdown()
                log_q_step(
                    client_id=self.client_id,
                    round_num=round_num,
                    episode=self.rl_selector_downlink.episode_count - 1,
                    state_comm_level=st.get("comm_level", ""),
                    state_resource=st.get("resource", ""),
                    state_battery_level=st.get("battery_level", ""),
                    **rl_state_network_kwargs(st),
                    action=proto,
                    reward=reward,
                    q_delta=q_delta,
                    epsilon=self.rl_selector_downlink.epsilon,
                    q_value=q_value,
                    avg_reward_last_100=avg_reward,
                    converged=q_converged,
                    metric_communication_time=reward_details.get("communication_time"),
                    metric_success=reward_details.get("success"),
                    metric_cpu_usage=reward_details.get("cpu_usage"),
                    metric_memory_usage=reward_details.get("memory_usage"),
                    metric_bandwidth_usage=reward_details.get("bandwidth_usage"),
                    metric_battery_level=reward_details.get("battery_level"),
                    metric_energy_usage=reward_details.get("energy_usage"),
                    reward_base=reward_details.get("reward_base"),
                    reward_communication_time=reward_details.get("reward_communication_time"),
                    reward_resource_penalty=reward_details.get("reward_resource_penalty"),
                    reward_battery_penalty=reward_details.get("reward_battery_penalty"),
                    reward_total=reward_details.get("reward_total"),
                    link_direction="downlink",
                )
        except Exception as e:
            print(f"[Downlink RL] gRPC-control fallback error: {e}")

    def _open_grpc_stub(self):
        if not GRPC_PROTO_AVAILABLE:
            return None, None
        options = [
            ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
            ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
        ]
        channel = grpc.insecure_channel(f"{self.grpc_host}:{self.grpc_port}", options=options)
        stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
        return channel, stub

    def _notify_rl_converged_grpc(self):
        """Send a zero-weight gRPC ModelUpdate with client_converged=1.0 to tell the server all RL phases are done."""
        if not GRPC_PROTO_AVAILABLE or federated_learning_pb2 is None:
            print(f"[Client {self.client_id}] gRPC protos unavailable; cannot send RL convergence notification.")
            return
        channel, stub = self._open_grpc_stub()
        if stub is None:
            return
        try:
            response = stub.SendModelUpdate(
                federated_learning_pb2.ModelUpdate(
                    client_id=self.client_id,
                    round=int(getattr(self, 'current_round', 0)) + 1,
                    weights=b"",
                    num_samples=0,
                    metrics={"client_converged": 1.0},
                )
            )
            if response.success:
                print(f"[Client {self.client_id}] RL convergence notification sent to server.")
            else:
                print(f"[Client {self.client_id}] RL convergence notification: server response={response.message}")
        except Exception as e:
            print(f"[Client {self.client_id}] Failed to send RL convergence notification: {e}")
        finally:
            try:
                channel.close()
            except Exception:
                pass

    def _fetch_server_fl_round_for_uplink(self) -> Optional[int]:
        """Server ``TrainingStatus.current_round`` is the FL round expecting this client's model update."""
        if not GRPC_PROTO_AVAILABLE or federated_learning_pb2 is None:
            return None
        channel, stub = self._open_grpc_stub()
        if stub is None:
            return None
        try:
            status = stub.CheckTrainingStatus(
                federated_learning_pb2.StatusRequest(
                    client_id=self.client_id,
                    current_round=int(getattr(self, "current_round", 0)),
                )
            )
            # If the server is complete, stop polling for rounds.
            if bool(getattr(status, "is_complete", False)):
                return None
            r = int(status.current_round)
            return r if r > 0 else None
        except Exception as e:
            print(f"[gRPC] CheckTrainingStatus (uplink round sync) failed: {e}")
            return None
        finally:
            try:
                channel.close()
            except Exception:
                pass

    def register_with_server_grpc(self) -> bool:
        """Register with unified server so it runs ``start_training()`` (sets ``current_round``). Matches emotion client."""
        if self.grpc_registered:
            return True
        if not GRPC_PROTO_AVAILABLE or federated_learning_pb2 is None or federated_learning_pb2_grpc is None:
            print(f"[gRPC] Client {self.client_id} registration skipped: protobuf stubs unavailable")
            return False
        try:
            options = [
                ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
                ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
                ("grpc.keepalive_time_ms", 600000),
                ("grpc.keepalive_timeout_ms", 60000),
            ]
            channel = grpc.insecure_channel(
                f"{self.grpc_host}:{self.grpc_port}", options=options
            )
            stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
            response = stub.RegisterClient(
                federated_learning_pb2.ClientRegistration(client_id=self.client_id)
            )
            channel.close()
            if response.success:
                self.grpc_registered = True
                print(f"[gRPC] Client {self.client_id} registered with unified server successfully")
                return True
            print(f"[gRPC] Registration failed: {response.message}")
            return False
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} registration error: {e}")
            return False

    def _select_downlink_protocol(self, round_id: int, global_model_id: int) -> str:
        # First global model bootstrap is always forced to gRPC.
        if round_id <= 1 or global_model_id <= 1 or not self.initial_global_model_downloaded:
            return 'grpc'
        # Use the dedicated DOWNLINK RL agent
        if USE_RL_SELECTION and self.rl_selector_downlink and self.env_manager:
            try:
                with tf.device('/CPU:0'):
                    import psutil
                    cpu = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory().percent
                    t_dl = float(getattr(self, "_last_downlink_comm_wall_s", 0.0) or 0.0)
                    if t_dl <= 0.0:
                        rm = self.round_metrics or {}
                        t_dl = float(rm.get("communication_time", 0) or 0.0)
                    state = self._rl_discrete_state_for_selector(
                        self.rl_selector_downlink, t_dl, cpu, memory
                    )
                    if getattr(self, "_rl_boundary_collection_phase", False):
                        self.rl_selector_downlink.epsilon = 1.0
                    selected = self.rl_selector_downlink.select_protocol(
                        state, training=USE_RL_EXPLORATION, record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING
                    )
                self._last_downlink_rl_state = state
                self._downlink_select_time = time.time()
                print(f"[Downlink RL] selected {selected.upper()} "
                      f"(epsilon={self.rl_selector_downlink.epsilon:.4f})")
            except Exception as e:
                print(f"[Downlink RL] Error: {e}, using default")
                selected = self.select_protocol()
                self._last_downlink_rl_state = None
                self._downlink_select_time = None
        else:
            selected = self.select_protocol()
            self._last_downlink_rl_state = None
            self._downlink_select_time = None
        return selected if selected in {'mqtt', 'amqp', 'grpc', 'quic', 'http3', 'dds'} else 'grpc'

    def _poll_and_respond_protocol_query_via_grpc(self):
        channel, stub = self._open_grpc_stub()
        if stub is None:
            return
        try:
            status = stub.CheckTrainingStatus(
                federated_learning_pb2.StatusRequest(client_id=self.client_id, current_round=self.current_round)
            )
            if not getattr(status, 'has_protocol_query', False):
                return
            query = status.protocol_query
            query_key = (int(query.round_id), int(query.global_model_id))
            if self.last_protocol_query_key == query_key:
                return
            selected = self._select_downlink_protocol(query.round_id, query.global_model_id)
            response = stub.SendProtocolSelection(
                federated_learning_pb2.ProtocolSelection(
                    client_id=self.client_id,
                    round_id=int(query.round_id),
                    global_model_id=int(query.global_model_id),
                    downlink_protocol_requested=selected,
                )
            )
            if response.success:
                self.selected_downlink_protocol = selected
                self.last_protocol_query_key = query_key
                # Real downlink reward is computed in _update_downlink_rl_after_reception()
                # after global model arrives. We do NOT log reward=0.0 here.
                _dl_eps = (
                    f"{self.rl_selector_downlink.epsilon:.4f}"
                    if self.rl_selector_downlink is not None
                    else "N/A"
                )
                print(
                    f"[gRPC] Client {self.client_id} protocol selection sent: "
                    f"round={query.round_id}, global_model_id={query.global_model_id}, downlink={selected} "
                    f"(downlink RL epsilon={_dl_eps})"
                )
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} protocol query handling failed: {e}")
        finally:
            channel.close()

    def _ensure_initial_global_model_via_grpc(self):
        if self.initial_global_model_downloaded:
            return
        channel, stub = self._open_grpc_stub()
        if stub is None:
            return
        try:
            response = stub.GetGlobalModel(
                federated_learning_pb2.ModelRequest(client_id=self.client_id, round=0, chunk_index=0)
            )
            if response.available and response.weights:
                # Apply the server's aggregated global weights to the local model.
                # Server sends pickle.dumps(list_of_numpy_arrays).
                try:
                    self.set_model_weights(response.weights)
                    print(f"[gRPC] Client {self.client_id} downloaded and applied global model via gRPC")
                except Exception as e_set:
                    print(f"[gRPC] Client {self.client_id} could not apply global model weights: {e_set}")
                self.initial_global_model_downloaded = True
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} initial global model download failed: {e}")
        finally:
            channel.close()

    def train_local_model(self) -> Dict:
        """Train model locally using real temperature data"""
        if self.model is None:
            input_dim = self.X_train.shape[1]
            self.model = self.build_model(input_dim)
        
        start_time = time.time()
        
        # Train with GPU acceleration (validation_split needs enough samples; same idea as emotion FL clients)
        n_samples = len(self.y_train)
        fit_kwargs = {
            "epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "verbose": 0,
        }
        if n_samples >= 5:
            fit_kwargs["validation_split"] = 0.2

        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            history = self.model.fit(self.X_train, self.y_train, **fit_kwargs)

        training_time = time.time() - start_time

        hist = history.history
        if "val_mae" in hist:
            val_mae = hist["val_mae"][-1]
            val_loss = hist["val_loss"][-1]
        else:
            val_mae = hist["mae"][-1]
            val_loss = hist["loss"][-1]
        
        metrics = {
            'training_time': training_time,
            'val_accuracy': 1.0 / (1.0 + val_mae),  # Convert MAE to accuracy-like metric
            'val_loss': val_loss,
            'val_mae': val_mae
        }
        
        print(f"[Training] Time: {training_time:.2f}s, MAE: {val_mae:.4f}")
        
        return metrics
    
    def get_model_weights(self) -> bytes:
        """Get serialized model weights"""
        weights = self.model.get_weights()
        return pickle.dumps(weights)
    
    def set_model_weights(self, weights_bytes: bytes):
        """Set model weights from serialized bytes"""
        if self.model is None:
            input_dim = self.X_train.shape[1]
            self.model = self.build_model(input_dim)
        weights = pickle.loads(weights_bytes)
        self.model.set_weights(weights)

    def _uplink_send_model_update(self, protocol: str, train_metrics: Dict) -> Tuple[bool, float]:
        """Serialize weights and send on the RL-selected uplink protocol; returns (ok, seconds).

        MQTT/AMQP use JSON + base64 weights like ``FL_Client_Unified`` (emotion). gRPC sends raw
        pickle bytes in ``ModelUpdate.weights``; the server wraps them into the same handler shape.
        """
        p = (protocol or "mqtt").lower()
        try:
            weights_bytes = self.get_model_weights()
        except Exception as e:
            print(f"[Uplink] Failed to serialize weights: {e}")
            return False, 0.0
        self.round_metrics["payload_bytes"] = len(weights_bytes)
        local_guess = int(getattr(self, "current_round", 0)) + 1
        srv = self._fetch_server_fl_round_for_uplink()
        if srv is not None:
            if srv != local_guess:
                print(
                    f"[Uplink] Server expects FL round {srv} for this update "
                    f"(local loop index suggests {local_guess}); using {srv}."
                )
            server_round = srv
        else:
            server_round = local_guess
        num_samples = int(self.X_train.shape[0])
        tm = train_metrics or {}
        metrics = {
            "val_accuracy": float(tm.get("val_accuracy", 0.0)),
            "val_loss": float(tm.get("val_loss", 0.0)),
            "val_mae": float(tm.get("val_mae", 0.0)),
            "training_time": float(tm.get("training_time", 0.0)),
            "training_time_sec": float(tm.get("training_time", 0.0)),
        }
        message = {
            "client_id": self.client_id,
            "round": server_round,
            "weights": base64.b64encode(weights_bytes).decode("ascii"),
            "num_samples": num_samples,
            "metrics": metrics,
            "use_case": _AMQP_UC,
        }
        t0 = time.time()
        ok = False
        if p in ("mqtt", "amqp"):
            handler = self.protocol_handlers.get(p)
            if handler is None:
                print(f"[Uplink] Unknown protocol {p!r}")
                return False, 0.0
            body = json.dumps(message).encode("utf-8")
            ok, _ = handler("send", body)
        elif p == "grpc":
            ok = self._send_grpc_model_update(message, weights_bytes)
        elif p in ("quic", "http3", "dds"):
            # These transport protocols are not yet implemented for the temperature
            # unified client uplink.  Fall back to AMQP which is always available.
            print(f"[Uplink] {p.upper()} uplink not implemented; falling back to AMQP.")
            handler = self.protocol_handlers.get("amqp")
            if handler is not None:
                body = json.dumps(message).encode("utf-8")
                ok, _ = handler("send", body)
            else:
                print("[Uplink] AMQP fallback handler unavailable.")
                ok = False
        else:
            print(f"[Uplink] Warning: {p.upper()} uplink is not implemented for temperature unified client; update not sent.")
            ok = False
        elapsed = time.time() - t0
        if ok:
            print(
                f"[Uplink] Model update sent via {p.upper()} "
                f"({len(weights_bytes)} bytes, {elapsed:.4f}s)"
            )
        else:
            print(f"[Uplink] Model update failed via {p.upper()}")
        return bool(ok), float(elapsed) if ok else 0.0

    def _send_grpc_model_update(self, message: Dict, weights_bytes: bytes) -> bool:
        """Send pickled weight blob via gRPC (same wire shape as emotion unified client)."""
        if not GRPC_PROTO_AVAILABLE or federated_learning_pb2 is None:
            print("[gRPC] Protobuf stubs unavailable; cannot send model update.")
            return False
        channel, stub = self._open_grpc_stub()
        if stub is None:
            return False
        try:
            metrics_dict = {k: float(v) for k, v in (message.get("metrics") or {}).items()}
            num_samples = int(message.get("num_samples", 0))
            payload_size = len(weights_bytes)
            if payload_size > GRPC_CHUNK_SIZE:
                chunks = [
                    weights_bytes[i : i + GRPC_CHUNK_SIZE]
                    for i in range(0, payload_size, GRPC_CHUNK_SIZE)
                ]
                total_chunks = len(chunks)
                for i, chunk_data in enumerate(chunks):
                    req = federated_learning_pb2.ModelUpdate(
                        client_id=message["client_id"],
                        round=message["round"],
                        weights=chunk_data,
                        num_samples=num_samples if i == 0 else 0,
                        metrics=metrics_dict if i == 0 else {},
                        chunk_index=i,
                        total_chunks=total_chunks,
                    )
                    response = stub.SendModelUpdate(req)
                    if not response.success:
                        print(f"[gRPC] Chunk {i + 1}/{total_chunks} failed: {response.message}")
                        return False
            else:
                response = stub.SendModelUpdate(
                    federated_learning_pb2.ModelUpdate(
                        client_id=message["client_id"],
                        round=message["round"],
                        weights=weights_bytes,
                        num_samples=num_samples,
                        metrics=metrics_dict,
                    )
                )
                if not response.success:
                    print(f"[gRPC] SendModelUpdate failed: {response.message}")
                    return False
            return True
        except Exception as e:
            print(f"[gRPC] Model update send failed: {e}")
            return False
        finally:
            try:
                channel.close()
            except Exception:
                pass
    
    # Protocol handlers (simplified versions)
    def _handle_mqtt(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle MQTT protocol communication"""
        try:
            broker = os.getenv("MQTT_BROKER", "mqtt-broker")
            port = int(os.getenv("MQTT_PORT", "1883"))
            # paho-mqtt ≥2.0 requires an explicit CallbackAPIVersion.
            _cbv = getattr(mqtt_client, "CallbackAPIVersion", None)
            if _cbv is not None:
                client = mqtt_client.Client(_cbv.VERSION1, f"temperature_client_{self.client_id}")
            else:
                client = mqtt_client.Client(f"temperature_client_{self.client_id}")
            client.connect(broker, port)
            
            if action == "send":
                if isinstance(data, (bytes, bytearray)):
                    payload = bytes(data)
                else:
                    payload = str(data).encode("utf-8")
                client.publish(TOPIC_CLIENT_UPDATE, payload=payload, qos=1)
                client.disconnect()
                return True, None
            elif action == "receive":
                # Simplified receive logic
                client.disconnect()
                return True, data
        except Exception as e:
            print(f"[MQTT] Error: {e}")
            return False, None
    
    def _handle_amqp(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle AMQP protocol communication"""
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=os.getenv("AMQP_BROKER", "amqp-broker"))
            )
            channel = connection.channel()
            channel.exchange_declare(exchange='fl_client_updates', exchange_type='direct', durable=True)
            if action == "send":
                body = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
                channel.basic_publish(
                    exchange='fl_client_updates',
                    routing_key=AMQP_MODEL_UPDATE_ROUTING_KEY,
                    body=body,
                    properties=pika.BasicProperties(
                        delivery_mode=2,
                        content_type="application/json",
                    ),
                )
                connection.close()
                return True, None
            connection.close()
            return True, data
        except Exception as e:
            print(f"[AMQP] Error: {e}")
            return False, None
    
    def _handle_grpc(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Uplink uses ``_send_grpc_model_update`` from ``_uplink_send_model_update``; this stub is unused for send."""
        print("[gRPC] Protocol handler (uplink path uses _send_grpc_model_update)")
        return True, data
    
    def _handle_quic(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle QUIC protocol communication"""
        print("[QUIC] Protocol handler")
        return True, data
    
    def _handle_dds(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle DDS protocol communication"""
        print("[DDS] Protocol handler")
        return True, data
    
    def federated_learning_round(self, protocol: Optional[str] = None) -> Dict:
        """Execute one round of federated learning"""
        round_start = time.time()
        round_idx = int(getattr(self, "current_round", 0))
        self._sync_rl_network_scenario_round_start()
        if USE_RL_SELECTION and self.env_manager:
            self.measure_network_condition()

        # gRPC control plane: protocol query first (emotion order), then model fetch.
        self._poll_and_respond_protocol_query_via_grpc()
        # If server armed downlink RL, re-stamp timer so comm_time is this round's gRPC segment only.
        if self._last_downlink_rl_state is not None and self._downlink_select_time is not None:
            self._downlink_select_time = time.time()
        self._ensure_initial_global_model_via_grpc()
        # Downlink Q + SQLite: server ProtocolQuery path, or gRPC fallback when server never queries.
        if USE_RL_SELECTION and self.rl_selector_downlink and self.env_manager:
            if USE_RL_EXPLORATION:
                if self._last_downlink_rl_state is not None:
                    self._update_downlink_rl_after_reception(round_num=round_idx)
                else:
                    self._downlink_rl_grpc_control_fallback(round_num=round_idx)
            elif self._last_downlink_rl_state is not None or self._downlink_select_time is not None:
                self._downlink_select_time = None
                self._last_downlink_rl_state = None

        if protocol is None:
            protocol = self.select_protocol()
        
        print(f"\n{'='*70}")
        print(f"FL ROUND - Using {protocol.upper()} Protocol")
        print(f"{'='*70}")
        
        try:
            # Train locally
            print(f"[Training] Starting local training...")
            train_metrics = self.train_local_model()

            upload_ok, uplink_sec = self._uplink_send_model_update(protocol, train_metrics)
            self.round_metrics["communication_time"] = float(uplink_sec)
            
            # Update metrics
            round_time = time.time() - round_start
            self.round_metrics['convergence_time'] = train_metrics['training_time']
            self.round_metrics['accuracy'] = train_metrics['val_accuracy']
            self.round_metrics['success'] = upload_ok
            
            # Q-learning updates only while exploring (USE_RL_EXPLORATION); inference is greedy-only, frozen Q.
            if USE_RL_SELECTION and self.rl_selector_uplink and self.env_manager:
                if USE_RL_EXPLORATION:
                    skip_uplink_training = False
                    if (
                        getattr(self, "_rl_boundary_collection_phase", False)
                        and finalize_rl_boundary_collection_and_start_training is not None
                    ):
                        import psutil
                        cpu_m = psutil.cpu_percent(interval=0.0)
                        mem_m = psutil.virtual_memory().percent
                        resource_load = (cpu_m + mem_m) / 2.0
                        batt = float(self.env_manager.battery_soc)
                        comm_wall_actual = float(self.round_metrics.get("communication_time", 0.0))
                        self._rl_boundary_comm_samples.append(comm_wall_actual)
                        self._rl_boundary_res_samples.append(resource_load)
                        self._rl_boundary_batt_samples.append(batt)
                        n_r = len(self._rl_boundary_comm_samples)
                        print(
                            f"[Client {self.client_id}] RL Phase 1 (uplink data collection): sample {n_r}/{RL_PHASE0_ROUNDS} "
                            f"(wall comm={comm_wall_actual:.3f}s, mean load={resource_load:.1f}%, SoC={batt:.3f})"
                        )
                        if n_r >= RL_PHASE0_ROUNDS:
                            finalize_rl_boundary_collection_and_start_training(
                                self.rl_selector_uplink,
                                self.rl_selector_downlink,
                                self.env_manager,
                                self._rl_boundary_comm_samples,
                                self._rl_boundary_res_samples,
                                self._rl_boundary_batt_samples,
                                client_id=self.client_id,
                                downlink_comm_times=self._rl_boundary_downlink_comm_samples,
                                resource_loads_downlink=self._rl_boundary_res_samples_dl,
                                battery_socs_downlink=self._rl_boundary_batt_samples_dl,
                            )
                            self._rl_boundary_collection_phase = False
                            self._rl_boundary_comm_samples = []
                            self._rl_boundary_downlink_comm_samples = []
                            self._rl_boundary_res_samples = []
                            self._rl_boundary_batt_samples = []
                            self._rl_boundary_res_samples_dl = []
                            self._rl_boundary_batt_samples_dl = []
                            print(
                                f"[Client {self.client_id}] RL Phase 3: Q-learning training started "
                                f"(epsilon decay on episodes)."
                            )
                        skip_uplink_training = True

                    if not skip_uplink_training:
                        resources = self.env_manager.get_resource_consumption()
                        payload_bytes = self.round_metrics.get('payload_bytes', 12 * 1024 * 1024)
                        t_calc = self._get_t_calc_for_reward(protocol, payload_bytes) if USE_COMMUNICATION_MODEL_REWARD else None
                        reward = self.rl_selector_uplink.calculate_reward(
                            communication_time=self.round_metrics['communication_time'],
                            success=self.round_metrics['success'],
                            resource_consumption=resources,
                            t_calc=t_calc,
                        )
                        next_state = None
                        if self.env_manager:
                            self.env_manager.update_comm_level_from_time(
                                float(self.round_metrics["communication_time"]),
                                self.rl_selector_uplink.comm_t_low,
                                self.rl_selector_uplink.comm_t_high,
                            )
                            self.env_manager.sync_battery_level_from_soc(None)
                            next_state = self.env_manager.get_current_state()
                        self.rl_selector_uplink.update_q_value(
                            reward,
                            next_state=next_state,
                            done=False if next_state is not None else True,
                        )
                        self.rl_selector_uplink.end_episode()
                        q_uplink_converged = self.rl_selector_uplink.check_q_converged(
                            threshold=Q_CONVERGENCE_THRESHOLD,
                            patience=Q_CONVERGENCE_PATIENCE,
                            state=self._last_uplink_rl_state,
                        )
                        q_downlink_converged = (
                            self.rl_selector_downlink.check_q_converged(
                                threshold=Q_CONVERGENCE_THRESHOLD,
                                patience=Q_CONVERGENCE_PATIENCE,
                                state=self._last_downlink_rl_state,
                            )
                            if self.rl_selector_downlink is not None
                            else True
                        )
                        q_both_converged = q_uplink_converged and q_downlink_converged
                        if log_q_step is not None and self._rl_q_logging_allowed():
                            st = self.env_manager.get_current_state()
                            q_delta = self.rl_selector_uplink.get_last_q_delta()
                            q_value = self.rl_selector_uplink.get_last_q_value()
                            avg_reward = (np.mean(self.rl_selector_uplink.total_rewards[-100:])
                                         if self.rl_selector_uplink.total_rewards else 0.0)
                            reward_details = self.rl_selector_uplink.get_last_reward_breakdown()
                            log_q_step(
                                client_id=self.client_id,
                                round_num=int(getattr(self, 'current_round', 0)),
                                episode=self.rl_selector_uplink.episode_count - 1,
                                state_comm_level=st.get('comm_level', ''),
                                state_resource=st.get('resource', ''),
                                state_battery_level=st.get('battery_level', ''),
                                **rl_state_network_kwargs(st),
                                action=protocol,
                                reward=reward,
                                q_delta=q_delta,
                                epsilon=self.rl_selector_uplink.epsilon,
                                q_value=q_value,
                                avg_reward_last_100=float(avg_reward),
                                converged=(
                                    q_both_converged
                                    if USE_QL_CONVERGENCE
                                    else q_uplink_converged
                                ),
                                metric_communication_time=reward_details.get('communication_time'),
                                metric_success=reward_details.get('success'),
                                metric_cpu_usage=reward_details.get('cpu_usage'),
                                metric_memory_usage=reward_details.get('memory_usage'),
                                metric_bandwidth_usage=reward_details.get('bandwidth_usage'),
                                metric_battery_level=reward_details.get('battery_level'),
                                metric_energy_usage=reward_details.get('energy_usage'),
                                reward_base=reward_details.get('reward_base'),
                                reward_communication_time=reward_details.get('reward_communication_time'),
                                reward_resource_penalty=reward_details.get('reward_resource_penalty'),
                                reward_battery_penalty=reward_details.get('reward_battery_penalty'),
                                reward_total=reward_details.get('reward_total'),
                                link_direction='uplink',
                            )
                        print(f"[Uplink RL] Reward: {reward:.2f}, epsilon: {self.rl_selector_uplink.epsilon:.4f}")
                        if USE_QL_CONVERGENCE and q_uplink_converged and not q_downlink_converged:
                            print(
                                f"[Client {self.client_id}] Uplink Q converged this round but downlink has not; "
                                f"continuing."
                            )
                        # Goal-aware convergence: stop only when TRAINING_TERMINATION_MODE
                        # allows it (mirrors Emotion Recognition behaviour).  Goals encoded in
                        # the RL reward — low comm time, low battery, high resource availability.
                        if USE_QL_CONVERGENCE and q_both_converged and stop_on_client_convergence():
                            _streak = os.getenv("RL_CONVERGENCE_SAME_PROTOCOL_STREAK", "5")
                            print(
                                f"[Client {self.client_id}] RL Q convergence (uplink+downlink, "
                                f"same-protocol streak={_streak}): goals met — stopping FL loop."
                            )
                            try:
                                self.rl_selector_uplink.save_q_table()
                            except Exception as e:
                                print(f"[Client {self.client_id}] Warning: could not save uplink Q-table: {e}")
                            try:
                                if self.rl_selector_downlink is not None:
                                    self.rl_selector_downlink.save_q_table()
                            except Exception as e:
                                print(f"[Client {self.client_id}] Warning: could not save downlink Q-table: {e}")
                            self._temperature_rl_converged = True
            
            _batt_soc = (
                float(self.env_manager.battery_soc)
                if (USE_RL_SELECTION and self.env_manager)
                else 1.0
            )
            append_client_fl_metrics_record(
                self.client_id,
                {
                    "client_id": self.client_id,
                    "round": int(getattr(self, "current_round", 0)),
                    "loss": float(train_metrics.get("val_loss", 0.0)),
                    "accuracy": float(train_metrics.get("val_accuracy", 0.0)),
                    "val_mae": float(train_metrics.get("val_mae", 0.0)),
                    "training_time_sec": float(train_metrics.get("training_time", 0.0)),
                    "total_fl_wall_time_sec": float(round_time),
                    "uplink_model_comm_sec": float(self.round_metrics.get("communication_time", 0.0)),
                    "battery_energy_joules": 0.0,
                    "battery_soc_after": _batt_soc,
                },
                use_case=use_case_from_env("temperature"),
                protocol=str(protocol),
            )
            
            return {
                'protocol': protocol,
                'round_time': round_time,
                **self.round_metrics,
                **train_metrics
            }
        except Exception as e:
            print(f"[Error] FL Round failed: {e}")
            return {'protocol': protocol, 'success': False}
    
    def run(self, num_rounds: int = 10):
        """Run federated learning for multiple rounds"""
        print(f"\n{'='*70}")
        print(f"STARTING FEDERATED LEARNING - {num_rounds} ROUNDS")
        print(f"{'='*70}\n")
        if not self.register_with_server_grpc():
            print(
                f"[Client {self.client_id}] WARNING: gRPC registration failed — "
                f"server will stay at current_round=0 and will reject all model updates (wrong round)."
            )

        # IMPORTANT: block until the server actually starts training (current_round > 0).
        # Otherwise a single client can race ahead and publish updates that the server
        # must reject while it is still waiting for MIN_CLIENTS registrations.
        rounds_completed = 0
        last_round_sent: Optional[int] = None
        while rounds_completed < num_rounds:
            server_round = self._fetch_server_fl_round_for_uplink()
            if server_round is None:
                if rounds_completed == 0:
                    print(
                        f"[Client {self.client_id}] Waiting for server to start training "
                        f"(server current_round=0; likely waiting for MIN_CLIENTS)..."
                    )
                time.sleep(1.0)
                continue

            if last_round_sent == int(server_round):
                time.sleep(0.25)
                continue

            self.current_round = int(server_round)
            print(f"\n{'#'*70}")
            print(f"# ROUND {rounds_completed + 1}/{num_rounds} (server_round={self.current_round})")
            print(f"{'#'*70}\n")

            metrics = self.federated_learning_round()

            last_round_sent = int(server_round)
            rounds_completed += 1

            print(f"\n[Round {rounds_completed}] Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")

            if getattr(self, "_temperature_rl_converged", False):
                print(
                    f"\n[Client {self.client_id}] Stopping after round {rounds_completed}: "
                    f"USE_QL_CONVERGENCE and uplink+downlink protocol-selection convergence (goals met)."
                )
                self._notify_rl_converged_grpc()
                break

            # Loss-based local convergence stop (goal: stable model quality).
            # Mirrors Emotion Recognition's _update_client_convergence_and_maybe_disconnect.
            if ENABLE_LOCAL_CONVERGENCE_STOP and not self.has_converged:
                loss_val = float(metrics.get("val_loss", float("inf")))
                if self._update_client_convergence_and_maybe_stop(loss_val):
                    print(
                        f"\n[Client {self.client_id}] Stopping FL loop after round {rounds_completed}: "
                        f"local loss-convergence reached (ENABLE_LOCAL_CONVERGENCE_STOP=true)."
                    )
                    break
            
        if USE_RL_SELECTION and self.rl_selector_uplink:
            print(f"\n{'='*70}\nUPLINK Q-learning (separate Q-table / epsilon)\n{'='*70}")
            self.rl_selector_uplink.print_statistics()
        if USE_RL_SELECTION and self.rl_selector_downlink:
            print(f"\n{'='*70}\nDOWNLINK Q-learning (separate Q-table / epsilon)\n{'='*70}")
            self.rl_selector_downlink.print_statistics()


def load_temperature_data(client_id: int) -> pd.DataFrame:
    """
    Load temperature regulation dataset
    
    Args:
        client_id: Client identifier
        
    Returns:
        DataFrame with temperature data
    """
    # Detect environment and construct dataset path
    if os.path.exists('/app'):
        dataset_path = '/app/Client/Temperature_Regulation/Dataset/base_data_baseline_unique.csv'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        dataset_path = os.path.join(project_root, 'Client', 'Temperature_Regulation', 'Dataset', 'base_data_baseline_unique.csv')
    
    print(f"[Dataset] Loading from: {dataset_path}")
    
    try:
        dataframe = pd.read_csv(dataset_path)
        print(f"[Dataset] Loaded successfully")
        print(f"[Dataset] Total samples: {len(dataframe)}")
        print(f"[Dataset] Features: {dataframe.shape[1]}")
        return dataframe
    except FileNotFoundError:
        print(f"[Warning] Dataset not found at {dataset_path}")
        print(f"[Info] Creating synthetic temperature data for testing...")
        
        # Create synthetic data as fallback
        np.random.seed(client_id + 42)
        n_samples = 1000
        
        features = {
            'outside_temperature': np.random.uniform(0, 40, n_samples),
            'inside_temperature': np.random.uniform(15, 30, n_samples),
            'humidity': np.random.uniform(30, 80, n_samples),
            'time_of_day': np.random.uniform(0, 24, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'heating_status': np.random.randint(0, 2, n_samples),
        }
        
        target = (features['inside_temperature'] * 0.7 + 
                 features['outside_temperature'] * 0.2 + 
                 features['humidity'] * 0.1 + 
                 np.random.normal(0, 1, n_samples))
        
        dataframe = pd.DataFrame(features)
        dataframe['target_temperature'] = target
        
        print(f"[Dataset] Synthetic data created: {dataframe.shape}")
        return dataframe


def main():
    """Main function"""
    print(f"Unified FL Client - Temperature Regulation (Client {CLIENT_ID})")
    
    # Load real temperature dataset
    print(f"\n{'='*70}")
    print("LOADING TEMPERATURE REGULATION DATASET")
    print(f"{'='*70}")
    
    dataframe = load_temperature_data(CLIENT_ID)
    
    # Create client
    client = UnifiedFLClient_Temperature(CLIENT_ID, NUM_CLIENTS, dataframe)
    
    # Run FL
    num_rounds = int(os.getenv("NUM_ROUNDS", "10"))
    client.run(num_rounds)


if __name__ == "__main__":
    main()
