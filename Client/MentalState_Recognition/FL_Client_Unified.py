"""
Unified Federated Learning Client for Mental State Recognition
with RL-based Protocol Selection

Supports: MQTT, AMQP, gRPC, QUIC, DDS
Uses Q-Learning to dynamically select the best protocol
"""

import os
import sys
import time
import threading
import numpy as np
_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
if _xla_flags:
    sanitized_flags = [f for f in _xla_flags.split() if f != "--xla_gpu_enable_command_buffer="]
    if sanitized_flags:
        os.environ["XLA_FLAGS"] = " ".join(sanitized_flags)
    else:
        os.environ.pop("XLA_FLAGS", None)
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Optional, List, Any
import json
import pickle
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
from rl_q_learning_selector import (
    QLearningProtocolSelector,
    EnvironmentStateManager,
    finalize_rl_boundary_collection_and_start_training,
)
from dynamic_network_controller import DynamicNetworkController
from MentalState_Recognition.data_partitioner import get_client_data, NUM_CLASSES
from fl_termination_env import stop_on_client_convergence

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
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set memory limit to prevent OOM with large CNN+BiLSTM+MHA model
        # Allow ~7GB per GPU (conservative for 10GB cards with overhead)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7168)]
        )
        print(f"[GPU] Configured with memory growth and 7GB limit")
    except RuntimeError as e:
        print(f"[GPU] Configuration error: {e}")

# Environment variables
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))
USE_RL_SELECTION = os.getenv("USE_RL_SELECTION", "true").lower() == "true"
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
DEFAULT_DATA_BATCH_SIZE = int(os.getenv("DEFAULT_DATA_BATCH_SIZE", "32"))
DEFAULT_LOCAL_EPOCHS = int(os.getenv("DEFAULT_LOCAL_EPOCHS", "20"))
ENABLE_LOCAL_CONVERGENCE_STOP = os.getenv("ENABLE_LOCAL_CONVERGENCE_STOP", "false").lower() == "true"
# Controls whether this client should signal/exit on local convergence (see fl_termination_env).
from fl_termination_env import stop_on_client_convergence
# When True, training ends when Q-learning value converges; when False, ends on accuracy convergence
USE_QL_CONVERGENCE = os.getenv("USE_QL_CONVERGENCE", "false").lower() == "true"
# Epsilon-greedy exploration for protocol selection (actual RL training). Independent of USE_QL_CONVERGENCE
# when USE_RL_EXPLORATION is set explicitly; if unset, matches USE_QL_CONVERGENCE for backward compatibility.
_ue = os.getenv("USE_RL_EXPLORATION", "").strip().lower()
if _ue in ("true", "1", "yes"):
    USE_RL_EXPLORATION = True
elif _ue in ("false", "0", "no"):
    USE_RL_EXPLORATION = False
else:
    USE_RL_EXPLORATION = USE_QL_CONVERGENCE
Q_CONVERGENCE_THRESHOLD = float(os.getenv("Q_CONVERGENCE_THRESHOLD", "0.01"))
Q_CONVERGENCE_PATIENCE = int(os.getenv("Q_CONVERGENCE_PATIENCE", "5"))
# Phase 1 — data collection: one sample per FL evaluation round with epsilon=1, no Q-updates (0 = skip → train immediately).
# The federated job length is NUM_ROUNDS (server). Phase 1 needs at least RL_PHASE0_ROUNDS FL rounds to finish
# collecting samples and run boundary computation; add more rounds for Phase 3 Q-learning after that.
RL_PHASE0_ROUNDS = int(os.getenv("RL_PHASE0_ROUNDS", "20"))
# When true (default), unified RL training runs: collect → compute boundaries → Q-learning training
RL_BOUNDARY_PIPELINE = os.getenv("RL_BOUNDARY_PIPELINE", "true").lower() in ("1", "true", "yes")
USE_COMMUNICATION_MODEL_REWARD = os.getenv("USE_COMMUNICATION_MODEL_REWARD", "true").lower() == "true"


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y")


RL_INFERENCE_ONLY = _env_truthy("RL_INFERENCE_ONLY")
_RL_PROTOCOL_SELECTION_RECORD_LEARNING = not RL_INFERENCE_ONLY

# Skip loading any on-disk Q-table (zeros + default epsilon). Overrides shared_data / PRETRAINED discovery.
RL_FRESH_Q_TABLE = _env_truthy("RL_FRESH_Q_TABLE")
# Optional explicit pickle paths for inference or a chosen checkpoint (highest priority if set and file exists).
_RL_Q_TABLE_UPLINK_FILE = (os.getenv("RL_Q_TABLE_UPLINK_PATH") or os.getenv("RL_Q_TABLE_UPLINK") or "").strip()
_RL_Q_TABLE_DOWNLINK_FILE = (os.getenv("RL_Q_TABLE_DOWNLINK_PATH") or os.getenv("RL_Q_TABLE_DOWNLINK") or "").strip()

_num_rounds_env = os.getenv("NUM_ROUNDS", "").strip()
if (
    _num_rounds_env
    and RL_BOUNDARY_PIPELINE
    and RL_PHASE0_ROUNDS > 0
    and not RL_FRESH_Q_TABLE
):
    try:
        _nr = int(_num_rounds_env)
        if _nr < RL_PHASE0_ROUNDS:
            print(
                f"[RL] Warning: NUM_ROUNDS={_nr} < RL_PHASE0_ROUNDS={RL_PHASE0_ROUNDS}. "
                f"Phase 1 (boundary data collection) will not complete before FL stops; "
                f"raise NUM_ROUNDS to at least {RL_PHASE0_ROUNDS} (plus rounds for Phase 3), "
                f"or lower RL_PHASE0_ROUNDS."
            )
    except ValueError:
        pass

TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(64 * 1024 * 1024)))


def _resolve_unified_grpc_host(default_host: str = "fl-server-unified-mentalstate") -> str:
    """Prefer GRPC_SERVER (compose) over GRPC_HOST (Client/Dockerfile may set fl-server)."""
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


# ---------------------------------------------------------------------------
# Model architecture helpers — must exactly match FL_Server_MQTT.py so that
# weights are compatible across server and all clients.
# ---------------------------------------------------------------------------
def _se_block(x, r=8):
    ch = x.shape[-1]
    s = keras.layers.GlobalAveragePooling1D()(x)
    s = keras.layers.Dense(max(ch // r, 8), activation='relu')(s)
    s = keras.layers.Dense(ch, activation='sigmoid', dtype='float32')(s)
    s = keras.layers.Reshape((1, ch))(s)
    return keras.layers.Multiply()([x, s])


def _conv_bn_relu(x, f, k, d=1):
    x = keras.layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    return x


def _res_block(x, f, k, d=1):
    sc = x
    y = _conv_bn_relu(x, f, k, d)
    y = keras.layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(y)
    y = keras.layers.BatchNormalization()(y)
    if sc.shape[-1] != f:
        sc = keras.layers.Conv1D(f, 1, padding="same", use_bias=False)(sc)
        sc = keras.layers.BatchNormalization()(sc)
    y = keras.layers.Add()([y, sc])
    y = keras.layers.ReLU()(y)
    y = _se_block(y)
    return y


class UnifiedFLClient_MentalState:
    """
    Unified Federated Learning Client for Mental State Recognition
    Integrates all 5 protocols with RL-based selection
    """
    
    def __init__(self, client_id: int, num_clients: int, X_train, y_train):
        """
        Initialize Unified FL Client
        
        Args:
            client_id: Unique client identifier
            num_clients: Total number of clients in FL
            X_train: Training data (EEG signals)
            y_train: Training labels
        """
        self.client_id = client_id
        self.num_clients = num_clients
        
        # Data
        self.X_train = X_train
        self.y_train = y_train
        
        # Model
        self.model = None
        self.local_epochs = DEFAULT_LOCAL_EPOCHS
        self.batch_size = DEFAULT_DATA_BATCH_SIZE
        
        # RL Components: two SEPARATE agents for UPLINK and DOWNLINK, each with own Q-table
        if USE_RL_SELECTION:
            # --- Uplink agent ---
            if os.path.exists("/shared_data"):
                save_path_uplink = "/shared_data/q_table_mentalstate_uplink_trained.pkl"
            else:
                save_path_uplink = f"q_table_mentalstate_uplink_client_{client_id}.pkl"
            initial_load_path_uplink = None
            if os.path.exists("/shared_data"):
                for _ul_cand in ("/shared_data/q_table_mentalstate_uplink_trained.pkl",
                                 "/shared_data/q_table_mentalstate_trained.pkl"):
                    if os.path.exists(_ul_cand):
                        initial_load_path_uplink = _ul_cand
                        break
            if initial_load_path_uplink is None:
                pretrained_dir = os.getenv("PRETRAINED_Q_TABLE_DIR")
                if pretrained_dir:
                    for candidate in (
                        os.path.join(pretrained_dir, "q_table_mentalstate_uplink_trained.pkl"),
                        os.path.join(pretrained_dir, "q_table_mentalstate_trained.pkl"),
                        os.path.join(pretrained_dir, f"q_table_mentalstate_uplink_client_{client_id}.pkl"),
                        os.path.join(pretrained_dir, f"q_table_mentalstate_client_{client_id}.pkl"),
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
                save_path_downlink = "/shared_data/q_table_mentalstate_downlink_trained.pkl"
            else:
                save_path_downlink = f"q_table_mentalstate_downlink_client_{client_id}.pkl"
            initial_load_path_downlink = None
            if os.path.exists("/shared_data") and os.path.exists("/shared_data/q_table_mentalstate_downlink_trained.pkl"):
                initial_load_path_downlink = "/shared_data/q_table_mentalstate_downlink_trained.pkl"
            if initial_load_path_downlink is None:
                pretrained_dir = os.getenv("PRETRAINED_Q_TABLE_DIR")
                if pretrained_dir:
                    for candidate in (
                        os.path.join(pretrained_dir, "q_table_mentalstate_downlink_trained.pkl"),
                        os.path.join(pretrained_dir, f"q_table_mentalstate_downlink_client_{client_id}.pkl"),
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
            self.env_manager = EnvironmentStateManager()
            self.env_manager.update_model_size('large')  # Mental state (LSTM model)
            if init_qlearning_db is not None:
                init_qlearning_db()
            
            # Boundary collection phase (similar to emotion recognition)
            # Phase 1 (data collection): collect uplink/downlink comm times, resource %, battery SoC
            # for RL_PHASE0_ROUNDS FL rounds with epsilon=1. Then Phase 2: compute boundary percentiles.
            # Finally Phase 3: Q-learning training with calculated boundaries.
            self._rl_boundary_collection_phase = bool(
                RL_BOUNDARY_PIPELINE
                and RL_PHASE0_ROUNDS > 0
                and not RL_FRESH_Q_TABLE
                and not RL_INFERENCE_ONLY
            )
            self._rl_boundary_uplink_comm_samples: List[float] = []
            self._rl_boundary_downlink_comm_samples: List[float] = []
            self._rl_boundary_res_samples: List[float] = []
            self._rl_boundary_batt_samples: List[float] = []
            self._rl_boundary_res_samples_dl: List[float] = []
            self._rl_boundary_batt_samples_dl: List[float] = []
            self._last_downlink_comm_wall_s: float = 0.0
            self._last_uplink_rl_state: Optional[Dict] = None
            if self._rl_boundary_collection_phase:
                self.rl_selector_uplink.epsilon = 1.0
                self.rl_selector_downlink.epsilon = 1.0
                print(
                    f"[Client {client_id}] RL pipeline: Phase 1 = first {RL_PHASE0_ROUNDS} FL rounds "
                    f"(uplink+downlink wall comm time, resource load, battery SoC; epsilon=1; no Q-updates). "
                    f"Then Phase 2 = boundaries, Phase 3 = Q-learning training."
                )
        else:
            self.rl_selector_uplink = None
            self.rl_selector_downlink = None
            self.rl_selector = None
            self.env_manager = None
            self._rl_boundary_collection_phase = False
        
        # Protocol handlers (same structure as emotion recognition)
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
        self._total_rounds = 0
        self.selected_downlink_protocol = 'grpc'
        self.initial_global_model_downloaded = False
        self.last_protocol_query_key = None
        self._registered_with_server = False
        # Downlink RL tracking
        self._last_downlink_rl_state = None
        self._downlink_select_time = None
        self.grpc_host = _resolve_unified_grpc_host()
        self.grpc_port = int(os.getenv("GRPC_PORT", "50051"))
        
        # Convergence tracking (for both accuracy and RL convergence)
        self.has_converged = False
        self.is_active = True
        self.shutdown_requested = False
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        
        # gRPC listener thread for background protocol query polling
        self.grpc_listener_thread = None
        self.grpc_stub = None
        self.grpc_lock = threading.Lock()  # Synchronize gRPC calls to prevent conflicts
        self.last_grpc_train_signal_round = -1
        self.last_grpc_eval_signal_round = -1
        
        print(f"\n{'='*70}")
        print(f"UNIFIED FL CLIENT - MENTAL STATE RECOGNITION")
        print(f"{'='*70}")
        print(f"Client ID: {self.client_id}/{self.num_clients}")
        print(f"gRPC (control / protocol negotiation): {self.grpc_host}:{self.grpc_port}")
        print(f"Training Samples: {len(self.y_train)}")
        print(f"RL Protocol Selection: {'ENABLED' if USE_RL_SELECTION else 'DISABLED'}")
        if USE_RL_SELECTION:
            print(f"RL Exploration (epsilon-greedy): {'ENABLED' if USE_RL_EXPLORATION else 'DISABLED (greedy)'}")
            print(f"Q-Convergence Stop: {'ENABLED' if USE_QL_CONVERGENCE else 'DISABLED'}")
        if USE_RL_SELECTION:
            print(f"Communication Model in Reward: {'ENABLED' if USE_COMMUNICATION_MODEL_REWARD else 'DISABLED'}")
        print(f"{'='*70}\n")
    
    def build_model(self, input_shape, num_classes) -> keras.Model:
        """Build mental state recognition model.

        Architecture is identical to FL_Server_MQTT.py so that weight tensors
        are fully compatible for FedAvg aggregation (both produce 88 tensors).
        Compiled with SparseCategoricalCrossentropy because client labels are
        integer-encoded; the server uses CategoricalCrossentropy but that only
        affects local training, not weight structure.
        """
        inp = keras.Input(shape=input_shape)

        x = _conv_bn_relu(inp, 64, 7, d=1)
        x = _res_block(x, 64, 7, d=1)
        x = keras.layers.MaxPooling1D(2)(x)

        for d in [1, 2, 4]:
            x = _res_block(x, 128, 5, d=d)

        x = keras.layers.Bidirectional(
            keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
        )(x)

        attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = keras.layers.Add()([x, attn])
        x = keras.layers.LayerNormalization()(x)

        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(0.35)(x)
        out = keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

        model = keras.Model(inp, out)

        lr_sched = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3, first_decay_steps=4,
            t_mul=2.0, m_mul=0.8, alpha=1e-5
        )
        opt = keras.optimizers.AdamW(
            learning_rate=lr_sched, weight_decay=1e-4, global_clipnorm=1.0
        )
        model.compile(
            optimizer=opt,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def select_protocol(self) -> str:
        """Select UPLINK protocol using the dedicated UPLINK RL agent (CPU-only Q-learning)"""
        if USE_RL_SELECTION and self.rl_selector_uplink and self.env_manager:
            try:
                # During Phase 1 (boundary collection), maintain epsilon=1.0 for pure random exploration
                if self._rl_boundary_collection_phase:
                    self.rl_selector_uplink.epsilon = 1.0
                
                # RL logic runs on CPU only (no GPU); keeps Q-table updates off GPU
                with tf.device('/CPU:0'):
                    import psutil
                    cpu = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory().percent
                    
                    resource_level = self.env_manager.detect_resource_level(cpu, memory)
                    self.env_manager.update_resource_level(resource_level)
                    
                    state = self.env_manager.get_current_state()
                    # Greedy vs explore from USE_RL_EXPLORATION; record_learning decoupled (see RL_INFERENCE_ONLY).
                    protocol = self.rl_selector_uplink.select_protocol(
                        state, training=USE_RL_EXPLORATION, record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING
                    )
                
                print(f"\n[Uplink RL Selection] State: {state}")
                print(f"[Uplink RL Selection] Selected Protocol: {protocol.upper()}")
                print(f"[Uplink RL Selection] Epsilon: {self.rl_selector_uplink.epsilon:.4f}")
                
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
        """
        Compute and log the downlink Q-learning reward after the global model is received.
        Uses the dedicated DOWNLINK RL agent (separate Q-table from uplink).
        Q-value is updated only during Phase 3 (after boundary collection).
        """
        if (not USE_RL_SELECTION
                or self.rl_selector_downlink is None
                or self._downlink_select_time is None
                or self._last_downlink_rl_state is None):
            return
        
        if not USE_RL_EXPLORATION:
            self._last_downlink_rl_state = None
            return
        
        try:
            comm_time = time.time() - self._downlink_select_time
            self._downlink_select_time = None
            self._last_downlink_comm_wall_s = float(comm_time)
            
            # Phase 1: Collect boundary samples for downlink (no Q-updates yet)
            if self._rl_boundary_collection_phase:
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
            resources = self.env_manager.get_resource_consumption() if self.env_manager else {}
            # Use actual downlink payload bytes from model reception
            payload_bytes = getattr(self, '_downlink_payload_bytes', None) or (12 * 1024 * 1024)
            t_calc = self._get_t_calc_for_reward(protocol, payload_bytes) if USE_COMMUNICATION_MODEL_REWARD else None
            
            reward = self.rl_selector_downlink.calculate_reward(
                communication_time=comm_time,
                success=True,
                resource_consumption=resources,
                t_calc=t_calc,
            )
            
            # Compute next state for Q-learning
            if self.env_manager:
                self.env_manager.update_comm_level_from_time(
                    comm_time,
                    self.rl_selector_downlink.comm_t_low,
                    self.rl_selector_downlink.comm_t_high,
                )
                self.env_manager.sync_battery_level_from_soc(None)
                next_state = self.env_manager.get_current_state()
            else:
                next_state = None
            
            # Only update Q-values during Phase 3 (after boundary calculation)
            if self._should_update_q_value():
                self.rl_selector_downlink.update_q_value(
                    reward,
                    next_state=next_state,
                    done=False if next_state is not None else True,
                )
            
            q_delta = self.rl_selector_downlink.get_last_q_delta()
            q_value = self.rl_selector_downlink.get_last_q_value()
            avg_reward = (
                float(np.mean(self.rl_selector_downlink.total_rewards[-100:]))
                if self.rl_selector_downlink.total_rewards else 0.0
            )
            
            if self._should_update_q_value():
                self.rl_selector_downlink.end_episode()
            
            q_converged = self.rl_selector_downlink.check_q_converged(
                threshold=Q_CONVERGENCE_THRESHOLD,
                patience=Q_CONVERGENCE_PATIENCE,
                state=downlink_state,
            )
            
            print(f"[Downlink RL] Client {self.client_id} | round={round_num} | protocol={protocol.upper()} | "
                  f"comm_time={comm_time:.3f}s | reward={reward:.2f} | q_delta={q_delta:.4f} | "
                  f"epsilon={self.rl_selector_downlink.epsilon:.4f} | q_conv_dl={q_converged}")
            
            # Only log Q-values during Phase 3 (Q-learning training), not during boundary collection
            if log_q_step is not None and self._should_update_q_value():
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
                    reward_total=reward_details.get('reward_total', reward),
                    link_direction='downlink',
                )
        except Exception as e:
            print(f"[Downlink RL] Client {self.client_id} reward update error: {e}")
            import traceback
            traceback.print_exc()

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

    def _register_with_server_via_grpc(self) -> bool:
        """Register this client with the FL server via gRPC (idempotent)."""
        if self._registered_with_server:
            return True
        channel, stub = self._open_grpc_stub()
        if stub is None:
            return False
        try:
            resp = stub.RegisterClient(
                federated_learning_pb2.ClientRegistration(client_id=self.client_id)
            )
            if resp.success:
                self._registered_with_server = True
                print(f"[gRPC] Client {self.client_id} registered with server (msg: {resp.message})")
                return True
            print(f"[gRPC] Registration rejected: {resp.message}")
            return False
        except Exception as e:
            print(f"[gRPC] Registration failed: {e}")
            return False
        finally:
            channel.close()

    def _upload_model_via_grpc(self, round_num: int, train_metrics: dict) -> float:
        """Serialize current model weights and send to server via gRPC SendModelUpdate.
        Returns elapsed uplink communication time in seconds (0.0 on failure)."""
        if self.model is None:
            return 0.0
        channel, stub = self._open_grpc_stub()
        if stub is None:
            return 0.0
        try:
            weights_bytes = self.get_model_weights()
            # Track model size for monitoring and RL reward calculation
            payload_bytes = len(weights_bytes)
            self.round_metrics['payload_bytes'] = payload_bytes
            print(f"[Model Size] Uplink: {payload_bytes / (1024*1024):.2f} MB")
            
            t0 = time.time()
            resp = stub.SendModelUpdate(
                federated_learning_pb2.ModelUpdate(
                    client_id=self.client_id,
                    round=round_num,
                    weights=weights_bytes,
                    num_samples=int(len(self.y_train)),
                    metrics={
                        'val_accuracy': float(train_metrics.get('val_accuracy', 0.0)),
                        'val_loss':     float(train_metrics.get('val_loss', 0.0)),
                        'training_time': float(train_metrics.get('training_time', 0.0)),
                    },
                    chunk_index=0,
                    total_chunks=1,
                )
            )
            comm_time = time.time() - t0
            print(f"[gRPC] Model uploaded (round={round_num}, {comm_time:.2f}s): {resp.message}")
            return comm_time
        except Exception as e:
            print(f"[gRPC] Model upload failed (round={round_num}): {e}")
            return 0.0
        finally:
            channel.close()

    def _wait_and_apply_global_model_via_grpc(self, target_round: int, timeout: int = 180) -> bool:
        """Poll server until it has a global model for target_round, then apply it.
        The background gRPC listener handles protocol query responses automatically.
        Returns True if the model was successfully downloaded and applied."""
        channel, stub = self._open_grpc_stub()
        if stub is None:
            return False
        deadline = time.time() + timeout
        poll_count = 0
        try:
            while time.time() < deadline:
                poll_count += 1
                if poll_count <= 10 or poll_count % 5 == 0:  # Log first 10 polls, then every 5th
                    print(f"[Client {self.client_id}] Poll #{poll_count} for round {target_round}: checking for model...")
                
                # Check for global model
                try:
                    response = stub.GetGlobalModel(
                        federated_learning_pb2.ModelRequest(
                            client_id=self.client_id, round=target_round, chunk_index=0
                        )
                    )
                    if response.available and response.weights and response.round >= target_round:
                        # Track downlink model size
                        downlink_payload_bytes = len(response.weights)
                        self._downlink_payload_bytes = downlink_payload_bytes
                        print(f"[Model Size] Downlink: {downlink_payload_bytes / (1024*1024):.2f} MB")
                        
                        self.set_model_weights(response.weights)
                        self.initial_global_model_downloaded = True
                        print(f"[gRPC] Applied global model (server round={response.round})")
                        return True
                except Exception as e:
                    if poll_count <= 3:  # Only log first few errors
                        print(f"[gRPC] GetGlobalModel error (target={target_round}): {e}")
                
                time.sleep(2.0)
            print(f"[gRPC] Timed out waiting for global model (target_round={target_round})")
            return False
        finally:
            channel.close()

    def _select_downlink_protocol(self, round_id: int, global_model_id: int) -> str:
        """Select downlink protocol using RL agent.
        
        Round 1 (initial model) always uses gRPC - no protocol selection.
        Protocol selection via RL starts from Round 2 onwards (matches Emotion Recognition).
        """
        # Round 1: Always use gRPC for initial model bootstrap
        if round_id <= 1 or global_model_id <= 1 or not self.initial_global_model_downloaded:
            print(f"[Downlink] Round {round_id}: Using gRPC (initial model, no RL selection)")
            return 'grpc'
        
        # Round 2+: Use dedicated DOWNLINK RL agent for protocol selection
        if USE_RL_SELECTION and self.rl_selector_downlink and self.env_manager:
            try:
                # During Phase 1 (boundary collection), maintain epsilon=1.0 for pure random exploration
                if self._rl_boundary_collection_phase:
                    self.rl_selector_downlink.epsilon = 1.0
                
                with tf.device('/CPU:0'):
                    state = self.env_manager.get_current_state()
                    print(f"[Downlink RL Selection] State: {state}")
                    selected = self.rl_selector_downlink.select_protocol(
                        state, training=USE_RL_EXPLORATION, record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING
                    )
                    print(f"[Downlink RL Selection] Selected Protocol: {selected.upper()}")
                    print(f"[Downlink RL Selection] Epsilon: {self.rl_selector_downlink.epsilon:.4f}")
                self._last_downlink_rl_state = state
                self._downlink_select_time = time.time()
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

    def _handle_grpc_protocol_query(self, protocol_query):
        """Respond to server ProtocolQuery via gRPC using RL-based downlink selection.
        Called automatically by the background gRPC listener thread."""
        if protocol_query is None:
            return

        query_key = (
            int(getattr(protocol_query, 'round_id', -1)),
            int(getattr(protocol_query, 'global_model_id', -1)),
        )
        if self.last_protocol_query_key == query_key:
            return

        round_id, global_model_id = query_key
        selected_downlink_protocol = self._select_downlink_protocol(round_id, global_model_id)

        with self.grpc_lock:
            if self.grpc_stub is None:
                return
            response = self.grpc_stub.SendProtocolSelection(
                federated_learning_pb2.ProtocolSelection(
                    client_id=self.client_id,
                    round_id=round_id,
                    global_model_id=global_model_id,
                    downlink_protocol_requested=selected_downlink_protocol,
                )
            )

        if response.success:
            self.selected_downlink_protocol = selected_downlink_protocol
            self.last_protocol_query_key = query_key
            # Real downlink reward is computed in _update_downlink_rl_after_reception()
            # after global model arrives. We do NOT log reward=0.0 here.
            _dl_eps = (
                f"{self.rl_selector_downlink.epsilon:.4f}"
                if self.rl_selector_downlink is not None
                else "N/A"
            )
            print(
                f"[gRPC] Client {self.client_id} replied ProtocolSelection: "
                f"round={round_id}, global_model_id={global_model_id}, "
                f"downlink={selected_downlink_protocol} "
                f"(downlink RL agent epsilon={_dl_eps})"
            )
        else:
            print(f"[gRPC] Client {self.client_id} ProtocolSelection rejected: {response.message}")

    def _poll_and_respond_protocol_query_via_grpc(self):
        """Manual polling version (kept for backwards compatibility during transition)."""
        with self.grpc_lock:
            if self.grpc_stub is None:
                return
            stub = self.grpc_stub
        
        try:
            status = stub.CheckTrainingStatus(
                federated_learning_pb2.StatusRequest(client_id=self.client_id, current_round=self.current_round)
            )
            
            has_query = getattr(status, 'has_protocol_query', False)
            if has_query:
                self._handle_grpc_protocol_query(status.protocol_query)
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} protocol query poll error: {e}")

    def start_grpc_listener(self):
        """Start gRPC polling thread for automatic protocol query handling."""
        if self.grpc_listener_thread is not None and self.grpc_listener_thread.is_alive():
            return
        
        def grpc_listener_loop():
            try:
                # Create gRPC channel and stub
                options = [
                    ('grpc.max_send_message_length', 4 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 4 * 1024 * 1024),
                    ('grpc.keepalive_time_ms', 600000),
                    ('grpc.keepalive_timeout_ms', 60000),
                ]
                channel = grpc.insecure_channel(f"{self.grpc_host}:{self.grpc_port}", options=options)
                with self.grpc_lock:
                    self.grpc_stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
                
                print(f"[gRPC] Listener started for client {self.client_id}")

                # Polling loop
                while not self.shutdown_requested:
                    try:
                        status_request = federated_learning_pb2.StatusRequest(client_id=self.client_id)
                        with self.grpc_lock:
                            if self.grpc_stub is None:
                                break
                            stub = self.grpc_stub
                        status = stub.CheckTrainingStatus(status_request)

                        if getattr(status, 'has_protocol_query', False):
                            self._handle_grpc_protocol_query(status.protocol_query)

                    except grpc.RpcError as e:
                        # Don't spam logs for connection errors during shutdown
                        if not self.shutdown_requested:
                            print(
                                f"[gRPC] Client {self.client_id} CheckTrainingStatus RpcError (will retry): "
                                f"{e.code()} - {e.details()}"
                            )
                    except Exception as e:
                        if not self.shutdown_requested:
                            print(f"[gRPC] Client {self.client_id} CheckTrainingStatus error: {e}")

                    time.sleep(1)  # Poll every second
                    
            except Exception as e:
                print(f"[gRPC] Listener error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                with self.grpc_lock:
                    self.grpc_stub = None
                try:
                    channel.close()
                except:
                    pass
        
        self.grpc_listener_thread = threading.Thread(target=grpc_listener_loop, daemon=True, name=f"gRPC-Listener-{self.client_id}")
        self.grpc_listener_thread.start()

    def _ensure_initial_global_model_via_grpc(self):
        """Download and apply the initial global model from the server via gRPC.
        
        The server always broadcasts Round 1 (initial model) via gRPC - no protocol negotiation.
        Protocol negotiation starts from Round 2 onwards.
        """
        if self.initial_global_model_downloaded:
            return
        print(f"[Client {self.client_id}] Downloading initial global model via gRPC (Round 1)...")
        self._wait_and_apply_global_model_via_grpc(target_round=0, timeout=180)  # Round 0 means "any round >= 0"

    def train_local_model(self) -> Dict:
        """Train model locally using real EEG data"""
        if self.model is None:
            input_shape = (self.X_train.shape[1], self.X_train.shape[2])
            self.model = self.build_model(input_shape, NUM_CLASSES)
        
        start_time = time.time()
        
        # Train with GPU acceleration (verbose=2 to show epoch progress)
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=2  # Show epoch progress like emotion recognition
            )
        
        training_time = time.time() - start_time
        
        # Get validation metrics
        val_accuracy = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        
        metrics = {
            'training_time': training_time,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss
        }
        
        print(f"[Training] Time: {training_time:.2f}s, Accuracy: {val_accuracy:.4f}")
        
        return metrics
    
    def get_model_weights(self) -> bytes:
        """Get serialized model weights"""
        weights = self.model.get_weights()
        return pickle.dumps(weights)
    
    def set_model_weights(self, weights_bytes: bytes):
        """Set model weights from serialized bytes"""
        if self.model is None:
            input_shape = (self.X_train.shape[1], self.X_train.shape[2])
            self.model = self.build_model(input_shape, NUM_CLASSES)
        weights = pickle.loads(weights_bytes)
        self.model.set_weights(weights)
    
    # Protocol handlers (simplified versions - same as emotion recognition)
    def _handle_mqtt(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle MQTT protocol communication"""
        try:
            broker = os.getenv("MQTT_BROKER", "mqtt-broker")
            port = int(os.getenv("MQTT_PORT", "1883"))
            client = mqtt_client.Client(f"mentalstate_client_{self.client_id}")
            client.connect(broker, port)
            
            if action == "send":
                client.publish(TOPIC_CLIENT_UPDATE, data)
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
                body = data.decode('utf-8') if isinstance(data, bytes) else data
                channel.basic_publish(
                    exchange='fl_client_updates',
                    routing_key='client.update',
                    body=body,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                connection.close()
                return True, None
            connection.close()
            return True, data
        except Exception as e:
            print(f"[AMQP] Error: {e}")
            return False, None
    
    def _handle_grpc(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle gRPC protocol communication"""
        print("[gRPC] Protocol handler")
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

        # Ensure first-round model bootstrap (protocol negotiation happens after upload)
        self._ensure_initial_global_model_via_grpc()
        
        if protocol is None:
            protocol = self.select_protocol()
        
        print(f"\n{'='*70}")
        print(f"FL ROUND - Using {protocol.upper()} Protocol")
        print(f"{'='*70}")
        
        try:
            # Train locally
            print(f"[Training] Starting local training...")
            train_metrics = self.train_local_model()

            # ---- Uplink: upload model to server via gRPC and record communication time ----
            comm_time = self._upload_model_via_grpc(self.current_round, train_metrics)

            # ---- Downlink: wait for server to aggregate and broadcast new global model ----
            # (Protocol negotiation polling happens automatically while waiting)
            next_round = self.current_round + 1
            if next_round <= self._total_rounds:
                self._wait_and_apply_global_model_via_grpc(target_round=next_round, timeout=180)

            # Update metrics
            round_time = time.time() - round_start
            self.round_metrics['convergence_time'] = train_metrics['training_time']
            self.round_metrics['accuracy'] = train_metrics['val_accuracy']
            self.round_metrics['communication_time'] = comm_time
            self.round_metrics['success'] = True
            
            # Uplink RL: compute reward always for logging; update Q only when not in boundary collection phase.
            if USE_RL_SELECTION and self.rl_selector_uplink and self.env_manager:
                import psutil
                resources = self.env_manager.get_resource_consumption()
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory_usage = psutil.virtual_memory().percent
                resource_avg = (cpu_usage + memory_usage) / 2.0
                battery_soc = self.env_manager.battery_soc
                
                # Phase 1: Collect boundary samples (no Q-updates yet)
                if self._rl_boundary_collection_phase:
                    self._rl_boundary_uplink_comm_samples.append(comm_time)
                    self._rl_boundary_res_samples.append(resource_avg)
                    self._rl_boundary_batt_samples.append(battery_soc)
                    print(f"[RL Phase 1] Collecting boundary data: comm={comm_time:.3f}s, resource={resource_avg:.1f}%, battery={battery_soc:.2f}")
                
                payload_bytes = self.round_metrics.get('payload_bytes', 12 * 1024 * 1024)
                t_calc = self._get_t_calc_for_reward(protocol, payload_bytes) if USE_COMMUNICATION_MODEL_REWARD else None
                reward = self.rl_selector_uplink.calculate_reward(
                    communication_time=self.round_metrics['communication_time'],
                    success=self.round_metrics['success'],
                    resource_consumption=resources,
                    t_calc=t_calc,
                )
                
                # Only update Q-values during Phase 3 (after boundary calculation)
                if USE_RL_EXPLORATION and self._should_update_q_value():
                    self.rl_selector_uplink.update_q_value(reward, done=False)
                
                # Only log Q-values during Phase 3 (Q-learning training), not during boundary collection
                if log_q_step is not None and self._should_update_q_value():
                    st = self.env_manager.get_current_state()
                    q_delta = self.rl_selector_uplink.get_last_q_delta()
                    q_value = self.rl_selector_uplink.get_last_q_value()
                    avg_reward = (np.mean(self.rl_selector_uplink.total_rewards[-100:])
                                 if self.rl_selector_uplink.total_rewards else 0.0)
                    reward_details = self.rl_selector_uplink.get_last_reward_breakdown()
                    log_q_step(
                        client_id=self.client_id,
                        round_num=int(getattr(self, 'current_round', 0)),
                        episode=self.rl_selector_uplink.episode_count,
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
                        converged=self.rl_selector_uplink.check_q_converged(),
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
                self._update_downlink_rl_after_reception(round_num=int(getattr(self, 'current_round', 0)))
            
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
                    "training_time_sec": float(train_metrics.get("training_time", 0.0)),
                    "total_fl_wall_time_sec": float(round_time),
                    "uplink_model_comm_sec": float(self.round_metrics.get("communication_time", 0.0)),
                    "battery_energy_joules": 0.0,
                    "battery_soc_after": _batt_soc,
                },
                use_case=use_case_from_env("mental_state"),
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
    
    def _should_update_q_value(self) -> bool:
        """Return True if we should perform Q-value updates (not during boundary collection phase)"""
        return not getattr(self, "_rl_boundary_collection_phase", False)
    
    def _compute_boundaries_and_start_q_learning(self):
        """
        Compute boundary percentiles from collected samples and transition from Phase 1 to Phase 3.
        Called after RL_PHASE0_ROUNDS rounds of data collection.
        Uses finalize_rl_boundary_collection_and_start_training to properly set boundaries 
        for both uplink and downlink RL agents.
        """
        if not self._rl_boundary_collection_phase:
            return
        
        print(f"\n{'='*70}")
        print("RL PIPELINE: COMPUTING BOUNDARIES (Phase 2)")
        print(f"{'='*70}")
        print(f"Collected {len(self._rl_boundary_uplink_comm_samples)} uplink comm samples")
        print(f"Collected {len(self._rl_boundary_downlink_comm_samples)} downlink comm samples")
        print(f"Collected {len(self._rl_boundary_res_samples)} uplink resource samples")
        print(f"Collected {len(self._rl_boundary_batt_samples)} uplink battery samples")
        print(f"Collected {len(self._rl_boundary_res_samples_dl)} downlink resource samples")
        print(f"Collected {len(self._rl_boundary_batt_samples_dl)} downlink battery samples")
        
        # Use the same function as emotion recognition to compute and apply boundaries
        if finalize_rl_boundary_collection_and_start_training is not None:
            finalize_rl_boundary_collection_and_start_training(
                self.rl_selector_uplink,
                self.rl_selector_downlink,
                self.env_manager,
                self._rl_boundary_uplink_comm_samples,
                self._rl_boundary_res_samples,
                self._rl_boundary_batt_samples,
                client_id=self.client_id,
                downlink_comm_times=self._rl_boundary_downlink_comm_samples,
                resource_loads_downlink=self._rl_boundary_res_samples_dl,
                battery_socs_downlink=self._rl_boundary_batt_samples_dl,
            )
        
        print(f"\n{'='*70}")
        print("RL PIPELINE: STARTING Q-LEARNING TRAINING (Phase 3)")
        print(f"{'='*70}\n")
        
        # Transition to Phase 3: enable Q-learning (finalize function resets epsilon to 1.0 for both agents)
        self._rl_boundary_collection_phase = False
        
        # Clear the collected samples to free memory
        self._rl_boundary_uplink_comm_samples = []
        self._rl_boundary_downlink_comm_samples = []
        self._rl_boundary_res_samples = []
        self._rl_boundary_batt_samples = []
        self._rl_boundary_res_samples_dl = []
        self._rl_boundary_batt_samples_dl = []
    
    def _would_converge_after_eval(self, loss: float) -> bool:
        """True if this eval loss would trigger local convergence in the same round (mirrors FL_Client_gRPC)."""
        if self.current_round < MIN_ROUNDS or self.has_converged:
            return False
        if self.best_loss - loss > CONVERGENCE_THRESHOLD:
            return False
        return (self.rounds_without_improvement + 1) >= CONVERGENCE_PATIENCE

    def _update_client_convergence_and_maybe_disconnect(self, loss: float):
        """Track local convergence and disconnect this client when converged."""
        if self.current_round < MIN_ROUNDS:
            self.best_loss = min(self.best_loss, float(loss))
            return

        if self.best_loss - float(loss) > CONVERGENCE_THRESHOLD:
            self.best_loss = float(loss)
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

        if self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
            self.has_converged = True
            print(f"[Client {self.client_id}] Local convergence reached at round {self.current_round}")
            if stop_on_client_convergence():
                self._notify_convergence_to_server()
                self._disconnect_after_convergence()

    def _notify_convergence_to_server(self):
        """Notify server this client is converged using gRPC control signal."""
        if grpc is None or federated_learning_pb2 is None or federated_learning_pb2_grpc is None:
            print(f"[gRPC] Client {self.client_id} convergence signal skipped: gRPC unavailable")
            return
        try:
            options = [
                ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
                ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
                ('grpc.keepalive_time_ms', 600000),
                ('grpc.keepalive_timeout_ms', 60000),
            ]
            channel = grpc.insecure_channel(f'{self.grpc_host}:{self.grpc_port}', options=options)
            stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
            response = stub.SendModelUpdate(
                federated_learning_pb2.ModelUpdate(
                    client_id=self.client_id,
                    round=self.current_round,
                    weights=b"",
                    num_samples=0,
                    metrics={"client_converged": 1.0}
                )
            )
            if response.success:
                print(f"[gRPC] Client {self.client_id} convergence notification sent")
            else:
                print(f"[gRPC] Convergence notification failed: {response.message}")
            channel.close()
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} failed to notify convergence: {e}")

    def _disconnect_after_convergence(self):
        """Stop participating once local convergence is reached."""
        self.is_active = False
        self.shutdown_requested = True
        print(f"[Client {self.client_id}] Disconnecting after local convergence")
    
    def run(self, num_rounds: int = 10):
        """Run federated learning for multiple rounds"""
        print(f"\n{'='*70}")
        print(f"STARTING FEDERATED LEARNING - {num_rounds} ROUNDS")
        print(f"{'='*70}\n")

        self._total_rounds = num_rounds

        # Register with server before starting (retry for up to 60 s)
        print("[gRPC] Registering with FL server...")
        for attempt in range(30):
            if self._register_with_server_via_grpc():
                break
            print(f"[gRPC] Registration attempt {attempt + 1}/30 failed, retrying in 2s...")
            time.sleep(2.0)
        else:
            print("[gRPC] Could not register with server – proceeding in stand-alone mode")

        # Start gRPC listener thread for automatic protocol query handling
        print("[gRPC] Starting background listener for protocol queries...")
        self.start_grpc_listener()
        
        # Wait for server to start training and deliver the initial global model
        print("[gRPC] Waiting for initial global model from server...")
        self._ensure_initial_global_model_via_grpc()

        for round_num in range(1, num_rounds + 1):
            self.current_round = round_num
            print(f"\n{'#'*70}")
            print(f"# ROUND {round_num}/{num_rounds}")
            print(f"{'#'*70}\n")
            
            # Check if we need to transition from Phase 1 (boundary collection) to Phase 3 (Q-learning)
            if (
                self._rl_boundary_collection_phase
                and round_num == RL_PHASE0_ROUNDS
            ):
                self._compute_boundaries_and_start_q_learning()
            
            metrics = self.federated_learning_round()
            
            print(f"\n[Round {round_num}] Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            
            # Handle RL convergence check (when USE_QL_CONVERGENCE is enabled and not in boundary collection phase)
            if USE_RL_SELECTION and self.rl_selector_uplink and USE_QL_CONVERGENCE and not self._rl_boundary_collection_phase:
                # Check Q-learning convergence for both uplink and downlink
                q_uplink_converged = self.rl_selector_uplink.check_q_converged(
                    threshold=Q_CONVERGENCE_THRESHOLD,
                    patience=Q_CONVERGENCE_PATIENCE,
                    state=self._last_uplink_rl_state if hasattr(self, '_last_uplink_rl_state') else None,
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
                
                if q_uplink_converged and not q_downlink_converged:
                    print(
                        f"[Client {self.client_id}] Uplink Q converged at round {round_num} but downlink "
                        f"has not; continuing (uplink Q-updates unchanged)."
                    )
                
                # End training only when both uplink and downlink Q-learning converged
                if q_both_converged and stop_on_client_convergence():
                    self.has_converged = True
                    print(
                        f"[Client {self.client_id}] RL convergence reached at round {round_num} "
                        f"(last {Q_CONVERGENCE_PATIENCE} consecutive protocol selections identical "
                        f"on uplink AND downlink)"
                    )
                    try:
                        self.rl_selector_uplink.save_q_table()
                    except Exception as e:
                        print(f"[Client {self.client_id}] Warning: could not save uplink Q-table on convergence: {e}")
                    try:
                        self.rl_selector_downlink.save_q_table()
                    except Exception as e:
                        print(f"[Client {self.client_id}] Warning: could not save downlink Q-table on convergence: {e}")
                    self._notify_convergence_to_server()
                    self._disconnect_after_convergence()
                    break  # Exit training loop after convergence
            
            # Handle accuracy convergence check (when not using RL convergence mode)
            if not USE_QL_CONVERGENCE and ENABLE_LOCAL_CONVERGENCE_STOP and stop_on_client_convergence():
                loss_f = float(metrics.get('val_loss', 0.0))
                self._update_client_convergence_and_maybe_disconnect(loss_f)
                if self.has_converged:
                    break  # Exit training loop after convergence
            
            if USE_RL_SELECTION and self.rl_selector and USE_QL_CONVERGENCE:
                self.rl_selector_uplink.end_episode()
                self.rl_selector_downlink.end_episode()
        
        if USE_RL_SELECTION and self.rl_selector:
            self.rl_selector.print_statistics()


def main():
    """Main function"""
    print(f"Unified FL Client - Mental State Recognition (Client {CLIENT_ID})")
    
    # Load real EEG dataset
    print(f"\n{'='*70}")
    print("LOADING MENTAL STATE RECOGNITION DATASET (EEG)")
    print(f"{'='*70}")
    
    try:
        ds_raw = os.getenv("DATASET_CLIENT_ID", "").strip()
        if ds_raw:
            partition_idx = int(ds_raw) - 1
            if partition_idx < 0 or partition_idx >= NUM_CLIENTS:
                print(
                    f"[Error] DATASET_CLIENT_ID={ds_raw} implies partition {partition_idx}, "
                    f"but NUM_CLIENTS={NUM_CLIENTS} (valid shards: 1..{NUM_CLIENTS})."
                )
                return
            X_train, y_train = get_client_data(partition_idx, NUM_CLIENTS)
        else:
            X_train, y_train = get_client_data(CLIENT_ID, NUM_CLIENTS)
        print(f"[Dataset] Loaded successfully")
        print(f"[Dataset] Shape: {X_train.shape}")
        print(f"[Dataset] Samples: {len(y_train)}")
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print(f"\nPlease ensure dataset exists at:")
        print(f"  Client/MentalState_Recognition/Dataset/")
        return
    
    # Create client
    client = UnifiedFLClient_MentalState(CLIENT_ID, NUM_CLIENTS, X_train, y_train)
    
    # Run FL
    num_rounds = int(os.getenv("NUM_ROUNDS", "10"))
    client.run(num_rounds)


if __name__ == "__main__":
    main()
