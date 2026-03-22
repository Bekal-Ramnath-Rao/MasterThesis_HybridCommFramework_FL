"""
Unified Federated Learning Client for Temperature Regulation
with RL-based Protocol Selection

Supports: MQTT, AMQP, gRPC, QUIC, DDS
Uses Q-Learning to dynamically select the best protocol
"""

import os
import sys
import time
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
from typing import Dict, Tuple, Optional
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
from rl_q_learning_selector import QLearningProtocolSelector, EnvironmentStateManager
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
    from q_learning_logger import init_db as init_qlearning_db, log_q_step
except ImportError:
    init_qlearning_db = None
    log_q_step = None

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
USE_COMMUNICATION_MODEL_REWARD = os.getenv("USE_COMMUNICATION_MODEL_REWARD", "true").lower() == "true"

TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))


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
        self.local_epochs = 5
        self.batch_size = 16
        
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
            )
            # Backward-compat alias
            self.rl_selector = self.rl_selector_uplink
            self.env_manager = EnvironmentStateManager()
            self.env_manager.update_model_size('small')  # Temperature (small model)
            if init_qlearning_db is not None:
                init_qlearning_db()
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
        self.selected_downlink_protocol = 'grpc'
        self.initial_global_model_downloaded = False
        self.last_protocol_query_key = None
        # Downlink RL tracking
        self._last_downlink_rl_state = None
        self._downlink_select_time = None
        self.grpc_host = os.getenv("GRPC_HOST", "fl-server-unified-temperature")
        self.grpc_port = int(os.getenv("GRPC_PORT", "50051"))
        
        print(f"\n{'='*70}")
        print(f"UNIFIED FL CLIENT - TEMPERATURE REGULATION")
        print(f"{'='*70}")
        print(f"Client ID: {self.client_id}/{self.num_clients}")
        print(f"Training Samples: {len(self.y_train)}")
        print(f"RL Protocol Selection: {'ENABLED' if USE_RL_SELECTION else 'DISABLED'}")
        if USE_RL_SELECTION:
            print(f"Communication Model in Reward: {'ENABLED' if USE_COMMUNICATION_MODEL_REWARD else 'DISABLED'}")
        print(f"{'='*70}\n")
    
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare temperature data for training"""
        # Split data for this client
        total_samples = len(self.dataframe)
        samples_per_client = total_samples // self.num_clients
        start_idx = self.client_id * samples_per_client
        end_idx = start_idx + samples_per_client if self.client_id < self.num_clients - 1 else total_samples
        
        client_data = self.dataframe.iloc[start_idx:end_idx]
        
        # Prepare features and target
        # Assuming the last column is the target
        X = client_data.iloc[:, :-1].values
        y = client_data.iloc[:, -1].values
        
        print(f"[Data Preparation] Client {self.client_id}")
        print(f"  Total dataset: {total_samples} samples")
        print(f"  Client subset: {len(y)} samples (indices {start_idx} to {end_idx})")
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
                    
                    state = self.env_manager.get_current_state()
                    # Inference: use learned Q-table only (training=False); training: epsilon-greedy (training=True)
                    protocol = self.rl_selector_uplink.select_protocol(state, training=USE_QL_CONVERGENCE)
                
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
        """Compute and log downlink Q-learning reward after global model is received."""
        if (not USE_RL_SELECTION
                or self.rl_selector_downlink is None
                or self._downlink_select_time is None
                or self._last_downlink_rl_state is None):
            return
        try:
            comm_time = time.time() - self._downlink_select_time
            self._downlink_select_time = None
            downlink_state = self._last_downlink_rl_state
            self._last_downlink_rl_state = None
            protocol = self.selected_downlink_protocol or 'grpc'
            resources = self.env_manager.get_resource_consumption() if self.env_manager else {}
            payload_bytes = 12 * 1024 * 1024
            t_calc = self._get_t_calc_for_reward(protocol, payload_bytes) if USE_COMMUNICATION_MODEL_REWARD else None
            accuracy_for_reward = self.round_metrics.get('accuracy', 0.0)
            reward = self.rl_selector_downlink.calculate_reward(
                communication_time=comm_time,
                success=True,
                accuracy=accuracy_for_reward,
                resource_consumption=resources,
                t_calc=t_calc,
            )
            self.rl_selector_downlink.update_q_value(reward, next_state=None, done=True)
            q_delta = self.rl_selector_downlink.get_last_q_delta()
            avg_reward = (
                float(np.mean(self.rl_selector_downlink.total_rewards[-100:]))
                if self.rl_selector_downlink.total_rewards else 0.0
            )
            self.rl_selector_downlink.end_episode()
            q_converged = self.rl_selector_downlink.check_q_converged()
            print(f"[Downlink RL] round={round_num} | protocol={protocol.upper()} | "
                  f"comm_time={comm_time:.3f}s | reward={reward:.2f} | "
                  f"epsilon={self.rl_selector_downlink.epsilon:.4f}")
            if log_q_step is not None:
                reward_details = self.rl_selector_downlink.get_last_reward_breakdown()
                log_q_step(
                    client_id=self.client_id,
                    round_num=round_num or int(getattr(self, 'current_round', 0)),
                    episode=self.rl_selector_downlink.episode_count - 1,
                    state_network=downlink_state.get('network', ''),
                    state_resource=downlink_state.get('resource', ''),
                    state_model_size=downlink_state.get('model_size', ''),
                    state_mobility=downlink_state.get('mobility', ''),
                    action=protocol,
                    reward=reward,
                    q_delta=q_delta,
                    epsilon=self.rl_selector_downlink.epsilon,
                    avg_reward_last_100=avg_reward,
                    converged=q_converged,
                    metric_communication_time=reward_details.get('communication_time'),
                    metric_convergence_time=None,
                    metric_accuracy=reward_details.get('accuracy'),
                    metric_success=reward_details.get('success'),
                    metric_cpu_usage=reward_details.get('cpu_usage'),
                    metric_memory_usage=reward_details.get('memory_usage'),
                    metric_bandwidth_usage=reward_details.get('bandwidth_usage'),
                    metric_battery_level=reward_details.get('battery_level'),
                    metric_energy_usage=reward_details.get('energy_usage'),
                    metric_t_calc=reward_details.get('t_calc'),
                    reward_base=reward_details.get('reward_base'),
                    reward_communication_time=reward_details.get('reward_communication_time'),
                    reward_convergence_time=None,
                    reward_accuracy=reward_details.get('reward_accuracy'),
                    reward_resource_penalty=reward_details.get('reward_resource_penalty'),
                    reward_battery_penalty=reward_details.get('reward_battery_penalty'),
                    reward_t_calc_penalty=reward_details.get('reward_t_calc_penalty'),
                    reward_total=reward_details.get('reward_total'),
                    link_direction='downlink',
                )
        except Exception as e:
            print(f"[Downlink RL] reward update error: {e}")

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

    def _select_downlink_protocol(self, round_id: int, global_model_id: int) -> str:
        # First global model bootstrap is always forced to gRPC.
        if round_id <= 1 or global_model_id <= 1 or not self.initial_global_model_downloaded:
            return 'grpc'
        # Use the dedicated DOWNLINK RL agent
        if USE_RL_SELECTION and self.rl_selector_downlink and self.env_manager:
            try:
                with tf.device('/CPU:0'):
                    state = self.env_manager.get_current_state()
                    selected = self.rl_selector_downlink.select_protocol(state, training=USE_QL_CONVERGENCE)
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
                self.initial_global_model_downloaded = True
                print(f"[gRPC] Client {self.client_id} downloaded initial global model via gRPC")
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
        
        # Train with GPU acceleration
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0
            )
        
        training_time = time.time() - start_time
        
        # Get validation metrics
        val_mae = history.history['val_mae'][-1]
        val_loss = history.history['val_loss'][-1]
        
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
    
    # Protocol handlers (simplified versions)
    def _handle_mqtt(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle MQTT protocol communication"""
        try:
            broker = os.getenv("MQTT_BROKER", "mqtt-broker")
            port = int(os.getenv("MQTT_PORT", "1883"))
            client = mqtt_client.Client(f"temperature_client_{self.client_id}")
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

        # Ensure first-round model bootstrap and handle pending protocol negotiation over gRPC.
        self._ensure_initial_global_model_via_grpc()
        self._poll_and_respond_protocol_query_via_grpc()
        
        if protocol is None:
            protocol = self.select_protocol()
        
        print(f"\n{'='*70}")
        print(f"FL ROUND - Using {protocol.upper()} Protocol")
        print(f"{'='*70}")
        
        try:
            # Train locally
            print(f"[Training] Starting local training...")
            train_metrics = self.train_local_model()
            
            # Update metrics
            round_time = time.time() - round_start
            self.round_metrics['convergence_time'] = train_metrics['training_time']
            self.round_metrics['accuracy'] = train_metrics['val_accuracy']
            self.round_metrics['success'] = True
            
            # Update RL unconditionally so GUI always shows Q-learning data
            # (USE_QL_CONVERGENCE only controls the *stopping* condition)
            if USE_RL_SELECTION and self.rl_selector_uplink and self.env_manager:
                resources = self.env_manager.get_resource_consumption()
                payload_bytes = self.round_metrics.get('payload_bytes', 12 * 1024 * 1024)
                t_calc = self._get_t_calc_for_reward(protocol, payload_bytes) if USE_COMMUNICATION_MODEL_REWARD else None
                reward = self.rl_selector_uplink.calculate_reward(
                    communication_time=self.round_metrics['communication_time'],
                    success=self.round_metrics['success'],
                    accuracy=self.round_metrics['accuracy'],
                    resource_consumption=resources,
                    t_calc=t_calc,
                )
                self.rl_selector_uplink.update_q_value(reward, done=False)
                if log_q_step is not None:
                    st = self.env_manager.get_current_state()
                    q_delta = self.rl_selector_uplink.get_last_q_delta()
                    avg_reward = (np.mean(self.rl_selector_uplink.total_rewards[-100:])
                                 if self.rl_selector_uplink.total_rewards else 0.0)
                    reward_details = self.rl_selector_uplink.get_last_reward_breakdown()
                    log_q_step(
                        client_id=self.client_id,
                        round_num=int(getattr(self, 'current_round', 0)),
                        episode=self.rl_selector_uplink.episode_count,
                        state_network=st.get('network', ''),
                        state_resource=st.get('resource', ''),
                        state_model_size=st.get('model_size', ''),
                        state_mobility=st.get('mobility', ''),
                        action=protocol,
                        reward=reward,
                        q_delta=q_delta,
                        epsilon=self.rl_selector_uplink.epsilon,
                        avg_reward_last_100=float(avg_reward),
                        converged=self.rl_selector_uplink.check_q_converged(),
                        metric_communication_time=reward_details.get('communication_time'),
                        metric_convergence_time=None,
                        metric_accuracy=reward_details.get('accuracy'),
                        metric_success=reward_details.get('success'),
                        metric_cpu_usage=reward_details.get('cpu_usage'),
                        metric_memory_usage=reward_details.get('memory_usage'),
                        metric_bandwidth_usage=reward_details.get('bandwidth_usage'),
                        metric_battery_level=reward_details.get('battery_level'),
                        metric_energy_usage=reward_details.get('energy_usage'),
                        metric_t_calc=reward_details.get('t_calc'),
                        reward_base=reward_details.get('reward_base'),
                        reward_communication_time=reward_details.get('reward_communication_time'),
                        reward_convergence_time=None,
                        reward_accuracy=reward_details.get('reward_accuracy'),
                        reward_resource_penalty=reward_details.get('reward_resource_penalty'),
                        reward_battery_penalty=reward_details.get('reward_battery_penalty'),
                        reward_t_calc_penalty=reward_details.get('reward_t_calc_penalty'),
                        reward_total=reward_details.get('reward_total'),
                        link_direction='uplink',
                    )
                print(f"[Uplink RL] Reward: {reward:.2f}, epsilon: {self.rl_selector_uplink.epsilon:.4f}")
                # Compute and log downlink reward if a downlink selection was made this round
                self._update_downlink_rl_after_reception(round_num=int(getattr(self, 'current_round', 0)))
            
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
        
        for round_num in range(num_rounds):
            self.current_round = round_num
            print(f"\n{'#'*70}")
            print(f"# ROUND {round_num + 1}/{num_rounds}")
            print(f"{'#'*70}\n")
            
            metrics = self.federated_learning_round()
            
            print(f"\n[Round {round_num + 1}] Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            
            if USE_RL_SELECTION and self.rl_selector and USE_QL_CONVERGENCE:
                self.rl_selector_uplink.end_episode()
                self.rl_selector_downlink.end_episode()
        if USE_RL_SELECTION and self.rl_selector:
            self.rl_selector.print_statistics()


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
