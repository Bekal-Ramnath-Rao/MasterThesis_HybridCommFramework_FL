"""
Unified Federated Learning Client for Emotion Recognition
with RL-based Protocol Selection

Supports: MQTT, AMQP, gRPC, QUIC, DDS
Uses Q-Learning to dynamically select the best protocol for DATA transmission
Uses MQTT for CONTROL signals (always)
Architecture: Event-driven, waits for server signals via MQTT callbacks
             Data transmission uses RL-selected protocol
"""

import os
import sys
import time
import json
import pickle
import base64
import logging
import threading
import socket
import re
import fcntl
from typing import Dict, Tuple, Optional, List, Sequence
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import protocol-specific modules
import paho.mqtt.client as mqtt
try:
    import pika
except ImportError:
    pika = None

try:
    import grpc
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))
    import federated_learning_pb2
    import federated_learning_pb2_grpc
except (ImportError, ModuleNotFoundError):
    grpc = None
    federated_learning_pb2 = None
    federated_learning_pb2_grpc = None

try:
    import asyncio
    from aioquic.asyncio import connect
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.quic.events import StreamDataReceived
    from aioquic.asyncio.protocol import QuicConnectionProtocol
except ImportError:
    asyncio = None
    connect = None
    QuicConfiguration = None
    StreamDataReceived = None
    QuicConnectionProtocol = None

try:
    import httpx
    HTTP3_CLIENT_AVAILABLE = True
except ImportError:
    HTTP3_CLIENT_AVAILABLE = False
    httpx = None

try:
    from aioquic.h3.connection import H3_ALPN, H3Connection
    from aioquic.h3.events import DataReceived, HeadersReceived, H3Event, StreamReset
    HTTP3_AVAILABLE = True
except ImportError:
    HTTP3_AVAILABLE = False
    H3Connection = None
    H3Event = None

try:
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.util import duration
    from cyclonedds.core import Qos, Policy
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.types import sequence
    from dataclasses import dataclass
    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False
    dataclass = lambda x: x
    IdlStruct = object

# Define QUIC protocol handler if available
if QuicConnectionProtocol is not None:
    class UnifiedClientQUICProtocol(QuicConnectionProtocol):
        """QUIC protocol handler for receiving server messages"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.client = None  # Set by factory function
            self._stream_buffers = {}  # Instance-specific stream buffers
        
        def quic_event_received(self, event):
            if isinstance(event, StreamDataReceived):
                try:
                    # Get or create buffer for this stream
                    stream_id = event.stream_id
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                    
                    # Append new data to buffer
                    self._stream_buffers[stream_id] += event.data
                    print(f"[QUIC] Client stream {stream_id}: received {len(event.data)} bytes, buffer now {len(self._stream_buffers[stream_id])} bytes")
                    
                    # Send flow control updates
                    self.transmit()
                    
                    # Try to decode complete messages (delimited by newline)
                    while b'\n' in self._stream_buffers[stream_id]:
                        message_data, self._stream_buffers[stream_id] = self._stream_buffers[stream_id].split(b'\n', 1)
                        if message_data:
                            try:
                                data_str = message_data.decode('utf-8')
                                message = json.loads(data_str)
                                msg_type = message.get('type', 'unknown')
                                print(f"[QUIC] Client decoded message type '{msg_type}' from stream {stream_id}")
                                # Handle message asynchronously
                                if self.client:
                                    asyncio.create_task(self.client._handle_quic_message_async(message))
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"[QUIC] Client error decoding message: {e}")
                    
                    # If stream ended and buffer has remaining data, try to process it
                    if event.end_stream and self._stream_buffers[stream_id]:
                        print(f"[QUIC] Client stream {stream_id} ended with {len(self._stream_buffers[stream_id])} bytes remaining")
                        try:
                            data_str = self._stream_buffers[stream_id].decode('utf-8')
                            message = json.loads(data_str)
                            msg_type = message.get('type', 'unknown')
                            print(f"[QUIC] Client decoded end-of-stream message type '{msg_type}'")
                            if self.client:
                                asyncio.create_task(self.client._handle_quic_message_async(message))
                            self._stream_buffers[stream_id] = b''
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"[QUIC] Client error decoding remaining buffer: {e}")
                except Exception as e:
                    print(f"[QUIC] Client error handling event: {e}")
                    import traceback
                    traceback.print_exc()
else:
    UnifiedClientQUICProtocol = None

# Define HTTP/3 protocol handler if available
if HTTP3_AVAILABLE and QuicConnectionProtocol is not None:
    class UnifiedClientHTTP3Protocol(QuicConnectionProtocol):
        """HTTP/3 protocol handler for receiving server messages"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.client = None  # Set by factory function
            self._http = None  # H3Connection instance
            self._stream_buffers = {}  # Instance-specific stream buffers
        
        def quic_event_received(self, event):
            """Handle QUIC events and convert to HTTP/3 events"""
            # Initialize H3 connection on first event
            if self._http is None:
                self._http = H3Connection(self._quic)
            
            # Convert QUIC events to HTTP/3 events
            for h3_event in self._http.handle_event(event):
                self._handle_h3_event(h3_event)
        
        def _handle_h3_event(self, event: H3Event):
            """Handle HTTP/3 events"""
            if isinstance(event, HeadersReceived):
                try:
                    stream_id = event.stream_id
                    headers = dict(event.headers)
                    status = headers.get(b":status", b"").decode()
                    print(f"[HTTP/3] Client received headers on stream {stream_id}, status: {status}")
                    
                    # Initialize buffer for this stream
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                except Exception as e:
                    print(f"[HTTP/3] Client error handling headers: {e}")
            
            elif isinstance(event, DataReceived):
                try:
                    stream_id = event.stream_id
                    # Get or create buffer for this stream
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                    
                    # Append new data to buffer
                    self._stream_buffers[stream_id] += event.data
                    print(f"[HTTP/3] Client stream {stream_id}: received {len(event.data)} bytes, buffer now {len(self._stream_buffers[stream_id])} bytes")
                    
                    # Send flow control updates
                    self.transmit()
                    
                    # If stream ended, process complete message
                    if event.end_stream:
                        try:
                            data_str = self._stream_buffers[stream_id].decode('utf-8')
                            message = json.loads(data_str)
                            msg_type = message.get('type', 'unknown')
                            print(f"[HTTP/3] Client decoded complete message type '{msg_type}' from stream {stream_id}")
                            
                            # Handle message asynchronously
                            if self.client:
                                asyncio.create_task(self.client._handle_http3_message_async(message))
                            
                            # Clear buffer
                            self._stream_buffers[stream_id] = b''
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"[HTTP/3] Client error decoding message: {e}")
                            self._stream_buffers[stream_id] = b''
                except Exception as e:
                    print(f"[HTTP/3] Client error handling data: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif isinstance(event, StreamReset):
                # Stream was reset, clear buffer
                stream_id = event.stream_id
                if stream_id in self._stream_buffers:
                    del self._stream_buffers[stream_id]
                    print(f"[HTTP/3] Client stream {stream_id} reset, cleared buffer")
else:
    UnifiedClientHTTP3Protocol = None

# Detect Docker environment and set project root
if os.path.exists('/app'):
    project_root = '/app'
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# packet_logger lives in scripts/utilities (Docker: /app/scripts/utilities, local: project_root/scripts/utilities)
_utilities_path = os.path.join(project_root, 'scripts', 'utilities')
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)

from packet_logger import init_db, log_sent_packet, log_received_packet
try:
    from q_learning_logger import init_db as init_qlearning_db, log_q_step
except ImportError:
    init_qlearning_db = None
    log_q_step = None

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from rl_q_learning_selector import QLearningProtocolSelector, EnvironmentStateManager
except ImportError:
    QLearningProtocolSelector = None
    EnvironmentStateManager = None

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Configure GPU - Force GPU 0 usage with memory growth
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Enable memory growth to prevent OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set GPU 0 as the only visible device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"[GPU] Configured to use GPU 0: {gpus[0]}")
        print(f"[GPU] Memory growth enabled")
    else:
        print("[WARNING] No GPU devices found, using CPU")
except Exception as e:
    print(f"[WARNING] GPU configuration failed: {e}, using CPU")

# Environment variables
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
USE_RL_SELECTION = os.getenv("USE_RL_SELECTION", "true").lower() == "true"
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
# When True, training ends when Q-learning value converges; when False, ends on accuracy convergence
USE_QL_CONVERGENCE = os.getenv("USE_QL_CONVERGENCE", "false").lower() == "true"
Q_CONVERGENCE_THRESHOLD = float(os.getenv("Q_CONVERGENCE_THRESHOLD", "0.01"))
Q_CONVERGENCE_PATIENCE = int(os.getenv("Q_CONVERGENCE_PATIENCE", "5"))

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", 'localhost')
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"
TOPIC_TRAINING_COMPLETE = "fl/training_complete"

# DDS Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
# FAIR CONFIG: 64 KB chunks for better DDS performance in poor networks
CHUNK_SIZE = 64 * 1024  # 64KB chunks

# DDS Data Structures (must be defined at module level for Python 3.8)
if DDS_AVAILABLE:
    from dataclasses import dataclass, field
    
    @dataclass
    class GlobalModel(IdlStruct):
        round: int
        weights: sequence[int]  # CycloneDDS sequence type for sequence<octet> in IDL
    
    @dataclass
    class GlobalModelChunk(IdlStruct):
        round: int
        chunk_id: int
        total_chunks: int
        payload: sequence[int]
        model_config_json: str = ""  # JSON string containing model configuration
    
    @dataclass
    class TrainingCommand(IdlStruct):
        round: int
        start_training: bool
        start_evaluation: bool
        training_complete: bool
    
    @dataclass
    class ModelUpdate(IdlStruct):
        client_id: int
        round: int
        weights: sequence[int]  # CycloneDDS sequence type for sequence<octet> in IDL
        num_samples: int
        loss: float
        mse: float
        mae: float
        mape: float
    
    @dataclass
    class ModelUpdateChunk(IdlStruct):
        client_id: int
        round: int
        chunk_id: int
        total_chunks: int
        payload: sequence[int]
        num_samples: int
        loss: float
        mse: float
        mae: float
        mape: float
    
    @dataclass
    class EvaluationMetrics(IdlStruct):
        client_id: int
        round: int
        num_samples: int
        loss: float
        accuracy: float
        mse: float
        mae: float
        mape: float


class UnifiedFLClient_Emotion:
    """
    Unified Federated Learning Client for Emotion Recognition
    Uses RL to select best protocol, but behaves identically to single-protocol clients
    """
    
    def __init__(self, client_id: int, num_clients: int, train_generator, validation_generator):
        """
        Initialize Unified FL Client
        
        Args:
            client_id: Unique client identifier
            num_clients: Total number of clients in FL
            train_generator: Training data generator
            validation_generator: Validation data generator
        """
        self.client_id = client_id
        self.num_clients = num_clients
        
        # Data generators
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        
        # Model
        self.model = None
        self.current_round = 0
        self.last_global_round = -1
        self.last_training_round = -1
        self.pending_start_training_round = None
        self.grpc_registered = False
        self.protocol_listeners_started = False
        self.evaluated_rounds = set()
        self.waiting_for_aggregated_model = False  # Track if we sent update and waiting for aggregated model
        self.is_active = True
        self.has_converged = False
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        
        # Training configuration
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        
        # RL Components (load from past experience when .pkl exists)
        if USE_RL_SELECTION and QLearningProtocolSelector is not None:
            # Persist Q-table in shared_data when in Docker so next run loads past experience
            if os.path.exists("/shared_data"):
                save_path = f"/shared_data/q_table_emotion_client_{client_id}.pkl"
            else:
                save_path = f"q_table_emotion_client_{client_id}.pkl"
            # Optional: load from pretrained dir first (e.g. scripts/experiments/pretrained_q_tables)
            initial_load_path = None
            pretrained_dir = os.getenv("PRETRAINED_Q_TABLE_DIR")
            if pretrained_dir:
                candidate = os.path.join(pretrained_dir, f"q_table_emotion_client_{client_id}.pkl")
                if os.path.exists(candidate):
                    initial_load_path = candidate
            self.rl_selector = QLearningProtocolSelector(
                save_path=save_path,
                initial_load_path=initial_load_path,
            )
            self.env_manager = EnvironmentStateManager()
        else:
            self.rl_selector = None
            self.env_manager = None

        # Track recent network latency samples for mobility estimation
        self.latency_history: List[float] = []
        
        # Track selected protocol and metrics
        self.selected_protocol = None
        self.round_metrics = {
            'communication_time': 0.0,
            'training_time': 0.0,
            'accuracy': 0.0,
            'success': False
        }
        self._last_rl_state = None  # for Q-learning log (state at time of action)
        
        # DDS chunk reassembly buffers (FAIR CONFIG: matching standalone)
        self.global_model_chunks = {}  # {chunk_id: payload}
        self.global_model_metadata = {}  # {round, total_chunks, model_config_json}
        
        # DDS Components
        if DDS_AVAILABLE:
            try:
                # Create DDS participant
                self.dds_participant = DomainParticipant(DDS_DOMAIN_ID)
                
                # Reliable QoS for critical control messages (registration, config, commands)
                reliable_qos = Qos(
                    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                    Policy.History.KeepLast(10),
                    Policy.Durability.TransientLocal
                )
                
                # Best effort QoS for non-critical bulk paths
                best_effort_qos = Qos(
                    Policy.Reliability.BestEffort,
                    Policy.History.KeepLast(1),
                )

                # Reliable QoS for chunked model transfer to prevent dropped chunks.
                chunk_qos = Qos(
                    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                    Policy.History.KeepLast(2048),
                    Policy.Durability.TransientLocal
                )
                
                # Create topics
                global_model_topic = Topic(self.dds_participant, "GlobalModel", GlobalModel)
                global_model_chunk_topic = Topic(self.dds_participant, "GlobalModelChunk", GlobalModelChunk)
                self.dds_update_topic = Topic(self.dds_participant, "ModelUpdate", ModelUpdate)
                update_chunk_topic = Topic(self.dds_participant, "ModelUpdateChunk", ModelUpdateChunk)
                self.dds_metrics_topic = Topic(self.dds_participant, "EvaluationMetrics", EvaluationMetrics)
                
                # Create readers (for receiving from server)
                # Use reliable QoS for chunked model data
                self.dds_global_model_reader = DataReader(self.dds_participant, global_model_topic, qos=best_effort_qos)
                self.dds_global_model_chunk_reader = DataReader(self.dds_participant, global_model_chunk_topic, qos=chunk_qos)
                
                # Create writers (for sending to server)
                # Use reliable QoS for chunked updates; metrics can remain best-effort
                self.dds_update_writer = DataWriter(self.dds_participant, self.dds_update_topic, qos=best_effort_qos)
                self.dds_update_chunk_writer = DataWriter(self.dds_participant, update_chunk_topic, qos=chunk_qos)
                self.dds_metrics_writer = DataWriter(self.dds_participant, self.dds_metrics_topic, qos=best_effort_qos)
                
                print(f"[DDS] Client {client_id} initialized on domain {DDS_DOMAIN_ID} with chunking support")
            except Exception as e:
                print(f"[DDS] Initialization failed: {e}")
                self.dds_participant = None
                self.dds_update_writer = None
                self.dds_update_chunk_writer = None
                self.dds_metrics_writer = None
                self.dds_global_model_reader = None
                self.dds_global_model_chunk_reader = None
        else:
            self.dds_participant = None
            self.dds_update_writer = None
            self.dds_update_chunk_writer = None
            self.dds_metrics_writer = None
            self.dds_global_model_reader = None
            self.dds_global_model_chunk_reader = None
        
        # Initialize packet logger and Q-learning logger
        init_db()
        if init_qlearning_db is not None:
            init_qlearning_db()
        
        # QUIC persistent connection components
        self.quic_protocol = None
        self.quic_connection_task = None
        self.quic_loop = None
        self.quic_thread = None
        # HTTP/3 persistent connection components
        self.http3_protocol = None
        self.http3_connection_task = None
        self.http3_loop = None
        self.http3_thread = None
        
        # AMQP listener components
        self.amqp_listener_connection = None
        self.amqp_listener_channel = None
        self.amqp_listener_thread = None
        
        # DDS listener components
        self.dds_listener_thread = None
        self.dds_global_model_reader = None
        self.dds_command_reader = None
        
        # gRPC listener components
        self.grpc_listener_thread = None
        self.grpc_stub = None
        
        # Initialize MQTT client for listening (always used for signal/sync)
        self.mqtt_client = mqtt.Client(client_id=f"fl_client_{client_id}", protocol=mqtt.MQTTv311)
        self.mqtt_client.max_inflight_messages_set(20)
        # FAIR CONFIG: Limited queue to 1000 messages (aligned with AMQP/gRPC)
        self.mqtt_client.max_queued_messages_set(1000)
        # FAIR CONFIG: Set max packet size to 128MB (aligned with AMQP default)
        self.mqtt_client._max_packet_size = 128 * 1024 * 1024  # 128 MB
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        print(f"\n{'='*70}")
        print(f"UNIFIED FL CLIENT - EMOTION RECOGNITION")
        print(f"{'='*70}")
        print(f"Client ID: {self.client_id}/{self.num_clients}")
        print(f"RL Protocol Selection: {'ENABLED' if USE_RL_SELECTION else 'DISABLED'}")
        print(f"{'='*70}\n")
        
        # Start protocol listeners in background threads
        self.start_all_protocol_listeners()
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print(f"Client {self.client_id} connected to MQTT broker")
            # Subscribe to topics
            self.mqtt_client.subscribe(TOPIC_GLOBAL_MODEL, qos=1)
            self.mqtt_client.subscribe(TOPIC_TRAINING_CONFIG, qos=1)
            self.mqtt_client.subscribe(TOPIC_START_TRAINING, qos=1)
            self.mqtt_client.subscribe(TOPIC_START_EVALUATION, qos=1)
            self.mqtt_client.subscribe(TOPIC_TRAINING_COMPLETE, qos=1)
            
            time.sleep(2)

            if not self.grpc_registered and not self.register_with_server_grpc():
                # Fallback only if gRPC is unavailable.
                registration_msg = json.dumps({"client_id": self.client_id})
                self.mqtt_client.publish("fl/client_register", registration_msg, qos=1)
                log_sent_packet(
                    packet_size=len(registration_msg),
                    peer="server",
                    protocol="MQTT",
                    round=0,
                    extra_info="registration_fallback"
                )
                print("  Registration fallback sent via MQTT\n")
        else:
            print(f"Client {self.client_id} failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        if not self.is_active:
            return
        try:
            log_received_packet(
                packet_size=len(msg.payload),
                peer="server",
                protocol="MQTT",
                round=self.current_round,
                extra_info=msg.topic
            )
            
            if msg.topic == TOPIC_GLOBAL_MODEL:
                self.handle_global_model(msg.payload)
            elif msg.topic == TOPIC_TRAINING_CONFIG:
                self.handle_training_config(msg.payload)
            elif msg.topic == TOPIC_START_TRAINING:
                self.handle_start_training(msg.payload)
            elif msg.topic == TOPIC_START_EVALUATION:
                self.handle_start_evaluation(msg.payload)
            elif msg.topic == TOPIC_TRAINING_COMPLETE:
                self.handle_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        # Cleanup QUIC connection
        self.cleanup()
        
        if rc == 0:
            print(f"\nClient {self.client_id} clean disconnect from broker")
            print(f"Client {self.client_id} exiting...")
            self.mqtt_client.loop_stop()
            import sys
            sys.exit(0)
        else:
            print(f"Client {self.client_id} unexpected disconnect, return code {rc}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.quic_connection_task and self.quic_loop:
            try:
                # Cancel the QUIC connection task
                self.quic_loop.call_soon_threadsafe(self.quic_connection_task.cancel)
            except:
                pass
        
        # Cleanup AMQP listener
        if self.amqp_listener_connection and not self.amqp_listener_connection.is_closed:
            try:
                self.amqp_listener_connection.close()
            except:
                pass
    
    # =========================================================================
    # PROTOCOL LISTENERS - Start all protocol listeners for receiving responses
    # =========================================================================
    
    def start_all_protocol_listeners(self):
        """Start listeners for all protocols (mirroring single-protocol implementations)"""
        if self.protocol_listeners_started:
            print("[Client] Protocol listeners already started - skipping duplicate start")
            return
        self.protocol_listeners_started = True
        print("[Client] Starting protocol listeners...")
        
        # MQTT already started in __init__
        
        # Get list of protocols that RL selector can use
        available_protocols = []
        if self.rl_selector:
            available_protocols = self.rl_selector.PROTOCOLS
            print(f"[Client] RL Selector configured for protocols: {available_protocols}")
        
        # Start AMQP listener only if in available protocols
        if pika is not None and (not available_protocols or 'amqp' in available_protocols):
            self.start_amqp_listener()
        
        # Start DDS listener only if in available protocols
        if DDS_AVAILABLE and (not available_protocols or 'dds' in available_protocols):
            self.start_dds_listener()
        
        # Start gRPC listener only if in available protocols
        if grpc is not None and (not available_protocols or 'grpc' in available_protocols):
            self.start_grpc_listener()
        
        # Start QUIC persistent connection listener only if in available protocols
        if asyncio is not None and connect is not None and (not available_protocols or 'quic' in available_protocols):
            self.start_quic_listener()
        
        if HTTP3_AVAILABLE and asyncio is not None and connect is not None and (not available_protocols or 'http3' in available_protocols):
            self.start_http3_listener()
        
        print("[Client] All protocol listeners started\n")
    
    # -------------------------------------------------------------------------
    # AMQP LISTENER
    # -------------------------------------------------------------------------
    
    def start_amqp_listener(self):
        """Start AMQP consumer thread (mirrors FL_Client_AMQP.py)"""
        def amqp_consumer_loop():
            # Retry with exponential backoff for startup race condition
            max_retries = 5
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"[AMQP] Retry {attempt}/{max_retries} after {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    
                    credentials = pika.PlainCredentials('guest', 'guest')
                    # FAIR CONFIG: heartbeat=600s for very_poor network scenarios
                    parameters = pika.ConnectionParameters(
                        host=os.getenv("AMQP_HOST", "rabbitmq-broker-unified"),
                        port=int(os.getenv("AMQP_PORT", "5672")),
                        credentials=credentials,
                        heartbeat=600,  # 10 minutes for very_poor network
                        blocked_connection_timeout=600  # Aligned with heartbeat
                    )
                    
                    self.amqp_listener_connection = pika.BlockingConnection(parameters)
                    self.amqp_listener_channel = self.amqp_listener_connection.channel()
                    
                    # Declare client-specific queues (server creates these)
                    queue_global_model = f'client_{self.client_id}_global_model'
                    queue_start_evaluation = f'client_{self.client_id}_start_evaluation'
                    queue_start_training = f'client_{self.client_id}_start_training'
                    
                    self.amqp_listener_channel.queue_declare(queue=queue_global_model, durable=True)
                    self.amqp_listener_channel.queue_declare(queue=queue_start_evaluation, durable=True)
                    self.amqp_listener_channel.queue_declare(queue=queue_start_training, durable=True)
                    
                    # Set up consumers
                    self.amqp_listener_channel.basic_consume(
                        queue=queue_global_model,
                        on_message_callback=self.on_amqp_global_model,
                        auto_ack=True
                    )
                    self.amqp_listener_channel.basic_consume(
                        queue=queue_start_evaluation,
                        on_message_callback=self.on_amqp_start_evaluation,
                        auto_ack=True
                    )
                    self.amqp_listener_channel.basic_consume(
                        queue=queue_start_training,
                        on_message_callback=self.on_amqp_start_training,
                        auto_ack=True
                    )
                    
                    print(f"[AMQP] Listener started for client {self.client_id}")
                    
                    # Start consuming (blocks in this thread)
                    self.amqp_listener_channel.start_consuming()
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"[AMQP] Listener failed after {max_retries} attempts: {e}")
                        import traceback
                        traceback.print_exc()
                    else:
                        print(f"[AMQP] Connection attempt {attempt + 1} failed: {e}")
        
        self.amqp_listener_thread = threading.Thread(target=amqp_consumer_loop, daemon=True, name=f"AMQP-Listener-{self.client_id}")
        self.amqp_listener_thread.start()
    
    def on_amqp_global_model(self, ch, method, properties, body):
        """AMQP callback: received global model"""
        try:
            log_received_packet(
                packet_size=len(body),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="global_model"
            )
            
            data = json.loads(body.decode())
            round_num = data['round']
            print(f"[AMQP] Client {self.client_id} received global model for round {round_num}")
            
            # Deserialize weights
            if 'quantized_data' in data:
                compressed_data = pickle.loads(base64.b64decode(data['quantized_data']))
                weights = self.quantizer.decompress_global_model(compressed_data) if self.quantizer else None
            else:
                weights = self.deserialize_weights(data['weights'])
            
            self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=data.get('model_config'),
                source="AMQP"
            )
                
        except Exception as e:
            print(f"[AMQP] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_amqp_start_evaluation(self, ch, method, properties, body):
        """AMQP callback: received start evaluation signal"""
        try:
            log_received_packet(
                packet_size=len(body),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="start_evaluation"
            )
            
            data = json.loads(body.decode())
            round_num = data['round']
            
            if round_num == self.current_round and round_num not in self.evaluated_rounds:
                self.evaluated_rounds.add(round_num)  # Add BEFORE evaluate to prevent race condition
                print(f"[AMQP] Client {self.client_id} starting evaluation for round {round_num}")
                self.evaluate_model()
                print(f"[AMQP] Client {self.client_id} evaluation completed for round {round_num}.")
                
        except Exception as e:
            print(f"[AMQP] Client {self.client_id} error handling evaluation signal: {e}")
    
    def on_amqp_start_training(self, ch, method, properties, body):
        """AMQP callback: received start training signal"""
        try:
            log_received_packet(
                packet_size=len(body),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="start_training"
            )
            
            data = json.loads(body.decode())
            round_num = data['round']
            
            # Check if model is initialized
            if self.model is None:
                self._defer_start_training(round_num, "AMQP")
                self.pending_start_training_round = round_num
                return
            
            # Check for duplicate training signals
            if self.last_training_round == round_num:
                print(f"[AMQP] Client {self.client_id} ignoring duplicate start training for round {round_num}")
                return
            
            # Check if we're ready for this round
            if self.current_round == 0 and round_num == 1:
                self.current_round = round_num
                self.last_training_round = round_num
                print(f"\n[AMQP] Client {self.client_id} starting training for round {round_num}...")
                self.train_local_model()
            elif round_num >= self.current_round and round_num <= self.current_round + 1:
                self.current_round = round_num
                self.last_training_round = round_num
                print(f"\n[AMQP] Client {self.client_id} starting training for round {round_num}...")
                self.train_local_model()
            else:
                print(f"[AMQP] Client {self.client_id} skipping training signal for round {round_num} (current: {self.current_round})")
                
        except Exception as e:
            print(f"[AMQP] Client {self.client_id} error handling training signal: {e}")
    
    # -------------------------------------------------------------------------
    # DDS LISTENER
    # -------------------------------------------------------------------------
    
    def start_dds_listener(self):
        """Start DDS reader polling thread (mirrors FL_Client_DDS.py)"""
        def dds_listener_loop():
            try:
                # Create readers for GlobalModel and TrainingCommand
                from cyclonedds.core import Qos, Policy
                from cyclonedds.util import duration
                from cyclonedds.topic import Topic
                
                reliable_qos = Qos(
                    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                    Policy.History.KeepLast(10),
                    Policy.Durability.TransientLocal
                )
                
                topic_global_model = Topic(self.dds_participant, "GlobalModel", GlobalModel)
                topic_command = Topic(self.dds_participant, "TrainingCommand", TrainingCommand)
                
                self.dds_global_model_reader = DataReader(self.dds_participant, topic_global_model, qos=reliable_qos)
                self.dds_command_reader = DataReader(self.dds_participant, topic_command, qos=reliable_qos)
                
                print(f"[DDS] Listener started for client {self.client_id}")
                
                # Polling loop
                while True:
                    # Check for global model
                    for sample in self.dds_global_model_reader.take():
                        if sample:
                            self.on_dds_global_model(sample)
                    
                    # Check for commands
                    for sample in self.dds_command_reader.take():
                        if sample:
                            self.on_dds_command(sample)
                    
                    time.sleep(0.1)  # Poll every 100ms
                    
            except Exception as e:
                print(f"[DDS] Listener error: {e}")
                import traceback
                traceback.print_exc()
        
        self.dds_listener_thread = threading.Thread(target=dds_listener_loop, daemon=True, name=f"DDS-Listener-{self.client_id}")
        self.dds_listener_thread.start()
    
    def on_dds_global_model(self, sample):
        """DDS callback: received global model"""
        try:
            round_num = sample.round
            print(f"[DDS] Client {self.client_id} received global model for round {round_num}")
            
            log_received_packet(
                packet_size=len(sample.weights),
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="global_model"
            )
            
            # Deserialize weights (DDS uses sequence[int])
            weights_bytes = bytes(sample.weights)
            weights = pickle.loads(weights_bytes)
            
            # Update model
            if self.model and round_num > self.last_global_round:
                self.model.set_weights(weights)
                self.last_global_round = round_num
                print(f"[DDS] Client {self.client_id} updated model weights for round {round_num}")
                
        except Exception as e:
            print(f"[DDS] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_dds_command(self, sample):
        """DDS callback: received training command"""
        try:
            log_received_packet(
                packet_size=32,  # Approximate size
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="command"
            )
            
            round_num = sample.round
            
            # Handle start_training command
            if sample.start_training:
                # Check if model is initialized
                if self.model is None:
                    self._defer_start_training(round_num, "DDS")
                    self.pending_start_training_round = round_num
                    return
                
                # Check for duplicate training signals
                if self.last_training_round == round_num:
                    print(f"[DDS] Client {self.client_id} ignoring duplicate start training for round {round_num}")
                    return
                
                # Check if we're ready for this round
                if self.current_round == 0 and round_num == 1:
                    self.current_round = round_num
                    self.last_training_round = round_num
                    print(f"\n[DDS] Client {self.client_id} starting training for round {round_num}...")
                    self.train_local_model()
                elif round_num >= self.current_round and round_num <= self.current_round + 1:
                    self.current_round = round_num
                    self.last_training_round = round_num
                    print(f"\n[DDS] Client {self.client_id} starting training for round {round_num}...")
                    self.train_local_model()
                else:
                    print(f"[DDS] Client {self.client_id} skipping training signal for round {round_num} (current: {self.current_round})")
            
            # Handle start_evaluation command
            if sample.start_evaluation and sample.round == self.current_round:
                if sample.round not in self.evaluated_rounds:
                    self.evaluated_rounds.add(sample.round)  # Add BEFORE evaluate to prevent race condition
                    print(f"[DDS] Client {self.client_id} starting evaluation for round {sample.round}")
                    self.evaluate_model()
                    print(f"[DDS] Client {self.client_id} evaluation completed for round {sample.round}.")
                    
        except Exception as e:
            print(f"[DDS] Client {self.client_id} error handling command: {e}")
    
    # -------------------------------------------------------------------------
    # gRPC LISTENER
    # -------------------------------------------------------------------------
    
    def start_grpc_listener(self):
        """Start gRPC polling thread (mirrors FL_Client_gRPC.py)"""
        def grpc_listener_loop():
            try:
                # Create gRPC channel and stub
                grpc_host = os.getenv("GRPC_HOST", "fl-server-unified-emotion")
                grpc_port = os.getenv("GRPC_PORT", "50051")
                options = [
                    ('grpc.max_send_message_length', 128 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 128 * 1024 * 1024),
                    ('grpc.keepalive_time_ms', 600000),
                    ('grpc.keepalive_timeout_ms', 60000),
                ]
                channel = grpc.insecure_channel(f"{grpc_host}:{grpc_port}", options=options)
                self.grpc_stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
                
                print(f"[gRPC] Listener started for client {self.client_id}")
                
                # Polling loop
                while True:
                    try:
                        # Poll control-plane status first (gRPC-only signaling in unified RL mode)
                        status_request = federated_learning_pb2.StatusRequest(client_id=self.client_id)
                        status = self.grpc_stub.CheckTrainingStatus(status_request)
                        if status.should_train and self.is_active:
                            self.handle_start_training(
                                json.dumps({'round': status.current_round}).encode(),
                                source="gRPC"
                            )
                        if status.should_evaluate and self.is_active:
                            self.handle_start_evaluation(json.dumps({'round': status.current_round}).encode())

                        # Poll for global model (request round 0 to get latest available)
                        # This allows client to get new model even if current_round hasn't updated yet
                        request = federated_learning_pb2.ModelRequest(
                            client_id=self.client_id,
                            round=0  # 0 means "give me latest model"
                        )
                        response = self.grpc_stub.GetGlobalModel(request)
                        
                        # Accept model if:
                        # 1. It's a newer round, OR
                        # 2. Same round but we're waiting for aggregated model (after sending update)
                        if response.available and (response.round > self.last_global_round or 
                                                   (response.round == self.current_round and self.waiting_for_aggregated_model)):
                            self.on_grpc_global_model(response)
                            
                    except grpc.RpcError:
                        pass  # Expected when no new model
                    except Exception as e:
                        print(f"[gRPC] Listener poll error: {e}")
                    
                    time.sleep(1)  # Poll every second
                    
            except Exception as e:
                print(f"[gRPC] Listener error: {e}")
                import traceback
                traceback.print_exc()
        
        self.grpc_listener_thread = threading.Thread(target=grpc_listener_loop, daemon=True, name=f"gRPC-Listener-{self.client_id}")
        self.grpc_listener_thread.start()
    
    def on_grpc_global_model(self, response):
        """gRPC callback: received global model"""
        try:
            round_num = response.round
            print(f"[gRPC] Client {self.client_id} received global model for round {round_num}")
            
            log_received_packet(
                packet_size=len(response.weights),
                peer="server",
                protocol="gRPC",
                round=self.current_round,
                extra_info="global_model"
            )
            
            # Deserialize weights
            weights = pickle.loads(response.weights)

            model_config = json.loads(response.model_config) if response.model_config else None
            applied = self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=model_config,
                source="gRPC"
            )
            if applied:
                self.waiting_for_aggregated_model = False  # Clear flag: received aggregated model

                # Check if should evaluate
                status_request = federated_learning_pb2.StatusRequest(client_id=self.client_id)
                status = self.grpc_stub.CheckTrainingStatus(status_request)

                if status.should_evaluate and round_num == self.current_round:
                    if round_num not in self.evaluated_rounds:
                        self.evaluated_rounds.add(round_num)  # Add BEFORE evaluate to prevent race condition
                        print(f"[gRPC] Client {self.client_id} starting evaluation for round {round_num}")
                        self.evaluate_model()
                        print(f"[gRPC] Client {self.client_id} evaluation completed for round {round_num}.")
                        
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # QUIC LISTENER
    # =========================================================================
    
    def start_quic_listener(self):
        """Start QUIC persistent connection in background thread"""
        if self.quic_thread is None or not self.quic_thread.is_alive():
            # Start the event loop thread
            self.quic_thread = threading.Thread(
                target=self._run_quic_loop,
                daemon=True,
                name=f"QUIC-Client-{self.client_id}"
            )
            self.quic_thread.start()
            
            # Wait for event loop to be ready
            max_wait = 2
            waited = 0
            while self.quic_loop is None and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1
            
            # Schedule the connection task in the event loop
            if self.quic_loop:
                self.quic_connection_task = asyncio.run_coroutine_threadsafe(
                    self._quic_connection_loop(),
                    self.quic_loop
                )
                time.sleep(2)  # Wait for connection attempt
            
            print(f"[QUIC] Listener started for client {self.client_id}")
    
    # =========================================================================
    # HTTP/3 LISTENER
    # =========================================================================
    
    def start_http3_listener(self):
        """Start HTTP/3 persistent connection in background thread"""
        if not HTTP3_AVAILABLE:
            return
        
        if self.http3_thread is None or not self.http3_thread.is_alive():
            # Start the event loop thread
            self.http3_thread = threading.Thread(
                target=self._run_http3_loop,
                daemon=True,
                name=f"HTTP3-Client-{self.client_id}"
            )
            self.http3_thread.start()
            
            # Wait for event loop to be ready
            max_wait = 2
            waited = 0
            while self.http3_loop is None and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1
            
            # Schedule the connection task in the event loop
            if self.http3_loop:
                self.http3_connection_task = asyncio.run_coroutine_threadsafe(
                    self._http3_connection_loop(),
                    self.http3_loop
                )
                time.sleep(2)  # Wait for connection attempt
            
            print(f"[HTTP/3] Listener started for client {self.client_id}")
    
    def handle_global_model(self, payload):
        """Receive and apply global model from server"""
        try:
            data = json.loads(payload.decode()) if isinstance(payload, bytes) else payload
            round_num = data.get('round', 0)
            
            # Check for duplicate (already processed this exact model)
            if hasattr(self, 'last_global_round') and self.last_global_round == round_num and self.model is not None:
                print(f"Client {self.client_id} ignoring duplicate global model for round {round_num}")
                return
            
            print(f"Client {self.client_id} received global model (round {round_num})")
            
            # Decompress/deserialize weights
            if 'quantized_data' in data:
                # Handle quantized/compressed data
                compressed_data = data['quantized_data']
                if isinstance(compressed_data, str):
                    import base64, pickle
                    compressed_data = pickle.loads(base64.b64decode(compressed_data.encode('utf-8')))
                if hasattr(self, 'quantization') and self.quantization is not None:
                    weights = self.quantization.decompress(compressed_data)
                elif hasattr(self, 'quantizer') and self.quantizer is not None:
                    weights = self.quantizer.decompress(compressed_data)
                else:
                    weights = compressed_data
                print(f"Client {self.client_id} decompressed quantized model")
            else:
                # Normal weights
                if 'weights' in data:
                    encoded_weights = data['weights']
                    if isinstance(encoded_weights, str):
                        import base64, pickle
                        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
                        weights = pickle.loads(serialized)
                    else:
                        weights = encoded_weights
                else:
                    weights = data.get('parameters', [])
            
            self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=data.get('model_config'),
                source="MQTT"
            )
            
        except Exception as e:
            print(f"Client {self.client_id} ERROR in handle_global_model: {e}")
            import traceback
            traceback.print_exc()

    def _apply_global_model_weights(self, round_num, weights, model_config=None, source="UNKNOWN"):
        """Initialize model if needed and apply incoming global weights."""
        if self.model is None:
            if model_config:
                print(f"[{source}] Client {self.client_id} initializing model from received configuration...")
                self.model = self.build_model_from_config(model_config)
                print(f"[{source}] Client {self.client_id} model built successfully")
            else:
                print(f"[{source}] Client {self.client_id} waiting for model_config to initialize model")
                return False

        if round_num >= self.last_global_round:
            self.model.set_weights(weights)
            self.current_round = max(self.current_round, round_num)
            self.last_global_round = round_num
            print(f"[{source}] Client {self.client_id} updated model weights (round {round_num})")

            if self.pending_start_training_round is not None:
                pending_round = self.pending_start_training_round
                self.pending_start_training_round = None
                if self.last_training_round != pending_round and self.is_active:
                    self.current_round = max(self.current_round, pending_round)
                    self.last_training_round = pending_round
                    print(f"[{source}] Client {self.client_id} processing deferred start_training for round {pending_round}")
                    self.train_local_model()
            return True

        return False
    
    def handle_training_config(self, payload):
        """Update training configuration"""
        self.training_config = json.loads(payload.decode())
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    def handle_start_training(self, payload, source="MQTT"):
        """Start local training when server signals"""
        if not self.is_active:
            return
        data = json.loads(payload.decode())
        round_num = data['round']
        
        # Check if model is initialized - WAIT for global model
        if self.model is None:
            self._defer_start_training(round_num, source)
            self.pending_start_training_round = round_num
            return
        
        # Check for duplicate training signals
        if self.last_training_round == round_num:
            print(f"Client {self.client_id} ignoring duplicate start training for round {round_num}")
            return
        
        # Check if we're ready for this round
        if self.current_round == 0 and round_num == 1:
            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            self.train_local_model()
        elif round_num >= self.current_round and round_num <= self.current_round + 1:
            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            self.train_local_model()
        else:
            print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
    
    def handle_start_evaluation(self, payload):
        """Start evaluation when server signals"""
        if not self.is_active:
            return
        data = json.loads(payload.decode())
        round_num = data['round']
        if round_num == self.current_round:
            if round_num in self.evaluated_rounds:
                print(f"Client {self.client_id} ignoring duplicate evaluation for round {round_num}")
                return
            self.evaluated_rounds.add(round_num)  # Add BEFORE evaluate to prevent race condition
            print(f"Client {self.client_id} starting evaluation for round {round_num}...")
            self.evaluate_model()
            print(f"Client {self.client_id} evaluation completed for round {round_num}.")
        else:
            print(f"Client {self.client_id} skipping evaluation signal for round {round_num} (current: {self.current_round})")
    
    def handle_training_complete(self):
        """Handle training completion signal from server"""
        self.is_active = False
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nDisconnecting from server...")
        time.sleep(1)
        self.mqtt_client.disconnect()
        print(f"Client {self.client_id} disconnected successfully.")

    def register_with_server_grpc(self) -> bool:
        """Register this client using gRPC to drive round-0 model delivery."""
        if self.grpc_registered:
            return True
        if grpc is None or federated_learning_pb2 is None or federated_learning_pb2_grpc is None:
            print(f"[gRPC] Client {self.client_id} registration skipped: gRPC not available")
            return False

        try:
            grpc_host = os.getenv("GRPC_HOST", "fl-server-unified-emotion")
            grpc_port = int(os.getenv("GRPC_PORT", "50051"))
            options = [
                ('grpc.max_send_message_length', 128 * 1024 * 1024),
                ('grpc.max_receive_message_length', 128 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 600000),
                ('grpc.keepalive_timeout_ms', 60000),
            ]
            channel = grpc.insecure_channel(f'{grpc_host}:{grpc_port}', options=options)
            stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
            response = stub.RegisterClient(
                federated_learning_pb2.ClientRegistration(client_id=self.client_id)
            )
            if response.success:
                self.grpc_registered = True
                print(f"[gRPC] Client {self.client_id} registered successfully")
                log_sent_packet(
                    packet_size=len(str(self.client_id)),
                    peer="server",
                    protocol="gRPC",
                    round=0,
                    extra_info="registration"
                )
                channel.close()
                return True

            print(f"[gRPC] Registration failed: {response.message}")
            channel.close()
            return False
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} registration error: {e}")
            return False

    def _defer_start_training(self, round_num: int, source: str):
        """Defer start_training until the initial/global model is available."""
        if self.pending_start_training_round != round_num:
            print(f"[{source}] Client {self.client_id} deferring start_training round {round_num} until global model arrives")
    
    def measure_network_condition(self):
        """
        Measure current network condition (latency / bandwidth estimate)
        and update the RL environment state manager.
        """
        if not self.env_manager:
            return

        try:
            target_host = MQTT_BROKER

            # Estimate latency by timing TCP connection to MQTT broker.
            # This avoids requiring external tools like `ping` inside the container.
            latencies: List[float] = []
            port = int(os.getenv("MQTT_PORT", "1883"))
            for _ in range(3):
                start = time.time()
                try:
                    with socket.create_connection(
                        (target_host, port), timeout=2
                    ):
                        pass
                    elapsed_ms = (time.time() - start) * 1000.0
                    latencies.append(elapsed_ms)
                except OSError:
                    # Treat failures as high latency samples
                    latencies.append(500.0)

            if latencies:
                latency_ms = sum(latencies) / len(latencies)
            else:
                # Fallback conservative default
                latency_ms = 300.0

            # Rough bandwidth estimate based on latency bucket
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

            condition = self.env_manager.detect_network_condition(
                latency_ms, bandwidth_mbps
            )
            self.env_manager.update_network_condition(condition)

            # Update mobility based on variability of recent latency samples.
            # Higher jitter -> higher inferred mobility level.
            try:
                self.latency_history.append(latency_ms)
                # Keep a rolling window to bound memory and smooth behaviour
                if len(self.latency_history) > 20:
                    self.latency_history.pop(0)

                mobility = "static"
                if len(self.latency_history) >= 5:
                    avg = sum(self.latency_history) / len(self.latency_history)
                    variance = sum(
                        (x - avg) ** 2 for x in self.latency_history
                    ) / len(self.latency_history)
                    stddev = variance ** 0.5

                    # Simple buckets for mobility based on latency jitter
                    if stddev < 5:
                        mobility = "static"
                    elif stddev < 20:
                        mobility = "low"
                    elif stddev < 50:
                        mobility = "medium"
                    else:
                        mobility = "high"

                    self.env_manager.update_mobility(mobility)
            except Exception as e:
                print(f"[Mobility] Failed to update mobility level: {e}")

            print(
                f"[Network] latency={latency_ms:.1f} ms, "
                f"est_bandwidth={bandwidth_mbps:.1f} Mbps -> "
                f"condition={condition}"
            )
        except Exception as e:
            print(f"[Network] Failed to measure network condition: {e}")

    def select_protocol(self) -> str:
        """
        Select protocol using RL based on current environment and network conditions
        
        Returns:
            Selected protocol name: 'mqtt', 'amqp', 'grpc', 'quic', or 'dds'
        """
        if USE_RL_SELECTION and self.rl_selector and self.env_manager:
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory().percent
                
                resource_level = self.env_manager.detect_resource_level(cpu, memory)
                self.env_manager.update_resource_level(resource_level)

                # Update network condition in the RL environment based on
                # current connectivity to the aggregation server / broker.
                self.measure_network_condition()

                state = self.env_manager.get_current_state()
                # Use training mode based on USE_QL_CONVERGENCE flag
                # If False: pure exploitation (use best known protocol)
                # If True: epsilon-greedy (explore and learn)
                training_mode = USE_QL_CONVERGENCE
                protocol = self.rl_selector.select_protocol(state, training=training_mode)
                
                print(f"\n[RL Protocol Selection]")
                print(f"  CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
                print(f"  State: {state}")
                print(f"  Selected Protocol: {protocol.upper()}")
                print(f"  Round: {self.current_round}\n")
                
                self.selected_protocol = protocol
                return protocol
            except Exception as e:
                print(f"[RL Selection] Error: {e}, using MQTT as fallback")
                return 'mqtt'
        else:
            # Default to MQTT if RL not enabled
            return 'mqtt'
    
    def build_model_from_config(self, model_config):
        """Build model from server's architecture definition"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential()
        model.add(Input(shape=tuple(model_config['input_shape'])))
        
        for layer_config in model_config['layers']:
            if layer_config['type'] == 'Conv2D':
                model.add(Conv2D(
                    filters=layer_config['filters'],
                    kernel_size=tuple(layer_config['kernel_size']),
                    activation=layer_config.get('activation')
                ))
            elif layer_config['type'] == 'MaxPooling2D':
                model.add(MaxPooling2D(pool_size=tuple(layer_config['pool_size'])))
            elif layer_config['type'] == 'Dropout':
                model.add(Dropout(layer_config['rate']))
            elif layer_config['type'] == 'Flatten':
                model.add(Flatten())
            elif layer_config['type'] == 'Dense':
                model.add(Dense(
                    units=layer_config['units'],
                    activation=layer_config.get('activation')
                ))
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )

        # Dynamically categorize model size for RL state based on parameter count
        try:
            if self.env_manager is not None:
                total_params = model.count_params()
                if total_params < 1e5:
                    size_category = "small"
                elif total_params < 1e7:
                    size_category = "medium"
                else:
                    size_category = "large"
                self.env_manager.update_model_size(size_category)
                print(
                    f"[RL] Model params={total_params} -> "
                    f"size_category={size_category}"
                )
        except Exception as e:
            print(f"[RL] Failed to update model size category: {e}")

        return model
    
    def serialize_weights(self, weights):
        """Serialize model weights for transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded
    
    def split_into_chunks(self, data):
        """Split serialized data into chunks of CHUNK_SIZE (for DDS)"""
        chunks = []
        for i in range(0, len(data), CHUNK_SIZE):
            chunks.append(data[i:i + CHUNK_SIZE])
        return chunks
    
    def send_model_update_chunked(self, round_num, serialized_weights, num_samples, loss, mse, mae, mape):
        """Send model update as chunks via DDS"""
        chunks = self.split_into_chunks(serialized_weights)
        total_chunks = len(chunks)
        
        print(f"Client {self.client_id}: Sending model update in {total_chunks} chunks ({len(serialized_weights)} bytes total)")
        
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = ModelUpdateChunk(
                client_id=self.client_id,
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                num_samples=num_samples,
                loss=loss,
                mse=mse,
                mae=mae,
                mape=mape
            )
            self.dds_update_chunk_writer.write(chunk)
            # Reliable QoS handles delivery, no need for artificial delay
            if (chunk_id + 1) % 20 == 0:  # Progress update every 20 chunks
                print(f"  Sent {chunk_id + 1}/{total_chunks} chunks")
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from server"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights
    
    def train_local_model(self):
        """Train model on local data and send updates to server via RL-selected protocol"""
        start_time = time.time()
        
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        
        try:
            steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
            val_steps = int(os.getenv("VAL_STEPS", "25"))
        except Exception:
            steps_per_epoch = 100
            val_steps = 25
        
        # Train the model using generator
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            verbose=2
        )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        num_samples = self.train_generator.n
        
        # Prepare training metrics
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1])
        }
        
        self.round_metrics['training_time'] = time.time() - start_time
        self.round_metrics['accuracy'] = metrics['val_accuracy']
        
        # Select protocol based on RL
        protocol = self.select_protocol()
        
        # Send model update via selected protocol
        update_message = {
            "client_id": self.client_id,
            "round": self.current_round,
            "weights": self.serialize_weights(updated_weights),
            "num_samples": num_samples,
            "metrics": metrics,
            "protocol": protocol
        }
        
        comm_start = time.time()
        success = False
        protocols_to_try = [protocol, 'amqp', 'mqtt', 'grpc', 'quic', 'http3', 'dds']  # AMQP second in fallback for testing
        
        for attempt_protocol in protocols_to_try:
            if success:
                break
            try:
                if attempt_protocol == 'mqtt':
                    self._send_via_mqtt(update_message)
                    success = True
                elif attempt_protocol == 'amqp' and pika is not None:
                    self._send_via_amqp(update_message)
                    success = True
                elif attempt_protocol == 'grpc' and grpc is not None:
                    self._send_via_grpc(update_message)
                    success = True
                elif attempt_protocol == 'quic' and asyncio is not None and self.quic_protocol is not None:
                    # Only try QUIC if connection is established
                    self._send_via_quic(update_message)
                    success = True
                elif attempt_protocol == 'http3' and HTTP3_AVAILABLE and asyncio is not None and self.http3_protocol is not None:
                    # Only try HTTP/3 if connection is established
                    self._send_via_http3(update_message)
                    success = True
                elif attempt_protocol == 'dds' and DDS_AVAILABLE:
                    self._send_via_dds(update_message)
                    success = True
            except Exception as e:
                if attempt_protocol == protocol:
                    print(f"Client {self.client_id} WARNING: {protocol} failed ({e}), trying fallback...")
                continue
        
        if success:
            self.round_metrics['communication_time'] = time.time() - comm_start
            self.round_metrics['success'] = True
        else:
            print(f"Client {self.client_id} ERROR: All protocols failed!")
            self.round_metrics['success'] = False
    
    def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server via RL-selected protocol"""
        if not self.is_active:
            return

        loss, accuracy = self.model.evaluate(
            self.validation_generator,
            verbose=0
        )
        
        num_samples = self.validation_generator.n
        
        # Select protocol based on RL (store state for Q-learning log)
        protocol = self.select_protocol()
        if self.env_manager is not None:
            self._last_rl_state = self.env_manager.get_current_state()
        
        metrics_message = {
            "client_id": self.client_id,
            "round": self.current_round,
            "num_samples": num_samples,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "protocol": protocol
        }
        
        comm_start = time.time()
        try:
            if protocol == 'mqtt':
                self._send_metrics_via_mqtt(metrics_message)
            elif protocol == 'amqp':
                self._send_metrics_via_amqp(metrics_message)
            elif protocol == 'grpc':
                self._send_metrics_via_grpc(metrics_message)
            elif protocol == 'quic' and self.quic_protocol is not None:
                # Only try QUIC if connection is established
                self._send_metrics_via_quic(metrics_message)
            elif protocol == 'quic':
                # QUIC not connected, fallback to MQTT
                print(f"Client {self.client_id} WARNING: QUIC not connected, falling back to MQTT for metrics")
                self._send_metrics_via_mqtt(metrics_message)
            elif protocol == 'http3' and self.http3_protocol is not None:
                # Only try HTTP/3 if connection is established
                self._send_metrics_via_http3(metrics_message)
            elif protocol == 'http3':
                # HTTP/3 not connected, fallback to MQTT
                print(f"Client {self.client_id} WARNING: HTTP/3 not connected, falling back to MQTT for metrics")
                self._send_metrics_via_mqtt(metrics_message)
            elif protocol == 'dds':
                self._send_metrics_via_dds(metrics_message)
            else:
                print(f"Client {self.client_id} ERROR: Unknown protocol {protocol}, falling back to MQTT")
                self._send_metrics_via_mqtt(metrics_message)
            
            self.round_metrics['communication_time'] = time.time() - comm_start
            print(f"Client {self.client_id} sent evaluation metrics for round {self.current_round}")
            print(f"Evaluation metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            # RL update and optional Q-convergence end condition
            if USE_RL_SELECTION and self.rl_selector and self.env_manager:
                try:
                    resources = self.env_manager.get_resource_consumption()
                    reward = self.rl_selector.calculate_reward(
                        self.round_metrics['communication_time'],
                        self.round_metrics['success'],
                        self.round_metrics.get('training_time', 0.0),
                        self.round_metrics['accuracy'],
                        resources,
                    )
                    self.rl_selector.update_q_value(reward, next_state=None, done=True)
                    q_delta = self.rl_selector.get_last_q_delta()
                    avg_reward = (np.mean(self.rl_selector.total_rewards[-100:])
                                 if self.rl_selector.total_rewards else 0.0)
                    self.rl_selector.end_episode()
                    q_converged = self.rl_selector.check_q_converged(
                        threshold=Q_CONVERGENCE_THRESHOLD,
                        patience=Q_CONVERGENCE_PATIENCE,
                    )
                    if log_q_step is not None and self._last_rl_state is not None:
                        st = self._last_rl_state
                        log_q_step(
                            client_id=self.client_id,
                            round_num=self.current_round,
                            episode=self.rl_selector.episode_count - 1,
                            state_network=st.get('network', ''),
                            state_resource=st.get('resource', ''),
                            state_model_size=st.get('model_size', ''),
                            state_mobility=st.get('mobility', ''),
                            action=self.selected_protocol or 'mqtt',
                            reward=reward,
                            q_delta=q_delta,
                            epsilon=self.rl_selector.epsilon,
                            avg_reward_last_100=float(avg_reward),
                            converged=USE_QL_CONVERGENCE and q_converged,
                        )
                    if USE_QL_CONVERGENCE and q_converged:
                        self.has_converged = True
                        print(f"[Client {self.client_id}] Q-learning convergence reached at round {self.current_round}")
                        # Reset epsilon for potential re-exploration if network conditions change
                        self.rl_selector.reset_epsilon()
                        self._notify_convergence_to_server()
                        self._disconnect_after_convergence()
                        return
                except Exception as rl_e:
                    print(f"[Client {self.client_id}] RL update error: {rl_e}")
            if not USE_QL_CONVERGENCE:
                self._update_client_convergence_and_maybe_disconnect(loss)
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics: {e}")
            import traceback
            traceback.print_exc()

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
            self._notify_convergence_to_server()
            self._disconnect_after_convergence()

    def _notify_convergence_to_server(self):
        """Notify server this client is converged using gRPC control signal."""
        if grpc is None or federated_learning_pb2 is None or federated_learning_pb2_grpc is None:
            print(f"[gRPC] Client {self.client_id} convergence signal skipped: gRPC unavailable")
            return
        try:
            grpc_host = os.getenv("GRPC_HOST", "fl-server-unified-emotion")
            grpc_port = int(os.getenv("GRPC_PORT", "50051"))
            options = [
                ('grpc.max_send_message_length', 128 * 1024 * 1024),
                ('grpc.max_receive_message_length', 128 * 1024 * 1024),
                ('grpc.keepalive_time_ms', 600000),
                ('grpc.keepalive_timeout_ms', 60000),
            ]
            channel = grpc.insecure_channel(f'{grpc_host}:{grpc_port}', options=options)
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
        print(f"[Client {self.client_id}] Disconnecting after local convergence")
        try:
            self.cleanup()
        except Exception:
            pass
        try:
            self.mqtt_client.disconnect()
        except Exception:
            pass
    
    # ============================================================================
    # Protocol-Specific Send Methods (Data Transmission)
    # ============================================================================
    
    def _send_via_mqtt(self, message: dict):
        """Send model update via MQTT"""
        try:
            payload = json.dumps(message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via MQTT - size: {payload_size_mb:.2f} MB")
            
            result = self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, payload, qos=1)
            # FAIR FIX: Use shorter timeout (5s) aligned with other protocols, or non-blocking check
            # MQTT QoS 1 ensures delivery, so we don't need to wait for full acknowledgment
            # This makes MQTT behavior similar to AMQP/gRPC which return immediately after send
            if result.rc == mqtt.MQTT_ERR_NO_CONN:
                raise Exception("MQTT not connected")
            # Only wait briefly to ensure message is queued (not blocking for full delivery)
            result.wait_for_publish(timeout=5)
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="MQTT",
                round=self.current_round,
                extra_info="model_update"
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Client {self.client_id} sent model update for round {self.current_round} via MQTT")
                print(f"Training metrics - Loss: {message['metrics']['loss']:.4f}, Accuracy: {message['metrics']['accuracy']:.4f}")
            else:
                raise Exception(f"MQTT publish failed with rc={result.rc}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via MQTT: {e}")
            raise
    
    def _send_metrics_via_mqtt(self, message: dict):
        """Send metrics via MQTT"""
        try:
            payload = json.dumps(message)
            result = self.mqtt_client.publish(TOPIC_CLIENT_METRICS, payload, qos=1)
            # FAIR FIX: Use shorter timeout (5s) aligned with other protocols
            # Metrics are small, so 5s is sufficient for queue confirmation
            if result.rc == mqtt.MQTT_ERR_NO_CONN:
                raise Exception("MQTT not connected")
            result.wait_for_publish(timeout=5)
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="MQTT",
                round=self.current_round,
                extra_info="metrics"
            )
            
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(f"MQTT publish failed with rc={result.rc}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via MQTT: {e}")
            raise
    
    def _send_via_amqp(self, message: dict):
        """Send model update via AMQP (RabbitMQ)"""
        if pika is None:
            raise ImportError("pika module not available for AMQP")
        
        try:
            # Get AMQP config
            amqp_host = os.getenv("AMQP_HOST", "localhost")
            amqp_port = int(os.getenv("AMQP_PORT", "5672"))
            amqp_user = os.getenv("AMQP_USER", "guest")
            amqp_password = os.getenv("AMQP_PASSWORD", "guest")
            
            # Connect to RabbitMQ
            credentials = pika.PlainCredentials(amqp_user, amqp_password)
            # FAIR CONFIG: heartbeat=600s for very_poor network scenarios
            parameters = pika.ConnectionParameters(
                host=amqp_host,
                port=amqp_port,
                credentials=credentials,
                heartbeat=600,  # 10 minutes for very_poor network
                blocked_connection_timeout=600,  # Aligned with heartbeat
                connection_attempts=3,
                retry_delay=2
            )
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # Declare exchange and send message
            channel.exchange_declare(exchange='fl_client_updates', exchange_type='direct', durable=True)
            
            payload = json.dumps(message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via AMQP - size: {payload_size_mb:.2f} MB")
            
            channel.basic_publish(
                exchange='fl_client_updates',
                routing_key=f'client_{self.client_id}_update',
                body=payload,
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="model_update"
            )
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via AMQP")
            connection.close()
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via AMQP: {e}")
            raise
    
    def _send_metrics_via_amqp(self, message: dict):
        """Send metrics via AMQP (RabbitMQ)"""
        if pika is None:
            raise ImportError("pika module not available for AMQP")
        
        try:
            amqp_host = os.getenv("AMQP_HOST", "localhost")
            amqp_port = int(os.getenv("AMQP_PORT", "5672"))
            amqp_user = os.getenv("AMQP_USER", "guest")
            amqp_password = os.getenv("AMQP_PASSWORD", "guest")
            
            credentials = pika.PlainCredentials(amqp_user, amqp_password)
            # FAIR CONFIG: heartbeat=600s for very_poor network scenarios
            parameters = pika.ConnectionParameters(
                host=amqp_host,
                port=amqp_port,
                credentials=credentials,
                heartbeat=600,  # 10 minutes for very_poor network
                blocked_connection_timeout=600,  # Aligned with heartbeat
                connection_attempts=3,
                retry_delay=2
            )
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            channel.exchange_declare(exchange='fl_client_updates', exchange_type='direct', durable=True)
            
            payload = json.dumps(message)
            channel.basic_publish(
                exchange='fl_client_updates',
                routing_key=f'client_{self.client_id}_metrics',
                body=payload,
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="metrics"
            )
            
            connection.close()
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via AMQP: {e}")
            raise
    
    def _send_via_grpc(self, message: dict):
        """Send model update via gRPC"""
        if grpc is None or federated_learning_pb2 is None:
            raise ImportError("grpc modules not available for gRPC")
        
        try:
            grpc_host = os.getenv("GRPC_HOST", "localhost")
            grpc_port = int(os.getenv("GRPC_PORT", "50051"))
            
            # FAIR CONFIG: Set max message size to 128MB (aligned with AMQP default)
            options = [
                ('grpc.max_send_message_length', 128 * 1024 * 1024),
                ('grpc.max_receive_message_length', 128 * 1024 * 1024),
                # FAIR CONFIG: Keepalive settings 600s for very_poor network
                ('grpc.keepalive_time_ms', 600000),  # 10 minutes
                ('grpc.keepalive_timeout_ms', 60000),  # 1 minute timeout
            ]
            channel = grpc.insecure_channel(f'{grpc_host}:{grpc_port}', options=options)
            stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
            
            # Send model update
            weights_str = message['weights']
            payload_size_mb = len(weights_str) / (1024 * 1024)
            print(f"Client {self.client_id} sending via gRPC - size: {payload_size_mb:.2f} MB")
            
            response = stub.SendModelUpdate(
                federated_learning_pb2.ModelUpdate(
                    client_id=message['client_id'],
                    round=message['round'],
                    weights=weights_str.encode() if isinstance(weights_str, str) else weights_str,
                    num_samples=message['num_samples'],
                    metrics={k: float(v) for k, v in message['metrics'].items()}
                )
            )
            
            # Set flag: we're now waiting for aggregated model
            self.waiting_for_aggregated_model = True
            
            log_sent_packet(
                packet_size=len(weights_str),
                peer="server",
                protocol="gRPC",
                round=self.current_round,
                extra_info="model_update"
            )
            
            if response.success:
                print(f"Client {self.client_id} sent model update for round {self.current_round} via gRPC")
            else:
                raise Exception(f"gRPC send failed: {response.message}")
            
            channel.close()
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via gRPC: {e}")
            raise
    
    def _send_metrics_via_grpc(self, message: dict):
        """Send metrics via gRPC"""
        if grpc is None or federated_learning_pb2 is None:
            raise ImportError("grpc modules not available for gRPC")
        
        try:
            grpc_host = os.getenv("GRPC_HOST", "localhost")
            grpc_port = int(os.getenv("GRPC_PORT", "50051"))
            
            # FAIR CONFIG: Set max message size to 128MB (aligned with AMQP default)
            options = [
                ('grpc.max_send_message_length', 128 * 1024 * 1024),
                ('grpc.max_receive_message_length', 128 * 1024 * 1024),
                # FAIR CONFIG: Keepalive settings 600s for very_poor network
                ('grpc.keepalive_time_ms', 600000),  # 10 minutes
                ('grpc.keepalive_timeout_ms', 60000),  # 1 minute timeout
            ]
            channel = grpc.insecure_channel(f'{grpc_host}:{grpc_port}', options=options)
            stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
            
            response = stub.SendMetrics(
                federated_learning_pb2.Metrics(
                    client_id=message['client_id'],
                    round=message['round'],
                    loss=message['loss'],
                    accuracy=message['accuracy'],
                    num_samples=message['num_samples']
                )
            )
            
            payload_size = len(json.dumps(message))
            log_sent_packet(
                packet_size=payload_size,
                peer="server",
                protocol="gRPC",
                round=self.current_round,
                extra_info="metrics"
            )
            
            if not response.success:
                raise Exception(f"gRPC send failed: {response.message}")
            
            channel.close()
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via gRPC: {e}")
            raise
    
    async def _ensure_quic_connection(self):
        """Establish persistent QUIC connection if not already connected"""
        if self.quic_protocol is not None:
            return  # Already connected
        
        # Start QUIC connection thread if not running
        if self.quic_thread is None or not self.quic_thread.is_alive():
            self.quic_thread = threading.Thread(
                target=self._run_quic_loop,
                daemon=True,
                name=f"QUIC-Client-{self.client_id}"
            )
            self.quic_thread.start()
            
            # Wait for connection to establish
            max_wait = 10  # seconds
            waited = 0
            while self.quic_protocol is None and waited < max_wait:
                await asyncio.sleep(0.1)
                waited += 0.1
            
            if self.quic_protocol is None:
                raise ConnectionError(f"QUIC connection not established after {max_wait}s")
            
            print(f"[QUIC] Client {self.client_id} connection ready")
    
    def _run_quic_loop(self):
        """Run QUIC event loop in a separate thread"""
        self.quic_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.quic_loop)
        try:
            # Keep loop running indefinitely
            self.quic_loop.run_forever()
        except Exception as e:
            print(f"[QUIC] Event loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Don't close the loop here - it should stay open for sending
            print(f"[QUIC] Event loop stopped for client {self.client_id}")
    
    async def _quic_connection_loop(self):
        """Maintain persistent QUIC connection (runs in background)"""
        import ssl
        quic_host = os.getenv("QUIC_HOST", "localhost")
        quic_port = int(os.getenv("QUIC_PORT", "4433"))
        
        # FAIR CONFIG: Aligned with MQTT/AMQP/gRPC/DDS for unbiased comparison
        config = QuicConfiguration(
            is_client=True, 
            alpn_protocols=["fl"],  # CRITICAL: Must match server's ALPN
            verify_mode=ssl.CERT_NONE,
            # FAIR CONFIG: Data limits 128MB per stream, 256MB total (aligned with AMQP)
            max_stream_data=128 * 1024 * 1024,  # 128 MB per stream
            max_data=256 * 1024 * 1024,  # 256 MB total connection
            # FAIR CONFIG: Timeout 600s for very_poor network scenarios
            idle_timeout=600.0  # 10 minutes
        )
        
        print(f"[QUIC] Client {self.client_id} connecting to {quic_host}:{quic_port}...")
        print(f"[QUIC] Configuration: verify_mode=CERT_NONE, idle_timeout=600s")
        
        # Create protocol factory that sets client reference
        def create_protocol(*args, **kwargs):
            protocol = UnifiedClientQUICProtocol(*args, **kwargs)
            protocol.client = self
            print(f"[QUIC] Client {self.client_id} created protocol instance")
            return protocol
        
        # Retry connection with exponential backoff, keep retrying indefinitely
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"[QUIC] Client {self.client_id} attempt {attempt + 1}/{max_retries} - calling connect()...")
                async with connect(
                    quic_host,
                    quic_port,
                    configuration=config,
                    create_protocol=create_protocol
                ) as protocol:
                    self.quic_protocol = protocol
                    print(f"[QUIC] ✓ Client {self.client_id} established persistent connection")
                    print(f"[QUIC] Connection state: {protocol._quic.get_timer() if hasattr(protocol, '_quic') else 'unknown'}")
                    
                    # Keep connection alive indefinitely
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        print(f"[QUIC] Client {self.client_id} connection cancelled")
                    break  # Connection successful, exit retry loop
                    
            except (ConnectionError, OSError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    print(f"[QUIC] ✗ Client {self.client_id} connection attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
                    print(f"[QUIC] Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"[QUIC] ✗✗✗ Client {self.client_id} connection FAILED after {max_retries} attempts: {type(e).__name__}: {e}")
                    print(f"[QUIC] QUIC protocol will NOT be available for this client")
                    print(f"[QUIC] Server status: listening on {quic_host}:{quic_port}")
                    print(f"[QUIC] Possible causes:")
                    print(f"[QUIC]   1. Server not responding to QUIC handshake")
                    print(f"[QUIC]   2. Firewall blocking UDP port {quic_port}")
                    print(f"[QUIC]   3. Server's asyncio loop not running properly")
                    self.quic_protocol = None
            except Exception as e:
                print(f"[QUIC] ✗ Client {self.client_id} unexpected connection error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.quic_protocol = None
                break
    
    async def _handle_quic_message_async(self, message: dict):
        """Handle QUIC message received from server asynchronously"""
        msg_type = message.get('type')
        print(f"[QUIC] Client {self.client_id} received message type: {msg_type}")
        
        if msg_type == 'global_model':
            self.on_global_model_received_quic(message)
        elif msg_type == 'start_training':
            self.on_start_training_quic(message)
        elif msg_type == 'start_evaluation':
            self.on_start_evaluation_quic(message)
    
    def _handle_quic_message(self, message: dict):
        """Handle QUIC message received from server (called from QUIC protocol)""" 
        msg_type = message.get('type')
        print(f"[QUIC] Client {self.client_id} received message type: {msg_type}")
        
        if msg_type == 'global_model':
            self.on_global_model_received_quic(message)
        elif msg_type == 'start_training':
            self.on_start_training_quic(message)
        elif msg_type == 'start_evaluation':
            self.on_start_evaluation_quic(message)
    
    def on_global_model_received_quic(self, message):
        """Handle global model received via QUIC"""
        try:
            round_num = message['round']
            print(f"[QUIC] Client {self.client_id} received global model for round {round_num}")
            
            # Deserialize weights
            if 'quantized_data' in message:
                compressed_data = pickle.loads(base64.b64decode(message['quantized_data']))
                weights = self.quantizer.decompress_global_model(compressed_data) if self.quantizer else None
            else:
                weights = self.deserialize_weights(message['weights'])
            
            self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=message.get('model_config'),
                source="QUIC"
            )
            
            # Set event to signal model is ready
            if hasattr(self, 'model_received'):
                self.model_received.set()
        except Exception as e:
            print(f"[QUIC] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_training_quic(self, message):
        """Handle start training signal via QUIC"""
        try:
            print(f"[QUIC] Client {self.client_id} starting training for round {message.get('round', self.current_round + 1)}")
            
            log_received_packet(
                packet_size=len(json.dumps(message)),
                peer="server",
                protocol="QUIC",
                round=self.current_round,
                extra_info="start_training"
            )
            
            # Call the standard training handler
            self.handle_start_training(json.dumps(message).encode())
        except Exception as e:
            print(f"[QUIC] Client {self.client_id} error handling start_training: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_evaluation_quic(self, message):
        """Handle start evaluation signal via QUIC"""
        try:
            round_num = message.get('round', self.current_round)
            print(f"[QUIC] Client {self.client_id} starting evaluation for round {round_num}")
            
            log_received_packet(
                packet_size=len(json.dumps(message)),
                peer="server",
                protocol="QUIC",
                round=self.current_round,
                extra_info="start_evaluation"
            )
            
            # Call the standard evaluation handler
            self.handle_start_evaluation(json.dumps(message).encode())
        except Exception as e:
            print(f"[QUIC] Client {self.client_id} error handling evaluation signal: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_via_quic(self, message: dict):
        """Send model update via QUIC using persistent connection"""
        if asyncio is None or connect is None:
            raise ImportError("aioquic module not available for QUIC")
        
        try:
            # Check if event loop is available and running
            if self.quic_loop is None:
                raise ConnectionError("QUIC event loop not available")
            
            if self.quic_loop.is_closed():
                raise ConnectionError("QUIC event loop is closed")
            
            # Check if protocol is connected
            if self.quic_protocol is None:
                raise ConnectionError("QUIC protocol not connected")
            
            # Add 'type' field for server to identify message type
            quic_message = {**message, 'type': 'update'}
            
            payload = json.dumps(quic_message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via QUIC - size: {payload_size_mb:.2f} MB")
            
            # Use persistent connection directly via run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._do_quic_send(payload),
                self.quic_loop
            )
            future.result(timeout=15)  # Wait for send to complete
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="QUIC",
                round=self.current_round,
                extra_info="model_update"
            )
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via QUIC")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via QUIC: {e}")
            raise
    
    def _send_metrics_via_quic(self, message: dict):
        """Send metrics via QUIC using persistent connection"""
        if asyncio is None or connect is None:
            raise ImportError("aioquic module not available for QUIC")
        
        try:
            # Check if event loop is available and running
            if self.quic_loop is None:
                raise ConnectionError("QUIC event loop not available")
            
            if self.quic_loop.is_closed():
                raise ConnectionError("QUIC event loop is closed")
            
            # Check if protocol is connected
            if self.quic_protocol is None:
                raise ConnectionError("QUIC protocol not connected")
            
            # Add 'type' field for server to identify message type
            quic_message = {**message, 'type': 'metrics'}
            
            payload = json.dumps(quic_message)
            
            # Use persistent connection directly via run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._do_quic_send(payload),
                self.quic_loop
            )
            future.result(timeout=15)  # Wait for send to complete
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="QUIC",
                round=self.current_round,
                extra_info="metrics"
            )
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via QUIC: {e}")
            raise
    
    async def _send_quic_persistent(self, payload: str):
        """Send data via persistent QUIC connection"""
        # Ensure connection exists
        await self._ensure_quic_connection()
        
        if self.quic_protocol is None:
            raise ConnectionError("QUIC connection not established")
        
        # Schedule send on QUIC thread's event loop
        future = asyncio.run_coroutine_threadsafe(
            self._do_quic_send(payload),
            self.quic_loop
        )
        # Wait for completion
        future.result(timeout=10)
    
    async def _do_quic_send(self, payload: str):
        """Actually send data via QUIC (runs in QUIC thread's event loop)"""
        # Ensure connection is ready
        if self.quic_protocol is None or self.quic_protocol._quic is None:
            raise ConnectionError("QUIC protocol not available")
        
        print(f"[QUIC] Client {self.client_id} preparing to send {len(payload)} bytes")
        
        # Send data via QUIC stream
        stream_id = self.quic_protocol._quic.get_next_available_stream_id()
        data = (payload + '\n').encode('utf-8')
        self.quic_protocol._quic.send_stream_data(stream_id, data, end_stream=True)
        self.quic_protocol.transmit()
        
        print(f"[QUIC] Client {self.client_id} sent on stream {stream_id}, transmitting...")
        
        # FAIR FIX: Removed artificial 1.5s delay for large messages
        # QUIC handles flow control automatically, so we don't need manual delays
        # This makes QUIC behavior similar to other protocols which don't have artificial delays
        # The transmit() call above is sufficient for immediate transmission
    
    async def _quic_send_data(self, host: str, port: int, payload: str, msg_type: str):
        """Async QUIC data send with timeout and retry (legacy method for registration)"""
        import ssl
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use ssl.CERT_NONE for self-signed certificate verification
                config = QuicConfiguration(
                    is_client=True,
                    alpn_protocols=["fl"],  # CRITICAL: Must match server's ALPN
                    verify_mode=ssl.CERT_NONE
                )
                
                # connect() returns a QuicConnectionProtocol
                # We use create_stream() to get reader/writer
                try:
                    # Try Python 3.11+ asyncio.timeout
                    async with asyncio.timeout(5):
                        async with connect(host, port, configuration=config) as protocol:
                            # Create a stream for sending data
                            reader, writer = await protocol.create_stream()
                            writer.write((payload + '\n').encode())
                            await writer.drain()
                            # Give time for data to be transmitted before closing
                            await asyncio.sleep(0.5)
                            writer.close()
                            # Wait for close to complete
                            try:
                                await writer.wait_closed()
                            except:
                                pass
                            return  # Success
                except AttributeError as e:
                    # Python 3.8 doesn't have asyncio.timeout
                    if "has no attribute 'timeout'" in str(e):
                        # Use manual context manager handling for Python 3.8
                        connection = connect(host, port, configuration=config)
                        protocol = await asyncio.wait_for(connection.__aenter__(), timeout=5)
                        try:
                            reader, writer = await protocol.create_stream()
                            writer.write((payload + '\n').encode())
                            await writer.drain()
                            # Give time for data to be transmitted
                            await asyncio.sleep(0.5)
                            writer.close()
                            try:
                                await writer.wait_closed()
                            except:
                                pass
                            return  # Success
                        finally:
                            await connection.__aexit__(None, None, None)
                    else:
                        raise
            except asyncio.TimeoutError:
                print(f"QUIC send timeout (attempt {attempt + 1}/{max_retries}): Connection took too long")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                else:
                    raise
            except (ConnectionError, OSError) as e:
                print(f"QUIC send connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
            except Exception as e:
                print(f"QUIC send error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
    
    # =========================================================================
    # HTTP/3 CLIENT METHODS
    # =========================================================================
    
    async def _ensure_http3_connection(self):
        """Establish persistent HTTP/3 connection if not already connected"""
        if self.http3_protocol is not None:
            return  # Already connected
        
        # Start HTTP/3 connection thread if not running
        if self.http3_thread is None or not self.http3_thread.is_alive():
            self.http3_thread = threading.Thread(
                target=self._run_http3_loop,
                daemon=True,
                name=f"HTTP3-Client-{self.client_id}"
            )
            self.http3_thread.start()
            
            # Wait for connection to establish
            max_wait = 10  # seconds
            waited = 0
            while self.http3_protocol is None and waited < max_wait:
                await asyncio.sleep(0.1)
                waited += 0.1
            
            if self.http3_protocol is None:
                raise ConnectionError(f"HTTP/3 connection not established after {max_wait}s")
            
            print(f"[HTTP/3] Client {self.client_id} connection ready")
    
    def _run_http3_loop(self):
        """Run HTTP/3 event loop in a separate thread"""
        self.http3_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.http3_loop)
        try:
            # Keep loop running indefinitely
            self.http3_loop.run_forever()
        except Exception as e:
            print(f"[HTTP/3] Event loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[HTTP/3] Event loop stopped for client {self.client_id}")
    
    async def _http3_connection_loop(self):
        """Maintain persistent HTTP/3 connection (runs in background)"""
        import ssl
        http3_host = os.getenv("HTTP3_HOST", "localhost")
        http3_port = int(os.getenv("HTTP3_PORT", "4434"))
        
        # FAIR CONFIG: Aligned with MQTT/AMQP/gRPC/QUIC/DDS for unbiased comparison
        config = QuicConfiguration(
            is_client=True, 
            alpn_protocols=H3_ALPN,  # HTTP/3 ALPN
            verify_mode=ssl.CERT_NONE,
            # FAIR CONFIG: Data limits 128MB per stream, 256MB total (aligned with AMQP)
            max_stream_data=128 * 1024 * 1024,  # 128 MB per stream
            max_data=256 * 1024 * 1024,  # 256 MB total connection
            # FAIR CONFIG: Timeout 600s for very_poor network scenarios
            idle_timeout=600.0  # 10 minutes
        )
        
        print(f"[HTTP/3] Client {self.client_id} connecting to {http3_host}:{http3_port}...")
        print(f"[HTTP/3] Configuration: verify_mode=CERT_NONE, idle_timeout=600s")
        
        # Create protocol factory that sets client reference
        def create_protocol(*args, **kwargs):
            protocol = UnifiedClientHTTP3Protocol(*args, **kwargs)
            protocol.client = self
            print(f"[HTTP/3] Client {self.client_id} created protocol instance")
            return protocol
        
        # Retry connection with exponential backoff
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"[HTTP/3] Client {self.client_id} attempt {attempt + 1}/{max_retries} - calling connect()...")
                async with connect(
                    http3_host,
                    http3_port,
                    configuration=config,
                    create_protocol=create_protocol
                ) as protocol:
                    self.http3_protocol = protocol
                    print(f"[HTTP/3] ✓ Client {self.client_id} established persistent connection")
                    
                    # Keep connection alive indefinitely
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        print(f"[HTTP/3] Client {self.client_id} connection cancelled")
                    break  # Connection successful, exit retry loop
                    
            except (ConnectionError, OSError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    print(f"[HTTP/3] ✗ Client {self.client_id} connection attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
                    print(f"[HTTP/3] Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"[HTTP/3] ✗✗✗ Client {self.client_id} connection FAILED after {max_retries} attempts: {type(e).__name__}: {e}")
                    self.http3_protocol = None
            except Exception as e:
                print(f"[HTTP/3] ✗ Client {self.client_id} unexpected connection error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.http3_protocol = None
                break
    
    async def _handle_http3_message_async(self, message: dict):
        """Handle HTTP/3 message received from server asynchronously"""
        msg_type = message.get('type')
        print(f"[HTTP/3] Client {self.client_id} received message type: {msg_type}")
        
        if msg_type == 'global_model':
            self.on_global_model_received_http3(message)
        elif msg_type == 'start_training':
            self.on_start_training_http3(message)
        elif msg_type == 'start_evaluation':
            self.on_start_evaluation_http3(message)
    
    def on_global_model_received_http3(self, message):
        """Handle global model received via HTTP/3"""
        try:
            round_num = message['round']
            print(f"[HTTP/3] Client {self.client_id} received global model for round {round_num}")
            
            # Deserialize weights
            if 'quantized_data' in message:
                compressed_data = pickle.loads(base64.b64decode(message['quantized_data']))
                weights = self.quantizer.decompress_global_model(compressed_data) if self.quantizer else None
            else:
                weights = self.deserialize_weights(message['weights'])
            
            self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=message.get('model_config'),
                source="HTTP/3"
            )
            
            # Set event to signal model is ready
            if hasattr(self, 'model_received'):
                self.model_received.set()
        except Exception as e:
            print(f"[HTTP/3] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_training_http3(self, message):
        """Handle start training signal via HTTP/3"""
        try:
            print(f"[HTTP/3] Client {self.client_id} starting training for round {message.get('round', self.current_round + 1)}")
            
            log_received_packet(
                packet_size=len(json.dumps(message)),
                peer="server",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info="start_training"
            )
            
            # Call the standard training handler
            self.handle_start_training(json.dumps(message).encode())
        except Exception as e:
            print(f"[HTTP/3] Client {self.client_id} error handling start_training: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_evaluation_http3(self, message):
        """Handle start evaluation signal via HTTP/3"""
        try:
            round_num = message.get('round', self.current_round)
            print(f"[HTTP/3] Client {self.client_id} starting evaluation for round {round_num}")
            
            log_received_packet(
                packet_size=len(json.dumps(message)),
                peer="server",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info="start_evaluation"
            )
            
            # Call the standard evaluation handler
            self.handle_start_evaluation(json.dumps(message).encode())
        except Exception as e:
            print(f"[HTTP/3] Client {self.client_id} error handling evaluation signal: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_via_http3(self, message: dict):
        """Send model update via HTTP/3 using persistent connection"""
        if not HTTP3_AVAILABLE or asyncio is None or connect is None:
            raise ImportError("HTTP/3 module not available")
        
        try:
            # Check if event loop is available and running
            if self.http3_loop is None:
                raise ConnectionError("HTTP/3 event loop not available")
            
            if self.http3_loop.is_closed():
                raise ConnectionError("HTTP/3 event loop is closed")
            
            # Check if protocol is connected
            if self.http3_protocol is None:
                raise ConnectionError("HTTP/3 protocol not connected")
            
            # Add 'type' field for server to identify message type
            http3_message = {**message, 'type': 'update'}
            
            payload = json.dumps(http3_message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via HTTP/3 - size: {payload_size_mb:.2f} MB")
            
            # Use persistent connection directly via run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._do_http3_send(payload),
                self.http3_loop
            )
            future.result(timeout=15)  # Wait for send to complete
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info="model_update"
            )
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via HTTP/3")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via HTTP/3: {e}")
            raise
    
    def _send_metrics_via_http3(self, message: dict):
        """Send metrics via HTTP/3 using persistent connection"""
        if not HTTP3_AVAILABLE or asyncio is None or connect is None:
            raise ImportError("HTTP/3 module not available")
        
        try:
            # Check if event loop is available and running
            if self.http3_loop is None:
                raise ConnectionError("HTTP/3 event loop not available")
            
            if self.http3_loop.is_closed():
                raise ConnectionError("HTTP/3 event loop is closed")
            
            # Check if protocol is connected
            if self.http3_protocol is None:
                raise ConnectionError("HTTP/3 protocol not connected")
            
            # Add 'type' field for server to identify message type
            http3_message = {**message, 'type': 'metrics'}
            
            payload = json.dumps(http3_message)
            
            # Use persistent connection directly via run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._do_http3_send(payload),
                self.http3_loop
            )
            future.result(timeout=15)  # Wait for send to complete
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info="metrics"
            )
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via HTTP/3: {e}")
            raise
    
    async def _do_http3_send(self, payload: str):
        """Actually send data via HTTP/3 (runs in HTTP/3 thread's event loop)"""
        if self.http3_protocol is None:
            raise ConnectionError("HTTP/3 protocol not connected")
        
        # Ensure HTTP connection is initialized
        if self.http3_protocol._http is None:
            self.http3_protocol._http = H3Connection(self.http3_protocol._quic)
        
        # Get next available stream ID
        stream_id = self.http3_protocol._quic.get_next_available_stream_id(is_unidirectional=False)
        
        # Prepare JSON payload
        payload_bytes = payload.encode('utf-8')
        
        # Send HTTP/3 request
        headers = [
            (b":method", b"POST"),
            (b":path", b"/fl/message"),
            (b":scheme", b"https"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload_bytes)).encode()),
        ]
        self.http3_protocol._http.send_headers(stream_id=stream_id, headers=headers)
        self.http3_protocol._http.send_data(stream_id=stream_id, data=payload_bytes, end_stream=True)
        self.http3_protocol.transmit()
        
        print(f"[HTTP/3] Client {self.client_id} sent data on stream {stream_id} ({len(payload_bytes)} bytes)")
    
    def _send_via_dds(self, message: dict):
        """Send model update via DDS with chunking (matching standalone)"""
        if not DDS_AVAILABLE or not self.dds_update_chunk_writer:
            raise NotImplementedError("DDS not available - triggering fallback")
        
        try:
            # Serialize weights and convert to list of integers for chunking
            weights_bytes = pickle.dumps(message['weights'])
            weights_list = list(weights_bytes)  # Convert bytes to List[int]
            
            # FAIR CONFIG: Use chunking to match standalone DDS implementation
            self.send_model_update_chunked(
                round_num=message['round'],
                serialized_weights=weights_list,
                num_samples=message.get('num_samples', 0),
                loss=message.get('loss', 0.0),
                mse=message.get('mse', 0.0),
                mae=message.get('mae', 0.0),
                mape=message.get('mape', 0.0)
            )
            
            # Log the packet (log total size, not individual chunks)
            log_sent_packet(
                packet_size=len(weights_bytes),
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="model_update_chunked"
            )
            
            print(f"Client {self.client_id} sent chunked model update for round {self.current_round} via DDS")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via DDS: {e}")
            raise
    
    def _send_metrics_via_dds(self, message: dict):
        """Send metrics via DDS"""
        if not DDS_AVAILABLE or not self.dds_metrics_writer:
            raise NotImplementedError("DDS not available - triggering fallback")
        
        try:
            # Create DDS metrics message
            dds_msg = EvaluationMetrics(
                client_id=self.client_id,
                round=message['round'],
                num_samples=message.get('num_samples', 0),
                loss=message.get('loss', 0.0),
                accuracy=message.get('accuracy', 0.0),
                mse=message.get('mse', 0.0),
                mae=message.get('mae', 0.0),
                mape=message.get('mape', 0.0)
            )
            
            # Write to DDS
            self.dds_metrics_writer.write(dds_msg)
            
            # Log the packet
            log_sent_packet(
                packet_size=len(str(dds_msg)),
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="metrics"
            )
            
            print(f"Client {self.client_id} sent metrics for round {self.current_round} via DDS")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via DDS: {e}")
            raise
    
    def check_global_model_chunks(self):
        """Check for global model chunks from server (matching standalone DDS)"""
        if not DDS_AVAILABLE or not self.dds_global_model_chunk_reader:
            return None
        
        try:
            # Check for chunked global model
            chunk_samples = self.dds_global_model_chunk_reader.take()
            
            for sample in chunk_samples:
                if not sample or not hasattr(sample, 'round'):
                    continue
                
                round_num = sample.round
                chunk_id = sample.chunk_id
                total_chunks = sample.total_chunks
                
                # Initialize buffers if needed
                if not self.global_model_metadata:
                    self.global_model_metadata = {
                        'round': round_num,
                        'total_chunks': total_chunks,
                        'model_config_json': sample.model_config_json if hasattr(sample, 'model_config_json') else ''
                    }
                    print(f"Client {self.client_id}: Receiving global model in {total_chunks} chunks...")
                
                # Store chunk
                self.global_model_chunks[chunk_id] = sample.payload
                
                # Progress logging - show every 20 chunks to reduce spam
                chunks_received = len(self.global_model_chunks)
                if chunks_received % 20 == 0 or chunks_received == total_chunks:
                    print(f"Client {self.client_id}: Received {chunks_received}/{total_chunks} chunks")
                
                # Check if all chunks received
                if chunks_received == total_chunks:
                    print(f"Client {self.client_id}: All chunks received, reassembling...")
                    
                    try:
                        # Reassemble chunks in order
                        reassembled_data = []
                        for i in range(total_chunks):
                            if i in self.global_model_chunks:
                                reassembled_data.extend(self.global_model_chunks[i])
                            else:
                                print(f"ERROR: Missing chunk {i}")
                                break
                        
                        # Only process if we have all chunks
                        if len(reassembled_data) > 0:
                            # Deserialize weights
                            weights = pickle.loads(bytes(reassembled_data))
                            
                            # Clear buffers
                            self.global_model_chunks.clear()
                            self.global_model_metadata.clear()
                            
                            return {'weights': weights, 'round': round_num}
                    
                    except Exception as e:
                        print(f"[ERROR] Client {self.client_id}: Exception during model reassembly: {e}")
                        import traceback
                        traceback.print_exc()
                        # Clear buffers on error
                        self.global_model_chunks.clear()
                        self.global_model_metadata.clear()
        
        except Exception as e:
            print(f"Client {self.client_id} ERROR checking global model chunks: {e}")
        
        return None
    
    def start(self):
        """Connect to MQTT broker and start listening"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
                self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=120)
                # FAIR CONFIG: keepalive 600s for very_poor network
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 600)
                print(f"Successfully connected to MQTT broker!\n")
                self.mqtt_client.loop_forever(retry_first_connection=True)
                break
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to MQTT broker after {max_retries} attempts.")
                    raise


def load_emotion_data(client_id: int):
    """
    Load emotion recognition dataset for a specific client
    
    Args:
        client_id: Client identifier
        
    Returns:
        Tuple of (train_generator, validation_generator)
    """
    # Detect environment: Docker uses /app prefix, local uses relative path
    if os.path.exists('/app'):
        base_path = '/app/Client/Emotion_Recognition/Dataset'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(project_root, 'Client', 'Emotion_Recognition', 'Dataset')
    
    train_path = os.path.join(base_path, f'client_{client_id}', 'train')
    validation_path = os.path.join(base_path, f'client_{client_id}', 'validation')
    
    print(f"Dataset base path: {base_path}")
    print(f"Train path: {train_path}")
    print(f"Validation path: {validation_path}")
    
    # Initialize image data generator with rescaling
    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    # Load training and validation data
    train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_data_gen.flow_from_directory(
        validation_path,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"[Dataset] Train samples: {train_generator.samples}")
    print(f"[Dataset] Validation samples: {validation_generator.samples}")
    print(f"[Dataset] Classes: {train_generator.num_classes}")
    
    return train_generator, validation_generator


def main():
    """Main function"""
    # Prevent duplicate client instance with same client_id in the same container/host.
    lock_path = f"/tmp/unified_emotion_client_{CLIENT_ID}.lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[Startup] Another unified client instance is already running for client_id={CLIENT_ID}. Exiting.")
        return

    print(f"Unified FL Client - Emotion Recognition (Client {CLIENT_ID})")
    
    # Load real emotion recognition dataset
    print(f"\n{'='*70}")
    print("LOADING EMOTION RECOGNITION DATASET")
    print(f"{'='*70}")
    
    try:
        train_generator, validation_generator = load_emotion_data(CLIENT_ID)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print(f"\nPlease ensure dataset exists at:")
        print(f"  Dataset/client_{CLIENT_ID}/train/")
        print(f"  Dataset/client_{CLIENT_ID}/validation/")
        return
    
    # Create client
    client = UnifiedFLClient_Emotion(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    # Start FL
    print(f"\n{'='*60}")
    print(f"Starting Unified FL Client {CLIENT_ID} with RL Protocol Selection")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"{'='*60}\n")
    
    try:
        client.start()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} shutting down...")
        client.mqtt_client.disconnect()


if __name__ == "__main__":
    main()
