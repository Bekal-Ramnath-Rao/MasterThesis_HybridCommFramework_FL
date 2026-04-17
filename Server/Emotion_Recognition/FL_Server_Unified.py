"""
Unified Federated Learning Server for Emotion Recognition
Handles all 5 protocols simultaneously: MQTT, AMQP, gRPC, QUIC, DDS

The server listens on all protocol channels and responds to clients
using whichever protocol they selected via RL.
"""

import os
import socket
import sys

# Configure GPU before importing TensorFlow
_gpu_id = os.environ.get('GPU_DEVICE_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_id
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure pip-installed CUDA 12 ptxas is on PATH (system ptxas may be too old)
_nvcc_bin = os.path.join(sys.prefix, 'lib', 'python' + '.'.join(map(str, sys.version_info[:2])),
                        'site-packages', 'nvidia', 'cuda_nvcc', 'bin')
if os.path.isdir(_nvcc_bin) and _nvcc_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _nvcc_bin + ':' + os.environ.get('PATH', '')

# Remove stale CUDA 10.x paths from LD_LIBRARY_PATH to avoid library conflicts
_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
_ld_path = ':'.join(p for p in _ld_path.split(':') if p and 'cuda-10' not in p)
os.environ['LD_LIBRARY_PATH'] = _ld_path

import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import threading
import asyncio
import fcntl
from typing import List, Dict, Sequence, TYPE_CHECKING
from pathlib import Path
from concurrent import futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf

# Configure GPU 0 with memory growth
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Enable memory growth to prevent OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set GPU 0 as the only visible device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"[GPU] Server configured to use GPU 0: {gpus[0]}")
        print(f"[GPU] Memory growth enabled")
    else:
        print("[WARNING] No GPU devices found, using CPU")
except Exception as e:
    print(f"[WARNING] GPU configuration failed: {e}, using CPU")

# Protocol-specific imports
import paho.mqtt.client as mqtt

try:
    import pika  # AMQP
    AMQP_AVAILABLE = True
except ImportError:
    AMQP_AVAILABLE = False
    print("Warning: pika not available, AMQP disabled")

try:
    import grpc  # gRPC
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))
    import federated_learning_pb2
    import federated_learning_pb2_grpc
    GRPC_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRPC_AVAILABLE = False
    print("Warning: grpc not available, gRPC disabled")

try:
    from aioquic.asyncio import serve
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.asyncio.protocol import QuicConnectionProtocol
    from aioquic.quic.events import StreamDataReceived
    QUIC_AVAILABLE = True
except ImportError as e:
    QUIC_AVAILABLE = False
    print(f"Warning: aioquic not available, QUIC disabled (ImportError: {e})")
except Exception as e:
    QUIC_AVAILABLE = False
    print(f"Warning: aioquic not available, QUIC disabled (Unexpected error: {type(e).__name__}: {e})")

# Set CycloneDDS config before any cyclonedds import (native lib may read at load time);
# same logic as FL_Server_DDS.py for DDS_PEER_* static unicast across hosts.
def _emotion_config_dir():
    if os.path.exists("/app"):
        return "/app/config"
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))


def _try_distributed_unicast_server():
    base = _emotion_config_dir()
    helper = os.path.join(base, "dds_distributed_unicast.py")
    if not os.path.isfile(helper):
        return False
    import importlib.util

    spec = importlib.util.spec_from_file_location("dds_distributed_unicast", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.try_apply_server_uri()


def _ensure_server_cyclonedds_uri():
    if os.environ.get("CYCLONEDDS_URI"):
        return
    if _try_distributed_unicast_server():
        return
    base = _emotion_config_dir()
    for name in ("cyclonedds-multicast-lan.xml", "cyclonedds-emotion-server.xml"):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            os.environ["CYCLONEDDS_URI"] = "file://" + os.path.abspath(p)
            return


_ensure_server_cyclonedds_uri()

try:
    from cyclonedds.domain import DomainParticipant  # DDS
    from cyclonedds.topic import Topic
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.util import duration
    from cyclonedds.core import Qos, Policy
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.types import sequence
    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False
    print("Warning: cyclonedds not available, DDS disabled")

try:
    from aioquic.h3.connection import H3_ALPN, H3Connection, Setting
    from aioquic.h3.events import DataReceived, HeadersReceived, H3Event
    from aioquic.quic.events import StreamReset
    HTTP3_AVAILABLE = True
except ImportError as e:
    HTTP3_AVAILABLE = False
    print(f"Warning: aioquic H3 not available, HTTP/3 disabled (ImportError: {e})")
except Exception as e:
    HTTP3_AVAILABLE = False
    print(f"Warning: aioquic H3 not available, HTTP/3 disabled (Unexpected error: {type(e).__name__}: {e})")

# Detect Docker environment and set project root accordingly
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
from experiment_results_path import get_experiment_results_dir
from battery_results_agg import avg_battery_model_drain_fraction

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

try:
    from quantization_server import ServerQuantizationHandler, QuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Quantization module not available")
    QUANTIZATION_AVAILABLE = False

# Server Configuration
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))
from fl_termination_env import stop_on_client_convergence
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))

# Protocol endpoints (auto-detect Docker vs local)
IN_DOCKER = os.path.exists('/app')
MQTT_BROKER = os.getenv("MQTT_BROKER", 'mqtt-broker' if IN_DOCKER else 'localhost')
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
AMQP_BROKER = os.getenv("AMQP_BROKER", 'amqp-broker' if IN_DOCKER else 'localhost')
AMQP_PORT = int(os.getenv("AMQP_PORT", "5672"))
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
QUIC_HOST = os.getenv("QUIC_HOST", '0.0.0.0')
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
HTTP3_HOST = os.getenv("HTTP3_HOST", '0.0.0.0')
HTTP3_PORT = int(os.getenv("HTTP3_PORT", "4434"))
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
DDS_CHUNK_MAX_BLOCKING_SEC = float(os.getenv("DDS_CHUNK_MAX_BLOCKING_SEC", "60.0"))

# Protocol max payload sizes (unified spec)
MQTT_MAX_PAYLOAD_BYTES = 128 * 1024   # 128 KB
AMQP_MAX_FRAME_BYTES = 128 * 1024     # 128 KB
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))  # 4 MB
HTTP3_MAX_STREAM_DATA = 16 * 1024     # HTTP/3: 16 KB per stream
CHUNK_SIZE = 64 * 1024                # DDS: 64 KB per chunk
# QUIC flow control (separate from HTTP/3; 128 MB for QUIC stream)
QUIC_HTTP3_MAX_DATA_BYTES = 128 * 1024 * 1024  # 128 MB for QUIC
PROTOCOL_NEGOTIATION_TIMEOUT_SEC = float(os.getenv("PROTOCOL_NEGOTIATION_TIMEOUT_SEC", "3.0"))
PROTOCOL_NEGOTIATION_POLL_SEC = float(os.getenv("PROTOCOL_NEGOTIATION_POLL_SEC", "0.1"))

# DDS Data Structures (must be defined at module level for Python 3.8)
if DDS_AVAILABLE:
    from dataclasses import dataclass, field
    
    @dataclass
    class GlobalModel(IdlStruct):
        round: int
        weights: sequence[int]
        model_config_json: str = ""
    
    @dataclass
    class GlobalModelChunk(IdlStruct):
        round: int
        chunk_id: int
        total_chunks: int
        payload: sequence[int]
        model_config_json: str = ""  # JSON string containing model configuration
        server_sent_unix: float = 0.0  # same on all chunks: time when server started this downlink
    
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
        accuracy: float
    
    @dataclass
    class ModelUpdateChunk(IdlStruct):
        client_id: int
        round: int
        chunk_id: int
        total_chunks: int
        payload: sequence[int]
        num_samples: int
        loss: float
        accuracy: float
    
    @dataclass
    class EvaluationMetrics(IdlStruct):
        client_id: int
        round: int
        num_samples: int
        loss: float
        accuracy: float
        client_converged: float = 0.0
        battery_soc: float = 1.0
        training_time_sec: float = 0.0
        round_time_sec: float = 0.0
        uplink_model_comm_sec: float = 0.0


class UnifiedFederatedLearningServer:
    """
    Unified FL Server that handles all 5 communication protocols
    """
    
    def __init__(self, min_clients, num_rounds, max_clients=100):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.num_clients = min_clients  # Start with minimum, will update as clients join
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = {}  # Maps client_id -> control-plane registration protocol
        self.client_delivery_protocols = {}  # Maps client_id -> preferred protocol for next global model downlink
        self.client_uplink_protocols = {}  # Maps client_id -> last protocol used for local model upload
        self.client_metric_protocols = {}  # Maps client_id -> last protocol used for metrics
        self.client_protocol_queries = {}  # Maps client_id -> {'round_id', 'global_model_id'} when negotiation query is pending
        self.active_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        self.model_config = None
        
        # Server state
        self.running = True
        self.training_started = False
        self._pre_training_update_notice_shown = False
        
        # Metrics storage
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        self.ROUND_TIMES = []
        self.BATTERY_CONSUMPTION = []
        self.BATTERY_MODEL_CONSUMPTION = []
        self.AVG_TRAINING_TIME_SEC = []
        self.AVG_BATTERY_SOC = []
        self.round_start_time = None

        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.start_time = None
        self.convergence_time = None
        
        # DDS chunk reassembly buffers (FAIR CONFIG: matching standalone)
        self.model_update_chunks = {}  # {client_id: {chunk_id: payload}}
        self.model_update_metadata = {}  # {client_id: {total_chunks, num_samples, loss, accuracy}}
        self.transport_update_chunks = {}  # {(protocol, client_id, round): {'chunks': {}, ...}}
        
        # Lock for thread-safe operations (reentrant for nested handlers)
        self.lock = threading.RLock()
        
        # Protocol handlers
        self.mqtt_client = None
        self.amqp_connection = None
        self.amqp_channel = None
        self.amqp_consumer_connection = None
        self.amqp_consumer_channel = None
        self.amqp_send_connection = None  # Separate connection for sending (thread-safe)
        self.amqp_send_channel = None
        self.grpc_server = None
        self.quic_server = None
        self.quic_clients = {}  # Maps client_id -> QuicConnectionProtocol (one connection per client; supports both clients using QUIC)
        self.http3_server = None
        self.http3_clients = {}  # Maps client_id -> HTTP3ServerProtocol (one connection per client)
        self.dds_participant = None
        self.dds_writers = {}
        self.dds_readers = {}
        
        # gRPC state tracking
        self.grpc_should_train = {}  # Maps client_id -> should_train (bool)
        self.grpc_should_evaluate = {}  # Maps client_id -> should_evaluate (bool)
        self.grpc_model_ready = {}  # Maps client_id -> model_ready_for_round (int)
        # (client_id, round) -> server wall time when gRPC downlink transfer started (chunk 0)
        self.grpc_downlink_sent_unix = {}
        # Dedup noisy GetGlobalModel chunk-0 logs (clients poll ~1 Hz per listener thread)
        self._grpc_chunk0_log_keys = set()
        
        # Training configuration
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        
        # Initialize packet logger
        init_db()
        
        # Initialize quantization if needed
        use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() in ("true", "1", "yes")
        if use_quantization and QUANTIZATION_AVAILABLE:
            self.quantization_handler = ServerQuantizationHandler(QuantizationConfig())
            print("[Unified Server] Quantization enabled")
        else:
            self.quantization_handler = None
            print("[Unified Server] Quantization disabled")
        
        # Initialize global model
        self.initialize_global_model()
        
        print(f"\n{'='*70}")
        print(f"UNIFIED FEDERATED LEARNING SERVER - EMOTION RECOGNITION")
        print(f"{'='*70}")
        print(f"Clients Expected: {self.num_clients}")
        print(f"Max Rounds: {self.num_rounds}")
        print(f"Protocols Enabled:")
        print(f"  - MQTT: ✓")
        print(f"  - AMQP: {'✓' if AMQP_AVAILABLE else '✗'}")
        print(f"  - gRPC: {'✓' if GRPC_AVAILABLE else '✗'}")
        print(f"  - QUIC: {'✓' if QUIC_AVAILABLE else '✗'}")
        print(f"  - HTTP/3: {'✓' if HTTP3_AVAILABLE else '✗'}")
        print(f"  - DDS: {'✓' if DDS_AVAILABLE else '✗'}")
        print(f"{'='*70}\n")
    
    def initialize_global_model(self):
        """Initialize global model weights (emotion recognition CNN)"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Input
        
        # Define model configuration for clients
        self.model_config = {
            "input_shape": [48, 48, 1],
            "layers": [
                {"type": "Conv2D", "filters": 32, "kernel_size": [3, 3], "activation": "relu"},
                {"type": "Conv2D", "filters": 64, "kernel_size": [3, 3], "activation": "relu"},
                {"type": "MaxPooling2D", "pool_size": [2, 2]},
                {"type": "Dropout", "rate": 0.25},
                {"type": "Conv2D", "filters": 128, "kernel_size": [3, 3], "activation": "relu"},
                {"type": "MaxPooling2D", "pool_size": [2, 2]},
                {"type": "Conv2D", "filters": 128, "kernel_size": [3, 3], "activation": "relu"},
                {"type": "MaxPooling2D", "pool_size": [2, 2]},
                {"type": "Dropout", "rate": 0.25},
                {"type": "Flatten"},
                {"type": "Dense", "units": 1024, "activation": "relu"},
                {"type": "Dropout", "rate": 0.5},
                {"type": "Dense", "units": 7, "activation": "softmax"}
            ],
            "num_classes": 7
        }
        
        # Create the CNN model structure
        model = Sequential()
        model.add(Input(shape=(48, 48, 1)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        
        # Get initial weights
        self.global_weights = model.get_weights()
        print("[Model] Global model initialized with random weights")
        print(f"[Model] Number of weight layers: {len(self.global_weights)}")
    
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
    
    def send_global_model_chunked(self, round_num, serialized_weights, model_config):
        """Send global model as chunks via DDS"""
        chunks = self.split_into_chunks(serialized_weights)
        total_chunks = len(chunks)
        server_sent_unix = time.time()
        
        print(f"Sending global model in {total_chunks} chunks ({len(serialized_weights)} bytes total)")
        
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = GlobalModelChunk(
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                model_config_json=model_config if chunk_id == 0 else "",  # Only send config with first chunk
                server_sent_unix=server_sent_unix,
            )
            self.dds_writers['global_model_chunk'].write(chunk)
            # Reliable QoS handles delivery, no artificial delay needed
            if (chunk_id + 1) % 20 == 0:  # Progress update every 20 chunks
                print(f"  Sent {chunk_id + 1}/{total_chunks} chunks")
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from clients"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights
    
    # =========================================================================
    # MQTT PROTOCOL HANDLERS
    # =========================================================================
    
    def start_mqtt_server(self):
        """Start MQTT protocol handler"""
        try:
            self.mqtt_client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="fl_unified_server_mqtt", 
                protocol=mqtt.MQTTv311, 
                clean_session=True
            )
            self.mqtt_client._max_packet_size = MQTT_MAX_PAYLOAD_BYTES  # 128 KB
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_message = self.on_mqtt_message
            # FAIR CONFIG: keepalive 600s for very_poor network
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 600)
            
            # Start MQTT loop in separate thread
            mqtt_thread = threading.Thread(target=self.mqtt_client.loop_forever, daemon=True)
            mqtt_thread.start()
            print("[MQTT] Server started")
        except Exception as e:
            print(f"[MQTT] Failed to start: {e}")
    
    def on_mqtt_connect(self, client, userdata, connect_flags, reason_code, properties):
        """MQTT connection callback"""
        if reason_code == 0:
            print(f"[MQTT] Connected to broker")
            client.subscribe("fl/client_register", qos=1)
            client.subscribe("fl/client/+/update", qos=1)
            client.subscribe("fl/client/+/metrics", qos=1)
        else:
            print(f"[MQTT] Connection failed with code {reason_code}")
    
    def on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            log_received_packet(
                packet_size=len(msg.payload),
                peer=f"client_{msg.topic}",
                protocol="MQTT",
                round=self.current_round,
                extra_info=msg.topic
            )
            
            if msg.topic == "fl/client_register":
                data = json.loads(msg.payload.decode())
                client_id = data['client_id']
                self.handle_client_registration(client_id, 'mqtt')
                
                # Set up AMQP queues for this client immediately (synchronous)
                try:
                    if AMQP_AVAILABLE:
                        credentials = pika.PlainCredentials('guest', 'guest')
                        # FAIR CONFIG: heartbeat=600s for very_poor network scenarios
                        parameters = pika.ConnectionParameters(
                            host=AMQP_BROKER,
                            port=AMQP_PORT,
                            credentials=credentials,
                            connection_attempts=5,
                            retry_delay=1,
                            heartbeat=600,  # 10 minutes for very_poor network
                            blocked_connection_timeout=600,  # Aligned with heartbeat
                            frame_max=AMQP_MAX_FRAME_BYTES  # 128 KB
                        )
                        conn = pika.BlockingConnection(parameters)
                        ch = conn.channel()
                        
                        # Declare exchange
                        ch.exchange_declare(
                            exchange='fl_client_updates',
                            exchange_type='direct',
                            durable=True
                        )
                        
                        # Queue for this client's updates
                        update_queue = f'client_{client_id}_updates'
                        ch.queue_declare(queue=update_queue, durable=True)
                        ch.queue_bind(
                            exchange='fl_client_updates',
                            queue=update_queue,
                            routing_key=f'client_{client_id}_update'
                        )
                        print(f"[AMQP] Declared queue: {update_queue}")
                        
                        # Queue for this client's metrics
                        metrics_queue = f'client_{client_id}_metrics'
                        ch.queue_declare(queue=metrics_queue, durable=True)
                        ch.queue_bind(
                            exchange='fl_client_updates',
                            queue=metrics_queue,
                            routing_key=f'client_{client_id}_metrics'
                        )
                        print(f"[AMQP] Declared queue: {metrics_queue}")
                        
                        conn.close()
                except Exception as e:
                    print(f"[AMQP] Error setting up queues for client {client_id}: {e}")
            elif "/update" in msg.topic:
                data = json.loads(msg.payload.decode())
                if data.get('type') == 'update_chunk':
                    reconstructed = self._handle_transport_update_chunk(data, 'mqtt')
                    if reconstructed is not None:
                        self.handle_client_update(reconstructed, 'mqtt')
                else:
                    self.handle_client_update(data, 'mqtt')
            elif "/metrics" in msg.topic:
                data = json.loads(msg.payload.decode())
                self.handle_client_metrics(data, 'mqtt')
        except Exception as e:
            print(f"[MQTT] Error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def send_via_mqtt(self, client_id, topic, message):
        """Send message to client via MQTT"""
        try:
            payload = json.dumps(message)
            self.mqtt_client.publish(topic, payload, qos=1)
            log_sent_packet(
                packet_size=len(payload),
                peer=f"client_{client_id}",
                protocol="MQTT",
                round=self.current_round,
                extra_info=topic
            )
        except Exception as e:
            print(f"[MQTT] Error sending to client {client_id}: {e}")
    
    # =========================================================================
    # AMQP PROTOCOL HANDLERS
    # =========================================================================
    
    def start_amqp_server(self):
        """Start AMQP protocol handler with application-level retry for broker startup race."""
        if not AMQP_AVAILABLE:
            return

        def _try_connect():
            credentials = pika.PlainCredentials('guest', 'guest')
            parameters = pika.ConnectionParameters(
                host=AMQP_BROKER,
                port=AMQP_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=600,
                connection_attempts=3,
                retry_delay=2,
                frame_max=AMQP_MAX_FRAME_BYTES
            )

            self.amqp_connection = pika.BlockingConnection(parameters)
            self.amqp_channel = self.amqp_connection.channel()

            self.amqp_send_connection = pika.BlockingConnection(parameters)
            self.amqp_send_channel = self.amqp_send_connection.channel()

            self.amqp_channel.exchange_declare(
                exchange='fl_client_updates',
                exchange_type='direct',
                durable=True
            )

            self.amqp_channel.queue_declare(queue='fl_client_register', durable=True)
            self.amqp_channel.queue_declare(queue='fl_client_update', durable=True)
            self.amqp_channel.queue_declare(queue='fl_client_metrics', durable=True)
            self.amqp_channel.queue_bind(
                exchange='fl_client_updates',
                queue='fl_client_update',
                routing_key='client.update'
            )
            self.amqp_channel.queue_bind(
                exchange='fl_client_updates',
                queue='fl_client_metrics',
                routing_key='client.metrics'
            )

            self.amqp_channel.basic_consume(
                queue='fl_client_register',
                on_message_callback=self.on_amqp_register,
                auto_ack=True
            )

            def consume():
                try:
                    self.amqp_channel.start_consuming()
                except Exception as e:
                    print(f"[AMQP] Consumer error: {e}")

            amqp_thread = threading.Thread(target=consume, daemon=True)
            amqp_thread.start()
            print("[AMQP] Server started with separate send/receive connections")

        def _start_with_retry():
            max_attempts = 15
            delay = 5
            for attempt in range(1, max_attempts + 1):
                try:
                    _try_connect()
                    return
                except Exception as e:
                    if attempt < max_attempts:
                        print(f"[AMQP] Server startup attempt {attempt}/{max_attempts} failed: {e} — retrying in {delay}s")
                        time.sleep(delay)
                        delay = min(delay * 2, 30)
                    else:
                        print(f"[AMQP] Server failed to start after {max_attempts} attempts: {e}")

        retry_thread = threading.Thread(target=_start_with_retry, daemon=True, name="AMQP-Server-Start")
        retry_thread.start()
    
    def on_amqp_register(self, ch, method, properties, body):
        """AMQP registration callback"""
        try:
            log_received_packet(
                packet_size=len(body),
                peer="amqp_client",
                protocol="AMQP",
                round=self.current_round,
                extra_info="registration"
            )
            
            data = json.loads(body.decode())
            client_id = data['client_id']
            self.handle_client_registration(client_id, 'amqp')
            
            # Just declare the queues but don't set up separate consumers
            # The messages will be processed via polling in the consumer thread
            try:
                # Queue for this client's updates
                update_queue = f'client_{client_id}_updates'
                ch.queue_declare(queue=update_queue, durable=True)
                ch.queue_bind(
                    exchange='fl_client_updates',
                    queue=update_queue,
                    routing_key=f'client_{client_id}_update'
                )
                print(f"[AMQP] Declared queue: {update_queue} with routing_key: client_{client_id}_update")
                
                # Queue for this client's metrics
                metrics_queue = f'client_{client_id}_metrics'
                ch.queue_declare(queue=metrics_queue, durable=True)
                ch.queue_bind(
                    exchange='fl_client_updates',
                    queue=metrics_queue,
                    routing_key=f'client_{client_id}_metrics'
                )
                print(f"[AMQP] Declared queue: {metrics_queue} with routing_key: client_{client_id}_metrics")
                
                print(f"[AMQP] Declared queues for client {client_id}")
            except Exception as e:
                print(f"[AMQP] Error declaring queues for client {client_id}: {e}")
        except Exception as e:
            print(f"[AMQP] Error handling registration: {e}")
            import traceback
            traceback.print_exc()
    
    # Note: AMQP update and metrics callbacks are now defined inline in on_amqp_register
    # to handle separate connections per client
    
    def _process_amqp_update_body(self, body):
        """Handle one AMQP update body using standalone routing keys."""
        log_received_packet(
            packet_size=len(body),
            peer="amqp_client",
            protocol="AMQP",
            round=self.current_round,
            extra_info="model_update"
        )
        data = json.loads(body.decode())
        if data.get('type') == 'update_chunk':
            reconstructed = self._handle_transport_update_chunk(data, 'amqp')
            if reconstructed is not None:
                self.handle_client_update(reconstructed, 'amqp')
                print(f"[AMQP] Received reassembled update from client {reconstructed.get('client_id')}")
        else:
            self.handle_client_update(data, 'amqp')
            print(f"[AMQP] Received update from client {data.get('client_id')}")
    
    def _process_amqp_metrics_body(self, body):
        """Handle one AMQP metrics body using standalone routing keys."""
        log_received_packet(
            packet_size=len(body),
            peer="amqp_client",
            protocol="AMQP",
            round=self.current_round,
            extra_info="metrics"
        )
        data = json.loads(body.decode())
        self.handle_client_metrics(data, 'amqp')
        print(f"[AMQP] Received metrics from client {data.get('client_id')}")
    
    async def handle_quic_message(self, message, protocol):
        """Handle incoming QUIC messages asynchronously"""
        try:
            msg_type = message.get('type')
            client_id = message.get('client_id', 'unknown')
            
            # Store QUIC protocol reference for ANY message (not just registration)
            if client_id and client_id != 'unknown':
                self.quic_clients[client_id] = protocol
                print(f"[QUIC] Stored protocol reference for client {client_id}")
            
            # Use asyncio.to_thread to call synchronous methods from async context
            # This ensures thread-safe execution of methods with locks
            loop = asyncio.get_event_loop()
            
            if msg_type == 'register':
                await loop.run_in_executor(None, self.handle_client_registration, client_id, 'quic')
                print(f"[QUIC] Received registration from client {client_id}")
            elif msg_type in ('update', 'model_update'):
                await loop.run_in_executor(None, self.handle_client_update, message, 'quic')
                print(f"[QUIC] Received update from client {client_id}")
            elif msg_type == 'metrics':
                await loop.run_in_executor(None, self.handle_client_metrics, message, 'quic')
                print(f"[QUIC] Received metrics from client {client_id}")
        except Exception as e:
            print(f"[QUIC] Error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def send_quic_message(self, client_id, message):
        """Send message to client via QUIC stream. Supports multiple QUIC clients (each has own connection in quic_clients)."""
        if client_id not in self.quic_clients:
            print(f"[QUIC] Warning: No QUIC protocol reference for client {client_id}")
            print(f"[QUIC] Available QUIC clients: {list(self.quic_clients.keys())}")
            return
        
        try:
            protocol = self.quic_clients[client_id]
            stream_id = protocol._quic.get_next_available_stream_id(is_unidirectional=False)
            # Add newline delimiter for message framing
            data = (json.dumps(message) + '\n').encode('utf-8')
            protocol._quic.send_stream_data(stream_id, data, end_stream=True)
            protocol.transmit()
            
            msg_type = message.get('type')
            print(f"[QUIC] Sent {msg_type} to client {client_id} on stream {stream_id} ({len(data)} bytes)")
            
            log_sent_packet(
                packet_size=len(data),
                peer=f"quic_client_{client_id}",
                protocol="QUIC",
                round=self.current_round,
                extra_info=msg_type
            )
        except Exception as e:
            print(f"[QUIC] Error sending message to client {client_id}: {e}")
            import traceback
            traceback.print_exc()
    
    def send_via_amqp(self, client_id, message_type, message):
        """Send message to client via AMQP using dedicated send connection"""
        if not AMQP_AVAILABLE:
            return
        
        try:
            # Use dedicated send connection (thread-safe, not shared with consumer)
            if not self.amqp_send_channel or not self.amqp_send_channel.is_open:
                print(f"[AMQP] Send channel closed, reopening...")
                if self.amqp_send_connection and self.amqp_send_connection.is_open:
                    self.amqp_send_channel = self.amqp_send_connection.channel()
                else:
                    # Recreate send connection
                    print(f"[AMQP] Send connection closed, recreating...")
                    credentials = pika.PlainCredentials('guest', 'guest')
                    # FAIR CONFIG: heartbeat=600s for very_poor network scenarios
                    parameters = pika.ConnectionParameters(
                        host=AMQP_BROKER,
                        port=AMQP_PORT,
                        credentials=credentials,
                        heartbeat=600,  # 10 minutes for very_poor network
                        blocked_connection_timeout=600,  # Aligned with heartbeat
                        frame_max=AMQP_MAX_FRAME_BYTES  # 128 KB
                    )
                    print("Before sending message type:", message_type)
                    self.amqp_send_connection = pika.BlockingConnection(parameters)
                    self.amqp_send_channel = self.amqp_send_connection.channel()
                    print("After sending message type:", message_type)
            
            payload = json.dumps(message)
            queue_name = f'client_{client_id}_{message_type}'
            
            # Declare queue if not exists (safe on send channel)
            self.amqp_send_channel.queue_declare(queue=queue_name, durable=True)
            
            self.amqp_send_channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=payload,
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            log_sent_packet(
                packet_size=len(payload),
                peer=f"client_{client_id}",
                protocol="AMQP",
                round=self.current_round,
                extra_info=message_type
            )
            print(f"[AMQP] Sent {message_type} to client {client_id} (queue: {queue_name})")
        except Exception as e:
            print(f"[AMQP] Error sending to client {client_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # gRPC PROTOCOL HANDLERS
    # =========================================================================
    
    def start_grpc_server(self):
        """Start gRPC protocol handler"""
        if not GRPC_AVAILABLE:
            return
        
        try:
            # FAIR CONFIG: Aligned with MQTT/AMQP/QUIC/DDS for unbiased comparison
            self.grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    # Realistic max payload: gRPC 4 MB
                    ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
                    ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
                    # FAIR CONFIG: Keepalive settings 600s for very_poor network
                    ('grpc.keepalive_time_ms', 600000),  # 10 minutes
                    ('grpc.keepalive_timeout_ms', 60000),  # 1 minute timeout
                    ('grpc.keepalive_permit_without_calls', 1),
                    ('grpc.http2.max_pings_without_data', 0),
                    ('grpc.http2.min_time_between_pings_ms', 10000),  # 10s
                    ('grpc.http2.max_ping_strikes', 2),
                ]
            )
            
            federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(
                FLServicer(self), self.grpc_server
            )
            
            self.grpc_server.add_insecure_port(f'[::]:{GRPC_PORT}')
            self.grpc_server.start()
            print(f"[gRPC] Server started on port {GRPC_PORT}")
        except Exception as e:
            print(f"[gRPC] Failed to start: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # QUIC PROTOCOL HANDLERS
    # =========================================================================
    
    def start_quic_server(self):
        """Start QUIC protocol handler"""
        if not QUIC_AVAILABLE:
            return
        
        try:
            # Run QUIC server in asyncio event loop
            def run_quic():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    print(f"[QUIC] Starting server on {QUIC_HOST}:{QUIC_PORT}")
                    # Run the QUIC server coroutine directly (not as a task)
                    loop.run_until_complete(self._run_quic_server())
                except Exception as e:
                    print(f"[QUIC] Thread error: {e}")
                    import traceback
                    traceback.print_exc()
            
            quic_thread = threading.Thread(target=run_quic, daemon=True)
            quic_thread.start()
            
            # Wait a bit for server to actually start listening
            time.sleep(3)
            print(f"[QUIC] Server initialized on {QUIC_HOST}:{QUIC_PORT}")
        except Exception as e:
            print(f"[QUIC] Failed to start: {e}")
            import traceback
            traceback.print_exc()
    
    async def _run_quic_server(self):
        """Async QUIC server"""
        try:
            # QUIC config: cubic congestion, 60s idle; flow control 128 MB (aligned with MQTT/gRPC for fair comparison)
            configuration = QuicConfiguration(
                is_client=False,
                alpn_protocols=["fl"],
                congestion_control_algorithm="cubic",
                idle_timeout=60.0,
                max_data=QUIC_HTTP3_MAX_DATA_BYTES,
                max_stream_data=QUIC_HTTP3_MAX_DATA_BYTES,
                max_datagram_frame_size=65536,
            )
            
            # Generate self-signed certificate for QUIC
            import ssl
            from cryptography import x509
            from cryptography.x509.oid import NameOID, ExtensionOID
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            import datetime
            import ipaddress
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Generate certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FL Server"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ])
            
            # Build SubjectAltName extension with multiple addresses
            san_list = [
                x509.DNSName("localhost"),
                x509.DNSName("fl-server-unified-emotion"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
            ]
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False
            ).sign(private_key, hashes.SHA256())
            
            # Save to project-local cert dir to avoid /tmp permission issues
            cert_dir = os.path.join(project_root, ".certs")
            os.makedirs(cert_dir, exist_ok=True)
            cert_path = os.path.join(cert_dir, "quic_cert.pem")
            key_path = os.path.join(cert_dir, "quic_key.pem")
            
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            with open(key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            configuration.load_cert_chain(cert_path, key_path)
            
            print(f"[QUIC] Server configuration complete:")
            print(f"[QUIC]   - Host: {QUIC_HOST}:{QUIC_PORT}")
            print(f"[QUIC]   - Certificate: {cert_path}")
            print(f"[QUIC]   - ALPN: fl")
            print(f"[QUIC]   - Max stream data: {QUIC_HTTP3_MAX_DATA_BYTES // (1024*1024)} MB")
            print(f"[QUIC]   - Idle timeout: 3600s")
            
            # Create protocol factory that sets server reference
            def create_protocol(*args, **kwargs):
                protocol = QUICServerProtocol(*args, **kwargs)
                protocol.server = self
                print(f"[QUIC] Server created protocol instance for new connection")
                return protocol
            
            print(f"[QUIC] Server starting serve() on {QUIC_HOST}:{QUIC_PORT}")
            await serve(
                QUIC_HOST,
                QUIC_PORT,
                configuration=configuration,
                create_protocol=create_protocol,
            )
            print(f"[QUIC] ✓ Server serve() completed, now running indefinitely")
            print(f"[QUIC] Server is ready to accept connections")
            # Keep server running indefinitely
            await asyncio.Future()  # Run forever
        except Exception as e:
            print(f"[QUIC] Server error: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # HTTP/3 PROTOCOL HANDLERS
    # =========================================================================
    
    def start_http3_server(self):
        """Start HTTP/3 protocol handler"""
        if not HTTP3_AVAILABLE:
            return
        
        try:
            # Run HTTP/3 server in asyncio event loop
            def run_http3():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    print(f"[HTTP/3] Starting server on {HTTP3_HOST}:{HTTP3_PORT}")
                    # Run the HTTP/3 server coroutine directly
                    loop.run_until_complete(self._run_http3_server())
                except Exception as e:
                    print(f"[HTTP/3] Thread error: {e}")
                    import traceback
                    traceback.print_exc()
            
            http3_thread = threading.Thread(target=run_http3, daemon=True)
            http3_thread.start()
            
            # Wait a bit for server to actually start listening
            time.sleep(3)
            print(f"[HTTP/3] Server initialized on {HTTP3_HOST}:{HTTP3_PORT}")
        except Exception as e:
            print(f"[HTTP/3] Failed to start: {e}")
            import traceback
            traceback.print_exc()
    
    async def _run_http3_server(self):
        """Async HTTP/3 server"""
        try:
            # Realistic max payload: HTTP/3 16 KB per stream
            configuration = QuicConfiguration(
                is_client=False,
                alpn_protocols=H3_ALPN,
                congestion_control_algorithm="cubic",
                idle_timeout=60.0,
                max_data=HTTP3_MAX_STREAM_DATA * 4,  # 64 KB total
                max_stream_data=HTTP3_MAX_STREAM_DATA,  # 16 KB per stream
                max_datagram_frame_size=65536,
            )
            
            # Generate self-signed certificate for HTTP/3 (reuse QUIC cert generation logic)
            import ssl
            from cryptography import x509
            from cryptography.x509.oid import NameOID, ExtensionOID
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            import datetime
            import ipaddress
            
            # Generate private key
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
            )
            
            # Generate certificate
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FL Server"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ])
            
            # Build SubjectAltName extension with multiple addresses
            san_list = [
                x509.DNSName("localhost"),
                x509.DNSName("fl-server-unified-emotion"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
            ]
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.datetime.utcnow()
            ).not_valid_after(
                datetime.datetime.utcnow() + datetime.timedelta(days=365)
            ).add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False
            ).sign(private_key, hashes.SHA256())
            
            # Save to project-local cert dir to avoid /tmp permission issues
            cert_dir = os.path.join(project_root, ".certs")
            os.makedirs(cert_dir, exist_ok=True)
            cert_path = os.path.join(cert_dir, "http3_cert.pem")
            key_path = os.path.join(cert_dir, "http3_key.pem")
            
            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            
            with open(key_path, "wb") as f:
                f.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            configuration.load_cert_chain(cert_path, key_path)
            
            print(f"[HTTP/3] Server configuration complete:")
            print(f"[HTTP/3]   - Host: {HTTP3_HOST}:{HTTP3_PORT}")
            print(f"[HTTP/3]   - Certificate: {cert_path}")
            print(f"[HTTP/3]   - ALPN: {H3_ALPN}")
            print(f"[HTTP/3]   - Max stream data: {HTTP3_MAX_STREAM_DATA // 1024} KB")
            print(f"[HTTP/3]   - Idle timeout: 600s")
            
            # Create protocol factory that sets server reference
            def create_protocol(*args, **kwargs):
                protocol = HTTP3ServerProtocol(*args, **kwargs)
                protocol.server = self
                return protocol
            
            print(f"[HTTP/3] Server starting serve() on {HTTP3_HOST}:{HTTP3_PORT}")
            await serve(
                HTTP3_HOST,
                HTTP3_PORT,
                configuration=configuration,
                create_protocol=create_protocol,
            )
            # Keep server running indefinitely
            await asyncio.Future()  # Run forever
        except Exception as e:
            print(f"[HTTP/3] Server error: {e}")
            import traceback
            traceback.print_exc()
    
    async def handle_http3_message(self, message, protocol):
        """Handle incoming HTTP/3 messages asynchronously"""
        try:
            msg_type = message.get('type')
            client_id = message.get('client_id', 'unknown')
            
            # Store HTTP/3 protocol reference for ANY message (not just registration)
            if client_id and client_id != 'unknown':
                self.http3_clients[client_id] = protocol
            
            # Use asyncio.to_thread to call synchronous methods from async context
            # This ensures thread-safe execution of methods with locks
            loop = asyncio.get_event_loop()
            
            if msg_type == 'register':
                await loop.run_in_executor(None, self.handle_client_registration, client_id, 'http3')
            elif msg_type in ('update', 'model_update'):
                await loop.run_in_executor(None, self.handle_client_update, message, 'http3')
            elif msg_type == 'update_chunk':
                reconstructed = await loop.run_in_executor(None, self._handle_transport_update_chunk, message, 'http3')
                if reconstructed is not None:
                    await loop.run_in_executor(None, self.handle_client_update, reconstructed, 'http3')
            elif msg_type == 'metrics':
                await loop.run_in_executor(None, self.handle_client_metrics, message, 'http3')
        except Exception as e:
            print(f"[HTTP/3] Error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def send_http3_message(self, client_id, message):
        """Send message to client via HTTP/3 stream. Supports multiple HTTP/3 clients."""
        if client_id not in self.http3_clients:
            print(f"[HTTP/3] Warning: No HTTP/3 protocol reference for client {client_id}")
            print(f"[HTTP/3] Available HTTP/3 clients: {list(self.http3_clients.keys())}")
            return
        
        try:
            protocol = self.http3_clients[client_id]
            # Ensure HTTP connection is initialized
            if protocol._http is None:
                protocol._http = H3Connection(protocol._quic)
            
            # Get next available stream ID (bidirectional for server push)
            stream_id = protocol._quic.get_next_available_stream_id(is_unidirectional=False)
            
            # Prepare JSON payload
            payload = json.dumps(message).encode('utf-8')
            
            # Send headers (server push)
            headers = [
                (b":status", b"200"),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ]
            protocol._http.send_headers(stream_id=stream_id, headers=headers)
            
            # Send data
            protocol._http.send_data(stream_id=stream_id, data=payload, end_stream=True)
            protocol.transmit()
            
            msg_type = message.get('type')
            
            log_sent_packet(
                packet_size=len(payload),
                peer=f"http3_client_{client_id}",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info=msg_type
            )
        except Exception as e:
            print(f"[HTTP/3] Error sending message to client {client_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # DDS PROTOCOL HANDLERS
    # =========================================================================
    
    def start_amqp_consumer(self):
        """Polling-based AMQP consumer for all clients, with startup retry."""
        max_connect_attempts = 15
        connect_delay = 5
        for connect_attempt in range(1, max_connect_attempts + 1):
            try:
                credentials = pika.PlainCredentials('guest', 'guest')
                parameters = pika.ConnectionParameters(
                    host=AMQP_BROKER,
                    port=AMQP_PORT,
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=600,
                    connection_attempts=3,
                    retry_delay=2,
                    frame_max=AMQP_MAX_FRAME_BYTES
                )
                self.amqp_consumer_connection = pika.BlockingConnection(parameters)
                self.amqp_consumer_channel = self.amqp_consumer_connection.channel()
                print("[AMQP] Consumer connection established")
                break  # connected — proceed to polling loop below
            except Exception as e:
                if connect_attempt < max_connect_attempts:
                    print(f"[AMQP] Consumer connect attempt {connect_attempt}/{max_connect_attempts} failed: {e} — retrying in {connect_delay}s")
                    time.sleep(connect_delay)
                    connect_delay = min(connect_delay * 2, 30)
                else:
                    print(f"[AMQP] Consumer failed to connect after {max_connect_attempts} attempts: {e}")
                    import traceback
                    traceback.print_exc()
                    return
        try:
            # Give time for initial setup
            time.sleep(3)
            
            # Run polling loop
            poll_count = 0
            consecutive_empty_checks = 0
            check_connection_health = 0
            
            while self.running:
                try:
                    # Periodically log polling status and check connection health
                    poll_count += 1
                    check_connection_health += 1
                    
                    if check_connection_health % 200 == 0:
                        print(f"[AMQP] Polling... ({poll_count} iterations, {len(self.registered_clients)} clients: {list(self.registered_clients.keys())})")
                        print(f"[AMQP] Connection status: open={self.amqp_consumer_connection.is_open}, channel open={self.amqp_consumer_channel.is_open}")
                    
                    # Poll ALL client queues (1 to num_clients), not just registered ones
                    # This ensures we receive messages even before/during registration
                    found_messages = False
                    try:
                        method, properties, body = self.amqp_consumer_channel.basic_get(
                            queue='fl_client_update',
                            auto_ack=False
                        )
                        if body is not None:
                            found_messages = True
                            consecutive_empty_checks = 0
                            try:
                                self._process_amqp_update_body(body)
                                self.amqp_consumer_channel.basic_ack(delivery_tag=method.delivery_tag)
                            except Exception as e:
                                print(f"[AMQP] Error processing shared update: {e}")
                                try:
                                    self.amqp_consumer_channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    
                    try:
                        method, properties, body = self.amqp_consumer_channel.basic_get(
                            queue='fl_client_metrics',
                            auto_ack=False
                        )
                        if body is not None:
                            found_messages = True
                            consecutive_empty_checks = 0
                            try:
                                self._process_amqp_metrics_body(body)
                                self.amqp_consumer_channel.basic_ack(delivery_tag=method.delivery_tag)
                            except Exception as e:
                                print(f"[AMQP] Error processing shared metrics: {e}")
                                try:
                                    self.amqp_consumer_channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    
                    for client_id in range(1, self.num_clients + 1):
                        update_queue = f'client_{client_id}_updates'
                        
                        try:
                            # Ensure queue exists and is bound to exchange
                            # (in case it wasn't declared yet)
                            try:
                                self.amqp_consumer_channel.queue_declare(
                                    queue=update_queue, 
                                    durable=True,
                                    passive=False  # Create if doesn't exist
                                )
                                self.amqp_consumer_channel.queue_bind(
                                    exchange='fl_client_updates',
                                    queue=update_queue,
                                    routing_key=f'client_{client_id}_update'
                                )
                            except:
                                # Queue already exists, that's fine
                                pass
                            
                            # Use explicit no_ack=False and auto_ack handling
                            method, properties, body = self.amqp_consumer_channel.basic_get(
                                queue=update_queue, 
                                auto_ack=False
                            )
                            
                            # Debug log every 500 iterations
                            if check_connection_health % 500 == 0:
                                print(f"[AMQP-DEBUG] basic_get({update_queue}): method={method}, body={'<data>' if body else None}")
                                
                            if body is not None:  # Important: check "is not None" not just "if body"
                                found_messages = True
                                consecutive_empty_checks = 0
                                try:
                                    log_received_packet(
                                        packet_size=len(body),
                                        peer=f"client_{client_id}",
                                        protocol="AMQP",
                                        round=self.current_round,
                                        extra_info="model_update"
                                    )
                                    data = json.loads(body.decode())
                                    if data.get('type') == 'update_chunk':
                                        reconstructed = self._handle_transport_update_chunk(data, 'amqp')
                                        if reconstructed is not None:
                                            self.handle_client_update(reconstructed, 'amqp')
                                            print(f"[AMQP] Received reassembled update from client {client_id}")
                                    else:
                                        self.handle_client_update(data, 'amqp')
                                        print(f"[AMQP] Received update from client {client_id}")
                                    # Acknowledge the message
                                    self.amqp_consumer_channel.basic_ack(delivery_tag=method.delivery_tag)
                                except Exception as e:
                                    print(f"[AMQP] Error processing update from client {client_id}: {e}")
                                    # Negative acknowledge with requeue
                                    try:
                                        self.amqp_consumer_channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                                    except:
                                        pass
                        except Exception as e:
                            # Queue may not exist yet, skip but log less frequently
                            if poll_count % 1000 == 0:
                                import traceback
                                print(f"[AMQP] Error polling {update_queue}: {e}")
                                traceback.print_exc()
                        
                        # Poll metrics queue
                        metrics_queue = f'client_{client_id}_metrics'
                        try:
                            # Ensure queue exists and is bound to exchange
                            try:
                                self.amqp_consumer_channel.queue_declare(
                                    queue=metrics_queue, 
                                    durable=True,
                                    passive=False
                                )
                                self.amqp_consumer_channel.queue_bind(
                                    exchange='fl_client_updates',
                                    queue=metrics_queue,
                                    routing_key=f'client_{client_id}_metrics'
                                )
                            except:
                                # Queue already exists, that's fine
                                pass
                            
                            method, properties, body = self.amqp_consumer_channel.basic_get(
                                queue=metrics_queue, 
                                auto_ack=False
                            )
                            
                            if body:
                                found_messages = True
                                try:
                                    log_received_packet(
                                        packet_size=len(body),
                                        peer=f"client_{client_id}",
                                        protocol="AMQP",
                                        round=self.current_round,
                                        extra_info="metrics"
                                    )
                                    data = json.loads(body.decode())
                                    self.handle_client_metrics(data, 'amqp')
                                    print(f"[AMQP] Received metrics from client {client_id}")
                                    # Acknowledge the message
                                    self.amqp_consumer_channel.basic_ack(delivery_tag=method.delivery_tag)
                                except Exception as e:
                                    print(f"[AMQP] Error processing metrics from client {client_id}: {e}")
                                    # Negative acknowledge with requeue
                                    try:
                                        self.amqp_consumer_channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
                                    except:
                                        pass
                        except Exception as e:
                            pass
                    
                    # If we found messages, keep polling immediately; otherwise, wait slightly
                    if not found_messages:
                        consecutive_empty_checks += 1
                        if consecutive_empty_checks > 100:
                            # Reduce console spam after lots of empty checks
                            time.sleep(0.2)
                    else:
                        time.sleep(0.01)  # Very short delay if we found messages
                        
                except Exception as e:
                    print(f"[AMQP] Consumer polling error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
        except Exception as e:
            print(f"[AMQP] Consumer polling loop failed: {e}")
            import traceback
            traceback.print_exc()
    
    def start_dds_server(self):
        """Start DDS protocol handler"""
        if not DDS_AVAILABLE:
            return
        
        try:
            # Same as FL_Server_DDS.setup_dds: refresh URI from DDS_PEER_* / temp XML before participant.
            _ensure_server_cyclonedds_uri()
            uri = os.environ.get("CYCLONEDDS_URI")
            print(f"[DDS] CYCLONEDDS_URI={uri or '(not set)'}")
            print(f"[DDS] Setting up DDS on domain {DDS_DOMAIN_ID}...")
            # Create DDS participant
            self.dds_participant = DomainParticipant(DDS_DOMAIN_ID)
            
            # Match FL_Server_DDS.setup_dds QoS so writers/readers interoperate with FL_Client_DDS / unified client.
            reliable_qos = Qos(
                Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                Policy.History.KeepLast(10),
                Policy.Durability.TransientLocal
            )
            reliable_qos_large = Qos(
                Policy.Reliability.Reliable(max_blocking_time=duration(seconds=600)),
                Policy.History.KeepLast(10),
                Policy.Durability.TransientLocal,
                Policy.ResourceLimits(max_samples=10, max_instances=10, max_samples_per_instance=10),
            )
            _chunk_blk = min(600.0, max(1.0, float(DDS_CHUNK_MAX_BLOCKING_SEC)))
            chunk_qos = Qos(
                Policy.Reliability.Reliable(max_blocking_time=duration(seconds=_chunk_blk)),
                Policy.History.KeepLast(2048),
                Policy.Durability.TransientLocal,
            )
            
            # Create topics and readers for model updates
            update_topic = Topic(self.dds_participant, "ModelUpdate", ModelUpdate)
            update_reader = DataReader(self.dds_participant, update_topic, qos=reliable_qos_large)
            update_chunk_topic = Topic(self.dds_participant, "ModelUpdateChunk", ModelUpdateChunk)
            update_chunk_reader = DataReader(self.dds_participant, update_chunk_topic, qos=chunk_qos)
            self._dds_update_chunk_reader = update_chunk_reader
            
            # Create topics and readers for metrics
            metrics_topic = Topic(self.dds_participant, "EvaluationMetrics", EvaluationMetrics)
            metrics_reader = DataReader(self.dds_participant, metrics_topic, qos=reliable_qos)
            
            # Create writers for sending global model and commands to clients
            global_model_topic = Topic(self.dds_participant, "GlobalModel", GlobalModel)
            self.dds_writers['global_model'] = DataWriter(self.dds_participant, global_model_topic, qos=reliable_qos)
            global_model_chunk_topic = Topic(self.dds_participant, "GlobalModelChunk", GlobalModelChunk)
            self.dds_writers['global_model_chunk'] = DataWriter(self.dds_participant, global_model_chunk_topic, qos=chunk_qos)
            
            command_topic = Topic(self.dds_participant, "TrainingCommand", TrainingCommand)
            self.dds_writers['command'] = DataWriter(self.dds_participant, command_topic, qos=reliable_qos)
            time.sleep(0.5)  # discovery ramp-up (FL_Server_DDS.setup_dds)
            
            # Create DDS listener thread that polls for messages
            def dds_listener():
                while self.running:
                    try:
                        # FAIR CONFIG: Read chunked model updates (matching standalone)
                        chunk_samples = list(update_chunk_reader.take(50))  # Read more chunks at once
                        for sample in chunk_samples:
                            if sample and hasattr(sample, 'client_id'):
                                client_id = sample.client_id
                                chunk_id = sample.chunk_id
                                total_chunks = sample.total_chunks
                                round_num = sample.round
                                # Same guards as FL_Server_DDS.check_model_updates
                                if round_num != self.current_round:
                                    continue
                                if client_id not in self.active_clients or client_id in self.client_updates:
                                    continue

                                meta = self.model_update_metadata.get(client_id)
                                buf = self.model_update_chunks.get(client_id)
                                need_reset = False
                                if meta is not None:
                                    if meta.get('round') != round_num or int(meta.get('total_chunks', -1)) != int(total_chunks):
                                        need_reset = True
                                if chunk_id == 0 and buf:
                                    need_reset = True
                                if need_reset:
                                    self.model_update_chunks.pop(client_id, None)
                                    self.model_update_metadata.pop(client_id, None)

                                if client_id not in self.model_update_chunks:
                                    self.model_update_chunks[client_id] = {}
                                    self.model_update_metadata[client_id] = {
                                        'round': round_num,
                                        'total_chunks': total_chunks,
                                        'num_samples': sample.num_samples,
                                        'loss': sample.loss,
                                        'accuracy': sample.accuracy,
                                    }

                                self.model_update_chunks[client_id][chunk_id] = sample.payload
                                if chunk_id == 0:
                                    print(
                                        f"[DDS] Chunked model update from client {client_id} "
                                        f"round {round_num} ({total_chunks} chunks) — receiving…"
                                    )

                                # Check if all chunks received for this client
                                if len(self.model_update_chunks[client_id]) == total_chunks:
                                    # Reassemble chunks in order
                                    reassembled_data = []
                                    for i in range(total_chunks):
                                        if i in self.model_update_chunks[client_id]:
                                            reassembled_data.extend(self.model_update_chunks[client_id][i])
                                        else:
                                            print(f"ERROR: Missing chunk {i} from client {client_id}")
                                            break
                                    
                                    # Only process if we have all chunks
                                    if len(reassembled_data) > 0:
                                        # Deserialize: may be weights (list) or quantized compressed_data (dict)
                                        unpacked = pickle.loads(bytes(reassembled_data))
                                        metadata = self.model_update_metadata[client_id]
                                        if isinstance(unpacked, dict) and ('quantization_params' in unpacked or 'compressed_data' in unpacked):
                                            data = {
                                                'client_id': client_id,
                                                'round': sample.round,
                                                'compressed_data': unpacked,
                                                'num_samples': metadata['num_samples'],
                                                'loss': metadata['loss'],
                                                'accuracy': metadata['accuracy'],
                                            }
                                        else:
                                            data = {
                                                'client_id': client_id,
                                                'round': sample.round,
                                                'weights': unpacked,
                                                'num_samples': metadata['num_samples'],
                                                'loss': metadata['loss'],
                                                'accuracy': metadata['accuracy'],
                                            }
                                        
                                        log_received_packet(
                                            packet_size=len(reassembled_data),
                                            peer=f"client_{client_id}",
                                            protocol="DDS",
                                            round=self.current_round,
                                            extra_info="model_update_chunked"
                                        )
                                        
                                        self.handle_client_update(data, 'dds')
                                        
                                        # Clear chunk buffers for this client
                                        del self.model_update_chunks[client_id]
                                        del self.model_update_metadata[client_id]
                                        
                                        print(f"Successfully reassembled and processed update from client {client_id}")
                        
                        # Read legacy non-chunked model updates (for backwards compatibility)
                        samples_read = list(update_reader.take(10))
                        if samples_read:
                            print(f"[DDS] Read {len(samples_read)} update samples from DDS")
                        for sample in samples_read:
                            if sample:
                                try:
                                    if sample.round != self.current_round:
                                        continue
                                    if (
                                        sample.client_id not in self.active_clients
                                        or sample.client_id in self.client_updates
                                    ):
                                        continue
                                    # Convert List[int] back to bytes and deserialize weights
                                    weights_bytes = bytes(sample.weights)
                                    weights = pickle.loads(weights_bytes)
                                    
                                    # Convert to dict format expected by handler
                                    data = {
                                        'client_id': sample.client_id,
                                        'round': sample.round,
                                        'weights': weights,
                                        'num_samples': sample.num_samples,
                                        'loss': sample.loss,
                                        'metrics': {
                                            'loss': sample.loss,
                                            'accuracy': sample.accuracy,
                                        }
                                    }
                                    
                                    print(f"[DDS] Processing update from client {sample.client_id}, round {sample.round}")
                                    
                                    log_received_packet(
                                        packet_size=len(sample.weights),
                                        peer=f"client_{sample.client_id}",
                                        protocol="DDS",
                                        round=self.current_round,
                                        extra_info="model_update"
                                    )
                                    
                                    self.handle_client_update(data, 'dds')
                                    print(f"[DDS] Received update from client {sample.client_id}")
                                except Exception as e:
                                    print(f"[DDS] Error processing update: {type(e).__name__}: {e}")
                                    import traceback
                                    traceback.print_exc()
                        
                        # Read metrics
                        samples_read = list(metrics_reader.take(10))
                        if samples_read:
                            print(f"[DDS] Read {len(samples_read)} metric samples from DDS")
                        for sample in samples_read:
                            if sample:
                                try:
                                    # Convert to dict format expected by handler
                                    data = {
                                        'client_id': sample.client_id,
                                        'round': sample.round,
                                        'num_samples': sample.num_samples,
                                        'loss': sample.loss,
                                        'accuracy': sample.accuracy,
                                        'battery_soc': sample.battery_soc,
                                        'training_time_sec': getattr(
                                            sample, 'training_time_sec', 0.0
                                        ),
                                        'round_time_sec': getattr(
                                            sample, 'round_time_sec', 0.0
                                        ),
                                        'uplink_model_comm_sec': getattr(
                                            sample, 'uplink_model_comm_sec', 0.0
                                        ),
                                        'metrics': {
                                            'loss': sample.loss,
                                            'accuracy': sample.accuracy,
                                            'client_converged': sample.client_converged,
                                            'battery_soc': sample.battery_soc,
                                            'training_time_sec': getattr(
                                                sample, 'training_time_sec', 0.0
                                            ),
                                            'round_time_sec': getattr(
                                                sample, 'round_time_sec', 0.0
                                            ),
                                            'uplink_model_comm_sec': getattr(
                                                sample, 'uplink_model_comm_sec', 0.0
                                            ),
                                        }
                                    }
                                    
                                    print(f"[DDS] Processing metrics from client {sample.client_id}, round {sample.round}")
                                    
                                    log_received_packet(
                                        packet_size=len(str(sample)),
                                        peer=f"client_{sample.client_id}",
                                        protocol="DDS",
                                        round=self.current_round,
                                        extra_info="metrics"
                                    )
                                    
                                    self.handle_client_metrics(data, 'dds')
                                    print(f"[DDS] Received metrics from client {sample.client_id}")
                                except Exception as e:
                                    print(f"[DDS] Error processing metrics: {type(e).__name__}: {e}")
                                    import traceback
                                    traceback.print_exc()
                        
                        # Small delay to avoid busy-waiting
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"[DDS] Listener error: {e}")
                        import traceback
                        traceback.print_exc()
                        time.sleep(1)
            
            dds_thread = threading.Thread(target=dds_listener, daemon=True)
            dds_thread.start()

            def _dds_chunk_match_diagnostic():
                time.sleep(12.0)
                try:
                    r = getattr(self, "_dds_update_chunk_reader", None)
                    if r is None:
                        return
                    n_matched = None
                    # cyclonedds-python versions differ: some DataReaders lack get_subscription_matched_status.
                    try:
                        pubs = r.get_matched_publications()
                        n_matched = len(pubs) if pubs is not None else 0
                    except Exception:
                        pass
                    if n_matched is None:
                        try:
                            st = r.get_subscription_matched_status()
                            n_matched = int(getattr(st, "current_count", 0))
                        except AttributeError:
                            n_matched = None
                    if n_matched is not None:
                        print(
                            f"[DDS] ModelUpdateChunk reader matched publications: {n_matched} "
                            f"(0 after clients start usually means UDP/locators: same-bridge containers "
                            f"often need DDS_DISABLE_EXTERNAL_ADVERTISE=1 on the server)"
                        )
                    else:
                        print(
                            "[DDS] ModelUpdateChunk reader: cannot query matched writers "
                            "(upgrade cyclonedds or ignore; DDS uplink may still work)"
                        )
                except Exception as ex:
                    print(f"[DDS] ModelUpdateChunk match diagnostic failed: {ex}")

            threading.Thread(target=_dds_chunk_match_diagnostic, daemon=True).start()
            
            print(f"[DDS] Server started on domain {DDS_DOMAIN_ID}")
            if os.path.exists("/.dockerenv"):
                print(
                    "[DDS] Running in Docker: if clients log DDS sends but this server never prints "
                    "'[DDS] Chunked model update', use network_mode:host for unified FL "
                    "(default Docker/docker-compose-unified-emotion.yml). Bridge layout: "
                    "docker-compose-unified-emotion.bridge.yml rarely supports DDS without extra UDP mapping."
                )

        except Exception as e:
            print(f"[DDS] Failed to start: {e}")
            import traceback
            traceback.print_exc()
    
    def receive_dds_update(self, data):
        """Handle DDS message received from client"""
        try:
            log_received_packet(
                packet_size=len(str(data)),
                peer="dds_client",
                protocol="DDS",
                round=self.current_round,
                extra_info="model_update"
            )
            self.handle_client_update(data, 'dds')
        except Exception as e:
            print(f"[DDS] Error handling DDS update: {e}")
    
    def receive_dds_metrics(self, data):
        """Handle DDS metrics received from client"""
        try:
            log_received_packet(
                packet_size=len(str(data)),
                peer="dds_client",
                protocol="DDS",
                round=self.current_round,
                extra_info="metrics"
            )
            self.handle_client_metrics(data, 'dds')
        except Exception as e:
            print(f"[DDS] Error handling DDS metrics: {e}")
    
    # =========================================================================
    # COMMON HANDLERS (Protocol-agnostic)
    # =========================================================================
    
    def handle_client_registration(self, client_id, protocol):
        """Handle client registration (thread-safe)"""
        with self.lock:
            existing_protocol = self.registered_clients.get(client_id)
            if existing_protocol == protocol:
                print(f"[{protocol.upper()}] Client {client_id} already registered via {protocol} - ignoring duplicate")
                return
            
            # Check if this is a late-joining client (training already started)
            is_late_join = self.training_started
            
            self.registered_clients[client_id] = protocol
            # Keep control-plane/bootstrap robust: default downlink preference is gRPC
            # until the client explicitly negotiates a different protocol.
            self.client_delivery_protocols.setdefault(client_id, 'grpc')
            self.active_clients.add(client_id)
            
            if is_late_join:
                print(f"[{protocol.upper()}] ⚡ LATE-JOINING Client {client_id} registered during training (round {self.current_round})")
                print(f"[{protocol.upper()}]   Active clients: {len(self.active_clients)}/{self.num_clients}")
                # Send current global model to late-joining client
                if self.global_weights is not None:
                    print(f"[{protocol.upper()}]   Sending current global model to late-joining client {client_id}")
                    # The client will receive the model via its protocol handler
            else:
                print(f"[{protocol.upper()}] Client {client_id} registered "
                      f"({len(self.registered_clients)}/{self.num_clients})")
            
            if not self.training_started and len(self.registered_clients) >= self.min_clients:
                print(f"\n[Server] All {self.num_clients} clients registered!")
                print("[Server] Distributing initial global model...\n")
                time.sleep(2)
                try:
                    self.distribute_initial_model()
                except Exception as e:
                    print(f"[Server] FATAL: distribute_initial_model() failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print("[Server] Training NOT started; fix the error and restart (or purge stale AMQP queues).")
                    return
                self.training_started = True
                self.start_time = time.time()
                print(f"[Server] Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    def mark_client_converged(self, client_id):
        """Remove converged client from active training set."""
        with self.lock:
            if not stop_on_client_convergence():
                # Fixed-round mode: ignore client-local convergence removal/disconnect.
                return
            if client_id in self.active_clients:
                self.active_clients.remove(client_id)
                self.client_updates.pop(client_id, None)
                self.client_metrics.pop(client_id, None)
                self.grpc_should_train[client_id] = False
                self.grpc_should_evaluate[client_id] = False
                print(f"[Server] Client {client_id} converged and disconnected. "
                      f"Active clients remaining: {len(self.active_clients)}")

                if not self.active_clients:
                    print("[Server] All clients converged. Stopping training.")
                    self.converged = True
                    self.signal_training_complete()
                    return

                # If a client converges mid-round, re-check aggregation thresholds
                # against the reduced active client set so training does not stall.
                if self.client_updates and len(self.client_updates) >= len(self.active_clients):
                    print("[Server] Continuing round after convergence: remaining model updates are sufficient.")
                    self.aggregate_models()
                elif self.client_metrics and len(self.client_metrics) >= len(self.active_clients):
                    print("[Server] Continuing round after convergence: remaining metrics are sufficient.")
                    self.aggregate_metrics()
    
    def handle_client_update(self, data, protocol):
        """Handle client model update (thread-safe)"""
        with self.lock:
            client_id = data['client_id']
            round_num = data['round']
            if not self.training_started:
                if not self._pre_training_update_notice_shown:
                    self._pre_training_update_notice_shown = True
                    print(
                        f"[{protocol.upper()}] Ignoring model updates until training starts "
                        f"(e.g. stale durable AMQP from a prior run). Purge client_*_updates queues or use a clean broker."
                    )
                return
            client_metrics = data.get('metrics') or {}
            if not client_metrics:
                client_metrics = {
                    'loss': float(data.get('loss', 0.0)),
                    'accuracy': float(data.get('accuracy', 0.0)),
                    'client_converged': float(data.get('client_converged', 0.0)),
                }
            
            # Check if this update is for the current round
            # Allow updates only for current round (not past or future)
            if round_num < self.current_round:
                print(f"[{protocol.upper()}] Ignoring old update from client {client_id} "
                      f"(round {round_num} < current {self.current_round})")
                return
            elif round_num > self.current_round:
                print(f"[{protocol.upper()}] WARNING: Received future update from client {client_id} "
                      f"(round {round_num} > current {self.current_round})")
                return
            
            # Track client uplink protocol separately from negotiated downlink preference.
            self.client_uplink_protocols[client_id] = protocol

            if client_id not in self.active_clients:
                print(f"[{protocol.upper()}] Ignoring update from inactive client {client_id}")
                return
            try:
                converged_flag = stop_on_client_convergence() and float(client_metrics.get('client_converged', 0.0)) >= 1.0
            except Exception:
                converged_flag = False
            if converged_flag:
                print(f"[{protocol.upper()}] Received convergence signal from client {client_id}")
                self.mark_client_converged(client_id)
                return
            
            # Deserialize weights
            if 'compressed_data' in data and self.quantization_handler is not None:
                compressed_update = data['compressed_data']
                if isinstance(compressed_update, str):
                    try:
                        compressed_update = pickle.loads(base64.b64decode(compressed_update.encode('utf-8')))
                    except Exception as e:
                        print(f"[Server] Error decoding compressed_data: {e}")
                # Keep quantized end-to-end: do NOT decompress/dequantize on server.
                self.client_updates[client_id] = {
                    'compressed_data': compressed_update,
                    'num_samples': data['num_samples'],
                    'metrics': client_metrics,
                    'protocol': protocol
                }
                weights = None
            else:
                raw_w = data['weights']
                # JSON transports (MQTT/AMQP/gRPC/…) send base64; DDS reassembly passes decoded tensors.
                if isinstance(raw_w, str):
                    weights = self.deserialize_weights(raw_w)
                else:
                    weights = raw_w
            
            # Store update (metrics are handled separately via handle_client_metrics)
            if 'compressed_data' not in data or self.quantization_handler is None:
                self.client_updates[client_id] = {
                    'weights': weights,
                    'num_samples': data['num_samples'],
                    'metrics': client_metrics,
                    'protocol': protocol
                }
            
            print(f"[{protocol.upper()}] Received update from client {client_id} "
                  f"({len(self.client_updates)}/{self.num_clients})")
            
            # Wait for all active clients only
            if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                self.aggregate_models()

    def _handle_transport_update_chunk(self, data, protocol):
        """Reassemble chunked updates received over text-based transports."""
        client_id = data['client_id']
        round_num = data['round']
        total_chunks = int(data.get('total_chunks', 1) or 1)
        chunk_index = int(data.get('chunk_index', 0) or 0)
        payload_chunk = data.get('payload_chunk', '')
        payload_key = data.get('payload_key', 'compressed_data')
        key = (protocol, client_id, round_num)

        if key not in self.transport_update_chunks:
            self.transport_update_chunks[key] = {
                'chunks': {},
                'total_chunks': total_chunks,
                'payload_key': payload_key,
                'num_samples': data.get('num_samples', 0),
                'metrics': data.get('metrics', {}),
                'protocol': data.get('protocol', protocol),
            }

        buf = self.transport_update_chunks[key]
        buf['chunks'][chunk_index] = payload_chunk

        received = len(buf['chunks'])
        if received % 20 == 0 or received == total_chunks:
            print(f"[{protocol.upper()}] Received {received}/{total_chunks} update chunks from client {client_id}")

        if received < total_chunks:
            return None

        try:
            payload = ''.join(buf['chunks'][i] for i in range(total_chunks))
        except KeyError as e:
            print(f"[{protocol.upper()}] Missing chunk {e} for client {client_id}, round {round_num}")
            return None

        reconstructed = {
            'client_id': client_id,
            'round': round_num,
            'num_samples': buf['num_samples'],
            'metrics': buf['metrics'],
            'protocol': buf['protocol'],
            payload_key: payload,
        }
        del self.transport_update_chunks[key]
        print(f"[{protocol.upper()}] Reassembled chunked update from client {client_id} for round {round_num}")
        return reconstructed
    
    def handle_client_metrics(self, data, protocol):
        """Handle client evaluation metrics (thread-safe)"""
        with self.lock:
            client_id = data['client_id']
            round_num = data['round']
            if not self.training_started:
                return
            metrics_payload = data.get('metrics') or {}
            loss_value = float(data.get('loss', metrics_payload.get('loss', 0.0)))
            accuracy_value = float(data.get('accuracy', metrics_payload.get('accuracy', 0.0)))
            battery_soc = float(data.get('battery_soc', metrics_payload.get('battery_soc', 1.0)))
            round_time_sec = float(data.get('round_time_sec', metrics_payload.get('round_time_sec', 0.0)))
            training_time_sec = float(data.get('training_time_sec', metrics_payload.get('training_time_sec', 0.0)))
            uplink_model_comm_sec = float(data.get('uplink_model_comm_sec', metrics_payload.get('uplink_model_comm_sec', 0.0)))
            client_converged = float(data.get('client_converged', metrics_payload.get('client_converged', 0.0)))
            
            print(f"[{protocol.upper()}] handle_client_metrics ENTRY: client_id={client_id}, round={round_num}, current_round={self.current_round}")
            print(f"[{protocol.upper()}] Current client_metrics keys: {list(self.client_metrics.keys())}")
            
            # Check if client already submitted metrics for this round
            if client_id in self.client_metrics:
                existing_round = self.client_metrics[client_id].get('round', -1)
                if existing_round == round_num:
                    print(f"[{protocol.upper()}] Ignoring duplicate metrics from client {client_id} "
                          f"for round {round_num}")
                    return
            
            # Accept metrics for current round only
            if round_num < self.current_round:
                print(f"[{protocol.upper()}] Ignoring old metrics from client {client_id} "
                      f"(round {round_num} < current {self.current_round})")
                return
            elif round_num > self.current_round:
                print(f"[{protocol.upper()}] WARNING: Received future metrics from client {client_id} "
                      f"(round {round_num} > current {self.current_round})")
                return
            
            self.client_metric_protocols[client_id] = protocol

            if client_id not in self.active_clients:
                print(f"[{protocol.upper()}] Ignoring metrics from inactive client {client_id}")
                return
            
            if stop_on_client_convergence() and client_converged >= 1.0:
                print(f"[{protocol.upper()}] Received convergence metrics from client {client_id}")
                self.mark_client_converged(client_id)
                return
            
            _cum_energy = data.get('cumulative_energy_j', (metrics_payload or {}).get('cumulative_energy_j'))
            self.client_metrics[client_id] = {
                'round': round_num,
                'num_samples': data['num_samples'],
                'loss': loss_value,
                'accuracy': accuracy_value,
                'protocol': protocol,
                'battery_soc': battery_soc,
                'round_time_sec': round_time_sec,
                'training_time_sec': training_time_sec,
                'uplink_model_comm_sec': uplink_model_comm_sec,
                'cumulative_energy_j': _cum_energy,
                'metrics': metrics_payload or {
                    'loss': loss_value,
                    'accuracy': accuracy_value,
                    'battery_soc': battery_soc,
                    'round_time_sec': round_time_sec,
                    'training_time_sec': training_time_sec,
                    'uplink_model_comm_sec': uplink_model_comm_sec,
                },
            }
            
            print(f"[{protocol.upper()}] Received metrics from client {client_id} "
                  f"({len(self.client_metrics)}/{self.num_clients})")
            
            # Wait for all active clients only
            if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                print(f"[{protocol.upper()}] All {len(self.active_clients)} active client(s) metrics received! Triggering aggregate_metrics()")
                self.aggregate_metrics()
    
    def distribute_initial_model(self):
        """Distribute initial global model and config to all clients"""
        # Prepare weights (quantized or not)
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        for client_id in self.registered_clients.keys():
            if client_id not in self.active_clients:
                continue
            try:
                # Unified mode always exposes the initial model through gRPC so the
                # client can bootstrap even when the selected data plane cannot carry
                # the full model payload.
                self.grpc_model_ready[client_id] = 0
                print(f"[gRPC] Initial model ready for client {client_id} to pull")
            except Exception as e:
                print(f"[gRPC] Error preparing initial model for client {client_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Start first round
        time.sleep(2)
        self.current_round = 1
        self.round_start_time = time.time()
        self.signal_start_training()

    def signal_start_training(self):
        """Signal all clients to start training for current round"""
        # Unified RL requirement: use gRPC control signaling only.
        for client_id in self.registered_clients.keys():
            if client_id not in self.active_clients:
                continue
            # gRPC - set one-shot flag for client polling
            try:
                self.grpc_should_train[client_id] = True
                self.grpc_should_evaluate[client_id] = False
                print(f"[gRPC] start_training flagged for client {client_id} (round {self.current_round})")
            except Exception as e:
                pass
    
    def signal_start_evaluation(self):
        """Signal all clients to start evaluation for current round"""
        # Unified RL requirement: use gRPC control signaling only.
        for client_id in self.registered_clients.keys():
            if client_id not in self.active_clients:
                continue
            # gRPC - set one-shot flag for client polling
            try:
                self.grpc_should_train[client_id] = False
                self.grpc_should_evaluate[client_id] = True
                print(f"[gRPC] start_evaluation flagged for client {client_id} (round {self.current_round})")
            except Exception as e:
                pass

    def _get_delivery_protocol(self, client_id):
        """Resolve negotiated downlink protocol for a specific client."""
        return self.client_delivery_protocols.get(
            client_id,
            'grpc'
        )

    def _normalize_protocol_name(self, protocol_name):
        """Normalize and validate protocol names used for downlink negotiation."""
        if protocol_name is None:
            return None
        normalized = str(protocol_name).strip().lower()
        allowed = {'mqtt', 'amqp', 'grpc', 'quic', 'http3', 'dds'}
        return normalized if normalized in allowed else None

    def prepare_downlink_protocol_negotiation(self, target_client_ids, round_id, global_model_id):
        """Queue per-client protocol queries and wait briefly for gRPC selections."""
        target_clients = [cid for cid in target_client_ids if cid in self.active_clients]
        if not target_clients:
            return

        print(
            f"[gRPC] Negotiating downlink protocol for global model {global_model_id} "
            f"(round {round_id}, clients={len(target_clients)})"
        )

        for client_id in target_clients:
            self.client_protocol_queries[client_id] = {
                'round_id': int(round_id),
                'global_model_id': int(global_model_id),
            }

        deadline = time.time() + max(PROTOCOL_NEGOTIATION_TIMEOUT_SEC, 0.0)
        while time.time() < deadline:
            pending = [cid for cid in target_clients if cid in self.client_protocol_queries]
            if not pending:
                break
            time.sleep(max(PROTOCOL_NEGOTIATION_POLL_SEC, 0.01))

        pending_after_wait = [cid for cid in target_clients if cid in self.client_protocol_queries]
        for client_id in pending_after_wait:
            # Fallback remains robust gRPC when no fresh selection arrives.
            self.client_delivery_protocols[client_id] = 'grpc'
            self.client_protocol_queries.pop(client_id, None)

        if pending_after_wait:
            print(
                f"[gRPC] Protocol negotiation timeout for clients {pending_after_wait}; "
                f"falling back to gRPC downlink"
            )
        else:
            print("[gRPC] Protocol negotiation complete for all target clients")

    def _payload_fits_protocol(self, protocol, message):
        """Check whether a JSON message fits the transport's payload budget."""
        if protocol == 'mqtt':
            return len(json.dumps(message).encode('utf-8')) <= MQTT_MAX_PAYLOAD_BYTES
        if protocol == 'amqp':
            return len(json.dumps(message).encode('utf-8')) <= AMQP_MAX_FRAME_BYTES
        if protocol == 'http3':
            return len(json.dumps(message).encode('utf-8')) <= HTTP3_MAX_STREAM_DATA
        return True
    
    def broadcast_global_model(self):
        """Broadcast updated global model to all clients via their registered protocols"""
        # Prepare message with weights
        if self.quantization_handler is not None:
            # Prefer latest recompressed global payload from aggregation; else compress float global_weights.
            compressed_data = getattr(self, "global_compressed", None)
            if not (isinstance(compressed_data, dict) and 'compressed_data' in compressed_data):
                compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        for client_id in self.registered_clients.keys():
            if client_id not in self.active_clients:
                continue
            try:
                # Keep the gRPC pull path current for every client as a reliable fallback.
                self.grpc_model_ready[client_id] = self.current_round
                print(f"[gRPC] Global model ready for client {client_id} to pull (round {self.current_round})")

                protocol = self._get_delivery_protocol(client_id)
                if protocol == 'mqtt':
                    message = {
                        'round': self.current_round,
                        weights_key: weights_data,
                        'server_sent_unix': time.time(),
                    }
                    if self._payload_fits_protocol('mqtt', message):
                        self.send_via_mqtt(client_id, "fl/global_model", message)
                        print(f"[MQTT] Sent global model to client {client_id}")
                    else:
                        print(f"[MQTT] Global model for client {client_id} exceeds payload limit; client will pull via gRPC")
                    
                elif protocol == 'amqp':
                    message = {
                        'round': self.current_round,
                        weights_key: weights_data,
                        'server_sent_unix': time.time(),
                    }
                    if self._payload_fits_protocol('amqp', message):
                        self.send_via_amqp(client_id, 'global_model', message)
                        print(f"[AMQP] Sent global model to client {client_id}")
                    else:
                        print(f"[AMQP] Global model for client {client_id} exceeds payload limit; client will pull via gRPC")
                    
                elif protocol == 'quic':
                    # Send via QUIC stream (like single-protocol server does)
                    message = {
                        'type': 'global_model',
                        'round': self.current_round,
                        weights_key: weights_data,
                        'server_sent_unix': time.time(),
                    }
                    self.send_quic_message(client_id, message)
                    print(f"[QUIC] Sent global model to client {client_id}")
                    
                elif protocol == 'http3':
                    # Send via HTTP/3 stream
                    message = {
                        'type': 'global_model',
                        'round': self.current_round,
                        weights_key: weights_data,
                        'server_sent_unix': time.time(),
                    }
                    if self._payload_fits_protocol('http3', message):
                        self.send_http3_message(client_id, message)
                        print(f"[HTTP/3] Sent global model to client {client_id}")
                    else:
                        print(f"[HTTP/3] Global model for client {client_id} exceeds payload limit; client will pull via gRPC")
                    
                elif protocol == 'grpc':
                    continue
                    
                elif protocol == 'dds':
                    # FAIR CONFIG: DDS uses pub/sub with chunking (matching standalone)
                    if DDS_AVAILABLE and 'global_model_chunk' in self.dds_writers:
                        try:
                            # Serialize weights to bytes for chunking
                            weights_bytes = base64.b64decode(weights_data if weights_key == 'quantized_data' else weights_data)
                            weights_list = list(weights_bytes)  # Convert bytes to List[int]
                            
                            # Send using chunking
                            model_config_json = json.dumps(self.model_config) if self.model_config else ""
                            self.send_global_model_chunked(
                                round_num=self.current_round,
                                serialized_weights=weights_list,
                                model_config=model_config_json
                            )
                            print(f"[DDS] Published chunked global model to DDS topic for client {client_id}")
                        except Exception as dds_error:
                            print(f"[DDS] Error publishing global model: {dds_error}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f"[DDS] DDS chunk writer not available for client {client_id}")
                    
            except Exception as e:
                print(f"[{protocol.upper()}] Error broadcasting to client {client_id}: {e}")
                import traceback
                traceback.print_exc()
    
    def aggregate_models(self):
        """Aggregate client model updates using FedAvg"""
        if not self.client_updates:
            return
        print(f"\n{'='*70}")
        print(f"ROUND {self.current_round}/{self.num_rounds} - AGGREGATING MODELS")
        print(f"{'='*70}")
        
        # FedAvg: weighted average by number of samples
        total_samples = sum(update['num_samples'] for update in self.client_updates.values())
        
        # Quantization end-to-end: aggregate directly on compressed quantized tensors.
        if (
            self.quantization_handler is not None
            and len(self.client_updates) > 0
            and 'compressed_data' in list(self.client_updates.values())[0]
        ):
            compressed_updates = {
                cid: {"compressed_data": upd["compressed_data"], "num_samples": upd.get("num_samples", 1)}
                for cid, upd in self.client_updates.items()
            }
            aggregated_compressed, _stats = self.quantization_handler.aggregate_compressed_updates(compressed_updates)
            self.global_compressed = aggregated_compressed
            lw = getattr(self.quantization_handler, "last_aggregated_float_weights", None)
            if lw is not None:
                self.global_weights = lw

            self.client_updates.clear()
            self.prepare_downlink_protocol_negotiation(
                target_client_ids=list(self.active_clients),
                round_id=self.current_round,
                global_model_id=self.current_round,
            )
            self.broadcast_global_model()
            time.sleep(1)
            self.signal_start_evaluation()
            return

        aggregated_weights = []
        
        for i in range(len(self.global_weights)):
            weighted_sum = None
            for client_id, update in self.client_updates.items():
                weight = update['weights'][i]
                weight_factor = update['num_samples'] / total_samples
                
                if weighted_sum is None:
                    weighted_sum = weight * weight_factor
                else:
                    weighted_sum += weight * weight_factor
            
            aggregated_weights.append(weighted_sum)
        
        self.global_weights = aggregated_weights
        
        # Clear updates
        self.client_updates.clear()

        # Query each active client for its preferred downlink protocol before sending
        # the next global model snapshot for this round.
        self.prepare_downlink_protocol_negotiation(
            target_client_ids=list(self.active_clients),
            round_id=self.current_round,
            global_model_id=self.current_round,
        )
        
        # Broadcast new global model
        self.broadcast_global_model()
        
        # Signal evaluation
        time.sleep(1)
        self.signal_start_evaluation()
    
    def aggregate_metrics(self):
        """Aggregate client evaluation metrics"""
        if not self.client_metrics:
            return
        if self.round_start_time is not None:
            self.ROUND_TIMES.append(time.time() - self.round_start_time)
        socs = [m.get('battery_soc', 1.0) for m in self.client_metrics.values()]
        avg_soc = sum(socs) / len(socs) if socs else 1.0
        self.BATTERY_CONSUMPTION.append(1.0 - avg_soc)
        self.BATTERY_MODEL_CONSUMPTION.append(avg_battery_model_drain_fraction(self.client_metrics))
        # Check if we still have active clients before aggregating
        if len(self.active_clients) == 0:
            print("[Server] No active clients remaining. Stopping training.")
            self.converged = True
            self.save_results()
            self.signal_training_complete()
            return
        
        total_samples = sum(m['num_samples'] for m in self.client_metrics.values())
        
        # Weighted average
        avg_loss = sum(m['loss'] * m['num_samples'] for m in self.client_metrics.values()) / total_samples
        avg_accuracy = sum(m['accuracy'] * m['num_samples'] for m in self.client_metrics.values()) / total_samples
        avg_training_time = (
            sum(
                m.get('training_time_sec', 0.0) * m['num_samples']
                for m in self.client_metrics.values()
            )
            / total_samples
            if total_samples
            else 0.0
        )
        self.AVG_TRAINING_TIME_SEC.append(float(avg_training_time))
        self.AVG_BATTERY_SOC.append(float(avg_soc))
        
        self.ACCURACY.append(avg_accuracy)
        self.LOSS.append(avg_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"\nRound {self.current_round} Results:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        print(f"  Avg training time (sample-weighted): {avg_training_time:.3f} s")
        print(f"  Avg battery SoC: {avg_soc:.4f}")
        print(f"  Active clients: {len(self.active_clients)}/{self.num_clients}")
        
        # Clear metrics
        self.client_metrics.clear()
        self.round_start_time = time.time()
        # Continue to next round ONLY if we still have active clients
        self.current_round += 1
        
        if len(self.active_clients) == 0:
            print("[Server] All clients converged. Stopping training.")
            self.converged = True
            self.save_results()
            self.signal_training_complete()
        elif self.current_round <= self.num_rounds:
            time.sleep(1)
            self.signal_start_training()
        else:
            print(f"\n✅ COMPLETED {self.num_rounds} ROUNDS")
            self.save_results()
            self.signal_training_complete()
    
    def signal_training_complete(self):
        """Signal all clients that training is complete"""
        message = {'status': 'complete'}
        
        for client_id, protocol in self.registered_clients.items():
            if client_id not in self.active_clients:
                continue
            try:
                if protocol == 'mqtt':
                    self.mqtt_client.publish("fl/training_complete", json.dumps(message), qos=1)
                elif protocol == 'amqp':
                    self.send_via_amqp(client_id, 'training_complete', message)
                # Other protocols handled separately
            except Exception as e:
                print(f"[{protocol.upper()}] Error signaling completion to client {client_id}: {e}")
        
        time.sleep(2)
        print("\n[Server] Training complete. Shutting down...")
    
    def save_results(self):
        """Save experiment results"""
        results_dir = get_experiment_results_dir("emotion", "unified")
        
        results = {
            'rounds': self.ROUNDS,
            'accuracy': self.ACCURACY,
            'loss': self.LOSS,
            'round_times_seconds': getattr(self, 'ROUND_TIMES', []),
            'battery_consumption': getattr(self, 'BATTERY_CONSUMPTION', []),
            'battery_model_consumption': getattr(self, 'BATTERY_MODEL_CONSUMPTION', []),
            'battery_model_consumption_source': 'client_battery_model',
            'avg_training_time_sec': getattr(self, 'AVG_TRAINING_TIME_SEC', []),
            'avg_battery_soc': getattr(self, 'AVG_BATTERY_SOC', []),
            'converged': self.converged,
            'total_time': time.time() - self.start_time,
            'convergence_time': self.convergence_time,
            'num_clients': self.num_clients,
            'protocols_used': dict(self.client_delivery_protocols or self.registered_clients),
            'protocol_distribution': {
                protocol: sum(1 for p in (self.client_delivery_protocols or self.registered_clients).values() if p == protocol)
                for protocol in set((self.client_delivery_protocols or self.registered_clients).values())
            }
        }
        n_done = len(self.ROUNDS)
        results['rounds_completed'] = n_done
        results['total_rounds'] = n_done
        results['final_loss'] = self.LOSS[-1] if self.LOSS else None
        results['final_accuracy'] = self.ACCURACY[-1] if self.ACCURACY else None
        ct = self.convergence_time
        if ct is not None:
            results['convergence_time_seconds'] = float(ct)
            results['convergence_time_minutes'] = float(ct) / 60.0
        else:
            results['convergence_time_seconds'] = None
            results['convergence_time_minutes'] = None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"unified_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        stable_runner = results_dir / 'rl_unified_training_results.json'
        with open(stable_runner, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to {results_file}")
        print(f"✅ Experiment runner snapshot: {stable_runner}")
        self.plot_results()

    def plot_results(self):
        """Plot battery consumption, round/convergence time, and loss/accuracy."""
        results_dir = get_experiment_results_dir("emotion", "unified")
        rounds = self.ROUNDS
        n = len(rounds)
        if n == 0:
            return
        conv_time = self.convergence_time if self.convergence_time is not None else (time.time() - self.start_time if self.start_time else 0)
        # 1) Battery consumption
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        bc = (self.BATTERY_CONSUMPTION + [0.0] * max(0, n - len(self.BATTERY_CONSUMPTION)))[:n] if self.BATTERY_CONSUMPTION else [0.0] * n
        if bc:
            ax1.plot(rounds, [c * 100 for c in bc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Battery consumption (%)', fontsize=12)
        ax1.set_title('Unified: Battery consumption till end of FL training', fontsize=14)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(results_dir / 'unified_battery_consumption.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Battery plot saved to {results_dir / 'unified_battery_consumption.png'}")
        # 2) Time per round and convergence time
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        rt = (self.ROUND_TIMES + [0.0] * max(0, n - len(self.ROUND_TIMES)))[:n] if self.ROUND_TIMES else [0.0] * n
        if rt:
            ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2, label=f'Total convergence time: {conv_time:.1f} s')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Time (s)', fontsize=12)
        ax2.set_title('Unified: Time per round and total convergence time', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(results_dir / 'unified_round_and_convergence_time.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Time plot saved to {results_dir / 'unified_round_and_convergence_time.png'}")
        # 3) Loss and Accuracy
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        ax3a.plot(rounds, self.LOSS, marker='o', linewidth=2, markersize=8, color='red')
        ax3a.set_xlabel('Round', fontsize=12)
        ax3a.set_ylabel('Loss', fontsize=12)
        ax3a.set_title('Unified: Loss over FL Rounds', fontsize=14)
        ax3a.grid(True, alpha=0.3)
        ax3b.plot(rounds, [a * 100 for a in self.ACCURACY], marker='s', linewidth=2, markersize=8, color='green')
        ax3b.set_xlabel('Round', fontsize=12)
        ax3b.set_ylabel('Accuracy (%)', fontsize=12)
        ax3b.set_title('Unified: Accuracy over FL Rounds', fontsize=14)
        ax3b.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(results_dir / 'unified_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Results plot saved to {results_dir / 'unified_training_metrics.png'}")

    def run(self):
        """Run unified server (all protocols)"""
        print("[Server] Starting all protocol handlers...\n")
        
        self.start_mqtt_server()
        self.start_amqp_server()
        
        # Start AMQP consumer thread (polling-based)
        amqp_consumer_thread = threading.Thread(target=self.start_amqp_consumer, daemon=True)
        amqp_consumer_thread.start()
        
        self.start_grpc_server()
        self.start_quic_server()
        self.start_http3_server()
        self.start_dds_server()
        
        print("\n[Server] All protocol handlers started")
        print("[Server] Waiting for client registrations...\n")
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
                if self.converged or self.current_round > self.num_rounds:
                    break
        except KeyboardInterrupt:
            print("\n[Server] Interrupted by user")
        
        print("[Server] Shutting down...")
        
        # Cleanup
        if self.mqtt_client:
            self.mqtt_client.disconnect()
        if self.amqp_connection:
            self.amqp_connection.close()
        if self.amqp_send_connection:  # Close send connection too
            self.amqp_send_connection.close()
        if hasattr(self, 'amqp_consumer_connection') and self.amqp_consumer_connection:
            self.amqp_consumer_connection.close()
        if self.grpc_server:
            self.grpc_server.stop(0)


# =========================================================================
# gRPC Servicer Implementation
# =========================================================================

if GRPC_AVAILABLE:
    GRPC_CHUNK_SIZE = GRPC_MAX_MESSAGE_BYTES - 4096  # leave room for proto framing

    class FLServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
        """gRPC service implementation (chunked transfer when payload > 4 MB)."""
        
        def __init__(self, server):
            self.server = server
            self._update_chunks = {}  # (client_id, round) -> {'chunks': {index: bytes}, 'num_samples': int, 'metrics': dict}
        
        def RegisterClient(self, request, context):
            """Handle client registration"""
            try:
                log_received_packet(
                    packet_size=request.ByteSize(),
                    peer=f"client_{request.client_id}",
                    protocol="gRPC",
                    round=0,
                    extra_info="registration"
                )
                
                self.server.handle_client_registration(request.client_id, 'grpc')
                
                return federated_learning_pb2.RegistrationResponse(
                    success=True,
                    message="Registration successful"
                )
            except Exception as e:
                return federated_learning_pb2.RegistrationResponse(
                    success=False,
                    message=str(e)
                )
        
        def GetGlobalModel(self, request, context):
            """Send global model to client (chunked when > 4 MB)."""
            try:
                client_id = request.client_id
                ready_round = self.server.grpc_model_ready.get(client_id)
                chunk_index = getattr(request, 'chunk_index', 0) or 0

                # Resolve what to serve: serialized_weights, model_config_json, round_to_serve
                if self.server.global_weights is None:
                    return federated_learning_pb2.GlobalModel(
                        round=0, weights=b"", available=False, model_config="",
                        chunk_index=0, total_chunks=1, server_sent_unix=0.0,
                    )
                if self.server.quantization_handler is not None:
                    compressed_data = self.server.quantization_handler.compress_global_model(self.server.global_weights)
                    serialized_weights = pickle.dumps(compressed_data)
                else:
                    serialized_weights = pickle.dumps(self.server.global_weights)
                total_size = len(serialized_weights)
                model_config_json = json.dumps(self.server.model_config)

                if ready_round == 0 and request.round == 0:
                    round_to_serve = 0
                elif ready_round is not None and request.round == 0:
                    round_to_serve = ready_round
                elif request.round == 0 or request.round == self.server.current_round or request.round == self.server.current_round - 1:
                    round_to_serve = self.server.current_round
                    if request.round != 0 and round_to_serve != 0:
                        model_config_json = ""
                else:
                    return federated_learning_pb2.GlobalModel(
                        round=request.round, weights=b"", available=False, model_config="",
                        chunk_index=0, total_chunks=1, server_sent_unix=0.0,
                    )

                sent_key = (int(client_id), int(round_to_serve))

                # Single message if within chunk size
                if total_size <= GRPC_CHUNK_SIZE:
                    server_sent_unix = time.time()
                    self.server.grpc_downlink_sent_unix[sent_key] = server_sent_unix
                    log_sent_packet(
                        packet_size=total_size, peer=f"client_{client_id}", protocol="gRPC",
                        round=round_to_serve, extra_info="global_model"
                    )
                    return federated_learning_pb2.GlobalModel(
                        round=round_to_serve,
                        weights=serialized_weights,
                        available=True,
                        model_config=model_config_json,
                        chunk_index=0,
                        total_chunks=1,
                        server_sent_unix=server_sent_unix,
                    )

                # Chunked transfer
                chunks = [serialized_weights[i:i + GRPC_CHUNK_SIZE] for i in range(0, total_size, GRPC_CHUNK_SIZE)]
                total_chunks = len(chunks)
                if chunk_index < 0 or chunk_index >= total_chunks:
                    return federated_learning_pb2.GlobalModel(
                        round=round_to_serve, weights=b"", available=False, model_config="",
                        chunk_index=chunk_index, total_chunks=total_chunks, server_sent_unix=0.0,
                    )
                chunk_data = chunks[chunk_index]
                cfg = model_config_json if chunk_index == 0 else ""
                if chunk_index == 0:
                    server_sent_unix = time.time()
                    self.server.grpc_downlink_sent_unix[sent_key] = server_sent_unix
                    _log_key = (int(client_id), int(round_to_serve), int(total_chunks))
                    if _log_key not in self.server._grpc_chunk0_log_keys:
                        self.server._grpc_chunk0_log_keys.add(_log_key)
                        print(
                            f"[gRPC] Client {client_id}: Sending global model "
                            f"(round {round_to_serve}, {total_size/1024:.2f} KB in {total_chunks} chunks) "
                            f"(further chunk-0 polls are silent until a new round/model size)"
                        )
                else:
                    server_sent_unix = float(self.server.grpc_downlink_sent_unix.get(sent_key, 0.0))
                    if server_sent_unix <= 0.0:
                        server_sent_unix = time.time()
                log_sent_packet(
                    packet_size=len(chunk_data), peer=f"client_{client_id}", protocol="gRPC",
                    round=round_to_serve, extra_info="global_model_chunk"
                )
                return federated_learning_pb2.GlobalModel(
                    round=round_to_serve,
                    weights=chunk_data,
                    available=True,
                    model_config=cfg,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    server_sent_unix=server_sent_unix,
                )
            except Exception as e:
                print(f"[gRPC] Error in GetGlobalModel: {e}")
                return federated_learning_pb2.GlobalModel(
                    round=getattr(request, 'round', 0),
                    weights=b"", available=False, model_config="",
                    chunk_index=0, total_chunks=1, server_sent_unix=0.0,
                )
        
        def SendModelUpdate(self, request, context):
            """Receive model update from client (chunked when > 4 MB)."""
            try:
                client_id = request.client_id
                round_num = request.round
                metrics = dict(request.metrics)
                total_chunks = getattr(request, 'total_chunks', 1) or 1
                chunk_index = getattr(request, 'chunk_index', 0) or 0

                log_received_packet(
                    packet_size=request.ByteSize(),
                    peer=f"client_{client_id}",
                    protocol="gRPC",
                    round=round_num,
                    extra_info="model_update" if total_chunks == 1 else "model_update_chunk"
                )

                # Convergence: accept even if chunked stream incomplete
                converged_flag = stop_on_client_convergence() and float(metrics.get('client_converged', 0.0)) >= 1.0
                if converged_flag and round_num <= self.server.current_round and client_id in self.server.active_clients:
                    self.server.mark_client_converged(client_id)
                    return federated_learning_pb2.UpdateResponse(success=True, message=f"Client {client_id} convergence acknowledged")
                if round_num != self.server.current_round:
                    return federated_learning_pb2.UpdateResponse(success=False, message=f"Round mismatch: expected {self.server.current_round}, got {round_num}")

                # Chunked: accumulate until complete
                if total_chunks > 1:
                    key = (client_id, round_num)
                    if key not in self._update_chunks:
                        self._update_chunks[key] = {'chunks': {}, 'num_samples': 0, 'metrics': {}}
                    buf = self._update_chunks[key]
                    buf['chunks'][chunk_index] = request.weights if request.weights else b''
                    if chunk_index == 0:
                        buf['num_samples'] = request.num_samples
                        buf['metrics'] = metrics
                    if len(buf['chunks']) < total_chunks:
                        return federated_learning_pb2.UpdateResponse(
                            success=True,
                            message=f"Chunk {chunk_index + 1}/{total_chunks} received for round {round_num}"
                        )
                    serialized_weights = b''.join(buf['chunks'][i] for i in range(total_chunks))
                    num_samples = buf['num_samples']
                    metrics = buf['metrics']
                    del self._update_chunks[key]
                else:
                    serialized_weights = request.weights
                    num_samples = request.num_samples

                # Build data for handle_client_update (chunked reassembly is raw bytes; single message may be JSON)
                if isinstance(serialized_weights, bytes):
                    try:
                        payload_str = serialized_weights.decode('utf-8')
                    except UnicodeDecodeError:
                        payload_str = base64.b64encode(serialized_weights).decode('utf-8')
                        data = {'client_id': client_id, 'round': round_num, 'weights': payload_str, 'num_samples': num_samples, 'metrics': metrics}
                        self.server.handle_client_update(data, 'grpc')
                        return federated_learning_pb2.UpdateResponse(success=True, message="Update received")
                else:
                    payload_str = serialized_weights
                try:
                    parsed = json.loads(payload_str)
                    if isinstance(parsed, dict) and ('compressed_data' in parsed or 'weights' in parsed):
                        data = {
                            'client_id': parsed.get('client_id', client_id),
                            'round': round_num,
                            'num_samples': parsed.get('num_samples', num_samples),
                            'metrics': parsed.get('metrics', metrics)
                        }
                        if 'compressed_data' in parsed:
                            data['compressed_data'] = parsed['compressed_data']
                        else:
                            data['weights'] = parsed['weights']
                    else:
                        data = None
                except (json.JSONDecodeError, TypeError):
                    data = None
                if data is None:
                    data = {
                        'client_id': client_id,
                        'round': round_num,
                        'weights': payload_str,
                        'num_samples': num_samples,
                        'metrics': metrics
                    }

                self.server.handle_client_update(data, 'grpc')
                return federated_learning_pb2.UpdateResponse(success=True, message="Update received")
            except Exception as e:
                print(f"[gRPC] Error in SendModelUpdate: {e}")
                import traceback
                traceback.print_exc()
                return federated_learning_pb2.UpdateResponse(success=False, message=str(e))
        
        def SendMetrics(self, request, context):
            """Receive evaluation metrics from client"""
            try:
                log_received_packet(
                    packet_size=request.ByteSize(),
                    peer=f"client_{request.client_id}",
                    protocol="gRPC",
                    round=request.round,
                    extra_info="metrics"
                )
                
                data = {
                    'client_id': request.client_id,
                    'round': request.round,
                    'num_samples': request.num_samples,
                    'loss': request.loss,
                    'accuracy': request.accuracy,
                    'battery_soc': getattr(request, 'battery_soc', 1.0),
                    'round_time_sec': getattr(request, 'round_time_sec', 0.0),
                    'training_time_sec': getattr(request, 'training_time_sec', 0.0),
                    'uplink_model_comm_sec': getattr(request, 'uplink_model_comm_sec', 0.0),
                }
                
                self.server.handle_client_metrics(data, 'grpc')
                
                return federated_learning_pb2.MetricsResponse(
                    success=True,
                    message="Metrics received"
                )
            except Exception as e:
                print(f"[gRPC] Error in SendMetrics: {e}")
                return federated_learning_pb2.MetricsResponse(
                    success=False,
                    message=str(e)
                )
        
        def CheckTrainingStatus(self, request, context):
            """Check if client should start training or evaluation"""
            try:
                client_id = request.client_id
                
                # Check if client should train or evaluate
                should_train = self.server.grpc_should_train.get(client_id, False)
                should_evaluate = self.server.grpc_should_evaluate.get(client_id, False)
                
                # Don't clear flags immediately - let them persist for multiple polls
                # This prevents race conditions where signals are lost if client has guards
                # Flags will be cleared when next round's signals are set
                pending_query = self.server.client_protocol_queries.get(client_id)
                has_protocol_query = pending_query is not None
                protocol_query = None
                if has_protocol_query:
                    protocol_query = federated_learning_pb2.ProtocolQuery(
                        client_id=client_id,
                        round_id=int(pending_query.get('round_id', self.server.current_round)),
                        global_model_id=int(pending_query.get('global_model_id', self.server.current_round)),
                    )
                
                status_kwargs = {
                    'should_train': should_train,
                    'should_evaluate': should_evaluate,
                    'current_round': self.server.current_round,
                    'is_complete': self.server.converged,
                    'has_protocol_query': has_protocol_query,
                }
                if protocol_query is not None:
                    status_kwargs['protocol_query'] = protocol_query

                return federated_learning_pb2.TrainingStatus(
                    **status_kwargs
                )
            except Exception as e:
                print(f"[gRPC] Error in CheckTrainingStatus: {e}")
                return federated_learning_pb2.TrainingStatus(
                    should_train=False,
                    should_evaluate=False,
                    current_round=0,
                    is_complete=True,
                    has_protocol_query=False,
                )

        def SendProtocolSelection(self, request, context):
            """Receive client's negotiated downlink protocol preference over gRPC control plane."""
            try:
                client_id = request.client_id
                selected_protocol = self.server._normalize_protocol_name(request.downlink_protocol_requested)
                if selected_protocol is None:
                    return federated_learning_pb2.ProtocolSelectionResponse(
                        success=False,
                        message=f"Unsupported downlink protocol: {request.downlink_protocol_requested}",
                    )

                pending_query = self.server.client_protocol_queries.get(client_id)
                if pending_query is None:
                    # Accept as best-effort preference refresh even if query already timed out/cleared.
                    self.server.client_delivery_protocols[client_id] = selected_protocol
                    return federated_learning_pb2.ProtocolSelectionResponse(
                        success=True,
                        message="Selection accepted (no pending query)",
                    )

                expected_round = int(pending_query.get('round_id', self.server.current_round))
                expected_model_id = int(pending_query.get('global_model_id', self.server.current_round))
                if request.round_id != expected_round or request.global_model_id != expected_model_id:
                    return federated_learning_pb2.ProtocolSelectionResponse(
                        success=False,
                        message=(
                            f"Selection mismatch: expected round={expected_round}, "
                            f"global_model_id={expected_model_id}"
                        ),
                    )

                self.server.client_delivery_protocols[client_id] = selected_protocol
                self.server.client_protocol_queries.pop(client_id, None)
                print(
                    f"[gRPC] Client {client_id} selected downlink protocol '{selected_protocol}' "
                    f"for round {request.round_id}, global model {request.global_model_id}"
                )
                return federated_learning_pb2.ProtocolSelectionResponse(
                    success=True,
                    message="Protocol selection recorded",
                )
            except Exception as e:
                print(f"[gRPC] Error in SendProtocolSelection: {e}")
                return federated_learning_pb2.ProtocolSelectionResponse(
                    success=False,
                    message=str(e),
                )


# =========================================================================
# HTTP/3 Protocol Handler
# =========================================================================

if HTTP3_AVAILABLE:
    class HTTP3ServerProtocol(QuicConnectionProtocol):
        """HTTP/3 protocol handler (supports multiple clients)"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server = None  # Set by factory function
            self._http = None  # H3Connection instance
            self._stream_buffers = {}  # Buffer data per stream ID (instance-specific)
            self._stream_content_lengths = {}
        
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
                    method = headers.get(b":method", b"").decode()
                    
                    # Initialize buffer for this stream
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                    
                    # Handle POST requests (client sending data)
                    if method == "POST":
                        content_length = int(headers.get(b"content-length", b"0"))
                        self._stream_content_lengths[stream_id] = content_length
                    
                except Exception as e:
                    print(f"[HTTP/3] Error handling headers: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif isinstance(event, DataReceived):
                try:
                    stream_id = event.stream_id
                    # Get or create buffer for this stream
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                    
                    # Append new data to buffer
                    self._stream_buffers[stream_id] += event.data
                    
                    # Send flow control updates to allow more data
                    self.transmit()
                    
                    expected_length = self._stream_content_lengths.get(stream_id, 0)
                    received_length = len(self._stream_buffers[stream_id])
                    
                    if expected_length > 0 and received_length >= expected_length:
                        try:
                            data_str = self._stream_buffers[stream_id].decode('utf-8')
                            message = json.loads(data_str)
                            client_id = message.get('client_id', 'unknown')
                            msg_type = message.get('type', 'unknown')
                            
                            log_received_packet(
                                packet_size=len(data_str),
                                peer=f"http3_client_{client_id}",
                                protocol="HTTP/3",
                                round=message.get('round', 0),
                                extra_info=msg_type
                            )
                            
                            # Send HTTP/3 response
                            response_body = json.dumps({"status": "ok"}).encode('utf-8')
                            response_headers = [
                                (b":status", b"200"),
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(response_body)).encode()),
                            ]
                            self._http.send_headers(stream_id=stream_id, headers=response_headers)
                            self._http.send_data(stream_id=stream_id, data=response_body, end_stream=True)
                            self.transmit()
                            
                            # Handle message asynchronously
                            if self.server:
                                asyncio.create_task(self.server.handle_http3_message(message, self))
                            
                            # Clear buffer
                            self._stream_buffers[stream_id] = b''
                            self._stream_content_lengths.pop(stream_id, None)
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"[HTTP/3] Error decoding message: {e}")
                            # Send error response
                            try:
                                error_headers = [
                                    (b":status", b"400"),
                                    (b"content-type", b"text/plain"),
                                ]
                                error_body = f"Error: {str(e)}".encode('utf-8')
                                error_headers.append((b"content-length", str(len(error_body)).encode()))
                                self._http.send_headers(stream_id=stream_id, headers=error_headers)
                                self._http.send_data(stream_id=stream_id, data=error_body, end_stream=True)
                                self.transmit()
                            except:
                                pass
                            self._stream_buffers[stream_id] = b''
                            self._stream_content_lengths.pop(stream_id, None)
                except Exception as e:
                    print(f"[HTTP/3] Error handling data: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif isinstance(event, StreamReset):
                # Stream was reset, clear buffer
                stream_id = event.stream_id
                if stream_id in self._stream_buffers:
                    del self._stream_buffers[stream_id]
                    print(f"[HTTP/3] Stream {stream_id} reset, cleared buffer")
                self._stream_content_lengths.pop(stream_id, None)

# =========================================================================
# QUIC Protocol Handler
# =========================================================================

if QUIC_AVAILABLE:
    def _parse_quic_message(message_data: bytes):
        """Parse JSON message in thread (avoids blocking event loop on large payloads)."""
        data_str = message_data.decode('utf-8')
        return json.loads(data_str)

    QUIC_SOCKET_BUFFER_BYTES = 7_500_000  # 7.5MB for SO_RCVBUF/SO_SNDBUF (poor network)

    class QUICServerProtocol(QuicConnectionProtocol):
        """QUIC protocol handler (supports multiple clients)"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server = None  # Set by factory function
            self._stream_buffers = {}  # Buffer data per stream ID (instance-specific)
            print(f"[QUIC] Protocol instance created")

        def connection_made(self, transport):
            super().connection_made(transport)
            sock = transport.get_extra_info("socket")
            if sock:
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, QUIC_SOCKET_BUFFER_BYTES)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, QUIC_SOCKET_BUFFER_BYTES)
                    print(f"[QUIC] UDP socket buffers set to {QUIC_SOCKET_BUFFER_BYTES // 1_000_000}MB")
                except OSError as e:
                    print(f"[QUIC] Could not set socket buffers: {e}")
        
        async def _process_parsed_message(self, message_data: bytes, stream_id: int):
            """Parse message in executor (non-blocking) and dispatch. Prevents event loop
            blocking from large JSON parsing, which was causing the second client's QUIC
            update to never be received when both clients sent updates simultaneously."""
            try:
                loop = asyncio.get_event_loop()
                message = await loop.run_in_executor(None, _parse_quic_message, message_data)
                print(f"[QUIC] Decoded message type '{message.get('type')}' from stream {stream_id}")
                log_received_packet(
                    packet_size=len(message_data),
                    peer=f"quic_client_{message.get('client_id', 'unknown')}",
                    protocol="QUIC",
                    round=message.get('round', 0),
                    extra_info=message.get('type', 'unknown')
                )
                if self.server:
                    await self.server.handle_quic_message(message, self)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[QUIC] Error decoding message: {e}")
        
        def quic_event_received(self, event):
            """Handle QUIC events"""
            if isinstance(event, StreamDataReceived):
                try:
                    # Get or create buffer for this stream
                    stream_id = event.stream_id
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                    
                    # Append new data to buffer
                    self._stream_buffers[stream_id] += event.data
                    
                    # Send flow control updates to allow more data
                    self.transmit()
                    
                    # Try to decode complete messages (delimited by newline)
                    # NOTE: Schedule parsing in executor - json.loads on ~12MB blocks the
                    # event loop and prevents processing the second client's packets.
                    while b'\n' in self._stream_buffers[stream_id]:
                        message_data, self._stream_buffers[stream_id] = self._stream_buffers[stream_id].split(b'\n', 1)
                        if message_data:
                            asyncio.create_task(self._process_parsed_message(message_data, stream_id))
                    
                    # If stream ended and buffer has remaining data, try to process it
                    if event.end_stream and self._stream_buffers[stream_id]:
                        print(f"[QUIC] Stream {stream_id} ended with {len(self._stream_buffers[stream_id])} bytes remaining")
                        message_data = self._stream_buffers[stream_id]
                        self._stream_buffers[stream_id] = b''
                        asyncio.create_task(self._process_parsed_message(message_data, stream_id))
                except Exception as e:
                    print(f"[QUIC] Error handling event: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Call parent handler for other events
                super().quic_event_received(event)


def main():
    """Main function"""
    # Prevent duplicate unified server instance in the same container/host.
    # Use user-writable path (home or cwd) to avoid PermissionError on /tmp.
    lock_path = os.path.join(os.path.expanduser("~"), ".unified_emotion_server.lock")
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("[Startup] Another unified server instance is already running. Exiting.")
        return

    # Use MIN_CLIENTS (dynamic clients) and configured NUM_ROUNDS.
    # MAX_CLIENTS controls the upper bound of concurrently registered clients.
    server = UnifiedFederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, max_clients=MAX_CLIENTS)
    server.run()


if __name__ == "__main__":
    main()
