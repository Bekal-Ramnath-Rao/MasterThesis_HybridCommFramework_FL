"""
Unified Federated Learning Server for Emotion Recognition
Handles all 5 protocols simultaneously: MQTT, AMQP, gRPC, QUIC, DDS

The server listens on all protocol channels and responds to clients
using whichever protocol they selected via RL.
"""

import os
import sys

# Configure GPU before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import threading
import asyncio
from typing import List, Dict, Sequence, TYPE_CHECKING
from pathlib import Path
from concurrent import futures

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

# Detect Docker environment and set project root accordingly
if os.path.exists('/app'):
    project_root = '/app'
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from packet_logger import init_db, log_sent_packet, log_received_packet

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
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
# FAIR CONFIG: 64 KB chunks for better DDS performance in poor networks
CHUNK_SIZE = 64 * 1024  # 64KB chunks

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
        self.registered_clients = {}  # Maps client_id -> protocol_used
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        self.model_config = None
        
        # Server state
        self.running = True
        
        # Metrics storage
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.start_time = None
        self.convergence_time = None
        
        # DDS chunk reassembly buffers (FAIR CONFIG: matching standalone)
        self.model_update_chunks = {}  # {client_id: {chunk_id: payload}}
        self.model_update_metadata = {}  # {client_id: {total_chunks, num_samples, loss, mse, mae, mape}}
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
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
        self.quic_clients = {}  # Maps client_id -> QuicConnectionProtocol for sending responses
        self.dds_participant = None
        self.dds_writers = {}
        self.dds_readers = {}
        
        # gRPC state tracking
        self.grpc_should_train = {}  # Maps client_id -> should_train (bool)
        self.grpc_should_evaluate = {}  # Maps client_id -> should_evaluate (bool)
        self.grpc_model_ready = {}  # Maps client_id -> model_ready_for_round (int)
        
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
        
        print(f"Sending global model in {total_chunks} chunks ({len(serialized_weights)} bytes total)")
        
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = GlobalModelChunk(
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                model_config_json=model_config if chunk_id == 0 else ""  # Only send config with first chunk
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
            # FAIR CONFIG: Set max packet size to 128MB (aligned with AMQP default)
            self.mqtt_client._max_packet_size = 128 * 1024 * 1024  # 128 MB
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
                            blocked_connection_timeout=600  # Aligned with heartbeat
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
        """Start AMQP protocol handler"""
        if not AMQP_AVAILABLE:
            return
        
        try:
            credentials = pika.PlainCredentials('guest', 'guest')
            # FAIR CONFIG: heartbeat=600s for very_poor network scenarios
            parameters = pika.ConnectionParameters(
                host=AMQP_BROKER,
                port=AMQP_PORT,
                credentials=credentials,
                heartbeat=600,  # 10 minutes for very_poor network
                blocked_connection_timeout=600,  # Aligned with heartbeat
                connection_attempts=5,
                retry_delay=2
            )
            
            # Connection 1: Consumer (owned by consumer thread)
            self.amqp_connection = pika.BlockingConnection(parameters)
            self.amqp_channel = self.amqp_connection.channel()
            
            # Connection 2: Sender (owned by main thread - thread-safe!)
            self.amqp_send_connection = pika.BlockingConnection(parameters)
            self.amqp_send_channel = self.amqp_send_connection.channel()
            
            # Declare exchanges and queues
            self.amqp_channel.exchange_declare(
                exchange='fl_client_updates',
                exchange_type='direct',
                durable=True
            )
            
            # Queue for client registrations
            self.amqp_channel.queue_declare(queue='fl_client_register', durable=True)
            
            # Set up registration consumer
            self.amqp_channel.basic_consume(
                queue='fl_client_register',
                on_message_callback=self.on_amqp_register,
                auto_ack=True
            )
            
            # Start consuming in separate thread
            def consume():
                try:
                    self.amqp_channel.start_consuming()
                except Exception as e:
                    print(f"[AMQP] Consumer error: {e}")
            
            amqp_thread = threading.Thread(target=consume, daemon=True)
            amqp_thread.start()
            
            print("[AMQP] Server started with separate send/receive connections")
        except Exception as e:
            print(f"[AMQP] Failed to start (will retry on client registration): {e}")
    
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
            elif msg_type == 'update':
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
        """Send message to client via QUIC stream"""
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
                        blocked_connection_timeout=600  # Aligned with heartbeat
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
                    # FAIR CONFIG: Message size limits 128MB (aligned with AMQP default)
                    ('grpc.max_send_message_length', 128 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 128 * 1024 * 1024),
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
            # FAIR CONFIG: Aligned with MQTT/AMQP/gRPC/DDS for unbiased comparison
            configuration = QuicConfiguration(
                is_client=False,
                alpn_protocols=["fl"],
                # FAIR CONFIG: Data limits 128MB per stream, 256MB total (aligned with AMQP)
                max_stream_data=128 * 1024 * 1024,  # 128 MB per stream
                max_data=256 * 1024 * 1024,  # 256 MB total connection
                # FAIR CONFIG: Timeout 600s for very_poor network scenarios
                idle_timeout=600.0,  # 10 minutes
                max_datagram_frame_size=65536,  # 64 KB frames
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
            
            # Save to temp files
            cert_path = "/tmp/quic_cert.pem"
            key_path = "/tmp/quic_key.pem"
            
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
            print(f"[QUIC]   - Max stream data: 50 MB")
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
    # DDS PROTOCOL HANDLERS
    # =========================================================================
    
    def start_amqp_consumer(self):
        """Polling-based AMQP consumer for all clients"""
        try:
            credentials = pika.PlainCredentials('guest', 'guest')
            # FAIR CONFIG: heartbeat=600s for very_poor network scenarios
            parameters = pika.ConnectionParameters(
                host=AMQP_BROKER,
                port=AMQP_PORT,
                credentials=credentials,
                heartbeat=600,  # 10 minutes for very_poor network
                blocked_connection_timeout=600,  # Aligned with heartbeat
                connection_attempts=5,
                retry_delay=2
            )
            self.amqp_consumer_connection = pika.BlockingConnection(parameters)
            self.amqp_consumer_channel = self.amqp_consumer_connection.channel()
            
            print("[AMQP] Consumer connection established")
            
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
            print(f"[AMQP] Consumer connection failed: {e}")
            import traceback
            traceback.print_exc()
    
    def start_dds_server(self):
        """Start DDS protocol handler"""
        if not DDS_AVAILABLE:
            return
        
        try:
            # Create DDS participant
            self.dds_participant = DomainParticipant(DDS_DOMAIN_ID)
            
            # FAIR CONFIG: Control QoS for registration/commands (60s for responsiveness)
            reliable_qos = Qos(
                Policy.Reliability.Reliable(max_blocking_time=duration(seconds=60)),
                Policy.History.KeepLast(10),
                Policy.Durability.TransientLocal
            )
            
            # FAIR CONFIG: Chunk QoS for data (600s timeout, 2048 chunks = 128 MB buffer)
            chunk_qos = Qos(
                Policy.Reliability.Reliable(max_blocking_time=duration(seconds=600)),  # 10 min for very_poor network
                Policy.History.KeepLast(2048),  # 2048 × 64KB = 128 MB buffer (aligned with AMQP)
                Policy.Durability.Volatile
            )
            
            # Best effort QoS for legacy non-chunked messages
            best_effort_qos = Qos(
                Policy.Reliability.BestEffort,
                Policy.History.KeepLast(1),
            )
            
            # Create topics and readers for model updates
            update_topic = Topic(self.dds_participant, "ModelUpdate", ModelUpdate)
            update_reader = DataReader(self.dds_participant, update_topic, qos=best_effort_qos)
            update_chunk_topic = Topic(self.dds_participant, "ModelUpdateChunk", ModelUpdateChunk)
            update_chunk_reader = DataReader(self.dds_participant, update_chunk_topic, qos=chunk_qos)
            
            # Create topics and readers for metrics
            metrics_topic = Topic(self.dds_participant, "EvaluationMetrics", EvaluationMetrics)
            metrics_reader = DataReader(self.dds_participant, metrics_topic, qos=reliable_qos)
            
            # Create writers for sending global model and commands to clients
            global_model_topic = Topic(self.dds_participant, "GlobalModel", GlobalModel)
            self.dds_writers['global_model'] = DataWriter(self.dds_participant, global_model_topic, qos=best_effort_qos)
            global_model_chunk_topic = Topic(self.dds_participant, "GlobalModelChunk", GlobalModelChunk)
            self.dds_writers['global_model_chunk'] = DataWriter(self.dds_participant, global_model_chunk_topic, qos=chunk_qos)
            
            command_topic = Topic(self.dds_participant, "TrainingCommand", TrainingCommand)
            self.dds_writers['command'] = DataWriter(self.dds_participant, command_topic, qos=reliable_qos)
            
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
                                
                                # Initialize chunk buffers if needed
                                if client_id not in self.model_update_chunks:
                                    self.model_update_chunks[client_id] = {}
                                    self.model_update_metadata[client_id] = {
                                        'total_chunks': total_chunks,
                                        'num_samples': sample.num_samples,
                                        'loss': sample.loss,
                                        'mse': sample.mse,
                                        'mae': sample.mae,
                                        'mape': sample.mape
                                    }
                                
                                # Store chunk
                                self.model_update_chunks[client_id][chunk_id] = sample.payload
                                
                                # Progress update every 20 chunks to reduce console spam
                                if (chunk_id + 1) % 20 == 0 or (chunk_id + 1) == total_chunks:
                                    print(f"Received {chunk_id + 1}/{total_chunks} chunks from client {client_id}")
                                
                                # Check if all chunks received for this client
                                if len(self.model_update_chunks[client_id]) == total_chunks:
                                    print(f"All chunks received from client {client_id}, reassembling...")
                                    
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
                                        # Deserialize client weights
                                        weights = pickle.loads(bytes(reassembled_data))
                                        
                                        metadata = self.model_update_metadata[client_id]
                                        data = {
                                            'client_id': client_id,
                                            'round': sample.round,
                                            'weights': weights,
                                            'num_samples': metadata['num_samples'],
                                            'loss': metadata['loss'],
                                            'mse': metadata['mse'],
                                            'mae': metadata['mae'],
                                            'mape': metadata['mape']
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
                                        'mse': sample.mse,
                                        'mae': sample.mae,
                                        'mape': sample.mape
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
                                        'mse': sample.mse,
                                        'mae': sample.mae,
                                        'mape': sample.mape
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
            
            print(f"[DDS] Server started on domain {DDS_DOMAIN_ID}")

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
            self.registered_clients[client_id] = protocol
            print(f"[{protocol.upper()}] Client {client_id} registered "
                  f"({len(self.registered_clients)}/{self.num_clients})")
            
            if len(self.registered_clients) >= self.min_clients:
                print(f"\n[Server] All {self.num_clients} clients registered!")
                print("[Server] Distributing initial global model...\n")
                time.sleep(2)
                self.distribute_initial_model()
                self.start_time = time.time()
                print(f"[Server] Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def handle_client_update(self, data, protocol):
        """Handle client model update (thread-safe)"""
        print(f"[DEBUG] handle_client_update ENTRY - protocol={protocol}, client_id={data.get('client_id')}, round={data.get('round')}, current_round={self.current_round}")
        with self.lock:
            client_id = data['client_id']
            round_num = data['round']
            
            print(f"[DEBUG] handle_client_update LOCKED - checking round: {round_num} vs {self.current_round}")
            
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
            
            # Update the protocol this client is using (RL agent may change protocol per round)
            self.registered_clients[client_id] = protocol
            
            # Deserialize weights
            if 'compressed_data' in data and self.quantization_handler is not None:
                compressed_update = data['compressed_data']
                if isinstance(compressed_update, str):
                    try:
                        compressed_update = pickle.loads(base64.b64decode(compressed_update.encode('utf-8')))
                    except Exception as e:
                        print(f"[Server] Error decoding compressed_data: {e}")
                weights = self.quantization_handler.decompress_client_update(client_id, compressed_update)
            else:
                weights = self.deserialize_weights(data['weights'])
            
            # Store update (metrics are handled separately via handle_client_metrics)
            self.client_updates[client_id] = {
                'weights': weights,
                'num_samples': data['num_samples'],
                'protocol': protocol
            }
            
            print(f"[{protocol.upper()}] Received update from client {client_id} "
                  f"({len(self.client_updates)}/{self.num_clients})")
            
            # Wait for all registered clients (dynamic)
            if len(self.client_updates) >= len(self.registered_clients):
                self.aggregate_models()
    
    def handle_client_metrics(self, data, protocol):
        """Handle client evaluation metrics (thread-safe)"""
        with self.lock:
            client_id = data['client_id']
            round_num = data['round']
            
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
            
            # Update the protocol this client is using (RL agent may change protocol per round)
            self.registered_clients[client_id] = protocol
            
            self.client_metrics[client_id] = {
                'round': round_num,
                'num_samples': data['num_samples'],
                'loss': data['loss'],
                'accuracy': data['accuracy'],
                'protocol': protocol
            }
            
            print(f"[{protocol.upper()}] Received metrics from client {client_id} "
                  f"({len(self.client_metrics)}/{self.num_clients})")
            
            # Wait for all registered clients (dynamic)
            if len(self.client_metrics) >= len(self.registered_clients):
                print(f"[{protocol.upper()}] All {self.num_clients} metrics received! Triggering aggregate_metrics()")
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
        
        for client_id, protocol in self.registered_clients.items():
            try:
                if protocol == 'mqtt':
                    message = {
                        'round': 0,
                        weights_key: weights_data,
                        'model_config': self.model_config
                    }
                    self.send_via_mqtt(client_id, "fl/global_model", message)
                    print(f"[MQTT] Sent initial model to client {client_id}")
                    
                elif protocol == 'amqp':
                    message = {
                        'round': 0,
                        weights_key: weights_data,
                        'model_config': self.model_config
                    }
                    self.send_via_amqp(client_id, 'global_model', message)
                    print(f"[AMQP] Sent initial model to client {client_id}")
                    
                elif protocol == 'quic':
                    # Send via QUIC stream
                    message = {
                        'type': 'global_model',
                        'round': 0,
                        weights_key: weights_data,
                        'model_config': self.model_config
                    }
                    self.send_quic_message(client_id, message)
                    print(f"[QUIC] Sent initial model to client {client_id}")
                    
                elif protocol == 'grpc':
                    # Mark initial model as ready for gRPC client to pull
                    self.grpc_model_ready[client_id] = 0
                    print(f"[gRPC] Initial model ready for client {client_id} to pull")
                    
                elif protocol == 'dds':
                    print(f"[DDS] Client {client_id} will receive initial model via pub/sub")
                    
            except Exception as e:
                print(f"[{protocol.upper()}] Error sending initial model to client {client_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Start first round
        time.sleep(2)
        self.current_round = 1
        self.signal_start_training()
    
    def signal_start_training(self):
        """Signal all clients to start training for current round"""
        # Broadcast to ALL protocols (clients listen to multiple protocols simultaneously)
        for client_id in self.registered_clients.keys():
            message = {'round': self.current_round}
            
            # MQTT
            try:
                self.send_via_mqtt(client_id, "fl/start_training", message)
            except Exception as e:
                pass  # Client may not be listening to this protocol
            
            # AMQP
            try:
                self.send_via_amqp(client_id, 'start_training', message)
            except Exception as e:
                pass
            
            # gRPC - set flag for polling
            try:
                self.grpc_should_train[client_id] = True
                self.grpc_should_evaluate[client_id] = False
            except Exception as e:
                pass
            
            # DDS
            try:
                if DDS_AVAILABLE and 'command' in self.dds_writers:
                    command = TrainingCommand(
                        round=self.current_round,
                        start_training=True,
                        start_evaluation=False,
                        training_complete=False
                    )
                    self.dds_writers['command'].write(command)
            except Exception as e:
                pass
    
    def signal_start_evaluation(self):
        """Signal all clients to start evaluation for current round"""
        # Broadcast to ALL protocols (clients listen to multiple protocols simultaneously)
        for client_id in self.registered_clients.keys():
            message = {'round': self.current_round}
            
            # MQTT
            try:
                self.send_via_mqtt(client_id, "fl/start_evaluation", message)
            except Exception as e:
                pass  # Client may not be listening to this protocol
            
            # AMQP
            try:
                self.send_via_amqp(client_id, 'start_evaluation', message)
            except Exception as e:
                pass
            
            # gRPC - set flag for polling
            try:
                self.grpc_should_train[client_id] = False
                self.grpc_should_evaluate[client_id] = True
            except Exception as e:
                pass
            
            # DDS
            try:
                if DDS_AVAILABLE and 'command' in self.dds_writers:
                    command = TrainingCommand(
                        round=self.current_round,
                        start_training=False,
                        start_evaluation=True,
                        training_complete=False
                    )
                    self.dds_writers['command'].write(command)
            except Exception as e:
                pass
    
    def broadcast_global_model(self):
        """Broadcast updated global model to all clients via their registered protocols"""
        # Prepare message with weights
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        for client_id, protocol in self.registered_clients.items():
            try:
                if protocol == 'mqtt':
                    message = {
                        'round': self.current_round,
                        weights_key: weights_data
                    }
                    self.send_via_mqtt(client_id, "fl/global_model", message)
                    print(f"[MQTT] Sent global model to client {client_id}")
                    
                elif protocol == 'amqp':
                    message = {
                        'round': self.current_round,
                        weights_key: weights_data
                    }
                    self.send_via_amqp(client_id, 'global_model', message)
                    print(f"[AMQP] Sent global model to client {client_id}")
                    
                elif protocol == 'quic':
                    # Send via QUIC stream (like single-protocol server does)
                    message = {
                        'type': 'global_model',
                        'round': self.current_round,
                        weights_key: weights_data
                    }
                    self.send_quic_message(client_id, message)
                    print(f"[QUIC] Sent global model to client {client_id}")
                    
                elif protocol == 'grpc':
                    # gRPC uses pull model - mark model as ready for this client
                    self.grpc_model_ready[client_id] = self.current_round
                    print(f"[gRPC] Global model ready for client {client_id} to pull (round {self.current_round})")
                    
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
        print(f"\n{'='*70}")
        print(f"ROUND {self.current_round}/{self.num_rounds} - AGGREGATING MODELS")
        print(f"{'='*70}")
        
        # FedAvg: weighted average by number of samples
        total_samples = sum(update['num_samples'] for update in self.client_updates.values())
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
        
        # Broadcast new global model
        self.broadcast_global_model()
        
        # Signal evaluation
        time.sleep(1)
        self.signal_start_evaluation()
    
    def aggregate_metrics(self):
        """Aggregate client evaluation metrics"""
        total_samples = sum(m['num_samples'] for m in self.client_metrics.values())
        
        # Weighted average
        avg_loss = sum(m['loss'] * m['num_samples'] for m in self.client_metrics.values()) / total_samples
        avg_accuracy = sum(m['accuracy'] * m['num_samples'] for m in self.client_metrics.values()) / total_samples
        
        self.ACCURACY.append(avg_accuracy)
        self.LOSS.append(avg_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"\nRound {self.current_round} Results:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Accuracy: {avg_accuracy:.4f}")
        
        # Check convergence
        if self.current_round >= MIN_ROUNDS:
            if self.best_loss - avg_loss > CONVERGENCE_THRESHOLD:
                self.best_loss = avg_loss
                self.rounds_without_improvement = 0
            else:
                self.rounds_without_improvement += 1
            
            if self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
                self.converged = True
                self.convergence_time = time.time() - self.start_time
                print(f"\n✅ CONVERGENCE ACHIEVED at round {self.current_round}")
                self.save_results()
                self.signal_training_complete()
                return
        else:
            self.best_loss = min(self.best_loss, avg_loss)
        
        # Clear metrics
        self.client_metrics.clear()
        
        # Continue to next round
        self.current_round += 1
        
        if self.current_round <= self.num_rounds:
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
        results_dir = Path("/app/results" if IN_DOCKER else "./results")
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'rounds': self.ROUNDS,
            'accuracy': self.ACCURACY,
            'loss': self.LOSS,
            'converged': self.converged,
            'total_time': time.time() - self.start_time,
            'convergence_time': self.convergence_time,
            'num_clients': self.num_clients,
            'protocols_used': dict(self.registered_clients),
            'protocol_distribution': {
                protocol: sum(1 for p in self.registered_clients.values() if p == protocol)
                for protocol in set(self.registered_clients.values())
            }
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"unified_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {results_file}")
    
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
        self.start_dds_server()
        
        print("\n[Server] All protocol handlers started")
        print("[Server] Waiting for client registrations...\n")
        
        # Keep main thread alive
        try:
            while not self.converged:
                time.sleep(1)
                if self.current_round > self.num_rounds:
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
    class FLServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
        """gRPC service implementation"""
        
        def __init__(self, server):
            self.server = server
        
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
            """Send global model to client"""
            try:
                # Round 0 means "give me the latest model" (for polling)
                if request.round == 0 or request.round == self.server.current_round or request.round == self.server.current_round - 1:
                    # Send current global model
                    if self.server.current_round == 0:
                        # Initial model with config
                        weights_bytes = pickle.dumps(self.server.global_weights)
                        model_config_json = json.dumps(self.server.model_config)
                        
                        response = federated_learning_pb2.GlobalModel(
                            round=0,
                            weights=weights_bytes,
                            available=True,
                            model_config=model_config_json
                        )
                    else:
                        # Updated model after aggregation
                        weights_bytes = pickle.dumps(self.server.global_weights)
                        
                        response = federated_learning_pb2.GlobalModel(
                            round=self.server.current_round,
                            weights=weights_bytes,
                            available=True,
                            model_config=""
                        )
                        # Log removed to avoid spam - clients poll every second
                else:
                    # Client requesting old/future round - not available
                    response = federated_learning_pb2.GlobalModel(
                        round=request.round,
                        weights=b"",
                        available=False,
                        model_config=""
                    )
                
                log_sent_packet(
                    packet_size=response.ByteSize(),
                    peer=f"client_{request.client_id}",
                    protocol="gRPC",
                    round=request.round,
                    extra_info="global_model"
                )
                
                return response
            except Exception as e:
                print(f"[gRPC] Error in GetGlobalModel: {e}")
                return federated_learning_pb2.GlobalModel(
                    round=request.round,
                    weights=b"",
                    available=False,
                    model_config=""
                )
        
        def SendModelUpdate(self, request, context):
            """Receive model update from client"""
            try:
                print(f"[gRPC] SendModelUpdate called - client_id={request.client_id}, round={request.round}, current_round={self.server.current_round}")
                
                log_received_packet(
                    packet_size=request.ByteSize(),
                    peer=f"client_{request.client_id}",
                    protocol="gRPC",
                    round=request.round,
                    extra_info="model_update"
                )
                
                # Decode the weights - they come as bytes or string
                if isinstance(request.weights, bytes):
                    weights_str = request.weights.decode('utf-8')
                else:
                    weights_str = request.weights
                
                metrics = dict(request.metrics)
                
                data = {
                    'client_id': request.client_id,
                    'round': request.round,
                    'weights': weights_str,  # Already base64-encoded from client
                    'num_samples': request.num_samples,
                    'metrics': metrics
                }
                
                print(f"[gRPC] Calling handle_client_update for client {request.client_id}, round {request.round}")
                self.server.handle_client_update(data, 'grpc')
                print(f"[gRPC] handle_client_update completed for client {request.client_id}")
                
                return federated_learning_pb2.UpdateResponse(
                    success=True,
                    message="Update received"
                )
            except Exception as e:
                print(f"[gRPC] Error in SendModelUpdate: {e}")
                import traceback
                traceback.print_exc()
                return federated_learning_pb2.UpdateResponse(
                    success=False,
                    message=str(e)
                )
        
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
                    'accuracy': request.accuracy
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
                
                # Check if client should train
                should_train = self.server.grpc_should_train.get(client_id, False)
                should_evaluate = self.server.grpc_should_evaluate.get(client_id, False)
                
                # Clear flags after reading (one-time signals)
                if should_train:
                    self.server.grpc_should_train[client_id] = False
                if should_evaluate:
                    self.server.grpc_should_evaluate[client_id] = False
                
                return federated_learning_pb2.TrainingStatus(
                    should_train=should_train,
                    should_evaluate=should_evaluate,
                    current_round=self.server.current_round,
                    is_complete=self.server.converged
                )
            except Exception as e:
                print(f"[gRPC] Error in CheckTrainingStatus: {e}")
                return federated_learning_pb2.TrainingStatus(
                    should_train=False,
                    should_evaluate=False,
                    current_round=0,
                    is_complete=True
                )


# =========================================================================
# QUIC Protocol Handler
# =========================================================================

if QUIC_AVAILABLE:
    class QUICServerProtocol(QuicConnectionProtocol):
        """QUIC protocol handler (supports multiple clients)"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server = None  # Set by factory function
            self._stream_buffers = {}  # Buffer data per stream ID (instance-specific)
            print(f"[QUIC] Protocol instance created")
        
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
                    while b'\n' in self._stream_buffers[stream_id]:
                        message_data, self._stream_buffers[stream_id] = self._stream_buffers[stream_id].split(b'\n', 1)
                        if message_data:
                            try:
                                data_str = message_data.decode('utf-8')
                                message = json.loads(data_str)
                                print(f"[QUIC] Decoded message type '{message.get('type')}' from stream {stream_id}")
                                
                                log_received_packet(
                                    packet_size=len(data_str),
                                    peer=f"quic_client_{message.get('client_id', 'unknown')}",
                                    protocol="QUIC",
                                    round=message.get('round', 0),
                                    extra_info=message.get('type', 'unknown')
                                )
                                
                                # Handle message asynchronously
                                if self.server:
                                    asyncio.create_task(self.server.handle_quic_message(message, self))
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"[QUIC] Error decoding message: {e}")
                    
                    # If stream ended and buffer has remaining data, try to process it
                    if event.end_stream and self._stream_buffers[stream_id]:
                        print(f"[QUIC] Stream {stream_id} ended with {len(self._stream_buffers[stream_id])} bytes remaining")
                        try:
                            data_str = self._stream_buffers[stream_id].decode('utf-8')
                            message = json.loads(data_str)
                            print(f"[QUIC] Decoded end-of-stream message type '{message.get('type')}'")
                            
                            log_received_packet(
                                packet_size=len(data_str),
                                peer=f"quic_client_{message.get('client_id', 'unknown')}",
                                protocol="QUIC",
                                round=message.get('round', 0),
                                extra_info=message.get('type', 'unknown')
                            )
                            
                            if self.server:
                                asyncio.create_task(self.server.handle_quic_message(message, self))
                            self._stream_buffers[stream_id] = b''
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"[QUIC] Error decoding remaining buffer: {e}")
                except Exception as e:
                    print(f"[QUIC] Error handling event: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Call parent handler for other events
                super().quic_event_received(event)


def main():
    """Main function"""
    # Use MIN_CLIENTS (dynamic clients) and configured NUM_ROUNDS.
    # MAX_CLIENTS controls the upper bound of concurrently registered clients.
    server = UnifiedFederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, max_clients=MAX_CLIENTS)
    server.run()


if __name__ == "__main__":
    main()
