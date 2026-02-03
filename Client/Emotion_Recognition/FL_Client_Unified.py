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

# Detect Docker environment and set project root
if os.path.exists('/app'):
    project_root = '/app'
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from packet_logger import init_db, log_sent_packet, log_received_packet

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

# DDS Data Structures (must be defined at module level for Python 3.8)
if DDS_AVAILABLE:
    from dataclasses import dataclass, field
    
    @dataclass
    class GlobalModel(IdlStruct):
        round: int
        weights: sequence[int]  # CycloneDDS sequence type for sequence<octet> in IDL
    
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
        self.evaluated_rounds = set()
        self.waiting_for_aggregated_model = False  # Track if we sent update and waiting for aggregated model
        
        # Training configuration
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        
        # RL Components
        if USE_RL_SELECTION and QLearningProtocolSelector is not None:
            self.rl_selector = QLearningProtocolSelector(
                save_path=f"q_table_emotion_client_{client_id}.pkl"
            )
            self.env_manager = EnvironmentStateManager()
            self.env_manager.update_model_size('medium')  # Emotion recognition
        else:
            self.rl_selector = None
            self.env_manager = None
        
        # Track selected protocol and metrics
        self.selected_protocol = None
        self.round_metrics = {
            'communication_time': 0.0,
            'training_time': 0.0,
            'accuracy': 0.0,
            'success': False
        }
        
        # DDS Components
        if DDS_AVAILABLE:
            try:
                # Create DDS participant
                self.dds_participant = DomainParticipant(DDS_DOMAIN_ID)
                
                # Create QoS for reliable communication
                qos = Qos(
                    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                    Policy.History.KeepLast(10),
                    Policy.Durability.TransientLocal
                )
                
                # Create topics and writers
                self.dds_update_topic = Topic(self.dds_participant, "ModelUpdate", ModelUpdate)
                self.dds_update_writer = DataWriter(self.dds_participant, self.dds_update_topic, qos=qos)
                
                self.dds_metrics_topic = Topic(self.dds_participant, "EvaluationMetrics", EvaluationMetrics)
                self.dds_metrics_writer = DataWriter(self.dds_participant, self.dds_metrics_topic, qos=qos)
                
                print(f"[DDS] Client {client_id} initialized on domain {DDS_DOMAIN_ID}")
            except Exception as e:
                print(f"[DDS] Initialization failed: {e}")
                self.dds_participant = None
                self.dds_update_writer = None
                self.dds_metrics_writer = None
        else:
            self.dds_participant = None
            self.dds_update_writer = None
            self.dds_metrics_writer = None
        
        # Initialize packet logger
        init_db()
        
        # QUIC persistent connection components
        self.quic_protocol = None
        self.quic_connection_task = None
        self.quic_loop = None
        self.quic_thread = None
        
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
        self.mqtt_client.max_queued_messages_set(0)
        self.mqtt_client._max_packet_size = 20 * 1024 * 1024
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
            
            # Send registration message
            registration_msg = json.dumps({"client_id": self.client_id})
            self.mqtt_client.publish("fl/client_register", registration_msg, qos=1)
            log_sent_packet(
                packet_size=len(registration_msg),
                peer="server",
                protocol="MQTT",
                round=0,
                extra_info="registration"
            )
            print(f"  Registration message sent\n")
        else:
            print(f"Client {self.client_id} failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
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
                    parameters = pika.ConnectionParameters(
                        host=os.getenv("AMQP_HOST", "rabbitmq-broker-unified"),
                        port=int(os.getenv("AMQP_PORT", "5672")),
                        credentials=credentials,
                        heartbeat=600,
                        blocked_connection_timeout=300
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
            
            # Update model
            if self.model and round_num > self.last_global_round:
                self.model.set_weights(weights)
                self.last_global_round = round_num
                print(f"[AMQP] Client {self.client_id} updated model weights for round {round_num}")
                
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
                print(f"[AMQP] Client {self.client_id} ERROR: Model not initialized yet, cannot start training for round {round_num}")
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
                    print(f"[DDS] Client {self.client_id} ERROR: Model not initialized yet, cannot start training for round {round_num}")
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
                channel = grpc.insecure_channel(f"{grpc_host}:{grpc_port}")
                self.grpc_stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
                
                print(f"[gRPC] Listener started for client {self.client_id}")
                
                # Polling loop
                while True:
                    try:
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
            
            # Update model
            if self.model and round_num > self.last_global_round:
                self.model.set_weights(weights)
                self.last_global_round = round_num
                self.waiting_for_aggregated_model = False  # Clear flag: received aggregated model
                print(f"[gRPC] Client {self.client_id} updated model weights for round {round_num}")
                
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
            self.quic_thread = threading.Thread(
                target=self._run_quic_loop,
                daemon=True,
                name=f"QUIC-Client-{self.client_id}"
            )
            self.quic_thread.start()
            
            # Wait briefly for connection to establish
            time.sleep(2)
            print(f"[QUIC] Listener started for client {self.client_id}")
    
    def handle_global_model(self, payload):
        """Receive and set global model weights from server"""
        try:
            data = json.loads(payload.decode())
            round_num = data['round']
            
            # Ignore duplicate global model for the same round
            if round_num <= self.last_global_round:
                print(f"Client {self.client_id} ignoring duplicate global model for round {round_num}")
                return
            
            encoded_weights = data['weights']
            weights = self.deserialize_weights(encoded_weights)
            
            if round_num == 0:
                # Initial model from server
                print(f"Client {self.client_id} received initial global model from server")
                
                model_config = data.get('model_config')
                if model_config:
                    self.model = self.build_model_from_config(model_config)
                    print(f"Client {self.client_id} built CNN model from server configuration")
                else:
                    raise ValueError("No model configuration received from server!")
                
                self.model.set_weights(weights)
                print(f"Client {self.client_id} model initialized with server weights")
                self.current_round = 0
                self.last_global_round = 0
            else:
                # Updated model after aggregation
                if self.model is None:
                    print(f"Client {self.client_id} ERROR: Received model for round {round_num} but local model not initialized!")
                    return
                self.model.set_weights(weights)
                self.current_round = round_num
                self.last_global_round = round_num
                self.waiting_for_aggregated_model = False  # Clear flag: received aggregated model
                print(f"Client {self.client_id} received global model for round {round_num}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR in handle_global_model: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_training_config(self, payload):
        """Update training configuration"""
        self.training_config = json.loads(payload.decode())
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    def handle_start_training(self, payload):
        """Start local training when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']
        
        # Check if model is initialized - WAIT for global model
        if self.model is None:
            print(f"Client {self.client_id} ERROR: Model not initialized yet, cannot start training for round {round_num}")
            print(f"Client {self.client_id} waiting for global model from server...")
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
        data = json.loads(payload.decode())
        round_num = data['round']
        
        if self.model is None:
            print(f"Client {self.client_id} ERROR: Model not initialized yet, cannot evaluate for round {round_num}")
            return
        
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
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nDisconnecting from server...")
        time.sleep(1)
        self.mqtt_client.disconnect()
        print(f"Client {self.client_id} disconnected successfully.")
    
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
                
                state = self.env_manager.get_current_state()
                protocol = self.rl_selector.select_protocol(state, training=True)
                
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
        
        return model
    
    def serialize_weights(self, weights):
        """Serialize model weights for transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded
    
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
        protocols_to_try = [protocol, 'amqp', 'mqtt', 'grpc', 'quic', 'dds']  # AMQP second in fallback for testing
        
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
                elif attempt_protocol == 'quic' and asyncio is not None:
                    self._send_via_quic(update_message)
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
        loss, accuracy = self.model.evaluate(
            self.validation_generator,
            verbose=0
        )
        
        num_samples = self.validation_generator.n
        
        # Select protocol based on RL
        protocol = self.select_protocol()
        
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
            elif protocol == 'quic':
                self._send_metrics_via_quic(metrics_message)
            elif protocol == 'dds':
                self._send_metrics_via_dds(metrics_message)
            else:
                print(f"Client {self.client_id} ERROR: Unknown protocol {protocol}, falling back to MQTT")
                self._send_metrics_via_mqtt(metrics_message)
            
            self.round_metrics['communication_time'] = time.time() - comm_start
            print(f"Client {self.client_id} sent evaluation metrics for round {self.current_round}")
            print(f"Evaluation metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics: {e}")
            import traceback
            traceback.print_exc()
    
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
            result.wait_for_publish(timeout=30)
            
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
            result.wait_for_publish(timeout=30)
            
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
            parameters = pika.ConnectionParameters(
                host=amqp_host,
                port=amqp_port,
                credentials=credentials,
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
            parameters = pika.ConnectionParameters(
                host=amqp_host,
                port=amqp_port,
                credentials=credentials,
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
            
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
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
            
            options = [
                ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ('grpc.max_receive_message_length', 100 * 1024 * 1024),
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
            self.quic_loop.run_until_complete(self._quic_connection_loop())
        except Exception as e:
            print(f"[QUIC] Event loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.quic_loop.close()
    
    async def _quic_connection_loop(self):
        """Maintain persistent QUIC connection (runs in background)"""
        import ssl
        quic_host = os.getenv("QUIC_HOST", "localhost")
        quic_port = int(os.getenv("QUIC_PORT", "4433"))
        
        config = QuicConfiguration(
            is_client=True, 
            verify_mode=ssl.CERT_NONE,
            max_stream_data=50 * 1024 * 1024,  # 50 MB per stream
            max_data=100 * 1024 * 1024,  # 100 MB total
            idle_timeout=3600.0  # 1 hour
        )
        
        print(f"[QUIC] Client {self.client_id} connecting to {quic_host}:{quic_port}...")
        
        # Create protocol factory that sets client reference
        def create_protocol(*args, **kwargs):
            protocol = UnifiedClientQUICProtocol(*args, **kwargs)
            protocol.client = self
            return protocol
        
        try:
            async with connect(
                quic_host,
                quic_port,
                configuration=config,
                create_protocol=create_protocol
            ) as protocol:
                self.quic_protocol = protocol
                print(f"[QUIC] Client {self.client_id} established persistent connection")
                
                # Keep connection alive indefinitely
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    print(f"[QUIC] Client {self.client_id} connection cancelled")
        except Exception as e:
            print(f"[QUIC] Client {self.client_id} connection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.quic_protocol = None
    
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
            
            # Update model
            if self.model:
                self.model.set_weights(weights)
                print(f"[QUIC] Client {self.client_id} updated model weights for round {round_num}")
            
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
            # Add 'type' field for server to identify message type
            quic_message = {**message, 'type': 'update'}
            
            payload = json.dumps(quic_message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via QUIC - size: {payload_size_mb:.2f} MB")
            
            # Use persistent connection directly via run_coroutine_threadsafe
            if self.quic_loop is None:
                raise ConnectionError("QUIC connection not established - loop not available")
            
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
            # Add 'type' field for server to identify message type
            quic_message = {**message, 'type': 'metrics'}
            
            payload = json.dumps(quic_message)
            
            # Use persistent connection directly via run_coroutine_threadsafe
            if self.quic_loop is None:
                raise ConnectionError("QUIC connection not established - loop not available")
            
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
        
        # For large messages, give time for transmission
        if len(data) > 1000000:  # > 1MB
            for _ in range(3):
                await asyncio.sleep(0.5)
                self.quic_protocol.transmit()
    
    async def _quic_send_data(self, host: str, port: int, payload: str, msg_type: str):
        """Async QUIC data send with timeout and retry (legacy method for registration)"""
        import ssl
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use ssl.CERT_NONE for self-signed certificate verification
                config = QuicConfiguration(is_client=True, verify_mode=ssl.CERT_NONE)
                
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
    
    def _send_via_dds(self, message: dict):
        """Send model update via DDS"""
        if not DDS_AVAILABLE or not self.dds_update_writer:
            raise NotImplementedError("DDS not available - triggering fallback")
        
        try:
            # Serialize weights and convert to list of integers
            weights_bytes = pickle.dumps(message['weights'])
            weights_list = list(weights_bytes)  # Convert bytes to List[int]
            
            # Create DDS message
            dds_msg = ModelUpdate(
                client_id=self.client_id,
                round=message['round'],
                weights=weights_list,
                num_samples=message.get('num_samples', 0),
                loss=message.get('loss', 0.0),
                mse=message.get('mse', 0.0),
                mae=message.get('mae', 0.0),
                mape=message.get('mape', 0.0)
            )
            
            # Write to DDS
            self.dds_update_writer.write(dds_msg)
            
            # Log the packet
            log_sent_packet(
                packet_size=len(weights_bytes),
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="model_update"
            )
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via DDS")
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
    
    def start(self):
        """Connect to MQTT broker and start listening"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
                self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=120)
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 3600)
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
