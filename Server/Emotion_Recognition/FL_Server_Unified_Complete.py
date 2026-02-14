"""
Unified Federated Learning Server for Emotion Recognition
Handles all 5 protocols simultaneously: MQTT, AMQP, gRPC, QUIC, DDS

The server listens on all protocol channels and responds to clients
using whichever protocol they selected via RL.
"""

import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import os
import sys
import threading
import asyncio
from typing import List, Dict
from pathlib import Path
from concurrent import futures

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
except ImportError:
    QUIC_AVAILABLE = False
    print("Warning: aioquic not available, QUIC disabled")

try:
    from cyclonedds.domain import DomainParticipant  # DDS
    from cyclonedds.topic import Topic
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.util import duration
    from cyclonedds.core import Qos, Policy
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

# packet_logger lives in scripts/utilities (Docker: /app/scripts/utilities, local: project_root/scripts/utilities)
_utilities_path = os.path.join(project_root, 'scripts', 'utilities')
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)

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
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
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


class UnifiedFederatedLearningServer:
    """
    Unified FL Server that handles all 5 communication protocols
    """
    
    def __init__(self, num_clients, num_rounds):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = {}  # Maps client_id -> protocol_used
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        self.model_config = None
        
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
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Protocol handlers
        self.mqtt_client = None
        self.amqp_connection = None
        self.amqp_channel = None
        self.grpc_server = None
        self.quic_server = None
        self.dds_participant = None
        self.dds_writers = {}
        self.dds_readers = {}
        
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
                client_id="fl_unified_server_mqtt", 
                protocol=mqtt.MQTTv311, 
                clean_session=True
            )
            self.mqtt_client._max_packet_size = 20 * 1024 * 1024
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_message = self.on_mqtt_message
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            
            # Start MQTT loop in separate thread
            mqtt_thread = threading.Thread(target=self.mqtt_client.loop_forever, daemon=True)
            mqtt_thread.start()
            print("[MQTT] Server started")
        except Exception as e:
            print(f"[MQTT] Failed to start: {e}")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            print(f"[MQTT] Connected to broker")
            client.subscribe("fl/client_register", qos=1)
            client.subscribe("fl/client/+/update", qos=1)
            client.subscribe("fl/client/+/metrics", qos=1)
        else:
            print(f"[MQTT] Connection failed with code {rc}")
    
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
                self.handle_client_registration(data['client_id'], 'mqtt')
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
            parameters = pika.ConnectionParameters(
                host=AMQP_BROKER,
                port=AMQP_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300
            )
            self.amqp_connection = pika.BlockingConnection(parameters)
            self.amqp_channel = self.amqp_connection.channel()
            
            # Declare exchanges and queues
            self.amqp_channel.exchange_declare(
                exchange='fl_client_updates',
                exchange_type='direct',
                durable=True
            )
            
            # Queue for client registrations
            self.amqp_channel.queue_declare(queue='fl_client_register', durable=True)
            
            # Set up consumers
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
            
            print("[AMQP] Server started")
        except Exception as e:
            print(f"[AMQP] Failed to start: {e}")
            import traceback
            traceback.print_exc()
    
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
            
            # Set up queues for this client
            self.amqp_channel.queue_declare(queue=f'client_{client_id}_updates', durable=True)
            self.amqp_channel.queue_bind(
                exchange='fl_client_updates',
                queue=f'client_{client_id}_updates',
                routing_key=f'client_{client_id}_update'
            )
            
            self.amqp_channel.basic_consume(
                queue=f'client_{client_id}_updates',
                on_message_callback=lambda ch, method, props, body: self.on_amqp_update(ch, method, props, body, client_id),
                auto_ack=True
            )
            
            # Queue for metrics
            self.amqp_channel.queue_declare(queue=f'client_{client_id}_metrics', durable=True)
            self.amqp_channel.queue_bind(
                exchange='fl_client_updates',
                queue=f'client_{client_id}_metrics',
                routing_key=f'client_{client_id}_metrics'
            )
            
            self.amqp_channel.basic_consume(
                queue=f'client_{client_id}_metrics',
                on_message_callback=lambda ch, method, props, body: self.on_amqp_metrics(ch, method, props, body, client_id),
                auto_ack=True
            )
        except Exception as e:
            print(f"[AMQP] Error handling registration: {e}")
            import traceback
            traceback.print_exc()
    
    def on_amqp_update(self, ch, method, properties, body, client_id):
        """AMQP update callback"""
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
        except Exception as e:
            print(f"[AMQP] Error handling update: {e}")
    
    def on_amqp_metrics(self, ch, method, properties, body, client_id):
        """AMQP metrics callback"""
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
        except Exception as e:
            print(f"[AMQP] Error handling metrics: {e}")
    
    def send_via_amqp(self, client_id, message_type, message):
        """Send message to client via AMQP"""
        if not AMQP_AVAILABLE or not self.amqp_channel:
            return
        
        try:
            payload = json.dumps(message)
            queue_name = f'client_{client_id}_{message_type}'
            
            # Declare queue if not exists
            self.amqp_channel.queue_declare(queue=queue_name, durable=True)
            
            self.amqp_channel.basic_publish(
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
        except Exception as e:
            print(f"[AMQP] Error sending to client {client_id}: {e}")
    
    # =========================================================================
    # gRPC PROTOCOL HANDLERS
    # =========================================================================
    
    def start_grpc_server(self):
        """Start gRPC protocol handler"""
        if not GRPC_AVAILABLE:
            return
        
        try:
            self.grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=10),
                options=[
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
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
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._run_quic_server())
            
            quic_thread = threading.Thread(target=run_quic, daemon=True)
            quic_thread.start()
            print(f"[QUIC] Server started on {QUIC_HOST}:{QUIC_PORT}")
        except Exception as e:
            print(f"[QUIC] Failed to start: {e}")
    
    async def _run_quic_server(self):
        """Async QUIC server"""
        try:
            configuration = QuicConfiguration(
                is_client=False,
                max_datagram_frame_size=65536,
            )
            
            # Generate self-signed certificate for QUIC
            import ssl
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.primitives import serialization
            import datetime
            
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
            
            await serve(
                QUIC_HOST,
                QUIC_PORT,
                configuration=configuration,
                create_protocol=lambda *args, **kwargs: QUICServerProtocol(self, *args, **kwargs),
            )
        except Exception as e:
            print(f"[QUIC] Server error: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # DDS PROTOCOL HANDLERS
    # =========================================================================
    
    def start_dds_server(self):
        """Start DDS protocol handler"""
        if not DDS_AVAILABLE:
            return
        
        try:
            # Create DDS participant
            self.dds_participant = DomainParticipant(DDS_DOMAIN_ID)
            
            # Define QoS for reliable communication
            qos = Qos(
                Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                Policy.History.KeepAll,
                Policy.Durability.Volatile
            )
            
            # Start DDS listener thread
            def dds_listener():
                while not self.converged:
                    try:
                        time.sleep(0.1)
                        # DDS uses callbacks, so just keep thread alive
                    except Exception as e:
                        print(f"[DDS] Listener error: {e}")
            
            dds_thread = threading.Thread(target=dds_listener, daemon=True)
            dds_thread.start()
            
            print(f"[DDS] Server started on domain {DDS_DOMAIN_ID}")
        except Exception as e:
            print(f"[DDS] Failed to start: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # COMMON HANDLERS (Protocol-agnostic)
    # =========================================================================
    
    def handle_client_registration(self, client_id, protocol):
        """Handle client registration (thread-safe)"""
        with self.lock:
            self.registered_clients[client_id] = protocol
            print(f"[{protocol.upper()}] Client {client_id} registered "
                  f"({len(self.registered_clients)}/{self.num_clients})")
            
            if len(self.registered_clients) == self.num_clients:
                print(f"\n[Server] All {self.num_clients} clients registered!")
                print("[Server] Distributing initial global model...\n")
                time.sleep(2)
                self.distribute_initial_model()
                self.start_time = time.time()
                print(f"[Server] Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def handle_client_update(self, data, protocol):
        """Handle client model update (thread-safe)"""
        with self.lock:
            client_id = data['client_id']
            round_num = data['round']
            
            if round_num != self.current_round:
                print(f"[{protocol.upper()}] Ignoring update from client {client_id} "
                      f"(round {round_num} != current {self.current_round})")
                return
            
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
            
            # Store update
            self.client_updates[client_id] = {
                'weights': weights,
                'num_samples': data['num_samples'],
                'metrics': data['metrics'],
                'protocol': protocol
            }
            
            print(f"[{protocol.upper()}] Received update from client {client_id} "
                  f"({len(self.client_updates)}/{self.num_clients})")
            
            if len(self.client_updates) == self.num_clients:
                self.aggregate_models()
    
    def handle_client_metrics(self, data, protocol):
        """Handle client evaluation metrics (thread-safe)"""
        with self.lock:
            client_id = data['client_id']
            round_num = data['round']
            
            if round_num != self.current_round:
                print(f"[{protocol.upper()}] Ignoring metrics from client {client_id} "
                      f"(round {round_num} != current {self.current_round})")
                return
            
            self.client_metrics[client_id] = {
                'num_samples': data['num_samples'],
                'loss': data['loss'],
                'accuracy': data['accuracy'],
                'protocol': protocol
            }
            
            print(f"[{protocol.upper()}] Received metrics from client {client_id} "
                  f"({len(self.client_metrics)}/{self.num_clients})")
            
            if len(self.client_metrics) == self.num_clients:
                self.aggregate_metrics()
    
    def distribute_initial_model(self):
        """Distribute initial global model and config to all clients"""
        message = {
            'round': 0,
            'weights': self.serialize_weights(self.global_weights),
            'model_config': self.model_config
        }
        
        for client_id, protocol in self.registered_clients.items():
            try:
                if protocol == 'mqtt':
                    self.send_via_mqtt(client_id, "fl/global_model", message)
                elif protocol == 'amqp':
                    self.send_via_amqp(client_id, 'global_model', message)
                elif protocol == 'grpc':
                    pass  # gRPC uses pull model, clients will request
                elif protocol == 'quic':
                    pass  # QUIC handled separately
                elif protocol == 'dds':
                    pass  # DDS uses pub/sub
                
                print(f"[{protocol.upper()}] Sent initial model to client {client_id}")
            except Exception as e:
                print(f"[{protocol.upper()}] Error sending initial model to client {client_id}: {e}")
        
        # Start first round
        time.sleep(2)
        self.current_round = 1
        self.signal_start_training()
    
    def signal_start_training(self):
        """Signal all clients to start training for current round"""
        message = {'round': self.current_round}
        
        for client_id, protocol in self.registered_clients.items():
            try:
                if protocol == 'mqtt':
                    self.send_via_mqtt(client_id, "fl/start_training", message)
                elif protocol == 'amqp':
                    self.send_via_amqp(client_id, 'start_training', message)
                # Other protocols handled via their mechanisms
            except Exception as e:
                print(f"[{protocol.upper()}] Error signaling training to client {client_id}: {e}")
    
    def signal_start_evaluation(self):
        """Signal all clients to start evaluation for current round"""
        message = {'round': self.current_round}
        
        for client_id, protocol in self.registered_clients.items():
            try:
                if protocol == 'mqtt':
                    self.send_via_mqtt(client_id, "fl/start_evaluation", message)
                elif protocol == 'amqp':
                    self.send_via_amqp(client_id, 'start_evaluation', message)
                # Other protocols handled via their mechanisms
            except Exception as e:
                print(f"[{protocol.upper()}] Error signaling evaluation to client {client_id}: {e}")
    
    def broadcast_global_model(self):
        """Broadcast updated global model to all clients"""
        message = {
            'round': self.current_round,
            'weights': self.serialize_weights(self.global_weights)
        }
        
        for client_id, protocol in self.registered_clients.items():
            try:
                if protocol == 'mqtt':
                    self.send_via_mqtt(client_id, "fl/global_model", message)
                elif protocol == 'amqp':
                    self.send_via_amqp(client_id, 'global_model', message)
                # Other protocols handled separately
                
                print(f"[{protocol.upper()}] Sent global model to client {client_id}")
            except Exception as e:
                print(f"[{protocol.upper()}] Error broadcasting to client {client_id}: {e}")
    
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
                if request.round == 0:
                    # Initial model with config
                    weights_bytes = pickle.dumps(self.server.global_weights)
                    model_config_json = json.dumps(self.server.model_config)
                    
                    response = federated_learning_pb2.GlobalModel(
                        round=0,
                        weights=weights_bytes,
                        available=True,
                        model_config=model_config_json
                    )
                elif request.round == self.server.current_round:
                    weights_bytes = pickle.dumps(self.server.global_weights)
                    
                    response = federated_learning_pb2.GlobalModel(
                        round=self.server.current_round,
                        weights=weights_bytes,
                        available=True,
                        model_config=""
                    )
                else:
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
                log_received_packet(
                    packet_size=request.ByteSize(),
                    peer=f"client_{request.client_id}",
                    protocol="gRPC",
                    round=request.round,
                    extra_info="model_update"
                )
                
                weights = pickle.loads(request.weights)
                metrics = dict(request.metrics)
                
                data = {
                    'client_id': request.client_id,
                    'round': request.round,
                    'weights': base64.b64encode(pickle.dumps(weights)).decode('utf-8'),
                    'num_samples': request.num_samples,
                    'metrics': metrics
                }
                
                self.server.handle_client_update(data, 'grpc')
                
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
            """Check if client should start training"""
            try:
                should_train = (request.round == self.server.current_round)
                
                return federated_learning_pb2.TrainingStatus(
                    should_train=should_train,
                    current_round=self.server.current_round,
                    is_complete=self.server.converged
                )
            except Exception as e:
                print(f"[gRPC] Error in CheckTrainingStatus: {e}")
                return federated_learning_pb2.TrainingStatus(
                    should_train=False,
                    current_round=0,
                    is_complete=True
                )


# =========================================================================
# QUIC Protocol Handler
# =========================================================================

if QUIC_AVAILABLE:
    class QUICServerProtocol(QuicConnectionProtocol):
        """QUIC protocol handler"""
        
        def __init__(self, server, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server = server
        
        def quic_event_received(self, event):
            """Handle QUIC events"""
            if isinstance(event, StreamDataReceived):
                try:
                    data = event.data.decode().strip()
                    message = json.loads(data)
                    
                    log_received_packet(
                        packet_size=len(data),
                        peer=f"quic_client_{message.get('client_id', 'unknown')}",
                        protocol="QUIC",
                        round=message.get('round', 0),
                        extra_info=message.get('type', 'unknown')
                    )
                    
                    if message['type'] == 'register':
                        self.server.handle_client_registration(message['client_id'], 'quic')
                    elif message['type'] == 'update':
                        self.server.handle_client_update(message, 'quic')
                    elif message['type'] == 'metrics':
                        self.server.handle_client_metrics(message, 'quic')
                except Exception as e:
                    print(f"[QUIC] Error handling event: {e}")


def main():
    """Main function"""
    server = UnifiedFederatedLearningServer(NUM_CLIENTS, NUM_ROUNDS)
    server.run()


if __name__ == "__main__":
    main()
