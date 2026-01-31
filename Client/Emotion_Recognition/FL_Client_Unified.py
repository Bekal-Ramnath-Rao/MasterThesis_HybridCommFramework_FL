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
from typing import Dict, Tuple, Optional
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
            self.mqtt_client.publish("fl/client_register", 
                                    json.dumps({"client_id": self.client_id}), qos=1)
            print(f"  Registration message sent\n")
        else:
            print(f"Client {self.client_id} failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
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
        if rc == 0:
            print(f"\nClient {self.client_id} clean disconnect from broker")
            print(f"Client {self.client_id} exiting...")
            self.mqtt_client.loop_stop()
            import sys
            sys.exit(0)
        else:
            print(f"Client {self.client_id} unexpected disconnect, return code {rc}")
    
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
            print(f"Client {self.client_id} starting evaluation for round {round_num}...")
            self.evaluate_model()
            self.evaluated_rounds.add(round_num)
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
        try:
            if protocol == 'mqtt':
                self._send_via_mqtt(update_message)
            elif protocol == 'amqp':
                self._send_via_amqp(update_message)
            elif protocol == 'grpc':
                self._send_via_grpc(update_message)
            elif protocol == 'quic':
                self._send_via_quic(update_message)
            elif protocol == 'dds':
                self._send_via_dds(update_message)
            else:
                print(f"Client {self.client_id} ERROR: Unknown protocol {protocol}, falling back to MQTT")
                self._send_via_mqtt(update_message)
            
            self.round_metrics['communication_time'] = time.time() - comm_start
            self.round_metrics['success'] = True
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending update via {protocol}: {e}")
            import traceback
            traceback.print_exc()
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
                    weights=weights_str,
                    num_samples=message['num_samples'],
                    metrics=json.dumps(message['metrics'])
                )
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
            
            if not response.success:
                raise Exception(f"gRPC send failed: {response.message}")
            
            channel.close()
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via gRPC: {e}")
            raise
    
    def _send_via_quic(self, message: dict):
        """Send model update via QUIC"""
        if asyncio is None or connect is None:
            raise ImportError("aioquic module not available for QUIC")
        
        try:
            quic_host = os.getenv("QUIC_HOST", "localhost")
            quic_port = int(os.getenv("QUIC_PORT", "4433"))
            
            payload = json.dumps(message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via QUIC - size: {payload_size_mb:.2f} MB")
            
            # Run async QUIC send
            asyncio.run(self._quic_send_data(quic_host, quic_port, payload, 'model_update'))
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via QUIC")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via QUIC: {e}")
            raise
    
    def _send_metrics_via_quic(self, message: dict):
        """Send metrics via QUIC"""
        if asyncio is None or connect is None:
            raise ImportError("aioquic module not available for QUIC")
        
        try:
            quic_host = os.getenv("QUIC_HOST", "localhost")
            quic_port = int(os.getenv("QUIC_PORT", "4433"))
            
            payload = json.dumps(message)
            asyncio.run(self._quic_send_data(quic_host, quic_port, payload, 'metrics'))
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via QUIC: {e}")
            raise
    
    async def _quic_send_data(self, host: str, port: int, payload: str, msg_type: str):
        """Async QUIC data send"""
        try:
            config = QuicConfiguration(is_client=True, verify_mode='unverified')
            
            async with await connect(
                host, port, configuration=config, local_host='0.0.0.0'
            ) as protocol:
                stream_id = protocol._quic_connection.get_next_available_stream_id()
                protocol._quic_connection.send_stream_data(stream_id, (payload + '\n').encode())
                
                # Wait for response
                time.sleep(0.5)
        except Exception as e:
            print(f"QUIC send error: {e}")
            raise
    
    def _send_via_dds(self, message: dict):
        """Send model update via DDS"""
        try:
            # DDS implementation - simplified for now
            # In real implementation, would use cyclonedds or opensplice
            import ddspython
            
            payload = json.dumps(message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via DDS - size: {payload_size_mb:.2f} MB")
            
            # TODO: Implement actual DDS send
            print(f"Client {self.client_id} sent model update for round {self.current_round} via DDS")
        except ImportError:
            print(f"Client {self.client_id} WARNING: DDS not available, falling back to MQTT")
            self._send_via_mqtt(message)
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via DDS: {e}")
            raise
    
    def _send_metrics_via_dds(self, message: dict):
        """Send metrics via DDS"""
        try:
            import ddspython
            
            payload = json.dumps(message)
            # TODO: Implement actual DDS send
        except ImportError:
            print(f"Client {self.client_id} WARNING: DDS not available, falling back to MQTT")
            self._send_metrics_via_mqtt(message)
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
