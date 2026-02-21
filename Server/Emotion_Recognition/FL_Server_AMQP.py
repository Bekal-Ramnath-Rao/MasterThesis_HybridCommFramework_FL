import numpy as np
import json
import pickle
import base64
import time
import pika
import os
import sys
from typing import List, Dict
import matplotlib.pyplot as plt
from pathlib import Path

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

from packet_logger import log_sent_packet, log_received_packet, init_db

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
AMQP_HOST = os.getenv("AMQP_HOST", "localhost")
AMQP_PORT = int(os.getenv("AMQP_PORT", "5672"))
AMQP_USER = os.getenv("AMQP_USER", "guest")
AMQP_PASSWORD = os.getenv("AMQP_PASSWORD", "guest")
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence

# Convergence Settings (primary stopping criterion)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))

# AMQP Exchanges and Queues
EXCHANGE_BROADCAST = "fl_broadcast"
EXCHANGE_CLIENT_UPDATES = "fl_client_updates"
QUEUE_CLIENT_REGISTER = "fl.client.register"
QUEUE_CLIENT_UPDATE = "fl.client.update"
QUEUE_CLIENT_METRICS = "fl.client.metrics"


class FederatedLearningServer:
    def __init__(self, min_clients, num_rounds, max_clients=100):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.num_clients = min_clients  # Start with minimum, will update as clients join
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.active_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        
        # Metrics storage for classification
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.training_started = False
        self.training_started = False
        self.start_time = None
        self.convergence_time = None

        # Initialize packet logging database
        init_db()
        
        # Initialize quantization handler (default: disabled unless explicitly enabled)
        uq_env = os.getenv("USE_QUANTIZATION", "false")
        use_quantization = uq_env.lower() in ("true", "1", "yes", "y")
        if use_quantization and QUANTIZATION_AVAILABLE:
            self.quantization_handler = ServerQuantizationHandler(QuantizationConfig())
            print("Server: Quantization enabled")
        else:
            self.quantization_handler = None
            if use_quantization and not QUANTIZATION_AVAILABLE:
                print("Server: Quantization requested but not available")
            else:
                print("Server: Quantization disabled")
        
        # Initialize global model
        self.initialize_global_model()
        
        # Training configuration
        # Training configuration broadcast to AMQP clients
        self.training_config = {
            "batch_size": 32,
            "local_epochs": 20  # Reduced from 20 for faster experiments
        }
        
        # AMQP connection
        self.connection = None
        self.channel = None
        self.consuming = False
    
    def initialize_global_model(self):
        """Initialize the global CNN model for emotion recognition"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
        from tensorflow.keras.optimizers import Adam
        
        # Create CNN model for emotion recognition (7 classes, 48x48 grayscale images)
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
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        # Get initial weights
        self.global_weights = model.get_weights()
        
        # Store model configuration for sending to clients
        self.model_config = {
            "input_shape": [48, 48, 1],
            "num_classes": 7,
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
            ]
        }
        
        print("\nGlobal CNN model initialized for emotion recognition")
        print(f"Model architecture: CNN with {len(self.global_weights)} weight layers")
        print(f"Input shape: 48x48x1 (grayscale images)")
        print(f"Output classes: 7 emotions")
    
    def serialize_weights(self, weights):
        """Serialize model weights for AMQP transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from AMQP"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights
    
    def connect(self):
        """Connect to RabbitMQ broker"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to RabbitMQ at {AMQP_HOST}:{AMQP_PORT}...")
                credentials = pika.PlainCredentials(AMQP_USER, AMQP_PASSWORD)
                # FAIR CONFIG: heartbeat=600s for very_poor network scenarios
                parameters = pika.ConnectionParameters(
                    host=AMQP_HOST,
                    port=AMQP_PORT,
                    credentials=credentials,
                    heartbeat=600,  # 10 minutes for very_poor network
                    blocked_connection_timeout=600  # Aligned with heartbeat
                )
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                
                # Declare exchanges
                self.channel.exchange_declare(exchange=EXCHANGE_BROADCAST, exchange_type='fanout', durable=True)
                self.channel.exchange_declare(exchange=EXCHANGE_CLIENT_UPDATES, exchange_type='direct', durable=True)
                
                # Declare queues for receiving from clients
                self.channel.queue_declare(queue=QUEUE_CLIENT_REGISTER, durable=True)
                self.channel.queue_declare(queue=QUEUE_CLIENT_UPDATE, durable=True)
                self.channel.queue_declare(queue=QUEUE_CLIENT_METRICS, durable=True)
                
                # Bind queues to exchange with routing keys
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_REGISTER, routing_key='client.register')
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_UPDATE, routing_key='client.update')
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_METRICS, routing_key='client.metrics')
                
                # Set up consumers
                self.channel.basic_consume(queue=QUEUE_CLIENT_REGISTER, on_message_callback=self.on_client_register, auto_ack=True)
                self.channel.basic_consume(queue=QUEUE_CLIENT_UPDATE, on_message_callback=self.on_client_update, auto_ack=True)
                self.channel.basic_consume(queue=QUEUE_CLIENT_METRICS, on_message_callback=self.on_client_metrics, auto_ack=True)

                print(f"Server connected to RabbitMQ broker\n")
                return True
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to RabbitMQ broker after {max_retries} attempts.")
                    print(f"\nPlease ensure:")
                    print(f"  1. RabbitMQ broker is running")
                    print(f"  2. Broker address is correct: {AMQP_HOST}:{AMQP_PORT}")
                    print(f"  3. Credentials are correct: {AMQP_USER}")
                    raise
    
    def on_client_register(self, ch, method, properties, body):
        """Handle client registration"""
        try:
            data = json.loads(body.decode())
            log_received_packet(
                packet_size=len(body),
                peer=EXCHANGE_CLIENT_UPDATES,
                protocol="AMQP",
                round=None,
                extra_info="Client registration"
            )
            client_id = data['client_id']
            self.registered_clients.add(client_id)
            self.active_clients.add(client_id)
            print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
            
            if len(self.registered_clients) > self.num_clients:
                self.update_client_count(len(self.registered_clients))
            
            if self.training_started:
                self.active_clients.add(client_id)
                print(f"[LATE JOIN] Client {client_id} joining during round {self.current_round}")
                if len(self.registered_clients) > self.num_clients:
                    self.update_client_count(len(self.registered_clients))
                if self.global_weights is not None:
                    self.send_current_model_to_client(client_id)
                return
            
            if len(self.registered_clients) >= self.min_clients:
                print("\nAll clients registered. Distributing initial global model...\n")
                time.sleep(2)
                self.distribute_initial_model()
                self.start_time = time.time()
                self.training_started = True
                print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Server error handling registration: {e}")
    
    def mark_client_converged(self, client_id):
        """Remove converged client from active federation."""
        if client_id in self.active_clients:
            self.active_clients.discard(client_id)
            self.client_updates.pop(client_id, None)
            self.client_metrics.pop(client_id, None)
            print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
            if not self.active_clients:
                self.converged = True
                print("All clients converged. Ending training.")
                self.convergence_time = time.time() - self.start_time if self.start_time else 0
                self.send_training_complete()
                self.plot_results()
                self.save_results()
                self.stop()
                return
            # Re-check: remaining active clients may have already sent metrics/updates
            if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                self.aggregate_metrics()
                self.continue_training()
                return
            if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                self.aggregate_models()
    
    def on_client_update(self, ch, method, properties, body):
        """Handle model update from client"""
        recv_start_cpu = time.perf_counter() if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1" else None
        log_received_packet(
            packet_size=len(body),
            peer=EXCHANGE_CLIENT_UPDATES,
            protocol="AMQP",
            round=None,
            extra_info="Model update"
        )
        try:
            data = json.loads(body.decode())
            client_id = data['client_id']
            round_num = data['round']
            
            if client_id not in self.active_clients:
                return
            if float(data.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
                self.mark_client_converged(client_id)
                return
            if round_num == self.current_round:
                # Check if update is compressed
                if 'compressed_data' in data and self.quantization_handler is not None:
                    compressed_update = data['compressed_data']
                    # If client sent serialized base64 string, decode and unpickle
                    if isinstance(compressed_update, str):
                        try:
                            compressed_update = pickle.loads(base64.b64decode(compressed_update.encode('utf-8')))
                        except Exception as e:
                            print(f"Server error decoding compressed_data from client {client_id}: {e}")
                    weights = self.quantization_handler.decompress_client_update(
                        client_id, 
                        compressed_update
                    )
                    print(f"Received and decompressed update from client {client_id}")
                else:
                    weights = self.deserialize_weights(data['weights'])
                
                if recv_start_cpu is not None:
                    O_recv = time.perf_counter() - recv_start_cpu
                    recv_end_ts = time.time()
                    send_start_ts = data.get("diagnostic_send_start_ts", recv_end_ts)
                    print(f"FL_DIAG client_id={client_id} O_recv={O_recv:.9f} recv_end_ts={recv_end_ts:.9f} send_start_ts={send_start_ts:.9f}")
                
                self.client_updates[client_id] = {
                    'weights': weights,
                    'num_samples': data['num_samples'],
                    'metrics': data['metrics']
                }
                
                print(f"Received update from client {client_id} "
                      f"({len(self.client_updates)}/{len(self.active_clients)})")
                
                if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                    self.aggregate_models()
        except Exception as e:
            print(f"Server error handling client update: {e}")
    
    def on_client_metrics(self, ch, method, properties, body):
        """Handle evaluation metrics from client"""
        log_received_packet(
            packet_size=len(body),
            peer=EXCHANGE_CLIENT_UPDATES,
            protocol="AMQP",
            round=None,
            extra_info="Evaluation metrics"
        )
        try:
            data = json.loads(body.decode())
            client_id = data['client_id']
            round_num = data['round']
            
            if client_id not in self.active_clients:
                return
            if float(data.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
                self.mark_client_converged(client_id)
                return
            if round_num == self.current_round:
                self.client_metrics[client_id] = {
                    'num_samples': data['num_samples'],
                    'metrics': data['metrics']
                }
                
                print(f"Received metrics from client {client_id} "
                      f"({len(self.client_metrics)}/{len(self.active_clients)})")
                
                if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                    self.aggregate_metrics()
                    self.continue_training()
        except Exception as e:
            print(f"Server error handling client metrics: {e}")
    
    def distribute_initial_model(self):
        """Distribute initial global model to all clients"""
        # Send training configuration to all clients
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "training_config",
                "config": self.training_config
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        self.current_round = 1
        
        print(f"\n{'='*70}")
        print(f"Distributing Initial Global Model")
        print(f"{'='*70}\n")
        
        # Optionally compress global model
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.get_compression_stats(self.global_weights, compressed_data)
            print(f"Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")

            # Serialize compressed data to JSON-safe base64 string
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            initial_model_message = {
                "message_type": "global_model",
                "round": 0,
                "quantized_data": serialized,
                "model_config": self.model_config
            }
        else:
            # Send initial global model with architecture configuration
            initial_model_message = {
                "message_type": "global_model",
                "round": 0,
                "weights": self.serialize_weights(self.global_weights),
            "model_config": {
                "input_shape": [48, 48, 1],
                "num_classes": 7,
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
                ]
            }
        }
        
        message_json = json.dumps(initial_model_message)
        message_size = len(message_json.encode('utf-8'))
        print(f"Initial model message size: {message_size / 1024:.2f} KB ({message_size} bytes)")
        print(f"Model config: {len(initial_model_message.get('model_config', {}).get('layers', []))} layers")
        
        print("\nPublishing initial model to clients...")
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=message_json,
            properties=pika.BasicProperties(delivery_mode=2)
        )
        log_sent_packet(
            packet_size=message_size,
            peer=EXCHANGE_BROADCAST,
            protocol="AMQP",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="Initial global model distribution"
        )
        
        print("Initial global model sent to all clients")
        
        # Wait for clients to receive and set the initial model
        print("Waiting for clients to receive and build the model...")
        time.sleep(3)
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Signal clients to start training with the global model
        print("Signaling clients to start training...")
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "start_training",
                "round": self.current_round
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print("Start training signal sent successfully\n")
    
    def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] 
                          for update in self.client_updates.values())
        
        # Initialize aggregated weights
        aggregated_weights = []
        
        # Get the structure from first client
        first_client_weights = list(self.client_updates.values())[0]['weights']
        
        # For each layer
        for layer_idx in range(len(first_client_weights)):
            # Weighted average of weights from all clients
            layer_weights = np.zeros_like(first_client_weights[layer_idx])
            
            for client_id, update in self.client_updates.items():
                weight = update['num_samples'] / total_samples
                layer_weights += weight * update['weights'][layer_idx]
            
            aggregated_weights.append(layer_weights)
        
        self.global_weights = aggregated_weights
        
        # Optionally compress before sending
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            # Serialize compressed data to JSON-safe base64 string
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            global_model_message = {
                "message_type": "global_model",
                "round": self.current_round,
                "quantized_data": serialized,
                "model_config": self.model_config
            }
        else:
            global_model_message = {
                "message_type": "global_model",
                "round": self.current_round,
                "weights": self.serialize_weights(self.global_weights)
        }
        
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps(global_model_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        message_size = len(json.dumps(global_model_message).encode('utf-8'))
        log_sent_packet(
            packet_size=message_size,
            peer=EXCHANGE_BROADCAST,
            protocol="AMQP",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="Aggregated global model distribution"
        )
        
        print(f"Aggregated global model from round {self.current_round} sent to all clients")
        
        # Request evaluation from clients
        time.sleep(1)
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "start_evaluation",
                "round": self.current_round
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        log_sent_packet(
            packet_size=len(json.dumps({
                "message_type": "start_evaluation",
                "round": self.current_round
            })),
            peer=EXCHANGE_BROADCAST,
            protocol="AMQP",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="Start evaluation signal"
        )
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics from all clients"""
        print(f"\nAggregating metrics from {len(self.client_metrics)} clients...")
        
        # Calculate total samples
        total_samples = sum(metric['num_samples'] 
                          for metric in self.client_metrics.values())
        
        # Weighted average of metrics
        aggregated_loss = sum(metric['metrics']['loss'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        aggregated_accuracy = sum(metric['metrics']['accuracy'] * metric['num_samples']
                                 for metric in self.client_metrics.values()) / total_samples
        
        # Store metrics
        self.LOSS.append(aggregated_loss)
        self.ACCURACY.append(aggregated_accuracy)
        self.ROUNDS.append(self.current_round)
        
        print(f"\nRound {self.current_round} Aggregated Metrics:")
        print(f"  Loss: {aggregated_loss:.4f}")
        print(f"  Accuracy: {aggregated_accuracy:.4f}\n")
    
    def continue_training(self):
        """Continue to next round or finish training"""
        # Clear updates and metrics for next round
        self.client_updates.clear()
        self.client_metrics.clear()
        
        # Stop only when no active clients remain or max rounds reached (no server-side convergence)
        if len(self.active_clients) == 0:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            self.converged = True
            print("\n" + "="*70)
            print("All clients converged locally. Training complete.")
            print("="*70 + "\n")
            self.send_training_complete()
            self.plot_results()
            self.save_results()
            self.stop()
            return
        
        # Check if more rounds needed
        if self.current_round < self.num_rounds:
            self.current_round += 1
            
            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"{'='*70}\n")
            
            time.sleep(1)
            
            # Send training command for next round
            self.channel.basic_publish(
                exchange=EXCHANGE_BROADCAST,
                routing_key='',
                body=json.dumps({
                    "message_type": "start_training",
                    "round": self.current_round
                }),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            log_sent_packet(
                packet_size=len(json.dumps({
                    "message_type": "start_training",
                    "round": self.current_round
                })),
                peer=EXCHANGE_BROADCAST,
                protocol="AMQP",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info="Start training signal for next round"
            )
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            
            # Send completion signal
            self.send_training_complete()
            self.plot_results()
            self.save_results()
            self.stop()
    
    def check_convergence(self):
        """Check if model has converged based on loss improvement"""
        if len(self.LOSS) == 0:
            return False
        
        current_loss = self.LOSS[-1]
        
        # Check if loss improved by at least the threshold
        improvement = self.best_loss - current_loss
        
        if improvement > CONVERGENCE_THRESHOLD:
            # Significant improvement
            self.best_loss = current_loss
            self.rounds_without_improvement = 0
            print(f"  → Loss improved by {improvement:.6f} (threshold: {CONVERGENCE_THRESHOLD})")
            return False
        else:
            # No significant improvement
            self.rounds_without_improvement += 1
            print(f"  → No significant improvement (improvement: {improvement:.6f}, threshold: {CONVERGENCE_THRESHOLD})")
            print(f"  → Rounds without improvement: {self.rounds_without_improvement}/{CONVERGENCE_PATIENCE}")
            
            if self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
                return True
            return False
    
    def send_training_complete(self):
        """Send training complete signal to all clients"""
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "training_complete"
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        log_sent_packet(
            packet_size=len(json.dumps({
                "message_type": "training_complete"
            })),
            peer=EXCHANGE_BROADCAST,
            protocol="AMQP",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="Training complete signal"
        )
        print("Training complete signal sent to all clients")
    
    def plot_results(self):
        """Plot training metrics"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.ROUNDS, self.LOSS, 'b-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Training Loss (Emotion Recognition)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.ROUNDS, self.ACCURACY, 'g-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy (Emotion Recognition)')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'amqp_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to {results_dir / 'amqp_training_metrics.png'}")
        plt.show()
        
        print("\nPlot closed. Training complete.")
    
    def save_results(self):
        """Save training results to JSON"""
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'rounds': self.ROUNDS,
            'loss': self.LOSS,
            'accuracy': self.ACCURACY,
            'summary': {
                'total_rounds': len(self.ROUNDS),
                'num_clients': self.num_clients,
                'final_loss': self.LOSS[-1] if self.LOSS else None,
                'final_accuracy': self.ACCURACY[-1] if self.ACCURACY else None,
                'convergence_time_seconds': self.convergence_time,
                'convergence_time_minutes': self.convergence_time / 60 if self.convergence_time else None,
                'converged': self.converged,
                'convergence_threshold': CONVERGENCE_THRESHOLD,
                'convergence_patience': CONVERGENCE_PATIENCE
            }
        }
        
        results_file = results_dir / 'amqp_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Training results saved to {results_file}")
    
    def start(self):
        """Start consuming messages"""
        print("="*70)
        print("Starting Federated Learning Server (AMQP) - Emotion Recognition")
        print(f"Broker: {AMQP_HOST}:{AMQP_PORT}")
        print(f"Number of Clients: {self.num_clients}")
        print(f"Number of Rounds: {self.num_rounds}")
        print("="*70)
        print("\nWaiting for clients to register...\n")
        
        self.consuming = True
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("\n\nServer interrupted by user")
            self.stop()
    
    def stop(self):
        """Stop consuming and close connection"""
        if self.consuming:
            self.channel.stop_consuming()
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        print("Server disconnected")


if __name__ == "__main__":
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    
    try:
        server.connect()
        server.start()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.stop()
