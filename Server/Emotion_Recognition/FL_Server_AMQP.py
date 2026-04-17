import io
import os
import sys
# Server uses CPU only (aggregation is numpy-only); saves GPU memory for clients
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import json
import pickle
import base64
import time
import pika
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
AMQP_HOST = os.getenv("AMQP_HOST", "localhost")
AMQP_PORT = int(os.getenv("AMQP_PORT", "5672"))
AMQP_USER = os.getenv("AMQP_USER", "guest")
AMQP_PASSWORD = os.getenv("AMQP_PASSWORD", "guest")

# AMQP delivery mode: 2=persistent (default), 1=non-persistent (for diagnostics/experiments)
try:
    AMQP_DELIVERY_MODE = int(os.getenv("AMQP_DELIVERY_MODE", "2"))
except (TypeError, ValueError):
    AMQP_DELIVERY_MODE = 2
if AMQP_DELIVERY_MODE not in (1, 2):
    AMQP_DELIVERY_MODE = 2
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence
from fl_termination_env import stop_on_client_convergence

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

AMQP_MAX_FRAME_BYTES = 128 * 1024
AMQP_CHUNK_PAYLOAD_BYTES = 96 * 1024


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
        self.ROUND_TIMES = []
        self.BATTERY_CONSUMPTION = []
        self.BATTERY_MODEL_CONSUMPTION = []
        self.round_start_time = None

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
        
        # Global model is initialized after connect() so RabbitMQ queues exist before clients send
        self.global_weights = None
        self.model_config = None
        
        # Training configuration
        # Training configuration broadcast to AMQP clients
        self.training_config = {
            "batch_size": int(os.getenv("BATCH_SIZE", "16")),
            "local_epochs": 20
        }
        
        # AMQP connection
        self.connection = None
        self.channel = None
        self.consuming = False
        self._model_update_chunk_buffers = {}
    
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
        buf = io.BytesIO()
        np.savez(buf, *weights)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from AMQP."""
        buf = io.BytesIO(base64.b64decode(encoded_weights.encode('utf-8')))
        try:
            loaded = np.load(buf, allow_pickle=False)
            weights = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]
        except Exception:
            buf.seek(0)
            weights = pickle.loads(buf.read())
        return weights

    def _chunk_model_payload(self, model_message):
        payload_key = "quantized_data" if "quantized_data" in model_message else "weights"
        payload_text = model_message[payload_key]
        if not isinstance(payload_text, str):
            raise TypeError(f"Expected string payload for {payload_key}, got {type(payload_text).__name__}")

        chunks = [
            payload_text[i:i + AMQP_CHUNK_PAYLOAD_BYTES]
            for i in range(0, len(payload_text), AMQP_CHUNK_PAYLOAD_BYTES)
        ] or [payload_text]

        total_chunks = len(chunks)
        chunk_messages = []
        for chunk_index, chunk_data in enumerate(chunks):
            chunk_msg = {
                "message_type": "global_model_chunk",
                "type": "global_model_chunk",
                "round": model_message["round"],
                "payload_key": payload_key,
                "payload_chunk": chunk_data,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
            }
            if chunk_index == 0 and "model_config" in model_message:
                chunk_msg["model_config"] = model_message["model_config"]
            chunk_messages.append(chunk_msg)
        return chunk_messages

    def _publish_global_model_with_chunking(self, model_message, extra_info):
        message_json = json.dumps(model_message)
        message_size = len(message_json.encode("utf-8"))

        if message_size <= AMQP_MAX_FRAME_BYTES:
            self.channel.basic_publish(
                exchange=EXCHANGE_BROADCAST,
                routing_key='',
                body=message_json,
                properties=pika.BasicProperties(delivery_mode=AMQP_DELIVERY_MODE)
            )
            log_sent_packet(
                packet_size=message_size,
                peer=EXCHANGE_BROADCAST,
                protocol="AMQP",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info=extra_info
            )
            return message_size

        chunks = self._chunk_model_payload(model_message)
        print(f"Chunking AMQP global model: {message_size} bytes total, {len(chunks)} chunks")
        sent_bytes = 0
        for chunk in chunks:
            chunk_payload = json.dumps(chunk)
            chunk_size = len(chunk_payload.encode("utf-8"))
            if chunk_size > AMQP_MAX_FRAME_BYTES:
                raise ValueError(
                    f"AMQP global-model chunk exceeds 128KB: {chunk_size} bytes "
                    f"(chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']})"
                )
            self.channel.basic_publish(
                exchange=EXCHANGE_BROADCAST,
                routing_key='',
                body=chunk_payload,
                properties=pika.BasicProperties(delivery_mode=AMQP_DELIVERY_MODE)
            )
            sent_bytes += chunk_size
            log_sent_packet(
                packet_size=chunk_size,
                peer=EXCHANGE_BROADCAST,
                protocol="AMQP",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info=f"{extra_info} chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']}"
            )
        return sent_bytes

    def _assemble_model_update_chunk(self, data):
        try:
            client_id = int(data["client_id"])
            round_num = int(data["round"])
            payload_key = data["payload_key"]
            chunk_index = int(data.get("chunk_index", 0))
            total_chunks = int(data.get("total_chunks", 1))
            payload_chunk = data.get("payload_chunk", "")
        except Exception as e:
            print(f"Server invalid AMQP update chunk metadata: {e}")
            return None

        if total_chunks <= 1:
            assembled = {
                "client_id": client_id,
                "round": round_num,
                payload_key: payload_chunk,
                "num_samples": data.get("num_samples", 0),
                "metrics": data.get("metrics", {}),
            }
            if "diagnostic_send_start_ts" in data:
                assembled["diagnostic_send_start_ts"] = data.get("diagnostic_send_start_ts")
            return assembled

        chunk_key = (client_id, round_num, payload_key)
        entry = self._model_update_chunk_buffers.setdefault(
            chunk_key,
            {
                "chunks": {},
                "total_chunks": total_chunks,
                "num_samples": data.get("num_samples", 0),
                "metrics": data.get("metrics", {}),
                "diagnostic_send_start_ts": data.get("diagnostic_send_start_ts"),
                "updated_at": time.time(),
            }
        )

        if chunk_index == 0:
            entry["num_samples"] = data.get("num_samples", entry["num_samples"])
            entry["metrics"] = data.get("metrics", entry["metrics"])
            if "diagnostic_send_start_ts" in data:
                entry["diagnostic_send_start_ts"] = data.get("diagnostic_send_start_ts")

        if chunk_index not in entry["chunks"]:
            entry["chunks"][chunk_index] = payload_chunk
        entry["updated_at"] = time.time()

        if len(entry["chunks"]) < total_chunks:
            return None

        assembled_payload = "".join(entry["chunks"].get(i, "") for i in range(total_chunks))
        assembled = {
            "client_id": client_id,
            "round": round_num,
            payload_key: assembled_payload,
            "num_samples": entry["num_samples"],
            "metrics": entry["metrics"],
        }
        if entry.get("diagnostic_send_start_ts") is not None:
            assembled["diagnostic_send_start_ts"] = entry["diagnostic_send_start_ts"]

        self._model_update_chunk_buffers.pop(chunk_key, None)
        print(
            f"Reassembled AMQP model update from client {client_id} for round {round_num} "
            f"from {total_chunks} chunks"
        )
        return assembled
    
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
                    blocked_connection_timeout=600,  # Aligned with heartbeat
                    frame_max=AMQP_MAX_FRAME_BYTES  # Realistic max payload: AMQP 128 KB
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
        if not stop_on_client_convergence():
            # Fixed-round mode: ignore client-local convergence removal/disconnect.
            print(f"Ignoring convergence signal from client {client_id} (STOP_ON_CLIENT_CONVERGENCE=false)")
            return
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
            msg_type = data.get("message_type") or data.get("type")
            if msg_type in ("model_update_chunk", "update_chunk"):
                data = self._assemble_model_update_chunk(data)
                if data is None:
                    return
            client_id = data['client_id']
            round_num = data['round']
            
            if client_id not in self.active_clients:
                return
            if stop_on_client_convergence() and float(data.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
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
                    # Keep quantized end-to-end: do NOT decompress/dequantize on server.
                    self.client_updates[client_id] = {
                        'compressed_data': compressed_update,
                        'num_samples': data['num_samples'],
                        'metrics': data['metrics']
                    }
                    print(f"Received quantized update from client {client_id} (kept quantized)")
                    weights = None
                else:
                    weights = self.deserialize_weights(data['weights'])
                
                if recv_start_cpu is not None:
                    O_recv = time.perf_counter() - recv_start_cpu
                    recv_end_ts = time.time()
                    send_start_ts = data.get("diagnostic_send_start_ts", recv_end_ts)
                    print(f"FL_DIAG client_id={client_id} O_recv={O_recv:.9f} recv_end_ts={recv_end_ts:.9f} send_start_ts={send_start_ts:.9f}")
                
                if 'compressed_data' not in data or self.quantization_handler is None:
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
            if stop_on_client_convergence() and float(data.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
                self.mark_client_converged(client_id)
                return
            if round_num == self.current_round:
                m = data.get('metrics', {})
                self.client_metrics[client_id] = {
                    'num_samples': data['num_samples'],
                    'metrics': data['metrics'],
                    'battery_soc': float(m.get('battery_soc', 1.0)),
                    'round_time_sec': float(m.get('round_time_sec', 0.0)),
                    'cumulative_energy_j': float(m.get('cumulative_energy_j', 0.0)),
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
            properties=pika.BasicProperties(delivery_mode=AMQP_DELIVERY_MODE)
        )
        
        self.current_round = 1
        self.round_start_time = time.time()
        
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
        sent_bytes = self._publish_global_model_with_chunking(
            initial_model_message,
            extra_info="Initial global model distribution"
        )
        
        print(f"Initial global model sent to all clients ({sent_bytes} bytes transmitted)")
        
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
            properties=pika.BasicProperties(delivery_mode=AMQP_DELIVERY_MODE)
        )
        print("Start training signal sent successfully\n")
    
    def send_current_model_to_client(self, client_id):
        """Send current global model to a single client (e.g. late joiner). Uses broadcast so the client receives it."""
        if self.global_weights is None or self.model_config is None:
            return
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            msg = {
                "message_type": "global_model",
                "round": self.current_round,
                "quantized_data": serialized,
                "model_config": self.model_config
            }
        else:
            msg = {
                "message_type": "global_model",
                "round": self.current_round,
                "weights": self.serialize_weights(self.global_weights),
                "model_config": self.model_config
            }
        self._publish_global_model_with_chunking(
            msg,
            extra_info=f"Late-join model broadcast (for client {client_id})"
        )
        print(f"Current global model (round {self.current_round}) sent to client {client_id} (broadcast)")
    
    def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")

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

            serialized = base64.b64encode(pickle.dumps(self.global_compressed)).decode('utf-8')
            global_model_message = {
                "message_type": "global_model",
                "round": self.current_round,
                "quantized_data": serialized,
                "model_config": self.model_config
            }

            self._publish_global_model_with_chunking(
                global_model_message,
                extra_info="Aggregated global model distribution (dequantize→FedAvg→requantize)"
            )

            print(f"Aggregated global model from round {self.current_round} sent to all clients (dequantize→FedAvg→requantize)")

            time.sleep(1)
            self.channel.basic_publish(
                exchange=EXCHANGE_BROADCAST,
                routing_key='',
                body=json.dumps({"message_type": "start_evaluation", "round": self.current_round}),
                properties=pika.BasicProperties(delivery_mode=AMQP_DELIVERY_MODE)
            )
            log_sent_packet(
                packet_size=len(json.dumps({"message_type": "start_evaluation", "round": self.current_round})),
                peer=EXCHANGE_BROADCAST,
                protocol="AMQP",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info="Start evaluation signal"
            )
            return
        
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
        
        message_size = self._publish_global_model_with_chunking(
            global_model_message,
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
            properties=pika.BasicProperties(delivery_mode=AMQP_DELIVERY_MODE)
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
        if getattr(self, 'round_start_time', None) is not None:
            self.ROUND_TIMES.append(time.time() - self.round_start_time)
        socs = [m.get('battery_soc', 1.0) for m in self.client_metrics.values()]
        self.BATTERY_CONSUMPTION.append(1.0 - (sum(socs) / len(socs) if socs else 1.0))
        self.BATTERY_MODEL_CONSUMPTION.append(avg_battery_model_drain_fraction(self.client_metrics))
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
            self.round_start_time = time.time()
            
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
                properties=pika.BasicProperties(delivery_mode=AMQP_DELIVERY_MODE)
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
            properties=pika.BasicProperties(delivery_mode=AMQP_DELIVERY_MODE)
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
        """Plot battery consumption, round/convergence time, and loss/accuracy."""
        results_dir = get_experiment_results_dir("emotion", "amqp")
        rounds = self.ROUNDS
        n = len(rounds)
        conv_time = self.convergence_time if self.convergence_time is not None else (time.time() - self.start_time if self.start_time else 0)
        # 1) Battery
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        bc = (self.BATTERY_CONSUMPTION + [0.0] * max(0, n - len(self.BATTERY_CONSUMPTION)))[:n] if getattr(self, 'BATTERY_CONSUMPTION', []) else [0.0] * n
        if bc:
            ax1.plot(rounds, [c * 100 for c in bc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round'); ax1.set_ylabel('Battery consumption (%)'); ax1.set_title('AMQP: Battery consumption till end of FL training'); ax1.grid(True, alpha=0.3)
        fig1.tight_layout(); fig1.savefig(results_dir / 'amqp_battery_consumption.png', dpi=300, bbox_inches='tight'); plt.close(fig1)
        # 2) Time per round and convergence
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        rt = (self.ROUND_TIMES + [0.0] * max(0, n - len(self.ROUND_TIMES)))[:n] if getattr(self, 'ROUND_TIMES', []) else [0.0] * n
        if rt:
            ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2, label=f'Total convergence: {conv_time:.1f} s')
        ax2.set_xlabel('Round'); ax2.set_ylabel('Time (s)'); ax2.set_title('AMQP: Time per round and convergence time'); ax2.legend(); ax2.grid(True, alpha=0.3)
        fig2.tight_layout(); fig2.savefig(results_dir / 'amqp_round_and_convergence_time.png', dpi=300, bbox_inches='tight'); plt.close(fig2)
        # 3) Loss and Accuracy
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        ax3a.plot(rounds, self.LOSS, 'b-', marker='o'); ax3a.set_xlabel('Round'); ax3a.set_ylabel('Loss'); ax3a.set_title('AMQP: Loss over Rounds'); ax3a.grid(True)
        ax3b.plot(rounds, self.ACCURACY, 'g-', marker='o'); ax3b.set_xlabel('Round'); ax3b.set_ylabel('Accuracy'); ax3b.set_title('AMQP: Accuracy over Rounds'); ax3b.grid(True)
        fig3.tight_layout()
        fig3.savefig(results_dir / 'amqp_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Training metrics plot saved to {results_dir / 'amqp_training_metrics.png'}")
        if not os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            plt.show(block=False)
        print("\nPlot closed. Training complete.")
    
    def save_results(self):
        """Save training results to JSON"""
        results_dir = get_experiment_results_dir("emotion", "amqp")
        
        results = {
            'rounds': self.ROUNDS,
            'loss': self.LOSS,
            'accuracy': self.ACCURACY,
            'round_times_seconds': getattr(self, 'ROUND_TIMES', []),
            'battery_consumption': getattr(self, 'BATTERY_CONSUMPTION', []),
            'battery_model_consumption': getattr(self, 'BATTERY_MODEL_CONSUMPTION', []),
            'battery_model_consumption_source': 'client_battery_model',
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
        # Connect first so registration queue exists before clients send (avoids lost registrations)
        server.connect()
        server.initialize_global_model()
        server.start()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.stop()
