import os
import sys
# Server uses CPU only (aggregation is numpy-only); saves GPU memory for clients
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import io
import pandas as pd
import json
import pickle
import base64
import time
import paho.mqtt.client as mqtt
from typing import List, Dict
import matplotlib.pyplot as plt
from pathlib import Path


# Detect Docker environment and set project root accordingly
if os.path.exists('/app'):
    # Likely running in Docker, code is under /app
    project_root = '/app'
else:
    # Local development: go up two levels from this file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# packet_logger lives in scripts/utilities (Docker: /app/scripts/utilities, local: project_root/scripts/utilities)
_utilities_path = os.path.join(project_root, 'scripts', 'utilities')
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)

print(f"Project root set to: {project_root}")
from packet_logger import init_db, log_sent_packet, log_received_packet
from battery_results_agg import avg_battery_model_drain_fraction
try:
    from experiment_results_path import get_experiment_results_dir
except ModuleNotFoundError:
    from scripts.utilities.experiment_results_path import get_experiment_results_dir

try:
    from fl_training_results_cpu_memory import (
        merge_cpu_memory_into_results,
        plot_cpu_memory_for_server_rounds,
    )
except ModuleNotFoundError:
    from scripts.utilities.fl_training_results_cpu_memory import (
        merge_cpu_memory_into_results,
        plot_cpu_memory_for_server_rounds,
    )

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
# Auto-detect environment: Docker (/app exists) or local
MQTT_BROKER = os.getenv("MQTT_BROKER", 'mqtt-broker' if os.path.exists('/app') else 'localhost')
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))  # MQTT broker port
# Keepalive (seconds). Long value avoids client rc=16 in very_poor / diagnostic pipeline. Max 65535.
_def_keepalive = 3600 if os.getenv("FL_DIAGNOSTIC_PIPELINE") == "1" else 600
MQTT_KEEPALIVE_SEC = min(65535, max(10, int(os.getenv("MQTT_KEEPALIVE_SEC", str(_def_keepalive)))))
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence
from fl_termination_env import stop_on_client_convergence

# Convergence Settings (primary stopping criterion)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))  # Loss improvement threshold
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))  # Rounds to wait for improvement
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))  # Minimum rounds before checking convergence

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_REGISTER = "fl/client_register"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"
TOPIC_TRAINING_COMPLETE = "fl/training_complete"

MQTT_MAX_PAYLOAD_BYTES = 128 * 1024
MQTT_CHUNK_PAYLOAD_BYTES = 96 * 1024


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
        
        # Track which clients are expected to send updates for current round
        self.round_participants = set()
        
        # Metrics storage (for classification)
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        # Per-round time (sec) and battery consumption (0–1, from client-reported SoC)
        self.ROUND_TIMES = []
        self.BATTERY_CONSUMPTION = []
        # Cumulative drain fraction from client BatteryModel (energy / capacity), see battery_results_agg
        self.BATTERY_MODEL_CONSUMPTION = []

        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.training_started = False  # Track if training has begun
        self.training_complete = False
        self.start_time = None
        self.convergence_time = None
        
        # Training timeout tracking (prevent waiting forever for stuck clients). Long in diagnostic/very_poor.
        self.round_start_time = None
        _def_training_timeout = 3600 if os.getenv("FL_DIAGNOSTIC_PIPELINE") == "1" else 600
        self.training_timeout = int(os.getenv("TRAINING_TIMEOUT", str(_def_training_timeout)))
        
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
        
        # Initialize packet logging database
        init_db()
        
        # Initialize global model
        self.initialize_global_model()
        
        # Training configuration
        # Training configuration broadcast to MQTT clients
        self.training_config = {
            "batch_size": int(os.getenv("BATCH_SIZE", "16")),
            "local_epochs": 20
        }
        
        # Unique MQTT client_id per process to avoid broker "already connected" (rc=7) reconnect loop
        _mqtt_client_id = f"fl_server_{os.getpid()}"
        self.mqtt_client = mqtt.Client(client_id=_mqtt_client_id, protocol=mqtt.MQTTv311, clean_session=True)
        # Realistic max payload: MQTT 128 KB
        self.mqtt_client._max_packet_size = MQTT_MAX_PAYLOAD_BYTES
        # Long keepalive to avoid client rc=16 in very_poor / diagnostic pipeline
        self.mqtt_client.keepalive = MQTT_KEEPALIVE_SEC
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self._model_update_chunk_buffers = {}


    
    def update_client_count(self, new_count):
        """Update expected client count when new clients join"""
        if new_count > self.num_clients and new_count <= self.max_clients:
            old_count = self.num_clients
            self.num_clients = new_count
            print(f"[DYNAMIC] Updated client count: {old_count} -> {new_count}")
            return True
        return False
    
    def handle_late_joining_client(self, client_id):
        """Handle a client joining after training has started"""
        if client_id not in self.registered_clients:
            self.registered_clients.add(client_id)
            print(f"[LATE JOIN] Client {client_id} joined after training started")
            
            # Update client count if needed
            if len(self.registered_clients) > self.num_clients:
                self.update_client_count(len(self.registered_clients))
            
            # Send current global model to late-joining client
            if self.global_weights is not None:
                self.send_global_model_to_client(client_id)
    
    def get_active_clients(self):
        """Get list of currently active (non-converged) clients; after one converges we proceed with the rest."""
        return list(self.active_clients)
    
    def adaptive_wait_for_clients(self, client_dict, timeout=300):
        """
        Adaptive waiting for client responses
        - Waits for minimum clients first
        - Then waits additional time for late-joining clients
        - Returns when all registered clients respond or timeout
        """
        import time
        start_time = time.time()
        min_received = len(client_dict) >= self.min_clients
        all_registered_received = len(client_dict) >= len(self.registered_clients)
        
        while not all_registered_received and (time.time() - start_time) < timeout:
            time.sleep(0.5)
            min_received = len(client_dict) >= self.min_clients
            all_registered_received = len(client_dict) >= len(self.registered_clients)
            
            # If we have minimum and haven't seen new clients for 10 seconds, proceed
            if min_received and (time.time() - start_time) > 10:
                break
        
        return len(client_dict) >= self.min_clients

    def initialize_global_model(self):
        """Initialize the global model structure (CNN for Emotion Recognition)"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers.schedules import ExponentialDecay
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
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
        model.add(Input(shape=(48, 48, 1)))  # Input layer
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
        
        print("\nGlobal model initialized with random weights")
        print(f"Model architecture: CNN for Emotion Recognition (7 classes)")
        print(f"Number of weight layers: {len(self.global_weights)}")
    
    def serialize_weights(self, weights):
        """Serialize model weights for MQTT transmission"""
        buf = io.BytesIO()
        np.savez(buf, *weights)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        return encoded

    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from MQTT.

        Uses numpy's native .npz format instead of pickle so that weights
        serialized on NumPy 2.x (numpy._core) can be loaded on NumPy 1.x
        (numpy.core) and vice-versa.
        """
        buf = io.BytesIO(base64.b64decode(encoded_weights.encode('utf-8')))
        loaded = np.load(buf, allow_pickle=False)
        weights = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]
        return weights

    def _chunk_model_payload(self, model_message):
        payload_key = "quantized_data" if "quantized_data" in model_message else "weights"
        payload_text = model_message[payload_key]
        if not isinstance(payload_text, str):
            raise TypeError(f"Expected string payload for {payload_key}, got {type(payload_text).__name__}")

        chunks = [
            payload_text[i:i + MQTT_CHUNK_PAYLOAD_BYTES]
            for i in range(0, len(payload_text), MQTT_CHUNK_PAYLOAD_BYTES)
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
            if chunk_index == 0:
                if "model_config" in model_message:
                    chunk_msg["model_config"] = model_message["model_config"]
                if "training_config" in model_message:
                    chunk_msg["training_config"] = model_message["training_config"]
            chunk_messages.append(chunk_msg)
        return chunk_messages

    def _publish_global_model_with_chunking(self, model_message, extra_info):
        """Publish global model while strictly respecting 128KB MQTT payload limit."""
        payload = json.dumps(model_message)
        payload_bytes = len(payload.encode("utf-8"))

        if payload_bytes <= MQTT_MAX_PAYLOAD_BYTES:
            result = self.mqtt_client.publish(TOPIC_GLOBAL_MODEL, payload, qos=1)
            log_sent_packet(
                packet_size=len(payload),
                peer=TOPIC_GLOBAL_MODEL,
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info=extra_info
            )
            return result, payload_bytes

        chunks = self._chunk_model_payload(model_message)
        print(
            f"Chunking global model for MQTT: {payload_bytes} bytes total, "
            f"{len(chunks)} chunks"
        )

        last_result = None
        total_sent = 0
        for chunk in chunks:
            chunk_payload = json.dumps(chunk)
            chunk_size = len(chunk_payload.encode("utf-8"))
            if chunk_size > MQTT_MAX_PAYLOAD_BYTES:
                raise ValueError(
                    f"MQTT global-model chunk exceeds 128KB: {chunk_size} bytes "
                    f"(chunk {chunk['chunk_index'] + 1}/{chunk['total_chunks']})"
                )
            last_result = self.mqtt_client.publish(TOPIC_GLOBAL_MODEL, chunk_payload, qos=1)
            total_sent += chunk_size
            log_sent_packet(
                packet_size=chunk_size,
                peer=TOPIC_GLOBAL_MODEL,
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info=(
                    f"{extra_info} chunk "
                    f"{chunk['chunk_index'] + 1}/{chunk['total_chunks']}"
                )
            )

        return last_result, total_sent

    def _assemble_model_update_chunk(self, data):
        """Buffer and reassemble chunked model updates from clients."""
        try:
            client_id = int(data["client_id"])
            round_num = int(data["round"])
            payload_key = data["payload_key"]
            chunk_index = int(data.get("chunk_index", 0))
            total_chunks = int(data.get("total_chunks", 1))
            payload_chunk = data.get("payload_chunk", "")
        except Exception as e:
            print(f"Server invalid model-update chunk metadata: {e}")
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
            f"Reassembled model update from client {client_id} for round {round_num} "
            f"from {total_chunks} chunks"
        )
        return assembled
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            # Only log first connection, not reconnections
            if not hasattr(self, '_connected_once'):
                print("Server connected to MQTT broker")
                self._connected_once = True
            
            # Subscribe to client topics with QoS 1 for reliable delivery
            self.mqtt_client.subscribe(TOPIC_CLIENT_REGISTER, qos=1)
            self.mqtt_client.subscribe("fl/client/+/update", qos=1)
            self.mqtt_client.subscribe("fl/client/+/metrics", qos=1)
        else:
            print(f"Server failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            log_received_packet(
                packet_size=len(msg.payload),
                peer=msg.topic,
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info="Received message"
            )
        except Exception as e:
            print(f"Server error logging received packet: {e}")

        try:
            if msg.topic == TOPIC_CLIENT_REGISTER:
                self.handle_client_registration(msg.payload)
            elif "/update" in msg.topic:
                self.handle_client_update(msg.payload)
            elif "/metrics" in msg.topic:
                self.handle_client_metrics(msg.payload)
        except Exception as e:
            print(f"Server error handling message: {e}")
    
    def handle_client_registration(self, payload):
        """Handle client registration"""
        data = json.loads(payload.decode())
        client_id = data['client_id']
        
        # Check if client is already registered
        if client_id in self.registered_clients:
            print(f"Client {client_id} re-registered (already known)")
            return
        
        self.registered_clients.add(client_id)
        self.active_clients.add(client_id)
        print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))
        
        # Check if this is a late-joining client (training already started)
        if self.training_started:
            print(f"[LATE JOIN] Client {client_id} joining during training (round {self.current_round})")
            # Send current global model to late-joining client
            if self.global_weights is not None:
                self.send_current_model_to_client(client_id)
            return
        
        # If minimum clients registered and training not started, begin training
        if len(self.registered_clients) >= self.min_clients and not self.training_started:
            print("\nMinimum clients registered. Distributing initial global model...\n")
            time.sleep(2)  # Give clients time to be ready
            self.distribute_initial_model()
            # Record training start time
            self.start_time = time.time()
            self.training_started = True
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def send_current_model_to_client(self, client_id):
        """Send current global model to a late-joining client"""
        try:
            print(f"📤 Sending current global model (round {self.current_round}) to late-joining client {client_id}")
            
            # Prepare payload with current model
            if self.quantization_handler is not None:
                compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
                serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
                model_message = {
                    'round': self.current_round,
                    'quantized_data': serialized,
                    'model_config': self.model_config,
                    'training_config': self.training_config
                }
            else:
                model_message = {
                    'round': self.current_round,
                    'weights': self.serialize_weights(self.global_weights),
                    'model_config': self.model_config,
                    'training_config': self.training_config
                }
            
            # Broadcast to all clients on general topic (late-joiner will receive it)
            # This is simpler than client-specific topics and works with our generic model handling
            result, sent_bytes = self._publish_global_model_with_chunking(
                model_message,
                extra_info=f"Late-join model broadcast (for client {client_id})"
            )
            
            log_sent_packet(
                packet_size=sent_bytes,
                peer=f"client_{client_id}",
                protocol="MQTT",
                round=self.current_round,
                extra_info=f"Late-join model broadcast (for client {client_id})"
            )
            
            print(f"✅ Current model (round {self.current_round}) broadcast to late-joining client {client_id}")
            
        except Exception as e:
            print(f"❌ Error sending current model to client {client_id}: {e}")
    
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
            self.round_participants.discard(client_id)
            print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
            if not self.active_clients:
                self.converged = True
                print("All clients converged. Ending training.")
                self.convergence_time = time.time() - self.start_time if self.start_time else 0
                self._send_training_complete_and_exit()
                return
            # Re-check: remaining active clients may have already sent metrics/updates
            if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                self.aggregate_metrics()
                self.continue_training()
                return
            active_in_round = self.round_participants & self.active_clients
            if len(self.client_updates) >= len(active_in_round) and len(active_in_round) > 0:
                self.aggregate_models()
    
    def handle_client_update(self, payload):
        """Handle model update from client"""
        recv_start_cpu = time.perf_counter() if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1" else None
        data = json.loads(payload.decode())

        msg_type = data.get("message_type") or data.get("type")
        if msg_type in ("model_update_chunk", "update_chunk"):
            data = self._assemble_model_update_chunk(data)
            if data is None:
                return

        client_id = data['client_id']
        round_num = data['round']
        
        if client_id not in self.active_clients:
            return
        converged = stop_on_client_convergence() and float(data.get('metrics', {}).get('client_converged', 0.0)) >= 1.0
        if converged:
            self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            # Check if update is compressed
            if 'compressed_data' in data and self.quantization_handler is not None:
                # Keep quantized end-to-end: do NOT decompress/dequantize on server.
                compressed_update = data['compressed_data']
                if isinstance(compressed_update, str):
                    try:
                        compressed_update = pickle.loads(base64.b64decode(compressed_update.encode('utf-8')))
                    except Exception as e:
                        print(f"Server error decoding compressed_data from client {client_id}: {e}")
                self.client_updates[client_id] = {
                    'compressed_data': compressed_update,
                    'num_samples': data['num_samples'],
                    'metrics': data['metrics']
                }
                print(f"Received quantized update from client {client_id} (kept quantized)")
            else:
                # Standard deserialization
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
                  f"({len(self.client_updates)}/{len(self.round_participants)})")
            
            # If all active participants sent updates, aggregate
            active_in_round = self.round_participants & self.active_clients
            if len(self.client_updates) >= len(active_in_round) and len(active_in_round) > 0:
                self.aggregate_models()
    
    def handle_client_metrics(self, payload):
        """Handle evaluation metrics from client"""
        data = json.loads(payload.decode())
        client_id = data['client_id']
        round_num = data['round']
        
        if client_id not in self.active_clients:
            return
        converged = stop_on_client_convergence() and float(data.get('metrics', {}).get('client_converged', 0.0)) >= 1.0
        if converged:
            self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            self.client_metrics[client_id] = {
                'num_samples': data['num_samples'],
                'metrics': data['metrics']
            }
            
            print(f"Received metrics from client {client_id} "
                  f"({len(self.client_metrics)}/{len(self.active_clients)})")
            
            # If all active clients sent metrics, aggregate and continue
            if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                self.aggregate_metrics()
                self.continue_training()
    
    def distribute_initial_model(self):
        """Distribute initial global model architecture and weights to all clients"""
        # Send training configuration to all clients
        self.mqtt_client.publish(TOPIC_TRAINING_CONFIG, 
                            json.dumps(self.training_config),
                            qos=1)
        log_sent_packet(
            packet_size=len(json.dumps(self.training_config)),
            peer=TOPIC_TRAINING_CONFIG,  # or client_id/server_id as appropriate
            protocol="MQTT",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="any additional info"
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
                "round": 0,
                "quantized_data": serialized,
                "model_config": self.model_config
            }
        else:
            # Send initial global model with architecture configuration
            initial_model_message = {
                "round": 0,
                "weights": self.serialize_weights(self.global_weights),
                "model_config": self.model_config  # Include model architecture
            }
        
        message_json = json.dumps(initial_model_message)
        message_size = len(message_json.encode('utf-8'))
        
        print(f"Initial model message size: {message_size / 1024:.2f} KB ({message_size} bytes)")
        print(f"Model config: {len(self.model_config['layers'])} layers, {self.model_config['num_classes']} classes")
        
        # Publish with QoS 1 so clients get at-least-once delivery (large payload)
        print("\nPublishing initial model to clients...")
        result, sent_bytes = self._publish_global_model_with_chunking(
            initial_model_message,
            extra_info="Initial global model distribution"
        )
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"  Initial global model sent successfully (QoS 1), bytes sent: {sent_bytes}")
        else:
            print(f"  FAILED to send initial model (return code: {result.rc})")

        # Give clients time to receive and build the model (large payload + model init)
        print("\nWaiting for clients to receive and build the model...")
        time.sleep(10)
        
        # Capture which active clients will participate in this round
        self.round_participants = self.active_clients.copy()
        self.round_start_time = time.time()
        print(f"Round {self.current_round} participants: {sorted(list(self.round_participants))}")
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Signal clients to start training with the global model
        print("Signaling clients to start training...")
        result = self.mqtt_client.publish(TOPIC_START_TRAINING,
                                json.dumps({"round": self.current_round}),
                                qos=1)
        log_sent_packet(
            packet_size=len(json.dumps({"round": self.current_round})),
            peer=TOPIC_START_TRAINING,  # or client_id/server_id as appropriate
            protocol="MQTT",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="Start training signal"
        )
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print("Start training signal sent successfully\n")
        else:
            print(f"ERROR: Failed to send start training signal (return code: {result.rc})\n")
    
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
                "round": self.current_round,
                "quantized_data": serialized,
                "model_config": self.model_config
            }

            print(f"Publishing aggregated global model for round {self.current_round} (dequantize→FedAvg→requantize)...")
            for i in range(3):
                result, _ = self._publish_global_model_with_chunking(
                    global_model_message,
                    extra_info="Aggregated global model distribution (dequantize→FedAvg→requantize)"
                )
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print(f"  Attempt {i+1}/3: Aggregated model sent")
                    break
                else:
                    print(f"  Attempt {i+1}/3: FAILED (rc={result.rc})")
                    time.sleep(0.5)

            print(f"Aggregated global model from round {self.current_round} sent to all clients (dequantize→FedAvg→requantize)")

            time.sleep(2)
            print("Requesting client evaluation...")
            self.mqtt_client.publish(TOPIC_START_EVALUATION, json.dumps({"round": self.current_round}), qos=1)
            log_sent_packet(
                packet_size=len(json.dumps({"round": self.current_round})),
                peer=TOPIC_START_EVALUATION,
                protocol="MQTT",
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
                "round": self.current_round,
                "quantized_data": serialized,
                "model_config": self.model_config  # Always include for late-joiners
            }
        else:
            # Send global model to all clients
            global_model_message = {
                "round": self.current_round,
                "weights": self.serialize_weights(self.global_weights),
                "model_config": self.model_config  # Always include for late-joiners
            }
        
        # Publish aggregated model (QoS 1 for at-least-once) and avoid duplicates
        print(f"Publishing aggregated model for round {self.current_round}...")
        for i in range(3):
            result, _ = self._publish_global_model_with_chunking(
                global_model_message,
                extra_info="Aggregated global model distribution"
            )
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"  Attempt {i+1}/3: Aggregated model sent")
                break
            else:
                print(f"  Attempt {i+1}/3: FAILED (rc={result.rc})")
                time.sleep(0.5)
        
        print(f"Aggregated global model from round {self.current_round} sent to all clients")
        
        # Request evaluation from clients
        time.sleep(2)
        print("Requesting client evaluation...")
        self.mqtt_client.publish(TOPIC_START_EVALUATION,
                                json.dumps({"round": self.current_round}), qos=1)
        log_sent_packet(
            packet_size=len(json.dumps({"round": self.current_round})),
            peer=TOPIC_START_EVALUATION,  # or client_id/server_id as appropriate
            protocol="MQTT",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="Start evaluation signal"
        )
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics from all clients"""
        print(f"\nAggregating metrics from {len(self.client_metrics)} clients...")

        # Round duration and battery (from client-reported metrics)
        if getattr(self, 'round_start_time', None) is not None:
            self.ROUND_TIMES.append(time.time() - self.round_start_time)
        socs = [m['metrics'].get('battery_soc', 1.0) for m in self.client_metrics.values() if isinstance(m.get('metrics'), dict)]
        if socs:
            avg_soc = sum(socs) / len(socs)
            self.BATTERY_CONSUMPTION.append(1.0 - avg_soc)  # consumption = 1 - SoC
        else:
            self.BATTERY_CONSUMPTION.append(0.0)
        self.BATTERY_MODEL_CONSUMPTION.append(avg_battery_model_drain_fraction(self.client_metrics))
        
        # Calculate total samples
        total_samples = sum(metric['num_samples'] 
                          for metric in self.client_metrics.values())
        
        # Weighted average of metrics (for classification)
        aggregated_accuracy = sum(metric['metrics']['accuracy'] * metric['num_samples']
                                 for metric in self.client_metrics.values()) / total_samples
        
        aggregated_loss = sum(metric['metrics']['loss'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        # Store metrics
        self.ACCURACY.append(aggregated_accuracy)
        self.LOSS.append(aggregated_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"\n{'='*70}")
        print(f"Round {self.current_round} - Aggregated Metrics:")
        print(f"  Loss:     {aggregated_loss:.6f}")
        print(f"  Accuracy: {aggregated_accuracy:.6f} ({aggregated_accuracy*100:.2f}%)")
        print(f"{'='*70}\n")
    
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
            self._send_training_complete_and_exit()
            return
        
        # Check if more rounds needed
        if self.current_round < self.num_rounds:
            self.current_round += 1
            self.round_start_time = time.time()
            
            # Capture active participants for this new round
            self.round_participants = self.active_clients.copy()
            
            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"Round {self.current_round} participants: {sorted(list(self.round_participants))}")
            print(f"{'='*70}\n")
            
            time.sleep(2)
            
            # Signal clients to start next training round (with retry, no duplicates)
            print(f"Signaling clients to start round {self.current_round}...")
            for i in range(3):
                result = self.mqtt_client.publish(TOPIC_START_TRAINING,
                                        json.dumps({"round": self.current_round}), qos=1)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print(f"  Attempt {i+1}/3: Start training signal sent")
                    break
                else:
                    print(f"  Attempt {i+1}/3: FAILED (rc={result.rc})")
                    time.sleep(0.5)
            print(f"Round {self.current_round} start signal sent\n")
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            self._send_training_complete_and_exit()
    
    def _send_training_complete_and_exit(self):
        """Send training completion signal and save/plot results."""
        if self.training_complete:
            return
        self.training_complete = True
        print("Sending training completion signal to all clients...")
        result = self.mqtt_client.publish(TOPIC_TRAINING_COMPLETE, json.dumps({"message": "Training completed"}), qos=1)
        log_sent_packet(
            packet_size=len(json.dumps({"message": "Training completed"})),
            peer=TOPIC_TRAINING_COMPLETE,
            protocol="MQTT",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="Training completion signal"
        )
        time.sleep(2)
        self.save_results()
        self.plot_results()
    
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
    
    def plot_results(self):
        """Plot training metrics: battery consumption, round/convergence time, loss & accuracy."""
        results_dir = get_experiment_results_dir("emotion", "mqtt")
        rounds = self.ROUNDS
        n = len(rounds)
        conv_time = self.convergence_time if self.convergence_time is not None else (time.time() - self.start_time if self.start_time else 0)

        # 1) Battery consumption till end of FL training
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        if self.BATTERY_CONSUMPTION and len(self.BATTERY_CONSUMPTION) == n:
            ax1.plot(rounds, [c * 100 for c in self.BATTERY_CONSUMPTION], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        else:
            bc = self.BATTERY_CONSUMPTION if len(self.BATTERY_CONSUMPTION) >= n else (self.BATTERY_CONSUMPTION + [0.0] * (n - len(self.BATTERY_CONSUMPTION)))[:n]
            if bc:
                ax1.plot(rounds, [c * 100 for c in bc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Battery consumption (%)', fontsize=12)
        ax1.set_title('Battery consumption till end of FL training', fontsize=14)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(results_dir / 'mqtt_battery_consumption.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Battery plot saved to {results_dir / 'mqtt_battery_consumption.png'}")

        # 2) Total time per round and convergence time
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if self.ROUND_TIMES and len(self.ROUND_TIMES) == n:
            ax2.bar(rounds, self.ROUND_TIMES, color='#a23b72', alpha=0.8, label='Time per round (s)')
        else:
            rt = self.ROUND_TIMES if len(self.ROUND_TIMES) >= n else (self.ROUND_TIMES + [0.0] * (n - len(self.ROUND_TIMES)))[:n]
            if rt:
                ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2, label=f'Total convergence time: {conv_time:.1f} s')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Time (s)', fontsize=12)
        ax2.set_title('Time per round and total convergence time', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(results_dir / 'mqtt_round_and_convergence_time.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Time plot saved to {results_dir / 'mqtt_round_and_convergence_time.png'}")

        # 3) Loss and Accuracy after each FL round
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        ax3a.plot(rounds, self.LOSS, marker='o', linewidth=2, markersize=8, color='red')
        ax3a.set_xlabel('Round', fontsize=12)
        ax3a.set_ylabel('Loss (Categorical Crossentropy)', fontsize=12)
        ax3a.set_title('Loss over Federated Learning Rounds', fontsize=14)
        ax3a.grid(True, alpha=0.3)
        ax3b.plot(rounds, [acc * 100 for acc in self.ACCURACY], marker='s', linewidth=2, markersize=8, color='green')
        ax3b.set_xlabel('Round', fontsize=12)
        ax3b.set_ylabel('Accuracy (%)', fontsize=12)
        ax3b.set_title('Accuracy over Federated Learning Rounds', fontsize=14)
        ax3b.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(results_dir / 'mqtt_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {results_dir / 'mqtt_training_metrics.png'}")
        plot_cpu_memory_for_server_rounds(
            results_dir,
            "mqtt_cpu_memory_per_round.png",
            self.ROUNDS,
            "emotion",
            title="MQTT (emotion): avg client CPU and RAM per round",
        )
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            plt.close('all')
        else:
            plt.show(block=False)  # Non-blocking show

        # Disconnect and exit
        print("\nTraining complete. Disconnecting...")
        time.sleep(2)  # Give time for message delivery
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop()
        print("Server disconnected successfully.")
        import sys
        sys.exit(0)
    
    def save_results(self):
        """Save results to file"""
        results_dir = get_experiment_results_dir("emotion", "mqtt")
        
        results = {
            "rounds": self.ROUNDS,
            "accuracy": self.ACCURACY,
            "loss": self.LOSS,
            "round_times_seconds": getattr(self, 'ROUND_TIMES', []),
            "battery_consumption": getattr(self, 'BATTERY_CONSUMPTION', []),
            "battery_model_consumption": getattr(self, 'BATTERY_MODEL_CONSUMPTION', []),
            "battery_model_consumption_source": "client_battery_model",
            "convergence_time_seconds": self.convergence_time,
            "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
            "total_rounds": len(self.ROUNDS),
            "num_clients": self.num_clients,
            "final_accuracy": self.ACCURACY[-1] if self.ACCURACY else None,
            "final_loss": self.LOSS[-1] if self.LOSS else None
        }
        merge_cpu_memory_into_results(results, "emotion")

        results_file = results_dir / 'mqtt_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def start(self):
        """Connect to MQTT broker and start server"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=MQTT_KEEPALIVE_SEC)
                print(f"Successfully connected to MQTT broker!\n")
                self.mqtt_client.loop_forever()
                break
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"Error: Could not connect to MQTT broker. {e}")
                    print(f"\nFailed to connect to MQTT broker after {max_retries} attempts.")
                    print(f"\nPlease ensure:")
                    print(f"  1. Mosquitto broker is running (service or container)")
                    print(f"  2. Broker address is correct: {MQTT_BROKER}:{MQTT_PORT}")
                    print(f"  3. Firewall allows connection on port {MQTT_PORT}")
                    raise


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Federated Learning Server with MQTT")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Clients: {MIN_CLIENTS} (min) - {MAX_CLIENTS} (max)")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"{'='*70}\n")
    print("Waiting for clients to connect...\n")
    
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.mqtt_client.disconnect()
