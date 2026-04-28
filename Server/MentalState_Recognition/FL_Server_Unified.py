"""
Unified Federated Learning Server for MentalState Recognition
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
import signal
import threading
from typing import List, Dict
from pathlib import Path

# Protocol-specific imports
import paho.mqtt.client as mqtt
import pika  # AMQP
import grpc  # gRPC
from concurrent import futures
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))
    import federated_learning_pb2
    import federated_learning_pb2_grpc
    GRPC_PROTO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    federated_learning_pb2 = None
    federated_learning_pb2_grpc = None
    GRPC_PROTO_AVAILABLE = False
from cyclonedds.domain import DomainParticipant  # DDS
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader

# Project root and utilities (for experiment_results path)
if os.path.exists("/app"):
    _project_root = "/app"
else:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_utilities_path = os.path.join(_project_root, "scripts", "utilities")
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)
from experiment_results_path import get_experiment_results_dir
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

try:
    from fl_termination_env import stop_on_client_convergence
except ImportError:
    def stop_on_client_convergence() -> bool:
        """Fallback if fl_termination_env is not found."""
        mode = (os.getenv("TRAINING_TERMINATION_MODE") or "").strip().lower()
        if mode == "fixed_rounds":
            return False
        if mode == "client_convergence":
            return True
        # Auto-detect RL mode
        use_ql = os.getenv("USE_QL_CONVERGENCE", "").strip().lower() in ("1", "true", "yes")
        if use_ql:
            print("[Server] Auto-detected USE_QL_CONVERGENCE=True → using fixed_rounds mode (no early stopping)")
            return False
        v = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").strip().lower()
        return v in ("1", "true", "yes")

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

# RL Configuration
USE_QL_CONVERGENCE = os.getenv("USE_QL_CONVERGENCE", "").strip().lower() in ("1", "true", "yes")

# Protocol endpoints (auto-detect Docker vs local)
IN_DOCKER = os.path.exists('/app')
MQTT_BROKER = os.getenv("MQTT_BROKER", 'mqtt-broker' if IN_DOCKER else 'localhost')
AMQP_BROKER = os.getenv("AMQP_BROKER", 'amqp-broker' if IN_DOCKER else 'localhost')
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))
PROTOCOL_NEGOTIATION_TIMEOUT_SEC = float(os.getenv("PROTOCOL_NEGOTIATION_TIMEOUT_SEC", "10.0"))  # Increased from 3.0 to 10.0 to wait for client training to finish
PROTOCOL_NEGOTIATION_POLL_SEC = float(os.getenv("PROTOCOL_NEGOTIATION_POLL_SEC", "0.2"))


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
        self.client_delivery_protocols = {}  # Per-client downlink protocol preference
        self.client_uplink_protocols = {}
        self.client_protocol_queries = {}  # Pending query keyed by client_id
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        
        # Metrics storage
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        self.ROUND_TIMES = []
        self.AVG_TRAINING_TIME_SEC = []
        self.AVG_BATTERY_SOC = []
        self.round_start_time = None
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.start_time = None
        self.convergence_time = None
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Shutdown flag for graceful termination
        self.shutdown_requested = False
        
        # Protocol handlers
        self.mqtt_client = None
        self.amqp_connection = None
        self.amqp_channel = None
        self.grpc_server = None
        self.dds_participant = None
        self.grpc_should_train = {}
        self.grpc_should_evaluate = {}
        self.grpc_model_ready = {}
        
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
        print(f"UNIFIED FEDERATED LEARNING SERVER - MENTAL STATE RECOGNITION")
        print(f"{'='*70}")
        print(f"Clients Expected: {self.num_clients}")
        print(f"Max Rounds: {self.num_rounds}")
        print(f"Protocols: MQTT, AMQP, gRPC, QUIC, HTTP/3, DDS")
        print(f"{'='*70}\n")
    
    def initialize_global_model(self):
        """Initialize global model weights using the actual mentalstate CNN+BiLSTM+MHA architecture."""
        t0 = time.time()
        print(f"[Model] initialize_global_model(): begin (pid={os.getpid()})", flush=True)
        print("[Model] Importing TensorFlow...", flush=True)
        import tensorflow as tf
        print(f"[Model] TensorFlow imported (v={getattr(tf, '__version__', 'unknown')}) in {time.time()-t0:.3f}s", flush=True)
        try:
            _server_dir = os.path.dirname(os.path.abspath(__file__))
            if _server_dir not in sys.path:
                sys.path.insert(0, _server_dir)
            print("[Model] Importing FL_Server_MQTT.build_model...", flush=True)
            from FL_Server_MQTT import build_model as _build_ms_model
            print("[Model] Building mentalstate model...", flush=True)
            _model = _build_ms_model()
            self.global_weights = _model.get_weights()
            print(f"[Model] Global model initialized (CNN+BiLSTM+MHA, {len(self.global_weights)} weight tensors) in {time.time()-t0:.3f}s", flush=True)
        except Exception as e:
            print(f"[Model] Could not load mentalstate build_model ({e}); falling back to random weights", flush=True)
            # Fallback: build the architecture inline so weight shapes are always correct
            inp = tf.keras.Input(shape=(256, 20))
            x = tf.keras.layers.Conv1D(64, 7, padding="same", use_bias=False)(inp)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling1D(2)(x)
            x = tf.keras.layers.Conv1D(128, 5, padding="same", use_bias=False)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
            )(x)
            attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
            x = tf.keras.layers.Add()([x, attn])
            x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.GlobalAveragePooling1D()(x)
            x = tf.keras.layers.Dense(256, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.4)(x)
            x = tf.keras.layers.Dense(128, activation="relu")(x)
            x = tf.keras.layers.Dropout(0.35)(x)
            out = tf.keras.layers.Dense(5, activation="softmax", dtype="float32")(x)
            _model = tf.keras.Model(inp, out)
            self.global_weights = _model.get_weights()
            print(f"[Model] Global model initialized via fallback ({len(self.global_weights)} weight tensors) in {time.time()-t0:.3f}s", flush=True)
    
    def start_mqtt_server(self):
        """Start MQTT protocol handler"""
        try:
            self.mqtt_client = mqtt.Client(client_id="fl_unified_server_mqtt", 
                                          protocol=mqtt.MQTTv311, 
                                          clean_session=True)
            self.mqtt_client.on_connect = self.on_mqtt_connect
            self.mqtt_client.on_message = self.on_mqtt_message
            self.mqtt_client.connect(MQTT_BROKER, 1883, 60)
            
            # Start MQTT loop in separate thread
            mqtt_thread = threading.Thread(target=self.mqtt_client.loop_forever, daemon=True)
            mqtt_thread.start()
            print("[MQTT] Server started")
        except Exception as e:
            print(f"[MQTT] Failed to start: {e}")
    
    def on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        print(f"[MQTT] Connected with result code {rc}")
        client.subscribe("fl/client_register")
        client.subscribe("fl/client/+/update")
        client.subscribe("fl/client/+/metrics")
    
    def on_mqtt_message(self, client, userdata, msg):
        """MQTT message callback"""
        try:
            payload = json.loads(msg.payload.decode())
            
            if msg.topic == "fl/client_register":
                self.handle_client_registration(payload['client_id'], 'mqtt')
            elif "/update" in msg.topic:
                self.handle_client_update(payload, 'mqtt')
            elif "/metrics" in msg.topic:
                self.handle_client_metrics(payload, 'mqtt')
        except Exception as e:
            print(f"[MQTT] Error handling message: {e}")
    
    def start_amqp_server(self):
        """Start AMQP protocol handler (with retry for broker startup delay)"""
        max_retries = 10
        retry_delay = 3
        connection = None
        for attempt in range(1, max_retries + 1):
            try:
                connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=AMQP_BROKER, connection_attempts=1, retry_delay=0)
                )
                break
            except Exception as e:
                print(f"[AMQP] Connection attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        if connection is None:
            print("[AMQP] Could not connect to broker after retries; AMQP disabled")
            return
        try:
            channel = connection.channel()
            
            channel.exchange_declare(exchange='fl_client_updates', exchange_type='direct', durable=True)
            channel.queue_declare(queue='fl.client.register')
            channel.queue_declare(queue='fl.client.update')
            channel.queue_declare(queue='fl.client.metrics')
            channel.queue_bind(exchange='fl_client_updates', queue='fl.client.register', routing_key='client.register')
            channel.queue_bind(exchange='fl_client_updates', queue='fl.client.update', routing_key='client.update')
            channel.queue_bind(exchange='fl_client_updates', queue='fl.client.metrics', routing_key='client.metrics')
            
            channel.basic_consume(queue='fl.client.register',
                                on_message_callback=self.on_amqp_register,
                                auto_ack=True)
            channel.basic_consume(queue='fl.client.update',
                                on_message_callback=self.on_amqp_update,
                                auto_ack=True)
            channel.basic_consume(queue='fl.client.metrics',
                                on_message_callback=self.on_amqp_metrics,
                                auto_ack=True)
            
            # Start consuming in separate thread
            amqp_thread = threading.Thread(target=channel.start_consuming, daemon=True)
            amqp_thread.start()
            
            self.amqp_connection = connection
            self.amqp_channel = channel
            print("[AMQP] Server started")
        except Exception as e:
            print(f"[AMQP] Failed to start: {e}")
    
    def on_amqp_register(self, ch, method, properties, body):
        """AMQP registration callback"""
        try:
            payload = json.loads(body.decode())
            self.handle_client_registration(payload['client_id'], 'amqp')
        except Exception as e:
            print(f"[AMQP] Error handling registration: {e}")
    
    def on_amqp_update(self, ch, method, properties, body):
        """AMQP update callback"""
        try:
            payload = json.loads(body.decode())
            self.handle_client_update(payload, 'amqp')
        except Exception as e:
            print(f"[AMQP] Error handling update: {e}")
    
    def on_amqp_metrics(self, ch, method, properties, body):
        """AMQP metrics callback"""
        try:
            payload = json.loads(body.decode())
            self.handle_client_metrics(payload, 'amqp')
        except Exception as e:
            print(f"[AMQP] Error handling metrics: {e}")
    
    def handle_client_registration(self, client_id, protocol):
        """Handle client registration (thread-safe)"""
        with self.lock:
            self.registered_clients[client_id] = protocol
            self.client_delivery_protocols.setdefault(client_id, 'grpc')
            print(f"[{protocol.upper()}] Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
            
            if len(self.registered_clients) >= self.min_clients:
                print(f"\n[Server] All {self.num_clients} clients registered!")
                self.start_training()
    
    def handle_client_update(self, payload, protocol):
        """Handle client model update (thread-safe)"""
        with self.lock:
            client_id = payload['client_id']
            round_num = payload['round']
            
            if round_num != self.current_round:
                print(f"[{protocol.upper()}] Ignoring update from client {client_id} (wrong round)")
                return
            
            self.client_uplink_protocols[client_id] = protocol

            # Store update
            self.client_updates[client_id] = {
                'weights': pickle.loads(base64.b64decode(payload['weights'])),
                'metrics': payload['metrics'],
                'protocol': protocol
            }
            
            print(f"[{protocol.upper()}] Received update from client {client_id} "
                  f"({len(self.client_updates)}/{self.num_clients})")
            
            # Wait for all registered clients (dynamic)
            if len(self.client_updates) >= len(self.registered_clients):
                self.aggregate_and_broadcast()
    
    def handle_client_metrics(self, payload, protocol):
        """Track client evaluation metrics using standalone message routing."""
        with self.lock:
            client_id = payload['client_id']
            merged = dict(payload.get('metrics') or {})
            for k in ('loss', 'accuracy', 'battery_soc', 'training_time_sec', 'training_time', 'round_time_sec'):
                if k in payload and k not in merged:
                    merged[k] = payload[k]
            self.client_metrics[client_id] = merged
    
    def start_training(self):
        """Start federated learning process"""
        self.start_time = time.time()
        self.current_round = 1
        self.round_start_time = time.time()
        for client_id in self.registered_clients.keys():
            self.grpc_model_ready[client_id] = 0
        self.broadcast_global_model()

    def _normalize_protocol_name(self, protocol_name):
        if protocol_name is None:
            return None
        normalized = str(protocol_name).strip().lower()
        allowed = {'mqtt', 'amqp', 'grpc', 'quic', 'http3', 'dds'}
        return normalized if normalized in allowed else None

    def _get_delivery_protocol(self, client_id):
        selected = self.client_delivery_protocols.get(client_id, 'grpc')
        print(f"[Server] _get_delivery_protocol({client_id}): returning '{selected}' (available protocols: {dict(self.client_delivery_protocols)})")
        return selected

    def prepare_downlink_protocol_negotiation(self, target_client_ids, round_id, global_model_id):
        target_clients = [cid for cid in target_client_ids if cid in self.registered_clients]
        if not target_clients:
            return
        print(f"[Server] Preparing downlink protocol negotiation for round {round_id}, global_model {global_model_id}, clients: {target_clients}")
        for client_id in target_clients:
            self.client_protocol_queries[client_id] = {
                'round_id': int(round_id),
                'global_model_id': int(global_model_id),
            }

        deadline = time.time() + max(PROTOCOL_NEGOTIATION_TIMEOUT_SEC, 0.0)
        while time.time() < deadline:
            pending = [cid for cid in target_clients if cid in self.client_protocol_queries]
            if not pending:
                print(f"[Server] All clients responded to downlink protocol negotiation")
                break
            time.sleep(max(PROTOCOL_NEGOTIATION_POLL_SEC, 0.01))

        pending_after_wait = [cid for cid in target_clients if cid in self.client_protocol_queries]
        if pending_after_wait:
            print(f"[Server] Timeout waiting for clients {pending_after_wait} to select downlink protocol, defaulting to gRPC")
        for client_id in pending_after_wait:
            self.client_delivery_protocols[client_id] = 'grpc'
            self.client_protocol_queries.pop(client_id, None)
    
    def aggregate_and_broadcast(self):
        """Aggregate client updates and broadcast new global model"""
        print(f"\n{'='*70}")
        print(f"ROUND {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}")
        
        # Aggregate weights (FedAvg) — global_weights is a list of numpy arrays
        updates_list = [update['weights'] for update in self.client_updates.values()]
        if isinstance(self.global_weights, list) and updates_list:
            aggregated_weights = [
                np.mean([upd[i] for upd in updates_list], axis=0)
                for i in range(len(self.global_weights))
            ]
        else:
            aggregated_weights = self.global_weights
        self.global_weights = aggregated_weights
        
        # Aggregate metrics (clients may report val_* keys)
        updates = list(self.client_updates.values())
        avg_accuracy = np.mean([
            float(u['metrics'].get('accuracy', u['metrics'].get('val_accuracy', 0.0)))
            for u in updates
        ])
        avg_loss = np.mean([
            float(u['metrics'].get('loss', u['metrics'].get('val_loss', 0.0)))
            for u in updates
        ])
        train_times = []
        socs = []
        for u in updates:
            m = u.get('metrics') or {}
            t = m.get('training_time_sec', m.get('training_time'))
            if t is not None:
                try:
                    train_times.append(float(t))
                except (TypeError, ValueError):
                    pass
            try:
                socs.append(float(m.get('battery_soc', 1.0)))
            except (TypeError, ValueError):
                socs.append(1.0)
        if self.round_start_time is not None:
            self.ROUND_TIMES.append(time.time() - self.round_start_time)
        self.AVG_TRAINING_TIME_SEC.append(float(np.mean(train_times)) if train_times else 0.0)
        self.AVG_BATTERY_SOC.append(float(np.mean(socs)) if socs else 1.0)
        
        self.ACCURACY.append(avg_accuracy)
        self.LOSS.append(avg_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"Avg Accuracy: {avg_accuracy:.4f}")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Avg training time (clients reporting): {self.AVG_TRAINING_TIME_SEC[-1]:.3f} s")
        print(f"Avg battery SoC: {self.AVG_BATTERY_SOC[-1]:.4f}")
        
        # Check for client-reported convergence signals (RL Q-convergence mode)
        if stop_on_client_convergence():
            # In client convergence mode (RL training), check if clients reported convergence
            for client_id, metrics in self.client_metrics.items():
                client_converged_flag = float(metrics.get('client_converged', 0.0))
                if client_converged_flag >= 1.0:
                    print(f"[Server] Client {client_id} reported convergence (Q-learning complete)")
                    # In single-client mode, stop immediately when client reports convergence
                    if len(self.registered_clients) == 1:
                        self.converged = True
                        self.convergence_time = time.time() - self.start_time
                        print(f"\n✅ CLIENT CONVERGENCE (Q-LEARNING) at round {self.current_round}")
                        self.save_results()
                        return
            # Don't check server-side accuracy convergence in client-convergence mode
        else:
            # Fixed-rounds mode: check server-side accuracy convergence ONLY if not using RL
            if not USE_QL_CONVERGENCE:
                if self.current_round >= MIN_ROUNDS:
                    if self.best_loss - avg_loss > CONVERGENCE_THRESHOLD:
                        self.best_loss = avg_loss
                        self.rounds_without_improvement = 0
                    else:
                        self.rounds_without_improvement += 1
                    
                    if self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
                        self.converged = True
                        self.convergence_time = time.time() - self.start_time
                        print(f"\n✅ ACCURACY CONVERGENCE at round {self.current_round}")
                        self.save_results()
                        return
                else:
                    self.best_loss = min(self.best_loss, avg_loss)
            # If USE_QL_CONVERGENCE=True, skip accuracy convergence entirely (run all NUM_ROUNDS)
        
        # Clear for next round
        self.client_updates.clear()
        self.current_round += 1
        self.round_start_time = time.time()
        
        if self.current_round <= self.num_rounds:
            self.broadcast_global_model()
        else:
            print(f"\n✅ COMPLETED {self.num_rounds} ROUNDS")
            self.save_results()
    
    def broadcast_global_model(self):
        """Broadcast global model to all clients via their registered protocols"""
        weights_b64 = base64.b64encode(pickle.dumps(self.global_weights)).decode()

        # Protocol negotiation only happens from round 2 onwards (round 1 always uses gRPC)
        if self.current_round > 1:
            print(f"[Server] Round {self.current_round}: Initiating downlink protocol negotiation...")
            self.prepare_downlink_protocol_negotiation(
                target_client_ids=list(self.registered_clients.keys()),
                round_id=self.current_round,
                global_model_id=self.current_round,
            )
        else:
            print(f"[Server] Round {self.current_round}: Initial model broadcast, using gRPC (no negotiation)")
        
        for client_id in self.registered_clients.keys():
            protocol = 'grpc' if self.current_round == 1 else self._get_delivery_protocol(client_id)
            print(f"[Server] Broadcasting round {self.current_round} to client {client_id} via {protocol.upper()}")
            message = {
                'round': self.current_round,
                'weights': weights_b64,
                'server_sent_unix': time.time(),
            }
            
            try:
                self.grpc_model_ready[client_id] = self.current_round
                if protocol == 'mqtt':
                    self.mqtt_client.publish(f"fl/global_model/{client_id}", 
                                            json.dumps(message))
                elif protocol == 'amqp':
                    # AMQP broadcast logic
                    pass
                elif protocol == 'grpc':
                    # Client pulls via GetGlobalModel over gRPC control/data plane.
                    pass
                # Other protocols can be added here as needed.
                
                print(f"[{protocol.upper()}] Sent global model to client {client_id}")
            except Exception as e:
                print(f"[{protocol.upper()}] Error broadcasting to client {client_id}: {e}")

    def start_grpc_server(self):
        if not GRPC_PROTO_AVAILABLE:
            print("[gRPC] Proto bindings unavailable; skipping gRPC server start")
            return
        options = [
            ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
            ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
        ]
        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
        federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(
            FLServicer(self), self.grpc_server
        )
        self.grpc_server.add_insecure_port(f'[::]:{GRPC_PORT}')
        self.grpc_server.start()
        print(f"[gRPC] Server started on port {GRPC_PORT}")
    
    def save_results(self):
        """Save experiment results"""
        results_dir = get_experiment_results_dir("mental_state", "unified")
        
        results = {
            'rounds': self.ROUNDS,
            'accuracy': self.ACCURACY,
            'loss': self.LOSS,
            'round_times_seconds': getattr(self, 'ROUND_TIMES', []),
            'avg_training_time_sec': getattr(self, 'AVG_TRAINING_TIME_SEC', []),
            'avg_battery_soc': getattr(self, 'AVG_BATTERY_SOC', []),
            'converged': self.converged,
            'total_time': time.time() - self.start_time,
            'convergence_time': self.convergence_time,
            'num_clients': self.num_clients,
            'protocols_used': dict(self.registered_clients)
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

        merge_cpu_memory_into_results(results, "mental_state")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"unified_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        stable_runner = results_dir / 'rl_unified_training_results.json'
        with open(stable_runner, 'w') as f:
            json.dump(results, f, indent=2)

        plot_cpu_memory_for_server_rounds(
            results_dir,
            "unified_cpu_memory_per_round.png",
            self.ROUNDS,
            "mental_state",
            title="Unified RL (mental_state): avg client CPU and RAM per round",
        )

        print(f"\n✅ Results saved to {results_file}")
        print(f"✅ Experiment runner snapshot: {stable_runner}")

    def cleanup(self):
        """Cleanup all protocol handlers gracefully"""
        print("\n[Server] Shutting down all protocol handlers...")
        
        # Stop gRPC server
        if self.grpc_server is not None:
            try:
                print("[gRPC] Stopping server...")
                self.grpc_server.stop(grace=2)
                print("[gRPC] Server stopped")
            except Exception as e:
                print(f"[gRPC] Error stopping server: {e}")
        
        # Disconnect MQTT client
        if self.mqtt_client is not None:
            try:
                print("[MQTT] Disconnecting client...")
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                print("[MQTT] Client disconnected")
            except Exception as e:
                print(f"[MQTT] Error disconnecting: {e}")
        
        # Close AMQP connection
        if self.amqp_channel is not None:
            try:
                print("[AMQP] Stopping channel...")
                self.amqp_channel.stop_consuming()
            except Exception as e:
                print(f"[AMQP] Error stopping channel: {e}")
        
        if self.amqp_connection is not None:
            try:
                print("[AMQP] Closing connection...")
                self.amqp_connection.close()
                print("[AMQP] Connection closed")
            except Exception as e:
                print(f"[AMQP] Error closing connection: {e}")
        
        # Cleanup DDS participant
        if self.dds_participant is not None:
            try:
                print("[DDS] Cleaning up participant...")
                # DDS participant cleanup is automatic when object is destroyed
                self.dds_participant = None
                print("[DDS] Participant cleaned up")
            except Exception as e:
                print(f"[DDS] Error cleaning up: {e}")
        
        print("[Server] All protocol handlers stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals (SIGTERM, SIGINT)"""
        sig_name = signal.Signals(signum).name
        print(f"\n[Server] Received {sig_name} signal, initiating graceful shutdown...")
        self.shutdown_requested = True

    def run(self):
        """Run unified server (all protocols)"""
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("[Server] Starting all protocol handlers...")
        
        self.start_mqtt_server()
        self.start_amqp_server()
        self.start_grpc_server()
        
        print("[Server] All protocol handlers started")
        print("[Server] Waiting for client registrations...")
        
        # Keep main thread alive until convergence or shutdown signal
        try:
            while not self.shutdown_requested and not self.converged and self.current_round <= self.num_rounds:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Server] Interrupted by user")
            self.shutdown_requested = True
        
        # Perform cleanup
        self.cleanup()


if GRPC_PROTO_AVAILABLE:
    class FLServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
        def __init__(self, server):
            self.server = server

        def RegisterClient(self, request, context):
            self.server.handle_client_registration(request.client_id, 'grpc')
            return federated_learning_pb2.RegistrationResponse(success=True, message='ok')

        def GetGlobalModel(self, request, context):
            if self.server.global_weights is None:
                return federated_learning_pb2.GlobalModel(
                    round=0, weights=b'', available=False, model_config='', chunk_index=0, total_chunks=1,
                    server_sent_unix=0.0,
                )
            ready_round = self.server.grpc_model_ready.get(request.client_id)
            if ready_round is None:
                return federated_learning_pb2.GlobalModel(
                    round=request.round, weights=b'', available=False, model_config='', chunk_index=0, total_chunks=1,
                    server_sent_unix=0.0,
                )
            serialized = pickle.dumps(self.server.global_weights)
            return federated_learning_pb2.GlobalModel(
                round=int(ready_round),
                weights=serialized,
                available=True,
                model_config='',
                chunk_index=0,
                total_chunks=1,
                server_sent_unix=time.time(),
            )

        def CheckTrainingStatus(self, request, context):
            pending_query = self.server.client_protocol_queries.get(request.client_id)
            has_protocol_query = pending_query is not None
            kwargs = {
                'should_train': False,
                'should_evaluate': False,
                'current_round': int(self.server.current_round),
                'is_complete': bool(self.server.converged),
                'has_protocol_query': has_protocol_query,
            }
            if has_protocol_query:
                kwargs['protocol_query'] = federated_learning_pb2.ProtocolQuery(
                    client_id=request.client_id,
                    round_id=int(pending_query['round_id']),
                    global_model_id=int(pending_query['global_model_id']),
                )
            return federated_learning_pb2.TrainingStatus(**kwargs)

        def SendProtocolSelection(self, request, context):
            print(f"[Server] Received SendProtocolSelection from client {request.client_id}: round={request.round_id}, model={request.global_model_id}, protocol={request.downlink_protocol_requested}")
            selected = self.server._normalize_protocol_name(request.downlink_protocol_requested)
            if selected is None:
                print(f"[Server] Invalid protocol requested: {request.downlink_protocol_requested}")
                return federated_learning_pb2.ProtocolSelectionResponse(success=False, message='invalid protocol')
            print(f"[Server] Setting client {request.client_id} delivery protocol to: {selected}")
            self.server.client_delivery_protocols[request.client_id] = selected
            self.server.client_protocol_queries.pop(request.client_id, None)
            print(f"[Server] Successfully recorded protocol selection for client {request.client_id}")
            return federated_learning_pb2.ProtocolSelectionResponse(success=True, message='recorded')

        def SendModelUpdate(self, request, context):
            try:
                payload = {
                    'client_id': request.client_id,
                    'round':     request.round,
                    # handle_client_update expects base64-encoded pickle bytes
                    'weights':   base64.b64encode(request.weights).decode('utf-8'),
                    'metrics':   dict(request.metrics),
                    'num_samples': request.num_samples,
                }
                self.server.handle_client_update(payload, 'grpc')
                return federated_learning_pb2.UpdateResponse(success=True, message='ok')
            except Exception as e:
                print(f"[gRPC] SendModelUpdate error: {e}")
                return federated_learning_pb2.UpdateResponse(success=False, message=str(e))

        def SendMetrics(self, request, context):
            return federated_learning_pb2.MetricsResponse(success=False, message='not implemented in this unified mode')

        def GetTrainingConfig(self, request, context):
            # Use configurable batch size and epochs (environment variables or defaults)
            batch_size = int(os.getenv("DEFAULT_DATA_BATCH_SIZE", "32"))
            local_epochs = int(os.getenv("DEFAULT_LOCAL_EPOCHS", "20"))
            return federated_learning_pb2.TrainingConfig(batch_size=batch_size, local_epochs=local_epochs)


def main():
    """Main function"""
    server = UnifiedFederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, max_clients=MAX_CLIENTS)
    server.run()


if __name__ == "__main__":
    main()
