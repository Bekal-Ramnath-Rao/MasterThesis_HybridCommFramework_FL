"""
Unified Federated Learning Server for Temperature Recognition
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
import fcntl
from typing import List, Dict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

try:
    from fl_termination_env import stop_on_client_convergence
except ImportError:
    def stop_on_client_convergence() -> bool:
        mode = (os.getenv("TRAINING_TERMINATION_MODE") or "").strip().lower()
        if mode == "fixed_rounds":
            return False
        if mode == "client_convergence":
            return True
        v = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").strip().lower()
        return v in ("1", "true", "yes")

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
AMQP_BROKER = os.getenv("AMQP_BROKER", 'amqp-broker' if IN_DOCKER else 'localhost')
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))
PROTOCOL_NEGOTIATION_TIMEOUT_SEC = float(os.getenv("PROTOCOL_NEGOTIATION_TIMEOUT_SEC", "3.0"))
PROTOCOL_NEGOTIATION_POLL_SEC = float(os.getenv("PROTOCOL_NEGOTIATION_POLL_SEC", "0.1"))


def _amqp_use_case_tag() -> str:
    """Isolate AMQP model-update traffic per use case (shared broker: emotion native uses ``client.update``)."""
    raw = (os.getenv("USE_CASE") or os.getenv("CLIENT_USE_CASE") or "temperature").strip().lower()
    return raw.replace(" ", "_").replace("-", "_") if raw else "temperature"


_AMQP_UC = _amqp_use_case_tag()
# Default namespaced keys so another stack (e.g. emotion on host) does not fill our consumer with foreign rounds.
AMQP_MODEL_UPDATE_ROUTING_KEY = os.getenv(
    "AMQP_MODEL_UPDATE_ROUTING_KEY", f"client.update.{_AMQP_UC}"
)
AMQP_MODEL_UPDATE_QUEUE = os.getenv(
    "AMQP_MODEL_UPDATE_QUEUE", f"fl.client.update.{_AMQP_UC}"
)
AMQP_MODEL_METRICS_ROUTING_KEY = os.getenv(
    "AMQP_MODEL_METRICS_ROUTING_KEY", f"client.metrics.{_AMQP_UC}"
)
AMQP_MODEL_METRICS_QUEUE = os.getenv(
    "AMQP_MODEL_METRICS_QUEUE", f"fl.client.metrics.{_AMQP_UC}"
)


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
        self.converged_clients = set()  # tracks clients that sent client_converged=1.0
        
        # Lock for thread-safe operations
        self.lock = threading.Lock()
        
        # Protocol handlers
        self.mqtt_client = None
        self.amqp_connection = None
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
        print(f"UNIFIED FEDERATED LEARNING SERVER - TEMPERATURE REGULATION")
        print(f"{'='*70}")
        print(f"Clients Expected: {self.num_clients}")
        print(f"Max Rounds: {self.num_rounds}")
        print(f"Protocols: MQTT, AMQP, gRPC, QUIC, DDS")
        print(f"{'='*70}\n")
    
    def initialize_global_model(self):
        """Initialize global model weights as a list (Keras get_weights() format).

        The temperature model is a small Dense network; initialise to None so the
        server does NOT broadcast oversized random CNN arrays that (a) overflow gRPC
        4 MB limit and (b) have the wrong layer structure.  global_weights is
        populated by list-based FedAvg after the first round of real client updates.
        """
        self.global_weights = None
        print("[Model] Global model initialised (weights=None; will be set after first aggregation)")
    
    def start_mqtt_server(self):
        """Start MQTT protocol handler"""
        try:
            # paho-mqtt ≥2.0 requires an explicit CallbackAPIVersion; use VERSION1
            # for backward-compatible on_connect(client, userdata, flags, rc) signature.
            _mqtt_version = getattr(mqtt, "CallbackAPIVersion", None)
            if _mqtt_version is not None:
                self.mqtt_client = mqtt.Client(
                    _mqtt_version.VERSION1,
                    client_id="fl_unified_server_mqtt",
                    protocol=mqtt.MQTTv311,
                    clean_session=True,
                )
            else:
                self.mqtt_client = mqtt.Client(
                    client_id="fl_unified_server_mqtt",
                    protocol=mqtt.MQTTv311,
                    clean_session=True,
                )
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
        """Start AMQP protocol handler"""
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=AMQP_BROKER)
            )
            channel = connection.channel()
            
            channel.exchange_declare(exchange='fl_client_updates', exchange_type='direct', durable=True)
            channel.queue_declare(queue='fl.client.register')
            channel.queue_declare(queue=AMQP_MODEL_UPDATE_QUEUE)
            channel.queue_declare(queue=AMQP_MODEL_METRICS_QUEUE)
            channel.queue_bind(exchange='fl_client_updates', queue='fl.client.register', routing_key='client.register')
            channel.queue_bind(
                exchange='fl_client_updates',
                queue=AMQP_MODEL_UPDATE_QUEUE,
                routing_key=AMQP_MODEL_UPDATE_ROUTING_KEY,
            )
            channel.queue_bind(
                exchange='fl_client_updates',
                queue=AMQP_MODEL_METRICS_QUEUE,
                routing_key=AMQP_MODEL_METRICS_ROUTING_KEY,
            )
            print(
                f"[AMQP] Unified temperature: update queue={AMQP_MODEL_UPDATE_QUEUE!r} "
                f"routing_key={AMQP_MODEL_UPDATE_ROUTING_KEY!r}; "
                f"metrics queue={AMQP_MODEL_METRICS_QUEUE!r} routing_key={AMQP_MODEL_METRICS_ROUTING_KEY!r}"
            )
            # Durable queues retain JSON updates from prior runs; purge ours on startup.
            if os.getenv("AMQP_PURGE_UPDATE_QUEUES_ON_START", "true").strip().lower() in ("1", "true", "yes"):
                for qname in (AMQP_MODEL_UPDATE_QUEUE, AMQP_MODEL_METRICS_QUEUE):
                    try:
                        purged = channel.queue_purge(qname)
                        mc = getattr(getattr(purged, "method", purged), "message_count", None)
                        if mc is not None:
                            print(f"[AMQP] Purged queue {qname!r} on startup ({mc} stale message(s))")
                        else:
                            print(f"[AMQP] Purged queue {qname!r} on startup")
                    except Exception as pe:
                        print(f"[AMQP] Purge {qname!r} skipped: {pe}")
            
            channel.basic_consume(queue='fl.client.register',
                                on_message_callback=self.on_amqp_register,
                                auto_ack=True)
            channel.basic_consume(queue=AMQP_MODEL_UPDATE_QUEUE,
                                on_message_callback=self.on_amqp_update,
                                auto_ack=True)
            channel.basic_consume(queue=AMQP_MODEL_METRICS_QUEUE,
                                on_message_callback=self.on_amqp_metrics,
                                auto_ack=True)
            
            # Start consuming in separate thread
            amqp_thread = threading.Thread(target=channel.start_consuming, daemon=True)
            amqp_thread.start()
            
            self.amqp_connection = connection
            print("[AMQP] Server started")
        except Exception as e:
            print(f"[AMQP] Failed to start: {e}")
    
    def on_amqp_register(self, ch, method, properties, body):
        """AMQP registration callback"""
        try:
            payload = json.loads(body.decode("utf-8"))
            self.handle_client_registration(payload['client_id'], 'amqp')
        except Exception as e:
            print(f"[AMQP] Error handling registration: {e}")
    
    def on_amqp_update(self, ch, method, properties, body):
        """AMQP update callback (JSON body; ignore stale binary pickles from legacy clients)."""
        try:
            payload = json.loads(body.decode("utf-8"))
            self.handle_client_update(payload, 'amqp')
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            hint = (
                f"[AMQP] Ignoring non-JSON update on {AMQP_MODEL_UPDATE_QUEUE!r} "
                "(often durable stale pickle from an older client build; purge the queue or restart RabbitMQ with a clean volume). "
                f"Detail: {e}"
            )
            print(hint)
        except Exception as e:
            print(f"[AMQP] Error handling update: {e}")
    
    def on_amqp_metrics(self, ch, method, properties, body):
        """AMQP metrics callback"""
        try:
            payload = json.loads(body.decode("utf-8"))
            self.handle_client_metrics(payload, 'amqp')
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"[AMQP] Ignoring non-JSON metrics message: {e}")
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
    
    def mark_client_converged(self, client_id: int):
        """Record that client reported RL convergence and stop training if all clients done."""
        self.converged_clients.add(client_id)
        print(f"[Server] Client {client_id} signaled RL convergence ({len(self.converged_clients)}/{len(self.registered_clients)})")
        if len(self.converged_clients) >= len(self.registered_clients):
            self.signal_training_complete()

    def signal_training_complete(self):
        """Mark training as complete and trigger save."""
        if not self.converged:
            print(f"\n✅ ALL CLIENTS RL-CONVERGED — stopping FL at round {self.current_round}")
            self.converged = True
            self.convergence_time = time.time() - self.start_time
            self.save_results()

    def handle_client_update(self, payload, protocol):
        """Handle client model update (thread-safe)"""
        with self.lock:
            client_id = payload['client_id']
            round_num = payload['round']

            # Check for client-driven RL convergence signal (empty-weights notification)
            client_converged_flag = float((payload.get('metrics') or {}).get('client_converged', 0.0))
            if stop_on_client_convergence() and client_converged_flag >= 1.0:
                print(f"[{protocol.upper()}] Received RL convergence signal from client {client_id}")
                self.mark_client_converged(client_id)
                # Convergence-only message has no real weights; skip normal update processing.
                if not payload.get('weights'):
                    return

            if round_num != self.current_round:
                if self.current_round == 0:
                    print(
                        f"[{protocol.upper()}] Ignoring update from client {client_id}: "
                        f"server not in training (current_round=0, update claims round={round_num}). "
                        f"Ensure the client calls gRPC RegisterClient (or publishes AMQP register) before FL rounds."
                    )
                else:
                    print(
                        f"[{protocol.upper()}] Ignoring update from client {client_id}: "
                        f"wrong round (server expects {self.current_round}, payload has {round_num})"
                    )
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
            for k in ('loss', 'accuracy', 'battery_soc', 'training_time_sec', 'training_time', 'round_time_sec', 'val_mae', 'val_loss'):
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
        return self.client_delivery_protocols.get(client_id, 'grpc')

    def prepare_downlink_protocol_negotiation(self, target_client_ids, round_id, global_model_id):
        target_clients = [cid for cid in target_client_ids if cid in self.registered_clients]
        if not target_clients:
            return
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
            self.client_delivery_protocols[client_id] = 'grpc'
            self.client_protocol_queries.pop(client_id, None)
    
    def aggregate_and_broadcast(self):
        """Aggregate client updates and broadcast new global model"""
        print(f"\n{'='*70}")
        print(f"ROUND {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}")
        
        # Aggregate weights using list-based FedAvg.
        # Clients send model.get_weights() which is a list of numpy arrays (one
        # per Keras layer), so we iterate by integer index, not string key.
        all_weights = [update['weights'] for update in self.client_updates.values()]
        if not all_weights:
            print("[Aggregation] No client weights received; skipping round.")
            return
        num_layers = len(all_weights[0])
        aggregated_weights = [
            np.mean([w[i] for w in all_weights], axis=0)
            for i in range(num_layers)
        ]
        self.global_weights = aggregated_weights
        
        # Aggregate metrics (temperature clients use val_accuracy / val_mae proxies)
        updates = list(self.client_updates.values())
        avg_accuracy = np.mean([
            float(u['metrics'].get(
                'accuracy', u['metrics'].get('val_accuracy', 0.0)
            ))
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
        
        print(f"Avg Accuracy (proxy): {avg_accuracy:.4f}")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Avg training time (clients reporting): {self.AVG_TRAINING_TIME_SEC[-1]:.3f} s")
        print(f"Avg battery SoC: {self.AVG_BATTERY_SOC[-1]:.4f}")

        # If all clients already signaled RL convergence, stop immediately
        if self.converged:
            return

        # Loss-based early stopping — disabled in client_convergence mode so the
        # RL agent can complete all 3 phases (Phase 1: 20+ rounds, Phase 2:
        # boundary setting, Phase 3: Q-learning until 5 consecutive same-protocol).
        if not stop_on_client_convergence():
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
                    return
            else:
                self.best_loss = min(self.best_loss, avg_loss)
        else:
            # client_convergence mode: track best loss for reporting but don't stop early
            self.best_loss = min(self.best_loss, avg_loss)
        
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

        if self.current_round > 1:
            self.prepare_downlink_protocol_negotiation(
                target_client_ids=list(self.registered_clients.keys()),
                round_id=self.current_round,
                global_model_id=self.current_round,
            )
        
        for client_id in self.registered_clients.keys():
            protocol = 'grpc' if self.current_round == 1 else self._get_delivery_protocol(client_id)
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
        results_dir = get_experiment_results_dir("temperature", "unified")
        
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

        merge_cpu_memory_into_results(results, "temperature")
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
        """Plot battery SoC, round/convergence time, loss, and accuracy."""
        results_dir = get_experiment_results_dir("temperature", "unified")
        rounds = self.ROUNDS
        n = len(rounds)
        if n == 0:
            print("[plot_results] No rounds recorded – skipping plots.")
            return
        conv_time = self.convergence_time if self.convergence_time is not None else (
            time.time() - self.start_time if self.start_time else 0
        )

        # 1) Battery SoC per round
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        soc = (self.AVG_BATTERY_SOC + [1.0] * max(0, n - len(self.AVG_BATTERY_SOC)))[:n] if self.AVG_BATTERY_SOC else [1.0] * n
        ax1.plot(rounds, [s * 100 for s in soc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Avg Battery SoC (%)', fontsize=12)
        ax1.set_title('Unified (temperature): Avg Battery SoC per FL round', fontsize=14)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(results_dir / 'unified_battery_soc.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Battery SoC plot saved to {results_dir / 'unified_battery_soc.png'}")

        # 2) Time per round and convergence time
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        rt = (self.ROUND_TIMES + [0.0] * max(0, n - len(self.ROUND_TIMES)))[:n] if self.ROUND_TIMES else [0.0] * n
        ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2,
                    label=f'Total convergence time: {conv_time:.1f} s')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Time (s)', fontsize=12)
        ax2.set_title('Unified (temperature): Time per round and total convergence time', fontsize=14)
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
        ax3a.set_title('Unified (temperature): Loss over FL Rounds', fontsize=14)
        ax3a.grid(True, alpha=0.3)
        ax3b.plot(rounds, [a * 100 for a in self.ACCURACY], marker='s', linewidth=2, markersize=8, color='green')
        ax3b.set_xlabel('Round', fontsize=12)
        ax3b.set_ylabel('Accuracy (%)', fontsize=12)
        ax3b.set_title('Unified (temperature): Accuracy over FL Rounds', fontsize=14)
        ax3b.grid(True, alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(results_dir / 'unified_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Training metrics plot saved to {results_dir / 'unified_training_metrics.png'}")

        # 4) Avg training time per round
        fig4, ax4 = plt.subplots(figsize=(7, 4))
        att = (self.AVG_TRAINING_TIME_SEC + [0.0] * max(0, n - len(self.AVG_TRAINING_TIME_SEC)))[:n] if self.AVG_TRAINING_TIME_SEC else [0.0] * n
        ax4.plot(rounds, att, marker='D', linewidth=2, markersize=6, color='#6a4c93')
        ax4.set_xlabel('Round', fontsize=12)
        ax4.set_ylabel('Avg client training time (s)', fontsize=12)
        ax4.set_title('Unified (temperature): Avg client training time per round', fontsize=14)
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(results_dir / 'unified_avg_training_time.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print(f"Training time plot saved to {results_dir / 'unified_avg_training_time.png'}")

        # 5) CPU and RAM from client JSONL
        plot_cpu_memory_for_server_rounds(
            results_dir,
            "unified_cpu_memory_per_round.png",
            self.ROUNDS,
            "temperature",
            title="Unified RL (temperature): avg client CPU and RAM per round",
        )

    def run(self):
        """Run unified server (all protocols)"""
        print("[Server] Starting all protocol handlers...")
        
        self.start_mqtt_server()
        self.start_amqp_server()
        self.start_grpc_server()
        
        print("[Server] All protocol handlers started")
        print("[Server] Waiting for client registrations...")
        
        # Keep main thread alive
        try:
            while not self.converged and self.current_round <= self.num_rounds:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Server] Interrupted by user")
        
        print("[Server] Shutting down...")
        if self.grpc_server is not None:
            self.grpc_server.stop(0)


if GRPC_PROTO_AVAILABLE:
    class FLServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
        def __init__(self, server):
            self.server = server
            self._grpc_update_chunks = {}

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
            selected = self.server._normalize_protocol_name(request.downlink_protocol_requested)
            if selected is None:
                return federated_learning_pb2.ProtocolSelectionResponse(success=False, message='invalid protocol')
            self.server.client_delivery_protocols[request.client_id] = selected
            self.server.client_protocol_queries.pop(request.client_id, None)
            return federated_learning_pb2.ProtocolSelectionResponse(success=True, message='recorded')

        def SendModelUpdate(self, request, context):
            """Receive model update (chunked when payload exceeds gRPC frame budget). Same contract as emotion unified."""
            try:
                client_id = request.client_id
                round_num = request.round
                metrics = dict(request.metrics)
                total_chunks = getattr(request, "total_chunks", 1) or 1
                chunk_index = getattr(request, "chunk_index", 0) or 0

                # Check for RL convergence signal BEFORE the round-mismatch guard.
                # The client sends this with round = current_round + 1 (which never
                # matches server's current_round), so it would be silently dropped if
                # we checked rounds first — leaving the server stuck in its run() loop.
                client_converged_flag = float(metrics.get("client_converged", 0.0))
                if client_converged_flag >= 1.0:
                    print(f"[gRPC] Received RL convergence signal from client {client_id} (round={round_num})")
                    self.server.mark_client_converged(client_id)
                    return federated_learning_pb2.UpdateResponse(
                        success=True,
                        message="RL convergence acknowledged",
                    )

                if round_num != self.server.current_round:
                    return federated_learning_pb2.UpdateResponse(
                        success=False,
                        message=f"Round mismatch: expected {self.server.current_round}, got {round_num}",
                    )

                if total_chunks > 1:
                    key = (client_id, round_num)
                    if key not in self._grpc_update_chunks:
                        self._grpc_update_chunks[key] = {"chunks": {}, "num_samples": 0, "metrics": {}}
                    buf = self._grpc_update_chunks[key]
                    buf["chunks"][chunk_index] = request.weights if request.weights else b""
                    if chunk_index == 0:
                        buf["num_samples"] = request.num_samples
                        buf["metrics"] = metrics
                    if len(buf["chunks"]) < total_chunks:
                        return federated_learning_pb2.UpdateResponse(
                            success=True,
                            message=f"Chunk {chunk_index + 1}/{total_chunks} received for round {round_num}",
                        )
                    serialized_weights = b"".join(buf["chunks"][i] for i in range(total_chunks))
                    num_samples = buf["num_samples"]
                    metrics = buf["metrics"]
                    del self._grpc_update_chunks[key]
                else:
                    serialized_weights = request.weights
                    num_samples = request.num_samples

                weights_b64 = base64.b64encode(serialized_weights).decode("ascii")
                data = {
                    "client_id": client_id,
                    "round": round_num,
                    "weights": weights_b64,
                    "num_samples": num_samples,
                    "metrics": metrics,
                }
                self.server.handle_client_update(data, "grpc")
                return federated_learning_pb2.UpdateResponse(success=True, message="Update received")
            except Exception as e:
                print(f"[gRPC] Error in SendModelUpdate: {e}")
                import traceback
                traceback.print_exc()
                return federated_learning_pb2.UpdateResponse(success=False, message=str(e))

        def SendMetrics(self, request, context):
            return federated_learning_pb2.MetricsResponse(success=False, message='not implemented in this unified mode')

        def GetTrainingConfig(self, request, context):
            return federated_learning_pb2.TrainingConfig(batch_size=16, local_epochs=20)


def main():
    """Main function"""
    # Prevent duplicate unified server instance in the same container/host.
    lock_path = os.path.join(os.path.expanduser("~"), ".unified_temperature_server.lock")
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
