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
AMQP_BROKER = os.getenv("AMQP_BROKER", 'amqp-broker' if IN_DOCKER else 'localhost')
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))
PROTOCOL_NEGOTIATION_TIMEOUT_SEC = float(os.getenv("PROTOCOL_NEGOTIATION_TIMEOUT_SEC", "3.0"))
PROTOCOL_NEGOTIATION_POLL_SEC = float(os.getenv("PROTOCOL_NEGOTIATION_POLL_SEC", "0.1"))


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
        print(f"UNIFIED FEDERATED LEARNING SERVER - EMOTION RECOGNITION")
        print(f"{'='*70}")
        print(f"Clients Expected: {self.num_clients}")
        print(f"Max Rounds: {self.num_rounds}")
        print(f"Protocols: MQTT, AMQP, gRPC, QUIC, DDS")
        print(f"{'='*70}\n")
    
    def initialize_global_model(self):
        """Initialize global model weights (mentalstate recognition CNN)"""
        # Simple initialization - clients will have actual trained weights
        # This creates a structure that matches the CNN architecture
        self.global_weights = {
            'conv1': np.random.randn(3, 3, 1, 32) * 0.01,
            'conv2': np.random.randn(3, 3, 32, 64) * 0.01,
            'conv3': np.random.randn(3, 3, 64, 128) * 0.01,
            'conv4': np.random.randn(3, 3, 128, 128) * 0.01,
            'dense1': np.random.randn(6272, 1024) * 0.01,  # After flatten
            'dense2': np.random.randn(1024, 7) * 0.01  # 7 mentalstates
        }
        print("[Model] Global model initialized")
    
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
        """Start AMQP protocol handler"""
        try:
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=AMQP_BROKER)
            )
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
            self.client_metrics[client_id] = payload.get('metrics', {})
    
    def start_training(self):
        """Start federated learning process"""
        self.start_time = time.time()
        self.current_round = 1
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
        
        # Aggregate weights (FedAvg)
        aggregated_weights = {}
        for key in self.global_weights.keys():
            weights_list = [update['weights'][key] for update in self.client_updates.values()]
            aggregated_weights[key] = np.mean(weights_list, axis=0)
        
        self.global_weights = aggregated_weights
        
        # Aggregate metrics
        avg_accuracy = np.mean([u['metrics']['accuracy'] for u in self.client_updates.values()])
        avg_loss = np.mean([u['metrics']['loss'] for u in self.client_updates.values()])
        
        self.ACCURACY.append(avg_accuracy)
        self.LOSS.append(avg_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"Avg Accuracy: {avg_accuracy:.4f}")
        print(f"Avg Loss: {avg_loss:.4f}")
        
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
                return
        else:
            self.best_loss = min(self.best_loss, avg_loss)
        
        # Clear for next round
        self.client_updates.clear()
        self.current_round += 1
        
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
                'weights': weights_b64
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
            'converged': self.converged,
            'total_time': time.time() - self.start_time,
            'convergence_time': self.convergence_time,
            'num_clients': self.num_clients,
            'protocols_used': dict(self.registered_clients)
        }
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"unified_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to {results_file}")
    
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

        def RegisterClient(self, request, context):
            self.server.handle_client_registration(request.client_id, 'grpc')
            return federated_learning_pb2.RegistrationResponse(success=True, message='ok')

        def GetGlobalModel(self, request, context):
            if self.server.global_weights is None:
                return federated_learning_pb2.GlobalModel(
                    round=0, weights=b'', available=False, model_config='', chunk_index=0, total_chunks=1
                )
            ready_round = self.server.grpc_model_ready.get(request.client_id)
            if ready_round is None:
                return federated_learning_pb2.GlobalModel(
                    round=request.round, weights=b'', available=False, model_config='', chunk_index=0, total_chunks=1
                )
            serialized = pickle.dumps(self.server.global_weights)
            return federated_learning_pb2.GlobalModel(
                round=int(ready_round),
                weights=serialized,
                available=True,
                model_config='',
                chunk_index=0,
                total_chunks=1,
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
            return federated_learning_pb2.UpdateResponse(success=False, message='not implemented in this unified mode')

        def SendMetrics(self, request, context):
            return federated_learning_pb2.MetricsResponse(success=False, message='not implemented in this unified mode')

        def GetTrainingConfig(self, request, context):
            return federated_learning_pb2.TrainingConfig(batch_size=16, local_epochs=5)


def main():
    """Main function"""
    server = UnifiedFederatedLearningServer(NUM_CLIENTS, NUM_ROUNDS)
    server.run()


if __name__ == "__main__":
    main()
