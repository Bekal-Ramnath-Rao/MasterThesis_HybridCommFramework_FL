import os
import sys
# Server uses CPU only (aggregation is numpy-only); saves GPU memory for clients
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import json
import pickle
import time
import grpc
from concurrent import futures
import threading
import matplotlib.pyplot as plt
from pathlib import Path

# Project root and utilities (for experiment_results path)
if os.path.exists("/app"):
    _project_root = "/app"
else:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
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


# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# Server Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "0.0.0.0")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))

# Convergence Settings
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))


class FederatedLearningServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
    def __init__(self, min_clients, num_rounds, max_clients=100, grpc_server=None):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self._grpc_server = grpc_server  # used to stop server after training in diagnostic pipeline
        self.num_clients = min_clients  # Start with minimum, will update as clients join
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.active_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.clients_evaluated = set()
        self.global_weights = None
        self.model_config = None
        self.lock = threading.Lock()
        
        # Metrics storage for classification
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        self.ROUND_TIMES = []
        self.BATTERY_CONSUMPTION = []
        self.round_start_time = None

        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.start_time = None
        self.convergence_time = None
        
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
        # Training configuration broadcast to gRPC clients
        self.training_config = {
            "batch_size": int(os.getenv("BATCH_SIZE", "16")),
            "local_epochs": 20
        }
        
        # Status flags
        self.training_started = False
        self.training_complete = False
        self.evaluation_phase = False
        # Chunked client updates: (client_id, round) -> {'chunks': {index: bytes}, 'num_samples': int, 'metrics': dict}
        self._update_chunks = {}
    
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
        
        # Store model configuration for clients to rebuild
        self.model_config = {
            'input_shape': [48, 48, 1],
            'num_classes': 7,
            'layers': [
                {'type': 'Conv2D', 'filters': 32, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'Conv2D', 'filters': 64, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
                {'type': 'Dropout', 'rate': 0.25},
                {'type': 'Conv2D', 'filters': 128, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
                {'type': 'Conv2D', 'filters': 128, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
                {'type': 'Dropout', 'rate': 0.25},
                {'type': 'Flatten'},
                {'type': 'Dense', 'units': 1024, 'activation': 'relu'},
                {'type': 'Dropout', 'rate': 0.5},
                {'type': 'Dense', 'units': 7, 'activation': 'softmax'}
            ]
        }
        
        print("\nGlobal CNN model initialized for emotion recognition")
        print(f"Model architecture: CNN with {len(self.global_weights)} weight layers")
        print(f"Input shape: 48x48x1 (grayscale images)")
        print(f"Output classes: 7 emotions")
    
    def serialize_weights(self, weights):
        """Serialize model weights for gRPC transmission"""
        serialized = pickle.dumps(weights)
        return serialized
    
    def deserialize_weights(self, serialized_weights):
        """Deserialize model weights received from gRPC"""
        weights = pickle.loads(serialized_weights)
        return weights
    
    def RegisterClient(self, request, context):
        """Handle client registration"""
        with self.lock:
            self.registered_clients.add(request.client_id)
            self.active_clients.add(request.client_id)
            print(f"Client {request.client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
            
            # Start when we have at least min_clients (e.g. 2); after one converges we proceed with remaining active clients
            if len(self.registered_clients) >= self.min_clients and not self.training_started:
                print(f"\nMinimum clients reached ({len(self.registered_clients)} >= {self.min_clients}). Starting federated learning in 2 seconds...")
                # Start training in a separate thread after a short delay
                threading.Timer(2.0, self.start_training).start()
            
            return federated_learning_pb2.RegistrationResponse(
                success=True,
                message=f"Client {request.client_id} registered successfully"
            )
    
    def mark_client_converged(self, client_id):
        """Remove converged client from active federation."""
        do_aggregate_metrics = False
        do_aggregate_updates = False
        with self.lock:
            if client_id in self.active_clients:
                self.active_clients.remove(client_id)
                self.client_updates.pop(client_id, None)
                self.client_metrics.pop(client_id, None)
                self.clients_evaluated.discard(client_id)
                print(f"Client {client_id} converged and disconnected. "
                      f"Active clients remaining: {len(self.active_clients)}")
                if not self.active_clients:
                    self.training_complete = True
                    self.evaluation_phase = False
                    self.converged = True
                    print("All clients converged. Ending training.")
                else:
                    # Re-check: remaining active clients may have already sent metrics/updates
                    if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                        do_aggregate_metrics = True
                    if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                        do_aggregate_updates = True
        if do_aggregate_metrics:
            threading.Thread(target=self.aggregate_metrics).start()
        if do_aggregate_updates:
            threading.Thread(target=self.aggregate_updates).start()

    def GetTrainingConfig(self, request, context):
        """Send training configuration to client"""
        return federated_learning_pb2.TrainingConfig(
            batch_size=self.training_config['batch_size'],
            local_epochs=self.training_config['local_epochs']
        )
    
    def GetGlobalModel(self, request, context):
        """Send global model weights and configuration to client (chunked when > 4 MB)."""
        with self.lock:
            if self.global_weights is None:
                print(f"Client {request.client_id}: Model not yet initialized")
                return federated_learning_pb2.GlobalModel(
                    round=0,
                    weights=b'',
                    available=False,
                    model_config='',
                    chunk_index=0,
                    total_chunks=1
                )
            
            # Compress or serialize global weights
            if self.quantization_handler is not None:
                compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
                stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
                print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
                serialized_weights = pickle.dumps(compressed_data)
            else:
                serialized_weights = self.serialize_weights(self.global_weights)
            model_config_json = json.dumps(self.model_config)
            total_size = len(serialized_weights)
            
            # Chunk if payload exceeds limit (stay under 4 MB per message)
            if total_size <= GRPC_CHUNK_SIZE:
                # Single message
                print(f"Client {request.client_id}: Sending global model (round {self.current_round}, {total_size/1024:.2f} KB)")
                return federated_learning_pb2.GlobalModel(
                    round=self.current_round,
                    weights=serialized_weights,
                    available=True,
                    model_config=model_config_json,
                    chunk_index=0,
                    total_chunks=1
                )
            
            # Chunked transfer
            chunk_index = getattr(request, 'chunk_index', 0) or 0
            chunks = []
            for i in range(0, total_size, GRPC_CHUNK_SIZE):
                chunks.append(serialized_weights[i:i + GRPC_CHUNK_SIZE])
            total_chunks = len(chunks)
            if chunk_index < 0 or chunk_index >= total_chunks:
                return federated_learning_pb2.GlobalModel(
                    round=self.current_round,
                    weights=b'',
                    available=False,
                    model_config='',
                    chunk_index=chunk_index,
                    total_chunks=total_chunks
                )
            chunk_data = chunks[chunk_index]
            # Only send model_config on first chunk
            cfg = model_config_json if chunk_index == 0 else ''
            if chunk_index == 0:
                print(f"Client {request.client_id}: Sending global model (round {self.current_round}, {total_size/1024:.2f} KB in {total_chunks} chunks)")
            return federated_learning_pb2.GlobalModel(
                round=self.current_round,
                weights=chunk_data,
                available=True,
                model_config=cfg,
                chunk_index=chunk_index,
                total_chunks=total_chunks
            )
    
    def SendModelUpdate(self, request, context):
        """Receive model update from client"""
        recv_start_cpu = time.perf_counter() if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1" else None
        with self.lock:
            client_id = request.client_id
            round_num = request.round
            metrics = dict(request.metrics)

            # Check convergence first: accept convergence even if round advanced (late-arriving message)
            converged_flag = float(metrics.get('client_converged', 0.0)) >= 1.0
            if converged_flag:
                # Allow convergence for current or previous round so we don't reject when server already advanced
                if round_num <= self.current_round and client_id in self.active_clients:
                    self.mark_client_converged(client_id)
                    return federated_learning_pb2.UpdateResponse(
                        success=True,
                        message=f"Client {client_id} convergence acknowledged"
                    )
                elif client_id not in self.active_clients:
                    return federated_learning_pb2.UpdateResponse(
                        success=True,
                        message=f"Client {client_id} already inactive"
                    )
                # else: same round, process below (should not store update)

            if round_num != self.current_round:
                return federated_learning_pb2.UpdateResponse(
                    success=False,
                    message=f"Round mismatch: expected {self.current_round}, got {round_num}"
                )
            
            total_chunks = getattr(request, 'total_chunks', 1) or 1
            chunk_index = getattr(request, 'chunk_index', 0) or 0
            
            # Chunked update: accumulate chunks until complete, then process
            if total_chunks > 1:
                key = (client_id, round_num)
                if key not in self._update_chunks:
                    self._update_chunks[key] = {'chunks': {}, 'num_samples': 0, 'metrics': {}}
                buf = self._update_chunks[key]
                buf['chunks'][chunk_index] = request.weights if request.weights else b''
                if chunk_index == 0:
                    buf['num_samples'] = request.num_samples
                    buf['metrics'] = metrics
                if len(buf['chunks']) < total_chunks:
                    return federated_learning_pb2.UpdateResponse(
                        success=True,
                        message=f"Chunk {chunk_index + 1}/{total_chunks} received for round {round_num}"
                    )
                # Reassemble
                serialized_weights = b''.join(buf['chunks'][i] for i in range(total_chunks))
                num_samples = buf['num_samples']
                metrics = buf['metrics']
                del self._update_chunks[key]
                # Fall through to deserialize and store (use serialized_weights as request.weights)
            else:
                serialized_weights = request.weights
                num_samples = request.num_samples
            
            # Deserialize weights
            if serialized_weights:
                if self.quantization_handler is not None:
                    try:
                        candidate = pickle.loads(serialized_weights)
                        if isinstance(candidate, dict) and 'compressed_data' in candidate:
                            weights = self.quantization_handler.decompress_client_update(client_id, candidate)
                            print(f"Server: Received and decompressed update from client {client_id}")
                        else:
                            weights = candidate
                    except Exception:
                        weights = self.deserialize_weights(serialized_weights)
                else:
                    weights = self.deserialize_weights(serialized_weights)
            else:
                weights = None
            
            if recv_start_cpu is not None:
                O_recv = time.perf_counter() - recv_start_cpu
                recv_end_ts = time.time()
                metrics_pre = metrics
                send_start_ts = metrics_pre.get("diagnostic_send_start_ts", recv_end_ts)
                print(f"FL_DIAG client_id={client_id} O_recv={O_recv:.9f} recv_end_ts={recv_end_ts:.9f} send_start_ts={send_start_ts:.9f}")
            
            self.client_updates[client_id] = {
                'weights': weights,
                'num_samples': num_samples,
                'metrics': metrics
            }
            
            print(f"Received update from client {client_id} for round {round_num}")
            print(f"  Training - Loss: {metrics.get('loss', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Progress: {len(self.client_updates)}/{len(self.active_clients)} active clients")
            
            if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                print(f"\nAll clients submitted updates for round {self.current_round}")
                threading.Thread(target=self.aggregate_updates).start()
            
            return federated_learning_pb2.UpdateResponse(
                success=True,
                message=f"Update received for round {round_num}"
            )
    
    def SendMetrics(self, request, context):
        """Receive evaluation metrics from client"""
        with self.lock:
            client_id = request.client_id
            round_num = request.round

            if client_id not in self.active_clients:
                return federated_learning_pb2.MetricsResponse(
                    success=True,
                    message=f"Client {client_id} already inactive"
                )
            
            # Proto Metrics has scalar loss/accuracy fields. Keep a safe fallback
            # if an old client still sends a metrics map message type.
            if hasattr(request, 'metrics'):
                metrics = dict(request.metrics)
                loss_value = float(metrics.get('loss', 0.0))
                acc_value = float(metrics.get('accuracy', 0.0))
            else:
                loss_value = float(getattr(request, 'loss', 0.0))
                acc_value = float(getattr(request, 'accuracy', 0.0))
            
            self.client_metrics[client_id] = {
                'loss': loss_value,
                'accuracy': acc_value,
                'num_samples': request.num_samples,
                'battery_soc': float(getattr(request, 'battery_soc', 1.0)),
                'round_time_sec': float(getattr(request, 'round_time_sec', 0.0)),
            }
            self.clients_evaluated.add(client_id)
            
            # Process convergence in same RPC (so active_clients is updated before "all evaluated" check)
            client_converged = float(getattr(request, 'client_converged', 0.0)) or 0.0
            all_just_converged = False
            if client_converged >= 1.0 and client_id in self.active_clients:
                self.active_clients.remove(client_id)
                self.client_updates.pop(client_id, None)
                # Keep last client's metrics so aggregate_metrics can record final round and then run completion (save, plot, stop server)
                if self.active_clients:
                    self.client_metrics.pop(client_id, None)
                self.clients_evaluated.discard(client_id)
                print(f"Client {client_id} converged (signalled in metrics). Active clients remaining: {len(self.active_clients)}")
                if not self.active_clients:
                    self.training_complete = True
                    self.evaluation_phase = False
                    self.converged = True
                    self.convergence_time = time.time() - self.start_time if self.start_time else None
                    print("All clients converged. Ending training.")
                    all_just_converged = True
            
            print(f"Received evaluation metrics from client {client_id}")
            print(f"  Loss: {loss_value:.4f}, Accuracy: {acc_value:.4f}")
            if client_converged >= 1.0:
                print(f"  Client {client_id} signalled convergence in this metrics message")
            print(f"  Progress: {len(self.clients_evaluated)}/{len(self.active_clients)} active clients evaluated")
            
            # Proceed when every active (non-converged) client has evaluated
            if self.active_clients.issubset(self.clients_evaluated) and len(self.active_clients) > 0:
                print(f"\nAll clients completed evaluation for round {self.current_round}")
                # Aggregate metrics and start next round
                threading.Thread(target=self.aggregate_metrics).start()
            # When last client just converged, run aggregate_metrics to record final round and run completion (save, plot, stop server)
            elif all_just_converged and self.client_metrics:
                threading.Thread(target=self.aggregate_metrics).start()
            
            return federated_learning_pb2.MetricsResponse(
                success=True,
                message="Metrics received"
            )
    
    def CheckTrainingStatus(self, request, context):
        """Let clients check if training should start"""
        return federated_learning_pb2.TrainingStatus(
            should_train=self.training_started,
            current_round=self.current_round,
            should_evaluate=self.evaluation_phase,
            is_complete=self.training_complete
        )
    
    def start_training(self):
        """Start the federated learning process"""
        with self.lock:
            if self.training_started:
                return
            
            self.training_started = True
            self.start_time = time.time()
            self.round_start_time = time.time()
            self.current_round = 1
        
        print("\n" + "="*70)
        print("STARTING FEDERATED LEARNING")
        print("="*70)
        print(f"Number of clients: {self.num_clients}")
        print(f"Maximum rounds: {self.num_rounds}")
        print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}")
        print(f"Patience: {CONVERGENCE_PATIENCE} rounds")
        print(f"Global model ready: {self.global_weights is not None}")
        print(f"Model layers: {len(self.global_weights) if self.global_weights else 0}")
        print("="*70 + "\n")
        print("Clients can now call GetGlobalModel to fetch initial weights...")
    
    def aggregate_updates(self):
        """Aggregate client model updates using FedAvg"""
        if not self.client_updates:
            return
        print(f"\n{'='*70}")
        print(f"AGGREGATING UPDATES - Round {self.current_round}")
        print(f"{'='*70}")
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in self.client_updates.values())
        
        # Weighted average of weights
        new_weights = []
        for i in range(len(self.global_weights)):
            layer_weights = []
            for client_id, update in self.client_updates.items():
                weight = update['weights'][i]
                num_samples = update['num_samples']
                layer_weights.append(weight * num_samples / total_samples)
            new_weights.append(np.sum(layer_weights, axis=0))
        
        # Update global weights
        with self.lock:
            self.global_weights = new_weights
            # Clear updates for next round
            self.client_updates.clear()
            # Enter evaluation phase
            self.evaluation_phase = True
        
        print(f"Global model updated for round {self.current_round}")
        print(f"Clients should now evaluate the model\n")
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics and advance rounds."""
        if not self.client_metrics:
            return
        if self.round_start_time is not None:
            self.ROUND_TIMES.append(time.time() - self.round_start_time)
        socs = [m.get('battery_soc', 1.0) for m in self.client_metrics.values()]
        avg_soc = sum(socs) / len(socs) if socs else 1.0
        self.BATTERY_CONSUMPTION.append(1.0 - avg_soc)
        # Calculate weighted average metrics
        total_samples = sum(m['num_samples'] for m in self.client_metrics.values())
        
        avg_loss = sum(m['loss'] * m['num_samples'] for m in self.client_metrics.values()) / total_samples
        avg_accuracy = sum(m['accuracy'] * m['num_samples'] for m in self.client_metrics.values()) / total_samples
        
        # Store metrics
        self.ROUNDS.append(self.current_round)
        self.LOSS.append(avg_loss)
        self.ACCURACY.append(avg_accuracy)
        
        print(f"\n{'='*70}")
        print(f"ROUND {self.current_round} SUMMARY")
        print(f"{'='*70}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        
        # Check stopping criteria
        should_stop = False
        stop_reason = ""

        if len(self.active_clients) == 0:
            should_stop = True
            stop_reason = "All clients converged locally"
            self.converged = True
            self.convergence_time = time.time() - self.start_time
        elif self.current_round >= self.num_rounds:
            should_stop = True
            stop_reason = f"Maximum rounds ({self.num_rounds}) reached"
        
        if should_stop:
            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETE: {stop_reason}")
            print(f"{'='*70}")
            print(f"Final Loss: {avg_loss:.4f}")
            print(f"Final Accuracy: {avg_accuracy:.4f}")
            print(f"Total rounds: {self.current_round}")
            print(f"Total time: {time.time() - self.start_time:.2f} seconds")
            if self.converged:
                print(f"Convergence time: {self.convergence_time:.2f} seconds")
            print(f"{'='*70}\n")
            
            with self.lock:
                self.training_complete = True
                self.evaluation_phase = False
            
            # Save results and plot
            self.save_results()
            self.plot_results()
            # Stop gRPC server so process exits (e.g. network_simulator / evaluate_all can continue)
            if self._grpc_server is not None:
                self._grpc_server.stop(0)
        else:
            print(f"{'='*70}\n")
            # Continue to next round
            with self.lock:
                self.current_round += 1
                self.round_start_time = time.time()
                self.client_metrics.clear()
                self.clients_evaluated.clear()
                self.evaluation_phase = False
            
            print(f"Starting Round {self.current_round}...\n")
    
    def save_results(self):
        """Save training results to JSON file"""
        results = {
            'rounds': self.ROUNDS,
            'loss': self.LOSS,
            'accuracy': self.ACCURACY,
            'round_times_seconds': getattr(self, 'ROUND_TIMES', []),
            'battery_consumption': getattr(self, 'BATTERY_CONSUMPTION', []),
            'converged': self.converged,
            'convergence_time': self.convergence_time if self.converged else None,
            'total_time': time.time() - self.start_time,
            'final_loss': self.LOSS[-1] if self.LOSS else None,
            'final_accuracy': self.ACCURACY[-1] if self.ACCURACY else None,
            'num_clients': self.num_clients,
            'config': self.training_config
        }
        
        # Create results directory if it doesn't exist
        results_dir = get_experiment_results_dir("emotion", "grpc")
        
        # Save to JSON
        filepath = results_dir / 'grpc_training_results.json'
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def plot_results(self):
        """Plot training metrics: battery, round/convergence time, loss & accuracy."""
        if not self.ROUNDS:
            print("No data to plot")
            return
        results_dir = get_experiment_results_dir("emotion", "grpc")
        rounds = self.ROUNDS
        n = len(rounds)
        conv_time = self.convergence_time if self.convergence_time is not None else (time.time() - self.start_time if self.start_time else 0)

        # 1) Battery consumption
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        bc = self.BATTERY_CONSUMPTION if len(self.BATTERY_CONSUMPTION) >= n else (self.BATTERY_CONSUMPTION + [0.0] * max(0, n - len(self.BATTERY_CONSUMPTION)))[:n]
        if bc:
            ax1.plot(rounds, [c * 100 for c in bc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Battery consumption (%)', fontsize=12)
        ax1.set_title('Battery consumption till end of FL training (gRPC)', fontsize=14)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        fig1.savefig(results_dir / 'grpc_battery_consumption.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Battery plot saved to {results_dir / 'grpc_battery_consumption.png'}")

        # 2) Time per round and convergence time
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        rt = self.ROUND_TIMES if len(self.ROUND_TIMES) >= n else (self.ROUND_TIMES + [0.0] * max(0, n - len(self.ROUND_TIMES)))[:n]
        if rt:
            ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2, label=f'Total convergence time: {conv_time:.1f} s')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Time (s)', fontsize=12)
        ax2.set_title('Time per round and total convergence time (gRPC)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(results_dir / 'grpc_round_and_convergence_time.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Time plot saved to {results_dir / 'grpc_round_and_convergence_time.png'}")

        # 3) Loss and Accuracy
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
        ax3a.plot(rounds, self.LOSS, 'b-', linewidth=2, marker='o')
        ax3a.set_xlabel('Round', fontsize=12)
        ax3a.set_ylabel('Loss', fontsize=12)
        ax3a.set_title('Federated Learning - Loss Over Rounds (gRPC)', fontsize=14, fontweight='bold')
        ax3a.grid(True, alpha=0.3)
        ax3b.plot(rounds, self.ACCURACY, 'g-', linewidth=2, marker='s')
        ax3b.set_xlabel('Round', fontsize=12)
        ax3b.set_ylabel('Accuracy', fontsize=12)
        ax3b.set_title('Federated Learning - Accuracy Over Rounds (gRPC)', fontsize=14, fontweight='bold')
        ax3b.grid(True, alpha=0.3)
        fig3.tight_layout()
        filepath = results_dir / 'grpc_training_plot.png'
        fig3.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Plot saved to {filepath}")
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            plt.close('all')
        else:
            plt.show(block=False)


# INT_MAX ms (~24 days) = effectively disable keepalive (gRPC C-core treats 0 as default)
GRPC_KEEPALIVE_DISABLED_MS = 2147483647
# Max message size (4 MB); larger models are sent in chunks via chunk_index/total_chunks
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))
# Chunk size for global model when serialized size exceeds limit (leave room for proto framing)
GRPC_CHUNK_SIZE = GRPC_MAX_MESSAGE_BYTES - 4096


def serve():
    """Start the gRPC server"""
    # 4 MB max message; global model is chunked when larger
    # Keepalive effectively disabled (INT_MAX) so poor/very-poor networks never get GOAWAY ping_timeout
    options = [
        ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
        ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
        ('grpc.keepalive_time_ms', GRPC_KEEPALIVE_DISABLED_MS),
        ('grpc.keepalive_timeout_ms', GRPC_KEEPALIVE_DISABLED_MS),
        ('grpc.http2.max_ping_strikes', 0),  # do not close on client pings
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    servicer = FederatedLearningServicer(MIN_CLIENTS, NUM_ROUNDS, max_clients=MAX_CLIENTS, grpc_server=server)
    federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    
    server.add_insecure_port(f'{GRPC_HOST}:{GRPC_PORT}')
    server.start()
    
    print(f"\n{'='*70}")
    print(f"Federated Learning gRPC Server - Emotion Recognition")
    print(f"{'='*70}")
    print(f"Server listening on {GRPC_HOST}:{GRPC_PORT}")
    print(f"Waiting for at least {MIN_CLIENTS} clients to register...")
    print(f"{'='*70}\n")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(0)


if __name__ == '__main__':
    serve()
