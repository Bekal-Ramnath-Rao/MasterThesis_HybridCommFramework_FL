import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import grpc
from concurrent import futures
import threading
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

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
    from pruning_server import ServerPruning
    from pruning_client import PruningConfig
    PRUNING_AVAILABLE = True
except ImportError:
    print("Warning: Pruning module not available")
    PRUNING_AVAILABLE = False


# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# Server Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "0.0.0.0")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
# Payload size limit: 4 MB (gRPC protocol constraint)
MAX_PAYLOAD_GRPC = 4 * 1024 * 1024  # 4 MB
GRPC_OPTIONS = [
    ('grpc.max_send_message_length', MAX_PAYLOAD_GRPC),
    ('grpc.max_receive_message_length', MAX_PAYLOAD_GRPC),
]
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence
NETWORK_SCENARIO = os.getenv("NETWORK_SCENARIO", "excellent")  # Network scenario for result filename

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
from battery_results_agg import avg_battery_model_drain_fraction

# Convergence Settings (primary stopping criterion)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
STOP_ON_CLIENT_CONVERGENCE = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").lower() in ("1", "true", "yes")


class FederatedLearningServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
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
        self.clients_evaluated = set()  # Track which clients have evaluated this round
        self.global_weights = None
        self.lock = threading.Lock()
        
        # Metrics storage
        self.MSE = []
        self.MAE = []
        self.MAPE = []
        self.LOSS = []
        self.ROUNDS = []
        self.AVG_TRAINING_TIME_SEC = []
        self.AVG_BATTERY_SOC = []
        self.BATTERY_CONSUMPTION = []
        self.BATTERY_MODEL_CONSUMPTION = []
        self.ROUND_TIMES = []
        self.round_start_time = None
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.start_time = None
        self.convergence_time = None
        
        # Initialize quantization handler
        use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() == "true"
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
        self.training_config = {
            "batch_size": 16,
            "local_epochs": 20
        }
        
        # Status flags
        self.training_started = False
        self.training_complete = False
        self.evaluation_phase = False

        # Optional compression handlers
        uq_env = os.getenv("USE_QUANTIZATION", "false")
        use_quantization = uq_env.lower() in ("true", "1", "yes", "y")
        if use_quantization and QUANTIZATION_AVAILABLE:
            self.quantization_handler = ServerQuantizationHandler(QuantizationConfig())
            print("Server: Quantization enabled")
        else:
            self.quantization_handler = None
            print("Server: Quantization disabled")

        up_env = os.getenv("USE_PRUNING", "false")
        use_pruning = up_env.lower() in ("true", "1", "yes", "y")
        if use_pruning and PRUNING_AVAILABLE:
            self.pruning_handler = ServerPruning(PruningConfig())
            print("Server: Pruning enabled")
        else:
            self.pruning_handler = None
            print("Server: Pruning disabled")
    
    def initialize_global_model(self):
        """Initialize the global model structure (LSTM for FL)"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM
        
        # Create the same LSTM model structure as clients
        # Input shape: (1, 4) - 1 time step, 4 features
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(1, 4)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', 
                     metrics=['mse', 'mae', 'mape'])
        
        # Get initial weights
        self.global_weights = model.get_weights()
        
        print("\nGlobal model initialized with random weights")
        print(f"Model architecture: LSTM(50) -> Dense(1)")
        print(f"Number of weight layers: {len(self.global_weights)}")
    
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
            client_id = request.client_id
            self.registered_clients.add(client_id)
            self.active_clients.add(client_id)
            print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))

        # Start when we have at least min_clients; after one converges we proceed with remaining active clients
        if len(self.registered_clients) >= self.min_clients and not self.training_started:
            print("\nMinimum clients reached. Distributing initial global model...\n")
            self.training_started = True
            self.current_round = 1

            print(f"\n{'='*70}")
            print(f"Distributing Initial Global Model")
            print(f"{'='*70}\n")
            print("Initial global model ready for clients")

            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"{'='*70}\n")

        return federated_learning_pb2.RegistrationResponse(
            success=True,
            message=f"Client {client_id} registered successfully"
        )
    
    def mark_client_converged(self, client_id):
        """Remove converged client from active federation; proceed with remaining clients."""
        if not STOP_ON_CLIENT_CONVERGENCE:
            # Fixed-round mode: ignore convergence signals for stopping/disconnecting.
            # Client implementations should also respect STOP_ON_CLIENT_CONVERGENCE and keep training.
            print(f"Ignoring convergence signal from client {client_id} (STOP_ON_CLIENT_CONVERGENCE=false)")
            return
        do_aggregate_metrics = False
        do_aggregate_models = False
        with self.lock:
            if client_id in self.active_clients:
                self.active_clients.discard(client_id)
                self.client_updates.pop(client_id, None)
                self.client_metrics.pop(client_id, None)
                self.clients_evaluated.discard(client_id)
                print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
                if not self.active_clients:
                    self.training_complete = True
                    self.converged = True
                    print("All clients converged. Ending training.")
                    self.convergence_time = time.time() - self.start_time if self.start_time else 0
                    self.plot_results()
                    self.save_results()
                else:
                    # Re-check: remaining active clients may have already sent updates/metrics
                    if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                        do_aggregate_metrics = True
                    if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                        do_aggregate_models = True
        if do_aggregate_metrics:
            self.aggregate_metrics()
            self.continue_training()
        if do_aggregate_models:
            self.aggregate_models()
    
    def GetTrainingConfig(self, request, context):
        """Return training configuration"""
        return federated_learning_pb2.TrainingConfig(
            batch_size=self.training_config["batch_size"],
            local_epochs=self.training_config["local_epochs"]
        )
    
    def CheckTrainingStatus(self, request, context):
        """Check if client should train or evaluate"""
        with self.lock:
            client_id = request.client_id
            client_round = request.current_round
            
            # Check if training is complete
            if self.training_complete:
                return federated_learning_pb2.TrainingStatus(
                    should_train=False,
                    current_round=self.current_round,
                    should_evaluate=False,
                    is_complete=True
                )
            
            # Check if client should start training
            if self.training_started and client_round == 0 and self.current_round == 1:
                return federated_learning_pb2.TrainingStatus(
                    should_train=True,
                    current_round=self.current_round,
                    should_evaluate=False,
                    is_complete=False
                )
            
            # Check if client should evaluate (after global model is ready)
            if self.evaluation_phase and client_round == self.current_round and self.global_weights is not None:
                # Only tell client to evaluate if it hasn't already
                if client_id not in self.clients_evaluated:
                    return federated_learning_pb2.TrainingStatus(
                        should_train=False,
                        current_round=self.current_round,
                        should_evaluate=True,
                        is_complete=False
                    )
            
            # Check if client should train next round
            if not self.evaluation_phase and client_round < self.current_round:
                return federated_learning_pb2.TrainingStatus(
                    should_train=True,
                    current_round=self.current_round,
                    should_evaluate=False,
                    is_complete=False
                )
            
            # No action needed
            return federated_learning_pb2.TrainingStatus(
                should_train=False,
                current_round=self.current_round,
                should_evaluate=False,
                is_complete=False
            )
    
    def GetGlobalModel(self, request, context):
        """Send global model to client"""
        with self.lock:
            # Wait for all clients to register before distributing model
            if not self.training_started:
                return federated_learning_pb2.GlobalModel(
                    round=0,
                    weights=b'',
                    available=False,
                    model_config=""
                )
            
            # Record training start time on first client request after all registered
            if self.start_time is None and request.round == 0:
                self.start_time = time.time()
                print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if self.global_weights is not None:
                # For initial distribution (client_round=0), send round 0
                # For updates after aggregation, send current_round
                round_to_send = 0 if request.round == 0 and self.training_started else self.current_round
                
                # Always include model_config for late-joining clients
                model_config = {
                    "architecture": "LSTM",
                    "layers": [
                        {
                            "type": "LSTM",
                            "units": 50,
                            "activation": "relu",
                            "input_shape": [1, 4]
                        },
                        {
                            "type": "Dense",
                            "units": 1
                        }
                    ],
                    "compile_config": {
                        "loss": "mean_squared_error",
                        "optimizer": "adam",
                        "metrics": ["mse", "mae", "mape"]
                    }
                }
                model_config_json = json.dumps(model_config)
                
                # Compress or serialize global weights (prune -> quantize order if both enabled).
                # If quantization is enabled, send quantized payload (may already apply pruning within handler).
                if self.quantization_handler is not None:
                    compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
                    stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
                    print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
                    # Send pickled compressed dict so clients receive metadata
                    serialized_weights = pickle.dumps(compressed_data)
                elif self.pruning_handler is not None:
                    pruned_bytes, _ = self.pruning_handler.compress_for_broadcast(self.global_weights)
                    serialized_weights = pickle.dumps({"pruned_data": pruned_bytes})
                else:
                    serialized_weights = self.serialize_weights(self.global_weights)
                
                return federated_learning_pb2.GlobalModel(
                    round=round_to_send,
                    weights=serialized_weights,
                    available=True,
                    model_config=model_config_json
                )
            else:
                return federated_learning_pb2.GlobalModel(
                    round=self.current_round,
                    weights=b'',
                    available=False
                )
    
    def SendModelUpdate(self, request, context):
        """Receive model update from client"""
        with self.lock:
            client_id = request.client_id
            round_num = request.round
            metrics = dict(request.metrics)
            if STOP_ON_CLIENT_CONVERGENCE and float(metrics.get('client_converged', 0.0)) >= 1.0:
                self.mark_client_converged(client_id)
                return federated_learning_pb2.UpdateResponse(
                    success=True,
                    message=f"Client {client_id} convergence acknowledged"
                )
            if client_id not in self.active_clients:
                return federated_learning_pb2.UpdateResponse(
                    success=True,
                    message=f"Client {client_id} already inactive"
                )
            if round_num == self.current_round:
                # Decompress or deserialize client weights.
                # Priority: quantized dict -> pruned_data dict -> raw weights list.
                weights = []
                try:
                    candidate = pickle.loads(request.weights) if request.weights else None
                    if isinstance(candidate, dict) and 'compressed_data' in candidate and self.quantization_handler is not None:
                        # Keep quantized end-to-end: do NOT decompress/dequantize on server.
                        self.client_updates[client_id] = {
                            'compressed_data': candidate,
                            'num_samples': request.num_samples,
                            'metrics': metrics
                        }
                        print(f"Server: Received quantized update from client {request.client_id} (kept quantized)")
                        candidate = None
                    elif isinstance(candidate, dict) and 'quantization_params' in candidate and self.quantization_handler is not None:
                        self.client_updates[client_id] = {
                            'compressed_data': candidate,
                            'num_samples': request.num_samples,
                            'metrics': metrics
                        }
                        print(f"Server: Received quantized update from client {request.client_id} (kept quantized)")
                        candidate = None
                    elif isinstance(candidate, dict) and 'pruned_data' in candidate and self.pruning_handler is not None:
                        weights = self.pruning_handler.decompress_client_update(candidate['pruned_data'])
                        print(f"Server: Received and decompressed pruned update from client {request.client_id}")
                    else:
                        weights = candidate if candidate is not None else self.deserialize_weights(request.weights)
                except Exception:
                    try:
                        weights = self.deserialize_weights(request.weights)
                    except Exception:
                        weights = []

                if client_id not in self.client_updates:
                    self.client_updates[client_id] = {
                        'weights': weights,
                        'num_samples': request.num_samples,
                        'metrics': metrics
                    }
                
                print(f"Received update from client {client_id} "
                      f"({len(self.client_updates)}/{len(self.active_clients)})")
                
                if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                    self.aggregate_models()
                
                return federated_learning_pb2.UpdateResponse(
                    success=True,
                    message="Model update received"
                )
            else:
                return federated_learning_pb2.UpdateResponse(
                    success=False,
                    message=f"Round mismatch: received {round_num}, current {self.current_round}"
                )
    
    def SendMetrics(self, request, context):
        """Receive evaluation metrics from client"""
        with self.lock:
            client_id = request.client_id
            round_num = request.round
            metrics = dict(request.metrics) if hasattr(request, 'metrics') and request.metrics else {}
            # Scalar Metrics fields (when clients use federated_learning_pb2.Metrics)
            for attr, key in (
                ('loss', 'loss'),
                ('accuracy', 'accuracy'),
                ('battery_soc', 'battery_soc'),
                ('training_time_sec', 'training_time_sec'),
                ('uplink_model_comm_sec', 'uplink_model_comm_sec'),
                ('round_time_sec', 'round_time_sec'),
            ):
                if hasattr(request, attr):
                    v = getattr(request, attr, None)
                    if v is not None and key not in metrics:
                        try:
                            metrics[key] = float(v)
                        except (TypeError, ValueError):
                            pass
            if client_id not in self.active_clients:
                return federated_learning_pb2.MetricsResponse(
                    success=True,
                    message=f"Client {client_id} already inactive"
                )
            if STOP_ON_CLIENT_CONVERGENCE and float(metrics.get('client_converged', 0.0)) >= 1.0:
                self.mark_client_converged(client_id)
                return federated_learning_pb2.MetricsResponse(
                    success=True,
                    message="Convergence acknowledged"
                )
            if round_num == self.current_round:
                if client_id in self.clients_evaluated:
                    return federated_learning_pb2.MetricsResponse(
                        success=True,
                        message="Metrics already received for this round"
                    )
                self.client_metrics[client_id] = {
                    'num_samples': request.num_samples,
                    'metrics': metrics
                }
                self.clients_evaluated.add(client_id)
                
                print(f"Received metrics from client {client_id} "
                      f"({len(self.client_metrics)}/{len(self.active_clients)})")
                
                if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                    self.aggregate_metrics()
                    self.continue_training()
                
                return federated_learning_pb2.MetricsResponse(
                    success=True,
                    message="Metrics received"
                )
            else:
                return federated_learning_pb2.MetricsResponse(
                    success=False,
                    message=f"Round mismatch: received {round_num}, current {self.current_round}"
                )
    
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
            print(f"Aggregated global model from round {self.current_round} (dequantize→FedAvg→requantize)")
            print(f"Global model ready for clients\n")
            self.evaluation_phase = True
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

        # Optionally apply server-side pruning to global weights (pruning-only mode or alongside quantization handler)
        if self.pruning_handler is not None and (self.quantization_handler is None):
            self.global_weights = self.pruning_handler.pruning_engine.prune_weights(
                self.global_weights, step=self.current_round
            )
        
        print(f"Aggregated global model from round {self.current_round}")
        print(f"Global model ready for clients\n")
        
        # Switch to evaluation phase
        self.evaluation_phase = True
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics from all clients"""
        print(f"\nAggregating metrics from {len(self.client_metrics)} clients...")
        
        # Calculate total samples
        total_samples = sum(metric['num_samples'] 
                          for metric in self.client_metrics.values())
        
        # Weighted average of metrics
        aggregated_mse = sum(metric['metrics'].get('mse', 0.0) * metric['num_samples']
                            for metric in self.client_metrics.values()) / total_samples
        
        aggregated_mae = sum(metric['metrics'].get('mae', 0.0) * metric['num_samples']
                            for metric in self.client_metrics.values()) / total_samples
        
        aggregated_mape = sum(metric['metrics'].get('mape', 0.0) * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        aggregated_loss = sum(metric['metrics'].get('loss', 0.0) * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        avg_training_time = (
            sum(
                float(metric['metrics'].get('training_time_sec', 0.0)) * metric['num_samples']
                for metric in self.client_metrics.values()
            )
            / total_samples
            if total_samples
            else 0.0
        )
        socs = [float(m['metrics'].get('battery_soc', 1.0)) for m in self.client_metrics.values()]
        avg_soc = sum(socs) / len(socs) if socs else 1.0
        self.AVG_TRAINING_TIME_SEC.append(float(avg_training_time))
        self.AVG_BATTERY_SOC.append(float(avg_soc))
        self.BATTERY_CONSUMPTION.append(1.0 - avg_soc)
        self.BATTERY_MODEL_CONSUMPTION.append(avg_battery_model_drain_fraction(self.client_metrics))
        if getattr(self, 'round_start_time', None) is not None:
            self.ROUND_TIMES.append(time.time() - self.round_start_time)
        
        # Store metrics
        self.MSE.append(aggregated_mse)
        self.MAE.append(aggregated_mae)
        self.MAPE.append(aggregated_mape)
        self.LOSS.append(aggregated_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"\nRound {self.current_round} Aggregated Metrics:")
        print(f"  Loss: {aggregated_loss:.4f}")
        print(f"  MSE: {aggregated_mse:.4f}")
        print(f"  MAE: {aggregated_mae:.4f}")
        print(f"  MAPE: {aggregated_mape:.4f}")
        print(f"  Avg training time (sample-weighted): {avg_training_time:.3f} s")
        print(f"  Avg battery SoC: {avg_soc:.4f}\n")
    
    def continue_training(self):
        """Continue to next round or finish training"""
        # Clear updates and metrics for next round
        self.client_updates.clear()
        self.client_metrics.clear()
        self.clients_evaluated.clear()
        self.evaluation_phase = False
        
        # Stop only when no active clients remain or max rounds reached (no server-side convergence)
        if len(self.active_clients) == 0:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            # If convergence-stopping is disabled, this indicates clients disconnected unexpectedly.
            self.converged = bool(STOP_ON_CLIENT_CONVERGENCE)
            self.training_complete = True
            if STOP_ON_CLIENT_CONVERGENCE:
                print("\n" + "="*70)
                print("All clients converged locally. Training complete.")
                print("="*70 + "\n")
            else:
                print("\n" + "="*70)
                print("All clients became inactive. Training complete (fixed-round mode).")
                print("="*70 + "\n")
            self.plot_results()
            self.save_results()
            return
        
        # Check if more rounds needed
        if self.current_round < self.num_rounds:
            self.current_round += 1
            
            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"{'='*70}\n")
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            self.training_complete = True
            self.plot_results()
            self.save_results()
    
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
        """Plot battery consumption, round/convergence time, loss, and regression metrics."""
        results_dir = get_experiment_results_dir("temperature", "grpc")
        rounds = self.ROUNDS
        n = len(rounds)
        conv_time = self.convergence_time if self.convergence_time is not None else (
            time.time() - self.start_time if self.start_time else 0)

        # 1) Battery
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        bc = (self.BATTERY_CONSUMPTION + [0.0] * max(0, n - len(self.BATTERY_CONSUMPTION)))[:n] \
            if getattr(self, 'BATTERY_CONSUMPTION', []) else [0.0] * n
        if bc:
            ax1.plot(rounds, [c * 100 for c in bc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round'); ax1.set_ylabel('Battery consumption (%)')
        ax1.set_title('gRPC (temperature): Battery consumption over FL rounds')
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout(); fig1.savefig(results_dir / 'grpc_battery_consumption.png', dpi=300, bbox_inches='tight'); plt.close(fig1)

        # 2) Time per round and convergence
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        rt = (self.ROUND_TIMES + [0.0] * max(0, n - len(self.ROUND_TIMES)))[:n] \
            if getattr(self, 'ROUND_TIMES', []) else [0.0] * n
        if rt:
            ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2,
                    label=f'Total convergence: {conv_time:.1f} s')
        ax2.set_xlabel('Round'); ax2.set_ylabel('Time (s)')
        ax2.set_title('gRPC (temperature): Time per round and convergence')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        fig2.tight_layout(); fig2.savefig(results_dir / 'grpc_round_and_convergence_time.png', dpi=300, bbox_inches='tight'); plt.close(fig2)

        # 3) Loss
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(rounds, self.LOSS, 'b-', marker='o', linewidth=2, markersize=6)
        ax3.set_xlabel('Round'); ax3.set_ylabel('Loss (MSE)')
        ax3.set_title('gRPC (temperature): Loss over FL Rounds')
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout(); fig3.savefig(results_dir / 'grpc_loss.png', dpi=300, bbox_inches='tight'); plt.close(fig3)

        # 4) Regression metrics
        fig4, axes = plt.subplots(1, 3, figsize=(17, 5))
        axes[0].plot(rounds, self.MSE, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Round'); axes[0].set_ylabel('MSE')
        axes[0].set_title('gRPC (temperature): MSE over Rounds'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(rounds, self.MAE, marker='s', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Round'); axes[1].set_ylabel('MAE')
        axes[1].set_title('gRPC (temperature): MAE over Rounds'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(rounds, self.MAPE, marker='^', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Round'); axes[2].set_ylabel('MAPE (%)')
        axes[2].set_title('gRPC (temperature): MAPE over Rounds'); axes[2].grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(results_dir / 'grpc_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print(f"Training metrics plots saved to {results_dir}")

        # 5) CPU and RAM
        plot_cpu_memory_for_server_rounds(
            results_dir,
            "grpc_cpu_memory_per_round.png",
            self.ROUNDS,
            "temperature",
            title="gRPC (temperature): avg client CPU and RAM per round",
        )
        if not os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            plt.show(block=False)
        print("\nPlot closed. Training complete.")
    
    def save_results(self):
        """Save training results to JSON"""
        results_dir = get_experiment_results_dir("temperature", "grpc")

        results = {
            "rounds": self.ROUNDS,
            "loss": self.LOSS,
            "mse": self.MSE,
            "mae": self.MAE,
            "mape": self.MAPE,
            "accuracy": [max(0.0, 1.0 - v) for v in self.MSE],
            "battery_consumption": getattr(self, 'BATTERY_CONSUMPTION', []),
            "battery_model_consumption": getattr(self, 'BATTERY_MODEL_CONSUMPTION', []),
            "battery_model_consumption_source": "client_battery_model",
            "round_times_seconds": getattr(self, 'ROUND_TIMES', []),
            "avg_training_time_sec": getattr(self, 'AVG_TRAINING_TIME_SEC', []),
            "avg_battery_soc": getattr(self, 'AVG_BATTERY_SOC', []),
            "convergence_time_seconds": self.convergence_time,
            "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
            "total_rounds": len(self.ROUNDS),
            "num_clients": self.num_clients,
            "summary": {
                "total_rounds": len(self.ROUNDS),
                "num_clients": self.num_clients,
                "final_loss": self.LOSS[-1] if self.LOSS else None,
                "final_mse": self.MSE[-1] if self.MSE else None,
                "final_mae": self.MAE[-1] if self.MAE else None,
                "final_mape": self.MAPE[-1] if self.MAPE else None,
                "convergence_time_seconds": self.convergence_time,
                "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
                "converged": self.converged,
            }
        }
        try:
            merge_cpu_memory_into_results(results, "temperature")
        except Exception:
            pass

        results_file = results_dir / 'grpc_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Training results saved to {results_file}")


def serve():
    """Start gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=GRPC_OPTIONS)
    servicer = FederatedLearningServicer(MIN_CLIENTS, NUM_ROUNDS, max_clients=MAX_CLIENTS)
    federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    
    server_address = f"{GRPC_HOST}:{GRPC_PORT}"
    server.add_insecure_port(server_address)
    server.start()
    
    print("="*70)
    print("Starting Federated Learning Server (gRPC)")
    print(f"Server Address: {server_address}")
    print(f"Number of Clients: {MIN_CLIENTS}")
    print(f"Number of Rounds: {NUM_ROUNDS}")
    print("="*70)
    print("\nWaiting for clients to register...\n")
    
    try:
        # Keep server running until training is complete
        while not servicer.training_complete:
            time.sleep(1)
        
        # Give clients time to receive completion signal
        print("\nTraining complete. Shutting down server in 5 seconds...")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.stop(0)


if __name__ == "__main__":
    serve()
