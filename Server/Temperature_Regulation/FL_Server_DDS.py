import numpy as np
import pandas as pd
import pickle
import time
import os
import sys
import json
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

import matplotlib.pyplot as plt

# Add CycloneDDS DLL path (only for Windows environments)
if sys.platform == 'win32':
    cyclone_path = os.getenv('CYCLONEDDS_HOME', r"C:\Masters_Infotech\Semester_5\MT_SW_Addons\vcpkg\buildtrees\cyclonedds\x64-windows-rel\bin")
    if cyclone_path and cyclone_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = cyclone_path + os.pathsep + os.environ.get('PATH', '')

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.util import duration
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from cyclonedds.core import Qos, Policy
from dataclasses import dataclass
from typing import List

# Server Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence
NETWORK_SCENARIO = os.getenv("NETWORK_SCENARIO", "excellent")  # Network scenario for result filename

# Chunking configuration for large messages
CHUNK_SIZE = 64 * 1024  # 64KB chunks for better DDS performance in poor networks

# Convergence Settings (primary stopping criterion)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))


# DDS Data Types (matching IDL)
@dataclass
class ClientRegistration(IdlStruct):
    client_id: int
    message: str


@dataclass
class TrainingConfig(IdlStruct):
    batch_size: int
    local_epochs: int


@dataclass
class TrainingCommand(IdlStruct):
    round: int
    start_training: bool
    start_evaluation: bool
    training_complete: bool


@dataclass
class GlobalModel(IdlStruct):
    round: int
    weights: sequence[int]
    model_config_json: str = ""  # JSON string for model configuration


@dataclass
class GlobalModelChunk(IdlStruct):
    round: int
    chunk_id: int
    total_chunks: int
    payload: sequence[int]
    model_config_json: str = ""  # JSON string for model configuration


@dataclass
class ModelUpdate(IdlStruct):
    client_id: int
    round: int
    weights: sequence[int]
    num_samples: int
    loss: float
    mse: float
    mae: float
    mape: float


@dataclass
class ModelUpdateChunk(IdlStruct):
    client_id: int
    round: int
    chunk_id: int
    total_chunks: int
    payload: sequence[int]
    num_samples: int
    loss: float
    mse: float
    mae: float
    mape: float


@dataclass
class EvaluationMetrics(IdlStruct):
    client_id: int
    round: int
    num_samples: int
    loss: float
    mse: float
    mae: float
    mape: float
    client_converged: float = 0.0


@dataclass
class ServerStatus(IdlStruct):
    current_round: int
    total_rounds: int
    training_started: bool
    training_complete: bool
    registered_clients: int


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
        
        # Chunk reassembly buffers
        self.model_update_chunks = {}  # {client_id: {chunk_id: payload}}
        self.model_update_metadata = {}  # {client_id: {total_chunks, num_samples, loss, mse, mae, mape}}
        
        # Metrics storage
        self.MSE = []
        self.MAE = []
        self.MAPE = []
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
            "batch_size": 32,
            "local_epochs": 5
        }
        
        # Status flags
        self.training_started = False
        self.training_complete = False
        self.evaluation_phase = False
        
        # DDS entities
        self.participant = None
        self.readers = {}
        self.writers = {}
    
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
        """Serialize model weights for DDS transmission"""
        serialized = pickle.dumps(weights)
        # Convert bytes to list of ints for DDS
        return list(serialized)
    
    def deserialize_weights(self, serialized_weights):
        """Deserialize model weights received from DDS"""
        # Convert list of ints back to bytes
        return pickle.loads(bytes(serialized_weights))
    
    def split_into_chunks(self, data):
        """Split serialized data into chunks of CHUNK_SIZE"""
        chunks = []
        for i in range(0, len(data), CHUNK_SIZE):
            chunks.append(data[i:i + CHUNK_SIZE])
        return chunks
    
    def send_global_model_chunked(self, round_num, serialized_weights, model_config):
        """Send global model as chunks"""
        chunks = self.split_into_chunks(serialized_weights)
        total_chunks = len(chunks)
        
        print(f"Sending global model in {total_chunks} chunks ({len(serialized_weights)} bytes total)")
        
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = GlobalModelChunk(
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                model_config_json=model_config if chunk_id == 0 else ""  # Only send config with first chunk
            )
            self.writers['global_model_chunk'].write(chunk)
            # Aligned with unified: Reliable QoS handles delivery, no artificial delay needed
            if (chunk_id + 1) % 20 == 0:
                print(f"  Sent {chunk_id + 1}/{total_chunks} chunks")
    
    def setup_dds(self):
        """Initialize DDS participant, topics, readers, and writers"""
        print(f"Setting up DDS on domain {DDS_DOMAIN_ID}...")
        
        # Create domain participant
        self.participant = DomainParticipant(DDS_DOMAIN_ID)
        
        # Reliable QoS for critical control messages (registration, config, commands)
        # TransientLocal durability ensures messages survive discovery delays
        reliable_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
            Policy.History.KeepLast(10),
            Policy.Durability.TransientLocal,
        )

        # Best effort QoS for large data transfers (model chunks)
        best_effort_qos = Qos(
            Policy.Reliability.BestEffort(),
            Policy.History.KeepLast(1),
        )
        
        # Create topics
        topic_registration = Topic(self.participant, "ClientRegistration", ClientRegistration)
        topic_config = Topic(self.participant, "TrainingConfig", TrainingConfig)
        topic_command = Topic(self.participant, "TrainingCommand", TrainingCommand)
        topic_global_model = Topic(self.participant, "GlobalModel", GlobalModel)
        topic_global_model_chunk = Topic(self.participant, "GlobalModelChunk", GlobalModelChunk)
        topic_model_update = Topic(self.participant, "ModelUpdate", ModelUpdate)
        topic_model_update_chunk = Topic(self.participant, "ModelUpdateChunk", ModelUpdateChunk)
        topic_metrics = Topic(self.participant, "EvaluationMetrics", EvaluationMetrics)
        topic_status = Topic(self.participant, "ServerStatus", ServerStatus)
        
        # Create readers (for receiving from clients)
        # Use Reliable QoS for registration to ensure delivery despite discovery delays
        self.readers['registration'] = DataReader(self.participant, topic_registration, qos=reliable_qos)
        # Use BestEffort for chunked data (many small messages, retransmission handled by chunking)
        self.readers['model_update'] = DataReader(self.participant, topic_model_update, qos=best_effort_qos)
        self.readers['model_update_chunk'] = DataReader(self.participant, topic_model_update_chunk, qos=best_effort_qos)
        self.readers['metrics'] = DataReader(self.participant, topic_metrics, qos=best_effort_qos)
        
        # Create writers (for sending to clients)
        # Use Reliable QoS for config and commands (critical control messages)
        self.writers['config'] = DataWriter(self.participant, topic_config, qos=reliable_qos)
        self.writers['command'] = DataWriter(self.participant, topic_command, qos=reliable_qos)
        # Use BestEffort for large model data and chunked transfers
        self.writers['global_model'] = DataWriter(self.participant, topic_global_model, qos=best_effort_qos)
        self.writers['global_model_chunk'] = DataWriter(self.participant, topic_global_model_chunk, qos=best_effort_qos)
        self.writers['status'] = DataWriter(self.participant, topic_status, qos=best_effort_qos)
        
        print("DDS setup complete (Reliable QoS for control, BestEffort for data chunks)\n")
        time.sleep(0.5)  # Allow time for discovery
    
    def publish_status(self):
        """Publish current server status"""
        status = ServerStatus(
            current_round=self.current_round,
            total_rounds=self.num_rounds,
            training_started=self.training_started,
            training_complete=self.training_complete,
            registered_clients=len(self.registered_clients)
        )
        self.writers['status'].write(status)
    
    def run(self):
        """Main server loop"""
        print("="*70)
        print("Starting Federated Learning Server (DDS)")
        print(f"DDS Domain ID: {DDS_DOMAIN_ID}")
        print(f"Number of Clients: {self.num_clients}")
        print(f"Number of Rounds: {self.num_rounds}")
        print("="*70)
        print("\nWaiting for clients to register...\n")
        
        # Setup DDS
        self.setup_dds()
                # Wait for DDS endpoint discovery to complete
        # This ensures readers/writers are matched before clients start sending
        print("Waiting for DDS endpoint discovery...")
        time.sleep(2.0)
        print("DDS endpoints ready\n")
                # Publish initial training config
        config = TrainingConfig(
            batch_size=self.training_config['batch_size'],
            local_epochs=self.training_config['local_epochs']
        )
        self.writers['config'].write(config)
        
        loop_count = 0
        try:
            while not self.training_complete:
                loop_count += 1
                if loop_count % 10 == 0:
                    print(f"[ServerLoop] Iteration {loop_count}, registered={len(self.registered_clients)}/{self.num_clients}, training_started={self.training_started}")
                    sys.stdout.flush()
                
                # Publish current status
                self.publish_status()
                
                # Check for client registrations
                self.check_registrations()
                
                # Check for model updates
                self.check_model_updates()
                
                # Check for evaluation metrics
                self.check_evaluation_metrics()
                
                time.sleep(0.5)
            
            print("\nServer shutting down...")
            
        except KeyboardInterrupt:
            print("\n\nServer interrupted by user")
        finally:
            self.cleanup()
    
    def check_registrations(self):
        """Check for new client registrations"""
        samples = self.readers['registration'].take()
        
        # Debug: Always log to show we're checking (every 20th call to avoid spam)
        if not hasattr(self, '_reg_check_count'):
            self._reg_check_count = 0
        self._reg_check_count += 1

        if self._reg_check_count % 20 == 1:
            print(f"[DEBUG] check_registrations called (count={self._reg_check_count}), samples received: {len(samples)}")

        # Debug: log how many samples received
        if len(samples) > 0:
            print(f"[DEBUG] *** RECEIVED {len(samples)} REGISTRATION SAMPLES ***")
        
        for sample in samples:
            # Some DDS implementations may emit InvalidSample entries; guard against those
            if not sample or not hasattr(sample, 'client_id'):
                # Debug: show what we're skipping
                print(f"[DEBUG] Skipping invalid registration sample: {type(sample).__name__}")
                continue
            client_id = sample.client_id
            print(f"[DEBUG] Processing registration from client {client_id}")
            if client_id not in self.registered_clients:
                self.registered_clients.add(client_id)
                self.active_clients.add(client_id)
                print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))
        
        # If all clients registered, distribute initial global model and start training
        if len(self.registered_clients) == self.num_clients and not self.training_started:
            print("\nAll clients registered. Distributing initial global model...\n")
            self.distribute_initial_model()
            # Record training start time
            self.start_time = time.time()
            self.training_started = True
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def mark_client_converged(self, client_id):
        """Remove converged client from active federation."""
        if client_id in self.active_clients:
            self.active_clients.discard(client_id)
            self.client_updates.pop(client_id, None)
            self.client_metrics.pop(client_id, None)
            print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
            if not self.active_clients:
                self.converged = True
                self.training_complete = True
                print("All clients converged. Ending training.")
                self.convergence_time = time.time() - self.start_time if self.start_time else 0
                command = TrainingCommand(
                    round=self.current_round,
                    start_training=False,
                    start_evaluation=False,
                    training_complete=True
                )
                self.writers['command'].write(command)
                self.plot_results()
                self.save_results()

    def distribute_initial_model(self):
        """Distribute initial global model to all clients"""
        self.training_started = True
        self.current_round = 1
        
        print(f"\n{'='*70}")
        print(f"Distributing Initial Global Model")
        print(f"{'='*70}\n")
        
        # Prepare model configuration
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
        
        # Compress or serialize global weights
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed initial global model - Ratio: {stats['compression_ratio']:.2f}x")
            serialized_weights = list(pickle.dumps(compressed_data))
        else:
            serialized_weights = self.serialize_weights(self.global_weights)
        
        # Wait for DDS chunk endpoints to be fully discovered (BestEffort needs this!)
        print("Waiting for chunk DataReader/DataWriter discovery...")
        time.sleep(5)  # Extra time for BestEffort endpoints to discover each other
        
        # Send initial global model to all clients in chunks
        print("Publishing initial model to clients in chunks...")
        self.send_global_model_chunked(0, serialized_weights, json.dumps(model_config))
        
        print("Initial global model (architecture + weights) sent to all clients in chunks")
        
        # Wait for clients to receive and set the initial model
        time.sleep(2)
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Send training command to start first round with retry for poor network conditions
        command = TrainingCommand(
            round=self.current_round,
            start_training=True,
            start_evaluation=False,
            training_complete=False
        )
        # Send multiple times to ensure delivery under poor network conditions
        for retry in range(3):
            self.writers['command'].write(command)
            if retry < 2:
                time.sleep(0.5)
    
    def check_model_updates(self):
        """Check for model updates from clients (chunked version)"""
        # Check for chunked model updates
        chunk_samples = self.readers['model_update_chunk'].take()
        
        for sample in chunk_samples:
            if not sample or not hasattr(sample, 'round'):
                continue
                
            if sample.round == self.current_round and hasattr(sample, 'client_id'):
                client_id = sample.client_id
                chunk_id = sample.chunk_id
                total_chunks = sample.total_chunks
                
                # Initialize buffers for this client if needed
                if client_id not in self.model_update_chunks:
                    self.model_update_chunks[client_id] = {}
                    self.model_update_metadata[client_id] = {
                        'total_chunks': total_chunks,
                        'num_samples': sample.num_samples,
                        'loss': sample.loss,
                        'mse': sample.mse,
                        'mae': sample.mae,
                        'mape': sample.mape
                    }
                
                # Store chunk
                self.model_update_chunks[client_id][chunk_id] = sample.payload
                
                print(f"Received chunk {chunk_id + 1}/{total_chunks} from client {client_id}")
                
                # Check if all chunks received for this client
                if len(self.model_update_chunks[client_id]) == total_chunks:
                    print(f"All chunks received from client {client_id}, reassembling...")
                    
                    # Reassemble chunks in order
                    reassembled_data = []
                    for i in range(total_chunks):
                        if i in self.model_update_chunks[client_id]:
                            reassembled_data.extend(self.model_update_chunks[client_id][i])
                        else:
                            print(f"ERROR: Missing chunk {i} from client {client_id}")
                            break
                    
                    # Only process if we have all chunks
                    if len(reassembled_data) > 0 and client_id not in self.client_updates:
                        # Decompress or deserialize client weights
                        if self.quantization_handler is not None:
                            try:
                                compressed_data = pickle.loads(bytes(reassembled_data))
                                weights = self.quantization_handler.decompress_client_update(client_id, compressed_data)
                                print(f"Server: Received and decompressed update from client {client_id}")
                            except Exception as e:
                                print(f"Server: Failed to decompress from client {client_id}, falling back: {e}")
                                weights = self.deserialize_weights(reassembled_data)
                        else:
                            weights = self.deserialize_weights(reassembled_data)
                        
                        metadata = self.model_update_metadata[client_id]
                        self.client_updates[client_id] = {
                            'weights': weights,
                            'num_samples': metadata['num_samples'],
                            'metrics': {
                                'loss': metadata['loss'],
                                'mse': metadata['mse'],
                                'mae': metadata['mae'],
                                'mape': metadata['mape']
                            }
                        }
                        
                        # Clear chunk buffers for this client
                        del self.model_update_chunks[client_id]
                        del self.model_update_metadata[client_id]
                        
                        print(f"Successfully reassembled update from client {client_id} "
                              f"({len(self.client_updates)}/{len(self.registered_clients)})")
        
        # If all clients sent updates, aggregate (ensure we have at least one client)
        if len(self.client_updates) > 0 and len(self.client_updates) >= len(self.registered_clients):
            self.aggregate_models()
    
    def check_evaluation_metrics(self):
        """Check for evaluation metrics from clients"""
        samples = self.readers['metrics'].take()
        
        for sample in samples:
            if sample:
                print(f"Server received metrics sample: client {sample.client_id}, round {sample.round} (current: {self.current_round})")
                
                if sample.round == self.current_round:
                    client_id = sample.client_id
                    
                    if client_id not in self.client_metrics:
                        self.client_metrics[client_id] = {
                            'num_samples': sample.num_samples,
                            'metrics': {
                                'loss': sample.loss,
                                'mse': sample.mse,
                                'mae': sample.mae,
                                'mape': sample.mape
                            }
                        }
                        
                        print(f"Received metrics from client {client_id} "
                              f"({len(self.client_metrics)}/{self.num_clients})")
                        
                        # If all clients sent metrics, aggregate and continue
                    # Wait for all registered clients (dynamic)
            if len(self.client_metrics) >= len(self.registered_clients):
                        self.aggregate_metrics()
                        self.continue_training()
    
    def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")
        
        # Safety check: ensure we have at least one client update
        if len(self.client_updates) == 0:
            print("ERROR: aggregate_models called with 0 client updates. Skipping aggregation.")
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
        
        print(f"Aggregated global model from round {self.current_round}")
        print(f"Sending global model to clients...\n")
        
        # Compress or serialize global weights
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
            # Serialize compressed data (pickle + convert to list of ints for DDS)
            serialized_weights = list(pickle.dumps(compressed_data))
        else:
            serialized_weights = self.serialize_weights(self.global_weights)
        
        # Publish global model using chunking (always include model_config for late-joiners)
        model_config = {
            "architecture": "LSTM",
            "layers": [
                {"type": "LSTM", "units": 50, "activation": "relu", "input_shape": [1, 4]},
                {"type": "Dense", "units": 1}
            ],
            "compile_config": {"loss": "mse", "optimizer": "adam", "metrics": ["mae"]}
        }
        self.send_global_model_chunked(self.current_round, serialized_weights, json.dumps(model_config))
        
        # Send evaluation command with retry for poor network conditions
        time.sleep(1)
        command = TrainingCommand(
            round=self.current_round,
            start_training=False,
            start_evaluation=True,
            training_complete=False
        )
        # Send multiple times to ensure delivery
        for retry in range(3):
            self.writers['command'].write(command)
            if retry < 2:
                time.sleep(0.5)
        
        self.evaluation_phase = True
        
        # Note: Evaluation metrics will be collected via check_evaluation_metrics() in main loop
        # No blocking wait here to allow server loop to continue polling
    
    def wait_for_evaluation_metrics(self):
        """Actively wait for evaluation metrics from all active clients (no timeout)"""
        print(f"\nWaiting for evaluation metrics from {len(self.active_clients)} active clients...")
        check_count = 0
        start_time = time.time()
        
        while len(self.client_metrics) < len(self.active_clients) and len(self.active_clients) > 0:
            samples = self.readers['metrics'].take()
            for sample in samples:
                client_id = sample.client_id
                if client_id not in self.active_clients:
                    continue
                if getattr(sample, 'client_converged', 0.0) >= 1.0:
                    self.mark_client_converged(client_id)
                    if len(self.active_clients) == 0:
                        return
                    continue
                if sample.round == self.current_round and client_id not in self.client_metrics:
                    print(f"Received evaluation metrics from client {client_id}")
                    metrics_dict = {
                        'mse': sample.mse,
                        'mae': sample.mae,
                        'mape': sample.mape,
                        'loss': sample.loss
                    }
                    self.client_metrics[client_id] = {
                        'metrics': metrics_dict,
                        'num_samples': sample.num_samples
                    }
                    print(f"Progress: {len(self.client_metrics)}/{len(self.active_clients)} clients")
            
            if len(self.client_metrics) < len(self.active_clients) and len(self.active_clients) > 0:
                time.sleep(0.5)
                check_count += 1
                if check_count % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"Still waiting for metrics ({len(self.client_metrics)}/{len(self.active_clients)}) - {elapsed:.1f}s elapsed")
        
        print(f"✓ All evaluation metrics received!")
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics from all clients"""
        print(f"\nAggregating metrics from {len(self.client_metrics)} clients...")
        
        # Calculate total samples
        total_samples = sum(metric['num_samples'] 
                          for metric in self.client_metrics.values())
        
        # Weighted average of metrics
        aggregated_mse = sum(metric['metrics']['mse'] * metric['num_samples']
                            for metric in self.client_metrics.values()) / total_samples
        
        aggregated_mae = sum(metric['metrics']['mae'] * metric['num_samples']
                            for metric in self.client_metrics.values()) / total_samples
        
        aggregated_mape = sum(metric['metrics']['mape'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        aggregated_loss = sum(metric['metrics']['loss'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
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
        print(f"  MAPE: {aggregated_mape:.4f}\n")
    
    def continue_training(self):
        """Continue to next round or finish training"""
        # Clear updates and metrics for next round
        self.client_updates.clear()
        self.client_metrics.clear()
        self.evaluation_phase = False
        
        # Stop only when no active clients or max rounds (no server-side convergence)
        if len(self.active_clients) == 0:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            self.converged = True
            print("\n" + "="*70)
            print("All clients converged locally. Training complete.")
            print("="*70 + "\n")
            command = TrainingCommand(
                round=self.current_round,
                start_training=False,
                start_evaluation=False,
                training_complete=True
            )
            self.writers['command'].write(command)
            self.training_complete = True
            self.plot_results()
            self.save_results()
            return
        
        if self.current_round >= MIN_ROUNDS and self.check_convergence():
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("CONVERGENCE ACHIEVED!")
            print(f"Training stopped early at round {self.current_round}/{self.num_rounds}")
            print(f"Loss improvement below threshold for {CONVERGENCE_PATIENCE} consecutive rounds")
            print(f"Time to Convergence: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            self.converged = True
            
            # Send completion signal
            command = TrainingCommand(
                round=self.current_round,
                start_training=False,
                start_evaluation=False,
                training_complete=True
            )
            self.writers['command'].write(command)
            
            self.training_complete = True
            self.plot_results()
            self.save_results()
            return
        
        # Check if more rounds needed
        if self.current_round < self.num_rounds:
            self.current_round += 1
            
            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"{'='*70}\n")
            
            time.sleep(2)
            
            # Send training command for next round with retry
            command = TrainingCommand(
                round=self.current_round,
                start_training=True,
                start_evaluation=False,
                training_complete=False
            )
            # Send multiple times to ensure delivery under poor network conditions
            for retry in range(3):
                self.writers['command'].write(command)
                if retry < 2:
                    time.sleep(0.5)
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            
            # Send completion signal with retry
            command = TrainingCommand(
                round=self.current_round,
                start_training=False,
                start_evaluation=False,
                training_complete=True
            )
            # Send multiple times to ensure all clients receive it
            for retry in range(3):
                self.writers['command'].write(command)
                if retry < 2:
                    time.sleep(0.5)
            
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
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.plot(self.ROUNDS, self.LOSS, 'b-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 4, 2)
        plt.plot(self.ROUNDS, self.MSE, 'r-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error')
        plt.grid(True)
        
        plt.subplot(1, 4, 3)
        plt.plot(self.ROUNDS, self.MAE, 'g-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')
        plt.grid(True)
        
        plt.subplot(1, 4, 4)
        plt.plot(self.ROUNDS, self.MAPE, 'm-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('MAPE')
        plt.title('Mean Absolute Percentage Error')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'dds_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to {results_dir / 'dds_training_metrics.png'}")
        plt.show()
        
        print("\nPlot closed. Training complete.")
    
    def save_results(self):
        """Save training results to CSV"""
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results_df = pd.DataFrame({
            'Round': self.ROUNDS,
            'Loss': self.LOSS,
            'MSE': self.MSE,
            'MAE': self.MAE,
            'MAPE': self.MAPE
        })
        
        # Add summary row with convergence time
        summary_df = pd.DataFrame([{
            'Round': 'SUMMARY',
            'Loss': self.LOSS[-1] if self.LOSS else None,
            'MSE': self.MSE[-1] if self.MSE else None,
            'MAE': self.MAE[-1] if self.MAE else None,
            'MAPE': self.MAPE[-1] if self.MAPE else None
        }])
        summary_df['Total Rounds'] = len(self.ROUNDS)
        summary_df['Num Clients'] = self.num_clients
        summary_df['Convergence Time (seconds)'] = self.convergence_time
        summary_df['Convergence Time (minutes)'] = self.convergence_time / 60 if self.convergence_time else None
        
        results_df = pd.concat([results_df, summary_df], ignore_index=True)
        
        results_file = results_dir / f'dds_{NETWORK_SCENARIO}_training_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Training results saved to {results_file}")
    
    def cleanup(self):
        """Cleanup DDS resources"""
        if self.participant:
            # DomainParticipant will be automatically cleaned up
            self.participant = None
        print("DDS resources cleaned up")


if __name__ == "__main__":
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    server.run()
