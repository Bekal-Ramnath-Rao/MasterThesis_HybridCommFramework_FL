import numpy as np
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

# Add CycloneDDS DLL path
cyclone_path = r"C:\Masters_Infotech\Semester_5\MT_SW_Addons\vcpkg\buildtrees\cyclonedds\x64-windows-rel\bin"
if cyclone_path not in os.environ.get('PATH', ''):
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
    accuracy: float


@dataclass
class ModelUpdateChunk(IdlStruct):
    client_id: int
    round: int
    chunk_id: int
    total_chunks: int
    payload: sequence[int]
    num_samples: int
    loss: float
    accuracy: float


@dataclass
class EvaluationMetrics(IdlStruct):
    client_id: int
    round: int
    num_samples: int
    loss: float
    accuracy: float


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
        self.num_classes = 7  # For emotion recognition
        self.current_round = 0
        self.registered_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        
        # Chunk reassembly buffers
        self.model_update_chunks = {}  # {client_id: {chunk_id: payload}}
        self.model_update_metadata = {}  # {client_id: {total_chunks, num_samples, loss, accuracy}}
        
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
        # Training configuration broadcast to DDS clients
        # Reduced batch size to 16 to prevent GPU OOM
        batch_size_env = int(os.getenv("BATCH_SIZE", "32"))
        # Default to 5 for faster smoke tests if LOCAL_EPOCHS not set
        local_epochs_env = int(os.getenv("LOCAL_EPOCHS", "20"))
        self.training_config = {
            "batch_size": batch_size_env,
            "local_epochs": local_epochs_env
        }
        print(f"Server DDS training config: batch_size={batch_size_env}, local_epochs={local_epochs_env}")
        
        # Status flags
        self.training_started = False
        self.training_complete = False
        self.evaluation_phase = False
        
        # DDS entities
        self.participant = None
        self.readers = {}
        self.writers = {}
    
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
        
        print("\nGlobal CNN model initialized for emotion recognition")
        print(f"Model architecture: CNN with {len(self.global_weights)} weight layers")
        print(f"Input shape: 48x48x1 (grayscale images)")
        print(f"Output classes: 7 emotions")
    
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
            print(f"  Sent chunk {chunk_id + 1}/{total_chunks} ({len(chunk_data)} bytes)")
            time.sleep(0.05)  # Small delay between chunks
    
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

        # Reliable QoS for chunks - need large history buffer for multiple clients × 144 chunks
        chunk_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=5)),
            Policy.History.KeepLast(500),  # Support multiple clients sending 144 chunks simultaneously
            Policy.Durability.Volatile
        )

        # Best effort QoS for large data transfers (model chunks)
        best_effort_qos = Qos(
            Policy.Reliability.BestEffort,
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
        
        # # Create readers (for receiving from clients) with reliable QoS
        # self.readers['registration'] = DataReader(self.participant, topic_registration, qos=reliable_qos)
        # self.readers['model_update'] = DataReader(self.participant, topic_model_update, qos=reliable_qos)
        # self.readers['metrics'] = DataReader(self.participant, topic_metrics, qos=reliable_qos)
        
        # # Create writers (for sending to clients) with reliable QoS
        # self.writers['config'] = DataWriter(self.participant, topic_config, qos=reliable_qos)
        # self.writers['command'] = DataWriter(self.participant, topic_command, qos=reliable_qos)
        # self.writers['global_model'] = DataWriter(self.participant, topic_global_model, qos=reliable_qos)
        # self.writers['status'] = DataWriter(self.participant, topic_status, qos=reliable_qos)

        # Create readers (for receiving from clients)
        # Use Reliable QoS for registration to ensure delivery despite discovery delays
        self.readers['registration'] = DataReader(self.participant, topic_registration, qos=reliable_qos)
        # Use chunk_qos for chunks with large history buffer to prevent loss
        self.readers['model_update'] = DataReader(self.participant, topic_model_update, qos=best_effort_qos)
        self.readers['model_update_chunk'] = DataReader(self.participant, topic_model_update_chunk, qos=chunk_qos)
        self.readers['metrics'] = DataReader(self.participant, topic_metrics, qos=reliable_qos)
        
        # Create writers (for sending to clients)
        # Use Reliable QoS for config and commands (critical control messages)
        self.writers['config'] = DataWriter(self.participant, topic_config, qos=reliable_qos)
        self.writers['command'] = DataWriter(self.participant, topic_command, qos=reliable_qos)
        # Use BestEffort for legacy global_model, chunk_qos for chunks with large buffer
        self.writers['global_model'] = DataWriter(self.participant, topic_global_model, qos=best_effort_qos)
        self.writers['global_model_chunk'] = DataWriter(self.participant, topic_global_model_chunk, qos=chunk_qos)
        self.writers['status'] = DataWriter(self.participant, topic_status, qos=best_effort_qos)
        
        print("DDS setup complete (Reliable QoS for control and chunks with KeepLast(200), BestEffort for status)\n")
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
        print("Starting Federated Learning Server (DDS) - Emotion Recognition")
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
                try:
                    loop_count += 1
                    # Print heartbeat every 10 iterations (5 seconds)
                    if loop_count % 10 == 0:
                        print(f"[ServerLoop] Iteration {loop_count}, registered={len(self.registered_clients)}/{self.num_clients}, training_started={self.training_started}")
                        sys.stdout.flush()
                    
                    # Publish current status
                    try:
                        self.publish_status()
                    except Exception as e:
                        print(f"[ERROR] publish_status failed: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                    
                    # Check for client registrations
                    try:
                        self.check_registrations()
                    except Exception as e:
                        print(f"[ERROR] check_registrations failed: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                    
                    # Check for model updates
                    try:
                        self.check_model_updates()
                    except Exception as e:
                        print(f"[ERROR] check_model_updates failed: {e}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                    
                    time.sleep(0.05)  # Fast polling for chunk reception
                    
                except Exception as loop_error:
                    print(f"[FATAL] Unhandled exception in server main loop iteration {loop_count}: {loop_error}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    # Continue loop despite error
                    time.sleep(1)
            
            #print(f"\n[ServerLoop] Loop exited normally: training_complete={self.training_complete}")
            sys.stdout.flush()
            print("\nServer shutting down...")
            
        except KeyboardInterrupt:
            print("\n\nServer interrupted by user")
            sys.stdout.flush()
        except Exception as fatal_error:
            print(f"\n[FATAL] Server crashed with exception: {fatal_error}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
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
    
    def distribute_initial_model(self):
        """Distribute initial global model to all clients"""
        self.training_started = True
        self.current_round = 1
        
        print(f"\n{'='*70}")
        print(f"Distributing Initial Global Model")
        print(f"{'='*70}\n")
        
        # Prepare model configuration
        model_config = {
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
        
        # Compress or serialize global weights
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed initial global model - Ratio: {stats['compression_ratio']:.2f}x")
            # Serialize compressed data (pickle + convert to list of ints for DDS)
            serialized_weights = list(pickle.dumps(compressed_data))
        else:
            serialized_weights = self.serialize_weights(self.global_weights)
        
        # Send initial global model to all clients using chunking
        print("Publishing initial model to clients in chunks...")
        self.send_global_model_chunked(0, serialized_weights, json.dumps(model_config))
        
        print("Initial global model (architecture + weights) sent to all clients in chunks")
        
        # Wait for clients to receive and set the initial model
        print("Waiting for clients to receive and build the model...")
        time.sleep(3)
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Send training command to start first round
        print("Signaling clients to start training...")
        command = TrainingCommand(
            round=self.current_round,
            start_training=True,
            start_evaluation=False,
            training_complete=False
        )
        self.writers['command'].write(command)
        print("Start training signal sent successfully\n")
    
    def check_model_updates(self):
        """Check for model updates from clients (chunked version)"""
        # Drain all pending chunks in one call for faster processing
        chunk_samples = self.readers['model_update_chunk'].take()
        
        if not chunk_samples:
            return  # No chunks to process
        
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
                        'accuracy': sample.accuracy
                    }
                
                # Store chunk
                self.model_update_chunks[client_id][chunk_id] = sample.payload
                
                # Progress update every 20 chunks to reduce console spam
                if (chunk_id + 1) % 20 == 0 or (chunk_id + 1) == total_chunks:
                    print(f"Received {chunk_id + 1}/{total_chunks} chunks from client {client_id}")
                
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
                                'accuracy': metadata['accuracy']
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
            if sample and hasattr(sample, 'client_id') and hasattr(sample, 'round'):
                print(f"Server received metrics sample: client {sample.client_id}, round {sample.round} (current: {self.current_round})")
                
                if sample.round == self.current_round:
                    client_id = sample.client_id
                    
                    if client_id not in self.client_metrics:
                        self.client_metrics[client_id] = {
                            'num_samples': sample.num_samples,
                            'metrics': {
                                'loss': sample.loss,
                                'accuracy': sample.accuracy
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
            'input_shape': [48, 48, 1],
            'num_classes': self.num_classes,
            'layers': [
                {'type': 'Conv2D', 'filters': 64, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
                {'type': 'Dropout', 'rate': 0.25},
                {'type': 'Conv2D', 'filters': 128, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
                {'type': 'Dropout', 'rate': 0.25},
                {'type': 'Flatten'},
                {'type': 'Dense', 'units': 512, 'activation': 'relu'},
                {'type': 'Dropout', 'rate': 0.5},
                {'type': 'Dense', 'units': self.num_classes, 'activation': 'softmax'}
            ]
        }
        self.send_global_model_chunked(self.current_round, serialized_weights, json.dumps(model_config))
        
        # Send evaluation command
        time.sleep(1)
        command = TrainingCommand(
            round=self.current_round,
            start_training=False,
            start_evaluation=True,
            training_complete=False
        )
        self.writers['command'].write(command)
        
        self.evaluation_phase = True
        
        # Wait for all evaluation metrics
        self.wait_for_evaluation_metrics()
        
        # After receiving all metrics, aggregate and continue
        # Wait for all registered clients (dynamic)
        if len(self.client_metrics) >= len(self.registered_clients):
            self.aggregate_metrics()
            self.continue_training()
    
    def wait_for_evaluation_metrics(self):
        """Actively wait for evaluation metrics from all clients"""
        print(f"\nWaiting for evaluation metrics from {self.num_clients} clients...")
        timeout = 60  # 60 seconds timeout
        start_time = time.time()
        
        while len(self.client_metrics) < self.num_clients:
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for metrics. Received {len(self.client_metrics)}/{self.num_clients}")
                break
            
            samples = self.readers['metrics'].take()
            for sample in samples:
                if sample.round == self.current_round:
                    client_id = sample.client_id
                    if client_id not in self.client_metrics:
                        print(f"Received evaluation metrics from client {client_id}")
                        metrics_dict = {
                            'loss': sample.loss,
                            'accuracy': sample.accuracy
                        }
                        self.client_metrics[client_id] = {
                            'metrics': metrics_dict,
                            'num_samples': sample.num_samples
                        }
                        print(f"Progress: {len(self.client_metrics)}/{self.num_clients} clients")
            
            if len(self.client_metrics) < self.num_clients:
                time.sleep(0.1)  # Short sleep before next check
        
        # Wait for all registered clients (dynamic)
        if len(self.client_metrics) >= len(self.registered_clients):
            print(f"✓ All evaluation metrics received!")
    
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
        self.evaluation_phase = False
        
        # Check for convergence (early stopping)
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
            
            # Send training command for next round
            command = TrainingCommand(
                round=self.current_round,
                start_training=True,
                start_evaluation=False,
                training_complete=False
            )
            self.writers['command'].write(command)
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            
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
        plt.savefig(results_dir / 'dds_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to {results_dir / 'dds_training_metrics.png'}")
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
        
        results_file = results_dir / 'dds_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Training results saved to {results_file}")
    
    def cleanup(self):
        """Cleanup DDS resources"""
        if self.participant:
            # DomainParticipant will be automatically cleaned up
            self.participant = None
        print("DDS resources cleaned up")


if __name__ == "__main__":
    # Clear any previous TensorFlow/CUDA errors
    import tensorflow as tf
    try:
        # Reset any existing sessions
        tf.keras.backend.clear_session()
        
        # Server runs on CPU only - clients use GPU for training
        # Disable GPU visibility for server (aggregation doesn't need GPU)
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("Server configured to use CPU only (GPU disabled)")
        
    except Exception as e:
        print(f"GPU initialization warning: {e}")
    
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    server.run()
