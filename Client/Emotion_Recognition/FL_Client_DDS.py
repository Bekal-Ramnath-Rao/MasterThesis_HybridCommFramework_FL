import numpy as np
import os
import sys
import logging
import json
import pickle
import time
import random

# GPU Configuration - Must be done BEFORE TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Get GPU device ID from environment variable (set by docker for multi-GPU isolation)
# Fallback strategy: GPU_DEVICE_ID -> (CLIENT_ID - 1) -> "0"
# This ensures different clients use different GPUs in multi-GPU setups
client_id_env = os.environ.get("CLIENT_ID", "0")
try:
    default_gpu = str(max(0, int(client_id_env) - 1))  # Client 1->GPU 0, Client 2->GPU 1, etc.
except (ValueError, TypeError):
    default_gpu = "0"
gpu_device = os.environ.get("GPU_DEVICE_ID", default_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device  # Isolate to specific GPU
print(f"GPU Configuration: CLIENT_ID={client_id_env}, GPU_DEVICE_ID={gpu_device}, CUDA_VISIBLE_DEVICES={gpu_device}")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow gradual GPU memory growth
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # GPU thread mode

# Disable Grappler layout optimizer to avoid NCHW transpose errors in logs
os.environ["TF_ENABLE_LAYOUT_OPTIMIZER"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Make TensorFlow logs less verbose
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Ensure Keras uses channels_last image data format
tf.keras.backend.set_image_data_format('channels_last')

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
        
        # Set GPU memory limit to avoid OOM (RTX 3080 has 10GB, reserve 7GB per process)
        # This prevents one process from consuming all GPU memory
        for gpu in gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=7000)]  # 7GB per GPU
                )
            except RuntimeError:
                pass  # GPU already configured
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPUs found. Running on CPU.")

# Add CycloneDDS DLL path
cyclone_path = r"C:\Masters_Infotech\Semester_5\MT_SW_Addons\vcpkg\buildtrees\cyclonedds\x64-windows-rel\bin"
if cyclone_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = cyclone_path + os.pathsep + os.environ.get('PATH', '')

from cyclonedds.domain import DomainParticipant

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.util import duration
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from cyclonedds.core import Qos, Policy
from dataclasses import dataclass
from typing import List

# DDS Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))

# Chunking configuration for large messages
CHUNK_SIZE = 64 * 1024  # 64KB chunks for better DDS performance in poor networks


# DDS Data Types (matching IDL)
@dataclass
class ClientRegistration(IdlStruct):
    client_id: int
    message: str


@dataclass
class ClientReady(IdlStruct):
    """Signal from client that it's ready to receive chunks"""
    client_id: int
    ready_for_chunks: bool


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
    model_config_json: str = ""  # JSON string containing model configuration


@dataclass
class GlobalModelChunk(IdlStruct):
    round: int
    chunk_id: int
    total_chunks: int
    payload: sequence[int]
    model_config_json: str = ""  # JSON string containing model configuration


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
    client_converged: float = 0.0


@dataclass
class ServerStatus(IdlStruct):
    current_round: int
    total_rounds: int
    training_started: bool
    training_complete: bool
    registered_clients: int


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, train_generator=None, validation_generator=None):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        
        # Initialize quantization compression (default: disabled unless explicitly enabled)
        uq_env = os.getenv("USE_QUANTIZATION", "false")
        use_quantization = uq_env.lower() in ("true", "1", "yes", "y")
        if use_quantization:
            self.quantizer = Quantization(QuantizationConfig())
            print(f"Client {self.client_id}: Quantization enabled")
        else:
            self.quantizer = None
            print(f"Client {self.client_id}: Quantization disabled")
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.current_round = 0
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        self.running = True
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.has_converged = False
        
        # Chunk reassembly buffers
        self.global_model_chunks = {}  # {chunk_id: payload}
        self.global_model_metadata = {}  # {round, total_chunks, model_config_json}
        
        # DDS entities
        self.participant = None
        self.readers = {}
        self.writers = {}
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {self.train_generator.n}")
        print(f"  Validation samples: {self.validation_generator.n}")
        print(f"  Waiting for initial global model from server...")
        
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
    
    def send_model_update_chunked(self, round_num, serialized_weights, num_samples, loss, accuracy):
        """Send model update as chunks"""
        chunks = self.split_into_chunks(serialized_weights)
        total_chunks = len(chunks)
        
        print(f"Client {self.client_id}: Sending model update in {total_chunks} chunks ({len(serialized_weights)} bytes total)")
        
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = ModelUpdateChunk(
                client_id=self.client_id,
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                num_samples=num_samples,
                loss=loss,
                accuracy=accuracy
            )
            self.writers['model_update_chunk'].write(chunk)
            
            # CRITICAL: Pacing for BestEffort QoS to prevent UDP buffer overflow
            time.sleep(0.01)  # 10ms between chunks
            
            if (chunk_id + 1) % 20 == 0:  # Progress update every 20 chunks
                print(f"  Sent {chunk_id + 1}/{total_chunks} chunks")
                time.sleep(0.1)  # Extra pause for buffer drain
    
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
            Policy.Durability.TransientLocal
        )

        # Best effort QoS for large data transfers (model chunks)
        # Increased history buffer to handle burst packet loss
        best_effort_qos = Qos(
            Policy.Reliability.BestEffort,
            Policy.History.KeepLast(50),  # Buffer up to 50 chunks to handle burst losses
        )
        # Reliable chunk QoS for receiving full global model chunk set.
        chunk_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
            Policy.History.KeepLast(2048),
            Policy.Durability.TransientLocal
        )
        
        # Create topics
        topic_registration = Topic(self.participant, "ClientRegistration", ClientRegistration)
        topic_ready = Topic(self.participant, "ClientReady", ClientReady)
        topic_config = Topic(self.participant, "TrainingConfig", TrainingConfig)
        topic_command = Topic(self.participant, "TrainingCommand", TrainingCommand)
        topic_global_model = Topic(self.participant, "GlobalModel", GlobalModel)
        topic_global_model_chunk = Topic(self.participant, "GlobalModelChunk", GlobalModelChunk)
        topic_model_update = Topic(self.participant, "ModelUpdate", ModelUpdate)
        topic_model_update_chunk = Topic(self.participant, "ModelUpdateChunk", ModelUpdateChunk)
        topic_metrics = Topic(self.participant, "EvaluationMetrics", EvaluationMetrics)
        topic_status = Topic(self.participant, "ServerStatus", ServerStatus)
        
        # # Create readers (for receiving from server) with reliable QoS
        # self.readers['config'] = DataReader(self.participant, topic_config, qos=reliable_qos)
        # self.readers['command'] = DataReader(self.participant, topic_command, qos=reliable_qos)
        # self.readers['global_model'] = DataReader(self.participant, topic_global_model, qos=reliable_qos)
        # self.readers['status'] = DataReader(self.participant, topic_status, qos=reliable_qos)
        
        # # Create writers (for sending to server) with reliable QoS
        # self.writers['registration'] = DataWriter(self.participant, topic_registration, qos=reliable_qos)
        # self.writers['model_update'] = DataWriter(self.participant, topic_model_update, qos=reliable_qos)
        # self.writers['metrics'] = DataWriter(self.participant, topic_metrics, qos=reliable_qos)
        
        # Create readers (for receiving from server)
        # Use Reliable QoS for config and commands (critical control messages)
        self.readers['config'] = DataReader(self.participant, topic_config, qos=reliable_qos)
        self.readers['command'] = DataReader(self.participant, topic_command, qos=reliable_qos)
        # Use BestEffort for chunked model data
        self.readers['global_model'] = DataReader(self.participant, topic_global_model, qos=best_effort_qos)
        self.readers['global_model_chunk'] = DataReader(self.participant, topic_global_model_chunk, qos=chunk_qos)
        self.readers['status'] = DataReader(self.participant, topic_status, qos=best_effort_qos)
        
        # Create writers (for sending to server)
        # Use Reliable QoS for registration and ready signals (critical to ensure server receives them)
        self.writers['registration'] = DataWriter(self.participant, topic_registration, qos=reliable_qos)
        self.writers['ready'] = DataWriter(self.participant, topic_ready, qos=reliable_qos)
        # Use Reliable+deep-history for chunk stream to avoid server-side partial updates.
        self.writers['model_update'] = DataWriter(self.participant, topic_model_update, qos=best_effort_qos)
        self.writers['model_update_chunk'] = DataWriter(self.participant, topic_model_update_chunk, qos=chunk_qos)
        self.writers['metrics'] = DataWriter(self.participant, topic_metrics, qos=best_effort_qos)

        print(f"Client {self.client_id} DDS setup complete (Reliable QoS for control, BestEffort for data)")
        
        # Wait for DDS endpoint discovery (critical for BestEffort QoS!)
        # BestEffort chunks are lost if sent before DataReader/DataWriter are matched
        print(f"Client {self.client_id} waiting for endpoint discovery...")
        time.sleep(3.0)  # Increased for BestEffort chunk endpoints
        
        # Register with server (send multiple times for reliability)
        registration = ClientRegistration(
            client_id=self.client_id,
            message=f"Client {self.client_id} ready"
        )
        print(f"Client {self.client_id} sending registration...")
        for i in range(3):
            self.writers['registration'].write(registration)
            print(f"  Registration attempt {i+1}/3")
            time.sleep(0.3)
        print(f"Client {self.client_id} registration sent")
        
        # Signal that we're ready to receive chunks (AFTER registration)
        time.sleep(2.0)  # Extra wait to ensure chunk endpoints are discovered
        ready_signal = ClientReady(
            client_id=self.client_id,
            ready_for_chunks=True
        )
        print(f"Client {self.client_id} signaling ready for chunk reception...")
        for i in range(3):
            self.writers['ready'].write(ready_signal)
            time.sleep(0.2)
        print(f"Client {self.client_id} ready signal sent\n")
    
    def run(self):
        """Main client loop"""
        print("="*60)
        print(f"Starting Federated Learning Client {self.client_id}")
        print(f"DDS Domain ID: {DDS_DOMAIN_ID}")
        print("="*60)
        print()
        
        # Setup DDS
        self.setup_dds()
        
        # Get training configuration
        self.get_training_config()
        
        print(f"Client {self.client_id} waiting for training to start...\n")
        
        loop_count = 0
        try:
            while self.running:
                loop_count += 1
                
                # Heartbeat every 5 seconds (50 iterations * 0.1s)
                if loop_count % 50 == 0:
                    print(f"[ClientLoop] Client {self.client_id} iteration {loop_count}, model={'received' if self.model else 'waiting'}, round={self.current_round}")
                
                # Check for global model updates
                self.check_global_model()
                
                # Check for training commands
                self.check_commands()
                
                time.sleep(0.1)  # Check more frequently
                
        except KeyboardInterrupt:
            print(f"\nClient {self.client_id} shutting down...")
        finally:
            self.cleanup()
    
    def get_training_config(self):
        """Get training configuration from server"""
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            samples = self.readers['config'].take()
            for sample in samples:
                if sample:
                    self.training_config = {
                        "batch_size": sample.batch_size,
                        "local_epochs": sample.local_epochs
                    }
                    print(f"Client {self.client_id} received config: {self.training_config}")
                    return
            time.sleep(0.5)
        
        print(f"Client {self.client_id} using default config: {self.training_config}")
    
    def check_commands(self):
        """Check for training commands from server"""
        samples = self.readers['command'].take()
        
        for sample in samples:
            if sample:
                print(f"Client {self.client_id} received command: round={sample.round}, start_training={sample.start_training}, start_evaluation={sample.start_evaluation}, training_complete={sample.training_complete}")
                if sample.training_complete:
                    print(f"\nClient {self.client_id} - Training completed!")
                    self.running = False
                    return
                
                # Check if we're ready for this round (should have received global model first)
                if sample.start_training:
                    if self.current_round == 0 and sample.round == 1:
                        # First training round with initial global model
                        if self.model is None:
                            print(f"Client {self.client_id} waiting for initial model before training...")
                            return
                        self.current_round = sample.round
                        print(f"\nClient {self.client_id} starting training for round {self.current_round} with initial global model...")
                        self.train_local_model()
                    elif sample.round > self.current_round:
                        # Subsequent rounds
                        self.current_round = sample.round
                        print(f"\nClient {self.client_id} starting training for round {self.current_round}...")
                        self.train_local_model()

        # Fallback: if we've received the initial global model but no command within a short window,
        # and server status indicates training has started, proactively begin round 1.
        status_samples = self.readers['status'].take()
        for status in status_samples:
            if status and not self.current_round and status.training_started and not status.training_complete:
                if self.model is not None:
                    self.current_round = max(1, status.current_round)
                    print(f"\n[Fallback] Client {self.client_id} starting training for round {self.current_round} based on server status...")
                    self.train_local_model()
                else:
                    print(f"[Fallback] Client {self.client_id} awaiting model before starting training...")
    
    def check_global_model(self):
        """Check for global model updates from server (chunked version)"""
        # Check for chunked global model
        chunk_samples = self.readers['global_model_chunk'].take()
        
        # DEBUG: Log if we're receiving anything
        if len(chunk_samples) > 0:
            print(f"[DEBUG] Client {self.client_id} received {len(chunk_samples)} chunk samples from DataReader")
        
        for sample in chunk_samples:
            if not sample or not hasattr(sample, 'round'):
                continue
            
            round_num = sample.round
            chunk_id = sample.chunk_id
            total_chunks = sample.total_chunks
            
            # Initialize buffers if needed
            if not self.global_model_metadata:
                self.global_model_metadata = {
                    'round': round_num,
                    'total_chunks': total_chunks,
                    'model_config_json': sample.model_config_json if hasattr(sample, 'model_config_json') else ''
                }
                print(f"Client {self.client_id}: Receiving global model in {total_chunks} chunks...")
            
            # Store chunk
            self.global_model_chunks[chunk_id] = sample.payload
            
            # Progress logging - show every 20 chunks to reduce spam
            chunks_received = len(self.global_model_chunks)
            if chunks_received % 20 == 0 or chunks_received == total_chunks:
                print(f"Client {self.client_id}: Received {chunks_received}/{total_chunks} chunks")
            
            # Check if all chunks received
            if chunks_received == total_chunks:
                print(f"Client {self.client_id}: All chunks received, reassembling...")
                
                try:
                    # Reassemble chunks in order
                    reassembled_data = []
                    for i in range(total_chunks):
                        if i in self.global_model_chunks:
                            reassembled_data.extend(self.global_model_chunks[i])
                        else:
                            print(f"ERROR: Missing chunk {i}")
                            break
                    
                    # Only process if we have all chunks
                    print(f"[DEBUG] Client {self.client_id}: Reassembled {len(reassembled_data)} bytes")
                    if len(reassembled_data) > 0:
                        print(f"[DEBUG] Client {self.client_id}: Starting deserialization...")
                        # Deserialize and potentially decompress weights
                        raw_weights = self.deserialize_weights(reassembled_data)
                        print(f"[DEBUG] Client {self.client_id}: Deserialization complete, type: {type(raw_weights)}")
                    
                    # Check if weights are compressed (quantized)
                    if isinstance(raw_weights, dict) and 'compressed_data' in raw_weights:
                        if self.quantizer is not None:
                            weights = self.quantizer.decompress(raw_weights)
                            print(f"Client {self.client_id}: Received and decompressed quantized global model")
                        else:
                            print(f"Client {self.client_id}: ERROR - Received quantized data but quantizer not initialized!")
                            # Clear buffers and continue
                            self.global_model_chunks.clear()
                            self.global_model_metadata.clear()
                            continue
                    else:
                        weights = raw_weights
                    
                    # Check if model needs initialization (works for late-joiners too)
                    print(f"[DEBUG] Client {self.client_id}: Model is None: {self.model is None}")
                    if self.model is None:
                        print(f"Client {self.client_id} received initial global model from server (round {round_num})")
                        
                        # Parse model config if available
                        model_config_json = self.global_model_metadata.get('model_config_json', '')
                        if model_config_json:
                            model_config = json.loads(model_config_json)
                            
                            # Build CNN model from server's architecture definition
                            self.model = Sequential()
                            self.model.add(Input(shape=tuple(model_config['input_shape'])))
                            
                            for layer_config in model_config['layers']:
                                if layer_config['type'] == 'Conv2D':
                                    self.model.add(Conv2D(
                                        filters=layer_config['filters'],
                                        kernel_size=tuple(layer_config['kernel_size']),
                                        activation=layer_config.get('activation')
                                    ))
                                elif layer_config['type'] == 'MaxPooling2D':
                                    self.model.add(MaxPooling2D(pool_size=tuple(layer_config['pool_size'])))
                                elif layer_config['type'] == 'Dropout':
                                    self.model.add(Dropout(layer_config['rate']))
                                elif layer_config['type'] == 'Flatten':
                                    self.model.add(Flatten())
                                elif layer_config['type'] == 'Dense':
                                    self.model.add(Dense(
                                        units=layer_config['units'],
                                        activation=layer_config.get('activation')
                                    ))
                            
                            # Compile model
                            self.model.compile(
                                loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate=0.0001),
                                metrics=['accuracy']
                            )
                            
                            print(f"Client {self.client_id} built CNN model from server configuration")
                            print(f"  Input shape: {model_config['input_shape']}")
                            print(f"  Output classes: {model_config['num_classes']}")
                        else:
                            # Fallback: hardcode architecture if server doesn't send config
                            self.model = Sequential()
                            self.model.add(Input(shape=(48, 48, 1)))
                            self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
                            self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
                            self.model.add(MaxPooling2D(pool_size=(2, 2)))
                            self.model.add(Dropout(0.25))
                            
                            self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
                            self.model.add(MaxPooling2D(pool_size=(2, 2)))
                            self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
                            self.model.add(MaxPooling2D(pool_size=(2, 2)))
                            self.model.add(Dropout(0.25))
                            
                            self.model.add(Flatten())
                            self.model.add(Dense(1024, activation='relu'))
                            self.model.add(Dropout(0.5))
                            self.model.add(Dense(7, activation='softmax'))
                            
                            self.model.compile(
                                loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate=0.0001),
                                metrics=['accuracy']
                            )
                            print(f"Client {self.client_id} using hardcoded CNN model architecture")
                        
                        # Set the initial weights from server
                        self.model.set_weights(weights)
                        print(f"Client {self.client_id} model initialized with server weights")
                        self.current_round = 0
                        # Start round 1 immediately after initial model is ready to avoid
                        # missing one-shot start_training command timing.
                        self.current_round = 1
                        print(f"Client {self.client_id} starting training for round 1 with initial global model...")
                        self.train_local_model()
                        # Training call handles further loop progression.
                        self.global_model_chunks.clear()
                        self.global_model_metadata.clear()
                        return
                    elif round_num == self.current_round:
                        # Updated model after aggregation
                        self.model.set_weights(weights)
                        print(f"Client {self.client_id} received global model for round {self.current_round}")
                        
                        # Evaluate immediately after receiving global model
                        print(f"Client {self.client_id} starting evaluation for round {self.current_round}...")
                        self.evaluate_model()
                    
                    # Clear chunk buffers
                    self.global_model_chunks.clear()
                    self.global_model_metadata.clear()
                    
                except Exception as e:
                    print(f"[ERROR] Client {self.client_id}: Exception during model reassembly: {e}")
                    import traceback
                    traceback.print_exc()
                    # Clear buffers on error
                    self.global_model_chunks.clear()
                    self.global_model_metadata.clear()
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        # Limit steps per epoch for faster training (configurable via env)
        try:
            steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
            val_steps = int(os.getenv("VAL_STEPS", "25"))
        except Exception:
            steps_per_epoch = 100
            val_steps = 25
        
        # Train the model
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            verbose=2
        )
        
        # Get final metrics
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        # Get model weights
        weights = self.model.get_weights()
        # Compress or serialize weights
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(weights, data_type="weights")
            stats = self.quantizer.get_compression_stats(weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Size: {stats['compressed_size_mb']:.2f}MB")
            # Serialize compressed data (pickle + convert to list of ints for DDS)
            serialized_weights = list(pickle.dumps(compressed_data))
        else:
            serialized_weights = self.serialize_weights(weights)
        
        # Introduce random delay before sending model update
        delay = random.uniform(0.5, 3.0)  # Random delay between 0.5 and 3.0 seconds
        print(f"Client {self.client_id} waiting {delay:.2f} seconds before sending update...")
        time.sleep(delay)
        
        # Send model update to server using chunking
        self.send_model_update_chunked(
            self.current_round,
            serialized_weights,
            self.train_generator.n,
            float(final_loss),
            float(final_accuracy)
        )
        
        print(f"Client {self.client_id} sent model update for round {self.current_round}")
        print(f"Training metrics - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        
        # Small delay to ensure message is sent
        time.sleep(0.5)
        
        # Wait for global model after training
        print(f"Client {self.client_id} waiting for global model for round {self.current_round}...")
        self.wait_for_global_model()
    
    def wait_for_global_model(self):
        """Actively wait for global model after training"""
        timeout = 30
        start_time = time.time()
        check_count = 0
        
        while time.time() - start_time < timeout:
            samples = self.readers['global_model'].take()
            check_count += 1
            
            if check_count % 50 == 0:  # Log every 5 seconds
                print(f"Client {self.client_id} still waiting... (checked {check_count} times)")
            
            for sample in samples:
                if sample:
                    print(f"Client {self.client_id} received sample for round {sample.round} (expecting {self.current_round})")
                    if sample.round == self.current_round:
                        # Update local model with global weights
                        # Deserialize and potentially decompress weights
                        raw_weights = self.deserialize_weights(sample.weights)
                        
                        # Check if weights are compressed (quantized)
                        if isinstance(raw_weights, dict) and 'compressed_data' in raw_weights:
                            if self.quantizer is not None:
                                weights = self.quantizer.decompress(raw_weights)
                                print(f"Client {self.client_id}: Received and decompressed quantized global model")
                            else:
                                print(f"Client {self.client_id}: ERROR - Received quantized data but quantizer not initialized!")
                                return
                        else:
                            weights = raw_weights
                        
                        self.model.set_weights(weights)
                        print(f"Client {self.client_id} received global model for round {self.current_round}")
                        
                        # Evaluate immediately
                        print(f"Client {self.client_id} starting evaluation for round {self.current_round}...")
                        self.evaluate_model()
                        return
            time.sleep(0.1)
        
        print(f"Client {self.client_id} WARNING: Timeout waiting for global model round {self.current_round}")
    
    def _update_local_convergence(self, loss: float):
        """Track client-local convergence and disconnect when converged."""
        if self.current_round < MIN_ROUNDS:
            self.best_loss = min(self.best_loss, loss)
            return
        if self.best_loss - loss > CONVERGENCE_THRESHOLD:
            self.best_loss = loss
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1
        if self.rounds_without_improvement >= CONVERGENCE_PATIENCE and not self.has_converged:
            self.has_converged = True
            print(f"Client {self.client_id} reached local convergence at round {self.current_round}")

    def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server"""
        # Evaluate on validation set
        loss, accuracy = self.model.evaluate(self.validation_generator, verbose=0)
        
        self._update_local_convergence(float(loss))
        client_converged = 1.0 if self.has_converged else 0.0
        
        # Send metrics to server
        metrics = EvaluationMetrics(
            client_id=self.client_id,
            round=self.current_round,
            num_samples=self.validation_generator.n,
            loss=float(loss),
            accuracy=float(accuracy),
            client_converged=client_converged
        )
        
        # Write with explicit return check
        result = self.writers['metrics'].write(metrics)
        
        # Wait to ensure message is sent
        time.sleep(0.5)
        
        print(f"Client {self.client_id} sent evaluation metrics for round {self.current_round}")
        print(f"Evaluation metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
        if self.has_converged:
            print(f"Client {self.client_id} notifying server of convergence and disconnecting")
            self.running = False
    
    def cleanup(self):
        """Cleanup DDS resources"""
        if self.participant:
            # DomainParticipant will be automatically cleaned up
            self.participant = None
        print(f"Client {self.client_id} DDS resources cleaned up")


def load_data(client_id):
    """Load emotion recognition data for this client"""
    # Detect environment: Docker uses /app prefix, local uses relative path
    if os.path.exists('/app'):
        base_path = '/app/Client/Emotion_Recognition/Dataset'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(project_root, 'Client', 'Emotion_Recognition', 'Dataset')
    
    train_path = os.path.join(base_path, f'client_{client_id}', 'train')
    validation_path = os.path.join(base_path, f'client_{client_id}', 'validation')
    print(f"Dataset base path: {base_path}")
    
    # Initialize image data generator with rescaling
    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    # Load training and validation data
    train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_data_gen.flow_from_directory(
        validation_path,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator


def main():
    """Main entry point"""
    # Load data
    print(f"Loading dataset for client {CLIENT_ID}...")
    train_generator, validation_generator = load_data(CLIENT_ID)
    print(f"Dataset loaded\n")
    
    # Create and run client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    client.run()


if __name__ == "__main__":
    main()
