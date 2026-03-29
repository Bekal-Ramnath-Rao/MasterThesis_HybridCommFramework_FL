import numpy as np
import os
import sys
import logging
import json
import pickle
import time
import random
import threading

# GPU Configuration - Must be done BEFORE TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
if _xla_flags:
    sanitized_flags = [f for f in _xla_flags.split() if f != "--xla_gpu_enable_command_buffer="]
    if sanitized_flags:
        os.environ["XLA_FLAGS"] = " ".join(sanitized_flags)
    else:
        os.environ.pop("XLA_FLAGS", None)
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
                    [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "4000")))]
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

# Set CycloneDDS config before any cyclonedds import (native lib may read at load time)
if not os.environ.get("CYCLONEDDS_URI") and os.path.exists("/app/config"):
    _cid = os.environ.get("CLIENT_ID", "1")
    _path = f"/app/config/cyclonedds-emotion-client{_cid}.xml"
    if os.path.exists(_path):
        os.environ["CYCLONEDDS_URI"] = f"file://{_path}"

from cyclonedds.domain import DomainParticipant

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

# Battery model (shared with gRPC/MQTT/AMQP/HTTP3)
_client_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _client_dir not in sys.path:
    sys.path.insert(0, _client_dir)
from battery_model import BatteryModel

from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.util import duration
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from cyclonedds.core import Qos, Policy
from dataclasses import dataclass
from typing import List

# DDS implementation vendor (CycloneDDS vs Fast DDS placeholder)
DDS_IMPL = os.getenv("DDS_IMPL", "cyclonedds").strip().lower()
print(f"DDS implementation (client {os.getenv('CLIENT_ID', '0')}): {DDS_IMPL}")
if DDS_IMPL not in ("cyclonedds", "fastdds"):
    print("Warning: Unknown DDS_IMPL; defaulting to CycloneDDS transport.")
elif DDS_IMPL == "fastdds":
    print("Note: Fast DDS integration is not yet implemented; using CycloneDDS stack for now.")

# DDS Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
DEFAULT_DATA_BATCH_SIZE = int(os.getenv("DEFAULT_DATA_BATCH_SIZE", "16"))

# Controls whether this client should signal/exit on local convergence.
# When false, clients keep training until the server indicates completion.
from fl_termination_env import stop_on_client_convergence

# Chunking configuration for large messages
CHUNK_SIZE = 64 * 1024


# DDS Data Types (matching IDL)
@dataclass
class ClientRegistration(IdlStruct):
    client_id: int
    message: str


@dataclass
class ClientReady(IdlStruct):
    """Signal from client that it's ready to receive model"""
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
    model_config_json: str = ""
    model_config_octets: sequence[int] = ()  # Primitive encoding; decode to get model_config_json


@dataclass
class GlobalModelChunk(IdlStruct):
    round: int
    chunk_id: int
    total_chunks: int
    payload: sequence[int]
    model_config_json: str = ""


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
    battery_soc: float = 1.0


@dataclass
class ServerStatus(IdlStruct):
    current_round: int
    total_rounds: int
    training_started: bool
    training_complete: bool
    registered_clients: int


def _octets_to_config(octets: list) -> str:
    """Decode primitive sequence[int] back to config JSON string (4 bytes per int)."""
    if not octets:
        return ""
    b = b"".join(int(x).to_bytes(4, "little") for x in octets)
    return b.rstrip(b"\x00").decode("utf-8", errors="replace")


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, train_generator=None, validation_generator=None):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        
        # Battery/energy model for consumption tracking (server uses for battery plot)
        self.battery_model = BatteryModel(protocol="dds")
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
        self.last_training_round = -1
        self.training_config = {"batch_size": 16, "local_epochs": 20}
        self.running = True
        self._training_lock = threading.Lock()
        self._training_thread = None
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.has_converged = False
        self.global_model_chunks = {}
        self.global_model_metadata = {}
        self._last_evaluated_round = -1
        
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

    def _apply_generator_batch_size(self, batch_size):
        """Sync DirectoryIterator batch size with training config."""
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            return
        if batch_size <= 0:
            return

        changed = False
        for gen_name in ("train_generator", "validation_generator"):
            generator = getattr(self, gen_name, None)
            if generator is None:
                continue
            current = getattr(generator, "batch_size", None)
            if current != batch_size:
                setattr(generator, "batch_size", batch_size)
                changed = True
        if changed:
            print(f"Client {self.client_id} synchronized generator batch_size to {batch_size}")

    def _launch_training_for_round(self, round_num):
        """Start local training in a background thread to keep DDS polling responsive."""
        with self._training_lock:
            if self._training_thread is not None and self._training_thread.is_alive():
                print(f"Client {self.client_id} training already in progress; ignoring start for round {round_num}")
                return
            if self.last_training_round == round_num:
                print(f"Client {self.client_id} ignoring duplicate start training for round {round_num}")
                return

            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")

            self._training_thread = threading.Thread(
                target=self._train_local_model_safe,
                args=(round_num,),
                daemon=True,
            )
            self._training_thread.start()

    def _train_local_model_safe(self, round_num):
        try:
            self.train_local_model()
        except Exception as e:
            print(f"Client {self.client_id} training thread error on round {round_num}: {e}")
            import traceback
            traceback.print_exc()
    
    def send_model_update(self, round_num, serialized_weights, num_samples, loss, accuracy):
        """Send model update in a single Reliable message."""
        msg = ModelUpdate(
            client_id=self.client_id,
            round=round_num,
            weights=serialized_weights,
            num_samples=num_samples,
            loss=loss,
            accuracy=accuracy,
        )
        self.writers['model_update'].write(msg)
        print(f"Client {self.client_id}: Sent model update for round {round_num} ({len(serialized_weights)} bytes)")

    def split_into_chunks(self, data):
        chunks = []
        for i in range(0, len(data), CHUNK_SIZE):
            chunks.append(data[i:i + CHUNK_SIZE])
        return chunks

    def send_model_update_chunked(self, round_num, serialized_weights, num_samples, loss, accuracy):
        """Send model update as DDS chunks to keep payloads bounded."""
        chunks = self.split_into_chunks(serialized_weights)
        total_chunks = len(chunks)
        print(
            f"Client {self.client_id}: Sending model update in {total_chunks} chunks "
            f"({len(serialized_weights)} bytes total)"
        )
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = ModelUpdateChunk(
                client_id=self.client_id,
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                num_samples=num_samples,
                loss=loss,
                accuracy=accuracy,
            )
            self.writers['model_update_chunk'].write(chunk)
            if (chunk_id + 1) % 20 == 0:
                print(f"  Sent {chunk_id + 1}/{total_chunks} chunks")
    
    def setup_dds(self):
        """Initialize DDS participant, topics, readers, and writers"""
        # Ensure CycloneDDS uses discovery server config (must be set before participant creation)
        uri = os.environ.get("CYCLONEDDS_URI")
        if not uri and os.path.exists("/app/config"):
            cid = os.environ.get("CLIENT_ID", "1")
            path = f"/app/config/cyclonedds-emotion-client{cid}.xml"
            if os.path.exists(path):
                uri = f"file://{path}"
                os.environ["CYCLONEDDS_URI"] = uri
        print(f"[DDS] CYCLONEDDS_URI={uri or os.environ.get('CYCLONEDDS_URI', '(not set)')}")
        print(f"Setting up DDS on domain {DDS_DOMAIN_ID}...")
        
        # Create domain participant
        self.participant = DomainParticipant(DDS_DOMAIN_ID)
        
        # Reliable QoS for control/small messages
        reliable_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
            Policy.History.KeepLast(10),
            Policy.Durability.TransientLocal
        )
        # Long blocking time + resource limits for large model updates (9MB; round 1 has no tc but discovery must be ready)
        reliable_qos_large = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=600)),  # 10 min for degraded networks
            Policy.History.KeepLast(10),
            Policy.Durability.TransientLocal,
            Policy.ResourceLimits(max_samples=10, max_instances=10, max_samples_per_instance=10),  # allow 9MB samples
        )
        # Dedicated QoS for chunked model transfer (matches unified path)
        chunk_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
            Policy.History.KeepLast(2048),
            Policy.Durability.TransientLocal,
        )
        # Create topics (single-message + chunked for bounded payloads)
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
        
        # Create readers (for receiving from server) — all Reliable
        self.readers['config'] = DataReader(self.participant, topic_config, qos=reliable_qos)
        self.readers['command'] = DataReader(self.participant, topic_command, qos=reliable_qos)
        self.readers['global_model'] = DataReader(self.participant, topic_global_model, qos=reliable_qos)
        self.readers['global_model_chunk'] = DataReader(self.participant, topic_global_model_chunk, qos=chunk_qos)
        self.readers['status'] = DataReader(self.participant, topic_status, qos=reliable_qos)
        
        # Create writers (for sending to server) — Reliable; model_update uses long timeout for large payload
        self.writers['registration'] = DataWriter(self.participant, topic_registration, qos=reliable_qos)
        self.writers['ready'] = DataWriter(self.participant, topic_ready, qos=reliable_qos)
        self.writers['model_update'] = DataWriter(self.participant, topic_model_update, qos=reliable_qos_large)
        self.writers['model_update_chunk'] = DataWriter(self.participant, topic_model_update_chunk, qos=chunk_qos)
        self.writers['metrics'] = DataWriter(self.participant, topic_metrics, qos=reliable_qos)

        print(f"Client {self.client_id} DDS setup complete (Reliable QoS; model_update 10 min blocking for large uploads)")
        
        # Wait for DDS endpoint discovery before sending
        print(f"Client {self.client_id} waiting for endpoint discovery...")
        time.sleep(2.0)
        
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
        
        # Signal ready to receive model
        time.sleep(2.0)
        ready_signal = ClientReady(
            client_id=self.client_id,
            ready_for_chunks=True
        )
        print(f"Client {self.client_id} signaling ready...")
        for i in range(3):
            self.writers['ready'].write(ready_signal)
            time.sleep(0.2)
        print(f"Client {self.client_id} ready signal sent")
        # Allow time for model_update reader (server) to be discovered so round 1 send is delivered (no tc yet)
        wait_s = int(os.environ.get("DDS_MODEL_UPDATE_DISCOVERY_WAIT", "12"))
        if wait_s > 0:
            print(f"Client {self.client_id} waiting {wait_s}s for model_update endpoint discovery...")
            time.sleep(wait_s)
        print()
    
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
        
        try:
            while self.running:
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
                    self._apply_generator_batch_size(self.training_config["batch_size"])
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
                        self._launch_training_for_round(sample.round)
                    elif sample.round > self.current_round:
                        # Subsequent rounds
                        self._launch_training_for_round(sample.round)

        # Fallback: if we've received the initial global model but no command within a short window,
        # and server status indicates training has started, proactively begin round 1.
        status_samples = self.readers['status'].take()
        for status in status_samples:
            if status and not self.current_round and status.training_started and not status.training_complete:
                if self.model is not None:
                    self._launch_training_for_round(max(1, status.current_round))
    
    def _apply_global_model_payload(self, round_num, serialized_weights, config_json=""):
        """Deserialize and apply global model payload (supports both single and chunked DDS paths)."""
        try:
            raw_weights = self.deserialize_weights(serialized_weights)
        except Exception as e:
            print(f"[ERROR] Client {self.client_id}: deserialize failed: {e}")
            return
        if isinstance(raw_weights, dict) and 'compressed_data' in raw_weights:
            if self.quantizer is not None:
                weights = self.quantizer.decompress(raw_weights)
                print(f"Client {self.client_id}: Received global model (dequantized for training)")
            else:
                print(f"Client {self.client_id}: ERROR - quantized data but quantizer not initialized")
                return
        else:
            weights = raw_weights

        def build_model_from_config(model_config_json):
            if not model_config_json:
                return
            model_config = json.loads(model_config_json)
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
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.0001),
                metrics=['accuracy']
            )
            print(f"Client {self.client_id} built CNN from server config")

        if self.model is None:
            print(f"Client {self.client_id} received initial global model (round {round_num})")
            build_model_from_config(config_json)
            if self.model is None:
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
            self.model.set_weights(weights)
            self.current_round = 0
            print(f"Client {self.client_id} initialized model; waiting for start_training command for round 1...")
            return
        if round_num == self.current_round:
            self.model.set_weights(weights)
            print(f"Client {self.client_id} received global model for round {self.current_round}")
            print(f"Client {self.client_id} starting evaluation for round {self.current_round}...")
            self._last_evaluated_round = self.current_round
            self.evaluate_model()

    def _apply_global_model(self, sample):
        """Process a single GlobalModel sample (legacy non-chunked path)."""
        round_num = sample.round
        config_json = ""
        if getattr(sample, 'model_config_octets', None) and len(sample.model_config_octets) > 0:
            config_json = _octets_to_config(sample.model_config_octets)
        if not config_json and getattr(sample, 'model_config_json', None):
            config_json = sample.model_config_json or ""
        self._apply_global_model_payload(round_num, sample.weights, config_json)

    def check_global_model(self):
        """Check for global model updates from server (chunked + legacy single-message)."""
        chunk_samples = self.readers['global_model_chunk'].take()
        for sample in chunk_samples:
            if not sample or not hasattr(sample, 'round'):
                continue
            round_num = sample.round
            chunk_id = sample.chunk_id
            total_chunks = sample.total_chunks
            if not self.global_model_metadata:
                self.global_model_metadata = {
                    'round': round_num,
                    'total_chunks': total_chunks,
                    'model_config_json': sample.model_config_json if hasattr(sample, 'model_config_json') else ''
                }
                print(f"Client {self.client_id}: Receiving global model in {total_chunks} DDS chunks...")

            self.global_model_chunks[chunk_id] = sample.payload

            if len(self.global_model_chunks) == total_chunks:
                reassembled_data = []
                missing_chunk = False
                for index in range(total_chunks):
                    if index in self.global_model_chunks:
                        reassembled_data.extend(self.global_model_chunks[index])
                    else:
                        missing_chunk = True
                        print(f"Client {self.client_id}: Missing global-model chunk {index}")
                        break

                if not missing_chunk and len(reassembled_data) > 0:
                    config_json = self.global_model_metadata.get('model_config_json', '')
                    self._apply_global_model_payload(round_num, reassembled_data, config_json)

                self.global_model_chunks.clear()
                self.global_model_metadata.clear()

        samples = self.readers['global_model'].take()
        for sample in samples:
            if not sample or not hasattr(sample, 'round') or not hasattr(sample, 'weights'):
                continue
            self._apply_global_model(sample)
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        self._apply_generator_batch_size(batch_size)
        epochs = self.training_config['local_epochs']
        # Limit steps per epoch for faster training (configurable via env)
        try:
            steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
            val_steps = int(os.getenv("VAL_STEPS", "25"))
        except Exception:
            steps_per_epoch = 100
            val_steps = 25
        
        training_start = time.time()
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
        
        # FAIR FIX: no artificial pre-send delay; align behavior with unified and other protocols.
        
        payload_bytes = len(serialized_weights)
        training_time = time.time() - training_start
        if payload_bytes > 1_000_000 and os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            print(f"Client {self.client_id} sending model update ({payload_bytes / 1e6:.2f} MB); may take 1–2 min on moderate network...")
        send_start_ts = time.time()
        send_start_cpu = time.perf_counter() if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1" else None
        # Send model update to server (chunked DDS path)
        self.send_model_update_chunked(
            self.current_round,
            serialized_weights,
            self.train_generator.n,
            float(final_loss),
            float(final_accuracy)
        )
        communication_time = time.time() - send_start_ts
        self.battery_model.update(payload_bytes, 0, training_time, communication_time)
        if send_start_cpu is not None:
            O_send = time.perf_counter() - send_start_cpu
            print(f"FL_DIAG O_send={O_send:.9f} payload_bytes={payload_bytes} send_start_ts={send_start_ts:.9f}")
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
        self._last_evaluated_round = -1
        check_count = 0
        
        while time.time() - start_time < timeout:
            check_count += 1
            self.check_global_model()
            if self._last_evaluated_round == self.current_round:
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
        client_converged = 1.0 if (self.has_converged and stop_on_client_convergence()) else 0.0
        
        # Send metrics to server (include battery_soc for server battery consumption plot)
        metrics = EvaluationMetrics(
            client_id=self.client_id,
            round=self.current_round,
            num_samples=self.validation_generator.n,
            loss=float(loss),
            accuracy=float(accuracy),
            client_converged=client_converged,
            battery_soc=float(self.battery_model.battery_soc),
        )
        
        # Write with explicit return check
        result = self.writers['metrics'].write(metrics)
        
        # Wait to ensure message is sent
        time.sleep(0.5)
        
        print(f"Client {self.client_id} sent evaluation metrics for round {self.current_round}")
        print(f"Evaluation metrics - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}\n")
        if self.has_converged and stop_on_client_convergence():
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
        batch_size=DEFAULT_DATA_BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_data_gen.flow_from_directory(
        validation_path,
        target_size=(48, 48),
        batch_size=DEFAULT_DATA_BATCH_SIZE,
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
