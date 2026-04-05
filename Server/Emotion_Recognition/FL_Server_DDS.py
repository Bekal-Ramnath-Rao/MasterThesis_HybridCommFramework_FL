import os
import sys
# Server uses CPU only (aggregation is numpy-only); must be set before any TensorFlow import
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pickle
import time
import json
from pathlib import Path

# Set CycloneDDS config before any cyclonedds import (native lib may read at load time)
def _emotion_config_dir():
    if os.path.exists("/app"):
        return "/app/config"
    # Server/Emotion_Recognition/FL_Server_DDS.py -> repo/config
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))


def _try_distributed_unicast_server():
    """Static SPDP peers (DDS_PEER_*); see config/dds_distributed_unicast.py."""
    base = _emotion_config_dir()
    helper = os.path.join(base, "dds_distributed_unicast.py")
    if not os.path.isfile(helper):
        return False
    import importlib.util

    spec = importlib.util.spec_from_file_location("dds_distributed_unicast", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.try_apply_server_uri()


def _ensure_server_cyclonedds_uri():
    """Explicit CYCLONEDDS_URI; else static unicast (DDS_PEER_*); else LAN multicast; else localhost peers."""
    if os.environ.get("CYCLONEDDS_URI"):
        return
    if _try_distributed_unicast_server():
        return
    base = _emotion_config_dir()
    for name in ("cyclonedds-multicast-lan.xml", "cyclonedds-emotion-server.xml"):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            os.environ["CYCLONEDDS_URI"] = "file://" + os.path.abspath(p)
            return


_ensure_server_cyclonedds_uri()

# Project root and utilities (for experiment_results path)
if os.path.exists("/app"):
    _project_root = "/app"
else:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_utilities_path = os.path.join(_project_root, "scripts", "utilities")
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)
try:
    from experiment_results_path import get_experiment_results_dir
except ModuleNotFoundError:
    def get_experiment_results_dir(use_case: str, protocol: str, scenario: str = None) -> Path:
        if scenario is None:
            scenario = os.getenv("NETWORK_SCENARIO", "default").strip() or "default"
        root = Path("/app") if os.path.exists("/app") else Path(_project_root)
        path = root / "experiment_results" / use_case / protocol / scenario
        path.mkdir(parents=True, exist_ok=True)
        return path

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

# DDS implementation vendor (CycloneDDS vs Fast DDS placeholder)
DDS_IMPL = os.getenv("DDS_IMPL", "cyclonedds").strip().lower()
print(f"DDS implementation (server): {DDS_IMPL}")
if DDS_IMPL not in ("cyclonedds", "fastdds"):
    print("Warning: Unknown DDS_IMPL; defaulting to CycloneDDS transport.")
elif DDS_IMPL == "fastdds":
    print("Note: Fast DDS integration is not yet implemented; using CycloneDDS stack for now.")

# Server Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence
from fl_termination_env import stop_on_client_convergence

# Chunking configuration for large messages
CHUNK_SIZE = 64 * 1024

# Convergence Settings (primary stopping criterion)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))


# DDS Data Types (primitive-heavy for low Tactual: int, long, double, bool, sequence<int>; avoid strings/nested)
@dataclass
class ClientRegistration(IdlStruct):
    client_id: int
    message: str  # Keep short; discovery only


@dataclass
class ClientReady(IdlStruct):
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
    model_config_octets: sequence[int] = ()  # Primitive encoding: JSON as 4-byte ints; use when non-empty


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


class FederatedLearningServer:
    def __init__(self, min_clients, num_rounds, max_clients=100):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.num_clients = min_clients  # Start with minimum, will update as clients join
        self.num_rounds = num_rounds
        self.num_classes = 7  # For emotion recognition
        self.current_round = 0
        self.registered_clients = set()
        self.active_clients = set()
        self.ready_clients = set()  # Track clients ready to receive model
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        
        # Chunk reassembly buffers
        self.model_update_chunks = {}
        self.model_update_metadata = {}
        
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
        batch_size_env = int(os.getenv("BATCH_SIZE", "16"))
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
        """Split serialized payload into bounded DDS chunks."""
        chunks = []
        for i in range(0, len(data), CHUNK_SIZE):
            chunks.append(data[i:i + CHUNK_SIZE])
        return chunks
    
    @staticmethod
    def _config_to_octets(s: str):
        """Encode config string as sequence[int] (4 bytes per int) for primitive-heavy DDS."""
        if not s:
            return []
        b = s.encode("utf-8")
        rem = len(b) % 4
        if rem:
            b += b"\x00" * (4 - rem)
        return [int.from_bytes(b[i : i + 4], "little") for i in range(0, len(b), 4)]

    def send_global_model(self, round_num, serialized_weights, model_config):
        """Send global model in a single Reliable message."""
        use_primitive = os.getenv("DDS_USE_PRIMITIVE_CONFIG", "1").strip().lower() in ("1", "true", "yes")
        config_octets = self._config_to_octets(model_config) if use_primitive else []
        config_json = "" if use_primitive else model_config
        msg = GlobalModel(
            round=round_num,
            weights=serialized_weights,
            model_config_json=config_json,
            model_config_octets=config_octets,
        )
        self.writers['global_model'].write(msg)
        print(f"Sent global model for round {round_num} ({len(serialized_weights)} bytes)")

    def send_global_model_chunked(self, round_num, serialized_weights, model_config):
        """Send global model as chunked DDS messages to enforce bounded payload size."""
        chunks = self.split_into_chunks(serialized_weights)
        total_chunks = len(chunks)
        print(
            f"Sending global model in {total_chunks} chunks "
            f"({len(serialized_weights)} bytes total)"
        )
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = GlobalModelChunk(
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                model_config_json=model_config if chunk_id == 0 else "",
            )
            self.writers['global_model_chunk'].write(chunk)
            if (chunk_id + 1) % 20 == 0:
                print(f"  Sent {chunk_id + 1}/{total_chunks} chunks")
    
    def setup_dds(self):
        """Initialize DDS participant, topics, readers, and writers"""
        # Ensure CycloneDDS uses discovery server config (must be set before participant creation)
        _ensure_server_cyclonedds_uri()
        uri = os.environ.get("CYCLONEDDS_URI")
        print(f"[DDS] CYCLONEDDS_URI={uri or '(not set)'}")
        print(f"Setting up DDS on domain {DDS_DOMAIN_ID}...")
        
        # Create domain participant
        self.participant = DomainParticipant(DDS_DOMAIN_ID)
        
        # Reliable QoS for control/small messages
        reliable_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
            Policy.History.KeepLast(10),
            Policy.Durability.TransientLocal,
        )
        # Long blocking + resource limits for large model updates (9MB; must accept large samples in round 1 too)
        reliable_qos_large = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=600)),
            Policy.History.KeepLast(10),
            Policy.Durability.TransientLocal,
            Policy.ResourceLimits(max_samples=10, max_instances=10, max_samples_per_instance=10),
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
        
        # # Create readers (for receiving from clients) with reliable QoS
        # self.readers['registration'] = DataReader(self.participant, topic_registration, qos=reliable_qos)
        # self.readers['model_update'] = DataReader(self.participant, topic_model_update, qos=reliable_qos)
        # self.readers['metrics'] = DataReader(self.participant, topic_metrics, qos=reliable_qos)
        
        # # Create writers (for sending to clients) with reliable QoS
        # self.writers['config'] = DataWriter(self.participant, topic_config, qos=reliable_qos)
        # self.writers['command'] = DataWriter(self.participant, topic_command, qos=reliable_qos)
        # self.writers['global_model'] = DataWriter(self.participant, topic_global_model, qos=reliable_qos)
        # self.writers['status'] = DataWriter(self.participant, topic_status, qos=reliable_qos)

        # Create readers (for receiving from clients) — Reliable; model_update uses long timeout for large payloads
        self.readers['registration'] = DataReader(self.participant, topic_registration, qos=reliable_qos)
        self.readers['ready'] = DataReader(self.participant, topic_ready, qos=reliable_qos)
        self.readers['model_update'] = DataReader(self.participant, topic_model_update, qos=reliable_qos_large)
        self.readers['model_update_chunk'] = DataReader(self.participant, topic_model_update_chunk, qos=chunk_qos)
        self.readers['metrics'] = DataReader(self.participant, topic_metrics, qos=reliable_qos)
        
        # Create writers (for sending to clients) — all Reliable
        self.writers['config'] = DataWriter(self.participant, topic_config, qos=reliable_qos)
        self.writers['command'] = DataWriter(self.participant, topic_command, qos=reliable_qos)
        self.writers['global_model'] = DataWriter(self.participant, topic_global_model, qos=reliable_qos)
        self.writers['global_model_chunk'] = DataWriter(self.participant, topic_global_model_chunk, qos=chunk_qos)
        self.writers['status'] = DataWriter(self.participant, topic_status, qos=reliable_qos)
        
        print("DDS setup complete (Reliable QoS; model_update reader 10 min for large uploads)\n")
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
        time.sleep(5.0)
        print("DDS endpoints ready\n")
        
        # Publish initial training config
        config = TrainingConfig(
            batch_size=self.training_config['batch_size'],
            local_epochs=self.training_config['local_epochs']
        )
        self.writers['config'].write(config)
        
        try:
            while not self.training_complete:
                try:
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
                    
                    time.sleep(0.05)
                    
                except Exception as loop_error:
                    print(f"[FATAL] Unhandled exception in server main loop: {loop_error}")
                    import traceback
                    traceback.print_exc()
                    sys.stdout.flush()
                    # Continue loop despite error
                    time.sleep(1)

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
        
        for sample in samples:
            # Some DDS implementations may emit InvalidSample entries; guard against those
            if not sample or not hasattr(sample, 'client_id'):
                continue
            client_id = sample.client_id
            if client_id not in self.registered_clients:
                self.registered_clients.add(client_id)
                self.active_clients.add(client_id)
                print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))
                
        # If all clients registered, distribute initial global model and start training
        # Start when we have at least min_clients; after one converges we proceed with remaining active clients
        if len(self.registered_clients) >= self.min_clients and not self.training_started:
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
        self.round_start_time = time.time()
        
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
        
        # Wait for all clients to signal ready to receive model
        print("Waiting for all clients to signal ready...")
        max_wait = 30
        start_time = time.time()
        while time.time() - start_time < max_wait:
            ready_samples = self.readers['ready'].take()
            for sample in ready_samples:
                if sample and sample.ready_for_chunks:
                    self.ready_clients.add(sample.client_id)
                    print(f"[READY] Client {sample.client_id} ready ({len(self.ready_clients)}/{self.num_clients})")
            if len(self.ready_clients) >= self.num_clients:
                print(f"[SUCCESS] All {self.num_clients} clients ready!")
                break
            time.sleep(0.5)
        if len(self.ready_clients) < self.num_clients:
            print(f"[WARNING] Only {len(self.ready_clients)}/{self.num_clients} clients signaled ready. Proceeding anyway.")

        # Send initial global model in chunked messages
        print("Publishing initial model to clients...")
        self.send_global_model_chunked(0, serialized_weights, json.dumps(model_config))
        print("Initial global model sent to all clients")
        
        # Wait for clients to receive and set the initial model
        print("Waiting for clients to receive and build the model...")
        time.sleep(2)
        
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
        """Check for model updates from clients (chunked + legacy single-message paths)."""
        chunk_samples = self.readers['model_update_chunk'].take()
        for sample in chunk_samples:
            if not sample or not hasattr(sample, 'round') or sample.round != self.current_round:
                continue
            client_id = sample.client_id
            if client_id not in self.active_clients or client_id in self.client_updates:
                continue

            chunk_id = sample.chunk_id
            total_chunks = sample.total_chunks
            if client_id not in self.model_update_chunks:
                self.model_update_chunks[client_id] = {}
                self.model_update_metadata[client_id] = {
                    'total_chunks': total_chunks,
                    'num_samples': sample.num_samples,
                    'loss': sample.loss,
                    'accuracy': sample.accuracy,
                }
            self.model_update_chunks[client_id][chunk_id] = sample.payload

            if len(self.model_update_chunks[client_id]) == total_chunks:
                reassembled_data = []
                missing_chunk = False
                for index in range(total_chunks):
                    if index in self.model_update_chunks[client_id]:
                        reassembled_data.extend(self.model_update_chunks[client_id][index])
                    else:
                        missing_chunk = True
                        print(f"Server: Missing update chunk {index} from client {client_id}")
                        break
                if not missing_chunk and len(reassembled_data) > 0:
                    recv_start_cpu = time.perf_counter() if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1" else None
                    try:
                        if self.quantization_handler is not None:
                            try:
                                compressed_data = pickle.loads(bytes(reassembled_data))
                                # Keep quantized end-to-end: do NOT decompress/dequantize on server.
                                metadata = self.model_update_metadata[client_id]
                                self.client_updates[client_id] = {
                                    'compressed_data': compressed_data,
                                    'num_samples': metadata['num_samples'],
                                    'metrics': {'loss': metadata['loss'], 'accuracy': metadata['accuracy']}
                                }
                                print(f"Server: Received quantized update from client {client_id} (kept quantized)")
                                weights = None
                            except Exception as e:
                                print(f"Server: Failed to decompress from client {client_id}, falling back: {e}")
                                weights = self.deserialize_weights(reassembled_data)
                        else:
                            weights = self.deserialize_weights(reassembled_data)
                    except Exception as e:
                        print(f"Server: Failed to deserialize chunked update from client {client_id}: {e}")
                        weights = None

                    if weights is not None and self.quantization_handler is None:
                        if recv_start_cpu is not None:
                            O_recv = time.perf_counter() - recv_start_cpu
                            recv_end_ts = time.time()
                            print(f"FL_DIAG client_id={client_id} O_recv={O_recv:.9f} recv_end_ts={recv_end_ts:.9f}")
                        metadata = self.model_update_metadata[client_id]
                        self.client_updates[client_id] = {
                            'weights': weights,
                            'num_samples': metadata['num_samples'],
                            'metrics': {'loss': metadata['loss'], 'accuracy': metadata['accuracy']}
                        }
                        print(f"Received update from client {client_id} ({len(self.client_updates)}/{len(self.active_clients)})")

                self.model_update_chunks.pop(client_id, None)
                self.model_update_metadata.pop(client_id, None)

        # Legacy single-message fallback
        samples = self.readers['model_update'].take()
        for sample in samples:
            if not sample or not hasattr(sample, 'round') or sample.round != self.current_round:
                continue
            client_id = sample.client_id
            if client_id not in self.active_clients or client_id in self.client_updates:
                continue
            recv_start_cpu = time.perf_counter() if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1" else None
            try:
                if self.quantization_handler is not None:
                    try:
                        compressed_data = pickle.loads(bytes(sample.weights))
                        # Keep quantized end-to-end: do NOT decompress/dequantize on server.
                        self.client_updates[client_id] = {
                            'compressed_data': compressed_data,
                            'num_samples': sample.num_samples,
                            'metrics': {'loss': sample.loss, 'accuracy': sample.accuracy}
                        }
                        print(f"Server: Received quantized update from client {client_id} (kept quantized)")
                        weights = None
                    except Exception as e:
                        print(f"Server: Failed to decompress from client {client_id}, falling back: {e}")
                        weights = self.deserialize_weights(sample.weights)
                else:
                    weights = self.deserialize_weights(sample.weights)
            except Exception as e:
                print(f"Server: Failed to deserialize update from client {client_id}: {e}")
                continue
            if recv_start_cpu is not None:
                O_recv = time.perf_counter() - recv_start_cpu
                recv_end_ts = time.time()
                print(f"FL_DIAG client_id={client_id} O_recv={O_recv:.9f} recv_end_ts={recv_end_ts:.9f}")
            if weights is not None and self.quantization_handler is None:
                self.client_updates[client_id] = {
                    'weights': weights,
                    'num_samples': sample.num_samples,
                    'metrics': {'loss': sample.loss, 'accuracy': sample.accuracy}
                }
                print(f"Received update from client {client_id} ({len(self.client_updates)}/{len(self.active_clients)})")
        if len(self.client_updates) > 0 and len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
            self.aggregate_models()
    
    def mark_client_converged(self, client_id):
        """Remove converged client from active federation; proceed with remaining clients."""
        if not stop_on_client_convergence():
            # Fixed-round mode: ignore client-local convergence removal/disconnect.
            print(f"Ignoring convergence signal from client {client_id} (STOP_ON_CLIENT_CONVERGENCE=false)")
            return
        if client_id in self.active_clients:
            self.active_clients.discard(client_id)
            self.client_updates.pop(client_id, None)
            self.client_metrics.pop(client_id, None)
            self.model_update_chunks.pop(client_id, None)
            self.model_update_metadata.pop(client_id, None)
            print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
            if not self.active_clients:
                self.converged = True
                print("All clients converged. Ending training.")
                self.convergence_time = time.time() - self.start_time if self.start_time else 0
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
            else:
                # Re-check: remaining active clients may have already sent updates/metrics
                if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                    self.aggregate_models()
                if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                    self.aggregate_metrics()
                    self.continue_training()
    
    def check_evaluation_metrics(self):
        """Check for evaluation metrics from clients"""
        samples = self.readers['metrics'].take()
        
        for sample in samples:
            if sample and hasattr(sample, 'client_id') and hasattr(sample, 'round'):
                print(f"Server received metrics sample: client {sample.client_id}, round {sample.round} (current: {self.current_round})")
                
                if sample.round == self.current_round:
                    client_id = sample.client_id
                    conv = getattr(sample, 'client_converged', 0.0) or 0.0
                    if stop_on_client_convergence() and float(conv) >= 1.0:
                        self.mark_client_converged(client_id)
                        continue
                    if client_id not in self.active_clients:
                        continue
                    if client_id not in self.client_metrics:
                        self.client_metrics[client_id] = {
                            'num_samples': sample.num_samples,
                            'metrics': {
                                'loss': sample.loss,
                                'accuracy': sample.accuracy
                            },
                            'battery_soc': float(getattr(sample, 'battery_soc', 1.0)),
                            'round_time_sec': float(getattr(sample, 'round_time_sec', 0.0)),
                        }
                        
                        print(f"Received metrics from client {client_id} "
                              f"({len(self.client_metrics)}/{len(self.active_clients)})")
                        
                        if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                            self.aggregate_metrics()
                            self.continue_training()
    
    def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")
        
        # Safety check: ensure we have at least one client update
        if len(self.client_updates) == 0:
            print("ERROR: aggregate_models called with 0 client updates. Skipping aggregation.")
            return

        # Quantization end-to-end: aggregate directly on compressed quantized tensors.
        if (
            self.quantization_handler is not None
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
            print(f"Sending global model to clients (dequantize→FedAvg→requantize on aggregate)...\n")

            serialized_weights = list(pickle.dumps(self.global_compressed))

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

            time.sleep(1)
            command = TrainingCommand(
                round=self.current_round,
                start_training=False,
                start_evaluation=True,
                training_complete=False
            )
            self.writers['command'].write(command)
            self.evaluation_phase = True

            self.wait_for_evaluation_metrics()
            if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                self.aggregate_metrics()
                self.continue_training()
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
        if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
            self.aggregate_metrics()
            self.continue_training()
    
    def wait_for_evaluation_metrics(self):
        """Actively wait for evaluation metrics from all active clients"""
        print(f"\nWaiting for evaluation metrics from {len(self.active_clients)} clients...")
        timeout = 60  # 60 seconds timeout
        start_time = time.time()
        
        while len(self.client_metrics) < len(self.active_clients) and len(self.active_clients) > 0:
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for metrics. Received {len(self.client_metrics)}/{len(self.active_clients)} active clients")
                break
            
            samples = self.readers['metrics'].take()
            for sample in samples:
                # Guard: DDS may return InvalidSample when a writer disconnects (e.g. client converged)
                if not sample or not hasattr(sample, 'round'):
                    continue
                if sample.round == self.current_round:
                    client_id = sample.client_id
                    conv = getattr(sample, 'client_converged', 0.0) or 0.0
                    if stop_on_client_convergence() and float(conv) >= 1.0:
                        self.mark_client_converged(client_id)
                        continue
                    if client_id not in self.active_clients:
                        continue
                    if client_id not in self.client_metrics:
                        print(f"Received evaluation metrics from client {client_id}")
                        metrics_dict = {
                            'loss': sample.loss,
                            'accuracy': sample.accuracy
                        }
                        self.client_metrics[client_id] = {
                            'metrics': metrics_dict,
                            'num_samples': sample.num_samples,
                            'battery_soc': float(getattr(sample, 'battery_soc', 1.0)),
                        }
                        print(f"Progress: {len(self.client_metrics)}/{len(self.active_clients)} clients")
            
            if len(self.client_metrics) < len(self.active_clients):
                time.sleep(0.1)  # Short sleep before next check
        
        if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
            print(f"✓ All evaluation metrics received!")
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics from all clients"""
        print(f"\nAggregating metrics from {len(self.client_metrics)} clients...")
        if getattr(self, 'round_start_time', None) is not None:
            self.ROUND_TIMES.append(time.time() - self.round_start_time)
        socs = [m.get('battery_soc', 1.0) for m in self.client_metrics.values()]
        self.BATTERY_CONSUMPTION.append(1.0 - (sum(socs) / len(socs) if socs else 1.0))
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
        
        # Stop only when no active clients remain or max rounds reached (no server-side convergence)
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
        
        # Check if more rounds needed
        if self.current_round < self.num_rounds:
            self.current_round += 1
            self.round_start_time = time.time()
            
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
        """Plot battery, round/convergence time, and loss/accuracy."""
        results_dir = get_experiment_results_dir("emotion", "dds")
        rounds = self.ROUNDS
        n = len(rounds)
        conv_time = self.convergence_time if self.convergence_time is not None else (time.time() - self.start_time if self.start_time else 0)
        bc = (getattr(self, 'BATTERY_CONSUMPTION', []) + [0.0] * max(0, n - len(getattr(self, 'BATTERY_CONSUMPTION', []))))[:n] or [0.0] * n
        rt = (getattr(self, 'ROUND_TIMES', []) + [0.0] * max(0, n - len(getattr(self, 'ROUND_TIMES', []))))[:n] or [0.0] * n
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        if bc: ax1.plot(rounds, [c * 100 for c in bc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round'); ax1.set_ylabel('Battery consumption (%)'); ax1.set_title('DDS: Battery consumption till end of FL'); ax1.grid(True, alpha=0.3)
        fig1.tight_layout(); fig1.savefig(results_dir / 'dds_battery_consumption.png', dpi=300, bbox_inches='tight'); plt.close(fig1)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if rt: ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2, label=f'Convergence: {conv_time:.1f} s')
        ax2.set_xlabel('Round'); ax2.set_ylabel('Time (s)'); ax2.set_title('DDS: Time per round and convergence'); ax2.legend(); ax2.grid(True, alpha=0.3)
        fig2.tight_layout(); fig2.savefig(results_dir / 'dds_round_and_convergence_time.png', dpi=300, bbox_inches='tight'); plt.close(fig2)
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        ax3a.plot(rounds, self.LOSS, 'b-', marker='o'); ax3a.set_xlabel('Round'); ax3a.set_ylabel('Loss'); ax3a.set_title('DDS: Loss over Rounds'); ax3a.grid(True)
        ax3b.plot(rounds, self.ACCURACY, 'g-', marker='o'); ax3b.set_xlabel('Round'); ax3b.set_ylabel('Accuracy'); ax3b.set_title('DDS: Accuracy over Rounds'); ax3b.grid(True)
        fig3.tight_layout(); fig3.savefig(results_dir / 'dds_training_metrics.png', dpi=300, bbox_inches='tight'); plt.close(fig3)
        print(f"Training metrics plot saved to {results_dir / 'dds_training_metrics.png'}")
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            plt.close()
        else:
            plt.show()

        print("\nPlot closed. Training complete.")
    
    def save_results(self):
        """Save training results to JSON"""
        results_dir = get_experiment_results_dir("emotion", "dds")
        
        results = {
            'rounds': self.ROUNDS,
            'loss': self.LOSS,
            'accuracy': self.ACCURACY,
            'round_times_seconds': getattr(self, 'ROUND_TIMES', []),
            'battery_consumption': getattr(self, 'BATTERY_CONSUMPTION', []),
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
    import tensorflow as tf
    try:
        tf.keras.backend.clear_session()
        print("Server configured to use CPU only (GPU disabled)")
    except Exception as e:
        print(f"GPU initialization warning: {e}")
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    server.run()
