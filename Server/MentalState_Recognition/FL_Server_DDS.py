import os
import sys
import glob
import json
import pickle
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "16"))

# Chunking configuration for large messages
CHUNK_SIZE = 64 * 1024  # 64KB chunks for better DDS performance in poor networks

# EEG Settings
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "Client", "MentalState_Recognition", "Dataset")
FS = 256
WIN_S = 1.0
WIN = int(FS * WIN_S)
STRIDE = 128
TEST_FRAC = 0.20
BATCH = 256

# Band-power features
BANDS = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
CHANS = ['TP9', 'AF7', 'AF8', 'TP10']
FEAT_COLS = [f'{b}_{c}' for b in BANDS for c in CHANS]

# Labels
CLASS_ORDER = ['alerted', 'concentrated', 'drowsy', 'neutral', 'relaxed']
LBL2ID = {c: i for i, c in enumerate(CLASS_ORDER)}
ID2LBL = {i: c for c, i in LBL2ID.items()}
NUM_CLASSES = len(CLASS_ORDER)

# GPU setup
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)


# ---------------- DDS Data Types ----------------
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
    model_config_json: str = ""


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


# ---------------- Data Loading Functions ----------------
def infer_label_from_name(path: str):
    n = os.path.basename(path).lower()
    if n.startswith("alerted"): return "alerted"
    if n.startswith("concentrated"): return "concentrated"
    if n.startswith("drowsy"): return "drowsy"
    if n.startswith("neutral"): return "neutral"
    if n.startswith("relaxed"): return "relaxed"
    return None


def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False, engine="c")
    except Exception:
        df = pd.read_csv(path, engine="python")
    miss = [c for c in FEAT_COLS if c not in df.columns]
    if miss:
        raise RuntimeError(f"{os.path.basename(path)} -> missing column: {miss[:5]}")
    df = df.dropna(subset=[FEAT_COLS[0]]).reset_index(drop=True)
    df[FEAT_COLS] = df[FEAT_COLS].astype('float32')
    return df


def csv_to_windows(path: str, file_id: int):
    label = infer_label_from_name(path)
    if label is None:
        return None
    df = read_csv_safe(path)
    x = df[FEAT_COLS].to_numpy(np.float32)
    # z-norm
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sd

    X, y, fidx = [], [], []
    for s in range(0, len(x) - WIN + 1, STRIDE):
        X.append(x[s:s + WIN])
        y.append(LBL2ID[label])
        fidx.append(file_id)
    if not X:
        return None
    return np.stack(X), np.array(y, np.int64), np.array(fidx, np.int64)


def find_all_csvs(folder):
    pats = ["Alerted-*.csv", "Concentrated-*.csv", "Drowsy-*.csv", "Neutral-*.csv", "Relaxed-*.csv"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(folder, p))
    files = sorted(files)
    if not files:
        raise SystemExit(f"CSV not found: {folder}")
    print(f"[{folder}] {len(files)} files found")
    return files


# ---------------- Model Building Functions ----------------
def se_block(x, r=8):
    ch = x.shape[-1]
    s = tf.keras.layers.GlobalAveragePooling1D()(x)
    s = tf.keras.layers.Dense(max(ch // r, 8), activation='relu')(s)
    s = tf.keras.layers.Dense(ch, activation='sigmoid', dtype='float32')(s)
    s = tf.keras.layers.Reshape((1, ch))(s)
    return tf.keras.layers.Multiply()([x, s])


def conv_bn_relu(x, f, k, d=1):
    x = tf.keras.layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x


def res_block(x, f, k, d=1):
    sc = x
    y = conv_bn_relu(x, f, k, d)
    y = tf.keras.layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    if sc.shape[-1] != f:
        sc = tf.keras.layers.Conv1D(f, 1, padding="same", use_bias=False)(sc)
        sc = tf.keras.layers.BatchNormalization()(sc)
    y = tf.keras.layers.Add()([y, sc])
    y = tf.keras.layers.ReLU()(y)
    y = se_block(y)
    return y


def build_model():
    inp = tf.keras.Input(shape=(256, 20))

    x = conv_bn_relu(inp, 64, 7, d=1)
    x = res_block(x, 64, 7, d=1)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    for d in [1, 2, 4]:
        x = res_block(x, 128, 5, d=d)

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
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inp, out)

    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3, first_decay_steps=4, t_mul=2.0, m_mul=0.8, alpha=1e-5
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4, global_clipnorm=1.0)

    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")])
    return model


# ---------------- Federated Learning Server ----------------
class FederatedLearningServer:
    def __init__(self, min_clients, num_rounds, max_clients=100):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.num_clients = min_clients  # Start with minimum, will update as clients join
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        
        # Chunk reassembly buffers
        self.model_update_chunks = {}  # {client_id: {chunk_id: payload}}
        self.model_update_metadata = {}  # {client_id: {total_chunks, num_samples, loss, accuracy}}

        # Metrics storage
        self.LOSS = []
        self.ACC = []
        self.TOP2 = []
        self.ROUNDS = []

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

        # Initialize global model and test data
        self.initialize_global_model()
        self.load_test_data()

        # Training configuration
        self.training_config = {
            "batch_size": BATCH,
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
        """Initialize the global EEG model"""
        print("\nInitializing global EEG model (CNN+BiLSTM+MHA)...")
        model = build_model()
        self.global_weights = model.get_weights()
        print(f"Model initialized with {len(self.global_weights)} weight layers")

    def load_test_data(self):
        """Load and prepare test data for evaluation"""
        print(f"\nLoading test data from {DATA_DIR}...")
        all_files = find_all_csvs(DATA_DIR)

        X_list, y_list = [], []
        for fid, f in enumerate(all_files):
            out = csv_to_windows(f, fid)
            if out is None:
                continue
            Xi, yi, fi = out
            X_list.append(Xi)
            y_list.append(yi)

        X_all = np.concatenate(X_list, axis=0).astype("float32")
        y_all = np.concatenate(y_list, axis=0).astype("int64")

        # Split into train/test
        _, self.X_test, _, self.y_test = train_test_split(
            X_all, y_all, test_size=TEST_FRAC, random_state=SEED, stratify=y_all
        )

        print(f"Test set: {len(self.y_test)} samples, shape: {self.X_test.shape}")
        print(f"Test class distribution: {dict(Counter(self.y_test))}")

    def serialize_weights(self, weights):
        """Serialize model weights for DDS transmission"""
        serialized = pickle.dumps(weights)
        return list(serialized)

    def deserialize_weights(self, serialized_weights):
        """Deserialize model weights received from DDS"""
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

        print("DDS setup complete with optimized QoS (KeepLast 10, 1s timeout)\n")
        time.sleep(0.5)

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
        print("=" * 70)
        print("Federated Learning Server - Mental State Recognition (DDS)")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  DDS Domain ID: {DDS_DOMAIN_ID}")
        print(f"  Clients: {self.num_clients}")
        print(f"  Rounds: {self.num_rounds}")
        print(f"  Model: CNN+BiLSTM+MHA")
        print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASS_ORDER)})")
        print("=" * 70 + "\n")
        print("Waiting for clients to register...\n")

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
                
                self.publish_status()
                self.check_registrations()
                self.check_model_updates()
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
                print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))
        
        print(f"[DEBUG] Registered clients: {sorted(self.registered_clients)}")
        
        if len(self.registered_clients) == self.num_clients and not self.training_started:
            print("\nAll clients registered. Distributing initial global model...\n")
            self.distribute_initial_model()

    def distribute_initial_model(self):
        """Distribute initial global model to all clients"""
        self.training_started = True
        self.current_round = 1

        print(f"\n{'=' * 70}")
        print(f"Distributing Initial Global Model")
        print(f"{'=' * 70}\n")

        model_config = {
            "architecture": "CNN+BiLSTM+MHA",
            "input_shape": [256, 20],
            "num_classes": NUM_CLASSES
        }

        serialized_weights = self.serialize_weights(self.global_weights)
        print("Publishing initial model to clients in chunks...")
        self.send_global_model_chunked(0, serialized_weights, json.dumps(model_config))
        
        print("Initial global model (architecture + weights) sent to all clients in chunks")
        # Give clients time to build and initialize model
        print("Waiting for clients to initialize models...")
        time.sleep(5)

        print(f"\n{'=' * 70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'=' * 70}\n")

        command = TrainingCommand(
            round=self.current_round,
            start_training=True,
            start_evaluation=False,
            training_complete=False
        )
        self.writers['command'].write(command)

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
                        'accuracy': sample.accuracy
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
                            'num_samples': metadata['num_samples']
                        }
                        
                        # Clear chunk buffers for this client
                        del self.model_update_chunks[client_id]
                        del self.model_update_metadata[client_id]
                        
                        print(f"Successfully reassembled update from client {client_id} "
                              f"({len(self.client_updates)}/{len(self.registered_clients)})")
        
        # If all clients sent updates, aggregate (ensure we have at least one client)
        if len(self.client_updates) > 0 and len(self.client_updates) >= len(self.registered_clients):
            self.aggregate_models()

    def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")
        
        # Safety check: ensure we have at least one client update
        if len(self.client_updates) == 0:
            print("ERROR: aggregate_models called with 0 client updates. Skipping aggregation.")
            return

        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in self.client_updates.values())

        # Initialize aggregated weights
        aggregated_weights = []
        first_client_weights = list(self.client_updates.values())[0]['weights']

        # FedAvg
        for layer_idx in range(len(first_client_weights)):
            layer_weights = np.zeros_like(first_client_weights[layer_idx])

            for client_id, update in self.client_updates.items():
                weight = update['num_samples'] / total_samples
                layer_weights += weight * update['weights'][layer_idx]

            aggregated_weights.append(layer_weights)

        self.global_weights = aggregated_weights

        # Evaluate on global test set
        self.evaluate_global_model()

        # Send global model to clients in chunks (always include model_config for late-joiners)
        model_config = {
            "architecture": "CNN+BiLSTM+MHA",
            "input_shape": [256, 20],
            "num_classes": NUM_CLASSES
        }
        serialized_weights = self.serialize_weights(self.global_weights)
        self.send_global_model_chunked(self.current_round, serialized_weights, json.dumps(model_config))
        print(f"Aggregated global model from round {self.current_round} sent to all clients in chunks\n")

        # Continue to next round
        self.continue_training()

    def evaluate_global_model(self):
        """Evaluate global model on test set"""
        print(f"\nEvaluating global model on test set...")

        # Create model and set weights
        model = build_model()
        model.set_weights(self.global_weights)

        # Prepare test data
        yte_oh = tf.one_hot(self.y_test, NUM_CLASSES, dtype=tf.float32)
        ds_te = tf.data.Dataset.from_tensor_slices((self.X_test, yte_oh))
        ds_te = ds_te.batch(BATCH).prefetch(tf.data.AUTOTUNE)

        # Evaluate
        loss, acc, top2 = model.evaluate(ds_te, verbose=0)

        # Store metrics
        self.LOSS.append(float(loss))
        self.ACC.append(float(acc))
        self.TOP2.append(float(top2))
        self.ROUNDS.append(self.current_round)

        print(f"\n{'=' * 70}")
        print(f"Round {self.current_round} - Global Test Metrics:")
        print(f"  Loss: {loss:.6f}")
        print(f"  Accuracy: {acc:.6f}")
        print(f"  Top-2 Accuracy: {top2:.6f}")
        print(f"{'=' * 70}\n")

    def continue_training(self):
        """Continue to next round or finish training"""
        self.client_updates.clear()
        self.evaluation_phase = False

        if self.current_round >= self.num_rounds:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED!")
            print(f"Total rounds: {self.num_rounds}")
            print(f"Best accuracy: {max(self.ACC):.6f}")
            print("=" * 70 + "\n")

            command = TrainingCommand(
                round=self.current_round,
                start_training=False,
                start_evaluation=False,
                training_complete=True
            )
            self.writers['command'].write(command)

            self.training_complete = True
            self.save_results()
            return

        # Start next round
        self.current_round += 1
        print(f"\n{'=' * 70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'=' * 70}\n")

        time.sleep(2)

        command = TrainingCommand(
            round=self.current_round,
            start_training=True,
            start_evaluation=False,
            training_complete=False
        )
        self.writers['command'].write(command)

    def save_results(self):
        """Save training results to JSON file"""
        results = {
            "rounds": self.ROUNDS,
            "loss": self.LOSS,
            "accuracy": self.ACC,
            "top2_accuracy": self.TOP2,
            "best_accuracy": max(self.ACC) if self.ACC else 0,
            "best_round": self.ROUNDS[np.argmax(self.ACC)] if self.ACC else 0,
            "num_clients": self.num_clients,
            "num_rounds": self.num_rounds
        }

        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "dds_training_results.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[RESULTS] Saved to {results_file}")
        print(f"[RESULTS] Best accuracy: {results['best_accuracy']:.6f} at round {results['best_round']}")

    def cleanup(self):
        """Clean up DDS resources"""
        print("Cleaning up DDS resources...")
        # DomainParticipant in CycloneDDS is automatically cleaned up
        # Just set references to None to allow garbage collection
        self.participant = None
        print("DDS cleanup complete")


if __name__ == "__main__":
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    server.run()
