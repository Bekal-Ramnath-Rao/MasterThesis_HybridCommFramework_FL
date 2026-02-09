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
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "16"))

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
class ModelUpdate(IdlStruct):
    client_id: int
    round: int
    weights: sequence[int]
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
    def __init__(self, num_clients, num_rounds):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None

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

    def setup_dds(self):
        """Initialize DDS participant, topics, readers, and writers"""
        print(f"Setting up DDS on domain {DDS_DOMAIN_ID}...")

        self.participant = DomainParticipant(DDS_DOMAIN_ID)

        # Infinite max_blocking_time for large model transfers (no timeout)
        reliable_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=3600)),  # 1 hour timeout
            Policy.History.KeepAll,
            Policy.Durability.TransientLocal
        )

        # Create topics
        topic_registration = Topic(self.participant, "ClientRegistration", ClientRegistration)
        topic_config = Topic(self.participant, "TrainingConfig", TrainingConfig)
        topic_command = Topic(self.participant, "TrainingCommand", TrainingCommand)
        topic_global_model = Topic(self.participant, "GlobalModel", GlobalModel)
        topic_model_update = Topic(self.participant, "ModelUpdate", ModelUpdate)
        topic_metrics = Topic(self.participant, "EvaluationMetrics", EvaluationMetrics)
        topic_status = Topic(self.participant, "ServerStatus", ServerStatus)

        # Create readers and writers
        self.readers['registration'] = DataReader(self.participant, topic_registration, qos=reliable_qos)
        self.readers['model_update'] = DataReader(self.participant, topic_model_update, qos=reliable_qos)
        self.readers['metrics'] = DataReader(self.participant, topic_metrics, qos=reliable_qos)

        self.writers['config'] = DataWriter(self.participant, topic_config, qos=reliable_qos)
        self.writers['command'] = DataWriter(self.participant, topic_command, qos=reliable_qos)
        self.writers['global_model'] = DataWriter(self.participant, topic_global_model, qos=reliable_qos)
        self.writers['status'] = DataWriter(self.participant, topic_status, qos=reliable_qos)

        print("DDS setup complete with RELIABLE QoS\n")
        time.sleep(2)

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

        config = TrainingConfig(
            batch_size=self.training_config['batch_size'],
            local_epochs=self.training_config['local_epochs']
        )
        self.writers['config'].write(config)

        try:
            while not self.training_complete:
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

        for sample in samples:
            if sample:
                client_id = sample.client_id
                print(f"[DEBUG] Received registration from client {client_id}")
                if client_id not in self.registered_clients:
                    self.registered_clients.add(client_id)
                    print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
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

        initial_model = GlobalModel(
            round=0,
            weights=self.serialize_weights(self.global_weights),
            model_config_json=json.dumps(model_config)
        )
        print(f"[DEBUG] Sending initial model - round=0, has_config=True, weights_size={len(self.serialize_weights(self.global_weights))}")
        self.writers['global_model'].write(initial_model)

        print("Initial global model sent to all clients")
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
        """Check for model updates from clients"""
        samples = self.readers['model_update'].take()

        for sample in samples:
            if sample and sample.round == self.current_round:
                client_id = sample.client_id

                if client_id not in self.client_updates:
                    # Decompress or deserialize client weights
                    if self.quantization_handler is not None:
                        try:
                            weights = self.quantization_handler.decompress_client_update(sample.client_id, bytes(sample.weights))
                            print(f"Server: Received and decompressed update from client {sample.client_id}")
                        except:
                            # Fallback to regular deserialization
                            weights = self.deserialize_weights(sample.weights)
                    else:
                        weights = self.deserialize_weights(sample.weights)
                    
                    self.client_updates[client_id] = {
                        'weights': weights,
                        'num_samples': sample.num_samples
                    }

                    print(f"Received update from client {client_id} "
                          f"({len(self.client_updates)}/{self.num_clients})")

                    if len(self.client_updates) == self.num_clients:
                        self.aggregate_models()

    def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")

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

        # Send global model to clients
        global_model = GlobalModel(
            round=self.current_round,
            weights=self.serialize_weights(self.global_weights)
        )
        self.writers['global_model'].write(global_model)
        print(f"Aggregated global model from round {self.current_round} sent to all clients\n")

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
    server = FederatedLearningServer(NUM_CLIENTS, NUM_ROUNDS)
    server.run()
