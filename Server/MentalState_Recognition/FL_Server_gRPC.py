import os
import glob
import json
import pickle
import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import sys
import grpc
from concurrent import futures
import threading
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# Server Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "0.0.0.0")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
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


# ---------------- Federated Learning Servicer ----------------
class FederatedLearningServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
    def __init__(self, num_clients, num_rounds):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        self.lock = threading.Lock()

        # Metrics storage
        self.LOSS = []
        self.ACC = []
        self.TOP2 = []
        self.ROUNDS = []

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

    def initialize_global_model(self):
        """Initialize the global EEG model"""
        print("\nInitializing global EEG model (CNN+BiLSTM+MHA)...")
        model = build_model()
        self.global_weights = model.get_weights()
        print(f"Model initialized with {len(self.global_weights)} weight layers")
        print(f"[DEBUG] Global weights initialized: {self.global_weights is not None}")

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
        """Serialize model weights for gRPC transmission"""
        serialized = pickle.dumps(weights)
        return serialized

    def deserialize_weights(self, serialized_weights):
        """Deserialize model weights received from gRPC"""
        weights = pickle.loads(serialized_weights)
        return weights

    def RegisterClient(self, request, context):
        """Handle client registration"""
        print(f"[DEBUG] RegisterClient called - Received registration request from client {request.client_id}")
        with self.lock:
            client_id = request.client_id
            print(f"[DEBUG] Acquired lock. Processing client {client_id}")
            self.registered_clients.add(client_id)
            print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
            print(f"[DEBUG] Currently registered clients: {sorted(self.registered_clients)}")

            if len(self.registered_clients) == self.num_clients and not self.training_started:
                print("\nAll clients registered. Distributing initial global model...\n")
                self.training_started = True
                self.current_round = 1

                print(f"\n{'=' * 70}")
                print(f"Distributing Initial Global Model")
                print(f"{'=' * 70}\n")
                print("Initial global model ready for clients")

                print(f"\n{'=' * 70}")
                print(f"Starting Round {self.current_round}/{self.num_rounds}")
                print(f"{'=' * 70}\n")

            return federated_learning_pb2.RegistrationResponse(
                success=True,
                message=f"Client {client_id} registered successfully"
            )

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

            print(f"[DEBUG] CheckTrainingStatus - Client {client_id}, client_round={client_round}, server_round={self.current_round}, training_started={self.training_started}")

            if self.training_complete:
                return federated_learning_pb2.TrainingStatus(
                    should_train=False,
                    round=self.current_round,
                    should_evaluate=False,
                    training_complete=True
                )

            # Check if client has already submitted update for current round
            client_has_submitted = client_id in self.client_updates and self.client_updates[client_id].get('round') == self.current_round

            # Tell client to train if:
            # 1. Training has started and client is behind the current round
            # 2. Client is at current round but hasn't submitted update yet
            if self.training_started and not self.evaluation_phase:
                if client_round < self.current_round or (client_round == self.current_round and not client_has_submitted):
                    print(f"[DEBUG] Telling client {client_id} to train for round {self.current_round}")
                    return federated_learning_pb2.TrainingStatus(
                        should_train=True,
                        round=self.current_round,
                        should_evaluate=False,
                        training_complete=False
                    )

            print(f"[DEBUG] Telling client {client_id} to wait (evaluation_phase={self.evaluation_phase}, has_submitted={client_has_submitted})")
            return federated_learning_pb2.TrainingStatus(
                should_train=False,
                round=self.current_round,
                should_evaluate=False,
                training_complete=False
            )

    def GetGlobalModel(self, request, context):
        """Send global model to client"""
        print(f"[DEBUG] GetGlobalModel called - client_id={request.client_id}, request.round={request.round}")
        with self.lock:
            print(f"[DEBUG] GetGlobalModel - training_started={self.training_started}, global_weights is None={self.global_weights is None}")
            
            if not self.training_started:
                print(f"[DEBUG] GetGlobalModel - Training not started yet, returning unavailable")
                return federated_learning_pb2.GlobalModel(
                    round=0,
                    weights=b'',
                    available=False,
                    model_config=""
                )

            if self.global_weights is not None:
                # Send current round model (not initial round 0)
                round_to_send = self.current_round
                model_config_json = ""
                
                # Only include model_config for initial model fetch
                if request.round == 0 and self.current_round == 1:
                    model_config = {
                        "architecture": "CNN+BiLSTM+MHA",
                        "input_shape": [256, 20],
                        "num_classes": NUM_CLASSES
                    }
                    model_config_json = json.dumps(model_config)
                    print(f"[DEBUG] GetGlobalModel - Sending initial model config")

                print(f"[DEBUG] GetGlobalModel - Sending round {round_to_send} to client {request.client_id} (request.round={request.round})")
                
                # Compress or serialize global weights
                if self.quantization_handler is not None:
                    compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
                    stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
                    print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
                    # Pickle compressed dict so gRPC bytes field can carry metadata
                    serialized_weights = pickle.dumps(compressed_data)
                else:
                    serialized_weights = self.serialize_weights(self.global_weights)
                
                return federated_learning_pb2.GlobalModel(
                    round=round_to_send,
                    weights=serialized_weights,
                    available=True,
                    model_config=model_config_json
                )
            else:
                print(f"[DEBUG] GetGlobalModel - global_weights is None!")
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

            if round_num == self.current_round:
                # Decompress or deserialize client weights (handle pickled compressed dicts)
                if request.weights:
                    if self.quantization_handler is not None:
                        try:
                            candidate = pickle.loads(request.weights)
                            if isinstance(candidate, dict) and 'compressed_data' in candidate:
                                weights = self.quantization_handler.decompress_client_update(request.client_id, candidate)
                                print(f"Server: Received and decompressed update from client {request.client_id}")
                            else:
                                weights = candidate
                        except Exception:
                            weights = self.deserialize_weights(request.weights)
                    else:
                        weights = self.deserialize_weights(request.weights)
                else:
                    weights = None
                
                self.client_updates[client_id] = {
                    'weights': weights,
                    'num_samples': request.num_samples,
                    'round': round_num
                }

                print(f"Received update from client {client_id} "
                      f"({len(self.client_updates)}/{self.num_clients})")

                if len(self.client_updates) == self.num_clients:
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
        """Receive evaluation metrics from client (not used for server evaluation)"""
        with self.lock:
            client_id = request.client_id
            print(f"Received metrics from client {client_id}")

            return federated_learning_pb2.MetricsResponse(
                success=True,
                message="Metrics received"
            )

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

        print(f"Aggregated global model from round {self.current_round}")
        print(f"Global model ready for clients\n")

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

        if self.current_round >= self.num_rounds:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED!")
            print(f"Total rounds: {self.num_rounds}")
            print(f"Best accuracy: {max(self.ACC):.6f}")
            print("=" * 70 + "\n")

            self.training_complete = True
            self.save_results()
            return

        # Start next round
        self.current_round += 1
        print(f"\n{'=' * 70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'=' * 70}\n")

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
        results_file = os.path.join(results_dir, "grpc_training_results.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[RESULTS] Saved to {results_file}")
        print(f"[RESULTS] Best accuracy: {results['best_accuracy']:.6f} at round {results['best_round']}")


def serve():
    print("\n" + "=" * 70)
    print("Federated Learning Server - Mental State Recognition (gRPC)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Host: {GRPC_HOST}:{GRPC_PORT}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Model: CNN+BiLSTM+MHA")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASS_ORDER)})")
    print("=" * 70 + "\n")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedLearningServicer(NUM_CLIENTS, NUM_ROUNDS)
    federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    server.add_insecure_port(f'{GRPC_HOST}:{GRPC_PORT}')
    server.start()

    print(f"gRPC Server started on {GRPC_HOST}:{GRPC_PORT}")
    print("Waiting for clients to connect...\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        server.stop(0)


if __name__ == "__main__":
    serve()
