import os
import sys
import glob
import json
import pickle
import base64
import time
import asyncio
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from typing import Dict, Optional
import tensorflow as tf
from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Add Compression_Technique to path for optional quantization support
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

try:
    from quantization_server import ServerQuantizationHandler, QuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Quantization module not available")
    QUANTIZATION_AVAILABLE = False

# Server Configuration
QUIC_HOST = os.getenv("QUIC_HOST", "localhost")
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
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


# ---------------- QUIC Protocol ----------------
class FederatedLearningServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = None
        self._stream_buffers = {}
        print(f"[DEBUG] New server protocol instance created: {id(self)}")

    def quic_event_received(self, event: QuicEvent):
        print(f"[DEBUG] quic_event_received called on protocol {id(self)}, event type: {type(event).__name__}")
        if isinstance(event, StreamDataReceived):
            print(f"[DEBUG] Server received data on stream {event.stream_id}, size={len(event.data)} bytes, end_stream={event.end_stream}")
            
            if event.stream_id not in self._stream_buffers:
                self._stream_buffers[event.stream_id] = b''

            self._stream_buffers[event.stream_id] += event.data

            # Send flow control updates to allow more data (critical for poor networks)
            self.transmit()

            # Process complete messages (delimited by newline)
            while b'\n' in self._stream_buffers[event.stream_id]:
                message_data, self._stream_buffers[event.stream_id] = self._stream_buffers[event.stream_id].split(
                    b'\n', 1)
                if message_data:
                    try:
                        data = message_data.decode('utf-8')
                        message = json.loads(data)
                        if self.server:
                            asyncio.create_task(self.server.handle_message(message, self))
                        else:
                            print(f"[DEBUG] Server not set in protocol, cannot handle message")
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding message: {e}")
                        print(f"Message data length: {len(message_data)}")
            
            # If stream ended and buffer has remaining data, try to process it
            if event.end_stream and self._stream_buffers[event.stream_id]:
                try:
                    data = self._stream_buffers[event.stream_id].decode('utf-8')
                    message = json.loads(data)
                    if self.server:
                        asyncio.create_task(self.server.handle_message(message, self))
                    self._stream_buffers[event.stream_id] = b''
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error decoding remaining buffer: {e}")


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
        self.registered_clients = {}
        self.active_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None

        # Metrics storage
        self.LOSS = []
        self.ACC = []
        self.TOP2 = []
        self.ROUNDS = []

        # Protocol reference
        self.protocol: Optional[FederatedLearningServerProtocol] = None

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
        """Serialize model weights for QUIC transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded

    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from QUIC"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights

    async def send_message(self, client_id, message):
        """Send message to client via QUIC stream"""
        if client_id in self.registered_clients:
            protocol = self.registered_clients[client_id]
            stream_id = protocol._quic.get_next_available_stream_id(is_unidirectional=False)
            data = (json.dumps(message) + '\n').encode('utf-8')
            # Set end_stream=True to ensure proper message delivery, especially for large messages
            protocol._quic.send_stream_data(stream_id, data, end_stream=True)
            protocol.transmit()
            
            msg_type = message.get('type')
            msg_size_mb = len(data) / (1024 * 1024)
            print(f"Sent message type '{msg_type}' to client {client_id} on stream {stream_id} ({len(data)} bytes = {msg_size_mb:.2f} MB)")
            
            # Multiple transmit calls for large messages (improved for poor networks)
            if len(data) > 1_000_000:  # > 1MB
                for _ in range(3):
                    await asyncio.sleep(0.5)
                    protocol.transmit()
            else:
                await asyncio.sleep(0.1)

    async def broadcast_message(self, message):
        """Broadcast message to all registered clients"""
        for client_id in self.registered_clients.keys():
            await self.send_message(client_id, message)

    async def handle_message(self, message, protocol):
        """Handle incoming messages from clients"""
        try:
            msg_type = message.get('type')
            print(f"[DEBUG] Server received message type: {msg_type}")

            if msg_type == 'register':
                await self.handle_client_registration(message, protocol)
            elif msg_type == 'model_update':
                await self.handle_client_update(message)
            elif msg_type == 'metrics':
                await self.handle_client_metrics(message)
        except Exception as e:
            print(f"Server error handling message: {e}")
            import traceback
            traceback.print_exc()

    async def handle_client_registration(self, message, protocol):
        """Handle client registration"""
        client_id = message['client_id']
        print(f"[DEBUG] Received registration from client {client_id}")
        self.registered_clients[client_id] = protocol
        self.active_clients.add(client_id)
        print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))
        print(f"[DEBUG] Registered clients: {sorted(self.registered_clients.keys())}")

        if self.training_started:
            self.active_clients.add(client_id)
            print(f"[LATE JOIN] Client {client_id} joining during round {self.current_round}")
            if len(self.registered_clients) > self.num_clients:
                self.update_client_count(len(self.registered_clients))
            if self.global_weights is not None:
                self.send_current_model_to_client(client_id)
            return
        
        if len(self.registered_clients) >= self.min_clients:
            print("\nAll clients registered. Distributing initial global model...\n")
            await asyncio.sleep(2)
            await self.distribute_initial_model()

    async def mark_client_converged(self, client_id):
        """Remove converged client from active federation."""
        if client_id in self.active_clients:
            self.active_clients.discard(client_id)
            self.registered_clients.pop(client_id, None)
            self.client_updates.pop(client_id, None)
            self.client_metrics.pop(client_id, None)
            print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
            if not self.active_clients:
                self.converged = True
                print("All clients converged. Ending training.")
                self.convergence_time = time.time() - self.start_time if self.start_time else 0
                await self.broadcast_message({'type': 'training_complete', 'message': 'Training completed'})
                await asyncio.sleep(2)
                self.save_results()
                self.plot_results()

    async def handle_client_update(self, message):
        """Handle model update from client"""
        client_id = message['client_id']
        round_num = message['round']
        m = message.get('metrics', {})
        if client_id not in self.active_clients:
            return
        if float(m.get('client_converged', 0.0)) >= 1.0:
            await self.mark_client_converged(client_id)
            return

        print(f"[DEBUG] Server received model_update - client_id={client_id}, round_num={round_num}, current_round={self.current_round}")

        if round_num == self.current_round:
            # Decompress or deserialize client weights
            if 'compressed_data' in message and self.quantization_handler is not None:
                weights = self.quantization_handler.decompress_client_update(message['client_id'], message['compressed_data'])
                print(f"Server: Received and decompressed update from client {message['client_id']}")
            else:
                weights = self.deserialize_weights(message['weights'])
            
            self.client_updates[client_id] = {
                'weights': weights,
                'num_samples': message['num_samples']
            }

            print(f"Received update from client {client_id} "
                  f"({len(self.client_updates)}/{len(self.active_clients)})")

            if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                await self.aggregate_models()
        else:
            print(f"[DEBUG] Ignoring update from client {client_id} - round mismatch (got {round_num}, expected {self.current_round})")

    async def handle_client_metrics(self, message):
        """Handle evaluation metrics from client (not used for server evaluation)"""
        client_id = message['client_id']
        print(f"Received metrics from client {client_id}")

    async def distribute_initial_model(self):
        """Distribute initial global model to all clients"""
        await self.broadcast_message({
            'type': 'training_config',
            'config': self.training_config
        })

        self.current_round = 1

        print(f"\n{'=' * 70}")
        print(f"Distributing Initial Global Model")
        print(f"{'=' * 70}\n")

        print(f"[DEBUG] Sending initial model - round=0, has_config=True")
        
        # Prepare global model (compress if quantization enabled)
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed initial global model - Ratio: {stats['compression_ratio']:.2f}x")
            weights_data = compressed_data
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        await self.broadcast_message({
            'type': 'global_model',
            'round': 0,
            weights_key: weights_data,
            'model_config': {
                "architecture": "CNN+BiLSTM+MHA",
                "input_shape": [256, 20],
                "num_classes": NUM_CLASSES
            }
        })

        print("Initial global model sent to all clients")
        # Give clients time to build and initialize model
        # Increased wait time for very poor network conditions (was 5s, now 30s)
        print("Waiting for clients to initialize models...")
        await asyncio.sleep(30)

        print(f"\n{'=' * 70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'=' * 70}\n")

        await self.broadcast_message({
            'type': 'start_training',
            'round': self.current_round
        })

    async def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")

        total_samples = sum(update['num_samples'] for update in self.client_updates.values())

        aggregated_weights = []
        first_client_weights = list(self.client_updates.values())[0]['weights']

        for layer_idx in range(len(first_client_weights)):
            layer_weights = np.zeros_like(first_client_weights[layer_idx])

            for client_id, update in self.client_updates.items():
                weight = update['num_samples'] / total_samples
                layer_weights += weight * update['weights'][layer_idx]

            aggregated_weights.append(layer_weights)

        self.global_weights = aggregated_weights

        # Evaluate on global test set
        await self.evaluate_global_model()

        # Send the aggregated model with the current round number
        # (the round that just completed)
        
        # Prepare global model (compress if quantization enabled)
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
            weights_data = compressed_data
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        # Define model_config for late-joiners
        model_config = {
            "architecture": "CNN+BiLSTM+MHA",
            "input_shape": [256, 20],
            "num_classes": NUM_CLASSES
        }
        
        await self.broadcast_message({
            'type': 'global_model',
            'round': self.current_round,
            weights_key: weights_data,
            'model_config': model_config  # Always include for late-joiners
        })

        print(f"Aggregated global model from round {self.current_round} sent to all clients\n")

        await self.continue_training()

    async def evaluate_global_model(self):
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

    async def continue_training(self):
        """Continue to next round or finish training"""
        self.client_updates.clear()

        if self.current_round >= self.num_rounds:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED!")
            print(f"Total rounds: {self.num_rounds}")
            print(f"Best accuracy: {max(self.ACC):.6f}")
            print("=" * 70 + "\n")

            await self.broadcast_message({
                'type': 'training_complete',
                'message': 'Training completed'
            })

            await asyncio.sleep(2)
            self.save_results()
            return

        # Increment round BEFORE sending start_training
        self.current_round += 1
        print(f"\n{'=' * 70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'=' * 70}\n")

        await asyncio.sleep(2)
        await self.broadcast_message({
            'type': 'start_training',
            'round': self.current_round
        })

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
        results_file = os.path.join(results_dir, "quic_training_results.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[RESULTS] Saved to {results_file}")
        print(f"[RESULTS] Best accuracy: {results['best_accuracy']:.6f} at round {results['best_round']}")


async def main():
    print(f"\n{'=' * 70}")
    print(f"Federated Learning Server - Mental State Recognition (QUIC)")
    print(f"{'=' * 70}")
    print(f"Configuration:")
    print(f"  Host: {QUIC_HOST}:{QUIC_PORT}")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Model: CNN+BiLSTM+MHA")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASS_ORDER)})")
    print(f"{'=' * 70}\n")
    print("Waiting for clients to connect...\n")

    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)

    # Configure QUIC
    configuration = QuicConfiguration(
        is_client=False,
        max_datagram_frame_size=65536,
        max_stream_data=20 * 1024 * 1024,  # 20MB per stream
        max_data=50 * 1024 * 1024,  # 50MB total
        idle_timeout=300.0,  # 5 minutes idle timeout
    )
    
    # Load certificates from certs directory
    # In Docker, certs are mounted at /app/certs/
    cert_dir = Path("/app/certs") if Path("/app/certs").exists() else Path(__file__).parent.parent.parent / "certs"
    cert_file = cert_dir / "server-cert.pem"
    key_file = cert_dir / "server-key.pem"
    
    if not cert_file.exists() or not key_file.exists():
        print("❌ Certificates not found. Please run generate_certs.py first.")
        print(f"   Expected location: {cert_dir}")
        import sys
        sys.exit(1)
    
    print(f"✓ Loading certificates from {cert_dir}")
    configuration.load_cert_chain(str(cert_file), str(key_file))

    # Create protocol factory
    def create_protocol(*args, **kwargs):
        protocol = FederatedLearningServerProtocol(*args, **kwargs)
        protocol.server = server
        server.protocol = protocol
        return protocol

    await serve(
        QUIC_HOST,
        QUIC_PORT,
        configuration=configuration,
        create_protocol=create_protocol,
    )

    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down...")
