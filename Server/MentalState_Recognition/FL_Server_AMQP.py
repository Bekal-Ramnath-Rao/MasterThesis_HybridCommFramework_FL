import os
import glob
import json
import pickle
import base64
import time
import sys
import numpy as np
import pandas as pd
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

from collections import Counter
from sklearn.model_selection import train_test_split
import pika
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Server Configuration
AMQP_HOST = os.getenv("AMQP_HOST", "localhost")
AMQP_PORT = int(os.getenv("AMQP_PORT", "5672"))
AMQP_USER = os.getenv("AMQP_USER", "guest")
AMQP_PASSWORD = os.getenv("AMQP_PASSWORD", "guest")
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "16"))

# Convergence Settings
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))

# AMQP Exchanges and Queues
EXCHANGE_BROADCAST = "fl_broadcast"
EXCHANGE_CLIENT_UPDATES = "fl_client_updates"
QUEUE_CLIENT_REGISTER = "fl.client.register"
QUEUE_CLIENT_UPDATE = "fl.client.update"
QUEUE_CLIENT_METRICS = "fl.client.metrics"

# EEG Settings
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

FS = 256
WIN_S = 1.0
WIN = int(FS * WIN_S)
STRIDE = 128
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


# ---------------- Federated Learning Server ----------------
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

        # Metrics storage (from client reports)
        self.LOSS = []
        self.ACC = []
        self.TOP2 = []
        self.ROUNDS = []
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.start_time = None
        self.convergence_time = None
        self.converged = False
        self.training_started = False
        self.training_started = False
        
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
        self.training_config = {
            "batch_size": BATCH,
            "local_epochs": 5
        }

        # AMQP connection
        self.connection = None
        self.channel = None

    def initialize_global_model(self):
        """Initialize the global EEG model"""
        print("\nInitializing global EEG model (CNN+BiLSTM+MHA)...")
        model = build_model()
        self.global_weights = model.get_weights()
        print(f"Model initialized with {len(self.global_weights)} weight layers")

    def serialize_weights(self, weights):
        """Serialize model weights for AMQP transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded

    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from AMQP"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights

    def connect(self):
        """Connect to RabbitMQ broker"""
        max_retries = 5
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to RabbitMQ at {AMQP_HOST}:{AMQP_PORT}...")
                credentials = pika.PlainCredentials(AMQP_USER, AMQP_PASSWORD)
                parameters = pika.ConnectionParameters(
                    host=AMQP_HOST,
                    port=AMQP_PORT,
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()

                # Declare exchanges
                self.channel.exchange_declare(exchange=EXCHANGE_BROADCAST, exchange_type='fanout', durable=True)
                self.channel.exchange_declare(exchange=EXCHANGE_CLIENT_UPDATES, exchange_type='direct', durable=True)

                # Declare queues
                self.channel.queue_declare(queue=QUEUE_CLIENT_REGISTER, durable=True)
                self.channel.queue_declare(queue=QUEUE_CLIENT_UPDATE, durable=True)
                self.channel.queue_declare(queue=QUEUE_CLIENT_METRICS, durable=True)

                # Bind queues
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_REGISTER,
                                        routing_key='client.register')
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_UPDATE,
                                        routing_key='client.update')
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_METRICS,
                                        routing_key='client.metrics')

                # Set up consumers
                self.channel.basic_consume(queue=QUEUE_CLIENT_REGISTER, on_message_callback=self.on_client_register,
                                           auto_ack=True)
                self.channel.basic_consume(queue=QUEUE_CLIENT_UPDATE, on_message_callback=self.on_client_update,
                                           auto_ack=True)
                self.channel.basic_consume(queue=QUEUE_CLIENT_METRICS, on_message_callback=self.on_client_metrics,
                                           auto_ack=True)

                print(f"Server connected to RabbitMQ broker\n")
                return True

            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to RabbitMQ broker after {max_retries} attempts.")
                    raise

    def on_client_register(self, ch, method, properties, body):
        """Handle client registration"""
        try:
            data = json.loads(body.decode())
            client_id = data['client_id']
            self.registered_clients.add(client_id)
            self.active_clients.add(client_id)
            print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
            
            if len(self.registered_clients) > self.num_clients:
                self.update_client_count(len(self.registered_clients))
            
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
                time.sleep(2)
                self.distribute_initial_model()
                self.start_time = time.time()
                self.training_started = True
        except Exception as e:
            print(f"Server error handling registration: {e}")

    def mark_client_converged(self, client_id):
        """Remove converged client from active federation."""
        if client_id in self.active_clients:
            self.active_clients.discard(client_id)
            self.client_updates.pop(client_id, None)
            self.client_metrics.pop(client_id, None)
            print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
            if not self.active_clients:
                self.converged = True
                print("All clients converged. Ending training.")
                self.convergence_time = time.time() - self.start_time if self.start_time else 0
                self.finish_training()

    def on_client_update(self, ch, method, properties, body):
        """Handle model update from client"""
        try:
            data = json.loads(body.decode())
            client_id = data['client_id']
            round_num = data['round']
            m = data.get('metrics', data) if isinstance(data.get('metrics'), dict) else data

            if client_id not in self.active_clients:
                return
            if float(m.get('client_converged', 0.0)) >= 1.0:
                self.mark_client_converged(client_id)
                return
            if round_num == self.current_round:
                # Check if update is compressed
                if 'compressed_data' in data and self.quantization_handler is not None:
                    compressed_update = data['compressed_data']
                    # If client sent serialized base64 string, decode and unpickle
                    if isinstance(compressed_update, str):
                        try:
                            compressed_update = pickle.loads(base64.b64decode(compressed_update.encode('utf-8')))
                        except Exception as e:
                            print(f"Server error decoding compressed_data from client {client_id}: {e}")
                    weights = self.quantization_handler.decompress_client_update(
                        client_id, 
                        compressed_update
                    )
                    print(f"Received and decompressed update from client {client_id}")
                else:
                    weights = self.deserialize_weights(data['weights'])
                
                self.client_updates[client_id] = {
                    'weights': weights,
                    'num_samples': data['num_samples'],
                    'metrics': data.get('metrics', {})
                }
                self.client_metrics[client_id] = data.get('metrics', {'loss': 0, 'accuracy': 0, 'top2_accuracy': 0})

                print(f"Received update from client {client_id} "
                      f"({len(self.client_updates)}/{len(self.active_clients)})")

                if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                    self.aggregate_models()
        except Exception as e:
            print(f"Server error handling client update: {e}")

    def on_client_metrics(self, ch, method, properties, body):
        """Handle evaluation metrics from client"""
        try:
            data = json.loads(body.decode())
            client_id = data['client_id']
            
            # Store client metrics
            self.client_metrics[client_id] = {
                'loss': data.get('loss', 0),
                'accuracy': data.get('accuracy', 0),
                'top2_accuracy': data.get('top2_accuracy', 0)
            }
            
            print(f"Received metrics from client {client_id}: "
                  f"Loss={data.get('loss', 0):.4f}, "
                  f"Acc={data.get('accuracy', 0):.4f}, "
                  f"Top2={data.get('top2_accuracy', 0):.4f}")
        except Exception as e:
            print(f"Server error handling client metrics: {e}")

    def distribute_initial_model(self):
        """Distribute initial global model to all clients"""
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "training_config",
                "config": self.training_config
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        self.current_round = 1
        self.start_time = time.time()  # Start timing

        print(f"\n{'=' * 70}")
        print(f"Distributing Initial Global Model")
        print(f"{'=' * 70}\n")

        initial_model_message = {
            "message_type": "global_model",
            "round": 0,
            "weights": self.serialize_weights(self.global_weights),
            "model_config": {
                "input_shape": [WIN, len(FEAT_COLS)],
                "num_classes": NUM_CLASSES
            }
        }

        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps(initial_model_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        print("Initial global model sent to all clients")
        # Give clients time to build and initialize model
        print("Waiting for clients to initialize models...")
        time.sleep(5)

        print(f"\n{'=' * 70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'=' * 70}\n")

        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "start_training",
                "round": self.current_round
            }),
            properties=pika.BasicProperties(delivery_mode=2)
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

        # Send global model to all clients
        global_model_message = {
            "message_type": "global_model",
            "round": self.current_round,
            "weights": self.serialize_weights(self.global_weights),
                "model_config": self.model_config  # Always include for late-joiners
        }

        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps(global_model_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )

        print(f"Aggregated global model from round {self.current_round} sent to all clients\n")

        # Continue to next round
        self.continue_training()

    def evaluate_global_model(self):
        """Aggregate and display client-reported metrics"""
        if not self.client_metrics:
            print("No client metrics available yet")
            return
        
        # Average client metrics (use 0 for missing top2_accuracy)
        avg_loss = np.mean([m.get('loss', 0) for m in self.client_metrics.values()])
        avg_acc = np.mean([m.get('accuracy', 0) for m in self.client_metrics.values()])
        avg_top2 = np.mean([m.get('top2_accuracy', 0) for m in self.client_metrics.values()])
        
        # Store aggregated metrics
        self.LOSS.append(float(avg_loss))
        self.ACC.append(float(avg_acc))
        self.TOP2.append(float(avg_top2))
        self.ROUNDS.append(self.current_round)
        
        print(f"\n{'=' * 70}")
        print(f"Round {self.current_round} - Aggregated Client Metrics:")
        print(f"  Avg Loss: {avg_loss:.6f}")
        print(f"  Avg Accuracy: {avg_acc:.6f}")
        print(f"  Avg Top-2 Accuracy: {avg_top2:.6f}")
        print(f"{'=' * 70}\n")
    
    def check_convergence(self, current_loss):
        """Check if training has converged"""
        if self.current_round < MIN_ROUNDS:
            return False
        
        improvement = self.best_loss - current_loss
        
        if improvement > CONVERGENCE_THRESHOLD:
            self.best_loss = current_loss
            self.rounds_without_improvement = 0
            print(f"Improvement detected: {improvement:.6f}")
        else:
            self.rounds_without_improvement += 1
            print(f"No significant improvement for {self.rounds_without_improvement} rounds")
        
        if self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print(f"\n{'=' * 70}")
            print("CONVERGENCE ACHIEVED!")
            print(f"Training stopped early at round {self.current_round}/{self.num_rounds}")
            print(f"No improvement for {CONVERGENCE_PATIENCE} consecutive rounds")
            print(f"Best loss: {self.best_loss:.6f}")
            print(f"Time to Convergence: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print(f"{'=' * 70}\n")
            self.converged = True
            self.finish_training()
            return True
        
        return False
    
    def finish_training(self):
        """Complete training and cleanup"""
        print("\nSending training completion signal to all clients...")
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "training_complete",
                "message": "Training completed"
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print("Training completion signal sent successfully")
        time.sleep(2)
        
        self.save_results()
        self.plot_results()

    def continue_training(self):
        """Continue to next round or finish training"""
        # Evaluate client metrics before continuing
        self.evaluate_global_model()
        
        self.client_updates.clear()
        self.client_metrics.clear()

        # Stop only when no active clients or max rounds (no server-side convergence)
        if len(self.active_clients) == 0:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            self.converged = True
            print("\n" + "=" * 70)
            print("All clients converged locally. Training complete.")
            print("=" * 70 + "\n")
            self.finish_training()
            return

        if self.current_round >= self.num_rounds:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "=" * 70)
            print("TRAINING COMPLETED!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Best accuracy: {max(self.ACC):.6f}")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("=" * 70 + "\n")

            self.finish_training()
            return

        # Start next round
        self.current_round += 1
        print(f"\n{'=' * 70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'=' * 70}\n")

        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "start_training",
                "round": self.current_round
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )

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
            "num_rounds": self.num_rounds,
            "convergence_time_seconds": self.convergence_time,
            "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
            "total_rounds": len(self.ROUNDS),
            "converged": self.converged
        }

        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        network_scenario = os.getenv("NETWORK_SCENARIO", "default")
        results_file = os.path.join(results_dir, f"amqp_{network_scenario}_training_results.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {results_file}")
        print(f"[RESULTS] Best accuracy: {results['best_accuracy']:.6f} at round {results['best_round']}")
    
    def plot_results(self):
        """Plot and save training metrics"""
        if not self.ROUNDS:
            print("No training data to plot")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.ROUNDS, self.LOSS, marker='o', linewidth=2, markersize=8, color='red')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Loss over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(self.ROUNDS, self.ACC, marker='s', linewidth=2, markersize=8, color='blue')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Top-2 Accuracy plot
        plt.subplot(1, 3, 3)
        plt.plot(self.ROUNDS, self.TOP2, marker='^', linewidth=2, markersize=8, color='green')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Top-2 Accuracy', fontsize=12)
        plt.title('Top-2 Accuracy over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to results folder
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'amqp_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {results_dir / 'amqp_training_metrics.png'}")
        plt.show(block=False)
        
        # Disconnect and exit
        print("\nTraining complete. Disconnecting...")
        time.sleep(2)
        self.connection.close()
        print("Server disconnected successfully.")
        sys.exit(0)

    def run(self):
        """Run the federated learning server"""
        print("\n" + "=" * 70)
        print("Federated Learning Server - Mental State Recognition (AMQP)")
        print("=" * 70)
        print(f"Configuration:")
        print(f"  Broker: {AMQP_HOST}:{AMQP_PORT}")
        print(f"  Clients: {self.num_clients}")
        print(f"  Rounds: {self.num_rounds}")
        print(f"  Model: CNN+BiLSTM+MHA")
        print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASS_ORDER)})")
        print("=" * 70 + "\n")

        self.connect()

        try:
            print("Waiting for clients to register...\n")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("\nServer stopped by user")
            if self.connection:
                self.connection.close()


if __name__ == "__main__":
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    server.run()
