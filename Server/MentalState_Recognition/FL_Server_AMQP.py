import os
import glob
import json
import pickle
import base64
import time
import numpy as np
import pandas as pd
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
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "16"))

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

        # Initialize global model and test data
        self.initialize_global_model()
        self.load_test_data()

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
            print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")

            if len(self.registered_clients) == self.num_clients:
                print("\nAll clients registered. Distributing initial global model...\n")
                time.sleep(2)
                self.distribute_initial_model()
        except Exception as e:
            print(f"Server error handling registration: {e}")

    def on_client_update(self, ch, method, properties, body):
        """Handle model update from client"""
        try:
            data = json.loads(body.decode())
            client_id = data['client_id']
            round_num = data['round']

            if round_num == self.current_round:
                self.client_updates[client_id] = {
                    'weights': self.deserialize_weights(data['weights']),
                    'num_samples': data['num_samples']
                }

                print(f"Received update from client {client_id} "
                      f"({len(self.client_updates)}/{self.num_clients})")

                if len(self.client_updates) == self.num_clients:
                    self.aggregate_models()
        except Exception as e:
            print(f"Server error handling client update: {e}")

    def on_client_metrics(self, ch, method, properties, body):
        """Handle evaluation metrics from client (not used for server evaluation)"""
        try:
            data = json.loads(body.decode())
            client_id = data['client_id']
            print(f"Received metrics from client {client_id}")
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
            "weights": self.serialize_weights(self.global_weights)
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

            self.channel.basic_publish(
                exchange=EXCHANGE_BROADCAST,
                routing_key='',
                body=json.dumps({
                    "message_type": "training_complete",
                    "message": "Training completed"
                }),
                properties=pika.BasicProperties(delivery_mode=2)
            )
            time.sleep(2)

            self.save_results()
            self.connection.close()
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
            "num_rounds": self.num_rounds
        }

        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "amqp_training_results.json")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n[RESULTS] Saved to {results_file}")
        print(f"[RESULTS] Best accuracy: {results['best_accuracy']:.6f} at round {results['best_round']}")

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
    server = FederatedLearningServer(NUM_CLIENTS, NUM_ROUNDS)
    server.run()
