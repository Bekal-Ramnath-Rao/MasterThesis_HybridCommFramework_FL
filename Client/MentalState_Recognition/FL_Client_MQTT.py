"""
Federated Learning Client for EEG Mental State Recognition using MQTT
Supports CNN+BiLSTM+MHA architecture with non-IID data partitioning
"""

import io
import os
import sys
import json
import pickle
import base64
import time
import random
import logging
import numpy as np
_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
if _xla_flags:
    sanitized_flags = [f for f in _xla_flags.split() if f != "--xla_gpu_enable_command_buffer="]
    if sanitized_flags:
        os.environ["XLA_FLAGS"] = " ".join(sanitized_flags)
    else:
        os.environ.pop("XLA_FLAGS", None)
import tensorflow as tf
import paho.mqtt.client as mqtt

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_utilities_path = os.path.join(_project_root, 'scripts', 'utilities')
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)
from client_fl_metrics_log import append_client_fl_metrics_record, use_case_from_env

try:
    from pruning_client import ModelPruning, PruningConfig
    PRUNING_AVAILABLE = True
except Exception:
    ModelPruning = None
    PruningConfig = None
    PRUNING_AVAILABLE = False

from collections import Counter

# Import data partitioner
from data_partitioner import get_client_data, NUM_CLASSES, ID2LBL, LBL2ID

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# GPU Configuration - Must be done BEFORE TensorFlow import
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

logging.getLogger("tensorflow").setLevel(logging.ERROR)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set memory limit to prevent OOM with large CNN+BiLSTM+MHA model
        # Allow ~7GB per GPU (conservative for 10GB cards with overhead)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7168)]
        )
        print(f"[GPU] Configured with memory growth and 7GB limit")
    except RuntimeError as e:
        print(f"[GPU] Configuration error: {e}")

# MQTT Configuration
# Auto-detect environment: Docker (/app exists) or local
MQTT_BROKER = os.getenv("MQTT_BROKER", 'mqtt-broker' if os.path.exists('/app') else 'localhost')
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "16"))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
STOP_ON_CLIENT_CONVERGENCE = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").lower() in ("1", "true", "yes")

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"
TOPIC_TRAINING_COMPLETE = "fl/training_complete"

# Training Configuration
AUTOTUNE = tf.data.AUTOTUNE
SMOOTH_EPS = 0.05


class FederatedLearningClient:
    def __init__(self, client_id, num_clients):
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

        # Initialize pruning compression (default: disabled unless explicitly enabled)
        up_env = os.getenv("USE_PRUNING", "false")
        use_pruning = up_env.lower() in ("true", "1", "yes", "y")
        if use_pruning and PRUNING_AVAILABLE and ModelPruning is not None:
            self.pruner = ModelPruning(PruningConfig())
            print(f"Client {self.client_id}: Pruning enabled")
        else:
            self.pruner = None
            if use_pruning and not PRUNING_AVAILABLE:
                print(f"Client {self.client_id}: Pruning requested but pruning module not available")
            else:
                print(f"Client {self.client_id}: Pruning disabled")
        self.x_train = None
        self.y_train = None
        self.current_round = 0
        self.training_config = {
            "batch_size": 16,
            "local_epochs": 5,
            "val_split": 0.10
        }
        self.class_weights = None
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.has_converged = False
        self.last_global_round = None
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id=f"fl_eeg_client_{client_id}")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Load and partition data
        self.prepare_data()
    
    def prepare_data(self):
        """Load and partition EEG data for this client using data_partitioner"""
        print(f"\n[Client {self.client_id}] Preparing data...")
        
        # Get data directory - detect environment
        if os.path.exists('/app'):
            data_dir = '/app/Client/MentalState_Recognition/Dataset'
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            data_dir = os.path.join(project_root, 'Client', 'MentalState_Recognition', 'Dataset')
        print(f"Dataset path: {data_dir}")
        
        # Use data partitioner to get non-IID data
        self.x_train, self.y_train = get_client_data(
            self.client_id, 
            self.num_clients, 
            data_dir
        )
        
        # Compute class weights for this client's data
        self.class_weights = self.compute_class_weights(self.y_train)
        
        print(f"[Client {self.client_id}] Data preparation complete!")
        print(f"  Training samples: {len(self.y_train)}")
        print(f"  Input shape: {self.x_train.shape}")
        print(f"  Class weights: {self.class_weights}")
    
    def compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        cc = Counter(y.tolist())
        total = max(1, sum(cc.values()))
        K = NUM_CLASSES
        return {cls: total / (K * cnt) for cls, cnt in cc.items()}
    
    def make_dataset(self, X, y, batch_size, training=True):
        """Create tf.data.Dataset with augmentations"""
        Xtf = tf.convert_to_tensor(X.astype('float32'))
        ytf = tf.convert_to_tensor(y.astype('int64'))
        
        # Sample weights
        sw = np.array([self.class_weights.get(int(k), 1.0) for k in y], dtype=np.float32)
        sww = tf.convert_to_tensor(sw, dtype=tf.float32)
        
        ds = tf.data.Dataset.from_tensor_slices((Xtf, ytf, sww))
        
        if training:
            ds = ds.shuffle(len(y), seed=42, reshuffle_each_iteration=True)
        
        def _augment(x):
            """Apply data augmentation"""
            # Time shift
            shift = tf.random.uniform([], -16, 17, dtype=tf.int32)
            x = tf.roll(x, shift=shift, axis=0)
            
            # Channel dropout
            if tf.random.uniform(()) < 0.30:
                C = tf.shape(x)[1]
                drop_n = tf.minimum(tf.random.uniform([], 1, 5, dtype=tf.int32), C)
                idx = tf.random.shuffle(tf.range(C))[:drop_n]
                mask = tf.ones([C], x.dtype)
                mask = tf.tensor_scatter_nd_update(
                    mask, tf.reshape(idx, [-1, 1]),
                    tf.zeros([drop_n], x.dtype)
                )
                x = x * mask[tf.newaxis, :]
            
            # Time masking
            if tf.random.uniform(()) < 0.30:
                T = tf.shape(x)[0]
                C = tf.shape(x)[1]
                L = tf.random.uniform([], 8, 33, dtype=tf.int32)
                s = tf.random.uniform([], 0, T - L, dtype=tf.int32)
                mask = tf.concat([
                    tf.ones([s, C], x.dtype),
                    tf.zeros([L, C], x.dtype),
                    tf.ones([T - s - L, C], x.dtype)
                ], axis=0)
                x = x * mask
            
            # Gaussian noise
            x = x + tf.random.normal(tf.shape(x), stddev=0.02, dtype=x.dtype)
            return x
        
        def _map(x, y, w):
            x = tf.cast(x, tf.float32)
            if training:
                x = _augment(x)
            y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES, dtype=x.dtype)
            if training and SMOOTH_EPS > 0:
                y = (1.0 - SMOOTH_EPS) * y + SMOOTH_EPS / NUM_CLASSES
            return x, y, w
        
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds
    
    def build_eeg_model(self, input_shape, num_classes):
        """Build CNN+BiLSTM+MHA model for EEG classification"""
        from tensorflow.keras import layers, Model
        
        def se_block(x, r=8):
            ch = x.shape[-1]
            s = layers.GlobalAveragePooling1D()(x)
            s = layers.Dense(max(ch // r, 8), activation='relu')(s)
            s = layers.Dense(ch, activation='sigmoid', dtype='float32')(s)
            s = layers.Reshape((1, ch))(s)
            return layers.Multiply()([x, s])
        
        def conv_bn_relu(x, f, k, d=1):
            x = layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            return x
        
        def res_block(x, f, k, d=1):
            sc = x
            y = conv_bn_relu(x, f, k, d)
            y = layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(y)
            y = layers.BatchNormalization()(y)
            if sc.shape[-1] != f:
                sc = layers.Conv1D(f, 1, padding="same", use_bias=False)(sc)
                sc = layers.BatchNormalization()(sc)
            y = layers.Add()([y, sc])
            y = layers.ReLU()(y)
            y = se_block(y)
            return y
        
        inp = layers.Input(shape=input_shape)
        
        x = conv_bn_relu(inp, 64, 7, d=1)
        x = res_block(x, 64, 7, d=1)
        x = layers.MaxPooling1D(2)(x)
        
        for d in [1, 2, 4]:
            x = res_block(x, 128, 5, d=d)
        
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.25)
        )(x)
        
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.35)(x)
        out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
        
        model = Model(inp, out)
        
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3, first_decay_steps=4,
            t_mul=2.0, m_mul=0.8, alpha=1e-5
        )
        opt = tf.keras.optimizers.AdamW(
            learning_rate=lr_sched, weight_decay=1e-4, global_clipnorm=1.0
        )
        
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="acc"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")
            ]
        )
        
        return model
    
    def serialize_weights(self, weights):
        """Serialize model weights for transmission."""
        buf = io.BytesIO()
        np.savez(buf, *weights)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights; uses numpy .npz format to avoid pickle/NumPy version mismatches."""
        buf = io.BytesIO(base64.b64decode(encoded_weights.encode('utf-8')))
        loaded = np.load(buf, allow_pickle=False)
        return [loaded[f'arr_{i}'] for i in range(len(loaded.files))]
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print(f"\n[Client {self.client_id}] Connected to MQTT broker")
            
            # Subscribe to topics
            self.mqtt_client.subscribe(TOPIC_GLOBAL_MODEL)
            print(f"  Subscribed to {TOPIC_GLOBAL_MODEL}")
            
            self.mqtt_client.subscribe(TOPIC_TRAINING_CONFIG)
            print(f"  Subscribed to {TOPIC_TRAINING_CONFIG}")
            
            self.mqtt_client.subscribe(TOPIC_START_TRAINING)
            print(f"  Subscribed to {TOPIC_START_TRAINING}")
            
            self.mqtt_client.subscribe(TOPIC_START_EVALUATION)
            print(f"  Subscribed to {TOPIC_START_EVALUATION}")
            
            self.mqtt_client.subscribe(TOPIC_TRAINING_COMPLETE)
            print(f"  Subscribed to {TOPIC_TRAINING_COMPLETE}")
            
            # Send registration message
            reg_msg = {
                "client_id": self.client_id,
                "num_samples": int(len(self.y_train))
            }
            self.mqtt_client.publish("fl/client_register", json.dumps(reg_msg))
            print(f"  Registration message sent")
        else:
            print(f"[Client {self.client_id}] Failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            if msg.topic == TOPIC_GLOBAL_MODEL:
                self.handle_global_model(msg.payload)
            elif msg.topic == TOPIC_TRAINING_CONFIG:
                self.handle_training_config(msg.payload)
            elif msg.topic == TOPIC_START_TRAINING:
                self.handle_start_training(msg.payload)
            elif msg.topic == TOPIC_START_EVALUATION:
                self.handle_start_evaluation(msg.payload)
            elif msg.topic == TOPIC_TRAINING_COMPLETE:
                self.handle_training_complete()
        except Exception as e:
            print(f"[Client {self.client_id}] Error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        if rc == 0:
            print(f"\n[Client {self.client_id}] Clean disconnect from broker")
            self.mqtt_client.loop_stop()
        else:
            print(f"[Client {self.client_id}] Unexpected disconnect, return code {rc}")
            self.mqtt_client.loop_stop()
    
    def handle_training_complete(self):
        """Handle training completion signal from server"""
        print("\n" + "=" * 70)
        print(f"[Client {self.client_id}] Training completed!")
        print("=" * 70)
        print("\nDisconnecting from server...")
        time.sleep(1)
        self.mqtt_client.disconnect()
    
    def handle_global_model(self, payload):
        """Receive and apply global model from server"""
        try:
            data = json.loads(payload.decode()) if isinstance(payload, bytes) else payload
            round_num = data.get('round', 0)
            
            if self.last_global_round == round_num and self.model is not None:
                print(f"Client {self.client_id} ignoring duplicate global model for round {round_num}")
                return
            
            print(f"Client {self.client_id} received global model (round {round_num})")
            
            if 'quantized_data' in data and self.quantizer is not None:
                compressed_data = data['quantized_data']
                if isinstance(compressed_data, str):
                    compressed_data = pickle.loads(base64.b64decode(compressed_data.encode('utf-8')))
                weights = self.quantizer.decompress(compressed_data)
                if round_num > 0:
                    print(f"Client {self.client_id}: Received quantized global model (dequantized for training)")
            else:
                weights = self.deserialize_weights(data['weights'])
            
            if self.model is None:
                model_config = data.get('model_config')
                if not model_config:
                    print(f"Client {self.client_id} WARNING: No model_config in global model, cannot initialize!")
                    return
                print(f"Client {self.client_id} building EEG model from server configuration...")
                self.model = self.build_eeg_model(
                    input_shape=tuple(model_config['input_shape']),
                    num_classes=model_config['num_classes']
                )
                self.model.set_weights(weights)
                _ = self.model(tf.zeros((1, *tuple(model_config['input_shape']))), training=False)
                print(f"Client {self.client_id} model initialized and verified with server weights")
                self.current_round = 0
            else:
                self.model.set_weights(weights)
                print(f"Client {self.client_id} received global model for round {round_num}, ready for next local round")
            
            self.last_global_round = round_num
            
        except Exception as e:
            print(f"\n[Client {self.client_id}] Error: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_training_config(self, payload):
        """Update training configuration"""
        cfg = json.loads(payload.decode())
        self.training_config.update(cfg)
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    def handle_start_training(self, payload):
        """Start local training when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']
        max_wait = 30
        wait_time = 0
        while self.model is None and wait_time < max_wait:
            print(f"Client {self.client_id} waiting for model initialization... ({wait_time}s)")
            time.sleep(1)
            wait_time += 1
        if self.model is None:
            print(f"Client {self.client_id} cannot train: model not initialized")
            return
        if round_num > self.current_round:
            self.current_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            self.train_local_model()
        else:
            print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
    
    def handle_start_evaluation(self, payload):
        """Handle evaluation signal (no held-out test set for this client)"""
        print(f"\n[Client {self.client_id}] Evaluation requested (no local test set)")
    
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
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        
        ds_train = self.make_dataset(
            self.x_train, self.y_train,
            batch_size, training=True
        )
        
        print(f"[Client {self.client_id}] Training for {epochs} epochs...")
        training_start = time.time()
        history = self.model.fit(
            ds_train,
            epochs=epochs,
            verbose=2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss", patience=3,
                    restore_best_weights=True, verbose=0
                )
            ]
        )
        training_time = time.time() - training_start
        
        updated_weights = self.model.get_weights()
        
        final_loss = float(history.history['loss'][-1]) if 'loss' in history.history else 0.0
        final_acc = float(history.history.get('acc', [0.0])[-1])
        self._update_local_convergence(final_loss)
        metrics = {
            "loss": final_loss,
            "accuracy": final_acc
        }
        if self.has_converged and STOP_ON_CLIENT_CONVERGENCE:
            metrics["client_converged"] = 1.0
        num_samples = int(len(self.y_train))
        
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
            stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - "
                  f"Ratio: {stats['compression_ratio']:.2f}x, "
                  f"Size: {stats['compressed_size_mb']:.2f}MB")
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "compressed_data": serialized,
                "num_samples": num_samples,
                "metrics": metrics
            }
        else:
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "weights": self.serialize_weights(updated_weights),
                "num_samples": num_samples,
                "metrics": metrics
            }
        
        delay = random.uniform(0.5, 3.0)
        print(f"Client {self.client_id} waiting {delay:.2f} seconds before sending update...")
        time.sleep(delay)
        
        comm_start = time.time()
        self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, json.dumps(update_message))
        uplink_comm_sec = time.time() - comm_start
        
        append_client_fl_metrics_record(
            self.client_id,
            {
                "client_id": self.client_id,
                "round": self.current_round,
                "loss": float(final_loss),
                "accuracy": float(final_acc),
                "training_time_sec": float(training_time),
                "pre_uplink_delay_sec": float(delay),
                "uplink_model_comm_sec": float(uplink_comm_sec),
                "total_fl_wall_time_sec": float(training_time + delay + uplink_comm_sec),
                "battery_energy_joules": 0.0,
                "battery_soc_after": 1.0,
            },
            use_case=use_case_from_env("mental_state"),
            protocol="mqtt",
        )
        
        print(f"Client {self.client_id} sent model update for round {self.current_round}")
        print(f"Training metrics - Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
        if self.has_converged and STOP_ON_CLIENT_CONVERGENCE:
            print(f"Client {self.client_id} notifying server of convergence and disconnecting")
            time.sleep(2)
            self.mqtt_client.disconnect()
    
    def start(self):
        """Connect to MQTT broker and listen for server messages"""
        max_retries = 5
        retry_delay = 2
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
                print(f"Successfully connected to MQTT broker!\n")
                self.mqtt_client.loop_forever()
                break
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to MQTT broker after {max_retries} attempts.")
                    raise


if __name__ == "__main__":
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS)
    client.start()
