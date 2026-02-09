"""
Federated Learning Client for EEG Mental State Recognition using MQTT
Supports CNN+BiLSTM+MHA architecture with non-IID data partitioning
"""

import os
import sys
import json
import pickle
import base64
import time
import logging
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

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
        self.x_train = None
        self.y_train = None
        self.current_round = 0
        self.training_config = {
            "batch_size": 16,
            "local_epochs": 5,
            "val_split": 0.10
        }
        self.class_weights = None
        
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
    
    def serialize_weights(self, weights):
        """Serialize model weights for MQTT transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from MQTT"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights
    
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
            
            # Check for duplicate (already processed this exact model)
            if hasattr(self, 'last_global_round') and self.last_global_round == round_num and self.model is not None:
                print(f"Client {self.client_id} ignoring duplicate global model for round {round_num}")
                return
            
            print(f"Client {self.client_id} received global model (round {round_num})")
            
            # Decompress/deserialize weights
            if 'quantized_data' in data:
                # Handle quantized/compressed data
                compressed_data = data['quantized_data']
                if isinstance(compressed_data, str):
                    import base64, pickle
                    compressed_data = pickle.loads(base64.b64decode(compressed_data.encode('utf-8')))
                if hasattr(self, 'quantization') and self.quantization is not None:
                    weights = self.quantization.decompress(compressed_data)
                elif hasattr(self, 'quantizer') and self.quantizer is not None:
                    weights = self.quantizer.decompress(compressed_data)
                else:
                    weights = compressed_data
                print(f"Client {self.client_id} decompressed quantized model")
            else:
                # Normal weights
                if 'weights' in data:
                    encoded_weights = data['weights']
                    if isinstance(encoded_weights, str):
                        import base64, pickle
                        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
                        weights = pickle.loads(serialized)
                    else:
                        weights = encoded_weights
                else:
                    weights = data.get('parameters', [])
            
            # Initialize model if not already done (for late-joining or first-time clients)
            if self.model is None:
                model_config = data.get('model_config')
                if model_config:
                    print(f"Client {self.client_id} initializing model from received configuration...")
                    self.model = self.build_model_from_config(model_config)
                    print(f"Client {self.client_id} model built successfully")
                else:
                    print(f"Client {self.client_id} WARNING: No model_config in global model, cannot initialize!")
                    return
            
            # Apply received weights
            self.model.set_weights(weights)
            self.current_round = round_num
            if hasattr(self, 'last_global_round'):
                self.last_global_round = round_num
            print(f"Client {self.client_id} updated model weights (round {round_num})")
            
        except Exception as e:
            print(f"\n[Client {self.client_id}] Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS)
    client.start()
