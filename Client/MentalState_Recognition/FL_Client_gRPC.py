"""
Federated Learning Client for EEG Mental State Recognition using gRPC
Supports CNN+BiLSTM+MHA architecture with non-IID data partitioning
"""

import os
import sys
import json
import pickle
import time
import random
import logging
import numpy as np
import tensorflow as tf
import grpc

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

import threading
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

# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# gRPC Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))

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
        
        # gRPC connection
        self.channel = None
        self.stub = None
        self.running = True
        
        # Load and partition data
        self.prepare_data()
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {len(self.y_train)}")
        print(f"  Input shape: {self.x_train.shape}")
        print(f"  Waiting for initial global model from server...")
    
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
        
        # Dilated conv stack
        for d in [1, 2, 4]:
            x = res_block(x, 128, 5, d=d)
        
        # Temporal modeling
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.25)
        )(x)
        
        # Self-attention
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        
        # Pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.35)(x)
        out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
        
        model = Model(inp, out)
        
        # Compile model
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
    
    def connect(self):
        """Connect to gRPC server"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to gRPC server at {GRPC_HOST}:{GRPC_PORT}...")
                options = [
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                ]
                self.channel = grpc.insecure_channel(f'{GRPC_HOST}:{GRPC_PORT}', options=options)
                self.stub = federated_learning_pb2_grpc.FederatedLearningStub(self.channel)
                
                # Test connection by registering
                print(f"[DEBUG] Client {self.client_id} sending registration request to server...")
                response = self.stub.RegisterClient(
                    federated_learning_pb2.ClientRegistration(client_id=self.client_id)
                )
                print(f"[DEBUG] Client {self.client_id} received registration response: success={response.success}")
                
                if response.success:
                    print(f"Client {self.client_id} connected to gRPC server")
                    print(f"Client {self.client_id} registration: {response.message}")
                    
                    # Get training configuration
                    config = self.stub.GetTrainingConfig(
                        federated_learning_pb2.ConfigRequest(client_id=self.client_id)
                    )
                    self.training_config.update({
                        "batch_size": config.batch_size,
                        "local_epochs": config.local_epochs
                    })
                    print(f"Client {self.client_id} received config: {self.training_config}")
                    
                    return True
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to gRPC server after {max_retries} attempts.")
                    print(f"\nPlease ensure:")
                    print(f"  1. gRPC server is running")
                    print(f"  2. Server address is correct: {GRPC_HOST}:{GRPC_PORT}")
                    raise
    
    def run(self):
        """Main client loop"""
        print(f"\nClient {self.client_id} waiting for server to start...\n")
        
        # Get initial global model
        initial_model_received = False
        retry_count = 0
        max_connection_retries = 30
        
        while not initial_model_received and retry_count < max_connection_retries:
            try:
                model_update = self.stub.GetGlobalModel(
                    federated_learning_pb2.ModelRequest(client_id=self.client_id, round=0)
                )
                
                print(f"[DEBUG] Initial model fetch - available={model_update.available}, has_weights={bool(model_update.weights)}, has_config={bool(model_update.model_config)}, round={model_update.round}")
                
                if model_update.available and model_update.weights and model_update.model_config:
                    self.receive_global_model(model_update)
                    initial_model_received = True
                    print(f"Client {self.client_id} received initial model from server\n")
                else:
                    retry_count += 1
                    if retry_count % 5 == 0:
                        print(f"Still waiting for initial model... ({retry_count}/{max_connection_retries})")
                    time.sleep(2)
                    
            except grpc.RpcError as e:
                print(f"Error getting initial model: {e}")
                retry_count += 1
                time.sleep(2)
        
        if not initial_model_received:
            print(f"Failed to receive initial model after {max_connection_retries} attempts")
            return
        
        # Main training loop
        while self.running:
            try:
                status = self.stub.CheckTrainingStatus(
                    federated_learning_pb2.StatusRequest(
                        client_id=self.client_id,
                        current_round=self.current_round
                    )
                )
                
                print(f"[DEBUG] Client {self.client_id} - Status: should_train={status.should_train}, round={status.round}, current_round={self.current_round}, training_complete={status.training_complete}")
                
                if status.training_complete:
                    print(f"\n{'='*70}")
                    print(f"Client {self.client_id} - Training completed!")
                    print(f"{'='*70}\n")
                    break
                
                if not status.should_train:
                    time.sleep(1)
                    continue
                
                print(f"[DEBUG] Client {self.client_id} - should_train is True, checking if need to fetch new model...")
                
                # If server's round is ahead, fetch new global model first
                if status.round > self.current_round:
                    print(f"[DEBUG] Client {self.client_id} - Fetching global model for round {status.round}...")
                    model_update = self.stub.GetGlobalModel(
                        federated_learning_pb2.ModelRequest(client_id=self.client_id, round=self.current_round)
                    )
                    
                    print(f"[DEBUG] Client {self.client_id} - Received model update, round={model_update.round}, status.round={status.round}")
                    
                    if model_update.round == status.round:
                        print(f"[DEBUG] Client {self.client_id} - Model round matches, updating...")
                        self.receive_global_model(model_update)
                        self.current_round = status.round
                    else:
                        print(f"[DEBUG] Client {self.client_id} - Model round mismatch!")
                        time.sleep(0.5)
                        continue
                
                # Ensure model is initialized before training
                if self.model is None:
                    print(f"Client {self.client_id} ERROR: Model not initialized!")
                    time.sleep(1)
                    continue
                
                print(f"\nClient {self.client_id} starting training for round {self.current_round}...")
                self.train_local_model()
                
                time.sleep(0.5)
                
            except grpc.RpcError as e:
                if "unavailable" in str(e).lower():
                    print(f"Connection lost to server. Exiting...")
                    break
                else:
                    print(f"RPC error in main loop: {e}")
                    time.sleep(1)
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)
        
        print(f"Client {self.client_id} shutting down...")
        if self.channel:
            self.channel.close()
    
    def receive_global_model(self, model_update):
        """Receive and set global model weights from server"""
        try:
            print(f"[DEBUG] Client {self.client_id} receive_global_model - round={model_update.round}, has_config={bool(model_update.model_config)}, model exists={self.model is not None}")
            
            # Decompress or deserialize weights (handle pickled compressed dicts)
            if model_update.weights:
                if self.quantizer is not None:
                    try:
                        candidate = pickle.loads(model_update.weights)
                        if isinstance(candidate, dict) and 'quantization_params' in candidate:
                            weights = self.quantizer.decompress(candidate)
                            print(f"Client {self.client_id}: Received and decompressed quantized global model")
                        else:
                            weights = candidate
                    except Exception:
                        weights = pickle.loads(model_update.weights)
                else:
                    weights = pickle.loads(model_update.weights)
            else:
                weights = None
            
            # Build model if we receive config (first time initialization)
            if model_update.model_config and self.model is None:
                model_config = json.loads(model_update.model_config)
                
                print(f"Client {self.client_id} building EEG model from server configuration...")
                self.model = self.build_eeg_model(
                    input_shape=tuple(model_config['input_shape']),
                    num_classes=model_config['num_classes']
                )
                
                print(f"Client {self.client_id} built EEG model from server configuration")
                print(f"  Input shape: {model_config['input_shape']}")
                print(f"  Output classes: {model_config['num_classes']}")
            
            if self.model is None:
                print(f"Client {self.client_id} ERROR: Model not initialized before setting weights!")
                return
            
            self.model.set_weights(weights)
            
            if model_update.model_config:
                # Verify model is ready (first initialization)
                model_config = json.loads(model_update.model_config)
                _ = self.model(tf.zeros((1, *model_config['input_shape'])), training=False)
                print(f"Client {self.client_id} initialized and verified with global model")
            else:
                print(f"Client {self.client_id} updated with global model for round {model_update.round}")
                
        except Exception as e:
            print(f"Error receiving global model: {e}")
            import traceback
            traceback.print_exc()
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        try:
            batch_size = self.training_config['batch_size']
            epochs = self.training_config['local_epochs']
            
            # Create dataset
            ds_train = self.make_dataset(
                self.x_train, self.y_train, 
                batch_size, training=True
            )
            
            # Train
            print(f"[Client {self.client_id}] Training for {epochs} epochs...")
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
            
            # Get updated weights
            updated_weights = self.model.get_weights()
            
            # Prepare metrics
            final_loss = float(history.history['loss'][-1]) if 'loss' in history.history else 0.0
            final_acc = float(history.history['acc'][-1]) if 'acc' in history.history else 0.0
            
            # Compress or serialize weights
            if self.quantizer is not None:
                compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
                stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
                print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Original: {stats['original_size_mb']:.2f}MB, Compressed: {stats['compressed_size_mb']:.2f}MB")
                # Send pickled compressed dict so server can read quantization metadata
                serialized_weights = pickle.dumps(compressed_data)
            else:
                serialized_weights = pickle.dumps(updated_weights)
            
            # Random delay before sending
            delay = random.uniform(0.5, 3.0)
            print(f"Client {self.client_id} waiting {delay:.2f} seconds before sending update...")
            time.sleep(delay)
            
            # Send update to server
            response = self.stub.SendModelUpdate(
                federated_learning_pb2.ModelUpdate(
                    client_id=self.client_id,
                    round=self.current_round,
                    weights=serialized_weights,
                    num_samples=int(len(self.y_train)),
                    metrics={
                        'loss': final_loss,
                        'accuracy': final_acc
                    }
                )
            )
            
            if response.success:
                print(f"Client {self.client_id} successfully sent update for round {self.current_round}")
                print(f"Training metrics - Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
            else:
                print(f"Failed to send update: {response.message}")
                
        except Exception as e:
            print(f"Error in train_local_model: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS)
    
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Client {CLIENT_ID}")
    print(f"Server: {GRPC_HOST}:{GRPC_PORT}")
    print(f"{'='*60}\n")
    
    try:
        if client.connect():
            client.run()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} interrupted by user")
    except Exception as e:
        print(f"\nClient {CLIENT_ID} error: {e}")
        import traceback
        traceback.print_exc()
