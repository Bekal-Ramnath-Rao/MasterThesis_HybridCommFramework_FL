"""
Federated Learning Client for EEG Mental State Recognition using QUIC
Supports CNN+BiLSTM+MHA architecture with non-IID data partitioning
"""

import os
import sys
import json
import pickle
import base64
import time
import asyncio
import logging
import numpy as np
import tensorflow as tf
from collections import Counter
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived
from aioquic.asyncio.protocol import QuicConnectionProtocol

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

# QUIC Configuration
QUIC_HOST = os.getenv("QUIC_HOST", "localhost")
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))

# Model initialization timeout (seconds) - longer for poor network conditions
# Default: 300s (5 minutes) for very poor network conditions with large models
MODEL_INIT_TIMEOUT = float(os.getenv("MODEL_INIT_TIMEOUT", "300"))

# Training Configuration
AUTOTUNE = tf.data.AUTOTUNE
SMOOTH_EPS = 0.05


class FederatedLearningClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = None
        self.stream_id = None
        self._stream_buffers = {}
    
    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, StreamDataReceived):
            if event.stream_id not in self._stream_buffers:
                self._stream_buffers[event.stream_id] = b''
            
            self._stream_buffers[event.stream_id] += event.data
            buffer_size = len(self._stream_buffers[event.stream_id])
            print(f"[DEBUG] Client stream {event.stream_id}: received {len(event.data)} bytes, buffer now {buffer_size} bytes, end_stream={event.end_stream}")
            
            # Send flow control updates to allow more data (critical for poor networks)
            self.transmit()
            
            while b'\n' in self._stream_buffers[event.stream_id]:
                message_data, self._stream_buffers[event.stream_id] = self._stream_buffers[event.stream_id].split(b'\n', 1)
                if message_data:
                    try:
                        data = message_data.decode('utf-8')
                        message = json.loads(data)
                        msg_type = message.get('type', 'unknown')
                        print(f"[DEBUG] Client decoded message from stream {event.stream_id}: type={msg_type}, size={len(message_data)} bytes")
                        if self.client:
                            asyncio.create_task(self.client.handle_message(message))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding message: {e}")
            
            # If stream ended and buffer has remaining data, try to process it
            if event.end_stream and self._stream_buffers[event.stream_id]:
                print(f"[DEBUG] Client stream {event.stream_id} ended with {len(self._stream_buffers[event.stream_id])} bytes remaining")
                try:
                    data = self._stream_buffers[event.stream_id].decode('utf-8')
                    message = json.loads(data)
                    msg_type = message.get('type', 'unknown')
                    print(f"[DEBUG] Client decoded end-of-stream message: type={msg_type}")
                    if self.client:
                        asyncio.create_task(self.client.handle_message(message))
                    self._stream_buffers[event.stream_id] = b''
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error decoding remaining buffer: {e}")


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
        self.protocol = None
        self.stream_id = 0
        self.model_ready = asyncio.Event()
        
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
    
    def build_model_from_config(self, model_config):
        """Build model from server-provided configuration"""
        input_shape = model_config.get('input_shape')
        num_classes = model_config.get('num_classes')
        layers = model_config.get('layers', [])
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        for layer in layers:
            if layer['type'] == 'conv':
                model.add(tf.keras.layers.Conv2D(
                    layer['filters'], 
                    layer['kernel'], 
                    activation=layer['activation'],
                    padding='same'
                ))
            elif layer['type'] == 'maxpool':
                model.add(tf.keras.layers.MaxPooling2D(layer['pool_size']))
            elif layer['type'] == 'flatten':
                model.add(tf.keras.layers.Flatten())
            elif layer['type'] == 'dense':
                model.add(tf.keras.layers.Dense(layer['units'], activation=layer['activation']))
            elif layer['type'] == 'dropout':
                model.add(tf.keras.layers.Dropout(layer['rate']))
            elif layer['type'] == 'lstm':
                model.add(tf.keras.layers.LSTM(layer['units'], return_sequences=layer.get('return_sequences', False)))
            elif layer['type'] == 'gru':
                model.add(tf.keras.layers.GRU(layer['units'], return_sequences=layer.get('return_sequences', False)))
        
        model.compile(
            optimizer='adam',
            loss=model_config.get('loss', 'categorical_crossentropy'),
            metrics=['accuracy']
        )
        
        return model
    
    
    async def send_message(self, message):
        """Send message to server via QUIC stream"""
        if self.protocol:
            msg_type = message.get('type')
            print(f"[DEBUG] Client {self.client_id} sending message type: {msg_type}")
            
            data = (json.dumps(message) + '\n').encode('utf-8')
            print(f"[DEBUG] Client {self.client_id} message size: {len(data)} bytes")
            
            self.stream_id = self.protocol._quic.get_next_available_stream_id()
            # End stream to ensure data is flushed and processed
            self.protocol._quic.send_stream_data(self.stream_id, data, end_stream=True)
            self.protocol.transmit()
            
            # FAIR FIX: Removed artificial delays (1.5s for large, 0.1s for small messages)
            # QUIC handles flow control automatically, so manual delays are unnecessary
            # This makes QUIC behavior similar to other protocols which don't have artificial delays
            print(f"[DEBUG] Client {self.client_id} sent {msg_type} on stream {self.stream_id}")
    
    async def handle_message(self, message):
        """Handle incoming messages from server"""
        try:
            msg_type = message.get('type')
            print(f"[DEBUG] Client {self.client_id} received message type: {msg_type}")
            
            if msg_type == 'training_config':
                await self.handle_training_config(message)
            elif msg_type == 'global_model':
                await self.handle_global_model(message)
            elif msg_type == 'start_training':
                await self.handle_start_training(message)
            elif msg_type == 'start_evaluation':
                await self.handle_start_evaluation(message)
            elif msg_type == 'training_complete':
                await self.handle_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    async def handle_training_config(self, message):
        """Update training configuration"""
        self.training_config.update(message['config'])
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    async def handle_global_model(self, message):
        """Receive and set global model weights"""
        round_num = message['round']
        encoded_weights = message['weights']
        
        print(f"[DEBUG] Client {self.client_id} handle_global_model - round={round_num}, has_config={bool(message.get('model_config'))}, model_exists={self.model is not None}")
        
        # Decompress or deserialize weights
        if 'quantized_data' in message and self.quantizer is not None:
            weights = self.quantizer.decompress(message['quantized_data'])
            print(f"Client {self.client_id}: Received and decompressed quantized global model")
        elif 'compressed_data' in message and self.quantizer is not None:
            weights = self.quantizer.decompress(message['compressed_data'])
            print(f"Client {self.client_id}: Received and decompressed quantized global model")
        else:
            weights = self.deserialize_weights(encoded_weights)
        
        # Initialize model if not yet created (works for any round)

        
        if self.model is None:

        
            print(f"Client {self.client_id} initializing model from server (round {round_num})")
            
            model_config = message.get('model_config')
            if model_config:
                self.model = self.build_eeg_model(
                    input_shape=tuple(model_config['input_shape']),
                    num_classes=model_config['num_classes']
                )
                print(f"Client {self.client_id} built EEG model from server configuration")
                print(f"  Input shape: {model_config['input_shape']}")
                print(f"  Output classes: {model_config['num_classes']}")
            else:
                raise ValueError("No model configuration received from server!")
            
            self.model.set_weights(weights)
            print(f"Client {self.client_id} model initialized with server weights")
            self.current_round = 0
            self.model_ready.set()
        else:
            # Update model weights
            self.model.set_weights(weights)
            # Only update current_round if this model is for a round >= current
            # (Don't go backwards if we've already moved to a later round)
            if round_num >= self.current_round:
                self.current_round = round_num
                print(f"Client {self.client_id} received global model for round {round_num}")
            else:
                print(f"Client {self.client_id} received late global model for round {round_num} (currently at round {self.current_round}), updating weights only")
            self.model_ready.set()
    
    async def handle_start_training(self, message):
        """Start local training when server signals"""
        round_num = message['round']
        
        print(f"[DEBUG] Client {self.client_id} received start_training - round={round_num}, model_ready={self.model_ready.is_set()}, current_round={self.current_round}")
        
        if not self.model_ready.is_set():
            print(f"Client {self.client_id} waiting for model initialization before training round {round_num}...")
            print(f"Client {self.client_id} using timeout of {MODEL_INIT_TIMEOUT}s (configured via MODEL_INIT_TIMEOUT env var)")
            try:
                await asyncio.wait_for(self.model_ready.wait(), timeout=MODEL_INIT_TIMEOUT)
                print(f"Client {self.client_id} model ready, proceeding with training")
            except asyncio.TimeoutError:
                print(f"Client {self.client_id} ERROR: Timeout waiting for model initialization after {MODEL_INIT_TIMEOUT}s")
                print(f"Client {self.client_id} TIP: Increase MODEL_INIT_TIMEOUT env var for very poor network conditions")
                return
        
        # Update to the new round and start training
        if round_num > self.current_round:
            self.current_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            await self.train_local_model()
        else:
            print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
    
    async def handle_start_evaluation(self, message):
        """Handle evaluation signal"""
        print(f"\n[Client {self.client_id}] Evaluation requested (no local test set)")
    
    async def handle_training_complete(self):
        """Handle training completion signal from server"""
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nClient shutting down...")
        await asyncio.sleep(1)
        import sys
        sys.exit(0)
    
    async def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        
        # Create dataset
        ds_train = self.make_dataset(
            self.x_train, self.y_train, 
            batch_size, training=True
        )
        
        print(f"[Client {self.client_id}] Training for {epochs} epochs...")
        
        # Train in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def train_model():
            return self.model.fit(
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
        
        history = await loop.run_in_executor(None, train_model)
        
        # Prepare metrics
        final_loss = float(history.history['loss'][-1]) if 'loss' in history.history else 0.0
        final_acc = float(history.history['acc'][-1]) if 'acc' in history.history else 0.0
        
        print(f"Client {self.client_id} training complete - "
              f"Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
        
        print(f"[DEBUG] Client {self.client_id} sending model_update for round {self.current_round}")
        
        # Prepare weights (compress if quantization enabled)
        updated_weights = self.model.get_weights()
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
            stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Size: {stats['compressed_size_mb']:.2f}MB")
            weights_data = compressed_data
            weights_key = 'compressed_data'
        else:
            weights_data = self.serialize_weights(updated_weights)
            weights_key = 'weights'
        
        # Send model update to server
        await self.send_message({
            'type': 'model_update',
            'client_id': self.client_id,
            'round': self.current_round,
            weights_key: weights_data,
            'num_samples': int(len(self.y_train)),
            'metrics': {
                'loss': final_loss,
                'accuracy': final_acc
            }
        })
    
    async def register_with_server(self):
        """Register with the federated learning server"""
        await self.send_message({
            'type': 'register',
            'client_id': self.client_id,
            'num_samples': int(len(self.y_train))
        })
        print(f"Client {self.client_id} registration sent to server")


async def main():
    print(f"\n{'='*70}")
    print(f"EEG Mental State Recognition - Federated Learning Client {CLIENT_ID}")
    print(f"Server: {QUIC_HOST}:{QUIC_PORT}")
    print(f"{'='*70}\n")
    
    # Create client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS)
    
    # Configure QUIC
    configuration = QuicConfiguration(
        is_client=True,
        alpn_protocols=["fl"],
        max_stream_data=50 * 1024 * 1024,  # 50 MB per stream
        max_data=100 * 1024 * 1024,  # 100 MB total
        idle_timeout=3600.0,  # 60 minutes idle timeout
        max_datagram_frame_size=65536,  # Larger frame size for better throughput
        initial_rtt=0.15,  # 150ms (account for 100ms latency + jitter)
    )
    
    # Load CA certificate for verification (optional - set verify_mode to False for testing)
    # cert_dir = Path(__file__).parent.parent.parent / "certs"
    # ca_cert = cert_dir / "server-cert.pem"
    # if ca_cert.exists():
    #     configuration.load_verify_locations(str(ca_cert))
    configuration.verify_mode = False
    
    # Create protocol factory
    def create_protocol(*args, **kwargs):
        protocol = FederatedLearningClientProtocol(*args, **kwargs)
        protocol.client = client
        client.protocol = protocol
        return protocol
    
    # Connect to server
    print(f"Connecting to QUIC server at {QUIC_HOST}:{QUIC_PORT}...")
    try:
        async with connect(
            QUIC_HOST,
            QUIC_PORT,
            configuration=configuration,
            create_protocol=create_protocol,
        ) as protocol:
            client.protocol = protocol
            print(f"✓ Connected to QUIC server successfully")
            
            # Register with server
            await client.register_with_server()
            
            print(f"Client {CLIENT_ID} waiting for training commands...")
            # Keep connection alive
            try:
                await asyncio.Future()
            except Exception as e:
                print(f"\n[ERROR] Client {CLIENT_ID} - Connection loop error: {e}")
                import traceback
                traceback.print_exc()
                raise
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print(f"Failed to connect to QUIC server at {QUIC_HOST}:{QUIC_PORT}")
        print("Make sure the server is running and reachable.")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nClient interrupted by user")
        import sys
        sys.exit(0)
