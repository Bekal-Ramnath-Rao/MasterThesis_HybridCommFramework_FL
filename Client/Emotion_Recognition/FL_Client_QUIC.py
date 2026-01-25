import numpy as np
import json
import pickle
import base64
import time
import asyncio
import os
import logging
import sys
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived
from aioquic.asyncio.protocol import QuicConnectionProtocol

# GPU Configuration - Must be done BEFORE TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

# Disable Grappler layout optimizer to avoid NCHW transpose errors in logs
os.environ["TF_ENABLE_LAYOUT_OPTIMIZER"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Make TensorFlow logs less verbose
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Ensure Keras uses channels_last image data format
tf.keras.backend.set_image_data_format('channels_last')

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
        
        # Set GPU memory limit to avoid OOM (RTX 3080 has 10GB, reserve 7GB per process)
        # This prevents one process from consuming all GPU memory
        for gpu in gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=7000)]  # 7GB per GPU
                )
            except RuntimeError:
                pass  # GPU already configured
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPUs found. Running on CPU.")

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

# QUIC Configuration
QUIC_HOST = os.getenv("QUIC_HOST", "fl-server-quic-emotion")
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))


class FederatedLearningClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = None
        self.stream_id = None
        self._stream_buffers = {}  # Buffer for incomplete messages
    
    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, StreamDataReceived):
            # Get or create buffer for this stream
            if event.stream_id not in self._stream_buffers:
                self._stream_buffers[event.stream_id] = b''
            
            # Append new data to buffer
            self._stream_buffers[event.stream_id] += event.data
            
            # Try to decode complete messages (delimited by newline)
            while b'\n' in self._stream_buffers[event.stream_id]:
                message_data, self._stream_buffers[event.stream_id] = self._stream_buffers[event.stream_id].split(b'\n', 1)
                if message_data:
                    try:
                        data = message_data.decode('utf-8')
                        message = json.loads(data)
                        if self.client:
                            asyncio.create_task(self.client.handle_message(message))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding message: {e}")
            
            # If stream ended and buffer has remaining data, try to process it
            if event.end_stream and self._stream_buffers[event.stream_id]:
                try:
                    data = self._stream_buffers[event.stream_id].decode('utf-8')
                    message = json.loads(data)
                    if self.client:
                        asyncio.create_task(self.client.handle_message(message))
                    self._stream_buffers[event.stream_id] = b''
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error decoding remaining buffer: {e}")


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, train_generator, validation_generator):
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
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.current_round = 0
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        self.protocol = None
        self.stream_id = 0
        self.model_ready = asyncio.Event()  # Event to signal model is ready
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {self.train_generator.n}")
        print(f"  Validation samples: {self.validation_generator.n}")
        print(f"  Waiting for initial global model from server...")
    
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
    
    async def send_message(self, message):
        """Send message to server via QUIC stream"""
        if self.protocol:
            msg_type = message.get('type')
            #print(f"[DEBUG] Client {self.client_id} sending message type: {msg_type}")
            
            # Add newline delimiter for message framing
            data = (json.dumps(message) + '\n').encode('utf-8')
            #print(f"[DEBUG] Client {self.client_id} message size: {len(data)} bytes")
            
            self.stream_id = self.protocol._quic.get_next_available_stream_id()
            # Send data with end_stream=True to ensure it's processed
            self.protocol._quic.send_stream_data(self.stream_id, data, end_stream=True)
            
            # Transmit immediately
            self.protocol.transmit()
            
            # For large messages, give more time for transmission
            if len(data) > 1000000:  # > 1MB
                #print(f"[DEBUG] Client {self.client_id} waiting for large message transmission...")
                # Multiple transmit calls with delays for very large messages
                for i in range(5):
                    await asyncio.sleep(1)
                    self.protocol.transmit()
                    #print(f"[DEBUG] Client {self.client_id} transmit call {i+1}/5")
            else:
                await asyncio.sleep(0.5)
            
            #print(f"[DEBUG] Client {self.client_id} sent {msg_type} on stream {self.stream_id}")
    
    async def handle_message(self, message):
        """Handle incoming messages from server"""
        try:
            msg_type = message.get('type')
            #print(f"[DEBUG] Client {self.client_id} received message type: {msg_type}")
            
            if msg_type == 'training_config':
                await self.handle_training_config(message)
            elif msg_type == 'global_model':
                await self.handle_global_model(message)
            elif msg_type == 'start_training':
                #print(f"[DEBUG] Client {self.client_id} handling start_training for round {message.get('round')}")
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
        self.training_config = message['config']
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    async def handle_global_model(self, message):
        """Receive and set global model weights and architecture from server"""
        round_num = message['round']
        encoded_weights = message['weights']
        
        # Decompress or deserialize weights
        if 'quantized_data' in message and self.quantizer is not None:
            weights = self.quantizer.decompress(message['quantized_data'])
            print(f"Client {self.client_id}: Received and decompressed quantized global model")
        elif 'compressed_data' in message and self.quantizer is not None:
            weights = self.quantizer.decompress(message['compressed_data'])
            print(f"Client {self.client_id}: Received and decompressed quantized global model")
        else:
            weights = self.deserialize_weights(encoded_weights)
        
        if round_num == 0:
            # If we've already moved past initialization, ignore repeated initial models
            if self.current_round > 0:
                #print(f"Client {self.client_id} ignoring duplicate initial global model (already in round {self.current_round})")
                return
            # Initial model from server - create model from server's config
            print(f"Client {self.client_id} received initial global model from server")
            
            model_config = message.get('model_config')
            if model_config:
                # Build model dynamically from server's layer configuration
                self.model = Sequential()
                
                for layer_info in model_config['layers']:
                    layer_type = layer_info['type']
                    
                    if layer_type == 'Input':
                        self.model.add(Input(shape=tuple(model_config['input_shape'])))
                    elif layer_type == 'Conv2D':
                        self.model.add(Conv2D(
                            filters=layer_info['filters'],
                            kernel_size=tuple(layer_info['kernel_size']),
                            activation=layer_info['activation']
                        ))
                    elif layer_type == 'MaxPooling2D':
                        self.model.add(MaxPooling2D(pool_size=tuple(layer_info['pool_size'])))
                    elif layer_type == 'Dropout':
                        self.model.add(Dropout(layer_info['rate']))
                    elif layer_type == 'Flatten':
                        self.model.add(Flatten())
                    elif layer_type == 'Dense':
                        if 'activation' in layer_info:
                            self.model.add(Dense(layer_info['units'], activation=layer_info['activation']))
                        else:
                            self.model.add(Dense(layer_info['units']))
                
                # Compile model
                self.model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=0.0001),
                    metrics=['accuracy']
                )
                print(f"Client {self.client_id} built CNN model from server configuration")
                print(f"  Model has {len(model_config['layers'])} layers, {model_config['num_classes']} output classes")
            else:
                raise ValueError("No model configuration received from server!")
            
            # Set the initial weights from server
            self.model.set_weights(weights)
            print(f"Client {self.client_id} model initialized with server weights")
            self.current_round = 0
            self.model_ready.set()  # Signal that model is ready
        else:
            # Updated model after aggregation
            self.model.set_weights(weights)
            # Only update current_round if this model is for a round >= current
            # (Don't go backwards if we've already moved to a later round)
            if round_num >= self.current_round:
                self.current_round = round_num
                print(f"Client {self.client_id} received global model for round {round_num}")
            else:
                print(f"Client {self.client_id} received late global model for round {round_num} (currently at round {self.current_round}), updating weights only")
            self.model_ready.set()  # Signal that model is ready
    
    async def handle_start_training(self, message):
        """Start local training when server signals"""
        round_num = message['round']
        
        # Wait for model to be initialized (with timeout)
        if not self.model_ready.is_set():
            print(f"Client {self.client_id} waiting for model initialization before training round {round_num}...")
            try:
                await asyncio.wait_for(self.model_ready.wait(), timeout=30.0)
                print(f"Client {self.client_id} model ready, proceeding with training")
            except asyncio.TimeoutError:
                print(f"Client {self.client_id} ERROR: Timeout waiting for model initialization")
                return
        
        if self.current_round == 0 and round_num == 1:
            self.current_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num} with initial global model...")
            await self.train_local_model()
        elif round_num == self.current_round:
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            await self.train_local_model()
        else:
            print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
    
    async def handle_start_evaluation(self, message):
        """Start evaluation when server signals"""
        round_num = message['round']
        
        # Ensure model is ready
        if not self.model_ready.is_set():
            print(f"Client {self.client_id} waiting for model before evaluation...")
            await self.model_ready.wait()
        
        if round_num == self.current_round:
            print(f"Client {self.client_id} starting evaluation for round {round_num}...")
            await self.evaluate_model()
            self.current_round = round_num + 1
            # Don't clear model_ready - the model is still valid for next round
            print(f"Client {self.client_id} ready for next round {self.current_round}")
        else:
            print(f"Client {self.client_id} skipping evaluation signal for round {round_num} (current: {self.current_round})")
    
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
        # Limit steps per epoch for faster training (configurable via env)
        try:
            steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
            val_steps = int(os.getenv("VAL_STEPS", "25"))
        except Exception:
            steps_per_epoch = 100
            val_steps = 25
        
        print(f"Training on {self.train_generator.n} samples for {epochs} epochs...")
        
        # Train the model directly (synchronous call is faster, no executor overhead)
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.validation_generator,
            validation_steps=val_steps,
            verbose=2
        )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        num_samples = self.train_generator.n  # Total number of training samples
        
        # Prepare training metrics (for classification)
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1])
        }
        
        # Compress weights if quantization is enabled
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
            stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - "
                  f"Ratio: {stats['compression_ratio']:.2f}x, "
                  f"Size: {stats['compressed_size_mb']:.2f}MB")
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
            'num_samples': num_samples,
            'metrics': metrics
        })
    
    async def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server"""
        print(f"Evaluating on {self.validation_generator.n} validation samples...")
        
        # Evaluate in thread pool (quick operation, <1s)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.model.evaluate(self.validation_generator, verbose=0)
        )
        
        loss, accuracy = results[0], results[1]
        
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Send evaluation metrics to server
        await self.send_message({
            'type': 'metrics',
            'client_id': self.client_id,
            'round': self.current_round,
            'num_samples': self.validation_generator.n,
            'metrics': {
                'loss': float(loss),
                'accuracy': float(accuracy)
            }
        })
    
    async def register_with_server(self):
        """Register with the federated learning server"""
        await self.send_message({
            'type': 'register',
            'client_id': self.client_id
        })
        print(f"Client {self.client_id} registration sent to server")


async def main():
    # Setup data generators
    # Detect environment: Docker uses /app prefix, local uses relative path
    if os.path.exists('/app'):
        base_path = '/app/Client/Emotion_Recognition/Dataset'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(project_root, 'Client', 'Emotion_Recognition', 'Dataset')
    
    client_data_dir = os.path.join(base_path, f'client_{CLIENT_ID}')
    print(f"Dataset base path: {base_path}")
    
    if not os.path.exists(client_data_dir):
        print(f"Error: Client data directory not found: {client_data_dir}")
        return
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(client_data_dir, 'train'),
        target_size=(48, 48),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(client_data_dir, 'validation'),
        target_size=(48, 48),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n{'='*70}")
    print(f"Federated Learning Client {CLIENT_ID} with QUIC")
    print(f"Server: {QUIC_HOST}:{QUIC_PORT}")
    print(f"{'='*70}\n")
    
    # Create client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    # Configure QUIC with large stream data limits for model weights
    configuration = QuicConfiguration(
        is_client=True,
        alpn_protocols=["fl"],
        max_stream_data=20 * 1024 * 1024,  # 20 MB per stream
        max_data=50 * 1024 * 1024,  # 50 MB total connection data
        idle_timeout=3600.0,  # 1 hour idle timeout (training can take long)
    )
    
    # Load CA certificate for verification (optional - set verify_mode to False for testing)
    # cert_dir = Path(__file__).parent.parent.parent / "certs"
    # ca_cert = cert_dir / "server-cert.pem"
    # if ca_cert.exists():
    #     configuration.load_verify_locations(str(ca_cert))
    configuration.verify_mode = False  # Disable certificate verification for testing
    
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
    except KeyboardInterrupt:
        print("\nClient shutting down...")
