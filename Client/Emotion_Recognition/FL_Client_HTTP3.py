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
from aioquic.quic.events import QuicEvent
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, HeadersReceived, H3Event
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

# HTTP/3 Configuration
HTTP3_HOST = os.getenv("HTTP3_HOST", "fl-server-http3-emotion")
HTTP3_PORT = int(os.getenv("HTTP3_PORT", "4434"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))

# Model initialization timeout (seconds) - longer for poor network conditions
# Default: 300s (5 minutes) for very poor network conditions with large models
MODEL_INIT_TIMEOUT = float(os.getenv("MODEL_INIT_TIMEOUT", "300"))


class FederatedLearningClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = None
        self.stream_id = None
        self._http = None  # H3Connection instance
        self._stream_buffers = {}  # Buffer for incomplete messages
        self._stream_content_lengths = {}  # Track expected content length per stream
    
    def quic_event_received(self, event: QuicEvent):
        """Handle QUIC events and convert to HTTP/3 events"""
        try:
            event_type = type(event).__name__
            # Log important events
            if event_type in ['ConnectionTerminated', 'StreamDataReceived', 'StreamReset']:
                print(f"[HTTP/3] Client QUIC event: {event_type}")
                if hasattr(event, 'stream_id'):
                    print(f"  Stream ID: {event.stream_id}")
                if event_type == 'ConnectionTerminated':
                    if hasattr(event, 'error_code'):
                        print(f"  Error code: {event.error_code}")
                    if hasattr(event, 'reason_phrase'):
                        print(f"  Reason: {event.reason_phrase}")
            
            # Initialize H3 connection on first event
            if self._http is None:
                print(f"[HTTP/3] Client initializing H3Connection")
                self._http = H3Connection(self._quic)
            
            # Convert QUIC events to HTTP/3 events
            if self._http is not None:
                try:
                    h3_events = self._http.handle_event(event)
                    if h3_events:
                        print(f"[HTTP/3] Client generated {len(h3_events)} H3 event(s) from {event_type}")
                    for h3_event in h3_events:
                        self._handle_h3_event(h3_event)
                except Exception as h3_error:
                    print(f"[HTTP/3] Client error converting QUIC to H3: {h3_error}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"[HTTP/3] Client error in quic_event_received: {e}")
            import traceback
            traceback.print_exc()
    
    def _handle_h3_event(self, event: H3Event):
        """Handle HTTP/3 events"""
        if isinstance(event, HeadersReceived):
            try:
                stream_id = event.stream_id
                headers = dict(event.headers)
                status = headers.get(b":status", b"").decode()
                method = headers.get(b":method", b"").decode()
                print(f"[HTTP/3] Client received headers on stream {stream_id}, status: {status}, method: {method}")
                
                # Initialize buffer for this stream
                if stream_id not in self._stream_buffers:
                    self._stream_buffers[stream_id] = b''
                
                # Track content length if present (for server responses)
                if b"content-length" in headers:
                    content_length = int(headers.get(b"content-length", b"0"))
                    self._stream_content_lengths[stream_id] = content_length
                    print(f"[HTTP/3] Client expecting {content_length} bytes on stream {stream_id}")
            except Exception as e:
                print(f"[HTTP/3] Client error handling headers: {e}")
        
        elif isinstance(event, DataReceived):
            try:
                stream_id = event.stream_id
                # Get or create buffer for this stream
                if stream_id not in self._stream_buffers:
                    self._stream_buffers[stream_id] = b''
                
                # Append new data to buffer
                self._stream_buffers[stream_id] += event.data
                buffer_size = len(self._stream_buffers[stream_id])
                if buffer_size % (100 * 1024) < len(event.data):
                    print(f"[DEBUG] Client stream {stream_id}: ~{buffer_size // 1024}KB received")
                
                # Send flow control updates
                self.transmit()
                
                # Check if we've received all expected data (HTTP/3 DataReceived doesn't have end_stream)
                expected_length = self._stream_content_lengths.get(stream_id, 0)
                received_length = len(self._stream_buffers[stream_id])
                
                # Process message when we've received all expected data
                # If content-length was specified, wait for all data; otherwise process when we have data
                if expected_length > 0 and received_length >= expected_length:
                    try:
                        data_str = self._stream_buffers[stream_id].decode('utf-8')
                        message = json.loads(data_str)
                        msg_type = message.get('type', 'unknown')
                        print(f"[DEBUG] Client decoded complete message type '{msg_type}' from stream {stream_id}")
                        
                        # Handle message asynchronously
                        if self.client:
                            asyncio.create_task(self.client.handle_message(message))
                        
                        # Clear buffer and content length tracking
                        self._stream_buffers[stream_id] = b''
                        self._stream_content_lengths.pop(stream_id, None)
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"[HTTP/3] Client error decoding message: {e}")
                        # Only clear buffer if we've received all expected data
                        if expected_length > 0 and received_length >= expected_length:
                            self._stream_buffers[stream_id] = b''
                            self._stream_content_lengths.pop(stream_id, None)
            except Exception as e:
                print(f"[HTTP/3] Client error handling data: {e}")
                import traceback
                traceback.print_exc()
        
        # Note: StreamReset is not available in aioquic.h3.events, 
        # so we handle stream resets via other event types
        # Stream resets are typically handled by the QUIC layer automatically


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
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.has_converged = False
        self.running = True
        self.protocol = None
        self.stream_id = 0
        self.model_ready = asyncio.Event()  # Event to signal model is ready
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {self.train_generator.n}")
        print(f"  Validation samples: {self.validation_generator.n}")
        print(f"  Waiting for initial global model from server...")
    
    def serialize_weights(self, weights):
        """Serialize model weights for HTTP/3 transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from HTTP/3"""
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
        """Send message to server via HTTP/3 stream"""
        if self.protocol:
            msg_type = message.get('type')
            
            # Ensure HTTP connection is initialized
            if self.protocol._http is None:
                self.protocol._http = H3Connection(self.protocol._quic)
            
            # Get next available stream ID
            self.stream_id = self.protocol._quic.get_next_available_stream_id(is_unidirectional=False)
            
            # Prepare JSON payload
            payload = json.dumps(message).encode('utf-8')
            
            # Send HTTP/3 request (must include :authority pseudo-header)
            headers = [
                (b":method", b"POST"),
                (b":path", b"/fl/message"),
                (b":scheme", b"https"),
                (b":authority", HTTP3_HOST.encode('utf-8')),  # Required HTTP/3 pseudo-header
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ]
            print(f"[HTTP/3] Client {self.client_id} sending {msg_type} on stream {self.stream_id}, payload size: {len(payload)} bytes")
            self.protocol._http.send_headers(stream_id=self.stream_id, headers=headers)
            self.protocol._http.send_data(stream_id=self.stream_id, data=payload, end_stream=True)
            self.protocol.transmit()
            print(f"[HTTP/3] Client {self.client_id} transmitted {msg_type} message")
            
            # FAIR FIX: Removed artificial delays (1.5s for large, 0.1s for small messages)
            # HTTP/3 handles flow control automatically, so manual delays are unnecessary
            # This makes HTTP/3 behavior similar to other protocols which don't have artificial delays
            # The transmit() call above is sufficient for immediate transmission
            
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
        print(f"Client {self.client_id}: Received global_model message for round {round_num}")
        
        # Decompress or deserialize weights
        if 'quantized_data' in message and self.quantizer is not None:
            # Deserialize base64+pickle encoded quantized data
            compressed_data = pickle.loads(base64.b64decode(message['quantized_data']))
            weights = self.quantizer.decompress(compressed_data)
            print(f"Client {self.client_id}: Received and decompressed quantized global model")
        elif 'compressed_data' in message and self.quantizer is not None:
            weights = self.quantizer.decompress(message['compressed_data'])
            print(f"Client {self.client_id}: Received and decompressed quantized global model")
        elif 'weights' in message:
            encoded_weights = message['weights']
            weights = self.deserialize_weights(encoded_weights)
            print(f"Client {self.client_id}: Deserialized global model weights ({len(encoded_weights)} bytes)")
        else:
            print(f"Client {self.client_id}: ERROR - No weights found in message!")
            return
        
        # Check if model needs initialization (works for late-joiners too)
        if self.model is None:
            # Initial model from server - create model from server's config
            print(f"Client {self.client_id} received initial global model from server (round {round_num})")
            
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
            print(f"Client {self.client_id} using timeout of {MODEL_INIT_TIMEOUT}s (configured via MODEL_INIT_TIMEOUT env var)")
            try:
                await asyncio.wait_for(self.model_ready.wait(), timeout=MODEL_INIT_TIMEOUT)
                print(f"Client {self.client_id} model ready, proceeding with training")
            except asyncio.TimeoutError:
                print(f"Client {self.client_id} ERROR: Timeout waiting for model initialization after {MODEL_INIT_TIMEOUT}s")
                print(f"Client {self.client_id} TIP: Increase MODEL_INIT_TIMEOUT env var for very poor network conditions")
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
        await asyncio.sleep(0.5)
        self.running = False
        if self.protocol:
            self.protocol.close()
    
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
            weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            weights_key = 'compressed_data'
        else:
            weights_data = self.serialize_weights(updated_weights)
            weights_key = 'weights'
        
        # Send model update to server
        update_message = {
            'type': 'model_update',
            'client_id': self.client_id,
            'round': self.current_round,
            weights_key: weights_data,
            'num_samples': num_samples,
            'metrics': metrics
        }
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            send_start_ts = time.time()
            send_start_cpu = time.perf_counter()
            update_message["diagnostic_send_start_ts"] = send_start_ts
        await self.send_message(update_message)
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            O_send = time.perf_counter() - send_start_cpu
            payload_bytes = len(json.dumps(update_message).encode('utf-8'))
            print(f"FL_DIAG O_send={O_send:.9f} payload_bytes={payload_bytes} send_start_ts={send_start_ts:.9f}")
    
    async def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server"""
        print(f"Evaluating on {self.validation_generator.n} validation samples...")
        
        # Evaluate in thread pool (http3k operation, <1s)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.model.evaluate(self.validation_generator, verbose=0)
        )
        
        loss, accuracy = results[0], results[1]
        self._update_local_convergence(float(loss))
        
        metrics_dict = {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
        if self.has_converged:
            metrics_dict['client_converged'] = 1.0
        
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        await self.send_message({
            'type': 'metrics',
            'client_id': self.client_id,
            'round': self.current_round,
            'num_samples': self.validation_generator.n,
            'metrics': metrics_dict
        })
        if self.has_converged:
            print(f"Client {self.client_id} notifying server of convergence and disconnecting")
            await asyncio.sleep(0.5)
            self.running = False
            if self.protocol:
                self.protocol.close()
    
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

    async def register_with_server(self):
        """Register with the federated learning server"""
        registration_msg = {
            'type': 'register',
            'client_id': self.client_id
        }
        print(f"[HTTP/3] Client {self.client_id} sending registration message...")
        await self.send_message(registration_msg)
        print(f"Client {self.client_id} registration sent to server")
        # Give server time to process registration
        await asyncio.sleep(1)


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
    print(f"Federated Learning Client {CLIENT_ID} with HTTP/3")
    print(f"Server: {HTTP3_HOST}:{HTTP3_PORT}")
    print(f"{'='*70}\n")
    
    # Create client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    # Configure HTTP/3 with large stream data limits for model weights
    # FAIR CONFIG: Aligned with MQTT/AMQP/gRPC/DDS for unbiased comparison
    configuration = QuicConfiguration(
        is_client=True,
        alpn_protocols=H3_ALPN,
        # FAIR CONFIG: Data limits 128MB per stream, 256MB total (aligned with AMQP)
        max_stream_data=128 * 1024 * 1024,  # 128 MB per stream
        max_data=256 * 1024 * 1024,  # 256 MB total connection
        # FAIR CONFIG: Timeout 600s for very_poor network scenarios
        idle_timeout=600.0,  # 10 minutes
        max_datagram_frame_size=65536,  # 64 KB frames
        # Poor network adjustments
        initial_rtt=0.15,  # 150ms (account for 100ms latency + jitter)
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
    print(f"Connecting to HTTP/3 server at {HTTP3_HOST}:{HTTP3_PORT}...")
    try:
        async with connect(
            HTTP3_HOST,
            HTTP3_PORT,
            configuration=configuration,
            create_protocol=create_protocol,
        ) as protocol:
            client.protocol = protocol
            print(f"✓ Connected to HTTP/3 server successfully")
            
            # Register with server
            await client.register_with_server()
            
            print(f"Client {CLIENT_ID} waiting for training commands...")
            print(f"[DEBUG] Client {CLIENT_ID} protocol._http initialized: {protocol._http is not None}")
            print(f"[DEBUG] Client {CLIENT_ID} protocol._quic initialized: {protocol._quic is not None}")
            
            # Keep connection alive until client is stopped
            # Process events by calling transmit periodically to allow QUIC to process incoming data
            while client.running:
                protocol.transmit()  # Process any pending QUIC events
                await asyncio.sleep(0.1)  # Shorter sleep for more responsive event processing
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print(f"Failed to connect to HTTP/3 server at {HTTP3_HOST}:{HTTP3_PORT}")
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
