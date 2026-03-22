import numpy as np
import os
import sys
import logging
import threading
import pickle
import time
import random
import json
import grpc

# GPU Configuration - Must be done BEFORE TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
if _xla_flags:
    sanitized_flags = [f for f in _xla_flags.split() if f != "--xla_gpu_enable_command_buffer="]
    if sanitized_flags:
        os.environ["XLA_FLAGS"] = " ".join(sanitized_flags)
    else:
        os.environ.pop("XLA_FLAGS", None)
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
                    [tf.config.LogicalDeviceConfiguration(memory_limit=int(os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "4000")))]
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
try:
    from pruning_client import ModelPruning, PruningConfig
    PRUNING_AVAILABLE = True
except Exception:
    ModelPruning = None
    PruningConfig = None
    PRUNING_AVAILABLE = False

# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))
# Battery model (shared with unified and MQTT)
_client_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _client_dir not in sys.path:
    sys.path.insert(0, _client_dir)
from battery_model import BatteryModel

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# gRPC Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
# Max message size (4 MB); server chunks larger global models and client reassembles; client chunks large updates
GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))
GRPC_CHUNK_SIZE = GRPC_MAX_MESSAGE_BYTES - 4096  # leave room for proto/metadata
# INT_MAX ms (~24 days) = effectively disable keepalive (gRPC C-core: 0 can mean "use default")
GRPC_KEEPALIVE_DISABLED_MS = 2147483647
# Retries for SendModelUpdate on UNAVAILABLE/DEADLINE_EXCEEDED (poor/very-poor network)
GRPC_SEND_UPDATE_MAX_RETRIES = int(os.getenv("GRPC_SEND_UPDATE_MAX_RETRIES", "4"))
# Timeout for SendModelUpdate in seconds. Default 3600 (1h) for very_poor. 0/none/inf = no deadline (wait indefinitely).
_env_timeout = os.getenv("GRPC_SEND_UPDATE_TIMEOUT_SEC", "3600").strip().lower()
if _env_timeout in ("0", "none", "inf", "infinity"):
    GRPC_SEND_UPDATE_TIMEOUT = None  # no deadline
else:
    try:
        GRPC_SEND_UPDATE_TIMEOUT = int(_env_timeout)
    except ValueError:
        GRPC_SEND_UPDATE_TIMEOUT = 3600  # 1 hour default
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
DEFAULT_DATA_BATCH_SIZE = int(os.getenv("DEFAULT_DATA_BATCH_SIZE", "16"))

# Controls whether this client should signal/exit on local convergence.
# When false, clients keep training until the server indicates completion.
STOP_ON_CLIENT_CONVERGENCE = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").lower() in ("1", "true", "yes")


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, train_generator=None, validation_generator=None):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        
        self.battery_model = BatteryModel(protocol="grpc")
        self._last_round_time_sec = 0.0
        # Initialize quantization compression (default: disabled unless explicitly enabled)
        uq_env = os.getenv("USE_QUANTIZATION", "false")
        use_quantization = uq_env.lower() in ("true", "1", "yes", "y")
        if use_quantization:
            self.quantizer = Quantization(QuantizationConfig())
            print(f"Client {self.client_id}: Quantization enabled")
        else:
            self.quantizer = None
            print(f"Client {self.client_id}: Quantization disabled")

        # Initialize pruning (default: disabled unless explicitly enabled)
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
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.current_round = 0
        self.training_config = {"batch_size": 16, "local_epochs": 20}
        
        # gRPC connection
        self.channel = None
        self.stub = None
        self.running = True
        self.last_training_round = -1
        self._training_lock = threading.Lock()
        self._training_thread = None
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.has_converged = False
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {self.train_generator.n}")
        print(f"  Validation samples: {self.validation_generator.n}")
        print(f"  Waiting for initial global model from server...")

    def _apply_generator_batch_size(self, batch_size):
        """Sync DirectoryIterator batch size with training config."""
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            return
        if batch_size <= 0:
            return

        changed = False
        for gen_name in ("train_generator", "validation_generator"):
            generator = getattr(self, gen_name, None)
            if generator is None:
                continue
            current = getattr(generator, "batch_size", None)
            if current != batch_size:
                setattr(generator, "batch_size", batch_size)
                changed = True
        if changed:
            print(f"Client {self.client_id} synchronized generator batch_size to {batch_size}")
        
    def connect(self):
        """Connect to gRPC server"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to gRPC server at {GRPC_HOST}:{GRPC_PORT}...")
                # Max message size must allow receiving global model (~9+ MB when quantization disabled)
                # Keepalive effectively disabled (INT_MAX) so poor/very-poor networks never hit ping_timeout
                options = [
                    ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
                    ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
                    ('grpc.keepalive_time_ms', GRPC_KEEPALIVE_DISABLED_MS),
                    ('grpc.keepalive_timeout_ms', GRPC_KEEPALIVE_DISABLED_MS),
                ]
                self.channel = grpc.insecure_channel(f'{GRPC_HOST}:{GRPC_PORT}', options=options)
                self.stub = federated_learning_pb2_grpc.FederatedLearningStub(self.channel)
                self._grpc_options = options  # for reconnect on UNAVAILABLE
                self._grpc_target = f'{GRPC_HOST}:{GRPC_PORT}'
                # Test connection by registering
                response = self.stub.RegisterClient(
                    federated_learning_pb2.ClientRegistration(client_id=self.client_id)
                )
                
                if response.success:
                    print(f"Client {self.client_id} connected to gRPC server")
                    print(f"Client {self.client_id} registration: {response.message}")
                    
                    # Get training configuration
                    config = self.stub.GetTrainingConfig(
                        federated_learning_pb2.ConfigRequest(client_id=self.client_id)
                    )
                    self.training_config = {
                        "batch_size": config.batch_size,
                        "local_epochs": config.local_epochs
                    }
                    self._apply_generator_batch_size(self.training_config["batch_size"])
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

    def _launch_training_for_round(self, round_num):
        """Start local training in a background thread to keep gRPC polling responsive."""
        with self._training_lock:
            if self._training_thread is not None and self._training_thread.is_alive():
                print(f"Client {self.client_id} training already in progress; ignoring start for round {round_num}")
                return
            if self.last_training_round == round_num:
                print(f"Client {self.client_id} ignoring duplicate start training for round {round_num}")
                return

            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")

            self._training_thread = threading.Thread(
                target=self._train_local_model_safe,
                args=(round_num,),
                daemon=True,
            )
            self._training_thread.start()

    def _train_local_model_safe(self, round_num):
        try:
            self.train_local_model()
        except Exception as e:
            print(f"Client {self.client_id} training thread error on round {round_num}: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """Main client loop - poll server for instructions"""
        print(f"\nClient {self.client_id} waiting for server to start...\n")
        
        # First, get initial global model from server
        initial_model_received = False
        retry_count = 0
        max_connection_retries = 30
        
        while not initial_model_received and retry_count < max_connection_retries:
            try:
                model_update = self._get_global_model_chunked(round_num=0)
                
                if model_update.available and model_update.weights and model_update.model_config:
                    # Initial model from server
                    self.receive_global_model(model_update)
                    initial_model_received = True
                    print(f"Client {self.client_id} received initial model from server\n")
                else:
                    retry_count += 1
                    #f retry_count % 5 == 0:
                        #print(f"Still waiting for initial model... ({retry_count}/{max_connection_retries})")
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
                # Check training status
                status = self.stub.CheckTrainingStatus(
                    federated_learning_pb2.StatusRequest(client_id=self.client_id)
                )
                
                if status.is_complete:
                    print(f"\n{'='*70}")
                    print(f"Client {self.client_id} - Training completed!")
                    print(f"{'='*70}\n")
                    break
                
                if not status.should_train:
                    time.sleep(1)
                    continue
                
                # Check if we need to start a new round
                if status.current_round > self.current_round and not status.should_evaluate:
                    # Get new global model (chunked when > 4 MB)
                    model_update = self._get_global_model_chunked(round_num=status.current_round)
                    
                    if model_update.round == status.current_round:
                        self.receive_global_model(model_update)
                        self._launch_training_for_round(status.current_round)
                
                # Check if we should evaluate
                elif status.should_evaluate and status.current_round == self.current_round:
                    print(f"Client {self.client_id} evaluating model for round {self.current_round}...")
                    self.evaluate_model()
                    # Mark that we've evaluated this round by incrementing our counter
                    # (server will move to next round after all clients evaluate)
                
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
    
    def _get_global_model_chunked(self, round_num=None):
        """Fetch global model from server, reassembling chunks when total_chunks > 1 (4 MB limit)."""
        round_val = self.current_round if round_num is None else round_num
        req = federated_learning_pb2.ModelRequest(client_id=self.client_id, round=round_val, chunk_index=0)
        first = self.stub.GetGlobalModel(req)
        if not first.available or not first.weights:
            return first
        total_chunks = getattr(first, 'total_chunks', 1) or 1
        if total_chunks <= 1:
            return first
        chunks = [first.weights]
        model_config = first.model_config
        for idx in range(1, total_chunks):
            req = federated_learning_pb2.ModelRequest(client_id=self.client_id, round=round_val, chunk_index=idx)
            part = self.stub.GetGlobalModel(req)
            if part.weights:
                chunks.append(part.weights)
        assembled = b''.join(chunks)
        return federated_learning_pb2.GlobalModel(
            round=first.round,
            weights=assembled,
            available=True,
            model_config=model_config,
            chunk_index=0,
            total_chunks=1
        )
    
    def receive_global_model(self, model_update):
        """Receive and set global model weights from server"""
        try:
            # Decompress/deserialize weights.
            # If both pruning and quantization are enabled, server should send quantized payload that already reflects pruning.
            candidate = pickle.loads(model_update.weights) if model_update.weights else None
            if isinstance(candidate, dict) and 'quantization_params' in candidate:
                weights = self.quantizer.as_training_weights(candidate) if self.quantizer is not None else []
                print(f"Client {self.client_id}: Received quantized global model (kept quantized)")
            elif isinstance(candidate, dict) and 'pruned_data' in candidate:
                if self.pruner is not None and candidate['pruned_data']:
                    weights = self.pruner.decompress_pruned_weights(candidate['pruned_data'])
                    print(f"Client {self.client_id}: Received and decompressed pruned global model")
                else:
                    weights = []
            else:
                weights = candidate
            
            if model_update.round == 0 and model_update.model_config:
                # Initial model - build from config
                model_config = json.loads(model_update.model_config)
                
                # Build CNN model from server's architecture definition
                self.model = Sequential()
                self.model.add(Input(shape=tuple(model_config['input_shape'])))
                
                for layer_config in model_config['layers']:
                    if layer_config['type'] == 'Conv2D':
                        self.model.add(Conv2D(
                            filters=layer_config['filters'],
                            kernel_size=tuple(layer_config['kernel_size']),
                            activation=layer_config.get('activation')
                        ))
                    elif layer_config['type'] == 'MaxPooling2D':
                        self.model.add(MaxPooling2D(pool_size=tuple(layer_config['pool_size'])))
                    elif layer_config['type'] == 'Dropout':
                        self.model.add(Dropout(layer_config['rate']))
                    elif layer_config['type'] == 'Flatten':
                        self.model.add(Flatten())
                    elif layer_config['type'] == 'Dense':
                        self.model.add(Dense(
                            units=layer_config['units'],
                            activation=layer_config.get('activation')
                        ))
                
                # Compile model
                self.model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=0.0001),
                    metrics=['accuracy']
                )
                
                print(f"Client {self.client_id} built CNN model from server configuration")
                print(f"  Input shape: {model_config['input_shape']}")
                print(f"  Output classes: {model_config['num_classes']}")
            
            # Set weights
            self.model.set_weights(weights)
            
            if model_update.round == 0:
                print(f"Client {self.client_id} initialized with global model")
            else:
                print(f"Client {self.client_id} updated with global model for round {model_update.round}")
                
        except Exception as e:
            print(f"Error receiving global model: {e}")
            import traceback
            traceback.print_exc()

    def _reconnect_grpc_channel(self):
        """Create a new gRPC channel and stub (e.g. after GOAWAY/ping_timeout)."""
        if self.channel:
            try:
                self.channel.close()
            except Exception:
                pass
            self.channel = None
        opts = getattr(self, '_grpc_options', None)
        target = getattr(self, '_grpc_target', f'{GRPC_HOST}:{GRPC_PORT}')
        if not opts:
            opts = [
                ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
                ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
                ('grpc.keepalive_time_ms', GRPC_KEEPALIVE_DISABLED_MS),
                ('grpc.keepalive_timeout_ms', GRPC_KEEPALIVE_DISABLED_MS),
            ]
        self.channel = grpc.insecure_channel(target, options=opts)
        self.stub = federated_learning_pb2_grpc.FederatedLearningStub(self.channel)
        print(f"Client {self.client_id} reconnected gRPC channel to {target}")
    
    def _send_model_update_with_retry(self, request, send_start_cpu=None):
        """Send model update; on UNAVAILABLE or DEADLINE_EXCEEDED reconnect and retry up to GRPC_SEND_UPDATE_MAX_RETRIES with backoff."""
        last_error = None
        timeout = GRPC_SEND_UPDATE_TIMEOUT  # None = no deadline (for very_poor)
        for attempt in range(GRPC_SEND_UPDATE_MAX_RETRIES):
            try:
                return self.stub.SendModelUpdate(request, timeout=timeout)
            except grpc.RpcError as e:
                last_error = e
                retryable = e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED)
                if not retryable or attempt == GRPC_SEND_UPDATE_MAX_RETRIES - 1:
                    raise
                backoff = (2 ** attempt) * 2  # 2, 4, 8 seconds
                reason = "ping_timeout" if e.code() == grpc.StatusCode.UNAVAILABLE else "deadline exceeded"
                print(f"Client {self.client_id} got {reason}, reconnecting and retrying in {backoff}s (attempt {attempt + 1}/{GRPC_SEND_UPDATE_MAX_RETRIES})...")
                time.sleep(backoff)
                self._reconnect_grpc_channel()
        raise last_error
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        try:
            batch_size = self.training_config['batch_size']
            self._apply_generator_batch_size(batch_size)
            epochs = self.training_config['local_epochs']
            # Limit steps per epoch for faster training (configurable via env)
            try:
                steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
                val_steps = int(os.getenv("VAL_STEPS", "25"))
            except Exception:
                steps_per_epoch = 100
                val_steps = 25

            training_start = time.time()
            # Train the model
            history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.validation_generator,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                verbose=2
            )
            training_time = time.time() - training_start

            # Get updated weights
            updated_weights = self.model.get_weights()
            num_samples = self.train_generator.n

            # Flow: Training -> Pruning -> Quantization -> Send
            if self.pruner is not None:
                updated_weights = self.pruner.prune_weights(updated_weights, step=self.current_round)
            
            # Serialize payload for gRPC:
            # - if quantization enabled -> send pickled quantized dict (may include pruning applied above)
            # - else if pruning enabled -> send pickled {"pruned_data": <bytes>} sparse payload
            # - else -> send pickled raw weights list
            if self.quantizer is not None:
                compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
                stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
                print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Original: {stats['original_size_mb']:.2f}MB, Compressed: {stats['compressed_size_mb']:.2f}MB")
                # Send pickled compressed dict so server can read quantization metadata
                serialized_weights = pickle.dumps(compressed_data)
            elif self.pruner is not None:
                pruned_bytes, _ = self.pruner.compress_pruned_weights(updated_weights)
                serialized_weights = pickle.dumps({"pruned_data": pruned_bytes})
            else:
                serialized_weights = pickle.dumps(updated_weights)
            
            # Prepare metrics
            metrics = {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "val_accuracy": float(history.history["val_accuracy"][-1])
            }
            
            send_start_ts = time.time()
            send_start_cpu = time.perf_counter() if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1" else None
            comm_start = time.time()
            # FAIR FIX: no artificial pre-send delay; align behavior with unified and other protocols.
            
            # Build metrics for server (include diagnostic_send_start_ts for pipeline T_actual)
            metrics_for_server = {
                'loss': metrics['loss'],
                'accuracy': metrics['accuracy'],
                'val_loss': metrics['val_loss'],
                'val_accuracy': metrics['val_accuracy']
            }
            if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
                metrics_for_server['diagnostic_send_start_ts'] = send_start_ts
            # Chunk payload if over gRPC limit (4 MB)
            payload_size = len(serialized_weights)
            if payload_size > GRPC_CHUNK_SIZE:
                chunks = []
                for start in range(0, payload_size, GRPC_CHUNK_SIZE):
                    chunks.append(serialized_weights[start:start + GRPC_CHUNK_SIZE])
                total_chunks = len(chunks)
                for i, chunk_data in enumerate(chunks):
                    req = federated_learning_pb2.ModelUpdate(
                        client_id=self.client_id,
                        round=self.current_round,
                        weights=chunk_data,
                        num_samples=num_samples if i == 0 else 0,
                        metrics=metrics_for_server if i == 0 else {},
                        chunk_index=i,
                        total_chunks=total_chunks
                    )
                    response = self._send_model_update_with_retry(req, send_start_cpu if i == 0 else None)
                    if not response.success:
                        raise RuntimeError(f"Chunk {i + 1}/{total_chunks} failed: {response.message}")
                print(f"Client {self.client_id} sent update in {total_chunks} chunks ({payload_size} bytes)")
            else:
                request = federated_learning_pb2.ModelUpdate(
                    client_id=self.client_id,
                    round=self.current_round,
                    weights=serialized_weights,
                    num_samples=num_samples,
                    metrics=metrics_for_server
                )
                response = self._send_model_update_with_retry(request, send_start_cpu)
            
            if send_start_cpu is not None:
                O_send = time.perf_counter() - send_start_cpu
                print(f"FL_DIAG O_send={O_send:.9f} payload_bytes={payload_size} send_start_ts={send_start_ts:.9f}")
            
            communication_time = time.time() - comm_start
            bytes_sent = payload_size
            self.battery_model.update(bytes_sent, 0, training_time, communication_time)
            self._last_round_time_sec = training_time + communication_time

            if response.success:
                print(f"Client {self.client_id} successfully sent update for round {self.current_round}")
                print(f"Training metrics - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            else:
                print(f"Failed to send update: {response.message}")
                
        except Exception as e:
            print(f"Error in train_local_model: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server"""
        try:
            loss, accuracy = self.model.evaluate(
                self.validation_generator,
                verbose=0
            )
            
            num_samples = self.validation_generator.n
            loss_f = float(loss)
            # Signal convergence in same message so server can proceed with remaining clients
            client_converged = 1.0 if (STOP_ON_CLIENT_CONVERGENCE and self._would_converge_after_eval(loss_f)) else 0.0
            
            # Send metrics to server (include battery, round time, and convergence flag)
            response = self.stub.SendMetrics(
                federated_learning_pb2.Metrics(
                    client_id=self.client_id,
                    round=self.current_round,
                    num_samples=num_samples,
                    loss=loss_f,
                    accuracy=float(accuracy),
                    battery_soc=float(self.battery_model.battery_soc),
                    round_time_sec=float(self._last_round_time_sec),
                    client_converged=client_converged,
                )
            )
            
            if response.success:
                print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
                self._update_local_convergence(loss_f)
            
        except Exception as e:
            print(f"Error in evaluate_model: {e}")
            import traceback
            traceback.print_exc()

    def _would_converge_after_eval(self, loss: float) -> bool:
        """Return True if this evaluation loss would trigger local convergence (so server can be signalled in same SendMetrics)."""
        if self.current_round < MIN_ROUNDS or self.has_converged:
            return False
        if self.best_loss - loss > CONVERGENCE_THRESHOLD:
            return False
        return (self.rounds_without_improvement + 1) >= CONVERGENCE_PATIENCE

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
            if STOP_ON_CLIENT_CONVERGENCE:
                self._notify_convergence_and_disconnect()

    def _notify_convergence_and_disconnect(self):
        """Notify server and stop this client."""
        try:
            response = self.stub.SendModelUpdate(
                federated_learning_pb2.ModelUpdate(
                    client_id=self.client_id,
                    round=self.current_round,
                    weights=b"",
                    num_samples=0,
                    metrics={"client_converged": 1.0}
                )
            )
            if response.success:
                print(f"Client {self.client_id} convergence notification acknowledged by server")
            else:
                print(f"Client {self.client_id} convergence notification failed: {response.message}")
        except Exception as e:
            print(f"Client {self.client_id} failed to notify convergence: {e}")
        finally:
            self.running = False


def load_data(client_id):
    """Load emotion recognition data for this client"""
    # Detect environment: Docker uses /app prefix, local uses relative path
    if os.path.exists('/app'):
        base_path = '/app/Client/Emotion_Recognition/Dataset'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(project_root, 'Client', 'Emotion_Recognition', 'Dataset')
    
    train_path = os.path.join(base_path, f'client_{client_id}', 'train')
    validation_path = os.path.join(base_path, f'client_{client_id}', 'validation')
    print(f"Dataset base path: {base_path}")
    
    # Initialize image data generator with rescaling
    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    # Load training and validation data
    train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=DEFAULT_DATA_BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_data_gen.flow_from_directory(
        validation_path,
        target_size=(48, 48),
        batch_size=DEFAULT_DATA_BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator


if __name__ == "__main__":
    # Load data
    print(f"Loading dataset for client {CLIENT_ID}...")
    train_generator, validation_generator = load_data(CLIENT_ID)
    print(f"Dataset loaded\n")
    
    # Create client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Client {CLIENT_ID}")
    print(f"Server: {GRPC_HOST}:{GRPC_PORT}")
    print(f"{'='*60}\n")
    
    # Connect and run
    try:
        if client.connect():
            client.run()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} interrupted by user")
    except Exception as e:
        print(f"\nClient {CLIENT_ID} error: {e}")
        import traceback
        traceback.print_exc()
