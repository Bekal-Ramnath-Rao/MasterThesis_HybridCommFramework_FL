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

# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# gRPC Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, train_generator=None, validation_generator=None):
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
        
        # gRPC connection
        self.channel = None
        self.stub = None
        self.running = True
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {self.train_generator.n}")
        print(f"  Validation samples: {self.validation_generator.n}")
        print(f"  Waiting for initial global model from server...")
        
    def connect(self):
        """Connect to gRPC server"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to gRPC server at {GRPC_HOST}:{GRPC_PORT}...")
                # FAIR CONFIG: Set max message size to 128MB (aligned with AMQP default)
                options = [
                    ('grpc.max_send_message_length', 128 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 128 * 1024 * 1024),
                    # FAIR CONFIG: Keepalive settings 600s for very_poor network
                    ('grpc.keepalive_time_ms', 600000),  # 10 minutes
                    ('grpc.keepalive_timeout_ms', 60000),  # 1 minute timeout
                ]
                self.channel = grpc.insecure_channel(f'{GRPC_HOST}:{GRPC_PORT}', options=options)
                self.stub = federated_learning_pb2_grpc.FederatedLearningStub(self.channel)
                
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
        """Main client loop - poll server for instructions"""
        print(f"\nClient {self.client_id} waiting for server to start...\n")
        
        # First, get initial global model from server
        initial_model_received = False
        retry_count = 0
        max_connection_retries = 30
        
        while not initial_model_received and retry_count < max_connection_retries:
            try:
                model_update = self.stub.GetGlobalModel(
                    federated_learning_pb2.ModelRequest(client_id=self.client_id)
                )
                
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
                
                if status.training_complete:
                    print(f"\n{'='*70}")
                    print(f"Client {self.client_id} - Training completed!")
                    print(f"{'='*70}\n")
                    break
                
                if not status.should_train:
                    time.sleep(1)
                    continue
                
                # Check if we need to start a new round
                if status.round > self.current_round and not status.should_evaluate:
                    # Get new global model
                    model_update = self.stub.GetGlobalModel(
                        federated_learning_pb2.ModelRequest(client_id=self.client_id)
                    )
                    
                    if model_update.round == status.round:
                        self.receive_global_model(model_update)
                        self.current_round = status.round
                        
                        # Train on local data
                        print(f"\nClient {self.client_id} starting training for round {self.current_round}...")
                        self.train_local_model()
                
                # Check if we should evaluate
                elif status.should_evaluate and status.round == self.current_round:
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
    
    def receive_global_model(self, model_update):
        """Receive and set global model weights from server"""
        try:
            # Decompress or deserialize weights
            if self.quantizer is not None and model_update.weights:
                try:
                    candidate = pickle.loads(model_update.weights)
                    if isinstance(candidate, dict) and 'quantization_params' in candidate:
                        weights = self.quantizer.decompress(candidate)
                        print(f"Client {self.client_id}: Received and decompressed quantized global model")
                    else:
                        weights = candidate
                except Exception:
                    # Fallback to regular deserialization if unpickle fails
                    weights = pickle.loads(model_update.weights)
            else:
                weights = pickle.loads(model_update.weights)
            
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
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        try:
            batch_size = self.training_config['batch_size']
            epochs = self.training_config['local_epochs']
            # Limit steps per epoch for faster training (configurable via env)
            try:
                steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
                val_steps = int(os.getenv("VAL_STEPS", "25"))
            except Exception:
                steps_per_epoch = 100
                val_steps = 25
            
            # Train the model
            history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.validation_generator,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                verbose=2
            )
            
            # Get updated weights
            updated_weights = self.model.get_weights()
            num_samples = self.train_generator.n
            
            # Compress or serialize weights
            if self.quantizer is not None:
                compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
                stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
                print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Original: {stats['original_size_mb']:.2f}MB, Compressed: {stats['compressed_size_mb']:.2f}MB")
                # Send pickled compressed dict so server can read quantization metadata
                serialized_weights = pickle.dumps(compressed_data)
            else:
                serialized_weights = pickle.dumps(updated_weights)
            
            # Prepare metrics
            metrics = {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
                "val_accuracy": float(history.history["val_accuracy"][-1])
            }
            
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
                    num_samples=num_samples,
                    metrics={
                        'loss': metrics['loss'],
                        'accuracy': metrics['accuracy'],
                        'val_loss': metrics['val_loss'],
                        'val_accuracy': metrics['val_accuracy']
                    }
                )
            )
            
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
            
            # Send metrics to server
            response = self.stub.SendMetrics(
                federated_learning_pb2.EvaluationMetrics(
                    client_id=self.client_id,
                    round=self.current_round,
                    num_samples=num_samples,
                    metrics={
                        'loss': float(loss),
                        'accuracy': float(accuracy)
                    }
                )
            )
            
            if response.success:
                print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"Error in evaluate_model: {e}")
            import traceback
            traceback.print_exc()


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
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_data_gen.flow_from_directory(
        validation_path,
        target_size=(48, 48),
        batch_size=32,
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
