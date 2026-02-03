import numpy as np
import os
import sys
import logging
import json
import pickle
import base64
import time
import random
import pika

# Detect Docker environment and set project root accordingly
if os.path.exists('/app'):
    project_root = '/app'
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from packet_logger import log_received_packet, log_sent_packet

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

# AMQP Configuration
AMQP_HOST = os.getenv("AMQP_HOST", "localhost")
AMQP_PORT = int(os.getenv("AMQP_PORT", "5672"))
AMQP_USER = os.getenv("AMQP_USER", "guest")
AMQP_PASSWORD = os.getenv("AMQP_PASSWORD", "guest")
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))

# AMQP Exchanges and Queues
EXCHANGE_BROADCAST = "fl_broadcast"
EXCHANGE_CLIENT_UPDATES = "fl_client_updates"


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, train_generator=None, validation_generator=None):
        self.client_id = client_id
        self.num_clients = num_clients
        self.current_round = 0
        # Default batch size adjusted for separate GPUs
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        # Deduplication tracking
        self.last_global_round = -1
        self.last_training_round = -1
        self.evaluated_rounds = set()
        
        # Store data generators
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        
        # Model will be initialized from server config
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
        
        # AMQP connection
        self.connection = None
        self.channel = None
        self.consuming = False
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {self.train_generator.n}")
        print(f"  Validation samples: {self.validation_generator.n}")
        print(f"  Waiting for initial global model from server...")
    
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
                
                # Create client-specific queue names to avoid round-robin distribution
                queue_broadcast = f"fl.client.{self.client_id}.broadcast"
                
                # Declare exclusive queue for this client to receive all broadcasts
                result = self.channel.queue_declare(queue=queue_broadcast, durable=False, exclusive=True, auto_delete=True)
                
                # Bind the client's queue to the fanout exchange
                self.channel.queue_bind(exchange=EXCHANGE_BROADCAST, queue=queue_broadcast)
                
                # Set up consumer for the broadcast queue
                self.channel.basic_consume(queue=queue_broadcast, on_message_callback=self.on_broadcast_message, auto_ack=True)
                
                print(f"Client {self.client_id} connected to RabbitMQ broker")
                
                # Send registration message
                self.send_registration()
                
                return True
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to RabbitMQ broker after {max_retries} attempts.")
                    print(f"\nPlease ensure:")
                    print(f"  1. RabbitMQ broker is running")
                    print(f"  2. Broker address is correct: {AMQP_HOST}:{AMQP_PORT}")
                    print(f"  3. Credentials are correct: {AMQP_USER}")
                    raise
    
    def reconnect(self):
        """Reconnect to RabbitMQ broker after connection loss"""
        print(f"Client {self.client_id} attempting to reconnect...")
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                if self.connection and not self.connection.is_closed:
                    try:
                        self.connection.close()
                    except:
                        pass
                
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
                
                # Redeclare exchanges
                self.channel.exchange_declare(exchange=EXCHANGE_BROADCAST, exchange_type='fanout', durable=True)
                self.channel.exchange_declare(exchange=EXCHANGE_CLIENT_UPDATES, exchange_type='direct', durable=True)
                
                print(f"Client {self.client_id} reconnected successfully")
                return True
                
            except Exception as e:
                print(f"Reconnection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        return False
    
    def publish_with_retry(self, exchange, routing_key, body, properties=None, max_retries=3):
        """Publish message with automatic retry and reconnection"""
        for attempt in range(max_retries):
            try:
                if properties is None:
                    properties = pika.BasicProperties(delivery_mode=2)
                    
                self.channel.basic_publish(
                    exchange=exchange,
                    routing_key=routing_key,
                    body=body,
                    properties=properties
                )

                log_sent_packet(
                    packet_size=len(body),
                    peer=exchange,
                    protocol="AMQP",
                    round=None,
                    extra_info=f"Published to {routing_key}"
                )
                
                return True
                
            except (pika.exceptions.StreamLostError, 
                    pika.exceptions.ConnectionWrongStateError,
                    pika.exceptions.AMQPConnectionError) as e:
                print(f"Client {self.client_id} publish failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    if self.reconnect():
                        print(f"Client {self.client_id} retrying publish...")
                        time.sleep(0.5)
                    else:
                        print(f"Client {self.client_id} reconnection failed")
                        return False
                else:
                    print(f"Client {self.client_id} failed to publish after {max_retries} attempts")
                    return False
        
        return False
    
    def send_registration(self):
        """Send registration message to server"""
        registration = {"client_id": self.client_id}
        if self.publish_with_retry(
            exchange=EXCHANGE_CLIENT_UPDATES,
            routing_key='client.register',
            body=json.dumps(registration),
            properties=pika.BasicProperties(delivery_mode=2)
        ):
            log_sent_packet(
                packet_size=len(json.dumps(registration)),
                peer=EXCHANGE_CLIENT_UPDATES,
                protocol="AMQP",
                round=None,
                extra_info="Client registration"
            )
            print(f"Client {self.client_id} registration sent")
        else:
            print(f"Client {self.client_id} ERROR: Failed to send registration")
    
    def on_broadcast_message(self, ch, method, properties, body):
        """Unified handler for all broadcast messages - routes based on message_type"""
        log_received_packet(
            packet_size=len(body),
            peer=EXCHANGE_BROADCAST,
            protocol="AMQP",
            round=None,
            extra_info="Broadcast message"
        )
        try:
            data = json.loads(body.decode())
            message_type = data.get('message_type')
            
            if message_type == 'global_model':
                self.on_global_model(ch, method, properties, body)
            elif message_type == 'training_config':
                self.on_training_config(ch, method, properties, body)
            elif message_type == 'start_training':
                self.on_start_training(ch, method, properties, body)
            elif message_type == 'start_evaluation':
                self.on_start_evaluation(ch, method, properties, body)
            elif message_type == 'training_complete':
                self.on_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling broadcast: {e}")
    
    def on_training_complete(self):
        """Handle training complete signal from server"""
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nDisconnecting from server...")
        self.channel.stop_consuming()
        self.connection.close()
        import sys
        sys.exit(0)
    
    def on_global_model(self, ch, method, properties, body):
        """Callback for receiving global model"""
        try:
            data = json.loads(body.decode())
            
            # Check message type
            if data.get('message_type') != 'global_model':
                return
            
            round_num = data['round']
            # Check if weights are quantized
            if 'quantized_data' in data and self.quantizer is not None:
                compressed_data = data['quantized_data']
                # If server sent serialized base64 string, decode and unpickle
                if isinstance(compressed_data, str):
                    try:
                        compressed_data = pickle.loads(base64.b64decode(compressed_data.encode('utf-8')))
                    except Exception as e:
                        print(f"Client {self.client_id} error decoding quantized_data: {e}")
                weights = self.quantizer.decompress(compressed_data)
                if round_num > 0:
                    print(f"Client {self.client_id}: Received and decompressed quantized global model")
            else:
                encoded_weights = data['weights']
                weights = self.deserialize_weights(encoded_weights)
            
            if round_num == 0:
                # Initial model from server - create model from server's config
                print(f"Client {self.client_id} received initial global model from server")
                
                model_config = data.get('model_config')
                if model_config:
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
                else:
                    raise ValueError("No model configuration received from server!")
                
                # Set the initial weights from server
                self.model.set_weights(weights)
                print(f"Client {self.client_id} model initialized with server weights")
                self.current_round = 0
            else:
                # Updated model after aggregation
                self.model.set_weights(weights)
                self.current_round = round_num
                print(f"Client {self.client_id} received global model for round {round_num}")
        except Exception as e:
            print(f"Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_training_config(self, ch, method, properties, body):
        """Callback for receiving training config"""
        try:
            data = json.loads(body.decode())
            
            # Check message type
            if data.get('message_type') != 'training_config':
                return
            
            self.training_config = data['config']
            print(f"Client {self.client_id} updated config: {self.training_config}")
        except Exception as e:
            print(f"Client {self.client_id} error handling config: {e}")
    
    def on_start_training(self, ch, method, properties, body):
        """Callback for starting training"""
        try:
            data = json.loads(body.decode())
            
            # Check message type
            if data.get('message_type') != 'start_training':
                return
            
            round_num = data['round']
            
            # Check if we're ready for this round (should have received global model first)
            if self.current_round == 0 and round_num == 1:
                if self.model is None:
                    print(f"Client {self.client_id} received start_training but model not initialized yet; waiting for global model.")
                    return
                # First training round with initial global model
                self.current_round = round_num
                print(f"\nClient {self.client_id} starting training for round {round_num} with initial global model...")
                self.train_local_model()
            elif round_num == self.current_round:
                if self.model is None:
                    print(f"Client {self.client_id} received start_training for round {round_num} but model not initialized; ignoring signal.")
                    return
                # Subsequent rounds
                print(f"\nClient {self.client_id} starting training for round {round_num}...")
                self.train_local_model()
            else:
                print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
        except Exception as e:
            print(f"Client {self.client_id} error starting training: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_evaluation(self, ch, method, properties, body):
        """Callback for starting evaluation"""
        try:
            data = json.loads(body.decode())
            
            # Check message type
            if data.get('message_type') != 'start_evaluation':
                return
            
            round_num = data['round']
            
            if round_num == self.current_round:
                print(f"Client {self.client_id} starting evaluation for round {round_num}...")
                self.evaluate_model()
                # After evaluation, prepare for next round
                self.current_round = round_num + 1
                print(f"Client {self.client_id} ready for next round {self.current_round}")
            else:
                print(f"Client {self.client_id} skipping evaluation signal for round {round_num} (current: {self.current_round})")
        except Exception as e:
            print(f"Client {self.client_id} error starting evaluation: {e}")
    
    def train_local_model(self):
        """Train model on local data with GPU optimization and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        # Limit steps per epoch for faster training (configurable via env)
        try:
            steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
            val_steps = int(os.getenv("VAL_STEPS", "25"))
        except Exception:
            steps_per_epoch = 100
            val_steps = 25
        
        # Use GPU device context for training
        with tf.device('/GPU:0'):
            # Train the model with GPU acceleration
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
        
        # Prepare training metrics
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
            
            # Serialize compressed data to JSON-safe base64 string
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "compressed_data": serialized,
                "num_samples": num_samples,
                "metrics": metrics
            }
        else:
            # Send model update without compression
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "weights": self.serialize_weights(updated_weights),
            "num_samples": num_samples,
            "metrics": metrics
        }
        
        # Introduce random delay before sending model update
        delay = random.uniform(0.5, 3.0)  # Random delay between 0.5 and 3.0 seconds
        print(f"Client {self.client_id} waiting {delay:.2f} seconds before sending update...")
        time.sleep(delay)
        
        # Publish with retry logic
        if self.publish_with_retry(
            exchange=EXCHANGE_CLIENT_UPDATES,
            routing_key='client.update',
            body=json.dumps(update_message),
            properties=pika.BasicProperties(delivery_mode=2)
        ):
            log_sent_packet(
                packet_size=len(json.dumps(update_message)),
                peer=EXCHANGE_CLIENT_UPDATES,
                protocol="AMQP",
                round=self.current_round,
                extra_info="Model update"
            )
            print(f"Client {self.client_id} sent model update for round {self.current_round}")
        else:
            print(f"Client {self.client_id} ERROR: Failed to send model update for round {self.current_round}")
            return
        print(f"Training metrics - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    
    def evaluate_model(self):
        """Evaluate model on validation data with GPU acceleration and send metrics to server"""
        # Use GPU device context for evaluation
        with tf.device('/GPU:0'):
            loss, accuracy = self.model.evaluate(
                self.validation_generator, 
                verbose=0
            )
        
        num_samples = self.validation_generator.n
        
        metrics_message = {
            "client_id": self.client_id,
            "round": self.current_round,
            "num_samples": num_samples,
            "metrics": {
                "loss": float(loss),
                "accuracy": float(accuracy)
            }
        }
        
        if self.publish_with_retry(
            exchange=EXCHANGE_CLIENT_UPDATES,
            routing_key='client.metrics',
            body=json.dumps(metrics_message),
            properties=pika.BasicProperties(delivery_mode=2)
        ):
            print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            log_sent_packet(
                packet_size=len(json.dumps(metrics_message)),
                peer=EXCHANGE_CLIENT_UPDATES,
                protocol="AMQP",
                round=self.current_round,
                extra_info="Evaluation metrics"
            )
        else:
            print(f"Client {self.client_id} ERROR: Failed to send evaluation metrics")
    
    def start(self):
        """Start consuming messages"""
        print(f"\nClient {self.client_id} waiting for messages...")
        self.consuming = True
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print(f"\nClient {self.client_id} stopping...")
            self.stop()
    
    def stop(self):
        """Stop consuming and close connection"""
        if self.consuming:
            self.channel.stop_consuming()
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        print(f"Client {self.client_id} disconnected")


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
    print(f"Broker: {AMQP_HOST}:{AMQP_PORT}")
    print(f"{'='*60}\n")
    
    try:
        client.connect()
        client.start()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} shutting down...")
        client.stop()
