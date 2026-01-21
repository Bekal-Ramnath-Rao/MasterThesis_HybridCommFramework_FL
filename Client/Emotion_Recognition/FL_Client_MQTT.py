import numpy as np
import pandas as pd
import math
import os
import sys
import logging
import json
import pickle
import base64
import time
import random
import paho.mqtt.client as mqtt

# GPU Configuration - Must be done BEFORE TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Get GPU device ID from environment variable (set by docker for multi-GPU isolation)
gpu_device = os.environ.get("GPU_DEVICE_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device  # Isolate to specific GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow gradual GPU memory growth
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # GPU thread mode

# Disable Grappler layout optimizer to avoid NCHW transpose errors in logs
os.environ["TF_ENABLE_LAYOUT_OPTIMIZER"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler

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
        # Lower limit for TensorFlow 2.20+ XLA command buffers
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

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")  # MQTT broker address
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))  # MQTT broker port
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))  # Can be set via environment variable
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"
TOPIC_TRAINING_COMPLETE = "fl/training_complete"

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
        
        # Initialize MQTT client with increased message size limit
        # Use MQTTv5 for better large message handling
        self.mqtt_client = mqtt.Client(client_id=f"fl_client_{client_id}", protocol=mqtt.MQTTv311)
        self.mqtt_client.max_inflight_messages_set(20)
        self.mqtt_client.max_queued_messages_set(0)  # Unlimited queue
        # Set max packet size to 20MB (20 * 1024 * 1024)
        self.mqtt_client._max_packet_size = 20 * 1024 * 1024
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
    
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
            print(f"Client {self.client_id} connected to MQTT broker")
            # Subscribe to topics - use QoS 0 for large model messages
            result1, mid1 = self.mqtt_client.subscribe(TOPIC_GLOBAL_MODEL, qos=1)
            print(f"  Subscribed to {TOPIC_GLOBAL_MODEL} (QoS 1) - Result: {result1}")
            
            result2, mid2 = self.mqtt_client.subscribe(TOPIC_TRAINING_CONFIG, qos=1)
            print(f"  Subscribed to {TOPIC_TRAINING_CONFIG} (QoS 1) - Result: {result2}")
            
            result3, mid3 = self.mqtt_client.subscribe(TOPIC_START_TRAINING, qos=1)
            print(f"  Subscribed to {TOPIC_START_TRAINING} (QoS 1) - Result: {result3}")
            
            result4, mid4 = self.mqtt_client.subscribe(TOPIC_START_EVALUATION, qos=1)
            print(f"  Subscribed to {TOPIC_START_EVALUATION} (QoS 1) - Result: {result4}")
            
            result5, mid5 = self.mqtt_client.subscribe(TOPIC_TRAINING_COMPLETE, qos=1)
            print(f"  Subscribed to {TOPIC_TRAINING_COMPLETE} (QoS 1) - Result: {result5}")
            
            # Wait longer for subscriptions to be fully processed
            print(f"  Waiting for subscriptions to be processed...")
            time.sleep(2)
            
            # Send registration message
            self.mqtt_client.publish("fl/client_register", 
                                    json.dumps({"client_id": self.client_id}), qos=1)
            print(f"  Registration message sent")
        else:
            print(f"Client {self.client_id} failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            print(f"Client {self.client_id} received message on topic: {msg.topic}, size: {len(msg.payload)} bytes")
            
            if msg.topic == TOPIC_GLOBAL_MODEL:
                self.handle_global_model(msg.payload)
            elif msg.topic == TOPIC_TRAINING_CONFIG:
                self.handle_training_config(msg.payload)
            elif msg.topic == TOPIC_START_TRAINING:
                self.handle_start_training(msg.payload)
            elif msg.topic == TOPIC_START_EVALUATION:
                self.handle_start_evaluation(msg.payload)
            elif msg.topic == 'fl/training_complete':
                self.handle_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling message on topic {msg.topic}: {e}")
            import traceback
            traceback.print_exc()
    
    def on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        if rc == 0:
            print(f"Client {self.client_id} clean disconnect from broker")
            print(f"Client {self.client_id} exiting...")
            # Stop the loop and exit
            self.mqtt_client.loop_stop()
            import sys
            sys.exit(0)
        else:
            print(f"Client {self.client_id} unexpected disconnect, return code {rc}")
            print(f"Client {self.client_id} attempting to reconnect...")
            # Don't stop the loop - paho-mqtt will automatically reconnect
            # The loop_forever() will handle reconnection attempts
    
    def handle_training_complete(self):
        """Handle training completion signal from server"""
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nDisconnecting from server...")
        time.sleep(1)  # Brief delay before disconnect
        self.mqtt_client.disconnect()
        print(f"Client {self.client_id} disconnected successfully.")
    
    def handle_global_model(self, payload):
        """Receive and set global model weights and architecture from server"""
        try:
            data = json.loads(payload.decode())
            round_num = data['round']
            # Ignore duplicate global model for the same round
            if round_num <= self.last_global_round:
                print(f"Client {self.client_id} ignoring duplicate global model for round {round_num}")
                return
            
            # Check if weights are quantized
            if 'quantized_data' in data and self.quantizer is not None:
                # Decompress quantized weights
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
                # Standard deserialization
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
                    
                    # Compile model for classification
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
                # Mark initial global model processed to ignore duplicates
                self.last_global_round = 0
            else:
                # Updated model after aggregation
                if self.model is None:
                    print(f"Client {self.client_id} ERROR: Received model for round {round_num} but local model not initialized!")
                    return
                self.model.set_weights(weights)
                self.current_round = round_num
                self.last_global_round = round_num
                print(f"Client {self.client_id} received global model for round {round_num}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR in handle_global_model: {e}")
            import traceback
            traceback.print_exc()
    
    def handle_training_config(self, payload):
        """Update training configuration"""
        self.training_config = json.loads(payload.decode())
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    def handle_start_training(self, payload):
        """Start local training when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']
        
        # Check if model is initialized
        if self.model is None:
            print(f"Client {self.client_id} ERROR: Model not initialized yet, cannot start training for round {round_num}")
            print(f"Client {self.client_id} waiting for global model from server...")
            return
        
        # Check for duplicate training signals
        if self.last_training_round == round_num:
            print(f"Client {self.client_id} ignoring duplicate start training for round {round_num}")
            return
        
        # Check if we're ready for this round
        if self.current_round == 0 and round_num == 1:
            # First training round with initial global model
            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num} with initial global model...")
            self.train_local_model()
        elif round_num >= self.current_round and round_num <= self.current_round + 1:
            # Next round - use model from current_round (might be round_num-1 if global model arrives late)
            # This handles race condition where start_training arrives before global_model
            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            self.train_local_model()
        else:
            print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
    
    def handle_start_evaluation(self, payload):
        """Start evaluation when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']
        
        # Check if model is initialized
        if self.model is None:
            print(f"Client {self.client_id} ERROR: Model not initialized yet, cannot evaluate for round {round_num}")
            return
        
        if round_num == self.current_round:
            if round_num in self.evaluated_rounds:
                print(f"Client {self.client_id} ignoring duplicate evaluation for round {round_num}")
                return
            print(f"Client {self.client_id} starting evaluation for round {round_num}...")
            self.evaluate_model()
            # After evaluation, prepare for next round
            # Do NOT increment current_round here; wait for server's start_training signal
            # to advance to the next round. This keeps round alignment consistent.
            self.evaluated_rounds.add(round_num)
            print(f"Client {self.client_id} evaluation completed for round {round_num}. Awaiting start_training for round {round_num + 1}.")
        else:
            print(f"Client {self.client_id} skipping evaluation signal for round {round_num} (current: {self.current_round})")
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        # Limit steps per epoch for faster smoke tests (configurable via env)
        try:
            steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
            val_steps = int(os.getenv("VAL_STEPS", "25"))
        except Exception:
            steps_per_epoch = 100
            val_steps = 25
        
        # Add training progress callbacks
        class BatchLogger(tf.keras.callbacks.Callback):
            def __init__(self, client_id: int, frequency: int = 100):
                super().__init__()
                self.client_id = client_id
                self.frequency = frequency
            def on_train_batch_end(self, batch, logs=None):
                if batch % self.frequency == 0:
                    logs = logs or {}
                    loss = logs.get('loss')
                    acc = logs.get('accuracy')
                    try:
                        print(f"[BatchLogger] Client {self.client_id} batch {batch}: loss={loss:.4f}, acc={acc:.4f}")
                    except Exception:
                        print(f"[BatchLogger] Client {self.client_id} batch {batch}: logs={logs}")

        class EpochLogger(tf.keras.callbacks.Callback):
            def __init__(self, client_id: int, total_epochs: int):
                super().__init__()
                self.client_id = client_id
                self.total_epochs = total_epochs
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                loss = logs.get('loss')
                acc = logs.get('accuracy')
                val_loss = logs.get('val_loss')
                val_acc = logs.get('val_accuracy')
                try:
                    print(f"[EpochLogger] Client {self.client_id} epoch {epoch+1}/{self.total_epochs}: "
                          f"loss={loss:.4f}, acc={acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
                except Exception:
                    print(f"[EpochLogger] Client {self.client_id} epoch {epoch+1}/{self.total_epochs}: logs={logs}")

        # Train the model using generator
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            verbose=2,
            callbacks=[BatchLogger(self.client_id), EpochLogger(self.client_id, epochs)]
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
            # Get compression stats
            stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - "
                  f"Ratio: {stats['compression_ratio']:.2f}x, "
                  f"Size: {stats['compressed_size_mb']:.2f}MB")
            
            # Serialize compressed data to JSON-safe base64 string
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            # Send compressed update
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "compressed_data": serialized,
                "num_samples": num_samples,
                "metrics": metrics
            }
        else:
            # Send model update to server without compression
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
        
        # Serialize and send model update with error handling
        try:
            payload = json.dumps(update_message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} serialized update size: {payload_size_mb:.2f} MB")
            
            # Use QoS 1 for reliable delivery of large model update messages
            result = self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, payload, qos=1)
            
            # Wait for the message to be published
            result.wait_for_publish(timeout=30)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Client {self.client_id} sent model update for round {self.current_round}")
                print(f"Training metrics - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            else:
                print(f"Client {self.client_id} ERROR: Failed to send model update, rc={result.rc}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR serializing/sending update: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server"""
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
        
        self.mqtt_client.publish(TOPIC_CLIENT_METRICS, json.dumps(metrics_message))
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    def start(self):
        """Connect to MQTT broker and start listening"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
                # Use 1 hour keepalive (3600 seconds) to prevent timeout during long training
                # Enable automatic reconnection on connection loss
                self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=120)
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 3600)
                print(f"Successfully connected to MQTT broker!\n")
                self.mqtt_client.loop_forever(retry_first_connection=True)
                break
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to MQTT broker after {max_retries} attempts.")
                    print(f"\nPlease ensure:")
                    print(f"  1. Mosquitto broker is running: net start mosquitto")
                    print(f"  2. Broker address is correct: {MQTT_BROKER}:{MQTT_PORT}")
                    print(f"  3. Firewall allows connection on port {MQTT_PORT}")
                    raise

def load_data(client_id):
    # Detect environment: Docker uses /app prefix, local uses relative path
    if os.path.exists('/app'):
        # Running in Docker container
        base_path = '/app/Client/Emotion_Recognition/Dataset'
    else:
        # Running locally - use path relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(project_root, 'Client', 'Emotion_Recognition', 'Dataset')
    
    train_path = os.path.join(base_path, f'client_{client_id}', 'train')
    validation_path = os.path.join(base_path, f'client_{client_id}', 'validation')
    
    print(f"Dataset base path: {base_path}")
    print(f"Train path: {train_path}")
    print(f"Validation path: {validation_path}")
    
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
    print(f"Dataset loaded")
    
    # Create and start client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Client {CLIENT_ID}")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"{'='*60}\n")
    
    try:
        client.start()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} shutting down...")
        client.mqtt_client.disconnect()
