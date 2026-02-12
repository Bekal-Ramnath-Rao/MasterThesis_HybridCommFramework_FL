import numpy as np
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
# Fallback strategy: GPU_DEVICE_ID -> (CLIENT_ID - 1) -> "0"
# This ensures different clients use different GPUs in multi-GPU setups
client_id_env = os.environ.get("CLIENT_ID", "0")
try:
    default_gpu = str(0)  # Client 1->GPU 0, Client 2->GPU 1, etc.
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

# Detect Docker environment and set project root accordingly
if os.path.exists('/app'):
    # Likely running in Docker, code is under /app
    project_root = '/app'
else:
    # Local development: go up two levels from this file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from packet_logger import log_sent_packet, log_received_packet, init_db
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
# Auto-detect environment: Docker (/app exists) or local
MQTT_BROKER = os.getenv("MQTT_BROKER", 'mqtt-broker' if os.path.exists('/app') else 'localhost')
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))  # MQTT broker port
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))  # Can be set via environment variable
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))

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
        # Client-side convergence
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.has_converged = False
        
        # Store data generators
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        
        # Model will be initialized from server config
        self.model = None

        # Initialize packet logger database
        init_db()
        
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
        # FAIR CONFIG: Limited queue to 1000 messages (aligned with AMQP/gRPC)
        self.mqtt_client.max_queued_messages_set(1000)
        # FAIR CONFIG: Set max packet size to 128MB (aligned with AMQP default)
        self.mqtt_client._max_packet_size = 128 * 1024 * 1024  # 128 MB
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
    
    def build_model_from_config(self, model_config):
        """Build CNN model from server-provided configuration"""
        input_shape = model_config.get('input_shape', (48, 48, 1))
        num_classes = model_config.get('num_classes', 7)
        layers = model_config.get('layers', [
            {'type': 'conv', 'filters': 64, 'kernel': 3, 'activation': 'relu'},
            {'type': 'maxpool', 'pool_size': 2},
            {'type': 'conv', 'filters': 128, 'kernel': 3, 'activation': 'relu'},
            {'type': 'maxpool', 'pool_size': 2},
            {'type': 'conv', 'filters': 256, 'kernel': 3, 'activation': 'relu'},
            {'type': 'maxpool', 'pool_size': 2},
            {'type': 'flatten'},
            {'type': 'dense', 'units': 256, 'activation': 'relu'},
            {'type': 'dropout', 'rate': 0.5},
            {'type': 'dense', 'units': num_classes, 'activation': 'softmax'}
        ])
        
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
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
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
            log_sent_packet(
                packet_size=len(json.dumps({"client_id": self.client_id})),
                peer="fl/client_register",  # or client_id/server_id as appropriate
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info="any additional info"
            )
            
            print(f"  Registration message sent")
        else:
            print(f"Client {self.client_id} failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            print(f"Client {self.client_id} received message on topic: {msg.topic}, size: {len(msg.payload)} bytes")
            log_received_packet(
                packet_size=len(msg.payload),
                peer=msg.topic,  # or client_id/server_id as appropriate
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info=msg.topic
            )
            
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
            
            # Initialize model if not yet created (works for any round)
            if self.model is None:
                print(f"Client {self.client_id} initializing model from server (round {round_num})")
                
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
                self.last_global_round = round_num
            else:
                # Updated model after aggregation (model already initialized)
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
            print(f"Client {self.client_id} waiting for global model (not yet initialized)...")
            return
        
        # Check for duplicate training signals
        if self.last_training_round == round_num:
            print(f"Client {self.client_id} ignoring duplicate start training for round {round_num}")
            return
        
        # Start training regardless of round number (generic approach)
        self.current_round = round_num
        self.last_training_round = round_num
        print(f"\nClient {self.client_id} starting training for round {round_num}...")
        self.train_local_model()
    
    def handle_start_evaluation(self, payload):
        """Start evaluation when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']
        
        # Check if model is initialized
        if self.model is None:
            print(f"Client {self.client_id} waiting for global model (not yet initialized)...")
            return
        
        # Check for duplicate evaluation signals
        if round_num in self.evaluated_rounds:
            print(f"Client {self.client_id} ignoring duplicate evaluation for round {round_num}")
            return
        
        # Evaluate regardless of round number (generic approach)
        self.current_round = round_num
        print(f"Client {self.client_id} starting evaluation for round {round_num}...")
        self.evaluate_model()
        self.evaluated_rounds.add(round_num)
        print(f"Client {self.client_id} evaluation completed for round {round_num}")
    
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
        
        # Train the model using generator
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
        
        # FAIR FIX: Removed random delay - this was causing unfair comparison with other protocols
        # Other protocols don't have random delays, so MQTT shouldn't either
        
        # Serialize and send model update with error handling
        try:
            payload = json.dumps(update_message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} serialized update size: {payload_size_mb:.2f} MB")
            
            # Use QoS 1 for reliable delivery of large model update messages
            result = self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, payload, qos=1)
            
            log_sent_packet(
                packet_size=len(payload),
                peer=TOPIC_CLIENT_UPDATE,  # or client_id/server_id as appropriate
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info="any additional info"
            )
            # FAIR FIX: Use shorter timeout (5s) aligned with other protocols
            # MQTT QoS 1 ensures delivery, so we only need to wait for queue confirmation
            if result.rc == mqtt.MQTT_ERR_NO_CONN:
                raise Exception("MQTT not connected")
            result.wait_for_publish(timeout=5)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Client {self.client_id} sent model update for round {self.current_round}")
                print(f"Training metrics - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
            else:
                print(f"Client {self.client_id} ERROR: Failed to send model update, rc={result.rc}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR serializing/sending update: {e}")
            import traceback
            traceback.print_exc()
    
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

    def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server"""
        loss, accuracy = self.model.evaluate(
            self.validation_generator,
            verbose=0
        )
        
        num_samples = self.validation_generator.n

        self._update_local_convergence(float(loss))
        
        metrics_dict = {
            "loss": float(loss),
            "accuracy": float(accuracy)
        }
        if self.has_converged:
            metrics_dict["client_converged"] = 1.0
        
        metrics_message = {
            "client_id": self.client_id,
            "round": self.current_round,
            "num_samples": num_samples,
            "metrics": metrics_dict
        }
        
        self.mqtt_client.publish(TOPIC_CLIENT_METRICS, json.dumps(metrics_message))
        log_sent_packet(
            packet_size=len(json.dumps(metrics_message)),
            peer=TOPIC_CLIENT_METRICS,  # or client_id/server_id as appropriate
            protocol="MQTT",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="any additional info"
        )
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        if self.has_converged:
            print(f"Client {self.client_id} notifying server of convergence and disconnecting")
            time.sleep(2)
            self.mqtt_client.disconnect()
    
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
                # FAIR CONFIG: keepalive 600s for very_poor network
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 600)
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
