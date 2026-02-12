import numpy as np
import pandas as pd
import math
import os
import sys
import logging

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
# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import json
import pickle
import base64
import time
import random
import paho.mqtt.client as mqtt

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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
    def __init__(self, client_id, num_clients, dataframe):
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
        self.x_test = None
        self.y_test = None
        self.current_round = 0
        self.training_config = {"batch_size": 16, "local_epochs": 20}
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.has_converged = False
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id=f"fl_client_{client_id}")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Prepare data and model
        self.prepare_data_and_model(dataframe)
        
    def prepare_data_and_model(self, dataframe):
        """Prepare data partition for this client"""
        # Extract relevant data
        X = dataframe[['Ambient_Temp', 'Cabin_Temp', 'Relative_Humidity', 'Solar_Load']]
        y = dataframe['Set_temp'].values.reshape(-1, 1)
        
        dataX = X.values
        datay = y
        
        # Fix random seed for reproducibility
        tf.random.set_seed(7)
        
        # Normalizing data
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_normalized = scaler_x.fit_transform(dataX)
        y_normalized = scaler_y.fit_transform(datay)
        
        x_train = dataX
        y_train = y_normalized
        
        # Reshape input to be [samples, time steps, features]
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        
        # Partition data for this client (use client_id - 1 for 0-based indexing)
        client_index = self.client_id - 1  # Convert 1-based to 0-based
        partition_size = math.floor(len(x_train) / self.num_clients)
        idx_from = client_index * partition_size
        idx_to = (client_index + 1) * partition_size
        full_x_train_cid = x_train[idx_from:idx_to] / 255.0
        full_y_train_cid = y_train[idx_from:idx_to]
        
        # Split into train and test (80/20)
        split_idx = math.floor(len(full_x_train_cid) * 0.8)
        self.x_train = full_x_train_cid[:split_idx]
        self.y_train = full_y_train_cid[:split_idx]
        self.x_test = full_x_train_cid[split_idx:]
        self.y_test = full_y_train_cid[split_idx:]
        
        # DO NOT create model here - wait for server to send it
        print(f"Client {self.client_id} initialized with {len(self.x_train)} training samples "
              f"and {len(self.x_test)} test samples")
        print(f"Client {self.client_id} waiting for initial global model from server...")
    
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
            # Subscribe to topics
            result1, mid1 = self.mqtt_client.subscribe(TOPIC_GLOBAL_MODEL)
            print(f"  Subscribed to {TOPIC_GLOBAL_MODEL} - Result: {result1}")
            
            result2, mid2 = self.mqtt_client.subscribe(TOPIC_TRAINING_CONFIG)
            print(f"  Subscribed to {TOPIC_TRAINING_CONFIG} - Result: {result2}")
            
            result3, mid3 = self.mqtt_client.subscribe(TOPIC_START_TRAINING)
            print(f"  Subscribed to {TOPIC_START_TRAINING} - Result: {result3}")
            
            result4, mid4 = self.mqtt_client.subscribe(TOPIC_START_EVALUATION)
            print(f"  Subscribed to {TOPIC_START_EVALUATION} - Result: {result4}")
            
            result5, mid5 = self.mqtt_client.subscribe(TOPIC_TRAINING_COMPLETE)
            print(f"  Subscribed to fl/training_complete (QoS 1) - Result: {result5}")
            
            # Send registration message
            self.mqtt_client.publish("fl/client_register", 
                                    json.dumps({"client_id": self.client_id}))
            print(f"  Registration message sent")
        else:
            print(f"Client {self.client_id} failed to connect, return code {rc}")
    
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
            elif msg.topic == 'fl/training_complete':
                self.handle_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling message: {e}")
    
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
            self.mqtt_client.loop_stop()
    
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
        data = json.loads(payload.decode())
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
        
        # Initialize model if not yet created (works for any round)
        if self.model is None:
            print(f"Client {self.client_id} initializing model from server (round {round_num})")
            
            model_config = data.get('model_config')
            if model_config:
                # Build model from server's architecture definition
                self.model = Sequential()
                for layer_config in model_config['layers']:
                    if layer_config['type'] == 'LSTM':
                        self.model.add(LSTM(
                            layer_config['units'],
                            activation=layer_config['activation'],
                            input_shape=tuple(layer_config['input_shape'])
                        ))
                    elif layer_config['type'] == 'Dense':
                        self.model.add(Dense(layer_config['units']))
                
                # Compile with server's config
                compile_cfg = model_config['compile_config']
                self.model.compile(
                    loss=compile_cfg['loss'],
                    optimizer=compile_cfg['optimizer'],
                    metrics=compile_cfg['metrics']
                )
                print(f"Client {self.client_id} built model from server configuration")
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
    
    def handle_training_config(self, payload):
        """Update training configuration"""
        self.training_config = json.loads(payload.decode())
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    def handle_start_training(self, payload):
        """Start local training when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']
        
        # Ensure model is initialized before training
        if self.model is None:
            print(f"Client {self.client_id} waiting for global model before training...")
            return
        
        # Check if we're ready for this round (should have received global model first)
        if self.current_round == 0 and round_num == 1:
            # First training round with initial global model
            self.current_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num} with initial global model...")
            self.train_local_model()
        elif round_num == self.current_round:
            # Subsequent rounds
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            self.train_local_model()
        else:
            print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
    
    def handle_start_evaluation(self, payload):
        """Start evaluation when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']
        
        if round_num == self.current_round:
            print(f"Client {self.client_id} starting evaluation for round {round_num}...")
            self.evaluate_model()
            # After evaluation, prepare for next round
            self.current_round = round_num + 1
            print(f"Client {self.client_id} ready for next round {self.current_round}")
        else:
            print(f"Client {self.client_id} skipping evaluation signal for round {round_num} (current: {self.current_round})")
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        
        # Train the model
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=2
        )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        num_samples = len(self.x_train)
        
        # Prepare training metrics
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "mse": float(history.history["mse"][-1]),
            "mae": float(history.history["mae"][-1]),
            "mape": float(history.history["mape"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_mse": float(history.history["val_mse"][-1]),
            "val_mae": float(history.history["val_mae"][-1]),
            "val_mape": float(history.history["val_mape"][-1]),
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
        
        # FAIR FIX: Removed random delay - this was causing unfair comparison with other protocols
        # Other protocols don't have random delays, so MQTT shouldn't either
        
        self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, json.dumps(update_message))
        print(f"Client {self.client_id} sent model update for round {self.current_round}")
        print(f"Training metrics - Loss: {metrics['loss']:.4f}, MSE: {metrics['mse']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.4f}")
    
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
        """Evaluate model on test data and send metrics to server"""
        loss, mse, mae, mape = self.model.evaluate(
            self.x_test, self.y_test, 
            batch_size=16, 
            verbose=0
        )
        
        num_samples = len(self.x_test)
        self._update_local_convergence(float(loss))
        
        metrics_dict = {
            "loss": float(loss),
            "mse": float(mse),
            "mae": float(mae),
            "mape": float(mape)
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
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, MSE: {mse:.4f}, "
              f"MAE: {mae:.4f}, MAPE: {mape:.4f}")
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
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
                print(f"Successfully connected to MQTT broker!\n")
                self.mqtt_client.loop_forever()
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


if __name__ == "__main__":
    # Load data
    print(f"Loading dataset for client {CLIENT_ID}...")
    # Detect environment and construct dataset path
    if os.path.exists('/app'):
        dataset_path = '/app/Client/Temperature_Regulation/Dataset/base_data_baseline_unique.csv'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        dataset_path = os.path.join(project_root, 'Client', 'Temperature_Regulation', 'Dataset', 'base_data_baseline_unique.csv')
    print(f"Dataset path: {dataset_path}")
    dataframe = pd.read_csv(dataset_path)
    print(f"Dataset loaded: {dataframe.shape}")
    
    # Create and start client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, dataframe)
    
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Client {CLIENT_ID}")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"{'='*60}\n")
    
    try:
        client.start()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} shutting down...")
        client.mqtt_client.disconnect()
