import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import json
import pickle
import base64
import time
import paho.mqtt.client as mqtt
import os
import logging

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

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


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, dataframe):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.current_round = 0
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id=f"fl_client_{client_id}")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        
        # Prepare data and model
        self.prepare_data_and_model(dataframe)
        
    def prepare_data_and_model(self, dataframe):
        """Prepare data partition and create LSTM model for this client"""
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
        
        # Partition data for this client
        partition_size = math.floor(len(x_train) / self.num_clients)
        idx_from = self.client_id * partition_size
        idx_to = (self.client_id + 1) * partition_size
        full_x_train_cid = x_train[idx_from:idx_to] / 255.0
        full_y_train_cid = y_train[idx_from:idx_to]
        
        # Split into train and test (80/20)
        split_idx = math.floor(len(full_x_train_cid) * 0.8)
        self.x_train = full_x_train_cid[:split_idx]
        self.y_train = full_y_train_cid[:split_idx]
        self.x_test = full_x_train_cid[split_idx:]
        self.y_test = full_y_train_cid[split_idx:]
        
        # Create LSTM model
        self.model = Sequential()
        self.model.add(LSTM(50, activation='relu', input_shape=(1, X_normalized.shape[1])))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam', 
                          metrics=['mse', 'mae', 'mape'])
        
        print(f"Client {self.client_id} initialized with {len(self.x_train)} training samples "
              f"and {len(self.x_test)} test samples")
    
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
            
            result5, mid5 = self.mqtt_client.subscribe('fl/training_complete')
            print(f"  Subscribed to fl/training_complete - Result: {result5}")
            
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
    
    def handle_training_complete(self):
        """Handle training completion signal from server"""
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nDisconnecting from server...")
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop()
        import sys
        sys.exit(0)
    
    def handle_global_model(self, payload):
        """Receive and set global model weights"""
        data = json.loads(payload.decode())
        round_num = data['round']
        encoded_weights = data['weights']
        
        weights = self.deserialize_weights(encoded_weights)
        self.model.set_weights(weights)
        
        if round_num == 0:
            # Initial model from server
            print(f"Client {self.client_id} received initial global model from server")
            self.current_round = 0
        else:
            # Updated model after aggregation
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
        
        # Send model update to server
        update_message = {
            "client_id": self.client_id,
            "round": self.current_round,
            "weights": self.serialize_weights(updated_weights),
            "num_samples": num_samples,
            "metrics": metrics
        }
        
        self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, json.dumps(update_message))
        print(f"Client {self.client_id} sent model update for round {self.current_round}")
        print(f"Training metrics - Loss: {metrics['loss']:.4f}, MSE: {metrics['mse']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.4f}")
    
    def evaluate_model(self):
        """Evaluate model on test data and send metrics to server"""
        loss, mse, mae, mape = self.model.evaluate(
            self.x_test, self.y_test, 
            batch_size=32, 
            verbose=0
        )
        
        num_samples = len(self.x_test)
        
        metrics_message = {
            "client_id": self.client_id,
            "round": self.current_round,
            "num_samples": num_samples,
            "metrics": {
                "loss": float(loss),
                "mse": float(mse),
                "mae": float(mae),
                "mape": float(mape)
            }
        }
        
        self.mqtt_client.publish(TOPIC_CLIENT_METRICS, json.dumps(metrics_message))
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, MSE: {mse:.4f}, "
              f"MAE: {mae:.4f}, MAPE: {mape:.4f}")
    
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
    dataframe = pd.read_csv("Dataset/base_data_baseline_unique.csv")
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
