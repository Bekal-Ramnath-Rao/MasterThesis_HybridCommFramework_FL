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
import random
import os
import sys
import logging
import threading
import pika

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# AMQP Configuration
AMQP_HOST = os.getenv("AMQP_HOST", "localhost")
AMQP_PORT = int(os.getenv("AMQP_PORT", "5672"))
AMQP_USER = os.getenv("AMQP_USER", "guest")
AMQP_PASSWORD = os.getenv("AMQP_PASSWORD", "guest")
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))

# AMQP Exchanges and Queues
EXCHANGE_BROADCAST = "fl_broadcast"
EXCHANGE_CLIENT_UPDATES = "fl_client_updates"


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, dataframe):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        
        # Initialize quantization compression
        use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
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
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        
        # AMQP connection
        self.connection = None
        self.channel = None
        self.consuming = False
        
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
                # Using exclusive=True ensures each client gets its own queue
                # Using auto_delete=True cleans up when client disconnects
                result = self.channel.queue_declare(queue=queue_broadcast, durable=False, exclusive=True, auto_delete=True)
                
                # Bind the client's queue to the fanout exchange
                self.channel.queue_bind(exchange=EXCHANGE_BROADCAST, queue=queue_broadcast)
                
                # Set up consumer for the broadcast queue
                # All message types will come through this single queue
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
    
    def send_registration(self):
        """Send registration message to server"""
        registration = {"client_id": self.client_id}
        self.channel.basic_publish(
            exchange=EXCHANGE_CLIENT_UPDATES,
            routing_key='client.register',
            body=json.dumps(registration),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print(f"Client {self.client_id} registration sent")
    
    def on_broadcast_message(self, ch, method, properties, body):
        """Unified handler for all broadcast messages - routes based on message_type"""
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
        except Exception as e:
            print(f"Client {self.client_id} error handling global model: {e}")
    
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
        except Exception as e:
            print(f"Client {self.client_id} error starting training: {e}")
    
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
            
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "compressed_data": compressed_data,
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
        
        self.channel.basic_publish(
            exchange=EXCHANGE_CLIENT_UPDATES,
            routing_key='client.update',
            body=json.dumps(update_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
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
        
        self.channel.basic_publish(
            exchange=EXCHANGE_CLIENT_UPDATES,
            routing_key='client.metrics',
            body=json.dumps(metrics_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, MSE: {mse:.4f}, "
              f"MAE: {mae:.4f}, MAPE: {mape:.4f}")
    
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


if __name__ == "__main__":
    # Load data
    print(f"Loading dataset for client {CLIENT_ID}...")
    dataframe = pd.read_csv("Client/Temperature_Regulation/Dataset/base_data_baseline_unique.csv")
    print(f"Dataset loaded: {dataframe.shape}")
    
    # Create and start client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, dataframe)
    
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
