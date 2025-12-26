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
import asyncio
import os
import logging
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived
from aioquic.asyncio.protocol import QuicConnectionProtocol

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# QUIC Configuration
QUIC_HOST = os.getenv("QUIC_HOST", "localhost")
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))


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
        self.protocol = None
        self.stream_id = 0
        
        # Prepare data and model
        self.prepare_data_and_model(dataframe)
    
    def prepare_data_and_model(self, dataframe):
        """Prepare data partition and create LSTM model for this client"""
        X = dataframe[['Ambient_Temp', 'Cabin_Temp', 'Relative_Humidity', 'Solar_Load']]
        y = dataframe['Set_temp'].values.reshape(-1, 1)
        
        dataX = X.values
        datay = y
        
        tf.random.set_seed(7)
        
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_normalized = scaler_x.fit_transform(dataX)
        y_normalized = scaler_y.fit_transform(datay)
        
        x_train = dataX
        y_train = y_normalized
        
        x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
        
        partition_size = math.floor(len(x_train) / self.num_clients)
        idx_from = self.client_id * partition_size
        idx_to = (self.client_id + 1) * partition_size
        full_x_train_cid = x_train[idx_from:idx_to] / 255.0
        full_y_train_cid = y_train[idx_from:idx_to]
        
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
            # Add newline delimiter for message framing
            data = (json.dumps(message) + '\n').encode('utf-8')
            self.stream_id = self.protocol._quic.get_next_available_stream_id()
            self.protocol._quic.send_stream_data(self.stream_id, data, end_stream=False)
            self.protocol.transmit()
    
    async def handle_message(self, message):
        """Handle incoming messages from server"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'training_config':
                await self.handle_training_config(message)
            elif msg_type == 'global_model':
                await self.handle_global_model(message)
            elif msg_type == 'start_training':
                await self.handle_start_training(message)
            elif msg_type == 'start_evaluation':
                await self.handle_start_evaluation(message)
            elif msg_type == 'training_complete':
                await self.handle_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling message: {e}")
    
    async def handle_training_config(self, message):
        """Update training configuration"""
        self.training_config = message['config']
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    async def handle_global_model(self, message):
        """Receive and set global model weights and architecture from server"""
        round_num = message['round']
        encoded_weights = message['weights']
        
        weights = self.deserialize_weights(encoded_weights)
        
        if round_num == 0:
            # Initial model from server - create model from server's config
            print(f"Client {self.client_id} received initial global model from server")
            
            model_config = message.get('model_config')
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
    
    async def handle_start_training(self, message):
        """Start local training when server signals"""
        round_num = message['round']
        
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
        
        if round_num == self.current_round:
            print(f"Client {self.client_id} starting evaluation for round {round_num}...")
            await self.evaluate_model()
            self.current_round = round_num + 1
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
        
        # Train in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        history = await loop.run_in_executor(
            None,
            lambda: self.model.fit(
                self.x_train,
                self.y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,
                verbose=2
            )
        )
        
        updated_weights = self.model.get_weights()
        num_samples = len(self.x_train)
        
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
        
        # Introduce random delay before sending model update
        delay = random.uniform(0.5, 3.0)
        print(f"Client {self.client_id} waiting {delay:.2f} seconds before sending update...")
        await asyncio.sleep(delay)
        
        update_message = {
            "type": "model_update",
            "client_id": self.client_id,
            "round": self.current_round,
            "weights": self.serialize_weights(updated_weights),
            "num_samples": num_samples,
            "metrics": metrics
        }
        
        await self.send_message(update_message)
        print(f"Client {self.client_id} sent model update for round {self.current_round}")
        print(f"Training metrics - Loss: {metrics['loss']:.4f}, MSE: {metrics['mse']:.4f}, "
              f"MAE: {metrics['mae']:.4f}, MAPE: {metrics['mape']:.4f}")
    
    async def evaluate_model(self):
        """Evaluate model on test data and send metrics to server"""
        loop = asyncio.get_event_loop()
        loss, mse, mae, mape = await loop.run_in_executor(
            None,
            lambda: self.model.evaluate(
                self.x_test, self.y_test, 
                batch_size=32, 
                verbose=0
            )
        )
        
        num_samples = len(self.x_test)
        
        metrics_message = {
            "type": "metrics",
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
        
        await self.send_message(metrics_message)
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, MSE: {mse:.4f}, "
              f"MAE: {mae:.4f}, MAPE: {mape:.4f}")
    
    async def start(self):
        """Connect to QUIC server and start client"""
        configuration = QuicConfiguration(is_client=True)
        configuration.verify_mode = False  # For testing; use proper certs in production
        
        print(f"Attempting to connect to QUIC server at {QUIC_HOST}:{QUIC_PORT}...")
        
        async with connect(
            QUIC_HOST,
            QUIC_PORT,
            configuration=configuration,
            create_protocol=lambda *args, **kwargs: FederatedLearningClientProtocol(*args, **kwargs),
        ) as client:
            self.protocol = client
            client.client = self
            
            print(f"Client {self.client_id} connected to QUIC server\n")
            
            # Register with server
            await self.send_message({
                'type': 'register',
                'client_id': self.client_id
            })
            print(f"Client {self.client_id} registration message sent")
            
            # Keep connection alive
            await asyncio.Future()


async def main():
    print(f"Loading dataset for client {CLIENT_ID}...")
    dataframe = pd.read_csv("Dataset/base_data_baseline_unique.csv")
    print(f"Dataset loaded: {dataframe.shape}")
    
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, dataframe)
    
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Client {CLIENT_ID}")
    print(f"Server: {QUIC_HOST}:{QUIC_PORT}")
    print(f"{'='*60}\n")
    
    try:
        await client.start()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} shutting down...")


if __name__ == "__main__":
    asyncio.run(main())
