import numpy as np
import pandas as pd
import math
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
            
            # Send flow control updates to allow more data (critical for poor networks)
            self.transmit()
            
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
        
        # Partition data for this client (use client_id - 1 for 0-based indexing)
        client_index = self.client_id - 1  # Convert 1-based to 0-based
        partition_size = math.floor(len(x_train) / self.num_clients)
        idx_from = client_index * partition_size
        idx_to = (client_index + 1) * partition_size
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
            self.protocol._quic.send_stream_data(self.stream_id, data, end_stream=True)
            self.protocol.transmit()
            
            # FAIR FIX: Removed artificial delays (1.5s for large, 0.1s for small messages)
            # QUIC handles flow control automatically, so manual delays are unnecessary
            # This makes QUIC behavior similar to other protocols which don't have artificial delays
    
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
        
        # Decompress or deserialize weights
        if 'quantized_data' in message and self.quantizer is not None:
            weights = self.quantizer.decompress(message['quantized_data'])
            print(f"Client {self.client_id}: Received and decompressed quantized global model")
        elif 'compressed_data' in message and self.quantizer is not None:
            weights = self.quantizer.decompress(message['compressed_data'])
            print(f"Client {self.client_id}: Received and decompressed quantized global model")
        else:
            weights = self.deserialize_weights(encoded_weights)
        
        # Initialize model if not yet created (works for any round)

        
        if self.model is None:

        
            print(f"Client {self.client_id} initializing model from server (round {round_num})")
            
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
            # Updated model after aggregation - don't change current_round
            # as it was already updated by handle_start_evaluation
            self.model.set_weights(weights)
            print(f"Client {self.client_id} received global model for round {round_num}")
    
    async def handle_start_training(self, message):
        """Start local training when server signals"""
        round_num = message['round']
        
        # Ensure model is initialized before training
        if self.model is None:
            print(f"Client {self.client_id} waiting for global model before training...")
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
                batch_size=16, 
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
        configuration = QuicConfiguration(
            is_client=True,
            alpn_protocols=["fl"],
            max_stream_data=50 * 1024 * 1024,  # 50 MB per stream
            max_data=100 * 1024 * 1024,  # 100 MB total
            idle_timeout=3600.0,  # 60 minutes idle timeout
            max_datagram_frame_size=65536,  # Larger frame size for better throughput
            initial_rtt=0.15,  # 150ms (account for 100ms latency + jitter)
        )
        
        # Load CA certificate for verification (optional - set verify_mode to False for testing)
        # cert_dir = Path(__file__).parent.parent.parent / "certs"
        # ca_cert = cert_dir / "server-cert.pem"
        # if ca_cert.exists():
        #     configuration.load_verify_locations(str(ca_cert))
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
