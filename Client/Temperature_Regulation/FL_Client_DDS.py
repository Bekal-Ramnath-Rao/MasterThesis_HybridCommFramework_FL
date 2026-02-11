import numpy as np
import pandas as pd
import math
import pickle
import time
import random
import os
import sys
import logging
import json

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

# Add CycloneDDS DLL path
cyclone_path = r"C:\Masters_Infotech\Semester_5\MT_SW_Addons\vcpkg\buildtrees\cyclonedds\x64-windows-rel\bin"
if cyclone_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = cyclone_path + os.pathsep + os.environ.get('PATH', '')

from cyclonedds.domain import DomainParticipant

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.util import duration
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from cyclonedds.core import Qos, Policy
from dataclasses import dataclass
from typing import List

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# DDS Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))

# Chunking configuration for large messages
CHUNK_SIZE = 64 * 1024  # 64KB chunks for better DDS performance in poor networks


# DDS Data Types (matching IDL)
@dataclass
class ClientRegistration(IdlStruct):
    client_id: int
    message: str


@dataclass
class TrainingConfig(IdlStruct):
    batch_size: int
    local_epochs: int


@dataclass
class TrainingCommand(IdlStruct):
    round: int
    start_training: bool
    start_evaluation: bool
    training_complete: bool


@dataclass
class GlobalModel(IdlStruct):
    round: int
    weights: sequence[int]
    model_config_json: str = ""  # JSON string containing model configuration


@dataclass
class GlobalModelChunk(IdlStruct):
    round: int
    chunk_id: int
    total_chunks: int
    payload: sequence[int]
    model_config_json: str = ""  # JSON string containing model configuration


@dataclass
class ModelUpdate(IdlStruct):
    client_id: int
    round: int
    weights: sequence[int]
    num_samples: int
    loss: float
    mse: float
    mae: float
    mape: float


@dataclass
class ModelUpdateChunk(IdlStruct):
    client_id: int
    round: int
    chunk_id: int
    total_chunks: int
    payload: sequence[int]
    num_samples: int
    loss: float
    mse: float
    mae: float
    mape: float


@dataclass
class EvaluationMetrics(IdlStruct):
    client_id: int
    round: int
    num_samples: int
    loss: float
    mse: float
    mae: float
    mape: float


@dataclass
class ServerStatus(IdlStruct):
    current_round: int
    total_rounds: int
    training_started: bool
    training_complete: bool
    registered_clients: int


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, dataframe):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        
        # Initialize quantization compression (default: enabled in Docker, disabled locally)
        uq_env = os.getenv("USE_QUANTIZATION")
        if uq_env is None:
            use_quantization = os.path.exists('/app')
        else:
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
        self.running = True
        
        # Chunk reassembly buffers
        self.global_model_chunks = {}  # {chunk_id: payload}
        self.global_model_metadata = {}  # {round, total_chunks, model_config_json}
        
        # DDS entities
        self.participant = None
        self.readers = {}
        self.writers = {}
        
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
        """Serialize model weights for DDS transmission"""
        serialized = pickle.dumps(weights)
        # Convert bytes to list of ints for DDS
        return list(serialized)
    
    def deserialize_weights(self, serialized_weights):
        """Deserialize model weights received from DDS"""
        # Convert list of ints back to bytes
        return pickle.loads(bytes(serialized_weights))
    
    def split_into_chunks(self, data):
        """Split serialized data into chunks of CHUNK_SIZE"""
        chunks = []
        for i in range(0, len(data), CHUNK_SIZE):
            chunks.append(data[i:i + CHUNK_SIZE])
        return chunks
    
    def send_model_update_chunked(self, round_num, serialized_weights, num_samples, loss, mse, mae, mape):
        """Send model update as chunks"""
        chunks = self.split_into_chunks(serialized_weights)
        total_chunks = len(chunks)
        
        print(f"Client {self.client_id}: Sending model update in {total_chunks} chunks ({len(serialized_weights)} bytes total)")
        
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = ModelUpdateChunk(
                client_id=self.client_id,
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                num_samples=num_samples,
                loss=loss,
                mse=mse,
                mae=mae,
                mape=mape
            )
            self.writers['model_update_chunk'].write(chunk)
            print(f"  Sent chunk {chunk_id + 1}/{total_chunks} ({len(chunk_data)} bytes)")
            time.sleep(0.05)  # Small delay between chunks
    
    def setup_dds(self):
        """Initialize DDS participant, topics, readers, and writers"""
        print(f"Setting up DDS on domain {DDS_DOMAIN_ID}...")
        
        # Create domain participant
        self.participant = DomainParticipant(DDS_DOMAIN_ID)
        
        # Reliable QoS for critical control messages (registration, config, commands)
        # TransientLocal durability ensures messages survive discovery delays
        reliable_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
            Policy.History.KeepLast(10),
            Policy.Durability.TransientLocal
        )

        # Best effort QoS for large data transfers (model chunks)
        best_effort_qos = Qos(
            Policy.Reliability.BestEffort(),
            Policy.History.KeepLast(1),
        )
        
        # Create topics
        topic_registration = Topic(self.participant, "ClientRegistration", ClientRegistration)
        topic_config = Topic(self.participant, "TrainingConfig", TrainingConfig)
        topic_command = Topic(self.participant, "TrainingCommand", TrainingCommand)
        topic_global_model = Topic(self.participant, "GlobalModel", GlobalModel)
        topic_global_model_chunk = Topic(self.participant, "GlobalModelChunk", GlobalModelChunk)
        topic_model_update = Topic(self.participant, "ModelUpdate", ModelUpdate)
        topic_model_update_chunk = Topic(self.participant, "ModelUpdateChunk", ModelUpdateChunk)
        topic_metrics = Topic(self.participant, "EvaluationMetrics", EvaluationMetrics)
        topic_status = Topic(self.participant, "ServerStatus", ServerStatus)
        
        # Create readers (for receiving from server)
        # Use Reliable QoS for config and commands (critical control messages)
        self.readers['config'] = DataReader(self.participant, topic_config, qos=reliable_qos)
        self.readers['command'] = DataReader(self.participant, topic_command, qos=reliable_qos)
        # Use BestEffort for chunked model data
        self.readers['global_model'] = DataReader(self.participant, topic_global_model, qos=best_effort_qos)
        self.readers['global_model_chunk'] = DataReader(self.participant, topic_global_model_chunk, qos=best_effort_qos)
        self.readers['status'] = DataReader(self.participant, topic_status, qos=best_effort_qos)
        
        # Create writers (for sending to server)
        # Use Reliable QoS for registration (critical to ensure server receives it)
        self.writers['registration'] = DataWriter(self.participant, topic_registration, qos=reliable_qos)
        # Use BestEffort for chunked data and metrics
        self.writers['model_update'] = DataWriter(self.participant, topic_model_update, qos=best_effort_qos)
        self.writers['model_update_chunk'] = DataWriter(self.participant, topic_model_update_chunk, qos=best_effort_qos)
        self.writers['metrics'] = DataWriter(self.participant, topic_metrics, qos=best_effort_qos)

        print(f"Client {self.client_id} DDS setup complete (Reliable QoS for control, BestEffort for data)")
        
        # Wait longer for DDS endpoint discovery (critical for BestEffort QoS!)
        # BestEffort chunks are lost if sent before DataReader/DataWriter are matched
        print(f"Client {self.client_id} waiting for endpoint discovery...")
        time.sleep(3.0)  # Increased for BestEffort chunk endpoints
        
        # Register with server (send multiple times for reliability)
        registration = ClientRegistration(
            client_id=self.client_id,
            message=f"Client {self.client_id} ready"
        )
        print(f"Client {self.client_id} sending registration...")
        for i in range(3):
            self.writers['registration'].write(registration)
            print(f"  Registration attempt {i+1}/3")
            time.sleep(0.3)
        print(f"Client {self.client_id} registration sent\n")
    
    def run(self):
        """Main client loop"""
        print("="*60)
        print(f"Starting Federated Learning Client {self.client_id}")
        print(f"DDS Domain ID: {DDS_DOMAIN_ID}")
        print("="*60)
        print()
        
        # Setup DDS
        self.setup_dds()
        
        # Get training configuration
        self.get_training_config()
        
        print(f"Client {self.client_id} waiting for training to start...\n")
        
        try:
            while self.running:
                # Check for global model updates
                self.check_global_model()
                
                # Check for training commands
                self.check_commands()
                
                time.sleep(0.1)  # Check more frequently
                
        except KeyboardInterrupt:
            print(f"\nClient {self.client_id} shutting down...")
        finally:
            self.cleanup()
    
    def get_training_config(self):
        """Get training configuration from server (no timeout)"""
        
        while True:  # Wait indefinitely for config
            samples = self.readers['config'].take()
            for sample in samples:
                if sample:
                    self.training_config = {
                        "batch_size": sample.batch_size,
                        "local_epochs": sample.local_epochs
                    }
                    print(f"Client {self.client_id} received config: {self.training_config}")
                    return
            time.sleep(0.5)
        
        print(f"Client {self.client_id} using default config: {self.training_config}")
    
    def check_commands(self):
        """Check for training commands from server"""
        samples = self.readers['command'].take()
        
        for sample in samples:
            if sample:
                if sample.training_complete:
                    print(f"\nClient {self.client_id} - Training completed!")
                    self.running = False
                    return
                
                # Check if we're ready for this round (should have received global model first)
                if sample.start_training:
                    if self.current_round == 0 and sample.round == 1:
                        # First training round with initial global model
                        if self.model is None:
                            print(f"Client {self.client_id} waiting for initial model before training...")
                            return
                        self.current_round = sample.round
                        print(f"\nClient {self.client_id} starting training for round {self.current_round} with initial global model...")
                        self.train_local_model()
                    elif sample.round > self.current_round:
                        # Subsequent rounds
                        self.current_round = sample.round
                        print(f"\nClient {self.client_id} starting training for round {self.current_round}...")
                        self.train_local_model()
    
    def check_global_model(self):
        """Check for global model updates from server (chunked version)"""
        # Check for chunked global model
        chunk_samples = self.readers['global_model_chunk'].take()
        
        for sample in chunk_samples:
            if not sample or not hasattr(sample, 'round'):
                continue
            
            round_num = sample.round
            chunk_id = sample.chunk_id
            total_chunks = sample.total_chunks
            
            # Initialize buffers if needed
            if not self.global_model_metadata:
                self.global_model_metadata = {
                    'round': round_num,
                    'total_chunks': total_chunks,
                    'model_config_json': sample.model_config_json if hasattr(sample, 'model_config_json') else ''
                }
                print(f"Client {self.client_id}: Receiving global model in {total_chunks} chunks...")
            
            # Store chunk
            self.global_model_chunks[chunk_id] = sample.payload
            print(f"Client {self.client_id}: Received chunk {chunk_id + 1}/{total_chunks}")
            
            # Check if all chunks received
            if len(self.global_model_chunks) == total_chunks:
                print(f"Client {self.client_id}: All chunks received, reassembling...")
                
                # Reassemble chunks in order
                reassembled_data = []
                for i in range(total_chunks):
                    if i in self.global_model_chunks:
                        reassembled_data.extend(self.global_model_chunks[i])
                    else:
                        print(f"ERROR: Missing chunk {i}")
                        break
                
                # Only process if we have all chunks
                if len(reassembled_data) > 0:
                    # Deserialize and potentially decompress weights
                    raw_weights = self.deserialize_weights(reassembled_data)
                    
                    # Check if weights are compressed (quantized)
                    if isinstance(raw_weights, dict) and 'compressed_data' in raw_weights:
                        if self.quantizer is not None:
                            weights = self.quantizer.decompress(raw_weights)
                            print(f"Client {self.client_id}: Received and decompressed quantized global model")
                        else:
                            print(f"Client {self.client_id}: ERROR - Received quantized data but quantizer not initialized!")
                            # Clear buffers and continue
                            self.global_model_chunks.clear()
                            self.global_model_metadata.clear()
                            continue
                    else:
                        weights = raw_weights
                    
                    # Check if model needs initialization (works for late-joiners too)
                    if self.model is None:
                        print(f"Client {self.client_id} received initial global model from server (round {round_num})")
                        
                        # Parse model config if available
                        model_config_json = self.global_model_metadata.get('model_config_json', '')
                        if model_config_json:
                            model_config = json.loads(model_config_json)
                            
                            # Build model from server's architecture
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
                            # Fallback: hardcode architecture if server doesn't send config
                            self.model = Sequential()
                            self.model.add(LSTM(50, activation='relu', input_shape=(1, 4)))
                            self.model.add(Dense(1))
                            self.model.compile(loss='mean_squared_error', optimizer='adam', 
                                              metrics=['mse', 'mae', 'mape'])
                            print(f"Client {self.client_id} using hardcoded model architecture")
                        
                        # Set the initial weights from server
                        self.model.set_weights(weights)
                        print(f"Client {self.client_id} model initialized with server weights")
                        self.current_round = 0
                    elif round_num == self.current_round:
                        # Updated model after aggregation
                        self.model.set_weights(weights)
                        print(f"Client {self.client_id} received global model for round {self.current_round}")
                        
                        # Evaluate immediately after receiving global model
                        print(f"Client {self.client_id} starting evaluation for round {self.current_round}...")
                        self.evaluate_model()
                    
                    # Clear chunk buffers
                    self.global_model_chunks.clear()
                    self.global_model_metadata.clear()
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        
        # Train the model
        history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            validation_data=(self.x_test, self.y_test)
        )
        
        # Get final metrics
        final_loss = history.history['loss'][-1]
        final_mse = history.history['mse'][-1]
        final_mae = history.history['mae'][-1]
        final_mape = history.history['mape'][-1]
        
        # Get model weights
        weights = self.model.get_weights()
        # Compress or serialize weights
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(weights, data_type="weights")
            stats = self.quantizer.get_compression_stats(weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Size: {stats['compressed_size_mb']:.2f}MB")
            # Serialize compressed data (pickle + convert to list of ints for DDS)
            serialized_weights = list(pickle.dumps(compressed_data))
        else:
            serialized_weights = self.serialize_weights(weights)
        
        # FAIR FIX: Removed random delay - this was causing unfair comparison with other protocols
        # Other protocols don't have random delays, so DDS shouldn't either
        
        # Send model update to server using chunking
        self.send_model_update_chunked(
            self.current_round,
            serialized_weights,
            len(self.x_train),
            float(final_loss),
            float(final_mse),
            float(final_mae),
            float(final_mape)
        )
        
        print(f"Client {self.client_id} sent model update for round {self.current_round}")
        print(f"Training metrics - Loss: {final_loss:.4f}, MSE: {final_mse:.4f}, "
              f"MAE: {final_mae:.4f}, MAPE: {final_mape:.4f}")
        
        # Small delay to ensure message is sent
        time.sleep(0.5)
        
        # Wait for global model after training
        print(f"Client {self.client_id} waiting for global model for round {self.current_round}...")
        self.wait_for_global_model()
    
    def wait_for_global_model(self):
        """Actively wait for global model after training (no timeout)"""
        check_count = 0
        
        while True:  # Wait indefinitely for global model
            samples = self.readers['global_model'].take()
            check_count += 1
            
            if check_count % 50 == 0:  # Log every 5 seconds
                print(f"Client {self.client_id} still waiting... (checked {check_count} times)")
            
            for sample in samples:
                if sample:
                    print(f"Client {self.client_id} received sample for round {sample.round} (expecting {self.current_round})")
                    if sample.round == self.current_round:
                        # Update local model with global weights
                        weights = self.deserialize_weights(sample.weights)
                        self.model.set_weights(weights)
                        print(f"Client {self.client_id} received global model for round {self.current_round}")
                        
                        # Evaluate immediately
                        print(f"Client {self.client_id} starting evaluation for round {self.current_round}...")
                        self.evaluate_model()
                        return
            time.sleep(0.1)
    
    def evaluate_model(self):
        """Evaluate model on test data and send metrics to server"""
        # Evaluate on test set
        results = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Extract metrics
        loss = results[0]
        mse = results[1]
        mae = results[2]
        mape = results[3]
        
        # Send metrics to server
        metrics = EvaluationMetrics(
            client_id=self.client_id,
            round=self.current_round,
            num_samples=len(self.x_test),
            loss=float(loss),
            mse=float(mse),
            mae=float(mae),
            mape=float(mape)
        )
        
        # Write with explicit return check
        result = self.writers['metrics'].write(metrics)
        
        # Wait to ensure message is sent
        time.sleep(0.5)
        
        print(f"Client {self.client_id} sent evaluation metrics for round {self.current_round}")
        print(f"Evaluation metrics - Loss: {loss:.4f}, MSE: {mse:.4f}, "
              f"MAE: {mae:.4f}, MAPE: {mape:.4f}\n")
    
    def cleanup(self):
        """Cleanup DDS resources"""
        if self.participant:
            # DomainParticipant will be automatically cleaned up
            self.participant = None
        print(f"Client {self.client_id} DDS resources cleaned up")


def main():
    """Main entry point"""
    # Load and prepare data
    # Detect environment and construct dataset path
    if os.path.exists('/app'):
        data_path = '/app/Client/Temperature_Regulation/Dataset/base_data_baseline_unique.csv'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        data_path = os.path.join(project_root, 'Client', 'Temperature_Regulation', 'Dataset', 'base_data_baseline_unique.csv')
    print(f"Dataset path: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} samples\n")
    
    # Create and run client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, df)
    client.run()


if __name__ == "__main__":
    main()
