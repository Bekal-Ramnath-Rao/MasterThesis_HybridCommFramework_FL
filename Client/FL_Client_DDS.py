import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
import os
import logging

# Add CycloneDDS DLL path
cyclone_path = r"C:\Masters_Infotech\Semester_5\MT_SW_Addons\vcpkg\buildtrees\cyclonedds\x64-windows-rel\bin"
if cyclone_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = cyclone_path + os.pathsep + os.environ.get('PATH', '')

from cyclonedds.domain import DomainParticipant
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
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.current_round = 0
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        self.running = True
        
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
        """Serialize model weights for DDS transmission"""
        serialized = pickle.dumps(weights)
        # Convert bytes to list of ints for DDS
        return list(serialized)
    
    def deserialize_weights(self, serialized_weights):
        """Deserialize model weights received from DDS"""
        # Convert list of ints back to bytes
        return pickle.loads(bytes(serialized_weights))
    
    def setup_dds(self):
        """Initialize DDS participant, topics, readers, and writers"""
        print(f"Setting up DDS on domain {DDS_DOMAIN_ID}...")
        
        # Create domain participant
        self.participant = DomainParticipant(DDS_DOMAIN_ID)
        
        # Create QoS policy for reliable communication
        reliable_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
            Policy.History.KeepAll,
            Policy.Durability.TransientLocal
        )
        
        # Create topics
        topic_registration = Topic(self.participant, "ClientRegistration", ClientRegistration)
        topic_config = Topic(self.participant, "TrainingConfig", TrainingConfig)
        topic_command = Topic(self.participant, "TrainingCommand", TrainingCommand)
        topic_global_model = Topic(self.participant, "GlobalModel", GlobalModel)
        topic_model_update = Topic(self.participant, "ModelUpdate", ModelUpdate)
        topic_metrics = Topic(self.participant, "EvaluationMetrics", EvaluationMetrics)
        topic_status = Topic(self.participant, "ServerStatus", ServerStatus)
        
        # Create readers (for receiving from server) with reliable QoS
        self.readers['config'] = DataReader(self.participant, topic_config, qos=reliable_qos)
        self.readers['command'] = DataReader(self.participant, topic_command, qos=reliable_qos)
        self.readers['global_model'] = DataReader(self.participant, topic_global_model, qos=reliable_qos)
        self.readers['status'] = DataReader(self.participant, topic_status, qos=reliable_qos)
        
        # Create writers (for sending to server) with reliable QoS
        self.writers['registration'] = DataWriter(self.participant, topic_registration, qos=reliable_qos)
        self.writers['model_update'] = DataWriter(self.participant, topic_model_update, qos=reliable_qos)
        self.writers['metrics'] = DataWriter(self.participant, topic_metrics, qos=reliable_qos)
        
        print(f"Client {self.client_id} DDS setup complete with RELIABLE QoS")
        time.sleep(2)  # Allow time for discovery
        
        # Register with server
        registration = ClientRegistration(
            client_id=self.client_id,
            message=f"Client {self.client_id} ready"
        )
        self.writers['registration'].write(registration)
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
                # Check for training commands
                self.check_commands()
                
                time.sleep(0.1)  # Check more frequently
                
        except KeyboardInterrupt:
            print(f"\nClient {self.client_id} shutting down...")
        finally:
            self.cleanup()
    
    def get_training_config(self):
        """Get training configuration from server"""
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
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
                
    def check_commands(self):
        """Check for training commands from server"""
        samples = self.readers['command'].take()
        
        for sample in samples:
            if sample:
                if sample.training_complete:
                    print(f"\nClient {self.client_id} - Training completed!")
                    print("Disconnecting from server...")
                    self.running = False
                    return
                
                # Check if we're ready for this round (should have received global model first)
                if sample.start_training:
                    if self.current_round == 0 and sample.round == 1:
                        # First training round with initial global model
                        self.current_round = sample.round
                        print(f"\nClient {self.client_id} starting training for round {self.current_round} with initial global model...")
                        self.train_local_model()
                    elif sample.round > self.current_round:
                        # Subsequent rounds
                        self.current_round = sample.round
                        print(f"\nClient {self.client_id} starting training for round {self.current_round}...")
                        self.train_local_model()
    
    def check_global_model(self):
        """Check for global model updates from server"""
        samples = self.readers['global_model'].take()
        
        for sample in samples:
            if sample:
                # Update local model with global weights
                weights = self.deserialize_weights(sample.weights)
                self.model.set_weights(weights)
                
                if sample.round == 0:
                    # Initial model from server
                    print(f"Client {self.client_id} received initial global model from server")
                    self.current_round = 0
                elif sample.round == self.current_round:
                    # Updated model after aggregation
                    print(f"Client {self.client_id} received global model for round {self.current_round}")
                    
                    # Evaluate immediately after receiving global model
                    print(f"Client {self.client_id} starting evaluation for round {self.current_round}...")
                    self.evaluate_model()
    
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
        serialized_weights = self.serialize_weights(weights)
        
        # Send model update to server
        update = ModelUpdate(
            client_id=self.client_id,
            round=self.current_round,
            weights=serialized_weights,
            num_samples=len(self.x_train),
            loss=float(final_loss),
            mse=float(final_mse),
            mae=float(final_mae),
            mape=float(final_mape)
        )
        self.writers['model_update'].write(update)
        
        print(f"Client {self.client_id} sent model update for round {self.current_round}")
        print(f"Training metrics - Loss: {final_loss:.4f}, MSE: {final_mse:.4f}, "
              f"MAE: {final_mae:.4f}, MAPE: {final_mape:.4f}")
        
        # Small delay to ensure message is sent
        time.sleep(0.5)
        
        # Wait for global model after training
        print(f"Client {self.client_id} waiting for global model for round {self.current_round}...")
        self.wait_for_global_model()
    
    def wait_for_global_model(self):
        """Actively wait for global model after training"""
        timeout = 30
        start_time = time.time()
        check_count = 0
        
        while time.time() - start_time < timeout:
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
        
        print(f"Client {self.client_id} WARNING: Timeout waiting for global model round {self.current_round}")
    
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
    data_path = os.path.join(os.path.dirname(__file__), 'Dataset/base_data_baseline_unique.csv')
    
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
