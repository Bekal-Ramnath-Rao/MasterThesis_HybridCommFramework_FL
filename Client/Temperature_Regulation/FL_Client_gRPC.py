import numpy as np
import pandas as pd
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import pickle
import time
import random
import grpc
import os
import sys
import logging
import threading

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# gRPC Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))


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
        
        # gRPC connection
        self.channel = None
        self.stub = None
        self.running = True
        
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
    
    def connect(self):
        """Connect to gRPC server"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to gRPC server at {GRPC_HOST}:{GRPC_PORT}...")
                self.channel = grpc.insecure_channel(f'{GRPC_HOST}:{GRPC_PORT}')
                self.stub = federated_learning_pb2_grpc.FederatedLearningStub(self.channel)
                
                # Test connection by registering
                response = self.stub.RegisterClient(
                    federated_learning_pb2.ClientRegistration(client_id=self.client_id)
                )
                
                if response.success:
                    print(f"Client {self.client_id} connected to gRPC server")
                    print(f"Client {self.client_id} registration: {response.message}")
                    
                    # Get training configuration
                    config = self.stub.GetTrainingConfig(
                        federated_learning_pb2.ConfigRequest(client_id=self.client_id)
                    )
                    self.training_config = {
                        "batch_size": config.batch_size,
                        "local_epochs": config.local_epochs
                    }
                    print(f"Client {self.client_id} received config: {self.training_config}")
                    
                    return True
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to gRPC server after {max_retries} attempts.")
                    print(f"\nPlease ensure:")
                    print(f"  1. gRPC server is running")
                    print(f"  2. Server address is correct: {GRPC_HOST}:{GRPC_PORT}")
                    raise
    
    def run(self):
        """Main client loop - poll server for instructions"""
        print(f"\nClient {self.client_id} waiting for server to start...\n")
        
        # First, get initial global model from server
        initial_model_received = False
        retry_count = 0
        max_connection_retries = 30
        
        while not initial_model_received and retry_count < max_connection_retries:
            try:
                global_model = self.stub.GetGlobalModel(
                    federated_learning_pb2.ModelRequest(
                        client_id=self.client_id,
                        round=0  # Request initial model
                    )
                )
                
                if global_model.available:
                    # Update local model with initial global weights
                    weights = pickle.loads(global_model.weights)
                    self.model.set_weights(weights)
                    if global_model.round == 0:
                        print(f"Client {self.client_id} received initial global model from server")
                        self.current_round = 0
                        initial_model_received = True
                    else:
                        print(f"Client {self.client_id} received global model for round {global_model.round}")
                        self.current_round = global_model.round
                        initial_model_received = True
                else:
                    time.sleep(0.5)  # Wait for server to initialize
            except grpc.RpcError as e:
                retry_count += 1
                if retry_count == 1:
                    print(f"Client {self.client_id} waiting for server to start...")
                if retry_count % 10 == 0:
                    print(f"  Still waiting... (attempt {retry_count}/{max_connection_retries})")
                time.sleep(2)
            except Exception as e:
                retry_count += 1
                print(f"Client {self.client_id} unexpected error: {e}")
                time.sleep(2)
        
        if not initial_model_received:
            print(f"\nClient {self.client_id} failed to connect to server after {max_connection_retries} attempts.")
            print("Please ensure the gRPC server is running.")
            self.running = False
            return
        
        print(f"\nClient {self.client_id} ready for training...\n")
        
        while self.running:
            try:
                # Check training status
                try:
                    status = self.stub.CheckTrainingStatus(
                        federated_learning_pb2.StatusRequest(
                            client_id=self.client_id,
                            current_round=self.current_round
                        )
                    )
                except grpc.RpcError as e:
                    if e.code() == grpc.StatusCode.UNAVAILABLE:
                        # Server has shut down, exit gracefully
                        print(f"\nClient {self.client_id} - Server has shut down.")
                        print("Training completed. Disconnecting...")
                        self.running = False
                        break
                    else:
                        raise  # Re-raise other RPC errors
                
                if status.training_complete:
                    print(f"\nClient {self.client_id} - Training completed!")
                    print("Disconnecting from server...")
                    self.running = False
                    break
                
                if status.should_train:
                    if self.current_round == 0 and status.round == 1:
                        # First training round with initial global model
                        self.current_round = status.round
                        print(f"\nClient {self.client_id} starting training for round {status.round} with initial global model...")
                        self.train_local_model()
                    elif status.round > self.current_round:
                        # Subsequent rounds
                        self.current_round = status.round
                        print(f"\nClient {self.client_id} starting training for round {status.round}...")
                        self.train_local_model()
                    time.sleep(1)  # Brief pause before checking for evaluation
                
                elif status.should_evaluate:
                    # Get global model
                    global_model = self.stub.GetGlobalModel(
                        federated_learning_pb2.ModelRequest(
                            client_id=self.client_id,
                            round=self.current_round
                        )
                    )
                    
                    if global_model.available:
                        # Update local model with global weights
                        weights = pickle.loads(global_model.weights)
                        self.model.set_weights(weights)
                        print(f"Client {self.client_id} received aggregated global model for round {self.current_round}")
                        
                        # Evaluate
                        print(f"Client {self.client_id} starting evaluation for round {self.current_round}...")
                        self.evaluate_model()
                
                else:
                    # Wait before polling again
                    time.sleep(0.5)
                    
            except KeyboardInterrupt:
                print(f"\nClient {self.client_id} shutting down...")
                self.running = False
                break
            except grpc.RpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    # Server has shut down, exit gracefully
                    print(f"\nClient {self.client_id} - Server unavailable, disconnecting...")
                    self.running = False
                    break
                else:
                    print(f"Client {self.client_id} RPC error: {e.code()}")
                    time.sleep(2)
            except Exception as e:
                print(f"Client {self.client_id} error: {type(e).__name__}")
                time.sleep(2)
        
        self.disconnect()
    
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
        serialized_weights = pickle.dumps(weights)
        
        # Introduce random delay before sending model update
        delay = random.uniform(0.5, 3.0)  # Random delay between 0.5 and 3.0 seconds
        print(f"Client {self.client_id} waiting {delay:.2f} seconds before sending update...")
        time.sleep(delay)
        
        # Send model update to server
        response = self.stub.SendModelUpdate(
            federated_learning_pb2.ModelUpdate(
                client_id=self.client_id,
                round=self.current_round,
                weights=serialized_weights,
                num_samples=len(self.x_train),
                metrics={
                    'loss': float(final_loss),
                    'mse': float(final_mse),
                    'mae': float(final_mae),
                    'mape': float(final_mape)
                }
            )
        )
        
        if response.success:
            print(f"Client {self.client_id} sent model update for round {self.current_round}")
            print(f"Training metrics - Loss: {final_loss:.4f}, MSE: {final_mse:.4f}, "
                  f"MAE: {final_mae:.4f}, MAPE: {final_mape:.4f}")
        else:
            print(f"Client {self.client_id} failed to send update: {response.message}")
    
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
        response = self.stub.SendMetrics(
            federated_learning_pb2.EvaluationMetrics(
                client_id=self.client_id,
                round=self.current_round,
                num_samples=len(self.x_test),
                metrics={
                    'loss': float(loss),
                    'mse': float(mse),
                    'mae': float(mae),
                    'mape': float(mape)
                }
            )
        )
        
        if response.success:
            print(f"Client {self.client_id} sent evaluation metrics for round {self.current_round}")
            print(f"Evaluation metrics - Loss: {loss:.4f}, MSE: {mse:.4f}, "
                  f"MAE: {mae:.4f}, MAPE: {mape:.4f}\n")
        else:
            print(f"Client {self.client_id} failed to send metrics: {response.message}")
    
    def disconnect(self):
        """Close gRPC channel"""
        if self.channel:
            self.channel.close()
            print(f"Client {self.client_id} disconnected from server")


def main():
    """Main entry point"""
    print("="*60)
    print(f"Starting Federated Learning Client {CLIENT_ID}")
    print(f"Server: {GRPC_HOST}:{GRPC_PORT}")
    print("="*60)
    print()
    
    # Load and prepare data
    data_path = os.path.join(os.path.dirname(__file__), '../Dataset/base_data_baseline_unique.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} samples\n")
    
    # Create and connect client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, df)
    client.connect()
    
    # Run federated learning
    client.run()


if __name__ == "__main__":
    main()
