import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import grpc
from concurrent import futures
import threading
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# Server Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "0.0.0.0")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence

# Convergence Settings (primary stopping criterion)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))


class FederatedLearningServicer(federated_learning_pb2_grpc.FederatedLearningServicer):
    def __init__(self, num_clients, num_rounds):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        self.lock = threading.Lock()
        
        # Metrics storage
        self.MSE = []
        self.MAE = []
        self.MAPE = []
        self.LOSS = []
        self.ROUNDS = []
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.start_time = None
        self.convergence_time = None
        
        # Initialize global model
        self.initialize_global_model()
        
        # Training configuration
        self.training_config = {
            "batch_size": 32,
            "local_epochs": 20
        }
        
        # Status flags
        self.training_started = False
        self.training_complete = False
        self.evaluation_phase = False
    
    def initialize_global_model(self):
        """Initialize the global model structure (LSTM for FL)"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM
        
        # Create the same LSTM model structure as clients
        # Input shape: (1, 4) - 1 time step, 4 features
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(1, 4)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', 
                     metrics=['mse', 'mae', 'mape'])
        
        # Get initial weights
        self.global_weights = model.get_weights()
        
        print("\nGlobal model initialized with random weights")
        print(f"Model architecture: LSTM(50) -> Dense(1)")
        print(f"Number of weight layers: {len(self.global_weights)}")
    
    def serialize_weights(self, weights):
        """Serialize model weights for gRPC transmission"""
        serialized = pickle.dumps(weights)
        return serialized
    
    def deserialize_weights(self, serialized_weights):
        """Deserialize model weights received from gRPC"""
        weights = pickle.loads(serialized_weights)
        return weights
    
    def RegisterClient(self, request, context):
        """Handle client registration"""
        with self.lock:
            client_id = request.client_id
            self.registered_clients.add(client_id)
            print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
            
            # If all clients registered, start distributing initial global model
            if len(self.registered_clients) == self.num_clients and not self.training_started:
                print("\nAll clients registered. Distributing initial global model...\n")
                self.training_started = True
                self.current_round = 1
                
                print(f"\n{'='*70}")
                print(f"Distributing Initial Global Model")
                print(f"{'='*70}\n")
                print("Initial global model ready for clients")
                
                print(f"\n{'='*70}")
                print(f"Starting Round {self.current_round}/{self.num_rounds}")
                print(f"{'='*70}\n")
            
            return federated_learning_pb2.RegistrationResponse(
                success=True,
                message=f"Client {client_id} registered successfully"
            )
    
    def GetTrainingConfig(self, request, context):
        """Return training configuration"""
        return federated_learning_pb2.TrainingConfig(
            batch_size=self.training_config["batch_size"],
            local_epochs=self.training_config["local_epochs"]
        )
    
    def CheckTrainingStatus(self, request, context):
        """Check if client should train or evaluate"""
        with self.lock:
            client_id = request.client_id
            client_round = request.current_round
            
            # Check if training is complete
            if self.training_complete:
                return federated_learning_pb2.TrainingStatus(
                    should_train=False,
                    round=self.current_round,
                    should_evaluate=False,
                    training_complete=True
                )
            
            # Check if client should start training
            if self.training_started and client_round == 0 and self.current_round == 1:
                return federated_learning_pb2.TrainingStatus(
                    should_train=True,
                    round=self.current_round,
                    should_evaluate=False,
                    training_complete=False
                )
            
            # Check if client should evaluate (after global model is ready)
            if self.evaluation_phase and client_round == self.current_round and self.global_weights is not None:
                return federated_learning_pb2.TrainingStatus(
                    should_train=False,
                    round=self.current_round,
                    should_evaluate=True,
                    training_complete=False
                )
            
            # Check if client should train next round
            if not self.evaluation_phase and client_round < self.current_round:
                return federated_learning_pb2.TrainingStatus(
                    should_train=True,
                    round=self.current_round,
                    should_evaluate=False,
                    training_complete=False
                )
            
            # No action needed
            return federated_learning_pb2.TrainingStatus(
                should_train=False,
                round=self.current_round,
                should_evaluate=False,
                training_complete=False
            )
    
    def GetGlobalModel(self, request, context):
        """Send global model to client"""
        with self.lock:
            # Record training start time on first client request
            if self.start_time is None and request.round == 0:
                self.start_time = time.time()
                print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if self.global_weights is not None:
                # For initial distribution (client_round=0), send round 0
                # For updates after aggregation, send current_round
                round_to_send = 0 if request.round == 0 and self.training_started else self.current_round
                
                return federated_learning_pb2.GlobalModel(
                    round=round_to_send,
                    weights=self.serialize_weights(self.global_weights),
                    available=True
                )
            else:
                return federated_learning_pb2.GlobalModel(
                    round=self.current_round,
                    weights=b'',
                    available=False
                )
    
    def SendModelUpdate(self, request, context):
        """Receive model update from client"""
        with self.lock:
            client_id = request.client_id
            round_num = request.round
            
            if round_num == self.current_round:
                self.client_updates[client_id] = {
                    'weights': self.deserialize_weights(request.weights),
                    'num_samples': request.num_samples,
                    'metrics': dict(request.metrics)
                }
                
                print(f"Received update from client {client_id} "
                      f"({len(self.client_updates)}/{self.num_clients})")
                
                # If all clients sent updates, aggregate
                if len(self.client_updates) == self.num_clients:
                    self.aggregate_models()
                
                return federated_learning_pb2.UpdateResponse(
                    success=True,
                    message="Model update received"
                )
            else:
                return federated_learning_pb2.UpdateResponse(
                    success=False,
                    message=f"Round mismatch: received {round_num}, current {self.current_round}"
                )
    
    def SendMetrics(self, request, context):
        """Receive evaluation metrics from client"""
        with self.lock:
            client_id = request.client_id
            round_num = request.round
            
            if round_num == self.current_round:
                self.client_metrics[client_id] = {
                    'num_samples': request.num_samples,
                    'metrics': dict(request.metrics)
                }
                
                print(f"Received metrics from client {client_id} "
                      f"({len(self.client_metrics)}/{self.num_clients})")
                
                # If all clients sent metrics, aggregate and continue
                if len(self.client_metrics) == self.num_clients:
                    self.aggregate_metrics()
                    self.continue_training()
                
                return federated_learning_pb2.MetricsResponse(
                    success=True,
                    message="Metrics received"
                )
            else:
                return federated_learning_pb2.MetricsResponse(
                    success=False,
                    message=f"Round mismatch: received {round_num}, current {self.current_round}"
                )
    
    def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] 
                          for update in self.client_updates.values())
        
        # Initialize aggregated weights
        aggregated_weights = []
        
        # Get the structure from first client
        first_client_weights = list(self.client_updates.values())[0]['weights']
        
        # For each layer
        for layer_idx in range(len(first_client_weights)):
            # Weighted average of weights from all clients
            layer_weights = np.zeros_like(first_client_weights[layer_idx])
            
            for client_id, update in self.client_updates.items():
                weight = update['num_samples'] / total_samples
                layer_weights += weight * update['weights'][layer_idx]
            
            aggregated_weights.append(layer_weights)
        
        self.global_weights = aggregated_weights
        
        print(f"Aggregated global model from round {self.current_round}")
        print(f"Global model ready for clients\n")
        
        # Switch to evaluation phase
        self.evaluation_phase = True
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics from all clients"""
        print(f"\nAggregating metrics from {len(self.client_metrics)} clients...")
        
        # Calculate total samples
        total_samples = sum(metric['num_samples'] 
                          for metric in self.client_metrics.values())
        
        # Weighted average of metrics
        aggregated_mse = sum(metric['metrics']['mse'] * metric['num_samples']
                            for metric in self.client_metrics.values()) / total_samples
        
        aggregated_mae = sum(metric['metrics']['mae'] * metric['num_samples']
                            for metric in self.client_metrics.values()) / total_samples
        
        aggregated_mape = sum(metric['metrics']['mape'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        aggregated_loss = sum(metric['metrics']['loss'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        # Store metrics
        self.MSE.append(aggregated_mse)
        self.MAE.append(aggregated_mae)
        self.MAPE.append(aggregated_mape)
        self.LOSS.append(aggregated_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"\nRound {self.current_round} Aggregated Metrics:")
        print(f"  Loss: {aggregated_loss:.4f}")
        print(f"  MSE: {aggregated_mse:.4f}")
        print(f"  MAE: {aggregated_mae:.4f}")
        print(f"  MAPE: {aggregated_mape:.4f}\n")
    
    def continue_training(self):
        """Continue to next round or finish training"""
        # Clear updates and metrics for next round
        self.client_updates.clear()
        self.client_metrics.clear()
        self.evaluation_phase = False
        
        # Check for convergence (early stopping)
        if self.current_round >= MIN_ROUNDS and self.check_convergence():
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("CONVERGENCE ACHIEVED!")
            print(f"Training stopped early at round {self.current_round}/{self.num_rounds}")
            print(f"Loss improvement below threshold for {CONVERGENCE_PATIENCE} consecutive rounds")
            print(f"Time to Convergence: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            self.converged = True
            self.training_complete = True
            self.plot_results()
            self.save_results()
            return
        
        # Check if more rounds needed
        if self.current_round < self.num_rounds:
            self.current_round += 1
            
            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"{'='*70}\n")
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            self.training_complete = True
            self.plot_results()
            self.save_results()
    
    def check_convergence(self):
        """Check if model has converged based on loss improvement"""
        if len(self.LOSS) == 0:
            return False
        
        current_loss = self.LOSS[-1]
        
        # Check if loss improved by at least the threshold
        improvement = self.best_loss - current_loss
        
        if improvement > CONVERGENCE_THRESHOLD:
            # Significant improvement
            self.best_loss = current_loss
            self.rounds_without_improvement = 0
            print(f"  → Loss improved by {improvement:.6f} (threshold: {CONVERGENCE_THRESHOLD})")
            return False
        else:
            # No significant improvement
            self.rounds_without_improvement += 1
            print(f"  → No significant improvement (improvement: {improvement:.6f}, threshold: {CONVERGENCE_THRESHOLD})")
            print(f"  → Rounds without improvement: {self.rounds_without_improvement}/{CONVERGENCE_PATIENCE}")
            
            if self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
                return True
            return False
    
    def plot_results(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.plot(self.ROUNDS, self.LOSS, 'b-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        plt.subplot(1, 4, 2)
        plt.plot(self.ROUNDS, self.MSE, 'r-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error')
        plt.grid(True)
        
        plt.subplot(1, 4, 3)
        plt.plot(self.ROUNDS, self.MAE, 'g-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('MAE')
        plt.title('Mean Absolute Error')
        plt.grid(True)
        
        plt.subplot(1, 4, 4)
        plt.plot(self.ROUNDS, self.MAPE, 'm-', marker='o')
        plt.xlabel('Round')
        plt.ylabel('MAPE')
        plt.title('Mean Absolute Percentage Error')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'grpc_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to {results_dir / 'grpc_training_metrics.png'}")
        plt.show()
        
        print("\nPlot closed. Training complete.")
    
    def save_results(self):
        """Save training results to CSV"""
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results_df = pd.DataFrame({
            'Round': self.ROUNDS,
            'Loss': self.LOSS,
            'MSE': self.MSE,
            'MAE': self.MAE,
            'MAPE': self.MAPE
        })
        
        # Add summary row with convergence time
        summary_df = pd.DataFrame([{
            'Round': 'SUMMARY',
            'Loss': self.LOSS[-1] if self.LOSS else None,
            'MSE': self.MSE[-1] if self.MSE else None,
            'MAE': self.MAE[-1] if self.MAE else None,
            'MAPE': self.MAPE[-1] if self.MAPE else None
        }])
        summary_df['Total Rounds'] = len(self.ROUNDS)
        summary_df['Num Clients'] = self.num_clients
        summary_df['Convergence Time (seconds)'] = self.convergence_time
        summary_df['Convergence Time (minutes)'] = self.convergence_time / 60 if self.convergence_time else None
        
        results_df = pd.concat([results_df, summary_df], ignore_index=True)
        
        results_file = results_dir / 'grpc_training_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Training results saved to {results_file}")


def serve():
    """Start gRPC server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FederatedLearningServicer(NUM_CLIENTS, NUM_ROUNDS)
    federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    
    server_address = f"{GRPC_HOST}:{GRPC_PORT}"
    server.add_insecure_port(server_address)
    server.start()
    
    print("="*70)
    print("Starting Federated Learning Server (gRPC)")
    print(f"Server Address: {server_address}")
    print(f"Number of Clients: {NUM_CLIENTS}")
    print(f"Number of Rounds: {NUM_ROUNDS}")
    print("="*70)
    print("\nWaiting for clients to register...\n")
    
    try:
        # Keep server running until training is complete
        while not servicer.training_complete:
            time.sleep(1)
        
        # Give clients time to receive completion signal
        print("\nTraining complete. Shutting down server in 5 seconds...")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.stop(0)


if __name__ == "__main__":
    serve()
