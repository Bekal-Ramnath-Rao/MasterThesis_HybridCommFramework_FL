import numpy as np
import pandas as pd
import pickle
import time
import os
from pathlib import Path
import matplotlib.pyplot as plt

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

# Server Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence

# Convergence Settings (primary stopping criterion)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))


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


class FederatedLearningServer:
    def __init__(self, num_clients, num_rounds):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        
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
        
        # DDS entities
        self.participant = None
        self.readers = {}
        self.writers = {}
    
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
        
        # Create readers (for receiving from clients) with reliable QoS
        self.readers['registration'] = DataReader(self.participant, topic_registration, qos=reliable_qos)
        self.readers['model_update'] = DataReader(self.participant, topic_model_update, qos=reliable_qos)
        self.readers['metrics'] = DataReader(self.participant, topic_metrics, qos=reliable_qos)
        
        # Create writers (for sending to clients) with reliable QoS
        self.writers['config'] = DataWriter(self.participant, topic_config, qos=reliable_qos)
        self.writers['command'] = DataWriter(self.participant, topic_command, qos=reliable_qos)
        self.writers['global_model'] = DataWriter(self.participant, topic_global_model, qos=reliable_qos)
        self.writers['status'] = DataWriter(self.participant, topic_status, qos=reliable_qos)
        
        print("DDS setup complete with RELIABLE QoS\n")
        time.sleep(2)  # Allow time for discovery
    
    def publish_status(self):
        """Publish current server status"""
        status = ServerStatus(
            current_round=self.current_round,
            total_rounds=self.num_rounds,
            training_started=self.training_started,
            training_complete=self.training_complete,
            registered_clients=len(self.registered_clients)
        )
        self.writers['status'].write(status)
    
    def run(self):
        """Main server loop"""
        print("="*70)
        print("Starting Federated Learning Server (DDS)")
        print(f"DDS Domain ID: {DDS_DOMAIN_ID}")
        print(f"Number of Clients: {self.num_clients}")
        print(f"Number of Rounds: {self.num_rounds}")
        print("="*70)
        print("\nWaiting for clients to register...\n")
        
        # Setup DDS
        self.setup_dds()
        
        # Publish initial training config
        config = TrainingConfig(
            batch_size=self.training_config['batch_size'],
            local_epochs=self.training_config['local_epochs']
        )
        self.writers['config'].write(config)
        
        try:
            while not self.training_complete:
                # Publish current status
                self.publish_status()
                
                # Check for client registrations
                self.check_registrations()
                
                # Check for model updates
                self.check_model_updates()
                
                time.sleep(0.5)
            
            print("\nServer shutting down...")
            
        except KeyboardInterrupt:
            print("\n\nServer interrupted by user")
        finally:
            self.cleanup()
    
    def check_registrations(self):
        """Check for new client registrations"""
        samples = self.readers['registration'].take()
        
        for sample in samples:
            if sample:
                client_id = sample.client_id
                if client_id not in self.registered_clients:
                    self.registered_clients.add(client_id)
                    print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
                    
                    # If all clients registered, distribute initial global model and start training
                    if len(self.registered_clients) == self.num_clients and not self.training_started:
                        print("\nAll clients registered. Distributing initial global model...\n")
                        self.distribute_initial_model()
                        # Record training start time
                        self.start_time = time.time()
                        print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def distribute_initial_model(self):
        """Distribute initial global model to all clients"""
        self.training_started = True
        self.current_round = 1
        
        print(f"\n{'='*70}")
        print(f"Distributing Initial Global Model")
        print(f"{'='*70}\n")
        
        # Send initial global model to all clients
        initial_model = GlobalModel(
            round=0,  # Round 0 = initial model distribution
            weights=self.serialize_weights(self.global_weights)
        )
        self.writers['global_model'].write(initial_model)
        
        print("Initial global model sent to all clients")
        
        # Wait for clients to receive and set the initial model
        time.sleep(2)
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Send training command to start first round
        command = TrainingCommand(
            round=self.current_round,
            start_training=True,
            start_evaluation=False,
            training_complete=False
        )
        self.writers['command'].write(command)
    
    def check_model_updates(self):
        """Check for model updates from clients"""
        samples = self.readers['model_update'].take()
        
        for sample in samples:
            if sample and sample.round == self.current_round:
                client_id = sample.client_id
                
                if client_id not in self.client_updates:
                    self.client_updates[client_id] = {
                        'weights': self.deserialize_weights(sample.weights),
                        'num_samples': sample.num_samples,
                        'metrics': {
                            'loss': sample.loss,
                            'mse': sample.mse,
                            'mae': sample.mae,
                            'mape': sample.mape
                        }
                    }
                    
                    print(f"Received update from client {client_id} "
                          f"({len(self.client_updates)}/{self.num_clients})")
                    
                    # If all clients sent updates, aggregate
                    if len(self.client_updates) == self.num_clients:
                        self.aggregate_models()
    
    def check_evaluation_metrics(self):
        """Check for evaluation metrics from clients"""
        samples = self.readers['metrics'].take()
        
        for sample in samples:
            if sample:
                print(f"Server received metrics sample: client {sample.client_id}, round {sample.round} (current: {self.current_round})")
                
                if sample.round == self.current_round:
                    client_id = sample.client_id
                    
                    if client_id not in self.client_metrics:
                        self.client_metrics[client_id] = {
                            'num_samples': sample.num_samples,
                            'metrics': {
                                'loss': sample.loss,
                                'mse': sample.mse,
                                'mae': sample.mae,
                                'mape': sample.mape
                            }
                        }
                        
                        print(f"Received metrics from client {client_id} "
                              f"({len(self.client_metrics)}/{self.num_clients})")
                        
                        # If all clients sent metrics, aggregate and continue
                    if len(self.client_metrics) == self.num_clients:
                        self.aggregate_metrics()
                        self.continue_training()
    
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
        print(f"Sending global model to clients...\n")
        
        # Publish global model
        global_model = GlobalModel(
            round=self.current_round,
            weights=self.serialize_weights(self.global_weights)
        )
        self.writers['global_model'].write(global_model)
        
        # Send evaluation command
        time.sleep(1)
        command = TrainingCommand(
            round=self.current_round,
            start_training=False,
            start_evaluation=True,
            training_complete=False
        )
        self.writers['command'].write(command)
        
        self.evaluation_phase = True
        
        # Wait for all evaluation metrics
        self.wait_for_evaluation_metrics()
        
        # After receiving all metrics, aggregate and continue
        if len(self.client_metrics) == self.num_clients:
            self.aggregate_metrics()
            self.continue_training()
    
    def wait_for_evaluation_metrics(self):
        """Actively wait for evaluation metrics from all clients"""
        print(f"\nWaiting for evaluation metrics from {self.num_clients} clients...")
        timeout = 60  # 60 seconds timeout
        start_time = time.time()
        
        while len(self.client_metrics) < self.num_clients:
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for metrics. Received {len(self.client_metrics)}/{self.num_clients}")
                break
            
            samples = self.readers['metrics'].take()
            for sample in samples:
                if sample.round == self.current_round:
                    client_id = sample.client_id
                    if client_id not in self.client_metrics:
                        print(f"Received evaluation metrics from client {client_id}")
                        metrics_dict = {
                            'mse': sample.mse,
                            'mae': sample.mae,
                            'mape': sample.mape,
                            'loss': sample.loss
                        }
                        self.client_metrics[client_id] = {
                            'metrics': metrics_dict,
                            'num_samples': sample.num_samples
                        }
                        print(f"Progress: {len(self.client_metrics)}/{self.num_clients} clients")
            
            if len(self.client_metrics) < self.num_clients:
                time.sleep(0.1)  # Short sleep before next check
        
        if len(self.client_metrics) == self.num_clients:
            print(f"✓ All evaluation metrics received!")
    
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
            
            # Send completion signal
            command = TrainingCommand(
                round=self.current_round,
                start_training=False,
                start_evaluation=False,
                training_complete=True
            )
            self.writers['command'].write(command)
            
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
            
            time.sleep(2)
            
            # Send training command for next round
            command = TrainingCommand(
                round=self.current_round,
                start_training=True,
                start_evaluation=False,
                training_complete=False
            )
            self.writers['command'].write(command)
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            
            # Send completion signal
            command = TrainingCommand(
                round=self.current_round,
                start_training=False,
                start_evaluation=False,
                training_complete=True
            )
            self.writers['command'].write(command)
            
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
        plt.savefig(results_dir / 'dds_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Training metrics plot saved to {results_dir / 'dds_training_metrics.png'}")
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
        
        results_file = results_dir / 'dds_training_results.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Training results saved to {results_file}")
    
    def cleanup(self):
        """Cleanup DDS resources"""
        if self.participant:
            # DomainParticipant will be automatically cleaned up
            self.participant = None
        print("DDS resources cleaned up")


if __name__ == "__main__":
    server = FederatedLearningServer(NUM_CLIENTS, NUM_ROUNDS)
    server.run()
