import numpy as np
import json
import pickle
import time
import grpc
from concurrent import futures
import threading
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

try:
    from quantization_server import ServerQuantizationHandler, QuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Quantization module not available")
    QUANTIZATION_AVAILABLE = False


# Add Protocols directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))

# Import generated gRPC code
import federated_learning_pb2
import federated_learning_pb2_grpc

# Server Configuration
GRPC_HOST = os.getenv("GRPC_HOST", "0.0.0.0")
GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))

# Convergence Settings
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
        self.clients_evaluated = set()
        self.global_weights = None
        self.model_config = None
        self.lock = threading.Lock()
        
        # Metrics storage for classification
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.start_time = None
        self.convergence_time = None
        
        # Initialize quantization handler (default: disabled unless explicitly enabled)
        uq_env = os.getenv("USE_QUANTIZATION", "false")
        use_quantization = uq_env.lower() in ("true", "1", "yes", "y")
        if use_quantization and QUANTIZATION_AVAILABLE:
            self.quantization_handler = ServerQuantizationHandler(QuantizationConfig())
            print("Server: Quantization enabled")
        else:
            self.quantization_handler = None
            if use_quantization and not QUANTIZATION_AVAILABLE:
                print("Server: Quantization requested but not available")
            else:
                print("Server: Quantization disabled")
        
        # Initialize global model
        self.initialize_global_model()
        
        # Training configuration
        # Training configuration broadcast to gRPC clients
        self.training_config = {
            "batch_size": 32,
            "local_epochs": 20  # Reduced from 20 for faster experiments
        }
        
        # Status flags
        self.training_started = False
        self.training_complete = False
        self.evaluation_phase = False
    
    def initialize_global_model(self):
        """Initialize the global CNN model for emotion recognition"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
        from tensorflow.keras.optimizers import Adam
        
        # Create CNN model for emotion recognition (7 classes, 48x48 grayscale images)
        model = Sequential()
        model.add(Input(shape=(48, 48, 1)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        # Get initial weights
        self.global_weights = model.get_weights()
        
        # Store model configuration for clients to rebuild
        self.model_config = {
            'input_shape': [48, 48, 1],
            'num_classes': 7,
            'layers': [
                {'type': 'Conv2D', 'filters': 32, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'Conv2D', 'filters': 64, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
                {'type': 'Dropout', 'rate': 0.25},
                {'type': 'Conv2D', 'filters': 128, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
                {'type': 'Conv2D', 'filters': 128, 'kernel_size': [3, 3], 'activation': 'relu'},
                {'type': 'MaxPooling2D', 'pool_size': [2, 2]},
                {'type': 'Dropout', 'rate': 0.25},
                {'type': 'Flatten'},
                {'type': 'Dense', 'units': 1024, 'activation': 'relu'},
                {'type': 'Dropout', 'rate': 0.5},
                {'type': 'Dense', 'units': 7, 'activation': 'softmax'}
            ]
        }
        
        print("\nGlobal CNN model initialized for emotion recognition")
        print(f"Model architecture: CNN with {len(self.global_weights)} weight layers")
        print(f"Input shape: 48x48x1 (grayscale images)")
        print(f"Output classes: 7 emotions")
    
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
            self.registered_clients.add(request.client_id)
            print(f"Client {request.client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
            
            if len(self.registered_clients) == self.num_clients and not self.training_started:
                print(f"\nAll {self.num_clients} clients registered!")
                print("Starting federated learning in 2 seconds...")
                # Start training in a separate thread after a short delay
                threading.Timer(2.0, self.start_training).start()
            
            return federated_learning_pb2.RegistrationResponse(
                success=True,
                message=f"Client {request.client_id} registered successfully"
            )
    
    def GetTrainingConfig(self, request, context):
        """Send training configuration to client"""
        return federated_learning_pb2.TrainingConfig(
            batch_size=self.training_config['batch_size'],
            local_epochs=self.training_config['local_epochs']
        )
    
    def GetGlobalModel(self, request, context):
        """Send global model weights and configuration to client"""
        with self.lock:
            if self.global_weights is None:
                print(f"Client {request.client_id}: Model not yet initialized")
                return federated_learning_pb2.GlobalModel(
                    round=0,
                    weights=b'',
                    available=False,
                    model_config=''
                )
            
            # Compress or serialize global weights
            if self.quantization_handler is not None:
                compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
                stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
                print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
                # Pickle compressed dict so gRPC bytes field can carry metadata
                serialized_weights = pickle.dumps(compressed_data)
            else:
                serialized_weights = self.serialize_weights(self.global_weights)
            model_config_json = json.dumps(self.model_config) if self.current_round == 0 else ''
            
            print(f"Client {request.client_id}: Sending global model (round {self.current_round}, {len(serialized_weights)/1024:.2f} KB)")
            
            return federated_learning_pb2.GlobalModel(
                round=self.current_round,
                weights=serialized_weights,
                available=True,
                model_config=model_config_json
            )
    
    def SendModelUpdate(self, request, context):
        """Receive model update from client"""
        with self.lock:
            client_id = request.client_id
            round_num = request.round
            
            if round_num != self.current_round:
                return federated_learning_pb2.UpdateResponse(
                    success=False,
                    message=f"Round mismatch: expected {self.current_round}, got {round_num}"
                )
            
            # Deserialize weights
            # Decompress or deserialize client weights (handle pickled compressed dicts)
            if request.weights:
                if self.quantization_handler is not None:
                    try:
                        candidate = pickle.loads(request.weights)
                        if isinstance(candidate, dict) and 'compressed_data' in candidate:
                            weights = self.quantization_handler.decompress_client_update(request.client_id, candidate)
                            print(f"Server: Received and decompressed update from client {request.client_id}")
                        else:
                            weights = candidate
                    except Exception:
                        # Fallback to regular deserialization
                        weights = self.deserialize_weights(request.weights)
                else:
                    weights = self.deserialize_weights(request.weights)
            else:
                weights = None
            
            # Extract metrics from map
            metrics = dict(request.metrics)
            
            # Store update
            self.client_updates[client_id] = {
                'weights': weights,
                'num_samples': request.num_samples,
                'metrics': metrics
            }
            
            print(f"Received update from client {client_id} for round {round_num}")
            print(f"  Training - Loss: {metrics.get('loss', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Progress: {len(self.client_updates)}/{self.num_clients} clients")
            
            # Check if all clients have submitted
            if len(self.client_updates) == self.num_clients:
                print(f"\nAll clients submitted updates for round {self.current_round}")
                # Aggregate in separate thread
                threading.Thread(target=self.aggregate_updates).start()
            
            return federated_learning_pb2.UpdateResponse(
                success=True,
                message=f"Update received for round {round_num}"
            )
    
    def SendMetrics(self, request, context):
        """Receive evaluation metrics from client"""
        with self.lock:
            client_id = request.client_id
            round_num = request.round
            
            # Extract metrics from map
            metrics = dict(request.metrics)
            
            self.client_metrics[client_id] = {
                'loss': metrics.get('loss', 0),
                'accuracy': metrics.get('accuracy', 0),
                'num_samples': request.num_samples
            }
            self.clients_evaluated.add(client_id)
            
            print(f"Received evaluation metrics from client {client_id}")
            print(f"  Loss: {metrics.get('loss', 0):.4f}, Accuracy: {metrics.get('accuracy', 0):.4f}")
            print(f"  Progress: {len(self.clients_evaluated)}/{self.num_clients} clients evaluated")
            
            # Check if all clients evaluated
            if len(self.clients_evaluated) == self.num_clients:
                print(f"\nAll clients completed evaluation for round {self.current_round}")
                # Aggregate metrics and start next round
                threading.Thread(target=self.aggregate_metrics).start()
            
            return federated_learning_pb2.MetricsResponse(
                success=True,
                message="Metrics received"
            )
    
    def CheckTrainingStatus(self, request, context):
        """Let clients check if training should start"""
        return federated_learning_pb2.TrainingStatus(
            should_train=self.training_started,
            training_complete=self.training_complete,
            round=self.current_round,
            should_evaluate=self.evaluation_phase
        )
    
    def start_training(self):
        """Start the federated learning process"""
        with self.lock:
            if self.training_started:
                return
            
            self.training_started = True
            self.start_time = time.time()
            self.current_round = 1
        
        print("\n" + "="*70)
        print("STARTING FEDERATED LEARNING")
        print("="*70)
        print(f"Number of clients: {self.num_clients}")
        print(f"Maximum rounds: {self.num_rounds}")
        print(f"Convergence threshold: {CONVERGENCE_THRESHOLD}")
        print(f"Patience: {CONVERGENCE_PATIENCE} rounds")
        print(f"Global model ready: {self.global_weights is not None}")
        print(f"Model layers: {len(self.global_weights) if self.global_weights else 0}")
        print("="*70 + "\n")
        print("Clients can now call GetGlobalModel to fetch initial weights...")
    
    def aggregate_updates(self):
        """Aggregate client model updates using FedAvg"""
        print(f"\n{'='*70}")
        print(f"AGGREGATING UPDATES - Round {self.current_round}")
        print(f"{'='*70}")
        
        # Calculate total samples
        total_samples = sum(update['num_samples'] for update in self.client_updates.values())
        
        # Weighted average of weights
        new_weights = []
        for i in range(len(self.global_weights)):
            layer_weights = []
            for client_id, update in self.client_updates.items():
                weight = update['weights'][i]
                num_samples = update['num_samples']
                layer_weights.append(weight * num_samples / total_samples)
            new_weights.append(np.sum(layer_weights, axis=0))
        
        # Update global weights
        with self.lock:
            self.global_weights = new_weights
            # Clear updates for next round
            self.client_updates.clear()
            # Enter evaluation phase
            self.evaluation_phase = True
        
        print(f"Global model updated for round {self.current_round}")
        print(f"Clients should now evaluate the model\n")
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics and check convergence"""
        # Calculate weighted average metrics
        total_samples = sum(m['num_samples'] for m in self.client_metrics.values())
        
        avg_loss = sum(m['loss'] * m['num_samples'] for m in self.client_metrics.values()) / total_samples
        avg_accuracy = sum(m['accuracy'] * m['num_samples'] for m in self.client_metrics.values()) / total_samples
        
        # Store metrics
        self.ROUNDS.append(self.current_round)
        self.LOSS.append(avg_loss)
        self.ACCURACY.append(avg_accuracy)
        
        print(f"\n{'='*70}")
        print(f"ROUND {self.current_round} SUMMARY")
        print(f"{'='*70}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Accuracy: {avg_accuracy:.4f}")
        
        # Check for convergence
        if avg_loss < self.best_loss - CONVERGENCE_THRESHOLD:
            self.best_loss = avg_loss
            self.rounds_without_improvement = 0
            print(f"✓ Loss improved! New best: {self.best_loss:.4f}")
        else:
            self.rounds_without_improvement += 1
            print(f"No significant improvement ({self.rounds_without_improvement}/{CONVERGENCE_PATIENCE})")
        
        # Check stopping criteria
        should_stop = False
        stop_reason = ""
        
        if self.current_round >= MIN_ROUNDS and self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
            should_stop = True
            stop_reason = f"Converged (no improvement for {CONVERGENCE_PATIENCE} rounds)"
            self.converged = True
            self.convergence_time = time.time() - self.start_time
        elif self.current_round >= self.num_rounds:
            should_stop = True
            stop_reason = f"Maximum rounds ({self.num_rounds}) reached"
        
        if should_stop:
            print(f"\n{'='*70}")
            print(f"TRAINING COMPLETE: {stop_reason}")
            print(f"{'='*70}")
            print(f"Final Loss: {avg_loss:.4f}")
            print(f"Final Accuracy: {avg_accuracy:.4f}")
            print(f"Total rounds: {self.current_round}")
            print(f"Total time: {time.time() - self.start_time:.2f} seconds")
            if self.converged:
                print(f"Convergence time: {self.convergence_time:.2f} seconds")
            print(f"{'='*70}\n")
            
            with self.lock:
                self.training_complete = True
                self.evaluation_phase = False
            
            # Save results and plot
            self.save_results()
            self.plot_results()
        else:
            print(f"{'='*70}\n")
            # Continue to next round
            with self.lock:
                self.current_round += 1
                self.client_metrics.clear()
                self.clients_evaluated.clear()
                self.evaluation_phase = False
            
            print(f"Starting Round {self.current_round}...\n")
    
    def save_results(self):
        """Save training results to JSON file"""
        results = {
            'rounds': self.ROUNDS,
            'loss': self.LOSS,
            'accuracy': self.ACCURACY,
            'converged': self.converged,
            'convergence_time': self.convergence_time if self.converged else None,
            'total_time': time.time() - self.start_time,
            'final_loss': self.LOSS[-1] if self.LOSS else None,
            'final_accuracy': self.ACCURACY[-1] if self.ACCURACY else None,
            'num_clients': self.num_clients,
            'config': self.training_config
        }
        
        # Create results directory if it doesn't exist
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        # Save to JSON
        filepath = results_dir / 'grpc_training_results.json'
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def plot_results(self):
        """Plot training metrics"""
        if not self.ROUNDS:
            print("No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Loss
        ax1.plot(self.ROUNDS, self.LOSS, 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Round', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Federated Learning - Loss Over Rounds (gRPC)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot Accuracy
        ax2.plot(self.ROUNDS, self.ACCURACY, 'g-', linewidth=2, marker='s')
        ax2.set_xlabel('Round', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Federated Learning - Accuracy Over Rounds (gRPC)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        filepath = results_dir / 'grpc_training_plot.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
        
        plt.show()


def serve():
    """Start the gRPC server"""
    # Set max message size to 100MB for large model weights
    options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    servicer = FederatedLearningServicer(NUM_CLIENTS, NUM_ROUNDS)
    federated_learning_pb2_grpc.add_FederatedLearningServicer_to_server(servicer, server)
    
    server.add_insecure_port(f'{GRPC_HOST}:{GRPC_PORT}')
    server.start()
    
    print(f"\n{'='*70}")
    print(f"Federated Learning gRPC Server - Emotion Recognition")
    print(f"{'='*70}")
    print(f"Server listening on {GRPC_HOST}:{GRPC_PORT}")
    print(f"Waiting for {NUM_CLIENTS} clients to register...")
    print(f"{'='*70}\n")
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop(0)


if __name__ == '__main__':
    serve()
