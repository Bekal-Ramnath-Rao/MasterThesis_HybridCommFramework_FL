import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import os
import sys
import paho.mqtt.client as mqtt
from typing import List, Dict
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


# Server Configuration
# Auto-detect environment: Docker (/app exists) or local
MQTT_BROKER = os.getenv("MQTT_BROKER", 'mqtt-broker' if os.path.exists('/app') else 'localhost')
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))  # MQTT broker port
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence
NETWORK_SCENARIO = os.getenv("NETWORK_SCENARIO", "excellent")  # Network scenario for result filename

# Convergence Settings (primary stopping criterion)
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))  # Loss improvement threshold
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))  # Rounds to wait for improvement
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))  # Minimum rounds before checking convergence

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_REGISTER = "fl/client_register"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"
TOPIC_TRAINING_COMPLETE = "fl/training_complete"


class FederatedLearningServer:
    def __init__(self, min_clients, num_rounds, max_clients=100):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.num_clients = min_clients  # Start with minimum, will update as clients join
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.active_clients = set()
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
        self.training_started = False
        self.training_complete = False
        self.start_time = None
        self.convergence_time = None
        
        # Initialize quantization handler
        use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() == "true"
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
        self.training_config = {
            "batch_size": 32,
            "local_epochs": 5
        }
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id="fl_server")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
    
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
            print("Server connected to MQTT broker")
            # Subscribe to client topics
            result1, mid1 = self.mqtt_client.subscribe(TOPIC_CLIENT_REGISTER)
            print(f"Subscribed to {TOPIC_CLIENT_REGISTER} - Result: {result1}")
            
            result2, mid2 = self.mqtt_client.subscribe("fl/client/+/update")
            print(f"Subscribed to fl/client/+/update - Result: {result2}")
            
            result3, mid3 = self.mqtt_client.subscribe("fl/client/+/metrics")
            print(f"Subscribed to fl/client/+/metrics - Result: {result3}")
        else:
            print(f"Server failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            if msg.topic == TOPIC_CLIENT_REGISTER:
                self.handle_client_registration(msg.payload)
            elif "/update" in msg.topic:
                self.handle_client_update(msg.payload)
            elif "/metrics" in msg.topic:
                self.handle_client_metrics(msg.payload)
        except Exception as e:
            print(f"Server error handling message: {e}")
    
    def handle_client_registration(self, payload):
        """Handle client registration"""
        data = json.loads(payload.decode())
        client_id = data['client_id']
        
        # Check if already registered
        if client_id in self.registered_clients:
            print(f"Client {client_id} re-registered")
            return
        
        self.registered_clients.add(client_id)
        self.active_clients.add(client_id)
        print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))
        
        # Check if this is a late-joining client
        if self.training_started:
            self.active_clients.add(client_id)
            print(f"[LATE JOIN] Client {client_id} joining during round {self.current_round}")
            if len(self.registered_clients) > self.num_clients:
                self.update_client_count(len(self.registered_clients))
            if self.global_weights is not None:
                self.send_current_model_to_client(client_id)
            return
        
        if len(self.registered_clients) >= self.min_clients:
            print("\nAll clients registered. Distributing initial global model...\n")
            time.sleep(2)  # Give clients time to be ready
            self.distribute_initial_model()
            # Record training start time
            self.start_time = time.time()
            self.training_started = True
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def send_current_model_to_client(self, client_id):
        """Send current global model to a late-joining client"""
        try:
            print(f"📤 Sending current global model (round {self.current_round}) to late-joining client {client_id}")
            
            # Prepare payload with current model
            model_message = {
                'round': self.current_round,
                'weights': self.serialize_weights(self.global_weights),
                'model_config': self.model_config
            }
            
            # Broadcast to all clients on general topic (late-joiner will receive it)
            payload = json.dumps(model_message).encode()
            self.mqtt_client.publish(TOPIC_GLOBAL_MODEL, payload, qos=1)
            
            print(f"✅ Current model (round {self.current_round}) broadcast to late-joining client {client_id}")
            
        except Exception as e:
            print(f"❌ Error sending current model to client {client_id}: {e}")
    
    def mark_client_converged(self, client_id):
        """Remove converged client from active federation."""
        if client_id in self.active_clients:
            self.active_clients.discard(client_id)
            self.client_updates.pop(client_id, None)
            self.client_metrics.pop(client_id, None)
            print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
            if not self.active_clients:
                self.converged = True
                print("All clients converged. Ending training.")
                self.convergence_time = time.time() - self.start_time if self.start_time else 0
                self._finish_training()
    
    def handle_client_update(self, payload):
        """Handle model update from client"""
        data = json.loads(payload.decode())
        client_id = data['client_id']
        round_num = data['round']
        
        if client_id not in self.active_clients:
            return
        if float(data.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
            self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            # Check if update is compressed
            if 'compressed_data' in data and self.quantization_handler is not None:
                compressed_update = data['compressed_data']
                # If client sent serialized base64 string, decode and unpickle
                if isinstance(compressed_update, str):
                    try:
                        compressed_update = pickle.loads(base64.b64decode(compressed_update.encode('utf-8')))
                    except Exception as e:
                        print(f"Server error decoding compressed_data from client {client_id}: {e}")
                weights = self.quantization_handler.decompress_client_update(
                    client_id, 
                    compressed_update
                )
                print(f"Received and decompressed update from client {client_id}")
            else:
                weights = self.deserialize_weights(data['weights'])
            
            self.client_updates[client_id] = {
                'weights': weights,
                'num_samples': data['num_samples'],
                'metrics': data['metrics']
            }
            
            print(f"Received update from client {client_id} "
                  f"({len(self.client_updates)}/{len(self.active_clients)})")
            
            if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                self.aggregate_models()
    
    def handle_client_metrics(self, payload):
        """Handle evaluation metrics from client"""
        data = json.loads(payload.decode())
        client_id = data['client_id']
        round_num = data['round']
        
        if client_id not in self.active_clients:
            return
        if float(data.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
            self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            self.client_metrics[client_id] = {
                'num_samples': data['num_samples'],
                'metrics': data['metrics']
            }
            
            print(f"Received metrics from client {client_id} "
                  f"({len(self.client_metrics)}/{len(self.active_clients)})")
            
            if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                self.aggregate_metrics()
                self.continue_training()
    
    def distribute_initial_model(self):
        """Distribute initial global model architecture and weights to all clients"""
        # Send training configuration to all clients
        self.mqtt_client.publish(TOPIC_TRAINING_CONFIG, 
                            json.dumps(self.training_config))
        
        self.current_round = 1
        
        print(f"\n{'='*70}")
        print(f"Distributing Initial Global Model")
        print(f"{'='*70}\n")
        
        # Optionally compress global model
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.get_compression_stats(self.global_weights, compressed_data)
            print(f"Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
            
            # Serialize compressed data to JSON-safe base64 string
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            initial_model_message = {
                "round": 0,
                "quantized_data": serialized,
                "model_config": self.model_config
            }
        else:
            # Send initial global model with architecture configuration
            initial_model_message = {
                "round": 0,
                "weights": self.serialize_weights(self.global_weights),
            "model_config": {
                "architecture": "LSTM",
                "layers": [
                    {
                        "type": "LSTM",
                        "units": 50,
                        "activation": "relu",
                        "input_shape": [1, 4]
                    },
                    {
                        "type": "Dense",
                        "units": 1
                    }
                ],
                "compile_config": {
                    "loss": "mean_squared_error",
                    "optimizer": "adam",
                    "metrics": ["mse", "mae", "mape"]
                }
            }
        }
        
        self.mqtt_client.publish(TOPIC_GLOBAL_MODEL, 
                                json.dumps(initial_model_message))
        
        print("Initial global model (architecture + weights) sent to all clients")
        
        # Wait for clients to receive and build the model
        time.sleep(2)
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Signal clients to start training with the global model
        self.mqtt_client.publish(TOPIC_START_TRAINING,
                                json.dumps({"round": self.current_round}))
    
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
        
        # Optionally compress before sending
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            # Serialize compressed data to JSON-safe base64 string
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            global_model_message = {
                "round": self.current_round,
                "quantized_data": serialized,
                "model_config": self.model_config
            }
        else:
            global_model_message = {
                "round": self.current_round,
                "weights": self.serialize_weights(self.global_weights)
        }
        
        self.mqtt_client.publish(TOPIC_GLOBAL_MODEL, 
                                json.dumps(global_model_message))
        
        print(f"Aggregated global model from round {self.current_round} sent to all clients")
        
        # Request evaluation from clients
        time.sleep(1)
        self.mqtt_client.publish(TOPIC_START_EVALUATION,
                                json.dumps({"round": self.current_round}))
    
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
        
        print(f"\n{'='*70}")
        print(f"Round {self.current_round} - Aggregated Metrics:")
        print(f"  Loss: {aggregated_loss:.6f}")
        print(f"  MSE:  {aggregated_mse:.6f}")
        print(f"  MAE:  {aggregated_mae:.6f}")
        print(f"  MAPE: {aggregated_mape:.6f}")
        print(f"{'='*70}\n")
    
    def _finish_training(self):
        """Send training completion signal and save/plot results."""
        if self.training_complete:
            return
        self.training_complete = True
        print("Sending training completion signal to all clients...")
        self.mqtt_client.publish(TOPIC_TRAINING_COMPLETE, json.dumps({"message": "Training completed"}), qos=1)
        time.sleep(2)
        self.save_results()
        self.plot_results()
    
    def continue_training(self):
        """Continue to next round or finish training"""
        # Clear updates and metrics for next round
        self.client_updates.clear()
        self.client_metrics.clear()
        
        # Stop only when no active clients remain or max rounds reached (no server-side convergence)
        if len(self.active_clients) == 0:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            self.converged = True
            print("\n" + "="*70)
            print("All clients converged locally. Training complete.")
            print("="*70 + "\n")
            self._finish_training()
            return
        
        # Check if more rounds needed
        if self.current_round < self.num_rounds:
            self.current_round += 1
            
            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"{'='*70}\n")
            
            time.sleep(2)
            
            # Signal clients to start next training round
            self.mqtt_client.publish(TOPIC_START_TRAINING,
                                    json.dumps({"round": self.current_round}))
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            self._finish_training()
    
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
        
        # MSE Plot
        plt.subplot(1, 3, 1)
        plt.plot(self.ROUNDS, self.MSE, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.title('MSE over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # MAE Plot
        plt.subplot(1, 3, 2)
        plt.plot(self.ROUNDS, self.MAE, marker='s', linewidth=2, markersize=8, color='orange')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
        plt.title('MAE over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # MAPE Plot
        plt.subplot(1, 3, 3)
        plt.plot(self.ROUNDS, self.MAPE, marker='^', linewidth=2, markersize=8, color='green')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Mean Absolute Percentage Error (MAPE)', fontsize=12)
        plt.title('MAPE over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save to results folder
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'mqtt_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {results_dir / 'mqtt_training_metrics.png'}")
        plt.show(block=False)  # Non-blocking show
        
        # Disconnect and exit
        print("\nTraining complete. Disconnecting...")
        time.sleep(2)  # Give time for message delivery
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop()
        print("Server disconnected successfully.")
        import sys
        sys.exit(0)
    
    def save_results(self):
        """Save results to file"""
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        
        results = {
            "rounds": self.ROUNDS,
            "mse": self.MSE,
            "mae": self.MAE,
            "mape": self.MAPE,
            "loss": self.LOSS,
            "convergence_time_seconds": self.convergence_time,
            "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
            "total_rounds": len(self.ROUNDS),
            "num_clients": self.num_clients
        }
        
        results_file = results_dir / f'mqtt_{NETWORK_SCENARIO}_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def start(self):
        """Connect to MQTT broker and start server"""
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
    print(f"\n{'='*70}")
    print(f"Federated Learning Server with MQTT")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Clients: {MIN_CLIENTS} (min) - {MAX_CLIENTS} (max)")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"{'='*70}\n")
    print("Waiting for clients to connect...\n")
    
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.mqtt_client.disconnect()
