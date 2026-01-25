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
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))  # High default - will stop at convergence

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
    def __init__(self, num_clients, num_rounds):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = set()
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        
        # Metrics storage (for classification)
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        
        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.start_time = None
        self.convergence_time = None
        
        # Training timeout tracking (prevent waiting forever for stuck clients)
        self.round_start_time = None
        self.training_timeout = int(os.getenv("TRAINING_TIMEOUT", "600"))  # 10 minutes default
        
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
        # Training configuration broadcast to MQTT clients
        self.training_config = {
            "batch_size": 32,
            "local_epochs": 20  # Reduced from 20 for faster experiments (configurable via env)
        }
        
        # Initialize MQTT client with fair comparison settings
        # clean_session=True for stateless operation (like other protocols)
        self.mqtt_client = mqtt.Client(client_id="fl_server", protocol=mqtt.MQTTv311, clean_session=True)
        # Set max packet size to 12MB+ for FL model weights
        self.mqtt_client._max_packet_size = 15 * 1024 * 1024  # 15MB with overhead
        # Set keepalive to 60s (aligned with AMQP/gRPC/QUIC/DDS)
        self.mqtt_client.keepalive = 60
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

    def initialize_global_model(self):
        """Initialize the global model structure (CNN for Emotion Recognition)"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers.schedules import ExponentialDecay
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        
        # Define model configuration for clients
        self.model_config = {
            "input_shape": [48, 48, 1],
            "layers": [
                {"type": "Conv2D", "filters": 32, "kernel_size": [3, 3], "activation": "relu"},
                {"type": "Conv2D", "filters": 64, "kernel_size": [3, 3], "activation": "relu"},
                {"type": "MaxPooling2D", "pool_size": [2, 2]},
                {"type": "Dropout", "rate": 0.25},
                {"type": "Conv2D", "filters": 128, "kernel_size": [3, 3], "activation": "relu"},
                {"type": "MaxPooling2D", "pool_size": [2, 2]},
                {"type": "Conv2D", "filters": 128, "kernel_size": [3, 3], "activation": "relu"},
                {"type": "MaxPooling2D", "pool_size": [2, 2]},
                {"type": "Dropout", "rate": 0.25},
                {"type": "Flatten"},
                {"type": "Dense", "units": 1024, "activation": "relu"},
                {"type": "Dropout", "rate": 0.5},
                {"type": "Dense", "units": 7, "activation": "softmax"}
            ],
            "num_classes": 7
        }
        
        # Create the CNN model structure
        model = Sequential()
        model.add(Input(shape=(48, 48, 1)))  # Input layer
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
        
        # Get initial weights
        self.global_weights = model.get_weights()
        
        print("\nGlobal model initialized with random weights")
        print(f"Model architecture: CNN for Emotion Recognition (7 classes)")
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
            # Only log first connection, not reconnections
            if not hasattr(self, '_connected_once'):
                print("Server connected to MQTT broker")
                self._connected_once = True
            
            # Subscribe to client topics with QoS 1 for reliable delivery
            self.mqtt_client.subscribe(TOPIC_CLIENT_REGISTER, qos=1)
            self.mqtt_client.subscribe("fl/client/+/update", qos=1)
            self.mqtt_client.subscribe("fl/client/+/metrics", qos=1)
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
        self.registered_clients.add(client_id)
        print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
        
        # If all clients registered, distribute initial global model and start federated learning
        if len(self.registered_clients) == self.num_clients:
            print("\nAll clients registered. Distributing initial global model...\n")
            time.sleep(2)  # Give clients time to be ready
            self.distribute_initial_model()
            # Record training start time
            self.start_time = time.time()
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def handle_client_update(self, payload):
        """Handle model update from client"""
        data = json.loads(payload.decode())
        client_id = data['client_id']
        round_num = data['round']
        
        if round_num == self.current_round:
            # Check if update is compressed
            if 'compressed_data' in data and self.quantization_handler is not None:
                # Decompress quantized weights (handle base64-serialized payloads)
                compressed_update = data['compressed_data']
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
                # Standard deserialization
                weights = self.deserialize_weights(data['weights'])
            
            self.client_updates[client_id] = {
                'weights': weights,
                'num_samples': data['num_samples'],
                'metrics': data['metrics']
            }
            
            print(f"Received update from client {client_id} "
                  f"({len(self.client_updates)}/{self.num_clients})")
            
            # If all clients sent updates, aggregate
            if len(self.client_updates) == self.num_clients:
                self.aggregate_models()
    
    def handle_client_metrics(self, payload):
        """Handle evaluation metrics from client"""
        data = json.loads(payload.decode())
        client_id = data['client_id']
        round_num = data['round']
        
        if round_num == self.current_round:
            self.client_metrics[client_id] = {
                'num_samples': data['num_samples'],
                'metrics': data['metrics']
            }
            
            print(f"Received metrics from client {client_id} "
                  f"({len(self.client_metrics)}/{self.num_clients})")
            
            # If all clients sent metrics, aggregate and continue
            if len(self.client_metrics) == self.num_clients:
                self.aggregate_metrics()
                self.continue_training()
    
    def distribute_initial_model(self):
        """Distribute initial global model architecture and weights to all clients"""
        # Send training configuration to all clients
        self.mqtt_client.publish(TOPIC_TRAINING_CONFIG, 
                            json.dumps(self.training_config),
                            qos=1)
        
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
                "model_config": self.model_config  # Include model architecture
            }
        
        message_json = json.dumps(initial_model_message)
        message_size = len(message_json.encode('utf-8'))
        
        print(f"Initial model message size: {message_size / 1024:.2f} KB ({message_size} bytes)")
        print(f"Model config: {len(self.model_config['layers'])} layers, {self.model_config['num_classes']} classes")
        
        # Publish multiple times to ensure delivery (QoS 0 can lose messages)
        print("\nPublishing initial model to clients...")
        for i in range(3):  # Send 3 times for reliability
            result = self.mqtt_client.publish(TOPIC_GLOBAL_MODEL, 
                                             message_json,
                                             qos=0)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"  Attempt {i+1}/3: Initial model sent successfully")
            else:
                print(f"  Attempt {i+1}/3: FAILED (return code: {result.rc})")
            
            time.sleep(0.5)  # Small delay between sends
        
        # Wait for clients to receive and build the model
        print("\nWaiting for clients to receive and build the model...")
        time.sleep(3)
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Signal clients to start training with the global model
        print("Signaling clients to start training...")
        result = self.mqtt_client.publish(TOPIC_START_TRAINING,
                                json.dumps({"round": self.current_round}),
                                qos=1)
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print("Start training signal sent successfully\n")
        else:
            print(f"ERROR: Failed to send start training signal (return code: {result.rc})\n")
    
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
                "quantized_data": serialized
            }
        else:
            # Send global model to all clients
            global_model_message = {
                "round": self.current_round,
                "weights": self.serialize_weights(self.global_weights)
            }
        
        # Publish aggregated model (QoS 1 for at-least-once) and avoid duplicates
        print(f"Publishing aggregated model for round {self.current_round}...")
        message_json = json.dumps(global_model_message)
        for i in range(3):
            result = self.mqtt_client.publish(TOPIC_GLOBAL_MODEL, message_json, qos=1)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"  Attempt {i+1}/3: Aggregated model sent")
                break
            else:
                print(f"  Attempt {i+1}/3: FAILED (rc={result.rc})")
                time.sleep(0.5)
        
        print(f"Aggregated global model from round {self.current_round} sent to all clients")
        
        # Request evaluation from clients
        time.sleep(2)
        print("Requesting client evaluation...")
        self.mqtt_client.publish(TOPIC_START_EVALUATION,
                                json.dumps({"round": self.current_round}), qos=1)
    
    def aggregate_metrics(self):
        """Aggregate evaluation metrics from all clients"""
        print(f"\nAggregating metrics from {len(self.client_metrics)} clients...")
        
        # Calculate total samples
        total_samples = sum(metric['num_samples'] 
                          for metric in self.client_metrics.values())
        
        # Weighted average of metrics (for classification)
        aggregated_accuracy = sum(metric['metrics']['accuracy'] * metric['num_samples']
                                 for metric in self.client_metrics.values()) / total_samples
        
        aggregated_loss = sum(metric['metrics']['loss'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        # Store metrics
        self.ACCURACY.append(aggregated_accuracy)
        self.LOSS.append(aggregated_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"\n{'='*70}")
        print(f"Round {self.current_round} - Aggregated Metrics:")
        print(f"  Loss:     {aggregated_loss:.6f}")
        print(f"  Accuracy: {aggregated_accuracy:.6f} ({aggregated_accuracy*100:.2f}%)")
        print(f"{'='*70}\n")
    
    def continue_training(self):
        """Continue to next round or finish training"""
        # Clear updates and metrics for next round
        self.client_updates.clear()
        self.client_metrics.clear()
        
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
            
            # Send training complete signal to all clients with QoS 1 (at least once delivery)
            print("Sending training completion signal to all clients...")
            print(f"Publishing to topic: {TOPIC_TRAINING_COMPLETE}")
            result = self.mqtt_client.publish(TOPIC_TRAINING_COMPLETE, json.dumps({"message": "Training completed"}), qos=1)
            print(f"Publish result: rc={result.rc}, mid={result.mid}")
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"ERROR: Failed to publish training completion message, rc={result.rc}")
            else:
                print(f"Training completion signal sent successfully (QoS 1)")
            time.sleep(2)  # Give clients time to receive and process the message
            
            self.save_results()
            self.plot_results()  # This will handle disconnection and exit
            return
        
        # Check if more rounds needed
        if self.current_round < self.num_rounds:
            self.current_round += 1
            
            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"{'='*70}\n")
            
            time.sleep(2)
            
            # Signal clients to start next training round (with retry, no duplicates)
            print(f"Signaling clients to start round {self.current_round}...")
            for i in range(3):
                result = self.mqtt_client.publish(TOPIC_START_TRAINING,
                                        json.dumps({"round": self.current_round}), qos=1)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print(f"  Attempt {i+1}/3: Start training signal sent")
                    break
                else:
                    print(f"  Attempt {i+1}/3: FAILED (rc={result.rc})")
                    time.sleep(0.5)
            print(f"Round {self.current_round} start signal sent\n")
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            
            # Send training complete signal to all clients with QoS 1 (at least once delivery)
            print("Sending training completion signal to all clients...")
            print(f"Publishing to topic: {TOPIC_TRAINING_COMPLETE}")
            result = self.mqtt_client.publish(TOPIC_TRAINING_COMPLETE, json.dumps({"message": "Training completed"}), qos=1)
            print(f"Publish result: rc={result.rc}, mid={result.mid}")
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                print(f"ERROR: Failed to publish training completion message, rc={result.rc}")
            else:
                print(f"Training completion signal sent successfully (QoS 1)")
            time.sleep(2)  # Give clients time to receive and process the message
            
            self.save_results()
            self.plot_results()  # This will handle disconnection and exit
    
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
        plt.figure(figsize=(12, 5))
        
        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.ROUNDS, self.LOSS, marker='o', linewidth=2, markersize=8, color='red')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss (Categorical Crossentropy)', fontsize=12)
        plt.title('Loss over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.ROUNDS, [acc*100 for acc in self.ACCURACY], marker='s', linewidth=2, markersize=8, color='green')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy over Federated Learning Rounds', fontsize=14)
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
            "accuracy": self.ACCURACY,
            "loss": self.LOSS,
            "convergence_time_seconds": self.convergence_time,
            "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
            "total_rounds": len(self.ROUNDS),
            "num_clients": self.num_clients,
            "final_accuracy": self.ACCURACY[-1] if self.ACCURACY else None,
            "final_loss": self.LOSS[-1] if self.LOSS else None
        }
        
        results_file = results_dir / 'mqtt_training_results.json'
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
                # Use 1 hour keepalive (3600 seconds) to prevent timeout during long training
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=3600)
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
                    print(f"  1. Mosquitto broker is running (service or container)")
                    print(f"  2. Broker address is correct: {MQTT_BROKER}:{MQTT_PORT}")
                    print(f"  3. Firewall allows connection on port {MQTT_PORT}")
                    raise


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Federated Learning Server with MQTT")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"{'='*70}\n")
    print("Waiting for clients to connect...\n")
    
    server = FederatedLearningServer(NUM_CLIENTS, NUM_ROUNDS)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.mqtt_client.disconnect()
