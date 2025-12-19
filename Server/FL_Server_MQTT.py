import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import os
import paho.mqtt.client as mqtt
from typing import List, Dict
import matplotlib.pyplot as plt
from pathlib import Path

# Server Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")  # MQTT broker address
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))  # MQTT broker port
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_REGISTER = "fl/client_register"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"


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
        
        # Training configuration
        self.training_config = {
            "batch_size": 32,
            "local_epochs": 20
        }
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id="fl_server")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
    
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
        self.registered_clients.add(client_id)
        print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
        
        # If all clients registered, start federated learning
        if len(self.registered_clients) == self.num_clients:
            print("\nAll clients registered. Starting federated learning...\n")
            time.sleep(2)  # Give clients time to be ready
            self.start_federated_learning()
    
    def handle_client_update(self, payload):
        """Handle model update from client"""
        data = json.loads(payload.decode())
        client_id = data['client_id']
        round_num = data['round']
        
        if round_num == self.current_round:
            self.client_updates[client_id] = {
                'weights': self.deserialize_weights(data['weights']),
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
    
    def start_federated_learning(self):
        """Initialize federated learning process"""
        # Send training configuration to all clients
        self.mqtt_client.publish(TOPIC_TRAINING_CONFIG, 
                                json.dumps(self.training_config))
        
        self.current_round = 1
        
        print(f"\n{'='*70}")
        print(f"Initializing Global Model")
        print(f"{'='*70}\n")
        
        # Wait for clients to send their initial model structure
        # Server will use first client's model as the initial global model
        print("Waiting for initial model from clients...")
        
        # Note: Server will receive initial weights from clients in first training round
        # and create the initial global model from aggregation
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Signal clients to start training with their initial weights
        # After first aggregation, all clients will have synchronized global model
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
        
        # Send global model to all clients
        global_model_message = {
            "round": self.current_round,
            "weights": self.serialize_weights(self.global_weights)
        }
        
        self.mqtt_client.publish(TOPIC_GLOBAL_MODEL, 
                                json.dumps(global_model_message))
        
        print(f"Global model for round {self.current_round} sent to all clients")
        
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
    
    def continue_training(self):
        """Continue to next round or finish training"""
        # Clear updates and metrics for next round
        self.client_updates.clear()
        self.client_metrics.clear()
        
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
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print("="*70 + "\n")
            self.plot_results()
            self.save_results()
    
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
        plt.savefig('fl_mqtt_results.png', dpi=300, bbox_inches='tight')
        print("Results plot saved as 'fl_mqtt_results.png'")
        plt.show()
    
    def save_results(self):
        """Save results to file"""
        results = {
            "rounds": self.ROUNDS,
            "mse": self.MSE,
            "mae": self.MAE,
            "mape": self.MAPE,
            "loss": self.LOSS
        }
        
        with open('fl_mqtt_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Results saved to 'fl_mqtt_results.json'")
    
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
