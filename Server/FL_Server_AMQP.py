import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import pika
import os
from typing import List, Dict
import matplotlib.pyplot as plt
from pathlib import Path

# Server Configuration
AMQP_HOST = os.getenv("AMQP_HOST", "localhost")
AMQP_PORT = int(os.getenv("AMQP_PORT", "5672"))
AMQP_USER = os.getenv("AMQP_USER", "guest")
AMQP_PASSWORD = os.getenv("AMQP_PASSWORD", "guest")
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))

# AMQP Exchanges and Queues
EXCHANGE_BROADCAST = "fl_broadcast"
EXCHANGE_CLIENT_UPDATES = "fl_client_updates"
QUEUE_CLIENT_REGISTER = "fl.client.register"
QUEUE_CLIENT_UPDATE = "fl.client.update"
QUEUE_CLIENT_METRICS = "fl.client.metrics"


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
        
        # AMQP connection
        self.connection = None
        self.channel = None
        self.consuming = False
    
    def serialize_weights(self, weights):
        """Serialize model weights for AMQP transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from AMQP"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights
    
    def connect(self):
        """Connect to RabbitMQ broker"""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to RabbitMQ at {AMQP_HOST}:{AMQP_PORT}...")
                credentials = pika.PlainCredentials(AMQP_USER, AMQP_PASSWORD)
                parameters = pika.ConnectionParameters(
                    host=AMQP_HOST,
                    port=AMQP_PORT,
                    credentials=credentials,
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
                self.connection = pika.BlockingConnection(parameters)
                self.channel = self.connection.channel()
                
                # Declare exchanges
                self.channel.exchange_declare(exchange=EXCHANGE_BROADCAST, exchange_type='fanout', durable=True)
                self.channel.exchange_declare(exchange=EXCHANGE_CLIENT_UPDATES, exchange_type='direct', durable=True)
                
                # Declare queues for receiving from clients
                self.channel.queue_declare(queue=QUEUE_CLIENT_REGISTER, durable=True)
                self.channel.queue_declare(queue=QUEUE_CLIENT_UPDATE, durable=True)
                self.channel.queue_declare(queue=QUEUE_CLIENT_METRICS, durable=True)
                
                # Bind queues to exchange with routing keys
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_REGISTER, routing_key='client.register')
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_UPDATE, routing_key='client.update')
                self.channel.queue_bind(exchange=EXCHANGE_CLIENT_UPDATES, queue=QUEUE_CLIENT_METRICS, routing_key='client.metrics')
                
                # Set up consumers
                self.channel.basic_consume(queue=QUEUE_CLIENT_REGISTER, on_message_callback=self.on_client_register, auto_ack=True)
                self.channel.basic_consume(queue=QUEUE_CLIENT_UPDATE, on_message_callback=self.on_client_update, auto_ack=True)
                self.channel.basic_consume(queue=QUEUE_CLIENT_METRICS, on_message_callback=self.on_client_metrics, auto_ack=True)
                
                print(f"Server connected to RabbitMQ broker\n")
                return True
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to RabbitMQ broker after {max_retries} attempts.")
                    print(f"\nPlease ensure:")
                    print(f"  1. RabbitMQ broker is running")
                    print(f"  2. Broker address is correct: {AMQP_HOST}:{AMQP_PORT}")
                    print(f"  3. Credentials are correct: {AMQP_USER}")
                    raise
    
    def on_client_register(self, ch, method, properties, body):
        """Handle client registration"""
        try:
            data = json.loads(body.decode())
            client_id = data['client_id']
            self.registered_clients.add(client_id)
            print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
            
            # If all clients registered, start federated learning
            if len(self.registered_clients) == self.num_clients:
                print("\nAll clients registered. Starting federated learning...\n")
                time.sleep(2)
                self.start_federated_learning()
        except Exception as e:
            print(f"Server error handling registration: {e}")
    
    def on_client_update(self, ch, method, properties, body):
        """Handle model update from client"""
        try:
            data = json.loads(body.decode())
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
        except Exception as e:
            print(f"Server error handling client update: {e}")
    
    def on_client_metrics(self, ch, method, properties, body):
        """Handle evaluation metrics from client"""
        try:
            data = json.loads(body.decode())
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
        except Exception as e:
            print(f"Server error handling client metrics: {e}")
    
    def start_federated_learning(self):
        """Initialize federated learning process"""
        # Send training configuration to all clients
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "training_config",
                "config": self.training_config
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        self.current_round = 1
        
        print(f"\n{'='*70}")
        print(f"Initializing Global Model")
        print(f"{'='*70}\n")
        
        print("Waiting for initial model from clients...")
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        # Signal clients to start training
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "start_training",
                "round": self.current_round
            }),
            properties=pika.BasicProperties(delivery_mode=2)
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
        
        # Send global model to all clients
        global_model_message = {
            "message_type": "global_model",
            "round": self.current_round,
            "weights": self.serialize_weights(self.global_weights)
        }
        
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps(global_model_message),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        
        print(f"Global model for round {self.current_round} sent to all clients")
        
        # Request evaluation from clients
        time.sleep(1)
        self.channel.basic_publish(
            exchange=EXCHANGE_BROADCAST,
            routing_key='',
            body=json.dumps({
                "message_type": "start_evaluation",
                "round": self.current_round
            }),
            properties=pika.BasicProperties(delivery_mode=2)
        )
    
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
            self.channel.basic_publish(
                exchange=EXCHANGE_BROADCAST,
                routing_key='',
                body=json.dumps({
                    "message_type": "start_training",
                    "round": self.current_round
                }),
                properties=pika.BasicProperties(delivery_mode=2)
            )
        else:
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print("="*70 + "\n")
            self.plot_results()
            self.save_results()
            self.stop()
    
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
        plt.savefig('fl_amqp_results.png', dpi=300, bbox_inches='tight')
        print("Results plot saved as 'fl_amqp_results.png'")
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
        
        with open('fl_amqp_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Results saved to 'fl_amqp_results.json'")
    
    def start(self):
        """Start consuming messages"""
        print("Waiting for clients to connect...\n")
        self.consuming = True
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("\nServer stopping...")
            self.stop()
    
    def stop(self):
        """Stop consuming and close connection"""
        if self.consuming:
            self.channel.stop_consuming()
        if self.connection and not self.connection.is_closed:
            self.connection.close()
        print("Server disconnected")


if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"Federated Learning Server with AMQP")
    print(f"Broker: {AMQP_HOST}:{AMQP_PORT}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"{'='*70}\n")
    
    server = FederatedLearningServer(NUM_CLIENTS, NUM_ROUNDS)
    
    try:
        server.connect()
        server.start()
    except KeyboardInterrupt:
        print("\nServer shutting down...")
        server.stop()
