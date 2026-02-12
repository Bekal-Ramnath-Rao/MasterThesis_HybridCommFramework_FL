import numpy as np
import pandas as pd
import json
import pickle
import base64
import time
import os
import sys
import asyncio
from typing import Dict, Optional
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

from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived

# Server Configuration
QUIC_HOST = os.getenv("QUIC_HOST", "localhost")
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))
NETWORK_SCENARIO = os.getenv("NETWORK_SCENARIO", "excellent")  # Network scenario for result filename

# Convergence Settings
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))


class FederatedLearningServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = None
        self._stream_buffers = {}  # Buffer for incomplete messages
    
    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, StreamDataReceived):
            # Get or create buffer for this stream
            if event.stream_id not in self._stream_buffers:
                self._stream_buffers[event.stream_id] = b''
            
            # Append new data to buffer
            self._stream_buffers[event.stream_id] += event.data
            
            # Send flow control updates to allow more data (critical for poor networks)
            self.transmit()
            
            # Try to decode complete messages (delimited by newline)
            while b'\n' in self._stream_buffers[event.stream_id]:
                message_data, self._stream_buffers[event.stream_id] = self._stream_buffers[event.stream_id].split(b'\n', 1)
                if message_data:
                    try:
                        data = message_data.decode('utf-8')
                        message = json.loads(data)
                        if self.server:
                            asyncio.create_task(self.server.handle_message(message, self))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding message: {e}")


class FederatedLearningServer:
    def __init__(self, min_clients, num_rounds, max_clients=100):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.num_clients = min_clients  # Start with minimum, will update as clients join
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = {}  # Maps client_id to protocol reference
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
        self.training_started = False
        self.start_time = None
        self.convergence_time = None
        
        # Protocol reference
        self.protocol: Optional[FederatedLearningServerProtocol] = None
        
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
    
    def initialize_global_model(self):
        """Initialize the global model structure (LSTM for FL)"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM
        
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(1, 4)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', 
                     metrics=['mse', 'mae', 'mape'])
        
        self.global_weights = model.get_weights()
        
        print("\nGlobal model initialized with random weights")
        print(f"Model architecture: LSTM(50) -> Dense(1)")
        print(f"Number of weight layers: {len(self.global_weights)}")
    
    def serialize_weights(self, weights):
        """Serialize model weights for QUIC transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from QUIC"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights
    
    async def send_message(self, client_id, message):
        """Send message to client via QUIC stream"""
        if client_id in self.registered_clients:
            protocol = self.registered_clients[client_id]
            # Create a new stream for each message
            stream_id = protocol._quic.get_next_available_stream_id(is_unidirectional=False)
            # Add newline delimiter for message framing
            data = (json.dumps(message) + '\n').encode('utf-8')
            protocol._quic.send_stream_data(stream_id, data, end_stream=True)
            protocol.transmit()
            print(f"Sent message type '{message.get('type')}' to client {client_id} on stream {stream_id}")
            
            # Multiple transmit calls for large messages (improved for poor networks)
            if len(data) > 1_000_000:  # > 1MB
                for _ in range(3):
                    await asyncio.sleep(0.5)
                    protocol.transmit()
            else:
                await asyncio.sleep(0.1)
    
    async def broadcast_message(self, message):
        """Broadcast message to all registered clients"""
        for client_id in self.registered_clients.keys():
            await self.send_message(client_id, message)
    
    async def handle_message(self, message, protocol):
        """Handle incoming messages from clients"""
        try:
            msg_type = message.get('type')
            
            if msg_type == 'register':
                await self.handle_client_registration(message, protocol)
            elif msg_type == 'model_update':
                await self.handle_client_update(message)
            elif msg_type == 'metrics':
                await self.handle_client_metrics(message)
        except Exception as e:
            print(f"Server error handling message: {e}")
    
    async def handle_client_registration(self, message, protocol):
        """Handle client registration"""
        client_id = message['client_id']
        self.registered_clients[client_id] = protocol  # Store protocol reference
        self.active_clients.add(client_id)
        print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))
        
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
            await asyncio.sleep(2)
            await self.distribute_initial_model()
            self.start_time = time.time()
            self.training_started = True
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    async def mark_client_converged(self, client_id):
        """Remove converged client from active federation."""
        if client_id in self.active_clients:
            self.active_clients.discard(client_id)
            self.registered_clients.pop(client_id, None)
            self.client_updates.pop(client_id, None)
            self.client_metrics.pop(client_id, None)
            print(f"Client {client_id} converged and disconnected. Active clients remaining: {len(self.active_clients)}")
            if not self.active_clients:
                self.converged = True
                print("All clients converged. Ending training.")
                self.convergence_time = time.time() - self.start_time if self.start_time else 0
                await self.broadcast_message({'type': 'training_complete', 'message': 'Training completed'})
                await asyncio.sleep(2)
                self.save_results()
                self.plot_results()
    
    async def handle_client_update(self, message):
        """Handle model update from client"""
        client_id = message['client_id']
        round_num = message['round']
        if client_id not in self.active_clients:
            return
        if float(message.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
            await self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            # Decompress or deserialize client weights
            if 'compressed_data' in message and self.quantization_handler is not None:
                weights = self.quantization_handler.decompress_client_update(message['client_id'], message['compressed_data'])
                print(f"Server: Received and decompressed update from client {message['client_id']}")
            else:
                weights = self.deserialize_weights(message['weights'])
            
            self.client_updates[client_id] = {
                'weights': weights,
                'num_samples': message['num_samples'],
                'metrics': message['metrics']
            }
            
            print(f"Received update from client {client_id} "
                  f"({len(self.client_updates)}/{len(self.active_clients)})")
            
            if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                await self.aggregate_models()
    
    async def handle_client_metrics(self, message):
        """Handle evaluation metrics from client"""
        client_id = message['client_id']
        round_num = message['round']
        if client_id not in self.active_clients:
            return
        if float(message.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
            await self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            self.client_metrics[client_id] = {
                'num_samples': message['num_samples'],
                'metrics': message['metrics']
            }
            
            print(f"Received metrics from client {client_id} "
                  f"({len(self.client_metrics)}/{len(self.active_clients)})")
            
            if len(self.client_metrics) >= len(self.active_clients) and len(self.active_clients) > 0:
                await self.aggregate_metrics()
                await self.continue_training()
    
    async def distribute_initial_model(self):
        """Distribute initial global model to all clients"""
        await self.broadcast_message({
            'type': 'training_config',
            'config': self.training_config
        })
        
        self.current_round = 1
        
        print(f"\n{'='*70}")
        print(f"Distributing Initial Global Model")
        print(f"{'='*70}\n")
        
        # Send initial global model with architecture configuration
        await self.broadcast_message({
            'type': 'global_model',
            'round': 0,
            'weights': self.serialize_weights(self.global_weights),
            'model_config': {
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
        })
        
        print("Initial global model (architecture + weights) sent to all clients")
        await asyncio.sleep(2)
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        await self.broadcast_message({
            'type': 'start_training',
            'round': self.current_round
        })
    
    async def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")
        
        total_samples = sum(update['num_samples'] 
                          for update in self.client_updates.values())
        
        aggregated_weights = []
        first_client_weights = list(self.client_updates.values())[0]['weights']
        
        for layer_idx in range(len(first_client_weights)):
            layer_weights = np.zeros_like(first_client_weights[layer_idx])
            
            for client_id, update in self.client_updates.items():
                weight = update['num_samples'] / total_samples
                layer_weights += weight * update['weights'][layer_idx]
            
            aggregated_weights.append(layer_weights)
        
        self.global_weights = aggregated_weights
        
        # Prepare global model (compress if quantization enabled)
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
            weights_data = compressed_data
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        # Define model_config for late-joiners
        model_config = {
            "architecture": "LSTM",
            "layers": [
                {"type": "LSTM", "units": 50, "activation": "relu", "input_shape": [1, 4]},
                {"type": "Dense", "units": 1}
            ],
            "compile_config": {"loss": "mse", "optimizer": "adam", "metrics": ["mae"]}
        }
        
        await self.broadcast_message({
            'type': 'global_model',
            'round': self.current_round,
            weights_key: weights_data,
            'model_config': model_config  # Always include for late-joiners
        })
        
        print(f"Aggregated global model from round {self.current_round} sent to all clients")
        
        await asyncio.sleep(1)
        await self.broadcast_message({
            'type': 'start_evaluation',
            'round': self.current_round
        })
    
    async def aggregate_metrics(self):
        """Aggregate evaluation metrics from all clients"""
        print(f"\nAggregating metrics from {len(self.client_metrics)} clients...")
        
        total_samples = sum(metric['num_samples'] 
                          for metric in self.client_metrics.values())
        
        aggregated_mse = sum(metric['metrics']['mse'] * metric['num_samples']
                            for metric in self.client_metrics.values()) / total_samples
        
        aggregated_mae = sum(metric['metrics']['mae'] * metric['num_samples']
                            for metric in self.client_metrics.values()) / total_samples
        
        aggregated_mape = sum(metric['metrics']['mape'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        aggregated_loss = sum(metric['metrics']['loss'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
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
    
    async def continue_training(self):
        """Continue to next round or finish training"""
        self.client_updates.clear()
        self.client_metrics.clear()
        
        if len(self.active_clients) == 0:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            self.converged = True
            print("\n" + "="*70)
            print("All clients converged locally. Training complete.")
            print("="*70 + "\n")
            await self.broadcast_message({'type': 'training_complete', 'message': 'Training completed'})
            await asyncio.sleep(2)
            self.save_results()
            self.plot_results()
            return
        
        if self.current_round >= MIN_ROUNDS and self.check_convergence():
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("CONVERGENCE ACHIEVED!")
            print(f"Training stopped early at round {self.current_round}/{self.num_rounds}")
            print(f"Loss improvement below threshold for {CONVERGENCE_PATIENCE} consecutive rounds")
            print(f"Time to Convergence: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            self.converged = True
            
            await self.broadcast_message({
                'type': 'training_complete',
                'message': 'Training completed'
            })
            
            await asyncio.sleep(2)
            self.save_results()
            self.plot_results()
            return
        
        if self.current_round < self.num_rounds:
            self.current_round += 1
            
            print(f"\n{'='*70}")
            print(f"Starting Round {self.current_round}/{self.num_rounds}")
            print(f"{'='*70}\n")
            
            await asyncio.sleep(2)
            await self.broadcast_message({
                'type': 'start_training',
                'round': self.current_round
            })
        else:
            self.convergence_time = time.time() - self.start_time if self.start_time else 0
            print("\n" + "="*70)
            print("Federated Learning Completed!")
            print(f"Maximum rounds ({self.num_rounds}) reached")
            print(f"Total Training Time: {self.convergence_time:.2f} seconds ({self.convergence_time/60:.2f} minutes)")
            print("="*70 + "\n")
            
            await self.broadcast_message({
                'type': 'training_complete',
                'message': 'Training completed'
            })
            
            await asyncio.sleep(2)
            self.save_results()
            self.plot_results()
    
    def check_convergence(self):
        """Check if model has converged based on loss improvement"""
        if len(self.LOSS) == 0:
            return False
        
        current_loss = self.LOSS[-1]
        improvement = self.best_loss - current_loss
        
        if improvement > CONVERGENCE_THRESHOLD:
            self.best_loss = current_loss
            self.rounds_without_improvement = 0
            print(f"  → Loss improved by {improvement:.6f} (threshold: {CONVERGENCE_THRESHOLD})")
            return False
        else:
            self.rounds_without_improvement += 1
            print(f"  → No significant improvement (improvement: {improvement:.6f}, threshold: {CONVERGENCE_THRESHOLD})")
            print(f"  → Rounds without improvement: {self.rounds_without_improvement}/{CONVERGENCE_PATIENCE}")
            
            if self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
                return True
            return False
    
    def plot_results(self):
        """Plot training metrics"""
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.ROUNDS, self.MSE, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.title('MSE over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(self.ROUNDS, self.MAE, marker='s', linewidth=2, markersize=8, color='orange')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
        plt.title('MAE over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(self.ROUNDS, self.MAPE, marker='^', linewidth=2, markersize=8, color='green')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Mean Absolute Percentage Error (MAPE)', fontsize=12)
        plt.title('MAPE over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)
        plt.savefig(results_dir / 'quic_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {results_dir / 'quic_training_metrics.png'}")
        print("\nDisplaying plot... Close the plot window to exit.")
        plt.show()
        
        print("\nPlot closed. Server shutting down...")
        import sys
        sys.exit(0)
    
    def save_results(self):
        """Save results to file"""
        results_dir = Path(__file__).parent.parent / 'results'
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
        
        results_file = results_dir / f'quic_{NETWORK_SCENARIO}_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")


async def main():
    print(f"\n{'='*70}")
    print(f"Federated Learning Server with QUIC")
    print(f"Host: {QUIC_HOST}:{QUIC_PORT}")
    print(f"Clients: {MIN_CLIENTS} (min) - {MAX_CLIENTS} (max)")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"{'='*70}\n")
    print("Waiting for clients to connect...\n")
    
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    
    # Configure QUIC
    configuration = QuicConfiguration(
        is_client=False,
        max_datagram_frame_size=65536,
    )
    
    # Load certificates from certs directory
    # In Docker, certs are mounted at /app/certs/
    cert_dir = Path("/app/certs") if Path("/app/certs").exists() else Path(__file__).parent.parent.parent / "certs"
    cert_file = cert_dir / "server-cert.pem"
    key_file = cert_dir / "server-key.pem"
    
    if not cert_file.exists() or not key_file.exists():
        print("❌ Certificates not found. Please run generate_certs.py first.")
        print(f"   Expected location: {cert_dir}")
        print(f"   Cert file: {cert_file}")
        print(f"   Key file: {key_file}")
        import sys
        sys.exit(1)
    
    print(f"✓ Loading certificates from {cert_dir}")
    configuration.load_cert_chain(str(cert_file), str(key_file))
    
    # Create protocol factory
    def create_protocol(*args, **kwargs):
        protocol = FederatedLearningServerProtocol(*args, **kwargs)
        protocol.server = server
        server.protocol = protocol
        return protocol
    
    await serve(
        QUIC_HOST,
        QUIC_PORT,
        configuration=configuration,
        create_protocol=create_protocol,
    )
    
    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutting down...")
