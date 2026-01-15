import numpy as np
import json
import pickle
import base64
import time
import os
import asyncio
from typing import Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path
from aioquic.asyncio import QuicConnectionProtocol, serve
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived

# Server Configuration
QUIC_HOST = os.getenv("QUIC_HOST", "localhost")
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))

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
        print(f"[DEBUG] quic_event_received called, event type: {type(event).__name__}")
        if isinstance(event, StreamDataReceived):
            print(f"[DEBUG] Server received data on stream {event.stream_id}, size={len(event.data)} bytes, end_stream={event.end_stream}")
            # Get or create buffer for this stream
            if event.stream_id not in self._stream_buffers:
                self._stream_buffers[event.stream_id] = b''
            
            # Append new data to buffer
            self._stream_buffers[event.stream_id] += event.data
            print(f"[DEBUG] Stream {event.stream_id} buffer now has {len(self._stream_buffers[event.stream_id])} bytes")
            
            # Send flow control updates to allow more data
            self.transmit()
            
            # Try to decode complete messages (delimited by newline)
            while b'\n' in self._stream_buffers[event.stream_id]:
                message_data, self._stream_buffers[event.stream_id] = self._stream_buffers[event.stream_id].split(b'\n', 1)
                if message_data:
                    try:
                        data = message_data.decode('utf-8')
                        message = json.loads(data)
                        print(f"[DEBUG] Server decoded message on stream {event.stream_id}: type={message.get('type')}")
                        if self.server:
                            asyncio.create_task(self.server.handle_message(message, self))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding message: {e}")
                        import traceback
                        traceback.print_exc()
            
            # If stream ended and buffer has remaining data, try to process it
            if event.end_stream:
                print(f"[DEBUG] Stream {event.stream_id} ended, processing remaining buffer ({len(self._stream_buffers[event.stream_id])} bytes)")
                if self._stream_buffers[event.stream_id]:
                    try:
                        data = self._stream_buffers[event.stream_id].decode('utf-8')
                        message = json.loads(data)
                        print(f"[DEBUG] Server decoded remaining buffer from stream {event.stream_id}: type={message.get('type')}")
                        if self.server:
                            asyncio.create_task(self.server.handle_message(message, self))
                        self._stream_buffers[event.stream_id] = b''
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding remaining buffer: {e}")
                        import traceback
                        traceback.print_exc()


class FederatedLearningServer:
    def __init__(self, num_clients, num_rounds):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.current_round = 0
        self.registered_clients = {}  # Maps client_id to protocol reference
        self.client_updates = {}
        self.client_metrics = {}
        self.global_weights = None
        
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
        
        # Protocol reference
        self.protocol: Optional[FederatedLearningServerProtocol] = None
        
        # Initialize global model
        self.initialize_global_model()
        
        # Training configuration
        self.training_config = {
            "batch_size": 32,
            "local_epochs": 20
        }
    
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
        
        self.global_weights = model.get_weights()
        
        print("\nGlobal CNN model initialized for emotion recognition")
        print(f"Model architecture: CNN with {len(self.global_weights)} weight layers")
        print(f"Input shape: 48x48x1 (grayscale images)")
        print(f"Output classes: 7 emotions")
    
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
            protocol._quic.send_stream_data(stream_id, data, end_stream=False)
            protocol.transmit()
            
            msg_type = message.get('type')
            print(f"Sent message type '{msg_type}' to client {client_id} on stream {stream_id} ({len(data)} bytes)")
            
            # Give event loop time to transmit large messages
            await asyncio.sleep(0.1)
    
    async def broadcast_message(self, message):
        """Broadcast message to all registered clients"""
        msg_type = message.get('type')
        print(f"[DEBUG] Server broadcasting message type: {msg_type} to {len(self.registered_clients)} clients")
        for client_id in self.registered_clients.keys():
            await self.send_message(client_id, message)
    
    async def handle_message(self, message, protocol):
        """Handle incoming messages from clients"""
        try:
            msg_type = message.get('type')
            client_id = message.get('client_id', 'unknown')
            print(f"[DEBUG] Server received message type: {msg_type} from client {client_id}")
            
            if msg_type == 'register':
                await self.handle_client_registration(message, protocol)
            elif msg_type == 'model_update':
                await self.handle_client_update(message)
            elif msg_type == 'metrics':
                await self.handle_client_metrics(message)
        except Exception as e:
            print(f"Server error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    async def handle_client_registration(self, message, protocol):
        """Handle client registration"""
        client_id = message['client_id']
        self.registered_clients[client_id] = protocol  # Store protocol reference
        print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients})")
        
        if len(self.registered_clients) == self.num_clients:
            print("\nAll clients registered. Distributing initial global model...\n")
            await asyncio.sleep(2)
            await self.distribute_initial_model()
            self.start_time = time.time()
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    async def handle_client_update(self, message):
        """Handle model update from client"""
        client_id = message['client_id']
        round_num = message['round']
        
        if round_num == self.current_round:
            self.client_updates[client_id] = {
                'weights': self.deserialize_weights(message['weights']),
                'num_samples': message['num_samples'],
                'metrics': message['metrics']
            }
            
            print(f"Received update from client {client_id} "
                  f"({len(self.client_updates)}/{self.num_clients})")
            
            if len(self.client_updates) == self.num_clients:
                await self.aggregate_models()
    
    async def handle_client_metrics(self, message):
        """Handle evaluation metrics from client"""
        client_id = message['client_id']
        round_num = message['round']
        
        if round_num == self.current_round:
            self.client_metrics[client_id] = {
                'num_samples': message['num_samples'],
                'metrics': message['metrics']
            }
            
            print(f"Received metrics from client {client_id} "
                  f"({len(self.client_metrics)}/{self.num_clients})")
            
            if len(self.client_metrics) == self.num_clients:
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
        model_config = {
            'input_shape': [48, 48, 1],
            'num_classes': 7,
            'architecture': 'CNN',
            'layers': [
                {'type': 'Input'},
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
        
        await self.broadcast_message({
            'type': 'global_model',
            'round': 0,
            'weights': self.serialize_weights(self.global_weights),
            'model_config': model_config
        })
        
        print("Initial global model (architecture + weights) sent to all clients")
        print("Waiting for clients to initialize their models (TensorFlow + CNN building)...")
        await asyncio.sleep(10)
        
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
        
        await self.broadcast_message({
            'type': 'global_model',
            'round': self.current_round,
            'weights': self.serialize_weights(self.global_weights)
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
        
        aggregated_accuracy = sum(metric['metrics']['accuracy'] * metric['num_samples']
                                 for metric in self.client_metrics.values()) / total_samples
        
        aggregated_loss = sum(metric['metrics']['loss'] * metric['num_samples']
                             for metric in self.client_metrics.values()) / total_samples
        
        self.ACCURACY.append(aggregated_accuracy)
        self.LOSS.append(aggregated_loss)
        self.ROUNDS.append(self.current_round)
        
        print(f"\n{'='*70}")
        print(f"Round {self.current_round} - Aggregated Metrics:")
        print(f"  Loss:     {aggregated_loss:.6f}")
        print(f"  Accuracy: {aggregated_accuracy:.6f}")
        print(f"{'='*70}\n")
    
    async def continue_training(self):
        """Continue to next round or finish training"""
        self.client_updates.clear()
        self.client_metrics.clear()
        
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
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.ROUNDS, self.LOSS, marker='o', linewidth=2, markersize=8, color='red')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Loss over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.ROUNDS, self.ACCURACY, marker='s', linewidth=2, markersize=8, color='green')
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Accuracy over Federated Learning Rounds', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        results_dir = Path(__file__).parent / 'results'
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
            "converged": self.converged
        }
        
        results_file = results_dir / 'quic_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")


async def main():
    print(f"\n{'='*70}")
    print(f"Federated Learning Server with QUIC - Emotion Recognition")
    print(f"Host: {QUIC_HOST}:{QUIC_PORT}")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"{'='*70}\n")
    
    server = FederatedLearningServer(NUM_CLIENTS, NUM_ROUNDS)
    
    # Configure QUIC with large stream data limits for model weights
    configuration = QuicConfiguration(
        is_client=False,
        max_datagram_frame_size=65536,
        max_stream_data=20 * 1024 * 1024,  # 20 MB per stream
        max_data=50 * 1024 * 1024,  # 50 MB total connection data
        idle_timeout=3600.0,  # 1 hour idle timeout (training can take long)
    )
    
    # Check if certificates exist in the certs directory
    cert_dir = Path(__file__).parent.parent.parent / "certs"
    cert_file = cert_dir / "server-cert.pem"
    key_file = cert_dir / "server-key.pem"
    
    if not cert_file.exists() or not key_file.exists():
        print("❌ Certificates not found. Please run generate_certs.py first.")
        print(f"   Expected location: {cert_dir}")
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
    
    print(f"✓ Starting QUIC server on {QUIC_HOST}:{QUIC_PORT}...")
    print("Waiting for clients to connect...\n")
    
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
        print("\n\nServer interrupted by user")
    except Exception as e:
        print(f"\n❌ Server Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer shutting down...")
