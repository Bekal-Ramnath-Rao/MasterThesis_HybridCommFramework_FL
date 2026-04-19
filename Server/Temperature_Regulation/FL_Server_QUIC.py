import io
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
STOP_ON_CLIENT_CONVERGENCE = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").lower() in ("1", "true", "yes")
NETWORK_SCENARIO = os.getenv("NETWORK_SCENARIO", "excellent")  # Network scenario for result filename

# Project root and utilities (for experiment_results path)
if os.path.exists("/app"):
    _project_root = "/app"
else:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_utilities_path = os.path.join(_project_root, "scripts", "utilities")
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)
from experiment_results_path import get_experiment_results_dir
try:
    from fl_training_results_cpu_memory import (
        merge_cpu_memory_into_results,
        plot_cpu_memory_for_server_rounds,
    )
except ModuleNotFoundError:
    from scripts.utilities.fl_training_results_cpu_memory import (
        merge_cpu_memory_into_results,
        plot_cpu_memory_for_server_rounds,
    )
from battery_results_agg import avg_battery_model_drain_fraction

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
        self.BATTERY_CONSUMPTION = []
        self.BATTERY_MODEL_CONSUMPTION = []
        self.ROUND_TIMES = []
        self.round_start_time = None
        
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
        """Handle convergence signal from a client."""
        if not STOP_ON_CLIENT_CONVERGENCE:
            # Fixed-round mode: ignore client-local convergence.
            print(f"Ignoring convergence signal from client {client_id} (STOP_ON_CLIENT_CONVERGENCE=false)")
            return
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
        if STOP_ON_CLIENT_CONVERGENCE and float(message.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
            await self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            # Decompress or deserialize client weights
            if 'compressed_data' in message and self.quantization_handler is not None:
                # Keep quantized end-to-end: do NOT decompress/dequantize on server.
                self.client_updates[client_id] = {
                    'compressed_data': message['compressed_data'],
                    'num_samples': message['num_samples'],
                    'metrics': message['metrics']
                }
                print(f"Server: Received quantized update from client {message['client_id']} (kept quantized)")
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
        if STOP_ON_CLIENT_CONVERGENCE and float(message.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
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
        
        self.round_start_time = time.time()
        await self.broadcast_message({
            'type': 'start_training',
            'round': self.current_round
        })
    
    async def aggregate_models(self):
        """Aggregate model weights using FedAvg algorithm"""
        print(f"\nAggregating models from {len(self.client_updates)} clients...")

        # Quantization end-to-end: aggregate directly on compressed quantized tensors.
        if (
            self.quantization_handler is not None
            and len(self.client_updates) > 0
            and 'compressed_data' in list(self.client_updates.values())[0]
        ):
            compressed_updates = {
                cid: {"compressed_data": upd["compressed_data"], "num_samples": upd.get("num_samples", 1)}
                for cid, upd in self.client_updates.items()
            }
            aggregated_compressed, _stats = self.quantization_handler.aggregate_compressed_updates(compressed_updates)
            self.global_compressed = aggregated_compressed
            lw = getattr(self.quantization_handler, "last_aggregated_float_weights", None)
            if lw is not None:
                self.global_weights = lw

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
                'quantized_data': self.global_compressed,
                'model_config': model_config
            })

            print(f"Aggregated (kept-quantized) global model from round {self.current_round} sent to all clients")

            await asyncio.sleep(1)
            await self.broadcast_message({'type': 'start_evaluation', 'round': self.current_round})
            return
        
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

        if getattr(self, 'round_start_time', None) is not None:
            self.ROUND_TIMES.append(time.time() - self.round_start_time)
        socs = [m['metrics'].get('battery_soc', 1.0) for m in self.client_metrics.values()]
        self.BATTERY_CONSUMPTION.append(1.0 - (sum(socs) / len(socs) if socs else 1.0))
        self.BATTERY_MODEL_CONSUMPTION.append(avg_battery_model_drain_fraction(self.client_metrics))
        
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
        
        if STOP_ON_CLIENT_CONVERGENCE and self.current_round >= MIN_ROUNDS and self.check_convergence():
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
            self.round_start_time = time.time()
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
        """Plot battery consumption, round/convergence time, loss, and regression metrics."""
        results_dir = get_experiment_results_dir("temperature", "quic")
        rounds = self.ROUNDS
        n = len(rounds)
        conv_time = self.convergence_time if self.convergence_time is not None else (
            time.time() - self.start_time if self.start_time else 0)

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        bc = (self.BATTERY_CONSUMPTION + [0.0] * max(0, n - len(self.BATTERY_CONSUMPTION)))[:n] \
            if getattr(self, 'BATTERY_CONSUMPTION', []) else [0.0] * n
        if bc:
            ax1.plot(rounds, [c * 100 for c in bc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round'); ax1.set_ylabel('Battery consumption (%)')
        ax1.set_title('QUIC (temperature): Battery consumption over FL rounds')
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout(); fig1.savefig(results_dir / 'quic_battery_consumption.png', dpi=300, bbox_inches='tight'); plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        rt = (self.ROUND_TIMES + [0.0] * max(0, n - len(self.ROUND_TIMES)))[:n] \
            if getattr(self, 'ROUND_TIMES', []) else [0.0] * n
        if rt:
            ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2,
                    label=f'Total convergence: {conv_time:.1f} s')
        ax2.set_xlabel('Round'); ax2.set_ylabel('Time (s)')
        ax2.set_title('QUIC (temperature): Time per round and convergence')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        fig2.tight_layout(); fig2.savefig(results_dir / 'quic_round_and_convergence_time.png', dpi=300, bbox_inches='tight'); plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ax3.plot(rounds, self.LOSS, 'b-', marker='o', linewidth=2, markersize=6)
        ax3.set_xlabel('Round'); ax3.set_ylabel('Loss (MSE)')
        ax3.set_title('QUIC (temperature): Loss over FL Rounds')
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout(); fig3.savefig(results_dir / 'quic_loss.png', dpi=300, bbox_inches='tight'); plt.close(fig3)

        fig4, axes = plt.subplots(1, 3, figsize=(17, 5))
        axes[0].plot(rounds, self.MSE, marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Round'); axes[0].set_ylabel('MSE')
        axes[0].set_title('QUIC (temperature): MSE over Rounds'); axes[0].grid(True, alpha=0.3)
        axes[1].plot(rounds, self.MAE, marker='s', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Round'); axes[1].set_ylabel('MAE')
        axes[1].set_title('QUIC (temperature): MAE over Rounds'); axes[1].grid(True, alpha=0.3)
        axes[2].plot(rounds, self.MAPE, marker='^', linewidth=2, markersize=8, color='green')
        axes[2].set_xlabel('Round'); axes[2].set_ylabel('MAPE (%)')
        axes[2].set_title('QUIC (temperature): MAPE over Rounds'); axes[2].grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(results_dir / 'quic_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print(f"Results plots saved to {results_dir}")
        plot_cpu_memory_for_server_rounds(
            results_dir,
            "quic_cpu_memory_per_round.png",
            self.ROUNDS,
            "temperature",
            title="QUIC (temperature): avg client CPU and RAM per round",
        )
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            plt.close()
        else:
            print("\nDisplaying plot... Close the plot window to exit.")
            plt.show(block=False)

        print("\nPlot closed. Server shutting down...")
        import sys
        sys.exit(0)
    
    def save_results(self):
        """Save results to file"""
        results_dir = get_experiment_results_dir("temperature", "quic")
        
        results = {
            "rounds": self.ROUNDS,
            "loss": self.LOSS,
            "mse": self.MSE,
            "mae": self.MAE,
            "mape": self.MAPE,
            "accuracy": [max(0.0, 1.0 - v) for v in self.MSE],
            "battery_consumption": getattr(self, 'BATTERY_CONSUMPTION', []),
            "battery_model_consumption": getattr(self, 'BATTERY_MODEL_CONSUMPTION', []),
            "battery_model_consumption_source": "client_battery_model",
            "round_times_seconds": getattr(self, 'ROUND_TIMES', []),
            "convergence_time_seconds": self.convergence_time,
            "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
            "total_rounds": len(self.ROUNDS),
            "num_clients": self.num_clients,
            "summary": {
                "total_rounds": len(self.ROUNDS),
                "num_clients": self.num_clients,
                "final_loss": self.LOSS[-1] if self.LOSS else None,
                "final_mse": self.MSE[-1] if self.MSE else None,
                "convergence_time_seconds": self.convergence_time,
                "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
                "converged": self.converged,
            }
        }
        merge_cpu_memory_into_results(results, "temperature")

        results_file = results_dir / 'quic_training_results.json'
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
