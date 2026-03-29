import os
import sys
# Server uses CPU only (aggregation is numpy-only); saves GPU memory for clients
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import json
import pickle
import base64
import time
import asyncio
import socket
from typing import Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path

# Project root and utilities (for experiment_results path)
if os.path.exists("/app"):
    _project_root = "/app"
else:
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_utilities_path = os.path.join(_project_root, "scripts", "utilities")
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)
from experiment_results_path import get_experiment_results_dir

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

# Server Configuration (use 0.0.0.0 for host network mode; 127.0.0.1 for local only)
QUIC_HOST = os.getenv("QUIC_HOST", "0.0.0.0")
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))
from fl_termination_env import stop_on_client_convergence

# Convergence Settings
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))


QUIC_SOCKET_BUFFER_BYTES = 7_500_000  # 7.5MB for SO_RCVBUF/SO_SNDBUF (poor network)


class FederatedLearningServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = None
        self._stream_buffers = {}  # Buffer for incomplete messages

    def connection_made(self, transport):
        super().connection_made(transport)
        sock = transport.get_extra_info("socket")
        if sock:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, QUIC_SOCKET_BUFFER_BYTES)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, QUIC_SOCKET_BUFFER_BYTES)
                print(f"[QUIC] UDP socket buffers set to {QUIC_SOCKET_BUFFER_BYTES // 1_000_000}MB")
            except OSError as e:
                print(f"[QUIC] Could not set socket buffers: {e}")

    def quic_event_received(self, event: QuicEvent):
        #print(f"[DEBUG] quic_event_received called, event type: {type(event).__name__}")
        if isinstance(event, StreamDataReceived):
            #print(f"[DEBUG] Server received data on stream {event.stream_id}, size={len(event.data)} bytes, end_stream={event.end_stream}")
            # Get or create buffer for this stream
            if event.stream_id not in self._stream_buffers:
                self._stream_buffers[event.stream_id] = b''
            
            # Append new data to buffer
            self._stream_buffers[event.stream_id] += event.data
            #print(f"[DEBUG] Stream {event.stream_id} buffer now has {len(self._stream_buffers[event.stream_id])} bytes")
            
            # Send flow control updates to allow more data
            self.transmit()
            
            # Try to decode complete messages (delimited by newline)
            while b'\n' in self._stream_buffers[event.stream_id]:
                message_data, self._stream_buffers[event.stream_id] = self._stream_buffers[event.stream_id].split(b'\n', 1)
                if message_data:
                    try:
                        data = message_data.decode('utf-8')
                        message = json.loads(data)
                        #print(f"[DEBUG] Server decoded message on stream {event.stream_id}: type={message.get('type')}")
                        if self.server:
                            asyncio.create_task(self.server.handle_message(message, self))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding message: {e}")
                        import traceback
                        traceback.print_exc()
            
            # If stream ended and buffer has remaining data, try to process it
            if event.end_stream:
                #print(f"[DEBUG] Stream {event.stream_id} ended, processing remaining buffer ({len(self._stream_buffers[event.stream_id])} bytes)")
                if self._stream_buffers[event.stream_id]:
                    try:
                        data = self._stream_buffers[event.stream_id].decode('utf-8')
                        message = json.loads(data)
                        #print(f"[DEBUG] Server decoded remaining buffer from stream {event.stream_id}: type={message.get('type')}")
                        if self.server:
                            asyncio.create_task(self.server.handle_message(message, self))
                        self._stream_buffers[event.stream_id] = b''
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding remaining buffer: {e}")
                        import traceback
                        traceback.print_exc()


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
        
        # Metrics storage for classification
        self.ACCURACY = []
        self.LOSS = []
        self.ROUNDS = []
        self.ROUND_TIMES = []
        self.BATTERY_CONSUMPTION = []
        self.round_start_time = None

        # Convergence tracking
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.converged = False
        self.training_started = False
        self.training_started = False
        self.start_time = None
        self.convergence_time = None
        self.model_config_json = None  # Will be set during distribute_initial_model
        
        # Protocol reference
        self.protocol: Optional[FederatedLearningServerProtocol] = None
        
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
        # Training configuration sent to clients
        # Batch size 16 to reduce client GPU memory; server runs on CPU
        # Batch size 16 to reduce client GPU memory; server runs on CPU
        self.training_config = {
            "batch_size": int(os.getenv("BATCH_SIZE", "16")),
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
        """Send message to client via QUIC stream with improved transmission for poor networks"""
        if client_id in self.registered_clients:
            protocol = self.registered_clients[client_id]
            # Create a new stream for each message
            stream_id = protocol._quic.get_next_available_stream_id(is_unidirectional=False)
            # Add newline delimiter for message framing
            data = (json.dumps(message) + '\n').encode('utf-8')
            # Set end_stream=True to ensure proper message delivery, especially for large messages
            protocol._quic.send_stream_data(stream_id, data, end_stream=True)
            protocol.transmit()
            
            msg_type = message.get('type')
            msg_size_mb = len(data) / (1024 * 1024)
            print(f"Sent message type '{msg_type}' to client {client_id} on stream {stream_id} ({len(data)} bytes = {msg_size_mb:.2f} MB)")
            
            # Multiple transmit calls for large messages (improved for poor networks)
            if len(data) > 1_000_000:  # > 1MB
                for _ in range(3):
                    await asyncio.sleep(0.5)
                    protocol.transmit()
            else:
                # Small delay for flow control
                await asyncio.sleep(0.1)
    
    async def broadcast_message(self, message):
        """Broadcast message to all registered clients"""
        msg_type = message.get('type')
        #print(f"[DEBUG] Server broadcasting message type: {msg_type} to {len(self.registered_clients)} clients")
        for client_id in self.registered_clients.keys():
            await self.send_message(client_id, message)
    
    async def handle_message(self, message, protocol):
        """Handle incoming messages from clients"""
        try:
            msg_type = message.get('type')
            client_id = message.get('client_id', 'unknown')
            #print(f"[DEBUG] Server received message type: {msg_type} from client {client_id}")
            
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
        self.active_clients.add(client_id)
        print(f"Client {client_id} registered ({len(self.registered_clients)}/{self.num_clients} expected, min: {self.min_clients})")
        
        # Update total client count if more clients join
        if len(self.registered_clients) > self.num_clients:
            self.update_client_count(len(self.registered_clients))
        
        # Check if this is a late-joining client
        if self.training_started:
            print(f"[LATE JOIN] Client {client_id} joining during round {self.current_round}")
            if len(self.registered_clients) > self.num_clients:
                self.update_client_count(len(self.registered_clients))
            if self.global_weights is not None:
                self.send_current_model_to_client(client_id)
            return
        
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
        if not stop_on_client_convergence():
            # Fixed-round mode: ignore client-local convergence removal/disconnect.
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
            elif len(self.client_metrics) >= len(self.active_clients):
                # If remaining active clients already sent metrics, do not stall.
                await self.aggregate_metrics()
                await self.continue_training()
            elif len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                # If remaining active clients already sent updates, aggregate and continue.
                await self.aggregate_models()
    
    async def handle_client_update(self, message):
        """Handle model update from client"""
        recv_start_cpu = time.perf_counter() if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1" else None
        client_id = message['client_id']
        round_num = message['round']
        if client_id not in self.active_clients:
            return
        if stop_on_client_convergence() and float(message.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
            await self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            # Decompress or deserialize client weights
            if 'compressed_data' in message and self.quantization_handler is not None:
                # Deserialize base64+pickle encoded compressed data
                compressed_data = pickle.loads(base64.b64decode(message['compressed_data']))
                # Keep quantized end-to-end: do NOT decompress/dequantize on server.
                self.client_updates[client_id] = {
                    'compressed_data': compressed_data,
                    'num_samples': message['num_samples'],
                    'metrics': message['metrics']
                }
                print(f"Server: Received quantized update from client {message['client_id']} (kept quantized)")
                weights = None
            else:
                # Offload heavy deserialization to thread pool to avoid blocking event loop
                encoded = message.get('weights')
                if encoded is None:
                    print(f"[ERROR] Missing 'weights' in model_update from client {client_id}")
                    return
                start_t = time.time()
                loop = asyncio.get_event_loop()
                try:
                    weights = await loop.run_in_executor(None, self.deserialize_weights, encoded)
                except Exception as e:
                    print(f"[ERROR] Failed to deserialize weights from client {client_id}: {e}")
                    import traceback; traceback.print_exc()
                    return
                dt = time.time() - start_t
                #print(f"[DEBUG] Deserialized weights from client {client_id} in {dt:.2f}s")
            
            if recv_start_cpu is not None:
                O_recv = time.perf_counter() - recv_start_cpu
                recv_end_ts = time.time()
                send_start_ts = message.get("diagnostic_send_start_ts", recv_end_ts)
                print(f"FL_DIAG client_id={client_id} O_recv={O_recv:.9f} recv_end_ts={recv_end_ts:.9f} send_start_ts={send_start_ts:.9f}")
            
            if 'compressed_data' not in message or self.quantization_handler is None:
                self.client_updates[client_id] = {
                    'weights': weights,
                    'num_samples': message['num_samples'],
                    'metrics': message['metrics']
                }
            
            print(f"Received update from client {client_id} "
                  f"({len(self.client_updates)}/{len(self.active_clients)})")
            
            if len(self.client_updates) >= len(self.active_clients) and len(self.active_clients) > 0:
                await self.aggregate_models()
        #else:
            #print(f"[DEBUG] Ignoring model_update from client {client_id} for round {round_num} (server at {self.current_round})")
    
    async def handle_client_metrics(self, message):
        """Handle evaluation metrics from client"""
        client_id = message['client_id']
        round_num = message['round']
        if client_id not in self.active_clients:
            return
        if stop_on_client_convergence() and float(message.get('metrics', {}).get('client_converged', 0.0)) >= 1.0:
            await self.mark_client_converged(client_id)
            return
        if round_num == self.current_round:
            m = message.get('metrics', {})
            self.client_metrics[client_id] = {
                'num_samples': message['num_samples'],
                'metrics': message['metrics'],
                'battery_soc': float(m.get('battery_soc', 1.0)),
                'round_time_sec': float(m.get('round_time_sec', 0.0)),
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
        self.round_start_time = time.time()
        
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
        
        # Store model config for late-joiners and aggregation
        self.model_config_json = model_config
        
        # Prepare global model (compress if quantization enabled)
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed initial global model - Ratio: {stats['compression_ratio']:.2f}x")
            # Serialize compressed data to JSON-safe base64 string
            weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        print("Publishing initial model to clients (sending multiple times for reliability)...")
        for i in range(3):
            await self.broadcast_message({
                'type': 'global_model',
                'round': 0,
                weights_key: weights_data,
                'model_config': model_config
            })
            print(f"  Attempt {i+1}/3: Initial model broadcast complete")
            # Longer delay for poor network conditions (was 0.5s, now 2s)
            await asyncio.sleep(2.0)
        
        print("Initial global model (architecture + weights) sent to all clients")
        print("Waiting for clients to initialize their models (TensorFlow + CNN building)...")
        # Increased wait time for very poor network conditions (was 8s, now 30s)
        # This allows time for large model transfer + TensorFlow initialization
        await asyncio.sleep(30)
        
        print(f"\n{'='*70}")
        print(f"Starting Round {self.current_round}/{self.num_rounds}")
        print(f"{'='*70}\n")
        
        print("Signaling clients to start training...")
        await self.broadcast_message({
            'type': 'start_training',
            'round': self.current_round
        })
        print("Start training signal sent successfully\n")
    
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

            weights_data = base64.b64encode(pickle.dumps(self.global_compressed)).decode('utf-8')
            await self.broadcast_message({
                'type': 'global_model',
                'round': self.current_round,
                'quantized_data': weights_data,
                'model_config': self.model_config_json
            })

            print(f"Aggregated global model from round {self.current_round} sent to all clients (dequantize→FedAvg→requantize)")

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
            # Serialize compressed data to JSON-safe base64 string
            weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        await self.broadcast_message({
            'type': 'global_model',
            'round': self.current_round,
            weights_key: weights_data,
            'model_config': self.model_config_json  # Always include for late-joiners
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
        socs = [m.get('battery_soc', 1.0) for m in self.client_metrics.values()]
        self.BATTERY_CONSUMPTION.append(1.0 - (sum(socs) / len(socs) if socs else 1.0))
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
        
        if self.current_round < self.num_rounds:
            self.current_round += 1
            self.round_start_time = time.time()
            
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
        """Plot battery, round/convergence time, and loss/accuracy."""
        results_dir = get_experiment_results_dir("emotion", "quic")
        rounds = self.ROUNDS
        n = len(rounds)
        conv_time = self.convergence_time if self.convergence_time is not None else (time.time() - self.start_time if self.start_time else 0)
        bc = (getattr(self, 'BATTERY_CONSUMPTION', []) + [0.0] * max(0, n - len(getattr(self, 'BATTERY_CONSUMPTION', []))))[:n] or [0.0] * n
        rt = (getattr(self, 'ROUND_TIMES', []) + [0.0] * max(0, n - len(getattr(self, 'ROUND_TIMES', []))))[:n] or [0.0] * n
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        if bc: ax1.plot(rounds, [c * 100 for c in bc], marker='o', linewidth=2, markersize=6, color='#2e86ab')
        ax1.set_xlabel('Round'); ax1.set_ylabel('Battery consumption (%)'); ax1.set_title('QUIC: Battery consumption till end of FL'); ax1.grid(True, alpha=0.3)
        fig1.tight_layout(); fig1.savefig(results_dir / 'quic_battery_consumption.png', dpi=300, bbox_inches='tight'); plt.close(fig1)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if rt: ax2.bar(rounds, rt, color='#a23b72', alpha=0.8, label='Time per round (s)')
        ax2.axhline(y=conv_time, color='#f18f01', linestyle='--', linewidth=2, label=f'Convergence: {conv_time:.1f} s')
        ax2.set_xlabel('Round'); ax2.set_ylabel('Time (s)'); ax2.set_title('QUIC: Time per round and convergence'); ax2.legend(); ax2.grid(True, alpha=0.3)
        fig2.tight_layout(); fig2.savefig(results_dir / 'quic_round_and_convergence_time.png', dpi=300, bbox_inches='tight'); plt.close(fig2)
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        ax3a.plot(rounds, self.LOSS, marker='o', linewidth=2, markersize=8, color='red'); ax3a.set_xlabel('Round'); ax3a.set_ylabel('Loss'); ax3a.set_title('QUIC: Loss over Rounds'); ax3a.grid(True, alpha=0.3)
        ax3b.plot(rounds, self.ACCURACY, marker='s', linewidth=2, markersize=8, color='green'); ax3b.set_xlabel('Round'); ax3b.set_ylabel('Accuracy'); ax3b.set_title('QUIC: Accuracy over Rounds'); ax3b.grid(True, alpha=0.3)
        fig3.tight_layout(); fig3.savefig(results_dir / 'quic_training_metrics.png', dpi=300, bbox_inches='tight'); plt.close(fig3)
        print(f"Results plot saved to {results_dir / 'quic_training_metrics.png'}")
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1": plt.close('all')
        else: plt.show(block=False)
        print("\nPlot closed. Server shutting down...")
        import sys
        sys.exit(0)
    
    def save_results(self):
        """Save results to file"""
        results_dir = get_experiment_results_dir("emotion", "quic")
        
        results = {
            "rounds": self.ROUNDS,
            "accuracy": self.ACCURACY,
            "loss": self.LOSS,
            "round_times_seconds": getattr(self, 'ROUND_TIMES', []),
            "battery_consumption": getattr(self, 'BATTERY_CONSUMPTION', []),
            "convergence_time_seconds": self.convergence_time,
            "convergence_time_minutes": self.convergence_time / 60 if self.convergence_time else None,
            "total_rounds": len(self.ROUNDS),
            "num_clients": self.num_clients,
            "converged": self.converged,
            "final_accuracy": self.ACCURACY[-1] if self.ACCURACY else None,
            "final_loss": self.LOSS[-1] if self.LOSS else None,
        }
        
        results_file = results_dir / 'quic_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")


async def main():
    print(f"\n{'='*70}")
    print(f"Federated Learning Server with QUIC - Emotion Recognition")
    print(f"Host: {QUIC_HOST}:{QUIC_PORT}")
    print(f"Clients: {MIN_CLIENTS} (min) - {MAX_CLIENTS} (max)")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"{'='*70}\n")
    
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    
    # QUIC config: cubic congestion, 60s idle; 128 MB flow control (aligned with MQTT/gRPC for fair FL comparison)
    QUIC_MAX_DATA_BYTES = 128 * 1024 * 1024  # 128 MB
    configuration = QuicConfiguration(
        is_client=False,
        alpn_protocols=["fl"],
        congestion_control_algorithm="cubic",
        idle_timeout=60.0,
        max_data=QUIC_MAX_DATA_BYTES,
        max_stream_data=QUIC_MAX_DATA_BYTES,
        max_datagram_frame_size=65536,
        initial_rtt=0.15,
    )
    
    # Check if certificates exist in the certs directory
    # In Docker, certs are mounted at /app/certs/
    cert_dir = Path("/app/certs") if Path("/app/certs").exists() else Path(__file__).parent.parent.parent / "certs"
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
