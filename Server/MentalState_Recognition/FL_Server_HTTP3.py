import io
import numpy as np
import json
import pickle
import base64
import time
import os
import asyncio
import sys
from typing import Dict, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import glob
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
from aioquic.quic.events import QuicEvent, StreamReset
from aioquic.h3.connection import H3_ALPN, H3Connection
from aioquic.h3.events import DataReceived, HeadersReceived, H3Event

# ============================================================================
# Data Loading Functions
# ============================================================================

def infer_label_from_name(path: str):
    n = os.path.basename(path).lower()
    if n.startswith("alerted"): return "alerted"
    if n.startswith("concentrated"): return "concentrated"
    if n.startswith("drowsy"): return "drowsy"
    if n.startswith("neutral"): return "neutral"
    if n.startswith("relaxed"): return "relaxed"
    return None

def read_csv_safe(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False, engine="c")
    except Exception:
        df = pd.read_csv(path, engine="python")
    miss = [c for c in FEAT_COLS if c not in df.columns]
    if miss:
        raise RuntimeError(f"{os.path.basename(path)} -> missing column: {miss[:5]}")
    df = df.dropna(subset=[FEAT_COLS[0]]).reset_index(drop=True)
    df[FEAT_COLS] = df[FEAT_COLS].astype('float32')
    return df

def csv_to_windows(path: str, file_id: int):
    label = infer_label_from_name(path)
    if label is None:
        return None
    df = read_csv_safe(path)
    x = df[FEAT_COLS].to_numpy(np.float32)
    # z-norm
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    x = (x - mu) / sd
    X, y, fidx = [], [], []
    for s in range(0, len(x) - WIN + 1, STRIDE):
        X.append(x[s:s + WIN])
        y.append(LBL2ID[label])
        fidx.append(file_id)
    if not X:
        return None
    return np.stack(X), np.array(y, np.int64), np.array(fidx, np.int64)

def find_all_csvs(folder):
    pats = ["Alerted-*.csv", "Concentrated-*.csv", "Drowsy-*.csv", "Neutral-*.csv", "Relaxed-*.csv"]
    files = []
    for p in pats:
        files += glob.glob(os.path.join(folder, p))
    files = sorted(files)
    if not files:
        raise SystemExit(f"CSV not found: {folder}")
    print(f"[{folder}] {len(files)} files found")
    return files

# ============================================================================
# Model Building Functions
# ============================================================================

def se_block(x, r=8):
    ch = x.shape[-1]
    s = tf.keras.layers.GlobalAveragePooling1D()(x)
    s = tf.keras.layers.Dense(max(ch // r, 8), activation='relu')(s)
    s = tf.keras.layers.Dense(ch, activation='sigmoid', dtype='float32')(s)
    s = tf.keras.layers.Reshape((1, ch))(s)
    return tf.keras.layers.Multiply()([x, s])

def conv_bn_relu(x, f, k, d=1):
    x = tf.keras.layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return x

def res_block(x, f, k, d=1):
    sc = x
    y = conv_bn_relu(x, f, k, d)
    y = tf.keras.layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    if sc.shape[-1] != f:
        sc = tf.keras.layers.Conv1D(f, 1, padding="same", use_bias=False)(sc)
        sc = tf.keras.layers.BatchNormalization()(sc)
    y = tf.keras.layers.Add()([y, sc])
    y = tf.keras.layers.ReLU()(y)
    y = se_block(y)
    return y

def build_model():
    inp = tf.keras.Input(shape=(256, 20))
    x = conv_bn_relu(inp, 64, 7, d=1)
    x = res_block(x, 64, 7, d=1)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    for d in [1, 2, 4]:
        x = res_block(x, 128, 5, d=d)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inp, out)
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3, first_decay_steps=4, t_mul=2.0, m_mul=0.8, alpha=1e-5
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4, global_clipnorm=1.0)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")])
    return model

# Server Configuration
HTTP3_HOST = os.getenv("HTTP3_HOST", "fl-server-http3-MentalState_Recognition")
HTTP3_PORT = int(os.getenv("HTTP3_PORT", "4434"))
# Dynamic client configuration
MIN_CLIENTS = int(os.getenv("MIN_CLIENTS", "2"))  # Minimum clients to start training
MAX_CLIENTS = int(os.getenv("MAX_CLIENTS", "100"))  # Maximum clients allowed
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "1000"))
STOP_ON_CLIENT_CONVERGENCE = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").lower() in ("1", "true", "yes")

# Convergence Settings
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))



def build_model():
    inp = tf.keras.Input(shape=(256, 20))

    x = conv_bn_relu(inp, 64, 7, d=1)
    x = res_block(x, 64, 7, d=1)
    x = tf.keras.layers.MaxPooling1D(2)(x)

    for d in [1, 2, 4]:
        x = res_block(x, 128, 5, d=d)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inp, out)

    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=1e-3, first_decay_steps=4, t_mul=2.0, m_mul=0.8, alpha=1e-5
    )
    opt = tf.keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=1e-4, global_clipnorm=1.0)

    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")])
    return model

class FederatedLearningServerProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server = None
        self._http = None  # H3Connection instance
        self._stream_buffers = {}  # Buffer for incomplete messages
    
    def quic_event_received(self, event: QuicEvent):
        """Handle QUIC events and convert to HTTP/3 events"""
        # Initialize H3 connection on first event
        if self._http is None:
            self._http = H3Connection(self._quic)
        
        # Convert QUIC events to HTTP/3 events
        for h3_event in self._http.handle_event(event):
            self._handle_h3_event(h3_event)
    
    def _handle_h3_event(self, event: H3Event):
        """Handle HTTP/3 events"""
        if isinstance(event, HeadersReceived):
            try:
                stream_id = event.stream_id
                headers = dict(event.headers)
                method = headers.get(b":method", b"").decode()
                path = headers.get(b":path", b"").decode()
                
                print(f"[HTTP/3] Received {method} request on stream {stream_id}, path: {path}")
                
                # Initialize buffer for this stream
                if stream_id not in self._stream_buffers:
                    self._stream_buffers[stream_id] = b''
                
                # Handle POST requests (client sending data)
                if method == "POST":
                    content_length = int(headers.get(b"content-length", b"0"))
                    print(f"[HTTP/3] Expecting {content_length} bytes on stream {stream_id}")
            except Exception as e:
                print(f"[HTTP/3] Error handling headers: {e}")
                import traceback
                traceback.print_exc()
        
        elif isinstance(event, DataReceived):
            try:
                stream_id = event.stream_id
                # Get or create buffer for this stream
                if stream_id not in self._stream_buffers:
                    self._stream_buffers[stream_id] = b''
                
                # Append new data to buffer
                self._stream_buffers[stream_id] += event.data
                
                # Send flow control updates to allow more data
                self.transmit()
                
                # If stream ended, process complete message
                if event.end_stream:
                    try:
                        data_str = self._stream_buffers[stream_id].decode('utf-8')
                        message = json.loads(data_str)
                        msg_type = message.get('type', 'unknown')
                        client_id = message.get('client_id', 'unknown')
                        print(f"[HTTP/3] Decoded complete message type '{msg_type}' from stream {stream_id}")
                        
                        # Send HTTP/3 response
                        response_headers = [
                            (b":status", b"200"),
                            (b"content-type", b"application/json"),
                        ]
                        response_body = json.dumps({"status": "ok"}).encode('utf-8')
                        self._http.send_headers(stream_id=stream_id, headers=response_headers)
                        self._http.send_data(stream_id=stream_id, data=response_body, end_stream=True)
                        self.transmit()
                        
                        # Handle message asynchronously
                        if self.server:
                            asyncio.create_task(self.server.handle_message(message, self))
                        
                        # Clear buffer
                        self._stream_buffers[stream_id] = b''
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"[HTTP/3] Error decoding message: {e}")
                        # Send error response
                        try:
                            error_headers = [
                                (b":status", b"400"),
                                (b"content-type", b"text/plain"),
                            ]
                            error_body = f"Error: {str(e)}".encode('utf-8')
                            self._http.send_headers(stream_id=stream_id, headers=error_headers)
                            self._http.send_data(stream_id=stream_id, data=error_body, end_stream=True)
                            self.transmit()
                        except:
                            pass
                        self._stream_buffers[stream_id] = b''
            except Exception as e:
                print(f"[HTTP/3] Error handling data: {e}")
                import traceback
                traceback.print_exc()
        
        elif isinstance(event, StreamReset):
            # Stream was reset, clear buffer
            stream_id = event.stream_id
            if stream_id in self._stream_buffers:
                del self._stream_buffers[stream_id]
                print(f"[HTTP/3] Stream {stream_id} reset, cleared buffer")


class FederatedLearningServer:
    def __init__(self, min_clients, num_rounds, max_clients=100):
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.num_clients = min_clients  # Start with minimum, will update as clients join
        self.num_rounds = num_rounds
        self.current_round = 0
        # Maps client_id -> set of active protocol references (handles duplicate WSL2 QUIC connections)
        self.registered_clients = {}
        self.active_clients = set()
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
        self.training_config = {
            "batch_size": 32,
            "local_epochs": 20
        }
    
    def initialize_global_model(self):
        """Initialize the global EEG model"""
        print("\nInitializing global EEG model (CNN+BiLSTM+MHA)...")
        model = build_model()
        self.global_weights = model.get_weights()
        print(f"Model initialized with {len(self.global_weights)} weight layers")

    
    def serialize_weights(self, weights):
        """Serialize model weights for HTTP/3 transmission"""
        buf = io.BytesIO()
        np.savez(buf, *weights)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from HTTP/3."""
        buf = io.BytesIO(base64.b64decode(encoded_weights.encode('utf-8')))
        try:
            loaded = np.load(buf, allow_pickle=False)
            weights = [loaded[f'arr_{i}'] for i in range(len(loaded.files))]
        except Exception:
            buf.seek(0)
            weights = pickle.loads(buf.read())
        return weights
    
    async def send_message(self, client_id, message):
        """Send message to all registered protocols for a client (handles duplicate WSL2 connections)."""
        if client_id not in self.registered_clients or not self.registered_clients[client_id]:
            print(f"[ERROR] Client {client_id} not in registered_clients. Available: {list(self.registered_clients.keys())}")
            return

        payload = json.dumps(message).encode('utf-8')
        headers = [
            (b":status", b"200"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ]

        protocols = list(self.registered_clients[client_id])
        dead_protocols = set()

        for protocol in protocols:
            if protocol._http is None:
                try:
                    protocol._http = H3Connection(protocol._quic)
                except Exception:
                    dead_protocols.add(protocol)
                    continue

            try:
                stream_id = protocol._quic.get_next_available_stream_id(is_unidirectional=False)
            except Exception as e:
                print(f"[ERROR] Failed to get stream ID for client {client_id}: {e}")
                dead_protocols.add(protocol)
                continue

            try:
                protocol._http.send_headers(stream_id=stream_id, headers=headers)
                protocol._http.send_data(stream_id=stream_id, data=payload, end_stream=True)
                protocol.transmit()
                msg_type = message.get('type')
                msg_size_mb = len(payload) / (1024 * 1024)
                print(f"Sent message type '{msg_type}' to client {client_id} on stream {stream_id} ({len(payload)} bytes = {msg_size_mb:.2f} MB)")
            except Exception as e:
                print(f"[ERROR] Failed to send on protocol for client {client_id}: {e}")
                dead_protocols.add(protocol)
                continue

        if dead_protocols:
            self.registered_clients[client_id] -= dead_protocols

        if len(payload) > 1_000_000:
            for _ in range(10):
                await asyncio.sleep(0.3)
                for protocol in list(self.registered_clients.get(client_id, set())):
                    try:
                        protocol.transmit()
                    except Exception:
                        pass
        else:
            await asyncio.sleep(0.05)
    
    async def broadcast_message(self, message):
        """Broadcast message to all registered clients."""
        for client_id in list(self.registered_clients.keys()):
            await self.send_message(client_id, message)
    
    async def handle_message(self, message, protocol):
        """Handle incoming messages from clients"""
        try:
            msg_type = message.get('type')
            client_id = message.get('client_id', 'unknown')
            
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
        """Handle client registration.

        Stores every distinct protocol object in a *set* per client_id so that
        WSL2 / NAT environments that create duplicate QUIC connections for the
        same logical client still receive downlink messages regardless of which
        underlying connection is live.
        """
        client_id = message['client_id']

        if client_id not in self.registered_clients:
            self.registered_clients[client_id] = set()

        existing = self.registered_clients[client_id]
        if protocol not in existing:
            if existing:
                print(
                    f"[HTTP/3] Client {client_id} registered on an additional QUIC connection "
                    f"(total protocols for this client: {len(existing) + 1}). "
                    f"All connections will receive downlink messages."
                )
            existing.add(protocol)

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

        if len(self.registered_clients) >= self.min_clients:
            # Set the flag BEFORE the async distribute call to prevent a concurrent
            # re-registration from triggering a second distribution round.
            self.training_started = True
            print("\nAll clients registered. Distributing initial global model...\n")
            self.start_time = time.time()
            await asyncio.sleep(2)
            await self.distribute_initial_model()
            print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    async def mark_client_converged(self, client_id):
        """Remove converged client from active federation."""
        if not STOP_ON_CLIENT_CONVERGENCE:
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
            await asyncio.sleep(2.0)
        
        print("Initial global model (architecture + weights) sent to all clients")
        print("Waiting for clients to initialize their models (TensorFlow + CNN building)...")
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
            weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        await self.broadcast_message({
            'type': 'global_model',
            'round': self.current_round,
            weights_key: weights_data,
            'model_config': self.model_config_json
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
        
        results_dir = get_experiment_results_dir("mental_state", "http3")
        plt.savefig(results_dir / 'http3_training_metrics.png', dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {results_dir / 'http3_training_metrics.png'}")
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            plt.close()
        else:
            print("\nDisplaying plot... Close the plot window to exit.")
            plt.show()

        print("\nPlot closed. Server shutting down...")
        import sys
        sys.exit(0)
    
    def save_results(self):
        """Save results to file"""
        results_dir = get_experiment_results_dir("mental_state", "http3")
        
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
        
        results_file = results_dir / 'http3_training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def update_client_count(self, new_count):
        """Update the number of expected clients dynamically"""
        if new_count > self.num_clients:
            self.num_clients = new_count
            print(f"[DYNAMIC] Updated expected client count to {self.num_clients}")
    
    async def send_current_model_to_client(self, client_id):
        """Send current global model to a late-joining client"""
        if self.global_weights is None:
            return
        
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        await self.send_message(client_id, {
            'type': 'global_model',
            'round': self.current_round,
            weights_key: weights_data,
            'model_config': self.model_config_json
        })


async def main():
    print(f"\n{'='*70}")
    print(f"Federated Learning Server with HTTP/3 - MentalState Recognition Recognition")
    print(f"Host: {HTTP3_HOST}:{HTTP3_PORT}")
    print(f"Clients: {MIN_CLIENTS} (min) - {MAX_CLIENTS} (max)")
    print(f"Rounds: {NUM_ROUNDS}")
    print(f"{'='*70}\n")
    
    server = FederatedLearningServer(MIN_CLIENTS, NUM_ROUNDS, MAX_CLIENTS)
    
    # FAIR CONFIG: Aligned with MQTT/AMQP/gRPC/QUIC/DDS for unbiased comparison
    configuration = QuicConfiguration(
        is_client=False,
        alpn_protocols=H3_ALPN,
        # Large windows so the full model payload fits in one flow-control window.
        # 16/32 KB stalled WSL2 clients that needed thousands of MAX_DATA round-trips.
        max_stream_data=32 * 1024 * 1024,   # 32 MB per stream
        max_data=128 * 1024 * 1024,         # 128 MB connection total
        # FAIR CONFIG: Timeout 600s for very_poor network scenarios
        idle_timeout=600.0,  # 10 minutes
        max_datagram_frame_size=65536,  # 64 KB frames
        initial_rtt=0.15,  # Account for network latency
    )
    
    # Check if certificates exist in the certs directory
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
    
    print(f"✓ Starting HTTP/3 server on {HTTP3_HOST}:{HTTP3_PORT}...")
    print("Waiting for clients to connect...\n")
    
    await serve(
        HTTP3_HOST,
        HTTP3_PORT,
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
