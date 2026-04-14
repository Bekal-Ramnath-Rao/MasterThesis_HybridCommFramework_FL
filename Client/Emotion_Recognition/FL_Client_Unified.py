"""
Unified Federated Learning Client for Emotion Recognition
with RL-based Protocol Selection

Supports: MQTT, AMQP, gRPC, QUIC, DDS
Uses Q-Learning to dynamically select the best protocol for DATA transmission
Uses gRPC for CONTROL signals and protocol negotiation
Architecture: Event-driven, polls gRPC control-plane for round signals/queries
             Data transmission uses RL-selected protocol
"""

import os
import sys
import time
import json
import pickle
import base64
import logging
import threading
import socket
import re
import fcntl
from typing import Dict, Tuple, Optional, List, Sequence
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import protocol-specific modules
import paho.mqtt.client as mqtt
try:
    import pika
except ImportError:
    pika = None

try:
    import grpc
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Protocols'))
    import federated_learning_pb2
    import federated_learning_pb2_grpc
    GRPC_MAX_MESSAGE_BYTES = int(os.getenv("GRPC_MAX_MESSAGE_BYTES", str(4 * 1024 * 1024)))
    GRPC_CHUNK_SIZE = GRPC_MAX_MESSAGE_BYTES - 4096  # leave room for proto framing
except (ImportError, ModuleNotFoundError):
    grpc = None
    federated_learning_pb2 = None
    federated_learning_pb2_grpc = None
    GRPC_CHUNK_SIZE = 4 * 1024 * 1024 - 4096

try:
    import asyncio
    from aioquic.asyncio import connect
    from aioquic.quic.configuration import QuicConfiguration
    from aioquic.quic.events import StreamDataReceived
    from aioquic.asyncio.protocol import QuicConnectionProtocol
except ImportError:
    asyncio = None
    connect = None
    QuicConfiguration = None
    StreamDataReceived = None
    QuicConnectionProtocol = None

try:
    import httpx
    HTTP3_CLIENT_AVAILABLE = True
except ImportError:
    HTTP3_CLIENT_AVAILABLE = False
    httpx = None

try:
    from aioquic.h3.connection import H3_ALPN, H3Connection
    from aioquic.h3.events import DataReceived, HeadersReceived, H3Event
    try:
        from aioquic.h3.events import StreamReset
    except ImportError:
        from aioquic.quic.events import StreamReset
    HTTP3_AVAILABLE = True
except ImportError:
    HTTP3_AVAILABLE = False
    H3Connection = None
    H3Event = None
    StreamReset = None

# Set CycloneDDS config before any cyclonedds import (native lib may read at load time);
# same logic as FL_Client_DDS.py for DDS_PEER_* static unicast across hosts.
def _emotion_config_dir():
    if os.path.exists("/app"):
        return "/app/config"
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "config"))


def _try_distributed_unicast_client():
    base = _emotion_config_dir()
    helper = os.path.join(base, "dds_distributed_unicast.py")
    if not os.path.isfile(helper):
        return False
    import importlib.util

    spec = importlib.util.spec_from_file_location("dds_distributed_unicast", helper)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.try_apply_client_uri()


def _ensure_client_cyclonedds_uri():
    if os.environ.get("CYCLONEDDS_URI"):
        return
    if _try_distributed_unicast_client():
        return
    base = _emotion_config_dir()
    mc = os.path.join(base, "cyclonedds-multicast-lan.xml")
    if os.path.isfile(mc):
        os.environ["CYCLONEDDS_URI"] = "file://" + os.path.abspath(mc)
        return
    _cid = os.environ.get("CLIENT_ID", "1")
    _path = os.path.join(base, f"cyclonedds-emotion-client{_cid}.xml")
    if os.path.isfile(_path):
        os.environ["CYCLONEDDS_URI"] = "file://" + os.path.abspath(_path)


_ensure_client_cyclonedds_uri()

try:
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.util import duration
    from cyclonedds.core import Qos, Policy
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.types import sequence
    from dataclasses import dataclass
    DDS_AVAILABLE = True
except ImportError:
    DDS_AVAILABLE = False
    dataclass = lambda x: x
    IdlStruct = object

# Compression: pruning and quantization (client flow: train -> prune -> quantize -> send)
_compression_root = os.path.join(os.path.dirname(__file__), "..", "Compression_Technique")
if _compression_root not in sys.path:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from Compression_Technique.pruning_client import ModelPruning, PruningConfig
    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False
    ModelPruning = None
    PruningConfig = None
try:
    from Compression_Technique.quantization_client import Quantization, QuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False
    Quantization = None
    QuantizationConfig = None

# Define QUIC protocol handler if available
if QuicConnectionProtocol is not None:
    class UnifiedClientQUICProtocol(QuicConnectionProtocol):
        """QUIC protocol handler for receiving server messages"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.client = None  # Set by factory function
            self._stream_buffers = {}  # Instance-specific stream buffers
        
        def quic_event_received(self, event):
            if isinstance(event, StreamDataReceived):
                try:
                    # Get or create buffer for this stream
                    stream_id = event.stream_id
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                    
                    # Append new data to buffer
                    self._stream_buffers[stream_id] += event.data
                    print(f"[QUIC] Client stream {stream_id}: received {len(event.data)} bytes, buffer now {len(self._stream_buffers[stream_id])} bytes")
                    
                    # Send flow control updates
                    self.transmit()
                    
                    # Try to decode complete messages (delimited by newline)
                    while b'\n' in self._stream_buffers[stream_id]:
                        message_data, self._stream_buffers[stream_id] = self._stream_buffers[stream_id].split(b'\n', 1)
                        if message_data:
                            try:
                                data_str = message_data.decode('utf-8')
                                message = json.loads(data_str)
                                msg_type = message.get('type', 'unknown')
                                print(f"[QUIC] Client decoded message type '{msg_type}' from stream {stream_id}")
                                # Handle message asynchronously
                                if self.client:
                                    asyncio.create_task(self.client._handle_quic_message_async(message))
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"[QUIC] Client error decoding message: {e}")
                    
                    # If stream ended and buffer has remaining data, try to process it
                    if event.end_stream and self._stream_buffers[stream_id]:
                        print(f"[QUIC] Client stream {stream_id} ended with {len(self._stream_buffers[stream_id])} bytes remaining")
                        try:
                            data_str = self._stream_buffers[stream_id].decode('utf-8')
                            message = json.loads(data_str)
                            msg_type = message.get('type', 'unknown')
                            print(f"[QUIC] Client decoded end-of-stream message type '{msg_type}'")
                            if self.client:
                                asyncio.create_task(self.client._handle_quic_message_async(message))
                            self._stream_buffers[stream_id] = b''
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"[QUIC] Client error decoding remaining buffer: {e}")
                except Exception as e:
                    print(f"[QUIC] Client error handling event: {e}")
                    import traceback
                    traceback.print_exc()
else:
    UnifiedClientQUICProtocol = None

# Define HTTP/3 protocol handler if available
if HTTP3_AVAILABLE and QuicConnectionProtocol is not None:
    class UnifiedClientHTTP3Protocol(QuicConnectionProtocol):
        """HTTP/3 protocol handler for receiving server messages"""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.client = None  # Set by factory function
            self._http = None  # H3Connection instance
            self._stream_buffers = {}  # Instance-specific stream buffers
            self._stream_content_lengths = {}
            self._stream_status_codes = {}
            self._response_waiters = {}

        def create_response_waiter(self, stream_id: int):
            waiter = asyncio.get_running_loop().create_future()
            self._response_waiters[stream_id] = waiter
            return waiter

        def _finish_response_waiter(self, stream_id: int, *, result=None, error: Exception = None):
            waiter = self._response_waiters.pop(stream_id, None)
            if waiter is None or waiter.done():
                return
            if error is not None:
                waiter.set_exception(error)
            else:
                waiter.set_result(result)
        
        def quic_event_received(self, event):
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
                    status = headers.get(b":status", b"").decode()
                    
                    # Initialize buffer for this stream
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                    content_length = headers.get(b"content-length")
                    self._stream_content_lengths[stream_id] = int(content_length) if content_length else None
                    self._stream_status_codes[stream_id] = int(status) if status else None
                except Exception as e:
                    print(f"[HTTP/3] Client error handling headers: {e}")
            
            elif isinstance(event, DataReceived):
                try:
                    stream_id = event.stream_id
                    # Get or create buffer for this stream
                    if stream_id not in self._stream_buffers:
                        self._stream_buffers[stream_id] = b''
                    
                    # Append new data to buffer
                    self._stream_buffers[stream_id] += event.data
                    
                    # Send flow control updates
                    self.transmit()
                    
                    expected_length = self._stream_content_lengths.get(stream_id)
                    received_length = len(self._stream_buffers[stream_id])
                    response_complete = (
                        (expected_length is not None and expected_length > 0 and received_length >= expected_length)
                        or getattr(event, "stream_ended", False)
                    )
                    
                    if response_complete:
                        try:
                            data_str = self._stream_buffers[stream_id].decode('utf-8')
                            message = json.loads(data_str) if data_str else {}

                            status_code = self._stream_status_codes.get(stream_id)
                            if stream_id in self._response_waiters:
                                if status_code is not None and status_code >= 400:
                                    self._finish_response_waiter(
                                        stream_id,
                                        error=ConnectionError(f"HTTP/3 request on stream {stream_id} failed with status {status_code}")
                                    )
                                else:
                                    self._finish_response_waiter(
                                        stream_id,
                                        result={"status": status_code, "body": message}
                                    )
                            
                            # Handle message asynchronously
                            if self.client and isinstance(message, dict) and 'type' in message:
                                asyncio.create_task(self.client._handle_http3_message_async(message))
                            
                            # Clear buffer
                            self._stream_buffers[stream_id] = b''
                            self._stream_content_lengths.pop(stream_id, None)
                            self._stream_status_codes.pop(stream_id, None)
                        except (json.JSONDecodeError, UnicodeDecodeError) as e:
                            print(f"[HTTP/3] Client error decoding message: {e}")
                            if stream_id in self._response_waiters:
                                self._finish_response_waiter(
                                    stream_id,
                                    error=ConnectionError(f"HTTP/3 response decoding failed on stream {stream_id}: {e}")
                                )
                            self._stream_buffers[stream_id] = b''
                            self._stream_content_lengths.pop(stream_id, None)
                            self._stream_status_codes.pop(stream_id, None)
                except Exception as e:
                    print(f"[HTTP/3] Client error handling data: {e}")
                    import traceback
                    traceback.print_exc()
            
            elif StreamReset is not None and isinstance(event, StreamReset):
                # Stream was reset, clear buffer
                stream_id = event.stream_id
                if stream_id in self._stream_buffers:
                    del self._stream_buffers[stream_id]
                self._stream_content_lengths.pop(stream_id, None)
                self._stream_status_codes.pop(stream_id, None)
                self._finish_response_waiter(
                    stream_id,
                    error=ConnectionError(f"HTTP/3 stream {stream_id} was reset")
                )
else:
    UnifiedClientHTTP3Protocol = None

# Detect Docker environment and set project root
if os.path.exists('/app'):
    project_root = '/app'
else:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# packet_logger lives in scripts/utilities (Docker: /app/scripts/utilities, local: project_root/scripts/utilities)
_utilities_path = os.path.join(project_root, 'scripts', 'utilities')
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)

from packet_logger import init_db, log_sent_packet, log_received_packet, get_round_bytes_sent_received
from client_fl_metrics_log import append_client_fl_metrics_record, use_case_from_env
try:
    from q_learning_logger import init_db as init_qlearning_db, log_q_step, rl_state_network_kwargs
except ImportError:
    init_qlearning_db = None
    log_q_step = None

    def rl_state_network_kwargs(_state=None):
        return {}

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from rl_q_learning_selector import (
        QLearningProtocolSelector,
        EnvironmentStateManager,
        finalize_rl_boundary_collection_and_start_training,
        normalize_coarse_network_scenario,
    )
except ImportError:
    QLearningProtocolSelector = None
    EnvironmentStateManager = None
    finalize_rl_boundary_collection_and_start_training = None
    normalize_coarse_network_scenario = None

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
if _xla_flags:
    sanitized_flags = [f for f in _xla_flags.split() if f != "--xla_gpu_enable_command_buffer="]
    if sanitized_flags:
        os.environ["XLA_FLAGS"] = " ".join(sanitized_flags)
    else:
        os.environ.pop("XLA_FLAGS", None)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Configure GPU - use GPU_DEVICE_ID env var if set, else CUDA_VISIBLE_DEVICES, else default to '0'
_gpu_id = os.environ.get('GPU_DEVICE_ID', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu_id

# Ensure pip-installed CUDA 12 ptxas is on PATH (system ptxas may be too old)
import sys as _sys
_nvcc_bin = os.path.join(_sys.prefix, 'lib', 'python' + '.'.join(map(str, _sys.version_info[:2])),
                        'site-packages', 'nvidia', 'cuda_nvcc', 'bin')
if os.path.isdir(_nvcc_bin) and _nvcc_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _nvcc_bin + ':' + os.environ.get('PATH', '')

# Remove stale CUDA 10.x paths from LD_LIBRARY_PATH to avoid library conflicts
_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
_ld_path = ':'.join(p for p in _ld_path.split(':') if p and 'cuda-10' not in p)
os.environ['LD_LIBRARY_PATH'] = _ld_path

try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Enable memory growth to prevent OOM errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Set GPU 0 as the only visible device
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"[GPU] Configured to use GPU 0: {gpus[0]}")
        print(f"[GPU] Memory growth enabled")
    else:
        print("[WARNING] No GPU devices found, using CPU")
except Exception as e:
    print(f"[WARNING] GPU configuration failed: {e}, using CPU")

# Environment variables
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
USE_RL_SELECTION = os.getenv("USE_RL_SELECTION", "true").lower() == "true"
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
DEFAULT_DATA_BATCH_SIZE = int(os.getenv("DEFAULT_DATA_BATCH_SIZE", "16"))
ENABLE_LOCAL_CONVERGENCE_STOP = os.getenv("ENABLE_LOCAL_CONVERGENCE_STOP", "false").lower() == "true"
# Controls whether this client should signal/exit on local convergence (see fl_termination_env).
from fl_termination_env import stop_on_client_convergence
# When True, training ends when Q-learning value converges; when False, ends on accuracy convergence
USE_QL_CONVERGENCE = os.getenv("USE_QL_CONVERGENCE", "false").lower() == "true"
# Epsilon-greedy exploration for protocol selection (actual RL training). Independent of USE_QL_CONVERGENCE
# when USE_RL_EXPLORATION is set explicitly; if unset, matches USE_QL_CONVERGENCE for backward compatibility.
_ue = os.getenv("USE_RL_EXPLORATION", "").strip().lower()
if _ue in ("true", "1", "yes"):
    USE_RL_EXPLORATION = True
elif _ue in ("false", "0", "no"):
    USE_RL_EXPLORATION = False
else:
    USE_RL_EXPLORATION = USE_QL_CONVERGENCE
Q_CONVERGENCE_THRESHOLD = float(os.getenv("Q_CONVERGENCE_THRESHOLD", "0.01"))
Q_CONVERGENCE_PATIENCE = int(os.getenv("Q_CONVERGENCE_PATIENCE", "5"))
# Phase 1 — data collection: one sample per FL evaluation round with epsilon=1, no Q-updates (0 = skip → train immediately).
# The federated job length is NUM_ROUNDS (server). Phase 1 needs at least RL_PHASE0_ROUNDS FL rounds to finish
# collecting samples and run boundary computation; add more rounds for Phase 3 Q-learning after that.
RL_PHASE0_ROUNDS = int(os.getenv("RL_PHASE0_ROUNDS", "20"))
# When true (default), unified RL training runs: collect → compute boundaries → Q-learning training
RL_BOUNDARY_PIPELINE = os.getenv("RL_BOUNDARY_PIPELINE", "true").lower() in ("1", "true", "yes")
USE_COMMUNICATION_MODEL_REWARD = os.getenv("USE_COMMUNICATION_MODEL_REWARD", "true").lower() == "true"

# One-shot hints when T_calc / R_tcalc stay zero in q_learning_log
_T_CALC_IMPORT_WARNED = False
_T_CALC_IPERF_MISSING_WARNED = False


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "y")


# When True, select_protocol uses only frozen converged_protocol_by_scenario (inference path).
# Default False so USE_RL_EXPLORATION=false still greedy-selects from the Q-table / converged map, not MQTT fallback.
RL_INFERENCE_ONLY = _env_truthy("RL_INFERENCE_ONLY")
_RL_PROTOCOL_SELECTION_RECORD_LEARNING = not RL_INFERENCE_ONLY

# Skip loading any on-disk Q-table (zeros + default epsilon). Overrides shared_data / PRETRAINED discovery.
RL_FRESH_Q_TABLE = _env_truthy("RL_FRESH_Q_TABLE")
# Optional explicit pickle paths for inference or a chosen checkpoint (highest priority if set and file exists).
_RL_Q_TABLE_UPLINK_FILE = (os.getenv("RL_Q_TABLE_UPLINK_PATH") or os.getenv("RL_Q_TABLE_UPLINK") or "").strip()
_RL_Q_TABLE_DOWNLINK_FILE = (os.getenv("RL_Q_TABLE_DOWNLINK_PATH") or os.getenv("RL_Q_TABLE_DOWNLINK") or "").strip()

_num_rounds_env = os.getenv("NUM_ROUNDS", "").strip()
if (
    _num_rounds_env
    and RL_BOUNDARY_PIPELINE
    and RL_PHASE0_ROUNDS > 0
    and not RL_FRESH_Q_TABLE
):
    try:
        _nr = int(_num_rounds_env)
        if _nr < RL_PHASE0_ROUNDS:
            print(
                f"[RL] Warning: NUM_ROUNDS={_nr} < RL_PHASE0_ROUNDS={RL_PHASE0_ROUNDS}. "
                f"Phase 1 (boundary data collection) will not complete before FL stops; "
                f"raise NUM_ROUNDS to at least {RL_PHASE0_ROUNDS} (plus rounds for Phase 3), "
                f"or lower RL_PHASE0_ROUNDS."
            )
    except ValueError:
        pass


def _coarse_network_bucket_for_scenario(name: str) -> str:
    """Map NetworkSimulator-style scenario names onto the three RL ``network`` buckets (for logs/metrics)."""
    if normalize_coarse_network_scenario is not None:
        return normalize_coarse_network_scenario(name)
    n = (name or "").strip().lower()
    if n in ("excellent", "good"):
        return "excellent"
    if n in ("moderate", "congested_light", "dynamic"):
        return "moderate"
    if n in ("poor", "very_poor", "satellite", "congested_moderate", "congested_heavy"):
        return "poor"
    return "moderate"


def _effective_rl_network_scenario_label_from_env() -> Optional[str]:
    """
    Explicit experiment/simulator label for Q-table ``network_scenario`` indexing.

    When set, used for **both** training and inference so greedy selection reads the same
    Q slices that were updated during training (avoids inference-only ``measure_network_condition``
    classifying broker RTT as one bucket while training used ``RL_REWARD_SCENARIO``).
    Priority: RL_REWARD_SCENARIO, then RL_STATE_NETWORK_SCENARIO, NETWORK_SCENARIO, SIMULATOR_SCENARIO.
    """
    rs = os.environ.get("RL_REWARD_SCENARIO", "").strip().lower()
    if rs:
        return rs
    for key in ("RL_STATE_NETWORK_SCENARIO", "NETWORK_SCENARIO", "SIMULATOR_SCENARIO"):
        v = os.environ.get(key, "").strip().lower()
        if v:
            return v
    return None


# Energy / battery model constants (tunable)
k_tx = 1e-8
k_rx = 1e-8
E_fixed = 0.1  # J
P_CPU_MAX = 10.0  # W
BATTERY_CAP_J = 60.0 * 3600.0  # 60 Wh example capacity (Joules)
PROTOCOL_ENERGY_ALPHA = {
    "mqtt": 1.0,
    "amqp": 1.1,
    "grpc": 1.2,
    "quic": 1.1,
    "http3": 1.25,
    "dds": 1.1,
}
PROTOCOL_CPU_BETA = {
    "mqtt": 1.0,
    "amqp": 1.05,
    "grpc": 1.1,
    "quic": 1.05,
    "http3": 1.15,
    "dds": 1.0,
}

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", 'localhost')
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"
TOPIC_TRAINING_COMPLETE = "fl/training_complete"

# DDS Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))

# Protocol max payload sizes (unified spec: AMQP/MQTT 128KB, gRPC 4MB, HTTP/3 16KB, DDS 64KB)
MQTT_MAX_PAYLOAD_BYTES = 128 * 1024   # 128 KB
AMQP_MAX_FRAME_BYTES = 128 * 1024    # 128 KB
# gRPC: set in grpc import block (GRPC_MAX_MESSAGE_BYTES / GRPC_CHUNK_SIZE)
HTTP3_MAX_STREAM_DATA_BYTES = 16 * 1024   # HTTP/3: 16 KB per stream
CHUNK_SIZE = 64 * 1024                # DDS: 64 KB per chunk
MQTT_UPDATE_CHUNK_BYTES = 96 * 1024
AMQP_UPDATE_CHUNK_BYTES = 96 * 1024
HTTP3_UPDATE_CHUNK_BYTES = 12 * 1024

# DDS Data Structures (must be defined at module level for Python 3.8)
if DDS_AVAILABLE:
    from dataclasses import dataclass, field
    
    @dataclass
    class GlobalModel(IdlStruct):
        round: int
        weights: sequence[int]  # CycloneDDS sequence type for sequence<octet> in IDL
    
    @dataclass
    class GlobalModelChunk(IdlStruct):
        round: int
        chunk_id: int
        total_chunks: int
        payload: sequence[int]
        model_config_json: str = ""  # JSON string containing model configuration
        server_sent_unix: float = 0.0
    
    @dataclass
    class TrainingCommand(IdlStruct):
        round: int
        start_training: bool
        start_evaluation: bool
        training_complete: bool
    
    @dataclass
    class ModelUpdate(IdlStruct):
        client_id: int
        round: int
        weights: sequence[int]  # CycloneDDS sequence type for sequence<octet> in IDL
        num_samples: int
        loss: float
        accuracy: float
    
    @dataclass
    class ModelUpdateChunk(IdlStruct):
        client_id: int
        round: int
        chunk_id: int
        total_chunks: int
        payload: sequence[int]
        num_samples: int
        loss: float
        accuracy: float
    
    @dataclass
    class EvaluationMetrics(IdlStruct):
        client_id: int
        round: int
        num_samples: int
        loss: float
        accuracy: float
        client_converged: float = 0.0
        battery_soc: float = 1.0
        training_time_sec: float = 0.0
        round_time_sec: float = 0.0
        uplink_model_comm_sec: float = 0.0


class UnifiedFLClient_Emotion:
    """
    Unified Federated Learning Client for Emotion Recognition
    Uses RL to select best protocol, but behaves identically to single-protocol clients
    """
    
    def __init__(self, client_id: int, num_clients: int, train_generator, validation_generator):
        """
        Initialize Unified FL Client
        
        Args:
            client_id: Unique client identifier
            num_clients: Total number of clients in FL
            train_generator: Training data generator
            validation_generator: Validation data generator
        """
        self.client_id = client_id
        self.num_clients = num_clients
        
        # Data generators
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        
        # Model
        self.model = None
        self.current_round = 0
        self.last_global_round = -1
        self.last_training_round = -1
        self.pending_start_training_round = None
        self.pending_start_evaluation_round = None
        self.grpc_registered = False
        self.protocol_listeners_started = False
        self.evaluated_rounds = set()
        self.waiting_for_aggregated_model = False  # Track if we sent update and waiting for aggregated model
        self.is_active = True
        self.has_converged = False
        self.shutdown_requested = False
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        
        # Track gRPC signals to avoid acting on persistent flags multiple times
        self.last_grpc_train_signal_round = -1
        self.last_grpc_eval_signal_round = -1
        self.last_protocol_query_key = None
        
        # Training configuration
        self.training_config = {"batch_size": 16, "local_epochs": 20}
        
        # RL Components: two SEPARATE agents - one for UPLINK, one for DOWNLINK
        # Each has its own Q-table (.pkl), episode history, and reward parameters
        if USE_RL_SELECTION and QLearningProtocolSelector is not None:
            # --- Uplink agent paths (client -> server model uploads) ---
            if os.path.exists("/shared_data"):
                save_path_uplink = "/shared_data/q_table_emotion_uplink_trained.pkl"
            else:
                save_path_uplink = f"q_table_emotion_uplink_client_{client_id}.pkl"
            initial_load_path_uplink = None
            if RL_FRESH_Q_TABLE:
                initial_load_path_uplink = None
            elif _RL_Q_TABLE_UPLINK_FILE:
                if os.path.isfile(_RL_Q_TABLE_UPLINK_FILE):
                    initial_load_path_uplink = _RL_Q_TABLE_UPLINK_FILE
                    print(
                        f"[Client {client_id}] RL: loading uplink Q-table from RL_Q_TABLE_UPLINK_PATH="
                        f"{_RL_Q_TABLE_UPLINK_FILE!r}"
                    )
                else:
                    print(
                        f"[Client {client_id}] RL: RL_Q_TABLE_UPLINK_PATH file not found "
                        f"({_RL_Q_TABLE_UPLINK_FILE!r}); falling back to shared/pretrained discovery"
                    )
            if initial_load_path_uplink is None and not RL_FRESH_Q_TABLE:
                if os.path.exists("/shared_data"):
                    # Prefer new uplink-specific table; fall back to old shared table for backwards compat
                    for _ul_cand in ("/shared_data/q_table_emotion_uplink_trained.pkl",
                                     "/shared_data/q_table_emotion_trained.pkl"):
                        if os.path.exists(_ul_cand):
                            initial_load_path_uplink = _ul_cand
                            break
            if initial_load_path_uplink is None and not RL_FRESH_Q_TABLE:
                pretrained_dir = os.getenv("PRETRAINED_Q_TABLE_DIR")
                if pretrained_dir:
                    for candidate in (
                        os.path.join(pretrained_dir, "q_table_emotion_uplink_trained.pkl"),
                        os.path.join(pretrained_dir, "q_table_emotion_trained.pkl"),
                        os.path.join(pretrained_dir, f"q_table_emotion_uplink_client_{client_id}.pkl"),
                        os.path.join(pretrained_dir, f"q_table_emotion_client_{client_id}.pkl"),
                    ):
                        if os.path.exists(candidate):
                            initial_load_path_uplink = candidate
                            break
            self.rl_selector_uplink = QLearningProtocolSelector(
                save_path=save_path_uplink,
                initial_load_path=initial_load_path_uplink,
                use_communication_model_reward=USE_COMMUNICATION_MODEL_REWARD,
            )

            # --- Downlink agent paths (server -> client model downloads) ---
            if os.path.exists("/shared_data"):
                save_path_downlink = "/shared_data/q_table_emotion_downlink_trained.pkl"
            else:
                save_path_downlink = f"q_table_emotion_downlink_client_{client_id}.pkl"
            initial_load_path_downlink = None
            if RL_FRESH_Q_TABLE:
                initial_load_path_downlink = None
            elif _RL_Q_TABLE_DOWNLINK_FILE:
                if os.path.isfile(_RL_Q_TABLE_DOWNLINK_FILE):
                    initial_load_path_downlink = _RL_Q_TABLE_DOWNLINK_FILE
                    print(
                        f"[Client {client_id}] RL: loading downlink Q-table from RL_Q_TABLE_DOWNLINK_PATH="
                        f"{_RL_Q_TABLE_DOWNLINK_FILE!r}"
                    )
                else:
                    print(
                        f"[Client {client_id}] RL: RL_Q_TABLE_DOWNLINK_PATH file not found "
                        f"({_RL_Q_TABLE_DOWNLINK_FILE!r}); falling back to shared/pretrained discovery"
                    )
            if initial_load_path_downlink is None and not RL_FRESH_Q_TABLE:
                if os.path.exists("/shared_data") and os.path.exists("/shared_data/q_table_emotion_downlink_trained.pkl"):
                    initial_load_path_downlink = "/shared_data/q_table_emotion_downlink_trained.pkl"
            if initial_load_path_downlink is None and not RL_FRESH_Q_TABLE:
                pretrained_dir = os.getenv("PRETRAINED_Q_TABLE_DIR")
                if pretrained_dir:
                    for candidate in (
                        os.path.join(pretrained_dir, "q_table_emotion_downlink_trained.pkl"),
                        os.path.join(pretrained_dir, f"q_table_emotion_downlink_client_{client_id}.pkl"),
                    ):
                        if os.path.exists(candidate):
                            initial_load_path_downlink = candidate
                            break
            self.rl_selector_downlink = QLearningProtocolSelector(
                save_path=save_path_downlink,
                initial_load_path=initial_load_path_downlink,
                use_communication_model_reward=USE_COMMUNICATION_MODEL_REWARD,
            )

            # Backward-compat alias: self.rl_selector always points to the uplink agent
            self.rl_selector = self.rl_selector_uplink
            
            # Reset epsilon if RESET_EPSILON environment variable is set OR reset flag file exists
            # This handles both new experiments and late-joining clients
            should_reset = False
            reset_flag_file = None
            scenario_info = None
            experiment_id = None
            reset_epsilon_value = None
            
            # Check environment variable first
            if os.getenv("RESET_EPSILON", "false").lower() == "true":
                should_reset = True
                print(f"[Client {client_id}] RESET_EPSILON environment variable detected")
                # Clear the environment variable so it doesn't reset again
                os.environ["RESET_EPSILON"] = "false"
            
            # Check reset flag file (for Docker containers where env vars might not be passed)
            # Check both Docker path (/shared_data) and local path
            flag_paths = [
                "/shared_data/reset_epsilon_flag.txt",  # Docker container path
                "./shared_data/reset_epsilon_flag.txt",  # Local development path
                os.path.join(os.path.dirname(__file__), "..", "..", "shared_data", "reset_epsilon_flag.txt")  # Alternative local path
            ]
            
            for check_path in flag_paths:
                if os.path.exists(check_path):
                    reset_flag_file = check_path
                    # Read scenario, experiment ID, and reset_epsilon flag from file
                    try:
                        with open(check_path, 'r') as f:
                            content = f.read()
                            # Parse all fields
                            for line in content.split('\n'):
                                if line.startswith('scenario='):
                                    scenario_info = line.split('=', 1)[1].strip()
                                elif line.startswith('experiment_id='):
                                    experiment_id = line.split('=', 1)[1]
                                elif line.startswith('reset_epsilon='):
                                    reset_epsilon_value = line.split('=', 1)[1].strip()
                        # Only set should_reset if reset_epsilon is explicitly enabled (1.0 or true)
                        if reset_epsilon_value and (reset_epsilon_value == "1.0" or reset_epsilon_value.lower() == "true"):
                            should_reset = True
                        else:
                            print(f"[Client {client_id}] Flag file exists but reset_epsilon={reset_epsilon_value} (resume mode)")
                            should_reset = False
                    except Exception as e:
                        print(f"[Client {client_id}] Warning: Could not read flag file content: {e}")
                    break
            
            # Check if we need to reset based on experiment ID change and reset_epsilon flag
            # Using experiment_id ensures epsilon resets for EACH new GUI experiment (when reset is enabled)
            # Both uplink and downlink agents get reset together
            if should_reset:
                # Check if this is a new experiment (avoid resetting multiple times for same experiment)
                last_experiment_id = getattr(self.rl_selector_uplink, 'last_experiment_id', None)
                if experiment_id and experiment_id == last_experiment_id:
                    print(f"[Client {client_id}] Already processed experiment '{experiment_id}', skipping reset")
                    print(f"[Client {client_id}] Scenario: {scenario_info}")
                    should_reset = False
                else:
                    print(f"\n{'='*70}")
                    print(
                        f"[Client {client_id}] 🔄 FRESH RL TRAINING: archive pickle, zero this scenario's Q-slice, "
                        f"epsilon→1.0 (uplink + downlink)"
                    )
                    if experiment_id:
                        print(f"[Client {client_id}]   New experiment ID: {experiment_id}")
                        if last_experiment_id:
                            print(f"[Client {client_id}]   Previous experiment ID: {last_experiment_id}")
                    if scenario_info:
                        print(f"[Client {client_id}]   Training scenario: {scenario_info}")
                    print(f"[Client {client_id}]   Uplink epsilon before reset: {self.rl_selector_uplink.epsilon:.4f}")
                    print(f"[Client {client_id}]   Downlink epsilon before reset: {self.rl_selector_downlink.epsilon:.4f}")
                    uplink_arch = self.rl_selector_uplink.begin_fresh_training_for_scenario(
                        scenario_info, experiment_id=experiment_id
                    )
                    downlink_arch = self.rl_selector_downlink.begin_fresh_training_for_scenario(
                        scenario_info, experiment_id=experiment_id
                    )
                    print(f"[Client {client_id}]   ✓ Uplink epsilon: {self.rl_selector_uplink.epsilon:.4f}")
                    print(f"[Client {client_id}]   ✓ Downlink epsilon: {self.rl_selector_downlink.epsilon:.4f}")
                    if uplink_arch:
                        print(f"[Client {client_id}]   Uplink archive: {uplink_arch}")
                    if downlink_arch:
                        print(f"[Client {client_id}]   Downlink archive: {downlink_arch}")
                    print(
                        f"[Client {client_id}]   Canonical Q-tables (multi-scenario, inference-ready): "
                        f"{self.rl_selector_uplink.save_path} / {self.rl_selector_downlink.save_path}"
                    )
                    print(f"{'='*70}\n")

                    # Save both Q-tables immediately to persist the reset and experiment tracking
                    try:
                        self.rl_selector_uplink.save_q_table()
                    except Exception as e:
                        print(f"[Client {client_id}] Warning: Could not save uplink Q-table after epsilon reset: {e}")
                    try:
                        self.rl_selector_downlink.save_q_table()
                    except Exception as e:
                        print(f"[Client {client_id}] Warning: Could not save downlink Q-table after epsilon reset: {e}")
            elif reset_flag_file and experiment_id:
                # Flag file exists but reset is disabled - this is resume mode
                last_experiment_id = getattr(self.rl_selector_uplink, 'last_experiment_id', None)
                if experiment_id != last_experiment_id:
                    print(f"\n{'='*70}")
                    print(f"[Client {client_id}] 📍 CONTINUING WITH PREVIOUS EPSILON (Resume Mode)")
                    if experiment_id:
                        print(f"[Client {client_id}]   Experiment ID: {experiment_id}")
                    if scenario_info:
                        print(f"[Client {client_id}]   Training scenario: {scenario_info}")
                    print(f"[Client {client_id}]   Uplink epsilon (preserved): {self.rl_selector_uplink.epsilon:.4f}")
                    print(f"[Client {client_id}]   Downlink epsilon (preserved): {self.rl_selector_downlink.epsilon:.4f}")
                    print(f"[Client {client_id}]   📊 Q-tables, rewards, and learning progress continue from previous state")
                    print(f"{'='*70}\n")
                    # Update experiment ID on both agents
                    self.rl_selector_uplink.last_experiment_id = experiment_id
                    self.rl_selector_downlink.last_experiment_id = experiment_id
                    if scenario_info:
                        _sn = scenario_info.strip().lower()
                        self.rl_selector_uplink.ensure_scenario(_sn)
                        self.rl_selector_downlink.ensure_scenario(_sn)
                    # Save both Q-tables to persist the tracking
                    try:
                        self.rl_selector_uplink.save_q_table()
                    except Exception as e:
                        print(f"[Client {client_id}] Warning: Could not save uplink Q-table: {e}")
                    try:
                        self.rl_selector_downlink.save_q_table()
                    except Exception as e:
                        print(f"[Client {client_id}] Warning: Could not save downlink Q-table: {e}")
            
            # Note: We DON'T delete the flag file here because:
                # 1. Late-joining clients also need to reset epsilon
                # 2. The flag file will be overwritten/cleaned when next experiment starts
                # 3. Experiment ID tracking prevents multiple resets for the same experiment
            
            self.env_manager = EnvironmentStateManager()
            self.env_manager.sync_comm_thresholds_from_selector(self.rl_selector_uplink)

            # Phase 1 (data collection) runs during real FL rounds in evaluate_model — not here.
            self._rl_boundary_collection_phase = bool(
                RL_BOUNDARY_PIPELINE
                and RL_PHASE0_ROUNDS > 0
                and USE_RL_EXPLORATION
                and finalize_rl_boundary_collection_and_start_training is not None
            )
            self._rl_boundary_comm_samples: List[float] = []
            self._rl_boundary_downlink_comm_samples: List[float] = []
            self._rl_boundary_res_samples: List[float] = []
            self._rl_boundary_batt_samples: List[float] = []
            self._rl_boundary_res_samples_dl: List[float] = []
            self._rl_boundary_batt_samples_dl: List[float] = []
            self._last_downlink_comm_wall_s: float = 0.0
            self._last_uplink_rl_state: Optional[Dict] = None
            if self._rl_boundary_collection_phase:
                self.rl_selector_uplink.epsilon = 1.0
                self.rl_selector_downlink.epsilon = 1.0
                print(
                    f"[Client {client_id}] RL pipeline: Phase 1 = first {RL_PHASE0_ROUNDS} FL rounds "
                    f"(uplink+downlink wall comm time, resource load, battery SoC; epsilon=1; no Q-updates). "
                    f"Then Phase 2 = boundaries, Phase 3 = Q-learning training."
                )

            if scenario_info:
                _sn = scenario_info.strip().lower()
                self.env_manager.update_network_scenario(
                    _sn,
                    rl_uplink=self.rl_selector_uplink,
                    rl_downlink=self.rl_selector_downlink,
                )
                self.env_manager.update_network_condition(_coarse_network_bucket_for_scenario(_sn))
                self.rl_selector_uplink.ensure_scenario(_sn)
                self.rl_selector_downlink.ensure_scenario(_sn)

            if os.getenv("RL_PRINT_CANONICAL_Q_VALUES", "").strip().lower() in (
                "1", "true", "yes", "y",
            ):
                print(f"\n[Client {client_id}] Canonical Q-row report — uplink:")
                self.rl_selector_uplink.print_canonical_q_values_report()
                print(f"\n[Client {client_id}] Canonical Q-row report — downlink:")
                self.rl_selector_downlink.print_canonical_q_values_report()
                print()
        else:
            self.rl_selector_uplink = None
            self.rl_selector_downlink = None
            self.rl_selector = None
            self.env_manager = None
            self._rl_boundary_collection_phase = False

        # Track recent network latency samples for mobility estimation
        self.latency_history: List[float] = []
        
        # Track selected protocol and metrics
        self.selected_protocol = None
        self.selected_downlink_protocol = 'grpc'
        self.round_metrics = {
            'communication_time': 0.0,
            # Client→server only (uplink); downlink is tracked separately in downlink RL / logs
            'uplink_model_comm_time': 0.0,
            'uplink_metrics_comm_time': 0.0,
            'training_time': 0.0,
            'accuracy': 0.0,
            'success': False
        }
        self._last_rl_state = None  # for Q-learning log (state at time of uplink action)
        self._last_update_protocol = None  # protocol used to send local model update
        self._last_downlink_rl_state = None  # state at time of downlink protocol selection
        self._downlink_select_time = None    # fallback when server_sent_unix is unavailable (legacy)
        self._downlink_server_sent_unix = None  # server wall time when downlink send started
        self._downlink_receive_complete_unix = None  # client wall time when model apply finished
        
        # DDS chunk reassembly buffers (FAIR CONFIG: matching standalone)
        self.global_model_chunks = {}  # {chunk_id: payload}
        self.global_model_metadata = {}  # {round, total_chunks, model_config_json}
        
        # DDS Components
        if DDS_AVAILABLE:
            try:
                # Create DDS participant
                self.dds_participant = DomainParticipant(DDS_DOMAIN_ID)
                
                # Reliable QoS for critical control messages (registration, config, commands)
                reliable_qos = Qos(
                    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                    Policy.History.KeepLast(10),
                    Policy.Durability.TransientLocal
                )
                
                # Best effort QoS for non-critical bulk paths
                best_effort_qos = Qos(
                    Policy.Reliability.BestEffort,
                    Policy.History.KeepLast(1),
                )

                # Reliable QoS for chunked model transfer to prevent dropped chunks.
                chunk_qos = Qos(
                    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                    Policy.History.KeepLast(2048),
                    Policy.Durability.TransientLocal
                )
                
                # Create topics
                global_model_topic = Topic(self.dds_participant, "GlobalModel", GlobalModel)
                global_model_chunk_topic = Topic(self.dds_participant, "GlobalModelChunk", GlobalModelChunk)
                self.dds_update_topic = Topic(self.dds_participant, "ModelUpdate", ModelUpdate)
                update_chunk_topic = Topic(self.dds_participant, "ModelUpdateChunk", ModelUpdateChunk)
                self.dds_metrics_topic = Topic(self.dds_participant, "EvaluationMetrics", EvaluationMetrics)
                
                # Create readers (for receiving from server)
                # Use reliable QoS for chunked model data
                self.dds_global_model_reader = DataReader(self.dds_participant, global_model_topic, qos=best_effort_qos)
                self.dds_global_model_chunk_reader = DataReader(self.dds_participant, global_model_chunk_topic, qos=chunk_qos)
                
                # Create writers (for sending to server)
                # Use reliable QoS for chunked updates; metrics can remain best-effort
                self.dds_update_writer = DataWriter(self.dds_participant, self.dds_update_topic, qos=best_effort_qos)
                self.dds_update_chunk_writer = DataWriter(self.dds_participant, update_chunk_topic, qos=chunk_qos)
                self.dds_metrics_writer = DataWriter(self.dds_participant, self.dds_metrics_topic, qos=best_effort_qos)
                
                print(f"[DDS] Client {client_id} initialized on domain {DDS_DOMAIN_ID} with chunking support")
            except Exception as e:
                print(f"[DDS] Initialization failed: {e}")
                self.dds_participant = None
                self.dds_update_writer = None
                self.dds_update_chunk_writer = None
                self.dds_metrics_writer = None
                self.dds_global_model_reader = None
                self.dds_global_model_chunk_reader = None
        else:
            self.dds_participant = None
            self.dds_update_writer = None
            self.dds_update_chunk_writer = None
            self.dds_metrics_writer = None
            self.dds_global_model_reader = None
            self.dds_global_model_chunk_reader = None
        
        # Initialize packet logger and Q-learning logger
        init_db()
        # Phase 1 (RL boundary data collection): do not create/migrate q_learning_*.db until Q-logging starts
        if init_qlearning_db is not None and self._rl_q_logging_allowed():
            init_qlearning_db()

        # Pruning and quantization (flow when both enabled: receive global model -> train -> prune -> quantize -> send)
        use_pruning = os.getenv("USE_PRUNING", "0").strip() in ("1", "true", "yes")
        use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() in ("true", "1", "yes")
        self.use_pruning = use_pruning
        self.use_quantization = use_quantization
        self.pruner = None
        self.quantizer = None
        if use_pruning and PRUNING_AVAILABLE and ModelPruning is not None:
            sparsity = float(os.getenv("PRUNING_SPARSITY", "0.5"))
            self.pruner = ModelPruning(PruningConfig(target_sparsity=sparsity))
            print(f"[Client {client_id}] Pruning enabled (sparsity={sparsity})")
        if use_quantization and QUANTIZATION_AVAILABLE and Quantization is not None:
            bits = int(os.getenv("QUANTIZATION_BITS", os.getenv("QUANT_BITS", "8")))
            strategy = os.getenv("QUANTIZATION_STRATEGY", "parameter_quantization")
            symmetric = os.getenv("QUANTIZATION_SYMMETRIC", "true").lower() in ("true", "1", "yes")
            self.quantizer = Quantization(QuantizationConfig(strategy=strategy, bits=bits, symmetric=symmetric))
            print(f"[Client {client_id}] Quantization enabled (bits={bits}, strategy={strategy})")
        
        # QUIC persistent connection components
        self.quic_protocol = None
        self.quic_connection_task = None
        self.quic_loop = None
        self.quic_thread = None
        # HTTP/3 persistent connection components
        self.http3_protocol = None
        self.http3_connection_task = None
        self.http3_loop = None
        self.http3_thread = None
        self.http3_send_lock = threading.Lock()
        
        # AMQP listener components
        self.amqp_listener_connection = None
        self.amqp_listener_channel = None
        self.amqp_listener_thread = None
        
        # DDS listener components
        self.dds_listener_thread = None
        self.dds_global_model_reader = None
        self.dds_command_reader = None
        
        # gRPC listener components
        self.grpc_listener_thread = None
        self.grpc_stub = None
        self.grpc_lock = threading.Lock()  # Synchronize gRPC calls to prevent conflicts
        
        # Initialize MQTT client for data-plane compatibility and fallback signaling
        self.mqtt_client = mqtt.Client(client_id=f"fl_client_{client_id}", protocol=mqtt.MQTTv311)
        self.mqtt_client.max_inflight_messages_set(20)
        # FAIR CONFIG: Limited queue to 1000 messages (aligned with AMQP/gRPC)
        self.mqtt_client.max_queued_messages_set(1000)
        self.mqtt_client._max_packet_size = MQTT_MAX_PAYLOAD_BYTES  # 128 KB
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        print(f"\n{'='*70}")
        print(f"UNIFIED FL CLIENT - EMOTION RECOGNITION")
        print(f"{'='*70}")
        print(f"Client ID: {self.client_id}/{self.num_clients}")
        print(f"RL Protocol Selection: {'ENABLED' if USE_RL_SELECTION else 'DISABLED'}")
        if USE_RL_SELECTION:
            print(f"RL Exploration (epsilon-greedy): {'ENABLED' if USE_RL_EXPLORATION else 'DISABLED (greedy)'}")
            if RL_INFERENCE_ONLY:
                print("RL Inference-only (RL_INFERENCE_ONLY): protocol pick from converged map / frozen Q inference path")
            print(f"Q-Convergence Stop (USE_QL_CONVERGENCE): {'ENABLED' if USE_QL_CONVERGENCE else 'DISABLED'}")
            print(f"Communication Model in Reward: {'ENABLED' if USE_COMMUNICATION_MODEL_REWARD else 'DISABLED'}")
            if getattr(self, "_rl_boundary_collection_phase", False):
                print(
                    f"RL boundary pipeline: ENABLED (Phase 1: {RL_PHASE0_ROUNDS} FL rounds collect metrics → "
                    f"Phase 2: min/max tercile boundaries → Phase 3: Q-learning). "
                    f"RL_BOUNDARY_PIPELINE={RL_BOUNDARY_PIPELINE}"
                )
        print(f"{'='*70}\n")
        
        # Start protocol listeners in background threads
        self.start_all_protocol_listeners()
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print(f"Client {self.client_id} connected to MQTT broker")
            # Subscribe to topics
            self.mqtt_client.subscribe(TOPIC_GLOBAL_MODEL, qos=1)
            self.mqtt_client.subscribe(TOPIC_TRAINING_CONFIG, qos=1)
            self.mqtt_client.subscribe(TOPIC_START_TRAINING, qos=1)
            self.mqtt_client.subscribe(TOPIC_START_EVALUATION, qos=1)
            self.mqtt_client.subscribe(TOPIC_TRAINING_COMPLETE, qos=1)
            
            time.sleep(2)

            if not self.grpc_registered and not self.register_with_server_grpc():
                # Fallback only if gRPC is unavailable.
                registration_msg = json.dumps({"client_id": self.client_id})
                self.mqtt_client.publish("fl/client_register", registration_msg, qos=1)
                log_sent_packet(
                    packet_size=len(registration_msg),
                    peer="server",
                    protocol="MQTT",
                    round=0,
                    extra_info="registration_fallback"
                )
                print("  Registration fallback sent via MQTT\n")
        else:
            print(f"Client {self.client_id} failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        if not self.is_active:
            return
        try:
            log_received_packet(
                packet_size=len(msg.payload),
                peer="server",
                protocol="MQTT",
                round=self.current_round,
                extra_info=msg.topic
            )
            
            if msg.topic == TOPIC_GLOBAL_MODEL:
                self.handle_global_model(msg.payload)
            elif msg.topic == TOPIC_TRAINING_CONFIG:
                self.handle_training_config(msg.payload)
            elif msg.topic == TOPIC_START_TRAINING:
                self.handle_start_training(msg.payload)
            elif msg.topic == TOPIC_START_EVALUATION:
                self.handle_start_evaluation(msg.payload)
            elif msg.topic == TOPIC_TRAINING_COMPLETE:
                self.handle_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        if rc == 0 and self.shutdown_requested:
            self.cleanup()
            print(f"\nClient {self.client_id} clean disconnect from broker")
            print(f"Client {self.client_id} exiting...")
            self.mqtt_client.loop_stop()
            return
        else:
            print(f"Client {self.client_id} MQTT disconnected, return code {rc}")
    
    def _get_amqp_connection_parameters(self):
        """Use one consistent AMQP endpoint for listener and sends."""
        amqp_host = os.getenv("AMQP_HOST") or os.getenv("AMQP_BROKER") or "localhost"
        amqp_port = int(os.getenv("AMQP_PORT", "5672"))
        amqp_user = os.getenv("AMQP_USER", "guest")
        amqp_password = os.getenv("AMQP_PASSWORD", "guest")
        credentials = pika.PlainCredentials(amqp_user, amqp_password)
        return pika.ConnectionParameters(
            host=amqp_host,
            port=amqp_port,
            credentials=credentials,
            heartbeat=600,
            blocked_connection_timeout=600,
            connection_attempts=3,
            retry_delay=2,
            frame_max=AMQP_MAX_FRAME_BYTES,
        )

    def cleanup(self):
        """Cleanup resources"""
        if self.quic_connection_task and self.quic_loop:
            try:
                # Cancel the QUIC connection task
                self.quic_loop.call_soon_threadsafe(self.quic_connection_task.cancel)
            except:
                pass
        
        # Cleanup AMQP listener
        if self.amqp_listener_connection and not self.amqp_listener_connection.is_closed:
            try:
                self.amqp_listener_connection.close()
            except Exception:
                pass
        self.amqp_listener_connection = None
        self.amqp_listener_channel = None
    
    # =========================================================================
    # PROTOCOL LISTENERS - Start all protocol listeners for receiving responses
    # =========================================================================
    
    def start_all_protocol_listeners(self):
        """Start listeners for all protocols (mirroring single-protocol implementations)"""
        if self.protocol_listeners_started:
            print("[Client] Protocol listeners already started - skipping duplicate start")
            return
        self.protocol_listeners_started = True
        print("[Client] Starting protocol listeners...")
        
        # MQTT already started in __init__
        
        # Get list of protocols that RL selector can use
        available_protocols = []
        if self.rl_selector:
            available_protocols = self.rl_selector.PROTOCOLS
            print(f"[Client] RL Selector configured for protocols: {available_protocols}")
        
        # Start AMQP listener only if in available protocols
        if pika is not None and (not available_protocols or 'amqp' in available_protocols):
            self.start_amqp_listener()
        
        # Start DDS listener only if in available protocols
        if DDS_AVAILABLE and (not available_protocols or 'dds' in available_protocols):
            self.start_dds_listener()
        
        # Start gRPC listener only if in available protocols
        if grpc is not None and (not available_protocols or 'grpc' in available_protocols):
            self.start_grpc_listener()
        
        # Start QUIC persistent connection listener only if in available protocols
        if asyncio is not None and connect is not None and (not available_protocols or 'quic' in available_protocols):
            self.start_quic_listener()
        
        if HTTP3_AVAILABLE and asyncio is not None and connect is not None and (not available_protocols or 'http3' in available_protocols):
            self.start_http3_listener()
        
        print("[Client] All protocol listeners started\n")
    
    # -------------------------------------------------------------------------
    # AMQP LISTENER
    # -------------------------------------------------------------------------
    
    def start_amqp_listener(self):
        """Start AMQP consumer thread (mirrors FL_Client_AMQP.py)"""
        if self.amqp_listener_thread and self.amqp_listener_thread.is_alive():
            return

        def amqp_consumer_loop():
            # Retry with exponential backoff for startup race condition
            max_retries = 5
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        print(f"[AMQP] Retry {attempt}/{max_retries} after {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    
                    parameters = self._get_amqp_connection_parameters()
                    self.amqp_listener_connection = pika.BlockingConnection(parameters)
                    self.amqp_listener_channel = self.amqp_listener_connection.channel()
                    
                    # Declare client-specific queues (server creates these)
                    queue_global_model = f'client_{self.client_id}_global_model'
                    queue_start_evaluation = f'client_{self.client_id}_start_evaluation'
                    queue_start_training = f'client_{self.client_id}_start_training'
                    
                    self.amqp_listener_channel.queue_declare(queue=queue_global_model, durable=True)
                    self.amqp_listener_channel.queue_declare(queue=queue_start_evaluation, durable=True)
                    self.amqp_listener_channel.queue_declare(queue=queue_start_training, durable=True)
                    
                    # Set up consumers
                    self.amqp_listener_channel.basic_consume(
                        queue=queue_global_model,
                        on_message_callback=self.on_amqp_global_model,
                        auto_ack=True
                    )
                    self.amqp_listener_channel.basic_consume(
                        queue=queue_start_evaluation,
                        on_message_callback=self.on_amqp_start_evaluation,
                        auto_ack=True
                    )
                    self.amqp_listener_channel.basic_consume(
                        queue=queue_start_training,
                        on_message_callback=self.on_amqp_start_training,
                        auto_ack=True
                    )
                    
                    print(f"[AMQP] Listener started for client {self.client_id}")
                    
                    # Start consuming (blocks in this thread)
                    self.amqp_listener_channel.start_consuming()
                    break  # Success - exit retry loop
                    
                except Exception as e:
                    self.amqp_listener_connection = None
                    self.amqp_listener_channel = None
                    if attempt == max_retries - 1:
                        print(f"[AMQP] Listener failed after {max_retries} attempts: {e}")
                        import traceback
                        traceback.print_exc()
                    else:
                        print(f"[AMQP] Connection attempt {attempt + 1} failed: {e}")
        
        self.amqp_listener_thread = threading.Thread(target=amqp_consumer_loop, daemon=True, name=f"AMQP-Listener-{self.client_id}")
        self.amqp_listener_thread.start()
    
    def on_amqp_global_model(self, ch, method, properties, body):
        """AMQP callback: received global model"""
        try:
            log_received_packet(
                packet_size=len(body),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="global_model"
            )
            
            data = json.loads(body.decode())
            round_num = data['round']
            print(f"[AMQP] Client {self.client_id} received global model for round {round_num}")
            
            # Deserialize weights
            if 'quantized_data' in data and self.quantizer is not None:
                raw = data['quantized_data']
                if isinstance(raw, str):
                    raw = base64.b64decode(raw.encode('utf-8'))
                compressed_data = pickle.loads(raw) if isinstance(raw, bytes) else raw
                weights = self.quantizer.decompress(compressed_data)
            else:
                weights = self.deserialize_weights(data['weights'])
            
            self._global_model_receive_t0 = time.time()
            self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=data.get('model_config'),
                source="AMQP",
                server_sent_unix=data.get('server_sent_unix'),
            )
                
        except Exception as e:
            print(f"[AMQP] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_amqp_start_evaluation(self, ch, method, properties, body):
        """AMQP callback: received start evaluation signal"""
        try:
            log_received_packet(
                packet_size=len(body),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="start_evaluation"
            )
            
            data = json.loads(body.decode())
            round_num = data['round']
            
            if round_num == self.current_round and round_num not in self.evaluated_rounds:
                self.evaluated_rounds.add(round_num)  # Add BEFORE evaluate to prevent race condition
                print(f"[AMQP] Client {self.client_id} starting evaluation for round {round_num}")
                self.evaluate_model(evaluation_round_num=round_num)
                print(f"[AMQP] Client {self.client_id} evaluation completed for round {round_num}.")
                
        except Exception as e:
            print(f"[AMQP] Client {self.client_id} error handling evaluation signal: {e}")
    
    def on_amqp_start_training(self, ch, method, properties, body):
        """AMQP callback: received start training signal"""
        try:
            log_received_packet(
                packet_size=len(body),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="start_training"
            )
            
            data = json.loads(body.decode())
            round_num = data['round']
            
            # Check if model is initialized
            if self.model is None:
                self._defer_start_training(round_num, "AMQP")
                self.pending_start_training_round = round_num
                return
            
            # Check for duplicate training signals
            if self.last_training_round == round_num:
                print(f"[AMQP] Client {self.client_id} ignoring duplicate start training for round {round_num}")
                return
            
            # Check if we're ready for this round
            if self.current_round == 0 and round_num == 1:
                self.current_round = round_num
                self.last_training_round = round_num
                print(f"\n[AMQP] Client {self.client_id} starting training for round {round_num}...")
                self.train_local_model()
            elif round_num >= self.current_round and round_num <= self.current_round + 1:
                self.current_round = round_num
                self.last_training_round = round_num
                print(f"\n[AMQP] Client {self.client_id} starting training for round {round_num}...")
                self.train_local_model()
            else:
                print(f"[AMQP] Client {self.client_id} skipping training signal for round {round_num} (current: {self.current_round})")
                
        except Exception as e:
            print(f"[AMQP] Client {self.client_id} error handling training signal: {e}")
    
    # -------------------------------------------------------------------------
    # DDS LISTENER
    # -------------------------------------------------------------------------
    
    def start_dds_listener(self):
        """Start DDS reader polling thread (mirrors FL_Client_DDS.py)"""
        def dds_listener_loop():
            try:
                # Create readers for GlobalModel and TrainingCommand
                from cyclonedds.core import Qos, Policy
                from cyclonedds.util import duration
                from cyclonedds.topic import Topic
                
                reliable_qos = Qos(
                    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
                    Policy.History.KeepLast(10),
                    Policy.Durability.TransientLocal
                )
                
                topic_global_model = Topic(self.dds_participant, "GlobalModel", GlobalModel)
                topic_command = Topic(self.dds_participant, "TrainingCommand", TrainingCommand)
                
                self.dds_global_model_reader = DataReader(self.dds_participant, topic_global_model, qos=reliable_qos)
                self.dds_command_reader = DataReader(self.dds_participant, topic_command, qos=reliable_qos)
                
                print(f"[DDS] Listener started for client {self.client_id}")
                
                # Polling loop
                while True:
                    # Check for global model
                    for sample in self.dds_global_model_reader.take():
                        if sample:
                            self.on_dds_global_model(sample)
                    
                    # Check for commands
                    for sample in self.dds_command_reader.take():
                        if sample:
                            self.on_dds_command(sample)
                    
                    time.sleep(0.1)  # Poll every 100ms
                    
            except Exception as e:
                print(f"[DDS] Listener error: {e}")
                import traceback
                traceback.print_exc()
        
        self.dds_listener_thread = threading.Thread(target=dds_listener_loop, daemon=True, name=f"DDS-Listener-{self.client_id}")
        self.dds_listener_thread.start()
    
    def on_dds_global_model(self, sample):
        """DDS callback: received global model"""
        try:
            round_num = sample.round
            print(f"[DDS] Client {self.client_id} received global model for round {round_num}")
            
            log_received_packet(
                packet_size=len(sample.weights),
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="global_model"
            )
            
            # Deserialize weights (DDS uses sequence[int])
            weights_bytes = bytes(sample.weights)
            weights = pickle.loads(weights_bytes)
            
            # Update model
            if self.model and round_num > self.last_global_round:
                self.model.set_weights(weights)
                self.last_global_round = round_num
                print(f"[DDS] Client {self.client_id} updated model weights for round {round_num}")
                
        except Exception as e:
            print(f"[DDS] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_dds_command(self, sample):
        """DDS callback: received training command"""
        try:
            log_received_packet(
                packet_size=32,  # Approximate size
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="command"
            )
            
            round_num = sample.round
            
            # Handle start_training command
            if sample.start_training:
                # Check if model is initialized
                if self.model is None:
                    self._defer_start_training(round_num, "DDS")
                    self.pending_start_training_round = round_num
                    return
                
                # Check for duplicate training signals
                if self.last_training_round == round_num:
                    print(f"[DDS] Client {self.client_id} ignoring duplicate start training for round {round_num}")
                    return
                
                # Check if we're ready for this round
                if self.current_round == 0 and round_num == 1:
                    self.current_round = round_num
                    self.last_training_round = round_num
                    print(f"\n[DDS] Client {self.client_id} starting training for round {round_num}...")
                    self.train_local_model()
                elif round_num >= self.current_round and round_num <= self.current_round + 1:
                    self.current_round = round_num
                    self.last_training_round = round_num
                    print(f"\n[DDS] Client {self.client_id} starting training for round {round_num}...")
                    self.train_local_model()
                else:
                    print(f"[DDS] Client {self.client_id} skipping training signal for round {round_num} (current: {self.current_round})")
            
            # Handle start_evaluation command
            if sample.start_evaluation and sample.round == self.current_round:
                if sample.round not in self.evaluated_rounds:
                    self.evaluated_rounds.add(sample.round)  # Add BEFORE evaluate to prevent race condition
                    print(f"[DDS] Client {self.client_id} starting evaluation for round {sample.round}")
                    self.evaluate_model(evaluation_round_num=sample.round)
                    print(f"[DDS] Client {self.client_id} evaluation completed for round {sample.round}.")
                    
        except Exception as e:
            print(f"[DDS] Client {self.client_id} error handling command: {e}")
    
    # -------------------------------------------------------------------------
    # gRPC LISTENER
    # -------------------------------------------------------------------------
    
    def start_grpc_listener(self):
        """Start gRPC polling thread (mirrors FL_Client_gRPC.py)"""
        def grpc_listener_loop():
            try:
                # Create gRPC channel and stub
                grpc_host = os.getenv("GRPC_HOST", "fl-server-unified-emotion")
                grpc_port = os.getenv("GRPC_PORT", "50051")
                options = [
                    ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
                    ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
                    ('grpc.keepalive_time_ms', 600000),
                    ('grpc.keepalive_timeout_ms', 60000),
                ]
                channel = grpc.insecure_channel(f"{grpc_host}:{grpc_port}", options=options)
                self.grpc_stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
                
                print(f"[gRPC] Listener started for client {self.client_id}")
                _unavail_log_t0 = [0.0]  # throttle "available=False" logs

                # Polling loop
                while True:
                    # Control plane: keep separate from GetGlobalModel so a CheckTrainingStatus failure
                    # still allows pulling the initial model in the same second.
                    try:
                        status_request = federated_learning_pb2.StatusRequest(client_id=self.client_id)
                        status = self.grpc_stub.CheckTrainingStatus(status_request)

                        if getattr(status, 'has_protocol_query', False):
                            self._handle_grpc_protocol_query(status.protocol_query)

                        if status.should_train and self.is_active and status.current_round != self.last_grpc_train_signal_round:
                            self.last_grpc_train_signal_round = status.current_round
                            self.handle_start_training(
                                json.dumps({'round': status.current_round}).encode(),
                                source="gRPC"
                            )
                        if status.should_evaluate and self.is_active and status.current_round != self.last_grpc_eval_signal_round:
                            self.last_grpc_eval_signal_round = status.current_round
                            self.handle_start_evaluation(json.dumps({'round': status.current_round}).encode())
                    except grpc.RpcError as e:
                        print(
                            f"[gRPC] Client {self.client_id} CheckTrainingStatus RpcError (will retry): "
                            f"{e.code()} - {e.details()}"
                        )
                    except Exception as e:
                        print(f"[gRPC] Client {self.client_id} CheckTrainingStatus error: {e}")

                    try:
                        # Data plane: global model (chunked when > 4 MB)
                        req = federated_learning_pb2.ModelRequest(client_id=self.client_id, round=0, chunk_index=0)
                        first = self.grpc_stub.GetGlobalModel(req)
                        if not first.available or not first.weights:
                            response = first
                            if self.model is None and (not first.available or not first.weights):
                                now = time.time()
                                if now - _unavail_log_t0[0] >= 12.0:
                                    _unavail_log_t0[0] = now
                                    wlen = len(first.weights) if first.weights else 0
                                    print(
                                        f"[gRPC] Client {self.client_id} still waiting for global model: "
                                        f"available={first.available}, weights_bytes={wlen}, round={first.round}"
                                    )
                        else:
                            total_chunks = getattr(first, 'total_chunks', 1) or 1
                            if total_chunks <= 1:
                                response = first
                            else:
                                chunks = [first.weights]
                                for idx in range(1, total_chunks):
                                    req = federated_learning_pb2.ModelRequest(client_id=self.client_id, round=0, chunk_index=idx)
                                    part = self.grpc_stub.GetGlobalModel(req)
                                    if part.weights:
                                        chunks.append(part.weights)
                                assembled = b''.join(chunks)
                                response = federated_learning_pb2.GlobalModel(
                                    round=first.round,
                                    weights=assembled,
                                    available=True,
                                    model_config=first.model_config,
                                    chunk_index=0,
                                    total_chunks=1,
                                    server_sent_unix=float(getattr(first, 'server_sent_unix', 0.0) or 0.0),
                                )
                        accept = response.available and (
                            (self.model is None) or
                            (response.round > self.last_global_round) or
                            (response.round == self.current_round and self.waiting_for_aggregated_model)
                        )
                        if accept:
                            self.on_grpc_global_model(response)

                    except grpc.RpcError as e:
                        if self.model is None:
                            print(f"[gRPC] Client {self.client_id} GetGlobalModel RpcError (will retry): {e.code()} - {e.details()}")
                    except Exception as e:
                        print(f"[gRPC] Client {self.client_id} GetGlobalModel / apply error: {e}")
                        import traceback
                        traceback.print_exc()

                    time.sleep(1)  # Poll every second
                    
            except Exception as e:
                print(f"[gRPC] Listener error: {e}")
                import traceback
                traceback.print_exc()
        
        self.grpc_listener_thread = threading.Thread(target=grpc_listener_loop, daemon=True, name=f"gRPC-Listener-{self.client_id}")
        self.grpc_listener_thread.start()
    
    def on_grpc_global_model(self, response):
        """gRPC callback: received global model"""
        try:
            self._global_model_receive_t0 = time.time()
            round_num = response.round
            print(f"[gRPC] Client {self.client_id} received global model for round {round_num}")
            
            log_received_packet(
                packet_size=len(response.weights),
                peer="server",
                protocol="gRPC",
                round=self.current_round,
                extra_info="global_model"
            )
            
            # Deserialize weights, dequantizing when the server sent compressed payloads.
            decoded = pickle.loads(response.weights)
            if isinstance(decoded, dict) and 'compressed_data' in decoded:
                if self.quantizer is None:
                    raise ValueError("Received quantized gRPC global model but client quantizer is not initialized")
                weights = self.quantizer.decompress(decoded)
                print(f"[gRPC] Client {self.client_id} received quantized global model (dequantized to float32)")
            else:
                weights = decoded

            model_config = json.loads(response.model_config) if response.model_config else None
            srv_sent = float(getattr(response, 'server_sent_unix', 0.0) or 0.0)
            applied = self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=model_config,
                source="gRPC",
                server_sent_unix=srv_sent if srv_sent > 0.0 else None,
            )
            if applied:
                self.waiting_for_aggregated_model = False  # Clear flag: received aggregated model

                # Check if should evaluate
                status_request = federated_learning_pb2.StatusRequest(client_id=self.client_id)
                status = self.grpc_stub.CheckTrainingStatus(status_request)

                if status.should_evaluate and round_num == self.current_round:
                    if round_num not in self.evaluated_rounds:
                        self.evaluated_rounds.add(round_num)  # Add BEFORE evaluate to prevent race condition
                        print(f"[gRPC] Client {self.client_id} starting evaluation for round {round_num}")
                        self.evaluate_model(evaluation_round_num=round_num)
                        print(f"[gRPC] Client {self.client_id} evaluation completed for round {round_num}.")
                        
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # QUIC LISTENER
    # =========================================================================
    
    def start_quic_listener(self):
        """Start QUIC persistent connection in background thread"""
        if self.quic_thread is None or not self.quic_thread.is_alive():
            # Start the event loop thread
            self.quic_thread = threading.Thread(
                target=self._run_quic_loop,
                daemon=True,
                name=f"QUIC-Client-{self.client_id}"
            )
            self.quic_thread.start()
            
            # Wait for event loop to be ready
            max_wait = 2
            waited = 0
            while self.quic_loop is None and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1
            
            # Schedule the connection task in the event loop
            if self.quic_loop:
                self.quic_connection_task = asyncio.run_coroutine_threadsafe(
                    self._quic_connection_loop(),
                    self.quic_loop
                )
                time.sleep(2)  # Wait for connection attempt
            
            print(f"[QUIC] Listener started for client {self.client_id}")
    
    # =========================================================================
    # HTTP/3 LISTENER
    # =========================================================================
    
    def start_http3_listener(self):
        """Start HTTP/3 persistent connection in background thread"""
        if not HTTP3_AVAILABLE:
            return
        
        if self.http3_thread is None or not self.http3_thread.is_alive():
            # Start the event loop thread
            self.http3_thread = threading.Thread(
                target=self._run_http3_loop,
                daemon=True,
                name=f"HTTP3-Client-{self.client_id}"
            )
            self.http3_thread.start()
            
            # Wait for event loop to be ready
            max_wait = 2
            waited = 0
            while self.http3_loop is None and waited < max_wait:
                time.sleep(0.1)
                waited += 0.1
            
            # Schedule the connection task in the event loop
            if self.http3_loop:
                self.http3_connection_task = asyncio.run_coroutine_threadsafe(
                    self._http3_connection_loop(),
                    self.http3_loop
                )
                time.sleep(2)  # Wait for connection attempt
            
            print(f"[HTTP/3] Listener started for client {self.client_id}")
    
    def handle_global_model(self, payload):
        """Receive and apply global model from server"""
        try:
            data = json.loads(payload.decode()) if isinstance(payload, bytes) else payload
            round_num = data.get('round', 0)
            
            # Check for duplicate (already processed this exact model)
            if hasattr(self, 'last_global_round') and self.last_global_round == round_num and self.model is not None:
                print(f"Client {self.client_id} ignoring duplicate global model for round {round_num}")
                return
            
            self._global_model_receive_t0 = time.time()
            print(f"Client {self.client_id} received global model (round {round_num})")
            
            # Decompress/deserialize weights
            if 'quantized_data' in data:
                # Handle quantized/compressed data
                compressed_data = data['quantized_data']
                if isinstance(compressed_data, str):
                    import base64, pickle
                    compressed_data = pickle.loads(base64.b64decode(compressed_data.encode('utf-8')))
                if self.quantizer is not None:
                    weights = self.quantizer.decompress(compressed_data)
                else:
                    weights = compressed_data
                print(f"Client {self.client_id} received quantized global model (dequantized to float32)")
            else:
                # Normal weights
                if 'weights' in data:
                    encoded_weights = data['weights']
                    if isinstance(encoded_weights, str):
                        import base64, pickle
                        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
                        weights = pickle.loads(serialized)
                    else:
                        weights = encoded_weights
                else:
                    weights = data.get('parameters', [])
            
            self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=data.get('model_config'),
                source="MQTT",
                server_sent_unix=data.get('server_sent_unix'),
            )
            
        except Exception as e:
            print(f"Client {self.client_id} ERROR in handle_global_model: {e}")
            import traceback
            traceback.print_exc()

    def _apply_global_model_weights(self, round_num, weights, model_config=None, source="UNKNOWN", server_sent_unix=None):
        """Initialize model if needed and apply incoming global weights."""
        recv_t0 = getattr(self, "_global_model_receive_t0", None)
        if recv_t0 is not None:
            self._global_model_receive_t0 = None

        if self.model is None and source != "gRPC":
            print(
                f"[{source}] Client {self.client_id} ignoring non-gRPC initial global model. "
                f"Bootstrap requires gRPC."
            )
            return False

        if self.model is None:
            if model_config:
                print(f"[{source}] Client {self.client_id} initializing model from received configuration...")
                self.model = self.build_model_from_config(model_config)
                print(f"[{source}] Client {self.client_id} model built successfully")
            else:
                print(f"[{source}] Client {self.client_id} waiting for model_config to initialize model")
                return False

        if round_num >= self.last_global_round:
            self.model.set_weights(weights)
            self.current_round = max(self.current_round, round_num)
            self.last_global_round = round_num
            self.waiting_for_aggregated_model = False
            print(f"[{source}] Client {self.client_id} updated model weights (round {round_num})")

            # Downlink comm time for RL: prefer server_sent_unix → client receive-complete (requires synced clocks).
            try:
                ss = float(server_sent_unix) if server_sent_unix is not None else None
            except (TypeError, ValueError):
                ss = None
            if ss is not None and ss <= 0.0:
                ss = None
            self._downlink_server_sent_unix = ss
            self._downlink_receive_complete_unix = time.time()

            # If gRPC protocol negotiation did not arm state/time (timeout, bootstrap, race), align
            # downlink Q-learning with the actual delivery path so updates match uplink cadence per round.
            t0_for_dl = recv_t0 if recv_t0 is not None else time.time()
            self._arm_downlink_rl_if_missing(source=source, receive_started_at=t0_for_dl)

            # Compute downlink RL reward now that the model has arrived
            self._update_downlink_rl_after_reception(round_num=round_num)

            if self.pending_start_training_round is not None:
                pending_round = self.pending_start_training_round
                self.pending_start_training_round = None
                if self.last_training_round != pending_round and self.is_active:
                    self.current_round = max(self.current_round, pending_round)
                    self.last_training_round = pending_round
                    print(f"[{source}] Client {self.client_id} processing deferred start_training for round {pending_round}")
                    self.train_local_model()
            if self.pending_start_evaluation_round == round_num and self.is_active:
                self.pending_start_evaluation_round = None
                if round_num not in self.evaluated_rounds:
                    self.evaluated_rounds.add(round_num)
                    print(f"[{source}] Client {self.client_id} processing deferred start_evaluation for round {round_num}")
                    self.evaluate_model(evaluation_round_num=round_num)
            return True

        return False

    def _sync_env_rl_comm_from_round_metrics(self) -> None:
        """Set env ``comm_level`` from last round uplink communication time (for next Q-state)."""
        if not self.env_manager:
            return
        # Phase 1: only collect raw times; discrete levels come after min/max over all rounds
        if getattr(self, "_rl_boundary_collection_phase", False):
            return
        try:
            rm = self.round_metrics or {}
            t = float(
                rm.get("communication_time")
                or (
                    float(rm.get("uplink_model_comm_time", 0) or 0)
                    + float(rm.get("uplink_metrics_comm_time", 0) or 0)
                )
            )
            if t > 0 and self.rl_selector_uplink is not None:
                self.env_manager.update_comm_level_from_time(
                    t,
                    self.rl_selector_uplink.comm_t_low,
                    self.rl_selector_uplink.comm_t_high,
                )
            elif t > 0:
                self.env_manager.update_comm_level_from_time(t)
        except Exception:
            pass

    def _rl_discrete_state_for_selector(
        self,
        selector,
        comm_time: float,
        cpu: float,
        memory: float,
    ):
        """Map measurements to Q-state, or a fixed placeholder during Phase 1 (no provisional thresholds)."""
        if getattr(self, "_rl_boundary_collection_phase", False) and self.env_manager:
            return self.env_manager.neutral_rl_state_before_boundaries()
        return self.env_manager.state_for_rl_selector(selector, comm_time, cpu, memory)

    def _gather_state_for_downlink_rl(self):
        """Build env state dict for downlink Q-learning (downlink-specific comm/resource/battery thresholds)."""
        if not self.env_manager or self.rl_selector_downlink is None:
            return None
        self._sync_env_rl_comm_from_round_metrics()
        with tf.device('/CPU:0'):
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory().percent
            self._apply_rl_network_scenario_for_selector_state()
            t_dl = float(getattr(self, "_last_downlink_comm_wall_s", 0.0) or 0.0)
            if t_dl <= 0.0:
                rm = self.round_metrics or {}
                t_dl = float(
                    (rm.get("communication_time") or 0)
                    or (
                        float(rm.get("uplink_model_comm_time", 0) or 0)
                        + float(rm.get("uplink_metrics_comm_time", 0) or 0)
                    )
                )
            return self._rl_discrete_state_for_selector(
                self.rl_selector_downlink, t_dl, cpu, memory
            )

    def _downlink_protocol_from_delivery_source(self, source: str) -> str:
        """Map global-model receive path to a Q-table action name."""
        s = (source or "").strip().upper().replace(" ", "")
        if s == "GRPC":
            return "grpc"
        if s == "MQTT":
            return "mqtt"
        if s == "AMQP":
            return "amqp"
        if s == "QUIC":
            return "http3"
        if s in ("HTTP/3", "HTTP3"):
            return "http3"
        if s == "DDS":
            return "dds"
        fallback = (self.selected_downlink_protocol or "grpc").strip().lower()
        if self.rl_selector_downlink is not None and fallback in self.rl_selector_downlink.PROTOCOLS:
            return fallback
        return "grpc"

    def _arm_downlink_rl_if_missing(self, source: str, receive_started_at: float) -> None:
        """
        When gRPC protocol negotiation did not run (or timed out before the client polled),
        still arm state/time + register (state, actual protocol) so downlink Q-updates run
        once per global model like uplink does per round.
        """
        if not USE_RL_SELECTION or self.rl_selector_downlink is None or self.env_manager is None:
            return
        if not USE_RL_EXPLORATION:
            return
        if self._downlink_select_time is not None and self._last_downlink_rl_state is not None:
            return
        try:
            state = self._gather_state_for_downlink_rl()
            if state is None:
                return
            proto = self._downlink_protocol_from_delivery_source(source)
            if proto not in self.rl_selector_downlink.PROTOCOLS:
                proto = "grpc"
            self.rl_selector_downlink.register_selection(state, proto, record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING)
            self._last_downlink_rl_state = state
            self._downlink_select_time = receive_started_at
            self.selected_downlink_protocol = proto
            print(
                f"[Downlink RL] Client {self.client_id} armed Q-update from delivery "
                f"(protocol={proto}, source={source})"
            )
        except Exception as e:
            print(f"[Downlink RL] Client {self.client_id} fallback arm failed: {e}")

    def _rl_q_logging_allowed(self) -> bool:
        """False during Phase 1 boundary data collection — no q_learning_log writes or DB init."""
        return not getattr(self, "_rl_boundary_collection_phase", False)

    def _update_downlink_rl_after_reception(self, round_num: int = 0):
        """
        Compute and log the downlink Q-learning reward after the global model is received.
        Called from _apply_global_model_weights() on successful model reception.
        Uses the dedicated DOWNLINK RL agent (separate Q-table from uplink).

        Downlink communication time uses server ``server_sent_unix`` (wall clock at send start on server)
        through client receive-complete time set in ``_apply_global_model_weights`` (requires time sync).
        Falls back to (receive_complete - protocol_selection_time) if the server did not send a timestamp.
        """
        if (not USE_RL_SELECTION
                or self.rl_selector_downlink is None
                or self._last_downlink_rl_state is None):
            return
        recv_end = getattr(self, '_downlink_receive_complete_unix', None)
        srv_sent = getattr(self, '_downlink_server_sent_unix', None)

        if recv_end is None:
            recv_end = time.time()
        if srv_sent is not None:
            comm_time = max(0.0, float(recv_end) - float(srv_sent))
        elif self._downlink_select_time is not None:
            comm_time = max(0.0, float(recv_end) - float(self._downlink_select_time))
        else:
            self._downlink_receive_complete_unix = None
            self._downlink_server_sent_unix = None
            return

        self._downlink_receive_complete_unix = None
        self._downlink_server_sent_unix = None
        self._downlink_select_time = None

        self._last_downlink_comm_wall_s = float(comm_time)

        if not USE_RL_EXPLORATION:
            self._last_downlink_rl_state = None
            return

        if getattr(self, "_rl_boundary_collection_phase", False):
            import psutil
            cpu_m = psutil.cpu_percent(interval=0.0)
            mem_m = psutil.virtual_memory().percent
            resource_load_dl = (cpu_m + mem_m) / 2.0
            batt_dl = float(self.env_manager.battery_soc) if self.env_manager else 1.0
            self._rl_boundary_downlink_comm_samples.append(comm_time)
            self._rl_boundary_res_samples_dl.append(resource_load_dl)
            self._rl_boundary_batt_samples_dl.append(batt_dl)
            nd = len(self._rl_boundary_downlink_comm_samples)
            print(
                f"[Client {self.client_id}] RL Phase 1 (downlink data collection): sample {nd} "
                f"(wall comm={comm_time:.3f}s, load={resource_load_dl:.1f}%, SoC={batt_dl:.3f})"
            )
            self._last_downlink_rl_state = None
            return

        try:
            downlink_state = self._last_downlink_rl_state
            self._last_downlink_rl_state = None  # reset

            protocol = self.selected_downlink_protocol or 'grpc'
            resources = (
                self.env_manager.get_resource_consumption(
                    protocol=protocol,
                    uplink_payload_bytes=self.round_metrics.get('payload_bytes'),
                    downlink_payload_bytes=getattr(self, '_downlink_payload_bytes', None),
                )
                if self.env_manager
                else {}
            )
            # Estimate payload of received global model (model weights size)
            payload_bytes = getattr(self, '_downlink_payload_bytes', None) or (12 * 1024 * 1024)
            t_calc = None
            if USE_COMMUNICATION_MODEL_REWARD:
                try:
                    reward_scenario = os.environ.get("RL_REWARD_SCENARIO", "").strip().lower()
                    if reward_scenario:
                        t_calc = self._get_t_calc_for_scenario(protocol, payload_bytes, reward_scenario)
                    if t_calc is None:
                        t_calc = self._get_t_calc_for_reward(protocol, payload_bytes)
                except Exception:
                    t_calc = None

            reward = self.rl_selector_downlink.calculate_reward(
                communication_time=comm_time,
                success=True,
                resource_consumption=resources,
                t_calc=t_calc,
            )
            if self.env_manager:
                self.env_manager.update_comm_level_from_time(
                    comm_time,
                    self.rl_selector_downlink.comm_t_low,
                    self.rl_selector_downlink.comm_t_high,
                )
                self.env_manager.sync_battery_level_from_soc(None)
                next_state = self.env_manager.get_current_state()
            else:
                next_state = None
            self.rl_selector_downlink.update_q_value(
                reward,
                next_state=next_state,
                done=False if next_state is not None else True,
            )
            q_delta = self.rl_selector_downlink.get_last_q_delta()
            q_value = self.rl_selector_downlink.get_last_q_value()
            avg_reward = (
                float(np.mean(self.rl_selector_downlink.total_rewards[-100:]))
                if self.rl_selector_downlink.total_rewards else 0.0
            )
            self.rl_selector_downlink.end_episode()
            q_converged = self.rl_selector_downlink.check_q_converged(
                threshold=Q_CONVERGENCE_THRESHOLD,
                patience=Q_CONVERGENCE_PATIENCE,
                state=downlink_state,
            )

            print(f"[Downlink RL] Client {self.client_id} | round={round_num} | protocol={protocol.upper()} | "
                  f"comm_time={comm_time:.3f}s | reward={reward:.2f} | q_delta={q_delta:.4f} | "
                  f"epsilon={self.rl_selector_downlink.epsilon:.4f} | q_conv_dl={q_converged}")

            if log_q_step is not None and self._rl_q_logging_allowed():
                reward_details = self.rl_selector_downlink.get_last_reward_breakdown()
                log_q_step(
                    client_id=self.client_id,
                    round_num=round_num or self.current_round,
                    episode=self.rl_selector_downlink.episode_count - 1,
                    state_comm_level=downlink_state.get('comm_level', ''),
                    state_resource=downlink_state.get('resource', ''),
                    state_battery_level=downlink_state.get('battery_level', ''),
                    **rl_state_network_kwargs(downlink_state),
                    action=protocol,
                    reward=reward,
                    q_delta=q_delta,
                    epsilon=self.rl_selector_downlink.epsilon,
                    q_value=q_value,
                    avg_reward_last_100=avg_reward,
                    converged=q_converged,
                    metric_communication_time=reward_details.get('communication_time'),
                    metric_success=reward_details.get('success'),
                    metric_cpu_usage=reward_details.get('cpu_usage'),
                    metric_memory_usage=reward_details.get('memory_usage'),
                    metric_bandwidth_usage=reward_details.get('bandwidth_usage'),
                    metric_battery_level=reward_details.get('battery_level'),
                    metric_energy_usage=reward_details.get('energy_usage'),
                    reward_base=reward_details.get('reward_base'),
                    reward_communication_time=reward_details.get('reward_communication_time'),
                    reward_resource_penalty=reward_details.get('reward_resource_penalty'),
                    reward_battery_penalty=reward_details.get('reward_battery_penalty'),
                    reward_total=reward_details.get('reward_total', reward),
                    link_direction='downlink',
                )
        except Exception as e:
            print(f"[Downlink RL] Client {self.client_id} reward update error: {e}")
            import traceback
            traceback.print_exc()

    def _apply_generator_batch_size(self, batch_size):
        """Sync DirectoryIterator batch size with training config."""
        try:
            batch_size = int(batch_size)
        except (TypeError, ValueError):
            return
        if batch_size <= 0:
            return

        changed = False
        for gen_name in ("train_generator", "validation_generator"):
            generator = getattr(self, gen_name, None)
            if generator is None:
                continue
            current = getattr(generator, "batch_size", None)
            if current != batch_size:
                setattr(generator, "batch_size", batch_size)
                changed = True
        if changed:
            print(f"Client {self.client_id} synchronized generator batch_size to {batch_size}")
    
    def handle_training_config(self, payload):
        """Update training configuration"""
        new_config = json.loads(payload.decode())
        if isinstance(new_config, dict):
            self.training_config.update(new_config)
        try:
            self.training_config["batch_size"] = int(self.training_config.get("batch_size", DEFAULT_DATA_BATCH_SIZE))
        except (TypeError, ValueError):
            self.training_config["batch_size"] = DEFAULT_DATA_BATCH_SIZE
        self._apply_generator_batch_size(self.training_config["batch_size"])
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    def handle_start_training(self, payload, source="MQTT"):
        """Start local training when server signals"""
        if not self.is_active:
            return
        data = json.loads(payload.decode())
        round_num = data['round']
        
        # Check if model is initialized - WAIT for global model
        if self.model is None:
            self._defer_start_training(round_num, source)
            self.pending_start_training_round = round_num
            return
        
        # Check for duplicate training signals
        if self.last_training_round == round_num:
            print(f"Client {self.client_id} ignoring duplicate start training for round {round_num}")
            return
        
        # Check if we're ready for this round
        if self.current_round == 0 and round_num == 1:
            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            self.train_local_model()
        elif round_num >= self.current_round and round_num <= self.current_round + 1:
            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            self.train_local_model()
        else:
            print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
    
    def handle_start_evaluation(self, payload):
        """Start evaluation when server signals"""
        if not self.is_active:
            return
        data = json.loads(payload.decode())
        round_num = data['round']
        # Server current_round can run ahead of local state when gRPC polls CheckTrainingStatus
        # before MQTT (or another thread) has applied handle_start_training for the same round.
        if round_num < self.current_round:
            print(
                f"Client {self.client_id} ignoring stale evaluation signal for round {round_num} "
                f"(current: {self.current_round})"
            )
            return
        if round_num > self.current_round:
            self.pending_start_evaluation_round = round_num
            print(
                f"Client {self.client_id} deferring evaluation for round {round_num} "
                f"until local round catches up (current: {self.current_round})"
            )
            return
        if round_num == self.current_round:
            if self.waiting_for_aggregated_model or self.last_global_round < round_num:
                self.pending_start_evaluation_round = round_num
                print(
                    f"Client {self.client_id} deferring evaluation for round {round_num} "
                    f"until aggregated model arrives (last_global_round={self.last_global_round})"
                )
                return
            if round_num in self.evaluated_rounds:
                print(f"Client {self.client_id} ignoring duplicate evaluation for round {round_num}")
                return
            self.evaluated_rounds.add(round_num)  # Add BEFORE evaluate to prevent race condition
            print(f"Client {self.client_id} starting evaluation for round {round_num}...")
            self.evaluate_model(evaluation_round_num=round_num)
            print(f"Client {self.client_id} evaluation completed for round {round_num}.")
    
    def handle_training_complete(self):
        """Handle training completion signal from server"""
        self.is_active = False
        self.shutdown_requested = True
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nDisconnecting from server...")
        time.sleep(1)
        self.mqtt_client.disconnect()
        print(f"Client {self.client_id} disconnected successfully.")

    def register_with_server_grpc(self) -> bool:
        """Register this client using gRPC to drive round-0 model delivery."""
        if self.grpc_registered:
            return True
        if grpc is None or federated_learning_pb2 is None or federated_learning_pb2_grpc is None:
            print(f"[gRPC] Client {self.client_id} registration skipped: gRPC not available")
            return False

        try:
            grpc_host = os.getenv("GRPC_HOST", "fl-server-unified-emotion")
            grpc_port = int(os.getenv("GRPC_PORT", "50051"))
            options = [
                ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
                ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
                ('grpc.keepalive_time_ms', 600000),
                ('grpc.keepalive_timeout_ms', 60000),
            ]
            channel = grpc.insecure_channel(f'{grpc_host}:{grpc_port}', options=options)
            stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
            response = stub.RegisterClient(
                federated_learning_pb2.ClientRegistration(client_id=self.client_id)
            )
            if response.success:
                self.grpc_registered = True
                print(f"[gRPC] Client {self.client_id} registered successfully")
                log_sent_packet(
                    packet_size=len(str(self.client_id)),
                    peer="server",
                    protocol="gRPC",
                    round=0,
                    extra_info="registration"
                )
                channel.close()
                return True

            print(f"[gRPC] Registration failed: {response.message}")
            channel.close()
            return False
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} registration error: {e}")
            return False

    def _is_protocol_available_for_downlink(self, protocol: str) -> bool:
        """Check whether a requested downlink protocol can be used by this client."""
        protocol = (protocol or '').lower()
        if protocol == 'mqtt':
            return True
        if protocol == 'amqp':
            return pika is not None
        if protocol == 'grpc':
            return grpc is not None and self.grpc_stub is not None
        if protocol == 'quic':
            return asyncio is not None and connect is not None
        if protocol == 'http3':
            return HTTP3_AVAILABLE and asyncio is not None and connect is not None
        if protocol == 'dds':
            return DDS_AVAILABLE and self.dds_participant is not None
        return False

    def _select_downlink_protocol(self, round_id: int, global_model_id: int) -> str:
        """Select downlink protocol using the DEDICATED downlink RL agent."""
        # Enforce gRPC bootstrap for initial model transfer.
        if self.model is None or global_model_id <= 0:
            return 'grpc'

        state = None
        downlink_rl_armed = False
        # Use the dedicated DOWNLINK Q-learning agent (separate Q-table from uplink)
        if USE_RL_SELECTION and self.rl_selector_downlink and self.env_manager:
            try:
                self._sync_env_rl_comm_from_round_metrics()
                with tf.device('/CPU:0'):
                    import psutil
                    cpu = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory().percent
                    self._apply_rl_network_scenario_for_selector_state()
                    t_dl = float(getattr(self, "_last_downlink_comm_wall_s", 0.0) or 0.0)
                    if t_dl <= 0.0:
                        rm = self.round_metrics or {}
                        t_dl = float(
                            (rm.get("communication_time") or 0)
                            or (
                                float(rm.get("uplink_model_comm_time", 0) or 0)
                                + float(rm.get("uplink_metrics_comm_time", 0) or 0)
                            )
                        )
                    state = self._rl_discrete_state_for_selector(
                        self.rl_selector_downlink, t_dl, cpu, memory
                    )
                    training_mode = USE_RL_EXPLORATION
                    if getattr(self, "_rl_boundary_collection_phase", False):
                        self.rl_selector_downlink.epsilon = 1.0
                    selected = self.rl_selector_downlink.select_protocol(
                        state, training=training_mode, record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING
                    )
                # Store downlink state and selection time so reward can be computed after model arrives
                self._last_downlink_rl_state = state
                self._downlink_select_time = time.time()
                downlink_rl_armed = True
                if getattr(self, "_rl_boundary_collection_phase", False):
                    print(
                        f"[Downlink RL] Client {self.client_id} selected {selected.upper()} "
                        f"(Phase 1: placeholder state for exploration; epsilon={self.rl_selector_downlink.epsilon:.4f})"
                    )
                else:
                    print(f"[Downlink RL] Client {self.client_id} selected {selected.upper()} "
                          f"(downlink agent, epsilon={self.rl_selector_downlink.epsilon:.4f})")
            except Exception as e:
                print(f"[Downlink RL] Error: {e}, falling back to uplink agent")
                selected = self.select_protocol()
                self._last_downlink_rl_state = None
                self._downlink_select_time = None
        else:
            selected = self.select_protocol()
            self._last_downlink_rl_state = None
            self._downlink_select_time = None

        if not self._is_protocol_available_for_downlink(selected):
            print(
                f"[gRPC] Client {self.client_id} downlink selection '{selected}' unavailable; "
                f"falling back to gRPC"
            )
            if (
                downlink_rl_armed
                and state is not None
                and self.rl_selector_downlink is not None
                and self.rl_selector_downlink.pop_last_selection()
            ):
                self.rl_selector_downlink.register_selection(state, 'grpc', record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING)
                self._last_downlink_rl_state = state
                self._downlink_select_time = time.time()
            else:
                self._last_downlink_rl_state = None
                self._downlink_select_time = None
            return 'grpc'
        return selected

    def _handle_grpc_protocol_query(self, protocol_query):
        """Respond to server ProtocolQuery via gRPC using RL-based downlink selection."""
        if protocol_query is None:
            return

        query_key = (
            int(getattr(protocol_query, 'round_id', -1)),
            int(getattr(protocol_query, 'global_model_id', -1)),
        )
        if self.last_protocol_query_key == query_key:
            return

        round_id, global_model_id = query_key
        selected_downlink_protocol = self._select_downlink_protocol(round_id, global_model_id)

        with self.grpc_lock:
            if self.grpc_stub is None:
                return
            response = self.grpc_stub.SendProtocolSelection(
                federated_learning_pb2.ProtocolSelection(
                    client_id=self.client_id,
                    round_id=round_id,
                    global_model_id=global_model_id,
                    downlink_protocol_requested=selected_downlink_protocol,
                )
            )

        if response.success:
            self.selected_downlink_protocol = selected_downlink_protocol
            self.last_protocol_query_key = query_key
            # Note: we do NOT log reward=0.0 here anymore.
            # The real downlink reward is computed and logged in _update_downlink_rl_after_reception()
            # once the global model actually arrives via the selected protocol.
            _dl_eps = (
                f"{self.rl_selector_downlink.epsilon:.4f}"
                if self.rl_selector_downlink is not None
                else "N/A"
            )
            print(
                f"[gRPC] Client {self.client_id} replied ProtocolSelection: "
                f"round={round_id}, global_model_id={global_model_id}, "
                f"downlink={selected_downlink_protocol} "
                f"(downlink RL agent epsilon={_dl_eps})"
            )
        else:
            print(f"[gRPC] Client {self.client_id} ProtocolSelection rejected: {response.message}")

    def _defer_start_training(self, round_num: int, source: str):
        """Defer start_training until the initial/global model is available."""
        if self.pending_start_training_round != round_num:
            print(f"[{source}] Client {self.client_id} deferring start_training round {round_num} until global model arrives")
    
    def measure_network_condition(self):
        """
        Measure current network condition (latency / bandwidth estimate)
        and update the RL environment state manager.
        """
        if not self.env_manager:
            return

        try:
            target_host = MQTT_BROKER

            # Estimate latency by timing TCP connection to MQTT broker.
            # This avoids requiring external tools like `ping` inside the container.
            latencies: List[float] = []
            port = int(os.getenv("MQTT_PORT", "1883"))
            for _ in range(3):
                start = time.time()
                try:
                    with socket.create_connection(
                        (target_host, port), timeout=2
                    ):
                        pass
                    elapsed_ms = (time.time() - start) * 1000.0
                    latencies.append(elapsed_ms)
                except OSError:
                    # Treat failures as high latency samples
                    latencies.append(500.0)

            if latencies:
                latency_ms = sum(latencies) / len(latencies)
            else:
                # Fallback conservative default
                latency_ms = 300.0

            # Rough bandwidth estimate based on latency bucket
            if latency_ms < 20:
                bandwidth_mbps = 100.0
            elif latency_ms < 50:
                bandwidth_mbps = 20.0
            elif latency_ms < 150:
                bandwidth_mbps = 5.0
            elif latency_ms < 400:
                bandwidth_mbps = 1.0
            else:
                bandwidth_mbps = 0.5

            condition = self.env_manager.detect_network_condition(
                latency_ms, bandwidth_mbps
            )
            self.env_manager.update_network_condition(condition)
            if USE_RL_SELECTION and self.env_manager and getattr(self, "rl_selector_uplink", None):
                self.env_manager.update_detected_network_scenario(
                    condition,
                    rl_uplink=self.rl_selector_uplink,
                    rl_downlink=self.rl_selector_downlink,
                )
                # Keep configured env label for metadata when set; else use measurement for both.
                if not _effective_rl_network_scenario_label_from_env():
                    self.env_manager.update_network_scenario(
                        condition,
                        rl_uplink=self.rl_selector_uplink,
                        rl_downlink=self.rl_selector_downlink,
                    )

            # Update mobility based on variability of recent latency samples.
            # Higher jitter -> higher inferred mobility level.
            try:
                self.latency_history.append(latency_ms)
                # Keep a rolling window to bound memory and smooth behaviour
                if len(self.latency_history) > 20:
                    self.latency_history.pop(0)

                mobility = "static"
                if len(self.latency_history) >= 5:
                    avg = sum(self.latency_history) / len(self.latency_history)
                    variance = sum(
                        (x - avg) ** 2 for x in self.latency_history
                    ) / len(self.latency_history)
                    stddev = variance ** 0.5

                    # Map jitter to Q-learning mobility: only {static, mobile} exist in
                    # QLearningProtocolSelector.MOBILITY_LEVELS; previous low/medium/high
                    # labels were ignored by EnvironmentStateManager.update_mobility.
                    if stddev < 5:
                        mobility = "static"
                    else:
                        mobility = "mobile"

                    self.env_manager.update_mobility(mobility)
            except Exception as e:
                print(f"[Mobility] Failed to update mobility level: {e}")

            print(
                f"[Network] latency={latency_ms:.1f} ms, "
                f"est_bandwidth={bandwidth_mbps:.1f} Mbps -> "
                f"condition={condition}"
            )
        except Exception as e:
            print(f"[Network] Failed to measure network condition: {e}")

    def _apply_rl_network_scenario_for_selector_state(self) -> None:
        """
        Set configured ``data_network_scenario`` from env when provided, then refresh
        client-detected scenario for Q-table indexing (``detected_network_scenario``;
        see ``RL_Q_USE_DETECTED_NETWORK``).
        """
        if not self.env_manager:
            return
        eff = _effective_rl_network_scenario_label_from_env()
        if eff:
            self.env_manager.update_network_scenario(
                eff,
                rl_uplink=self.rl_selector_uplink,
                rl_downlink=self.rl_selector_downlink,
            )
            self.env_manager.update_network_condition(
                _coarse_network_bucket_for_scenario(eff)
            )
        elif USE_QL_CONVERGENCE:
            self.env_manager.update_network_scenario(
                None,
                rl_uplink=self.rl_selector_uplink,
                rl_downlink=self.rl_selector_downlink,
            )
        # Always measure so Q-table can use client-detected network slice alongside configured env.
        self.measure_network_condition()

    def _shared_data_dir(self):
        """Resolve shared_data path (Docker: /shared_data; native: project shared_data)."""
        if os.path.exists("/shared_data"):
            return "/shared_data"
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "shared_data")

    def _get_t_calc_for_reward(self, protocol: str, payload_bytes: int) -> Optional[float]:
        """Get T_calc from communication model (iperf3 JSON) for reward; None if unavailable."""
        global _T_CALC_IMPORT_WARNED, _T_CALC_IPERF_MISSING_WARNED
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Network_Simulation"))
            from communication_model import get_t_calc_for_reward
            from pathlib import Path
            shared = self._shared_data_dir()
            path = Path(shared) / "iperf3_network_params.json"
            out = get_t_calc_for_reward(protocol, payload_bytes, json_path=path)
            if (
                out is None
                and USE_COMMUNICATION_MODEL_REWARD
                and not _T_CALC_IPERF_MISSING_WARNED
            ):
                _T_CALC_IPERF_MISSING_WARNED = True
                if not path.is_file():
                    print(
                        f"[RL] T_calc / R_tcalc will stay 0: missing {path}. "
                        "Populate via iperf3 (diagnostic pipeline) or set RL_REWARD_SCENARIO for analytic T_calc."
                    )
                else:
                    print(
                        f"[RL] T_calc / R_tcalc: {path} present but model returned None "
                        "(empty JSON or missing bandwidth_bps / unusable params)."
                    )
            return out
        except Exception as e:
            if not _T_CALC_IMPORT_WARNED:
                _T_CALC_IMPORT_WARNED = True
                print(
                    f"[RL] T_calc / R_tcalc will stay 0: could not load communication_model ({e!r}). "
                    "Ensure /app/Network_Simulation exists in the container (Dockerfile COPY or compose volume)."
                )
            return None

    def _get_t_calc_for_scenario(self, protocol: str, payload_bytes: int, scenario_name: str) -> Optional[float]:
        """Get T_calc for a named network scenario (e.g. good, moderate, poor) for simulated reward.
        Used when RL_REWARD_SCENARIO is set: train in excellent conditions but reward as if in that scenario."""
        global _T_CALC_IMPORT_WARNED
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Network_Simulation"))
            from communication_model import get_t_calc_for_scenario
            return get_t_calc_for_scenario(protocol, payload_bytes, scenario_name)
        except Exception as e:
            if not _T_CALC_IMPORT_WARNED:
                _T_CALC_IMPORT_WARNED = True
                print(
                    f"[RL] T_calc (scenario) unavailable ({e!r}). "
                    "Ensure Network_Simulation is on PYTHONPATH or mounted at /app/Network_Simulation."
                )
            return None

    def _protocol_payload_limit_bytes(self, protocol: str) -> Optional[int]:
        """Return the per-protocol payload cap for a single non-chunked send."""
        return {
            'mqtt': MQTT_MAX_PAYLOAD_BYTES,
            'amqp': AMQP_MAX_FRAME_BYTES,
            'grpc': GRPC_MAX_MESSAGE_BYTES,
            'http3': HTTP3_MAX_STREAM_DATA_BYTES,
            'dds': CHUNK_SIZE,
        }.get(protocol)

    def _estimate_update_payload_bytes(self, protocol: str, message: dict) -> int:
        """Estimate transport payload size for protocol gating before send."""
        if protocol == 'http3':
            return len(json.dumps({**message, 'type': 'update'}).encode('utf-8'))
        if protocol in ('mqtt', 'amqp', 'quic'):
            return len(json.dumps(message).encode('utf-8'))
        if protocol == 'grpc':
            if 'compressed_data' in message:
                return len(base64.b64decode(message['compressed_data'].encode('utf-8')))
            weights = message.get('weights', b'')
            return len(weights.encode('utf-8')) if isinstance(weights, str) else len(weights)
        if protocol == 'dds':
            if 'compressed_data' in message:
                return len(base64.b64decode(message['compressed_data'].encode('utf-8')))
            return len(pickle.dumps(message.get('weights')))
        return len(json.dumps(message).encode('utf-8'))

    def _protocol_can_send_update(self, protocol: str, message: dict) -> bool:
        """Unified mode supports chunked updates on every transport."""
        return True

    def _build_update_protocol_order(self, preferred_protocol: str, message: dict) -> List[str]:
        """Send updates only on the RL-selected protocol."""
        return [preferred_protocol]

    def _assert_protocol_payload_limit(self, protocol: str, payload_size_bytes: int):
        """Reject oversize non-chunked sends before they hit the transport."""
        if protocol in ('grpc', 'dds', 'quic'):
            return
        limit = self._protocol_payload_limit_bytes(protocol)
        if limit is not None and payload_size_bytes > limit:
            raise ValueError(
                f"{protocol.upper()} payload {payload_size_bytes} B exceeds configured limit {limit} B"
            )

    def _ensure_protocol_connection_sync(self, protocol: str):
        """Ensure persistent transports are connected before sending."""
        if protocol == 'quic':
            if asyncio is None:
                raise ConnectionError("QUIC asyncio support not available")
            if self.quic_loop is None or self.quic_loop.is_closed():
                self.start_quic_listener()
            if self.quic_loop is None:
                raise ConnectionError("QUIC event loop not available")
            future = asyncio.run_coroutine_threadsafe(self._ensure_quic_connection(), self.quic_loop)
            future.result(timeout=15)
        elif protocol == 'http3':
            if asyncio is None:
                raise ConnectionError("HTTP/3 asyncio support not available")
            if self.http3_loop is None or self.http3_loop.is_closed():
                self.start_http3_listener()
            if self.http3_loop is None:
                raise ConnectionError("HTTP/3 event loop not available")
            future = asyncio.run_coroutine_threadsafe(self._ensure_http3_connection(), self.http3_loop)
            future.result(timeout=15)

    def _update_payload_field(self, message: dict) -> str:
        """Return the serialized payload field used for model updates."""
        return 'compressed_data' if 'compressed_data' in message else 'weights'

    def _chunked_update_messages(self, message: dict, chunk_bytes: int, message_type: str) -> List[dict]:
        """Split a serialized model update into transport-safe chunks."""
        payload_key = self._update_payload_field(message)
        payload_text = message[payload_key]
        if not isinstance(payload_text, str):
            raise TypeError(f"Chunked update payload for {payload_key} must be a base64 string")
        chunks = [payload_text[i:i + chunk_bytes] for i in range(0, len(payload_text), chunk_bytes)] or [payload_text]
        total_chunks = len(chunks)
        return [
            {
                "type": message_type,
                "client_id": message["client_id"],
                "round": message["round"],
                "protocol": message.get("protocol"),
                "payload_key": payload_key,
                "payload_chunk": chunk_data,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "num_samples": message.get("num_samples", 0),
                "metrics": message.get("metrics", {}),
            }
            for chunk_index, chunk_data in enumerate(chunks)
        ]

    def _run_iperf3_if_diagnostic(self):
        """When FL_DIAGNOSTIC_PIPELINE=1 and current_round==1, run iperf3 from client and write to shared_data."""
        if not os.environ.get("FL_DIAGNOSTIC_PIPELINE", "").strip() in ("1", "true", "yes"):
            return
        if getattr(self, "current_round", 0) != 1:
            return
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "Network_Simulation"))
            from communication_model import run_iperf3_native
            from pathlib import Path
            server_host = os.getenv("MQTT_BROKER") or os.getenv("GRPC_HOST") or "127.0.0.1"
            shared = self._shared_data_dir()
            os.makedirs(shared, exist_ok=True)
            out_path = Path(shared) / "iperf3_network_params.json"
            run_iperf3_native(server_host, duration_sec=5, use_udp=True, output_json_path=out_path)
            print(f"[Client {self.client_id}] Diagnostic: iperf3 run complete, params written to {out_path}")
        except Exception as e:
            print(f"[Client {self.client_id}] Diagnostic iperf3 failed: {e}")

    def select_protocol(self) -> str:
        """
        Select UPLINK protocol using the dedicated UPLINK RL agent (CPU-only Q-learning).
        Returns: Selected protocol name: 'mqtt', 'amqp', 'grpc', 'quic', or 'dds'
        """
        if USE_RL_SELECTION and self.rl_selector_uplink and self.env_manager:
            try:
                self._sync_env_rl_comm_from_round_metrics()
                # RL logic runs on CPU only (no GPU); keeps Q-table updates off GPU
                with tf.device('/CPU:0'):
                    import psutil
                    cpu = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory().percent

                    # Explicit RL_REWARD_SCENARIO / NETWORK_SCENARIO (see _effective_rl_network_scenario_label_from_env)
                    # applies to **inference** as well as training so Q-state matches learned slices.
                    self._apply_rl_network_scenario_for_selector_state()

                    rm = self.round_metrics or {}
                    t_comm = float(
                        (rm.get("communication_time") or 0)
                        or (
                            float(rm.get("uplink_model_comm_time", 0) or 0)
                            + float(rm.get("uplink_metrics_comm_time", 0) or 0)
                        )
                    )
                    state = self._rl_discrete_state_for_selector(
                        self.rl_selector_uplink, t_comm, cpu, memory
                    )
                    self._last_uplink_rl_state = state
                    # During RL-table training we keep epsilon-greedy exploration enabled.
                    # Once training is done, clients load and use the converged Q-table greedily.
                    training_mode = USE_RL_EXPLORATION
                    if getattr(self, "_rl_boundary_collection_phase", False):
                        self.rl_selector_uplink.epsilon = 1.0
                    protocol = self.rl_selector_uplink.select_protocol(
                        state, training=training_mode, record_learning=_RL_PROTOCOL_SELECTION_RECORD_LEARNING
                    )
                
                print(f"\n[Uplink RL Protocol Selection]")
                print(f"  CPU: {cpu:.1f}%, Memory: {memory:.1f}%")
                if getattr(self, "_rl_boundary_collection_phase", False):
                    soc = float(self.env_manager.battery_soc) if self.env_manager else float("nan")
                    print(
                        f"  Phase 1 (boundary collection): raw uplink wall comm ≈ {t_comm:.3f}s, "
                        f"SoC={soc:.3f} — discrete levels are assigned only after all "
                        f"{RL_PHASE0_ROUNDS} rounds (min/max → thresholds)."
                    )
                    print(f"  Placeholder state for random exploration only: {state}")
                else:
                    print(f"  State: {state}")
                print(f"  Selected Protocol: {protocol.upper()}")
                print(f"  Round: {self.current_round}")
                print(f"  Uplink epsilon: {self.rl_selector_uplink.epsilon:.4f}\n")
                
                self.selected_protocol = protocol
                return protocol
            except Exception as e:
                print(f"[Uplink RL Selection] Error: {e}, using MQTT as fallback")
                return 'mqtt'
        else:
            # Default to MQTT if RL not enabled
            return 'mqtt'
    
    def build_model_from_config(self, model_config):
        """Build model from server's architecture definition"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential()
        model.add(Input(shape=tuple(model_config['input_shape'])))
        
        for layer_config in model_config['layers']:
            if layer_config['type'] == 'Conv2D':
                model.add(Conv2D(
                    filters=layer_config['filters'],
                    kernel_size=tuple(layer_config['kernel_size']),
                    activation=layer_config.get('activation')
                ))
            elif layer_config['type'] == 'MaxPooling2D':
                model.add(MaxPooling2D(pool_size=tuple(layer_config['pool_size'])))
            elif layer_config['type'] == 'Dropout':
                model.add(Dropout(layer_config['rate']))
            elif layer_config['type'] == 'Flatten':
                model.add(Flatten())
            elif layer_config['type'] == 'Dense':
                model.add(Dense(
                    units=layer_config['units'],
                    activation=layer_config.get('activation')
                ))
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )

        # Dynamically categorize model size for RL state based on parameter count
        try:
            if self.env_manager is not None:
                total_params = model.count_params()
                if total_params < 1e5:
                    size_category = "small"
                elif total_params < 1e7:
                    size_category = "medium"
                else:
                    size_category = "large"
                self.env_manager.update_model_size(size_category)
                print(
                    f"[RL] Model params={total_params} -> "
                    f"size_category={size_category}"
                )
        except Exception as e:
            print(f"[RL] Failed to update model size category: {e}")

        return model
    
    def serialize_weights(self, weights):
        """Serialize model weights for transmission"""
        serialized = pickle.dumps(weights)
        encoded = base64.b64encode(serialized).decode('utf-8')
        return encoded
    
    def split_into_chunks(self, data):
        """Split serialized data into chunks of CHUNK_SIZE (for DDS)"""
        chunks = []
        for i in range(0, len(data), CHUNK_SIZE):
            chunks.append(data[i:i + CHUNK_SIZE])
        return chunks

    def _dds_serialized_weights_bytes(self, message: dict) -> bytes:
        """Bytes to chunk over DDS: same pickle payload JSON transports wrap as base64 strings."""
        if 'compressed_data' in message:
            return base64.b64decode(message['compressed_data'].encode('utf-8'))
        w = message['weights']
        if isinstance(w, str):
            return base64.b64decode(w.encode('utf-8'))
        return pickle.dumps(w)
    
    def send_model_update_chunked(self, round_num, serialized_weights, num_samples, loss, accuracy):
        """Send model update as chunks via DDS"""
        chunks = self.split_into_chunks(serialized_weights)
        total_chunks = len(chunks)
        
        print(f"Client {self.client_id}: Sending model update in {total_chunks} chunks ({len(serialized_weights)} bytes total)")
        
        for chunk_id, chunk_data in enumerate(chunks):
            chunk = ModelUpdateChunk(
                client_id=self.client_id,
                round=round_num,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                payload=chunk_data,
                num_samples=num_samples,
                loss=loss,
                accuracy=accuracy
            )
            self.dds_update_chunk_writer.write(chunk)
            # Reliable QoS handles delivery, no need for artificial delay
            if (chunk_id + 1) % 20 == 0:  # Progress update every 20 chunks
                print(f"  Sent {chunk_id + 1}/{total_chunks} chunks")
        
        return total_chunks
    
    def deserialize_weights(self, encoded_weights):
        """Deserialize model weights received from server"""
        serialized = base64.b64decode(encoded_weights.encode('utf-8'))
        weights = pickle.loads(serialized)
        return weights
    
    def train_local_model(self):
        """Train model on local data and send updates to server via RL-selected protocol"""
        self._last_update_protocol = None
        # In diagnostic pipeline (native): run iperf3 from client on round 1 to measure network params
        self._run_iperf3_if_diagnostic()
        start_time = time.time()

        batch_size = self.training_config['batch_size']
        self._apply_generator_batch_size(batch_size)
        epochs = self.training_config['local_epochs']
        
        try:
            steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
            val_steps = int(os.getenv("VAL_STEPS", "25"))
        except Exception:
            steps_per_epoch = 100
            val_steps = 25
        
        # Train the model on GPU (RL runs on CPU elsewhere)
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            history = self.model.fit(
                self.train_generator,
                epochs=epochs,
                validation_data=self.validation_generator,
                steps_per_epoch=steps_per_epoch,
                validation_steps=val_steps,
                verbose=2
            )
        
        # Get updated weights
        updated_weights = self.model.get_weights()
        num_samples = self.train_generator.n

        # Flow: Training -> Pruning -> Quantization -> Send
        if self.pruner is not None:
            updated_weights = self.pruner.prune_weights(updated_weights, step=self.current_round)
            if self.current_round == 0 or (self.current_round % 5 == 0):
                stats = self.pruner.get_pruning_statistics(updated_weights)
                print(f"[Client {self.client_id}] Pruning applied (round {self.current_round}): "
                      f"sparsity={stats['overall_sparsity']:.2%}, non_zero={stats['non_zero_params']}")

        # Prepare training metrics
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1])
        }
        
        self.round_metrics['training_time'] = time.time() - start_time
        self.round_metrics['accuracy'] = metrics['val_accuracy']
        
        # Select protocol based on RL
        protocol = self.select_protocol()
        
        # Build update message: quantized (compressed_data) or raw (weights)
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode("utf-8")
            self.round_metrics['payload_bytes'] = len(serialized.encode("utf-8"))
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "compressed_data": serialized,
                "num_samples": num_samples,
                "metrics": metrics,
                "protocol": protocol
            }
            if getattr(self.quantizer, "get_compression_stats", None):
                try:
                    stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
                    print(f"[Client {self.client_id}] Quantized: {stats.get('compression_ratio', 0):.2f}x, "
                          f"{stats.get('compressed_size_mb', 0):.2f} MB")
                except Exception:
                    pass
        else:
            serialized_weights = self.serialize_weights(updated_weights)
            self.round_metrics['payload_bytes'] = len(serialized_weights.encode('utf-8') if isinstance(serialized_weights, str) else len(serialized_weights))
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "weights": serialized_weights,
                "num_samples": num_samples,
                "metrics": metrics,
                "protocol": protocol
            }

        comm_start = time.time()
        success = False
        protocols_to_try = self._build_update_protocol_order(protocol, update_message)
        
        for attempt_protocol in protocols_to_try:
            if success:
                break
            try:
                attempt_message = dict(update_message)
                attempt_message['protocol'] = attempt_protocol
                payload_size_bytes = self._estimate_update_payload_bytes(attempt_protocol, attempt_message)
                if attempt_protocol == 'mqtt':
                    if payload_size_bytes > MQTT_MAX_PAYLOAD_BYTES:
                        self._send_chunked_update_via_mqtt(attempt_message)
                    else:
                        self._send_via_mqtt(attempt_message)
                    success = True
                elif attempt_protocol == 'amqp' and pika is not None:
                    if payload_size_bytes > AMQP_MAX_FRAME_BYTES:
                        self._send_chunked_update_via_amqp(attempt_message)
                    else:
                        self._send_via_amqp(attempt_message)
                    success = True
                elif attempt_protocol == 'grpc' and grpc is not None:
                    self._send_via_grpc(attempt_message)
                    success = True
                elif attempt_protocol == 'quic' and asyncio is not None:
                    self._ensure_protocol_connection_sync('quic')
                    self._send_via_quic(attempt_message)
                    success = True
                elif attempt_protocol == 'http3' and HTTP3_AVAILABLE and asyncio is not None:
                    self._ensure_protocol_connection_sync('http3')
                    if payload_size_bytes > HTTP3_MAX_STREAM_DATA_BYTES:
                        self._send_chunked_update_via_http3(attempt_message)
                    else:
                        self._send_via_http3(attempt_message)
                    success = True
                elif attempt_protocol == 'dds' and DDS_AVAILABLE:
                    if payload_size_bytes > CHUNK_SIZE:
                        self._send_chunked_update_via_dds(attempt_message)
                    else:
                        self._send_via_dds(attempt_message)
                    success = True
                if success:
                    self.selected_protocol = attempt_protocol
                    self._last_update_protocol = attempt_protocol
            except Exception as e:
                print(f"Client {self.client_id} ERROR: {attempt_protocol} update send failed: {e}")
                continue
        
        if success:
            self.round_metrics['uplink_model_comm_time'] = time.time() - comm_start
            # Total uplink time is completed after evaluation metrics are sent
            self.round_metrics['uplink_metrics_comm_time'] = 0.0
            self.round_metrics['communication_time'] = self.round_metrics['uplink_model_comm_time']
            self.round_metrics['success'] = True
            self.waiting_for_aggregated_model = True
        else:
            print(f"Client {self.client_id} ERROR: All protocols failed!")
            self.round_metrics['success'] = False
            self.round_metrics['uplink_model_comm_time'] = 0.0
            self.round_metrics['uplink_metrics_comm_time'] = 0.0
    
    def evaluate_model(self, evaluation_round_num: Optional[int] = None):
        """Evaluate model on validation data and send metrics to server via RL-selected protocol.

        Args:
            evaluation_round_num: Round index for metrics payload and uplink Q-learning log.
                When omitted, uses ``self.current_round``. Pass explicitly when evaluation
                follows ``_apply_global_model_weights`` and ``current_round`` may already
                have been advanced by a deferred ``start_training`` for the *next* round —
                otherwise downlink rows (tied to the global model round) and uplink rows
                disagree in the database, and the server can reject metrics.
        """
        if not self.is_active:
            return

        report_round = (
            int(evaluation_round_num)
            if evaluation_round_num is not None
            else int(self.current_round)
        )

        loss, accuracy = self.model.evaluate(
            self.validation_generator,
            verbose=0
        )
        loss_f = float(loss)
        accuracy_f = float(accuracy)
        client_converged = (
            1.0 if (stop_on_client_convergence() and self._would_converge_after_eval(loss_f)) else 0.0
        )
        
        num_samples = self.validation_generator.n
        
        # Use the same transport as the model upload for this round. train_local_model() already
        # called select_protocol() once and registered (state, action) for the uplink; calling
        # select_protocol() again would append a second pair while update_q_value() only trains
        # the last entry — pairing round rewards with the wrong action (e.g. HTTP/3 for metrics
        # while the slow uplink used another stack).
        protocol = (self._last_update_protocol or self.selected_protocol or "mqtt")
        self.selected_protocol = protocol
        if self.env_manager is not None:
            self._last_rl_state = (
                getattr(self, "_last_uplink_rl_state", None) or self.env_manager.get_current_state()
            )
        
        training_time_sec = float(self.round_metrics.get('training_time', 0.0))
        uplink_model_comm_sec = float(self.round_metrics.get('uplink_model_comm_time', 0.0))
        # This round's evaluation-metrics uplink is sent below; do not include stale prior-round timing.
        round_time_sec = training_time_sec + uplink_model_comm_sec
        battery_soc = self.env_manager.battery_soc if (USE_RL_SELECTION and self.env_manager) else 1.0
        battery_soc_before_energy = float(battery_soc)

        metrics_message = {
            "client_id": self.client_id,
            "round": report_round,
            "num_samples": num_samples,
            "loss": loss_f,
            "accuracy": accuracy_f,
            "battery_soc": float(battery_soc),
            "training_time_sec": training_time_sec,
            "uplink_model_comm_sec": uplink_model_comm_sec,
            "round_time_sec": float(round_time_sec),
            "client_converged": float(client_converged),
            "metrics": {
                "loss": loss_f,
                "accuracy": accuracy_f,
                "battery_soc": float(battery_soc),
                "training_time_sec": training_time_sec,
                "uplink_model_comm_sec": uplink_model_comm_sec,
                "round_time_sec": float(round_time_sec),
                "client_converged": float(client_converged),
            },
        }
        energy_j_total = 0.0
        
        metrics_comm_start = time.time()
        try:
            if protocol == 'mqtt':
                self._send_metrics_via_mqtt(metrics_message)
            elif protocol == 'amqp':
                self._send_metrics_via_amqp(metrics_message)
            elif protocol == 'grpc':
                self._send_metrics_via_grpc(metrics_message)
            elif protocol == 'quic':
                self._ensure_protocol_connection_sync('quic')
                self._send_metrics_via_quic(metrics_message)
            elif protocol == 'http3':
                self._ensure_protocol_connection_sync('http3')
                self._send_metrics_via_http3(metrics_message)
            elif protocol == 'dds':
                self._send_metrics_via_dds(metrics_message)
            else:
                raise ValueError(f"Unknown protocol {protocol}")
            
            self.round_metrics['uplink_metrics_comm_time'] = time.time() - metrics_comm_start
            self.round_metrics['communication_time'] = (
                self.round_metrics.get('uplink_model_comm_time', 0.0)
                + self.round_metrics['uplink_metrics_comm_time']
            )
            # Battery from radio (TX and RX bits) + CPU; use all protocols' bytes for the round
            if USE_RL_SELECTION and self.env_manager:
                try:
                    protocol_for_energy = self.selected_protocol or 'mqtt'
                    try:
                        bytes_sent, bytes_recv = get_round_bytes_sent_received(
                            report_round, protocol=None
                        )
                    except Exception:
                        bytes_sent, bytes_recv = 0, 0
                    t_round = (
                        self.round_metrics.get('training_time', 0.0)
                        + self.round_metrics['communication_time']
                    )
                    try:
                        import psutil
                        cpu_util = psutil.cpu_percent(interval=0.0)
                    except Exception:
                        cpu_util = 50.0
                    alpha = PROTOCOL_ENERGY_ALPHA.get(protocol_for_energy, 1.0)
                    beta = PROTOCOL_CPU_BETA.get(protocol_for_energy, 1.0)
                    E_radio_baseline = k_tx * (bytes_sent * 8) + k_rx * (bytes_recv * 8) + E_fixed
                    E_radio = alpha * E_radio_baseline
                    E_cpu = P_CPU_MAX * (cpu_util / 100.0) * t_round * beta
                    E_total = E_radio + E_cpu
                    energy_j_total = float(E_total)
                    soc = self.env_manager.battery_soc
                    delta_soc = E_total / BATTERY_CAP_J
                    new_soc = soc - delta_soc
                    self.env_manager.update_battery(new_soc, E_total)
                except Exception:
                    pass
            uplink_metrics_comm_sec = float(self.round_metrics.get('uplink_metrics_comm_time', 0.0))
            total_fl_wall_time_sec = (
                training_time_sec
                + uplink_model_comm_sec
                + uplink_metrics_comm_sec
            )
            battery_soc_after = (
                float(self.env_manager.battery_soc)
                if (USE_RL_SELECTION and self.env_manager)
                else float(battery_soc_before_energy)
            )
            append_client_fl_metrics_record(
                self.client_id,
                {
                    "client_id": self.client_id,
                    "round": report_round,
                    "loss": loss_f,
                    "accuracy": accuracy_f,
                    "training_time_sec": training_time_sec,
                    "uplink_model_comm_sec": uplink_model_comm_sec,
                    "uplink_metrics_comm_sec": uplink_metrics_comm_sec,
                    "total_fl_wall_time_sec": float(total_fl_wall_time_sec),
                    "battery_energy_joules": float(energy_j_total),
                    "battery_soc_before": float(battery_soc_before_energy),
                    "battery_soc_after": float(battery_soc_after),
                },
                use_case=use_case_from_env("emotion"),
                protocol=protocol,
            )
            print(f"Client {self.client_id} sent evaluation metrics for round {report_round}")
            print(f"Evaluation metrics - Loss: {loss_f:.4f}, Accuracy: {accuracy_f:.4f}")
            # RL update and optional Q-convergence end condition (battery already updated above for metrics)
            # When USE_RL_EXPLORATION: update Q, log, epsilon decay (training). When false: greedy
            # protocol from Q-table / converged map (still records selection unless RL_INFERENCE_ONLY); Q unchanged.
            # Only the *stopping* condition differs: USE_QL_CONVERGENCE -> stop when Q converges; else stop on loss.
            # Uses the dedicated UPLINK Q-learning agent.
            if USE_RL_SELECTION and self.rl_selector_uplink and self.env_manager:
                try:
                    resources = self.env_manager.get_resource_consumption(
                        protocol=self.selected_protocol or 'mqtt',
                        uplink_payload_bytes=self.round_metrics.get('payload_bytes'),
                        downlink_payload_bytes=getattr(self, '_downlink_payload_bytes', None),
                    )
                    payload_bytes = self.round_metrics.get('payload_bytes') or (12 * 1024 * 1024)
                    protocol = self.selected_protocol or 'mqtt'
                    comm_wall_actual = float(
                        self.round_metrics.get('uplink_model_comm_time', 0.0)
                        + self.round_metrics.get('uplink_metrics_comm_time', 0.0)
                    )
                    # Uplink reward may use communication model (T_calc); data collection always uses wall time
                    comm_time_for_reward = comm_wall_actual
                    t_calc_for_reward = None
                    reward_scenario = os.environ.get("RL_REWARD_SCENARIO", "").strip().lower()
                    if USE_COMMUNICATION_MODEL_REWARD and reward_scenario:
                        simulated_t_calc = self._get_t_calc_for_scenario(protocol, payload_bytes, reward_scenario)
                        if simulated_t_calc is not None:
                            comm_time_for_reward = simulated_t_calc
                            t_calc_for_reward = simulated_t_calc
                        else:
                            t_calc_for_reward = self._get_t_calc_for_reward(protocol, payload_bytes)
                    elif USE_COMMUNICATION_MODEL_REWARD:
                        t_calc_for_reward = self._get_t_calc_for_reward(protocol, payload_bytes)

                    skip_uplink_training = False
                    if (
                        getattr(self, "_rl_boundary_collection_phase", False)
                        and finalize_rl_boundary_collection_and_start_training is not None
                    ):
                        import psutil
                        cpu_m = psutil.cpu_percent(interval=0.0)
                        mem_m = psutil.virtual_memory().percent
                        resource_load = (cpu_m + mem_m) / 2.0
                        batt = float(self.env_manager.battery_soc)
                        self._rl_boundary_comm_samples.append(comm_wall_actual)
                        self._rl_boundary_res_samples.append(resource_load)
                        self._rl_boundary_batt_samples.append(batt)
                        n_r = len(self._rl_boundary_comm_samples)
                        print(
                            f"[Client {self.client_id}] RL Phase 1 (uplink data collection): sample {n_r}/{RL_PHASE0_ROUNDS} "
                            f"(wall comm={comm_wall_actual:.3f}s, mean load={resource_load:.1f}%, SoC={batt:.3f})"
                        )
                        if n_r >= RL_PHASE0_ROUNDS:
                            finalize_rl_boundary_collection_and_start_training(
                                self.rl_selector_uplink,
                                self.rl_selector_downlink,
                                self.env_manager,
                                self._rl_boundary_comm_samples,
                                self._rl_boundary_res_samples,
                                self._rl_boundary_batt_samples,
                                client_id=self.client_id,
                                downlink_comm_times=self._rl_boundary_downlink_comm_samples,
                                resource_loads_downlink=self._rl_boundary_res_samples_dl,
                                battery_socs_downlink=self._rl_boundary_batt_samples_dl,
                            )
                            self._rl_boundary_collection_phase = False
                            self._rl_boundary_comm_samples = []
                            self._rl_boundary_downlink_comm_samples = []
                            self._rl_boundary_res_samples = []
                            self._rl_boundary_batt_samples = []
                            self._rl_boundary_res_samples_dl = []
                            self._rl_boundary_batt_samples_dl = []
                            print(
                                f"[Client {self.client_id}] RL Phase 3: Q-learning training started "
                                f"(epsilon decay on episodes)."
                            )
                        skip_uplink_training = True

                    if not skip_uplink_training and USE_RL_EXPLORATION:
                        reward = self.rl_selector_uplink.calculate_reward(
                            communication_time=comm_time_for_reward,
                            success=self.round_metrics['success'],
                            resource_consumption=resources,
                            t_calc=t_calc_for_reward,
                        )
                        if self.env_manager:
                            self.env_manager.update_comm_level_from_time(
                                comm_time_for_reward,
                                self.rl_selector_uplink.comm_t_low,
                                self.rl_selector_uplink.comm_t_high,
                            )
                            self.env_manager.sync_battery_level_from_soc(None)
                            next_state = self.env_manager.get_current_state()
                        else:
                            next_state = None
                        self.rl_selector_uplink.update_q_value(
                            reward,
                            next_state=next_state,
                            done=False if next_state is not None else True,
                        )
                        q_delta = self.rl_selector_uplink.get_last_q_delta()
                        q_value = self.rl_selector_uplink.get_last_q_value()
                        avg_reward = (np.mean(self.rl_selector_uplink.total_rewards[-100:])
                                     if self.rl_selector_uplink.total_rewards else 0.0)
                        self.rl_selector_uplink.end_episode()
                        q_uplink_converged = self.rl_selector_uplink.check_q_converged(
                            threshold=Q_CONVERGENCE_THRESHOLD,
                            patience=Q_CONVERGENCE_PATIENCE,
                            state=self._last_uplink_rl_state,
                        )
                        q_downlink_converged = (
                            self.rl_selector_downlink.check_q_converged(
                                threshold=Q_CONVERGENCE_THRESHOLD,
                                patience=Q_CONVERGENCE_PATIENCE,
                                state=self._last_downlink_rl_state,
                            )
                            if self.rl_selector_downlink is not None
                            else True
                        )
                        q_both_converged = q_uplink_converged and q_downlink_converged
                        # Log uplink Q-step (skipped during Phase 1 data collection)
                        if (
                            log_q_step is not None
                            and self._last_rl_state is not None
                            and self._rl_q_logging_allowed()
                        ):
                            st = self._last_rl_state
                            reward_details = self.rl_selector_uplink.get_last_reward_breakdown()
                            log_q_step(
                                client_id=self.client_id,
                                round_num=report_round,
                                episode=self.rl_selector_uplink.episode_count - 1,
                                state_comm_level=st.get('comm_level', ''),
                                state_resource=st.get('resource', ''),
                                state_battery_level=st.get('battery_level', ''),
                                **rl_state_network_kwargs(st),
                                action=self._last_update_protocol or self.selected_protocol or 'mqtt',
                                reward=reward,
                                q_delta=q_delta,
                                epsilon=self.rl_selector_uplink.epsilon,
                                q_value=q_value,
                                avg_reward_last_100=float(avg_reward),
                                converged=(
                                    q_both_converged
                                    if USE_QL_CONVERGENCE
                                    else q_uplink_converged
                                ),
                                metric_communication_time=reward_details.get('communication_time'),
                                metric_success=reward_details.get('success'),
                                metric_cpu_usage=reward_details.get('cpu_usage'),
                                metric_memory_usage=reward_details.get('memory_usage'),
                                metric_bandwidth_usage=reward_details.get('bandwidth_usage'),
                                metric_battery_level=reward_details.get('battery_level'),
                                metric_energy_usage=reward_details.get('energy_usage'),
                                reward_base=reward_details.get('reward_base'),
                                reward_communication_time=reward_details.get('reward_communication_time'),
                                reward_resource_penalty=reward_details.get('reward_resource_penalty'),
                                reward_battery_penalty=reward_details.get('reward_battery_penalty'),
                                reward_total=reward_details.get('reward_total', reward),
                                link_direction='uplink',
                            )
                        if (
                            USE_QL_CONVERGENCE
                            and q_uplink_converged
                            and not q_downlink_converged
                        ):
                            print(
                                f"[Client {self.client_id}] Uplink Q converged at round {report_round} but downlink "
                                f"has not; continuing (uplink Q-updates unchanged)."
                            )
                        # End training only when both uplink and downlink Q-learning converged
                        if USE_QL_CONVERGENCE and q_both_converged and stop_on_client_convergence():
                            self.has_converged = True
                            print(
                                f"[Client {self.client_id}] RL convergence reached at round {report_round} "
                                f"(last {Q_CONVERGENCE_PATIENCE} consecutive protocol selections identical "
                                f"on uplink AND downlink)"
                            )
                            try:
                                self.rl_selector_uplink.save_q_table()
                            except Exception as e:
                                print(f"[Client {self.client_id}] Warning: could not save uplink Q-table on convergence: {e}")
                            try:
                                self.rl_selector_downlink.save_q_table()
                            except Exception as e:
                                print(f"[Client {self.client_id}] Warning: could not save downlink Q-table on convergence: {e}")
                            self._notify_convergence_to_server()
                            self._disconnect_after_convergence()
                            return
                except Exception as rl_e:
                    print(f"[Client {self.client_id}] Uplink RL update error: {rl_e}")
            if not USE_QL_CONVERGENCE and ENABLE_LOCAL_CONVERGENCE_STOP and stop_on_client_convergence():
                self._update_client_convergence_and_maybe_disconnect(loss_f)
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics: {e}")
            import traceback
            traceback.print_exc()

    def _would_converge_after_eval(self, loss: float) -> bool:
        """True if this eval loss would trigger local convergence in the same round (mirrors FL_Client_gRPC)."""
        if self.current_round < MIN_ROUNDS or self.has_converged:
            return False
        if self.best_loss - loss > CONVERGENCE_THRESHOLD:
            return False
        return (self.rounds_without_improvement + 1) >= CONVERGENCE_PATIENCE

    def _update_client_convergence_and_maybe_disconnect(self, loss: float):
        """Track local convergence and disconnect this client when converged."""
        if self.current_round < MIN_ROUNDS:
            self.best_loss = min(self.best_loss, float(loss))
            return

        if self.best_loss - float(loss) > CONVERGENCE_THRESHOLD:
            self.best_loss = float(loss)
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

        if self.rounds_without_improvement >= CONVERGENCE_PATIENCE:
            self.has_converged = True
            print(f"[Client {self.client_id}] Local convergence reached at round {self.current_round}")
            if stop_on_client_convergence():
                self._notify_convergence_to_server()
                self._disconnect_after_convergence()

    def _notify_convergence_to_server(self):
        """Notify server this client is converged using gRPC control signal."""
        if grpc is None or federated_learning_pb2 is None or federated_learning_pb2_grpc is None:
            print(f"[gRPC] Client {self.client_id} convergence signal skipped: gRPC unavailable")
            return
        try:
            grpc_host = os.getenv("GRPC_HOST", "fl-server-unified-emotion")
            grpc_port = int(os.getenv("GRPC_PORT", "50051"))
            options = [
                ('grpc.max_send_message_length', GRPC_MAX_MESSAGE_BYTES),
                ('grpc.max_receive_message_length', GRPC_MAX_MESSAGE_BYTES),
                ('grpc.keepalive_time_ms', 600000),
                ('grpc.keepalive_timeout_ms', 60000),
            ]
            channel = grpc.insecure_channel(f'{grpc_host}:{grpc_port}', options=options)
            stub = federated_learning_pb2_grpc.FederatedLearningStub(channel)
            response = stub.SendModelUpdate(
                federated_learning_pb2.ModelUpdate(
                    client_id=self.client_id,
                    round=self.current_round,
                    weights=b"",
                    num_samples=0,
                    metrics={"client_converged": 1.0}
                )
            )
            if response.success:
                print(f"[gRPC] Client {self.client_id} convergence notification sent")
            else:
                print(f"[gRPC] Convergence notification failed: {response.message}")
            channel.close()
        except Exception as e:
            print(f"[gRPC] Client {self.client_id} failed to notify convergence: {e}")

    def _disconnect_after_convergence(self):
        """Stop participating once local convergence is reached."""
        self.is_active = False
        self.shutdown_requested = True
        print(f"[Client {self.client_id}] Disconnecting after local convergence")
        try:
            self.cleanup()
        except Exception:
            pass
        try:
            self.mqtt_client.disconnect()
        except Exception:
            pass
    
    # ============================================================================
    # Protocol-Specific Send Methods (Data Transmission)
    # ============================================================================

    def _send_chunked_update_via_mqtt(self, message: dict):
        """Send a large model update over MQTT in multiple chunks."""
        chunk_messages = self._chunked_update_messages(message, MQTT_UPDATE_CHUNK_BYTES, "update_chunk")
        total_chunks = len(chunk_messages)
        print(f"Client {self.client_id}: Sending model update in {total_chunks} MQTT chunks")
        for chunk in chunk_messages:
            payload = json.dumps(chunk)
            result = self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_NO_CONN:
                raise Exception("MQTT not connected")
            result.wait_for_publish(timeout=5)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(f"MQTT publish failed with rc={result.rc}")
            if (chunk["chunk_index"] + 1) % 20 == 0 or (chunk["chunk_index"] + 1) == total_chunks:
                print(f"  Sent {chunk['chunk_index'] + 1}/{total_chunks} MQTT chunks")
        log_sent_packet(
            packet_size=len(json.dumps(message)),
            peer="server",
            protocol="MQTT",
            round=self.current_round,
            extra_info=f"model_update_chunked total_chunks={total_chunks}"
        )
        print(f"Client {self.client_id} sent chunked model update for round {self.current_round} via MQTT")

    def _send_chunked_update_via_amqp(self, message: dict):
        """Send a large model update over AMQP in multiple chunks."""
        chunk_messages = self._chunked_update_messages(message, AMQP_UPDATE_CHUNK_BYTES, "update_chunk")
        total_chunks = len(chunk_messages)
        print(f"Client {self.client_id}: Sending model update in {total_chunks} AMQP chunks")
        parameters = self._get_amqp_connection_parameters()
        connection = pika.BlockingConnection(parameters)
        try:
            channel = connection.channel()
            channel.exchange_declare(exchange='fl_client_updates', exchange_type='direct', durable=True)
            for chunk in chunk_messages:
                payload = json.dumps(chunk)
                channel.basic_publish(
                    exchange='fl_client_updates',
                    routing_key=f'client_{self.client_id}_update',
                    body=payload,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                if (chunk["chunk_index"] + 1) % 20 == 0 or (chunk["chunk_index"] + 1) == total_chunks:
                    print(f"  Sent {chunk['chunk_index'] + 1}/{total_chunks} AMQP chunks")
        finally:
            connection.close()
        log_sent_packet(
            packet_size=len(json.dumps(message)),
            peer="server",
            protocol="AMQP",
            round=self.current_round,
            extra_info=f"model_update_chunked total_chunks={total_chunks}"
        )
        print(f"Client {self.client_id} sent chunked model update for round {self.current_round} via AMQP")

    def _send_chunked_update_via_http3(self, message: dict):
        """Send a large model update over HTTP/3 in multiple chunks."""
        chunk_messages = self._chunked_update_messages(message, HTTP3_UPDATE_CHUNK_BYTES, "update_chunk")
        total_chunks = len(chunk_messages)
        print(f"Client {self.client_id}: Sending model update in {total_chunks} HTTP/3 chunks")
        for chunk in chunk_messages:
            payload = json.dumps(chunk)
            self._send_http3_payload(payload, timeout=20)
            if (chunk["chunk_index"] + 1) % 20 == 0 or (chunk["chunk_index"] + 1) == total_chunks:
                print(f"  Sent {chunk['chunk_index'] + 1}/{total_chunks} HTTP/3 chunks")
        log_sent_packet(
            packet_size=len(json.dumps(message)),
            peer="server",
            protocol="HTTP/3",
            round=self.current_round,
            extra_info=f"model_update_chunked total_chunks={total_chunks}"
        )
        print(f"Client {self.client_id} sent chunked model update for round {self.current_round} via HTTP/3")

    def _send_chunked_update_via_dds(self, message: dict):
        """Send a large model update over DDS in multiple chunks."""
        if not DDS_AVAILABLE or not self.dds_update_chunk_writer:
            raise NotImplementedError("DDS chunk writer not available")

        weights_bytes = self._dds_serialized_weights_bytes(message)

        metrics = message.get('metrics', {})
        total_chunks = self.send_model_update_chunked(
            round_num=message['round'],
            serialized_weights=weights_bytes,
            num_samples=message.get('num_samples', 0),
            loss=float(metrics.get('loss', message.get('loss', 0.0))),
            accuracy=float(metrics.get('accuracy', message.get('accuracy', 0.0))),
        )

        log_sent_packet(
            packet_size=len(weights_bytes),
            peer="server",
            protocol="DDS",
            round=self.current_round,
            extra_info=f"model_update_chunked total_chunks={total_chunks}"
        )
        print(f"Client {self.client_id} sent chunked model update for round {self.current_round} via DDS")
    
    def _send_via_mqtt(self, message: dict):
        """Send model update via MQTT"""
        try:
            payload = json.dumps(message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via MQTT - size: {payload_size_mb:.2f} MB")
            
            result = self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, payload, qos=1)
            # FAIR FIX: Use shorter timeout (5s) aligned with other protocols, or non-blocking check
            # MQTT QoS 1 ensures delivery, so we don't need to wait for full acknowledgment
            # This makes MQTT behavior similar to AMQP/gRPC which return immediately after send
            if result.rc == mqtt.MQTT_ERR_NO_CONN:
                raise Exception("MQTT not connected")
            # Only wait briefly to ensure message is queued (not blocking for full delivery)
            result.wait_for_publish(timeout=5)
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="MQTT",
                round=self.current_round,
                extra_info="model_update"
            )
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"Client {self.client_id} sent model update for round {self.current_round} via MQTT")
                print(f"Training metrics - Loss: {message['metrics']['loss']:.4f}, Accuracy: {message['metrics']['accuracy']:.4f}")
            else:
                raise Exception(f"MQTT publish failed with rc={result.rc}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via MQTT: {e}")
            raise
    
    def _send_metrics_via_mqtt(self, message: dict):
        """Send metrics via MQTT"""
        try:
            payload = json.dumps(message)
            result = self.mqtt_client.publish(TOPIC_CLIENT_METRICS, payload, qos=1)
            # FAIR FIX: Use shorter timeout (5s) aligned with other protocols
            # Metrics are small, so 5s is sufficient for queue confirmation
            if result.rc == mqtt.MQTT_ERR_NO_CONN:
                raise Exception("MQTT not connected")
            result.wait_for_publish(timeout=5)
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="MQTT",
                round=self.current_round,
                extra_info="metrics"
            )
            
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(f"MQTT publish failed with rc={result.rc}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via MQTT: {e}")
            raise
    
    def _send_via_amqp(self, message: dict):
        """Send model update via AMQP (RabbitMQ)"""
        if pika is None:
            raise ImportError("pika module not available for AMQP")
        
        try:
            estimated_payload = json.dumps(message).encode('utf-8')
            # Get AMQP config
            parameters = self._get_amqp_connection_parameters()
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            # Declare exchange and send message
            channel.exchange_declare(exchange='fl_client_updates', exchange_type='direct', durable=True)
            
            payload = estimated_payload.decode('utf-8')
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via AMQP - size: {payload_size_mb:.2f} MB")
            
            channel.basic_publish(
                exchange='fl_client_updates',
                routing_key='client.update',
                body=payload,
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="model_update"
            )
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via AMQP")
            connection.close()
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via AMQP: {e}")
            raise
    
    def _send_metrics_via_amqp(self, message: dict):
        """Send metrics via AMQP (RabbitMQ)"""
        if pika is None:
            raise ImportError("pika module not available for AMQP")
        
        try:
            parameters = self._get_amqp_connection_parameters()
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            
            channel.exchange_declare(exchange='fl_client_updates', exchange_type='direct', durable=True)
            
            payload = json.dumps(message)
            channel.basic_publish(
                exchange='fl_client_updates',
                routing_key='client.metrics',
                body=payload,
                properties=pika.BasicProperties(delivery_mode=2)
            )
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="AMQP",
                round=self.current_round,
                extra_info="metrics"
            )
            
            connection.close()
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via AMQP: {e}")
            raise
    
    def _send_via_grpc(self, message: dict):
        """Send model update via gRPC using persistent stub"""
        if grpc is None or federated_learning_pb2 is None:
            raise ImportError("grpc modules not available for gRPC")
        
        # Use persistent grpc_stub with lock to prevent concurrent calls
        with self.grpc_lock:
            try:
                if self.grpc_stub is None:
                    raise Exception("gRPC stub not initialized. Listener may not be started.")
                
                stub = self.grpc_stub
            
                # Send model update (chunked when > 4 MB)
                if 'compressed_data' in message:
                    payload = json.dumps(message).encode('utf-8')
                else:
                    payload = (message['weights'].encode('utf-8') if isinstance(message['weights'], str) else message['weights'])
                payload_size = len(payload)
                payload_size_mb = payload_size / (1024 * 1024)
                print(f"Client {self.client_id} sending via gRPC - size: {payload_size_mb:.2f} MB")
                metrics_dict = {k: float(v) for k, v in message['metrics'].items()}

                if payload_size > GRPC_CHUNK_SIZE:
                    chunks = [payload[i:i + GRPC_CHUNK_SIZE] for i in range(0, payload_size, GRPC_CHUNK_SIZE)]
                    total_chunks = len(chunks)
                    for i, chunk_data in enumerate(chunks):
                        req = federated_learning_pb2.ModelUpdate(
                            client_id=message['client_id'],
                            round=message['round'],
                            weights=chunk_data,
                            num_samples=message['num_samples'] if i == 0 else 0,
                            metrics=metrics_dict if i == 0 else {},
                            chunk_index=i,
                            total_chunks=total_chunks
                        )
                        response = stub.SendModelUpdate(req)
                        if not response.success:
                            raise Exception(f"gRPC chunk {i + 1}/{total_chunks} failed: {response.message}")
                    print(f"Client {self.client_id} sent model update in {total_chunks} chunks ({payload_size} bytes) via gRPC")
                else:
                    response = stub.SendModelUpdate(
                        federated_learning_pb2.ModelUpdate(
                            client_id=message['client_id'],
                            round=message['round'],
                            weights=payload,
                            num_samples=message['num_samples'],
                            metrics=metrics_dict
                        )
                    )
                    if not response.success:
                        raise Exception(f"gRPC send failed: {response.message}")
                    print(f"Client {self.client_id} sent model update for round {self.current_round} via gRPC")

                # Set flag: we're now waiting for aggregated model
                self.waiting_for_aggregated_model = True

                log_sent_packet(
                    packet_size=payload_size,
                    peer="server",
                    protocol="gRPC",
                    round=self.current_round,
                    extra_info="model_update"
                )
                
            except Exception as e:
                print(f"Client {self.client_id} ERROR sending via gRPC: {e}")
                raise
    
    def _send_metrics_via_grpc(self, message: dict):
        """Send metrics via gRPC using persistent stub"""
        if grpc is None or federated_learning_pb2 is None:
            raise ImportError("grpc modules not available for gRPC")
        
        # Use persistent grpc_stub with lock to prevent concurrent calls
        with self.grpc_lock:
            try:
                if self.grpc_stub is None:
                    raise Exception("gRPC stub not initialized. Listener may not be started.")
                
                stub = self.grpc_stub
                
                response = stub.SendMetrics(
                    federated_learning_pb2.Metrics(
                        client_id=message['client_id'],
                        round=message['round'],
                        loss=message.get('loss', message.get('metrics', {}).get('loss', 0.0)),
                        accuracy=message.get('accuracy', message.get('metrics', {}).get('accuracy', 0.0)),
                        num_samples=message['num_samples'],
                        battery_soc=float(message.get('battery_soc', message.get('metrics', {}).get('battery_soc', 1.0))),
                        round_time_sec=float(message.get('round_time_sec', message.get('metrics', {}).get('round_time_sec', 0.0))),
                        client_converged=float(
                            message.get('client_converged', message.get('metrics', {}).get('client_converged', 0.0))
                        ),
                        training_time_sec=float(
                            message.get('training_time_sec', message.get('metrics', {}).get('training_time_sec', 0.0))
                        ),
                        uplink_model_comm_sec=float(
                            message.get(
                                'uplink_model_comm_sec',
                                message.get('metrics', {}).get('uplink_model_comm_sec', 0.0),
                            )
                        ),
                    )
                )
                
                payload_size = len(json.dumps(message))
                log_sent_packet(
                    packet_size=payload_size,
                    peer="server",
                    protocol="gRPC",
                    round=self.current_round,
                    extra_info="metrics"
                )
                
                if not response.success:
                    raise Exception(f"gRPC send failed: {response.message}")
                
            except Exception as e:
                print(f"Client {self.client_id} ERROR sending metrics via gRPC: {e}")
                raise
    
    async def _ensure_quic_connection(self):
        """Establish persistent QUIC connection if not already connected"""
        if self.quic_protocol is not None:
            return  # Already connected
        
        # Start QUIC connection thread if not running
        if self.quic_thread is None or not self.quic_thread.is_alive():
            self.quic_thread = threading.Thread(
                target=self._run_quic_loop,
                daemon=True,
                name=f"QUIC-Client-{self.client_id}"
            )
            self.quic_thread.start()
            
            # Wait for connection to establish
            max_wait = 10  # seconds
            waited = 0
            while self.quic_protocol is None and waited < max_wait:
                await asyncio.sleep(0.1)
                waited += 0.1
            
            if self.quic_protocol is None:
                raise ConnectionError(f"QUIC connection not established after {max_wait}s")
            
            print(f"[QUIC] Client {self.client_id} connection ready")
    
    def _run_quic_loop(self):
        """Run QUIC event loop in a separate thread"""
        self.quic_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.quic_loop)
        try:
            # Keep loop running indefinitely
            self.quic_loop.run_forever()
        except Exception as e:
            print(f"[QUIC] Event loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Don't close the loop here - it should stay open for sending
            print(f"[QUIC] Event loop stopped for client {self.client_id}")
    
    async def _quic_connection_loop(self):
        """Maintain persistent QUIC connection (runs in background)"""
        import ssl
        quic_host = os.getenv("QUIC_HOST", "localhost")
        quic_port = int(os.getenv("QUIC_PORT", "4433"))
        
        # QUIC config: cubic congestion, 60s idle; 128 MB flow control (aligned with server for fair FL comparison)
        QUIC_MAX_DATA_BYTES = 128 * 1024 * 1024  # 128 MB
        config = QuicConfiguration(
            is_client=True,
            alpn_protocols=["fl"],
            verify_mode=ssl.CERT_NONE,
            congestion_control_algorithm="cubic",
            idle_timeout=60.0,
            max_data=QUIC_MAX_DATA_BYTES,
            max_stream_data=QUIC_MAX_DATA_BYTES,
        )
        
        print(f"[QUIC] Client {self.client_id} connecting to {quic_host}:{quic_port}...")
        print(f"[QUIC] Configuration: verify_mode=CERT_NONE, idle_timeout=600s")
        
        # Create protocol factory that sets client reference
        def create_protocol(*args, **kwargs):
            protocol = UnifiedClientQUICProtocol(*args, **kwargs)
            protocol.client = self
            print(f"[QUIC] Client {self.client_id} created protocol instance")
            return protocol
        
        # Retry connection with exponential backoff, keep retrying indefinitely
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"[QUIC] Client {self.client_id} attempt {attempt + 1}/{max_retries} - calling connect()...")
                async with connect(
                    quic_host,
                    quic_port,
                    configuration=config,
                    create_protocol=create_protocol
                ) as protocol:
                    self.quic_protocol = protocol
                    print(f"[QUIC] ✓ Client {self.client_id} established persistent connection")
                    print(f"[QUIC] Connection state: {protocol._quic.get_timer() if hasattr(protocol, '_quic') else 'unknown'}")
                    
                    # Keep connection alive indefinitely
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        print(f"[QUIC] Client {self.client_id} connection cancelled")
                    break  # Connection successful, exit retry loop
                    
            except (ConnectionError, OSError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    print(f"[QUIC] ✗ Client {self.client_id} connection attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
                    print(f"[QUIC] Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"[QUIC] ✗✗✗ Client {self.client_id} connection FAILED after {max_retries} attempts: {type(e).__name__}: {e}")
                    print(f"[QUIC] QUIC protocol will NOT be available for this client")
                    print(f"[QUIC] Server status: listening on {quic_host}:{quic_port}")
                    print(f"[QUIC] Possible causes:")
                    print(f"[QUIC]   1. Server not responding to QUIC handshake")
                    print(f"[QUIC]   2. Firewall blocking UDP port {quic_port}")
                    print(f"[QUIC]   3. Server's asyncio loop not running properly")
                    self.quic_protocol = None
            except Exception as e:
                print(f"[QUIC] ✗ Client {self.client_id} unexpected connection error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.quic_protocol = None
                break
    
    async def _handle_quic_message_async(self, message: dict):
        """Handle QUIC message received from server asynchronously"""
        msg_type = message.get('type')
        print(f"[QUIC] Client {self.client_id} received message type: {msg_type}")
        
        if msg_type == 'global_model':
            self.on_global_model_received_quic(message)
        elif msg_type == 'start_training':
            self.on_start_training_quic(message)
        elif msg_type == 'start_evaluation':
            self.on_start_evaluation_quic(message)
    
    def _handle_quic_message(self, message: dict):
        """Handle QUIC message received from server (called from QUIC protocol)""" 
        msg_type = message.get('type')
        print(f"[QUIC] Client {self.client_id} received message type: {msg_type}")
        
        if msg_type == 'global_model':
            self.on_global_model_received_quic(message)
        elif msg_type == 'start_training':
            self.on_start_training_quic(message)
        elif msg_type == 'start_evaluation':
            self.on_start_evaluation_quic(message)
    
    def on_global_model_received_quic(self, message):
        """Handle global model received via QUIC"""
        try:
            round_num = message['round']
            print(f"[QUIC] Client {self.client_id} received global model for round {round_num}")
            
            # Deserialize weights
            if 'quantized_data' in message and self.quantizer is not None:
                raw = message['quantized_data']
                if isinstance(raw, str):
                    raw = base64.b64decode(raw.encode('utf-8'))
                compressed_data = pickle.loads(raw) if isinstance(raw, bytes) else raw
                weights = self.quantizer.decompress(compressed_data)
            else:
                weights = self.deserialize_weights(message['weights'])
            
            self._global_model_receive_t0 = time.time()
            self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=message.get('model_config'),
                source="QUIC",
                server_sent_unix=message.get('server_sent_unix'),
            )
            
            # Set event to signal model is ready
            if hasattr(self, 'model_received'):
                self.model_received.set()
        except Exception as e:
            print(f"[QUIC] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_training_quic(self, message):
        """Handle start training signal via QUIC"""
        try:
            print(f"[QUIC] Client {self.client_id} starting training for round {message.get('round', self.current_round + 1)}")
            
            log_received_packet(
                packet_size=len(json.dumps(message)),
                peer="server",
                protocol="QUIC",
                round=self.current_round,
                extra_info="start_training"
            )
            
            # Call the standard training handler
            self.handle_start_training(json.dumps(message).encode())
        except Exception as e:
            print(f"[QUIC] Client {self.client_id} error handling start_training: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_evaluation_quic(self, message):
        """Handle start evaluation signal via QUIC"""
        try:
            round_num = message.get('round', self.current_round)
            print(f"[QUIC] Client {self.client_id} starting evaluation for round {round_num}")
            
            log_received_packet(
                packet_size=len(json.dumps(message)),
                peer="server",
                protocol="QUIC",
                round=self.current_round,
                extra_info="start_evaluation"
            )
            
            # Call the standard evaluation handler
            self.handle_start_evaluation(json.dumps(message).encode())
        except Exception as e:
            print(f"[QUIC] Client {self.client_id} error handling evaluation signal: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_via_quic(self, message: dict):
        """Send model update via QUIC using persistent connection"""
        if asyncio is None or connect is None:
            raise ImportError("aioquic module not available for QUIC")
        
        try:
            # Check if event loop is available and running
            if self.quic_loop is None:
                raise ConnectionError("QUIC event loop not available")
            
            if self.quic_loop.is_closed():
                raise ConnectionError("QUIC event loop is closed")
            
            # Check if protocol is connected
            if self.quic_protocol is None:
                raise ConnectionError("QUIC protocol not connected")
            
            # Add 'type' field for server to identify message type
            quic_message = {**message, 'type': 'model_update'}
            
            payload = json.dumps(quic_message)
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via QUIC - size: {payload_size_mb:.2f} MB")
            
            # Use persistent connection directly via run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._do_quic_send(payload),
                self.quic_loop
            )
            future.result(timeout=15)  # Wait for send to complete
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="QUIC",
                round=self.current_round,
                extra_info="model_update"
            )
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via QUIC")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via QUIC: {e}")
            raise
    
    def _send_metrics_via_quic(self, message: dict):
        """Send metrics via QUIC using persistent connection"""
        if asyncio is None or connect is None:
            raise ImportError("aioquic module not available for QUIC")
        
        try:
            # Check if event loop is available and running
            if self.quic_loop is None:
                raise ConnectionError("QUIC event loop not available")
            
            if self.quic_loop.is_closed():
                raise ConnectionError("QUIC event loop is closed")
            
            # Check if protocol is connected
            if self.quic_protocol is None:
                raise ConnectionError("QUIC protocol not connected")
            
            # Add 'type' field for server to identify message type
            quic_message = {**message, 'type': 'metrics'}
            
            payload = json.dumps(quic_message)
            
            # Use persistent connection directly via run_coroutine_threadsafe
            future = asyncio.run_coroutine_threadsafe(
                self._do_quic_send(payload),
                self.quic_loop
            )
            future.result(timeout=15)  # Wait for send to complete
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="QUIC",
                round=self.current_round,
                extra_info="metrics"
            )
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via QUIC: {e}")
            raise
    
    async def _send_quic_persistent(self, payload: str):
        """Send data via persistent QUIC connection"""
        # Ensure connection exists
        await self._ensure_quic_connection()
        
        if self.quic_protocol is None:
            raise ConnectionError("QUIC connection not established")
        
        # Schedule send on QUIC thread's event loop
        future = asyncio.run_coroutine_threadsafe(
            self._do_quic_send(payload),
            self.quic_loop
        )
        # Wait for completion
        future.result(timeout=10)
    
    async def _do_quic_send(self, payload: str):
        """Actually send data via QUIC (runs in QUIC thread's event loop)"""
        # Ensure connection is ready
        if self.quic_protocol is None or self.quic_protocol._quic is None:
            raise ConnectionError("QUIC protocol not available")
        
        print(f"[QUIC] Client {self.client_id} preparing to send {len(payload)} bytes")
        
        # Send data via QUIC stream
        stream_id = self.quic_protocol._quic.get_next_available_stream_id()
        data = (payload + '\n').encode('utf-8')
        self.quic_protocol._quic.send_stream_data(stream_id, data, end_stream=True)
        self.quic_protocol.transmit()
        
        print(f"[QUIC] Client {self.client_id} sent on stream {stream_id}, transmitting...")
        
        # FAIR FIX: Removed artificial 1.5s delay for large messages
        # QUIC handles flow control automatically, so we don't need manual delays
        # This makes QUIC behavior similar to other protocols which don't have artificial delays
        # The transmit() call above is sufficient for immediate transmission
    
    async def _quic_send_data(self, host: str, port: int, payload: str, msg_type: str):
        """Async QUIC data send with timeout and retry (legacy method for registration)"""
        import ssl
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Use ssl.CERT_NONE for self-signed certificate verification
                config = QuicConfiguration(
                    is_client=True,
                    alpn_protocols=["fl"],  # CRITICAL: Must match server's ALPN
                    verify_mode=ssl.CERT_NONE
                )
                
                # connect() returns a QuicConnectionProtocol
                # We use create_stream() to get reader/writer
                try:
                    # Try Python 3.11+ asyncio.timeout
                    async with asyncio.timeout(5):
                        async with connect(host, port, configuration=config) as protocol:
                            # Create a stream for sending data
                            reader, writer = await protocol.create_stream()
                            writer.write((payload + '\n').encode())
                            await writer.drain()
                            # Give time for data to be transmitted before closing
                            await asyncio.sleep(0.5)
                            writer.close()
                            # Wait for close to complete
                            try:
                                await writer.wait_closed()
                            except:
                                pass
                            return  # Success
                except AttributeError as e:
                    # Python 3.8 doesn't have asyncio.timeout
                    if "has no attribute 'timeout'" in str(e):
                        # Use manual context manager handling for Python 3.8
                        connection = connect(host, port, configuration=config)
                        protocol = await asyncio.wait_for(connection.__aenter__(), timeout=5)
                        try:
                            reader, writer = await protocol.create_stream()
                            writer.write((payload + '\n').encode())
                            await writer.drain()
                            # Give time for data to be transmitted
                            await asyncio.sleep(0.5)
                            writer.close()
                            try:
                                await writer.wait_closed()
                            except:
                                pass
                            return  # Success
                        finally:
                            await connection.__aexit__(None, None, None)
                    else:
                        raise
            except asyncio.TimeoutError:
                print(f"QUIC send timeout (attempt {attempt + 1}/{max_retries}): Connection took too long")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                else:
                    raise
            except (ConnectionError, OSError) as e:
                print(f"QUIC send connection error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
            except Exception as e:
                print(f"QUIC send error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    raise
    
    # =========================================================================
    # HTTP/3 CLIENT METHODS
    # =========================================================================
    
    async def _ensure_http3_connection(self):
        """Establish persistent HTTP/3 connection if not already connected"""
        if self.http3_protocol is not None and self.http3_connection_task is not None and not self.http3_connection_task.done():
            return  # Already connected
        if self.http3_connection_task is not None and self.http3_connection_task.done():
            self.http3_protocol = None
            self.http3_connection_task = None
        
        # Start HTTP/3 connection thread if not running
        if self.http3_thread is None or not self.http3_thread.is_alive():
            self.http3_thread = threading.Thread(
                target=self._run_http3_loop,
                daemon=True,
                name=f"HTTP3-Client-{self.client_id}"
            )
            self.http3_thread.start()
        if self.http3_loop is None or self.http3_loop.is_closed():
            raise ConnectionError("HTTP/3 event loop not available")
        if self.http3_connection_task is None:
            self.http3_connection_task = asyncio.run_coroutine_threadsafe(
                self._http3_connection_loop(),
                self.http3_loop
            )
        
        # Wait for connection to establish
        max_wait = 10  # seconds
        waited = 0
        while self.http3_protocol is None and waited < max_wait:
            await asyncio.sleep(0.1)
            waited += 0.1
        
        if self.http3_protocol is None:
            raise ConnectionError(f"HTTP/3 connection not established after {max_wait}s")
        
        print(f"[HTTP/3] Client {self.client_id} connection ready")
    
    def _run_http3_loop(self):
        """Run HTTP/3 event loop in a separate thread"""
        self.http3_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.http3_loop)
        try:
            # Keep loop running indefinitely
            self.http3_loop.run_forever()
        except Exception as e:
            print(f"[HTTP/3] Event loop error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"[HTTP/3] Event loop stopped for client {self.client_id}")
    
    async def _http3_connection_loop(self):
        """Maintain persistent HTTP/3 connection (runs in background)"""
        import ssl
        http3_host = os.getenv("HTTP3_HOST", "localhost")
        http3_port = int(os.getenv("HTTP3_PORT", "4434"))
        
        # FAIR CONFIG: Aligned with MQTT/AMQP/gRPC/QUIC/DDS for unbiased comparison
        config = QuicConfiguration(
            is_client=True, 
            alpn_protocols=H3_ALPN,  # HTTP/3 ALPN
            verify_mode=ssl.CERT_NONE,
            max_stream_data=HTTP3_MAX_STREAM_DATA_BYTES,  # 16 KB per stream
            max_data=HTTP3_MAX_STREAM_DATA_BYTES * 4,      # 64 KB total connection
            # FAIR CONFIG: Timeout 600s for very_poor network scenarios
            idle_timeout=600.0  # 10 minutes
        )
        
        print(f"[HTTP/3] Client {self.client_id} connecting to {http3_host}:{http3_port}...")
        print(f"[HTTP/3] Configuration: verify_mode=CERT_NONE, idle_timeout=600s")
        
        # Create protocol factory that sets client reference
        def create_protocol(*args, **kwargs):
            protocol = UnifiedClientHTTP3Protocol(*args, **kwargs)
            protocol.client = self
            print(f"[HTTP/3] Client {self.client_id} created protocol instance")
            return protocol
        
        # Retry connection with exponential backoff
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"[HTTP/3] Client {self.client_id} attempt {attempt + 1}/{max_retries} - calling connect()...")
                async with connect(
                    http3_host,
                    http3_port,
                    configuration=config,
                    create_protocol=create_protocol
                ) as protocol:
                    self.http3_protocol = protocol
                    print(f"[HTTP/3] ✓ Client {self.client_id} established persistent connection")
                    try:
                        register_payload = json.dumps({
                            "type": "register",
                            "client_id": self.client_id,
                        })
                        await self._do_http3_send(register_payload)
                        print(f"[HTTP/3] Client {self.client_id} registered over persistent connection")
                    except Exception as register_error:
                        print(f"[HTTP/3] Client {self.client_id} registration over HTTP/3 failed: {register_error}")
                    
                    # Keep connection alive indefinitely
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        print(f"[HTTP/3] Client {self.client_id} connection cancelled")
                    break  # Connection successful, exit retry loop
                    
            except (ConnectionError, OSError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    print(f"[HTTP/3] ✗ Client {self.client_id} connection attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {e}")
                    print(f"[HTTP/3] Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"[HTTP/3] ✗✗✗ Client {self.client_id} connection FAILED after {max_retries} attempts: {type(e).__name__}: {e}")
                    self.http3_protocol = None
            except Exception as e:
                print(f"[HTTP/3] ✗ Client {self.client_id} unexpected connection error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                self.http3_protocol = None
                break
    
    async def _handle_http3_message_async(self, message: dict):
        """Handle HTTP/3 message received from server asynchronously"""
        msg_type = message.get('type')
        print(f"[HTTP/3] Client {self.client_id} received message type: {msg_type}")
        
        if msg_type == 'global_model':
            self.on_global_model_received_http3(message)
        elif msg_type == 'start_training':
            self.on_start_training_http3(message)
        elif msg_type == 'start_evaluation':
            self.on_start_evaluation_http3(message)
    
    def on_global_model_received_http3(self, message):
        """Handle global model received via HTTP/3"""
        try:
            round_num = message['round']
            print(f"[HTTP/3] Client {self.client_id} received global model for round {round_num}")
            
            # Deserialize weights
            if 'quantized_data' in message and self.quantizer is not None:
                raw = message['quantized_data']
                if isinstance(raw, str):
                    raw = base64.b64decode(raw.encode('utf-8'))
                compressed_data = pickle.loads(raw) if isinstance(raw, bytes) else raw
                weights = self.quantizer.decompress(compressed_data)
            else:
                weights = self.deserialize_weights(message['weights'])
            
            self._global_model_receive_t0 = time.time()
            self._apply_global_model_weights(
                round_num=round_num,
                weights=weights,
                model_config=message.get('model_config'),
                source="HTTP/3",
                server_sent_unix=message.get('server_sent_unix'),
            )
            
            # Set event to signal model is ready
            if hasattr(self, 'model_received'):
                self.model_received.set()
        except Exception as e:
            print(f"[HTTP/3] Client {self.client_id} error handling global model: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_training_http3(self, message):
        """Handle start training signal via HTTP/3"""
        try:
            print(f"[HTTP/3] Client {self.client_id} starting training for round {message.get('round', self.current_round + 1)}")
            
            log_received_packet(
                packet_size=len(json.dumps(message)),
                peer="server",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info="start_training"
            )
            
            # Call the standard training handler
            self.handle_start_training(json.dumps(message).encode())
        except Exception as e:
            print(f"[HTTP/3] Client {self.client_id} error handling start_training: {e}")
            import traceback
            traceback.print_exc()
    
    def on_start_evaluation_http3(self, message):
        """Handle start evaluation signal via HTTP/3"""
        try:
            round_num = message.get('round', self.current_round)
            print(f"[HTTP/3] Client {self.client_id} starting evaluation for round {round_num}")
            
            log_received_packet(
                packet_size=len(json.dumps(message)),
                peer="server",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info="start_evaluation"
            )
            
            # Call the standard evaluation handler
            self.handle_start_evaluation(json.dumps(message).encode())
        except Exception as e:
            print(f"[HTTP/3] Client {self.client_id} error handling evaluation signal: {e}")
            import traceback
            traceback.print_exc()
    
    def _send_via_http3(self, message: dict):
        """Send model update via HTTP/3 using persistent connection"""
        if not HTTP3_AVAILABLE or asyncio is None or connect is None:
            raise ImportError("HTTP/3 module not available")
        
        try:
            # Check if event loop is available and running
            if self.http3_loop is None:
                raise ConnectionError("HTTP/3 event loop not available")
            
            if self.http3_loop.is_closed():
                raise ConnectionError("HTTP/3 event loop is closed")
            
            # Add 'type' field for server to identify message type
            http3_message = {**message, 'type': 'model_update'}
            
            payload = json.dumps(http3_message)
            if len(payload.encode('utf-8')) > HTTP3_MAX_STREAM_DATA_BYTES:
                self._send_chunked_update_via_http3(message)
                return
            payload_size_mb = len(payload) / (1024 * 1024)
            print(f"Client {self.client_id} sending via HTTP/3 - size: {payload_size_mb:.2f} MB")
            
            self._send_http3_payload(payload, timeout=20)
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info="model_update"
            )
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via HTTP/3")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via HTTP/3: {e}")
            raise
    
    def _send_metrics_via_http3(self, message: dict):
        """Send metrics via HTTP/3 using persistent connection"""
        if not HTTP3_AVAILABLE or asyncio is None or connect is None:
            raise ImportError("HTTP/3 module not available")
        
        try:
            # Check if event loop is available and running
            if self.http3_loop is None:
                raise ConnectionError("HTTP/3 event loop not available")
            
            if self.http3_loop.is_closed():
                raise ConnectionError("HTTP/3 event loop is closed")
            
            # Add 'type' field for server to identify message type
            http3_message = {**message, 'type': 'metrics'}
            
            payload = json.dumps(http3_message)
            
            self._send_http3_payload(payload, timeout=20)
            
            log_sent_packet(
                packet_size=len(payload),
                peer="server",
                protocol="HTTP/3",
                round=self.current_round,
                extra_info="metrics"
            )
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via HTTP/3: {e}")
            raise

    def _restart_http3_connection(self):
        """Force-close a stale HTTP/3 connection so the next send reconnects cleanly."""
        if self.http3_connection_task is not None and not self.http3_connection_task.done():
            self.http3_connection_task.cancel()
            try:
                self.http3_connection_task.result(timeout=5)
            except Exception:
                pass
        self.http3_protocol = None
        self.http3_connection_task = None

    def _send_http3_payload(self, payload: str, timeout: float = 20.0):
        """Send a single HTTP/3 request and wait for its response."""
        last_error = None
        for attempt in range(2):
            try:
                self._ensure_protocol_connection_sync('http3')
                with self.http3_send_lock:
                    future = asyncio.run_coroutine_threadsafe(
                        self._do_http3_send(payload),
                        self.http3_loop
                    )
                    return future.result(timeout=timeout)
            except Exception as e:
                last_error = e
                print(f"[HTTP/3] Client {self.client_id} send attempt {attempt + 1}/2 failed: {e}")
                self._restart_http3_connection()
        raise last_error
    
    async def _do_http3_send(self, payload: str):
        """Actually send data via HTTP/3 (runs in HTTP/3 thread's event loop)"""
        if self.http3_protocol is None:
            raise ConnectionError("HTTP/3 protocol not connected")
        
        # Ensure HTTP connection is initialized
        if self.http3_protocol._http is None:
            self.http3_protocol._http = H3Connection(self.http3_protocol._quic)
        
        # Get next available stream ID
        stream_id = self.http3_protocol._quic.get_next_available_stream_id(is_unidirectional=False)
        response_waiter = self.http3_protocol.create_response_waiter(stream_id)
        
        # Prepare JSON payload
        payload_bytes = payload.encode('utf-8')
        
        # Send HTTP/3 request
        headers = [
            (b":method", b"POST"),
            (b":path", b"/fl/message"),
            (b":scheme", b"https"),
            (b":authority", os.getenv('HTTP3_HOST', 'localhost').encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload_bytes)).encode()),
        ]
        self.http3_protocol._http.send_headers(stream_id=stream_id, headers=headers)
        self.http3_protocol._http.send_data(stream_id=stream_id, data=payload_bytes, end_stream=True)
        self.http3_protocol.transmit()
        await asyncio.sleep(0.05)
        self.http3_protocol.transmit()

        try:
            return await asyncio.wait_for(response_waiter, timeout=10.0)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"HTTP/3 response timeout on stream {stream_id}") from e
    
    def _send_via_dds(self, message: dict):
        """Send model update via DDS"""
        if not DDS_AVAILABLE or not self.dds_update_writer:
            raise NotImplementedError("DDS not available - triggering fallback")
        
        try:
            weights_bytes = self._dds_serialized_weights_bytes(message)
            metrics = message.get('metrics', {})
            dds_msg = ModelUpdate(
                client_id=self.client_id,
                round=message['round'],
                weights=list(weights_bytes),
                num_samples=message.get('num_samples', 0),
                loss=float(metrics.get('loss', message.get('loss', 0.0))),
                accuracy=float(metrics.get('accuracy', message.get('accuracy', 0.0)))
            )
            self.dds_update_writer.write(dds_msg)
            
            log_sent_packet(
                packet_size=len(weights_bytes),
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="model_update"
            )
            
            print(f"Client {self.client_id} sent model update for round {self.current_round} via DDS")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending via DDS: {e}")
            raise
    
    def _send_metrics_via_dds(self, message: dict):
        """Send metrics via DDS"""
        if not DDS_AVAILABLE or not self.dds_metrics_writer:
            raise NotImplementedError("DDS not available - triggering fallback")
        
        try:
            # Create DDS metrics message
            dds_msg = EvaluationMetrics(
                client_id=self.client_id,
                round=message['round'],
                num_samples=message.get('num_samples', 0),
                loss=message.get('loss', message.get('metrics', {}).get('loss', 0.0)),
                accuracy=message.get('accuracy', message.get('metrics', {}).get('accuracy', 0.0)),
                client_converged=float(
                    message.get('client_converged', message.get('metrics', {}).get('client_converged', 0.0))
                ),
                battery_soc=float(message.get('battery_soc', message.get('metrics', {}).get('battery_soc', 1.0))),
                training_time_sec=float(
                    message.get('training_time_sec', message.get('metrics', {}).get('training_time_sec', 0.0))
                ),
                round_time_sec=float(
                    message.get('round_time_sec', message.get('metrics', {}).get('round_time_sec', 0.0))
                ),
                uplink_model_comm_sec=float(
                    message.get(
                        'uplink_model_comm_sec',
                        message.get('metrics', {}).get('uplink_model_comm_sec', 0.0),
                    )
                ),
            )
            
            # Write to DDS
            self.dds_metrics_writer.write(dds_msg)
            
            # Log the packet
            log_sent_packet(
                packet_size=len(str(dds_msg)),
                peer="server",
                protocol="DDS",
                round=self.current_round,
                extra_info="metrics"
            )
            
            print(f"Client {self.client_id} sent metrics for round {self.current_round} via DDS")
        except Exception as e:
            print(f"Client {self.client_id} ERROR sending metrics via DDS: {e}")
            raise
    
    def check_global_model_chunks(self):
        """Check for global model chunks from server (matching standalone DDS)"""
        if not DDS_AVAILABLE or not self.dds_global_model_chunk_reader:
            return None
        
        try:
            # Check for chunked global model
            chunk_samples = self.dds_global_model_chunk_reader.take()
            
            for sample in chunk_samples:
                if not sample or not hasattr(sample, 'round'):
                    continue
                
                round_num = sample.round
                chunk_id = sample.chunk_id
                total_chunks = sample.total_chunks
                
                # Initialize buffers if needed
                if not self.global_model_metadata:
                    self.global_model_metadata = {
                        'round': round_num,
                        'total_chunks': total_chunks,
                        'model_config_json': sample.model_config_json if hasattr(sample, 'model_config_json') else ''
                    }
                
                # Store chunk
                self.global_model_chunks[chunk_id] = sample.payload
                
                chunks_received = len(self.global_model_chunks)
                
                # Check if all chunks received
                if chunks_received == total_chunks:
                    try:
                        # Reassemble chunks in order
                        reassembled_data = []
                        for i in range(total_chunks):
                            if i in self.global_model_chunks:
                                reassembled_data.extend(self.global_model_chunks[i])
                            else:
                                print(f"ERROR: Missing chunk {i}")
                                break
                        
                        # Only process if we have all chunks
                        if len(reassembled_data) > 0:
                            # Deserialize weights
                            weights = pickle.loads(bytes(reassembled_data))
                            
                            # Clear buffers
                            self.global_model_chunks.clear()
                            self.global_model_metadata.clear()
                            
                            return {'weights': weights, 'round': round_num}
                    
                    except Exception as e:
                        print(f"[ERROR] Client {self.client_id}: Exception during model reassembly: {e}")
                        import traceback
                        traceback.print_exc()
                        # Clear buffers on error
                        self.global_model_chunks.clear()
                        self.global_model_metadata.clear()
        
        except Exception as e:
            print(f"Client {self.client_id} ERROR checking global model chunks: {e}")
        
        return None
    
    def start(self):
        """Connect to MQTT broker and keep the unified client alive."""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
                self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=120)
                # FAIR CONFIG: keepalive 600s for very_poor network
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 600)
                print(f"Successfully connected to MQTT broker!\n")
                self.mqtt_client.loop_start()
                try:
                    while not self.shutdown_requested:
                        time.sleep(1)
                finally:
                    self.mqtt_client.loop_stop()
                break
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    print(f"\nFailed to connect to MQTT broker after {max_retries} attempts.")
                    raise


def emotion_dataset_folder_id() -> int:
    """Which Dataset/client_N folder to load (defaults to CLIENT_ID). Set DATASET_CLIENT_ID to override."""
    raw = os.getenv("DATASET_CLIENT_ID", "").strip()
    return int(raw) if raw else CLIENT_ID


def load_emotion_data(client_id: int):
    """
    Load emotion recognition dataset for a specific client
    
    Args:
        client_id: Client identifier (folder Dataset/client_{client_id}/)
        
    Returns:
        Tuple of (train_generator, validation_generator)
    """
    # Detect environment: Docker uses /app prefix, local uses relative path
    if os.path.exists('/app'):
        base_path = '/app/Client/Emotion_Recognition/Dataset'
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        base_path = os.path.join(project_root, 'Client', 'Emotion_Recognition', 'Dataset')
    
    train_path = os.path.join(base_path, f'client_{client_id}', 'train')
    validation_path = os.path.join(base_path, f'client_{client_id}', 'validation')
    
    print(f"Dataset base path: {base_path}")
    print(f"Train path: {train_path}")
    print(f"Validation path: {validation_path}")
    
    # Initialize image data generator with rescaling
    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    # Load training and validation data
    train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=DEFAULT_DATA_BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_data_gen.flow_from_directory(
        validation_path,
        target_size=(48, 48),
        batch_size=DEFAULT_DATA_BATCH_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"[Dataset] Train samples: {train_generator.samples}")
    print(f"[Dataset] Validation samples: {validation_generator.samples}")
    print(f"[Dataset] Classes: {train_generator.num_classes}")
    
    return train_generator, validation_generator


def main():
    """Main function"""
    # Prevent duplicate client instance with same client_id in the same container/host.
    # Use project-local lock dir to avoid /tmp permission issues (e.g. file owned by another user).
    _lock_dir = os.path.join(project_root, ".locks")
    os.makedirs(_lock_dir, exist_ok=True)
    lock_path = os.path.join(_lock_dir, f"unified_emotion_client_{CLIENT_ID}.lock")
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[Startup] Another unified client instance is already running for client_id={CLIENT_ID}. Exiting.")
        return

    print(f"Unified FL Client - Emotion Recognition (Client {CLIENT_ID})")
    
    # Load real emotion recognition dataset
    print(f"\n{'='*70}")
    print("LOADING EMOTION RECOGNITION DATASET")
    print(f"{'='*70}")
    
    data_folder_id = emotion_dataset_folder_id()
    try:
        train_generator, validation_generator = load_emotion_data(data_folder_id)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print(f"\nPlease ensure dataset exists at:")
        print(f"  Dataset/client_{data_folder_id}/train/")
        print(f"  Dataset/client_{data_folder_id}/validation/")
        return
    
    # Create client
    client = UnifiedFLClient_Emotion(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    # Start FL
    print(f"\n{'='*60}")
    print(f"Starting Unified FL Client {CLIENT_ID} with RL Protocol Selection")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"{'='*60}\n")
    
    try:
        client.start()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} shutting down...")
        client.mqtt_client.disconnect()


if __name__ == "__main__":
    main()
