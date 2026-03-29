import numpy as np
import os
import sys
import logging
import json
import pickle
import base64
import time
import random
import threading
import queue
import paho.mqtt.client as mqtt

# GPU Configuration - Must be done BEFORE TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
_xla_flags = os.environ.get("XLA_FLAGS", "").strip()
if _xla_flags:
    sanitized_flags = [f for f in _xla_flags.split() if f != "--xla_gpu_enable_command_buffer="]
    if sanitized_flags:
        os.environ["XLA_FLAGS"] = " ".join(sanitized_flags)
    else:
        os.environ.pop("XLA_FLAGS", None)
# Get GPU device ID from environment variable (set by docker for multi-GPU isolation)
# Fallback strategy: GPU_DEVICE_ID -> (CLIENT_ID - 1) -> "0"
# This ensures different clients use different GPUs in multi-GPU setups
client_id_env = os.environ.get("CLIENT_ID", "0")
try:
    default_gpu = str(0)  # Client 1->GPU 0, Client 2->GPU 1, etc.
except (ValueError, TypeError):
    default_gpu = "0"
gpu_device = os.environ.get("GPU_DEVICE_ID", default_gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device  # Isolate to specific GPU
print(f"GPU Configuration: CLIENT_ID={client_id_env}, GPU_DEVICE_ID={gpu_device}, CUDA_VISIBLE_DEVICES={gpu_device}")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow gradual GPU memory growth
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # GPU thread mode

# Disable Grappler layout optimizer to avoid NCHW transpose errors in logs
os.environ["TF_ENABLE_LAYOUT_OPTIMIZER"] = "0"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Detect Docker environment and set project root accordingly
if os.path.exists('/app'):
    # Likely running in Docker, code is under /app
    project_root = '/app'
else:
    # Local development: go up two levels from this file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# packet_logger lives in scripts/utilities (Docker: /app/scripts/utilities, local: project_root/scripts/utilities)
_utilities_path = os.path.join(project_root, 'scripts', 'utilities')
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)

from packet_logger import log_sent_packet, log_received_packet, init_db, get_round_bytes_sent_received
# Make TensorFlow logs less verbose
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Ensure Keras uses channels_last image data format
tf.keras.backend.set_image_data_format('channels_last')

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
        
        # Limit GPU memory per process so multiple clients fit (e.g. 2 clients on one GPU)
        memory_limit_mb = int(os.environ.get("TF_GPU_MEMORY_LIMIT_MB", "4000"))
        for gpu in gpus:
            try:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
            except RuntimeError:
                pass  # GPU already configured
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPUs found. Running on CPU.")

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig
try:
    from pruning_client import ModelPruning, PruningConfig
    PRUNING_AVAILABLE = True
except Exception:
    ModelPruning = None
    PruningConfig = None
    PRUNING_AVAILABLE = False

# Battery model (shared with unified client)
_client_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _client_dir not in sys.path:
    sys.path.insert(0, _client_dir)
from battery_model import BatteryModel

# MQTT Configuration
# Auto-detect environment: Docker (/app exists) or local
MQTT_BROKER = os.getenv("MQTT_BROKER", 'mqtt-broker' if os.path.exists('/app') else 'localhost')
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))  # MQTT broker port
# Keepalive (seconds). Long value avoids rc=16 (keepalive timeout) in very_poor / diagnostic pipeline. Max 65535.
_def_keepalive = 3600 if os.getenv("FL_DIAGNOSTIC_PIPELINE") == "1" else 600
MQTT_KEEPALIVE_SEC = min(65535, max(10, int(os.getenv("MQTT_KEEPALIVE_SEC", str(_def_keepalive)))))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))  # Can be set via environment variable
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
CONVERGENCE_THRESHOLD = float(os.getenv("CONVERGENCE_THRESHOLD", "0.001"))
CONVERGENCE_PATIENCE = int(os.getenv("CONVERGENCE_PATIENCE", "2"))
MIN_ROUNDS = int(os.getenv("MIN_ROUNDS", "3"))
DEFAULT_DATA_BATCH_SIZE = int(os.getenv("DEFAULT_DATA_BATCH_SIZE", "16"))

# Controls whether this client should signal/exit on local convergence.
# When false, clients keep training until the server indicates completion.
from fl_termination_env import stop_on_client_convergence

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"
TOPIC_TRAINING_COMPLETE = "fl/training_complete"

MQTT_MAX_PAYLOAD_BYTES = 128 * 1024
MQTT_CHUNK_PAYLOAD_BYTES = 96 * 1024

class FederatedLearningClient:
    def __init__(self, client_id, num_clients, train_generator=None, validation_generator=None):
        self.client_id = client_id
        self.num_clients = num_clients
        self.current_round = 0
        # Default batch size adjusted for separate GPUs
        self.training_config = {"batch_size": 16, "local_epochs": 20}
        # Deduplication tracking
        self.last_global_round = -1
        self.last_training_round = -1
        self.evaluated_rounds = set()
        # Client-side convergence
        self.best_loss = float('inf')
        self.rounds_without_improvement = 0
        self.has_converged = False
        
        # Store data generators
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        
        # Model will be initialized from server config
        self.model = None
        self._pending_start_training_round = None  # set if start_training arrives before model ready
        self._model_lock = threading.Lock()
        self._training_lock = threading.Lock()
        self._training_thread = None
        self.shutdown_requested = False

        # Initialize packet logger database
        init_db()
        # Battery/energy model (same as unified use case)
        self.battery_model = BatteryModel(protocol="mqtt")
        self._last_round_time_sec = 0.0  # training + communication time for last round

        # Initialize quantization compression (default: disabled unless explicitly enabled)
        uq_env = os.getenv("USE_QUANTIZATION", "false")
        use_quantization = uq_env.lower() in ("true", "1", "yes", "y")
        if use_quantization:
            self.quantizer = Quantization(QuantizationConfig())
            print(f"Client {self.client_id}: Quantization enabled")
        else:
            self.quantizer = None
            print(f"Client {self.client_id}: Quantization disabled")

        # Initialize pruning (default: disabled unless explicitly enabled)
        up_env = os.getenv("USE_PRUNING", "false")
        use_pruning = up_env.lower() in ("true", "1", "yes", "y")
        if use_pruning and PRUNING_AVAILABLE and ModelPruning is not None:
            self.pruner = ModelPruning(PruningConfig())
            print(f"Client {self.client_id}: Pruning enabled")
        else:
            self.pruner = None
            if use_pruning and not PRUNING_AVAILABLE:
                print(f"Client {self.client_id}: Pruning requested but pruning module not available")
            else:
                print(f"Client {self.client_id}: Pruning disabled")
        
        # Unique MQTT client_id per process to avoid broker "already connected" (rc=7) reconnect loop
        _mqtt_client_id = f"fl_client_{client_id}_{os.getpid()}"
        self.mqtt_client = mqtt.Client(client_id=_mqtt_client_id, protocol=mqtt.MQTTv311)
        # Long keepalive to avoid rc=16 (keepalive timeout) in very_poor / diagnostic pipeline
        self.mqtt_client.keepalive = MQTT_KEEPALIVE_SEC
        self.mqtt_client.max_inflight_messages_set(20)
        # FAIR CONFIG: Limited queue to 1000 messages (aligned with AMQP/gRPC)
        self.mqtt_client.max_queued_messages_set(1000)
        # Realistic max payload: MQTT 128 KB
        self.mqtt_client._max_packet_size = MQTT_MAX_PAYLOAD_BYTES
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        self._global_model_chunk_buffers = {}
        self._global_model_queue = queue.Queue()
        self._global_model_worker = threading.Thread(target=self._global_model_worker_loop, daemon=True)
        self._global_model_worker.start()
        
    
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

    def _global_model_worker_loop(self):
        """Serialize global model processing to avoid multiple concurrent TF graphs."""
        while True:
            payload = self._global_model_queue.get()
            if payload is None:
                break
            try:
                self._handle_global_model_thread(payload)
            except Exception as e:
                print(f"Client {self.client_id} global model worker error: {e}")
            finally:
                self._global_model_queue.task_done()

    def _launch_training_for_round(self, round_num):
        """Start local training in a background thread so MQTT loop remains responsive."""
        with self._training_lock:
            if self._training_thread is not None and self._training_thread.is_alive():
                print(f"Client {self.client_id} training already in progress; ignoring start for round {round_num}")
                return

            self.current_round = round_num
            self.last_training_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num}...")

            self._training_thread = threading.Thread(
                target=self._train_local_model_safe,
                args=(round_num,),
                daemon=True,
            )
            self._training_thread.start()

    def _train_local_model_safe(self, round_num):
        try:
            self.train_local_model()
        except Exception as e:
            print(f"Client {self.client_id} training thread error on round {round_num}: {e}")
            import traceback
            traceback.print_exc()
    
    def build_model_from_config(self, model_config):
        """Build CNN model from server-provided configuration"""
        input_shape = model_config.get('input_shape', (48, 48, 1))
        num_classes = model_config.get('num_classes', 7)
        layers = model_config.get('layers', [
            {'type': 'conv', 'filters': 64, 'kernel': 3, 'activation': 'relu'},
            {'type': 'maxpool', 'pool_size': 2},
            {'type': 'conv', 'filters': 128, 'kernel': 3, 'activation': 'relu'},
            {'type': 'maxpool', 'pool_size': 2},
            {'type': 'conv', 'filters': 256, 'kernel': 3, 'activation': 'relu'},
            {'type': 'maxpool', 'pool_size': 2},
            {'type': 'flatten'},
            {'type': 'dense', 'units': 256, 'activation': 'relu'},
            {'type': 'dropout', 'rate': 0.5},
            {'type': 'dense', 'units': num_classes, 'activation': 'softmax'}
        ])
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))
        
        for layer in layers:
            if layer['type'] == 'conv':
                model.add(tf.keras.layers.Conv2D(
                    layer['filters'], 
                    layer['kernel'], 
                    activation=layer['activation'],
                    padding='same'
                ))
            elif layer['type'] == 'maxpool':
                model.add(tf.keras.layers.MaxPooling2D(layer['pool_size']))
            elif layer['type'] == 'flatten':
                model.add(tf.keras.layers.Flatten())
            elif layer['type'] == 'dense':
                model.add(tf.keras.layers.Dense(layer['units'], activation=layer['activation']))
            elif layer['type'] == 'dropout':
                model.add(tf.keras.layers.Dropout(layer['rate']))
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker"""
        if rc == 0:
            print(f"Client {self.client_id} connected to MQTT broker")
            # Subscribe to topics - use QoS 0 for large model messages
            result1, mid1 = self.mqtt_client.subscribe(TOPIC_GLOBAL_MODEL, qos=1)
            print(f"  Subscribed to {TOPIC_GLOBAL_MODEL} (QoS 1) - Result: {result1}")
            
            result2, mid2 = self.mqtt_client.subscribe(TOPIC_TRAINING_CONFIG, qos=1)
            print(f"  Subscribed to {TOPIC_TRAINING_CONFIG} (QoS 1) - Result: {result2}")
            
            result3, mid3 = self.mqtt_client.subscribe(TOPIC_START_TRAINING, qos=1)
            print(f"  Subscribed to {TOPIC_START_TRAINING} (QoS 1) - Result: {result3}")
            
            result4, mid4 = self.mqtt_client.subscribe(TOPIC_START_EVALUATION, qos=1)
            print(f"  Subscribed to {TOPIC_START_EVALUATION} (QoS 1) - Result: {result4}")
            
            result5, mid5 = self.mqtt_client.subscribe(TOPIC_TRAINING_COMPLETE, qos=1)
            print(f"  Subscribed to {TOPIC_TRAINING_COMPLETE} (QoS 1) - Result: {result5}")
            
            # Wait longer for subscriptions to be fully processed
            print(f"  Waiting for subscriptions to be processed...")
            time.sleep(2)
            
            # Send registration message
            self.mqtt_client.publish("fl/client_register", 
                                    json.dumps({"client_id": self.client_id}), qos=1)
            log_sent_packet(
                packet_size=len(json.dumps({"client_id": self.client_id})),
                peer="fl/client_register",  # or client_id/server_id as appropriate
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info="any additional info"
            )
            
            print(f"  Registration message sent")
        else:
            print(f"Client {self.client_id} failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            print(f"Client {self.client_id} received message on topic: {msg.topic}, size: {len(msg.payload)} bytes")
            log_received_packet(
                packet_size=len(msg.payload),
                peer=msg.topic,
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info=msg.topic
            )

            if msg.topic == TOPIC_GLOBAL_MODEL:
                # Queue payload for a single worker thread to avoid concurrent graph creation/OOM
                payload_copy = bytes(msg.payload)
                self._global_model_queue.put(payload_copy)
            elif msg.topic == TOPIC_TRAINING_CONFIG:
                self.handle_training_config(msg.payload)
            elif msg.topic == TOPIC_START_TRAINING:
                self.handle_start_training(msg.payload)
            elif msg.topic == TOPIC_START_EVALUATION:
                self.handle_start_evaluation(msg.payload)
            elif msg.topic == 'fl/training_complete':
                self.handle_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling message on topic {msg.topic}: {e}")
            import traceback
            traceback.print_exc()
    
    def on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        if rc == 0 and self.shutdown_requested:
            print(f"Client {self.client_id} clean disconnect from broker")
            print(f"Client {self.client_id} exiting...")
            self.mqtt_client.loop_stop()
            return
        else:
            # rc=16: keepalive timeout (broker didn't see traffic in time). rc=7: duplicate client ID / connection lost.
            print(f"Client {self.client_id} unexpected disconnect, return code {rc}")
            if rc == 7:
                self._reconnect_after_rc7 = getattr(self, "_reconnect_after_rc7", 0) + 1
                if self._reconnect_after_rc7 <= 2:
                    print(f"Client {self.client_id} will reconnect once after short delay...")
                self.mqtt_client.loop_stop()
            else:
                # rc=16 or other: let paho auto-reconnect (loop_forever with retry_first_connection=True)
                print(f"Client {self.client_id} attempting to reconnect...")

    def _assemble_global_model_chunk(self, data):
        """Buffer and reassemble chunked global-model messages."""
        try:
            chunk_index = int(data.get("chunk_index", 0))
            total_chunks = int(data.get("total_chunks", 1))
            round_num = int(data["round"])
            payload_key = data["payload_key"]
            payload_chunk = data.get("payload_chunk", "")
        except Exception as e:
            print(f"Client {self.client_id} invalid global model chunk metadata: {e}")
            return None

        if total_chunks <= 1:
            assembled = dict(data)
            assembled[payload_key] = payload_chunk
            assembled.pop("payload_chunk", None)
            assembled.pop("payload_key", None)
            assembled["message_type"] = "global_model"
            assembled["type"] = "global_model"
            return assembled

        chunk_key = (round_num, payload_key)
        entry = self._global_model_chunk_buffers.setdefault(
            chunk_key,
            {"chunks": {}, "total_chunks": total_chunks, "base": None, "updated_at": time.time()}
        )

        if entry["base"] is None:
            entry["base"] = {
                "round": round_num,
                "message_type": "global_model",
                "type": "global_model",
                "model_config": data.get("model_config")
            }
            if "training_config" in data:
                entry["base"]["training_config"] = data.get("training_config")

        if data.get("model_config") and not entry["base"].get("model_config"):
            entry["base"]["model_config"] = data.get("model_config")
        if "training_config" in data and not entry["base"].get("training_config"):
            entry["base"]["training_config"] = data.get("training_config")

        if chunk_index not in entry["chunks"]:
            entry["chunks"][chunk_index] = payload_chunk
        entry["updated_at"] = time.time()

        if len(entry["chunks"]) < total_chunks:
            return None

        assembled_payload = "".join(entry["chunks"].get(i, "") for i in range(total_chunks))
        assembled = dict(entry["base"])
        assembled[payload_key] = assembled_payload
        self._global_model_chunk_buffers.pop(chunk_key, None)
        print(
            f"Client {self.client_id} reassembled global model for round {round_num} "
            f"from {total_chunks} chunks"
        )
        return assembled

    def _chunk_model_update_messages(self, update_message):
        payload_key = "compressed_data" if "compressed_data" in update_message else "weights"
        payload_text = update_message[payload_key]
        if not isinstance(payload_text, str):
            raise TypeError(f"Expected string payload for {payload_key}, got {type(payload_text).__name__}")

        chunks = [
            payload_text[i:i + MQTT_CHUNK_PAYLOAD_BYTES]
            for i in range(0, len(payload_text), MQTT_CHUNK_PAYLOAD_BYTES)
        ] or [payload_text]

        chunk_messages = []
        total_chunks = len(chunks)
        for chunk_index, chunk_data in enumerate(chunks):
            chunk_msg = {
                "message_type": "model_update_chunk",
                "type": "update_chunk",
                "protocol": "mqtt",
                "client_id": update_message["client_id"],
                "round": update_message["round"],
                "payload_key": payload_key,
                "payload_chunk": chunk_data,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "num_samples": update_message["num_samples"] if chunk_index == 0 else 0,
                "metrics": update_message["metrics"] if chunk_index == 0 else {},
            }
            if chunk_index == 0 and "diagnostic_send_start_ts" in update_message:
                chunk_msg["diagnostic_send_start_ts"] = update_message["diagnostic_send_start_ts"]
            chunk_messages.append(chunk_msg)
        return chunk_messages

    def _publish_update_with_chunking(self, update_message):
        payload = json.dumps(update_message)
        payload_bytes = len(payload.encode("utf-8"))

        if payload_bytes <= MQTT_MAX_PAYLOAD_BYTES:
            result = self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_NO_CONN:
                raise Exception("MQTT not connected")
            result.wait_for_publish(timeout=30)
            if not result.is_published():
                raise TimeoutError("Timed out waiting for MQTT publish acknowledgment")
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(f"MQTT publish failed with rc={result.rc}")
            log_sent_packet(
                packet_size=len(payload),
                peer=TOPIC_CLIENT_UPDATE,
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info="Model update"
            )
            return payload_bytes

        chunk_messages = self._chunk_model_update_messages(update_message)
        print(
            f"Client {self.client_id} chunking model update: {payload_bytes} bytes total, "
            f"{len(chunk_messages)} chunks"
        )
        bytes_sent = 0
        for chunk_msg in chunk_messages:
            chunk_payload = json.dumps(chunk_msg)
            chunk_size = len(chunk_payload.encode("utf-8"))
            if chunk_size > MQTT_MAX_PAYLOAD_BYTES:
                raise ValueError(
                    f"MQTT chunk exceeds 128KB limit: {chunk_size} bytes "
                    f"(chunk {chunk_msg['chunk_index'] + 1}/{chunk_msg['total_chunks']})"
                )
            result = self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, chunk_payload, qos=1)
            if result.rc == mqtt.MQTT_ERR_NO_CONN:
                raise Exception("MQTT not connected")
            result.wait_for_publish(timeout=30)
            if not result.is_published():
                raise TimeoutError(
                    f"Timed out waiting for MQTT chunk publish acknowledgment "
                    f"for chunk {chunk_msg['chunk_index'] + 1}/{chunk_msg['total_chunks']}"
                )
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                raise Exception(
                    f"MQTT chunk publish failed with rc={result.rc} "
                    f"for chunk {chunk_msg['chunk_index'] + 1}/{chunk_msg['total_chunks']}"
                )
            bytes_sent += chunk_size
            log_sent_packet(
                packet_size=chunk_size,
                peer=TOPIC_CLIENT_UPDATE,
                protocol="MQTT",
                round=self.current_round if hasattr(self, 'current_round') else None,
                extra_info=f"Model update chunk {chunk_msg['chunk_index'] + 1}/{chunk_msg['total_chunks']}"
            )
        return bytes_sent
    
    def handle_training_complete(self):
        """Handle training completion signal from server"""
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nDisconnecting from server...")
        self.shutdown_requested = True
        time.sleep(1)  # Brief delay before disconnect
        self.mqtt_client.disconnect()
        print(f"Client {self.client_id} disconnected successfully.")
    
    def _handle_global_model_thread(self, payload):
        """Run handle_global_model in a thread; then trigger training if start_training arrived early."""
        self.handle_global_model(payload)
        with self._model_lock:
            if self._pending_start_training_round is not None:
                r = self._pending_start_training_round
                self._pending_start_training_round = None
                if self.model is not None:
                    print(f"Client {self.client_id} model ready; starting pending training for round {r}")
                    self._launch_training_for_round(r)

    def handle_global_model(self, payload):
        """Receive and set global model weights and architecture from server"""
        try:
            data = json.loads(payload.decode())
            message_type = data.get("message_type") or data.get("type")
            if message_type == "global_model_chunk":
                assembled = self._assemble_global_model_chunk(data)
                if assembled is None:
                    return
                data = assembled

            round_num = data['round']
            # Ignore duplicate global model for the same round
            if round_num <= self.last_global_round:
                print(f"Client {self.client_id} ignoring duplicate global model for round {round_num}")
                return
            
            # Decompress/deserialize weights: dequantize full-precision weights for local training.
            if 'quantized_data' in data and self.quantizer is not None:
                compressed_data = data['quantized_data']
                if isinstance(compressed_data, str):
                    try:
                        compressed_data = pickle.loads(base64.b64decode(compressed_data.encode('utf-8')))
                    except Exception as e:
                        print(f"Client {self.client_id} error decoding quantized_data: {e}")
                weights = self.quantizer.decompress(compressed_data)
                if round_num > 0:
                    print(f"Client {self.client_id}: Received global model (dequantized for training)")
            elif 'pruned_data' in data and PRUNING_AVAILABLE and ModelPruning is not None:
                try:
                    compressed_bytes = base64.b64decode(data['pruned_data'].encode('utf-8'))
                    pruning_codec = self.pruner or ModelPruning(PruningConfig())
                    weights = pruning_codec.decompress_pruned_weights(compressed_bytes)
                    if round_num > 0:
                        print(f"Client {self.client_id}: Received and decompressed pruned global model")
                except Exception as e:
                    print(f"Client {self.client_id} error decoding pruned_data: {e}")
                    encoded_weights = data['weights']
                    weights = self.deserialize_weights(encoded_weights)
            else:
                # Standard deserialization
                encoded_weights = data['weights']
                weights = self.deserialize_weights(encoded_weights)
            
            # Initialize model if not yet created (works for any round)
            if self.model is None:
                print(f"Client {self.client_id} initializing model from server (round {round_num})")
                
                model_config = data.get('model_config')
                if model_config:
                    # Build in a local variable so self.model stays None until compile()+set_weights are done.
                    # That avoids start_training racing and calling train_local_model() before compile().
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
                    model.set_weights(weights)
                    # Assign only after compile + set_weights so start_training never sees an unready model
                    self.model = model
                    print(f"Client {self.client_id} built CNN model from server configuration")
                    print(f"  Input shape: {model_config['input_shape']}")
                    print(f"  Output classes: {model_config['num_classes']}")
                    print(f"Client {self.client_id} model initialized with server weights")
                else:
                    raise ValueError("No model configuration received from server!")
                
                self.current_round = 0
                self.last_global_round = round_num
            else:
                # Updated model after aggregation (model already initialized)
                self.model.set_weights(weights)
                self.current_round = round_num
                self.last_global_round = round_num
                print(f"Client {self.client_id} received global model for round {round_num}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR in handle_global_model: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    def handle_start_training(self, payload):
        """Start local training when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']

        # If model not ready yet (global model still processing in thread), queue this round
        if self.model is None:
            print(f"Client {self.client_id} waiting for global model; will start round {round_num} when ready")
            self._pending_start_training_round = round_num
            return

        # Check for duplicate training signals
        if self.last_training_round == round_num:
            print(f"Client {self.client_id} ignoring duplicate start training for round {round_num}")
            return

        self._launch_training_for_round(round_num)
    
    def handle_start_evaluation(self, payload):
        """Start evaluation when server signals"""
        data = json.loads(payload.decode())
        round_num = data['round']
        
        # Check if model is initialized
        if self.model is None:
            print(f"Client {self.client_id} waiting for global model (not yet initialized)...")
            return
        
        # Check for duplicate evaluation signals
        if round_num in self.evaluated_rounds:
            print(f"Client {self.client_id} ignoring duplicate evaluation for round {round_num}")
            return
        
        # Evaluate regardless of round number (generic approach)
        self.current_round = round_num
        print(f"Client {self.client_id} starting evaluation for round {round_num}...")
        self.evaluate_model()
        self.evaluated_rounds.add(round_num)
        print(f"Client {self.client_id} evaluation completed for round {round_num}")
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        self._apply_generator_batch_size(batch_size)
        epochs = self.training_config['local_epochs']
        # Limit steps per epoch for faster smoke tests (configurable via env)
        try:
            steps_per_epoch = int(os.getenv("STEPS_PER_EPOCH", "100"))
            val_steps = int(os.getenv("VAL_STEPS", "25"))
        except Exception:
            steps_per_epoch = 100
            val_steps = 25

        training_start = time.time()
        # Train the model using generator
        history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.validation_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=val_steps,
            verbose=2
        )
        training_time = time.time() - training_start

        # Get updated weights
        updated_weights = self.model.get_weights()
        num_samples = self.train_generator.n  # Total number of training samples

        # Apply pruning before quantization/transmission when enabled
        if self.pruner is not None:
            updated_weights = self.pruner.prune_weights(updated_weights, step=self.current_round)
            if self.current_round == 0 or (self.current_round % 5 == 0):
                stats = self.pruner.get_pruning_statistics(updated_weights)
                print(
                    f"Client {self.client_id}: Pruned weights - "
                    f"Sparsity: {stats['overall_sparsity']:.2%}, "
                    f"Compression: {stats['compression_ratio']:.2f}x"
                )
        
        # Prepare training metrics (for classification)
        metrics = {
            "loss": float(history.history["loss"][-1]),
            "accuracy": float(history.history["accuracy"][-1]),
            "val_loss": float(history.history["val_loss"][-1]),
            "val_accuracy": float(history.history["val_accuracy"][-1])
        }
        
        # Compress weights for transmission:
        # - if quantization enabled -> quantize pruned weights
        # - else if pruning enabled -> sparse-compress pruned weights
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
            # Get compression stats
            stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - "
                  f"Ratio: {stats['compression_ratio']:.2f}x, "
                  f"Size: {stats['compressed_size_mb']:.2f}MB")
            
            # Serialize compressed data to JSON-safe base64 string
            serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
            # Send compressed update
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "compressed_data": serialized,
                "num_samples": num_samples,
                "metrics": metrics
            }
        elif self.pruner is not None:
            try:
                pruned_bytes, _ = self.pruner.compress_pruned_weights(updated_weights)
                pruned_b64 = base64.b64encode(pruned_bytes).decode("utf-8")
                update_message = {
                    "client_id": self.client_id,
                    "round": self.current_round,
                    "pruned_data": pruned_b64,
                    "num_samples": num_samples,
                    "metrics": metrics
                }
            except Exception as e:
                print(f"Client {self.client_id} error compressing pruned weights: {e}")
                update_message = {
                    "client_id": self.client_id,
                    "round": self.current_round,
                    "weights": self.serialize_weights(updated_weights),
                    "num_samples": num_samples,
                    "metrics": metrics
                }
        else:
            # Send model update to server without compression
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "weights": self.serialize_weights(updated_weights),
                "num_samples": num_samples,
                "metrics": metrics
            }
        
        # Diagnostic pipeline: high-precision timing for O_send (serialize + send)
        if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
            send_start_ts = time.time()
            send_start_cpu = time.perf_counter()
            update_message["diagnostic_send_start_ts"] = send_start_ts
        
        # FAIR FIX: Removed random delay - this was causing unfair comparison with other protocols
        # Other protocols don't have random delays, so MQTT shouldn't either

        comm_start = time.time()
        # Serialize and send model update with error handling
        try:
            payload_size_mb = len(json.dumps(update_message).encode('utf-8')) / (1024 * 1024)
            print(f"Client {self.client_id} serialized update size: {payload_size_mb:.2f} MB")
            bytes_sent = self._publish_update_with_chunking(update_message)
            
            if os.environ.get("FL_DIAGNOSTIC_PIPELINE") == "1":
                send_end_cpu = time.perf_counter()
                O_send = send_end_cpu - send_start_cpu
                print(f"FL_DIAG O_send={O_send:.9f} payload_bytes={bytes_sent} send_start_ts={send_start_ts:.9f}")

            print(f"Client {self.client_id} sent model update for round {self.current_round}")
            print(f"Training metrics - Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"Client {self.client_id} ERROR serializing/sending update: {e}")
            import traceback
            traceback.print_exc()
        communication_time = time.time() - comm_start

        # Battery model update (same as unified use case)
        try:
            bytes_sent, bytes_recv = get_round_bytes_sent_received(self.current_round, "MQTT")
        except Exception:
            bytes_sent, bytes_recv = 0, 0
        self.battery_model.update(
            bytes_sent, bytes_recv, training_time, communication_time
        )
        self._last_round_time_sec = training_time + communication_time
    
    def _update_local_convergence(self, loss: float):
        """Track client-local convergence and disconnect when converged."""
        if self.current_round < MIN_ROUNDS:
            self.best_loss = min(self.best_loss, loss)
            return

        if self.best_loss - loss > CONVERGENCE_THRESHOLD:
            self.best_loss = loss
            self.rounds_without_improvement = 0
        else:
            self.rounds_without_improvement += 1

        if self.rounds_without_improvement >= CONVERGENCE_PATIENCE and not self.has_converged:
            self.has_converged = True
            print(f"Client {self.client_id} reached local convergence at round {self.current_round}")

    def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server"""
        loss, accuracy = self.model.evaluate(
            self.validation_generator,
            verbose=0
        )
        
        num_samples = self.validation_generator.n

        self._update_local_convergence(float(loss))

        metrics_dict = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "battery_soc": float(self.battery_model.battery_soc),
            "round_time_sec": float(self._last_round_time_sec),
            "cumulative_energy_j": float(self.battery_model.cumulative_energy_j),
        }
        if self.has_converged:
            # Avoid sending client_converged=1.0 when fixed-round mode is enabled.
            metrics_dict["client_converged"] = 1.0 if stop_on_client_convergence() else 0.0
        
        metrics_message = {
            "client_id": self.client_id,
            "round": self.current_round,
            "num_samples": num_samples,
            "metrics": metrics_dict
        }
        
        self.mqtt_client.publish(TOPIC_CLIENT_METRICS, json.dumps(metrics_message))
        log_sent_packet(
            packet_size=len(json.dumps(metrics_message)),
            peer=TOPIC_CLIENT_METRICS,  # or client_id/server_id as appropriate
            protocol="MQTT",
            round=self.current_round if hasattr(self, 'current_round') else None,
            extra_info="any additional info"
        )
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        if self.has_converged and stop_on_client_convergence():
            print(f"Client {self.client_id} notifying server of convergence and disconnecting")
            time.sleep(2)
            self.mqtt_client.disconnect()
    
    def start(self):
        """Connect to MQTT broker and keep client alive (aligned with unified behavior)."""
        max_retries = 10
        retry_delay = 2
        self._reconnect_after_rc7 = 0

        for attempt in range(max_retries):
            try:
                print(f"Attempting to connect to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}...")
                self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=120)
                self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=MQTT_KEEPALIVE_SEC)
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
                    print(f"\nFailed to connect to MQTT broker. Ensure Mosquitto is running at {MQTT_BROKER}:{MQTT_PORT}")
                    raise

def load_data(client_id):
    # Detect environment: Docker uses /app prefix, local uses relative path
    if os.path.exists('/app'):
        # Running in Docker container
        base_path = '/app/Client/Emotion_Recognition/Dataset'
    else:
        # Running locally - use path relative to project root
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

    return train_generator, validation_generator

if __name__ == "__main__":
    # Load data
    print(f"Loading dataset for client {CLIENT_ID}...")
    train_generator, validation_generator = load_data(CLIENT_ID)
    print(f"Dataset loaded")
    
    # Create and start client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    print(f"\n{'='*60}")
    print(f"Starting Federated Learning Client {CLIENT_ID}")
    print(f"Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"{'='*60}\n")
    
    try:
        client.start()
    except KeyboardInterrupt:
        print(f"\nClient {CLIENT_ID} shutting down...")
        client.mqtt_client.disconnect()
