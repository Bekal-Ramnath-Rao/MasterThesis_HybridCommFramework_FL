"""
Federated Learning Client for EEG Mental State Recognition using DDS
Supports CNN+BiLSTM+MHA architecture with non-IID data partitioning
"""

import os
import sys
import json
import pickle
import time
import random
import logging
import numpy as np
import tensorflow as tf
from collections import Counter

# Import data partitioner
from data_partitioner import get_client_data, NUM_CLASSES, ID2LBL, LBL2ID

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# Add CycloneDDS DLL path
cyclone_path = r"C:\Masters_Infotech\Semester_5\MT_SW_Addons\vcpkg\buildtrees\cyclonedds\x64-windows-rel\bin"
if cyclone_path not in os.environ.get('PATH', ''):
    os.environ['PATH'] = cyclone_path + os.pathsep + os.environ.get('PATH', '')

from cyclonedds.domain import DomainParticipant

# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig

from cyclonedds.topic import Topic
from cyclonedds.sub import DataReader
from cyclonedds.pub import DataWriter
from cyclonedds.util import duration
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from cyclonedds.core import Qos, Policy
from dataclasses import dataclass
from typing import List

# DDS Configuration
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))

# Training Configuration
AUTOTUNE = tf.data.AUTOTUNE
SMOOTH_EPS = 0.05


# DDS Data Types
@dataclass
class ClientRegistration(IdlStruct):
    client_id: int
    message: str


@dataclass
class TrainingConfig(IdlStruct):
    batch_size: int
    local_epochs: int


@dataclass
class TrainingCommand(IdlStruct):
    round: int
    start_training: bool
    start_evaluation: bool
    training_complete: bool


@dataclass
class GlobalModel(IdlStruct):
    round: int
    weights: sequence[int]
    model_config_json: str = ""


@dataclass
class ModelUpdate(IdlStruct):
    client_id: int
    round: int
    weights: sequence[int]
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


@dataclass
class ServerStatus(IdlStruct):
    current_round: int
    total_rounds: int
    training_started: bool
    training_complete: bool
    registered_clients: int


class FederatedLearningClient:
    def __init__(self, client_id, num_clients):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        
        # Initialize quantization compression
        use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
        if use_quantization:
            self.quantizer = Quantization(QuantizationConfig())
            print(f"Client {self.client_id}: Quantization enabled")
        else:
            self.quantizer = None
            print(f"Client {self.client_id}: Quantization disabled")
        self.x_train = None
        self.y_train = None
        self.current_round = 0
        self.training_config = {
            "batch_size": 256,
            "local_epochs": 5,
            "val_split": 0.10
        }
        self.class_weights = None
        self.running = True
        
        # DDS entities
        self.participant = None
        self.readers = {}
        self.writers = {}
        
        # Load and partition data
        self.prepare_data()
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {len(self.y_train)}")
        print(f"  Input shape: {self.x_train.shape}")
        print(f"  Waiting for initial global model from server...")
    
    def prepare_data(self):
        """Load and partition EEG data for this client using data_partitioner"""
        print(f"\n[Client {self.client_id}] Preparing data...")
        
        # Get data directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "Dataset")
        
        # Use data partitioner to get non-IID data
        self.x_train, self.y_train = get_client_data(
            self.client_id, 
            self.num_clients, 
            data_dir
        )
        
        # Compute class weights for this client's data
        self.class_weights = self.compute_class_weights(self.y_train)
        
        print(f"[Client {self.client_id}] Data preparation complete!")
        print(f"  Class weights: {self.class_weights}")
    
    def compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        cc = Counter(y.tolist())
        total = max(1, sum(cc.values()))
        K = NUM_CLASSES
        return {cls: total / (K * cnt) for cls, cnt in cc.items()}
    
    def make_dataset(self, X, y, batch_size, training=True):
        """Create tf.data.Dataset with augmentations"""
        Xtf = tf.convert_to_tensor(X.astype('float32'))
        ytf = tf.convert_to_tensor(y.astype('int64'))
        
        # Sample weights
        sw = np.array([self.class_weights.get(int(k), 1.0) for k in y], dtype=np.float32)
        sww = tf.convert_to_tensor(sw, dtype=tf.float32)
        
        ds = tf.data.Dataset.from_tensor_slices((Xtf, ytf, sww))
        
        if training:
            ds = ds.shuffle(len(y), seed=42, reshuffle_each_iteration=True)
        
        def _augment(x):
            """Apply data augmentation"""
            # Time shift
            shift = tf.random.uniform([], -16, 17, dtype=tf.int32)
            x = tf.roll(x, shift=shift, axis=0)
            
            # Channel dropout
            if tf.random.uniform(()) < 0.30:
                C = tf.shape(x)[1]
                drop_n = tf.minimum(tf.random.uniform([], 1, 5, dtype=tf.int32), C)
                idx = tf.random.shuffle(tf.range(C))[:drop_n]
                mask = tf.ones([C], x.dtype)
                mask = tf.tensor_scatter_nd_update(
                    mask, tf.reshape(idx, [-1, 1]),
                    tf.zeros([drop_n], x.dtype)
                )
                x = x * mask[tf.newaxis, :]
            
            # Time masking
            if tf.random.uniform(()) < 0.30:
                T = tf.shape(x)[0]
                C = tf.shape(x)[1]
                L = tf.random.uniform([], 8, 33, dtype=tf.int32)
                s = tf.random.uniform([], 0, T - L, dtype=tf.int32)
                mask = tf.concat([
                    tf.ones([s, C], x.dtype),
                    tf.zeros([L, C], x.dtype),
                    tf.ones([T - s - L, C], x.dtype)
                ], axis=0)
                x = x * mask
            
            # Gaussian noise
            x = x + tf.random.normal(tf.shape(x), stddev=0.02, dtype=x.dtype)
            return x
        
        def _map(x, y, w):
            x = tf.cast(x, tf.float32)
            if training:
                x = _augment(x)
            y = tf.one_hot(tf.cast(y, tf.int32), NUM_CLASSES, dtype=x.dtype)
            if training and SMOOTH_EPS > 0:
                y = (1.0 - SMOOTH_EPS) * y + SMOOTH_EPS / NUM_CLASSES
            return x, y, w
        
        ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(AUTOTUNE)
        return ds
    
    def build_eeg_model(self, input_shape, num_classes):
        """Build CNN+BiLSTM+MHA model for EEG classification"""
        from tensorflow.keras import layers, Model
        
        def se_block(x, r=8):
            ch = x.shape[-1]
            s = layers.GlobalAveragePooling1D()(x)
            s = layers.Dense(max(ch // r, 8), activation='relu')(s)
            s = layers.Dense(ch, activation='sigmoid', dtype='float32')(s)
            s = layers.Reshape((1, ch))(s)
            return layers.Multiply()([x, s])
        
        def conv_bn_relu(x, f, k, d=1):
            x = layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            return x
        
        def res_block(x, f, k, d=1):
            sc = x
            y = conv_bn_relu(x, f, k, d)
            y = layers.Conv1D(f, k, padding="same", dilation_rate=d, use_bias=False)(y)
            y = layers.BatchNormalization()(y)
            if sc.shape[-1] != f:
                sc = layers.Conv1D(f, 1, padding="same", use_bias=False)(sc)
                sc = layers.BatchNormalization()(sc)
            y = layers.Add()([y, sc])
            y = layers.ReLU()(y)
            y = se_block(y)
            return y
        
        inp = layers.Input(shape=input_shape)
        
        x = conv_bn_relu(inp, 64, 7, d=1)
        x = res_block(x, 64, 7, d=1)
        x = layers.MaxPooling1D(2)(x)
        
        # Dilated conv stack
        for d in [1, 2, 4]:
            x = res_block(x, 128, 5, d=d)
        
        # Temporal modeling
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.25)
        )(x)
        
        # Self-attention
        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)(x, x)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)
        
        # Pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.35)(x)
        out = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
        
        model = Model(inp, out)
        
        # Compile model
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-3, first_decay_steps=4, 
            t_mul=2.0, m_mul=0.8, alpha=1e-5
        )
        opt = tf.keras.optimizers.AdamW(
            learning_rate=lr_sched, weight_decay=1e-4, global_clipnorm=1.0
        )
        
        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name="acc"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top2")
            ]
        )
        
        return model
    
    def serialize_weights(self, weights):
        """Serialize model weights for DDS transmission"""
        serialized = pickle.dumps(weights)
        return list(serialized)
    
    def deserialize_weights(self, serialized_weights):
        """Deserialize model weights received from DDS"""
        return pickle.loads(bytes(serialized_weights))
    
    def setup_dds(self):
        """Initialize DDS participant, topics, readers, and writers"""
        print(f"Setting up DDS on domain {DDS_DOMAIN_ID}...")
        
        # Create domain participant
        self.participant = DomainParticipant(DDS_DOMAIN_ID)
        
        # Create QoS policy for reliable communication
        reliable_qos = Qos(
            Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
            Policy.History.KeepAll,
            Policy.Durability.TransientLocal
        )
        
        # Create topics
        topic_registration = Topic(self.participant, "ClientRegistration", ClientRegistration)
        topic_config = Topic(self.participant, "TrainingConfig", TrainingConfig)
        topic_command = Topic(self.participant, "TrainingCommand", TrainingCommand)
        topic_global_model = Topic(self.participant, "GlobalModel", GlobalModel)
        topic_model_update = Topic(self.participant, "ModelUpdate", ModelUpdate)
        topic_metrics = Topic(self.participant, "EvaluationMetrics", EvaluationMetrics)
        topic_status = Topic(self.participant, "ServerStatus", ServerStatus)
        
        # Create readers with reliable QoS
        self.readers['config'] = DataReader(self.participant, topic_config, qos=reliable_qos)
        self.readers['command'] = DataReader(self.participant, topic_command, qos=reliable_qos)
        self.readers['global_model'] = DataReader(self.participant, topic_global_model, qos=reliable_qos)
        self.readers['status'] = DataReader(self.participant, topic_status, qos=reliable_qos)
        
        # Create writers with reliable QoS
        self.writers['registration'] = DataWriter(self.participant, topic_registration, qos=reliable_qos)
        self.writers['model_update'] = DataWriter(self.participant, topic_model_update, qos=reliable_qos)
        self.writers['metrics'] = DataWriter(self.participant, topic_metrics, qos=reliable_qos)
        
        print(f"Client {self.client_id} DDS setup complete with RELIABLE QoS")
        time.sleep(2)
        
        # Register with server
        registration = ClientRegistration(
            client_id=self.client_id,
            message=f"Client {self.client_id} ready"
        )
        print(f"[DEBUG] Client {self.client_id} sending registration to server")
        self.writers['registration'].write(registration)
        print(f"Client {self.client_id} registered with server\n")
    
    def run(self):
        """Main client loop"""
        print("="*60)
        print(f"Starting Federated Learning Client {self.client_id}")
        print(f"DDS Domain ID: {DDS_DOMAIN_ID}")
        print("="*60)
        print()
        
        # Setup DDS
        self.setup_dds()
        
        # Get training configuration
        self.get_training_config()
        
        print(f"Client {self.client_id} waiting for training to start...\n")
        
        try:
            while self.running:
                # Check for global model updates
                self.check_global_model()
                
                # Check for training commands
                self.check_commands()
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\nClient {self.client_id} shutting down...")
        finally:
            self.cleanup()
    
    def get_training_config(self):
        """Get training configuration from server"""
        timeout = 10
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            samples = self.readers['config'].take()
            for sample in samples:
                if sample:
                    self.training_config.update({
                        "batch_size": sample.batch_size,
                        "local_epochs": sample.local_epochs
                    })
                    print(f"Client {self.client_id} received config: {self.training_config}")
                    return
            time.sleep(0.5)
        
        print(f"Client {self.client_id} using default config: {self.training_config}")
    
    def check_commands(self):
        """Check for training commands from server"""
        samples = self.readers['command'].take()
        
        for sample in samples:
            if sample:
                print(f"[DEBUG] Client {self.client_id} received command - round={sample.round}, start_training={sample.start_training}, current_round={self.current_round}")
                if sample.training_complete:
                    print(f"\nClient {self.client_id} - Training completed!")
                    self.running = False
                    return
                
                if sample.start_training:
                    if self.current_round == 0 and sample.round == 1:
                        if self.model is None:
                            print(f"Client {self.client_id} waiting for initial model before training...")
                            return
                        self.current_round = sample.round
                        print(f"\nClient {self.client_id} starting training for round {self.current_round} with initial global model...")
                        self.train_local_model()
                    elif sample.round > self.current_round:
                        if self.model is None:
                            print(f"Client {self.client_id} ERROR: Model not initialized!")
                            return
                        self.current_round = sample.round
                        print(f"\nClient {self.client_id} starting training for round {self.current_round}...")
                        self.train_local_model()
    
    def check_global_model(self):
        """Check for global model updates from server"""
        samples = self.readers['global_model'].take()
        
        for sample in samples:
            if sample:
                round_num = sample.round
                weights = self.deserialize_weights(sample.weights)
                
                if round_num == 0:
                    print(f"[DEBUG] Client {self.client_id} received initial model - round={round_num}, has_config={hasattr(sample, 'model_config_json') and bool(sample.model_config_json)}")
                    print(f"Client {self.client_id} received initial global model from server")
                    
                    if hasattr(sample, 'model_config_json') and sample.model_config_json:
                        model_config = json.loads(sample.model_config_json)
                        self.model = self.build_eeg_model(
                            input_shape=tuple(model_config['input_shape']),
                            num_classes=model_config['num_classes']
                        )
                        print(f"Client {self.client_id} built EEG model from server configuration")
                        print(f"  Input shape: {model_config['input_shape']}")
                        print(f"  Output classes: {model_config['num_classes']}")
                    else:
                        # Default EEG model
                        self.model = self.build_eeg_model((256, 20), NUM_CLASSES)
                        print(f"Client {self.client_id} using default EEG model architecture")
                    
                    self.model.set_weights(weights)
                    print(f"Client {self.client_id} model initialized with server weights")
                    self.current_round = 0
                elif round_num == self.current_round:
                    self.model.set_weights(weights)
                    print(f"Client {self.client_id} received global model for round {self.current_round}")
    
    def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        
        # Create dataset
        ds_train = self.make_dataset(
            self.x_train, self.y_train, 
            batch_size, training=True
        )
        
        # Train
        print(f"[Client {self.client_id}] Training for {epochs} epochs...")
        history = self.model.fit(
            ds_train,
            epochs=epochs,
            verbose=2,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="loss", patience=3,
                    restore_best_weights=True, verbose=0
                )
            ]
        )
        
        # Get metrics
        final_loss = float(history.history['loss'][-1]) if 'loss' in history.history else 0.0
        final_acc = float(history.history['acc'][-1]) if 'acc' in history.history else 0.0
        
        # Get model weights
        weights = self.model.get_weights()
        # Compress or serialize weights
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(weights, data_type="weights")
            stats = self.quantizer.get_compression_stats(weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Size: {stats['compressed_size_mb']:.2f}MB")
            serialized_weights = compressed_data
        else:
            serialized_weights = self.serialize_weights(weights)
        
        # Random delay before sending
        delay = random.uniform(0.5, 3.0)
        print(f"Client {self.client_id} waiting {delay:.2f} seconds before sending update...")
        time.sleep(delay)
        
        # Send model update to server
        update = ModelUpdate(
            client_id=self.client_id,
            round=self.current_round,
            weights=serialized_weights,
            num_samples=int(len(self.y_train)),
            loss=final_loss,
            accuracy=final_acc
        )
        self.writers['model_update'].write(update)
        
        print(f"Client {self.client_id} sent model update for round {self.current_round}")
        print(f"Training metrics - Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
        
        time.sleep(0.5)
    
    def cleanup(self):
        """Cleanup DDS resources"""
        if self.participant:
            self.participant = None
        print(f"Client {self.client_id} DDS resources cleaned up")


if __name__ == "__main__":
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS)
    client.run()
