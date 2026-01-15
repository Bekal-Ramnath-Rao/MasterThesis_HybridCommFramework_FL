"""
Federated Learning Client for EEG Mental State Recognition using MQTT
Supports CNN+BiLSTM+MHA architecture with non-IID data partitioning
"""

import os
import sys
import json
import pickle
import base64
import time
import logging
import numpy as np
import tensorflow as tf
import paho.mqtt.client as mqtt
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

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "16"))

# MQTT Topics
TOPIC_GLOBAL_MODEL = "fl/global_model"
TOPIC_CLIENT_UPDATE = f"fl/client/{CLIENT_ID}/update"
TOPIC_CLIENT_METRICS = f"fl/client/{CLIENT_ID}/metrics"
TOPIC_TRAINING_CONFIG = "fl/training_config"
TOPIC_START_TRAINING = "fl/start_training"
TOPIC_START_EVALUATION = "fl/start_evaluation"
TOPIC_TRAINING_COMPLETE = "fl/training_complete"

# Training Configuration
AUTOTUNE = tf.data.AUTOTUNE
SMOOTH_EPS = 0.05


class FederatedLearningClient:
    def __init__(self, client_id, num_clients):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        self.x_train = None
        self.y_train = None
        self.current_round = 0
        self.training_config = {
            "batch_size": 256,
            "local_epochs": 5,
            "val_split": 0.10
        }
        self.class_weights = None
        
        # Initialize MQTT client
        self.mqtt_client = mqtt.Client(client_id=f"fl_eeg_client_{client_id}")
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Load and partition data
        self.prepare_data()
    
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
        print(f"  Training samples: {len(self.y_train)}")
        print(f"  Input shape: {self.x_train.shape}")
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
            print(f"\n[Client {self.client_id}] Connected to MQTT broker")
            
            # Subscribe to topics
            self.mqtt_client.subscribe(TOPIC_GLOBAL_MODEL)
            print(f"  Subscribed to {TOPIC_GLOBAL_MODEL}")
            
            self.mqtt_client.subscribe(TOPIC_TRAINING_CONFIG)
            print(f"  Subscribed to {TOPIC_TRAINING_CONFIG}")
            
            self.mqtt_client.subscribe(TOPIC_START_TRAINING)
            print(f"  Subscribed to {TOPIC_START_TRAINING}")
            
            self.mqtt_client.subscribe(TOPIC_START_EVALUATION)
            print(f"  Subscribed to {TOPIC_START_EVALUATION}")
            
            self.mqtt_client.subscribe(TOPIC_TRAINING_COMPLETE)
            print(f"  Subscribed to {TOPIC_TRAINING_COMPLETE}")
            
            # Send registration message
            reg_msg = {
                "client_id": self.client_id,
                "num_samples": int(len(self.y_train))
            }
            self.mqtt_client.publish("fl/client_register", json.dumps(reg_msg))
            print(f"  Registration message sent")
        else:
            print(f"[Client {self.client_id}] Failed to connect, return code {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
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
            print(f"[Client {self.client_id}] Error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    def on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker"""
        if rc == 0:
            print(f"\n[Client {self.client_id}] Clean disconnect from broker")
            self.mqtt_client.loop_stop()
        else:
            print(f"[Client {self.client_id}] Unexpected disconnect, return code {rc}")
            self.mqtt_client.loop_stop()
    
    def handle_training_complete(self):
        """Handle training completion signal from server"""
        print("\n" + "=" * 70)
        print(f"[Client {self.client_id}] Training completed!")
        print("=" * 70)
        print("\nDisconnecting from server...")
        time.sleep(1)
        self.mqtt_client.disconnect()
    
    def handle_global_model(self, payload):
        """Receive and set global model weights"""
        data = json.loads(payload.decode())
        round_num = data['round']
        encoded_weights = data['weights']
        
        weights = self.deserialize_weights(encoded_weights)
        
        if round_num == 0:
            print(f"\n[Client {self.client_id}] Received initial global model")
            
            # Build model from config
            model_config = data.get('model_config')
            if model_config:
                print(f"[Client {self.client_id}] Building model architecture...")
                self.model = self.build_model_from_config(model_config)
                self.model.set_weights(weights)
                # Verify model is ready
                _ = self.model(tf.zeros((1, *model_config['input_shape'])), training=False)
                print(f"[Client {self.client_id}] Model initialized and verified with global weights")
        else:
            if self.model is not None:
                self.model.set_weights(weights)
                print(f"\n[Client {self.client_id}] Updated model weights for round {round_num}")
            else:
                print(f"[Client {self.client_id}] ERROR: Model not initialized!")
    
    def build_model_from_config(self, config):
        """Build model from server configuration"""
        # For simplicity, rebuild the full model architecture
        # This should match the server's model exactly
        return self.build_eeg_model(
            input_shape=tuple(config['input_shape']),
            num_classes=config['num_classes']
        )
    
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
    
    def handle_training_config(self, payload):
        """Handle training configuration update"""
        config = json.loads(payload.decode())
        self.training_config.update(config)
        print(f"\n[Client {self.client_id}] Updated training config: {self.training_config}")
    
    def handle_start_training(self, payload):
        """Handle start training command"""
        data = json.loads(payload.decode())
        round_num = data['round']
        self.current_round = round_num
        
        print(f"\n{'=' * 70}")
        print(f"[Client {self.client_id}] Starting training for round {round_num}")
        print(f"{'=' * 70}")
        
        # Wait for model to be initialized (race condition protection)
        max_wait = 30  # seconds
        wait_time = 0
        while self.model is None and wait_time < max_wait:
            print(f"[Client {self.client_id}] Waiting for model initialization... ({wait_time}s)")
            time.sleep(1)
            wait_time += 1
        
        if self.model is None:
            print(f"[Client {self.client_id}] ERROR: Model not initialized after {max_wait}s!")
            return
        
        # Train the model
        start_time = time.time()
        history = self.train_local_model()
        train_time = time.time() - start_time
        
        # Send update to server
        self.send_update_to_server(history, train_time, round_num)
    
    def train_local_model(self):
        """Train the local model"""
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
        
        return history
    
    def send_update_to_server(self, history, train_time, round_num):
        """Send model update and metrics to server"""
        # Get updated weights
        weights = self.model.get_weights()
        encoded_weights = self.serialize_weights(weights)
        
        # Prepare update message
        update_msg = {
            "client_id": self.client_id,
            "round": round_num,
            "weights": encoded_weights,
            "num_samples": int(len(self.y_train)),
            "train_time": float(train_time)
        }
        
        # Send update
        print(f"[Client {self.client_id}] Sending update to server...")
        self.mqtt_client.publish(TOPIC_CLIENT_UPDATE, json.dumps(update_msg), qos=1)
        
        # Send metrics
        final_loss = float(history.history['loss'][-1]) if 'loss' in history.history else 0.0
        final_acc = float(history.history['acc'][-1]) if 'acc' in history.history else 0.0
        
        metrics_msg = {
            "client_id": self.client_id,
            "round": round_num,
            "loss": final_loss,
            "accuracy": final_acc,
            "train_time": float(train_time)
        }
        
        self.mqtt_client.publish(TOPIC_CLIENT_METRICS, json.dumps(metrics_msg), qos=1)
        
        print(f"[Client {self.client_id}] Update sent successfully")
        print(f"  Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
        print(f"  Training time: {train_time:.2f}s")
    
    def handle_start_evaluation(self, payload):
        """Handle start evaluation command"""
        print(f"\n[Client {self.client_id}] Evaluation requested (no local test set)")
    
    def start(self):
        """Start the client"""
        print(f"\n{'=' * 70}")
        print(f"EEG Mental State Recognition - Federated Learning Client {self.client_id}")
        print(f"{'=' * 70}")
        print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
        print(f"Number of clients: {self.num_clients}")
        print(f"Number of rounds: {NUM_ROUNDS}")
        print(f"{'=' * 70}\n")
        
        try:
            # Keepalive set to 3600s (1 hour) to prevent timeout during long training
            self.mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=3600)
            self.mqtt_client.loop_forever()
        except KeyboardInterrupt:
            print(f"\n[Client {self.client_id}] Interrupted by user")
            self.mqtt_client.disconnect()
        except Exception as e:
            print(f"\n[Client {self.client_id}] Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS)
    client.start()
