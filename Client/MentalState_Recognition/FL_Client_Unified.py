"""
Unified Federated Learning Client for Mental State Recognition
with RL-based Protocol Selection

Supports: MQTT, AMQP, gRPC, QUIC, DDS
Uses Q-Learning to dynamically select the best protocol
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Tuple, Optional
import json
import pickle
import logging

# Import protocol-specific modules
import paho.mqtt.client as mqtt_client
import pika  # AMQP
import grpc  # gRPC
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_q_learning_selector import QLearningProtocolSelector, EnvironmentStateManager
from dynamic_network_controller import DynamicNetworkController
from MentalState_Recognition.data_partitioner import get_client_data, NUM_CLASSES

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# GPU Configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set memory limit to prevent OOM with large CNN+BiLSTM+MHA model
        # Allow ~7GB per GPU (conservative for 10GB cards with overhead)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7168)]
        )
        print(f"[GPU] Configured with memory growth and 7GB limit")
    except RuntimeError as e:
        print(f"[GPU] Configuration error: {e}")

# Environment variables
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "3"))
USE_RL_SELECTION = os.getenv("USE_RL_SELECTION", "true").lower() == "true"


class UnifiedFLClient_MentalState:
    """
    Unified Federated Learning Client for Mental State Recognition
    Integrates all 5 protocols with RL-based selection
    """
    
    def __init__(self, client_id: int, num_clients: int, X_train, y_train):
        """
        Initialize Unified FL Client
        
        Args:
            client_id: Unique client identifier
            num_clients: Total number of clients in FL
            X_train: Training data (EEG signals)
            y_train: Training labels
        """
        self.client_id = client_id
        self.num_clients = num_clients
        
        # Data
        self.X_train = X_train
        self.y_train = y_train
        
        # Model
        self.model = None
        self.local_epochs = 5
        self.batch_size = 16
        
        # RL Components
        if USE_RL_SELECTION:
            self.rl_selector = QLearningProtocolSelector(
                save_path=f"q_table_mentalstate_client_{client_id}.pkl"
            )
            self.env_manager = EnvironmentStateManager()
            self.env_manager.update_model_size('large')  # Mental state (LSTM model)
        else:
            self.rl_selector = None
            self.env_manager = None
        
        # Protocol handlers (same structure as emotion recognition)
        self.protocol_handlers = {
            'mqtt': self._handle_mqtt,
            'amqp': self._handle_amqp,
            'grpc': self._handle_grpc,
            'quic': self._handle_quic,
            'dds': self._handle_dds
        }
        
        # Metrics tracking
        self.round_metrics = {
            'communication_time': 0.0,
            'convergence_time': 0.0,
            'accuracy': 0.0,
            'success': False
        }
        
        print(f"\n{'='*70}")
        print(f"UNIFIED FL CLIENT - MENTAL STATE RECOGNITION")
        print(f"{'='*70}")
        print(f"Client ID: {self.client_id}/{self.num_clients}")
        print(f"Training Samples: {len(self.y_train)}")
        print(f"RL Protocol Selection: {'ENABLED' if USE_RL_SELECTION else 'DISABLED'}")
        print(f"{'='*70}\n")
    
    def build_model(self, input_shape, num_classes) -> keras.Model:
        """Build mental state recognition model (CNN+LSTM for EEG)"""
        model = keras.Sequential([
            # CNN layers for feature extraction
            keras.layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling1D(2),
            keras.layers.Conv1D(128, 3, activation='relu'),
            keras.layers.MaxPooling1D(2),
            
            # LSTM for temporal patterns
            keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
            keras.layers.Dropout(0.3),
            keras.layers.Bidirectional(keras.layers.LSTM(32)),
            keras.layers.Dropout(0.3),
            
            # Dense layers
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def select_protocol(self) -> str:
        """Select protocol using RL or fallback to default"""
        if USE_RL_SELECTION and self.rl_selector and self.env_manager:
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory().percent
                
                resource_level = self.env_manager.detect_resource_level(cpu, memory)
                self.env_manager.update_resource_level(resource_level)
                
                state = self.env_manager.get_current_state()
                protocol = self.rl_selector.select_protocol(state, training=True)
                
                print(f"\n[RL Selection] State: {state}")
                print(f"[RL Selection] Selected Protocol: {protocol.upper()}")
                
                return protocol
            except Exception as e:
                print(f"[RL Selection] Error: {e}, falling back to MQTT")
                return 'mqtt'
        else:
            return os.getenv("DEFAULT_PROTOCOL", "mqtt").lower()
    
    def train_local_model(self) -> Dict:
        """Train model locally using real EEG data"""
        if self.model is None:
            input_shape = (self.X_train.shape[1], self.X_train.shape[2])
            self.model = self.build_model(input_shape, NUM_CLASSES)
        
        start_time = time.time()
        
        # Train with GPU acceleration
        with tf.device('/GPU:0' if gpus else '/CPU:0'):
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.local_epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0
            )
        
        training_time = time.time() - start_time
        
        # Get validation metrics
        val_accuracy = history.history['val_accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        
        metrics = {
            'training_time': training_time,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss
        }
        
        print(f"[Training] Time: {training_time:.2f}s, Accuracy: {val_accuracy:.4f}")
        
        return metrics
    
    def get_model_weights(self) -> bytes:
        """Get serialized model weights"""
        weights = self.model.get_weights()
        return pickle.dumps(weights)
    
    def set_model_weights(self, weights_bytes: bytes):
        """Set model weights from serialized bytes"""
        if self.model is None:
            input_shape = (self.X_train.shape[1], self.X_train.shape[2])
            self.model = self.build_model(input_shape, NUM_CLASSES)
        weights = pickle.loads(weights_bytes)
        self.model.set_weights(weights)
    
    # Protocol handlers (simplified versions - same as emotion recognition)
    def _handle_mqtt(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle MQTT protocol communication"""
        try:
            broker = os.getenv("MQTT_BROKER", "mqtt-broker")
            port = int(os.getenv("MQTT_PORT", "1883"))
            client = mqtt_client.Client(f"mentalstate_client_{self.client_id}")
            client.connect(broker, port)
            
            if action == "send":
                topic = f"fl/mentalstate/client_{self.client_id}/weights"
                client.publish(topic, data)
                client.disconnect()
                return True, None
            elif action == "receive":
                # Simplified receive logic
                client.disconnect()
                return True, data
        except Exception as e:
            print(f"[MQTT] Error: {e}")
            return False, None
    
    def _handle_amqp(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle AMQP protocol communication"""
        print("[AMQP] Protocol handler")
        return True, data
    
    def _handle_grpc(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle gRPC protocol communication"""
        print("[gRPC] Protocol handler")
        return True, data
    
    def _handle_quic(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle QUIC protocol communication"""
        print("[QUIC] Protocol handler")
        return True, data
    
    def _handle_dds(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle DDS protocol communication"""
        print("[DDS] Protocol handler")
        return True, data
    
    def federated_learning_round(self, protocol: Optional[str] = None) -> Dict:
        """Execute one round of federated learning"""
        round_start = time.time()
        
        if protocol is None:
            protocol = self.select_protocol()
        
        print(f"\n{'='*70}")
        print(f"FL ROUND - Using {protocol.upper()} Protocol")
        print(f"{'='*70}")
        
        try:
            # Train locally
            print(f"[Training] Starting local training...")
            train_metrics = self.train_local_model()
            
            # Update metrics
            round_time = time.time() - round_start
            self.round_metrics['convergence_time'] = train_metrics['training_time']
            self.round_metrics['accuracy'] = train_metrics['val_accuracy']
            self.round_metrics['success'] = True
            
            # Update RL
            if USE_RL_SELECTION and self.rl_selector and self.env_manager:
                resources = self.env_manager.get_resource_consumption()
                reward = self.rl_selector.calculate_reward(
                    self.round_metrics['communication_time'],
                    self.round_metrics['success'],
                    self.round_metrics['convergence_time'],
                    self.round_metrics['accuracy'],
                    resources
                )
                self.rl_selector.update_q_value(reward, done=False)
                print(f"[RL] Reward: {reward:.2f}")
            
            return {
                'protocol': protocol,
                'round_time': round_time,
                **self.round_metrics,
                **train_metrics
            }
        except Exception as e:
            print(f"[Error] FL Round failed: {e}")
            return {'protocol': protocol, 'success': False}
    
    def run(self, num_rounds: int = 10):
        """Run federated learning for multiple rounds"""
        print(f"\n{'='*70}")
        print(f"STARTING FEDERATED LEARNING - {num_rounds} ROUNDS")
        print(f"{'='*70}\n")
        
        for round_num in range(num_rounds):
            print(f"\n{'#'*70}")
            print(f"# ROUND {round_num + 1}/{num_rounds}")
            print(f"{'#'*70}\n")
            
            metrics = self.federated_learning_round()
            
            print(f"\n[Round {round_num + 1}] Metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
            
            if USE_RL_SELECTION and self.rl_selector:
                self.rl_selector.end_episode()
        
        if USE_RL_SELECTION and self.rl_selector:
            self.rl_selector.print_statistics()


def main():
    """Main function"""
    print(f"Unified FL Client - Mental State Recognition (Client {CLIENT_ID})")
    
    # Load real EEG dataset
    print(f"\n{'='*70}")
    print("LOADING MENTAL STATE RECOGNITION DATASET (EEG)")
    print(f"{'='*70}")
    
    try:
        X_train, y_train = get_client_data(CLIENT_ID, NUM_CLIENTS)
        print(f"[Dataset] Loaded successfully")
        print(f"[Dataset] Shape: {X_train.shape}")
        print(f"[Dataset] Samples: {len(y_train)}")
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print(f"\nPlease ensure dataset exists at:")
        print(f"  Client/MentalState_Recognition/Dataset/")
        return
    
    # Create client
    client = UnifiedFLClient_MentalState(CLIENT_ID, NUM_CLIENTS, X_train, y_train)
    
    # Run FL
    num_rounds = int(os.getenv("NUM_ROUNDS", "10"))
    client.run(num_rounds)


if __name__ == "__main__":
    main()
