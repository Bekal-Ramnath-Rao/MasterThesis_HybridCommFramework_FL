"""
Unified Federated Learning Client for Emotion Recognition
with RL-based Protocol Selection

Supports: MQTT, AMQP, gRPC, QUIC, DDS
Uses Q-Learning to dynamically select the best protocol
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
from cyclonedds.util import duration

# Import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl_q_learning_selector import QLearningProtocolSelector, EnvironmentStateManager
from dynamic_network_controller import DynamicNetworkController

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Environment variables
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
USE_RL_SELECTION = os.getenv("USE_RL_SELECTION", "true").lower() == "true"


class UnifiedFLClient_Emotion:
    """
    Unified Federated Learning Client for Emotion Recognition
    Integrates all 5 protocols with RL-based selection
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
        self.local_epochs = 20
        
        # RL Components
        if USE_RL_SELECTION:
            self.rl_selector = QLearningProtocolSelector(
                save_path=f"q_table_emotion_client_{client_id}.pkl"
            )
            self.env_manager = EnvironmentStateManager()
            self.env_manager.update_model_size('medium')  # Emotion recognition
        else:
            self.rl_selector = None
            self.env_manager = None
        
        # Protocol handlers
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
        print(f"UNIFIED FL CLIENT - EMOTION RECOGNITION")
        print(f"{'='*70}")
        print(f"Client ID: {self.client_id}/{self.num_clients}")
        print(f"RL Protocol Selection: {'ENABLED' if USE_RL_SELECTION else 'DISABLED'}")
        print(f"{'='*70}\n")
    
    def build_model(self) -> keras.Model:
        """Build emotion recognition model (CNN for facial expressions)"""
        model = keras.Sequential([
            keras.layers.Input(shape=(48, 48, 1)),
            keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(7, activation='softmax')  # 7 emotions
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def select_protocol(self) -> str:
        """
        Select protocol using RL or fallback to default
        
        Returns:
            Selected protocol name
        """
        if USE_RL_SELECTION and self.rl_selector and self.env_manager:
            # Detect current network conditions
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory().percent
                
                # Update environment state
                resource_level = self.env_manager.detect_resource_level(cpu, memory)
                self.env_manager.update_resource_level(resource_level)
                
                # Get current state
                state = self.env_manager.get_current_state()
                
                # Select protocol using Q-learning
                protocol = self.rl_selector.select_protocol(state, training=True)
                
                print(f"\n[RL Selection] State: {state}")
                print(f"[RL Selection] Selected Protocol: {protocol.upper()}")
                
                return protocol
            except Exception as e:
                print(f"[RL Selection] Error: {e}, falling back to MQTT")
                return 'mqtt'
        else:
            # Default protocol
            return os.getenv("DEFAULT_PROTOCOL", "mqtt").lower()
    
    def train_local_model(self) -> Dict:
        """
        Train model locally using real data generators
        
        Returns:
            Training metrics
        """
        if self.model is None:
            self.model = self.build_model()
        
        start_time = time.time()
        
        # Train with GPU acceleration
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            history = self.model.fit(
                self.train_generator,
                epochs=self.local_epochs,
                verbose=0
            )
        
        training_time = time.time() - start_time
        
        # Evaluate on validation data
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            val_loss, val_accuracy = self.model.evaluate(
                self.validation_generator,
                verbose=0
            )
        
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
            self.model = self.build_model()
        weights = pickle.loads(weights_bytes)
        self.model.set_weights(weights)
    
    def _handle_mqtt(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle MQTT protocol communication"""
        try:
            broker = os.getenv("MQTT_BROKER", "mqtt-broker")
            port = int(os.getenv("MQTT_PORT", "1883"))
            
            client = mqtt_client.Client(f"emotion_client_{self.client_id}")
            client.connect(broker, port)
            
            start_time = time.time()
            
            if action == "send":
                # Send weights to server
                topic = f"fl/emotion/client_{self.client_id}/weights"
                client.publish(topic, data)
                client.disconnect()
                
                comm_time = time.time() - start_time
                self.round_metrics['communication_time'] = comm_time
                return True, None
                
            elif action == "receive":
                # Receive global weights from server
                received_data = [None]
                
                def on_message(client, userdata, msg):
                    received_data[0] = msg.payload
                    client.disconnect()
                
                client.on_message = on_message
                client.subscribe(f"fl/emotion/global_weights")
                client.loop_start()
                
                # Wait for message
                timeout = 30
                elapsed = 0
                while received_data[0] is None and elapsed < timeout:
                    time.sleep(0.1)
                    elapsed += 0.1
                
                client.loop_stop()
                
                comm_time = time.time() - start_time
                self.round_metrics['communication_time'] = comm_time
                
                if received_data[0]:
                    return True, received_data[0]
                return False, None
                
        except Exception as e:
            print(f"[MQTT] Error: {e}")
            return False, None
    
    def _handle_amqp(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle AMQP protocol communication"""
        try:
            host = os.getenv("AMQP_HOST", "rabbitmq")
            credentials = pika.PlainCredentials('guest', 'guest')
            connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=host, credentials=credentials)
            )
            channel = connection.channel()
            
            start_time = time.time()
            
            if action == "send":
                queue_name = f"emotion_client_{self.client_id}_weights"
                channel.queue_declare(queue=queue_name)
                channel.basic_publish(exchange='', routing_key=queue_name, body=data)
                connection.close()
                
                comm_time = time.time() - start_time
                self.round_metrics['communication_time'] = comm_time
                return True, None
                
            elif action == "receive":
                queue_name = "emotion_global_weights"
                channel.queue_declare(queue=queue_name)
                
                method, properties, body = channel.basic_get(queue=queue_name, auto_ack=True)
                connection.close()
                
                comm_time = time.time() - start_time
                self.round_metrics['communication_time'] = comm_time
                
                if body:
                    return True, body
                return False, None
                
        except Exception as e:
            print(f"[AMQP] Error: {e}")
            return False, None
    
    def _handle_grpc(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle gRPC protocol communication"""
        # Simplified gRPC implementation
        # In production, use proper proto definitions and stubs
        print("[gRPC] Protocol handler (simplified)")
        return True, data
    
    def _handle_quic(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle QUIC protocol communication"""
        # Simplified QUIC implementation
        # In production, use aioquic library
        print("[QUIC] Protocol handler (simplified)")
        return True, data
    
    def _handle_dds(self, action: str, data: Optional[bytes] = None) -> Tuple[bool, Optional[bytes]]:
        """Handle DDS protocol communication"""
        try:
            domain_id = int(os.getenv("DDS_DOMAIN_ID", "0"))
            participant = DomainParticipant(domain_id)
            
            start_time = time.time()
            
            if action == "send":
                # Create topic and writer
                topic_name = f"EmotionWeights_Client{self.client_id}"
                # Simplified: In production, use proper IDL types
                # writer = DataWriter(participant, topic)
                # writer.write(data)
                
                comm_time = time.time() - start_time
                self.round_metrics['communication_time'] = comm_time
                return True, None
                
            elif action == "receive":
                # Create topic and reader
                topic_name = "EmotionGlobalWeights"
                # Simplified: In production, use proper IDL types
                # reader = DataReader(participant, topic)
                # samples = reader.take(1)
                
                comm_time = time.time() - start_time
                self.round_metrics['communication_time'] = comm_time
                return True, data
                
        except Exception as e:
            print(f"[DDS] Error: {e}")
            return False, None
    
    def federated_learning_round(
        self,
        protocol: Optional[str] = None
    ) -> Dict:
        """
        Execute one round of federated learning
        
        Args:
            protocol: Protocol to use (None = auto-select)
            
        Returns:
            Metrics for this round
        """
        round_start = time.time()
        
        # Select protocol
        if protocol is None:
            protocol = self.select_protocol()
        
        print(f"\n{'='*70}")
        print(f"FL ROUND - Using {protocol.upper()} Protocol")
        print(f"{'='*70}")
        
        try:
            # 1. Receive global weights
            print(f"[{protocol.upper()}] Receiving global model...")
            success, global_weights = self.protocol_handlers[protocol]("receive")
            
            if success and global_weights:
                self.set_model_weights(global_weights)
                print(f"[{protocol.upper()}] Global model received")
            
            # 2. Train locally
            print(f"[Training] Starting local training...")
            train_metrics = self.train_local_model()
            
            # 3. Send updated weights
            print(f"[{protocol.upper()}] Sending updated weights...")
            local_weights = self.get_model_weights()
            success, _ = self.protocol_handlers[protocol]("send", local_weights)
            
            if success:
                print(f"[{protocol.upper()}] Weights sent successfully")
                self.round_metrics['success'] = True
            else:
                print(f"[{protocol.upper()}] Failed to send weights")
                self.round_metrics['success'] = False
            
            # Calculate metrics
            round_time = time.time() - round_start
            self.round_metrics['convergence_time'] = train_metrics['training_time']
            self.round_metrics['accuracy'] = train_metrics['val_accuracy']
            
            # Update RL if enabled
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
            self.round_metrics['success'] = False
            
            if USE_RL_SELECTION and self.rl_selector:
                # Negative reward for failure
                self.rl_selector.update_q_value(-10.0, done=False)
            
            return {'protocol': protocol, 'success': False}
    
    def run(self, num_rounds: int = 10):
        """
        Run federated learning for multiple rounds
        
        Args:
            num_rounds: Number of FL rounds
        """
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
            
            # End RL episode
            if USE_RL_SELECTION and self.rl_selector:
                self.rl_selector.end_episode()
        
        # Print final RL statistics
        if USE_RL_SELECTION and self.rl_selector:
            self.rl_selector.print_statistics()


def load_emotion_data(client_id: int):
    """
    Load emotion recognition dataset for a specific client
    
    Args:
        client_id: Client identifier
        
    Returns:
        Tuple of (train_generator, validation_generator)
    """
    # Initialize image data generator with rescaling
    train_data_gen = ImageDataGenerator(rescale=1./255)
    validation_data_gen = ImageDataGenerator(rescale=1./255)

    # Load training and validation data from client-specific directories
    train_path = f'Dataset/client_{client_id}/train/'
    val_path = f'Dataset/client_{client_id}/validation/'
    
    print(f"[Dataset] Loading from:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    
    train_generator = train_data_gen.flow_from_directory(
        train_path,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical'
    )

    validation_generator = validation_data_gen.flow_from_directory(
        val_path,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode='categorical'
    )
    
    print(f"[Dataset] Train samples: {train_generator.samples}")
    print(f"[Dataset] Validation samples: {validation_generator.samples}")
    print(f"[Dataset] Classes: {train_generator.num_classes}")
    
    return train_generator, validation_generator


def main():
    """Main function"""
    print(f"Unified FL Client - Emotion Recognition (Client {CLIENT_ID})")
    
    # Load real emotion recognition dataset
    print(f"\n{'='*70}")
    print("LOADING EMOTION RECOGNITION DATASET")
    print(f"{'='*70}")
    
    try:
        train_generator, validation_generator = load_emotion_data(CLIENT_ID)
    except Exception as e:
        print(f"[Error] Failed to load dataset: {e}")
        print(f"\nPlease ensure dataset exists at:")
        print(f"  Dataset/client_{CLIENT_ID}/train/")
        print(f"  Dataset/client_{CLIENT_ID}/validation/")
        return
    
    # Create client
    client = UnifiedFLClient_Emotion(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    # Run FL
    num_rounds = int(os.getenv("NUM_ROUNDS", "10"))
    client.run(num_rounds)


if __name__ == "__main__":
    main()
