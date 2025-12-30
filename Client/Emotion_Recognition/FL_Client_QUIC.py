import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import pickle
import base64
import time
import asyncio
import os
import logging
from aioquic.asyncio import connect
from aioquic.quic.configuration import QuicConfiguration
from aioquic.quic.events import QuicEvent, StreamDataReceived
from aioquic.asyncio.protocol import QuicConnectionProtocol

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# QUIC Configuration
QUIC_HOST = os.getenv("QUIC_HOST", "localhost")
QUIC_PORT = int(os.getenv("QUIC_PORT", "4433"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "1"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))


class FederatedLearningClientProtocol(QuicConnectionProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = None
        self.stream_id = None
        self._stream_buffers = {}  # Buffer for incomplete messages
    
    def quic_event_received(self, event: QuicEvent):
        if isinstance(event, StreamDataReceived):
            # Get or create buffer for this stream
            if event.stream_id not in self._stream_buffers:
                self._stream_buffers[event.stream_id] = b''
            
            # Append new data to buffer
            self._stream_buffers[event.stream_id] += event.data
            
            # Try to decode complete messages (delimited by newline)
            while b'\n' in self._stream_buffers[event.stream_id]:
                message_data, self._stream_buffers[event.stream_id] = self._stream_buffers[event.stream_id].split(b'\n', 1)
                if message_data:
                    try:
                        data = message_data.decode('utf-8')
                        message = json.loads(data)
                        if self.client:
                            asyncio.create_task(self.client.handle_message(message))
                    except (json.JSONDecodeError, UnicodeDecodeError) as e:
                        print(f"Error decoding message: {e}")
            
            # If stream ended and buffer has remaining data, try to process it
            if event.end_stream and self._stream_buffers[event.stream_id]:
                try:
                    data = self._stream_buffers[event.stream_id].decode('utf-8')
                    message = json.loads(data)
                    if self.client:
                        asyncio.create_task(self.client.handle_message(message))
                    self._stream_buffers[event.stream_id] = b''
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"Error decoding remaining buffer: {e}")


class FederatedLearningClient:
    def __init__(self, client_id, num_clients, train_generator, validation_generator):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model = None
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.current_round = 0
        self.training_config = {"batch_size": 32, "local_epochs": 20}
        self.protocol = None
        self.stream_id = 0
        self.model_ready = asyncio.Event()  # Event to signal model is ready
        
        print(f"Client {self.client_id} initialized with:")
        print(f"  Training samples: {self.train_generator.n}")
        print(f"  Validation samples: {self.validation_generator.n}")
        print(f"  Waiting for initial global model from server...")
    
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
    
    async def send_message(self, message):
        """Send message to server via QUIC stream"""
        if self.protocol:
            msg_type = message.get('type')
            print(f"[DEBUG] Client {self.client_id} sending message type: {msg_type}")
            
            # Add newline delimiter for message framing
            data = (json.dumps(message) + '\n').encode('utf-8')
            print(f"[DEBUG] Client {self.client_id} message size: {len(data)} bytes")
            
            self.stream_id = self.protocol._quic.get_next_available_stream_id()
            # Send data with end_stream=True to ensure it's processed
            self.protocol._quic.send_stream_data(self.stream_id, data, end_stream=True)
            
            # Transmit immediately
            self.protocol.transmit()
            
            # For large messages, give more time for transmission
            if len(data) > 1000000:  # > 1MB
                print(f"[DEBUG] Client {self.client_id} waiting for large message transmission...")
                # Multiple transmit calls with delays for very large messages
                for i in range(5):
                    await asyncio.sleep(1)
                    self.protocol.transmit()
                    print(f"[DEBUG] Client {self.client_id} transmit call {i+1}/5")
            else:
                await asyncio.sleep(0.5)
            
            print(f"[DEBUG] Client {self.client_id} sent {msg_type} on stream {self.stream_id}")
    
    async def handle_message(self, message):
        """Handle incoming messages from server"""
        try:
            msg_type = message.get('type')
            print(f"[DEBUG] Client {self.client_id} received message type: {msg_type}")
            
            if msg_type == 'training_config':
                await self.handle_training_config(message)
            elif msg_type == 'global_model':
                await self.handle_global_model(message)
            elif msg_type == 'start_training':
                await self.handle_start_training(message)
            elif msg_type == 'start_evaluation':
                await self.handle_start_evaluation(message)
            elif msg_type == 'training_complete':
                await self.handle_training_complete()
        except Exception as e:
            print(f"Client {self.client_id} error handling message: {e}")
            import traceback
            traceback.print_exc()
    
    async def handle_training_config(self, message):
        """Update training configuration"""
        self.training_config = message['config']
        print(f"Client {self.client_id} updated config: {self.training_config}")
    
    async def handle_global_model(self, message):
        """Receive and set global model weights and architecture from server"""
        round_num = message['round']
        encoded_weights = message['weights']
        
        weights = self.deserialize_weights(encoded_weights)
        
        if round_num == 0:
            # Initial model from server - create model from server's config
            print(f"Client {self.client_id} received initial global model from server")
            
            model_config = message.get('model_config')
            if model_config:
                # Build model dynamically from server's layer configuration
                self.model = Sequential()
                
                for layer_info in model_config['layers']:
                    layer_type = layer_info['type']
                    
                    if layer_type == 'Input':
                        self.model.add(Input(shape=tuple(model_config['input_shape'])))
                    elif layer_type == 'Conv2D':
                        self.model.add(Conv2D(
                            filters=layer_info['filters'],
                            kernel_size=tuple(layer_info['kernel_size']),
                            activation=layer_info['activation']
                        ))
                    elif layer_type == 'MaxPooling2D':
                        self.model.add(MaxPooling2D(pool_size=tuple(layer_info['pool_size'])))
                    elif layer_type == 'Dropout':
                        self.model.add(Dropout(layer_info['rate']))
                    elif layer_type == 'Flatten':
                        self.model.add(Flatten())
                    elif layer_type == 'Dense':
                        if 'activation' in layer_info:
                            self.model.add(Dense(layer_info['units'], activation=layer_info['activation']))
                        else:
                            self.model.add(Dense(layer_info['units']))
                
                # Compile model
                self.model.compile(
                    loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=0.0001),
                    metrics=['accuracy']
                )
                print(f"Client {self.client_id} built CNN model from server configuration")
                print(f"  Model has {len(model_config['layers'])} layers, {model_config['num_classes']} output classes")
            else:
                raise ValueError("No model configuration received from server!")
            
            # Set the initial weights from server
            self.model.set_weights(weights)
            print(f"Client {self.client_id} model initialized with server weights")
            self.current_round = 0
            self.model_ready.set()  # Signal that model is ready
        else:
            # Updated model after aggregation
            self.model.set_weights(weights)
            # Only update current_round if this model is for a round >= current
            # (Don't go backwards if we've already moved to a later round)
            if round_num >= self.current_round:
                self.current_round = round_num
                print(f"Client {self.client_id} received global model for round {round_num}")
            else:
                print(f"Client {self.client_id} received late global model for round {round_num} (currently at round {self.current_round}), updating weights only")
            self.model_ready.set()  # Signal that model is ready
    
    async def handle_start_training(self, message):
        """Start local training when server signals"""
        round_num = message['round']
        
        # Wait for model to be initialized (with timeout)
        if not self.model_ready.is_set():
            print(f"Client {self.client_id} waiting for model initialization before training round {round_num}...")
            try:
                await asyncio.wait_for(self.model_ready.wait(), timeout=30.0)
                print(f"Client {self.client_id} model ready, proceeding with training")
            except asyncio.TimeoutError:
                print(f"Client {self.client_id} ERROR: Timeout waiting for model initialization")
                return
        
        if self.current_round == 0 and round_num == 1:
            self.current_round = round_num
            print(f"\nClient {self.client_id} starting training for round {round_num} with initial global model...")
            await self.train_local_model()
        elif round_num == self.current_round:
            print(f"\nClient {self.client_id} starting training for round {round_num}...")
            await self.train_local_model()
        else:
            print(f"Client {self.client_id} round mismatch - received signal for round {round_num}, currently at {self.current_round}")
    
    async def handle_start_evaluation(self, message):
        """Start evaluation when server signals"""
        round_num = message['round']
        
        # Ensure model is ready
        if not self.model_ready.is_set():
            print(f"Client {self.client_id} waiting for model before evaluation...")
            await self.model_ready.wait()
        
        if round_num == self.current_round:
            print(f"Client {self.client_id} starting evaluation for round {round_num}...")
            await self.evaluate_model()
            self.current_round = round_num + 1
            # Don't clear model_ready - the model is still valid for next round
            print(f"Client {self.client_id} ready for next round {self.current_round}")
        else:
            print(f"Client {self.client_id} skipping evaluation signal for round {round_num} (current: {self.current_round})")
    
    async def handle_training_complete(self):
        """Handle training completion signal from server"""
        print("\n" + "="*70)
        print(f"Client {self.client_id} - Training completed!")
        print("="*70)
        print("\nClient shutting down...")
        await asyncio.sleep(1)
        import sys
        sys.exit(0)
    
    async def train_local_model(self):
        """Train model on local data and send updates to server"""
        batch_size = self.training_config['batch_size']
        epochs = self.training_config['local_epochs']
        
        print(f"Training on {self.train_generator.n} samples for {epochs} epochs...")
        
        # Start a keep-alive task to prevent idle timeout during training
        keep_alive_task = asyncio.create_task(self._keep_alive_during_training())
        
        try:
            # Train in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Create a function with all parameters for fit
            def train_model():
                return self.model.fit(
                    self.train_generator,
                    epochs=epochs,
                    verbose=1
                )
            
            history = await loop.run_in_executor(None, train_model)
        finally:
            # Cancel keep-alive task
            keep_alive_task.cancel()
            try:
                await keep_alive_task
            except asyncio.CancelledError:
                pass
        
        # Prepare training metrics
        final_loss = float(history.history['loss'][-1])
        final_accuracy = float(history.history['accuracy'][-1])
        
        print(f"Client {self.client_id} training complete - "
              f"Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
        
        # Send model update to server
        await self.send_message({
            'type': 'model_update',
            'client_id': self.client_id,
            'round': self.current_round,
            'weights': self.serialize_weights(self.model.get_weights()),
            'num_samples': self.train_generator.n,
            'metrics': {
                'loss': final_loss,
                'accuracy': final_accuracy
            }
        })
    
    async def _keep_alive_during_training(self):
        """Send periodic pings to keep connection alive during training"""
        while True:
            await asyncio.sleep(30)  # Ping every 30 seconds
            if self.protocol:
                self.protocol.transmit()
    
    async def evaluate_model(self):
        """Evaluate model on validation data and send metrics to server"""
        print(f"Evaluating on {self.validation_generator.n} validation samples...")
        
        # Evaluate in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def evaluate():
            return self.model.evaluate(
                self.validation_generator,
                verbose=1
            )
        
        results = await loop.run_in_executor(None, evaluate)
        
        loss, accuracy = results[0], results[1]
        
        print(f"Client {self.client_id} evaluation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Send evaluation metrics to server
        await self.send_message({
            'type': 'metrics',
            'client_id': self.client_id,
            'round': self.current_round,
            'num_samples': self.validation_generator.n,
            'metrics': {
                'loss': float(loss),
                'accuracy': float(accuracy)
            }
        })
    
    async def register_with_server(self):
        """Register with the federated learning server"""
        await self.send_message({
            'type': 'register',
            'client_id': self.client_id
        })
        print(f"Client {self.client_id} registration sent to server")


async def main():
    # Setup data generators
    client_data_dir = f'Dataset/client_{CLIENT_ID}'
    
    if not os.path.exists(client_data_dir):
        print(f"Error: Client data directory not found: {client_data_dir}")
        return
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        f'{client_data_dir}/train',
        target_size=(48, 48),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        f'{client_data_dir}/validation',
        target_size=(48, 48),
        batch_size=32,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n{'='*70}")
    print(f"Federated Learning Client {CLIENT_ID} with QUIC")
    print(f"Server: {QUIC_HOST}:{QUIC_PORT}")
    print(f"{'='*70}\n")
    
    # Create client
    client = FederatedLearningClient(CLIENT_ID, NUM_CLIENTS, train_generator, validation_generator)
    
    # Configure QUIC with large stream data limits for model weights
    configuration = QuicConfiguration(
        is_client=True,
        alpn_protocols=["fl"],
        max_stream_data=20 * 1024 * 1024,  # 20 MB per stream
        max_data=50 * 1024 * 1024,  # 50 MB total connection data
        idle_timeout=3600.0,  # 1 hour idle timeout (training can take long)
    )
    configuration.verify_mode = False  # Disable certificate verification for testing
    
    # Create protocol factory
    def create_protocol(*args, **kwargs):
        protocol = FederatedLearningClientProtocol(*args, **kwargs)
        protocol.client = client
        client.protocol = protocol
        return protocol
    
    # Connect to server
    print(f"Connecting to QUIC server at {QUIC_HOST}:{QUIC_PORT}...")
    try:
        async with connect(
            QUIC_HOST,
            QUIC_PORT,
            configuration=configuration,
            create_protocol=create_protocol,
        ) as protocol:
            client.protocol = protocol
            print(f"✓ Connected to QUIC server successfully")
            
            # Register with server
            await client.register_with_server()
            
            print(f"Client {CLIENT_ID} waiting for training commands...")
            # Keep connection alive
            try:
                await asyncio.Future()
            except Exception as e:
                print(f"\n[ERROR] Client {CLIENT_ID} - Connection loop error: {e}")
                import traceback
                traceback.print_exc()
                raise
    except ConnectionError as e:
        print(f"\n❌ Connection Error: {e}")
        print(f"Failed to connect to QUIC server at {QUIC_HOST}:{QUIC_PORT}")
        print("Make sure the server is running and reachable.")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nClient interrupted by user")
        import sys
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nClient shutting down...")
