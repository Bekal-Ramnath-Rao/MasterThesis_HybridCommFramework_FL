#!/usr/bin/env python3
"""
Fix all client and server files to use generic broadcast model:
- Servers always send model_config with global models
- Clients initialize from any global model if not initialized
- Remove strict round validation - allow late-joiners
"""

import os
import re

# Define all client and server file paths
CLIENTS = [
    "Client/Emotion_Recognition/FL_Client_AMQP.py",
    "Client/Emotion_Recognition/FL_Client_gRPC.py",
    "Client/Emotion_Recognition/FL_Client_QUIC.py",
    "Client/Emotion_Recognition/FL_Client_DDS.py",
    "Client/Emotion_Recognition/FL_Client_Unified.py",
    "Client/MentalState_Recognition/FL_Client_MQTT.py",
    "Client/MentalState_Recognition/FL_Client_AMQP.py",
    "Client/MentalState_Recognition/FL_Client_gRPC.py",
    "Client/MentalState_Recognition/FL_Client_QUIC.py",
    "Client/MentalState_Recognition/FL_Client_DDS.py",
    "Client/MentalState_Recognition/FL_Client_Unified.py",
    "Client/Temperature_Regulation/FL_Client_MQTT.py",
    "Client/Temperature_Regulation/FL_Client_AMQP.py",
    "Client/Temperature_Regulation/FL_Client_gRPC.py",
    "Client/Temperature_Regulation/FL_Client_QUIC.py",
    "Client/Temperature_Regulation/FL_Client_DDS.py",
    "Client/Temperature_Regulation/FL_Client_Unified.py",
]

SERVERS = [
    "Server/Emotion_Recognition/FL_Server_AMQP.py",
    "Server/Emotion_Recognition/FL_Server_gRPC.py",
    "Server/Emotion_Recognition/FL_Server_QUIC.py",
    "Server/Emotion_Recognition/FL_Server_DDS.py",
    "Server/Emotion_Recognition/FL_Server_Unified.py",
    "Server/MentalState_Recognition/FL_Server_MQTT.py",
    "Server/MentalState_Recognition/FL_Server_AMQP.py",
    "Server/MentalState_Recognition/FL_Server_gRPC.py",
    "Server/MentalState_Recognition/FL_Server_QUIC.py",
    "Server/MentalState_Recognition/FL_Server_DDS.py",
    "Server/MentalState_Recognition/FL_Server_Unified.py",
    "Server/Temperature_Regulation/FL_Server_MQTT.py",
    "Server/Temperature_Regulation/FL_Server_AMQP.py",
    "Server/Temperature_Regulation/FL_Server_gRPC.py",
    "Server/Temperature_Regulation/FL_Server_QUIC.py",
    "Server/Temperature_Regulation/FL_Server_DDS.py",
    "Server/Temperature_Regulation/FL_Server_Unified.py",
]

def fix_client_file(filepath):
    """Fix client file to support generic broadcast"""
    print(f"Fixing client: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # 1. Add build_model_from_config if not present
    if 'def build_model_from_config' not in content:
        # Find deserialize_weights method
        pattern = r'(    def deserialize_weights\(self.*?\n(?:.*?\n)*?        return weights\n)'
        match = re.search(pattern, content)
        if match:
            insert_after = match.group(1)
            build_model_method = '''    
    def build_model_from_config(self, model_config):
        """Build model from server-provided configuration"""
        input_shape = model_config.get('input_shape')
        num_classes = model_config.get('num_classes')
        layers = model_config.get('layers', [])
        
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
            elif layer['type'] == 'lstm':
                model.add(tf.keras.layers.LSTM(layer['units'], return_sequences=layer.get('return_sequences', False)))
            elif layer['type'] == 'gru':
                model.add(tf.keras.layers.GRU(layer['units'], return_sequences=layer.get('return_sequences', False)))
        
        model.compile(
            optimizer='adam',
            loss=model_config.get('loss', 'categorical_crossentropy'),
            metrics=['accuracy']
        )
        
        return model
    
'''
            content = content.replace(insert_after, insert_after + build_model_method)
            print(f"  ✓ Added build_model_from_config method")
    
    # 2. Fix handle_global_model - remove round 0 special case and strict validation
    # Pattern matches handle_global_model with various implementations
    pattern = r'(    def handle_global_model\(self.*?\n)((?:.*?\n)*?)(        except Exception as e:.*?\n(?:.*?\n)*?            traceback\.print_exc\(\))'
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        # Check if it has the old validation pattern
        old_body = match.group(2)
        if 'if round_num == 0:' in old_body or 'ERROR: Received model for round' in old_body:
            new_body = '''        """Receive and apply global model from server"""
        try:
            data = json.loads(payload.decode()) if isinstance(payload, bytes) else payload
            round_num = data.get('round', 0)
            
            # Check for duplicate (already processed this exact model)
            if hasattr(self, 'last_global_round') and self.last_global_round == round_num and self.model is not None:
                print(f"Client {self.client_id} ignoring duplicate global model for round {round_num}")
                return
            
            print(f"Client {self.client_id} received global model (round {round_num})")
            
            # Decompress/deserialize weights
            if 'quantized_data' in data:
                # Handle quantized/compressed data
                compressed_data = data['quantized_data']
                if isinstance(compressed_data, str):
                    import base64, pickle
                    compressed_data = pickle.loads(base64.b64decode(compressed_data.encode('utf-8')))
                if hasattr(self, 'quantization') and self.quantization is not None:
                    weights = self.quantization.decompress(compressed_data)
                elif hasattr(self, 'quantizer') and self.quantizer is not None:
                    weights = self.quantizer.decompress(compressed_data)
                else:
                    weights = compressed_data
                print(f"Client {self.client_id} decompressed quantized model")
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
            
            # Initialize model if not already done (for late-joining or first-time clients)
            if self.model is None:
                model_config = data.get('model_config')
                if model_config:
                    print(f"Client {self.client_id} initializing model from received configuration...")
                    self.model = self.build_model_from_config(model_config)
                    print(f"Client {self.client_id} model built successfully")
                else:
                    print(f"Client {self.client_id} WARNING: No model_config in global model, cannot initialize!")
                    return
            
            # Apply received weights
            self.model.set_weights(weights)
            self.current_round = round_num
            if hasattr(self, 'last_global_round'):
                self.last_global_round = round_num
            print(f"Client {self.client_id} updated model weights (round {round_num})")
            
'''
            content = content.replace(match.group(0), match.group(1) + new_body + match.group(3))
            print(f"  ✓ Fixed handle_global_model for generic broadcast")
    
    # 3. Fix handle_start_training - remove round validation
    pattern = r'    def handle_start_training\(self.*?\n(?:.*?\n)*?        self\.train_local_model\(\)(?:\n.*?print\(.*?\))*'
    match = re.search(pattern, content)
    if match:
        old_method = match.group(0)
        if 'round_num >= self.current_round' in old_method or 'round mismatch' in old_method:
            new_method = '''    def handle_start_training(self, payload):
        """Start local training when server signals"""
        data = json.loads(payload.decode()) if isinstance(payload, bytes) else payload
        round_num = data.get('round', data.get('round_num', 0))
        
        # Check if model is initialized
        if self.model is None:
            print(f"Client {self.client_id} waiting for global model (not yet initialized)...")
            return
        
        # Check for duplicate training signals
        if hasattr(self, 'last_training_round') and self.last_training_round == round_num:
            print(f"Client {self.client_id} ignoring duplicate start training for round {round_num}")
            return
        
        # Start training regardless of round number (generic approach)
        self.current_round = round_num
        if hasattr(self, 'last_training_round'):
            self.last_training_round = round_num
        print(f"\\nClient {self.client_id} starting training for round {round_num}...")
        self.train_local_model()'''
            content = content.replace(old_method, new_method)
            print(f"  ✓ Fixed handle_start_training for generic broadcast")
    
    # 4. Fix handle_start_evaluation - remove round validation
    pattern = r'    def handle_start_evaluation\(self.*?\n(?:.*?\n)*?        self\.evaluate_model\(\)(?:\n.*?(?:evaluated_rounds|print).*?)*'
    match = re.search(pattern, content)
    if match:
        old_method = match.group(0)
        if 'if round_num == self.current_round:' in old_method or 'skipping evaluation' in old_method:
            new_method = '''    def handle_start_evaluation(self, payload):
        """Start evaluation when server signals"""
        data = json.loads(payload.decode()) if isinstance(payload, bytes) else payload
        round_num = data.get('round', data.get('round_num', 0))
        
        # Check if model is initialized
        if self.model is None:
            print(f"Client {self.client_id} waiting for global model (not yet initialized)...")
            return
        
        # Check for duplicate evaluation signals
        if hasattr(self, 'evaluated_rounds') and round_num in self.evaluated_rounds:
            print(f"Client {self.client_id} ignoring duplicate evaluation for round {round_num}")
            return
        
        # Evaluate regardless of round number (generic approach)
        self.current_round = round_num
        print(f"Client {self.client_id} starting evaluation for round {round_num}...")
        self.evaluate_model()
        if hasattr(self, 'evaluated_rounds'):
            self.evaluated_rounds.add(round_num)
        print(f"Client {self.client_id} evaluation completed for round {round_num}")'''
            content = content.replace(old_method, new_method)
            print(f"  ✓ Fixed handle_start_evaluation for generic broadcast")
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Client file updated successfully")
        return True
    else:
        print(f"  ⏭️  No changes needed")
        return False

def fix_server_file(filepath):
    """Fix server file to always send model_config with global models"""
    print(f"Fixing server: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Find global model broadcast (non-quantized version) - make sure it includes model_config
    pattern = r'(global_model_message = \{\s*"round": self\.current_round,\s*"weights":.*?\n\s*\})'
    matches = list(re.finditer(pattern, content))
    
    for match in matches:
        old_dict = match.group(1)
        if '"model_config"' not in old_dict:
            new_dict = old_dict.replace(
                '}',
                ',\n                "model_config": self.model_config  # Always include for late-joiners\n            }'
            )
            content = content.replace(old_dict, new_dict)
            print(f"  ✓ Added model_config to global model broadcast (weights)")
    
    # Find global model broadcast (quantized version) - make sure it includes model_config
    pattern = r'(global_model_message = \{\s*"round": self\.current_round,\s*"quantized_data":.*?\n\s*\})'
    matches = list(re.finditer(pattern, content))
    
    for match in matches:
        old_dict = match.group(1)
        if '"model_config"' not in old_dict:
            new_dict = old_dict.replace(
                '}',
                ',\n                "model_config": self.model_config  # Always include for late-joiners\n            }'
            )
            content = content.replace(old_dict, new_dict)
            print(f"  ✓ Added model_config to global model broadcast (quantized)")
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Server file updated successfully")
        return True
    else:
        print(f"  ⏭️  No changes needed")
        return False

def main():
    print("=" * 80)
    print("GENERIC BROADCAST FIX - Apply to all servers and clients")
    print("=" * 80)
    print()
    
    # Fix all clients
    print("\n" + "=" * 80)
    print("FIXING CLIENT FILES")
    print("=" * 80)
    client_count = 0
    for client_file in CLIENTS:
        if os.path.exists(client_file):
            if fix_client_file(client_file):
                client_count += 1
        else:
            print(f"⚠️  File not found: {client_file}")
        print()
    
    # Fix all servers
    print("\n" + "=" * 80)
    print("FIXING SERVER FILES")
    print("=" * 80)
    server_count = 0
    for server_file in SERVERS:
        if os.path.exists(server_file):
            if fix_server_file(server_file):
                server_count += 1
        else:
            print(f"⚠️  File not found: {server_file}")
        print()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Clients updated: {client_count}/{len(CLIENTS)}")
    print(f"Servers updated: {server_count}/{len(SERVERS)}")
    print("\n✅ Generic broadcast model implemented!")
    print("\nKey changes:")
    print("  - Servers always send model_config with global models")
    print("  - Clients initialize from any global model if not initialized")
    print("  - Removed strict round validation - late-joiners can join anytime")
    print("  - Simplified to broadcast model: server sends → clients receive/train")

if __name__ == "__main__":
    main()
