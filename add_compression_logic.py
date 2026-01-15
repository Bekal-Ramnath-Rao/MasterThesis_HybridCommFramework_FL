"""
Add compression/decompression logic to all FL implementations
Handles clients and servers for all protocols
"""

import os
import re
from pathlib import Path


# ============ CLIENT MODIFICATIONS ============

CLIENT_SEND_COMPRESSION_PATTERN = r'(# Send model update.*?\n\s+update_message = \{[\s\S]*?"weights": self\.serialize_weights\(updated_weights\),)'

CLIENT_SEND_COMPRESSION_REPLACEMENT = '''# Compress weights if quantization is enabled
        if self.quantizer is not None:
            compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
            stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
            print(f"Client {self.client_id}: Compressed weights - "
                  f"Ratio: {stats['compression_ratio']:.2f}x, "
                  f"Size: {stats['compressed_size_mb']:.2f}MB")
            
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "compressed_data": compressed_data,
                "num_samples": num_samples,
                "metrics": metrics
            }
        else:
            # Send model update without compression
            update_message = {
                "client_id": self.client_id,
                "round": self.current_round,
                "weights": self.serialize_weights(updated_weights),'''

# Pattern for decompressing global model  
CLIENT_RECEIVE_PATTERN = r'(encoded_weights = data\[\'weights\'\]\s+weights = self\.deserialize_weights\(encoded_weights\))'

CLIENT_RECEIVE_REPLACEMENT = '''# Check if weights are quantized
            if 'quantized_data' in data and self.quantizer is not None:
                compressed_data = data['quantized_data']
                weights = self.quantizer.decompress(compressed_data)
                if round_num > 0:
                    print(f"Client {self.client_id}: Received and decompressed quantized global model")
            else:
                encoded_weights = data['weights']
                weights = self.deserialize_weights(encoded_weights)'''


# ============ SERVER MODIFICATIONS ============

# Pattern for decompressing client updates
SERVER_RECEIVE_PATTERN = r'(self\.client_updates\[client_id\] = \{\s+[\'"]weights[\'"]: self\.deserialize_weights\(data\[[\'"]weights[\'"]\]\),)'

SERVER_RECEIVE_REPLACEMENT = '''# Check if update is compressed
            if 'compressed_data' in data and self.quantization_handler is not None:
                weights = self.quantization_handler.decompress_client_update(
                    client_id, 
                    data['compressed_data']
                )
                print(f"Received and decompressed update from client {client_id}")
            else:
                weights = self.deserialize_weights(data['weights'])
            
            self.client_updates[client_id] = {
                'weights': weights,'''

# Pattern for compressing global model after aggregation
SERVER_SEND_AGGREGATED_PATTERN = r'(self\.global_weights = aggregated_weights\s+# Send global model.*?\n\s+global_model_message = \{[\s\S]*?"weights": self\.serialize_weights\(self\.global_weights\))'

SERVER_SEND_AGGREGATED_REPLACEMENT = '''self.global_weights = aggregated_weights
        
        # Optionally compress before sending
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            global_model_message = {
                "round": self.current_round,
                "quantized_data": compressed_data
            }
        else:
            global_model_message = {
                "round": self.current_round,
                "weights": self.serialize_weights(self.global_weights)'''

# Pattern for compressing initial model distribution
SERVER_SEND_INITIAL_PATTERN = r'(# Send initial global model.*?\n\s+initial_model_message = \{[\s\S]*?"weights": self\.serialize_weights\(self\.global_weights\),)'

SERVER_SEND_INITIAL_REPLACEMENT = '''# Optionally compress global model
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.get_compression_stats(self.global_weights, compressed_data)
            print(f"Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
            
            initial_model_message = {
                "round": 0,
                "quantized_data": compressed_data,
                "model_config": self.model_config
            }
        else:
            # Send initial global model with architecture configuration
            initial_model_message = {
                "round": 0,
                "weights": self.serialize_weights(self.global_weights),'''


def add_client_compression_logic(filepath):
    """Add compression/decompression logic to client file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Add send compression
    if '"weights": self.serialize_weights(updated_weights),' in content and 'compressed_data' not in content:
        content = re.sub(CLIENT_SEND_COMPRESSION_PATTERN, CLIENT_SEND_COMPRESSION_REPLACEMENT, content, flags=re.DOTALL)
        print(f"    ✓ Added send compression logic")
        modified = True
    elif 'compressed_data' in content and 'self.quantizer.compress' in content:
        print(f"    Already has send compression")
    
    # Add receive decompression
    if 'encoded_weights = data[\'weights\']' in content and 'quantized_data' not in content:
        content = re.sub(CLIENT_RECEIVE_PATTERN, CLIENT_RECEIVE_REPLACEMENT, content)
        print(f"    ✓ Added receive decompression logic")
        modified = True
    elif 'quantized_data' in content:
        print(f"    Already has receive decompression")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def add_server_compression_logic(filepath):
    """Add compression/decompression logic to server file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Add receive decompression
    if "'weights': self.deserialize_weights(data['weights'])" in content:
        if 'compressed_data' not in content or 'quantization_handler.decompress_client_update' not in content:
            content = re.sub(SERVER_RECEIVE_PATTERN, SERVER_RECEIVE_REPLACEMENT, content)
            print(f"    ✓ Added client update decompression logic")
            modified = True
    if "'weights': self.deserialize_weights(data['weights'])" in content and 'quantization_handler.decompress_client_update' in content:
        print(f"    Already has client update decompression")
    
    # Add aggregated model compression
    if 'self.global_weights = aggregated_weights' in content:
        if re.search(SERVER_SEND_AGGREGATED_PATTERN, content, re.DOTALL):
            content = re.sub(SERVER_SEND_AGGREGATED_PATTERN, SERVER_SEND_AGGREGATED_REPLACEMENT, content, flags=re.DOTALL)
            print(f"    ✓ Added aggregated model compression logic")
            modified = True
        elif 'quantization_handler.compress_global_model' in content:
            print(f"    Already has aggregated model compression")
    
    # Add initial model compression
    if 'initial_model_message' in content:
        if re.search(SERVER_SEND_INITIAL_PATTERN, content, re.DOTALL):
            content = re.sub(SERVER_SEND_INITIAL_PATTERN, SERVER_SEND_INITIAL_REPLACEMENT, content, flags=re.DOTALL)
            print(f"    ✓ Added initial model compression logic")
            modified = True
        elif '"quantized_data": compressed_data' in content and 'initial_model_message' in content:
            print(f"    Already has initial model compression")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False


def main():
    """Main function"""
    print("="*70)
    print("Adding Compression/Decompression Logic to All Files")
    print("="*70)
    print("\nThis script adds quantization compression logic to:")
    print("  - Client: compress before send, decompress on receive")
    print("  - Server: decompress client updates, compress global model")
    print("="*70)
    
    base_path = Path(__file__).parent
    use_cases = ['Emotion_Recognition', 'MentalState_Recognition', 'Temperature_Regulation']
    protocols = ['MQTT', 'AMQP', 'gRPC', 'QUIC', 'DDS']
    
    # Process clients
    print("\n" + "="*70)
    print("ADDING CLIENT COMPRESSION/DECOMPRESSION")
    print("="*70)
    
    for use_case in use_cases:
        print(f"\n{use_case}:")
        for protocol in protocols:
            client_file = base_path / 'Client' / use_case / f'FL_Client_{protocol}.py'
            if client_file.exists():
                print(f"  {protocol}:")
                add_client_compression_logic(str(client_file))
    
    # Process servers
    print("\n" + "="*70)
    print("ADDING SERVER COMPRESSION/DECOMPRESSION")
    print("="*70)
    
    for use_case in use_cases:
        print(f"\n{use_case}:")
        for protocol in protocols:
            server_file = base_path / 'Server' / use_case / f'FL_Server_{protocol}.py'
            if server_file.exists():
                print(f"  {protocol}:")
                add_server_compression_logic(str(server_file))
    
    print("\n" + "="*70)
    print("COMPRESSION LOGIC INTEGRATION COMPLETE")
    print("="*70)
    print("\nQuantization is now integrated into all protocols!")
    print("Enable/disable with: $env:USE_QUANTIZATION=\"true\" or \"false\"")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
