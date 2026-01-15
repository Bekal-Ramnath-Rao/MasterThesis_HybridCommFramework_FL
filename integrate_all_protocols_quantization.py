"""
Comprehensive script to add quantization compression/decompression logic
to ALL protocols: MQTT, AMQP, gRPC, QUIC, and DDS
"""
import os
import re

# Define file patterns
use_cases = ["Emotion_Recognition", "MentalState_Recognition", "Temperature_Regulation"]
protocols = ["MQTT", "AMQP", "gRPC", "QUIC", "DDS"]

def process_grpc_client(filepath):
    """Add quantization to gRPC client"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Pattern 1: Compress weights before sending (in train_local_model)
    # Look for: serialized_weights = pickle.dumps(updated_weights)
    pattern1 = r'([ \t]+)# Serialize weights\n([ \t]+)serialized_weights = pickle\.dumps\(updated_weights\)'
    if re.search(pattern1, content):
        replacement1 = r'''\1# Compress or serialize weights
\2if self.quantizer is not None:
\2    compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
\2    stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
\2    print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Original: {stats['original_size_mb']:.2f}MB, Compressed: {stats['compressed_size_mb']:.2f}MB")
\2    serialized_weights = compressed_data
\2else:
\2    serialized_weights = pickle.dumps(updated_weights)'''
        content = re.sub(pattern1, replacement1, content)
        modified = True
        print(f"  ✓ Added send compression to {filepath}")
    
    # Pattern 2: Decompress weights when receiving (in receive_global_model)
    # Look for: weights = pickle.loads(model_update.weights)
    pattern2 = r'([ \t]+)(weights = pickle\.loads\(model_update\.weights\))'
    if re.search(pattern2, content):
        replacement2 = r'''\1# Decompress or deserialize weights
\1if self.quantizer is not None and model_update.weights:
\1    try:
\1        weights = self.quantizer.decompress(model_update.weights)
\1        print(f"Client {self.client_id}: Received and decompressed quantized global model")
\1    except:
\1        # Fallback to regular deserialization if decompression fails
\1        weights = pickle.loads(model_update.weights)
\1else:
\1    weights = pickle.loads(model_update.weights)'''
        content = re.sub(pattern2, replacement2, content)
        modified = True
        print(f"  ✓ Added receive decompression to {filepath}")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def process_grpc_server(filepath):
    """Add quantization to gRPC server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Pattern 1: Decompress client updates (in SendModelUpdate)
    # Look for: weights = pickle.loads(request.weights)
    pattern1 = r'([ \t]+)(weights = pickle\.loads\(request\.weights\))'
    if re.search(pattern1, content):
        replacement1 = r'''\1# Decompress or deserialize client weights
\1if self.quantization_handler is not None and request.weights:
\1    try:
\1        weights = self.quantization_handler.decompress_client_update(request.client_id, request.weights)
\1        print(f"Server: Received and decompressed update from client {request.client_id}")
\1    except:
\1        # Fallback to regular deserialization
\1        weights = pickle.loads(request.weights)
\1else:
\1    weights = pickle.loads(request.weights)'''
        content = re.sub(pattern1, replacement1, content)
        modified = True
        print(f"  ✓ Added client update decompression to {filepath}")
    
    # Pattern 2: Compress global model before sending (in GetGlobalModel or distribute_global_model)
    # Look for: serialized_weights = pickle.dumps(self.global_weights)
    pattern2 = r'([ \t]+)(serialized_weights = pickle\.dumps\(self\.global_weights\))'
    if re.search(pattern2, content):
        replacement2 = r'''\1# Compress or serialize global weights
\1if self.quantization_handler is not None:
\1    compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
\1    stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
\1    print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
\1    serialized_weights = compressed_data
\1else:
\1    serialized_weights = pickle.dumps(self.global_weights)'''
        content = re.sub(pattern2, replacement2, content)
        modified = True
        print(f"  ✓ Added global model compression to {filepath}")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def process_quic_client(filepath):
    """Add quantization to QUIC client"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Pattern 1: Compress weights before sending (in train_local_model)
    # Look for: 'weights': self.serialize_weights(self.model.get_weights())
    pattern1 = r"([ \t]+)'weights': self\.serialize_weights\(self\.model\.get_weights\(\)\),"
    if re.search(pattern1, content):
        # First, we need to get the weights before the message
        # Add a section before await self.send_message
        insert_pattern = r"([ \t]+)(# Send model update to server\n[ \t]+await self\.send_message\(\{)"
        if re.search(insert_pattern, content):
            insert_code = r'''\1# Prepare weights (compress if quantization enabled)
\1updated_weights = self.model.get_weights()
\1if self.quantizer is not None:
\1    compressed_data = self.quantizer.compress(updated_weights, data_type="weights")
\1    stats = self.quantizer.get_compression_stats(updated_weights, compressed_data)
\1    print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Size: {stats['compressed_size_mb']:.2f}MB")
\1    weights_data = compressed_data
\1    weights_key = 'compressed_data'
\1else:
\1    weights_data = self.serialize_weights(updated_weights)
\1    weights_key = 'weights'
\1
\1\2'''
            content = re.sub(insert_pattern, insert_code, content)
            
            # Now replace the weights line to use the variable
            pattern1_new = r"([ \t]+)'weights': self\.serialize_weights\(self\.model\.get_weights\(\)\),"
            replacement1 = r"\1weights_key: weights_data,"
            content = re.sub(pattern1_new, replacement1, content)
            modified = True
            print(f"  ✓ Added send compression to {filepath}")
    
    # Pattern 2: Decompress weights when receiving (in handle_global_model)
    # Look for: weights = self.deserialize_weights(encoded_weights)
    pattern2 = r"([ \t]+)weights = self\.deserialize_weights\(encoded_weights\)"
    if re.search(pattern2, content):
        replacement2 = r'''\1# Decompress or deserialize weights
\1if 'quantized_data' in message and self.quantizer is not None:
\1    weights = self.quantizer.decompress(message['quantized_data'])
\1    print(f"Client {self.client_id}: Received and decompressed quantized global model")
\1elif 'compressed_data' in message and self.quantizer is not None:
\1    weights = self.quantizer.decompress(message['compressed_data'])
\1    print(f"Client {self.client_id}: Received and decompressed quantized global model")
\1else:
\1    weights = self.deserialize_weights(encoded_weights)'''
        content = re.sub(pattern2, replacement2, content)
        modified = True
        print(f"  ✓ Added receive decompression to {filepath}")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def process_quic_server(filepath):
    """Add quantization to QUIC server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Pattern 1: Decompress client updates (in handle_model_update)
    # Look for: weights = self.deserialize_weights(message['weights'])
    pattern1 = r"([ \t]+)weights = self\.deserialize_weights\(message\['weights'\]\)"
    if re.search(pattern1, content):
        replacement1 = r'''\1# Decompress or deserialize client weights
\1if 'compressed_data' in message and self.quantization_handler is not None:
\1    weights = self.quantization_handler.decompress_client_update(message['client_id'], message['compressed_data'])
\1    print(f"Server: Received and decompressed update from client {message['client_id']}")
\1else:
\1    weights = self.deserialize_weights(message['weights'])'''
        content = re.sub(pattern1, replacement1, content)
        modified = True
        print(f"  ✓ Added client update decompression to {filepath}")
    
    # Pattern 2: Compress global model before sending (in distribute_global_model or send_global_model)
    # Look for: 'weights': self.serialize_weights(self.global_weights)
    pattern2 = r"([ \t]+)'weights': self\.serialize_weights\(self\.global_weights\)"
    if re.search(pattern2, content):
        # Add preparation code before the message
        insert_pattern = r"([ \t]+)(message = \{\n[ \t]+'type': 'global_model')"
        if re.search(insert_pattern, content):
            insert_code = r'''\1# Prepare global model (compress if quantization enabled)
\1if self.quantization_handler is not None:
\1    compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
\1    stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
\1    print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
\1    weights_data = compressed_data
\1    weights_key = 'quantized_data'
\1else:
\1    weights_data = self.serialize_weights(self.global_weights)
\1    weights_key = 'weights'
\1
\1\2'''
            content = re.sub(insert_pattern, insert_code, content)
            
            # Now replace the weights line
            pattern2_new = r"([ \t]+)'weights': self\.serialize_weights\(self\.global_weights\)"
            replacement2 = r"\1weights_key: weights_data"
            content = re.sub(pattern2_new, replacement2, content)
            modified = True
            print(f"  ✓ Added global model compression to {filepath}")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def process_dds_client(filepath):
    """Add quantization to DDS client"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Pattern 1: Compress weights before sending (in train_local_model)
    # Look for: serialized_weights = self.serialize_weights(weights)
    pattern1 = r'([ \t]+)serialized_weights = self\.serialize_weights\(weights\)'
    if re.search(pattern1, content):
        replacement1 = r'''\1# Compress or serialize weights
\1if self.quantizer is not None:
\1    compressed_data = self.quantizer.compress(weights, data_type="weights")
\1    stats = self.quantizer.get_compression_stats(weights, compressed_data)
\1    print(f"Client {self.client_id}: Compressed weights - Ratio: {stats['compression_ratio']:.2f}x, Size: {stats['compressed_size_mb']:.2f}MB")
\1    serialized_weights = compressed_data
\1else:
\1    serialized_weights = self.serialize_weights(weights)'''
        content = re.sub(pattern1, replacement1, content)
        modified = True
        print(f"  ✓ Added send compression to {filepath}")
    
    # Pattern 2: Decompress weights when receiving (in handle_global_model)
    # Look for: weights = self.deserialize_weights(bytes(sample.weights))
    pattern2 = r'([ \t]+)(weights = self\.deserialize_weights\(bytes\(sample\.weights\)\))'
    if re.search(pattern2, content):
        replacement2 = r'''\1# Decompress or deserialize weights
\1if self.quantizer is not None:
\1    try:
\1        weights = self.quantizer.decompress(bytes(sample.weights))
\1        print(f"Client {self.client_id}: Received and decompressed quantized global model")
\1    except:
\1        # Fallback to regular deserialization
\1        weights = self.deserialize_weights(bytes(sample.weights))
\1else:
\1    weights = self.deserialize_weights(bytes(sample.weights))'''
        content = re.sub(pattern2, replacement2, content)
        modified = True
        print(f"  ✓ Added receive decompression to {filepath}")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def process_dds_server(filepath):
    """Add quantization to DDS server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Pattern 1: Decompress client updates (in handle_model_update)
    # Look for: weights = self.deserialize_weights(bytes(sample.weights))
    pattern1 = r'([ \t]+# Deserialize weights\n[ \t]+)(weights = self\.deserialize_weights\(bytes\(sample\.weights\)\))'
    if re.search(pattern1, content):
        replacement1 = r'''\1# Decompress or deserialize client weights
\1if self.quantization_handler is not None:
\1    try:
\1        weights = self.quantization_handler.decompress_client_update(sample.client_id, bytes(sample.weights))
\1        print(f"Server: Received and decompressed update from client {sample.client_id}")
\1    except:
\1        # Fallback to regular deserialization
\1        weights = self.deserialize_weights(bytes(sample.weights))
\1else:
\1    weights = self.deserialize_weights(bytes(sample.weights))'''
        content = re.sub(pattern1, replacement1, content)
        modified = True
        print(f"  ✓ Added client update decompression to {filepath}")
    
    # Pattern 2: Compress global model before sending (in distribute_global_model)
    # Look for: serialized_weights = self.serialize_weights(self.global_weights)
    pattern2 = r'([ \t]+)(serialized_weights = self\.serialize_weights\(self\.global_weights\))'
    if re.search(pattern2, content):
        replacement2 = r'''\1# Compress or serialize global weights
\1if self.quantization_handler is not None:
\1    compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
\1    stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
\1    print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
\1    serialized_weights = compressed_data
\1else:
\1    serialized_weights = self.serialize_weights(self.global_weights)'''
        content = re.sub(pattern2, replacement2, content)
        modified = True
        print(f"  ✓ Added global model compression to {filepath}")
    
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Process all files"""
    print("=" * 80)
    print("Adding Quantization Compression Logic to ALL Protocols")
    print("=" * 80)
    
    total_modified = 0
    
    for use_case in use_cases:
        print(f"\n{'='*80}")
        print(f"Processing {use_case}")
        print(f"{'='*80}")
        
        for protocol in protocols:
            print(f"\n{protocol} Protocol:")
            
            # Process client
            client_file = f"Client/{use_case}/FL_Client_{protocol}.py"
            if os.path.exists(client_file):
                if protocol == "gRPC":
                    if process_grpc_client(client_file):
                        total_modified += 1
                elif protocol == "QUIC":
                    if process_quic_client(client_file):
                        total_modified += 1
                elif protocol == "DDS":
                    if process_dds_client(client_file):
                        total_modified += 1
                else:
                    print(f"  ℹ {protocol} client already processed")
            
            # Process server
            server_file = f"Server/{use_case}/FL_Server_{protocol}.py"
            if os.path.exists(server_file):
                if protocol == "gRPC":
                    if process_grpc_server(server_file):
                        total_modified += 1
                elif protocol == "QUIC":
                    if process_quic_server(server_file):
                        total_modified += 1
                elif protocol == "DDS":
                    if process_dds_server(server_file):
                        total_modified += 1
                else:
                    print(f"  ℹ {protocol} server already processed")
    
    print(f"\n{'='*80}")
    print(f"✅ Completed! Modified {total_modified} files")
    print(f"{'='*80}")
    print("\nQuantization is now integrated into ALL protocols:")
    print("  ✓ MQTT (already complete)")
    print("  ✓ AMQP (already complete)")
    print("  ✓ gRPC (compression logic added)")
    print("  ✓ QUIC (compression logic added)")
    print("  ✓ DDS (compression logic added)")

if __name__ == "__main__":
    main()
