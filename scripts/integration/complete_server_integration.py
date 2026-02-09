"""
Apply the same quantization integration patterns to all QUIC and DDS server files
"""
import re

files_to_process = [
    # MentalState Recognition
    ("Server/MentalState_Recognition/FL_Server_QUIC.py", "QUIC"),
    ("Server/MentalState_Recognition/FL_Server_DDS.py", "DDS"),
    # Temperature Regulation
    ("Server/Temperature_Regulation/FL_Server_QUIC.py", "QUIC"),
    ("Server/Temperature_Regulation/FL_Server_DDS.py", "DDS"),
]

def add_quic_server_decompression(filepath):
    """Add decompression for client updates in QUIC server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for handle_client_update
    pattern = r"(    async def handle_client_update\(self, message\):\n        \"\"\"Handle model update from client\"\"\"\n        client_id = message\['client_id'\]\n        round_num = message\['round'\]\n        \n        if round_num == self\.current_round:\n            self\.client_updates\[client_id\] = \{\n                )'weights': self\.deserialize_weights\(message\['weights'\]\),"
    
    replacement = r"""\1# Decompress or deserialize client weights
            if 'compressed_data' in message and self.quantization_handler is not None:
                weights = self.quantization_handler.decompress_client_update(message['client_id'], message['compressed_data'])
                print(f"Server: Received and decompressed update from client {message['client_id']}")
            else:
                weights = self.deserialize_weights(message['weights'])
            
            self.client_updates[client_id] = {
                'weights': weights,"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"  ✓ Added client update decompression to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def add_quic_server_initial_compression(filepath):
    """Add compression for initial global model in QUIC server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for distribute_initial_model
    pattern = r"([ \t]+)await self\.broadcast_message\(\{\n([ \t]+)'type': 'global_model',\n([ \t]+)'round': 0,\n([ \t]+)'weights': self\.serialize_weights\(self\.global_weights\),\n([ \t]+)'model_config': model_config\n([ \t]+)\}\)"
    
    replacement = r"""\1# Prepare global model (compress if quantization enabled)
\1if self.quantization_handler is not None:
\1    compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
\1    stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
\1    print(f"Server: Compressed initial global model - Ratio: {stats['compression_ratio']:.2f}x")
\1    weights_data = compressed_data
\1    weights_key = 'quantized_data'
\1else:
\1    weights_data = self.serialize_weights(self.global_weights)
\1    weights_key = 'weights'
\1
\1await self.broadcast_message({
\2'type': 'global_model',
\3'round': 0,
\1weights_key: weights_data,
\5'model_config': model_config
\6})"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"  ✓ Added initial model compression to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def add_quic_server_aggregated_compression(filepath):
    """Add compression for aggregated global model in QUIC server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for sending aggregated model
    pattern = r"([ \t]+self\.global_weights = aggregated_weights\n        \n        )await self\.broadcast_message\(\{\n([ \t]+)'type': 'global_model',\n([ \t]+)'round': self\.current_round,\n([ \t]+)'weights': self\.serialize_weights\(self\.global_weights\)\n([ \t]+)\}\)"
    
    replacement = r"""\1# Prepare global model (compress if quantization enabled)
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
            weights_data = compressed_data
            weights_key = 'quantized_data'
        else:
            weights_data = self.serialize_weights(self.global_weights)
            weights_key = 'weights'
        
        await self.broadcast_message({
\2'type': 'global_model',
\3'round': self.current_round,
        weights_key: weights_data
\5})"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"  ✓ Added aggregated model compression to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def add_dds_server_decompression(filepath):
    """Add decompression for client updates in DDS server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for handle_model_update
    pattern = r"([ \t]+if client_id not in self\.client_updates:\n                    )self\.client_updates\[client_id\] = \{\n([ \t]+)'weights': self\.deserialize_weights\(sample\.weights\),"
    
    replacement = r"""\1# Decompress or deserialize client weights
                    if self.quantization_handler is not None:
                        try:
                            weights = self.quantization_handler.decompress_client_update(sample.client_id, bytes(sample.weights))
                            print(f"Server: Received and decompressed update from client {sample.client_id}")
                        except:
                            # Fallback to regular deserialization
                            weights = self.deserialize_weights(sample.weights)
                    else:
                        weights = self.deserialize_weights(sample.weights)
                    
                    self.client_updates[client_id] = {
\2'weights': weights,"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"  ✓ Added client update decompression to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def add_dds_server_initial_compression(filepath):
    """Add compression for initial global model in DDS server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for distribute_initial_model
    pattern = r"([ \t]+# Send initial global model to all clients\n        )initial_model = GlobalModel\(\n([ \t]+)round=0,  # Round 0 = initial model distribution\n([ \t]+)weights=self\.serialize_weights\(self\.global_weights\),\n([ \t]+)model_config_json=json\.dumps\(model_config\)\n([ \t]+)\)\n([ \t]+)self\.writers\['global_model'\]\.write\(initial_model\)"
    
    replacement = r"""# Compress or serialize global weights
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed initial global model - Ratio: {stats['compression_ratio']:.2f}x")
            serialized_weights = compressed_data
        else:
            serialized_weights = self.serialize_weights(self.global_weights)
        
        \1initial_model = GlobalModel(
\2round=0,  # Round 0 = initial model distribution
\2weights=serialized_weights,
\4model_config_json=json.dumps(model_config)
\5)
\6self.writers['global_model'].write(initial_model)"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"  ✓ Added initial model compression to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def add_dds_server_aggregated_compression(filepath):
    """Add compression for aggregated global model in DDS server"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for sending aggregated model
    pattern = r"([ \t]+print\(f\"Sending global model to clients\.\.\.\\n\"\)\n        \n        # Publish global model\n        )global_model = GlobalModel\(\n([ \t]+)round=self\.current_round,\n([ \t]+)weights=self\.serialize_weights\(self\.global_weights\)\n([ \t]+)\)\n([ \t]+)self\.writers\['global_model'\]\.write\(global_model\)"
    
    replacement = r"""\1# Compress or serialize global weights
        if self.quantization_handler is not None:
            compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
            stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
            print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
            serialized_weights = compressed_data
        else:
            serialized_weights = self.serialize_weights(self.global_weights)
        
        global_model = GlobalModel(
\2round=self.current_round,
\2weights=serialized_weights
\4)
\5self.writers['global_model'].write(global_model)"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"  ✓ Added aggregated model compression to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    print("="*80)
    print("Adding Quantization to Remaining QUIC and DDS Server Files")
    print("="*80)
    
    total_modified = 0
    
    for filepath, protocol in files_to_process:
        print(f"\nProcessing {filepath}...")
        
        if protocol == "QUIC":
            modified = False
            if add_quic_server_decompression(filepath):
                modified = True
            if add_quic_server_initial_compression(filepath):
                modified = True
            if add_quic_server_aggregated_compression(filepath):
                modified = True
            if modified:
                total_modified += 1
        
        elif protocol == "DDS":
            modified = False
            if add_dds_server_decompression(filepath):
                modified = True
            if add_dds_server_initial_compression(filepath):
                modified = True
            if add_dds_server_aggregated_compression(filepath):
                modified = True
            if modified:
                total_modified += 1
    
    print(f"\n{'='*80}")
    print(f"✅ Completed! Modified {total_modified} server files")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
