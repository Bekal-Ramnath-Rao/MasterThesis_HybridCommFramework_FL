"""
Add quantization compression/decompression to all gRPC server files
"""
import re
import os

grpc_server_files = [
    "Server/Emotion_Recognition/FL_Server_gRPC.py",
    "Server/MentalState_Recognition/FL_Server_gRPC.py",
    "Server/Temperature_Regulation/FL_Server_gRPC.py",
]

def add_grpc_server_get_global_model_compression(filepath):
    """Add compression when sending global model"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for GetGlobalModel method - serialize weights
    pattern = r"([ \t]+)serialized_weights = self\.serialize_weights\(self\.global_weights\)\n([ \t]+)model_config_json = json\.dumps\(self\.model_config\) if self\.current_round == 0 else ''\n([ \t]+)\n([ \t]+)return federated_learning_pb2\.GlobalModel\(\n([ \t]+)round=self\.current_round,\n([ \t]+)weights=serialized_weights,"
    
    replacement = r"""\1# Compress or serialize global weights
\1if self.quantization_handler is not None:
\1    compressed_data = self.quantization_handler.compress_global_model(self.global_weights)
\1    stats = self.quantization_handler.quantizer.get_compression_stats(self.global_weights, compressed_data)
\1    print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")
\1    serialized_weights = compressed_data
\1else:
\1    serialized_weights = self.serialize_weights(self.global_weights)
\2model_config_json = json.dumps(self.model_config) if self.current_round == 0 else ''
\3
\4return federated_learning_pb2.GlobalModel(
\5round=self.current_round,
\6weights=serialized_weights,"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"  ✓ Added GetGlobalModel compression to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def add_grpc_server_send_model_update_decompression(filepath):
    """Add decompression when receiving client updates"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern for SendModelUpdate method - deserialize weights
    pattern = r"([ \t]+# Deserialize weights\n[ \t]+)weights = self\.deserialize_weights\(request\.weights\)"
    
    replacement = r"""\1# Decompress or deserialize client weights
            if self.quantization_handler is not None and request.weights:
                try:
                    weights = self.quantization_handler.decompress_client_update(request.client_id, request.weights)
                    print(f"Server: Received and decompressed update from client {request.client_id}")
                except:
                    # Fallback to regular deserialization
                    weights = self.deserialize_weights(request.weights)
            else:
                weights = self.deserialize_weights(request.weights)"""
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content, count=1)
        print(f"  ✓ Added SendModelUpdate decompression to {filepath}")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    print("="*80)
    print("Adding Quantization to gRPC Server Files")
    print("="*80)
    
    total_modified = 0
    
    for filepath in grpc_server_files:
        if os.path.exists(filepath):
            print(f"\nProcessing {filepath}...")
            modified = False
            
            if add_grpc_server_get_global_model_compression(filepath):
                modified = True
            
            if add_grpc_server_send_model_update_decompression(filepath):
                modified = True
            
            if modified:
                total_modified += 1
        else:
            print(f"  ⚠ File not found: {filepath}")
    
    print(f"\n{'='*80}")
    print(f"✅ Completed! Modified {total_modified} gRPC server files")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
