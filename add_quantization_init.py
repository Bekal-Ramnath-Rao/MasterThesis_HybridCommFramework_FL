"""
Complete Quantization Integration Script
Adds initialization and compression logic to all FL client and server files
"""

import os
import re
from pathlib import Path


def add_client_quantization_init(filepath):
    """Add quantization initialization to client __init__"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if already has quantization init
    if 'self.quantizer' in content:
        print(f"  Already has quantization init")
        return False
    
    # Find self.model = None and add quantization init after it
    pattern = r'(self\.model = None)'
    
    replacement = """self.model = None
        
        # Initialize quantization compression
        use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() == "true"
        if use_quantization:
            self.quantizer = Quantization(QuantizationConfig())
            print(f"Client {self.client_id}: Quantization enabled")
        else:
            self.quantizer = None
            print(f"Client {self.client_id}: Quantization disabled")"""
    
    content = re.sub(pattern, replacement, content, count=1)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Added quantization init")
    return True


def add_server_quantization_init(filepath):
    """Add quantization handler initialization to server __init__"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Skip if already has quantization handler
    if 'self.quantization_handler' in content:
        print(f"  Already has quantization handler init")
        return False
    
    # Find self.initialize_global_model() and add quantization handler before it
    pattern = r'(\n\s+# Initialize global model\s+self\.initialize_global_model\(\))'
    
    replacement = """
        
        # Initialize quantization handler
        use_quantization = os.getenv("USE_QUANTIZATION", "false").lower() == "true"
        if use_quantization and QUANTIZATION_AVAILABLE:
            self.quantization_handler = ServerQuantizationHandler(QuantizationConfig())
            print("Server: Quantization enabled")
        else:
            self.quantization_handler = None
            if use_quantization and not QUANTIZATION_AVAILABLE:
                print("Server: Quantization requested but not available")
            else:
                print("Server: Quantization disabled")
        
        # Initialize global model
        self.initialize_global_model()"""
    
    content = re.sub(pattern, replacement, content, count=1)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Added quantization handler init")
    return True


def main():
    """Main function to add initialization to all files"""
    print("="*70)
    print("Adding Quantization Initialization to All Files")
    print("="*70)
    
    base_path = Path(__file__).parent
    use_cases = ['Emotion_Recognition', 'MentalState_Recognition', 'Temperature_Regulation']
    protocols = ['MQTT', 'AMQP', 'gRPC', 'QUIC', 'DDS']
    
    # Process clients
    print("\n" + "="*70)
    print("ADDING CLIENT INITIALIZATION")
    print("="*70)
    
    for use_case in use_cases:
        print(f"\n{use_case}:")
        for protocol in protocols:
            client_file = base_path / 'Client' / use_case / f'FL_Client_{protocol}.py'
            if client_file.exists():
                print(f"  Processing {client_file.name}...")
                add_client_quantization_init(str(client_file))
    
    # Process servers
    print("\n" + "="*70)
    print("ADDING SERVER INITIALIZATION")
    print("="*70)
    
    for use_case in use_cases:
        print(f"\n{use_case}:")
        for protocol in protocols:
            server_file = base_path / 'Server' / use_case / f'FL_Server_{protocol}.py'
            if server_file.exists():
                print(f"  Processing {server_file.name}...")
                add_server_quantization_init(str(server_file))
    
    print("\n" + "="*70)
    print("INITIALIZATION COMPLETE")
    print("="*70)
    print("\nNote: Compression/decompression logic must still be added manually.")
    print("See FL_Client_MQTT.py and FL_Server_MQTT.py for reference patterns.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
