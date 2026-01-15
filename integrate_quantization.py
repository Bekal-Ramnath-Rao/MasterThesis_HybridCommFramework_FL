"""
Script to integrate quantization into all FL client and server implementations
Applies the same quantization integration pattern to all protocols and use cases
"""

import os
import re
import shutil
from pathlib import Path


def backup_file(filepath):
    """Create backup of file before modification"""
    backup_path = f"{filepath}.backup"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        print(f"  Backup created: {backup_path}")


def add_quantization_imports_to_client(filepath):
    """Add quantization imports to client file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already has quantization imports
    if 'from quantization_client import' in content or 'from quantization import' in content:
        print(f"  Skipping {filepath} - already has quantization imports")
        return False
    
    # Find the import section
    import_pattern = r'(import paho\.mqtt\.client as mqtt|import grpc|from cyclonedds|import pika)'
    
    # Add imports after existing imports
    quantization_imports = """
# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

from quantization_client import Quantization, QuantizationConfig
"""
    
    # Add sys import if not present
    if 'import sys' not in content:
        content = content.replace('import os\n', 'import os\nimport sys\n')
    
    # Add quantization imports
    match = re.search(import_pattern, content)
    if match:
        insert_pos = content.find('\n', match.end()) + 1
        content = content[:insert_pos] + quantization_imports + '\n' + content[insert_pos:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Added quantization imports to {filepath}")
    return True


def add_quantization_to_client_init(filepath):
    """Add quantization initialization to client __init__"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already initialized
    if 'self.quantizer' in content:
        print(f"  Skipping init - already has quantizer")
        return False
    
    # Find the __init__ method and add quantization initialization
    init_pattern = r'(self\.model = None)'
    
    quantization_init = """self.model = None
        
        # Initialize quantization compression
        use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
        if use_quantization:
            self.quantizer = Quantization(QuantizationConfig())
            print(f"Client {self.client_id}: Quantization enabled")
        else:
            self.quantizer = None
            print(f"Client {self.client_id}: Quantization disabled")"""
    
    content = re.sub(init_pattern, quantization_init, content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Added quantization init to {filepath}")
    return True


def integrate_client_file(filepath):
    """Integrate quantization into a client file"""
    print(f"\nProcessing client: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return
    
    backup_file(filepath)
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Check if already integrated
    if any('self.quantizer' in line for line in lines):
        print(f"  Already integrated")
        return
    
    # Add imports
    add_quantization_imports_to_client(filepath)
    
    print(f"  ✓ Integrated quantization into {filepath}")


def add_quantization_imports_to_server(filepath):
    """Add quantization imports to server file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already has quantization imports
    if 'ServerQuantizationHandler' in content or 'from quantization_server import' in content:
        print(f"  Skipping {filepath} - already has quantization imports")
        return False
    
    # Add imports
    quantization_imports = """
# Add Compression_Technique to path
compression_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Compression_Technique')
if compression_path not in sys.path:
    sys.path.insert(0, compression_path)

try:
    from quantization_server import ServerQuantizationHandler, QuantizationConfig
    QUANTIZATION_AVAILABLE = True
except ImportError:
    print("Warning: Quantization module not available")
    QUANTIZATION_AVAILABLE = False
"""
    
    # Add sys import if not present
    if 'import sys' not in content:
        content = content.replace('import os\n', 'import os\nimport sys\n')
    
    # Add after imports
    import_pattern = r'(from pathlib import Path)'
    match = re.search(import_pattern, content)
    if match:
        insert_pos = content.find('\n', match.end()) + 1
        content = content[:insert_pos] + quantization_imports + '\n' + content[insert_pos:]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  ✓ Added quantization imports to {filepath}")
    return True


def integrate_server_file(filepath):
    """Integrate quantization into a server file"""
    print(f"\nProcessing server: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return
    
    backup_file(filepath)
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already integrated
    if 'self.quantization_handler' in content:
        print(f"  Already integrated")
        return
    
    # Add imports
    add_quantization_imports_to_server(filepath)
    
    print(f"  ✓ Integrated quantization into {filepath}")


def main():
    """Main integration function"""
    print("="*70)
    print("Quantization Integration Script")
    print("="*70)
    print("\nThis script integrates quantization into all FL implementations")
    print("It will:")
    print("  1. Add quantization imports")
    print("  2. Initialize quantization handlers")
    print("  3. Create backups of modified files")
    print("\n" + "="*70 + "\n")
    
    base_path = Path(__file__).parent
    
    # Use cases
    use_cases = ['Emotion_Recognition', 'MentalState_Recognition', 'Temperature_Regulation']
    protocols = ['MQTT', 'AMQP', 'gRPC', 'QUIC', 'DDS']
    
    # Integrate clients
    print("\n" + "="*70)
    print("INTEGRATING CLIENT FILES")
    print("="*70)
    
    for use_case in use_cases:
        print(f"\n{use_case}:")
        for protocol in protocols:
            client_file = base_path / 'Client' / use_case / f'FL_Client_{protocol}.py'
            if client_file.exists():
                integrate_client_file(str(client_file))
    
    # Integrate servers
    print("\n" + "="*70)
    print("INTEGRATING SERVER FILES")
    print("="*70)
    
    for use_case in use_cases:
        print(f"\n{use_case}:")
        for protocol in protocols:
            server_file = base_path / 'Server' / use_case / f'FL_Server_{protocol}.py'
            if server_file.exists():
                integrate_server_file(str(server_file))
    
    print("\n" + "="*70)
    print("INTEGRATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the changes in the modified files")
    print("  2. Manually add compression/decompression logic to:")
    print("     - train_local_model() methods in clients")
    print("     - handle_global_model() methods in clients")
    print("     - handle_client_update() methods in servers")
    print("     - aggregate_models() methods in servers")
    print("  3. Follow the pattern from FL_Client_MQTT.py and FL_Server_MQTT.py")
    print("  4. Test with different quantization strategies")
    print("  5. Restore from .backup files if needed")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
