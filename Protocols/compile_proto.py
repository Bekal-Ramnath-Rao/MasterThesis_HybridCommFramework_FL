"""
Compile Protocol Buffer definitions for gRPC
Run this script to generate Python code from .proto files
"""
import subprocess
import sys
from pathlib import Path

def compile_proto():
    """Compile the federated_learning.proto file"""
    
    # Get the Protocols directory path
    protocols_dir = Path(__file__).parent
    proto_file = protocols_dir / "federated_learning.proto"
    
    if not proto_file.exists():
        print(f"Error: Proto file not found at {proto_file}")
        return False
    
    print(f"Compiling {proto_file}...")
    
    # Compile the proto file
    # Output will be generated in the same directory
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={protocols_dir}",
        f"--python_out={protocols_dir}",
        f"--grpc_python_out={protocols_dir}",
        str(proto_file)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Successfully compiled protocol buffers!")
        print(f"  Generated: federated_learning_pb2.py")
        print(f"  Generated: federated_learning_pb2_grpc.py")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error compiling proto file:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("gRPC Protocol Buffer Compiler")
    print("="*60)
    print()
    
    success = compile_proto()
    
    print()
    if success:
        print("You can now run the gRPC server and clients!")
    else:
        print("Please fix the errors and try again.")
        print()
        print("Make sure grpcio-tools is installed:")
        print("  pip install grpcio-tools")
    
    sys.exit(0 if success else 1)
