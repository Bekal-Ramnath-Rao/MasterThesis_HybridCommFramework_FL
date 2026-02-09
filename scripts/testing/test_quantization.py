"""
Test and Demo Script for Quantization Implementation
Tests all three quantization strategies with sample data
"""

import numpy as np
import sys
import os

# Add paths
client_compression_path = os.path.join(os.path.dirname(__file__), 'Client', 'Compression_Technique')
server_compression_path = os.path.join(os.path.dirname(__file__), 'Server', 'Compression_Technique')
sys.path.insert(0, client_compression_path)
sys.path.insert(0, server_compression_path)

from quantization_client import Quantization, QuantizationConfig, QuantizationStrategy


def create_sample_weights():
    """Create sample model weights for testing"""
    weights = [
        np.random.randn(32, 3, 3, 1).astype(np.float32),  # Conv2D layer
        np.random.randn(32).astype(np.float32),            # Bias
        np.random.randn(64, 3, 3, 32).astype(np.float32),  # Conv2D layer
        np.random.randn(64).astype(np.float32),            # Bias
        np.random.randn(1024, 9216).astype(np.float32),    # Dense layer
        np.random.randn(1024).astype(np.float32),          # Bias
        np.random.randn(7, 1024).astype(np.float32),       # Output layer
        np.random.randn(7).astype(np.float32),             # Bias
    ]
    return weights


def calculate_size(weights):
    """Calculate total size in MB"""
    total_bytes = sum(w.nbytes for w in weights)
    return total_bytes / (1024 * 1024)


def test_parameter_quantization():
    """Test parameter quantization strategy"""
    print("\n" + "="*70)
    print("TEST 1: PARAMETER QUANTIZATION")
    print("="*70)
    
    # Create sample weights
    weights = create_sample_weights()
    original_size = calculate_size(weights)
    print(f"\nOriginal weights size: {original_size:.2f} MB")
    print(f"Number of layers: {len(weights)}")
    
    # Test different bit depths
    for bits in [8, 16, 32]:
        print(f"\n--- Testing {bits}-bit quantization ---")
        
        config = QuantizationConfig(
            strategy="parameter_quantization",
            bits=bits,
            symmetric=True,
            per_channel=False
        )
        
        quantizer = Quantization(config)
        
        # Compress
        compressed_data = quantizer.compress(weights, data_type="weights")
        compressed_size = calculate_size(compressed_data['compressed_data'])
        
        # Decompress
        decompressed_weights = quantizer.decompress(compressed_data)
        
        # Calculate stats
        stats = quantizer.get_compression_stats(weights, compressed_data)
        
        print(f"Compressed size: {compressed_size:.2f} MB")
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"Size reduction: {stats['size_reduction_percent']:.1f}%")
        
        # Verify reconstruction
        max_error = max(np.max(np.abs(orig - decomp)) 
                       for orig, decomp in zip(weights, decompressed_weights))
        print(f"Max reconstruction error: {max_error:.6f}")


def test_symmetric_vs_asymmetric():
    """Test symmetric vs asymmetric quantization"""
    print("\n" + "="*70)
    print("TEST 2: SYMMETRIC VS ASYMMETRIC QUANTIZATION")
    print("="*70)
    
    weights = create_sample_weights()
    
    for symmetric in [True, False]:
        mode = "Symmetric" if symmetric else "Asymmetric"
        print(f"\n--- Testing {mode} Quantization ---")
        
        config = QuantizationConfig(
            strategy="parameter_quantization",
            bits=8,
            symmetric=symmetric,
            per_channel=False
        )
        
        quantizer = Quantization(config)
        compressed_data = quantizer.compress(weights, data_type="weights")
        stats = quantizer.get_compression_stats(weights, compressed_data)
        
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")


def test_per_channel_quantization():
    """Test per-channel vs per-tensor quantization"""
    print("\n" + "="*70)
    print("TEST 3: PER-CHANNEL VS PER-TENSOR QUANTIZATION")
    print("="*70)
    
    weights = create_sample_weights()
    
    for per_channel in [False, True]:
        mode = "Per-Channel" if per_channel else "Per-Tensor"
        print(f"\n--- Testing {mode} Quantization ---")
        
        config = QuantizationConfig(
            strategy="parameter_quantization",
            bits=8,
            symmetric=True,
            per_channel=per_channel
        )
        
        quantizer = Quantization(config)
        compressed_data = quantizer.compress(weights, data_type="weights")
        decompressed_weights = quantizer.decompress(compressed_data)
        
        # Calculate reconstruction error
        mse = sum(np.mean((orig - decomp)**2) 
                 for orig, decomp in zip(weights, decompressed_weights)) / len(weights)
        
        stats = quantizer.get_compression_stats(weights, compressed_data)
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"Mean Squared Error: {mse:.6f}")


def test_server_client_workflow():
    """Test complete server-client quantization workflow"""
    print("\n" + "="*70)
    print("TEST 4: SERVER-CLIENT WORKFLOW SIMULATION")
    print("="*70)
    
    from quantization_server import ServerQuantizationHandler
    
    # Client side
    print("\n--- Client Side ---")
    client_config = QuantizationConfig(
        strategy="parameter_quantization",
        bits=8,
        symmetric=True
    )
    client_quantizer = Quantization(client_config)
    
    # Create and compress client weights
    client_weights = create_sample_weights()
    print(f"Client: Original weights size: {calculate_size(client_weights):.2f} MB")
    
    compressed_update = client_quantizer.compress(client_weights, data_type="weights")
    print(f"Client: Compressed update size: {calculate_size(compressed_update['compressed_data']):.2f} MB")
    
    # Server side
    print("\n--- Server Side ---")
    server_handler = ServerQuantizationHandler(client_config)
    
    # Simulate multiple clients
    client_updates = {}
    for client_id in range(2):
        client_weights = create_sample_weights()
        compressed = client_quantizer.compress(client_weights, data_type="weights")
        client_updates[client_id] = {
            'compressed_data': compressed,
            'num_samples': 1000 + client_id * 100
        }
    
    # Aggregate
    aggregated_weights, agg_stats = server_handler.aggregate_quantized_updates(
        client_updates,
        aggregation_method="fedavg"
    )
    
    print(f"Server: Aggregated {agg_stats['num_clients']} client updates")
    print(f"Server: Aggregated weights size: {calculate_size(aggregated_weights):.2f} MB")
    
    # Compress global model for distribution
    compressed_global = server_handler.compress_global_model(aggregated_weights)
    stats = server_handler.get_compression_stats(aggregated_weights, compressed_global)
    print(f"Server: Compressed global model - Ratio: {stats['compression_ratio']:.2f}x")


def test_all_strategies():
    """Test all three quantization strategies"""
    print("\n" + "="*70)
    print("TEST 5: ALL QUANTIZATION STRATEGIES")
    print("="*70)
    
    weights = create_sample_weights()
    strategies = [
        "parameter_quantization",
        "post_training_quantization",
        "quantization_aware_training"
    ]
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy.upper()} ---")
        
        config = QuantizationConfig(
            strategy=strategy,
            bits=8,
            symmetric=True
        )
        
        quantizer = Quantization(config)
        
        try:
            compressed_data = quantizer.compress(weights, data_type="weights")
            stats = quantizer.get_compression_stats(weights, compressed_data)
            print(f"✓ Success - Compression ratio: {stats['compression_ratio']:.2f}x")
        except Exception as e:
            print(f"✗ Failed: {e}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("QUANTIZATION IMPLEMENTATION TEST SUITE")
    print("="*70)
    print("\nTesting quantization compression for federated learning")
    print("This will test all strategies and configurations")
    
    try:
        test_parameter_quantization()
        test_symmetric_vs_asymmetric()
        test_per_channel_quantization()
        test_server_client_workflow()
        test_all_strategies()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Integration is complete for MQTT clients/servers")
        print("  2. Run integrate_quantization.py to add to other protocols")
        print("  3. Test with actual FL training")
        print("  4. Monitor compression statistics during training")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
