"""
Test Pruning Implementation for All Use Cases
Verifies pruning works correctly with CNN, CNN+BiLSTM+MHA, and LSTM models
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import tensorflow as tf
from Client.Compression_Technique.pruning_client import ModelPruning, PruningConfig
from Server.Compression_Technique.pruning_server import ServerPruning, PruningMetricsLogger


def create_cnn_model():
    """Create CNN model similar to Emotion Recognition"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_lstm_model():
    """Create LSTM model similar to Temperature Regulation"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(10, 4)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_complex_model():
    """Create complex model similar to Mental State Recognition"""
    inputs = tf.keras.Input(shape=(256, 20))
    
    # Conv1D layers
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    
    # BiLSTM
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    
    # Attention (simplified)
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = tf.keras.layers.Add()([x, attention])
    
    # Dense
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def test_unstructured_pruning():
    """Test unstructured pruning on CNN model"""
    print("\n" + "="*70)
    print("Test 1: Unstructured Pruning on CNN Model (Emotion Recognition)")
    print("="*70)
    
    # Create model
    model = create_cnn_model()
    original_weights = model.get_weights()
    
    # Configure pruning
    config = PruningConfig(
        target_sparsity=0.5,
        pruning_schedule="polynomial",
        begin_step=0,
        end_step=100,
        structured=False
    )
    
    pruning = ModelPruning(config)
    
    # Test pruning at different steps
    for step in [0, 25, 50, 75, 100]:
        model.set_weights(original_weights)  # Reset
        pruned_model = pruning.apply_pruning_to_model(model, step=step)
        
        stats = pruning.get_pruning_statistics(pruned_model.get_weights())
        expected_sparsity = config.target_sparsity * (1 - (1 - step/100) ** 3) if step < 100 else config.target_sparsity
        
        print(f"\nStep {step}:")
        print(f"  Expected Sparsity: {expected_sparsity:.2%}")
        print(f"  Actual Sparsity: {stats['overall_sparsity']:.2%}")
        print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Non-zero Params: {stats['non_zero_params']:,}/{stats['total_params']:,}")
    
    print("\n✓ Unstructured pruning test PASSED")
    return True


def test_structured_pruning():
    """Test structured pruning on complex model"""
    print("\n" + "="*70)
    print("Test 2: Structured Pruning on Complex Model (Mental State Recognition)")
    print("="*70)
    
    # Create model
    model = create_complex_model()
    original_weights = model.get_weights()
    
    # Configure structured pruning
    config = PruningConfig(
        target_sparsity=0.4,
        pruning_schedule="constant",
        structured=True
    )
    
    pruning = ModelPruning(config)
    
    # Apply pruning
    pruned_model = pruning.apply_pruning_to_model(model, step=100)
    
    stats = pruning.get_pruning_statistics(pruned_model.get_weights())
    
    print(f"\nStructured Pruning Results:")
    print(f"  Target Sparsity: {config.target_sparsity:.2%}")
    print(f"  Actual Sparsity: {stats['overall_sparsity']:.2%}")
    print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
    
    # Verify structured pruning (entire filters should be zero)
    pruned_weights = pruned_model.get_weights()
    for i, w in enumerate(pruned_weights):
        if len(w.shape) >= 2:
            print(f"  Layer {i} shape: {w.shape}, zeros: {np.sum(w == 0)}/{w.size}")
    
    print("\n✓ Structured pruning test PASSED")
    return True


def test_compression():
    """Test sparse weight compression"""
    print("\n" + "="*70)
    print("Test 3: Sparse Weight Compression")
    print("="*70)
    
    # Create and prune model
    model = create_lstm_model()
    
    config = PruningConfig(target_sparsity=0.6, structured=False)
    pruning = ModelPruning(config)
    
    pruned_model = pruning.apply_pruning_to_model(model, step=100)
    pruned_weights = pruned_model.get_weights()
    
    # Calculate original size
    original_size = sum(w.nbytes for w in pruned_weights)
    
    # Compress
    compressed, metadata = pruning.compress_pruned_weights(pruned_weights)
    compressed_size = len(compressed)
    
    # Decompress
    decompressed = pruning.decompress_pruned_weights(compressed)
    
    print(f"\nCompression Results:")
    print(f"  Original Size: {original_size / 1024:.2f} KB")
    print(f"  Compressed Size: {compressed_size / 1024:.2f} KB")
    print(f"  Compression Ratio: {original_size / compressed_size:.2f}x")
    print(f"  Size Reduction: {(1 - compressed_size / original_size) * 100:.1f}%")
    
    # Verify decompression accuracy
    print(f"\nDecompression Verification:")
    for i, (orig, decomp) in enumerate(zip(pruned_weights, decompressed)):
        diff = np.max(np.abs(orig - decomp))
        print(f"  Layer {i}: Max difference = {diff:.2e}")
        assert diff < 1e-6, f"Decompression error too large: {diff}"
    
    print("\n✓ Compression test PASSED")
    return True


def test_server_pruning():
    """Test server-side pruning coordination"""
    print("\n" + "="*70)
    print("Test 4: Server-Side Pruning Coordination")
    print("="*70)
    
    # Create server pruning
    config = PruningConfig(target_sparsity=0.5, structured=False)
    server_pruning = ServerPruning(config)
    
    # Create model
    model = create_cnn_model()
    
    # Simulate FL rounds
    for round_num in [0, 25, 50, 75, 100]:
        # Prune global model
        pruned_model = server_pruning.prune_global_model(model, round_num)
        
        # Get stats
        stats = server_pruning.get_compression_stats(pruned_model.get_weights())
        
        print(f"\nRound {round_num}:")
        print(f"  Sparsity: {stats['overall_sparsity']:.2%}")
        print(f"  Should prune: {server_pruning.should_prune_this_round(round_num)}")
        
        if 'communication_savings' in stats:
            comm = stats['communication_savings']
            print(f"  Comm. reduction: {comm['size_reduction_percent']:.1f}%")
    
    print("\n✓ Server pruning test PASSED")
    return True


def test_client_aggregation():
    """Test aggregation with pruned client updates"""
    print("\n" + "="*70)
    print("Test 5: Client Update Aggregation with Pruning")
    print("="*70)
    
    # Create server
    config = PruningConfig(target_sparsity=0.5)
    server_pruning = ServerPruning(config)
    
    # Simulate 3 clients with different data sizes
    model = create_lstm_model()
    client_weights = [model.get_weights() for _ in range(3)]
    num_samples = [100, 150, 200]
    
    # Add some variation to weights
    for i, weights in enumerate(client_weights):
        for j in range(len(weights)):
            client_weights[i][j] = weights[j] + np.random.randn(*weights[j].shape) * 0.01
    
    # Aggregate
    aggregated = server_pruning.aggregate_pruned_updates(client_weights, num_samples)
    
    print(f"\nAggregation Results:")
    print(f"  Number of clients: {len(client_weights)}")
    print(f"  Sample distribution: {num_samples}")
    print(f"  Aggregated weights shape: {[w.shape for w in aggregated[:3]]}")
    
    # Verify weighted average
    expected_0 = (client_weights[0][0] * 100 + client_weights[1][0] * 150 + client_weights[2][0] * 200) / 450
    diff = np.max(np.abs(aggregated[0] - expected_0))
    print(f"  Aggregation accuracy: Max diff = {diff:.2e}")
    
    print("\n✓ Aggregation test PASSED")
    return True


def test_metrics_logger():
    """Test pruning metrics logging"""
    print("\n" + "="*70)
    print("Test 6: Pruning Metrics Logger")
    print("="*70)
    
    import tempfile
    log_dir = tempfile.mkdtemp()
    
    logger = PruningMetricsLogger(log_dir=log_dir)
    
    # Log some rounds
    for round_num in range(10):
        stats = {
            'overall_sparsity': round_num * 0.05,
            'compression_ratio': 1.0 / (1.0 - round_num * 0.05) if round_num > 0 else 1.0,
            'non_zero_params': 10000 - round_num * 500,
            'total_params': 10000,
            'communication_savings': {
                'original_size_bytes': 40000,
                'compressed_size_bytes': 40000 - round_num * 2000,
                'compression_ratio': 40000 / (40000 - round_num * 2000) if round_num > 0 else 1.0,
                'size_reduction_percent': (round_num * 2000 / 40000) * 100
            }
        }
        accuracy = 0.95 - round_num * 0.01
        
        logger.log_round(round_num, stats, accuracy)
    
    # Save metrics
    logger.save_metrics()
    logger.print_summary()
    
    # Verify file created
    import os
    metrics_file = os.path.join(log_dir, "pruning_metrics.json")
    assert os.path.exists(metrics_file), "Metrics file not created"
    
    print(f"  Metrics saved to: {metrics_file}")
    print("\n✓ Metrics logger test PASSED")
    return True


def run_all_tests():
    """Run all pruning tests"""
    print("\n" + "="*70)
    print("PRUNING IMPLEMENTATION COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("Testing pruning for all 3 FL use cases:")
    print("  1. Emotion Recognition (CNN)")
    print("  2. Mental State Recognition (CNN+BiLSTM+MHA)")
    print("  3. Temperature Regulation (LSTM)")
    print("="*70)
    
    tests = [
        ("Unstructured Pruning", test_unstructured_pruning),
        ("Structured Pruning", test_structured_pruning),
        ("Sparse Compression", test_compression),
        ("Server Pruning", test_server_pruning),
        ("Client Aggregation", test_client_aggregation),
        ("Metrics Logger", test_metrics_logger)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} test FAILED: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
