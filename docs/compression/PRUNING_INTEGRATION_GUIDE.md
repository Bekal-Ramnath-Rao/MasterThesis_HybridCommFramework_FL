# Model Pruning Integration Guide

## Overview
Comprehensive pruning implementation for all 3 Federated Learning use cases:
- **Emotion Recognition**: CNN model (Conv2D layers)
- **Mental State Recognition**: CNN+BiLSTM+MultiHeadAttention model
- **Temperature Regulation**: LSTM model

## Features

### Pruning Types
1. **Unstructured Pruning**: Removes individual weights based on magnitude
   - Higher compression ratios
   - Requires sparse tensor support for speedup

2. **Structured Pruning**: Removes entire filters/channels/neurons
   - Lower compression but hardware-friendly
   - Immediate speedup without special hardware

### Pruning Schedules
- **Polynomial Decay**: Gradually increases sparsity (recommended)
- **Constant**: Fixed sparsity from start

## Client-Side Integration

### 1. Emotion Recognition (CNN Model)

```python
# In Client/Emotion_Recognition/FL_Client_Emotion.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Compression_Technique.pruning_client import ModelPruning, PruningConfig

class EmotionFLClient:
    def __init__(self):
        # Initialize pruning
        self.use_pruning = os.getenv("USE_PRUNING", "0") == "1"
        
        if self.use_pruning:
            pruning_config = PruningConfig(
                target_sparsity=0.5,  # 50% sparsity
                pruning_schedule="polynomial",
                begin_step=10,  # Start pruning after round 10
                end_step=100,   # Reach target by round 100
                frequency=5,    # Update pruning every 5 rounds
                structured=False  # Unstructured for CNNs
            )
            self.pruning_engine = ModelPruning(pruning_config)
            print("[Emotion Client] Pruning enabled")
    
    def train_local_model(self, round_num):
        # ... existing training code ...
        
        # Apply pruning after training
        if self.use_pruning:
            self.model = self.pruning_engine.apply_pruning_to_model(
                self.model, 
                step=round_num
            )
            
            # Get statistics
            stats = self.pruning_engine.get_pruning_statistics(
                self.model.get_weights()
            )
            print(f"[Pruning] Round {round_num}: "
                  f"Sparsity {stats['overall_sparsity']:.2%}")
        
        return self.model.get_weights()
    
    def compress_weights_for_upload(self, weights):
        """Compress weights before sending to server"""
        if self.use_pruning:
            compressed, metadata = self.pruning_engine.compress_pruned_weights(weights)
            return compressed
        else:
            import pickle
            return pickle.dumps(weights)
```

### 2. Mental State Recognition (CNN+BiLSTM+MHA Model)

```python
# In Client/MentalState_Recognition/FL_Client_MentalState.py

from Compression_Technique.pruning_client import ModelPruning, PruningConfig

class MentalStateFLClient:
    def __init__(self):
        self.use_pruning = os.getenv("USE_PRUNING", "0") == "1"
        
        if self.use_pruning:
            # Structured pruning works better for LSTM/Attention
            pruning_config = PruningConfig(
                target_sparsity=0.4,  # 40% for complex models
                pruning_schedule="polynomial",
                begin_step=15,
                end_step=150,
                frequency=10,
                structured=True  # Structured for LSTM layers
            )
            self.pruning_engine = ModelPruning(pruning_config)
            print("[MentalState Client] Structured pruning enabled")
    
    def train_local_model(self, round_num):
        # ... existing training ...
        
        if self.use_pruning:
            # Apply pruning
            self.model = self.pruning_engine.apply_pruning_to_model(
                self.model,
                step=round_num
            )
            
            # Optional: Fine-tune after aggressive pruning
            if round_num % 50 == 0 and round_num > 0:
                print(f"[Round {round_num}] Fine-tuning pruned model...")
                self.model = self.pruning_engine.fine_tune_pruned_model(
                    self.model,
                    self.X_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=3
                )
        
        return self.model.get_weights()
```

### 3. Temperature Regulation (LSTM Model)

```python
# In Client/Temperature_Regulation/FL_Client_Temperature.py

from Compression_Technique.pruning_client import ModelPruning, PruningConfig

class TemperatureFLClient:
    def __init__(self):
        self.use_pruning = os.getenv("USE_PRUNING", "0") == "1"
        
        if self.use_pruning:
            # Conservative pruning for regression tasks
            pruning_config = PruningConfig(
                target_sparsity=0.3,  # 30% for regression
                pruning_schedule="polynomial",
                begin_step=20,
                end_step=200,
                frequency=10,
                structured=False
            )
            self.pruning_engine = ModelPruning(pruning_config)
            print("[Temperature Client] Pruning enabled for LSTM")
    
    def train_local_model(self, round_num):
        # ... existing training ...
        
        if self.use_pruning:
            self.model = self.pruning_engine.apply_pruning_to_model(
                self.model,
                step=round_num
            )
        
        return self.model.get_weights()
```

## Server-Side Integration

### All Use Cases

```python
# In Server/<UseCase>_Regulation/FL_Server_<UseCase>.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Compression_Technique.pruning_server import ServerPruning, PruningMetricsLogger

class FLServer:
    def __init__(self):
        self.use_pruning = os.getenv("USE_PRUNING", "0") == "1"
        
        if self.use_pruning:
            # Match client configuration
            from Compression_Technique.pruning_client import PruningConfig
            
            pruning_config = PruningConfig(
                target_sparsity=float(os.getenv("PRUNING_SPARSITY", "0.5")),
                structured=os.getenv("PRUNING_STRUCTURED", "false").lower() == "true"
            )
            
            self.server_pruning = ServerPruning(pruning_config)
            self.pruning_logger = PruningMetricsLogger(
                log_dir=f"results/{self.protocol}/pruning_logs"
            )
            print("[Server] Pruning coordinator initialized")
    
    def aggregate_client_updates(self, client_weights_list, num_samples_list):
        """Aggregate client updates with pruning awareness"""
        
        if self.use_pruning:
            # Decompress if clients sent compressed updates
            decompressed_weights = []
            for cw in client_weights_list:
                if isinstance(cw, bytes):
                    weights = self.server_pruning.decompress_client_update(cw)
                else:
                    weights = cw
                decompressed_weights.append(weights)
            
            # Aggregate with pruning awareness
            aggregated = self.server_pruning.aggregate_pruned_updates(
                decompressed_weights,
                num_samples_list
            )
        else:
            # Standard weighted average
            aggregated = self.weighted_average(client_weights_list, num_samples_list)
        
        return aggregated
    
    def update_global_model(self, aggregated_weights, round_num):
        """Update global model and apply server-side pruning"""
        
        # Set aggregated weights
        self.global_model.set_weights(aggregated_weights)
        
        # Apply server-side pruning
        if self.use_pruning and self.server_pruning.should_prune_this_round(round_num):
            print(f"\n[Server] Applying pruning at round {round_num}")
            self.global_model = self.server_pruning.prune_global_model(
                self.global_model,
                round_num
            )
            
            # Log metrics
            stats = self.server_pruning.get_compression_stats(
                self.global_model.get_weights()
            )
            accuracy = self.evaluate_global_model()
            
            self.pruning_logger.log_round(round_num, stats, accuracy)
            
            # Print communication savings
            if round_num % 10 == 0:
                comm_stats = stats['communication_savings']
                print(f"\n[Communication Efficiency]")
                print(f"  Original: {comm_stats['original_size_bytes'] / 1024:.1f} KB")
                print(f"  Compressed: {comm_stats['compressed_size_bytes'] / 1024:.1f} KB")
                print(f"  Reduction: {comm_stats['size_reduction_percent']:.1f}%")
    
    def broadcast_global_model(self):
        """Broadcast global model to clients"""
        
        weights = self.global_model.get_weights()
        
        if self.use_pruning:
            # Compress before broadcasting
            compressed, metadata = self.server_pruning.compress_for_broadcast(weights)
            return compressed
        else:
            return weights
    
    def finalize_training(self):
        """Called after all rounds complete"""
        
        if self.use_pruning:
            # Save final pruning state
            self.server_pruning.save_pruning_state(
                f"results/{self.protocol}/final_pruning_state.pkl"
            )
            
            # Save and print metrics
            self.pruning_logger.save_metrics()
            self.pruning_logger.print_summary()
```

## Docker Environment Variables

Add to your `docker-compose.yml` files:

```yaml
services:
  fl_server_mqtt:
    environment:
      # ... existing variables ...
      - USE_PRUNING=1
      - PRUNING_SPARSITY=0.5      # Target sparsity (0.0 to 1.0)
      - PRUNING_STRUCTURED=false   # true for structured, false for unstructured
      - PRUNING_BEGIN_STEP=10      # Start pruning at this round
      - PRUNING_END_STEP=100       # Reach target by this round
      - PRUNING_FREQUENCY=5        # Update pruning every N rounds

  fl_client_mqtt_1:
    environment:
      # ... existing variables ...
      - USE_PRUNING=1
      - PRUNING_SPARSITY=0.5
      - PRUNING_STRUCTURED=false
```

## Recommended Configurations

### Emotion Recognition (CNN)
```yaml
USE_PRUNING=1
PRUNING_SPARSITY=0.5
PRUNING_STRUCTURED=false
PRUNING_BEGIN_STEP=10
PRUNING_END_STEP=100
PRUNING_FREQUENCY=5
```
- **Rationale**: CNNs handle high sparsity well, unstructured gives better compression

### Mental State Recognition (CNN+BiLSTM+MHA)
```yaml
USE_PRUNING=1
PRUNING_SPARSITY=0.4
PRUNING_STRUCTURED=true
PRUNING_BEGIN_STEP=15
PRUNING_END_STEP=150
PRUNING_FREQUENCY=10
```
- **Rationale**: Complex model needs gradual pruning, structured pruning preserves LSTM structure

### Temperature Regulation (LSTM)
```yaml
USE_PRUNING=1
PRUNING_SPARSITY=0.3
PRUNING_STRUCTURED=false
PRUNING_BEGIN_STEP=20
PRUNING_END_STEP=200
PRUNING_FREQUENCY=10
```
- **Rationale**: Regression tasks sensitive to pruning, conservative sparsity target

## Testing Pruning

### Test Individual Model
```python
# test_pruning_<usecase>.py
from Compression_Technique.pruning_client import ModelPruning, PruningConfig
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model('path/to/model.h5')

# Configure pruning
config = PruningConfig(target_sparsity=0.5)
pruning = ModelPruning(config)

# Apply pruning
pruned_model = pruning.apply_pruning_to_model(model, step=100)

# Get statistics
stats = pruning.get_pruning_statistics(pruned_model.get_weights())
print(f"Sparsity: {stats['overall_sparsity']:.2%}")
print(f"Compression: {stats['compression_ratio']:.2f}x")

# Test compression
compressed, metadata = pruning.compress_pruned_weights(pruned_model.get_weights())
print(f"Compressed size: {len(compressed) / 1024:.1f} KB")
```

## Performance Metrics

### What to Monitor
1. **Model Sparsity**: Percentage of zero weights
2. **Compression Ratio**: Original params / Non-zero params
3. **Communication Savings**: Size reduction in bytes
4. **Model Accuracy**: Ensure minimal degradation
5. **Training Time**: Pruning adds minimal overhead

### Expected Results
- **Communication**: 40-70% reduction in model size
- **Accuracy**: <2% degradation with proper schedule
- **Convergence**: May need 10-20% more rounds

## Troubleshooting

### Issue: Accuracy drops too much
**Solution**: Reduce target_sparsity or increase end_step for gradual pruning

### Issue: Not seeing compression benefits
**Solution**: 
- Enable compression: Use `compress_pruned_weights()` before transmission
- Check sparsity: Ensure pruning is actually applied

### Issue: LSTM/Attention layers unstable after pruning
**Solution**: Switch to structured pruning (`PRUNING_STRUCTURED=true`)

## Combining with Quantization

You can use both pruning and quantization together:

```python
# Apply in order: Quantization → Pruning
if self.use_quantization:
    self.model = self.quantization_engine.apply_quantization(self.model)

if self.use_pruning:
    self.model = self.pruning_engine.apply_pruning_to_model(self.model, round_num)
```

Set both in docker-compose:
```yaml
environment:
  - USE_QUANTIZATION=1
  - USE_PRUNING=1
  - PRUNING_SPARSITY=0.4
```

## Next Steps

1. **Integrate into existing FL clients** (see examples above)
2. **Update docker-compose files** with pruning variables
3. **Run experiments** comparing pruning vs. no pruning
4. **Analyze results** using the pruning metrics logger
5. **Tune sparsity targets** based on your accuracy requirements

## References

- Magnitude-based Pruning: [Han et al., 2015]
- Structured Pruning: [Liu et al., 2017]
- Lottery Ticket Hypothesis: [Frankle & Carbin, 2019]
