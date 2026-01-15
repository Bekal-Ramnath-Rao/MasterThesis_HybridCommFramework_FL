# Pruning Quick Reference

## Enable Pruning in Docker

```yaml
# docker-compose-<protocol>-pruned.yml
environment:
  - USE_PRUNING=1
  - PRUNING_SPARSITY=0.5
  - PRUNING_STRUCTURED=false
```

## Client Code Integration (3 lines)

```python
# In FL_Client_<UseCase>.py

# 1. Import
from Compression_Technique.pruning_client import ModelPruning, PruningConfig

# 2. Initialize (in __init__)
if os.getenv("USE_PRUNING", "0") == "1":
    self.pruning_engine = ModelPruning(PruningConfig(
        target_sparsity=float(os.getenv("PRUNING_SPARSITY", "0.5"))
    ))

# 3. Apply (in train_local_model)
if hasattr(self, 'pruning_engine'):
    self.model = self.pruning_engine.apply_pruning_to_model(self.model, round_num)
```

## Server Code Integration (3 lines)

```python
# In FL_Server_<UseCase>.py

# 1. Import
from Compression_Technique.pruning_server import ServerPruning

# 2. Initialize (in __init__)
if os.getenv("USE_PRUNING", "0") == "1":
    self.server_pruning = ServerPruning(PruningConfig(
        target_sparsity=float(os.getenv("PRUNING_SPARSITY", "0.5"))
    ))

# 3. Apply (in update_global_model)
if hasattr(self, 'server_pruning'):
    self.global_model = self.server_pruning.prune_global_model(
        self.global_model, round_num
    )
```

## Recommended Configs

| Use Case | Sparsity | Structured | Begin | End | Frequency |
|----------|----------|------------|-------|-----|-----------|
| Emotion (CNN) | 0.5 | false | 10 | 100 | 5 |
| MentalState (CNN+LSTM+MHA) | 0.4 | true | 15 | 150 | 10 |
| Temperature (LSTM) | 0.3 | false | 20 | 200 | 10 |

## Test Pruning

```bash
# Test implementation
python test_pruning.py

# Expected: 6/6 tests passed
```

## Check Results

```python
# Get pruning statistics
stats = pruning_engine.get_pruning_statistics(model.get_weights())

print(f"Sparsity: {stats['overall_sparsity']:.2%}")
print(f"Compression: {stats['compression_ratio']:.2f}x")
print(f"Active params: {stats['non_zero_params']:,}")
```

## Compression for Communication

```python
# Client: Compress before sending
compressed = pruning.compress_pruned_weights(weights)

# Server: Decompress
weights = server_pruning.decompress_client_update(compressed)
```

## Combine with Quantization

```yaml
environment:
  - USE_QUANTIZATION=1
  - QUANTIZATION_BITS=8
  - USE_PRUNING=1
  - PRUNING_SPARSITY=0.5
```

Apply order: Quantization → Pruning

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Accuracy drops >5% | Reduce `PRUNING_SPARSITY` or increase `PRUNING_END_STEP` |
| No compression seen | Enable `compress_pruned_weights()` before transmission |
| LSTM unstable | Set `PRUNING_STRUCTURED=true` |
| Too slow convergence | Increase `PRUNING_FREQUENCY` (update less often) |

## Key Files

- `Client/Compression_Technique/pruning_client.py` - Client implementation
- `Server/Compression_Technique/pruning_server.py` - Server implementation
- `PRUNING_INTEGRATION_GUIDE.md` - Full integration guide
- `test_pruning.py` - Test suite

## Expected Results

- **Communication**: 40-70% size reduction
- **Accuracy**: <2% degradation
- **Compression**: 1.5-3x depending on sparsity
