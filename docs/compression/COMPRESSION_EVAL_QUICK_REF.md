# Compression Evaluation Quick Reference

## Quick Commands

### Evaluate Experiment (Auto-Detects Compression)
```bash
python Network_Simulation/evaluate_all.py \
  --experiment-folder <folder_name>
```

### Compare Pruning Results
```bash
python Network_Simulation/compare_pruning_results.py \
  --baseline experiment_results/<baseline_folder> \
  --pruned experiment_results/<pruned_folder>
```

### Compare Quantization Results
```bash
python Network_Simulation/compare_quantization_results.py \
  --baseline experiment_results/<baseline_folder> \
  --quantized experiment_results/<quantized_folder>
```

### Compare All Compression Techniques
```bash
python Network_Simulation/compare_compression.py --mode both \
  --baseline experiment_results/<baseline_folder> \
  --quantized experiment_results/<quantized_folder> \
  --pruned experiment_results/<pruned_folder>
```

## What Gets Detected

| Compression | Detected Metrics | Display Format |
|-------------|------------------|----------------|
| **Quantization** | Bits (8/16), compression ratio | `Q:8bit` |
| **Pruning** | Sparsity %, compression ratio, comm. savings | `P:50.0%` |
| **Combined** | All of the above | `Q:8bit, P:50.0%` |

## Output Files

### evaluate_all.py
- JSON files: `Server/<UseCase>_Regulation/results/<protocol>_<scenario>_training_results.json`
- Console: Shows `[Q:8bit, P:50.0%]` inline

### compare_pruning_results.py
- CSV: `pruning_comparison_results.csv`
- Report: `pruning_comparison_report.txt`
- Charts: `pruning_comparison_charts.png`

### compare_compression.py
- Quantization folder: `<output>/quantization/`
- Pruning folder: `<output>/pruning/`

## Sample Output

### Consolidation
```
EXCELLENT Network Scenario:
  - MQTT: 1000 rounds, 234.56s [Q:8bit, P:50.0%]
  - GRPC: 1000 rounds, 223.45s [P:40.0%]
  - QUIC: 1000 rounds, 212.34s
```

### Comparison Table
```
Protocol   Rounds   Time (s)   Final Loss   Compression         
------------------------------------------------------------------
MQTT       1000     234.56     0.011890     Q:8bit, P:50.0%     
GRPC       1000     223.45     0.011234     P:40.0%             
QUIC       1000     212.34     0.010987     None                
```

## Metrics in JSON

```json
{
  "uses_quantization": true,
  "quantization_bits": 8,
  "avg_quantization_compression": 4.0,
  
  "uses_pruning": true,
  "avg_sparsity_pct": 50.0,
  "avg_pruning_compression": 2.0,
  "communication_savings_pct": 45.0
}
```

## Legend

- **Q:8bit** = 8-bit Quantization
- **Q:16bit** = 16-bit Quantization
- **P:50.0%** = 50% Pruning Sparsity
- **None** = No compression
