# Compression Evaluation Guide

## Overview

The evaluation pipeline now supports automatic detection and comparison of compression techniques:
- **Quantization**: Reduced precision (8-bit, 16-bit)
- **Pruning**: Sparse models (sparsity percentages)
- **Combined**: Both quantization and pruning together

## Files Updated

### 1. consolidate_results.py
- ✅ Detects quantization metrics (bits, compression ratio)
- ✅ Detects pruning metrics (sparsity, communication savings)
- ✅ Shows compression info in summary output

### 2. compare_protocols.py
- ✅ Displays compression techniques in summary tables
- ✅ Shows "Q:8bit" for quantization, "P:50.0%" for pruning

### 3. New: compare_pruning_results.py
- ✅ Dedicated pruning comparison tool
- ✅ Analyzes sparsity, compression ratios, communication savings
- ✅ Generates detailed reports and visualizations

### 4. New: compare_compression.py
- ✅ Unified comparison for all compression techniques
- ✅ Supports quantization, pruning, or both

## Usage Examples

### Standard Evaluation with Compression Detection

```bash
# Evaluate experiment folder (automatically detects compression)
python Network_Simulation/evaluate_all.py \
  --experiment-folder temperature_pruned_50pct_20260111_120000

# Output will show compression info:
# MQTT: 1000 rounds, 234.56s [P:50.0%]
# AMQP: 1000 rounds, 245.67s [Q:8bit, P:40.0%]
```

### Compare Pruning Results

```bash
# Compare baseline vs pruned experiments
python Network_Simulation/compare_pruning_results.py \
  --baseline experiment_results/temperature_baseline_20260110_100000 \
  --pruned experiment_results/temperature_pruned_50pct_20260111_120000

# Output includes:
# - CSV results: pruning_comparison_results.csv
# - Text report: pruning_comparison_report.txt
# - Charts: pruning_comparison_charts.png
```

### Compare Quantization Results

```bash
# Compare baseline vs quantized experiments
python Network_Simulation/compare_quantization_results.py \
  --baseline experiment_results/temperature_baseline \
  --quantized experiment_results/temperature_quantized_8bit
```

### Unified Compression Comparison

```bash
# Compare all compression techniques
python Network_Simulation/compare_compression.py --mode both \
  --baseline experiment_results/temperature_baseline \
  --quantized experiment_results/temperature_quantized_8bit \
  --pruned experiment_results/temperature_pruned_50pct \
  --output comparison_results/all_techniques

# Or compare just one technique
python Network_Simulation/compare_compression.py --mode pruning \
  --baseline experiment_results/temperature_baseline \
  --compressed experiment_results/temperature_pruned_50pct
```

## Detected Metrics

### Quantization Metrics
- **quantization_bits**: Number of bits (8, 16, etc.)
- **avg_quantization_compression**: Average compression ratio
- **uses_quantization**: Boolean flag

### Pruning Metrics
- **avg_sparsity_pct**: Average model sparsity percentage
- **avg_pruning_compression**: Average compression ratio
- **communication_savings_pct**: Communication size reduction
- **uses_pruning**: Boolean flag

## Sample Output

### Consolidation Summary
```
EXCELLENT Network Scenario:
  - MQTT: 1000 rounds, 234.56s [Q:8bit, P:50.0%]
  - AMQP: 1000 rounds, 245.67s [Q:8bit]
  - GRPC: 1000 rounds, 223.45s [P:40.0%]
  - QUIC: 1000 rounds, 212.34s
  - DDS: 1000 rounds, 256.78s [Q:8bit, P:50.0%]
```

### Protocol Comparison Table
```
====================================================================================================
Protocol Performance Summary - Temperature (Excellent Network)
====================================================================================================
Protocol   Rounds   Time (s)     Time (min)   Final Loss   Final MSE    Compression         
----------------------------------------------------------------------------------------------------
AMQP       1000     245.67       4.09         0.012345     0.000123     Q:8bit              
DDS        1000     256.78       4.28         0.012567     0.000126     Q:8bit, P:50.0%     
GRPC       1000     223.45       3.72         0.011234     0.000112     P:40.0%             
MQTT       1000     234.56       3.91         0.011890     0.000119     Q:8bit, P:50.0%     
QUIC       1000     212.34       3.54         0.010987     0.000110     None                
====================================================================================================
```

### Pruning Comparison Report
```
================================================================================
PRUNING COMPARISON REPORT
================================================================================
Generated: 2026-01-11 12:00:00
Total Experiments: 30

OVERALL STATISTICS:
--------------------------------------------------------------------------------
Average Model Sparsity: 50.00%
Average Compression Ratio: 2.00x
Average Model Size Reduction: 61.50%
Average Communication Savings: 45.00%

Average Loss Change: -0.50%
Best Loss Improvement: 1.20%
Worst Loss Degradation: -2.10%
Experiments with Improved Loss: 15/30
Experiments with Degraded Loss: 12/30
Experiments with Same Loss: 3/30

PER-PROTOCOL ANALYSIS:
================================================================================

MQTT:
--------------------------------------------------------------------------------
  Experiments: 6
  Average Sparsity: 50.00%
  Average Compression: 2.00x
  Average Loss Change: -0.30%
  Best: excellent (none) - 1.20% improvement
  Worst: very_poor (none) - -1.50% change
```

## Integration with Existing Workflows

### Step 1: Run Experiments with Compression
```bash
# With pruning
docker-compose -f docker-compose-temperature-pruned.yml up

# With quantization
docker-compose -f docker-compose-temperature-quantized.yml up

# With both
docker-compose -f docker-compose-temperature-combined.yml up
```

### Step 2: Evaluate Results
```bash
# Standard evaluation (detects compression automatically)
python Network_Simulation/evaluate_all.py \
  --experiment-folder temperature_pruned_50pct_20260111_120000
```

### Step 3: Compare Techniques
```bash
# Compare with baseline
python Network_Simulation/compare_compression.py --mode pruning \
  --baseline experiment_results/temperature_baseline \
  --compressed experiment_results/temperature_pruned_50pct
```

## Visualization Charts

### Pruning Comparison Charts Include:
1. **Final Loss Comparison** - Baseline vs Pruned by protocol
2. **Pruning Effectiveness** - Sparsity % and compression ratio
3. **Communication Savings** - Size reduction by protocol
4. **Accuracy Impact** - Loss change by network scenario

### Chart Output
- Saved as PNG at 300 DPI
- 16x12 inch size for publication quality
- Color-coded (green = improvement, red = degradation)

## Tips & Best Practices

### 1. Organizing Experiments
Use descriptive folder names:
- `temperature_baseline_20260111_100000` - No compression
- `temperature_quantized_8bit_20260111_110000` - Quantization only
- `temperature_pruned_50pct_20260111_120000` - Pruning only
- `temperature_combined_8bit_50pct_20260111_130000` - Both techniques

### 2. Comparison Workflow
```bash
# Run all variants
docker-compose -f docker-compose-baseline.yml up
docker-compose -f docker-compose-quantized.yml up
docker-compose -f docker-compose-pruned.yml up
docker-compose -f docker-compose-combined.yml up

# Evaluate each
for folder in temperature_*_20260111_*; do
  python Network_Simulation/evaluate_all.py --experiment-folder $folder
done

# Compare all
python Network_Simulation/compare_compression.py --mode both \
  --baseline experiment_results/temperature_baseline_20260111_100000 \
  --quantized experiment_results/temperature_quantized_8bit_20260111_110000 \
  --pruned experiment_results/temperature_pruned_50pct_20260111_120000
```

### 3. Analyzing Results
Look for:
- **Compression effectiveness**: Higher sparsity/better compression
- **Accuracy preservation**: Minimal loss degradation (<2%)
- **Communication savings**: >40% for good pruning
- **Protocol sensitivity**: Which protocols benefit most

## Troubleshooting

### Issue: Compression not detected
**Solution**: Check server logs contain compression metrics:
- Quantization: Look for "QUANTIZATION_BITS" or "Compression ratio"
- Pruning: Look for "Sparsity" or "Pruning" messages

### Issue: No matching experiments in comparison
**Solution**: Ensure experiment folder structures match:
- Both should have same protocol_scenario subdirectories
- E.g., both need `mqtt_excellent/`, `amqp_excellent/`, etc.

### Issue: Charts not generated
**Solution**: Install required packages:
```bash
pip install matplotlib seaborn pandas
```

## Output Files

### From evaluate_all.py
- `Server/<UseCase>_Regulation/results/<protocol>_<scenario>_training_results.json`
- Contains all metrics including compression flags

### From compare_pruning_results.py
- `comparison_results/pruning_comparison_<timestamp>/`
  - `pruning_comparison_results.csv` - Full comparison data
  - `pruning_comparison_report.txt` - Text summary
  - `pruning_comparison_charts.png` - Visualizations

### From compare_compression.py (both mode)
- `comparison_results/<output>/quantization/` - Quantization comparison
- `comparison_results/<output>/pruning/` - Pruning comparison

## Next Steps

1. ✅ Run experiments with different compression techniques
2. ✅ Use `evaluate_all.py` to consolidate results (auto-detects compression)
3. ✅ Compare with `compare_compression.py` for detailed analysis
4. ✅ Review charts and reports for insights
5. ✅ Iterate on compression parameters based on findings

---

**Last Updated**: January 11, 2026
**Status**: ✅ Fully integrated and tested
