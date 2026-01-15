# Evaluation Script Usage Guide

## Overview
The updated `evaluate_all.py` script now accepts an experiment folder name and automatically organizes comparison results with matching folder names.

## Current Experiment Folders
Based on your experiment_results directory:
1. `temperature_quantized_8bit_20260110_195711` - Quantized experiments
2. `temperature_quantized_8bit_congestion_20260111_100707` - Quantized with congestion

## Usage Examples

### Evaluate Quantized Experiments
```bash
# On Windows PowerShell (recommended to set encoding first):
$env:PYTHONIOENCODING="utf-8"
python Network_Simulation/evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711

# Or in one line:
$env:PYTHONIOENCODING="utf-8"; python Network_Simulation/evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711
```

**Output location:** `comparison_results/temperature_quantized_8bit_20260110_195711/`

### Evaluate Quantized + Congestion Experiments
```bash
$env:PYTHONIOENCODING="utf-8"; python Network_Simulation/evaluate_all.py --experiment-folder temperature_quantized_8bit_congestion_20260111_100707
```

**Output location:** `comparison_results/temperature_quantized_8bit_congestion_20260111_100707/`

### Evaluate Specific Scenarios Only
```bash
$env:PYTHONIOENCODING="utf-8"; python Network_Simulation/evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711 --scenarios excellent good moderate
```

### Skip Comparison Plots (Faster)
```bash
$env:PYTHONIOENCODING="utf-8"; python Network_Simulation/evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711 --skip-comparison
```

### Custom Output Directory
```bash
$env:PYTHONIOENCODING="utf-8"; python Network_Simulation/evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711 --output-dir my_custom_results
```

### Export to CSV
```bash
$env:PYTHONIOENCODING="utf-8"; python Network_Simulation/evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711 --export-csv
```

## Required Parameter
- `--experiment-folder` / `-e`: **REQUIRED** - Name of the folder in `experiment_results/` to evaluate

## Optional Parameters
- `--use-case` / `-u`: Use case type (default: temperature)
- `--scenarios` / `-s`: Specific scenarios to evaluate (default: all available)
- `--output-dir` / `-o`: Custom output directory (default: comparison_results/<experiment-folder>)
- `--skip-consolidate`: Skip consolidation if already done
- `--skip-comparison`: Skip individual scenario comparisons
- `--export-csv`: Export results to CSV format

## Evaluation Pipeline Steps

The script runs three main steps:

1. **Consolidate Results**: Extract metrics from server logs
2. **Compare Protocols**: Generate comparison plots for each scenario
3. **Comprehensive Analysis**: Cross-scenario evaluation with heatmaps and rankings

## Output Structure

```
comparison_results/
├── temperature_quantized_8bit_20260110_195711/
│   ├── temperature_excellent_convergence_time.png
│   ├── temperature_excellent_loss_comparison.png
│   ├── temperature_excellent_summary_comparison.png
│   ├── temperature_good_convergence_time.png
│   ├── ...
│   ├── comprehensive_convergence_heatmap.png
│   └── comprehensive_protocol_rankings.png
└── temperature_quantized_8bit_congestion_20260111_100707/
    ├── temperature_excellent_convergence_time.png
    ├── ...
    └── comprehensive_protocol_rankings.png
```

## Notes

- The script automatically validates that the experiment folder exists
- Results are organized using the same folder name for easy identification
- If experiment folder is not found, it lists all available folders
- All dependent scripts (consolidate_results.py, compare_protocols.py, evaluate_network_scenarios.py) have been updated to support the experiment-folder parameter
