# Network Scenario Evaluation Guide

## Overview

This guide explains how to evaluate your network simulation experiments across all network conditions. The evaluation system consists of three main scripts that work together to provide comprehensive analysis.

## Available Network Scenarios

The system supports evaluation across six network conditions:

1. **Excellent** (LAN) - 2ms latency, 1Gbps bandwidth, 0.01% loss
2. **Good** (Broadband) - 10ms latency, 100Mbps bandwidth, 0.1% loss
3. **Moderate** (4G/LTE) - 50ms latency, 20Mbps bandwidth, 1% loss
4. **Poor** (3G) - 100ms latency, 2Mbps bandwidth, 3% loss
5. **Very Poor** (2G/Edge) - 300ms latency, 384Kbps bandwidth, 5% loss
6. **Satellite** - 600ms latency, 5Mbps bandwidth, 2% loss

## Evaluation Scripts

### 1. consolidate_results.py
Extracts and consolidates training results from experiment log files.

**Usage:**
```bash
# Consolidate all scenarios
python consolidate_results.py --use-case temperature

# Consolidate specific scenarios
python consolidate_results.py --use-case temperature --scenarios excellent good moderate
```

**Output:**
- JSON files in `Server/<UseCase>_Regulation/results/`
- Format: `{protocol}_{scenario}_training_results.json`

### 2. compare_protocols.py
Compares protocol performance within each scenario.

**Usage:**
```bash
# Compare protocols for specific scenarios
python compare_protocols.py --use-case temperature --scenarios excellent good

# Specify custom output directory
python compare_protocols.py --use-case temperature --scenarios excellent --output-dir my_results
```

**Output per scenario:**
- `{usecase}_{scenario}_convergence_time.png` - Time vs rounds plot
- `{usecase}_{scenario}_loss_comparison.png` - Loss convergence plot
- `{usecase}_{scenario}_summary_comparison.png` - Bar charts summary
- Console output: Performance summary table

### 3. evaluate_network_scenarios.py
Provides comprehensive cross-scenario analysis and comparison.

**Usage:**
```bash
# Full evaluation with CSV export
python evaluate_network_scenarios.py --use-case temperature --export-csv

# Custom output directory
python evaluate_network_scenarios.py --use-case temperature --output-dir network_analysis
```

**Output:**
- `{usecase}_heatmap_time.png` - Training time heatmap across all scenarios
- `{usecase}_heatmap_loss.png` - Final loss heatmap
- `{usecase}_heatmap_rounds.png` - Rounds to convergence heatmap
- `{usecase}_cross_scenario_comparison.png` - Grouped bar chart
- `{usecase}_{protocol}_across_scenarios.png` - Per-protocol analysis (5 files)
- `{usecase}_all_scenarios_comparison.csv` - Complete data export (if --export-csv used)
- Console output: Comprehensive table for all scenarios

### 4. evaluate_all.py (Master Pipeline)
Runs the complete evaluation pipeline automatically.

**Usage:**
```bash
# Evaluate all available scenarios
python evaluate_all.py --use-case temperature --export-csv

# Evaluate specific scenarios only
python evaluate_all.py --use-case temperature --scenarios excellent good moderate

# Skip already-completed steps
python evaluate_all.py --use-case temperature --skip-consolidate --skip-comparison
```

**Options:**
- `--use-case`: Choose emotion, mentalstate, or temperature (default: temperature)
- `--scenarios`: Specific scenarios to evaluate (default: all)
- `--output-dir`: Output directory (default: comparison_results)
- `--skip-consolidate`: Skip consolidation if already done
- `--skip-comparison`: Skip individual scenario comparisons (faster)
- `--export-csv`: Export results to CSV format

**Pipeline Steps:**
1. Consolidate experiment results from logs
2. Compare protocols for each scenario (optional)
3. Generate comprehensive cross-scenario analysis
4. Export to CSV (optional)

## Typical Workflow

### After Running Experiments

After running experiments with `run_network_experiments.py`:

```bash
# Your experiments created data in experiment_results/
# For example: experiment_results/temperature_20260103_110426/

# Step 1: Run the master evaluation pipeline
python evaluate_all.py --use-case temperature --export-csv
```

This will:
- Extract all results from experiment logs
- Generate comparison plots for each scenario
- Create heatmaps and cross-scenario analysis
- Export CSV for further analysis
- Save everything to `comparison_results/`

### Evaluating Specific Scenarios

If you only ran experiments for certain scenarios:

```bash
# Only evaluate excellent and good scenarios
python evaluate_all.py --use-case temperature --scenarios excellent good --export-csv
```

### Re-running Analysis

If you already consolidated results and just want new visualizations:

```bash
# Skip consolidation, just generate new plots
python evaluate_all.py --use-case temperature --skip-consolidate
```

### Quick Scenario-Specific Analysis

For analyzing just one scenario:

```bash
# Consolidate all scenarios
python consolidate_results.py --use-case temperature

# Compare only the excellent scenario
python compare_protocols.py --use-case temperature --scenarios excellent
```

## Generated Outputs

### Visualization Files

**Per-Scenario Comparisons** (from compare_protocols.py):
- Shows protocol performance within a single network condition
- 3 plots per scenario: time, loss, and summary bars

**Cross-Scenario Analysis** (from evaluate_network_scenarios.py):
- Heatmaps showing performance across all scenarios
- Individual protocol trend analysis
- Grouped comparisons

### Data Files

**JSON Results** (`Server/<UseCase>_Regulation/results/`):
- Structured results for each protocol-scenario combination
- Contains: rounds, times, loss values, MSE, MAE, MAPE

**CSV Export** (`comparison_results/`):
- Consolidated table for all scenarios
- Easy import to Excel, R, Python pandas, etc.
- Columns: use_case, scenario, protocol, rounds, time_seconds, final_loss, final_mse, etc.

## Understanding the Results

### Performance Metrics

- **Rounds**: Number of federated learning rounds to convergence
- **Time (seconds/minutes)**: Total training time
- **Final Loss**: Model loss at convergence
- **Final MSE**: Mean Squared Error at convergence
- **Final MAE**: Mean Absolute Error
- **Final MAPE**: Mean Absolute Percentage Error

### What to Look For

1. **Protocol Performance**: Which protocol is fastest/most accurate?
2. **Network Impact**: How do different network conditions affect each protocol?
3. **Stability**: Do protocols maintain performance across scenarios?
4. **Trade-offs**: Time vs accuracy, robustness vs speed

### Interpreting Heatmaps

- **Darker colors** = higher values (longer time, higher loss)
- **Lighter colors** = lower values (faster, better accuracy)
- Look for protocols that maintain consistent (light) colors across scenarios

### Interpreting Cross-Scenario Plots

- Each protocol has one bar per scenario
- Compare bar heights to see network impact
- Steeper increases = more sensitive to network degradation

## Advanced Usage

### Custom Analysis

Use the exported CSV for custom analysis:

```python
import pandas as pd

# Load results
df = pd.read_csv('comparison_results/temperature_all_scenarios_comparison.csv')

# Filter specific protocols
mqtt_data = df[df['protocol'] == 'mqtt']

# Compare scenarios
pivot = df.pivot_table(index='scenario', columns='protocol', values='time_seconds')
print(pivot)
```

### Batch Evaluation

Evaluate multiple use cases:

```bash
for use_case in temperature emotion mentalstate; do
    python evaluate_all.py --use-case $use_case --export-csv
done
```

### Selective Re-evaluation

If you've run new experiments for specific scenarios:

```bash
# Consolidate just the new scenarios
python consolidate_results.py --use-case temperature --scenarios poor very_poor

# Re-run comprehensive analysis
python evaluate_network_scenarios.py --use-case temperature --export-csv
```

## Troubleshooting

### "No results found"

**Problem**: Script reports no results for a scenario.
**Solution**: 
1. Verify experiments actually ran: `ls experiment_results/`
2. Check that experiment completed successfully
3. Verify log files exist: `ls experiment_results/*/server_logs.txt`

### Missing Visualizations

**Problem**: Some plots aren't generated.
**Solution**:
1. Install required packages: `pip install matplotlib pandas numpy`
2. Check that results were consolidated first
3. Verify JSON files exist in `Server/<UseCase>_Regulation/results/`

### Incomplete Data

**Problem**: Some protocol-scenario combinations are missing.
**Solution**:
- Run missing experiments with `run_network_experiments.py`
- Specify only available scenarios: `--scenarios excellent good`

## Quick Reference

```bash
# Complete workflow from scratch
python evaluate_all.py --use-case temperature --export-csv

# Update after new experiments
python consolidate_results.py --use-case temperature
python evaluate_network_scenarios.py --use-case temperature --export-csv

# Quick comparison for one scenario
python compare_protocols.py --use-case temperature --scenarios excellent

# Export data only
python evaluate_network_scenarios.py --use-case temperature --export-csv
```

## Next Steps

After evaluation:

1. **Review Results**: Check the generated plots and CSV
2. **Identify Best Protocols**: Based on your requirements (speed, accuracy, stability)
3. **Document Findings**: Note which protocols work best under which conditions
4. **Plan Further Experiments**: If needed, run additional scenarios
5. **Publish Results**: Use the plots and CSV in your thesis/paper

## Support

For issues or questions:
1. Check that all scripts are in the root directory
2. Verify Python packages are installed: `pip install -r requirements.txt`
3. Ensure experiments completed successfully before evaluation
4. Review the COMPLETE_EVALUATION_GUIDE.md for more details
