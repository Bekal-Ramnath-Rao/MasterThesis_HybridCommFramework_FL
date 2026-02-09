# FL Baseline & Network Comparison Guide

## Overview

This guide explains how to use the **baseline experiment system** to establish reference metrics and compare network condition impacts on FL training.

## What is Baseline?

**Baseline** refers to FL experiments run under **ideal network conditions** (no latency, no packet loss, no bandwidth limits). These results serve as a reference point for:

- Comparing protocol performance under different network conditions
- Measuring network degradation impact
- Calculating round trip time (RTT) overhead
- Evaluating communication efficiency

## Key Features

### 1. **Baseline Mode** (`--baseline`)
- Runs experiments without applying network conditions
- Saves results to dedicated baseline folder
- Tracks round trip time (RTT) for each FL round
- Creates reference data for comparison

### 2. **RTT Tracking**
- **Round Trip Time (RTT)**: Time from sending global model to clients receiving it, training, and server receiving updates back
- Automatically measured for each FL round
- Averaged across all rounds for comparison
- Stored in JSON format for analysis

### 3. **Dashboard Comparison**
- Real-time comparison with baseline metrics
- Shows expected vs actual performance degradation
- Highlights protocol-specific impacts
- Visual indication of network condition effects

## Directory Structure

```
experiment_results_baseline/
├── README.md
├── emotion/
│   ├── mqtt_baseline/
│   │   ├── mqtt_baseline_rtt.json          # RTT data
│   │   ├── mqtt_training_results.json      # Training metrics
│   │   ├── metadata.json                   # Experiment config
│   │   └── server_logs.txt                 # Server logs
│   ├── amqp_baseline/
│   ├── grpc_baseline/
│   ├── quic_baseline/
│   └── dds_baseline/
├── temperature/
│   └── [same structure]
└── mentalstate/
    └── [same structure]
```

## Workflow

### Step 1: Run Baseline Experiments

Run baseline for all protocols (recommended):

```bash
# Emotion recognition baseline
cd /path/to/project
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --rounds 10 \
    --baseline

# Temperature prediction baseline
python3 Network_Simulation/run_network_experiments.py \
    --use-case temperature \
    --enable-gpu \
    --rounds 10 \
    --baseline

# Mental state baseline
python3 Network_Simulation/run_network_experiments.py \
    --use-case mentalstate \
    --enable-gpu \
    --rounds 10 \
    --baseline
```

For specific protocol baseline:

```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --protocols mqtt \
    --rounds 10 \
    --baseline
```

### Step 2: Run Experiments with Network Conditions

After establishing baseline, run experiments with network conditions:

```bash
# Single experiment with poor network
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --single \
    --protocol mqtt \
    --scenario poor \
    --rounds 10

# All protocols, all scenarios
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --rounds 10
```

### Step 3: Monitor with Dashboard

Start the monitoring dashboard with baseline comparison:

```bash
# Terminal 1: Start dashboard
python3 Network_Simulation/fl_training_dashboard.py --use-case emotion

# Terminal 2: Apply network conditions (optional)
python3 Network_Simulation/fl_network_monitor.py --client-id 1 --latency 200ms
```

## RTT Measurement Details

### What is Measured

**Round Trip Time (RTT)** includes:
1. Server sends global model to clients
2. Clients receive model (network latency)
3. Clients train on local data
4. Clients send updates back to server (network latency)
5. Server receives updates
6. **RTT = Time until next round starts**

### How It's Tracked

The system monitors server logs for round completion patterns:
- `Round X completed`
- `[Round X] completed`
- `Completed round X`
- `Round X finished`

Time between consecutive round completions = RTT for that round

### RTT Data Format

```json
{
  "protocol": "mqtt",
  "scenario": "baseline",
  "use_case": "emotion",
  "num_rounds": 10,
  "rtt_per_round": [2.3, 2.5, 2.4, 2.6, 2.5, 2.4, 2.5, 2.4, 2.5, 2.4],
  "avg_rtt_per_round": 2.45,
  "min_rtt": 2.3,
  "max_rtt": 2.6,
  "total_rtt": 24.5,
  "timestamp": "2026-01-28T14:30:00",
  "baseline_mode": true
}
```

## Dashboard Output Example

### With Baseline Data

```
================================================================================
         FL TRAINING DASHBOARD - Use Case: EMOTION
                    Last Update: 2026-01-28 14:30:45
================================================================================

📈 BASELINE COMPARISON
--------------------------------------------------------------------------------
Protocol        Baseline Avg RTT     Current Network                      Expected Impact
--------------------------------------------------------------------------------
MQTT            2.45s                L:200ms BW:1Mbit Loss:2%             High degradation expected (+150-250% RTT) + Retransmissions
AMQP            2.67s                None                                 Baseline (ideal)
GRPC            2.12s                L:100ms                              Moderate degradation expected (+80-120% RTT)
--------------------------------------------------------------------------------
```

## Comparison Analysis

### Baseline vs Network Conditions

| Scenario | Expected RTT Increase | Reason |
|----------|----------------------|---------|
| Excellent (baseline) | 0% | No added latency |
| 20ms latency | +20-50% | Minimal delay |
| 50ms latency | +50-80% | Moderate delay |
| 100ms latency | +80-120% | Significant delay |
| 200ms latency | +150-250% | High delay + timeouts |
| 300ms latency | +250-400% | Very high delay + retransmissions |
| Packet loss | +Variable | Depends on loss % and retransmission protocol |

### Example Calculation

**Baseline**: MQTT avg RTT = 2.5s/round

**With 200ms latency + 2% loss**:
- Expected RTT: 2.5s × 2.0 (200% overhead) = **5.0s/round**
- Actual measured: **5.2s/round**
- Overhead: +108% (close to expected)

## Using Baseline Data

### Programmatic Access

```python
import json
from pathlib import Path

# Load baseline RTT
baseline_file = Path("experiment_results_baseline/emotion/mqtt_baseline/mqtt_baseline_rtt.json")
with open(baseline_file) as f:
    baseline = json.load(f)

baseline_rtt = baseline['avg_rtt_per_round']
print(f"Baseline RTT: {baseline_rtt:.2f}s")

# Load current experiment RTT
current_file = Path("experiment_results/emotion_20260128_143000/mqtt_poor/mqtt_rtt.json")
with open(current_file) as f:
    current = json.load(f)

current_rtt = current['avg_rtt_per_round']
degradation = ((current_rtt - baseline_rtt) / baseline_rtt) * 100

print(f"Current RTT: {current_rtt:.2f}s")
print(f"Degradation: +{degradation:.1f}%")
```

### Comparison Script

```bash
# Compare baseline with experiment
python3 Network_Simulation/compare_with_baseline.py \
    --baseline experiment_results_baseline/emotion \
    --experiment experiment_results/emotion_20260128_143000
```

## Best Practices

### 1. **Run Baseline First**
Always establish baseline before running network condition experiments:
```bash
python3 run_network_experiments.py --use-case emotion --baseline
```

### 2. **Consistent Configuration**
Use same settings for baseline and experiments:
- Same number of rounds (`--rounds`)
- Same GPU settings (`--enable-gpu`)
- Same model architecture
- Same number of clients

### 3. **Multiple Baselines**
Create separate baselines for different configurations:
```bash
# Baseline without quantization
python3 run_network_experiments.py --use-case emotion --baseline

# Baseline with quantization
python3 run_network_experiments.py --use-case emotion --baseline --use-quantization --quantization-bits 8
```

### 4. **Refresh Periodically**
Re-run baseline if you change:
- Model architecture
- Training hyperparameters
- Number of clients
- Hardware setup
- Docker configuration

### 5. **Document Baseline**
Keep track of when baseline was generated and what configuration was used.

## Troubleshooting

### Issue: No Baseline Data Found

**Symptom**: Dashboard shows "No baseline data found"

**Solution**:
```bash
# Run baseline first
python3 run_network_experiments.py --use-case emotion --baseline

# Verify baseline files exist
ls -la experiment_results_baseline/emotion/
```

### Issue: RTT Not Being Tracked

**Symptom**: `rtt_per_round` is empty in JSON

**Solution**: Check server logs for round completion messages. The system looks for:
- `Round X completed`
- `[Round X] completed`

Ensure your server prints these messages.

### Issue: Baseline Comparison Shows Wrong Data

**Symptom**: Dashboard compares wrong protocol or use case

**Solution**: Specify correct use case:
```bash
python3 fl_training_dashboard.py --use-case emotion
```

## Advanced Usage

### Custom Baseline Location

Modify `run_network_experiments.py` to change baseline folder:
```python
# Line ~65
self.results_dir = project_root / "my_custom_baseline" / folder_name
```

### Automated Baseline + Experiments

```bash
#!/bin/bash
# Run baseline then all experiments

USE_CASE="emotion"

# Step 1: Baseline
echo "Running baseline for $USE_CASE..."
python3 run_network_experiments.py \
    --use-case $USE_CASE \
    --rounds 10 \
    --baseline

# Step 2: Experiments with network conditions
echo "Running network condition experiments..."
python3 run_network_experiments.py \
    --use-case $USE_CASE \
    --rounds 10 \
    --scenarios poor very_poor

echo "Done! Check results in experiment_results/"
```

## Summary

The baseline system provides:
- ✅ **Reference metrics** for comparison
- ✅ **RTT tracking** for each FL round
- ✅ **Automated comparison** in dashboard
- ✅ **Protocol-specific** baseline data
- ✅ **Easy to use** with `--baseline` flag

Workflow:
1. Run baseline: `--baseline`
2. Run experiments with network conditions
3. Monitor with dashboard: automatic comparison
4. Analyze RTT overhead and performance degradation

All baseline data stored in `experiment_results_baseline/<use_case>/`
