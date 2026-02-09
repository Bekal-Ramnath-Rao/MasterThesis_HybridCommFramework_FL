# FL Baseline & Network Experiments - Quick Reference

## Baseline Experiments

### Run Baseline (All Protocols)
```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --rounds 10 \
    --baseline
```

### Run Baseline (Specific Protocol)
```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --protocols mqtt \
    --rounds 10 \
    --baseline
```

### Baseline Output Location
```
experiment_results_baseline/
└── emotion/
    ├── mqtt_baseline/
    │   ├── mqtt_baseline_rtt.json
    │   └── mqtt_training_results.json
    ├── amqp_baseline/
    ├── grpc_baseline/
    ├── quic_baseline/
    └── dds_baseline/
```

## Network Condition Experiments

### Single Experiment
```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --single \
    --protocol mqtt \
    --scenario poor \
    --rounds 10
```

### All Protocols, Specific Scenarios
```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --scenarios poor very_poor \
    --rounds 10
```

### All Protocols, All Scenarios
```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --rounds 10
```

## Monitoring with Baseline Comparison

### Start Dashboard
```bash
python3 Network_Simulation/fl_training_dashboard.py --use-case emotion
```

### Dashboard Shows:
- ✅ Baseline RTT for each protocol
- ✅ Current network conditions
- ✅ Expected vs actual performance impact
- ✅ Real-time container stats

## RTT (Round Trip Time)

### What is RTT?
Time from global model distribution → client training → update aggregation → next round

### Example RTT Data
```json
{
  "protocol": "mqtt",
  "avg_rtt_per_round": 2.45,
  "rtt_per_round": [2.3, 2.5, 2.4, 2.6, ...]
}
```

## Network Condition Impact

| Network Scenario | Expected RTT Increase | Use Case |
|-----------------|----------------------|----------|
| Excellent (baseline) | 0% | Reference |
| 20ms latency | +20-50% | Good broadband |
| 100ms latency | +80-120% | 4G/LTE |
| 200ms latency | +150-250% | Poor network |
| 300ms latency | +250-400% | Very poor/edge |
| + Packet loss | +Variable | Retransmissions |

## Complete Workflow

### 1. Run Baseline
```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --baseline
```

### 2. Run Experiments
```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --scenarios poor
```

### 3. Monitor (Terminal 1)
```bash
python3 Network_Simulation/fl_training_dashboard.py \
    --use-case emotion
```

### 4. Control Network (Terminal 2)
```bash
# Apply conditions dynamically
python3 Network_Simulation/fl_network_monitor.py \
    --client-id 1 \
    --latency 200ms \
    --loss 2
```

## Directory Structure

```
project/
├── experiment_results_baseline/    # Baseline (no network conditions)
│   ├── emotion/
│   ├── temperature/
│   └── mentalstate/
│
├── experiment_results/              # Experiments with network conditions
│   ├── emotion_20260128_143000/
│   │   ├── mqtt_poor/
│   │   │   ├── mqtt_rtt.json       # RTT with network conditions
│   │   │   └── mqtt_training_results.json
│   │   └── ...
│   └── ...
│
└── Network_Simulation/
    ├── run_network_experiments.py  # Main experiment runner
    ├── fl_training_dashboard.py    # Monitoring dashboard
    └── fl_network_monitor.py       # Network control tool
```

## Comparison Commands

### Load Baseline RTT
```python
import json
baseline = json.load(open('experiment_results_baseline/emotion/mqtt_baseline/mqtt_baseline_rtt.json'))
print(f"Baseline RTT: {baseline['avg_rtt_per_round']:.2f}s")
```

### Calculate Degradation
```python
baseline_rtt = 2.5  # from baseline
current_rtt = 5.2   # from experiment
degradation = ((current_rtt - baseline_rtt) / baseline_rtt) * 100
print(f"Degradation: +{degradation:.1f}%")
```

## Quick Tips

### ✅ Best Practices
1. **Run baseline first** before any experiments
2. **Use same configuration** (rounds, GPU, etc.)
3. **Refresh baseline** when changing model/hardware
4. **Monitor in real-time** with dashboard
5. **Document conditions** applied during experiments

### ⚠️ Common Mistakes
1. Running experiments before baseline
2. Using different number of rounds
3. Changing model architecture between baseline and experiments
4. Not specifying use-case in dashboard

### 🔧 Troubleshooting
```bash
# Check baseline exists
ls experiment_results_baseline/emotion/

# Verify RTT files
cat experiment_results_baseline/emotion/mqtt_baseline/mqtt_baseline_rtt.json

# Re-run baseline if needed
python3 run_network_experiments.py --use-case emotion --baseline --protocols mqtt
```

## Example Session

```bash
# 1. Run baseline (once)
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion --baseline --rounds 10

# 2. Start monitoring
python3 Network_Simulation/fl_training_dashboard.py --use-case emotion

# 3. Run experiment with network conditions (new terminal)
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --single \
    --protocol mqtt \
    --scenario poor \
    --rounds 10

# 4. Observe comparison in dashboard
# Dashboard shows:
#   Baseline RTT: 2.5s
#   Current conditions: L:200ms BW:1Mbit Loss:2%
#   Expected impact: +150-250% RTT
```

## File Naming Convention

| Mode | Folder | RTT File | Example |
|------|--------|----------|---------|
| Baseline | `<protocol>_baseline/` | `<protocol>_baseline_rtt.json` | `mqtt_baseline_rtt.json` |
| Experiment | `<protocol>_<scenario>/` | `<protocol>_rtt.json` | `mqtt_rtt.json` |

## Flags Summary

| Flag | Purpose | Example |
|------|---------|---------|
| `--baseline` | Run baseline mode | `--baseline` |
| `--use-case` | Select use case | `--use-case emotion` |
| `--protocols` | Select protocols | `--protocols mqtt grpc` |
| `--scenarios` | Select scenarios | `--scenarios poor very_poor` |
| `--rounds` | Number of FL rounds | `--rounds 10` |
| `--enable-gpu` | Use GPU | `--enable-gpu` |
| `--single` | Single experiment | `--single --protocol mqtt --scenario poor` |

---

**Remember**: Baseline = Reference point. Run once per configuration, then compare all experiments against it.
