# FL Baseline & Network Control System - Summary

## 🎯 What Was Added

### 1. **Baseline Experiment System**
- Dedicated folder for baseline results: `experiment_results_baseline/`
- Run experiments without network conditions (`--baseline` flag)
- Establishes reference metrics for comparison
- Organized by use case (emotion, temperature, mentalstate)

### 2. **Round Trip Time (RTT) Tracking**
- Automatically measures time for each FL round
- Tracks from global model distribution to next round start
- Calculates average, min, max RTT
- Saves detailed RTT data in JSON format

### 3. **Enhanced Dashboard with Baseline Comparison**
- Loads baseline metrics automatically
- Compares current network conditions with baseline
- Shows expected vs actual performance degradation
- Protocol-specific comparison
- Real-time degradation estimation

## 📁 New Folder Structure

```
project/
├── experiment_results_baseline/          # NEW: Baseline folder
│   ├── README.md
│   ├── emotion/
│   │   ├── mqtt_baseline/
│   │   │   ├── mqtt_baseline_rtt.json   # NEW: RTT data
│   │   │   ├── mqtt_training_results.json
│   │   │   └── metadata.json
│   │   ├── amqp_baseline/
│   │   ├── grpc_baseline/
│   │   ├── quic_baseline/
│   │   └── dds_baseline/
│   ├── temperature/
│   └── mentalstate/
│
├── experiment_results/                   # Existing: Network experiments
│   └── emotion_20260128_143000/
│       ├── mqtt_poor/
│       │   ├── mqtt_rtt.json            # NEW: RTT with conditions
│       │   └── mqtt_training_results.json
│       └── ...
```

## 📝 Modified Files

### 1. `Network_Simulation/run_network_experiments.py`
**Changes:**
- ✅ Added `baseline_mode` parameter
- ✅ Added `--baseline` flag to CLI
- ✅ Modified results directory logic (baseline → dedicated folder)
- ✅ Added RTT tracking in `wait_for_completion()`
- ✅ Modified `collect_results()` to save RTT data
- ✅ Updated `run_single_experiment()` to skip network conditions in baseline mode
- ✅ In baseline mode, runs all protocols with excellent scenario (no network conditions)

**Key Code Additions:**
```python
# RTT tracking
round_trip_times = []
last_round_complete_time = None
current_round = 0

# Baseline folder selection
if baseline_mode:
    self.results_dir = project_root / "experiment_results_baseline" / folder_name
else:
    self.results_dir = project_root / "experiment_results" / folder_name

# RTT data storage
rtt_data = {
    "avg_rtt_per_round": avg_rtt,
    "rtt_per_round": round_trip_times,
    "min_rtt": min(round_trip_times),
    "max_rtt": max(round_trip_times),
    ...
}
```

### 2. `Network_Simulation/fl_training_dashboard.py`
**Changes:**
- ✅ Added `use_case` parameter
- ✅ Added `load_baseline_data()` method
- ✅ Added `get_protocol_from_container()` helper
- ✅ Enhanced `display_dashboard()` with baseline comparison section
- ✅ Added `--use-case` CLI argument
- ✅ Shows baseline RTT, current network conditions, expected impact

**New Dashboard Section:**
```
📈 BASELINE COMPARISON
Protocol    Baseline Avg RTT    Current Network              Expected Impact
MQTT        2.45s               L:200ms BW:1Mbit Loss:2%     High degradation (+150-250%)
AMQP        2.67s               None                         Baseline (ideal)
```

## 🚀 New Commands

### Run Baseline
```bash
# All protocols
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --baseline \
    --rounds 10

# Specific protocol
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --protocols mqtt \
    --baseline \
    --rounds 10
```

### Monitor with Baseline Comparison
```bash
# Dashboard with baseline comparison
python3 Network_Simulation/fl_training_dashboard.py --use-case emotion

# Different use case
python3 Network_Simulation/fl_training_dashboard.py --use-case temperature
```

## 📊 RTT Data Format

### Baseline RTT File: `mqtt_baseline_rtt.json`
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

### Experiment RTT File: `mqtt_rtt.json`
```json
{
  "protocol": "mqtt",
  "scenario": "poor",
  "use_case": "emotion",
  "num_rounds": 10,
  "rtt_per_round": [5.1, 5.3, 5.2, 5.4, 5.0, 5.3, 5.2, 5.1, 5.3, 5.2],
  "avg_rtt_per_round": 5.21,
  "min_rtt": 5.0,
  "max_rtt": 5.4,
  "total_rtt": 52.1,
  "timestamp": "2026-01-28T15:00:00",
  "baseline_mode": false
}
```

**Degradation**: (5.21 - 2.45) / 2.45 × 100 = **+112.7%**

## 📖 New Documentation

### 1. `experiment_results_baseline/README.md`
- Explains baseline concept
- Usage instructions
- File format description
- Comparison methodology

### 2. `FL_BASELINE_GUIDE.md`
- Comprehensive baseline guide (7.5KB)
- Detailed workflow
- RTT measurement explanation
- Comparison analysis
- Best practices
- Troubleshooting

### 3. `FL_BASELINE_QUICK_REF.md`
- Quick reference (4.2KB)
- Common commands
- Directory structure
- RTT impact table
- Example session

## 🔄 Complete Workflow

### Step 1: Run Baseline (Once)
```bash
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --rounds 10 \
    --baseline
```

**Output**: `experiment_results_baseline/emotion/mqtt_baseline/`

### Step 2: Start Monitoring (Terminal 1)
```bash
python3 Network_Simulation/fl_training_dashboard.py --use-case emotion
```

**Dashboard loads baseline and shows**:
- Baseline RTT for each protocol
- Current network conditions
- Expected performance impact

### Step 3: Run Experiments (Terminal 2)
```bash
# Single experiment with poor network
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --single \
    --protocol mqtt \
    --scenario poor \
    --rounds 10
```

**Output**: `experiment_results/emotion_20260128_143000/mqtt_poor/`

### Step 4: Apply Dynamic Network Conditions (Terminal 3)
```bash
# Change network conditions during training
python3 Network_Simulation/fl_network_monitor.py \
    --client-id 1 \
    --latency 200ms \
    --loss 2
```

### Step 5: Observe Comparison
Dashboard shows real-time comparison:
- Baseline RTT vs Current RTT
- Network degradation percentage
- Expected vs actual impact

## 📈 RTT Tracking Details

### What is Measured
**Round Trip Time (RTT) for each FL round:**
1. Server sends global model
2. Client receives model (network latency)
3. Client trains locally
4. Client sends update (network latency)
5. Server receives update
6. **Next round starts** ← RTT measured here

### How It Works
```python
# In wait_for_completion()
round_patterns = [
    r'Round (\d+)/\d+ completed',
    r'\[Round (\d+)\] completed',
    r'Completed round (\d+)',
]

# Track time between round completions
if latest_round > current_round:
    rtt = current_time - last_round_complete_time
    round_trip_times.append(rtt)
```

### Average Calculation
```python
avg_rtt = sum(round_trip_times) / len(round_trip_times)
```

## 🎯 Use Cases

### 1. Protocol Comparison
Compare different protocols under same network conditions:
```bash
# Baseline for all protocols
python3 run_network_experiments.py --use-case emotion --baseline

# Run all with poor network
python3 run_network_experiments.py --use-case emotion --scenario poor

# Compare RTTs in dashboard
python3 fl_training_dashboard.py --use-case emotion
```

### 2. Network Impact Analysis
Measure network degradation impact:
```bash
# Baseline
python3 run_network_experiments.py --use-case emotion --baseline

# Test different latencies
for latency in 50 100 200 300; do
    python3 fl_network_monitor.py --all --latency ${latency}ms
    # Run experiment
    python3 run_network_experiments.py --single --protocol mqtt --scenario moderate
done
```

### 3. Real-time Monitoring
Monitor FL training with baseline comparison:
```bash
# Start training
docker-compose up -d

# Monitor with baseline comparison
python3 fl_training_dashboard.py --use-case emotion

# Apply network conditions dynamically
python3 fl_network_monitor.py --client-id 1 --latency 200ms
```

## 📊 Expected Performance Impact

| Network Scenario | Latency | Bandwidth | Loss | Expected RTT Increase |
|-----------------|---------|-----------|------|----------------------|
| Baseline | 0ms | Unlimited | 0% | 0% (reference) |
| Excellent | 5ms | 100mbit | 0.1% | +10-20% |
| Good | 20ms | 50mbit | 0.5% | +20-50% |
| Moderate | 50ms | 10mbit | 1% | +50-80% |
| Poor | 100ms | 5mbit | 2% | +80-120% |
| Very Poor | 200ms | 1mbit | 3% | +150-250% |
| Satellite | 600ms | 5mbit | 2% | +300-500% |

## 🔍 Verification

### Check Baseline Exists
```bash
ls -lR experiment_results_baseline/emotion/
```

### Verify RTT Data
```bash
cat experiment_results_baseline/emotion/mqtt_baseline/mqtt_baseline_rtt.json | jq
```

### Compare with Experiment
```bash
# Baseline
cat experiment_results_baseline/emotion/mqtt_baseline/mqtt_baseline_rtt.json | jq '.avg_rtt_per_round'

# Experiment
cat experiment_results/emotion_20260128/mqtt_poor/mqtt_rtt.json | jq '.avg_rtt_per_round'
```

## 🎉 Summary

**What You Can Do Now:**

1. ✅ **Run baseline experiments** without network conditions
2. ✅ **Track RTT** for each FL round automatically
3. ✅ **Compare performance** with baseline in real-time
4. ✅ **Measure network impact** quantitatively
5. ✅ **Analyze degradation** per protocol
6. ✅ **Monitor dynamically** during training

**Key Files:**
- `run_network_experiments.py` - Run experiments with `--baseline`
- `fl_training_dashboard.py` - Monitor with baseline comparison
- `FL_BASELINE_GUIDE.md` - Complete documentation
- `FL_BASELINE_QUICK_REF.md` - Quick commands

**Key Folders:**
- `experiment_results_baseline/` - Baseline results
- `experiment_results/` - Network experiments

**Start Here:**
```bash
# 1. Run baseline
python3 Network_Simulation/run_network_experiments.py --use-case emotion --baseline

# 2. Monitor
python3 Network_Simulation/fl_training_dashboard.py --use-case emotion
```

All features implemented and tested! ✅
