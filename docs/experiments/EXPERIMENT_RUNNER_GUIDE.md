# Comprehensive Experiment Runner - Usage Guide

## Overview

The `run_comprehensive_experiments.sh` script automates running federated learning experiments across:
- **3 Use Cases**: Emotion Recognition, Mental State Recognition, Temperature Regulation
- **5 Protocols**: MQTT, AMQP, gRPC, QUIC, DDS
- **9 Network Scenarios**: excellent, good, moderate, poor, very_poor, satellite, congested_light, congested_moderate, congested_heavy
- **2 Quantization Scenarios**: Disabled and Enabled

## Quick Start

### Run All Experiments (Complete Suite)
```bash
./run_comprehensive_experiments.sh
```

### Run with Custom Rounds
```bash
./run_comprehensive_experiments.sh --rounds 100
```

### Run Single Use Case
```bash
./run_comprehensive_experiments.sh --use-case emotion
./run_comprehensive_experiments.sh --use-case mentalstate
./run_comprehensive_experiments.sh --use-case temperature
```

### Run Single Scenario
```bash
# Scenario 1: Without Quantization
./run_comprehensive_experiments.sh --scenario 1

# Scenario 2: With Quantization
./run_comprehensive_experiments.sh --scenario 2
```

### Run Single Protocol for All Scenarios
```bash
./run_comprehensive_experiments.sh --protocol mqtt
./run_comprehensive_experiments.sh --protocol amqp
./run_comprehensive_experiments.sh --protocol grpc
./run_comprehensive_experiments.sh --protocol quic
./run_comprehensive_experiments.sh --protocol dds
```

## Advanced Usage

### Combine Filters
```bash
# Emotion use case, scenario 1 only, MQTT protocol
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --protocol mqtt

# Mental state, 50 rounds, skip consolidation
./run_comprehensive_experiments.sh --use-case mentalstate --rounds 50 --skip-consolidate

# All experiments, skip Docker build (use cached images)
./run_comprehensive_experiments.sh --skip-build
```

## Script Features

### ✓ Automatic Environment Verification
- Checks Docker installation
- Verifies Docker Compose V2
- Detects GPU availability
- Validates project structure

### ✓ Comprehensive Logging
- Separate log file for each experiment
- Build logs for each use case
- Consolidation logs
- All logs stored in `logs/experiments_YYYYMMDD_HHMMSS/`

### ✓ Progress Tracking
- Real-time status updates with color coding
- Summary statistics for each scenario
- Estimated total time calculation

### ✓ Error Handling
- Graceful failure handling
- Detailed error messages
- Failed experiment logs preserved

### ✓ Result Consolidation
- Automatically consolidates results after each use case
- Can be skipped with `--skip-consolidate`
- Results saved in `experiment_results/`

## Experiment Matrix

### Total Experiments

| Use Cases | Protocols | Scenarios | Quantization | Total |
|-----------|-----------|-----------|--------------|-------|
| 3 | 5 | 9 | 2 | **270** |

Each experiment = 100+ training rounds (configurable)

### Breakdown by Use Case

For **each** use case:
- 9 network scenarios × 5 protocols × 2 quantization modes = **90 experiments**
- Total across 3 use cases = **270 experiments**

## Execution Timeline

### With 100 rounds per experiment:
- **Estimated duration**: ~45 hours (continuous)
- **Per use case**: ~15 hours
- **Per scenario**: ~7.5 hours

### With 50 rounds per experiment (faster):
- **Estimated duration**: ~22 hours
- **Per use case**: ~7 hours

## Output Structure

```
├── logs/experiments_YYYYMMDD_HHMMSS/
│   ├── emotion_scenario1_excellent_disabled_mqtt.log
│   ├── emotion_scenario1_good_disabled_mqtt.log
│   ├── ...
│   ├── build_emotion.log
│   ├── build_mentalstate.log
│   ├── build_temperature.log
│   ├── consolidate_emotion.log
│   ├── consolidate_mentalstate.log
│   └── consolidate_temperature.log
│
├── experiment_results/
│   ├── emotion_YYYYMMDD_HHMMSS/
│   │   ├── mqtt_excellent/
│   │   ├── mqtt_good/
│   │   └── ...
│   ├── mentalstate_YYYYMMDD_HHMMSS/
│   └── temperature_YYYYMMDD_HHMMSS/
```

## Monitoring During Execution

### Watch GPU Usage
```bash
nvidia-smi -l 1  # Update every 1 second
```

### Monitor Docker Containers
```bash
docker stats  # Real-time container stats
```

### View Current Logs
```bash
tail -f logs/experiments_YYYYMMDD_HHMMSS/*.log
```

## Common Use Cases

### 1. Quick Validation Run
Test setup with minimal experiments:
```bash
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --protocol mqtt --rounds 5
```

### 2. Single Use Case Full Suite
Run all experiments for one use case only:
```bash
./run_comprehensive_experiments.sh --use-case emotion
```

### 3. Protocol Comparison
Compare all protocols on same scenario:
```bash
./run_comprehensive_experiments.sh --scenario 1 --rounds 100
```

### 4. Quantization Impact Analysis
Run with and without quantization:
```bash
# Without quantization
./run_comprehensive_experiments.sh --scenario 1

# With quantization
./run_comprehensive_experiments.sh --scenario 2
```

### 5. Resume After Interruption
Skip build (images already created), continue with different options:
```bash
./run_comprehensive_experiments.sh --skip-build --use-case temperature --rounds 100
```

## Troubleshooting

### Build Failures
Check the build log:
```bash
cat logs/experiments_YYYYMMDD_HHMMSS/build_emotion.log
```

### Experiment Failures
Check specific experiment log:
```bash
cat logs/experiments_YYYYMMDD_HHMMSS/emotion_scenario1_excellent_disabled_mqtt.log
```

### GPU Memory Issues
Reduce rounds and use `--skip-consolidate`:
```bash
./run_comprehensive_experiments.sh --rounds 20 --skip-consolidate
```

### Consolidation Failures
Run manually after fixing issues:
```bash
python3 Network_Simulation/consolidate_results.py --use-case emotion
```

## Notes

- **GPU Required**: Experiments run with GPU enabled (--enable-gpu)
- **Disk Space**: Ensure sufficient disk space for logs and results (~10GB+ recommended)
- **Network**: Stable network required for multi-container Docker networking
- **CPU Cores**: Minimum 8 cores recommended (experiments run in parallel containers)
- **Memory**: Minimum 32GB RAM recommended (for running all services)

## Environment Variables

Override defaults:
```bash
# Custom rounds
export ROUNDS=150

# Disable GPU (use CPU only)
export ENABLE_GPU=false

# Skip consolidation by default
export SKIP_CONSOLIDATE=true

./run_comprehensive_experiments.sh
```

## Support

For issues or questions:
1. Check logs in `logs/experiments_YYYYMMDD_HHMMSS/`
2. Review error messages in console output
3. Verify environment with `./run_comprehensive_experiments.sh --help`
4. Check project documentation in repository

