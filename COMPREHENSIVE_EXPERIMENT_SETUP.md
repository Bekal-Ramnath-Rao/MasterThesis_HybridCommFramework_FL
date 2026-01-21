# Comprehensive Experiment Runner - Complete Setup

## Summary

A complete shell script automation system has been created to run federated learning experiments across:

### Test Scope
- **3 Use Cases**: Emotion Recognition, Mental State Recognition, Temperature Regulation
- **5 Communication Protocols**: MQTT, AMQP, gRPC, QUIC, DDS
- **9 Network Scenarios**: excellent, good, moderate, poor, very_poor, satellite, congested_light, congested_moderate, congested_heavy
- **2 Quantization Configurations**: Disabled (Scenario 1), Enabled (Scenario 2)

**Total: 270 experiments** (3 × 5 × 9 × 2)

## Files Created

### 1. **run_comprehensive_experiments.sh** (13KB, executable)
Main experiment automation script with features:
- ✅ Automatic environment verification (Docker, Docker Compose V2, GPU, Python3)
- ✅ Flexible filtering (use-case, scenario, protocol, rounds)
- ✅ Comprehensive logging system
- ✅ Automatic Docker image building
- ✅ Result consolidation
- ✅ Error handling and progress tracking
- ✅ Color-coded output for clarity

### 2. **EXPERIMENT_RUNNER_GUIDE.md**
Comprehensive guide including:
- Quick start instructions
- Advanced usage patterns
- Experiment matrix breakdown
- Execution timeline estimates
- Output structure documentation
- Monitoring instructions
- Troubleshooting guide
- Environment variable options

### 3. **EXPERIMENT_QUICK_COMMANDS.sh** 
Ready-to-copy command reference with:
- 30+ pre-configured experiment commands
- Scenario 1 variations (without quantization)
- Scenario 2 variations (with quantization)
- Validation commands for quick testing
- Advanced option combinations
- Network scenario definitions
- Monitoring commands
- Result analysis commands
- Sequential execution recommendations

## Quick Start

### 1. Verify Setup
```bash
./run_comprehensive_experiments.sh --help
```

### 2. Quick Validation (2 minutes)
Test that everything works:
```bash
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --protocol mqtt --rounds 5
```

### 3. Run Scenario 1 - Without Quantization (~15 hours)
```bash
./run_comprehensive_experiments.sh --scenario 1 --rounds 100
```

### 4. Run Scenario 2 - With Quantization (~15 hours)
```bash
./run_comprehensive_experiments.sh --scenario 2 --rounds 100
```

## Usage Examples

### Run All Experiments (Complete Suite)
```bash
./run_comprehensive_experiments.sh
```
- Builds all Docker images
- Runs 270 experiments (45+ hours estimated)
- Consolidates all results

### Emotion Recognition Only
```bash
./run_comprehensive_experiments.sh --use-case emotion --rounds 100
```
- 90 experiments (Scenario 1 + Scenario 2)
- ~7.5 hours estimated

### Specific Protocol (MQTT) - All Scenarios
```bash
./run_comprehensive_experiments.sh --protocol mqtt --rounds 100
```
- 54 experiments (all use cases + both scenarios)
- All 9 network scenarios

### Scenario 1 Only (Without Quantization)
```bash
./run_comprehensive_experiments.sh --scenario 1 --rounds 100
```
- 135 experiments
- ~22-23 hours estimated

### Skip Docker Build (Use Cached Images)
```bash
./run_comprehensive_experiments.sh --skip-build --rounds 100
```
- Speeds up start time
- Uses pre-built Docker images

## Experiment Matrix

### By Use Case (90 experiments each)

```
Emotion Recognition:
  9 network scenarios × 5 protocols × 2 quantization modes = 90 experiments
  Estimated time: 7.5 hours @ 100 rounds

Mental State Recognition:
  9 network scenarios × 5 protocols × 2 quantization modes = 90 experiments
  Estimated time: 7.5 hours @ 100 rounds

Temperature Regulation:
  9 network scenarios × 5 protocols × 2 quantization modes = 90 experiments
  Estimated time: 7.5 hours @ 100 rounds
```

### By Protocol (54 experiments each)

```
MQTT:    9 scenarios × 3 use cases × 2 quant modes = 54 experiments
AMQP:    9 scenarios × 3 use cases × 2 quant modes = 54 experiments
gRPC:    9 scenarios × 3 use cases × 2 quant modes = 54 experiments
QUIC:    9 scenarios × 3 use cases × 2 quant modes = 54 experiments
DDS:     9 scenarios × 3 use cases × 2 quant modes = 54 experiments
```

### By Scenario (135 experiments each)

```
Scenario 1 (No Quantization):
  9 scenarios × 5 protocols × 3 use cases = 135 experiments
  Estimated time: 22.5 hours @ 100 rounds

Scenario 2 (With Quantization):
  9 scenarios × 5 protocols × 3 use cases = 135 experiments
  Estimated time: 22.5 hours @ 100 rounds
```

## Features

### Automatic Environment Verification
```
✓ Docker installation check
✓ Docker Compose V2 availability
✓ NVIDIA GPU detection and count
✓ Python3 availability
✓ Project structure validation
```

### Comprehensive Logging
- **Per-experiment logs**: `emotion_scenario1_excellent_disabled_mqtt.log`
- **Build logs**: `build_emotion.log`, `build_mentalstate.log`, `build_temperature.log`
- **Consolidation logs**: `consolidate_emotion.log`, etc.
- **All organized** in: `logs/experiments_YYYYMMDD_HHMMSS/`

### Real-Time Progress Tracking
- Color-coded status messages (success/warning/error/info)
- Per-scenario summary statistics
- Per-use-case failure tracking
- Experiment completion status

### Automated Result Consolidation
- Automatically consolidates results after each use case
- Extracts training metrics from logs
- Saves to `Server/{UseCase}_Regulation/results/`
- Can be skipped with `--skip-consolidate`

### Smart Docker Management
- Automatic image building with `--no-cache`
- Separate compose files per use case
- GPU device isolation (Client 1 → GPU 0, Client 2 → GPU 1)
- Can skip build with cached images

## Configuration Options

### Command Line Arguments
```bash
--use-case [emotion|mentalstate|temperature]
  Run only specific use case (default: all 3)

--scenario [1|2]
  Run only specific scenario (default: both)
  1 = Without quantization
  2 = With quantization

--protocol [mqtt|amqp|grpc|quic|dds]
  Run only specific protocol (default: all 5)

--rounds N
  Number of training rounds per experiment (default: 100)

--skip-build
  Skip Docker image building, use cached images

--skip-consolidate
  Skip automatic result consolidation

--help
  Show help message and usage examples
```

### Environment Variables
```bash
export ROUNDS=150                # Override round count
export ENABLE_GPU=false          # Force CPU mode
export SKIP_CONSOLIDATE=true     # Skip consolidation by default
./run_comprehensive_experiments.sh
```

## Execution Timeline

### Estimated Durations (with 100 rounds per experiment)

| Configuration | Time | Details |
|---|---|---|
| Quick validation | 2 min | 1 use case, 1 protocol, 1 scenario, 5 rounds |
| Single protocol | 7.5 hours | All scenarios, all use cases, 1 protocol |
| Single use case | 7.5 hours | All scenarios, all protocols, 1 use case |
| Scenario 1 only | 22.5 hours | All 9 scenarios, all 5 protocols, all 3 use cases, no quantization |
| Scenario 2 only | 22.5 hours | All 9 scenarios, all 5 protocols, all 3 use cases, with quantization |
| **Complete suite** | **45+ hours** | All 270 experiments |

### Speedup Options
- Use `--rounds 50` for ~50% faster execution
- Use `--skip-build` to avoid rebuild time
- Run specific use cases sequentially instead of all at once

## Output Structure

```
Project Root/
├── run_comprehensive_experiments.sh       (Main script)
├── EXPERIMENT_RUNNER_GUIDE.md            (Full documentation)
├── EXPERIMENT_QUICK_COMMANDS.sh           (Command reference)
│
├── logs/experiments_YYYYMMDD_HHMMSS/     (All experiment logs)
│   ├── emotion_scenario1_excellent_disabled_mqtt.log
│   ├── emotion_scenario1_good_disabled_mqtt.log
│   ├── ... (one log per experiment)
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
│   │   │   ├── metadata.json
│   │   │   ├── mqtt_training_results.json
│   │   │   └── server_logs.txt
│   │   ├── mqtt_good/
│   │   └── ... (all 9 scenarios × 5 protocols)
│   ├── mentalstate_YYYYMMDD_HHMMSS/
│   └── temperature_YYYYMMDD_HHMMSS/
│
└── Server/{UseCase}_Regulation/results/   (Consolidated metrics)
    ├── mqtt_excellent_training_results.json
    ├── mqtt_good_training_results.json
    └── ... (consolidated per scenario & protocol)
```

## Monitoring

### Watch Real-Time GPU Usage
```bash
nvidia-smi -l 1
```

### Monitor Container Resources
```bash
docker stats
```

### View Live Experiment Logs
```bash
tail -f logs/experiments_*/emotion_scenario1*.log
```

### Count Completed Experiments
```bash
ls logs/experiments_*/*.log | wc -l
```

## Common Workflows

### Workflow 1: Validate & Full Run
```bash
# 1. Quick validation (2 minutes)
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --protocol mqtt --rounds 5

# 2. Full scenario 1 (22.5 hours)
./run_comprehensive_experiments.sh --scenario 1

# 3. Full scenario 2 (22.5 hours)
./run_comprehensive_experiments.sh --scenario 2
```

### Workflow 2: Protocol Comparison
```bash
# Run each protocol separately to compare
for protocol in mqtt amqp grpc quic dds; do
    ./run_comprehensive_experiments.sh --protocol $protocol --scenario 1 --rounds 100
done
```

### Workflow 3: Use Case Analysis
```bash
# Run each use case separately
for use_case in emotion mentalstate temperature; do
    ./run_comprehensive_experiments.sh --use-case $use_case --rounds 100
done
```

## Troubleshooting

### Docker Build Failures
```bash
cat logs/experiments_YYYYMMDD_HHMMSS/build_emotion.log
```

### Specific Experiment Failure
```bash
cat logs/experiments_YYYYMMDD_HHMMSS/emotion_scenario1_excellent_disabled_mqtt.log
```

### GPU Memory Issues
```bash
# Reduce rounds and skip consolidation
./run_comprehensive_experiments.sh --rounds 20 --skip-consolidate
```

### Resume After Interruption
```bash
# Skip build (images cached) and continue
./run_comprehensive_experiments.sh --skip-build
```

## Requirements

- **Docker**: Version 20.10+
- **Docker Compose V2**: Already installed
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **CPU**: Minimum 8 cores (16+ recommended)
- **RAM**: Minimum 32GB (more for parallel containers)
- **Disk**: 10GB+ free space for logs and results
- **Network**: Stable network for multi-container setup

## Success Indicators

✅ Script runs without errors
✅ Docker images build successfully
✅ Experiments complete with training logs
✅ Results consolidate without errors
✅ GPU memory stays within limits
✅ No OOM (Out of Memory) errors
✅ All log files created in logs directory
✅ Result JSON files created in results directory

