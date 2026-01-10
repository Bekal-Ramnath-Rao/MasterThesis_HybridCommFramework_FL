# Network Congestion Testing Guide

## Overview

This guide explains how to test your Federated Learning protocols under realistic network congestion using traffic generator containers.

## 🎯 What's New

We've added **artificial network congestion** capabilities to complement the existing network condition simulation:

1. **Network Condition Simulation** (tc-based) - Simulates latency, jitter, bandwidth limits, packet loss
2. **Traffic Generation** (container-based) - Creates actual network congestion with background traffic
3. **Combined Testing** - Tests protocols under both degraded conditions AND competing traffic

## 📁 New Files Created

### 1. `congestion_manager.py`
Manages traffic generator containers to create different levels of network congestion.

**Congestion Levels:**
- `none` - No traffic generators (baseline)
- `light` - 1-2 HTTP traffic generators
- `moderate` - HTTP + bandwidth hog
- `heavy` - All generators + packet spammer  
- `extreme` - Maximum congestion with all generators

### 2. `docker-compose-traffic-generators.yml`
Docker Compose file defining various traffic generator containers:
- **HTTP Traffic Generators** - Simulates web browsing
- **Bandwidth Hog** - Simulates large file transfers
- **Packet Spammer** - Creates many small packets
- **Connection Flooder** - Opens many concurrent connections

### 3. `run_congestion_experiments.py`
Automated experiment runner that combines network simulation with traffic generation.

**Predefined Scenarios:**
- `baseline` - Good network, no congestion
- `light_load` - Good network with light traffic
- `moderate_load` - Moderate network with moderate traffic
- `heavy_load` - Moderate network with heavy traffic
- `extreme_load` - Poor network with extreme traffic
- `congestion_only` - Excellent network with heavy congestion (isolates effect)
- `degraded_congested` - Uses congested_moderate scenario + generators

### 4. Updated Network Scenarios
Added three new network scenarios to `network_simulator.py`:
- `congested_light` - Light congestion (30ms latency, 10mbit, 1.5% loss)
- `congested_moderate` - Moderate congestion (75ms latency, 5mbit, 3.5% loss)
- `congested_heavy` - Heavy congestion (150ms latency, 2mbit, 6% loss)

## 🚀 Quick Start

### 1. List Available Options

```bash
# List congestion scenarios
python Network_Simulation/run_congestion_experiments.py --list

# List congestion levels
python Network_Simulation/congestion_manager.py --list
```

### 2. Run a Single Congestion Test

```bash
# Test MQTT under moderate load
python Network_Simulation/run_congestion_experiments.py \
  --use-case temperature \
  --protocols mqtt \
  --scenario moderate_load \
  --rounds 100
```

### 3. Run Multiple Protocols Under Congestion

```bash
# Compare all protocols under heavy load
python Network_Simulation/run_congestion_experiments.py \
  --use-case temperature \
  --protocols mqtt amqp grpc dds quic \
  --scenario heavy_load \
  --rounds 100
```

### 4. Run All Congestion Scenarios

```bash
# Comprehensive congestion testing
python Network_Simulation/run_congestion_experiments.py \
  --use-case temperature \
  --protocols mqtt grpc \
  --all-scenarios \
  --rounds 50
```

## 🔧 Manual Control

### Start Traffic Generators

```bash
# Start moderate congestion
python Network_Simulation/congestion_manager.py --start --level moderate

# Start heavy congestion
python Network_Simulation/congestion_manager.py --start --level heavy
```

### Check Status

```bash
python Network_Simulation/congestion_manager.py --status
```

### Stop Traffic Generators

```bash
python Network_Simulation/congestion_manager.py --stop
```

### Scale Congestion

```bash
# Gradually increase congestion
python Network_Simulation/congestion_manager.py --scale-from light --scale-to heavy
```

## 🧪 Experiment Design

### Recommended Test Matrix

| Scenario | Network Condition | Congestion Level | Purpose |
|----------|------------------|------------------|---------|
| baseline | good | none | Baseline performance |
| light_load | good | light | Minimal competition |
| moderate_load | moderate | moderate | Typical shared network |
| heavy_load | moderate | heavy | Peak usage |
| extreme_load | poor | extreme | Worst case |
| congestion_only | excellent | heavy | Isolate congestion effect |

### What Gets Tested

1. **Protocol Overhead** - How well protocols handle competing traffic
2. **Congestion Control** - Built-in congestion avoidance mechanisms
3. **Connection Stability** - Ability to maintain connections under load
4. **Throughput** - Actual data transfer rates with background traffic
5. **Latency** - Real-world delays including queueing from congestion

## 📊 Example Usage Workflow

```bash
# 1. List available scenarios
python Network_Simulation/run_congestion_experiments.py --list

# 2. Test one protocol under baseline
python Network_Simulation/run_congestion_experiments.py \
  --use-case temperature \
  --protocols mqtt \
  --scenario baseline \
  --rounds 100

# 3. Test same protocol under extreme load
python Network_Simulation/run_congestion_experiments.py \
  --use-case temperature \
  --protocols mqtt \
  --scenario extreme_load \
  --rounds 100

# 4. Compare results to see congestion impact
# Results will be in experiment_results/temperature_TIMESTAMP/
```

## 🔍 Understanding Results

Results will be saved in `experiment_results/` with the following structure:

```
experiment_results/
└── temperature_20260108_150000/
    ├── mqtt_baseline_good/
    │   ├── server_logs.txt
    │   ├── metadata.json
    │   └── mqtt_training_results.json
    ├── mqtt_moderate_load_moderate/
    │   └── ...
    └── mqtt_extreme_load_poor/
        └── ...
```

### Key Metrics to Compare

- **Training Time** - How long each scenario took
- **Communication Overhead** - Bytes transferred vs. baseline
- **Convergence** - Model accuracy achieved
- **Round Duration** - Time per FL round
- **Error Rates** - Failed communications or timeouts

## 💡 Advanced Usage

### Custom Congestion Test

```python
# Create your own test script
from Network_Simulation.congestion_manager import CongestionManager
from Network_Simulation.run_network_experiments import ExperimentRunner

# Initialize
manager = CongestionManager(use_case="temperature")
runner = ExperimentRunner(use_case="temperature", num_rounds=100)

# Start FL containers
runner.start_containers("mqtt", "moderate")

# Apply network conditions
runner.apply_network_scenario("moderate", "mqtt")

# Add congestion
manager.start_traffic_generators("heavy")

# Wait for completion
runner.wait_for_completion("mqtt")

# Cleanup
manager.stop_traffic_generators()
runner.stop_containers("mqtt")
```

### Modify Traffic Generators

Edit `Docker/docker-compose-traffic-generators.yml` to:
- Adjust traffic intensity
- Add new traffic patterns
- Target specific services
- Change traffic timing

## 🐛 Troubleshooting

### Issue: Networks don't exist

```bash
# Manually create networks
docker network create fl-mqtt-network-temp
docker network create fl-amqp-network-temp
docker network create fl-grpc-network-temp
docker network create fl-quic-network-temp
docker network create fl-dds-network-temp
```

### Issue: Traffic generators won't start

```bash
# Check Docker logs
docker logs http-traffic-gen-1

# Ensure FL containers are running first
docker ps | grep fl-
```

### Issue: Cleanup needed

```bash
# Stop everything
docker-compose -f Docker/docker-compose-temperature.yml down
docker-compose -f Docker/docker-compose-traffic-generators.yml down

# Remove all networks
docker network prune -f
```

## 📈 Integration with Existing Tests

The new congestion scenarios are fully integrated with your existing setup:

```bash
# Use new scenarios in existing experiment runner
python Network_Simulation/run_network_experiments.py \
  --use-case temperature \
  --protocols mqtt amqp grpc dds quic \
  --scenarios excellent good moderate congested_light congested_moderate congested_heavy \
  --rounds 1000
```

This adds the congestion scenarios alongside your standard network scenarios.

## 🎓 Research Implications

This congestion testing allows you to evaluate:

1. **Real-World Performance** - Beyond idealized network conditions
2. **Scalability** - How protocols handle network competition
3. **Robustness** - Stability under varying loads
4. **Protocol Comparison** - Which handles congestion best
5. **Adaptive Behavior** - How protocols respond to changing conditions

## 🔄 Next Steps

1. Run baseline tests without congestion
2. Run tests with increasing congestion levels
3. Compare metrics across scenarios
4. Identify protocol strengths/weaknesses
5. Document findings in your thesis

---

**Note:** All traffic generators run in the background and automatically reconnect. They can run indefinitely or until manually stopped.
