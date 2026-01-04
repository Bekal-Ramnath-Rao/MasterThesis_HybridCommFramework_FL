# Network Simulation for Federated Learning Protocol Evaluation

This guide explains how to simulate various network conditions when testing your FL protocols in Docker containers using Linux traffic control (`tc`) and network emulation (`netem`).

## 📋 Overview

The framework provides two main tools for network simulation:

1. **[network_simulator.py](network_simulator.py)** - Apply network conditions to running containers
2. **[run_network_experiments.py](run_network_experiments.py)** - Automated experiment runner across all protocols and network scenarios

## 🔧 How It Works

### Linux Traffic Control (tc) + Network Emulation (netem)

Docker containers run on Linux (even on Windows/Mac via Docker Desktop's Linux VM). The `tc` (traffic control) command with the `netem` (network emulator) module allows you to:

- **Add latency/delay** - Simulate distance or processing delays
- **Add jitter** - Simulate variable latency
- **Limit bandwidth** - Simulate slow connections
- **Add packet loss** - Simulate unreliable networks
- **Reorder packets** - Simulate out-of-order delivery

These conditions are applied to the network interface inside each container, affecting all traffic to/from that container.

## 🌐 Predefined Network Scenarios

The framework includes 6 realistic network scenarios:

| Scenario | Latency | Jitter | Bandwidth | Loss | Description |
|----------|---------|--------|-----------|------|-------------|
| `excellent` | 2ms | 0.5ms | 1000mbit | 0.01% | Local Area Network (LAN) |
| `good` | 10ms | 2ms | 100mbit | 0.1% | Broadband/Fiber |
| `moderate` | 50ms | 10ms | 20mbit | 1% | 4G/LTE Mobile |
| `poor` | 100ms | 30ms | 2mbit | 3% | 3G Mobile |
| `very_poor` | 300ms | 100ms | 384kbit | 5% | Edge/2G |
| `satellite` | 600ms | 50ms | 5mbit | 2% | Satellite Internet |

## 🚀 Quick Start

### 1. Install Required Tools in Containers

The Docker images need to have `iproute2` package installed (provides `tc` command). Add this to your Dockerfiles:

```dockerfile
# Add to Server/Dockerfile and Client/Dockerfile
RUN apt-get update && apt-get install -y iproute2 && rm -rf /var/lib/apt/lists/*
```

### 2. Run Containers with Elevated Privileges

To use `tc`, containers need `NET_ADMIN` capability. Update your docker-compose files:

```yaml
services:
  fl-client-mqtt-emotion-1:
    # ... existing config ...
    cap_add:
      - NET_ADMIN
```

### 3. Apply Network Conditions

```bash
# Show all available scenarios
python network_simulator.py --list

# Apply 'poor' network conditions to all FL containers
python network_simulator.py --scenario poor --pattern fl-

# Apply 'moderate' network to only MQTT clients
python network_simulator.py --scenario moderate --pattern mqtt-client

# Apply custom conditions
python network_simulator.py --custom --latency 100ms --jitter 20ms --loss 2% --pattern fl-client

# Reset all containers to normal
python network_simulator.py --reset
```

## 📊 Running Automated Experiments

### Run All Experiments

Test all protocols across all network conditions:

```bash
# Run all experiments for Emotion Recognition
python run_network_experiments.py --use-case emotion --rounds 10

# Run all experiments for Mental State Recognition
python run_network_experiments.py --use-case mentalstate --rounds 5

# Test specific protocols only
python run_network_experiments.py --protocols mqtt amqp grpc

# Test specific network scenarios only
python run_network_experiments.py --scenarios good moderate poor
```

### Run Single Experiment

```bash
# Test MQTT under poor network conditions
python run_network_experiments.py --single --protocol mqtt --scenario poor --use-case emotion
```

### What Happens Automatically

For each protocol + network scenario combination:

1. ✓ Starts Docker containers for the protocol
2. ✓ Applies network conditions to all FL containers
3. ✓ Waits for FL training to complete
4. ✓ Collects results and logs
5. ✓ Stops containers
6. ✓ Moves to next experiment

Results are saved in `experiment_results/[use_case]_[timestamp]/[protocol]_[scenario]/`

## 📁 Experiment Results Structure

```
experiment_results/
└── emotion_20250130_143022/
    ├── mqtt_excellent/
    │   ├── metadata.json
    │   ├── server_logs.txt
    │   └── mqtt_training_results.json
    ├── mqtt_good/
    │   └── ...
    ├── amqp_excellent/
    │   └── ...
    └── ...
```

## 🔍 Manual Network Condition Management

### Apply Conditions to Specific Container

```python
from network_simulator import NetworkSimulator

sim = NetworkSimulator(verbose=True)

# Custom conditions
conditions = {
    "latency": "50ms",
    "jitter": "10ms", 
    "bandwidth": "10mbit",
    "loss": "2%"
}

sim.apply_network_conditions("fl-client-mqtt-emotion-1", conditions)
```

### Verify Applied Conditions

```bash
# Check tc rules on a container
docker exec fl-client-mqtt-emotion-1 tc qdisc show dev eth0
```

Expected output:
```
qdisc netem 1: root refcnt 2 limit 1000 delay 50.0ms 10.0ms loss 2%
```

### Remove Conditions

```bash
# Reset specific container
docker exec fl-client-mqtt-emotion-1 tc qdisc del dev eth0 root

# Or use the script
python network_simulator.py --reset --pattern fl-client-mqtt
```

## 🎯 Best Practices

### 1. Apply to Clients Only (Recommended)

Apply network conditions to client containers to simulate real-world edge devices:

```bash
python network_simulator.py --scenario moderate --pattern fl-client
```

This simulates clients with poor connectivity while keeping server-broker communication fast.

### 2. Apply to All FL Components

For complete network simulation:

```bash
python network_simulator.py --scenario poor --pattern fl-
```

### 3. Gradual Testing

Start with better networks and gradually test worse conditions:

```bash
# Test progression
python network_simulator.py --scenario excellent --pattern fl-
# Run experiment, collect results
python network_simulator.py --scenario good --pattern fl-
# Run experiment, collect results
python network_simulator.py --scenario moderate --pattern fl-
# And so on...
```

## 🛠️ Troubleshooting

### Error: "Operation not permitted"

**Problem**: Container doesn't have NET_ADMIN capability

**Solution**: Add to docker-compose:
```yaml
cap_add:
  - NET_ADMIN
```

### Network conditions not applying

**Problem**: `tc` command not found in container

**Solution**: Install `iproute2` in Dockerfile:
```dockerfile
RUN apt-get update && apt-get install -y iproute2
```

### Containers can't communicate

**Problem**: Too aggressive network conditions

**Solution**: 
- Reduce packet loss: `--loss 1%` instead of `--loss 10%`
- Increase bandwidth: `--bandwidth 5mbit` instead of `--bandwidth 100kbit`
- Check broker timeouts in your FL code

### Bandwidth limit not working with netem

**Problem**: Can't combine `tbf` (token bucket filter) and `netem` on same interface

**Solution**: The script applies them separately. If you need both:
```bash
# Use hierarchical tc setup (advanced)
docker exec container tc qdisc add dev eth0 root handle 1: tbf rate 10mbit burst 32kbit latency 400ms
docker exec container tc qdisc add dev eth0 parent 1:1 handle 10: netem delay 50ms loss 2%
```

## 📈 Analyzing Results

### Compare Protocol Performance

After running experiments, compare results:

```python
import json
import pandas as pd

# Load results
mqtt_good = json.load(open("experiment_results/.../mqtt_good/mqtt_training_results.json"))
mqtt_poor = json.load(open("experiment_results/.../mqtt_poor/mqtt_training_results.json"))

# Compare metrics
print(f"MQTT Good Network - Avg Round Time: {mqtt_good['avg_round_time']}")
print(f"MQTT Poor Network - Avg Round Time: {mqtt_poor['avg_round_time']}")
```

### Key Metrics to Compare

- **Round completion time** - How network affects training speed
- **Model convergence** - Whether poor networks affect accuracy
- **Message delivery rate** - Packet loss impact
- **Total training time** - Overall efficiency
- **Bandwidth usage** - Protocol overhead in constrained networks

## 🔬 Advanced Scenarios

### Asymmetric Network Conditions

Apply different conditions to different clients:

```bash
# Client 1: Good network
python network_simulator.py --scenario good --pattern fl-client-mqtt-emotion-1

# Client 2: Poor network  
python network_simulator.py --scenario poor --pattern fl-client-mqtt-emotion-2
```

### Dynamic Conditions During Training

```python
# Apply changing network conditions during training
import time
from network_simulator import NetworkSimulator

sim = NetworkSimulator()

# Start with good network
sim.apply_scenario_to_containers("good", "fl-client")
time.sleep(300)  # 5 minutes

# Degrade to moderate
sim.apply_scenario_to_containers("moderate", "fl-client")
time.sleep(300)  # 5 minutes

# Further degrade to poor
sim.apply_scenario_to_containers("poor", "fl-client")
```

### Packet Reordering

```bash
docker exec fl-client-mqtt-1 tc qdisc add dev eth0 root netem delay 10ms reorder 25% 50%
```

### Packet Duplication

```bash
docker exec fl-client-mqtt-1 tc qdisc add dev eth0 root netem duplicate 1%
```

### Packet Corruption

```bash
docker exec fl-client-mqtt-1 tc qdisc add dev eth0 root netem corrupt 0.1%
```

## 📚 Additional Resources

- [Linux tc man page](https://man7.org/linux/man-pages/man8/tc.8.html)
- [netem documentation](https://wiki.linuxfoundation.org/networking/netem)
- [Docker networking](https://docs.docker.com/network/)

## 💡 Example Workflow

Complete workflow for comprehensive protocol evaluation:

```bash
# 1. Update Dockerfiles to include iproute2
# 2. Update docker-compose files to add NET_ADMIN capability
# 3. Rebuild images
docker-compose build

# 4. Run automated experiments for all protocols and conditions
python run_network_experiments.py --use-case emotion --rounds 10

# 5. Analyze results
# Results in: experiment_results/emotion_[timestamp]/

# 6. Optional: Run specific test with custom conditions
python run_network_experiments.py --single \
  --protocol mqtt \
  --scenario moderate \
  --use-case emotion \
  --rounds 5
```

## 🎓 Understanding the Results

Network conditions affect FL protocols differently:

- **MQTT/AMQP**: Broker-based, may queue messages during poor conditions
- **gRPC**: HTTP/2 based, built-in flow control and multiplexing
- **QUIC**: UDP-based with built-in congestion control, may perform better under packet loss
- **DDS**: Peer-to-peer, may have different behavior with discovery under latency

Your experiments will reveal which protocol is most robust under specific conditions!
