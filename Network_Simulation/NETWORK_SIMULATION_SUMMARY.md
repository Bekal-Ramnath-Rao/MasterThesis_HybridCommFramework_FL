# Network Simulation Tools - Quick Reference

## 📦 What Was Created

I've created a complete network simulation framework for your Federated Learning thesis project:

### 🛠️ New Tools

1. **[network_simulator.py](network_simulator.py)**
   - Apply network conditions (latency, jitter, bandwidth, packet loss) to running Docker containers
   - 6 predefined scenarios: excellent, good, moderate, poor, very_poor, satellite
   - Custom condition support

2. **[run_network_experiments.py](run_network_experiments.py)**
   - Automated experiment runner
   - Tests all protocols × all network scenarios
   - Collects results automatically

3. **[update_compose_for_network_sim.py](update_compose_for_network_sim.py)**
   - Helper to add NET_ADMIN capability to docker-compose files

### 📖 Documentation

1. **[README_NETWORK_SIMULATION.md](README_NETWORK_SIMULATION.md)** - Complete guide to network simulation
2. **[COMPLETE_EVALUATION_GUIDE.md](COMPLETE_EVALUATION_GUIDE.md)** - Step-by-step evaluation workflow
3. **[HOWTO_ADD_NET_ADMIN.md](HOWTO_ADD_NET_ADMIN.md)** - Guide for updating docker-compose files

### ✏️ Modified Files

1. **[Server/Dockerfile](Server/Dockerfile)** - Added `iproute2` package for `tc` command
2. **[Client/Dockerfile](Client/Dockerfile)** - Added `iproute2` package for `tc` command
3. **[README_DOCKER.md](README_DOCKER.md)** - Added link to network simulation guide

## 🚀 Quick Start

### 1. Prepare Docker Environment

```powershell
# Add cap_add: NET_ADMIN to all FL services in:
# - docker-compose-emotion.yml
# - docker-compose-mentalstate.yml  
# - docker-compose-temperature.yml
# See HOWTO_ADD_NET_ADMIN.md for details

# Rebuild images
docker-compose -f docker-compose-emotion.yml build
```

### 2. Test Network Simulation

```powershell
# Start test containers
docker-compose -f docker-compose-emotion.yml up -d fl-server-mqtt-emotion fl-client-mqtt-emotion-1

# Apply network conditions
python network_simulator.py --scenario moderate --pattern fl-client

# Verify
docker exec fl-client-mqtt-emotion-1 tc qdisc show dev eth0

# Cleanup
docker-compose -f docker-compose-emotion.yml down
```

### 3. Run Experiments

```powershell
# Single experiment (quick test)
python run_network_experiments.py --single --protocol mqtt --scenario poor --rounds 5

# Full evaluation (all protocols × all networks)
python run_network_experiments.py --use-case emotion --rounds 10
```

## 🌐 Available Network Scenarios

| Scenario | Latency | Jitter | Bandwidth | Loss | Use Case |
|----------|---------|--------|-----------|------|----------|
| `excellent` | 2ms | 0.5ms | 1000mbit | 0.01% | LAN |
| `good` | 10ms | 2ms | 100mbit | 0.1% | Broadband |
| `moderate` | 50ms | 10ms | 20mbit | 1% | 4G/LTE |
| `poor` | 100ms | 30ms | 2mbit | 3% | 3G |
| `very_poor` | 300ms | 100ms | 384kbit | 5% | Edge/2G |
| `satellite` | 600ms | 50ms | 5mbit | 2% | Satellite |

## 📊 Example Commands

### List Available Scenarios
```powershell
python network_simulator.py --list
```

### Apply Predefined Scenario
```powershell
# Apply to all FL containers
python network_simulator.py --scenario poor --pattern fl-

# Apply to MQTT clients only
python network_simulator.py --scenario moderate --pattern mqtt-client
```

### Apply Custom Conditions
```powershell
python network_simulator.py --custom `
    --latency 75ms `
    --jitter 20ms `
    --bandwidth 5mbit `
    --loss 2% `
    --pattern fl-client
```

### Reset to Normal
```powershell
python network_simulator.py --reset
```

### Run Specific Experiments
```powershell
# Test specific protocols
python run_network_experiments.py --protocols mqtt grpc --rounds 10

# Test specific scenarios
python run_network_experiments.py --scenarios moderate poor --rounds 10

# Combination
python run_network_experiments.py `
    --protocols mqtt amqp `
    --scenarios good moderate `
    --use-case emotion `
    --rounds 5
```

## 🔬 How It Works

### Linux Traffic Control (tc + netem)

Docker containers run Linux (even on Windows). The framework uses:

- **`tc`** (traffic control) - Linux kernel's traffic shaping utility
- **`netem`** (network emulator) - Kernel module for network emulation
- **`iproute2`** - Package providing tc command

Applied to container's `eth0` interface to simulate:
- Latency/delay
- Jitter (variable latency)
- Bandwidth limits
- Packet loss
- Packet reordering
- Packet corruption

### Container Privileges

Containers need `NET_ADMIN` capability to modify network settings:

```yaml
services:
  fl-client-mqtt-emotion-1:
    # ... other config ...
    cap_add:
      - NET_ADMIN
```

## 📁 Results Structure

```
experiment_results/
└── emotion_20250130_143022/
    ├── mqtt_excellent/
    │   ├── metadata.json
    │   ├── server_logs.txt
    │   └── mqtt_training_results.json
    ├── mqtt_poor/
    ├── amqp_excellent/
    └── ...
```

## ⏱️ Expected Duration

- **Single experiment**: 10-30 minutes (depending on rounds)
- **Minimal evaluation** (3 protocols × 3 scenarios × 5 rounds): ~3-4 hours
- **Standard evaluation** (5 protocols × 4 scenarios × 10 rounds): ~8-12 hours
- **Comprehensive** (5 protocols × 6 scenarios × 20 rounds): 20-40 hours

## 🎯 Research Questions You Can Answer

1. Which protocol performs best under high latency?
2. How does packet loss affect convergence?
3. What's the minimum bandwidth needed for each protocol?
4. Which protocol has lowest overhead?
5. Do network conditions affect model accuracy?
6. Which protocol is most resilient to poor networks?

## 📚 Documentation Hierarchy

1. **Start here**: [COMPLETE_EVALUATION_GUIDE.md](COMPLETE_EVALUATION_GUIDE.md)
2. **Technical details**: [README_NETWORK_SIMULATION.md](README_NETWORK_SIMULATION.md)
3. **Docker setup**: [README_DOCKER.md](README_DOCKER.md)
4. **Setup help**: [HOWTO_ADD_NET_ADMIN.md](HOWTO_ADD_NET_ADMIN.md)

## ⚠️ Important Notes

1. **Must rebuild images** after modifying Dockerfiles
2. **Must add cap_add** to docker-compose before running
3. **Reset conditions** between experiments or they persist
4. **Monitor disk space** - results can grow large
5. **Plan for time** - full evaluation takes hours/days

## 🆘 Common Issues

| Problem | Solution |
|---------|----------|
| "Permission denied" | Add `cap_add: [NET_ADMIN]` to service |
| "tc: command not found" | Rebuild images with updated Dockerfiles |
| Experiments timeout | Reduce network severity or increase timeout |
| Containers crash | Check Docker resources, reduce packet loss |

## 💡 Tips

- **Test incrementally**: Start with one protocol, one scenario
- **Start mild**: Use "good" or "moderate" scenarios first
- **Monitor logs**: Use `docker logs -f container-name`
- **Asymmetric testing**: Apply different conditions to different clients
- **Save results**: Backup `experiment_results/` regularly

## 🎓 Thesis Applications

Use these experiments to:
- Compare protocol performance quantitatively
- Create performance tables and graphs
- Analyze trade-offs (speed vs reliability)
- Justify protocol selection for specific use cases
- Demonstrate real-world applicability

## 📞 Next Steps

1. ✅ Read [COMPLETE_EVALUATION_GUIDE.md](COMPLETE_EVALUATION_GUIDE.md)
2. ✅ Update docker-compose files (add NET_ADMIN)
3. ✅ Rebuild Docker images
4. ✅ Test with single experiment
5. ✅ Run full evaluation
6. ✅ Analyze results
7. ✅ Write thesis chapter! 📝

Good luck with your research! 🚀
