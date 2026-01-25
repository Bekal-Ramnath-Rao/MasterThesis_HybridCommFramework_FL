# Network Conditions - Quick Reference

## 🚀 Quick Start

### Apply Network Condition (Local Execution)

```bash
# Method 1: Using environment variables (recommended)
export APPLY_NETWORK_CONDITION=true
export NETWORK_CONDITION=poor
python Server/Emotion_Recognition/FL_Server_MQTT.py

# Method 2: Apply before running
python network_condition_manager.py apply --condition poor
python Server/Emotion_Recognition/FL_Server_MQTT.py
python network_condition_manager.py reset
```

### Apply Network Condition (Docker)

```bash
# Set in docker-compose.yml or override:
APPLY_NETWORK_CONDITION=true NETWORK_CONDITION=poor \
  docker-compose -f Docker/docker-compose-emotion.yml up
```

---

## 📋 Available Conditions

| Condition | Use Case | Latency | Bandwidth | Loss |
|-----------|----------|---------|-----------|------|
| `excellent` | LAN/Data Center | 2ms | 1000 Mbps | 0.01% |
| `good` | Broadband/Fiber | 10ms | 100 Mbps | 0.1% |
| `moderate` | 4G/LTE | 50ms | 20 Mbps | 1% |
| `poor` | 3G Mobile | 100ms | 2 Mbps | 3% |
| `very_poor` | Edge/2G | 300ms | 384 Kbps | 5% |
| `satellite` | Satellite Link | 600ms | 5 Mbps | 2% |
| `none` | No conditions | 0ms | Unlimited | 0% |

---

## 🔧 Commands

```bash
# List all conditions
python network_condition_manager.py list

# Apply predefined condition
python network_condition_manager.py apply --condition poor

# Apply custom condition
python network_condition_manager.py apply \
  --latency 100 \
  --jitter 30 \
  --bandwidth 2 \
  --loss 3

# Reset (remove all conditions)
python network_condition_manager.py reset

# Check current conditions
sudo tc qdisc show dev lo  # Local
sudo tc qdisc show dev eth0  # Docker
```

---

## 🎯 Environment Variables

```bash
# Enable network conditions
export APPLY_NETWORK_CONDITION=true

# Set condition name
export NETWORK_CONDITION=poor

# Custom conditions (alternative)
export NETWORK_LATENCY_MS=100
export NETWORK_JITTER_MS=30
export NETWORK_BANDWIDTH_MBPS=2
export NETWORK_LOSS_PERCENT=3
```

---

## 📝 Examples

### Example 1: Test All Protocols with Poor Network

```bash
#!/bin/bash
export APPLY_NETWORK_CONDITION=true
export NETWORK_CONDITION=poor

# Test MQTT
python Server/Emotion_Recognition/FL_Server_MQTT.py &
sleep 5
python Client/Emotion_Recognition/FL_Client_MQTT.py

# Test gRPC
python Server/Emotion_Recognition/FL_Server_gRPC.py &
sleep 5
python Client/Emotion_Recognition/FL_Client_gRPC.py

# Reset
python network_condition_manager.py reset
```

### Example 2: Compare Good vs Poor Network

```bash
# Run with good network
python network_condition_manager.py apply --condition good
python Server/Emotion_Recognition/FL_Server_DDS.py > results_good.log 2>&1

# Run with poor network
python network_condition_manager.py apply --condition poor
python Server/Emotion_Recognition/FL_Server_DDS.py > results_poor.log 2>&1

# Reset
python network_condition_manager.py reset

# Compare results
diff results_good.log results_poor.log
```

### Example 3: Gradual Network Degradation

```bash
#!/bin/bash
for condition in excellent good moderate poor very_poor; do
    echo "Testing with $condition network..."
    python network_condition_manager.py apply --condition $condition
    python Server/Emotion_Recognition/FL_Server_MQTT.py
    sleep 2
done
python network_condition_manager.py reset
```

---

## ⚠️ Important Notes

1. **Requires sudo**: Network conditions need root privileges
2. **Install iproute2**: `sudo apt-get install iproute2`
3. **Persistent**: Conditions remain until reset
4. **Interface**: 
   - Local: `lo` (loopback)
   - Docker: `eth0` (container)
5. **Fair Testing**: Apply same conditions to all protocols

---

## 🔍 Troubleshooting

**"tc command not available"**
```bash
sudo apt-get install iproute2
```

**"Permission denied"**
```bash
# Ensure you can use sudo, or run with sudo:
sudo python network_condition_manager.py apply --condition poor
```

**"Conditions not working"**
```bash
# Verify they're applied:
sudo tc qdisc show dev lo

# Reset and try again:
python network_condition_manager.py reset
python network_condition_manager.py apply --condition poor
```

---

## ✅ Verification Checklist

- [ ] iproute2 installed (`tc` command available)
- [ ] Can run with sudo
- [ ] Environment variables set (if using env method)
- [ ] Conditions applied before experiment
- [ ] Conditions verified with `tc qdisc show`
- [ ] Conditions reset after experiment

---

**Quick Test:**
```bash
chmod +x test_network_conditions.sh
./test_network_conditions.sh
```
