# Network Condition Management for Local and Docker Execution

## Overview

The **Network Condition Manager** allows you to apply realistic network conditions (latency, jitter, bandwidth limits, packet loss) to your Federated Learning experiments, whether running **locally** or in **Docker containers**.

## Features

✅ **Dual Mode Support**: Works for both local and Docker execution  
✅ **Predefined Conditions**: excellent, good, moderate, poor, very_poor, satellite  
✅ **Custom Conditions**: Set specific latency, jitter, bandwidth, loss  
✅ **Environment Variable Control**: Apply conditions only when explicitly requested  
✅ **Auto-Reset**: Clean up network conditions after experiments  

---

## Installation Requirements

### For Local Execution

Install `iproute2` package (contains `tc` - traffic control):

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install iproute2

# Already installed in most Linux distributions
```

### For Docker Execution

Docker containers need `iproute2` and `NET_ADMIN` capability (already configured in your docker-compose files).

---

## Usage

### Method 1: Using Environment Variables (Recommended)

Set environment variables before running your FL experiments:

```bash
# Enable network conditions
export APPLY_NETWORK_CONDITION=true
export NETWORK_CONDITION=poor

# Run your FL experiment
python Server/Emotion_Recognition/FL_Server_MQTT.py
```

### Method 2: Using CLI Tool Directly

```bash
# Apply predefined condition
python network_condition_manager.py apply --condition poor

# Run your experiments (conditions are now active)
python Server/Emotion_Recognition/FL_Server_MQTT.py

# Reset conditions after experiments
python network_condition_manager.py reset
```

### Method 3: Apply Custom Conditions

```bash
# Apply custom conditions
python network_condition_manager.py apply --latency 100 --jitter 30 --bandwidth 2 --loss 3

# Or with environment variables
export APPLY_NETWORK_CONDITION=true
export NETWORK_LATENCY_MS=100
export NETWORK_JITTER_MS=30
export NETWORK_BANDWIDTH_MBPS=2
export NETWORK_LOSS_PERCENT=3
```

---

## Available Network Conditions

| Condition | Latency | Jitter | Bandwidth | Loss | Description |
|-----------|---------|--------|-----------|------|-------------|
| **excellent** | 2ms | 0.5ms | 1000 Mbps | 0.01% | LAN/Data Center |
| **good** | 10ms | 2ms | 100 Mbps | 0.1% | Broadband/Fiber |
| **moderate** | 50ms | 10ms | 20 Mbps | 1% | 4G/LTE |
| **poor** | 100ms | 30ms | 2 Mbps | 3% | 3G Mobile |
| **very_poor** | 300ms | 100ms | 384 Kbps | 5% | Edge/2G |
| **satellite** | 600ms | 50ms | 5 Mbps | 2% | Satellite Link |
| **none** | 0ms | 0ms | Unlimited | 0% | No conditions |

---

## Docker Usage

For Docker containers, the conditions are applied automatically if environment variables are set in `docker-compose.yml`:

```yaml
services:
  fl-server-mqtt-emotion:
    environment:
      - APPLY_NETWORK_CONDITION=true
      - NETWORK_CONDITION=poor
```

Or override when starting:

```bash
cd Docker
APPLY_NETWORK_CONDITION=true NETWORK_CONDITION=poor \
  docker-compose -f docker-compose-emotion.yml up
```

---

## Local Execution Examples

### Example 1: Run MQTT Server with Poor Network

```bash
# Terminal 1: MQTT Server
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL
export APPLY_NETWORK_CONDITION=true
export NETWORK_CONDITION=poor
python Server/Emotion_Recognition/FL_Server_MQTT.py
```

```bash
# Terminal 2: MQTT Client 1
export APPLY_NETWORK_CONDITION=true
export NETWORK_CONDITION=poor
export CLIENT_ID=1
python Client/Emotion_Recognition/FL_Client_MQTT.py
```

### Example 2: Run DDS with Very Poor Network

```bash
# Apply conditions first
python network_condition_manager.py apply --condition very_poor

# Run DDS server
python Server/Emotion_Recognition/FL_Server_DDS.py

# In another terminal, run client
python Client/Emotion_Recognition/FL_Client_DDS.py

# Reset after done
python network_condition_manager.py reset
```

### Example 3: Custom Network Conditions

```bash
# Apply custom conditions (100ms latency, 2Mbps bandwidth, 3% loss)
python network_condition_manager.py apply \
  --latency 100 \
  --jitter 30 \
  --bandwidth 2 \
  --loss 3

# Run experiments
python Server/Emotion_Recognition/FL_Server_gRPC.py

# Reset
python network_condition_manager.py reset
```

---

## Integration in Code

To integrate network condition manager in your FL code:

```python
from network_condition_manager import NetworkConditionManager

# At the start of your server/client
def main():
    # Apply network conditions from environment if requested
    network_manager = NetworkConditionManager.from_environment(verbose=True)
    
    if network_manager:
        print("Network conditions applied!")
    else:
        print("Running with normal network conditions")
    
    # Your FL code here...
    run_federated_learning()
    
    # Reset at the end (optional, will auto-reset on next apply)
    if network_manager:
        network_manager.reset_network_conditions()
```

---

## Verification

### Check if Conditions Are Applied

```bash
# View current tc rules
sudo tc qdisc show dev lo  # For local execution
sudo tc qdisc show dev eth0  # For Docker
```

Example output when conditions are applied:
```
qdisc netem 1: root refcnt 2 limit 1000 delay 100.0ms  30.0ms loss 3%
```

### List All Available Conditions

```bash
python network_condition_manager.py list
```

---

## Troubleshooting

### Issue: "tc command not available"

**Solution:**
```bash
sudo apt-get install iproute2
```

### Issue: "Permission denied"

**Solution:** The tool uses `sudo` internally. Ensure your user has sudo privileges.

Alternatively, run with sudo:
```bash
sudo python network_condition_manager.py apply --condition poor
```

### Issue: Conditions not working in Docker

**Solution:** Ensure containers have `NET_ADMIN` capability in docker-compose.yml:
```yaml
services:
  fl-server:
    cap_add:
      - NET_ADMIN
```

### Issue: Want to test without actually applying conditions

**Solution:** Simply don't set the environment variable:
```bash
# This will run normally without network conditions
python Server/Emotion_Recognition/FL_Server_MQTT.py
```

---

## Automation Script

Create a helper script to run experiments with different network conditions:

```bash
#!/bin/bash
# run_with_network_condition.sh

CONDITION=$1
PROTOCOL=$2

if [ -z "$CONDITION" ] || [ -z "$PROTOCOL" ]; then
    echo "Usage: $0 <condition> <protocol>"
    echo "Example: $0 poor mqtt"
    exit 1
fi

# Apply network condition
python network_condition_manager.py apply --condition $CONDITION

# Run experiment based on protocol
case $PROTOCOL in
    mqtt)
        python Server/Emotion_Recognition/FL_Server_MQTT.py
        ;;
    grpc)
        python Server/Emotion_Recognition/FL_Server_gRPC.py
        ;;
    dds)
        python Server/Emotion_Recognition/FL_Server_DDS.py
        ;;
    *)
        echo "Unknown protocol: $PROTOCOL"
        ;;
esac

# Reset network conditions
python network_condition_manager.py reset
```

Usage:
```bash
chmod +x run_with_network_condition.sh
./run_with_network_condition.sh poor mqtt
```

---

## Important Notes

1. **Sudo Required**: Network conditions require root privileges (uses `sudo tc`)
2. **Interface**: 
   - Local: Applies to `lo` (loopback)
   - Docker: Applies to `eth0` (container interface)
3. **Persistent**: Conditions remain until reset or system reboot
4. **Fair Comparison**: All protocols experience the same conditions
5. **Reset After**: Always reset conditions after experiments

---

## Quick Reference

```bash
# List conditions
python network_condition_manager.py list

# Apply condition
python network_condition_manager.py apply --condition poor

# Reset
python network_condition_manager.py reset

# With environment variables
export APPLY_NETWORK_CONDITION=true
export NETWORK_CONDITION=poor
python Server/Emotion_Recognition/FL_Server_MQTT.py

# Custom conditions
python network_condition_manager.py apply --latency 100 --bandwidth 2 --loss 3
```

---

**Status**: ✅ Ready for local and Docker execution!
