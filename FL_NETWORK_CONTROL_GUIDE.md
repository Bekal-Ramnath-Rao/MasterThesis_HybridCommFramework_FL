# FL Network Control & Monitoring Guide

## Overview

This guide covers the new FL Network Control & Monitoring system that allows you to:
1. **Select specific clients** to apply network conditions
2. **Control network parameters** (latency, jitter, packet loss, bandwidth)
3. **Make changes in parallel** during FL training
4. **Monitor the FL setup** in real-time

## Key Features

### 🎯 Host-Level Control
- Uses **veth interfaces** for reliable network control
- Changes apply from the host side (no container modification needed)
- Supports any running FL client container

### 🔧 Flexible Parameters
- **Latency**: Any value (e.g., 200ms, 300ms, 500ms)
- **Bandwidth**: Any value (e.g., 1mbit, 500kbit, 10mbit)
- **Packet Loss**: Any percentage (e.g., 2%, 5%, 10%)
- **Jitter**: Any value (e.g., 10ms, 30ms, 50ms)

### 📊 Real-Time Monitoring
- Live FL training dashboard
- Container resource monitoring (CPU, memory, network)
- Current network conditions display
- Log streaming

## Tools Overview

### 1. FL Network Monitor (`fl_network_monitor.py`)
Main tool for applying network conditions during training.

**Features:**
- Select specific clients by ID
- Apply custom network conditions
- Clear conditions
- Show current status
- Interactive mode

### 2. FL Training Dashboard (`fl_training_dashboard.py`)
Real-time monitoring dashboard for FL training.

**Features:**
- Live server/client status
- Resource usage monitoring
- Network condition display
- Recent activity logs

### 3. FL Network Control Script (`fl_network_control.sh`)
Quick access wrapper script for common operations.

## Quick Start

### Prerequisites

```bash
# Make scripts executable
cd Network_Simulation/
chmod +x fl_network_control.sh
chmod +x fl_network_monitor.py
chmod +x fl_training_dashboard.py

# Ensure sudo access for tc commands
sudo echo "Testing sudo access..."
```

### Step 1: Start FL Training

```bash
# Start your FL training as usual
cd ..
docker-compose up -d

# Wait for containers to be ready
sleep 5
```

### Step 2: Start Monitoring Dashboard

```bash
# In Terminal 1 - Start the monitoring dashboard
cd Network_Simulation/
python3 fl_training_dashboard.py

# Or with custom refresh rate
python3 fl_training_dashboard.py --interval 3
```

### Step 3: Apply Network Conditions

```bash
# In Terminal 2 - Control network conditions

# Option A: Quick commands
./fl_network_control.sh status              # Check current conditions
./fl_network_control.sh list-clients        # List all clients
./fl_network_control.sh quick-change 1      # Quick change for client 1

# Option B: Direct commands
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 2
python3 fl_network_monitor.py --client-id 2 --bandwidth 1mbit
python3 fl_network_monitor.py --all --jitter 30ms

# Option C: Interactive mode
python3 fl_network_monitor.py --monitor
```

## Usage Examples

### Example 1: Simulate Poor Network on Client 1

```bash
# Apply poor network conditions
python3 fl_network_monitor.py \
    --client-id 1 \
    --latency 200ms \
    --bandwidth 1mbit \
    --loss 3 \
    --jitter 50ms

# Verify conditions
python3 fl_network_monitor.py --show-status
```

### Example 2: Change Single Parameter During Training

```bash
# Increase latency to client 2 during round 5
python3 fl_network_monitor.py --client-id 2 --latency 300ms

# Later, reduce bandwidth
python3 fl_network_monitor.py --client-id 2 --bandwidth 500kbit
```

### Example 3: Apply to All Clients

```bash
# Apply moderate conditions to all clients
python3 fl_network_monitor.py \
    --all \
    --latency 100ms \
    --bandwidth 5mbit \
    --loss 1
```

### Example 4: Clear Conditions

```bash
# Clear from specific client
python3 fl_network_monitor.py --client-id 1 --clear

# Clear from all clients
python3 fl_network_monitor.py --all --clear
```

## Interactive Mode

The interactive mode provides a menu-driven interface for network control:

```bash
python3 fl_network_monitor.py --monitor
```

**Interactive Menu:**
```
1. Apply network conditions to specific client
2. Apply conditions to all clients
3. Change single parameter for specific client
4. Show current network conditions
5. Clear conditions from specific client
6. Clear all conditions
7. Start FL training monitor
8. Refresh container/veth mapping
9. Test veth interface connectivity
0. Exit
```

## Technical Details

### How It Works

1. **Container Detection**: Automatically finds all running FL client containers
2. **Veth Mapping**: Maps each container to its host veth interface
   ```bash
   container_id=$(docker ps -qf name=<container_name>)
   container_ifindex=$(docker exec $container_id cat /sys/class/net/eth0/iflink)
   veth=$(ip link | grep "^$container_ifindex:" | awk -F': ' '{print $2}')
   ```
3. **TC Application**: Applies traffic control rules to the veth interface
   ```bash
   sudo tc qdisc add dev <veth> root netem delay 200ms loss 2%
   ```

### Network Condition Architecture

```
┌─────────────────┐
│ FL Client 1     │
│  eth0           │
└────────┬────────┘
         │
    ┌────▼─────┐
    │ veth-abc │  ◄── Host applies tc rules here
    └──────────┘
         │
    ┌────▼─────┐
    │ docker0  │
    │ bridge   │
    └──────────┘
```

### Command Examples with TC

```bash
# Latency only
sudo tc qdisc add dev veth123 root netem delay 200ms

# Latency + Jitter
sudo tc qdisc add dev veth123 root netem delay 200ms 50ms

# Latency + Loss
sudo tc qdisc add dev veth123 root netem delay 200ms loss 2%

# Bandwidth limit
sudo tc qdisc add dev veth123 root tbf rate 1mbit burst 32kbit latency 400ms

# Bandwidth + Latency + Loss
sudo tc qdisc add dev veth123 root handle 1: tbf rate 1mbit burst 32kbit latency 400ms
sudo tc qdisc add dev veth123 parent 1:1 handle 10: netem delay 200ms loss 2%

# Clear rules
sudo tc qdisc del dev veth123 root
```

## Workflow Scenarios

### Scenario 1: Dynamic Network During Training

```bash
# Terminal 1: Start FL training
docker-compose up

# Terminal 2: Monitor training
python3 fl_training_dashboard.py

# Terminal 3: Apply conditions dynamically
# Round 1-5: Good network
python3 fl_network_monitor.py --all --latency 20ms --bandwidth 50mbit

# Round 6-10: Degrade client 1
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 3

# Round 11-15: Recover client 1, degrade client 2
python3 fl_network_monitor.py --client-id 1 --clear
python3 fl_network_monitor.py --client-id 2 --bandwidth 1mbit

# After training: Clear all
python3 fl_network_monitor.py --all --clear
```

### Scenario 2: Simulate Gradual Network Degradation

```bash
# Script to gradually increase latency
for latency in 50 100 150 200 250 300; do
    echo "Setting latency to ${latency}ms..."
    python3 fl_network_monitor.py --client-id 1 --latency ${latency}ms
    sleep 60  # Wait 1 minute between changes
done
```

### Scenario 3: Protocol Comparison Under Varying Conditions

```bash
# Start experiments with different conditions per client
# Client 1: Excellent
python3 fl_network_monitor.py --client-id 1 --latency 10ms --bandwidth 100mbit

# Client 2: Moderate
python3 fl_network_monitor.py --client-id 2 --latency 50ms --bandwidth 20mbit

# Client 3: Poor
python3 fl_network_monitor.py --client-id 3 --latency 200ms --bandwidth 2mbit

# Monitor and compare protocol performance
python3 fl_training_dashboard.py
```

## Monitoring Dashboard

The dashboard displays:

### Server Status
- Container name
- CPU usage
- Memory usage
- Network I/O
- Status

### Client Status
- Container name
- CPU usage
- Memory usage
- Network I/O
- Current network conditions

### Recent Activity
- Last log entries from containers
- Training progress updates
- Error messages

### Dashboard Example Output

```
================================================================================
                          FL TRAINING DASHBOARD
                    Last Update: 2026-01-28 14:30:45
================================================================================

📊 SERVER STATUS
--------------------------------------------------------------------------------
Container                                CPU        Memory              Network I/O               Status
--------------------------------------------------------------------------------
fl-server-temperature-mqtt               2.5%       450MB / 2GB         125MB / 89MB              Running
--------------------------------------------------------------------------------

👥 CLIENT STATUS
--------------------------------------------------------------------------------
Container                           CPU      Memory            Net I/O              Network Conditions
--------------------------------------------------------------------------------
fl-client-temp-mqtt-1               5.2%     380MB / 1GB       45MB / 32MB          L:200ms BW:1Mbit Loss:2%
fl-client-temp-mqtt-2               4.8%     390MB / 1GB       48MB / 35MB          None
--------------------------------------------------------------------------------

📝 RECENT ACTIVITY (Last 2 lines per container)
--------------------------------------------------------------------------------

fl-server-temperature-mqtt:
  [Round 5] Aggregating models from 2 clients...
  [Round 5] Global model updated, accuracy: 87.5%

fl-client-temp-mqtt-1:
  Training on local data... loss: 0.245
  Uploading model to server...
```

## Troubleshooting

### Issue: "No veth found for container"

**Solution:**
```bash
# Refresh container/veth mapping
python3 fl_network_monitor.py --monitor
# Then select option 8: Refresh container/veth mapping

# Or manually check
docker ps -qf name=<container> | xargs -I {} docker exec {} cat /sys/class/net/eth0/iflink
ip link | grep "^<ifindex>:"
```

### Issue: "Permission denied" when applying tc

**Solution:**
```bash
# Ensure you have sudo access
sudo echo "Testing..."

# Or run with sudo
sudo python3 fl_network_monitor.py --client-id 1 --latency 200ms
```

### Issue: "RTNETLINK answers: File exists"

**Solution:**
```bash
# Clear existing rules first
python3 fl_network_monitor.py --client-id 1 --clear

# Then apply new rules
python3 fl_network_monitor.py --client-id 1 --latency 200ms
```

### Issue: Container not found

**Solution:**
```bash
# List all running containers
./fl_network_control.sh list-clients

# Or manually check
docker ps --filter "name=client"
```

## Advanced Usage

### Scripted Network Changes

Create a script to apply dynamic conditions:

```bash
#!/bin/bash
# dynamic_network_scenario.sh

# Round 1-3: Good network
python3 fl_network_monitor.py --all --latency 20ms --bandwidth 50mbit
sleep 180  # 3 minutes

# Round 4-6: Moderate degradation
python3 fl_network_monitor.py --all --latency 100ms --bandwidth 10mbit --loss 1
sleep 180

# Round 7-9: Poor network
python3 fl_network_monitor.py --all --latency 200ms --bandwidth 2mbit --loss 3
sleep 180

# Round 10: Recovery
python3 fl_network_monitor.py --all --clear
```

### Python Integration

```python
import subprocess
import time

def apply_network_condition(client_id, latency=None, bandwidth=None, loss=None):
    """Apply network condition to specific client"""
    cmd = ['python3', 'fl_network_monitor.py', '--client-id', str(client_id)]
    
    if latency:
        cmd.extend(['--latency', latency])
    if bandwidth:
        cmd.extend(['--bandwidth', bandwidth])
    if loss:
        cmd.extend(['--loss', str(loss)])
    
    subprocess.run(cmd)

# Example usage
apply_network_condition(1, latency='200ms', loss=2)
time.sleep(60)
apply_network_condition(1, bandwidth='1mbit')
```

## Best Practices

1. **Start monitoring before applying conditions**: Launch dashboard first to observe effects

2. **Apply changes gradually**: Don't change all parameters at once

3. **Clear conditions after experiments**: Reset to baseline for fair comparison

4. **Document conditions**: Keep track of what conditions were applied when

5. **Use meaningful client IDs**: Know which client corresponds to which protocol/use case

6. **Monitor resource usage**: Ensure network conditions don't cause container crashes

7. **Test veth connectivity**: Periodically verify veth interfaces are correctly mapped

## Summary

The FL Network Control & Monitoring system provides:
- ✅ **Client selection**: Target specific clients by ID
- ✅ **Flexible parameters**: Latency, bandwidth, loss, jitter
- ✅ **Parallel changes**: Modify conditions during training
- ✅ **Real-time monitoring**: Dashboard with live updates
- ✅ **Host-level control**: Reliable veth interface approach
- ✅ **Easy to use**: Interactive mode and quick commands

Use `./fl_network_control.sh` for quick access or `python3 fl_network_monitor.py --monitor` for full interactive control.
