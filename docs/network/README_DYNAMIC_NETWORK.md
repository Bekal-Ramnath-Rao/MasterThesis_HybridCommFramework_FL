# Dynamic Network Parameter Controller

## Overview
This script allows you to dynamically change network parameters (latency, bandwidth, packet loss, jitter) for FL client containers **in real-time** while training is running.

## Features
- ✅ Change network conditions while FL training is active
- ✅ Target specific protocols (MQTT, AMQP, gRPC, QUIC, DDS)
- ✅ Apply predefined scenarios or custom parameters
- ✅ Interactive mode for easy management
- ✅ Only affects client containers, not servers
- ✅ View and clear network rules

## Prerequisites
- Docker containers must be running with `NET_ADMIN` capability
- Python 3.6+
- Running FL system

## Usage

### Interactive Mode (Recommended)
```bash
python dynamic_network_controller.py --interactive
```

This opens a menu where you can:
1. Apply scenarios to all clients
2. Apply scenarios to specific protocols
3. Set custom parameters
4. View current network rules
5. Clear network rules

### Command-Line Examples

#### Apply Predefined Scenario
```bash
# Apply 'poor' network conditions to all clients
python dynamic_network_controller.py --scenario poor

# Apply 'moderate' to only MQTT clients
python dynamic_network_controller.py --protocol mqtt --scenario moderate

# Apply 'excellent' to temperature use case clients
python dynamic_network_controller.py --usecase temperature --scenario excellent
```

#### Apply Custom Parameters
```bash
# Set specific latency and packet loss for all clients
python dynamic_network_controller.py --latency 150ms --loss 3

# Set bandwidth and jitter for AMQP clients
python dynamic_network_controller.py --protocol amqp --bandwidth 2mbit --jitter 25ms

# Apply to specific containers
python dynamic_network_controller.py --clients client-temp-1 client-temp-2 --latency 200ms
```

#### View and Manage Rules
```bash
# Show current network rules for all clients
python dynamic_network_controller.py --show-rules

# Clear all network rules from all clients
python dynamic_network_controller.py --clear
```

## Network Scenarios

| Scenario | Latency | Bandwidth | Loss | Jitter |
|----------|---------|-----------|------|--------|
| **excellent** | 5ms | 100mbit | 0.1% | 1ms |
| **good** | 20ms | 50mbit | 0.5% | 5ms |
| **moderate** | 50ms | 10mbit | 1% | 10ms |
| **poor** | 100ms | 5mbit | 2% | 20ms |
| **very_poor** | 200ms | 1mbit | 5% | 50ms |

## Evaluation Workflow

### 1. Start FL Training
```bash
# Start with excellent conditions
python run_network_experiments.py --use-case temperature --protocols mqtt amqp grpc quic dds --scenarios excellent --rounds 1000
```

### 2. Monitor Training
Check server logs or metrics to see current round progress.

### 3. Change Network Conditions Dynamically
```bash
# While training is running, degrade network to 'poor'
python dynamic_network_controller.py --scenario poor

# Or apply custom degradation
python dynamic_network_controller.py --latency 150ms --loss 4 --bandwidth 3mbit
```

### 4. Observe Impact
- Watch how different protocols handle network changes
- Monitor convergence time changes
- Observe packet retransmission and error handling

### 5. Restore or Change Again
```bash
# Improve conditions back to 'good'
python dynamic_network_controller.py --scenario good

# Or clear all limitations
python dynamic_network_controller.py --clear
```

## Advanced Usage

### Target Specific Protocols During Training
```bash
# Degrade only MQTT clients
python dynamic_network_controller.py --protocol mqtt --scenario very_poor

# Keep other protocols at good conditions
python dynamic_network_controller.py --protocol amqp --scenario good
python dynamic_network_controller.py --protocol grpc --scenario good
```

### Simulate Network Fluctuations
```bash
# Create a script to periodically change conditions
while true; do
    python dynamic_network_controller.py --scenario good
    sleep 60
    python dynamic_network_controller.py --scenario poor
    sleep 60
done
```

### Progressive Degradation Test
```bash
# Start excellent
python dynamic_network_controller.py --scenario excellent
sleep 120

# Move to good
python dynamic_network_controller.py --scenario good
sleep 120

# Move to moderate
python dynamic_network_controller.py --scenario moderate
sleep 120

# Move to poor
python dynamic_network_controller.py --scenario poor
```

## How It Works

The script uses Linux `tc` (traffic control) to apply network shaping rules inside Docker containers:

1. **Traffic Shaping (tbf)**: Controls bandwidth
2. **Network Emulation (netem)**: Adds latency, jitter, and packet loss

These rules are applied to the `eth0` interface inside each client container without affecting the server.

## Troubleshooting

### "No client containers found"
- Ensure FL containers are running: `docker ps`
- Check container names contain "client"

### "Permission denied" or tc errors
- Ensure containers have `NET_ADMIN` capability
- Check if `add_net_admin_capability.py` was run before starting containers

### Rules not applying
- Verify containers are running: `docker ps | grep client`
- Check container logs for network-related errors
- Try clearing rules first: `python dynamic_network_controller.py --clear`

## Notes

- Network parameters only affect **egress** traffic (outgoing from clients)
- Server containers are automatically excluded
- Changes take effect immediately
- Rules persist until container restart or explicit clearing
- Safe to run while training is active

## Example Evaluation Session

```bash
# Terminal 1: Start FL training
python run_network_experiments.py --use-case temperature --protocols mqtt amqp --scenarios excellent --rounds 500

# Terminal 2: Wait for training to start, then change conditions
sleep 30
python dynamic_network_controller.py --interactive

# In interactive mode:
# - Option 1: Apply scenario to all clients → Select "poor"
# - Wait 60 seconds
# - Option 1: Apply scenario to all clients → Select "excellent"
# - Wait 60 seconds
# - Option 2: Apply to MQTT only → Select "very_poor"
```

This allows you to see how each protocol handles dynamic network changes during active training!
