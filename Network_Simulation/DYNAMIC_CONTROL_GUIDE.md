# Dynamic Network Control Guide

## Overview
The Dynamic Network Controller allows you to change network parameters **in real-time** while FL training is running. You can set **ANY custom values** - not just predefined scenarios.

## Key Features
- ✅ Set **ANY** custom values (e.g., 75ms latency, 8.5mbit bandwidth, 1.5% loss)
- ✅ Change parameters **dynamically** during FL training
- ✅ Modify **single parameters** without affecting others
- ✅ View **current status** of all network parameters
- ✅ Target specific protocols or containers
- ✅ Interactive mode for easy control

## Quick Start

### 1. Interactive Mode (Recommended)
```bash
python dynamic_network_controller.py --interactive
```

**Interactive menu options:**
- Apply preset scenarios
- Set ANY custom values
- Modify single parameters
- View current status
- Clear rules

### 2. Command Line Examples

#### Set Custom Values (All Clients)
```bash
# Any custom values you want
python dynamic_network_controller.py --latency 75ms --bandwidth 8.5mbit --loss 1.5 --jitter 12.3ms

# Another example
python dynamic_network_controller.py --latency 125.5ms --bandwidth 3.2mbit --loss 0.75 --jitter 18ms
```

#### Change Single Parameter
```bash
# Only change latency (keeps bandwidth, loss, jitter unchanged)
python dynamic_network_controller.py --latency 95ms

# Only change bandwidth
python dynamic_network_controller.py --bandwidth 6.8mbit

# Only change packet loss
python dynamic_network_controller.py --loss 2.3
```

#### Target Specific Protocol
```bash
# Change MQTT clients only
python dynamic_network_controller.py --protocol mqtt --latency 80ms --bandwidth 7.5mbit

# Change gRPC clients only
python dynamic_network_controller.py --protocol grpc --loss 1.8 --jitter 15ms
```

#### View Current Status
```bash
# Show current network parameters for all clients
python dynamic_network_controller.py --show-status
```

Output:
```
====================================================================================================
Container                      Latency         Bandwidth       Loss       Jitter         
====================================================================================================
client-temp-1                  75.0ms          8.5Mbit         1.5        12.3ms         
client-temp-2                  75.0ms          8.5Mbit         1.5        12.3ms         
====================================================================================================
```

## Supported Values

### Latency
- **Format:** `<number>ms`
- **Examples:** `5ms`, `75ms`, `125.5ms`, `200ms`
- **Range:** Any positive value (typically 0-500ms)

### Bandwidth
- **Format:** `<number>mbit` or `<number>kbit`
- **Examples:** `100mbit`, `8.5mbit`, `500kbit`, `1.2mbit`
- **Range:** Any positive value (typically 0.5mbit - 100mbit)

### Packet Loss
- **Format:** `<number>` (percentage)
- **Examples:** `0.1`, `1.5`, `2.3`, `5`, `7.8`
- **Range:** 0-100 (typically 0-10%)

### Jitter
- **Format:** `<number>ms`
- **Examples:** `1ms`, `12.3ms`, `15.5ms`, `50ms`
- **Range:** Any positive value (typically 0-100ms)

## Workflow Example: Dynamic FL Evaluation

### Step 1: Start FL Training
```bash
# Start all protocol servers/clients
docker-compose -f docker-compose-temperature.yml up
```

### Step 2: Start with Good Conditions
```bash
# In another terminal, set initial good conditions
python dynamic_network_controller.py --latency 10ms --bandwidth 50mbit --loss 0.2 --jitter 2ms
```

### Step 3: Change During Training
```bash
# After a few rounds, degrade to moderate
python dynamic_network_controller.py --latency 60ms --bandwidth 12mbit --loss 1.2 --jitter 8ms

# After more rounds, degrade further
python dynamic_network_controller.py --latency 110ms --bandwidth 4.5mbit --loss 2.8 --jitter 18ms

# Then improve again
python dynamic_network_controller.py --latency 35ms --bandwidth 25mbit --loss 0.6 --jitter 5ms
```

### Step 4: Monitor Status
```bash
# Check current parameters anytime
python dynamic_network_controller.py --show-status
```

### Step 5: Test Specific Scenario
```bash
# Simulate WiFi congestion
python dynamic_network_controller.py --latency 85ms --bandwidth 7.2mbit --jitter 22ms

# Simulate poor 4G connection
python dynamic_network_controller.py --latency 150ms --bandwidth 2.8mbit --loss 3.5 --jitter 35ms

# Simulate satellite link
python dynamic_network_controller.py --latency 600ms --bandwidth 5mbit --loss 0.5 --jitter 100ms
```

## Interactive Mode Workflow

### Using Option 3: Set ANY Custom Values
```
Select option: 3

--- Set ANY Custom Values ---
Examples: latency=75ms, bandwidth=8.5mbit, loss=1.5, jitter=12.3ms
Leave blank to skip a parameter

  Latency (e.g., 75ms, 125.5ms): 85ms
  Bandwidth (e.g., 8.5mbit, 500kbit): 6.2mbit
  Packet loss % (e.g., 1.5, 0.3): 1.8
  Jitter (e.g., 12.5ms, 8ms): 14.5ms

✓ client-temp-1: Custom parameters applied - {'latency': '85ms', 'bandwidth': '6.2mbit', 'loss': '1.8', 'jitter': '14.5ms'}
✓ client-temp-2: Custom parameters applied - {'latency': '85ms', 'bandwidth': '6.2mbit', 'loss': '1.8', 'jitter': '14.5ms'}
```

### Using Option 7: Modify Single Parameter
```
Select option: 7

--- Modify Single Parameter ---
This will change ONLY the selected parameter, keeping others unchanged

Select parameter to modify:
  1. Latency
  2. Bandwidth
  3. Packet Loss
  4. Jitter

Parameter: 2
Enter new bandwidth (e.g., 8.5mbit, 500kbit): 4.3mbit

✓ client-temp-1: Custom parameters applied - {'bandwidth': '4.3mbit'}
✓ client-temp-2: Custom parameters applied - {'bandwidth': '4.3mbit'}
```

### Using Option 5: Show Current Status
```
Select option: 5

====================================================================================================
Container                      Latency         Bandwidth       Loss       Jitter         
====================================================================================================
client-temp-1                  85.0ms          4.3Mbit         1.8        14.5ms         
client-temp-2                  85.0ms          4.3Mbit         1.8        14.5ms         
====================================================================================================
```

## Advanced Usage

### Protocol-Specific Testing
```bash
# Test MQTT under different conditions
python dynamic_network_controller.py --protocol mqtt --latency 45ms --bandwidth 15mbit

# Test gRPC with high latency
python dynamic_network_controller.py --protocol grpc --latency 180ms --loss 2.1

# Test QUIC with packet loss
python dynamic_network_controller.py --protocol quic --loss 3.5 --jitter 25ms
```

### Gradual Degradation Script
Create a bash script for gradual changes:
```bash
#!/bin/bash
# gradual_test.sh - Gradually degrade network conditions

echo "Starting with excellent conditions..."
python dynamic_network_controller.py --latency 5ms --bandwidth 100mbit --loss 0.1 --jitter 1ms
sleep 30

echo "Degrading to good..."
python dynamic_network_controller.py --latency 25ms --bandwidth 40mbit --loss 0.6 --jitter 6ms
sleep 30

echo "Degrading to moderate..."
python dynamic_network_controller.py --latency 65ms --bandwidth 15mbit --loss 1.3 --jitter 11ms
sleep 30

echo "Degrading to poor..."
python dynamic_network_controller.py --latency 120ms --bandwidth 6mbit --loss 2.5 --jitter 22ms
sleep 30

echo "Improving back to good..."
python dynamic_network_controller.py --latency 30ms --bandwidth 35mbit --loss 0.7 --jitter 7ms
```

### Real-World Scenarios

#### WiFi Congestion
```bash
python dynamic_network_controller.py --latency 70ms --bandwidth 8.5mbit --loss 1.2 --jitter 18ms
```

#### Mobile 4G (Good)
```bash
python dynamic_network_controller.py --latency 45ms --bandwidth 20mbit --loss 0.5 --jitter 8ms
```

#### Mobile 4G (Poor)
```bash
python dynamic_network_controller.py --latency 150ms --bandwidth 3.5mbit --loss 2.8 --jitter 35ms
```

#### 3G Network
```bash
python dynamic_network_controller.py --latency 200ms --bandwidth 2mbit --loss 4 --jitter 50ms
```

#### Satellite Link
```bash
python dynamic_network_controller.py --latency 600ms --bandwidth 5mbit --loss 0.3 --jitter 80ms
```

#### IoT Low Power
```bash
python dynamic_network_controller.py --latency 100ms --bandwidth 500kbit --loss 3 --jitter 30ms
```

## Tips

1. **Start FL training first**, then use controller to change parameters
2. **Use --show-status** frequently to verify current settings
3. **Interactive mode** is best for exploring different scenarios
4. **Command line** is best for scripted/automated testing
5. You can change parameters **as frequently as you want** - even every few seconds
6. **Single parameter modification** preserves other settings
7. Use **decimal values** for fine-grained control (e.g., 1.5%, 8.5mbit)

## Troubleshooting

### "No client containers found"
- Make sure Docker containers are running: `docker ps`
- Containers must have 'client' in their name

### "Error applying network params"
- Containers need NET_ADMIN capability
- Check with: `docker inspect <container> | grep -i cap_add`

### Changes not taking effect
- Use `--show-status` to verify current settings
- Check with `--show-rules` for detailed tc output
- Clear rules first with `--clear`, then reapply

## See Also
- [README_NETWORK_SIMULATION.md](../README_NETWORK_SIMULATION.md) - Overall network simulation setup
- [README_DYNAMIC_NETWORK.md](README_DYNAMIC_NETWORK.md) - Detailed architecture
- [network_simulator.py](network_simulator.py) - Automated scenario testing
