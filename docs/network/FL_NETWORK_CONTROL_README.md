# FL Network Control & Monitoring System

## 🎯 Overview

A comprehensive system for **real-time network condition control** and **FL training monitoring** that allows you to:

1. **Select specific clients** (by ID, protocol, or all)
2. **Control network parameters** (packet loss, jitter, latency, bandwidth)
3. **Apply changes in parallel** during FL training (no restart needed)
4. **Monitor the FL setup** with a real-time dashboard

## ✨ Key Features

- ✅ **Host-level control** using veth interfaces (reliable and accurate)
- ✅ **Flexible parameters** - any custom values, not limited to presets
- ✅ **Dynamic changes** - modify conditions while training is running
- ✅ **Real-time monitoring** - live dashboard with container stats
- ✅ **Easy to use** - interactive mode and quick commands
- ✅ **Well documented** - comprehensive guides and quick reference

## 🚀 Quick Start

### 1. Verify Setup
```bash
cd Network_Simulation/
python3 test_fl_network_system.py
```
Expected: All 10 tests should pass ✓

### 2. Start FL Training
```bash
cd ..
docker-compose up -d
```

### 3. Start Monitoring (Terminal 1)
```bash
cd Network_Simulation/
python3 fl_training_dashboard.py
```

### 4. Control Network (Terminal 2)
```bash
# Interactive mode (recommended)
python3 fl_network_monitor.py --monitor

# Or direct commands
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 2
```

## 📁 Files Structure

### Tools (Network_Simulation/)
```
fl_network_monitor.py          Main network control tool (24KB)
fl_training_dashboard.py       Real-time monitoring dashboard (9.5KB)
fl_network_control.sh          Quick command wrapper (6.8KB)
test_fl_network_system.py      System verification (5.9KB)
demo_fl_network_control.sh     Interactive demo (3.3KB)
```

### Documentation
```
FL_NETWORK_CONTROL_GUIDE.md              Full guide (13KB)
FL_NETWORK_CONTROL_QUICK_REF.md          Quick reference (5.6KB)
FL_NETWORK_CONTROL_IMPLEMENTATION.md     Implementation details (10KB)
```

## 📖 Documentation

| Document | Purpose | When to Use |
|----------|---------|-------------|
| [FL_NETWORK_CONTROL_GUIDE.md](FL_NETWORK_CONTROL_GUIDE.md) | Comprehensive usage guide with examples | Learning the system |
| [FL_NETWORK_CONTROL_QUICK_REF.md](FL_NETWORK_CONTROL_QUICK_REF.md) | Quick command reference | Daily usage |
| [FL_NETWORK_CONTROL_IMPLEMENTATION.md](FL_NETWORK_CONTROL_IMPLEMENTATION.md) | Technical implementation details | Understanding internals |

## 🎬 Usage Examples

### Example 1: Apply Poor Network to Client 1
```bash
python3 fl_network_monitor.py \
    --client-id 1 \
    --latency 200ms \
    --bandwidth 1mbit \
    --loss 3 \
    --jitter 50ms

# Verify
python3 fl_network_monitor.py --show-status
```

### Example 2: Dynamic Changes During Training
```bash
# Round 1-5: Good network
python3 fl_network_monitor.py --all --latency 20ms --bandwidth 50mbit

# Round 6-10: Degrade client 1
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 3

# Round 11+: Clear all
python3 fl_network_monitor.py --all --clear
```

### Example 3: Interactive Mode
```bash
python3 fl_network_monitor.py --monitor

# Interactive menu:
# 1. Apply to specific client
# 2. Apply to all clients
# 3. Change single parameter
# 4. Show status
# 5. Clear specific client
# 6. Clear all
# 7. Start FL monitor
```

### Example 4: Using Helper Script
```bash
cd Network_Simulation/

# List clients
./fl_network_control.sh list-clients

# Show status
./fl_network_control.sh status

# Quick change
./fl_network_control.sh quick-change 1

# Start monitoring
./fl_network_control.sh monitor
```

## 🛠️ Technical Details

### How It Works

The system uses **host-level traffic control** on veth interfaces:

```bash
# 1. Find container
container_id=$(docker ps -qf name=<client>)

# 2. Get veth interface
container_ifindex=$(docker exec $container_id cat /sys/class/net/eth0/iflink)
veth=$(ip link | grep "^$container_ifindex:" | awk '{print $2}')

# 3. Apply TC rules
sudo tc qdisc add dev $veth root netem delay 200ms
sudo tc qdisc change dev $veth root netem delay 300ms
```

### Architecture
```
┌─────────────────┐
│ FL Client       │
│  Container      │
│  eth0           │
└────────┬────────┘
         │
    ┌────▼─────┐
    │ veth-xyz │  ◄── TC rules applied here (host level)
    └──────────┘
         │
    ┌────▼─────┐
    │ docker0  │
    │ bridge   │
    └──────────┘
```

## 📊 Dashboard Example

```
================================================================================
                          FL TRAINING DASHBOARD
                    Last Update: 2026-01-28 14:30:45
================================================================================

📊 SERVER STATUS
Container                                CPU        Memory              Network I/O
fl-server-temperature-mqtt               2.5%       450MB / 2GB         125MB / 89MB

👥 CLIENT STATUS
Container                           CPU      Memory            Network Conditions
fl-client-temp-mqtt-1               5.2%     380MB / 1GB       L:200ms BW:1Mbit Loss:2%
fl-client-temp-mqtt-2               4.8%     390MB / 1GB       None

📝 RECENT ACTIVITY
fl-server-temperature-mqtt:
  [Round 5] Aggregating models from 2 clients...
  [Round 5] Global model updated, accuracy: 87.5%
```

## 🎓 Demo

Run the interactive demo to see all features:

```bash
cd Network_Simulation/
./demo_fl_network_control.sh
```

This will guide you through:
1. System verification
2. Client listing
3. Status checking
4. Applying conditions
5. Changing parameters
6. Clearing conditions

## 🔍 Common Operations

### Check Status
```bash
python3 fl_network_monitor.py --show-status
```

### Apply to Specific Client
```bash
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 2
```

### Apply to All Clients
```bash
python3 fl_network_monitor.py --all --bandwidth 1mbit
```

### Change Single Parameter
```bash
python3 fl_network_monitor.py --client-id 2 --latency 300ms
```

### Clear Conditions
```bash
python3 fl_network_monitor.py --client-id 1 --clear
python3 fl_network_monitor.py --all --clear
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No veth found | Refresh: Option 8 in interactive mode |
| Permission denied | Use `sudo` or check sudo access |
| File exists error | Clear first: `--clear` then apply |
| Container not found | Check: `docker ps --filter "name=client"` |

## 💡 Best Practices

1. **Start monitoring before applying conditions** - See effects in real-time
2. **Use `--show-status` to verify** - Confirm conditions were applied
3. **Clear conditions after experiments** - Reset to baseline
4. **Document what you change** - Keep track for research
5. **Use interactive mode for exploration** - Easier for testing
6. **Use direct commands for automation** - Better for scripts

## 📋 Typical Workflow

```bash
# Terminal 1: Start monitoring
cd Network_Simulation/
python3 fl_training_dashboard.py

# Terminal 2: Control network conditions
cd Network_Simulation/

# Check initial status
python3 fl_network_monitor.py --show-status

# Apply conditions (e.g., poor network to client 1)
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 3

# Monitor effects in Terminal 1
# Wait for a few FL rounds...

# Change conditions dynamically
python3 fl_network_monitor.py --client-id 1 --bandwidth 500kbit

# After training, clear all
python3 fl_network_monitor.py --all --clear
```

## 🎯 Features Comparison

| Feature | Old System | New System |
|---------|-----------|------------|
| Client selection | All clients only | Specific or all |
| Parameter control | Preset scenarios | Any custom values |
| Control location | Inside containers | Host veth (reliable) |
| Monitoring | Limited | Real-time dashboard |
| Ease of use | CLI only | Interactive + CLI |
| Documentation | Basic | Comprehensive |

## 📦 Requirements

- ✅ Docker
- ✅ Python 3
- ✅ sudo access (for tc commands)
- ✅ iproute2 (tc utility)

All checked by `test_fl_network_system.py`

## 🎉 Summary

The FL Network Control System provides everything you need to:

- **Control network conditions** for specific clients or all clients
- **Monitor FL training** in real-time with a live dashboard
- **Apply changes dynamically** while training is running
- **Use host-level control** via veth interfaces (reliable and accurate)
- **Access comprehensive documentation** and quick references

**Ready to use!** Just run the verification test and start experimenting.

---

For more information:
- Read: [FL_NETWORK_CONTROL_GUIDE.md](FL_NETWORK_CONTROL_GUIDE.md)
- Quick ref: [FL_NETWORK_CONTROL_QUICK_REF.md](FL_NETWORK_CONTROL_QUICK_REF.md)
- Implementation: [FL_NETWORK_CONTROL_IMPLEMENTATION.md](FL_NETWORK_CONTROL_IMPLEMENTATION.md)
