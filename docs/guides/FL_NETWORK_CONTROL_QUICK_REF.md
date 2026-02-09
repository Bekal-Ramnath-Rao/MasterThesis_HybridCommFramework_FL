# FL Network Control - Quick Reference

## 🚀 Quick Start

```bash
# Make scripts executable
chmod +x Network_Simulation/fl_network_control.sh
chmod +x Network_Simulation/fl_network_monitor.py
chmod +x Network_Simulation/fl_training_dashboard.py
```

## 📊 Common Commands

### Start Monitoring
```bash
# Terminal 1: Dashboard
cd Network_Simulation/
python3 fl_training_dashboard.py

# Terminal 2: Network Control
python3 fl_network_monitor.py --monitor
```

### Quick Network Changes
```bash
# Apply to specific client
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 2

# Apply to all clients
python3 fl_network_monitor.py --all --bandwidth 1mbit

# Change single parameter
python3 fl_network_monitor.py --client-id 2 --latency 300ms

# Show current status
python3 fl_network_monitor.py --show-status

# Clear conditions
python3 fl_network_monitor.py --client-id 1 --clear
python3 fl_network_monitor.py --all --clear
```

### Using Helper Script
```bash
cd Network_Simulation/

# Show status
./fl_network_control.sh status

# List clients
./fl_network_control.sh list-clients

# Quick change
./fl_network_control.sh quick-change 1

# Apply to client
./fl_network_control.sh apply-client 1

# Apply to all
./fl_network_control.sh apply-all

# Clear conditions
./fl_network_control.sh clear-client 1
./fl_network_control.sh clear-all

# Start monitoring
./fl_network_control.sh monitor

# Interactive control
./fl_network_control.sh control
```

## 🎯 Network Parameters

| Parameter | Example Values | Description |
|-----------|---------------|-------------|
| Latency | `20ms`, `100ms`, `200ms`, `500ms` | Network delay |
| Bandwidth | `1mbit`, `5mbit`, `10mbit`, `100mbit` | Link speed |
| Packet Loss | `1`, `2.5`, `5`, `10` | Loss percentage |
| Jitter | `10ms`, `30ms`, `50ms` | Delay variation |

## 📋 Typical Workflows

### Workflow 1: Apply Poor Network to Client 1
```bash
# Terminal 1: Monitor
python3 fl_training_dashboard.py

# Terminal 2: Apply conditions
python3 fl_network_monitor.py \
    --client-id 1 \
    --latency 200ms \
    --bandwidth 1mbit \
    --loss 3 \
    --jitter 50ms

# Verify
python3 fl_network_monitor.py --show-status
```

### Workflow 2: Dynamic Changes During Training
```bash
# Round 1-5: Good network
python3 fl_network_monitor.py --all --latency 20ms --bandwidth 50mbit

# Round 6-10: Degrade client 1
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 3

# Round 11+: Clear all
python3 fl_network_monitor.py --all --clear
```

### Workflow 3: Interactive Control
```bash
python3 fl_network_monitor.py --monitor

# Then use menu:
# 1 - Apply to specific client
# 2 - Apply to all clients
# 3 - Change single parameter
# 4 - Show status
# 5 - Clear specific client
# 6 - Clear all
# 7 - Start FL monitor
```

## 🔧 Technical Commands

### Get Container Veth Interface
```bash
# For specific container
container_id=$(docker ps -qf name=<container_name>)
container_ifindex=$(docker exec $container_id cat /sys/class/net/eth0/iflink)
ip link | grep "^$container_ifindex:"
```

### Manual TC Commands
```bash
# Find veth
veth=$(ip link | grep "^<ifindex>:" | awk -F': ' '{print $2}')

# Apply latency
sudo tc qdisc add dev $veth root netem delay 200ms

# Change latency
sudo tc qdisc change dev $veth root netem delay 300ms

# Add loss
sudo tc qdisc change dev $veth root netem delay 200ms loss 2%

# Clear rules
sudo tc qdisc del dev $veth root

# Show rules
sudo tc qdisc show dev $veth
```

## 📈 Monitoring

### Dashboard Output
- Server status (CPU, memory, network I/O)
- Client status with network conditions
- Recent activity logs
- Real-time updates

### Check Container Stats
```bash
docker stats --no-stream
docker stats <container_name> --no-stream
```

### View Logs
```bash
docker logs -f <container_name>
docker logs --tail 50 <container_name>
```

## 🎬 Complete Example

```bash
# Step 1: Start FL training
cd /path/to/project
docker-compose up -d

# Step 2: Start monitoring (Terminal 1)
cd Network_Simulation/
python3 fl_training_dashboard.py

# Step 3: Apply initial conditions (Terminal 2)
python3 fl_network_monitor.py --all --latency 50ms --bandwidth 20mbit

# Step 4: Monitor and adjust during training
# Wait for a few rounds...
python3 fl_network_monitor.py --client-id 1 --latency 200ms --loss 3

# Step 5: Clear after training
python3 fl_network_monitor.py --all --clear

# Step 6: Stop containers
cd ..
docker-compose down
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No veth found | Refresh: Option 8 in interactive mode |
| Permission denied | Use `sudo` or check sudo access |
| File exists error | Clear first: `--clear` then apply |
| Container not found | Check: `docker ps --filter "name=client"` |

## 💡 Tips

1. **Always start monitoring before applying conditions**
2. **Use `--show-status` to verify conditions**
3. **Clear conditions after experiments for clean slate**
4. **Keep Terminal 1 for monitoring, Terminal 2 for control**
5. **Use interactive mode for exploratory testing**
6. **Use direct commands for scripted experiments**

## 🔗 Files

| File | Purpose |
|------|---------|
| `fl_network_monitor.py` | Main network control tool |
| `fl_training_dashboard.py` | Real-time monitoring dashboard |
| `fl_network_control.sh` | Quick command wrapper |
| `FL_NETWORK_CONTROL_GUIDE.md` | Full documentation |

## 📞 Getting Help

```bash
# Tool help
python3 fl_network_monitor.py --help
python3 fl_training_dashboard.py --help
./fl_network_control.sh

# Interactive mode (guided)
python3 fl_network_monitor.py --monitor
```

---

**Remember**: Network changes apply in **parallel** during training. Monitor effects in real-time!
