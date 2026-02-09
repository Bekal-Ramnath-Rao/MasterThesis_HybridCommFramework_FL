# Distributed Federated Learning System - README

## 🌐 Overview

This system enables **true distributed federated learning** across multiple physical machines on the same network. Run the main experiment on one PC and connect additional clients from other PCs to simulate realistic heterogeneous FL scenarios.

## 📁 Key Files Created

### GUI Applications
1. **[Network_Simulation/distributed_client_gui.py](Network_Simulation/distributed_client_gui.py)**
   - GUI for running FL clients on remote PCs
   - Features: connection testing, network simulation, real-time monitoring
   - Use: `python3 distributed_client_gui.py`

2. **[Network_Simulation/launch_distributed_client.sh](Network_Simulation/launch_distributed_client.sh)**
   - Launch script for distributed client GUI
   - Use: `./launch_distributed_client.sh`

### Update Scripts
3. **[update_dynamic_client_support.py](update_dynamic_client_support.py)**
   - Updates FL servers to support dynamic client joining
   - Adds late-joining client handling and adaptive convergence
   - Use: `python3 update_dynamic_client_support.py`

### Documentation
4. **[DISTRIBUTED_CLIENT_SETUP.md](DISTRIBUTED_CLIENT_SETUP.md)**
   - Complete setup guide with architecture, troubleshooting, best practices
   - 800+ lines of comprehensive documentation

5. **[DISTRIBUTED_CLIENT_QUICK_START.md](DISTRIBUTED_CLIENT_QUICK_START.md)**
   - Quick reference guide (TL;DR version)
   - Essential commands and configurations

## 🚀 Quick Start

### Step 1: Main Experiment PC
```bash
cd Network_Simulation
python3 experiment_gui.py
```
- Set "Number of Clients" to total (e.g., 4 for 2 local + 2 remote)
- Note server IP: `hostname -I`

### Step 2: Remote PC(s)
```bash
cd Network_Simulation
./launch_distributed_client.sh
```
- Enter server IP
- Configure client ID (unique)
- Select same use case as main experiment
- Start client

## ✨ Features

### Distributed Architecture
- **Multi-PC Support**: Run clients on different machines
- **Network Heterogeneity**: Each client can simulate different network conditions
- **Real-World Scenarios**: Test FL with actual network latency and limited bandwidth

### Dynamic Client Management
- **Late Joining**: Clients can join mid-experiment
- **Adaptive Server**: Automatically adjusts to variable client count (min 2)
- **Convergence Adaptation**: Resets convergence checks when new clients join
- **Graceful Handling**: Server waits for all registered clients each round

### Network Simulation (Per Client)
Each client can independently simulate:
- Excellent (5ms latency, 100Mbps)
- Good (20ms, 50Mbps)
- Moderate (50ms, 20Mbps)
- Poor (100ms, 5Mbps)
- Very Poor (200ms, 1Mbps)
- Satellite (600ms, 10Mbps)
- Congestion (Light/Moderate/Heavy)

### Protocol Support
All 5 communication protocols supported:
- MQTT (port 1883)
- AMQP/RabbitMQ (port 5672)
- gRPC (port 50051)
- QUIC (port 4433)
- DDS (auto-discovery)

**RL-Unified Mode**: Clients automatically select best protocol per round

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│      Main Experiment PC             │
│  IP: 192.168.1.100                  │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  FL Server (Unified)        │   │
│  │  • MQTT Broker              │   │
│  │  • RabbitMQ Broker          │   │
│  │  • gRPC Server              │   │
│  │  • QUIC Server              │   │
│  │  • DDS Domain               │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌──────────┐  ┌──────────┐        │
│  │Client 1  │  │Client 2  │        │
│  │(Docker)  │  │(Docker)  │        │
│  └──────────┘  └──────────┘        │
└──────────────┬──────────────────────┘
               │
        Network│(LAN/WiFi)
               │
      ┌────────┴────────┬────────────┐
      │                 │            │
┌─────▼──────┐   ┌─────▼──────┐   ┌─▼───────┐
│Remote PC 1 │   │Remote PC 2 │   │Remote PC│
│192.168.1.2 │   │192.168.1.3 │   │...      │
│            │   │            │   │         │
│  Client 3  │   │  Client 4  │   │Client N │
│  (Docker)  │   │  (Docker)  │   │(Docker) │
│            │   │            │   │         │
│ Network:   │   │ Network:   │   │Network: │
│  Poor      │   │  Satellite │   │Excellent│
└────────────┘   └────────────┘   └─────────┘
```

## 📋 Requirements

### Server PC (Main Experiment)
- Ubuntu 20.04+
- Docker & Docker Compose
- Python 3.8+
- 32GB RAM, 8+ CPU cores
- NVIDIA GPU (8GB+ VRAM)
- Gigabit Ethernet

### Client PCs (Remote)
- Ubuntu 20.04+ or similar Linux
- Docker installed
- Python 3.8+ with PyQt5
- 8GB RAM, 4+ CPU cores
- 100Mbps+ network (WiFi acceptable)
- GPU optional (4GB+ VRAM if enabled)

### Network
- All PCs on same network (LAN or WiFi)
- Firewall ports open: 1883, 5672, 50051, 4433
- Minimum 100Mbps connectivity recommended

## 🔧 Installation

### Main PC
```bash
# Already have experiment_gui.py
cd Network_Simulation
python3 experiment_gui.py
```

### Remote PCs
```bash
# Copy distributed client files
git clone <repository-url>
cd MasterThesis_HybridCommFramework_FL/Network_Simulation

# Install dependencies
pip3 install PyQt5

# Launch client GUI
./launch_distributed_client.sh
```

## 🎯 Use Cases

### Scenario 1: Heterogeneous Network Study
- **Main PC**: 2 clients with excellent network
- **Remote PC 1**: 1 client with poor network (mobile simulation)
- **Remote PC 2**: 1 client with satellite network (high latency)
- **Goal**: Study impact of network heterogeneity on FL convergence

### Scenario 2: Scalability Testing
- **Main PC**: Server only (or 1 client)
- **Remote PCs**: 5-10 clients distributed across machines
- **Goal**: Test server scalability with many distributed clients

### Scenario 3: Dynamic Participation
- **Phase 1**: Start with 2 clients
- **Phase 2**: After 10 rounds, add 2 remote clients
- **Phase 3**: Observe convergence adaptation
- **Goal**: Study impact of clients joining mid-training

### Scenario 4: Protocol Comparison
- **Client 1-2** (Main PC): Use RL-Unified
- **Client 3** (Remote): Fixed to gRPC
- **Client 4** (Remote): Fixed to MQTT
- **Goal**: Compare RL-selected vs fixed protocol performance

## 🔐 Security Notes

⚠️ **Warning**: This setup is for research/development on trusted networks.

For production:
- Enable SSL/TLS for all protocols
- Use authentication (MQTT, AMQP)
- Configure restrictive firewall rules
- Use VPN for remote connections
- Implement client authentication

## 📊 Monitoring

### Main Experiment GUI
- **Experiment Output**: Overall progress
- **FL Training Monitor**: Per-round metrics
- **Server Logs**: Server-side events
- **Client Logs**: Select any client to view logs
- **Packet Logs**: Network traffic analysis

### Distributed Client GUI
- **Connection Status**: Real-time connectivity indicator
- **Client Logs**: Local client container logs
- **Status Info**: Training progress, round number

## 🐛 Troubleshooting

### Connection Failed
```bash
# Check server IP
hostname -I

# Test connectivity
ping <server-ip>
nc -zv <server-ip> 1883

# Check firewall
sudo ufw status
sudo ufw allow 1883/tcp
```

### Container Won't Start
```bash
# Check if image exists
docker images | grep fl-client

# Remove conflicting container
docker rm fl-client-X-distributed

# Check Docker logs
docker logs fl-client-X-distributed
```

### Server Not Waiting
```bash
# Check server environment
docker exec fl-server-unified env | grep NUM_CLIENTS

# Update server
python3 update_dynamic_client_support.py
docker-compose -f Docker/docker-compose-unified-<usecase>.yml build
```

## 📚 Documentation Index

| Document | Purpose | Length |
|----------|---------|--------|
| [DISTRIBUTED_CLIENT_SETUP.md](DISTRIBUTED_CLIENT_SETUP.md) | Complete guide | Full |
| [DISTRIBUTED_CLIENT_QUICK_START.md](DISTRIBUTED_CLIENT_QUICK_START.md) | Quick reference | Quick |
| This README | Overview | Summary |

## 🔄 Workflow

```
┌─────────────────────────────────────────────────────┐
│                   SETUP PHASE                       │
├─────────────────────────────────────────────────────┤
│ 1. Start experiment_gui.py on main PC              │
│ 2. Configure experiment (protocols, use case, etc.)│
│ 3. Set total client count (local + remote)         │
│ 4. Note server IP address                          │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│              REMOTE PC SETUP                        │
├─────────────────────────────────────────────────────┤
│ 1. Launch distributed_client_gui.py                │
│ 2. Enter server IP and test connection             │
│ 3. Configure client (ID, use case, network)        │
│ 4. Repeat for each remote PC                       │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│             START EXPERIMENT                        │
├─────────────────────────────────────────────────────┤
│ 1. Click "Start Experiment" on main PC             │
│ 2. Server waits for all clients to register        │
│ 3. Click "Start Client" on each remote GUI         │
│ 4. Training begins when all clients connected      │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│          TRAINING & MONITORING                      │
├─────────────────────────────────────────────────────┤
│ • Main GUI shows aggregated metrics                │
│ • Remote GUIs show individual client logs          │
│ • Network conditions applied per client            │
│ • RL selects protocols dynamically                 │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│            COMPLETION                               │
├─────────────────────────────────────────────────────┤
│ • Convergence achieved or max rounds reached       │
│ • Results saved to shared_data/                    │
│ • Clients automatically stopped                    │
│ • Review metrics in experiment GUI                 │
└─────────────────────────────────────────────────────┘
```

## 🎓 Research Applications

This distributed setup enables:
- **Realistic FL Experiments**: True network conditions, not just simulation
- **Heterogeneity Studies**: Mix of powerful/weak devices, good/poor networks
- **Scalability Research**: Test with many geographically distributed clients
- **Protocol Comparison**: Compare performance across real network conditions
- **Dynamic Participation**: Study client churn and late-joining behavior
- **Resource Utilization**: Leverage multiple machines for parallel experiments

## 🤝 Contributing

When adding features:
1. Update [distributed_client_gui.py](Network_Simulation/distributed_client_gui.py) for client-side
2. Update [FL_Server_Unified.py](Server/*/FL_Server_Unified.py) for server-side
3. Run [update_dynamic_client_support.py](update_dynamic_client_support.py) if needed
4. Update documentation
5. Test with at least 2 remote PCs

## 📝 License

Same as main project

## 👨‍💻 Author

Part of Master Thesis: Hybrid Communication Framework for Federated Learning

---

**Quick Commands**:
```bash
# Main PC
python3 Network_Simulation/experiment_gui.py

# Remote PC
./Network_Simulation/launch_distributed_client.sh

# Update server (if needed)
python3 update_dynamic_client_support.py
```

For detailed instructions, see [DISTRIBUTED_CLIENT_SETUP.md](DISTRIBUTED_CLIENT_SETUP.md)
