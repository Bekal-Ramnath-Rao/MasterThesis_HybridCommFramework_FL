# Distributed FL Client Setup Guide

## Overview

This guide explains how to run federated learning clients on multiple PCs that connect to a central experiment server. This allows for true distributed FL experiments where clients can join from different machines on the same network.

## Architecture

```
┌─────────────────────────┐
│   Main Experiment PC    │
│  (Server + Initial      │
│   Clients)              │
│                         │
│  • FL Server            │
│  • MQTT Broker          │
│  • RabbitMQ Broker      │
│  • gRPC Server          │
│  • QUIC Server          │
│  • DDS Domain           │
│  • Client 1             │
│  • Client 2             │
└────────┬────────────────┘
         │
         │  Network Connection
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
┌───▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
│Remote │ │Remote│ │Remote│ │Remote│
│ PC 1  │ │ PC 2 │ │ PC 3 │ │ PC 4 │
│       │ │      │ │      │ │      │
│Client │ │Client│ │Client│ │Client│
│  3    │ │  4   │ │  5   │ │  6   │
└───────┘ └──────┘ └──────┘ └──────┘
```

## Features

### 1. Distributed Client GUI
- **Location**: `Network_Simulation/distributed_client_gui.py`
- **Purpose**: Run on remote PCs to connect clients to the central server
- **Features**:
  - Configure server IP address
  - Test connection to server
  - Select client ID (unique per client)
  - Choose use case (emotion, mentalstate, temperature)
  - Select protocol mode (RL-Unified or specific protocol)
  - Apply network conditions locally to simulate poor connectivity
  - Monitor client status and logs in real-time

### 2. Dynamic Client Support
- Server adapts to variable number of clients
- Minimum of 2 clients required
- Clients can join mid-experiment (with convergence adaptation)
- Server waits for all registered clients before starting each round

### 3. Network Scenario Support
Each remote client can apply different network conditions:
- Excellent (5ms latency, 100Mbps)
- Good (20ms, 50Mbps)
- Moderate (50ms, 20Mbps)
- Poor (100ms, 5Mbps)
- Very Poor (200ms, 1Mbps)
- Satellite (600ms, 10Mbps)
- Congestion (Light/Moderate/Heavy)

## Setup Instructions

### On the Main Experiment PC

1. **Start the Experiment Server**:
   ```bash
   cd Network_Simulation
   python3 experiment_gui.py
   ```

2. **Configure the Experiment**:
   - Set **Number of Clients** to the total expected (e.g., 4 clients)
   - This includes both local clients (started by experiment_gui) and remote clients
   - For example, if experiment starts 2 local clients and you want 2 remote clients, set to 4

3. **Start the Experiment**:
   - Click "Start Experiment"
   - Server will wait for all clients to register before beginning training

4. **Note the Server IP**:
   ```bash
   # Find your IP address
   hostname -I
   # Example output: 192.168.1.100
   ```

### On Each Remote PC

1. **Copy the Project** (or just the necessary files):
   ```bash
   # Option 1: Clone the full repository
   git clone <repository-url>
   
   # Option 2: Copy just the client GUI and Docker files
   scp user@server:/path/to/Network_Simulation/distributed_client_gui.py .
   scp user@server:/path/to/Network_Simulation/launch_distributed_client.sh .
   ```

2. **Ensure Docker Images are Available**:
   ```bash
   # Option 1: Build on remote PC (if you have the full project)
   cd Docker
   docker-compose -f docker-compose-unified-emotion.yml build
   
   # Option 2: Export/import images from main PC
   # On main PC:
   docker save docker-fl-client-unified-emotion:latest | gzip > fl-client-emotion.tar.gz
   scp fl-client-emotion.tar.gz user@remote-pc:/tmp/
   
   # On remote PC:
   docker load < /tmp/fl-client-emotion.tar.gz
   ```

3. **Launch the Distributed Client GUI**:
   ```bash
   cd Network_Simulation
   ./launch_distributed_client.sh
   
   # Or directly:
   python3 distributed_client_gui.py
   ```

4. **Configure and Start Client**:
   - Enter server IP (e.g., 192.168.1.100)
   - Click "Test Connection" to verify connectivity
   - Set unique Client ID (e.g., 3 if main experiment has clients 1-2)
   - Select same use case as main experiment
   - Choose protocol mode (RL-Unified recommended)
   - Optionally select network scenario to simulate
   - Click "Start Client"

## Network Requirements

### Firewall Configuration

Ensure the following ports are accessible from remote PCs to the server:

| Protocol | Port  | Description           |
|----------|-------|-----------------------|
| MQTT     | 1883  | Message Broker        |
| AMQP     | 5672  | RabbitMQ              |
| gRPC     | 50051 | gRPC Communication    |
| QUIC     | 4433  | QUIC Protocol         |
| DDS      | N/A   | Auto-discovery        |

### Server Firewall Rules (Ubuntu/Linux)

```bash
# Allow MQTT
sudo ufw allow 1883/tcp

# Allow AMQP
sudo ufw allow 5672/tcp

# Allow gRPC
sudo ufw allow 50051/tcp

# Allow QUIC
sudo ufw allow 4433/udp

# For DDS multicast (if needed)
sudo ufw allow from 192.168.1.0/24 to any port 7400:7500 proto udp
```

### Testing Connectivity

From remote PC, test each port:

```bash
# Test MQTT
nc -zv 192.168.1.100 1883

# Test AMQP
nc -zv 192.168.1.100 5672

# Test gRPC
nc -zv 192.168.1.100 50051

# Test QUIC
nc -zvu 192.168.1.100 4433
```

## Dynamic Client Joining

### How It Works

1. **Initial Setup**: Experiment starts with minimum clients (e.g., 2)
2. **Late Joining**: Additional clients can join during training
3. **Server Adaptation**: 
   - Server updates expected client count
   - Waits for all currently registered clients
   - Convergence check includes late-joining clients

### Increasing Client Count During Experiment

#### Method 1: Via Experiment GUI (Main PC)

1. While experiment is running, modify the "Min Clients" setting
2. The server will adapt on the next round
3. Late-joining clients will participate in subsequent rounds

#### Method 2: Environment Variable

Edit the docker-compose file or update the running container:

```bash
# Update NUM_CLIENTS for the server container
docker exec fl-server-unified sh -c 'export NUM_CLIENTS=4'
```

### Convergence with Dynamic Clients

The server implements intelligent convergence checking:

1. **Baseline Metrics**: Established with initial client set
2. **Late Joiner Integration**: 
   - Late-joining clients receive current global model
   - Their data contributes to subsequent aggregations
   - Convergence patience resets when new clients join
3. **Fair Evaluation**: All active clients must meet convergence criteria

## Example Scenarios

### Scenario 1: 2 Local + 2 Remote Clients

**Main PC (experiment_gui.py)**:
- Set "Number of Clients" to 4
- Start experiment (creates clients 1-2 locally)

**Remote PC 1 (distributed_client_gui.py)**:
- Server IP: 192.168.1.100
- Client ID: 3
- Start client

**Remote PC 2 (distributed_client_gui.py)**:
- Server IP: 192.168.1.100
- Client ID: 4
- Start client

### Scenario 2: Poor Network Simulation

**Remote PC 1**:
- Network Scenario: "Poor" (100ms latency, 5Mbps)
- Simulates mobile network conditions

**Remote PC 2**:
- Network Scenario: "Satellite" (600ms latency)
- Simulates extreme delay conditions

**Main PC Clients**:
- Network Scenario: "Excellent"
- Baseline performance

This allows studying how heterogeneous network conditions affect FL convergence.

### Scenario 3: Dynamic Joining

1. Start experiment with 2 clients
2. Run for 5 rounds
3. Start 2 additional remote clients mid-experiment
4. Server adapts and includes new clients from round 6 onwards
5. Convergence considers all 4 clients

## Troubleshooting

### Client Cannot Connect to Server

**Problem**: "Cannot reach MQTT broker" error

**Solutions**:
1. Verify server IP is correct
2. Check firewall rules on server
3. Ensure server containers are running:
   ```bash
   docker ps | grep broker
   ```
4. Test network connectivity:
   ```bash
   ping 192.168.1.100
   ```

### Client Container Fails to Start

**Problem**: Docker error when starting client

**Solutions**:
1. Check if image exists:
   ```bash
   docker images | grep fl-client
   ```
2. Verify container name isn't already in use:
   ```bash
   docker ps -a | grep fl-client
   docker rm fl-client-X-distributed
   ```
3. Check Docker logs:
   ```bash
   docker logs fl-client-X-distributed
   ```

### Network Conditions Not Applied

**Problem**: tc command fails inside container

**Solutions**:
1. Ensure container has NET_ADMIN capability (automatically added by GUI)
2. Check if network interface exists:
   ```bash
   docker exec fl-client-X-distributed ip link show
   ```
3. Use host network mode (automatically used by distributed GUI)

### Server Not Waiting for All Clients

**Problem**: Training starts before all clients connect

**Solutions**:
1. Ensure NUM_CLIENTS environment variable matches total expected clients
2. Check server logs for registration messages
3. Verify client IDs are unique

## Advanced Configuration

### Custom Network Conditions

Edit `distributed_client_gui.py` to add custom scenarios:

```python
scenarios = {
    "custom": {
        "latency": "150ms",
        "bandwidth": "10mbit",
        "jitter": "20ms",
        "loss": "2"
    }
}
```

### GPU Support for Remote Clients

If remote PC has NVIDIA GPU:

1. Install nvidia-docker2
2. Enable GPU in distributed client GUI (checkbox)
3. Client container will use GPU acceleration

### Monitoring Multiple Remote Clients

From main PC, monitor all clients:

```bash
# List all FL containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep fl-client

# Monitor specific remote client logs
docker logs -f fl-client-3-distributed
```

## Best Practices

1. **Start Server First**: Always start the main experiment before remote clients
2. **Unique Client IDs**: Ensure each client has a unique ID (1, 2, 3, 4, ...)
3. **Matching Use Cases**: All clients must use the same use case (emotion/mentalstate/temperature)
4. **Network Stability**: For production experiments, use wired connections for server
5. **Resource Allocation**: Remote PCs should have sufficient RAM (8GB+) and CPU cores (4+)
6. **Docker Images**: Keep images synchronized across all PCs
7. **Time Synchronization**: Use NTP to keep all PCs synchronized

## Performance Considerations

### Bandwidth Requirements

Per client, per round (approximate):
- Model size: 50-200 MB (depending on architecture)
- Metrics: <1 MB
- Control messages: <1 KB

Total bandwidth per round: ~100-400 MB per client

### Latency Impact

- LAN (Excellent): 1-5ms - Minimal impact
- WiFi (Good): 20-50ms - Slight impact
- Poor Network: 100-200ms - Noticeable delays
- Satellite: 600ms+ - Significant round time increase

### Recommended Hardware

**Server PC**:
- CPU: 8+ cores
- RAM: 32GB+
- GPU: NVIDIA with 8GB+ VRAM
- Network: Gigabit Ethernet

**Client PCs**:
- CPU: 4+ cores
- RAM: 8GB+
- GPU: Optional (4GB+ VRAM if enabled)
- Network: 100Mbps+ (WiFi acceptable)

## Security Considerations

**Warning**: This setup is designed for research/development environments on trusted networks.

For production:
1. Enable SSL/TLS for all protocols
2. Use authentication for MQTT and AMQP
3. Configure firewall rules restrictively
4. Use VPN for remote connections
5. Implement client authentication
6. Encrypt model weights in transit

## Conclusion

The distributed FL client setup enables realistic federated learning experiments across multiple machines with heterogeneous network conditions. This is essential for studying real-world FL scenarios where clients may have varying connectivity quality.

For questions or issues, refer to the main project documentation or check the troubleshooting section above.
