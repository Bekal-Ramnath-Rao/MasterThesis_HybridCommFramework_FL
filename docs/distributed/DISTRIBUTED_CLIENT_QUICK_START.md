# Distributed FL Client - Quick Start Guide

## TL;DR

Run FL clients on multiple PCs connected to same network as experiment server.

## Setup in 3 Steps

### Step 1: On Main Experiment PC

```bash
cd Network_Simulation
python3 experiment_gui.py
```

- Set "Number of Clients" = total expected (e.g., 4 for 2 local + 2 remote)
- Start experiment
- Note server IP: `hostname -I` (e.g., 192.168.1.100)

### Step 2: On Remote PC(s)

```bash
cd Network_Simulation
./launch_distributed_client.sh
```

### Step 3: In Distributed Client GUI

1. **Enter Server IP**: `192.168.1.100`
2. **Test Connection**: Click "Test Connection"
3. **Configure**:
   - Client ID: `3` (unique, sequential)
   - Use Case: `emotion` (match main experiment)
   - Protocol: `RL-Unified (Auto Select)`
   - Network: Choose scenario (optional)
4. **Start**: Click "Start Client"

## Key Points

✅ **DO**:
- Use unique Client IDs (1, 2, 3, 4...)
- Match use case with main experiment
- Test connection before starting
- Use wired network for best results

❌ **DON'T**:
- Use duplicate Client IDs
- Mix different use cases
- Start clients before server
- Forget to open firewall ports

## Required Ports

| Protocol | Port  |
|----------|-------|
| MQTT     | 1883  |
| AMQP     | 5672  |
| gRPC     | 50051 |
| QUIC     | 4433  |

## Firewall Setup (Server)

```bash
sudo ufw allow 1883/tcp  # MQTT
sudo ufw allow 5672/tcp  # AMQP
sudo ufw allow 50051/tcp # gRPC
sudo ufw allow 4433/udp  # QUIC
```

## Test Connection

```bash
# From remote PC
nc -zv <SERVER_IP> 1883
nc -zv <SERVER_IP> 5672
nc -zv <SERVER_IP> 50051
```

## Common Issues

### "Cannot reach MQTT broker"
- Check server IP is correct
- Verify firewall rules
- Ensure server is running: `docker ps`

### "Client already exists"
- Use different Client ID
- Stop existing client: `docker stop fl-client-X-distributed`

### Network conditions not working
- Container has NET_ADMIN (automatic)
- Using host network mode (automatic)

## Example Scenarios

### 2 Local + 2 Remote

**Main PC**: Start with 4 clients total
**Remote PC 1**: Client ID 3
**Remote PC 2**: Client ID 4

### Different Network Conditions

**Remote PC 1**: "Poor Network" (100ms latency)
**Remote PC 2**: "Satellite" (600ms latency)
**Main PC**: "Excellent" (baseline)

### Dynamic Joining

1. Start with 2 clients
2. Run 5 rounds
3. Start 2 remote clients
4. Server adapts automatically

## Monitoring

### Check Running Clients

```bash
docker ps | grep fl-client
```

### View Client Logs

```bash
docker logs -f fl-client-3-distributed
```

### Monitor from Main PC

All logs visible in experiment GUI's "Client Logs" tab

## Performance Tips

- **LAN**: <5ms latency ✓ Best
- **WiFi**: 20-50ms ✓ Good  
- **Poor**: 100-200ms ⚠️ Slow
- **Satellite**: 600ms+ ⚠️ Very slow

## Docker Images

### Check if images exist

```bash
docker images | grep fl-client
```

### Build if missing

```bash
cd Docker
docker-compose -f docker-compose-unified-emotion.yml build
```

### Transfer from main PC

```bash
# On main PC
docker save docker-fl-client-unified-emotion:latest | gzip > client.tar.gz
scp client.tar.gz user@remote:/tmp/

# On remote PC
docker load < /tmp/client.tar.gz
```

## Minimum Requirements

**Server PC**:
- 8+ CPU cores
- 32GB RAM
- NVIDIA GPU (8GB+ VRAM)
- Gigabit Ethernet

**Client PCs**:
- 4+ CPU cores
- 8GB RAM
- 100Mbps network (WiFi ok)
- GPU optional

## Network Scenarios

| Scenario            | Latency | Bandwidth | Loss |
|---------------------|---------|-----------|------|
| Excellent           | 5ms     | 100Mbps   | 0%   |
| Good                | 20ms    | 50Mbps    | 0.1% |
| Moderate            | 50ms    | 20Mbps    | 0.5% |
| Poor                | 100ms   | 5Mbps     | 1%   |
| Very Poor           | 200ms   | 1Mbps     | 3%   |
| Satellite           | 600ms   | 10Mbps    | 2%   |
| Light Congestion    | 30ms    | 30Mbps    | 0.5% |
| Moderate Congestion | 75ms    | 15Mbps    | 1.5% |
| Heavy Congestion    | 150ms   | 5Mbps     | 3%   |

## Advanced

### Update Server for Dynamic Clients

```bash
python3 update_dynamic_client_support.py
# Then rebuild Docker images
```

### Custom Network Conditions

Edit `distributed_client_gui.py` to add custom scenarios

### GPU on Remote Client

- Ensure nvidia-docker installed
- Check "Enable GPU" in GUI
- Client uses GPU if available

## Support

For detailed docs: [DISTRIBUTED_CLIENT_SETUP.md](DISTRIBUTED_CLIENT_SETUP.md)

For issues: Check server/client logs in GUI

## Summary

```
┌───────────────────┐
│ Main PC (Server)  │ ← hostname -I → 192.168.1.100
│ Clients: 1-2      │
└─────────┬─────────┘
          │
    ┌─────┴─────┬─────────┐
    │           │         │
┌───▼───┐  ┌───▼───┐  ┌──▼────┐
│Remote │  │Remote │  │Remote │
│PC 1   │  │PC 2   │  │PC 3   │
│       │  │       │  │       │
│Client │  │Client │  │Client │
│  3    │  │  4    │  │  5    │
└───────┘  └───────┘  └───────┘
```

**Quick Command Sequence**:

```bash
# Main PC
cd Network_Simulation && python3 experiment_gui.py
# Set clients to 4, start experiment

# Remote PC
cd Network_Simulation && ./launch_distributed_client.sh
# Server IP: 192.168.1.100, Client ID: 3, Start
```

Done! 🎉
