# 🚀 Dynamic Client Support - Quick Reference

## What Changed?
All FL servers (MQTT, AMQP, gRPC, QUIC, DDS, Unified) now support **2-100 clients dynamically**.

## Key Features
✅ Start with minimum 2 clients, scale to 100  
✅ Clients can join during training  
✅ All registered clients included in aggregation & evaluation  
✅ Late-joining clients receive current global model  

## Quick Start

### 1. Update Experiment GUI (Optional)
Edit `Network_Simulation/experiment_gui.py` line 475:
```python
self.min_clients.setValue(3)  # Change from 2 to 3+
```

### 2. Rebuild Docker Images (REQUIRED)
```bash
cd Docker
# For current experiment (emotion):
docker-compose -f docker-compose-emotion.gpu-isolated.yml build

# Or use GUI: Docker Build tab → Build button
```

### 3. Start Experiment (With 2 Clients)
```bash
cd Network_Simulation
python3 experiment_gui.py
# Set Min Clients: 3 (or desired)
# Click "Start Experiment"
```

### 4. Add Distributed Client (From Another PC)
```bash
cd Network_Simulation
./launch_distributed_client.sh

# Configure:
Server IP: 129.69.102.245
Client ID: 3
Use Case: emotion
Protocol: mqtt
Network Scenario: excellent

# Click "Test Connection" → Should succeed
# Click "Start Client" → Joins experiment
```

## Environment Variables

```bash
# Set before starting experiment:
export MIN_CLIENTS=3      # Minimum to start training (default: 2)
export MAX_CLIENTS=10     # Maximum allowed (default: 100)

cd Network_Simulation
python3 experiment_gui.py
```

## Port Mappings (IMPORTANT!)

| Protocol | Internal Port | External Port (Host) |
|----------|---------------|---------------------|
| MQTT     | 1883          | **31883**           |
| AMQP     | 5672          | **35672**           |
| gRPC     | 50051         | 50051               |
| QUIC     | 4433          | 4433                |
| DDS      | N/A           | Domain-based        |

**Note**: Distributed clients connect to **external ports** (31883, 35672)

## Verification

### Check Server Logs:
```bash
docker logs -f fl-server-mqtt-emotion
```

Look for:
```
Client 1 registered (1/3 expected, min: 3)
Client 2 registered (2/3 expected, min: 3)
Client 3 registered (3/3 expected, min: 3)
All clients registered. Distributing initial global model...
Training started at: 2026-02-04 15:30:00
```

### Late-Joining Client:
```
[LATE JOIN] Client 4 joined after training started
[DYNAMIC] Updated client count: 3 -> 4
```

### Aggregation with All Clients:
```
Received update from client 1 (1/4)
Received update from client 2 (2/4)
Received update from client 3 (3/4)
Received update from client 4 (4/4)
Aggregating models from 4 clients...
```

## Troubleshooting

### "Training not starting"
**Problem**: Not enough clients registered  
**Solution**: Check MIN_CLIENTS setting matches number of available clients

### "Cannot connect from distributed client"
**Problem**: Wrong port (1883 vs 31883)  
**Solution**: Use port **31883** for MQTT, **35672** for AMQP

### "Client image not found"
**Problem**: Docker image doesn't exist  
**Solution**: Build images first (see step 2)

### "Aggregation waiting forever"
**Problem**: One client not responding  
**Solution**: Check all client containers are running:
```bash
docker ps | grep client
```

## File Backups

All original files backed up:
- Servers: `*.bak_dynamic_clients`
- Docker: `*.bak_clients`

To restore:
```bash
# Example for MQTT server:
cp Server/Emotion_Recognition/FL_Server_MQTT.py.bak_dynamic_clients \
   Server/Emotion_Recognition/FL_Server_MQTT.py
```

## Testing Checklist

- [ ] Rebuild Docker images
- [ ] Start experiment with MIN_CLIENTS=3
- [ ] Wait for 2 local clients to register
- [ ] Verify training NOT started yet (waiting for 3rd)
- [ ] Launch distributed client (ID=3) from GUI
- [ ] Verify training starts after 3rd client joins
- [ ] Check all 3 clients participate in aggregation
- [ ] (Optional) Add 4th client during training
- [ ] Verify 4th client receives current model
- [ ] Verify all 4 clients in next round's aggregation

## Updated Files Summary

- **18** Server files (all protocols × all use cases)
- **17** Docker Compose files
- **1** Distributed Client GUI
- **2** Documentation files

Total: **38 files** updated ✅

## Support

For detailed information, see:
- `DYNAMIC_CLIENT_SUPPORT_COMPLETE.md` (full documentation)
- `DISTRIBUTED_CLIENT_SETUP.md` (distributed setup guide)
- `DISTRIBUTED_CLIENT_QUICK_START.md` (quick start guide)

---
**Last Updated**: 2026-02-04  
**Status**: ✅ Ready for use (after rebuilding images)
