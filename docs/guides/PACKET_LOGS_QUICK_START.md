# Quick Start: Packet Logs Integration

## What's Been Done ✅

### 1. GUI Integration
- ✅ PacketLogsTab component integrated into experiment_gui.py
- ✅ Advanced packet visualization with filtering and auto-refresh
- ✅ Graceful fallback if component unavailable

### 2. Docker Configuration
- ✅ All 3 unified docker-compose files updated:
  - `docker-compose-unified-emotion.yml`
  - `docker-compose-unified-mentalstate.yml`
  - `docker-compose-unified-temperature.yml`
- ✅ shared_data volume mounted for all services (server + 2 clients each)
- ✅ NODE_TYPE environment variable added (server/client)

## Quick Test

### Terminal 1: Start Docker
```bash
cd Docker
docker-compose -f docker-compose-unified-emotion.yml up --build
```

### Terminal 2: Launch GUI
```bash
cd Network_Simulation
python3 experiment_gui.py
```

### In GUI:
1. Go to "Packet Logs" tab
2. Select "Server" or "Client 1/2" from dropdown
3. Watch packets appear in real-time!

## Database Files

After starting the system, you'll see:
```bash
shared_data/
├── packet_logs_server.db
├── packet_logs_client_1.db
└── packet_logs_client_2.db
```

## Features

### PacketLogsTab Component
- 📊 Real-time packet visualization
- 🔄 Auto-refresh every 2 seconds
- 🎯 Protocol filtering (MQTT, AMQP, gRPC, QUIC, DDS)
- 🖥️ Node selection (Server/Client 1/Client 2)
- 📝 Detailed packet information:
  - Timestamp
  - Protocol used
  - Message type
  - Size in bytes
  - Direction (sent/received)
  - Payload preview

### Logged Protocols
All 5 protocols are logged:
1. **MQTT** - Control signals (always used)
2. **AMQP** - RabbitMQ messaging
3. **gRPC** - High-performance RPC
4. **QUIC** - UDP-based transport
5. **DDS** - CycloneDDS pub/sub

## Architecture

```
Docker Container (Server)     Docker Container (Client 1)     Docker Container (Client 2)
       ↓                              ↓                               ↓
   /shared_data              /shared_data                    /shared_data
       ↓                              ↓                               ↓
packet_logs_server.db   packet_logs_client_1.db      packet_logs_client_2.db
                                     ↓
                             Host: ./shared_data/
                                     ↓
                            experiment_gui.py
                         (PacketLogsTab reads all DBs)
```

## File Changes Summary

| File | Changes |
|------|---------|
| `Network_Simulation/experiment_gui.py` | Added PacketLogsTab import and integration |
| `Docker/docker-compose-unified-emotion.yml` | Added volume mounts + NODE_TYPE (6 places) |
| `Docker/docker-compose-unified-mentalstate.yml` | Added volume mounts + NODE_TYPE (6 places) |
| `Docker/docker-compose-unified-temperature.yml` | Added volume mounts + NODE_TYPE (6 places) |

**Total Services Updated:** 9 (3 servers + 6 clients)

## Success Indicators

✅ GUI shows "Packet Logs" tab
✅ Dropdown shows Server/Client 1/Client 2 options
✅ Protocol filter shows MQTT/AMQP/gRPC/QUIC/DDS
✅ Packets appear when FL training runs
✅ Auto-refresh updates table every 2 seconds
✅ Database path shown in status label

## Common Issues

**Issue:** "No database found"
**Fix:** Start Docker containers first, wait for databases to be created

**Issue:** No packets showing
**Fix:** Wait for FL training to start - packets only logged during communication

**Issue:** Permission denied
**Fix:** `chmod 777 shared_data/`

## Documentation

For detailed information, see:
- 📄 [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - Full integration details
- 📄 [UNIFIED_IMPLEMENTATION_COMPLETE.md](UNIFIED_IMPLEMENTATION_COMPLETE.md) - Unified server/client implementation
- 📄 [GUI/packet_logs_tab.py](GUI/packet_logs_tab.py) - Component source code
- 📄 [packet_logger.py](packet_logger.py) - Logging implementation

---

**Status:** ✅ READY TO USE
**Last Updated:** Now
**Integration:** COMPLETE
