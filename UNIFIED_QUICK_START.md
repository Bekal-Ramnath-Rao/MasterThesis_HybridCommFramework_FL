# Unified FL Scenario - Quick Start Guide

## 🚀 What's New

This unified implementation enables **all 5 communication protocols** (MQTT, AMQP, gRPC, QUIC, DDS) to work simultaneously, with clients using RL-based protocol selection for optimal performance.

## ✅ What's Been Implemented

### 1. Complete Protocol Support
- **MQTT** ✓ - Broker-based pub/sub
- **AMQP** ✓ - RabbitMQ message queue
- **gRPC** ✓ - High-performance RPC
- **QUIC** ✓ - UDP-based low-latency
- **DDS** ✓ - Data Distribution Service with fallback

### 2. Packet Logger Integration
- Separate databases for server and clients
- Real-time packet tracking
- Protocol-specific logging
- Volume-mounted for GUI access

### 3. RL Protocol Selection
- Q-Learning based selection
- Dynamic protocol switching per round
- Reward-based optimization
- State management (network, resource, model size)

## 📂 File Structure

```
MasterThesis_HybridCommFramework_FL/
├── Server/Emotion_Recognition/
│   ├── FL_Server_Unified.py          ← Complete unified server
│   └── FL_Server_Unified_backup.py   ← Original backup
├── Client/Emotion_Recognition/
│   └── FL_Client_Unified.py          ← Complete unified client with packet logging
├── Protocols/
│   ├── federated_learning.proto      ← Updated gRPC definitions
│   ├── federated_learning_pb2.py     ← Regenerated
│   └── federated_learning_pb2_grpc.py ← Regenerated
├── GUI/
│   └── packet_logs_tab.py            ← New packet visualization tab
├── shared_data/                      ← Packet log databases (volume mounted)
│   ├── packet_logs_server.db
│   ├── packet_logs_client_1.db
│   └── packet_logs_client_2.db
├── packet_logger.py                  ← Fixed database path logic
├── docker-compose-unified.yml        ← Docker configuration
└── UNIFIED_IMPLEMENTATION_COMPLETE.md ← Full documentation

```

## 🛠️ Setup Instructions

### Step 1: Prepare Environment

```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL

# Ensure shared_data directory exists
mkdir -p shared_data

# Ensure results directory exists
mkdir -p results
```

### Step 2: Build Docker Images

```bash
# Build server image
docker build -t fl-server-unified -f Dockerfile.server .

# Build client image
docker build -t fl-client-unified -f Dockerfile.client .
```

### Step 3: Start Services

```bash
# Start all services with docker-compose
docker-compose -f docker-compose-unified.yml up --build
```

Or manually:

```bash
# Start MQTT broker
docker run -d --name mqtt-broker -p 1883:1883 eclipse-mosquitto

# Start AMQP broker
docker run -d --name amqp-broker -p 5672:5672 -p 15672:15672 rabbitmq:3-management

# Start FL server
docker run --name fl-server-unified \
  -e NODE_TYPE=server \
  -e NUM_CLIENTS=2 \
  -v $(pwd)/shared_data:/shared_data \
  -p 50051:50051 -p 4433:4433 \
  --link mqtt-broker --link amqp-broker \
  fl-server-unified

# Start FL clients
docker run --name fl-client-1 \
  -e NODE_TYPE=client \
  -e CLIENT_ID=1 \
  -e USE_RL_SELECTION=true \
  -v $(pwd)/shared_data:/shared_data \
  --link mqtt-broker --link amqp-broker --link fl-server-unified \
  --gpus device=0 \
  fl-client-unified
```

## 📊 Monitoring Packet Logs

### Option 1: GUI (Recommended)

```bash
# Run the GUI
python GUI/main_gui.py

# Navigate to "Packet Logs" tab
# Select "Server" or "Client 1", "Client 2", etc.
# Choose protocol filter
# Watch real-time updates!
```

### Option 2: Command Line

```bash
# View server packets
sqlite3 shared_data/packet_logs_server.db "SELECT * FROM sent_packets ORDER BY id DESC LIMIT 10;"

# View client packets
sqlite3 shared_data/packet_logs_client_1.db "SELECT * FROM received_packets ORDER BY id DESC LIMIT 10;"

# Count packets by protocol
sqlite3 shared_data/packet_logs_server.db "SELECT protocol, COUNT(*) FROM sent_packets GROUP BY protocol;"

# View specific round
sqlite3 shared_data/packet_logs_client_1.db "SELECT * FROM sent_packets WHERE round = 1;"
```

### Option 3: Python Script

```python
import sqlite3

def view_packets(db_path="shared_data/packet_logs_server.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query sent packets
    cursor.execute("SELECT * FROM sent_packets ORDER BY id DESC LIMIT 10")
    print("Sent Packets:")
    for row in cursor.fetchall():
        print(f"  {row}")
    
    # Query by protocol
    cursor.execute("SELECT protocol, COUNT(*), AVG(packet_size) FROM sent_packets GROUP BY protocol")
    print("\nProtocol Statistics:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} packets, avg size: {row[2]:.2f} bytes")
    
    conn.close()

view_packets()
```

## 🔍 Verifying Communication

### 1. Check Server Logs

```bash
docker logs fl-server-unified | grep "Protocol Selection"
docker logs fl-server-unified | grep "Received update"
docker logs fl-server-unified | grep "MQTT\|AMQP\|gRPC\|QUIC\|DDS"
```

Expected output:
```
[MQTT] Server started
[AMQP] Server started
[gRPC] Server started on port 50051
[QUIC] Server started on 0.0.0.0:4433
[DDS] Server started on domain 0
...
[MQTT] Client 1 registered
[gRPC] Received update from client 1
[QUIC] Received metrics from client 1
```

### 2. Check Client Logs

```bash
docker logs fl-client-1-unified | grep "RL Protocol Selection"
docker logs fl-client-1-unified | grep "selected protocol"
```

Expected output:
```
[RL Protocol Selection]
  CPU: 45.2%, Memory: 32.1%
  State: (medium, normal_network, low_resources)
  Selected Protocol: GRPC
  Round: 1

Client 1 sending via gRPC - size: 12.45 MB
Client 1 sent model update for round 1 via gRPC
```

## 🧪 Testing Each Protocol

### Test MQTT
```bash
# In client logs, look for:
docker logs fl-client-1-unified | grep "MQTT"
# Expected: Registration, control signals
```

### Test AMQP
```bash
# Set RL to prefer AMQP temporarily by modifying reward function
# Or check logs for natural AMQP selection:
docker logs fl-client-1-unified | grep "AMQP"
```

### Test gRPC
```bash
# Check gRPC communications:
docker logs fl-client-1-unified | grep "gRPC"
docker logs fl-server-unified | grep "gRPC"
```

### Test QUIC
```bash
# Check QUIC communications:
docker logs fl-client-1-unified | grep "QUIC"
```

### Test DDS
```bash
# Check DDS communications (or fallback to MQTT):
docker logs fl-client-1-unified | grep "DDS"
```

## 📈 Viewing Results

### Training Metrics

```bash
# Check results directory
ls -lh results/

# View JSON results
cat results/unified_results_*.json | jq
```

### Packet Logs

```bash
# Server packet stats
sqlite3 shared_data/packet_logs_server.db "
  SELECT 
    protocol,
    COUNT(*) as count,
    SUM(packet_size) as total_bytes,
    AVG(packet_size) as avg_bytes
  FROM sent_packets 
  GROUP BY protocol;
"

# Client packet stats
sqlite3 shared_data/packet_logs_client_1.db "
  SELECT 
    protocol,
    COUNT(*) as count,
    SUM(packet_size) as total_bytes,
    AVG(packet_size) as avg_bytes
  FROM sent_packets 
  GROUP BY protocol;
"
```

## 🐛 Troubleshooting

### Database Not Updating

**Problem:** GUI shows empty tables or old data

**Solutions:**
1. Check file permissions:
   ```bash
   ls -l shared_data/
   chmod 666 shared_data/*.db
   ```

2. Verify volume mount:
   ```bash
   docker exec fl-server-unified ls -l /shared_data/
   ```

3. Check NODE_TYPE environment variable:
   ```bash
   docker exec fl-server-unified printenv | grep NODE_TYPE
   ```

### Protocol Not Available

**Problem:** "WARNING: DDS not available, falling back to MQTT"

**Solutions:**
1. Install missing dependencies in Docker image
2. Check Dockerfile includes required packages
3. Verify library imports in container:
   ```bash
   docker exec fl-client-1-unified python -c "from cyclonedds import *"
   ```

### RL Always Selects Same Protocol

**Problem:** RL selector always picks MQTT

**Solutions:**
1. Check RL reward function in `rl_q_learning_selector.py`
2. Adjust exploration rate (epsilon)
3. Modify reward calculation to favor exploration
4. Check environment state detection

### Packet Logs Missing

**Problem:** Database exists but no packets logged

**Solutions:**
1. Check packet_logger initialization:
   ```bash
   docker logs fl-server-unified | grep "PacketLogger"
   ```

2. Verify database path:
   ```bash
   docker exec fl-server-unified python -c "from packet_logger import DB_PATH; print(DB_PATH)"
   ```

3. Check write permissions:
   ```bash
   docker exec fl-server-unified touch /shared_data/test.txt
   ```

## 🎯 Next Steps

1. **Run Experiments:**
   - Compare RL-selected vs fixed protocol
   - Measure protocol distribution across rounds
   - Analyze packet sizes and latencies

2. **Enhance GUI:**
   - Add protocol distribution charts
   - Add latency visualization
   - Add real-time network metrics

3. **Optimize RL:**
   - Tune reward function
   - Add more environmental factors
   - Implement adaptive exploration

4. **Performance Testing:**
   - Test with poor network conditions
   - Test with resource constraints
   - Test with large model sizes

## 📝 Important Notes

1. **MQTT is always used for control signals** (registration, training start/stop, etc.)
2. **RL selects protocol only for data transmission** (model updates, metrics)
3. **Packet logs are separate per node** (server vs each client)
4. **Volume mounting is essential** for GUI to access packet logs
5. **All protocols must be available** for full RL selection (graceful fallback if not)

## 🎉 Summary

✅ **All 5 protocols implemented and tested**
✅ **Packet logging integrated throughout**
✅ **GUI ready for visualization**
✅ **RL protocol selection working**
✅ **Docker compose configured**
✅ **Documentation complete**

**The unified FL system is now fully operational!**

For detailed implementation information, see [UNIFIED_IMPLEMENTATION_COMPLETE.md](UNIFIED_IMPLEMENTATION_COMPLETE.md)
