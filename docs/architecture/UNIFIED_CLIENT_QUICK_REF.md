# Unified FL Client - Quick Reference

## What Changed?

The unified client **now supports all 5 protocols** for data transmission while using MQTT for control signals.

### Before (Limited)
```
Client → MQTT → Server (always)
```

### After (Intelligent)
```
CONTROL SIGNALS: Client ←→ MQTT ←→ Server (always)
                           ↓
                    RL Selector
                           ↓
DATA TRANSMISSION:  Client → [Protocol] → Server
                    ├─ MQTT  (RL selects)
                    ├─ AMQP
                    ├─ gRPC
                    ├─ QUIC
                    └─ DDS
```

## Key Architectural Principles

### 1. Two-Level Communication
- **MQTT (Control)**: Always used for server signals (event-driven)
  - Server tells client when to train, evaluate, update config
  - Client waits for MQTT messages, never initiates independently
  
- **RL-Selected (Data)**: Dynamic protocol per round for model/metrics transfer
  - Client uses Q-Learning to select best protocol
  - Selection based on: CPU, memory, network conditions, model size
  - Gracefully falls back to MQTT if protocol unavailable

### 2. Similar to Single Protocol Clients
- Architecture is **identical** to MQTT, AMQP, gRPC, QUIC, DDS single clients
- **Only difference**: Protocol selection is dynamic (RL) instead of fixed
- Maintains event-driven pattern, waits for server signals
- Handles model initialization, training, evaluation exactly the same

### 3. Event-Driven (Not Loop-Based)
- Client calls `mqtt_client.loop_forever()` - blocking, event-driven
- All actions triggered by server MQTT messages via callbacks:
  - `on_connect()` → subscribe, register
  - `on_message()` → route to handler
  - `handle_global_model()` → initialize model
  - `handle_start_training()` → select protocol, train, send
  - `handle_start_evaluation()` → evaluate, send metrics
  - `handle_training_complete()` → cleanup

## Protocol Selection in Action

### Each Training Round

```python
1. Server sends "start_training" via MQTT
   ↓
2. Client's on_message() triggers handle_start_training()
   ↓
3. Client trains local model
   ↓
4. Client calls select_protocol()
   - Measures CPU: 45%, Memory: 52%
   - State: MEDIUM resources, MEDIUM model
   - Q-Learning recommends: gRPC
   ↓
5. Client sends model update via gRPC
   ↓
6. Server processes update, aggregates, sends global model
   ↓
7. Next round repeats (might select MQTT, AMQP, QUIC, etc.)
```

## Code Organization

### File: `FL_Client_Unified.py` (907 lines)

**Class Structure:**
```python
class UnifiedFLClient_Emotion:
    
    # Initialization
    __init__()                          # Setup RL selector, MQTT client
    
    # MQTT Control Signal Handlers
    on_connect()                        # Subscribe, register
    on_message()                        # Route to specific handler
    on_disconnect()                     # Clean shutdown
    handle_global_model()               # Receive/initialize model
    handle_training_config()            # Update parameters
    handle_start_training()             # Train (with protocol selection)
    handle_start_evaluation()           # Evaluate (with protocol selection)
    handle_training_complete()          # Exit cleanly
    
    # Model Operations
    build_model_from_config()           # Build from server config
    train_local_model()                 # Train + select protocol + send
    evaluate_model()                    # Evaluate + select protocol + send
    
    # Protocol Selection
    select_protocol()                   # RL-based selection
    
    # Protocol-Specific Senders (Data Transmission)
    _send_via_mqtt()                    # MQTT update
    _send_metrics_via_mqtt()            # MQTT metrics
    _send_via_amqp()                    # AMQP update
    _send_metrics_via_amqp()            # AMQP metrics
    _send_via_grpc()                    # gRPC update
    _send_metrics_via_grpc()            # gRPC metrics
    _send_via_quic()                    # QUIC update
    _send_metrics_via_quic()            # QUIC metrics
    _send_via_dds()                     # DDS update
    _send_metrics_via_dds()             # DDS metrics
    _quic_send_data()                   # Async QUIC helper
    
    # Utilities
    serialize_weights()                 # Pickle + Base64 encode
    deserialize_weights()               # Base64 decode + Unpickle
    start()                             # Main loop (mqtt_client.loop_forever)
```

## Protocol Methods Implementation

### MQTT (_send_via_mqtt)
```python
def _send_via_mqtt(self, message: dict):
    # Publish to fl/client/{id}/update
    # QoS=1 for reliable delivery
    mqtt_client.publish(TOPIC_CLIENT_UPDATE, json.dumps(message), qos=1)
```

### AMQP (_send_via_amqp)
```python
def _send_via_amqp(self, message: dict):
    # Connect to RabbitMQ
    # Publish to fl_client_updates exchange
    # Graceful fallback if unavailable
```

### gRPC (_send_via_grpc)
```python
def _send_via_grpc(self, message: dict):
    # Connect to gRPC server
    # Call SendModelUpdate RPC
    # Use protobuf serialization
```

### QUIC (_send_via_quic)
```python
async def _send_via_quic(self, message: dict):
    # Async connect via QUIC (UDP)
    # Send JSON-delimited message
    # Low latency UDP-based transfer
```

### DDS (_send_via_dds)
```python
def _send_via_dds(self, message: dict):
    # Publish to DDS topic
    # Real-time middleware (optional)
    # Falls back to MQTT if unavailable
```

## Control Flow Example (Round 1)

```
Initial State:
- Client connects to MQTT broker
- Subscribes: fl/global_model, fl/training_config, fl/start_training, 
              fl/start_evaluation, fl/training_complete
- Publishes: "fl/client_register" with client_id

Server Sends (MQTT):
- Topic: fl/global_model
- Message: {round: 0, weights: [...], model_config: {...}}
- Client handler: handle_global_model()
  → Creates model
  → Sets weights
  → self.model is now initialized

Server Sends (MQTT):
- Topic: fl/training_config
- Message: {batch_size: 32, local_epochs: 20}
- Client handler: handle_training_config()
  → Updates self.training_config

Server Sends (MQTT):
- Topic: fl/start_training
- Message: {round: 1}
- Client handler: handle_start_training()
  → Check: self.model is not None ✓
  → Train on local data
  → Select protocol (e.g., gRPC)
  → Send weights via gRPC
  → Client sends "model_update" message

Server Sends (MQTT):
- Topic: fl/start_evaluation
- Message: {round: 1}
- Client handler: handle_start_evaluation()
  → Evaluate on validation data
  → Select protocol (e.g., MQTT this time)
  → Send metrics via MQTT

Server Sends (MQTT):
- Topic: fl/training_complete
- Message: {}
- Client handler: handle_training_complete()
  → Print completion message
  → Disconnect from MQTT
  → Exit
```

## Comparison: Unified vs Single-Protocol Clients

### Unified Client Architecture
```python
# Control: MQTT (always)
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.loop_forever()  # Blocks, event-driven

# Data: RL-Selected Protocol
def train_local_model():
    protocol = select_protocol()  # RL
    if protocol == 'mqtt':
        _send_via_mqtt(msg)
    elif protocol == 'amqp':
        _send_via_amqp(msg)
    # ... etc
```

### MQTT Single-Protocol Client
```python
# All communication: MQTT
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.loop_forever()  # Blocks, event-driven

def train_local_model():
    mqtt_client.publish(TOPIC_CLIENT_UPDATE, msg)
```

### gRPC Single-Protocol Client
```python
# Control: gRPC polling (active)
def run():
    while True:
        model = stub.GetGlobalModel()  # Polling
        if model.available:
            train_local_model()
            stub.SendModelUpdate(update)
            evaluate_model()
            stub.SendMetrics(metrics)

# Data: gRPC (always)
stub.SendModelUpdate(weights)
stub.SendMetrics(accuracy)
```

## Key Differences

| Aspect | MQTT Client | gRPC Client | Unified Client |
|---|---|---|---|
| Control Method | MQTT events | gRPC polling | MQTT events ✓ |
| Data Transport | MQTT | gRPC | RL-selected ✓ |
| Architecture | Event-driven | Polling | Event-driven ✓ |
| Protocol Count | 1 | 1 | 5 (dynamic) ✓ |
| RL Selection | No | No | Yes ✓ |
| Backward Compatible | Yes | Yes | Yes ✓ |

## Configuration for Docker

### Environment Variables
```yaml
fl-client-unified-emotion-1:
  environment:
    CLIENT_ID: 1
    NUM_CLIENTS: 2
    USE_RL_SELECTION: "true"
    MQTT_BROKER: fl-mqtt-broker
    MQTT_PORT: 1883
    AMQP_HOST: fl-amqp-broker
    AMQP_PORT: 5672
    GRPC_HOST: fl-server-unified-emotion
    GRPC_PORT: 50051
    QUIC_HOST: fl-server-unified-emotion
    QUIC_PORT: 4433
```

## Testing Locally

### 1. Test RL Selector Standalone
```bash
# Just test the Q-Learning algorithm
python3 Client/rl_q_learning_selector.py
```

### 2. Test Unified Client (needs MQTT broker)
```bash
# Start MQTT broker
docker run -d -p 1883:1883 eclipse-mosquitto

# Set environment
export MQTT_BROKER=localhost
export USE_RL_SELECTION=true

# Run client (will wait for server)
python3 Client/Emotion_Recognition/FL_Client_Unified.py
```

### 3. Test Full System
```bash
docker-compose -f docker-compose-unified-emotion.yml up
```

## Common Issues & Troubleshooting

### Issue: Client not starting training
**Solution:** Server hasn't sent `fl/start_training` signal yet. Check server logs.

### Issue: "Model not initialized yet"
**Solution:** Client received training signal before global model. Server should send model first.

### Issue: Protocol timeout
**Solution:** Selected protocol not available. Check that gRPC, AMQP servers are running.

### Issue: Large model update fails
**Solution:** Increase gRPC message size limit in config, or RL will select MQTT.

### Issue: RL selector not loaded
**Solution:** `rl_q_learning_selector.py` in parent directory. Defaults to MQTT if missing.

## Files Modified

- ✅ `FL_Client_Unified.py` (907 lines) - Complete refactor with all 5 protocols
- ✅ `UNIFIED_CLIENT_ARCHITECTURE.md` - Comprehensive design documentation
- ✅ `UNIFIED_CLIENT_QUICK_REF.md` - This file

## Summary

**The unified client is now a true multi-protocol federation client that:**

1. ✅ Uses MQTT for control signals (server orchestration)
2. ✅ Dynamically selects protocols for data transmission (RL-based)
3. ✅ Supports all 5 protocols: MQTT, AMQP, gRPC, QUIC, DDS
4. ✅ Maintains event-driven architecture (waits for server signals)
5. ✅ Matches behavior of single-protocol clients (just smarter protocol selection)
6. ✅ Gracefully degrades if protocols unavailable (falls back to MQTT)
7. ✅ Can be configured via environment variables
8. ✅ Compatible with existing Docker compose setup
