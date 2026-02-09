# Unified FL Client Architecture - Multi-Protocol with RL Selection

## Overview

The unified FL client now supports **all 5 protocols** (MQTT, AMQP, gRPC, QUIC, DDS) with **RL-based dynamic protocol selection** for data transmission, while maintaining a consistent event-driven architecture.

## Architecture Design

### Two-Level Communication Pattern

```
┌─────────────────────────────────────────────────────┐
│         CONTROL SIGNALS (Server → Client)           │
│                                                       │
│  MQTT ONLY (Always, Event-Driven)                   │
│  ├─ fl/global_model      → Initialize/Update model  │
│  ├─ fl/training_config   → Update parameters        │
│  ├─ fl/start_training    → Begin training           │
│  ├─ fl/start_evaluation  → Begin evaluation         │
│  └─ fl/training_complete → Cleanup & exit           │
└─────────────────────────────────────────────────────┘
                            ↕
                   MQTT Callbacks Handler
                    (on_message routes)
                            ↕
┌─────────────────────────────────────────────────────┐
│      DATA TRANSMISSION (Client → Server)            │
│                                                       │
│  RL-SELECTED PROTOCOL (Dynamic)                     │
│  ├─ Model Updates    → Weights + Metrics            │
│  └─ Evaluation Metrics → Loss + Accuracy            │
│                                                       │
│  Protocol Options:                                  │
│  • MQTT  - Reliable queue, good for poor networks   │
│  • AMQP  - Advanced routing, fault-tolerant         │
│  • gRPC  - Fast, efficient serialization            │
│  • QUIC  - UDP-based, low latency                   │
│  • DDS   - Real-time, publish-subscribe             │
└─────────────────────────────────────────────────────┘
```

## Key Features

### 1. **Event-Driven Control Flow**
- Client waits indefinitely for server signals via `mqtt_client.loop_forever()`
- Server determines training schedule (when to train, when to evaluate)
- Client only acts on server signals - never initiates training independently

### 2. **RL-Based Protocol Selection**
- Each round, client uses RL (Q-Learning) to select best protocol
- Selection considers: CPU load, memory usage, network conditions, model size
- Independent selection for model updates and metrics
- Automatic fallback to MQTT if selected protocol unavailable

### 3. **Unified Interface**
- Protocol selection hidden from training code
- Training and evaluation logic unchanged
- Easy to add new protocols - just add `_send_via_<protocol>()` method

### 4. **Graceful Degradation**
- Missing protocol libraries don't crash client
- Falls back to MQTT for missing protocols
- RL modules optional - defaults to MQTT if not installed

## Implementation Details

### Control Signal Handlers (MQTT)

```python
def on_connect():
    # Subscribe to all control topics
    mqtt_client.subscribe([
        (TOPIC_GLOBAL_MODEL, 1),
        (TOPIC_TRAINING_CONFIG, 1),
        (TOPIC_START_TRAINING, 1),
        (TOPIC_START_EVALUATION, 1),
        (TOPIC_TRAINING_COMPLETE, 1)
    ])

def on_message(msg.topic):
    if msg.topic == TOPIC_GLOBAL_MODEL:
        handle_global_model()        # Set self.model
    elif msg.topic == TOPIC_START_TRAINING:
        if self.model is None:
            print("ERROR: waiting for global model")
            return
        train_local_model()           # Uses RL to select protocol
    elif msg.topic == TOPIC_START_EVALUATION:
        evaluate_model()              # Uses RL to select protocol
    elif msg.topic == TOPIC_TRAINING_COMPLETE:
        disconnect()                  # Clean shutdown
```

### Training with Protocol Selection

```python
def train_local_model():
    # ... training code ...
    
    # Select best protocol
    protocol = self.select_protocol()  # Uses RL Q-Learning
    
    # Send via selected protocol
    if protocol == 'mqtt':
        self._send_via_mqtt(update_message)
    elif protocol == 'amqp':
        self._send_via_amqp(update_message)
    elif protocol == 'grpc':
        self._send_via_grpc(update_message)
    elif protocol == 'quic':
        self._send_via_quic(update_message)
    elif protocol == 'dds':
        self._send_via_dds(update_message)
    else:
        self._send_via_mqtt(update_message)  # Fallback
```

### Protocol Selection Logic

```python
def select_protocol() -> str:
    if USE_RL_SELECTION and RL_available:
        # Measure environment
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        
        # Get RL state
        state = env_manager.get_current_state()
        
        # Query Q-Learning for best action
        protocol = rl_selector.select_protocol(state, training=True)
        
        return protocol
    else:
        return 'mqtt'  # Default fallback
```

## Protocol-Specific Implementation

### MQTT (_send_via_mqtt)
- Uses MQTT publish on `fl/client/{id}/update` topic
- QoS=1 for reliable delivery
- Good for: Unreliable networks, message persistence

### AMQP (_send_via_amqp)
- Connects to RabbitMQ, publishes to `fl_client_updates` exchange
- Direct routing with durable messages
- Good for: Complex routing, high reliability

### gRPC (_send_via_grpc)
- Calls `SendModelUpdate` RPC on server
- Efficient protobuf serialization
- Good for: LAN, high-speed connections

### QUIC (_send_via_quic)
- Connects asynchronously to QUIC server (UDP port 4433)
- Sends JSON-delimited messages
- Good for: Mobile, UDP-friendly networks

### DDS (_send_via_dds)
- Publish-subscribe on DDS topics (optional)
- Falls back to MQTT if unavailable
- Good for: Real-time systems, standardized middleware

## RL Protocol Selection Details

### State Space
The RL selector observes:
- **Resource Level**: LOW, MEDIUM, HIGH (based on CPU/memory)
- **Model Size**: SMALL, MEDIUM, LARGE (emotion=medium)
- **Network Conditions**: Implied from previous round performance

### Action Space
Five protocol choices:
- Action 0: MQTT
- Action 1: AMQP
- Action 2: gRPC
- Action 3: QUIC
- Action 4: DDS

### Reward Function
Rewards based on:
- **Success**: Communication successful (+1.0)
- **Failure**: Communication failed (-1.0)
- **Round time**: Faster = higher reward
- **Resource efficiency**: Lower CPU/memory = higher reward

### Q-Table Learning
- Initial Q-table: uniform random
- Update per round: Q(s,a) = Q(s,a) + α[R + γ·max(Q(s',a)) - Q(s,a)]
- Saved to `q_table_emotion_client_{id}.pkl` between runs
- Exploration: ε-greedy (explore with probability ε, exploit with 1-ε)

## Backward Compatibility

### Unified Client vs Single Protocol Clients

**Same:**
- Event-driven architecture
- Wait for server signals
- Training and evaluation logic
- Model initialization from server config
- Quantization support

**Different:**
- Single clients: Hardcoded to one protocol
- Unified client: Dynamic protocol selection per round

**Result:** Unified client can match or exceed performance of best single-protocol client

## Configuration

### Environment Variables

```bash
# RL Selection
export USE_RL_SELECTION=true          # Enable RL protocol selection
export CLIENT_ID=1                     # Client identifier
export NUM_CLIENTS=2                   # Total clients

# MQTT (Control signals, always used)
export MQTT_BROKER=localhost
export MQTT_PORT=1883

# Optional: AMQP
export AMQP_HOST=localhost
export AMQP_PORT=5672
export AMQP_USER=guest
export AMQP_PASSWORD=guest

# Optional: gRPC
export GRPC_HOST=localhost
export GRPC_PORT=50051

# Optional: QUIC
export QUIC_HOST=localhost
export QUIC_PORT=4433

# Training
export STEPS_PER_EPOCH=100
export VAL_STEPS=25
```

## Testing

### Test RL Protocol Selection Locally

```bash
# 1. Test Q-Learning selector standalone
python3 -m debugpy.adapter 49209 -- Client/Emotion_Recognition/rl_q_learning_selector.py

# 2. Test unified client with RL (local MQTT broker)
export MQTT_BROKER=localhost
export USE_RL_SELECTION=true
python3 Client/Emotion_Recognition/FL_Client_Unified.py

# 3. Test unified server
export MQTT_BROKER=localhost
python3 Server/Emotion_Recognition/FL_Server_Unified.py

# 4. Docker testing
docker-compose -f docker-compose-unified-emotion.yml up
```

## Error Handling & Fallbacks

| Error Scenario | Behavior |
|---|---|
| Protocol library unavailable | Falls back to MQTT |
| Protocol send fails | Raises exception, triggers retry |
| RL module unavailable | Defaults to MQTT |
| RL selector error | Prints warning, uses MQTT |
| Network timeout | Protocol-specific timeout handling |
| Model not initialized | Rejects training signal with clear error |

## Performance Considerations

### Memory
- Each protocol connection is on-demand
- MQTT connection persistent (for control)
- Other protocols connected only when needed for data

### Latency
- MQTT: ~50-100ms (broker overhead)
- AMQP: ~50-100ms (similar to MQTT)
- gRPC: ~10-50ms (more efficient)
- QUIC: ~5-30ms (UDP-based, lowest latency)
- DDS: ~1-10ms (real-time optimized)

### Throughput
- For 1MB weights: QUIC fastest, MQTT slowest
- Compression helps all protocols equally
- Protocol selection crucial for large models

## Future Enhancements

1. **Per-Round Feedback**: Reward RL based on actual round metrics
2. **Adaptive Learning Rate**: Adjust RL exploration vs exploitation
3. **Ensemble Methods**: Use multiple protocols for redundancy
4. **Protocol-Specific Compression**: Optimize compression per protocol
5. **Dynamic Server-Side Protocol**: Server also adapts protocol
6. **Multi-Path Transmission**: Split updates across protocols

## Architecture Consistency with Single-Protocol Clients

| Aspect | MQTT Client | Unified Client |
|---|---|---|
| Control signals | MQTT | MQTT ✓ |
| Wait for server | Yes | Yes ✓ |
| Async callbacks | Yes | Yes ✓ |
| Model initialization | Server config | Server config ✓ |
| Training trigger | Server signal | Server signal ✓ |
| Evaluation trigger | Server signal | Server signal ✓ |
| Data transmission | MQTT | RL-selected ✓ |
| Graceful shutdown | Yes | Yes ✓ |

**Conclusion:** Architecture is consistent and backward-compatible. The only difference is data transmission protocol selection, which is the intended design.
