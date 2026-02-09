# Protocol Implementation Details

## Summary of Protocol-Specific Methods in FL_Client_Unified.py

### MQTT Protocol

#### _send_via_mqtt(message: dict)
- **Purpose**: Send model weights and metrics via MQTT
- **Server Endpoint**: MQTT broker at MQTT_BROKER:MQTT_PORT
- **Message Routing**: 
  - Model update: `fl/client/{client_id}/update`
  - Metrics: `fl/client/{client_id}/metrics`
- **QoS**: 1 (Acknowledged delivery)
- **Payload**: JSON string with weights (base64 encoded)
- **Timeout**: 30 seconds
- **Fallback**: N/A (default protocol)

#### _send_metrics_via_mqtt(message: dict)
- Sends metrics (loss, accuracy) via MQTT
- Smaller payload than model updates
- Same timeout and QoS as model updates

---

### AMQP Protocol (RabbitMQ)

#### _send_via_amqp(message: dict)
- **Purpose**: Send model via AMQP message broker
- **Server Endpoint**: AMQP_HOST:AMQP_PORT (default: localhost:5672)
- **Connection**:
  - Username: AMQP_USER (default: guest)
  - Password: AMQP_PASSWORD (default: guest)
  - Connection timeout: 3 attempts, 2s delay
- **Message Routing**:
  - Exchange: `fl_client_updates` (type: direct)
  - Routing key: `client_{client_id}_update`
- **Delivery**: Persistent (survives broker restart)
- **Connection Cleanup**: Closes after message sent

#### _send_metrics_via_amqp(message: dict)
- Sends evaluation metrics via same broker
- Routing key: `client_{client_id}_metrics`
- Same persistence and cleanup as model updates

---

### gRPC Protocol

#### _send_via_grpc(message: dict)
- **Purpose**: Send model via gRPC remote procedure call
- **Server Endpoint**: GRPC_HOST:GRPC_PORT (default: localhost:50051)
- **Message Size**: Configured for 100MB max (for large models)
- **RPC Method**: `SendModelUpdate(ModelUpdate)`
- **Message Format**:
  ```protobuf
  message ModelUpdate {
    int32 client_id = 1;
    int32 round = 2;
    string weights = 3;           // base64 encoded weights
    int32 num_samples = 4;
    string metrics = 5;           // JSON metrics
  }
  ```
- **Response**: `AckResponse` with success flag
- **Error Handling**: Raises exception if server responds with success=false

#### _send_metrics_via_grpc(message: dict)
- **RPC Method**: `SendMetrics(Metrics)`
- **Message Format**:
  ```protobuf
  message Metrics {
    int32 client_id = 1;
    int32 round = 2;
    float loss = 3;
    float accuracy = 4;
    int32 num_samples = 5;
  }
  ```

---

### QUIC Protocol

#### _send_via_quic(message: dict)
- **Purpose**: Send model via QUIC (UDP-based, low latency)
- **Server Endpoint**: QUIC_HOST:QUIC_PORT (default: localhost:4433)
- **Protocol**: QUIC with UDP
- **Connection Type**: Asynchronous via asyncio
- **Message Format**: JSON string, newline-delimited
- **Certificate**: Unverified (is_client=True, verify_mode='unverified')
- **Implementation**: 
  1. Creates QUIC configuration
  2. Connects to server
  3. Gets next available stream ID
  4. Sends data on stream (JSON + newline)
  5. Waits briefly for response
- **Async Method**: `_quic_send_data(host, port, payload, msg_type)`

#### _send_metrics_via_quic(message: dict)
- Same implementation as model send
- Message type: 'metrics' instead of 'model_update'
- Payload contains loss, accuracy, num_samples

#### _quic_send_data(host, port, payload, msg_type) [ASYNC]
- **Parameters**:
  - `host`: QUIC server hostname
  - `port`: QUIC server port (UDP)
  - `payload`: JSON string message
  - `msg_type`: 'model_update' or 'metrics'
- **Operation**:
  1. Creates QUIC config (unverified client)
  2. Initiates async connection
  3. Gets next stream ID
  4. Sends payload + newline on stream
  5. Connection closes automatically
- **Error Handling**: Catches connection errors, prints to console

---

### DDS Protocol (Placeholder)

#### _send_via_dds(message: dict)
- **Purpose**: Send model via DDS real-time middleware
- **Status**: Placeholder implementation
- **Current Behavior**:
  - Attempts to import `ddspython` (optional dependency)
  - If unavailable: Falls back to MQTT with warning
  - If available: Prints placeholder message
- **Future Enhancement**: Implement actual DDS publish on topic
- **Topics** (planned):
  - Model updates: `fl/client/{id}/model_update`
  - Metrics: `fl/client/{id}/metrics`

#### _send_metrics_via_dds(message: dict)
- Same pattern as model send
- Falls back to MQTT if DDS unavailable

---

## Protocol Selection Logic (select_protocol)

### Algorithm
1. **Check if RL is enabled**: `USE_RL_SELECTION` env var
2. **If enabled**:
   - Measure CPU percentage: `psutil.cpu_percent(interval=0.1)`
   - Measure memory percentage: `psutil.virtual_memory().percent`
   - Determine resource level: `env_manager.detect_resource_level()`
   - Get RL state: `env_manager.get_current_state()`
   - Query Q-Learning: `rl_selector.select_protocol(state, training=True)`
3. **Output Information**:
   - CPU and memory usage
   - Current state
   - Selected protocol (MQTT, AMQP, gRPC, QUIC, DDS)
   - Current round number
4. **If RL disabled or error**: Default to MQTT

### Return Value
- String: 'mqtt' | 'amqp' | 'grpc' | 'quic' | 'dds'
- Never returns None; always has fallback to MQTT

---

## Error Handling Strategy

### Per-Protocol Error Handling

#### MQTT
- Checks publish return code
- Raises exception if rc != MQTT_ERR_SUCCESS
- Timeout: 30 seconds per message

#### AMQP
- Connection attempts: 3 with 2s delay between
- Catches pika exceptions
- Raises on any error
- Auto-closes connection after each send

#### gRPC
- Catches grpc.RpcError exceptions
- Max message size: 100MB
- Checks response.success flag
- Closes channel after each send

#### QUIC
- Catches asyncio and aioquic exceptions
- Implicit timeout: 0.5s wait after send
- Connection auto-closes on error

#### DDS
- Catches ImportError for missing dependency
- Catches ddspython exceptions
- Falls back to MQTT on any error

### Global Error Handling

In `train_local_model()` and `evaluate_model()`:
```python
try:
    # Send via selected protocol
    ...
except Exception as e:
    print(f"ERROR sending via {protocol}: {e}")
    raise  # Propagate to caller
```

---

## Message Formats

### Model Update Message
```python
{
    "client_id": 1,
    "round": 1,
    "weights": "AAECAw==...",  # base64 encoded pickle bytes
    "num_samples": 5000,
    "metrics": {
        "loss": 0.4321,
        "accuracy": 0.8765,
        "val_loss": 0.4567,
        "val_accuracy": 0.8654
    },
    "protocol": "grpc"  # Which protocol was used
}
```

### Metrics Message
```python
{
    "client_id": 1,
    "round": 1,
    "num_samples": 5000,
    "loss": 0.4567,
    "accuracy": 0.8654,
    "protocol": "mqtt"  # Which protocol was used
}
```

---

## Protocol Availability Check

Each protocol method gracefully handles unavailable dependencies:

```python
if pika is None:
    raise ImportError("pika module not available for AMQP")

if grpc is None or federated_learning_pb2 is None:
    raise ImportError("grpc modules not available for gRPC")

if asyncio is None or connect is None:
    raise ImportError("aioquic module not available for QUIC")

# DDS tries import, falls back to MQTT if unavailable
try:
    import ddspython
except ImportError:
    # Fall back to MQTT
```

---

## Configuration Requirements

### Minimal (MQTT only)
```bash
export MQTT_BROKER=localhost
export MQTT_PORT=1883
```

### Full (All protocols)
```bash
# MQTT (Control + Data)
export MQTT_BROKER=localhost
export MQTT_PORT=1883

# AMQP
export AMQP_HOST=localhost
export AMQP_PORT=5672
export AMQP_USER=guest
export AMQP_PASSWORD=guest

# gRPC
export GRPC_HOST=localhost
export GRPC_PORT=50051

# QUIC
export QUIC_HOST=localhost
export QUIC_PORT=4433

# RL Selection
export USE_RL_SELECTION=true
```

---

## Protocol Performance Characteristics

| Protocol | Latency | Throughput | Overhead | Best For |
|---|---|---|---|---|
| MQTT | 50-100ms | Slow | High | Unreliable networks |
| AMQP | 50-100ms | Slow | High | Reliable queuing |
| gRPC | 10-50ms | Fast | Low | LAN environments |
| QUIC | 5-30ms | Very Fast | Very Low | Mobile, UDP-friendly |
| DDS | 1-10ms | Very Fast | Very Low | Real-time systems |

---

## Testing Protocol-Specific Sends

### Unit Test Example
```python
# Test MQTT send
msg = {"client_id": 1, "round": 1, "weights": "abc...", "metrics": {...}}
client._send_via_mqtt(msg)

# Test gRPC send
client._send_via_grpc(msg)

# Test QUIC send
asyncio.run(client._send_via_quic(msg))

# Test protocol selection
protocol = client.select_protocol()
print(f"Selected: {protocol}")
```

### Docker Test
```bash
docker-compose -f docker-compose-unified-emotion.yml up

# Check logs
docker logs fl-client-unified-emotion-1
# Should see: [RL Protocol Selection] Selected Protocol: ...
```

---

## Future Enhancement Ideas

1. **Adaptive Timeouts**: Set protocol-specific timeouts based on network latency
2. **Compression**: Use protocol-specific compression (gRPC has native compression)
3. **Retry Logic**: Implement exponential backoff for failed sends
4. **Metrics Tracking**: Record which protocol was used, latency, success rate
5. **Dynamic Weights**: Weight RL rewards by protocol suitability
6. **Multi-Send**: Send via multiple protocols for redundancy
7. **DDS Implementation**: Full DDS sender using cyclonedds-python
8. **Protocol Ranking**: Track historical performance, use for RL reward function
