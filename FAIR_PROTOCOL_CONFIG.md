# Fair Protocol Configuration Standard

## Purpose
This document defines standardized configuration parameters for **fair, unbiased comparison** of all communication protocols (MQTT, AMQP, gRPC, QUIC, DDS) in the Federated Learning evaluation.

## Date: 2026-02-09
## Status: MANDATORY for all protocol implementations

---

## 1. MESSAGE SIZE LIMITS

**Standard: 128 MB** (based on AMQP's RabbitMQ default, which is the most permissive)

| Protocol | Configuration | Value |
|----------|--------------|-------|
| **MQTT** | `_max_packet_size` | 128 * 1024 * 1024 (128 MB) |
| **AMQP** | RabbitMQ default | 128 MB (no explicit config needed) |
| **gRPC** | `max_send_message_length`<br>`max_receive_message_length` | 128 * 1024 * 1024 (128 MB) |
| **QUIC** | `max_stream_data`<br>`max_data` | 128 MB per stream<br>256 MB total connection |
| **DDS** | Chunking with buffer | KeepLast(2048) × 64KB = 128 MB |

**Rationale:** Using maximum supported limit eliminates artificial constraints that could bias results. AMQP's 128 MB default is chosen as the common ceiling.

---

## 2. QUEUE/BUFFER MANAGEMENT

**Standard: Limited queueing with 1000 message cap**

| Protocol | Configuration | Value |
|----------|--------------|-------|
| **MQTT** | `max_queued_messages_set` | 1000 (was unlimited/0) |
| **AMQP** | RabbitMQ queue length | 1000 (default) |
| **gRPC** | Streaming flow control | Dynamic, effectively ~1000 |
| **QUIC** | Connection buffer | 256 MB (2× max_stream) |
| **DDS** | History policy | KeepLast(1000) for updates |

**Rationale:** Unlimited queuing (MQTT's previous setting) masks network issues and creates unfair advantage. Standardizing to 1000 messages ensures comparable buffering behavior.

---

## 3. TIMEOUT/HEARTBEAT SETTINGS

**Standard: 600 seconds (10 minutes) for very_poor network scenarios**

| Protocol | Configuration | Value |
|----------|--------------|-------|
| **MQTT** | `keepalive` | 600 seconds |
| **AMQP** | `heartbeat` | 600 seconds |
| **gRPC** | `keepalive_time_ms` | 600000 ms (600s) |
| **QUIC** | `idle_timeout` | 600.0 seconds |
| **DDS** | `max_blocking_time` | 600 seconds (data)<br>60 seconds (control) |

**Rationale:** Very poor network conditions (high packet loss, long interruptions) require generous timeouts. 10 minutes allows protocols to survive temporary outages while still detecting true failures. Control messages remain at 60s for responsiveness.

---

## 4. CHUNKING IMPLEMENTATION

**Standard: All protocols implement 64 KB chunking with consistent reassembly**

| Protocol | Chunk Size | Buffer Size | Implementation |
|----------|-----------|-------------|----------------|
| **MQTT** | 64 KB | 2048 chunks (128 MB) | NEW: Add chunking |
| **AMQP** | 64 KB | 2048 chunks (128 MB) | NEW: Add chunking |
| **gRPC** | 64 KB | 2048 chunks (128 MB) | NEW: Add chunking |
| **QUIC** | 64 KB | 2048 chunks (128 MB) | NEW: Add chunking |
| **DDS** | 64 KB | 2048 chunks (128 MB) | EXISTING: Already implemented |

**Rationale:** 
- **Fair comparison:** DDS already uses chunking for reliability. Adding chunking to all protocols ensures equal footing.
- **Better reliability:** 64 KB chunks work better in poor network conditions than large monolithic messages.
- **Consistent behavior:** All protocols now handle large models identically.

---

## 5. CONSISTENCY BETWEEN STANDALONE AND UNIFIED

**CRITICAL REQUIREMENT:** Configuration must be **identical** between:
- Standalone protocol implementations (`FL_Client_MQTT.py`, `FL_Server_MQTT.py`, etc.)
- Unified RL-based implementation (`FL_Client_Unified.py`, `FL_Server_Unified.py`)

**Verification checklist:**
- [ ] Message size limits match
- [ ] Queue/buffer sizes match
- [ ] Timeout settings match
- [ ] Chunking configuration matches
- [ ] QoS policies match (for DDS)

---

## 6. STANDARDIZED CONFIGURATION CODE

### MQTT Configuration
```python
# Client and Server
mqtt_client._max_packet_size = 128 * 1024 * 1024  # 128 MB
mqtt_client.max_inflight_messages_set(20)
mqtt_client.max_queued_messages_set(1000)  # Limited to 1000 messages
mqtt_client.keepalive = 600  # 10 minutes for very_poor network
```

### AMQP Configuration
```python
# Client and Server
parameters = pika.ConnectionParameters(
    host=AMQP_HOST,
    port=AMQP_PORT,
    credentials=credentials,
    heartbeat=600,  # 10 minutes for very_poor network
    blocked_connection_timeout=600  # Aligned with heartbeat
)
# RabbitMQ default max message: 128 MB (no explicit config needed)
# Queue max length: 1000 (RabbitMQ default)
```

### gRPC Configuration
```python
# Client and Server - SAME FOR STANDALONE AND UNIFIED
options = [
    ('grpc.max_send_message_length', 128 * 1024 * 1024),     # 128 MB
    ('grpc.max_receive_message_length', 128 * 1024 * 1024),  # 128 MB
    ('grpc.keepalive_time_ms', 600000),  # 10 minutes
    ('grpc.keepalive_timeout_ms', 60000),  # 1 minute timeout
    ('grpc.keepalive_permit_without_calls', 1),
    ('grpc.http2.max_pings_without_data', 0),
    ('grpc.http2.min_time_between_pings_ms', 10000),
    ('grpc.http2.max_ping_strikes', 2),
]
```

### QUIC Configuration
```python
# Client and Server
config = QuicConfiguration(
    is_client=True/False,  # Client/Server specific
    alpn_protocols=["fl"],
    verify_mode=ssl.CERT_NONE,
    max_stream_data=128 * 1024 * 1024,  # 128 MB per stream
    max_data=256 * 1024 * 1024,         # 256 MB total connection
    idle_timeout=600.0,                 # 10 minutes for very_poor network
    max_datagram_frame_size=65536       # 64 KB frames
)
```

### DDS Configuration
```python
# Chunking configuration
CHUNK_SIZE = 64 * 1024  # 64 KB

# QoS for control messages (registration, commands)
control_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=60)),
    Policy.History.KeepLast(10),
    Policy.Durability.TransientLocal
)

# QoS for data chunks (model updates)
chunk_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=600)),  # 10 min
    Policy.History.KeepLast(2048),  # 2048 × 64KB = 128 MB buffer
    Policy.Durability.Volatile
)

# QoS for metrics (small messages)
metrics_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=60)),
    Policy.History.KeepLast(10),
    Policy.Durability.TransientLocal
)
```

---

## 7. IMPLEMENTATION CHECKLIST

For each protocol implementation, verify:

### Standalone Implementations
- [ ] `FL_Client_MQTT.py` - Updated
- [ ] `FL_Server_MQTT.py` - Updated
- [ ] `FL_Client_AMQP.py` - Updated
- [ ] `FL_Server_AMQP.py` - Updated
- [ ] `FL_Client_gRPC.py` - Updated
- [ ] `FL_Server_gRPC.py` - Updated
- [ ] `FL_Client_QUIC.py` - Updated
- [ ] `FL_Server_QUIC.py` - Updated
- [ ] `FL_Client_DDS.py` - Verified
- [ ] `FL_Server_DDS.py` - Verified

### Unified RL Implementation
- [ ] `FL_Client_Unified.py` - MQTT section
- [ ] `FL_Client_Unified.py` - AMQP section
- [ ] `FL_Client_Unified.py` - gRPC section
- [ ] `FL_Client_Unified.py` - QUIC section
- [ ] `FL_Client_Unified.py` - DDS section
- [ ] `FL_Server_Unified.py` - MQTT section
- [ ] `FL_Server_Unified.py` - AMQP section
- [ ] `FL_Server_Unified.py` - gRPC section
- [ ] `FL_Server_Unified.py` - QUIC section
- [ ] `FL_Server_Unified.py` - DDS section

---

## 8. TESTING REQUIREMENTS

After applying these configurations, test under:

1. **Excellent Network** (0% loss, 10ms latency, 100 Mbps)
2. **Good Network** (1% loss, 50ms latency, 50 Mbps)
3. **Fair Network** (5% loss, 100ms latency, 10 Mbps)
4. **Poor Network** (10% loss, 200ms latency, 1 Mbps)
5. **Very Poor Network** (20% loss, 500ms latency, 256 Kbps)

**Expected behavior:** All protocols should:
- Successfully transmit 128 MB messages (either directly or via chunking)
- Survive 10-minute idle periods
- Handle queue pressure identically
- Show performance differences due to **protocol design**, not configuration bias

---

## 9. VALIDATION

Run comparison tests with:
```bash
# Test each protocol independently
python experiment_runner.py --protocol mqtt --network very_poor
python experiment_runner.py --protocol amqp --network very_poor
python experiment_runner.py --protocol grpc --network very_poor
python experiment_runner.py --protocol quic --network very_poor
python experiment_runner.py --protocol dds --network very_poor

# Test unified RL-based protocol selection
python experiment_runner.py --protocol rl_unified --network very_poor
```

**Success criteria:**
- All protocols complete FL training successfully
- Performance differences reflect protocol design, not config bias
- Unified RL implementation shows same behavior as standalone when selecting each protocol

---

## 10. MAINTENANCE

**Last Updated:** 2026-02-09
**Next Review:** After any protocol implementation changes

**Approval Required:** Any deviation from these standards must be documented with:
1. Technical justification
2. Impact analysis on fairness
3. Alternative fair comparison approach
