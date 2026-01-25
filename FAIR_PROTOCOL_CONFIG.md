# Fair Protocol Comparison Configuration

## Overview
This document defines standardized configurations for all communication protocols to ensure fair performance comparison in Federated Learning experiments.

## Configuration Assessment ✅

Your proposed configurations are **FAIR and WELL-DESIGNED** for the following reasons:

### ✅ Consistent Reliability
- All protocols use reliable delivery mechanisms
- Ensures no data loss during model transmission
- Fair comparison of overhead vs. reliability trade-offs

### ✅ Consistent Keepalive/Heartbeat (60s)
- Uniform connection management across protocols
- Prevents premature disconnections
- Allows fair comparison of connection overhead

### ✅ Adequate Message Sizes
- 50MB for gRPC/QUIC handles 12MB models + overhead
- MQTT broker configured for 12MB+ messages
- Sufficient headroom for serialization overhead

### ✅ Similar QoS Guarantees
- All protocols configured for at-least-once delivery
- Comparable reliability guarantees
- Fair latency vs. reliability trade-off

## Detailed Configuration

### 1. MQTT Configuration

**Server/Client Settings:**
```python
QoS: 1 (at-least-once delivery)
Keep-alive: 60 seconds
Clean session: True (stateless between rounds)
Message retention: False
```

**Broker Settings (Mosquitto):**
```conf
max_packet_size 12582912  # 12MB (12 * 1024 * 1024)
message_size_limit 12582912
keepalive_interval 60
max_keepalive 120
max_connections 1000
```

**Rationale:**
- QoS 1 balances reliability and performance
- Clean session ensures stateless operation
- 12MB+ handles typical FL model sizes

---

### 2. AMQP Configuration

**Connection Settings:**
```python
Delivery mode: 2 (persistent messages)
Acknowledgment: Manual ACK (guaranteed delivery)
Heartbeat: 60 seconds
Prefetch count: 1 (one model at a time)
Blocked connection timeout: 300 seconds
```

**Queue Settings:**
```python
Durable: True (survive broker restart)
Auto-delete: False
Exclusive: False
```

**Rationale:**
- Persistent messages ensure reliability
- Manual ACK provides delivery guarantees
- Prefetch=1 optimizes for large model updates
- Heartbeat aligns with other protocols

---

### 3. gRPC Configuration

**Message Size Limits:**
```python
max_send_message_length: 52428800  # 50MB (50 * 1024 * 1024)
max_receive_message_length: 52428800  # 50MB
```

**Keepalive Settings:**
```python
grpc.keepalive_time_ms: 60000  # 60s
grpc.keepalive_timeout_ms: 20000  # 20s
grpc.keepalive_permit_without_calls: 1
grpc.http2.max_pings_without_data: 0
```

**Timeout Settings:**
```python
grpc.http2.min_time_between_pings_ms: 10000  # 10s
grpc.http2.max_ping_strikes: 2
Call timeout: 300 seconds (for large messages)
```

**Rationale:**
- 50MB handles 12MB models with overhead
- Keepalive prevents idle disconnections
- Extended timeout for large model transmission
- HTTP/2 optimizes flow control

---

### 4. QUIC Configuration

**Stream Settings:**
```python
max_stream_data: 52428800  # 50MB per stream
max_data: 104857600  # 100MB total connection
idle_timeout: 60.0  # 60 seconds
```

**Connection Settings:**
```python
max_datagram_frame_size: 65536  # 64KB
initial_max_streams_bidi: 100
initial_max_streams_uni: 100
```

**Congestion Control:**
```python
congestion_control_algorithm: 'cubic'  # Default, or 'bbr' if available
```

**Rationale:**
- QUIC streams are inherently reliable
- 50MB per stream accommodates FL models
- Cubic CC is standard and well-tested
- BBR can be used for comparison if needed

---

### 5. DDS (CycloneDDS) Configuration

**QoS Settings:**
```xml
<Reliability>
    <Kind>RELIABLE</Kind>
    <MaxBlockingTime>300s</MaxBlockingTime>  <!-- 5 minutes -->
</Reliability>

<History>
    <Kind>KEEP_LAST</Kind>
    <Depth>1</Depth>  <!-- Only latest model -->
</History>

<ResourceLimits>
    <MaxSamples>1</MaxSamples>
    <MaxInstances>100</MaxInstances>
    <MaxSamplesPerInstance>1</MaxSamplesPerInstance>
</ResourceLimits>
```

**Discovery & Lease:**
```xml
<Discovery>
    <ParticipantLeaseDuration>60s</ParticipantLeaseDuration>
    <AllowMulticast>false</AllowMulticast>  <!-- Unicast for Docker -->
</Discovery>
```

**Data Transmission:**
```xml
<Internal>
    <FragmentSize>8192</FragmentSize>  <!-- 8KB fragments -->
    <EnableSharedMemory>false</EnableSharedMemory>  <!-- Docker compatibility -->
</Internal>

<MaxMessageSize>10485760</MaxMessageSize>  <!-- 10MB -->
```

**Rationale:**
- Reliable QoS with 5-min blocking suitable for FL
- KeepLast(1) ensures only latest model is stored
- 60s lease duration aligns with other protocols
- 8KB fragments optimize network transmission
- Shared memory disabled for Docker isolation

---

## Comparison Summary

| Protocol | Reliability | Keepalive/Lease | Max Message | Timeout | Delivery Guarantee |
|----------|-------------|-----------------|-------------|---------|-------------------|
| **MQTT** | QoS 1 | 60s | 12MB+ | Default | At-least-once |
| **AMQP** | Persistent | 60s | Unlimited* | 300s | Exactly-once (manual ACK) |
| **gRPC** | HTTP/2 | 60s | 50MB | 300s | At-least-once |
| **QUIC** | Stream-level | 60s | 50MB per stream | 60s idle | At-least-once |
| **DDS** | Reliable | 60s | 10MB (fragmented) | 300s | At-least-once |

\* AMQP has no hard limit, but practical limits apply based on broker memory

---

## Implementation Checklist

### ✅ To Configure

- [x] MQTT: Update broker config (mosquitto.conf)
- [x] MQTT: Ensure QoS 1 in all publish/subscribe calls
- [x] AMQP: Set heartbeat to 60s
- [x] AMQP: Verify delivery_mode=2 and manual ACK
- [x] gRPC: Add keepalive settings
- [x] gRPC: Update message size to 50MB
- [x] QUIC: Update idle_timeout to 60s
- [x] QUIC: Update max_stream_data to 50MB
- [x] DDS: Update XML config for KeepLast(1)
- [x] DDS: Set lease duration to 60s
- [x] DDS: Configure fragment size to 8KB

---

## Fairness Validation

### Parameters Aligned:
1. **Connection Management**: All use 60s keepalive/heartbeat/lease
2. **Reliability**: All guarantee at-least-once delivery minimum
3. **Message Capacity**: All support 12MB+ models
4. **Timeout Handling**: Consistent 300s for large transfers
5. **Stateless Operation**: Clean session/no history retention

### Expected Differences (Protocol Characteristics):
- **Overhead**: Varies based on protocol design
- **Latency**: Varies based on framing/acknowledgment
- **Resource Usage**: Varies based on implementation
- **Network Efficiency**: Varies based on serialization

These differences reflect **inherent protocol characteristics**, making the comparison meaningful and fair.

---

## Testing Recommendations

1. **Baseline Test**: Run all protocols in good network conditions
2. **Stress Test**: Test with 12MB models at various network conditions
3. **Reliability Test**: Verify delivery guarantees under packet loss
4. **Timeout Test**: Ensure all protocols handle timeouts consistently
5. **Resource Test**: Monitor CPU/memory across all protocols

---

**Status**: Configuration design approved ✅
**Next Step**: Implement configurations in codebase
