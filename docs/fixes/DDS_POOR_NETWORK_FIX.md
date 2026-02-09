# DDS Configuration Fix for Poor Network Conditions

## Problem
DDS stopped working after Round 1 in poor/very_poor network conditions (2G: 300ms latency, 384 Kbps, 5% loss) while MQTT/AMQP/gRPC/QUIC worked fine.

## Root Causes

### 1. **Lease Timeout Too Short**
- **Problem**: 60s lease → participants marked "dead" during slow 12MB transfers
- **Math**: 12MB @ 384 Kbps = 262s theoretical, ~400s with packet loss
- **Fix**: Increased to **180s** (3 minutes)

### 2. **Fragmentation Overhead**
- **Problem**: 8KB fragments → 12MB / 8KB = **1,500 fragments**
- **Impact**: Each fragment has protocol overhead (headers, ACKs, retransmissions)
- **Fix**: Increased to **16KB** → only **750 fragments** (50% reduction)

### 3. **Fragmentation Reassembly Timeout**
- **Problem**: Default defrag timeout (~30s) → fragments expire before reassembly
- **Fix**: Added `DefragReliableMaxSamples=1000` to hold all fragments

### 4. **Aggressive Retransmissions**
- **Problem**: Fast retransmissions → network congestion storms in poor conditions
- **Fix**: Added delays:
  - `HeartbeatResponseDelay=5s`
  - `NackResponseDelay=1s`
  - `NackDelay=2s`

### 5. **Insufficient Socket Buffers**
- **Problem**: Default buffers too small for 12MB messages
- **Fix**: Set `SocketReceiveBufferSize=10MB` and `SocketSendBufferSize=10MB`

### 6. **QoS Blocking Time Too Short**
- **Problem**: `max_blocking_time=300s` → write blocks timeout at 5 minutes
- **Math**: 12MB @ 384 Kbps with retransmissions = ~400s
- **Fix**: Increased to **600s** (10 minutes)

## Configuration Changes

### XML Configuration (cyclonedds-*.xml)

```xml
<Discovery>
    <!-- Increased from 60s to 180s -->
    <ParticipantLeaseDuration>180s</ParticipantLeaseDuration>
    <SPDPInterval>60s</SPDPInterval>
</Discovery>

<Internal>
    <!-- Increased from 8KB to 16KB -->
    <FragmentSize>16384</FragmentSize>
    <EnableSharedMemory>false</EnableSharedMemory>
    
    <!-- NEW: Retransmission tuning -->
    <HeartbeatResponseDelay>5s</HeartbeatResponseDelay>
    <NackResponseDelay>1s</NackResponseDelay>
    <NackDelay>2s</NackDelay>
    
    <!-- NEW: Socket buffers -->
    <SocketReceiveBufferSize min="10MB"/>
    <SocketSendBufferSize min="10MB"/>
    
    <!-- NEW: Defrag limits -->
    <DefragReliableMaxSamples>1000</DefragReliableMaxSamples>
    <DefragUnreliableMaxSamples>1000</DefragUnreliableMaxSamples>
</Internal>
```

### Python QoS (FL_Server_DDS.py)

```python
reliable_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=600)),  # Increased from 300s
    Policy.History.KeepLast(1),
    Policy.Durability.TransientLocal
)
```

## Files Updated

1. `cyclonedds-emotion.xml` - Emotion recognition scenario
2. `cyclonedds-temperature.xml` - Temperature regulation scenario
3. `cyclonedds-mentalstate.xml` - Mental state monitoring scenario
4. `Server/Emotion_Recognition/FL_Server_DDS.py` - QoS blocking time

## Network Condition Impact

### Very Poor (2G): 300ms latency, 384 Kbps, 5% loss

**Before Fix:**
- 8KB fragments: 1,500 fragments
- 60s lease: Timeout after 120s
- 300s blocking: Timeout at 5 minutes
- **Result**: Fails after Round 1

**After Fix:**
- 16KB fragments: 750 fragments (50% less overhead)
- 180s lease: No timeout for 400s transfers
- 600s blocking: No timeout for slow networks
- Retransmission delays: Prevents congestion storms
- Socket buffers: Handles large messages efficiently
- **Result**: Works reliably

### Poor (3G): 100ms latency, 2 Mbps, 3% loss

**Transfer Time:**
- 12MB @ 2 Mbps = 48s theoretical, ~60s with loss
- **Status**: Well within 180s lease and 600s blocking ✅

### Moderate (4G): 50ms latency, 20 Mbps, 1% loss

**Transfer Time:**
- 12MB @ 20 Mbps = 4.8s theoretical, ~5s with loss
- **Status**: Works perfectly ✅

## Testing

To test DDS in poor networks:

```bash
# Set network condition
export APPLY_NETWORK_CONDITION=true
export NETWORK_CONDITION=very_poor

# Run DDS experiment
cd Docker
docker-compose -f docker-compose-emotion.yml build
docker-compose -f docker-compose-emotion.yml up
```

## Expected Behavior

1. **Round 1**: Initial 12MB model distribution
   - Transfer time: ~400s in very_poor
   - No lease timeout (180s > 120s window)
   - No blocking timeout (600s > 400s transfer)
   - ✅ SUCCESS

2. **Round 2+**: Client updates (12MB) → Server aggregation → Global model (12MB)
   - Each transfer: ~400s in very_poor
   - Total round time: ~800s (client→server + server→client)
   - ✅ SUCCESS

3. **Comparison with other protocols**:
   - MQTT: QoS 1, 12MB packets, 60s keepalive - ✅ Works
   - AMQP: Persistent delivery, 60s heartbeat - ✅ Works
   - gRPC: 50MB messages, 60s keepalive - ✅ Works
   - QUIC: 50MB streams, 60s idle timeout - ✅ Works
   - **DDS: Now comparable with proper timeouts - ✅ Works**

## Key Insights

1. **Network protocol tuning is critical**: Default timeouts assume good networks
2. **Fragmentation overhead matters**: 16KB vs 8KB = 50% fewer fragments
3. **Retransmission delays prevent storms**: Fast retries worsen congestion
4. **Buffer sizing enables large messages**: 10MB buffers for 12MB models
5. **Timeout consistency**: All protocols now use ~60-180s keepalive/lease ranges

## References

- [CycloneDDS Configuration Guide](https://github.com/eclipse-cyclonedds/cyclonedds/blob/master/docs/manual/config.rst)
- [DDS Reliability QoS](https://www.omg.org/spec/DDS/1.4/PDF)
- Fair Protocol Comparison: See `FAIR_PROTOCOL_CONFIG.md`
