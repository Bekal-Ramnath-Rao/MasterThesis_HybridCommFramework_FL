# DDS Write Timeout Fix for Large Model Transmission

## Problem
When running DDS-based federated learning in **very poor network conditions**, the server was getting a write timeout error when trying to send the initial global model (~12.5 MB):

```
[DDS_RETCODE_TIMEOUT] A timeout has occurred. 
Occurred while writing sample in <Entity, type=cyclonedds.pub.DataWriter>
```

**Root cause**: The DataWriter QoS policy had `max_blocking_time=duration(seconds=1)` which is too short for transmitting large models in poor network conditions.

## Solution

### Server-Side Fix
**File**: [Server/Emotion_Recognition/FL_Server_DDS.py](Server/Emotion_Recognition/FL_Server_DDS.py)

**Before:**
```python
reliable_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
    Policy.History.KeepAll,
    Policy.Durability.TransientLocal
)
```

**After:**
```python
reliable_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=300)),  # 5 minutes
    Policy.History.KeepAll,
    Policy.Durability.TransientLocal,
    Policy.ResourceLimits(
        max_samples=100,
        max_instances=100,
        max_samples_per_instance=100
    )
)
```

### Client-Side Fix
**File**: [Client/Emotion_Recognition/FL_Client_DDS.py](Client/Emotion_Recognition/FL_Client_DDS.py)

Applied the same QoS changes to ensure both sides have sufficient timeout for large data transfers.

## Changes Applied

### Emotion Recognition (Fixed)
- ✅ **Server**: Increased from 1s to 300s (5 minutes)
- ✅ **Client**: Increased from 1s to 300s (5 minutes)
- ✅ Added ResourceLimits policy for large samples

### Temperature & MentalState (Already OK)
- ✅ **Server**: Already had 3600s (1 hour) timeout
- ✅ **Client**: Already had 3600s (1 hour) timeout

## Key Changes

1. **Increased max_blocking_time**: 1s → 300s
   - Allows DDS to wait up to 5 minutes for data transmission
   - Critical for 12+ MB models in poor network conditions

2. **Added ResourceLimits policy**:
   - `max_samples=100`: Maximum number of samples in queue
   - `max_instances=100`: Maximum instances
   - `max_samples_per_instance=100`: Samples per instance
   - Prevents resource exhaustion with large data

## Network Condition Guidelines

| Network Condition | Model Size | Recommended Timeout |
|------------------|------------|---------------------|
| Good (>10 Mbps) | 12 MB | 30-60s |
| Poor (1-10 Mbps) | 12 MB | 120-300s |
| Very Poor (<1 Mbps) | 12 MB | 300-600s |
| Extremely Poor (<100 Kbps) | 12 MB | 600-1800s |

## Testing

### Expected Behavior After Fix

**Server logs:**
```
======================================================================
Distributing Initial Global Model
======================================================================

Publishing initial model to clients (sending multiple times for reliability)...
  Attempt 1/3: Initial model published via DDS
  Attempt 2/3: Initial model published via DDS
  Attempt 3/3: Initial model published via DDS
Initial global model sent to all clients
Waiting for clients to initialize their models (TensorFlow + CNN building)...
```

**Client logs:**
```
Client 1: Received initial global model from server
Client 1: Building CNN model from server configuration...
Model initialized successfully
Client 1 model ready for training
```

### Troubleshooting

**If timeout still occurs:**
1. Check network bandwidth between containers
2. Increase timeout further in QoS policy
3. Enable quantization to reduce model size:
   ```bash
   export USE_QUANTIZATION=true
   ```
4. Check CycloneDDS configuration for fragment size limits

**Check DDS discovery:**
```bash
# Inside container
ddsperf ping
```

**Monitor network usage:**
```bash
docker stats
```

## Configuration Options

### Environment Variables (Optional)
While the timeout is now hardcoded to 300s, you can modify the source if needed:

```python
# In FL_Server_DDS.py and FL_Client_DDS.py
# Change this line:
Policy.Reliability.Reliable(max_blocking_time=duration(seconds=300))

# To use environment variable:
timeout_seconds = int(os.getenv("DDS_WRITE_TIMEOUT", "300"))
Policy.Reliability.Reliable(max_blocking_time=duration(seconds=timeout_seconds))
```

## Technical Details

### DDS QoS Policies Explained

**Reliability.Reliable**:
- Ensures guaranteed delivery of samples
- `max_blocking_time`: How long writer waits for acknowledgment
- Critical for large data transfers

**History.KeepAll**:
- Keeps all samples until delivered
- Ensures no data loss
- May use more memory

**Durability.TransientLocal**:
- Late-joining readers receive historical data
- Important for federated learning initialization

**ResourceLimits**:
- Prevents unbounded resource usage
- Important for large models
- Configures queue sizes

### Why This Happens in Poor Networks

1. **Large Sample Size**: 12.5 MB model serialized to bytes
2. **Network Fragmentation**: DDS fragments large samples
3. **Slow Transmission**: Poor network takes time to send all fragments
4. **ACK Timeout**: Writer waits for acknowledgment from readers
5. **Blocking**: With 1s timeout, writer gives up before transmission completes

### DDS Fragment Size

CycloneDDS automatically fragments large messages. Default fragment size is typically 64KB. For a 12.5 MB model:
- Fragments: ~196 fragments (12.5 MB / 64 KB)
- At 100 Kbps: ~1000 seconds to transmit
- At 1 Mbps: ~100 seconds to transmit
- At 10 Mbps: ~10 seconds to transmit

## Related Files
- [QUIC_TIMEOUT_FIX.md](QUIC_TIMEOUT_FIX.md) - Similar fix for QUIC protocol
- [DDS_POOR_NETWORK_FIX.md](DDS_POOR_NETWORK_FIX.md) - Additional DDS network optimizations

## Summary

✅ **Fixed**: DDS write timeout from 1s to 300s for Emotion Recognition  
✅ **Added**: ResourceLimits policy for large sample support  
✅ **Verified**: Temperature & MentalState already had proper timeouts  
🎯 **Result**: DDS can now successfully transmit 12+ MB models in very poor network conditions
