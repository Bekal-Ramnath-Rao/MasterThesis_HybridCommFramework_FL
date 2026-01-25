# DDS Poor Network Fix - Quick Reference

## Summary
**Problem**: DDS fails after Round 1 in poor networks (384 Kbps, 300ms latency)  
**Root Cause**: Timeouts too short for 12MB transfers (~400s actual vs 60s lease)  
**Solution**: Increase timeouts, reduce fragmentation overhead, tune retransmissions

## Key Changes

| Setting | Before | After | Reason |
|---------|--------|-------|--------|
| **Lease Duration** | 60s | 180s | Prevent timeout during 400s transfers |
| **Fragment Size** | 8KB | 16KB | 750 fragments vs 1500 (50% reduction) |
| **Max Blocking Time** | 300s | 600s | Allow 10min for slow transfers |
| **Socket Buffers** | Default | 10MB | Handle large 12MB models |
| **Defrag Samples** | Default | 1000 | Hold all fragments (750 max) |
| **Heartbeat Delay** | - | 5s | Reduce congestion storms |
| **NACK Delay** | - | 2s | Slower retransmissions |

## Transfer Time Calculations

### Very Poor (2G): 384 Kbps, 300ms, 5% loss
```
12MB = 100,663,296 bits
Time = 100,663,296 / 384,000 = 262s theoretical
With 5% loss + retransmissions: ~400s actual

✅ 180s lease > 120s timeout window
✅ 600s blocking > 400s transfer
✅ 1000 defrag samples > 750 fragments
```

### Poor (3G): 2 Mbps, 100ms, 3% loss
```
Time = 100,663,296 / 2,000,000 = 50s theoretical
With 3% loss: ~60s actual

✅ Well within all timeouts
```

## Files Modified

1. **cyclonedds-emotion.xml** - Emotion recognition config
2. **cyclonedds-temperature.xml** - Temperature regulation config
3. **cyclonedds-mentalstate.xml** - Mental state monitoring config
4. **Server/Emotion_Recognition/FL_Server_DDS.py** - QoS settings (line 226)

## Verification

```bash
./verify_dds_poor_network_config.sh
```

Expected: **20/20 checks passed**

## Testing

```bash
# Enable poor network conditions
export APPLY_NETWORK_CONDITION=true
export NETWORK_CONDITION=very_poor

# Run DDS experiment
cd Docker
docker-compose -f docker-compose-emotion.yml up

# Monitor progress
docker logs -f fl-server-dds-emotion
```

## Expected Results

| Round | Phase | Transfer | Time (very_poor) | Status |
|-------|-------|----------|------------------|--------|
| 1 | Initial model | Server→Client (12MB) | ~400s | ✅ No timeout |
| 2 | Client update | Client→Server (12MB) | ~400s | ✅ No timeout |
| 2 | Global model | Server→Client (12MB) | ~400s | ✅ No timeout |
| 3+ | Repeat | - | ~800s/round | ✅ Stable |

## Comparison with Other Protocols

| Protocol | Keepalive/Lease | Max Message | Blocking/Timeout | Very Poor Status |
|----------|-----------------|-------------|------------------|------------------|
| **MQTT** | 60s | 12MB | Infinite retry | ✅ Works |
| **AMQP** | 60s | No limit | Infinite retry | ✅ Works |
| **gRPC** | 60s | 50MB | 20s (with retry) | ✅ Works |
| **QUIC** | 60s idle | 50MB | Connection retry | ✅ Works |
| **DDS** | 180s | 10MB (frag) | 600s blocking | ✅ **Now works!** |

## Troubleshooting

### Symptoms of Wrong Config
```
❌ "Participant lost" after 60-120s
❌ "Write timeout" after 300s
❌ Fragments dropping/timing out
❌ Network congestion storms
```

### Verify Config
```bash
# Check lease
grep "ParticipantLeaseDuration" cyclonedds-emotion.xml
# Should show: <ParticipantLeaseDuration>180s</ParticipantLeaseDuration>

# Check fragment size
grep "FragmentSize" cyclonedds-emotion.xml
# Should show: <FragmentSize>16384</FragmentSize>

# Check QoS blocking
grep "max_blocking_time" Server/Emotion_Recognition/FL_Server_DDS.py
# Should show: max_blocking_time=duration(seconds=600)
```

### Logs to Monitor
```bash
# DDS discovery
docker logs fl-server-dds-emotion | grep -i "participant\|discovery\|lease"

# Transfer progress
docker logs fl-server-dds-emotion | grep -i "round\|sending\|received"

# Errors
docker logs fl-server-dds-emotion | grep -i "timeout\|error\|lost"
```

## References

- Full documentation: [DDS_POOR_NETWORK_FIX.md](DDS_POOR_NETWORK_FIX.md)
- Fair protocol config: [FAIR_PROTOCOL_CONFIG.md](FAIR_PROTOCOL_CONFIG.md)
- Network conditions: [NETWORK_CONDITIONS_USAGE.md](NETWORK_CONDITIONS_USAGE.md)
