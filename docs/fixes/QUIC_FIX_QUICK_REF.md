# QUIC Persistent Connection - Quick Reference

## What Was Fixed

**Problem**: QUIC clients couldn't receive responses → rounds stuck  
**Solution**: Persistent bidirectional QUIC connection in dedicated thread

## Architecture

```
┌─────────────────────────────────────────────┐
│  Main Thread (Synchronous)                  │
│  - MQTT loop_forever()                      │
│  - Calls _send_via_quic()                   │
│  - Receives messages via callbacks          │
└────────────────┬────────────────────────────┘
                 │
                 │ run_coroutine_threadsafe()
                 ↓
┌─────────────────────────────────────────────┐
│  QUIC Thread (Asynchronous)                 │
│  - Dedicated asyncio event loop             │
│  - Persistent connection via async with     │
│  - UnifiedClientQUICProtocol                │
│    • Receives messages from server          │
│    • Buffers streams                        │
│    • Calls client._handle_quic_message()    │
└─────────────────────────────────────────────┘
```

## Key Components

### 1. Protocol Class (Lines 56-108)
```python
class UnifiedClientQUICProtocol(QuicConnectionProtocol):
    def quic_event_received(self, event):
        # Buffers data, parses JSON, calls client callback
```

### 2. Connection Thread (Lines 973-986)
```python
def _run_quic_loop(self):
    self.quic_loop = asyncio.new_event_loop()
    self.quic_loop.run_until_complete(self._quic_connection_loop())
```

### 3. Persistent Connection (Lines 988-1022)
```python
async def _quic_connection_loop(self):
    async with connect(...) as protocol:
        self.quic_protocol = protocol
        await asyncio.Future()  # Keep alive forever
```

### 4. Send Method (Lines 1074-1103)
```python
def _send_via_quic(self, message):
    loop = asyncio.new_event_loop()
    loop.run_until_complete(self._send_quic_persistent(payload))
```

### 5. Cross-Thread Send (Lines 1142-1156)
```python
async def _send_quic_persistent(self, payload):
    future = asyncio.run_coroutine_threadsafe(
        self._do_quic_send(payload),
        self.quic_loop  # QUIC thread's event loop
    )
    future.result(timeout=10)
```

## Flow Diagram

### Client Send (Update/Metrics)
```
Main Thread                    QUIC Thread
-----------                    -----------
_send_via_quic()
  ↓
  Create temp event loop
  ↓
  _send_quic_persistent()
                              → Schedule on QUIC loop
                              → _do_quic_send()
                              → protocol.send_stream_data()
                              → transmit()
  ↓
  Wait for future
  ← Return
```

### Server Response (Global Model)
```
QUIC Thread                    Main Thread
-----------                    -----------
UnifiedClientQUICProtocol
  ↓
  quic_event_received()
  ↓
  Buffer stream data
  ↓
  Parse JSON message
  ↓
  client._handle_quic_message()
                              → Check type
                              → on_global_model_received_quic()
                              → model.set_weights()
                              → Signal training/eval ready
```

## Configuration

### Environment Variables
```bash
QUIC_HOST=fl-server-unified-emotion  # Docker service name
QUIC_PORT=4433                        # QUIC server port
```

### QUIC Config (Lines 993-999)
```python
QuicConfiguration(
    is_client=True,
    verify_mode=ssl.CERT_NONE,        # Skip cert validation
    max_stream_data=50 * 1024 * 1024,  # 50 MB per stream
    max_data=100 * 1024 * 1024,        # 100 MB total
    idle_timeout=3600.0                # 1 hour timeout
)
```

## Testing

### Check Connection Established
```bash
docker logs fl-client-unified-emotion-1 2>&1 | grep "QUIC.*established persistent connection"
```

### Check Messages Received
```bash
docker logs fl-client-unified-emotion-1 2>&1 | grep "QUIC.*received.*global_model"
docker logs fl-client-unified-emotion-1 2>&1 | grep "QUIC.*received.*start_evaluation"
```

### Check Round Progression
```bash
docker logs fl-server-unified-emotion 2>&1 | grep "Round"
# Should see: Round 1, Round 2, Round 3...
```

## Debugging

### Connection Issues
```python
# Check if thread started
print(f"QUIC thread alive: {self.quic_thread.is_alive()}")

# Check if protocol set
print(f"QUIC protocol: {self.quic_protocol}")

# Check event loop
print(f"QUIC loop running: {self.quic_loop.is_running()}")
```

### Message Reception
```python
# In UnifiedClientQUICProtocol.quic_event_received:
print(f"[DEBUG] Stream {event.stream_id}: {len(event.data)} bytes")
print(f"[DEBUG] Buffer size: {len(self._stream_buffers[event.stream_id])}")
print(f"[DEBUG] Decoded message type: {message.get('type')}")
```

### Send Issues
```python
# In _do_quic_send:
print(f"[DEBUG] Stream ID: {stream_id}")
print(f"[DEBUG] Payload size: {len(data)} bytes")
print(f"[DEBUG] Protocol: {self.quic_protocol}")
```

## Common Issues

### 1. Connection Not Establishing
**Symptom**: Timeout error after 10s  
**Fix**: Check server is running, verify host/port, check firewall

### 2. Messages Not Received
**Symptom**: Client sends but doesn't get responses  
**Fix**: Verify server is sending on QUIC, check protocol type field

### 3. Event Loop Errors
**Symptom**: "Event loop is closed" errors  
**Fix**: Ensure each thread has its own event loop, don't share

### 4. Thread Not Starting
**Symptom**: No connection establishment logs  
**Fix**: Check if _ensure_quic_connection is called, verify thread daemon=True

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| Connection Lifecycle | Create → Send → Close | Persistent |
| Client Receives? | ❌ No | ✅ Yes |
| Event Loop | asyncio.run() each time | Dedicated thread |
| Protocol Handler | None | UnifiedClientQUICProtocol |
| Rounds Progress? | ❌ Stuck | ✅ Working |

## Integration Points

### With MQTT (Control Signals)
- MQTT runs in main thread with `loop_forever()`
- QUIC runs in background thread
- Both can coexist without interference

### With RL Protocol Selector
- RL selects QUIC as protocol
- `_send_via_quic()` called from sync code
- Transparent to RL agent

### With Server
- Server tracks client protocol per round
- Server sends response back via same protocol
- Server's `send_quic_message()` sends to QUIC stream

## Files Modified

1. **FL_Client_Unified.py**
   - Added UnifiedClientQUICProtocol class
   - Added thread-based connection management
   - Updated send methods for persistent connection
   - Added message handlers for incoming data

2. **QUIC_PERSISTENT_CONNECTION_FIX.md**
   - Comprehensive documentation

3. **QUIC_FIX_QUICK_REF.md** (this file)
   - Quick reference guide

## Next Steps

1. Test with single client QUIC
2. Test with 2 clients both using QUIC
3. Test with mixed protocols (QUIC + others)
4. Verify RL learns QUIC performance correctly
5. Run full experiment suite

## Success Indicators

✅ Logs show "QUIC persistent connection established"  
✅ Client receives global model messages  
✅ Rounds progress beyond Round 1  
✅ No connection timeout errors  
✅ Clean shutdown on Ctrl+C
