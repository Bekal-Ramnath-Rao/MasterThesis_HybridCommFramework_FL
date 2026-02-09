# QUIC Persistent Connection Implementation

## Problem Statement

QUIC clients in unified mode were unable to receive responses from the server, causing rounds to not progress:

1. **Client sends → closes connection immediately** - No way to receive server's response
2. **Server sends response via QUIC** - But client connection already closed
3. **Client waits indefinitely** - Never receives global model or evaluation signal

## Root Cause

The unified QUIC client implementation used a **request-response pattern** instead of a **persistent bidirectional connection**:

```python
# OLD BROKEN APPROACH
async def _quic_send_data(self, host, port, payload, msg_type):
    async with connect(...) as protocol:
        # Send data
        protocol.send_data(...)
        # Connection closes here!
    # No way to receive response
```

## Solution Architecture

Implemented **persistent QUIC connection** with dedicated thread and event loop:

### 1. Thread-Based Event Loop

Since the unified client runs in synchronous MQTT `loop_forever()`, we cannot use the main thread's event loop for QUIC. Solution:

- **Dedicated thread** for QUIC with its own event loop
- **Persistent connection** maintained via `async with connect()`
- **Event loop stays alive** via `await asyncio.Future()`

### 2. Bidirectional Protocol

Added `UnifiedClientQUICProtocol` to handle incoming messages:

```python
class UnifiedClientQUICProtocol(QuicConnectionProtocol):
    def quic_event_received(self, event: QuicEvent):
        # Buffers stream data
        # Parses complete messages (newline-delimited)
        # Calls client._handle_quic_message()
```

### 3. Message Routing

Server responses are now properly received and routed:

```python
def _handle_quic_message(self, message: dict):
    msg_type = message.get('type')
    if msg_type == 'global_model':
        self.on_global_model_received_quic(message)
    elif msg_type == 'start_evaluation':
        self.on_start_evaluation_quic(message)
```

## Implementation Details

### File: `Client/Emotion_Recognition/FL_Client_Unified.py`

#### 1. Connection Variables (Lines 287-290)
```python
self.quic_protocol = None        # Protocol reference
self.quic_connection_task = None # Connection task
self.quic_loop = None           # Dedicated event loop
self.quic_thread = None         # Background thread
```

#### 2. Protocol Class (Lines 56-108)
```python
class UnifiedClientQUICProtocol(QuicConnectionProtocol):
    """Handles incoming QUIC messages from server"""
    - Stream buffering with newline delimiter
    - JSON message parsing
    - Callback to client._handle_quic_message()
```

#### 3. Connection Management (Lines 947-1022)

**_ensure_quic_connection()**: Starts QUIC thread if not running
- Creates daemon thread with dedicated event loop
- Waits up to 10s for connection establishment
- Raises error if connection fails

**_run_quic_loop()**: Runs in background thread
- Creates new event loop
- Runs `_quic_connection_loop()` until completion
- Handles exceptions and cleanup

**_quic_connection_loop()**: Maintains persistent connection
- Uses `async with connect()` context manager
- Keeps connection alive with `await asyncio.Future()`
- Proper cleanup on cancellation

#### 4. Send Methods (Lines 1074-1169)

**_send_via_quic()**: Called from sync code
- Creates temporary event loop
- Calls `_send_quic_persistent()` via `run_until_complete()`
- Logs packet transmission

**_send_quic_persistent()**: Async send wrapper
- Ensures connection is established
- Schedules `_do_quic_send()` on QUIC thread's event loop
- Waits for completion with timeout

**_do_quic_send()**: Actually sends data
- Runs in QUIC thread's event loop
- Gets stream ID and sends data
- Handles large message transmission

#### 5. Message Handlers (Lines 1023-1073)

**_handle_quic_message()**: Routes incoming messages
**on_global_model_received_quic()**: Updates model weights
**on_start_evaluation_quic()**: Signals evaluation can start

#### 6. Cleanup (Lines 374-383)

**cleanup()**: Cancels QUIC connection task
**on_disconnect()**: Calls cleanup before exit

## Pattern Copied From

`Client/Emotion_Recognition/FL_Client_QUIC.py` (single-protocol reference):

- Persistent connection via `async with connect()`
- `FederatedLearningClientProtocol` with `quic_event_received`
- Stream buffering and message parsing
- Connection kept alive with `await asyncio.Future()`

## Testing Checklist

- [ ] Client establishes QUIC connection on startup
- [ ] Client can send model updates via QUIC
- [ ] Client receives global model via QUIC
- [ ] Client receives evaluation signal via QUIC
- [ ] Rounds progress correctly with QUIC protocol
- [ ] Connection remains stable across multiple rounds
- [ ] Proper cleanup on shutdown

## Verification Commands

```bash
# Start server
docker-compose up fl-server-unified-emotion

# Start clients in QUIC mode
export PROTOCOL_SELECTION="false"  # Force QUIC for testing
docker-compose up fl-client-unified-emotion-1 fl-client-unified-emotion-2

# Check logs for:
# - "QUIC persistent connection established"
# - "Client received global model for round N"
# - "Client starting evaluation for round N"
# - Rounds should progress: 1 → 2 → 3...
```

## Key Differences from Other Protocols

| Protocol | Connection Type | Event Loop |
|----------|----------------|------------|
| MQTT | Persistent (loop_forever) | paho.mqtt |
| AMQP | Per-request | Blocking |
| gRPC | Per-request | Threading |
| QUIC | Persistent | Dedicated thread + asyncio |
| DDS | Pub/Sub | DDS middleware |

## Success Criteria

✅ **Before**: Rounds stuck after Round 1 - clients waiting for responses  
✅ **After**: Rounds progress normally - bidirectional QUIC communication working  

✅ **Before**: Connection created and closed for each send  
✅ **After**: Single persistent connection throughout training session  

✅ **Before**: No way to receive server responses  
✅ **After**: UnifiedClientQUICProtocol receives and routes messages  

## Notes

1. **Thread Safety**: QUIC operations run in dedicated thread to avoid blocking MQTT loop
2. **Event Loop Isolation**: Each thread has its own event loop (MQTT main thread, QUIC background thread)
3. **Cross-Thread Communication**: Uses `asyncio.run_coroutine_threadsafe()` for thread-safe async calls
4. **Graceful Shutdown**: Cleanup method cancels QUIC task before exit
5. **Same Pattern as Single-Protocol**: Implementation mirrors working FL_Client_QUIC.py

## Related Files

- `Client/Emotion_Recognition/FL_Client_Unified.py` - Updated client
- `Client/Emotion_Recognition/FL_Client_QUIC.py` - Reference implementation
- `Server/Emotion_Recognition/FL_Server_Unified.py` - Server with QUIC send capability
- `COMPREHENSIVE_EXPERIMENT_SETUP.md` - Testing documentation
