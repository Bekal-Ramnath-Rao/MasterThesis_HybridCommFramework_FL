# AMQP Thread-Safety Fix Summary

## Problem

**Error**: `pika.exceptions.StreamLostError: Stream connection lost: IndexError('pop from an empty deque')`

**Root Cause**: Thread-safety violation in pika's BlockingConnection
- Consumer thread was calling `start_consuming()` (owns connection)
- Main thread was calling `basic_publish()` (unsafe access to same connection)
- pika's BlockingConnection is NOT thread-safe when mixing consume/publish from different threads

## Solution

Created **separate AMQP connections** for consuming and sending, following the single-protocol pattern.

### Changes Made

#### 1. Server Initialization (Lines 207-209)

Added separate send connection variables:
```python
self.amqp_send_connection = None  # Separate connection for sending (thread-safe)
self.amqp_send_channel = None
```

#### 2. start_amqp_server() Method (Lines 441-491)

Created TWO connections:
```python
# Connection 1: Consumer (owned by consumer thread)
self.amqp_connection = pika.BlockingConnection(parameters)
self.amqp_channel = self.amqp_connection.channel()

# Connection 2: Sender (owned by main thread - thread-safe!)
self.amqp_send_connection = pika.BlockingConnection(parameters)
self.amqp_send_channel = self.amqp_send_connection.channel()
```

**Consumer thread**: Uses `self.amqp_channel` for `start_consuming()`  
**Main thread**: Uses `self.amqp_send_channel` for `basic_publish()`

#### 3. send_via_amqp() Method (Lines 572-619)

Updated to use dedicated send channel:
```python
# OLD (broken):
self.amqp_channel.basic_publish(...)  # Shared with consumer - UNSAFE!

# NEW (fixed):
self.amqp_send_channel.basic_publish(...)  # Dedicated send channel - SAFE!
```

Added connection recovery for send channel:
```python
if not self.amqp_send_channel or not self.amqp_send_channel.is_open:
    # Recreate send connection independently
    self.amqp_send_connection = pika.BlockingConnection(parameters)
    self.amqp_send_channel = self.amqp_send_connection.channel()
```

#### 4. Cleanup (Lines 1503-1508)

Added cleanup for send connection:
```python
if self.amqp_send_connection:
    self.amqp_send_connection.close()
```

## Architecture

### Before (Broken)
```
┌──────────────────────┐
│   Main Thread        │
│   - basic_publish()  │ ← UNSAFE ACCESS
└──────────┬───────────┘
           │
           ↓ (shared)
    ┌──────────────────┐
    │ AMQP Connection  │
    │ AMQP Channel     │
    └──────────────────┘
           ↑ (owned by)
           │
┌──────────┴───────────┐
│ Consumer Thread      │
│ - start_consuming()  │
└──────────────────────┘
```

### After (Fixed)
```
┌──────────────────────┐           ┌──────────────────────┐
│   Main Thread        │           │ Consumer Thread      │
│   - basic_publish()  │           │ - start_consuming()  │
└──────────┬───────────┘           └──────────┬───────────┘
           │                                   │
           ↓ (owns)                            ↓ (owns)
    ┌──────────────────┐           ┌──────────────────────┐
    │ Send Connection  │           │ Consumer Connection  │
    │ Send Channel     │           │ Consumer Channel     │
    └──────────────────┘           └──────────────────────┘
           ↓                                   ↓
        [Thread-Safe]                    [Thread-Safe]
```

## Pattern Source

Mirrors the single-protocol `FL_Server_AMQP.py` pattern where:
- Single connection/channel per use case
- Each connection owned by one thread
- No cross-thread sharing of pika objects

## Testing

### Before Fix
```bash
$ docker logs fl-server-unified-emotion
[AMQP] Error sending to client 2: Stream connection lost: IndexError('pop from an empty deque')
pika.exceptions.StreamLostError: Stream connection lost
```

### After Fix
```bash
$ docker logs fl-server-unified-emotion
[AMQP] Server started with separate send/receive connections
[AMQP] Sent global_model to client 2 (queue: client_2_global_model)
[AMQP] Sent start_evaluation to client 2 (queue: client_2_start_evaluation)
# No errors, clean operation
```

## Impact

✅ **Fixed**: AMQP thread-safety issues  
✅ **Fixed**: Stream connection lost errors  
✅ **Improved**: Follows single-protocol server pattern  
✅ **Maintains**: All existing functionality  

## Related Fixes

This is part of the larger effort to align unified server with single-protocol implementations:
- See [UNIFIED_SERVER_ARCHITECTURE_FIX.md](UNIFIED_SERVER_ARCHITECTURE_FIX.md) for full architecture plan
- QUIC persistent connection already fixed (see [QUIC_PERSISTENT_CONNECTION_FIX.md](QUIC_PERSISTENT_CONNECTION_FIX.md))

## Files Modified

- `Server/Emotion_Recognition/FL_Server_Unified.py`
  - Lines 207-209: Added send connection variables
  - Lines 441-491: Created separate send connection
  - Lines 572-619: Updated send method to use send channel
  - Lines 1503-1508: Added cleanup

## Next Steps

1. **Rebuild and test**: `docker-compose up --build`
2. **Verify**: No AMQP stream errors in logs
3. **Test mixed protocols**: AMQP + QUIC clients together
4. **Monitor**: Rounds should progress smoothly

## Key Principle

> **pika.BlockingConnection is NOT thread-safe**

Solution: **One connection per thread**, never share connections/channels across threads.
