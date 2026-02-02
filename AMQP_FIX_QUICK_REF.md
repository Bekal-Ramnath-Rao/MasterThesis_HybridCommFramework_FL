# AMQP Fix - Quick Reference

## What Was Fixed

**Problem**: `Stream connection lost: IndexError('pop from an empty deque')`  
**Cause**: Thread-unsafe access to same pika connection  
**Solution**: Separate connections for send/receive

## Changes

### 1. Two Connections Instead of One

```python
# Consumer connection (consumer thread owns)
self.amqp_connection = pika.BlockingConnection(...)
self.amqp_channel = connection.channel()

# Send connection (main thread owns)  
self.amqp_send_connection = pika.BlockingConnection(...)
self.amqp_send_channel = send_connection.channel()
```

### 2. Send Method Uses Send Channel

```python
# OLD (broken):
self.amqp_channel.basic_publish(...)  # Shared with consumer

# NEW (fixed):
self.amqp_send_channel.basic_publish(...)  # Dedicated channel
```

## Testing

```bash
# Rebuild and restart
cd Docker
docker ps -a --filter "name=unified" --format "{{.Names}}" | xargs -r docker rm -f
docker-compose -f docker-compose-unified-emotion.yml up --build -d

# Wait for startup
sleep 60

# Check logs - should see NO stream errors
docker logs fl-server-unified-emotion 2>&1 | grep -i "stream\|error\|amqp"

# Should see successful AMQP sends:
# [AMQP] Server started with separate send/receive connections
# [AMQP] Sent global_model to client X
# [AMQP] Sent start_evaluation to client X
```

## Verification

✅ No "Stream connection lost" errors  
✅ "[AMQP] Server started with separate send/receive connections"  
✅ AMQP clients receive global models  
✅ Rounds progress normally

## Files Changed

- **Server/Emotion_Recognition/FL_Server_Unified.py**
  - Added `amqp_send_connection` and `amqp_send_channel`
  - Updated `start_amqp_server()` to create both connections
  - Updated `send_via_amqp()` to use send channel
  - Added cleanup for send connection

## Key Principle

**pika.BlockingConnection = NOT thread-safe**

✅ One connection per thread  
❌ Never share across threads
