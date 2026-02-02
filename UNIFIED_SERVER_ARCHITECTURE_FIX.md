# Unified Server Architecture Fix

## Problem Analysis

### Current Issues

1. **AMQP Thread Safety**: 
   - Consumer thread calls `start_consuming()` 
   - Main thread calls `basic_publish()`
   - pika's BlockingConnection is NOT thread-safe → Stream connection lost errors

2. **Inconsistent with Single-Protocol Pattern**:
   - Single-protocol servers: each protocol owns its connection/thread
   - Unified server: mixing thread access to same connection

3. **Connection Management**:
   - Trying to recreate connections/channels on error
   - Not following the persistent connection pattern of single-protocol servers

## Solution: Mirror Single-Protocol Architecture

### Architecture Principle

**Each protocol listener should be a complete, independent copy of its single-protocol implementation**

```
Unified Server
├── MQTT Thread (loop_forever)
├── AMQP Thread (start_consuming) - OWN connection/channel
├── gRPC Thread (wait_for_termination)
├── QUIC Thread (asyncio.Future)
└── DDS Thread (waitset loop)
```

### Key Changes Required

#### 1. AMQP: Separate Connection for Sending

**Problem**: Consumer thread owns the connection, send operations from main thread fail

**Solution**: Create TWO connections like single-protocol AMQP server pattern:
- Connection 1 (consumer thread): For `start_consuming()`
- Connection 2 (main/send thread): For `basic_publish()`

OR use callback-based sending within the consumer thread

**Reference**: `FL_Server_AMQP.py` lines 112-148 (connect method)

#### 2. QUIC: Complete Server Protocol

**Current**: Partial implementation  
**Needed**: Full copy of `FL_Server_QUIC.py` server protocol class

**Reference**: `FL_Server_QUIC.py` lines 28-140 (FederatedLearningServerProtocol)

#### 3. Thread-Safe Message Queue

Instead of directly calling send methods from main thread, use queues:

```python
class UnifiedServer:
    def __init__(self):
        self.send_queue_mqtt = queue.Queue()
        self.send_queue_amqp = queue.Queue()
        self.send_queue_grpc = queue.Queue()
        self.send_queue_quic = queue.Queue()
        self.send_queue_dds = queue.Queue()
    
    def send_via_amqp(self, client_id, message_type, message):
        # Put in queue instead of direct send
        self.send_queue_amqp.put((client_id, message_type, message))
    
    def _amqp_sender_thread(self):
        # Thread that owns AMQP connection
        connection = pika.BlockingConnection(...)
        channel = connection.channel()
        
        while True:
            client_id, msg_type, message = self.send_queue_amqp.get()
            # Send using this thread's owned connection
            channel.basic_publish(...)
```

## Detailed Implementation Plan

### Step 1: Copy Single-Protocol Server Classes

Each protocol should have its complete server class from single-protocol implementation:

```python
# From FL_Server_MQTT.py
class MQTTServerHandler:
    def __init__(self, unified_server):
        self.server = unified_server  # Reference to unified server
        self.mqtt_client = mqtt.Client()
        # ... exact copy of single-protocol setup
    
    def start(self):
        self.mqtt_client.loop_forever()  # Blocks in its own thread
```

### Step 2: Unified Server Coordinates

```python
class UnifiedFederatedLearningServer:
    def __init__(self):
        # Core FL logic (aggregation, convergence, etc.)
        self.mqtt_handler = MQTTServerHandler(self)
        self.amqp_handler = AMQPServerHandler(self)
        self.grpc_handler = gRPCServerHandler(self)
        self.quic_handler = QUICServerHandler(self)
        self.dds_handler = DDSServerHandler(self)
    
    def run(self):
        # Start each protocol in its own thread
        threading.Thread(target=self.mqtt_handler.start, daemon=True).start()
        threading.Thread(target=self.amqp_handler.start, daemon=True).start()
        # ... etc
        
        # Main thread waits
        while not self.converged:
            time.sleep(1)
```

### Step 3: Callback Pattern

Protocol handlers call back to unified server for FL logic:

```python
class AMQPServerHandler:
    def on_client_update(self, ch, method, properties, body):
        data = json.loads(body.decode())
        # Call unified server's method
        self.server.handle_client_update(data, 'amqp')
    
    def send_global_model(self, client_id, model_data):
        # Use THIS handler's owned connection
        self.channel.basic_publish(...)
```

## Specific Fixes for Current Errors

### AMQP Stream Connection Lost

**Current Code** (FL_Server_Unified.py:568-590):
```python
def send_via_amqp(self, client_id, message_type, message):
    # Uses self.amqp_channel which is owned by consumer thread
    self.amqp_channel.basic_publish(...)  # UNSAFE - different thread!
```

**Fix Option A - Separate Connection**:
```python
def start_amqp_sender(self):
    # New thread with own connection
    self.amqp_send_connection = pika.BlockingConnection(parameters)
    self.amqp_send_channel = self.amqp_send_connection.channel()

def send_via_amqp(self, client_id, message_type, message):
    # Use send_channel, not consumer channel
    self.amqp_send_channel.basic_publish(...)
```

**Fix Option B - Queue-Based**:
```python
def send_via_amqp(self, client_id, message_type, message):
    self.amqp_send_queue.put((client_id, message_type, message))

def _amqp_sender_loop(self):
    # Runs in separate thread with own connection
    while True:
        item = self.amqp_send_queue.get()
        self.send_channel.basic_publish(...)
```

### QUIC Client Protocol Storage

**Current**: Storing in `self.quic_clients` dict  
**Better**: Let QUIC handler manage its own clients like single-protocol server

**Reference**: `FL_Server_QUIC.py` - server stores protocols internally

## File Structure After Fix

```
FL_Server_Unified.py
├── Core FL Logic
│   ├── __init__
│   ├── initialize_global_model()
│   ├── aggregate_weights()
│   ├── aggregate_metrics()
│   └── continue_training()
│
├── Protocol Handlers (each copied from single-protocol)
│   ├── MQTTHandler (from FL_Server_MQTT.py)
│   ├── AMQPHandler (from FL_Server_AMQP.py)
│   ├── gRPCHandler (from FL_Server_gRPC.py)
│   ├── QUICHandler (from FL_Server_QUIC.py)
│   └── DDSHandler (from FL_Server_DDS.py)
│
└── Coordination
    ├── run() - starts all handlers
    └── Callbacks from handlers to FL logic
```

## Testing Checklist

After implementing fixes:

- [ ] AMQP: No "Stream connection lost" errors
- [ ] AMQP: Multiple sends without channel errors
- [ ] QUIC: Clients can send and receive
- [ ] All protocols: Can run simultaneously
- [ ] Mixed clients: Some AMQP, some QUIC, some others
- [ ] Rounds progress with mixed protocols
- [ ] No thread-safety issues
- [ ] Clean shutdown of all threads

## Implementation Priority

1. **Immediate**: Fix AMQP thread-safety (Option A or B above)
2. **High**: Verify QUIC persistent connections work
3. **Medium**: Refactor to handler pattern for cleaner separation
4. **Low**: Optimize with connection pooling if needed

## Key Principle

> **"Each protocol handler should be indistinguishable from running its single-protocol server independently"**

The only difference: callbacks go to unified server instead of local methods.

## References

- `FL_Server_MQTT.py` - MQTT pattern
- `FL_Server_AMQP.py` - AMQP connection management
- `FL_Server_gRPC.py` - gRPC server pattern
- `FL_Server_QUIC.py` - QUIC protocol and asyncio
- `FL_Server_DDS.py` - DDS waitset pattern

All handlers should mirror these implementations exactly.
