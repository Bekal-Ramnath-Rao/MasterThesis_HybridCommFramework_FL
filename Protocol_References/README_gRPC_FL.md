# Federated Learning with gRPC

This implementation uses gRPC (Google Remote Procedure Call) for federated learning communication, providing a direct RPC-based approach.

## Protocol Comparison

| Feature | MQTT | AMQP | gRPC |
|---------|------|------|------|
| Pattern | Pub/Sub | Message Queue | RPC (Request-Response) |
| Broker | Required | Required | Not Required |
| Port | 1883 | 5672 | 50051 (default) |
| Communication | Async | Async | Sync/Async |
| Message Format | Binary | Binary | Protocol Buffers |
| Connection | Many-to-many | Many-to-many | Point-to-point |
| Discovery | Broker-based | Broker-based | Client must know server address |

## Architecture

### Communication Flow

```
┌──────────────────────────────────────────────────────────────┐
│                       gRPC Server                            │
│                    (FL_Server_gRPC.py)                       │
│                      Port: 50051                             │
│                                                              │
│  RPCs:                                                       │
│  ├─ RegisterClient(ClientRegistration)                      │
│  ├─ GetGlobalModel(ClientInfo) → GlobalModel                │
│  ├─ SendModelUpdate(ModelUpdate) → Ack                      │
│  ├─ SendEvaluationMetrics(EvaluationMetrics) → Ack          │
│  └─ CheckTrainingStatus(ClientInfo) → TrainingStatus        │
└──────────────────┬───────────────────────────────────────────┘
                   │
       ┌───────────┴──────────┬────────────────┐
       │                      │                │
       ▼                      ▼                ▼
┌─────────────┐      ┌─────────────┐   ┌─────────────┐
│  Client 0   │      │  Client 1   │   │  Client N   │
│             │      │             │   │             │
│ Polling     │      │ Polling     │   │ Polling     │
│ loop (0.5s) │      │ loop (0.5s) │   │ loop (0.5s) │
└─────────────┘      └─────────────┘   └─────────────┘
```

### gRPC Services

**FederatedLearningService:**
1. **RegisterClient**: Clients register with server
2. **GetGlobalModel**: Clients request latest global model
3. **SendModelUpdate**: Clients send trained model weights
4. **SendEvaluationMetrics**: Clients send evaluation results
5. **CheckTrainingStatus**: Clients poll for training instructions

### Client Polling Pattern
Since gRPC doesn't have native pub/sub, clients poll the server every 0.5 seconds to check:
- Should I start training?
- Should I evaluate the model?
- Is training complete?

## Setup

### 1. Install gRPC Dependencies

```powershell
pip install grpcio grpcio-tools protobuf
```

### 2. Compile Protocol Buffers

The `.proto` file defines the service interface and message types. Compile it to generate Python code:

```powershell
cd Protocols
python compile_proto.py
```

This generates:
- `federated_learning_pb2.py` - Message classes
- `federated_learning_pb2_grpc.py` - Service stubs

### 3. Verify Generated Files

Check that these files exist:
```
Protocols/
├── federated_learning.proto
├── federated_learning_pb2.py
└── federated_learning_pb2_grpc.py
```

## Running Federated Learning

### Local Testing

**Terminal 1 - Start Server:**
```powershell
cd Server
python FL_Server_gRPC.py
```

Output:
```
======================================================================
Starting Federated Learning Server (gRPC)
Server Address: localhost:50051
Number of Clients: 2
Number of Rounds: 5
======================================================================
gRPC Server started on localhost:50051
```

**Terminal 2 - Start Client 0:**
```powershell
$env:CLIENT_ID="0"
cd Client
python FL_Client_gRPC.py
```

**Terminal 3 - Start Client 1:**
```powershell
$env:CLIENT_ID="1"
cd Client
python FL_Client_gRPC.py
```

### Distributed Setup (Different PCs)

#### **PC 1 (Server - 192.168.0.101):**
```powershell
$env:GRPC_HOST="0.0.0.0"  # Listen on all interfaces
$env:GRPC_PORT="50051"
$env:NUM_CLIENTS="2"
$env:NUM_ROUNDS="5"

cd Server
python FL_Server_gRPC.py
```

#### **PC 2 (Client 0):**
```powershell
$env:CLIENT_ID="0"
$env:GRPC_HOST="192.168.0.101"  # Server IP
$env:GRPC_PORT="50051"
$env:NUM_CLIENTS="2"

cd Client
python FL_Client_gRPC.py
```

#### **PC 3 (Client 1):**
```powershell
$env:CLIENT_ID="1"
$env:GRPC_HOST="192.168.0.101"  # Server IP
$env:GRPC_PORT="50051"
$env:NUM_CLIENTS="2"

cd Client
python FL_Client_gRPC.py
```

## Configuration

### Server Configuration (FL_Server_gRPC.py)

```python
# Environment Variables
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = os.getenv("GRPC_PORT", "50051")
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
```

### Client Configuration (FL_Client_gRPC.py)

```python
# Environment Variables
GRPC_HOST = os.getenv("GRPC_HOST", "localhost")
GRPC_PORT = os.getenv("GRPC_PORT", "50051")
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
```

### Training Configuration
Modify in server file:
```python
training_config = {
    "batch_size": 32,
    "local_epochs": 20
}
```

## Protocol Buffer Definition

The `federated_learning.proto` file defines:

### Messages
```protobuf
message ClientRegistration {
    int32 client_id = 1;
    string message = 2;
}

message GlobalModel {
    int32 round = 1;
    bytes weights = 2;  // Serialized numpy arrays
}

message ModelUpdate {
    int32 client_id = 1;
    int32 round = 2;
    bytes weights = 3;
    int32 num_samples = 4;
    float loss = 5;
    float mse = 6;
    float mae = 7;
    float mape = 8;
}

message EvaluationMetrics {
    int32 client_id = 1;
    int32 round = 2;
    int32 num_samples = 3;
    float loss = 4;
    float mse = 5;
    float mae = 6;
    float mape = 7;
}

message TrainingStatus {
    bool start_training = 1;
    bool start_evaluation = 2;
    bool training_complete = 3;
    int32 current_round = 4;
}
```

### Service
```protobuf
service FederatedLearningService {
    rpc RegisterClient(ClientRegistration) returns (Acknowledgment);
    rpc GetGlobalModel(ClientInfo) returns (GlobalModel);
    rpc SendModelUpdate(ModelUpdate) returns (Acknowledgment);
    rpc SendEvaluationMetrics(EvaluationMetrics) returns (Acknowledgment);
    rpc CheckTrainingStatus(ClientInfo) returns (TrainingStatus);
}
```

## Key Features

### 1. Strongly Typed Communication
- Protocol Buffers provide type safety
- Automatic serialization/deserialization
- Efficient binary encoding

### 2. Direct Client-Server Communication
- No broker required
- Lower latency than broker-based systems
- Simpler deployment

### 3. Request-Response Pattern
- Clear RPC semantics
- Easy error handling
- Built-in timeout support

### 4. Polling-Based Synchronization
- Clients poll server for instructions
- 0.5-second polling interval
- Ensures clients stay synchronized

## Output

### Server Output
- Client registrations
- Model aggregation status
- Round completion
- Plots: `results/grpc_training_metrics.png`
- Results: `results/grpc_training_results.csv`

### Client Output
- Connection status
- Training progress
- Evaluation metrics
- Round updates

## VS Code Debug Configurations

The project includes debug configurations in `.vscode/launch.json`:

### Individual Configurations
- `FL Server (gRPC) - Local`
- `FL Server (gRPC) - Remote`
- `FL Client 0 (gRPC) - Local/Remote`
- `FL Client 1 (gRPC) - Local/Remote`

### Compound Configurations
- `FL Server + Clients (gRPC - Local)` - Launches all components

## Troubleshooting

### Port Already in Use
```
Error: [Errno 10048] Only one usage of each socket address
```
**Solution:**
```powershell
# Find process using port 50051
netstat -ano | findstr :50051

# Kill the process
taskkill /PID <PID> /F
```

### Connection Refused
```
grpc._channel._InactiveRpcError: failed to connect to all addresses
```
**Solutions:**
1. Ensure server is running
2. Check firewall allows port 50051
3. Verify GRPC_HOST and GRPC_PORT match

### Protocol Buffer Compilation Errors
```
ModuleNotFoundError: No module named 'federated_learning_pb2'
```
**Solution:**
```powershell
cd Protocols
python compile_proto.py
```

### Stub Import Errors
```
ImportError: cannot import name 'FederatedLearningServiceServicer'
```
**Solution:** Recompile proto files with latest grpcio-tools

## Performance Considerations

### Advantages
✅ **Low Latency**: Direct RPC calls, no broker overhead
✅ **Type Safety**: Protocol Buffers catch errors early
✅ **Efficiency**: Binary encoding is compact
✅ **HTTP/2**: Multiplexing, header compression

### Limitations
❌ **No Pub/Sub**: Clients must poll for updates
❌ **Point-to-Point**: Each client needs server address
❌ **No Message Queue**: No built-in message persistence
❌ **Polling Overhead**: 0.5s polling interval adds overhead

## Security

### Enable TLS/SSL

**Server:**
```python
server_credentials = grpc.ssl_server_credentials([
    (private_key, certificate_chain)
])
server.add_secure_port(f'{host}:{port}', server_credentials)
```

**Client:**
```python
credentials = grpc.ssl_channel_credentials(root_certificates)
channel = grpc.secure_channel(f'{host}:{port}', credentials)
```

## Advanced Configuration

### Change Polling Interval
In `FL_Client_gRPC.py`:
```python
time.sleep(0.5)  # Change to 0.1 for faster polling, 1.0 for less frequent
```

### Add More Clients
1. Set `NUM_CLIENTS` environment variable
2. Start each client with unique `CLIENT_ID` (0, 1, 2, ...)

### Increase Training Rounds
```powershell
$env:NUM_ROUNDS="10"
```

### Use Different Port
```powershell
$env:GRPC_PORT="8080"
```

## Protocol Comparison Summary

| Aspect | gRPC | MQTT | AMQP | DDS |
|--------|------|------|------|-----|
| Setup Complexity | Low | Medium | High | High |
| Network Discovery | Manual | Broker | Broker | Automatic |
| Message Pattern | RPC | Pub/Sub | Queue | Pub/Sub |
| Latency | Low | Low | Medium | Very Low |
| Reliability | High | Medium | High | Very High |
| Scalability | Medium | High | High | High |

## References

- [gRPC Documentation](https://grpc.io/docs/)
- [Protocol Buffers Guide](https://developers.google.com/protocol-buffers)
- [gRPC Python Tutorial](https://grpc.io/docs/languages/python/basics/)
