# Federated Learning with DDS (Data Distribution Service)

This implementation uses DDS (Data Distribution Service) with Eclipse CycloneDDS for federated learning, providing decentralized peer-to-peer communication.

## Protocol Comparison

| Feature | MQTT | AMQP | gRPC | DDS |
|---------|------|------|------|-----|
| Pattern | Pub/Sub | Message Queue | RPC | Pub/Sub (P2P) |
| Broker | Required | Required | Not Required | Not Required |
| Discovery | Broker | Broker | Manual | Automatic Multicast |
| Port | 1883 | 5672 | 50051 | 7400+ (dynamic) |
| QoS | 3 levels | Multiple | Limited | 22+ policies |
| Latency | Low | Medium | Low | **Very Low** |
| Reliability | Medium | High | High | **Very High** |
| Use Case | IoT | Enterprise | Microservices | **Real-time Systems** |

## Architecture

### DDS Decentralized Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    DDS Domain (ID: 0)                           │
│                   Automatic P2P Discovery                        │
│                                                                  │
│  ┌────────────┐         ┌────────────┐         ┌────────────┐  │
│  │  Server    │         │  Client 0  │         │  Client 1  │  │
│  │  (Peer)    │◄───────►│  (Peer)    │◄───────►│  (Peer)    │  │
│  └────────────┘         └────────────┘         └────────────┘  │
│       │                       │                       │          │
│       └───────────────────────┴───────────────────────┘         │
│                    All peers discover each other                │
│                    No central broker needed                     │
└─────────────────────────────────────────────────────────────────┘

Topics (Global):
├─ ClientRegistration     - Client → Server
├─ TrainingConfig         - Server → Clients
├─ TrainingCommand        - Server → Clients
├─ GlobalModel            - Server → Clients
├─ ModelUpdate            - Client → Server
├─ EvaluationMetrics      - Client → Server
└─ ServerStatus           - Server → Clients
```

### Key DDS Concepts

1. **Domain Participant**: Entry point to DDS (one per application)
2. **Topic**: Named data stream (e.g., "GlobalModel")
3. **DataWriter**: Publishes data to a topic
4. **DataReader**: Subscribes to data from a topic
5. **QoS Policies**: 22+ policies for reliability, durability, history, etc.

### DDS Discovery
- **Multicast-based**: Peers automatically find each other on the network
- **No configuration needed**: Just use same Domain ID
- **Dynamic**: Peers can join/leave at any time
- **Resilient**: No single point of failure

## Setup

### 1. Install CycloneDDS

**Windows (vcpkg):**
```powershell
# Install vcpkg if not already installed
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install CycloneDDS
.\vcpkg.exe install cyclonedds:x64-windows
```

**Windows (Pre-built):**
Download from: https://github.com/eclipse-cyclonedds/cyclonedds/releases

**Linux:**
```bash
sudo apt-get update
sudo apt-get install cyclonedds libcyclonedds-dev
```

### 2. Install Python Bindings

```powershell
pip install cyclonedds
```

### 3. Configure DLL Path (Windows)

The code automatically adds CycloneDDS binary path. Update if your installation differs:

**Server/FL_Server_DDS.py and Client/FL_Client_DDS.py:**
```python
# Update this path to match your CycloneDDS installation
cyclone_path = r"C:\Masters_Infotech\Semester_5\MT_SW_Addons\vcpkg\buildtrees\cyclonedds\x64-windows-rel\bin"
```

### 4. Verify Installation

```powershell
python -c "from cyclonedds.domain import DomainParticipant; print('DDS OK')"
```

## Running Federated Learning

### Local Testing

**Terminal 1 - Start Server:**
```powershell
cd Server
python FL_Server_DDS.py
```

Output:
```
======================================================================
Starting Federated Learning Server (DDS)
DDS Domain ID: 0
Number of Clients: 2
Number of Rounds: 5
======================================================================
Setting up DDS on domain 0...
DDS setup complete with RELIABLE QoS
```

**Terminal 2 - Start Client 0:**
```powershell
$env:CLIENT_ID="0"
cd Client
python FL_Client_DDS.py
```

**Terminal 3 - Start Client 1:**
```powershell
$env:CLIENT_ID="1"
cd Client
python FL_Client_DDS.py
```

### Distributed Setup (Different PCs)

**✨ DDS automatically discovers peers on the same network - no configuration needed!**

#### **PC 1 (Server):**
```powershell
$env:DDS_DOMAIN_ID="0"
$env:NUM_CLIENTS="2"
$env:NUM_ROUNDS="5"

cd Server
python FL_Server_DDS.py
```

#### **PC 2 (Client 0):**
```powershell
$env:CLIENT_ID="0"
$env:DDS_DOMAIN_ID="0"  # Must match server
$env:NUM_CLIENTS="2"

cd Client
python FL_Client_DDS.py
```

#### **PC 3 (Client 1):**
```powershell
$env:CLIENT_ID="1"
$env:DDS_DOMAIN_ID="0"  # Must match server
$env:NUM_CLIENTS="2"

cd Client
python FL_Client_DDS.py
```

**Note:** All participants with same `DDS_DOMAIN_ID` on the same network will automatically discover each other via multicast!

## Configuration

### Server Configuration (FL_Server_DDS.py)

```python
# Environment Variables
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
NUM_ROUNDS = int(os.getenv("NUM_ROUNDS", "5"))
```

### Client Configuration (FL_Client_DDS.py)

```python
# Environment Variables
DDS_DOMAIN_ID = int(os.getenv("DDS_DOMAIN_ID", "0"))
CLIENT_ID = int(os.getenv("CLIENT_ID", "0"))
NUM_CLIENTS = int(os.getenv("NUM_CLIENTS", "2"))
```

### Training Configuration
```python
training_config = {
    "batch_size": 32,
    "local_epochs": 20
}
```

## DDS Data Types

All messages are defined as Python `@dataclass` with `IdlStruct` base:

```python
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from dataclasses import dataclass

@dataclass
class GlobalModel(IdlStruct):
    round: int
    weights: sequence[int]  # Serialized numpy arrays
```

### Defined Types
1. **ClientRegistration** - Client announces presence
2. **TrainingConfig** - Server sends hyperparameters
3. **TrainingCommand** - Server instructs clients (train/evaluate/complete)
4. **GlobalModel** - Server sends aggregated weights
5. **ModelUpdate** - Clients send trained weights
6. **EvaluationMetrics** - Clients send evaluation results
7. **ServerStatus** - Server broadcasts current state

## Quality of Service (QoS)

DDS QoS policies ensure reliable communication:

```python
reliable_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
    Policy.History.KeepAll,
    Policy.Durability.TransientLocal
)
```

### QoS Policies Used

| Policy | Setting | Purpose |
|--------|---------|---------|
| **Reliability** | Reliable | Guarantees message delivery (vs BEST_EFFORT) |
| **History** | KeepAll | Keeps all messages until read (vs KEEP_LAST) |
| **Durability** | TransientLocal | Late joiners get historical data |

### Available QoS Policies (22+)
- Reliability, Durability, History, Deadline, Latency Budget
- Lifespan, Liveliness, Ownership, Time-based Filter
- Partition, Presentation, Resource Limits, and more

## Key Features

### 1. **Automatic Peer Discovery**
- No broker or server address configuration
- Multicast-based discovery (UDP)
- Dynamic join/leave

### 2. **Decentralized Architecture**
- No single point of failure
- True peer-to-peer
- Broker-less

### 3. **High Reliability**
- Configurable QoS policies
- Built-in acknowledgments
- Message persistence

### 4. **Low Latency**
- Direct peer-to-peer communication
- Shared memory optimization (same host)
- Zero-copy transfers (when possible)

### 5. **Type Safety**
- Strongly typed data structures
- IDL-based definitions
- Compile-time checks

## Output

### Server Output
- DDS domain and QoS information
- Client registration notifications
- Model aggregation status
- Plots: `results/dds_training_metrics.png`
- Results: `results/dds_training_results.csv`

### Client Output
- DDS setup confirmation
- Training progress per round
- Evaluation metrics
- Round synchronization status

## VS Code Debug Configurations

The project includes debug configurations in `.vscode/launch.json`:

### Individual Configurations
- `FL Server (DDS) - Local`
- `FL Server (DDS) - Remote` (same as local - automatic discovery)
- `FL Client 0 (DDS) - Local/Remote`
- `FL Client 1 (DDS) - Local/Remote`

### Compound Configurations
- `FL Server + Clients (DDS - Local)` - Launches all components

**Note:** Local and Remote configs are identical because DDS uses automatic peer discovery!

## Troubleshooting

### DDS DLL Not Found
```
CycloneDDSLoaderException: Could not load DLL
```
**Solution:**
1. Verify CycloneDDS installation path
2. Update `cyclone_path` in both server and client files
3. Ensure the DLL is at the specified location

### No Peer Discovery
```
Waiting for clients to register... (timeout)
```
**Solutions:**
1. **Check Domain ID**: All participants must use same `DDS_DOMAIN_ID`
2. **Firewall**: Allow UDP multicast (ports 7400-7500)
3. **Network**: Ensure multicast is enabled on network interface
4. **Subnet**: Peers must be on same subnet for default multicast

### Multicast Issues on Windows
```powershell
# Check multicast routes
route print

# Add multicast route if needed
route add 239.255.0.1 mask 255.255.255.255 <your_ip>
```

### AttributeError: 'DomainParticipant' has no attribute 'close'
```
AttributeError: 'DomainParticipant' object has no attribute 'close'
```
**Solution:** Already fixed in code. DomainParticipant auto-cleans up. If you see this, update your code.

### Type Errors with Weights
```
Exception: Cannot construct typeobject for bytes
```
**Solution:** Use `sequence[int]` instead of `bytes` or `List[int]` in IdlStruct definitions.

## Network Configuration

### Default Multicast
CycloneDDS uses multicast for discovery:
- **Address**: 239.255.0.1
- **Port Range**: 7400-7500 (UDP)

### Custom Network Interface
Create `cyclonedds.xml`:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain>
    <General>
      <NetworkInterfaceAddress>192.168.1.100</NetworkInterfaceAddress>
    </General>
  </Domain>
</CycloneDDS>
```

Set environment variable:
```powershell
$env:CYCLONEDDS_URI="file://path/to/cyclonedds.xml"
```

### Disable Multicast (Unicast Only)
For networks without multicast support:
```xml
<CycloneDDS>
  <Domain>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <Peers>
        <Peer address="192.168.1.100"/>
        <Peer address="192.168.1.101"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
```

## Performance Considerations

### Advantages
✅ **Lowest Latency**: Direct P2P, no broker overhead
✅ **Highest Reliability**: 22+ QoS policies
✅ **Automatic Discovery**: Zero configuration
✅ **Real-time Capable**: Used in aerospace, automotive
✅ **Scalable**: Thousands of peers
✅ **Resilient**: No single point of failure

### Limitations
❌ **Complex Setup**: Requires native DLL installation
❌ **Multicast Dependency**: Network must support multicast
❌ **Platform-specific**: Binary compatibility issues
❌ **Learning Curve**: More complex than MQTT

## Security

### Enable DDS Security
Create security configuration:
```xml
<CycloneDDS>
  <Domain>
    <Security>
      <Authentication>
        <Library>dds_security_auth</Library>
        <IdentityCertificate>file://identity_ca.pem</IdentityCertificate>
        <PrivateKey>file://identity.key</PrivateKey>
      </Authentication>
    </Security>
  </Domain>
</CycloneDDS>
```

## Advanced Configuration

### Change Domain ID
Use different domains to isolate different FL experiments:
```powershell
$env:DDS_DOMAIN_ID="1"  # Domain 1 won't see Domain 0 traffic
```

### Increase History Depth
```python
qos = Qos(Policy.History.KeepLast(100))  # Keep last 100 messages
```

### Add Deadlines
Detect slow clients:
```python
qos = Qos(Policy.Deadline(duration(seconds=5)))  # Must send within 5s
```

### Content Filtering
```python
reader = DataReader(
    participant, 
    topic,
    qos=qos,
    filter=Query("client_id = 0")  # Only read client 0 data
)
```

## DDS vs Other Protocols

### When to Use DDS
✅ Real-time requirements (< 1ms latency)
✅ High reliability needed (aerospace, medical)
✅ Decentralized architecture preferred
✅ Network supports multicast
✅ Need advanced QoS policies

### When to Use MQTT
✅ IoT devices with limited resources
✅ Internet-scale (WAN) communication
✅ Simple pub/sub sufficient

### When to Use AMQP
✅ Enterprise message queuing
✅ Complex routing needed
✅ Message persistence critical

### When to Use gRPC
✅ Microservices architecture
✅ Strong typing with protobuf
✅ HTTP/2 infrastructure

## Protocol Performance Comparison

| Metric | MQTT | AMQP | gRPC | **DDS** |
|--------|------|------|------|---------|
| Latency (ms) | 5-10 | 10-20 | 2-5 | **< 1** |
| Throughput | High | Medium | High | **Very High** |
| Reliability | Medium | High | High | **Very High** |
| Setup Time | Fast | Medium | Fast | **Slow** |
| Complexity | Low | High | Medium | **High** |

## IDL (Interface Definition Language)

While Python uses `@dataclass`, here's the equivalent IDL:

```idl
module FederatedLearning {
    struct GlobalModel {
        long round;
        sequence<octet> weights;
    };
    
    struct ModelUpdate {
        long client_id;
        long round;
        sequence<octet> weights;
        long num_samples;
        float loss;
        float mse;
        float mae;
        float mape;
    };
};
```

## References

- [Eclipse CycloneDDS](https://github.com/eclipse-cyclonedds/cyclonedds)
- [DDS Specification](https://www.omg.org/spec/DDS/)
- [CycloneDDS Python](https://github.com/eclipse-cyclonedds/cyclonedds-python)
- [DDS Tutorial](https://cyclonedds.io/docs/cyclonedds/latest/)

## Real-World DDS Applications

- **Automotive**: ADAS, autonomous driving
- **Aerospace**: Flight control, avionics
- **Medical**: Surgical robots, monitoring
- **Industrial**: Factory automation, robotics
- **Military**: Command & control systems
- **Financial**: High-frequency trading

DDS is the protocol of choice when **reliability and real-time performance are non-negotiable**!
