# Dynamic Protocol Selection Engine for Federated Learning

## Overview

This decision engine dynamically selects the optimal communication protocol (MQTT, AMQP, gRPC, QUIC, DDS) for federated learning based on real-time network conditions, device resources, model characteristics, and mobility patterns.

## Architecture

### 1. Decision Criteria (Weighted Multi-Criteria Scoring)

```
Total Score = W₁×Network + W₂×Resources + W₃×Model + W₄×Mobility

Where:
- W₁ (Network) = 0.40 (40%)
- W₂ (Resources) = 0.25 (25%)  
- W₃ (Model) = 0.20 (20%)
- W₄ (Mobility) = 0.15 (15%)
```

### 2. Measurement Parameters

#### Network Conditions
- **Latency**: Round-trip time to server (ms) - measured via ping
- **Bandwidth**: Available bandwidth (Mbps) - estimated from network interface
- **Packet Loss**: Percentage of lost packets - measured via ping statistics
- **Jitter**: Latency variance (ms) - calculated from multiple pings

#### Resource Availability
- **CPU Usage**: Current CPU utilization (%) - via psutil
- **Memory**: Available RAM (MB) and usage (%) - via psutil
- **Battery Level**: Battery percentage if on mobile device - via psutil.sensors_battery()
- **Charging Status**: Whether device is plugged in - via psutil

#### Model Characteristics
- **Model Size**: Size of FL model in MB - provided by FL framework
- **Update Frequency**: How often model updates occur (seconds)

#### Mobility Patterns
- **Is Mobile**: Whether device is moving or stationary
- **Velocity**: Speed of movement (km/h) - from GPS/accelerometer
- **Connection Stability**: Score 0-100 based on network handoffs
- **Handoff Rate**: Network handoffs per hour

## Protocol Profiles

Each protocol has characteristics suited for specific scenarios:

### MQTT (Message Queuing Telemetry Transport)
**Best For**: IoT devices, low bandwidth, mobile scenarios
- ✅ Very low overhead and battery usage
- ✅ Excellent for constrained devices
- ✅ Handles disconnections well (QoS levels)
- ✅ Good for sporadic connectivity
- ⚠️ Moderate latency
- ❌ Not optimized for large payloads

**Use When**:
- Low bandwidth (< 5 Mbps)
- Resource-constrained devices
- High mobility scenarios
- Battery-powered devices

### AMQP (Advanced Message Queuing Protocol)
**Best For**: Reliable message delivery, enterprise scenarios
- ✅ Strong reliability guarantees
- ✅ Good message routing capabilities
- ✅ Moderate resource usage
- ⚠️ More overhead than MQTT
- ⚠️ Moderate latency

**Use When**:
- Reliability is critical
- Need advanced routing
- Moderate network conditions
- Stable power supply

### gRPC (Google Remote Procedure Call)
**Best For**: High-performance, stable networks, large models
- ✅ HTTP/2 multiplexing and streaming
- ✅ Excellent for large model transfers
- ✅ Low latency on good networks
- ✅ Efficient serialization (Protocol Buffers)
- ❌ Higher resource usage
- ❌ Struggles with poor connectivity

**Use When**:
- Large models (> 50 MB)
- Stable, high-bandwidth networks
- Sufficient CPU/memory resources
- Stationary devices

### QUIC (Quick UDP Internet Connections)
**Best For**: Mobile scenarios, packet loss tolerance, low latency
- ✅ Excellent low-latency performance
- ✅ Best-in-class packet loss handling
- ✅ Fast connection establishment (0-RTT)
- ✅ Excellent for mobility (connection migration)
- ⚠️ Moderate resource usage
- ⚠️ May be blocked by some firewalls

**Use When**:
- High packet loss (> 1%)
- Mobile/unstable connections
- Low latency required
- Frequent network changes

### DDS (Data Distribution Service)
**Best For**: Real-time systems, distributed scenarios
- ✅ Real-time performance
- ✅ Data-centric pub-sub
- ✅ QoS policies
- ❌ Higher overhead
- ❌ More complex setup

**Use When**:
- Real-time requirements
- Complex data flows
- Distributed architectures
- Sufficient resources available

## Scoring Algorithm

### Network Score Calculation

```python
def score_network(protocol, conditions):
    score = 50  # baseline
    
    if latency > 100ms:
        score = protocol.high_latency_score
    
    if packet_loss > 1%:
        score = avg(score, protocol.packet_loss_score)
    
    if bandwidth < 5 Mbps:
        score = avg(score, protocol.low_bandwidth_score)
    
    return min(100, max(0, score))
```

### Resource Score Calculation

```python
def score_resources(protocol, resources):
    score = 70  # baseline
    
    if cpu_usage > 70%:
        score = avg(score, protocol.low_cpu_score)
    
    if memory_usage > 80%:
        score = avg(score, protocol.low_memory_score)
    
    if battery < 30% and not charging:
        score = avg(score, protocol.battery_efficient_score)
    
    return score
```

### Example Scoring Matrix

| Condition | MQTT | AMQP | gRPC | QUIC | DDS |
|-----------|------|------|------|------|-----|
| Low Bandwidth | 95 | 75 | 70 | 80 | 60 |
| High Latency | 80 | 75 | 85 | 95 | 90 |
| Packet Loss | 85 | 80 | 65 | 95 | 80 |
| Low CPU | 95 | 80 | 60 | 70 | 50 |
| Low Memory | 95 | 80 | 65 | 70 | 55 |
| Large Model | 60 | 70 | 90 | 85 | 75 |
| High Mobility | 90 | 75 | 50 | 95 | 70 |
| Battery Efficient | 95 | 80 | 60 | 70 | 55 |

## Usage

### Basic Protocol Selection

```python
from protocol_selector import ProtocolSelector

# Create selector
selector = ProtocolSelector(server_address="192.168.1.100")

# Select best protocol
protocol, score = selector.select_best_protocol(
    model_size_mb=25.0,
    verbose=True
)

print(f"Selected: {protocol}")  # e.g., "mqtt"
print(f"Score: {score.total_score}")  # e.g., 87.5
```

### Dynamic FL Client with Auto-Selection

```python
from dynamic_fl_client import DynamicFLClient

# Create dynamic client
client = DynamicFLClient(
    client_id=1,
    server_address="192.168.1.100",
    model_size_mb=25.0,
    reevaluate_every=5  # Re-evaluate every 5 rounds
)

# Initialize (selects initial protocol)
client.initialize()

# Training loop
for round in range(100):
    client.run_training_round(round)
    # Protocol automatically re-evaluated every 5 rounds

# View statistics
client.get_statistics()
```

### Manual Condition Specification

```python
# Specify conditions manually for testing
scores = selector.calculate_protocol_scores(
    network={
        "latency": 120,      # ms
        "bandwidth": 2.0,    # Mbps
        "packet_loss": 2.0,  # %
        "jitter": 15         # ms
    },
    resources={
        "cpu_usage": 75,           # %
        "memory_available": 512,    # MB
        "memory_percent": 85,       # %
        "battery_level": 25,        # %
        "is_charging": False
    },
    model_size_mb=50.0,
    mobility={
        "is_mobile": True,
        "velocity": 30,              # km/h
        "connection_stability": 50,  # 0-100
        "handoffs_per_hour": 5
    }
)

# Get best protocol
best = max(scores.items(), key=lambda x: x[1].total_score)
print(f"Best: {best[0]} with score {best[1].total_score}")
```

## Integration with FL Framework

### Option 1: Client-Side Selection (Recommended)

Each client independently selects protocol based on local conditions:

```python
# In FL_Client.py
from protocol_selector import ProtocolSelector

class FederatedLearningClient:
    def __init__(self, client_id):
        self.selector = ProtocolSelector()
        self.protocol = self.selector.select_best_protocol()[0]
        self.init_protocol_client(self.protocol)
```

**Advantages**:
- Adapts to each client's unique conditions
- No server coordination needed
- Scales well with many clients

**Considerations**:
- Server must support all protocols
- Clients must have all protocol libraries

### Option 2: Server-Guided Selection

Server recommends protocol based on client profiles:

```python
# Server sends recommendation
client_profile = client.get_conditions()
recommended_protocol = server.recommend_protocol(client_profile)
client.switch_to(recommended_protocol)
```

### Option 3: Hybrid Approach

Client selects from server-approved protocols:

```python
# Server provides allowed protocols
allowed = server.get_allowed_protocols()  # ['mqtt', 'quic']
protocol = selector.select_best_protocol(
    allowed_protocols=allowed
)[0]
```

## Mobility Detection Methods

### Method 1: GPS-Based (Mobile Devices)

```python
import gpsd

def detect_mobility_gps():
    gpsd.connect()
    packet = gpsd.get_current()
    
    return {
        "velocity": packet.speed() * 3.6,  # m/s to km/h
        "is_mobile": packet.speed() > 1.0   # > 1 m/s
    }
```

### Method 2: Network Interface Monitoring

```python
def detect_mobility_network():
    # Monitor WiFi signal strength changes
    # Track network handoffs (SSID changes)
    # Check interface type (WiFi vs Cellular)
    
    handoffs = count_network_changes(last_5_minutes)
    signal_variance = get_signal_strength_variance()
    
    return {
        "is_mobile": handoffs > 2 or signal_variance > 20,
        "handoffs_per_hour": handoffs * 12
    }
```

### Method 3: Accelerometer-Based

```python
from adafruit_lsm6ds import LSM6DS33

def detect_mobility_accelerometer():
    sensor = LSM6DS33(i2c)
    accel_x, accel_y, accel_z = sensor.acceleration
    
    movement = sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    
    return {
        "is_mobile": movement > threshold,
        "movement_intensity": movement
    }
```

## Evaluation Scenarios

### Scenario 1: Stationary IoT Device
```yaml
Network: Good (10ms latency, 0% loss)
Resources: Constrained (80% CPU, 90% memory)
Model: Small (10 MB)
Mobility: Stationary

Expected: MQTT (low resource usage)
Score: MQTT=92, AMQP=78, gRPC=65, QUIC=72, DDS=58
```

### Scenario 2: Mobile Phone on 4G
```yaml
Network: Moderate (80ms latency, 1% loss)
Resources: Adequate (40% CPU, 50% memory)
Model: Medium (25 MB)
Mobility: Mobile (30 km/h, 3 handoffs/hour)

Expected: QUIC (mobility + packet loss handling)
Score: QUIC=88, MQTT=82, AMQP=72, gRPC=58, DDS=65
```

### Scenario 3: Server with Stable Connection
```yaml
Network: Excellent (5ms latency, 0% loss, 100 Mbps)
Resources: Abundant (20% CPU, 30% memory)
Model: Large (100 MB)
Mobility: Stationary

Expected: gRPC (large model transfer)
Score: gRPC=94, QUIC=88, DDS=82, AMQP=75, MQTT=68
```

### Scenario 4: Satellite Connection
```yaml
Network: Poor (500ms latency, 3% loss, 2 Mbps)
Resources: Moderate (50% CPU, 60% memory)
Model: Medium (30 MB)
Mobility: Stationary

Expected: MQTT (latency tolerance + low bandwidth)
Score: MQTT=86, QUIC=82, AMQP=74, DDS=68, gRPC=52
```

## Advanced Features

### 1. Adaptive Weight Tuning

Automatically adjust weights based on past performance:

```python
class AdaptiveSelector(ProtocolSelector):
    def tune_weights(self, performance_history):
        # Use reinforcement learning or heuristics
        # to optimize weights based on actual results
        pass
```

### 2. Hysteresis for Protocol Switching

Prevent frequent switching (flapping):

```python
def should_switch(current, new, threshold=10):
    # Only switch if new protocol is significantly better
    return new.total_score - current.total_score > threshold
```

### 3. Time-of-Day Awareness

Consider network congestion patterns:

```python
def get_time_factor():
    hour = datetime.now().hour
    # Peak hours: reduce bandwidth-intensive protocols
    if 9 <= hour <= 17:
        return {"bandwidth_penalty": 0.8}
    return {"bandwidth_penalty": 1.0}
```

### 4. Cost Awareness

Factor in data costs (important for cellular):

```python
protocol_costs = {
    "mqtt": 1.0,   # Low data usage
    "amqp": 1.2,
    "grpc": 1.5,   # High data usage
    "quic": 1.3,
    "dds": 1.4
}
```

## Testing & Validation

### Unit Tests

```bash
python -m pytest tests/test_protocol_selector.py
```

### Integration Tests

```bash
python tests/test_dynamic_client.py
```

### Benchmarking

```bash
python benchmarks/compare_protocols.py \
    --scenarios all \
    --repetitions 10
```

## Future Enhancements

1. **Machine Learning Integration**: Train ML model to predict best protocol from historical data
2. **Multi-Objective Optimization**: Use Pareto optimization for conflicting objectives
3. **Federated Protocol Learning**: Learn optimal selection across all clients
4. **Energy Modeling**: Detailed battery consumption models per protocol
5. **Security Scoring**: Add security/privacy as selection criteria

## References

- MQTT Specification: https://mqtt.org/mqtt-specification/
- AMQP Specification: https://www.amqp.org/
- gRPC Documentation: https://grpc.io/docs/
- QUIC Protocol: https://www.chromium.org/quic/
- DDS Specification: https://www.omg.org/spec/DDS/

## Citation

If you use this decision engine in your research, please cite:

```bibtex
@software{fl_protocol_selector,
  title={Dynamic Protocol Selection Engine for Federated Learning},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/fl-protocol-selector}
}
```
