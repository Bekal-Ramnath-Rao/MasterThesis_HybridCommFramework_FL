# Real Network Congestion Testing - Implementation Guide

## 🎯 Overview

Your FL experiments now support **REAL network congestion** using actual traffic-generating containers that compete for network resources. This goes beyond simulated delays (tc-based) to create authentic network load conditions.

## 🔧 What Was Implemented

### **1. Traffic Generator Containers**
Five types of containers that generate real network traffic:

#### **HTTP Traffic Generators** (`http-traffic-gen-1`, `http-traffic-gen-2`)
- **Purpose**: Simulate web browsing and API calls
- **Technology**: Alpine Linux with `curl`
- **Behavior**: Continuously makes HTTP requests to FL servers
- **Impact**: Creates TCP connection overhead, consumes bandwidth
- **Real-world analog**: Multiple users accessing web services

#### **Bandwidth Hog** (`bandwidth-hog`)
- **Purpose**: Simulate large file transfers
- **Technology**: Alpine Linux with `iperf3`
- **Behavior**: Sends UDP traffic at 20 Mbit/s for 10-second bursts
- **Impact**: Saturates network bandwidth, creates queuing delays
- **Real-world analog**: Video streaming, file downloads, backups

#### **Packet Spammer** (`packet-spammer`)
- **Purpose**: Create many small packets rapidly
- **Technology**: Alpine Linux with `ping`
- **Behavior**: Sends 64-byte ICMP packets at 10/sec (100 packets every 10s)
- **Impact**: Increases packet processing overhead, fills router queues
- **Real-world analog**: VoIP calls, gaming traffic, IoT devices

#### **Connection Flooder** (`connection-flooder`)
- **Purpose**: Simulate many concurrent connections
- **Technology**: Python with socket programming
- **Behavior**: Opens 10 concurrent TCP connections per target, repeatedly
- **Impact**: Exhausts connection pools, increases CPU load on servers
- **Real-world analog**: High user concurrency, DDoS-like conditions

### **2. Congestion Levels**

| Level | Containers Active | Real Impact |
|-------|------------------|-------------|
| **none** | 0 | Baseline - no artificial congestion |
| **light** | 1 (HTTP Gen 1) | Minimal - occasional HTTP requests |
| **moderate** | 3 (HTTP 1+2, Bandwidth Hog) | Noticeable - web traffic + file transfers |
| **heavy** | 4 (+ Packet Spammer) | Significant - all above + packet floods |
| **extreme** | 5 (+ Connection Flooder) | Severe - maximum load with connection saturation |

### **3. Integration Points**

#### **All Use Cases Supported**
✅ **Temperature** - `docker-compose-temperature.yml`
✅ **Emotion** - `docker-compose-emotion.yml`  
✅ **MentalState** - `docker-compose-mentalstate.yml`

Each has its own set of traffic generators with unique names:
- Temperature: `*-temp` (e.g., `http-traffic-gen-1-temp`)
- Emotion: `*-emotion` (e.g., `http-traffic-gen-1-emotion`)
- MentalState: `*-mental` (e.g., `http-traffic-gen-1-mental`)

#### **Docker Compose Profiles**
Traffic generators use the `congestion` profile:
```yaml
profiles:
  - congestion
```

This means they:
- **Don't start** with normal `docker-compose up`
- **Only start** when explicitly requested with `--profile congestion`
- **Won't interfere** with standard experiments

## 🚀 How to Use

### **Option 1: Integrated with Experiments (Recommended)**

```bash
# Run with moderate congestion
python Network_Simulation/run_network_experiments.py \
  --use-case temperature \
  --protocols mqtt grpc \
  --scenarios excellent moderate \
  --enable-congestion \
  --congestion-level moderate \
  --rounds 100
```

### **Option 2: Manual Control**

```bash
# Start moderate congestion manually
python Network_Simulation/congestion_manager.py \
  --use-case temperature \
  --start \
  --level moderate

# Check status
python Network_Simulation/congestion_manager.py \
  --use-case temperature \
  --status

# Stop all
python Network_Simulation/congestion_manager.py \
  --use-case temperature \
  --stop
```

### **Option 3: Direct Docker Compose**

```bash
# Start specific traffic generators with profile
docker-compose -f Docker/docker-compose-temperature.yml \
  --profile congestion \
  up -d http-traffic-gen-1-temp bandwidth-hog-temp

# Stop all congestion containers
docker-compose -f Docker/docker-compose-temperature.yml \
  --profile congestion \
  down
```

## 📊 Real Congestion vs. Simulated Conditions

### **TC-Based Simulation (network_simulator.py)**
- ✅ Precise control (exact latency, loss, bandwidth)
- ✅ Deterministic and repeatable
- ✅ Low overhead
- ❌ Doesn't test protocol behavior under load
- ❌ Doesn't simulate queuing dynamics
- ❌ Missing real-world interactions

### **Traffic-Based Congestion (NEW)**
- ✅ Tests real protocol congestion control
- ✅ Simulates competing applications
- ✅ Realistic queuing and buffering
- ✅ Tests connection management under load
- ✅ Reveals protocol robustness
- ❌ Less predictable/repeatable
- ❌ Higher resource usage
- ❌ Results may vary

### **Combined Approach (BEST)**
```bash
# Use BOTH for comprehensive testing
python Network_Simulation/run_network_experiments.py \
  --use-case temperature \
  --protocols mqtt amqp grpc \
  --scenarios moderate poor \
  --enable-congestion \
  --congestion-levels none moderate heavy \
  --rounds 100
```

This tests each protocol under:
1. `moderate` network (50ms latency, 20mbit, 1% loss) - NO congestion
2. `moderate` network + MODERATE congestion (HTTP + bandwidth hog)
3. `moderate` network + HEAVY congestion (+ packet spammer)

## 🔬 What This Tests

### **1. Congestion Control Mechanisms**
- **TCP-based protocols (MQTT, AMQP)**: How well do they handle packet loss and retransmission?
- **QUIC**: Does its built-in congestion control perform better than TCP?
- **gRPC (HTTP/2)**: How does multiplexing behave under congestion?

### **2. Resource Competition**
- Can FL traffic compete with background applications?
- Which protocol gets "starved" first under heavy load?
- How do protocols share bandwidth fairly?

### **3. Connection Resilience**
- Do connections stay alive under stress?
- How many reconnections occur?
- Which protocol handles connection floods better?

### **4. Throughput Under Load**
- Actual data transfer rates vs. theoretical
- Model convergence time with competing traffic
- Communication overhead increases

### **5. Scalability Indicators**
- How would protocols perform in shared networks?
- Real-world deployment considerations
- Multi-tenant environment readiness

## 📈 Expected Results

### **Protocols Under Congestion**

**MQTT (TCP-based)**
- Expected: Moderate impact, TCP backpressure helps
- Concern: Head-of-line blocking on congested connections
- Advantage: Simple protocol, low overhead

**AMQP (TCP-based)**  
- Expected: Similar to MQTT but higher overhead
- Concern: Complex protocol may add latency
- Advantage: Reliable message queuing

**gRPC (HTTP/2)**
- Expected: Good performance, multiplexing helps
- Concern: Connection setup overhead
- Advantage: Stream management, flow control

**QUIC (UDP-based)**
- Expected: Best under congestion
- Concern: May be too aggressive vs. TCP
- Advantage: Built-in congestion control, 0-RTT

**DDS**
- Expected: Variable, depends on QoS settings
- Concern: Discovery overhead in congested networks
- Advantage: Multicast efficiency (if used)

## 🎓 Research Value

### **Novel Contribution**
Most FL papers test under **idealized** network conditions. Your work evaluates protocols under:
- ✅ Simulated network degradation (tc)
- ✅ **Real traffic competition** (containers)
- ✅ Combined scenarios (both)

This is **closer to real deployments** where:
- Edge devices share network with other apps
- IoT networks have competing traffic
- Public networks are unpredictable

### **Thesis Impact**
You can claim:
1. **Comprehensive evaluation methodology**
2. **Real-world applicability** of results
3. **Protocol selection guidance** for specific scenarios
4. **Evidence of robustness** (or lack thereof)

## ⚙️ Dependencies

### **Already Installed in Containers**
- ✅ `curl` (Alpine `apk` package)
- ✅ `iperf3` (Alpine `apk` package)
- ✅ `ping` (Built into Alpine)
- ✅ Python 3.9 (Python Alpine image)

### **Auto-Installed at Runtime**
- ✅ `paho-mqtt` (Connection flooder)
- ✅ `pika` (Connection flooder)

### **No Additional Setup Required!**
Everything is self-contained in the Docker containers. They:
1. Pull base images (alpine, python:3.9-alpine)
2. Install tools via `apk add` or `pip install`
3. Start generating traffic immediately

## 🐛 Troubleshooting

### **Traffic Generators Won't Start**
```bash
# Check if profile is being used
docker ps -a --filter "name=traffic-gen"

# Manual start with profile
cd Docker
docker-compose -f docker-compose-temperature.yml --profile congestion up -d

# Check logs
docker logs http-traffic-gen-1-temp
```

### **No Visible Impact**
- Increase congestion level: `--congestion-level heavy` or `extreme`
- Combine with degraded network: `--scenarios poor congested_heavy`
- Check network bandwidth: Traffic may not saturate gigabit connections

### **Containers Exit Immediately**
```bash
# Check for errors
docker logs connection-flooder-temp

# Verify networks exist
docker network ls | grep fl-

# Restart with logs
docker-compose -f Docker/docker-compose-temperature.yml --profile congestion up http-traffic-gen-1-temp
```

### **Cleanup Issues**
```bash
# Force remove all congestion containers (temperature)
docker rm -f http-traffic-gen-1-temp http-traffic-gen-2-temp bandwidth-hog-temp packet-spammer-temp connection-flooder-temp

# For all use cases
docker ps -a | grep "traffic-gen\|bandwidth-hog\|packet-spammer\|connection-flooder" | awk '{print $1}' | xargs docker rm -f
```

## 📝 Summary

✅ **All 3 use cases** now have integrated traffic generators
✅ **5 congestion levels** from none to extreme  
✅ **4 types of traffic** simulating real applications
✅ **No additional dependencies** - everything is self-contained
✅ **Docker profiles** prevent interference with normal experiments
✅ **Automatic management** via congestion_manager.py
✅ **Full integration** with run_network_experiments.py

Your FL protocol evaluation now includes **real network congestion** - a unique and valuable addition to your research! 🎉
