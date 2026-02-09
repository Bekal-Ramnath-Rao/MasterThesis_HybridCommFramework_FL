# RL-Based Protocol Selection for Federated Learning

## Overview

This system implements **Q-Learning based dynamic protocol selection** for Federated Learning across three use cases:
- **Emotion Recognition** (Facial expression classification)
- **Mental State Recognition** (EEG-based classification)  
- **Temperature Regulation** (IoT control system)

### Supported Protocols
1. **MQTT** - Lightweight messaging for IoT
2. **AMQP** - Advanced message queuing
3. **gRPC** - High-performance RPC
4. **QUIC** - Fast, secure transport
5. **DDS** - Real-time data distribution

---

## Architecture

### 1. Q-Learning Protocol Selector (`rl_q_learning_selector.py`)

**Purpose**: Intelligently select the best communication protocol based on environment conditions

**Actions**: 5 protocols (MQTT, AMQP, gRPC, QUIC, DDS)

**Rewards**:
- ✅ **Communication Time**: Faster round-trip = higher reward
- ✅ **Success Rate**: Successful communication = positive reward  
- ✅ **Convergence Time**: Faster model convergence = higher reward
- ✅ **Model Accuracy**: Higher accuracy = higher reward
- ✅ **Resource Consumption**: Lower CPU/memory/bandwidth = higher reward

**State Space**:
- **Network Condition**: excellent, good, moderate, poor, very_poor
- **Resource Level**: high, medium, low
- **Model Size**: small, medium, large
- **Mobility**: static, low, medium, high

**Q-Learning Parameters**:
```python
learning_rate = 0.1      # Alpha: how much to update Q-values
discount_factor = 0.95   # Gamma: importance of future rewards
epsilon = 1.0            # Initial exploration rate
epsilon_decay = 0.995    # Decay rate for exploration
epsilon_min = 0.01       # Minimum exploration rate
```

**Reward Calculation**:
```python
reward = base_reward (10 for success, -10 for failure)
       + time_reward (0-5 based on communication time)
       + convergence_reward (0-5 based on training time)
       + accuracy_reward (0-10 based on model accuracy)
       + resource_penalty (-5 to 0 based on resource usage)
```

---

### 2. Dynamic Network Controller (`dynamic_network_controller.py`)

**Purpose**: Change network conditions dynamically while containers are running (simulates mobility)

**Network Scenarios**:
| Scenario | Latency | Jitter | Bandwidth | Packet Loss |
|----------|---------|--------|-----------|-------------|
| Excellent | 5ms | 1ms | 100 Mbps | 0% |
| Good | 20ms | 5ms | 50 Mbps | 0.1% |
| Moderate | 50ms | 10ms | 20 Mbps | 0.5% |
| Poor | 150ms | 30ms | 5 Mbps | 1% |
| Very Poor | 300ms | 50ms | 1 Mbps | 3% |
| Satellite | 500ms | 100ms | 10 Mbps | 0.5% |

**Mobility Patterns**:
- **Static**: No changes (excellent → excellent → excellent)
- **Low**: Minimal variation (excellent → good → excellent)
- **Medium**: Moderate variation (good → moderate → good)
- **High**: High variation (good → poor → moderate → excellent)

**Usage**:
```bash
# Apply a scenario to all FL client containers
python dynamic_network_controller.py --scenario moderate

# Simulate high mobility (changes every 30 seconds)
python dynamic_network_controller.py --mobility high --interval 30

# Apply custom conditions
python dynamic_network_controller.py --custom --latency 100 --bandwidth 10 --loss 2

# Clear all network conditions
python dynamic_network_controller.py --clear
```

---

### 3. Unified FL Clients

Each use case has a unified client that integrates all 5 protocols:

#### **Emotion Recognition** (`FL_Client_Unified.py`)
- **Model**: CNN for facial expression classification (48x48 images)
- **Classes**: 7 emotions (happy, sad, angry, neutral, surprise, fear, disgust)
- **Model Size**: ~2MB (medium)

#### **Mental State Recognition** (To be created)
- **Model**: LSTM/CNN for EEG signal classification
- **Classes**: Mental states (focused, relaxed, stressed, etc.)
- **Model Size**: ~5MB (large)

#### **Temperature Regulation** (To be created)
- **Model**: Dense neural network for temperature control
- **Classes**: Regression (target temperature)
- **Model Size**: ~500KB (small)

**Client Features**:
- ✅ Automatic protocol selection using Q-Learning
- ✅ Multiple protocol handlers (MQTT, AMQP, gRPC, QUIC, DDS)
- ✅ Metrics tracking (time, accuracy, resources)
- ✅ Adaptive learning from experience
- ✅ Fallback mechanisms for failures

---

### 4. Unified FL Servers (To be created)

Server-side implementation that:
- Receives weights from multiple clients via different protocols
- Performs FedAvg aggregation
- Sends global weights back to clients
- Tracks protocol usage and performance

---

## Setup and Installation

### Prerequisites
```bash
# System dependencies
sudo apt-get install -y build-essential cmake iproute2

# Python environment (use conda)
conda activate base

# Verify packages
pip list | grep -E "tensorflow|paho-mqtt|pika|grpcio|aioquic|cyclonedds"
```

### Install Components
```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL

# Test Q-Learning Selector
python Client/rl_q_learning_selector.py

# Test Network Controller
python Client/dynamic_network_controller.py --scenario good
```

---

## Usage Examples

### 1. Run with Automatic Protocol Selection (RL-based)

```bash
# Set environment
export USE_RL_SELECTION=true
export CLIENT_ID=1
export NUM_CLIENTS=2
export NUM_ROUNDS=10

# Run emotion recognition client
cd Client/Emotion_Recognition
python FL_Client_Unified.py
```

### 2. Run with Fixed Protocol

```bash
# Force MQTT
export USE_RL_SELECTION=false
export DEFAULT_PROTOCOL=mqtt

python FL_Client_Unified.py
```

### 3. Simulate Mobility During Training

**Terminal 1** (Run FL):
```bash
docker compose -f Docker/docker-compose-emotion.yml up
```

**Terminal 2** (Simulate mobility):
```bash
# Start with excellent conditions
python Client/dynamic_network_controller.py --scenario excellent

# Wait 30 seconds, then change to poor
sleep 30
python Client/dynamic_network_controller.py --scenario poor

# Wait 30 seconds, then change to moderate
sleep 30
python Client/dynamic_network_controller.py --scenario moderate
```

**Terminal 3** (Automated mobility simulation):
```bash
# Automatically cycle through high mobility pattern every 30 seconds
python Client/dynamic_network_controller.py --mobility high --interval 30
```

---

## Q-Learning Training Process

### Episode Structure
1. **Observe State**: Network, Resources, Model Size, Mobility
2. **Select Protocol**: Epsilon-greedy (explore vs exploit)
3. **Execute FL Round**: Train model, communicate
4. **Measure Reward**: Time, Success, Accuracy, Resources
5. **Update Q-Table**: Q-learning update rule
6. **Decay Epsilon**: Reduce exploration over time

### Learning Progress
```
Episode 1-10: High exploration (ε ≈ 1.0) → Try all protocols
Episode 10-50: Balanced (ε ≈ 0.5) → Learn best protocols per state
Episode 50+: Low exploration (ε ≈ 0.01) → Exploit best protocols
```

### Q-Table Persistence
- Automatically saved every 10 episodes
- Loaded on startup if exists
- Continues learning across sessions

---

## Monitoring and Analysis

### View Q-Learning Statistics
```python
from Client.rl_q_learning_selector import QLearningProtocolSelector

selector = QLearningProtocolSelector(save_path="q_table_emotion_client_1.pkl")
selector.print_statistics()
```

**Output**:
```
======================================================================
Q-LEARNING PROTOCOL SELECTOR - STATISTICS
======================================================================
Episodes: 100
Epsilon (exploration): 0.0605
Average Reward (last 100): 15.43

Protocol Usage:
  MQTT: 45 times | Success: 42 | Failure: 3 | Rate: 93.33%
  AMQP: 23 times | Success: 20 | Failure: 3 | Rate: 86.96%
  GRPC: 18 times | Success: 17 | Failure: 1 | Rate: 94.44%
  QUIC: 10 times | Success: 9 | Failure: 1 | Rate: 90.00%
  DDS: 4 times | Success: 3 | Failure: 1 | Rate: 75.00%
======================================================================
```

### Extract Q-Table for Analysis
```python
import pickle
import numpy as np

# Load Q-table
with open("q_table_emotion_client_1.pkl", "rb") as f:
    data = pickle.load(f)

q_table = data['q_table']

# Get best protocol for excellent network + high resources
state_idx = (0, 0, 1, 0)  # (excellent, high, medium_model, static)
best_protocol_idx = np.argmax(q_table[state_idx])
protocols = ['mqtt', 'amqp', 'grpc', 'quic', 'dds']
print(f"Best protocol for state {state_idx}: {protocols[best_protocol_idx]}")
```

---

## Docker Integration

### Enable NET_ADMIN Capability
All containers need NET_ADMIN to allow tc (traffic control):

```yaml
services:
  fl-client-emotion-1:
    build: ...
    cap_add:
      - NET_ADMIN  # Required for dynamic network control
```

### Environment Variables
```yaml
environment:
  # RL Configuration
  - USE_RL_SELECTION=true
  - DEFAULT_PROTOCOL=mqtt
  
  # FL Configuration
  - CLIENT_ID=1
  - NUM_CLIENTS=2
  - NUM_ROUNDS=10
  
  # Protocol Endpoints
  - MQTT_BROKER=mqtt-broker
  - MQTT_PORT=1883
  - AMQP_HOST=rabbitmq
  - GRPC_HOST=fl-server-grpc
  - DDS_DOMAIN_ID=0
```

---

## Performance Metrics

### Tracked Metrics per Round
- **Communication Time**: Round-trip time for weight exchange
- **Training Time**: Local model training duration
- **Model Accuracy**: Validation accuracy
- **Resource Usage**: CPU, Memory, Bandwidth
- **Success Rate**: Communication success/failure

### Reward Components
```
Total Reward = Success Bonus (10 or -10)
             + Communication Speed Bonus (0-5)
             + Convergence Speed Bonus (0-5)
             + Accuracy Bonus (0-10)
             + Resource Efficiency Bonus (-5 to 0)

Maximum Reward: ~30
Typical Reward: 10-20 for good performance
Failure Penalty: -10
```

---

## Troubleshooting

### Q-Learning not improving
- **Check**: Epsilon decay (should decrease over episodes)
- **Check**: Reward calculation (should be balanced)
- **Solution**: Increase episodes, adjust learning rate

### Network conditions not applying
- **Check**: Container has NET_ADMIN capability
- **Check**: tc command is available in container
- **Solution**: Rebuild Docker images with iproute2 installed

### Protocol selection always the same
- **Check**: Epsilon value (if too low, only exploitation)
- **Check**: Q-table values (one protocol might dominate)
- **Solution**: Reset Q-table or increase epsilon

### Communication failures
- **Check**: Protocol endpoints (broker/server addresses)
- **Check**: Docker network connectivity
- **Solution**: Verify services are running, check logs

---

## Future Enhancements

1. **Multi-Agent RL**: Coordinate protocol selection across clients
2. **Deep Q-Networks (DQN)**: Handle continuous state spaces
3. **Transfer Learning**: Share Q-tables across similar use cases
4. **Protocol Fusion**: Use multiple protocols simultaneously
5. **Predictive Selection**: Anticipate network changes

---

## Files Created

### Core Components
- ✅ `Client/rl_q_learning_selector.py` - Q-Learning agent
- ✅ `Client/dynamic_network_controller.py` - Network condition controller
- ✅ `Client/Emotion_Recognition/FL_Client_Unified.py` - Unified emotion client

### To Be Created
- ⏳ `Client/MentalState_Recognition/FL_Client_Unified.py`
- ⏳ `Client/Temperature_Regulation/FL_Client_Unified.py`
- ⏳ `Server/Emotion_Recognition/FL_Server_Unified.py`
- ⏳ `Server/MentalState_Recognition/FL_Server_Unified.py`
- ⏳ `Server/Temperature_Regulation/FL_Server_Unified.py`

---

## References

- Q-Learning: Watkins & Dayan (1992)
- Federated Learning: McMahan et al. (2017)
- Network Simulation: Linux tc (Traffic Control)
- Communication Protocols: MQTT, AMQP, gRPC, QUIC, DDS specifications

---

## Contact

For questions or issues, please check:
- Project documentation in `/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL`
- Protocol references in `Protocol_References/`
- Troubleshooting guides in `Docker/README_DOCKER.md`
