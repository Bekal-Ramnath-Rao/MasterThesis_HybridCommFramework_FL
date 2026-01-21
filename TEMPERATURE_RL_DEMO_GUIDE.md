# Temperature FL Demo with Dynamic Network & RL Selection

## Quick Start

### 1. Run the Demo

```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL

# Option A: With RL Monitor (Recommended)
python Client/Temperature_Regulation/FL_Client_Unified.py 2>&1 | python Client/rl_monitor.py

# Option B: Direct run
./run_temperature_rl_demo.sh
```

### 2. Change Network Conditions (In Another Terminal)

While the FL client is running, open a **new terminal** and execute:

```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL

# Start with excellent network
python Client/dynamic_network_controller.py --scenario excellent

# Wait a few rounds, then degrade to fair
python Client/dynamic_network_controller.py --scenario fair

# Degrade further to poor
python Client/dynamic_network_controller.py --scenario poor

# Simulate high mobility (rapid changes)
python Client/dynamic_network_controller.py --mobility high

# Custom: 200ms latency, 50ms jitter, 5Mbps bandwidth, 3% loss
python Client/dynamic_network_controller.py --custom 200,50,5,3

# Clear all limitations
python Client/dynamic_network_controller.py --clear
```

## What to Watch For

### RL Protocol Selection
```
🎯 State: (3, 1, 0, 2)  # (network, resource, model_size, mobility)
📨 PROTOCOL: MQTT       # Selected protocol for this round
💰 Reward: +12.50 🟡 GOOD
```

### Protocol Usage Patterns
- **Early rounds**: Random exploration (MQTT, gRPC, DDS, AMQP, QUIC)
- **Mid rounds**: Learning phase (tries different protocols)
- **Late rounds**: Exploitation (converges to best protocol)

### Network Adaptation
When you change network conditions, observe:
1. **Degraded network** → RL may switch from DDS/gRPC to MQTT
2. **Improved network** → RL may switch from MQTT to gRPC/QUIC
3. **High mobility** → RL favors lightweight protocols

## Network Scenarios

| Scenario | Latency | Jitter | Bandwidth | Loss |
|----------|---------|--------|-----------|------|
| excellent | 5ms | 1ms | 100 Mbps | 0% |
| good | 20ms | 5ms | 50 Mbps | 0.1% |
| fair | 50ms | 10ms | 20 Mbps | 0.5% |
| poor | 150ms | 30ms | 5 Mbps | 1% |
| very_poor | 300ms | 50ms | 1 Mbps | 2% |

## Expected Behavior

### Example Timeline (20 rounds)

```
Rounds 1-5:   [EXPLORATION] Random protocols (MQTT, gRPC, DDS, ...)
              Network: excellent
              
Rounds 6-10:  [LEARNING] Trying different combinations
              You change: --scenario fair
              RL adapts: May switch to lighter protocols
              
Rounds 11-15: [MIXED] Exploration decreasing (ε decay)
              You change: --mobility high
              RL adapts: Favors robust protocols
              
Rounds 16-20: [EXPLOITATION] Converges to optimal protocol
              Consistent choice based on learned Q-values
```

## Configuration

Edit environment variables in [run_temperature_rl_demo.sh](run_temperature_rl_demo.sh):

```bash
export NUM_ROUNDS=20           # More rounds = better learning
export USE_RL_SELECTION=true   # Enable RL (false = use DEFAULT_PROTOCOL)
export DEFAULT_PROTOCOL=mqtt   # Fallback protocol
```

## Q-Learning Parameters

Located in [rl_q_learning_selector.py](Client/rl_q_learning_selector.py):

- **Learning rate (α)**: 0.1 (how much to update Q-values)
- **Discount factor (γ)**: 0.95 (future reward importance)
- **Epsilon (ε)**: Starts at 1.0, decays by 0.995 each round
- **Min epsilon**: 0.01 (always 1% random exploration)

## Monitoring Tools

### Real-Time Monitor
```bash
python Client/Temperature_Regulation/FL_Client_Unified.py 2>&1 | python Client/rl_monitor.py
```

Shows:
- 📨 Protocol selections with emojis
- 💰 Rewards with color indicators (🟢🟡🟠🔴)
- 📊 Training metrics (time, MAE)
- 📈 Final statistics and protocol distribution

### Manual Observation
Look for these log entries:
- `[RL Selection] Selected Protocol: MQTT`
- `[RL] Reward: +12.50`
- `[Training] Time: 2.34s, MAE: 0.1234`

## Troubleshooting

### MQTT Broker Not Running
```bash
mosquitto -d -c mqtt-config/mosquitto.conf
```

### CycloneDDS Not Available
```bash
# Activate conda environment
conda activate base  # or your conda env with cyclonedds

# Verify installation
python -c "import cyclonedds; print(cyclonedds.__version__)"
```

### Dataset Not Found
The script will automatically use synthetic fallback data if `base_data_baseline_unique.csv` is not found.

## Files Created

- ✅ [run_temperature_rl_demo.sh](run_temperature_rl_demo.sh) - Main demo script
- ✅ [Client/rl_monitor.py](Client/rl_monitor.py) - Real-time RL monitoring tool
- ✅ Q-table saved as `q_table_temperature_client_0.pkl` after run
