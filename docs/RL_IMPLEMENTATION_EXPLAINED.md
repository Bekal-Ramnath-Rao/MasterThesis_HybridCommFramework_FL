# Reinforcement Learning Implementation Explained

## Overview

This document provides a comprehensive explanation of the Q-Learning based Reinforcement Learning (RL) implementation for dynamic protocol selection in the Federated Learning framework.

## Table of Contents

1. [RL Components Overview](#rl-components-overview)
2. [Hyperparameters](#hyperparameters)
3. [State Space](#state-space)
4. [Action Space](#action-space)
5. [Reward Function](#reward-function)
6. [Q-Learning Algorithm](#q-learning-algorithm)
7. [State Transitions](#state-transitions)
8. [Training Process](#training-process)

---

## RL Components Overview

The RL system consists of two main components:

### 1. `QLearningProtocolSelector` (`Client/rl_q_learning_selector.py`)
- Implements the Q-Learning algorithm
- Manages the Q-table (state-action value function)
- Handles action selection (epsilon-greedy)
- Updates Q-values based on rewards

### 2. `EnvironmentStateManager` (`Client/rl_q_learning_selector.py`)
- Tracks and manages environment state
- Detects network conditions, resource levels, model size, and mobility
- Provides state information to the RL agent

---

## Hyperparameters

### 1. Learning Rate (α) = 0.1

**Location**: `Client/rl_q_learning_selector.py`, line 38

```python
learning_rate: float = 0.1
```

**Purpose**: Controls how quickly the agent updates its Q-values based on new experiences.

**How it works**:
- Higher learning rate (e.g., 0.5): Agent learns quickly but may be unstable
- Lower learning rate (e.g., 0.01): Agent learns slowly but more stable
- **Current value (0.1)**: Balanced approach - moderate learning speed with reasonable stability

**In Q-update equation**:
```
Q(s,a) = Q(s,a) + α * [reward + γ * max(Q(s',a')) - Q(s,a)]
```

The learning rate multiplies the TD-error (temporal difference error), determining how much the Q-value changes.

---

### 2. Discount Factor (γ) = 0.95

**Location**: `Client/rl_q_learning_selector.py`, line 39

```python
discount_factor: float = 0.95
```

**Purpose**: Determines the importance of future rewards compared to immediate rewards.

**How it works**:
- γ = 0: Agent only cares about immediate rewards (myopic)
- γ = 1: Agent values future rewards equally to immediate rewards
- **Current value (0.95)**: High value indicates the agent considers long-term consequences, which is important for protocol selection where decisions affect multiple rounds

**In Q-update equation**:
```
Q(s,a) = Q(s,a) + α * [reward + γ * max(Q(s',a')) - Q(s,a)]
```

The discount factor multiplies the maximum Q-value of the next state, weighting future rewards.

**Why 0.95?**
- Protocol selection decisions have lasting effects (affects multiple FL rounds)
- High discount factor encourages the agent to learn protocols that work well over time
- Close to 1.0 but not exactly 1.0 to ensure convergence

---

### 3. Epsilon (ε) = 1.0 (Initial)

**Location**: `Client/rl_q_learning_selector.py`, line 40

```python
epsilon: float = 1.0
epsilon_decay: float = 0.995
epsilon_min: float = 0.01
```

**Purpose**: Controls the exploration vs exploitation trade-off in epsilon-greedy action selection.

**How it works**:
- **ε = 1.0 (initial)**: 100% exploration - agent always selects random actions
- **ε = 0.0**: 100% exploitation - agent always selects best-known action
- **Current decay (0.995)**: After each episode, ε = ε × 0.995
- **Minimum ε (0.01)**: Agent always explores 1% of the time (prevents complete stagnation)
- **Epsilon Reset**: When Q-values converge, epsilon is reset to 1.0 to allow re-exploration if network conditions change

**Epsilon-Greedy Strategy** (line 127-132):
```python
if training and np.random.random() < self.epsilon:
    # Explore: random action
    action_idx = np.random.randint(len(self.PROTOCOLS))
else:
    # Exploit: best known action
    action_idx = np.argmax(self.q_table[state_idx])
```

**Decay Schedule**:
- Episode 0: ε = 1.0 (100% exploration)
- Episode 10: ε ≈ 0.95 (95% exploration)
- Episode 100: ε ≈ 0.61 (61% exploration)
- Episode 500: ε ≈ 0.08 (8% exploration)
- Episode 1000+: ε ≈ 0.01 (1% exploration, minimum)

**Why this setup?**
- Starts with full exploration to discover all protocols
- Gradually shifts to exploitation as agent learns
- Maintains small exploration to adapt to changing conditions

**Epsilon Reset on Convergence**:
- When Q-values converge (detected via `check_q_converged()`), epsilon is reset to 1.0
- This allows re-exploration if network conditions change after convergence
- Useful for adapting to new network scenarios or changing environments
- Method: `reset_epsilon()` (line 257-260)

---

## State Space

The state space is **4-dimensional**, representing the current environment conditions:

### State Dimensions

**Location**: `Client/rl_q_learning_selector.py`, lines 30-34

```python
NETWORK_CONDITIONS = ['excellent', 'good', 'moderate', 'poor', 'very_poor']  # 5 values
RESOURCE_LEVELS = ['high', 'medium', 'low']                                  # 3 values
MODEL_SIZES = ['small', 'medium', 'large']                                    # 3 values
MOBILITY_LEVELS = ['static', 'low', 'medium', 'high']                        # 4 values
```

**Total State Space Size**: 5 × 3 × 3 × 4 = **180 unique states**

### 1. Network Condition (5 states)

**Detection Method**: `EnvironmentStateManager.detect_network_condition()` (lines 476-497)

Based on **latency** (ms) and **bandwidth** (Mbps):

| Condition    | Latency (ms) | Bandwidth (Mbps) |
|--------------|--------------|------------------|
| excellent    | < 10         | > 50             |
| good         | < 30         | > 20             |
| moderate     | < 100        | > 5              |
| poor         | < 300        | > 1              |
| very_poor    | ≥ 300        | ≤ 1              |

**How it's measured** (in `FL_Client_Unified.py`, lines 1360-1409):
- Measures TCP connection latency to MQTT broker (3 samples, average)
- Estimates bandwidth based on connection speed
- Updates state before each protocol selection

### 2. Resource Level (3 states)

**Detection Method**: `EnvironmentStateManager.detect_resource_level()` (lines 499-517)

Based on **CPU** and **Memory** usage (%):

| Level   | Average CPU+Memory Usage |
|---------|--------------------------|
| high    | < 30%                    |
| medium  | 30% - 70%                |
| low     | ≥ 70%                    |

**How it's measured**:
- Uses `psutil` to get CPU and memory percentages
- Calculates average: `(cpu_percent + memory_percent) / 2`
- Updated before each protocol selection

### 3. Model Size (3 states)

**Detection Method**: Based on total model parameters (lines 1522-1531 in `FL_Client_Unified.py`)

| Size    | Parameters        |
|---------|-------------------|
| small   | < 100,000         |
| medium  | 100,000 - 10M     |
| large   | ≥ 10,000,000      |

**How it's determined**:
- Calculated when model is built: `model.count_params()`
- Automatically categorized and stored in environment state

### 4. Mobility Level (4 states)

**Detection Method**: Based on latency variance (lines 1411-1430 in `FL_Client_Unified.py`)

| Level   | Latency Variance (ms²) |
|---------|------------------------|
| static  | Low variance           |
| low     | Moderate variance      |
| medium  | Higher variance        |
| high    | Very high variance     |

**How it's determined**:
- Tracks last 20 latency samples
- Calculates variance: `sum((x - avg)²) / n`
- Higher variance → higher mobility (network conditions changing)

### State Index Conversion

**Method**: `get_state_index()` (lines 96-111)

Converts state dictionary to Q-table indices:

```python
def get_state_index(self, state: Dict) -> Tuple[int, int, int, int]:
    network_idx = self.NETWORK_CONDITIONS.index(state.get('network', 'moderate'))
    resource_idx = self.RESOURCE_LEVELS.index(state.get('resource', 'medium'))
    model_idx = self.MODEL_SIZES.index(state.get('model_size', 'medium'))
    mobility_idx = self.MOBILITY_LEVELS.index(state.get('mobility', 'static'))
    return (network_idx, resource_idx, model_idx, mobility_idx)
```

**Q-table Shape**: `(5, 3, 3, 4, 6)` = (network, resource, model_size, mobility, actions)

---

## Action Space

**Location**: `Client/rl_q_learning_selector.py`, line 28

```python
PROTOCOLS = ['mqtt', 'amqp', 'grpc', 'quic', 'http3', 'dds']
```

**Total Actions**: **6 protocols**

### Available Actions

1. **MQTT** (Message Queuing Telemetry Transport)
   - Lightweight, pub/sub protocol
   - Good for IoT, low bandwidth

2. **AMQP** (Advanced Message Queuing Protocol)
   - Message broker protocol
   - Reliable, feature-rich

3. **gRPC** (Google Remote Procedure Call)
   - High-performance RPC framework
   - Good for low latency

4. **QUIC** (Quick UDP Internet Connections)
   - UDP-based, multiplexed
   - Good for poor networks

5. **HTTP/3**
   - HTTP over QUIC
   - Modern web protocol

6. **DDS** (Data Distribution Service)
   - Real-time, pub/sub
   - Good for high-performance systems

### Action Selection

**Method**: `select_protocol()` (lines 113-141)

**Epsilon-Greedy Strategy**:
```python
if training and np.random.random() < self.epsilon:
    # Explore: random action
    action_idx = np.random.randint(len(self.PROTOCOLS))
else:
    # Exploit: best known action
    action_idx = np.argmax(self.q_table[state_idx])
```

**During Training**:
- With probability ε: Select random protocol (exploration)
- With probability (1-ε): Select protocol with highest Q-value (exploitation)

**During Inference** (`training=False`):
- Always selects best-known protocol (pure exploitation)
- Used when `USE_QL_CONVERGENCE=False` (training ends on accuracy convergence, use learned knowledge)

---

## Reward Function

**Location**: `Client/rl_q_learning_selector.py`, lines 143-195

**Method**: `calculate_reward()`

### Reward Components

The reward is a **weighted combination** of multiple metrics:

```python
def calculate_reward(
    self,
    communication_time: float,      # Round-trip time (seconds)
    success: bool,                   # Communication success
    convergence_time: float,         # Model convergence time (seconds)
    accuracy: float,                 # Model accuracy (0-1)
    resource_consumption: Dict[str, float]  # CPU, memory, bandwidth (0-1)
) -> float:
```

### Reward Calculation Breakdown

#### 1. Base Reward for Success/Failure

```python
if not success:
    return -10.0  # Large penalty for failure

reward = 10.0  # Base reward for success
```

- **Success**: +10.0 points
- **Failure**: -10.0 points (large penalty to discourage failed protocols)

#### 2. Communication Time Reward

```python
# Normalize: 0-3600 seconds (1 hour) -> reward 5 to 0
# This allows differentiation even in poor/very_poor network conditions
time_reward = max(0, 5.0 - (communication_time / 720.0))
reward += time_reward
```

**Scaling**:
- 0 seconds: +5.0 points
- 6 minutes (360s): +4.5 points
- 12 minutes (720s): +4.0 points
- 30 minutes (1800s): +2.5 points
- 1 hour (3600s): +0.0 points
- >1 hour: +0.0 points

**Purpose**: Encourages faster communication while allowing differentiation between protocols even in poor network conditions. Previously, protocols taking longer than 5 seconds all received 0 reward, preventing the RL agent from learning which protocol performs better in challenging network conditions.

#### 3. Convergence Time Reward

```python
# Normalize: 0-100 seconds -> reward 5 to 0
conv_reward = max(0, 5.0 - (convergence_time / 20.0))
reward += conv_reward
```

**Scaling**:
- 0 seconds: +5.0 points
- 20 seconds: +4.0 points
- 40 seconds: +3.0 points
- 60 seconds: +2.0 points
- 80 seconds: +1.0 points
- ≥100 seconds: +0.0 points

**Purpose**: Encourages faster model convergence

#### 4. Accuracy Reward

```python
# Scale accuracy (0-1) to reward (0-10)
accuracy_reward = accuracy * 10.0
reward += accuracy_reward
```

**Scaling**:
- 0.0 accuracy: +0.0 points
- 0.5 accuracy: +5.0 points
- 0.9 accuracy: +9.0 points
- 1.0 accuracy: +10.0 points

**Purpose**: Encourages higher model accuracy

#### 5. Resource Consumption Penalty

```python
cpu_usage = resource_consumption.get('cpu', 0.5)
memory_usage = resource_consumption.get('memory', 0.5)
bandwidth_usage = resource_consumption.get('bandwidth', 0.5)

# Average resource usage (0-1) converted to penalty (0 to -5)
avg_resource = (cpu_usage + memory_usage + bandwidth_usage) / 3.0
resource_penalty = -5.0 * avg_resource
reward += resource_penalty
```

**Scaling**:
- 0% usage: +0.0 points (no penalty)
- 50% usage: -2.5 points
- 100% usage: -5.0 points

**Purpose**: Discourages high resource consumption

### Total Reward Range

**Best Case Scenario**:
- Success: +10.0
- Fast communication (0s): +5.0
- Fast convergence (0s): +5.0
- High accuracy (1.0): +10.0
- Low resources (0%): +0.0
- **Total: +30.0 points**

**Poor Network Scenario** (e.g., 30 minutes communication time):
- Success: +10.0
- Slow communication (1800s): +2.5 (previously would be +0.0)
- Fast convergence (0s): +5.0
- Moderate accuracy (0.8): +8.0
- Medium resources (50%): -2.5
- **Total: +23.0 points** (allows differentiation between protocols)

**Worst Case Scenario**:
- Failure: -10.0
- **Total: -10.0 points**

**Typical Reward Range**: -10.0 to +30.0

**Note**: The updated communication time normalization (0-3600s instead of 0-5s) ensures that protocols can still be differentiated even in poor/very_poor network conditions, where communication may take minutes or tens of minutes. Previously, any protocol taking longer than 5 seconds would receive 0 reward, preventing the RL agent from learning which protocol performs relatively better in challenging conditions.

### Reward Usage

**Location**: `FL_Client_Unified.py`, lines 1738-1745

Reward is calculated after each FL round evaluation:

```python
reward = self.rl_selector.calculate_reward(
    self.round_metrics['communication_time'],
    self.round_metrics['success'],
    self.round_metrics.get('training_time', 0.0),
    self.round_metrics['accuracy'],
    resources,
)
```

---

## Q-Learning Algorithm

### Q-Table Structure

**Location**: `Client/rl_q_learning_selector.py`, lines 68-75

```python
state_space_size = (
    len(self.NETWORK_CONDITIONS),    # 5
    len(self.RESOURCE_LEVELS),       # 3
    len(self.MODEL_SIZES),           # 3
    len(self.MOBILITY_LEVELS)        # 4
)
self.q_table = np.zeros(state_space_size + (len(self.PROTOCOLS),))
```

**Q-table Shape**: `(5, 3, 3, 4, 6)`
- Dimensions: (network, resource, model_size, mobility, actions)
- Total Q-values: 5 × 3 × 3 × 4 × 6 = **1,080 Q-values**
- Initialized to zeros (tabula rasa)

### Q-Value Update Rule

**Location**: `Client/rl_q_learning_selector.py`, lines 197-250

**Method**: `update_q_value()`

#### Bellman Equation for Q-Learning

```
Q(s,a) ← Q(s,a) + α × [R + γ × max(Q(s',a')) - Q(s,a)]
```

Where:
- **Q(s,a)**: Current Q-value for state s and action a
- **α**: Learning rate (0.1)
- **R**: Immediate reward
- **γ**: Discount factor (0.95)
- **max(Q(s',a'))**: Maximum Q-value in next state s'

#### Implementation

**For Non-Terminal States** (lines 224-230):
```python
if done or next_state is None:
    # Terminal state
    new_q = current_q + self.learning_rate * (reward - current_q)
else:
    # Non-terminal state: use Bellman equation
    next_state_idx = self.get_state_index(next_state)
    max_next_q = np.max(self.q_table[next_state_idx])
    new_q = current_q + self.learning_rate * (
        reward + self.discount_factor * max_next_q - current_q
    )
```

**For Terminal States** (lines 221-223):
```python
# Terminal state (episode ended)
new_q = current_q + self.learning_rate * (reward - current_q)
```

### Q-Value Update Process

1. **Get Current Q-Value**:
   ```python
   state_idx = self.state_history[-1]
   action_idx = self.action_history[-1]
   current_q = self.q_table[state_idx][action_idx]
   ```

2. **Calculate TD-Error**:
   ```python
   td_error = reward + γ * max_next_q - current_q
   ```

3. **Update Q-Value**:
   ```python
   new_q = current_q + α * td_error
   self.q_table[state_idx][action_idx] = new_q
   ```

4. **Track Convergence**:
   ```python
   q_delta = abs(new_q - current_q)
   self._q_deltas.append(q_delta)
   ```

### Q-Value Interpretation

**High Q-value** (> 0): Protocol performs well in this state
- Agent is likely to select this protocol again

**Low Q-value** (< 0): Protocol performs poorly in this state
- Agent is unlikely to select this protocol

**Q-value convergence**: When Q-values stop changing significantly
- Indicates agent has learned optimal policy
- Used as training end condition (`USE_QL_CONVERGENCE`)

---

## State Transitions

### Current State

**Location**: `FL_Client_Unified.py`, lines 1449-1479

**When**: Before each protocol selection

**Process**:
1. Measure CPU and memory usage
2. Detect resource level
3. Measure network condition (latency, bandwidth)
4. Update network condition in state
5. Detect mobility from latency variance
6. Get current state dictionary

**Example Current State**:
```python
state = {
    'network': 'good',        # Based on latency < 30ms, bandwidth > 20Mbps
    'resource': 'medium',      # Based on CPU+Memory avg = 50%
    'model_size': 'medium',    # Based on 500K parameters
    'mobility': 'low'          # Based on moderate latency variance
}
```

### Future State (Next State)

**Location**: `FL_Client_Unified.py`, line 1745

**When**: After receiving reward and updating Q-value

**Current Implementation**:
```python
self.rl_selector.update_q_value(reward, next_state=None, done=True)
```

**Note**: Currently, `next_state=None` and `done=True` are used, meaning:
- Each FL round is treated as a **terminal state**
- Q-learning update uses simplified equation (no future state)
- This is a **one-step Q-learning** approach

**Why Terminal States?**
- Each FL round is independent
- Protocol selection for round N doesn't directly affect round N+1 state
- State changes are driven by environment (network, resources), not by actions

### State Update Flow

```
Round N:
1. Measure environment → Current State S_t
2. Select protocol (action A_t) using epsilon-greedy
3. Execute action (send model update via protocol)
4. Measure performance → Reward R_t
5. Update Q(S_t, A_t) using reward
6. End episode (done=True)

Round N+1:
1. Measure environment → New State S_{t+1} (may differ due to environment changes)
2. Select protocol (action A_{t+1}) using epsilon-greedy
3. ...
```

**State Changes Between Rounds**:
- Network condition may change (latency varies)
- Resource levels may change (CPU/memory usage varies)
- Model size stays constant (same model architecture)
- Mobility may change (latency variance changes)

---

## Training Process

### Episode Structure

**One Episode** = One FL round

**Steps per Episode**:

1. **State Observation** (`select_protocol()`):
   - Measure network condition
   - Measure resource levels
   - Get model size
   - Detect mobility
   - Form state vector

2. **Action Selection** (`select_protocol()`):
   - Use epsilon-greedy strategy
   - Select protocol (action)

3. **Action Execution** (`train_local_model()` or `evaluate_model()`):
   - Send model update via selected protocol
   - Measure communication time
   - Track success/failure

4. **Reward Calculation** (`calculate_reward()`):
   - Calculate reward based on:
     - Communication time
     - Success status
     - Convergence time
     - Accuracy
     - Resource consumption

5. **Q-Value Update** (`update_q_value()`):
   - Update Q-table using Bellman equation
   - Track Q-value changes for convergence

6. **Episode End** (`end_episode()`):
   - Decay epsilon
   - Save Q-table (every 10 episodes)
   - Update statistics

### Training Loop

**Location**: `FL_Client_Unified.py`, federated learning rounds

```python
for round in range(max_rounds):
    # 1. Receive global model
    global_model = receive_global_model()
    
    # 2. Select protocol (RL)
    protocol = self.select_protocol()  # Uses RL
    
    # 3. Train local model
    self.train_local_model()
    
    # 4. Send update via selected protocol
    self.send_model_update(protocol)
    
    # 5. Evaluate model
    self.evaluate_model()
    
    # 6. Calculate reward and update Q-value
    reward = calculate_reward(...)
    self.rl_selector.update_q_value(reward, done=True)
    self.rl_selector.end_episode()
```

### Convergence Detection

**Q-Learning Convergence** (`check_q_converged()`, lines 414-429):

```python
def check_q_converged(self, threshold=0.01, patience=5) -> bool:
    if len(self._q_deltas) < patience:
        return False
    return all(d <= threshold for d in self._q_deltas[-patience:])
```

**Convergence Criteria**:
- Last `patience` Q-updates all have delta ≤ `threshold`
- Default: threshold = 0.01, patience = 5
- Means: Last 5 Q-updates changed by ≤ 0.01 each

**Usage** (`USE_QL_CONVERGENCE`):
- **When `True`**: 
  - Uses `training=True` (epsilon-greedy exploration)
  - Training ends when Q-values converge
  - Epsilon resets to 1.0 when convergence is detected
- **When `False`**: 
  - Uses `training=False` (pure exploitation, greedy policy)
  - Always selects best-known protocol for current state
  - Training ends on accuracy convergence
  - No exploration, uses learned knowledge only

### Q-Table Persistence

**Save Path**: `shared_data/q_table_emotion_client_{id}.pkl`

**Saved Data**:
- Q-table (numpy array)
- Epsilon value
- Episode count
- Total rewards history
- Protocol usage statistics
- Protocol success/failure counts

**Load on Startup**:
- Tries to load from `initial_load_path` (pretrained)
- Falls back to `save_path` (previous run)
- If neither exists, starts with zeros (fresh Q-table)

---

## Summary

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate (α) | 0.1 | Controls Q-value update speed |
| Discount Factor (γ) | 0.95 | Weight of future rewards |
| Initial Epsilon (ε) | 1.0 | Initial exploration rate |
| Epsilon Decay | 0.995 | Exploration decay per episode |
| Min Epsilon | 0.01 | Minimum exploration rate |

### State Space

- **Dimensions**: 4 (network, resource, model_size, mobility)
- **Total States**: 180 unique states
- **State Detection**: Automatic based on measurements

### Action Space

- **Actions**: 6 protocols (MQTT, AMQP, gRPC, QUIC, HTTP/3, DDS)
- **Selection**: Epsilon-greedy (exploration vs exploitation)

### Reward Function

- **Range**: -10.0 to +30.0
- **Components**: Success, communication time, convergence time, accuracy, resources
- **Goal**: Maximize reward = optimize protocol selection

### Q-Learning

- **Q-table Size**: 1,080 Q-values (180 states × 6 actions)
- **Update Rule**: Bellman equation with learning rate and discount factor
- **Convergence**: Based on Q-value stability

### Training

- **Episode**: One FL round
- **Process**: Observe → Act → Reward → Update Q-value
- **Convergence**: Q-values stabilize or accuracy converges

---

## References

- **Q-Learning Algorithm**: Watkins & Dayan (1992)
- **Implementation**: `Client/rl_q_learning_selector.py`
- **Usage**: `Client/Emotion_Recognition/FL_Client_Unified.py`
- **Documentation**: `docs/guides/README_RL_PROTOCOL_SELECTION.md`
