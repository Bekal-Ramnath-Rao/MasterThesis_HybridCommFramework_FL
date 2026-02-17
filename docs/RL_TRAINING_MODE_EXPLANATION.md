# RL Training Mode Explanation & Proposed Changes

## Current Implementation Analysis

### Current Behavior

1. **Protocol Selection** (`FL_Client_Unified.py`, line 1470):
   ```python
   protocol = self.rl_selector.select_protocol(state, training=True)
   ```
   - Always uses `training=True` regardless of `USE_QL_CONVERGENCE` setting
   - This means epsilon-greedy strategy is always active (exploration + exploitation)

2. **USE_QL_CONVERGENCE Flag**:
   - **When `True`**: Training ends when Q-values converge (multiple episodes)
   - **When `False`**: Training ends on accuracy convergence (current behavior)
   - **Current issue**: Doesn't affect exploration strategy, only end condition

3. **Epsilon Management**:
   - Starts at 1.0 (100% exploration)
   - Decays by 0.995 per episode
   - Minimum 0.01 (1% exploration)
   - **Current issue**: Never resets, even when Q-values converge

## Your Understanding (Verified)

### ✅ Correct Understanding:

1. **When `USE_QL_CONVERGENCE = True`**:
   - Use `training=True` → Epsilon-greedy (exploration + exploitation)
   - Continue learning until Q-values converge
   - This allows the agent to explore and learn optimal protocols

2. **When `USE_QL_CONVERGENCE = False`**:
   - Use `training=False` → Pure exploitation (greedy policy)
   - Always select best-known protocol for current state
   - No exploration, just use learned knowledge
   - Training ends on accuracy convergence

3. **Epsilon Reset**:
   - When Q-values converge, reset epsilon to 1.0
   - This allows re-exploration if network conditions change
   - Useful for adapting to new network scenarios

## Proposed Changes

### Change 1: Training Mode Based on USE_QL_CONVERGENCE

**File**: `Client/Emotion_Recognition/FL_Client_Unified.py`

**Current**:
```python
protocol = self.rl_selector.select_protocol(state, training=True)
```

**Proposed**:
```python
# Use training mode based on USE_QL_CONVERGENCE flag
# If False: pure exploitation (use best known protocol)
# If True: epsilon-greedy (explore and learn)
training_mode = USE_QL_CONVERGENCE
protocol = self.rl_selector.select_protocol(state, training=training_mode)
```

### Change 2: Add Epsilon Reset Method

**File**: `Client/rl_q_learning_selector.py`

**Add method**:
```python
def reset_epsilon(self):
    """Reset epsilon to initial value (1.0) for re-exploration"""
    self.epsilon = 1.0
    print(f"[Q-Learning] Epsilon reset to {self.epsilon:.4f} for re-exploration")
```

### Change 3: Reset Epsilon on Convergence

**File**: `Client/Emotion_Recognition/FL_Client_Unified.py`

**In convergence detection** (around line 1771):
```python
if USE_QL_CONVERGENCE and q_converged:
    self.has_converged = True
    print(f"[Client {self.client_id}] Q-learning convergence reached at round {self.current_round}")
    # Reset epsilon for potential re-exploration if conditions change
    self.rl_selector.reset_epsilon()
    self._notify_convergence_to_server()
    self._disconnect_after_convergence()
    return
```

## Benefits

1. **Pure Exploitation Mode**: When `USE_QL_CONVERGENCE=False`, agent uses learned knowledge without exploration
2. **Epsilon Reset**: Allows re-exploration when Q-values converge, useful if network conditions change
3. **Clear Separation**: Training mode explicitly tied to convergence end condition

## Questions for Clarification

1. **Per-Network-Scenario Convergence**: Currently, convergence is checked globally (all Q-updates). Do you want per-network-condition convergence tracking? (e.g., reset epsilon when Q-values converge for "poor" network condition specifically)

2. **Epsilon Reset Timing**: Should epsilon reset happen:
   - Immediately when convergence is detected? (proposed)
   - Only if network condition changes after convergence?
   - Both?

## Implementation Plan

1. ✅ Update `select_protocol()` calls to use `training=USE_QL_CONVERGENCE`
2. ✅ Add `reset_epsilon()` method to `QLearningProtocolSelector`
3. ✅ Call `reset_epsilon()` when Q-values converge
4. ✅ Update other unified clients (Temperature, MentalState) if they use RL
5. ✅ Update documentation
