# Resume Training Feature - Continue from Previous Epsilon

## Overview

This feature allows you to **resume RL training** from where it left off, preserving epsilon values, Q-table progress, accumulated rewards, and all learning state. This is especially useful when:

- Training was interrupted unexpectedly (container crash, power loss, etc.)
- You want to continue training the same scenario with more rounds
- You want to save time by not resetting exploration to 1.0

## Two Training Modes

### 1. Fresh Training (Default)
**When to use**: Starting new scenario training from scratch

- ✅ **Epsilon resets to 1.0** (full exploration)
- ✅ **Q-table preserved** (accumulates across scenarios)
- ✅ **Rewards start fresh** for this experiment
- 📊 Recommended for training on a new network scenario

**GUI Setting**: ✅ Check "Reset Epsilon to 1.0 (Fresh Training)"

### 2. Resume Training
**When to use**: Continuing interrupted or incomplete training

- ✅ **Epsilon preserved** (continues from previous value, e.g., 0.3)
- ✅ **Q-table preserved** (all learning maintained)
- ✅ **Rewards and episode count continue** from previous state
- 📊 Saves training time by continuing exploration from where you left off

**GUI Setting**: ❌ Uncheck "Reset Epsilon to 1.0 (Fresh Training)"

## How to Use

### Using the GUI

1. **Open Experiment GUI**:
   ```bash
   cd Network_Simulation
   python3 experiment_gui.py
   ```

2. **Configure RL Training**:
   - Select **rl_unified** protocol
   - Select **Training** mode (not Inference)
   - Choose your scenario (e.g., "Poor")

3. **Choose Training Mode**:
   
   **For Fresh Training** (default):
   - ✅ **Check** "Reset Epsilon to 1.0 (Fresh Training)"
   - Epsilon will reset to 1.0 on experiment start
   
   **For Resume Training**:
   - ❌ **Uncheck** "Reset Epsilon to 1.0 (Fresh Training)"
   - Epsilon will continue from previous value (e.g., 0.3)

4. **Start Experiment**

### Example Scenarios

#### Scenario 1: Multi-Scenario Fresh Training

```
1. Train on "Poor" network:
   - Check "Reset Epsilon to 1.0" ✅
   - Start → Epsilon: 1.0 → 0.01
   
2. Train on "Good" network:
   - Check "Reset Epsilon to 1.0" ✅
   - Start → Epsilon: 1.0 → 0.01 (fresh exploration)
   
3. Train on "Satellite":
   - Check "Reset Epsilon to 1.0" ✅
   - Start → Epsilon: 1.0 → 0.01 (fresh exploration)

Result: Q-table with learning from all 3 scenarios
```

#### Scenario 2: Resume Interrupted Training

```
1. Start training on "Poor" network:
   - Check "Reset Epsilon to 1.0" ✅
   - Start → Epsilon: 1.0 → 0.5
   - [Training interrupted at epsilon=0.5]
   
2. Resume training on same scenario:
   - Uncheck "Reset Epsilon to 1.0" ❌
   - Start → Epsilon: 0.5 → 0.01 (continues)
   - [Training completes]

Result: Time saved by continuing from 0.5 instead of restarting from 1.0
```

#### Scenario 3: Extend Training on Same Scenario

```
1. Complete training on "Poor" network:
   - Check "Reset Epsilon to 1.0" ✅
   - Start → Epsilon: 1.0 → 0.01 (10 rounds)
   
2. Want more training on same scenario:
   - Uncheck "Reset Epsilon to 1.0" ❌
   - Increase rounds to 20
   - Start → Epsilon: 0.01 → 0.01 (continues)
   
Result: Additional training without resetting exploration
```

## Technical Details

### Flag File Format

The experiment runner creates a flag file with the reset control:

**Fresh Training** (`reset_epsilon=1.0`):
```
experiment_id=a1b2c3d4
scenario=poor
timestamp=1709856234.567
reset_epsilon=1.0
```

**Resume Training** (`reset_epsilon=0.0`):
```
experiment_id=e5f6g7h8
scenario=poor
timestamp=1709856789.012
reset_epsilon=0.0
```

### Client Behavior

The FL client reads this flag file and acts accordingly:

**When `reset_epsilon=1.0`** (Fresh Training):
```
[Client 1] 🔄 RESETTING EPSILON TO 1.0 (Fresh Training)
[Client 1]   New experiment ID: a1b2c3d4
[Client 1]   Training scenario: poor
[Client 1]   Current epsilon before reset: 0.3456
[Client 1]   ✓ Epsilon reset to: 1.0000
[Client 1]   📊 Q-table will accumulate learning across all scenarios
```

**When `reset_epsilon=0.0`** (Resume Training):
```
[Client 1] 📍 CONTINUING WITH PREVIOUS EPSILON (Resume Mode)
[Client 1]   Experiment ID: e5f6g7h8
[Client 1]   Training scenario: poor
[Client 1]   Current epsilon (preserved): 0.3456
[Client 1]   📊 Q-table, rewards, and learning progress will continue from previous state
```

### What Gets Preserved in Resume Mode

| Item | Fresh Training | Resume Training |
|------|---------------|-----------------|
| Epsilon | Reset to 1.0 | Preserved (e.g., 0.3) |
| Q-table | Preserved | Preserved |
| Episode count | Continues | Continues |
| Accumulated rewards | Continues | Continues |
| Protocol usage stats | Preserved | Preserved |
| Learning rate | Same | Same |

**Key Point**: The Q-table is ALWAYS preserved, even in fresh training mode. Only epsilon resets.

## Command Line Usage

For direct command-line execution:

### Fresh Training (default)
```bash
python3 run_network_experiments.py \
  --use-case emotion \
  --protocols rl_unified \
  --scenarios poor \
  --rounds 10 \
  --use-ql-convergence \
  --enable-gpu
```

### Resume Training
```bash
python3 run_network_experiments.py \
  --use-case emotion \
  --protocols rl_unified \
  --scenarios poor \
  --rounds 10 \
  --use-ql-convergence \
  --enable-gpu \
  --no-reset-epsilon    # Add this flag to preserve epsilon
```

## Verification

### Check Current Epsilon Value

Before starting resume mode, check the current epsilon in the Q-table:

```bash
# View recent Q-learning logs
tail -20 shared_data/ql_learning_log_client_1_emotion.csv | column -t -s,
```

Look at the "Epsilon" column in the latest entries to see the current value.

### Monitor During Training

Watch the client logs to confirm the correct mode:

```bash
# Fresh Training - should see:
docker logs fl-client-unified-emotion-1 2>&1 | grep "RESETTING EPSILON"

# Resume Training - should see:
docker logs fl-client-unified-emotion-1 2>&1 | grep "CONTINUING WITH PREVIOUS EPSILON"
```

## When to Use Each Mode

### Use Fresh Training When:
- ✅ Starting training on a NEW network scenario
- ✅ Want full exploration (epsilon 1.0 → 0.01)
- ✅ Building a multi-scenario Q-table from scratch
- ✅ Testing the agent's learning ability

### Use Resume Training When:
- ✅ Training was interrupted (crash, timeout, etc.)
- ✅ Want to extend training duration on same scenario
- ✅ Epsilon already converged to low value (e.g., 0.01)
- ✅ Saving time by avoiding full re-exploration

## Best Practices

### 1. Multi-Scenario Training
```
Scenario 1: Fresh (1.0 → 0.01)
Scenario 2: Fresh (1.0 → 0.01)
Scenario 3: Fresh (1.0 → 0.01)
```
Each scenario gets full exploration, Q-table accumulates.

### 2. Interrupted Training Recovery
```
Attempt 1: Fresh (1.0 → 0.5) [interrupted]
Attempt 2: Resume (0.5 → 0.01) [completed]
```
Save hours by not restarting from 1.0.

### 3. Extended Training
```
Round 1-10: Fresh (1.0 → 0.05)
Round 11-20: Resume (0.05 → 0.01)
```
Continue training without resetting exploration.

## Troubleshooting

### Issue: Epsilon not preserved in resume mode

**Symptom**: Epsilon resets to 1.0 even with "Resume" unchecked

**Causes**:
1. Flag file has `reset_epsilon=1.0` instead of `0.0`
2. Q-table file was deleted
3. Experiment started in fresh mode

**Solution**:
1. Check flag file: `cat shared_data/reset_epsilon_flag.txt`
2. Verify Q-table exists: `ls -lh shared_data/q_table_emotion_trained.pkl`
3. Check GUI setting: "Reset Epsilon to 1.0" should be unchecked

### Issue: Q-table lost between experiments

**Symptom**: Episode count resets to 1

**Solution**:
1. Verify Docker volume mount in compose file:
   ```yaml
   volumes:
     - ./shared_data:/shared_data
   ```
2. Check file permissions: `ls -la shared_data/`
3. Don't use "Reset Q-Table" button between experiments

### Issue: Can't find previous epsilon value

**Solution**:
```bash
# Check latest Q-learning log entry
tail -1 shared_data/ql_learning_log_client_1_emotion.csv | column -t -s,

# Or load the Q-table in Python
python3 -c "
import pickle
with open('shared_data/q_table_emotion_trained.pkl', 'rb') as f:
    data = pickle.load(f)
    print(f'Epsilon: {data.get(\"epsilon\", \"N/A\")}')
    print(f'Episodes: {data.get(\"episode_count\", \"N/A\")}')
"
```

## Implementation Files

**Modified Files**:
- [`experiment_gui.py`](../Network_Simulation/experiment_gui.py) - Added epsilon reset checkbox
- [`run_network_experiments.py`](../Network_Simulation/run_network_experiments.py) - Added `--no-reset-epsilon` flag
- [`FL_Client_Unified.py`](../Client/Emotion_Recognition/FL_Client_Unified.py) - Checks reset_epsilon flag value

**New Arguments**:
- GUI: "Reset Epsilon to 1.0 (Fresh Training)" checkbox
- CLI: `--no-reset-epsilon` flag

## Related Documentation

- [Epsilon Reset Fix](./EPSILON_RESET_FIX.md) - Multi-scenario training with epsilon reset
- [RL Training Mode](./RL_TRAINING_MODE_EXPLANATION.md) - Training vs inference modes
- [Q-Table Reset Feature](./Q_TABLE_RESET_FEATURE.md) - Complete Q-table reset (separate from epsilon)

## Summary

🔄 **Two Modes Available**:
1. **Fresh Training**: Epsilon → 1.0 (full exploration)
2. **Resume Training**: Epsilon preserved (continue learning)

✅ **Benefits**:
- Save training time when interrupted
- Continue complex experiments without restart
- Flexible control over exploration strategy

📊 **Q-table always preserved**: Accumulates learning across all experiments

🚀 **Easy to use**: Simple checkbox in GUI, or `--no-reset-epsilon` flag in CLI
