# Epsilon Reset Fix for Multi-Scenario RL Training

## Problem

When selecting network scenarios in the GUI and starting RL agent training, epsilon was not being reset to 1.0 for each new experiment. This occurred because:

1. The reset mechanism tracked only the **scenario name** (`last_scenario`)
2. If you ran the same scenario twice, the second run would skip the epsilon reset
3. This prevented fresh exploration when restarting training with the same network conditions

## Solution

The fix introduces a **unique experiment ID** for each GUI experiment start:

### Key Changes

1. **`run_network_experiments.py`** (lines 866-898):
   - Generates a unique `experiment_id` (UUID) for each experiment
   - Writes both `experiment_id` and `scenario` to the reset flag file
   - This ensures each GUI "Start Experiment" click creates a fresh training session

2. **`FL_Client_Unified.py`** (lines 547-609):
   - Now reads both `experiment_id` and `scenario` from flag file
   - Compares `experiment_id` instead of just `scenario` to decide if reset is needed
   - Prevents duplicate resets within the same experiment while allowing resets across experiments

3. **`rl_q_learning_selector.py`** (lines 285-307):
   - Updated `reset_epsilon()` to track both `experiment_id` and `scenario`
   - Added clear documentation that **only epsilon resets, not the Q-table**
   - This enables multi-scenario training with one accumulated Q-table

## How It Works

### Multi-Scenario Training Flow

1. **First Experiment - "Poor" Network**:
   ```
   GUI Start → experiment_id=abc123 → epsilon=1.0 → Train on "poor" → Q-table updated
   ```

2. **Second Experiment - "Good" Network**:
   ```
   GUI Start → experiment_id=def456 → epsilon=1.0 → Train on "good" → Q-table updated (accumulated)
   ```

3. **Third Experiment - "Poor" Network Again**:
   ```
   GUI Start → experiment_id=ghi789 → epsilon=1.0 → Train on "poor" → Q-table updated (accumulated)
   ```

### Key Benefits

✅ **Fresh exploration for each GUI experiment** (epsilon always resets to 1.0)  
✅ **Q-table accumulates across all scenarios** (learns optimal protocols for all network conditions)  
✅ **No duplicate resets within same experiment** (late-joining clients use same experiment_id)  
✅ **Ready for inference** (converged Q-table contains experience from all scenarios)  

## Usage

### Training Across Multiple Scenarios

1. Open the FL Experiment GUI
2. Select **rl_unified** protocol
3. Check **"End training when Q-learning value converges"** (recommended)
4. Select one network scenario (e.g., "Poor")
5. Click **"▶️ Start Experiment"**
6. Wait for training to complete
7. Select a different scenario (e.g., "Good")
8. Click **"▶️ Start Experiment"** again
9. Repeat for all scenarios you want to train on

**Result**: After training on all scenarios, you'll have a `q_table_emotion_trained.pkl` file in `shared_data/` that contains optimal protocol selections for ALL trained network conditions.

### Inference with Converged Q-Table

Once you have a converged Q-table from multi-scenario training:

1. Uncheck **"End training when Q-learning value converges"** (inference mode)
2. The agent will use the trained Q-table for exploitation (greedy policy)
3. No exploration, just optimal protocol selection based on learned experience

## Verification

### Check Epsilon Reset in Logs

When starting a new experiment, you should see in the server terminal:

```
==================================================
[Q-Learning] Preparing epsilon reset for new experiment: poor
==================================================
[Q-Learning] ✓ Created reset flag file: .../shared_data/reset_epsilon_flag.txt
[Q-Learning]   Experiment ID: abc12345
[Q-Learning]   Scenario: poor
[Q-Learning]   All clients will reset epsilon to 1.0 on initialization
[Q-Learning]   Note: Q-table will persist across scenarios for multi-scenario training
==================================================
```

And in the client logs:

```
==================================================
[Client 1] 🔄 RESETTING EPSILON TO 1.0 (New Experiment)
[Client 1]   New experiment ID: abc12345
[Client 1]   Previous experiment ID: xyz98765
[Client 1]   Training scenario: poor
[Client 1]   Current epsilon before reset: 0.2345
[Client 1]   ✓ Epsilon reset to: 1.0000
[Client 1]   📊 Q-table will accumulate learning across all scenarios
==================================================
```

### Verify Q-Table Persistence

After training on multiple scenarios, check the Q-table file:

```bash
# Check if Q-table file exists
ls -lh shared_data/q_table_emotion_trained.pkl

# View Q-learning logs to see accumulated episodes
cat shared_data/ql_learning_log_client_1_emotion.csv | tail -20
```

The Q-table file should grow in size as you train on more scenarios, and the episode count should increase across experiments.

## Technical Details

### Experiment ID Generation

```python
import uuid
experiment_id = str(uuid.uuid4())[:8]  # Short unique ID (e.g., "a1b2c3d4")
```

### Flag File Format

```
experiment_id=a1b2c3d4
scenario=poor
timestamp=1709856234.567
reset_epsilon=1.0
```

### Client-Side Tracking

The `QLearningProtocolSelector` now tracks:
- `last_experiment_id`: Prevents duplicate resets within same experiment
- `last_scenario`: For logging/debugging purposes
- `epsilon`: Always reset to 1.0 when new `experiment_id` is detected
- `q_table`: Never reset, accumulates across all experiments

## Troubleshooting

### Epsilon Not Resetting

**Symptoms**: Client logs show "Already reset epsilon for experiment '...', skipping reset"

**Cause**: Same experiment_id detected (flag file not updated between experiments)

**Fix**: Manually delete the flag file before starting new experiment:
```bash
rm shared_data/reset_epsilon_flag.txt
```

### Q-Table Not Accumulating

**Symptoms**: Q-table file size doesn't grow, episode count resets

**Cause**: Q-table file being deleted or overwritten

**Fix**: Check Docker compose file - ensure `shared_data/` is mounted as a volume:
```yaml
volumes:
  - ./shared_data:/shared_data
```

## Related Files

- [`run_network_experiments.py`](../Network_Simulation/run_network_experiments.py) - Creates flag file with experiment ID
- [`FL_Client_Unified.py`](../Client/Emotion_Recognition/FL_Client_Unified.py) - Reads flag file and resets epsilon
- [`rl_q_learning_selector.py`](../Client/rl_q_learning_selector.py) - Implements epsilon reset logic
- [`experiment_gui.py`](../Network_Simulation/experiment_gui.py) - GUI that triggers experiments

## See Also

- [Q-Learning Implementation Explained](./RL_IMPLEMENTATION_EXPLAINED.md)
- [RL Training Mode Explanation](./RL_TRAINING_MODE_EXPLANATION.md)
- [Q-Table Reset Feature](./Q_TABLE_RESET_FEATURE.md)
