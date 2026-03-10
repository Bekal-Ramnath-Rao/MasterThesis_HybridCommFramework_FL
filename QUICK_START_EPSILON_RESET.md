# Quick Start: Multi-Scenario RL Training with Epsilon Reset Fix

## What Was Fixed

✅ **Epsilon now resets to 1.0 for EACH new experiment** from the GUI, regardless of whether you're using the same scenario or a different one.

✅ **Q-table accumulates across ALL scenarios**, allowing you to train on multiple network conditions and build a comprehensive protocol selection policy.

## How to Train RL Agent Across Multiple Scenarios

### Step 1: First Scenario Training

1. Open the experiment GUI:
   ```bash
   cd Network_Simulation
   python3 experiment_gui.py
   ```

2. Configure:
   - **Use Case**: Emotion Recognition
   - **Protocol**: rl_unified
   - **Training Mode**: ✅ Check "End training when Q-learning value converges"
   - **Communication Model Reward**: ✅ Enabled (recommended)
   - **Scenario**: Select ONE scenario (e.g., "Poor")
   - **Rounds**: 10-50 (will train multiple episodes per round)
   - **GPU**: ✅ Enabled

3. Click **"▶️ Start Experiment"**

4. Monitor the logs - you should see:
   ```
   [Q-Learning] Experiment ID: abc12345
   [Client 1] 🔄 RESETTING EPSILON TO 1.0 (New Experiment)
   [Client 1]   ✓ Epsilon reset to: 1.0000
   ```

5. Wait for training to complete (epsilon will decay from 1.0 → 0.01)

### Step 2: Additional Scenarios

6. After first scenario completes, **repeat for each scenario**:
   - Select different scenario (e.g., "Good")
   - Click **"▶️ Start Experiment"** again
   - Epsilon will reset to 1.0 automatically
   - Q-table will accumulate new learning

7. Train on all scenarios you want:
   - Poor
   - Good
   - Moderate
   - Satellite
   - etc.

### Step 3: Converged Q-Table for Inference

8. After training on all scenarios, you'll have a **converged Q-table** in:
   ```
   shared_data/q_table_emotion_trained.pkl
   ```

9. This Q-table contains optimal protocol selections for ALL trained scenarios!

## How to Use Converged Q-Table for Inference

1. **Uncheck** "End training when Q-learning value converges" in GUI
2. Select any scenario (the Q-table already knows all scenarios)
3. Start experiment
4. The RL agent will use **greedy policy** (no exploration, only exploitation)
5. It will select the best protocol based on learned experience

## Verification

### Test the Fix

Run the verification script:
```bash
cd Network_Simulation
python3 verify_epsilon_reset.py
```

Expected output:
```
✅ ALL TESTS PASSED - Epsilon reset logic is working correctly!
```

### Check Logs

During training, look for these messages:

**Server terminal** (when experiment starts):
```
[Q-Learning] Preparing epsilon reset for new experiment: poor
[Q-Learning]   Experiment ID: abc12345
[Q-Learning]   Note: Q-table will persist across scenarios for multi-scenario training
```

**Client logs** (during initialization):
```
[Client 1] 🔄 RESETTING EPSILON TO 1.0 (New Experiment)
[Client 1]   New experiment ID: abc12345
[Client 1]   Training scenario: poor
[Client 1]   ✓ Epsilon reset to: 1.0000
[Client 1]   📊 Q-table will accumulate learning across all scenarios
```

### Monitor Q-Table Growth

Check the Q-table file size after each scenario:
```bash
ls -lh shared_data/q_table_emotion_trained.pkl
```

View recent Q-learning episodes:
```bash
tail -20 shared_data/ql_learning_log_client_1_emotion.csv
```

## Example Training Sequence

Here's a complete multi-scenario training workflow:

```bash
# Scenario 1: Poor Network (4 hours)
# GUI: Select "Poor" → Start
# Wait for completion, epsilon: 1.0 → 0.01
# Episodes: 1-500

# Scenario 2: Good Network (1 hour)
# GUI: Select "Good" → Start
# Epsilon resets: 1.0 → 0.01
# Episodes: 501-700

# Scenario 3: Satellite (3 hours)
# GUI: Select "Satellite" → Start
# Epsilon resets: 1.0 → 0.01
# Episodes: 701-1000

# Result: Q-table with 1000 episodes, covering all 3 scenarios
# Ready for inference mode!
```

## Troubleshooting

### Issue: Epsilon not resetting

**Symptom**: Logs show "Already reset epsilon for experiment '...'"

**Solution**: 
1. Stop all containers: `docker-compose down`
2. Delete flag file: `rm shared_data/reset_epsilon_flag.txt`
3. Restart experiment from GUI

### Issue: Q-table gets reset between scenarios

**Symptom**: Episode count goes back to 1 after new scenario

**Solution**:
1. Check Docker Compose file has volume mount:
   ```yaml
   volumes:
     - ./shared_data:/shared_data
   ```
2. Don't use the "Reset Q-Table" button between scenarios
3. Only reset Q-table when starting completely fresh training

## Files Modified

- ✅ `Network_Simulation/run_network_experiments.py` - Generates unique experiment ID
- ✅ `Client/Emotion_Recognition/FL_Client_Unified.py` - Reads experiment ID and resets epsilon
- ✅ `Client/rl_q_learning_selector.py` - Updated reset logic with experiment tracking

## Documentation

For more details, see:
- [EPSILON_RESET_FIX.md](../docs/EPSILON_RESET_FIX.md) - Complete technical explanation
- [RL_IMPLEMENTATION_EXPLAINED.md](../docs/RL_IMPLEMENTATION_EXPLAINED.md) - RL system overview
- [RL_TRAINING_MODE_EXPLANATION.md](../docs/RL_TRAINING_MODE_EXPLANATION.md) - Training vs inference modes

## Summary

🎯 **Goal**: Train RL agent on multiple network scenarios, accumulate learning in one Q-table

✅ **Fixed**: Epsilon now resets for each new experiment (not just new scenario)

📊 **Result**: Converged Q-table ready for inference across all trained scenarios

🚀 **Next Steps**: Start GUI, train on your desired scenarios, use converged Q-table!
