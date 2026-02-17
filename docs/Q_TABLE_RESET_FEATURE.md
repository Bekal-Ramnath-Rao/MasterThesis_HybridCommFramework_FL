# Q-Table Reset Feature

## Overview

This feature allows you to reset the Q-table to start fresh training, which is useful when:
- New protocols are added (e.g., HTTP/3)
- You want to retrain the RL agent from scratch
- You need to clear learned knowledge for a new experiment

## Features

### 1. Automatic Excel Backup Before Reset

**Location**: `GUI/q_learning_logs_tab.py`

Before resetting the Q-table, the system automatically:
- Checks if there's any Q-learning data to backup
- Exports all Q-learning logs to an Excel file with timestamp
- Saves the backup to `shared_data/q_learning_backup_YYYYMMDD_HHMMSS.xlsx`
- Only proceeds with reset after backup is confirmed

### 2. Q-Table Reset Button

**Location**: `GUI/q_learning_logs_tab.py` → Q-Learning Tab

A new **"🔄 Reset Q-Table"** button is available in the Q-Learning logs tab:
- Red background to indicate destructive action
- Tooltip explains the feature
- Confirmation dialog before reset
- Automatically exports Excel before resetting

### 3. Reset Functionality

**Location**: `Client/rl_q_learning_selector.py` → `reset_q_table()`

The reset method:
- Resets Q-table to all zeros (fresh start)
- Resets epsilon to 1.0 (full exploration)
- Clears all statistics (episode count, rewards, protocol usage)
- Clears history (state, action, reward history)
- Deletes saved `.pkl` Q-table files

## Usage

### Via GUI

1. Open the **Q-Learning** tab in the experiment GUI
2. Click **"🔄 Reset Q-Table"** button
3. Confirm the reset (Excel backup will be created automatically)
4. Next experiment will start with a fresh Q-table

### Via Code

```python
from Client.rl_q_learning_selector import QLearningProtocolSelector

selector = QLearningProtocolSelector(save_path="q_table.pkl")

# Reset Q-table
selector.reset_q_table()
```

## Excel Backup Format

The automatic backup includes:
- **File name**: `q_learning_backup_YYYYMMDD_HHMMSS.xlsx`
- **Location**: `shared_data/` directory
- **Format**: One sheet per client (Client 1, Client 2, etc.)
- **Columns**: 
  - ID, Timestamp, Round, Episode
  - State (network, resource, model_size, mobility)
  - Action, Reward, Q Delta, Epsilon
  - Avg Reward (last 100), Converged

## Inference Uses Learned Q-Table

**Important**: During inference (when `USE_QL_CONVERGENCE=False`), the system uses the learned Q-table:

```python
# In select_protocol() when training=False
action_idx = np.argmax(self.q_table[state_idx])  # Uses learned Q-table
```

The Q-table is:
- **Loaded at startup** from `.pkl` files
- **Used for inference** when `training=False` (pure exploitation)
- **Updated during training** when `training=True` (epsilon-greedy)

## Files Modified

1. **`Client/rl_q_learning_selector.py`**
   - Added `reset_q_table()` method

2. **`GUI/q_learning_logs_tab.py`**
   - Added "Reset Q-Table" button
   - Added `reset_q_table_with_backup()` method
   - Automatic Excel export before reset

## Example Workflow

### Scenario: Adding HTTP/3 Protocol

1. **Before Reset**:
   - Q-table learned with 5 protocols: MQTT, AMQP, gRPC, QUIC, DDS
   - HTTP/3 added to protocol list

2. **Reset Process**:
   - Click "Reset Q-Table" button
   - System exports Excel backup automatically
   - Q-table reset to zeros
   - All `.pkl` files deleted

3. **After Reset**:
   - Next experiment starts with fresh Q-table
   - All 6 protocols (including HTTP/3) start with equal Q-values
   - RL agent learns optimal protocol selection from scratch

## Safety Features

- **Confirmation dialog**: Prevents accidental resets
- **Automatic backup**: Excel export before reset
- **Error handling**: Graceful handling of missing files or errors
- **User feedback**: Clear messages about what was deleted/backed up

## Notes

- **Q-learning logs (database)**: Not deleted, only Q-table files (`.pkl`) are deleted
- **Excel backup**: Always created if data exists, even if reset fails
- **Multiple clients**: All client Q-tables are reset (Client 1, Client 2, etc.)
- **Fresh start**: After reset, epsilon starts at 1.0 (100% exploration)
