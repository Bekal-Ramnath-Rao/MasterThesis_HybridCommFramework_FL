# Quick Reference: RL Training Modes

## Two Training Modes in GUI

### ✅ Fresh Training (Reset Epsilon to 1.0)
**Default Mode** - Start exploration from scratch

```
Epsilon: 1.0 → 0.01
Q-table: Preserved & Accumulated
Rewards: Fresh accumulation
Episodes: Continue counting

Use when: Training NEW scenario
```

**GUI**: ✅ **Check** "Reset Epsilon to 1.0 (Fresh Training)"

---

### 📍 Resume Training (Preserve Epsilon)
**Resume Mode** - Continue from previous state

```
Epsilon: 0.3 → 0.01 (preserved)
Q-table: Preserved & Accumulated
Rewards: Continued accumulation
Episodes: Continue counting

Use when: Interrupted/Extending training
```

**GUI**: ❌ **Uncheck** "Reset Epsilon to 1.0 (Fresh Training)"

---

## Quick Examples

### Multi-Scenario Training (Fresh each time)
```
✅ Poor network    → Epsilon: 1.0 → 0.01
✅ Good network    → Epsilon: 1.0 → 0.01
✅ Satellite       → Epsilon: 1.0 → 0.01

Result: Comprehensive Q-table with all scenarios
```

### Resume Interrupted Training
```
1st attempt: ✅ Poor → 1.0 → 0.5 [interrupted]
2nd attempt: ❌ Poor → 0.5 → 0.01 [completed]

Result: Saved time, continued from 0.5
```

### Extend Training Duration
```
Rounds 1-10:  ✅ Poor → 1.0 → 0.05
Rounds 11-20: ❌ Poor → 0.05 → 0.01

Result: More training episodes without reset
```

---

## Command Line Flags

**Fresh Training** (default):
```bash
python3 run_network_experiments.py --protocols rl_unified --scenarios poor
```

**Resume Training**:
```bash
python3 run_network_experiments.py --protocols rl_unified --scenarios poor --no-reset-epsilon
```

---

## Log Messages

**Fresh Training**:
```
🔄 RESETTING EPSILON TO 1.0 (Fresh Training)
   Current epsilon before reset: 0.3456
   ✓ Epsilon reset to: 1.0000
```

**Resume Training**:
```
📍 CONTINUING WITH PREVIOUS EPSILON (Resume Mode)
   Current epsilon (preserved): 0.3456
   📊 Q-table, rewards, and learning progress will continue
```

---

## When to Use Each

| Situation | Mode | Why |
|-----------|------|-----|
| New scenario | Fresh ✅ | Full exploration needed |
| Training crashed | Resume ❌ | Don't waste progress |
| Extend training | Resume ❌ | Continue from where left off |
| Multi-scenario | Fresh ✅ | Each scenario needs exploration |

---

## Quick Verification

**Check current epsilon before resuming**:
```bash
tail -1 shared_data/ql_learning_log_client_1_emotion.csv | column -t -s,
```

**Check flag file**:
```bash
cat shared_data/reset_epsilon_flag.txt
# Look for: reset_epsilon=1.0 (fresh) or 0.0 (resume)
```

---

## Key Benefit

**Time Savings**: Resume mode can save hours by not restarting exploration from 1.0 when interrupted or extending training.

**Example**: If training was interrupted at epsilon=0.3, resuming saves ~1000 exploration steps compared to restarting from 1.0.

---

## Full Documentation

See [RESUME_TRAINING_FEATURE.md](./docs/RESUME_TRAINING_FEATURE.md) for complete details.
