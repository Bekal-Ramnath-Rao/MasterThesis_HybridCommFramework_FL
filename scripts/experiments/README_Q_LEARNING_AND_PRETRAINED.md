# Q-Learning Data: Excel Log vs Q-Table, and How a Pre-Learned Q-Table Helps

## What You Have in This Folder

- **`Q_Learning_Excellent.xlsx`** – Exported **Q-learning log** from an experiment run under **excellent** network conditions (via the GUI “Download to Excel” on the Q-Learning tab).  
  It contains **one sheet per client** (e.g. Client 1, Client 2) with columns such as:  
  ID, Timestamp, Round, Episode, State (net/res/size/mob), Action, Reward, Q Delta, Epsilon, Avg R(100), Converged.

This is **not** the same as the **Q-table** (the matrix of Q(s,a) values). The Excel is a **history of steps** (state, action, reward, q_delta, etc.); the Q-table is what the agent uses to choose protocols and is saved as a **.pkl** file.

---

## Two Kinds of “Saved Q-Learning Data”

| What              | Format | Where it lives                    | Purpose |
|-------------------|--------|-----------------------------------|--------|
| **Q-learning log**| Excel  | e.g. `scripts/experiments/Q_Learning_Excellent.xlsx` | Analysis, plots, documentation, optional warm-start from log. |
| **Q-table**       | `.pkl` | Client run dir, e.g. `q_table_emotion_client_1.pkl`  | Actual policy; loaded at startup to reuse learned behaviour. |

---

## How an Already-Learned Q-Table Helps in Future Runs

The **Q-table** is the only thing the client **loads** to “reuse” past learning. The current implementation already supports this.

### 1. **Automatic load on startup**

- When a unified client starts, it creates a `QLearningProtocolSelector` with a `save_path` (e.g. `q_table_emotion_client_1.pkl`).
- In `__init__`, the selector calls **`load_q_table()`**.
- If that file exists and its shape matches (same state/action dimensions), the client **starts with that Q-table** instead of zeros.

So: **any run that leaves behind a `.pkl` in the same path will be reused next time** the client runs with that path.

### 2. **Benefits of reusing a pre-learned Q-table**

- **Faster convergence** – The agent does not start from zero; it starts from a policy that already associates (state, action) with value.
- **Better early behaviour** – From round 1, protocol choices are informed by past experience (e.g. “excellent” run) instead of random exploration.
- **Transfer across scenarios** – You can take a Q-table learned in one scenario (e.g. excellent) and use it as the starting point for another (e.g. good or poor). Same state/action space, so the same `.pkl` shape works; the agent then adapts with further updates.

### 3. **How to use a pre-learned Q-table in practice**

- **Option A – Same machine / same run dir**  
  Run an “excellent” (or any) experiment once. The client writes `q_table_emotion_client_<id>.pkl` in its working directory. The next run that uses the same working directory will automatically load it.

- **Option B – Copy into a known location**  
  After a good run (e.g. excellent), copy the `.pkl` into a folder you keep for “pretrained” tables (e.g. `scripts/experiments/pretrained_q_tables/`). When launching a **new** experiment, configure the client so its `save_path` points to that file (or copy that file to the client’s run dir as `q_table_emotion_client_<id>.pkl`). Then the client starts with that policy.

- **Option C – Docker / shared volume**  
  If the client runs in Docker, mount a host directory (e.g. `scripts/experiments/pretrained_q_tables/`) where you put the `.pkl` and set the client’s working dir or `save_path` so it reads/writes there. Then every run on that setup can reuse the same pre-learned Q-table.

So: **the Excel file you saved does not replace the .pkl**; it is for analysis and documentation. The **already-learned Q-table** that helps in future is the **.pkl** file, and it helps by being **loaded automatically at client startup** when the path and shape match.

---

## How the Excel “Excellent” Log Can Be Used

- **Analysis** – Inspect which (state, action) pairs were tried, rewards, q_delta over time, and which client converged (Converged column).
- **Documentation** – Keep one Excel per scenario (e.g. `Q_Learning_Excellent.xlsx`, `Q_Learning_Good.xlsx`) to compare behaviour across network conditions.
- **Future extension** – You could add a script that:
  - Reads the Excel (or the same data from the DB),
  - Builds or approximates a Q-table from the log (e.g. from rewards and state/action columns),
  - Saves it as a `.pkl` in `scripts/experiments/pretrained_q_tables/` (e.g. `q_table_emotion_excellent.pkl`),
  - Then use that `.pkl` as the initial Q-table for new runs (by pointing the client’s `save_path` to it or copying it to the client’s run dir).

The current code does **not** build a Q-table from the Excel; it only **loads** an existing `.pkl` Q-table. So today, the Excel is for analysis and record-keeping; the **pre-learned Q-table** that directly helps future runs is the **.pkl** file, and it helps by being loaded at startup as described above.

---

## Evaluating unified (with past experience) vs single-protocol implementation

Once you have learned from past experience (Q-tables saved as `.pkl`, e.g. in `shared_data/` or from a previous run), you can **evaluate** the unified use case (RL-based protocol selection using that experience) against **single-protocol** runs on the same scenario for a fair comparison.

### What you are comparing

| Run type | What it does | Result folder pattern |
|----------|--------------|------------------------|
| **Unified (with past experience)** | Clients load existing `.pkl`, use RL to choose protocol each round | `rl_unified_<scenario>` |
| **Single protocol** | One protocol only (e.g. MQTT or DDS), no RL | `mqtt_<scenario>`, `dds_<scenario>`, etc. |

Use the **same** use case, **same** network scenario(s), and **same** number of rounds (and same GPU/quantization settings if applicable) so the comparison is fair.

### Step 1: Ensure unified runs use past experience

- **Docker (GUI or `run_network_experiments.py`):** The unified compose file mounts `shared_data` at `/shared_data`. Unified clients save and load Q-tables there (`q_table_emotion_client_1.pkl`, etc.). So:
  - After **one** RL-unified run, the next RL-unified run **automatically** loads from `shared_data/` (no extra step).
  - To start from a **specific** pretrained set: copy your `.pkl` files into `shared_data/` with names `q_table_emotion_client_1.pkl`, `q_table_emotion_client_2.pkl`, then start the experiment. Alternatively, set env `PRETRAINED_Q_TABLE_DIR` to a path that is mounted in the container and contains those filenames (see Option C in “How to use a pre-learned Q-table” above).

So: for “unified with past experience”, either run unified once and then run again (same setup), or put the desired `.pkl` files in `shared_data/` (or in a mounted dir pointed to by `PRETRAINED_Q_TABLE_DIR`) before starting.

### Step 2: Run unified experiment (with loaded Q-table)

**Via GUI (Network Simulation / Experiment GUI):**

1. Use case: e.g. Emotion (or Mental State / Temperature).
2. Under **Communication Protocols**, select **🤖 RL-Unified (Dynamic Selection)** only (or include it with others if you want multiple protocol types in one batch).
3. Select **scenario(s)** (e.g. excellent, good, poor) and **rounds**.
4. Start experiment. Results go to e.g. `experiment_results/<folder>/rl_unified_<scenario>/`.

**Via CLI:**

```bash
cd Network_Simulation
python run_network_experiments.py --use-case emotion --protocols rl_unified \
  --scenarios excellent good poor --rounds 50
```

Results are under `experiment_results/` (or `experiment_results_baseline/` if baseline mode) in subdirs like `rl_unified_excellent`, `rl_unified_good`, etc.

### Step 3: Run single-protocol experiments (same scenarios and rounds)

Use the **same** scenarios and rounds as in Step 2, but with one protocol per run.

**Via GUI:**

1. Same use case and scenario(s) as Step 2.
2. Under **Communication Protocols**, select **only one** protocol (e.g. MQTT, or DDS), **not** RL-Unified.
3. Same rounds. Start. Repeat for each protocol you want (MQTT, AMQP, gRPC, QUIC, DDS).

**Via CLI (one protocol per call):**

```bash
# Same scenarios and rounds as unified
python run_network_experiments.py --use-case emotion --protocols mqtt \
  --scenarios excellent good poor --rounds 50

python run_network_experiments.py --use-case emotion --protocols dds \
  --scenarios excellent good poor --rounds 50
# ... repeat for amqp, grpc, quic as needed
```

Results will be in subdirs like `mqtt_excellent`, `dds_excellent`, etc., under the same parent folder.

### Step 4: Compare results

- **Same parent folder:** e.g. `experiment_results/emotion_<timestamp>/` contains both `rl_unified_<scenario>` and `mqtt_<scenario>`, `dds_<scenario>`, etc.
- **Metrics:** Use the JSON/CSV result files in each subdir (e.g. `rl_unified_training_results.json`, `mqtt_training_results.json`) and compare:
  - Rounds to convergence (if applicable)
  - Final accuracy / loss
  - Training time or round trip times (if you collect them)
- **Scripts:** You can use existing comparison/visualization scripts (e.g. under `Network_Simulation/`: `evaluate_all.py`, `compare_protocols.py`, `visualize_results.py`) by pointing them at that experiment folder so unified and single-protocol runs are compared together.

### Summary checklist

1. **Past experience in place:** Q-tables in `shared_data/` (or loaded via `PRETRAINED_Q_TABLE_DIR`) so unified clients start from learned policy.
2. **Unified run:** Select RL-Unified, run with chosen scenario(s) and rounds → get `rl_unified_<scenario>`.
3. **Single-protocol runs:** Same scenario(s) and rounds, one protocol at a time (MQTT, DDS, etc.) → get `mqtt_<scenario>`, `dds_<scenario>`, etc.
4. **Compare:** Same experiment folder; use result JSONs/CSVs and/or evaluation scripts to compare unified (with past experience) vs each single-protocol implementation.
