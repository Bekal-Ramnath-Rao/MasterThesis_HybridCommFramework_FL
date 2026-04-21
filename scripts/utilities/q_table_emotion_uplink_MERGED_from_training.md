# Emotion uplink Q-table — merged from training archives

Each **12×5** block is one coarse network scenario. Values come from real Q-learning checkpoints; **unvisited (s,a) pairs remain 0**.

## How this merge was built

| Scenario slice | Source archive (strongest updates on that slice) |
|---|---|
| excellent | `q_table_emotion_uplink_trained_archive_20260412_182029_good.pkl` (good → coarse excellent) |
| moderate | `q_table_emotion_uplink_trained_archive_20260412_192556_moderate.pkl` |
| poor | `q_table_emotion_uplink_trained_archive_20260412_234520_congested_light.pkl` (updates on poor index) |

**Merged pickle (loadable):** `scripts/utilities/q_table_emotion_uplink_merged_from_training_archives.pkl`

**Source converged maps:** `{'good': {'excellent': 'amqp'}, 'moderate': {'moderate': 'dds'}, 'congested_light': {'poor': 'dds'}}`

## excellent — 12 states × 5 actions

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 6.356513 | 30.124878 | 0.000000 | 3.527708 | 6.999733 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 0.000000 | 0.000000 | 2.152827 | 8.034860 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
## moderate — 12 states × 5 actions

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 6.290428 | 4.901059 | 1.859994 | 5.077052 | 20.523484 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 1.351949 | 2.451820 | 1.213991 | 9.987494 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
## poor — 12 states × 5 actions

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 8.424595 | 1.937185 | 5.845079 | 2.627193 | 32.956445 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 2.439603 | 3.796002 | 1.105020 | 0.000000 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
