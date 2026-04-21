# Emotion downlink Q-table — merged from training archives

Each block is **12 states × 5 actions** (comm × resource × battery × protocol). Values come from Q-learning; **(s,a) pairs that were never updated stay 0**.

## Source per scenario slice (auto-picked: max |Q| on that slice)

| Scenario | Archive |
|---|---|
| excellent | `/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL/shared_data/q_table_emotion_downlink_trained_archive_20260412_182029_good.pkl` |
| moderate | `/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL/shared_data/q_table_emotion_downlink_trained_archive_20260412_192556_moderate.pkl` |
| poor | `/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL/shared_data/q_table_emotion_downlink_trained_archive_20260412_234520_congested_light.pkl` |

**Pickle:** `/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL/scripts/utilities/q_table_emotion_downlink_merged_from_training_archives.pkl`

**converged_protocol_by_scenario (from sources):** `{'q_table_emotion_downlink_trained_archive_20260412_182029_good.pkl': {'excellent': 'http3'}, 'q_table_emotion_downlink_trained_archive_20260412_234520_congested_light.pkl': {'poor': 'grpc'}, 'q_table_emotion_downlink_trained_archive_20260412_192556_moderate.pkl': {'moderate': 'grpc'}}`

## excellent

| row | state | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 3.445435 | 7.075261 | 3.669053 | 24.721581 | 4.567788 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 0.000000 | 2.023504 | 3.347900 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 2.754826 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
## moderate

| row | state | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 0.000000 | 0.000000 | 3.981980 | 0.000000 | 0.000000 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 1.092804 | 1.366776 | 15.153831 | 1.302277 | 2.083756 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 1.234372 | 0.000000 | 6.929845 | 2.420786 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
## poor

| row | state | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 1.087083 | 0.000000 | 5.401143 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 8.483398 | 1.033719 | 0.000000 |
