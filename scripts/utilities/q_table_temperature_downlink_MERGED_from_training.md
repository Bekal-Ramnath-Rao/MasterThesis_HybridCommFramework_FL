# Temperature regulation downlink Q-table — from training export (Excel)

Each block is **12 states × 5 actions** (comm × resource × battery × protocol). Values are the **last logged `q_value`** for each distinct *(coarse scenario, discrete state, action)* tuple in the export. **(s,a) pairs that never received an update stay 0.**

## Source

| Field | Value |
|---|---|
| Workbook | `scripts/experiments/Temperature_final_result.xlsx` |
| Sheet | `Client 1` |
| Rows used | 197 downlink rows (after dropping incomplete state/action rows) |

## Coarse scenario mapping (matches emotion-style slices)

| Excel `state_data_network_scenario` | Coarse block |
|---|---|
| `excellent`, `good` | **excellent** |
| `moderate` | **moderate** |
| `congested_light` | **poor** |

**Largest total Q per coarse scenario** (sum of `Q(s,a)` over the 12 discrete states, one winner per scenario): `{'excellent': 'amqp', 'moderate': 'mqtt', 'poor': 'mqtt'}`

## excellent

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 15.832397 | 25.599737 | 3.730810 | 10.616780 | 8.961320 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 2.428084 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 11.735669 | 6.359447 | 2.533520 | 2.316338 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 3.527245 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
## moderate

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 37.114481 | 14.238041 | 7.112136 | 13.662249 | 9.600604 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
## poor

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 40.619643 | 15.671956 | 16.573898 | 13.533110 | 5.759418 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 4.547173 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 8.867929 | 5.204991 | 7.201327 | 0.000000 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
