# Temperature regulation uplink Q-table — from training export (Excel)

Each **12×5** block is one coarse network scenario. Values are the **last logged `q_value`** per *(coarse scenario, discrete state, action)* in the export; **unvisited (s,a) pairs remain 0**.

## Source

| Field | Value |
|---|---|
| Workbook | `scripts/experiments/Temperature_final_result.xlsx` |
| Sheet | `Client 1` |
| Rows used | 197 uplink rows (after dropping incomplete state/action rows) |

## Coarse scenario mapping (matches emotion-style slices)

| Excel `state_data_network_scenario` | Coarse block |
|---|---|
| `excellent`, `good` | **excellent** |
| `moderate` | **moderate** |
| `congested_light` | **poor** |

**Largest total Q per coarse scenario** (sum over the 12 discrete states): `{'excellent': 'http3', 'moderate': 'mqtt', 'poor': 'mqtt'}`

## excellent

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 2.524861 | 2.773342 | 0.000000 | 35.759589 | 5.210714 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 0.000000 | 4.562755 | 0.000000 | 4.570430 | 2.740996 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 0.000000 | 10.049336 | 0.000000 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 2.185120 | 0.000000 | 0.000000 |
## moderate

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 39.873961 | 3.746188 | 0.000000 | 6.324499 | 9.520665 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 7.345448 | 4.567548 | 0.000000 | 0.000000 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 0.000000 | 2.542768 | 0.000000 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 2.175854 | 0.000000 | 0.000000 |
## poor

| row | state (comm / resource / battery) | mqtt | amqp | grpc | http3 | dds |
|---:|---|---:|---:|---:|---:|---:|
| 0 | low/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 1 | low/high/low | 46.744207 | 5.844722 | 0.000000 | 16.395611 | 10.820459 |
| 2 | low/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 3 | low/low/low | 4.802150 | 2.533954 | 0.000000 | 2.312731 | 0.000000 |
| 4 | mid/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 5 | mid/high/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 6 | mid/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 7 | mid/low/low | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 8 | high/high/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 9 | high/high/low | 0.000000 | 0.000000 | 8.455067 | 0.000000 | 0.000000 |
| 10 | high/low/high | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 11 | high/low/low | 0.000000 | 0.000000 | 2.184947 | 0.000000 | 0.000000 |
