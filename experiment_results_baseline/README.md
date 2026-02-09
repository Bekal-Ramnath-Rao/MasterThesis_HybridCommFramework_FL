# Baseline Experiment Results

This folder contains baseline FL experiment results for all protocols **without any network conditions applied**.

## Purpose

The baseline results serve as a reference point for comparing:
- Network condition experiments (latency, bandwidth, packet loss, jitter)
- Protocol performance under ideal conditions
- Round trip time (RTT) for each FL round
- Training convergence without network interference

## Structure

```
experiment_results_baseline/
├── emotion/          # Emotion recognition baseline results
├── mentalstate/      # Mental state baseline results
├── temperature/      # Temperature prediction baseline results
└── README.md         # This file
```

## Running Baseline Experiments

To generate baseline results (without network conditions):

```bash
# For emotion recognition
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --rounds 10 \
    --baseline

# For temperature prediction
python3 Network_Simulation/run_network_experiments.py \
    --use-case temperature \
    --enable-gpu \
    --rounds 10 \
    --baseline

# For mental state recognition
python3 Network_Simulation/run_network_experiments.py \
    --use-case mentalstate \
    --enable-gpu \
    --rounds 10 \
    --baseline
```

Results will be automatically stored in `experiment_results_baseline/<use_case>/`

## Baseline Data Collected

For each protocol (MQTT, AMQP, gRPC, QUIC, DDS), the following metrics are collected:

### Round Trip Time (RTT)
- **Definition**: Time from receiving global model to completing training and receiving next global model
- **Measurement**: Start = global model received, End = next start training signal
- **Stored as**: Average RTT across all FL rounds

### Training Metrics
- Accuracy per round
- Loss per round
- Model size
- Communication overhead

### Resource Usage
- CPU usage
- Memory usage  
- Network I/O

## Usage in Comparison

The FL Training Dashboard (`fl_training_dashboard.py`) automatically:
1. Loads baseline results from this folder
2. Compares current experiment with baseline
3. Shows performance delta (improvement/degradation)
4. Highlights RTT differences

Example comparison output:
```
Baseline RTT: 2.5s/round
Current RTT:  5.8s/round
Delta:        +132% (degradation due to 200ms latency)
```

## Notes

- **Run baseline once** for each use case and protocol combination
- **Re-run baseline** if you change:
  - Model architecture
  - Number of clients
  - Training hyperparameters
  - Hardware setup
- **Keep baseline updated** with your experiment versions
- Baseline assumes **excellent network conditions** (minimal latency, no packet loss)

## File Format

Each baseline experiment creates:
```
<protocol>_baseline_results.json    # Training metrics
<protocol>_baseline_rtt.json        # Round trip time data
<protocol>_baseline_metadata.json   # Experiment configuration
```

Example structure:
```json
{
  "protocol": "mqtt",
  "use_case": "emotion",
  "num_rounds": 10,
  "avg_rtt_per_round": 2.5,
  "rtt_per_round": [2.3, 2.5, 2.4, 2.6, ...],
  "timestamp": "2026-01-28T14:30:00",
  "network_conditions": "none"
}
```
