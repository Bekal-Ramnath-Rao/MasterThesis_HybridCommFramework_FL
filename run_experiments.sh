#!/bin/bash

LOG_DIR="experiment_logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Logging to: $LOG_DIR"

# Experiment 1: Without Quantization
echo "Running Experiment 1 (without quantization - baseline)..."
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion --enable-gpu \
    --rounds 1000 --protocols mqtt amqp grpc quic dds --scenarios excellent good moderate poor very_poor congested_light congested_moderate congested_heavy\
    2>&1 | tee "$LOG_DIR/exp1_baseline_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Experiment 1 failed!"
    exit 1
fi

# Experiment 2: With Quantization
echo "Running Experiment 2 (with quantization)..."
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion --enable-gpu --use-quantization \
    --quantization-strategy parameter_quantization \
    --quantization-bits 16 --rounds 1000 \
    --protocols mqtt amqp grpc quic dds --scenarios excellent good moderate poor very_poor congested_light congested_moderate congested_heavy \
    2>&1 | tee "$LOG_DIR/exp2_quantized_${TIMESTAMP}.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "Experiment 2 failed!"
    exit 1
fi

echo "All experiments completed! Logs saved in $LOG_DIR"
