#!/bin/bash
# Quick Reference - Experiment Commands
# Copy and paste these commands to run various experiment configurations

################################################################################
# SCENARIO 1: Without Quantization (All 9 Network Scenarios)
################################################################################

# 1.1 Run all use cases, all protocols, all scenarios - NO quantization
./run_comprehensive_experiments.sh --scenario 1 --rounds 100

# 1.2 Emotion recognition only - NO quantization
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --rounds 100

# 1.3 Mental state recognition only - NO quantization
./run_comprehensive_experiments.sh --use-case mentalstate --scenario 1 --rounds 100

# 1.4 Temperature regulation only - NO quantization
./run_comprehensive_experiments.sh --use-case temperature --scenario 1 --rounds 100

# 1.5 MQTT protocol only - NO quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 1 --protocol mqtt --rounds 100

# 1.6 AMQP protocol only - NO quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 1 --protocol amqp --rounds 100

# 1.7 gRPC protocol only - NO quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 1 --protocol grpc --rounds 100

# 1.8 QUIC protocol only - NO quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 1 --protocol quic --rounds 100

# 1.9 DDS protocol only - NO quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 1 --protocol dds --rounds 100

################################################################################
# SCENARIO 2: With Quantization (All 9 Network Scenarios)
################################################################################

# 2.1 Run all use cases, all protocols, all scenarios - WITH quantization
./run_comprehensive_experiments.sh --scenario 2 --rounds 100

# 2.2 Emotion recognition only - WITH quantization
./run_comprehensive_experiments.sh --use-case emotion --scenario 2 --rounds 100

# 2.3 Mental state recognition only - WITH quantization
./run_comprehensive_experiments.sh --use-case mentalstate --scenario 2 --rounds 100

# 2.4 Temperature regulation only - WITH quantization
./run_comprehensive_experiments.sh --use-case temperature --scenario 2 --rounds 100

# 2.5 MQTT protocol only - WITH quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 2 --protocol mqtt --rounds 100

# 2.6 AMQP protocol only - WITH quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 2 --protocol amqp --rounds 100

# 2.7 gRPC protocol only - WITH quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 2 --protocol grpc --rounds 100

# 2.8 QUIC protocol only - WITH quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 2 --protocol quic --rounds 100

# 2.9 DDS protocol only - WITH quantization, all scenarios
./run_comprehensive_experiments.sh --scenario 2 --protocol dds --rounds 100

################################################################################
# VALIDATION & TESTING
################################################################################

# V1 Quick validation - Single protocol, single scenario, 5 rounds
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --protocol mqtt --rounds 5

# V2 Verify emotion setup - All protocols, scenario 1, 10 rounds
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --rounds 10

# V3 Verify all use cases - Single protocol, scenario 1, 10 rounds each
./run_comprehensive_experiments.sh --scenario 1 --protocol mqtt --rounds 10

# V4 Test quantization - Single use case, protocol, 5 rounds
./run_comprehensive_experiments.sh --use-case emotion --protocol mqtt --rounds 5

################################################################################
# ADVANCED OPTIONS
################################################################################

# A1 Run all experiments with custom 50 rounds (faster execution)
./run_comprehensive_experiments.sh --rounds 50

# A2 Skip Docker build (use cached images) - speeds up start
./run_comprehensive_experiments.sh --skip-build --rounds 100

# A3 Skip result consolidation (run experiments only)
./run_comprehensive_experiments.sh --skip-consolidate --rounds 100

# A4 Combine options - emotion, scenario 1, skip build, 75 rounds
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --skip-build --rounds 75

################################################################################
# NETWORK SCENARIOS BREAKDOWN
################################################################################

# Network Scenarios in Script:
# 1. excellent        - No latency, no packet loss
# 2. good             - 20ms latency, 0.1% loss
# 3. moderate         - 50ms latency, 0.5% loss
# 4. poor             - 100ms latency, 1% loss
# 5. very_poor        - 200ms latency, 2% loss
# 6. satellite        - 600ms latency, 0.5% loss
# 7. congested_light  - Light traffic congestion
# 8. congested_moderate - Moderate traffic congestion
# 9. congested_heavy  - Heavy traffic congestion

################################################################################
# MONITORING DURING EXPERIMENTS
################################################################################

# M1 Watch GPU memory usage (real-time)
nvidia-smi -l 1

# M2 Monitor container resource usage
docker stats

# M3 View live logs from all experiments
tail -f logs/experiments_*/emotion_scenario1_excellent_disabled_mqtt.log

# M4 Count how many experiments have completed
ls logs/experiments_*/*.log | wc -l

################################################################################
# RESULT ANALYSIS
################################################################################

# R1 View consolidation results for emotion
tail -20 Server/Emotion_Regulation/results/*.json

# R2 List all experiment result folders
ls -lh experiment_results/

# R3 Check latest results directory
ls -td experiment_results/*/ | head -1

# R4 View experiment logs for specific protocol
ls logs/experiments_*/emotion_scenario1*mqtt*.log

################################################################################
# SEQUENTIAL RECOMMENDATION
################################################################################

# Step 1: Quick validation to verify setup works
echo "Step 1: Quick validation..."
./run_comprehensive_experiments.sh --use-case emotion --scenario 1 --protocol mqtt --rounds 5

# Step 2: Run Scenario 1 (without quantization) - takes ~15 hours
echo "Step 2: Running Scenario 1 (without quantization)..."
./run_comprehensive_experiments.sh --scenario 1 --rounds 100

# Step 3: Run Scenario 2 (with quantization) - takes ~15 hours  
echo "Step 3: Running Scenario 2 (with quantization)..."
./run_comprehensive_experiments.sh --scenario 2 --rounds 100

# Step 4: Analyze results
echo "Step 4: Analyzing results..."
ls -lh experiment_results/
