#!/bin/bash
# Quick Test Script for GPU OOM Fix
# Tests the moderate scenario with reduced batch size and GPU isolation

set -e  # Exit on error

PROJECT_ROOT="/home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL"
cd "$PROJECT_ROOT"

echo "=============================================================================="
echo "GPU OOM Fix - Testing Moderate Scenario"
echo "=============================================================================="
echo ""
echo "Step 1: Verify GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

echo "Step 2: Rebuild Docker images with batch size fixes..."
echo "Running: docker-compose build"
docker-compose -f Docker/docker-compose-emotion.gpu-isolated.yml build --no-cache
echo ""

echo "Step 3: Run quick test (2 rounds, moderate scenario, MQTT protocol)..."
echo "Running: python3 Network_Simulation/run_network_experiments.py"
echo "         --use-case emotion"
echo "         --enable-gpu"
echo "         --single"
echo "         --protocol mqtt"
echo "         --scenario moderate"
echo "         --rounds 2"
echo ""

python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --single \
    --protocol mqtt \
    --scenario moderate \
    --rounds 2

echo ""
echo "=============================================================================="
echo "Test completed successfully!"
echo "=============================================================================="
echo ""
echo "Improvements Applied:"
echo "  ✓ Batch size reduced from 32 → 16 (50% memory reduction)"
echo "  ✓ Data augmentation batch size reduced from 64 → 16"
echo "  ✓ GPU device isolation (Client 1 → GPU 0, Client 2 → GPU 1)"
echo "  ✓ TensorFlow memory limit set to 8GB per GPU"
echo ""
echo "If no OOM errors occurred, the fix was successful!"
echo ""
