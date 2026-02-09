#!/bin/bash

# QUICK START - DEPENDENCY FIXED VERSION
# ============================================================================
# Run this for a complete build and test of the dependency-fixed system
# ============================================================================

cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL

echo "=========================================="
echo "🔨 REBUILDING ALL DOCKER IMAGES"
echo "=========================================="
echo ""
echo "This will rebuild all 45 images with proper dependency resolution."
echo "Estimated time: 10-15 minutes"
echo ""

cd Docker

# Build all use cases
for compose_file in docker-compose-emotion.yml docker-compose-mentalstate.yml docker-compose-temperature.yml; do
    echo "Building images from $compose_file..."
    docker compose -f $compose_file build
    echo "✅ Done"
    echo ""
done

cd ..

echo "=========================================="
echo "✅ ALL IMAGES BUILT"
echo "=========================================="
echo ""
echo "Now running quick GPU test..."
echo ""

python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --single \
    --protocol mqtt \
    --scenario excellent \
    --rounds 2

echo ""
echo "=========================================="
echo "✅ TEST COMPLETE"
echo "=========================================="
echo ""
echo "All dependencies fixed! Ready to run experiments."
echo ""
echo "Quick commands:"
echo "  - Full emotion test:  python3 Network_Simulation/run_network_experiments.py --use-case emotion --enable-gpu --rounds 10"
echo "  - All 3 use cases:    bash build_and_test.sh (or manually run for each use case)"
echo ""
echo "See commands.txt for all available options."
echo ""
