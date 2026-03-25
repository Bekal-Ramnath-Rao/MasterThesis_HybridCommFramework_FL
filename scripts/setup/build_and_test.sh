#!/usr/bin/env bash

# ============================================================================
# Comprehensive Dependency Fix - Build and Test Script
# ============================================================================
# This script rebuilds all Docker images with proper dependency resolution
# and tests the GPU-accelerated federated learning experiments.
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../../scripts/lib/resolve_python.sh
source "$PROJECT_DIR/scripts/lib/resolve_python.sh" || exit 1
DOCKER_DIR="$PROJECT_DIR/Docker"

echo "==============================================="
echo "🔨 Building All Docker Images with Dependencies"
echo "==============================================="

cd "$DOCKER_DIR"

# Build emotion images
echo ""
echo "Building Emotion Recognition Images..."
docker compose -f docker-compose-emotion.yml build

echo ""
echo "Building Mental State Recognition Images..."
docker compose -f docker-compose-mentalstate.yml build

echo ""
echo "Building Temperature Regulation Images..."
docker compose -f docker-compose-temperature.yml build

echo ""
echo "==============================================="
echo "✅ All images built successfully!"
echo "==============================================="

echo ""
echo "==============================================="
echo "🧪 Running GPU Test Experiment"
echo "==============================================="

cd "$PROJECT_DIR"

"$PYTHON" Network_Simulation/run_network_experiments.py \
  --use-case emotion \
  --enable-gpu \
  --single \
  --protocol mqtt \
  --scenario excellent \
  --rounds 2

echo ""
echo "==============================================="
echo "✅ Test Experiment Completed!"
echo "==============================================="
echo ""
echo "To run comprehensive experiments:"
echo ""
echo "# All 3 use cases with GPU (3 x 5 protocols x 9 scenarios = 135 experiments)"
echo "for use_case in emotion mentalstate temperature; do"
echo "  $PYTHON Network_Simulation/run_network_experiments.py --use-case \$use_case --enable-gpu --rounds 10"
echo "done"
echo ""
