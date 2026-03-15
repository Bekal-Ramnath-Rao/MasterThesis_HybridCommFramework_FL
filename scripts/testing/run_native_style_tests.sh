#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/app}"
if [[ ! -d "$PROJECT_ROOT" ]]; then
  PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
fi

cd "$PROJECT_ROOT"

echo "==============================================="
echo "Running Native-Style Test Suite"
echo "Project root: $PROJECT_ROOT"
echo "==============================================="

echo "[1/4] Packet logger test"
python scripts/testing/test_packet_logger.py

echo "[2/4] Quantization test"
python scripts/testing/test_quantization.py

echo "[3/4] Pruning test"
python scripts/testing/test_pruning.py

echo "[4/4] AMQP direct test"
python scripts/testing/test_amqp_direct.py

echo "==============================================="
echo "All native-style tests completed"
echo "==============================================="
