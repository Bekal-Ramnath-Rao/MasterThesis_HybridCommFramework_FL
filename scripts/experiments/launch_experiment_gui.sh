#!/usr/bin/env bash

# ===============================================================================
# Federated Learning Experiment GUI Launcher
# ===============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/lib/resolve_python.sh
source "${SCRIPT_DIR}/../../scripts/lib/resolve_python.sh" || exit 1

echo "🚀 Launching FL Experiment GUI..."
echo ""

# Check if PyQt5 is installed
if ! "$PYTHON" -c "import PyQt5" 2>/dev/null; then
    echo "📦 PyQt5 not found. Installing..."
    "$PYTHON" -m pip install -r "${SCRIPT_DIR}/../../Network_Simulation/gui_requirements.txt"
    echo ""
fi

# Launch the GUI
"$PYTHON" "${SCRIPT_DIR}/../../Network_Simulation/experiment_gui.py"

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ GUI closed successfully"
else
    echo "❌ GUI encountered an error"
    exit 1
fi
