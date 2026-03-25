#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=../../scripts/lib/resolve_python.sh
source "$REPO_ROOT/scripts/lib/resolve_python.sh" || exit 1

# ===============================================================================
# FL EXPERIMENT GUI - DEMO MODE
# Launches the GUI in demo/preview mode with pre-configured settings
# ===============================================================================

echo "🚀 Launching FL Experiment GUI in Demo Mode..."
echo ""
echo "Demo Configuration:"
echo "  ✓ Use Case: Emotion Recognition"
echo "  ✓ Protocol: MQTT (pre-selected)"
echo "  ✓ Scenario: Excellent (pre-selected)"
echo "  ✓ GPU: Enabled"
echo "  ✓ Rounds: 3 (quick test)"
echo ""
echo "This is perfect for testing the GUI without running a full experiment."
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Check PyQt5
if ! "$PYTHON" -c "import PyQt5" 2>/dev/null; then
    echo "📦 Installing PyQt5..."
    "$PYTHON" -m pip install PyQt5
fi

# Launch GUI
"$PYTHON" "$REPO_ROOT/Network_Simulation/experiment_gui.py"

echo ""
echo "✅ GUI Demo Complete!"
