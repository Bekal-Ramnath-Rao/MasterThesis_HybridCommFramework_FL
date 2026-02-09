#!/bin/bash

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
if ! python3 -c "import PyQt5" 2>/dev/null; then
    echo "📦 Installing PyQt5..."
    pip install PyQt5
fi

# Launch GUI
python3 Network_Simulation/experiment_gui.py

echo ""
echo "✅ GUI Demo Complete!"
