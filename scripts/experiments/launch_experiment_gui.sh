#!/bin/bash

# ===============================================================================
# Federated Learning Experiment GUI Launcher
# ===============================================================================

echo "🚀 Launching FL Experiment GUI..."
echo ""

# Check if PyQt5 is installed
if ! python3 -c "import PyQt5" 2>/dev/null; then
    echo "📦 PyQt5 not found. Installing..."
    pip install -r ../../Network_Simulation/gui_requirements.txt
    echo ""
fi

# Launch the GUI
python3 ../../Network_Simulation/experiment_gui.py

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ GUI closed successfully"
else
    echo "❌ GUI encountered an error"
    exit 1
fi
