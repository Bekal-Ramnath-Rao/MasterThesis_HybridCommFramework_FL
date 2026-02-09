#!/bin/bash
# Launch script for Distributed FL Client GUI
# Run this on a remote PC to connect a client to the central experiment server

echo "🌐 Starting Distributed FL Client GUI..."
echo ""

# Check if PyQt5 is installed
if ! python3 -c "import PyQt5" 2>/dev/null; then
    echo "❌ PyQt5 not found. Installing..."
    pip3 install PyQt5
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Launch the GUI
cd "$SCRIPT_DIR"
python3 distributed_client_gui.py


echo ""
echo "👋 Distributed Client GUI closed"
