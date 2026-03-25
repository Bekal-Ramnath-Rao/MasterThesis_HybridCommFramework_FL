#!/usr/bin/env bash
# Launch script for Distributed FL Client GUI
# Run this on a remote PC to connect a client to the central experiment server

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../scripts/lib/resolve_python.sh
source "${SCRIPT_DIR}/../scripts/lib/resolve_python.sh" || exit 1

echo "🌐 Starting Distributed FL Client GUI..."
echo ""

# Check if PyQt5 is installed
if ! "$PYTHON" -c "import PyQt5" 2>/dev/null; then
    echo "❌ PyQt5 not found. Installing..."
    "$PYTHON" -m pip install PyQt5
fi

# Launch the GUI
cd "$SCRIPT_DIR" || exit 1
"$PYTHON" distributed_client_gui.py


echo ""
echo "👋 Distributed Client GUI closed"
