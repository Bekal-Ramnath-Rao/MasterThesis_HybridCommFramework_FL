#!/bin/bash
# Quick start script to create packet logger databases and test the setup

echo "=================================================="
echo "Packet Logger Database Quick Start"
echo "=================================================="
echo ""

# Step 1: Ensure shared_data directory exists
echo "Step 1: Creating shared_data directory..."
mkdir -p shared_data
chmod 777 shared_data
echo "✓ Directory created: $(pwd)/shared_data"
echo ""

# Step 2: Show current state
echo "Step 2: Current state of shared_data..."
if [ -z "$(ls -A shared_data)" ]; then
    echo "⚠️  shared_data is empty (as expected before running containers)"
else
    echo "Files in shared_data:"
    ls -lh shared_data/
fi
echo ""

# Step 3: Explain how to create databases
echo "Step 3: How to create the .db files"
echo "=================================================="
echo ""
echo "The database files are created automatically when"
echo "Docker containers run. Choose one option:"
echo ""
echo "OPTION 1: Start Unified Containers (Recommended)"
echo "  cd Docker"
echo "  docker-compose -f docker-compose-unified-emotion.yml up"
echo ""
echo "OPTION 2: Use the GUI"
echo "  cd Network_Simulation"
echo "  python3 experiment_gui.py"
echo "  → Go to 'Basic Config' tab"
echo "  → Check '🤖 RL-Unified (Dynamic Selection)'"
echo "  → Click '▶️ Start Experiment'"
echo ""
echo "OPTION 3: Build and run manually"
echo "  cd Docker"
echo "  docker-compose -f docker-compose-unified-emotion.yml build"
echo "  docker-compose -f docker-compose-unified-emotion.yml up -d"
echo ""
echo "=================================================="
echo ""

# Step 4: Offer to start containers now
read -p "Would you like to start unified containers now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting unified emotion containers..."
    echo "This will:"
    echo "  1. Build Docker images (first time only)"
    echo "  2. Start FL server and 2 clients"
    echo "  3. Create packet_logs_*.db files in shared_data/"
    echo "  4. Begin federated learning with RL-based protocol selection"
    echo ""
    echo "Press Ctrl+C to stop when you're done testing."
    echo ""
    sleep 2
    
    cd Docker
    docker-compose -f docker-compose-unified-emotion.yml up
else
    echo ""
    echo "No problem! Run one of the commands above when ready."
    echo ""
    echo "After starting containers, check databases with:"
    echo "  ls -lh shared_data/"
    echo ""
    echo "View packets in GUI:"
    echo "  python3 Network_Simulation/experiment_gui.py"
    echo "  → Go to 'Packet Logs' tab"
    echo ""
fi

echo ""
echo "=================================================="
echo "For more information, see:"
echo "  - DATABASE_AND_GUI_UPDATES.md"
echo "  - PACKET_LOGS_QUICK_START.md"
echo "=================================================="
