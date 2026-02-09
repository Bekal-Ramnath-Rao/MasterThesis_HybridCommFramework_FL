#!/bin/bash

#######################################################################
# Temperature FL Demo with Dynamic Network Control & RL Monitoring
#######################################################################

echo "=========================================================================="
echo "Temperature Regulation FL with RL Protocol Selection Demo"
echo "=========================================================================="
echo ""

# Configuration
export CLIENT_ID=0
export NUM_CLIENTS=2
export NUM_ROUNDS=20
export USE_RL_SELECTION=true
export DEFAULT_PROTOCOL=mqtt

# MQTT Configuration
export MQTT_BROKER=localhost
export MQTT_PORT=1883

# Check if MQTT broker is running
if ! pgrep -x "mosquitto" > /dev/null; then
    echo "[Warning] MQTT broker (mosquitto) not running"
    echo "[Info] Starting mosquitto..."
    mosquitto -d -c mqtt-config/mosquitto.conf
    sleep 2
fi

echo "[✓] MQTT broker running"
echo ""

# Verify dataset exists
DATASET_PATH="Client/Temperature_Regulation/Dataset/base_data_baseline_unique.csv"
if [ -f "$DATASET_PATH" ]; then
    echo "[✓] Temperature dataset found: $DATASET_PATH"
else
    echo "[!] Dataset not found, will use synthetic fallback"
fi
echo ""

echo "=========================================================================="
echo "Configuration:"
echo "=========================================================================="
echo "  Clients: $NUM_CLIENTS"
echo "  Rounds: $NUM_ROUNDS"
echo "  RL Selection: $USE_RL_SELECTION"
echo "  Default Protocol: $DEFAULT_PROTOCOL"
echo "=========================================================================="
echo ""

# Instructions
echo "=========================================================================="
echo "DYNAMIC NETWORK CONTROL"
echo "=========================================================================="
echo ""
echo "Open a NEW TERMINAL and run these commands to change network conditions:"
echo ""
echo "  # Apply network scenarios (excellent → very_poor):"
echo "  python Client/dynamic_network_controller.py --scenario excellent"
echo "  python Client/dynamic_network_controller.py --scenario good"
echo "  python Client/dynamic_network_controller.py --scenario fair"
echo "  python Client/dynamic_network_controller.py --scenario poor"
echo "  python Client/dynamic_network_controller.py --scenario very_poor"
echo ""
echo "  # Simulate mobility patterns:"
echo "  python Client/dynamic_network_controller.py --mobility static"
echo "  python Client/dynamic_network_controller.py --mobility low"
echo "  python Client/dynamic_network_controller.py --mobility medium"
echo "  python Client/dynamic_network_controller.py --mobility high"
echo ""
echo "  # Custom conditions:"
echo "  python Client/dynamic_network_controller.py --custom 100,20,10,2"
echo "  #                                            latency,jitter,bandwidth,loss"
echo ""
echo "  # Clear all network limitations:"
echo "  python Client/dynamic_network_controller.py --clear"
echo ""
echo "=========================================================================="
echo ""

read -p "Press ENTER to start FL Client (or Ctrl+C to cancel)..."

# Activate conda environment (if needed for CycloneDDS)
echo ""
echo "[Info] Running in current environment"
echo "[Note] For DDS protocol, ensure you're in conda environment with cyclonedds"
echo ""

# Run the unified client
echo "=========================================================================="
echo "Starting Temperature FL Client with RL Protocol Selection"
echo "=========================================================================="
echo ""
echo "Watch for:"
echo "  [RL Selection] - Shows which protocol is selected each round"
echo "  [RL] Reward - Shows the reward received for the protocol choice"
echo "  Q-Learning Statistics - Summary at the end"
echo ""
echo "=========================================================================="
echo ""

python Client/Temperature_Regulation/FL_Client_Unified.py

echo ""
echo "=========================================================================="
echo "Demo Complete!"
echo "=========================================================================="
echo ""
echo "Check Q-table saved at: q_table_temperature_client_${CLIENT_ID}.pkl"
echo ""
