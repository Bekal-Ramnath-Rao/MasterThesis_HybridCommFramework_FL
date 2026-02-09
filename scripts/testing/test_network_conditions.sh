#!/bin/bash

echo "========================================="
echo "Network Condition Manager - Demo & Test"
echo "========================================="
echo ""

# Check if tc is available
echo "1. Checking prerequisites..."
if ! command -v tc &> /dev/null; then
    echo "❌ tc command not found. Installing iproute2..."
    sudo apt-get update && sudo apt-get install -y iproute2
else
    echo "✅ tc command is available"
fi
echo ""

# List available conditions
echo "2. Available network conditions:"
echo "-----------------------------------"
python network_condition_manager.py list | grep "Condition:" | head -7
echo ""

# Test applying a condition
echo "3. Testing network condition application..."
echo "-----------------------------------"
echo ""
echo "Applying 'poor' network condition..."
python network_condition_manager.py apply --condition poor --verbose
echo ""

# Check if applied
echo "4. Verifying applied conditions..."
echo "-----------------------------------"
echo "Current tc rules on loopback interface:"
sudo tc qdisc show dev lo
echo ""

# Reset
echo "5. Resetting network conditions..."
echo "-----------------------------------"
python network_condition_manager.py reset --verbose
echo ""

echo "6. Verifying reset..."
echo "-----------------------------------"
echo "Current tc rules on loopback interface (should be default):"
sudo tc qdisc show dev lo
echo ""

echo "========================================="
echo "Demo Complete!"
echo "========================================="
echo ""
echo "Usage Examples:"
echo ""
echo "1. Apply condition and run FL experiment:"
echo "   export APPLY_NETWORK_CONDITION=true"
echo "   export NETWORK_CONDITION=poor"
echo "   python Server/Emotion_Recognition/FL_Server_MQTT.py"
echo ""
echo "2. Apply custom condition:"
echo "   python network_condition_manager.py apply --latency 100 --bandwidth 2 --loss 3"
echo ""
echo "3. Reset conditions:"
echo "   python network_condition_manager.py reset"
echo ""
