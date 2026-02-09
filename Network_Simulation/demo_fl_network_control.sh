#!/bin/bash
# FL Network Control Demo
# Demonstrates the network control system in action

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "========================================================================"
echo "  FL Network Control & Monitoring System - Demo"
echo "========================================================================"
echo ""

# Step 1: Verify system
echo -e "${BLUE}Step 1: Verifying system setup...${NC}"
echo ""
python3 "$SCRIPT_DIR/test_fl_network_system.py"
echo ""
read -p "Press Enter to continue..."

# Step 2: List clients
echo ""
echo -e "${BLUE}Step 2: Listing available FL clients...${NC}"
echo ""
docker ps --filter "name=client" --format "table {{.Names}}\t{{.Status}}"
echo ""
read -p "Press Enter to continue..."

# Step 3: Show current status
echo ""
echo -e "${BLUE}Step 3: Checking current network conditions...${NC}"
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" --show-status
echo ""
read -p "Press Enter to continue..."

# Step 4: Apply conditions to client 1
echo ""
echo -e "${BLUE}Step 4: Applying network conditions to client 1...${NC}"
echo -e "${YELLOW}  Latency: 200ms, Bandwidth: 1mbit, Loss: 2%, Jitter: 30ms${NC}"
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" \
    --client-id 1 \
    --latency 200ms \
    --bandwidth 1mbit \
    --loss 2 \
    --jitter 30ms
echo ""
read -p "Press Enter to continue..."

# Step 5: Show updated status
echo ""
echo -e "${BLUE}Step 5: Verifying conditions were applied...${NC}"
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" --show-status
echo ""
read -p "Press Enter to continue..."

# Step 6: Change single parameter
echo ""
echo -e "${BLUE}Step 6: Changing only latency to 300ms...${NC}"
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" --client-id 1 --latency 300ms
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" --show-status
echo ""
read -p "Press Enter to continue..."

# Step 7: Apply to all clients
echo ""
echo -e "${BLUE}Step 7: Applying moderate conditions to ALL clients...${NC}"
echo -e "${YELLOW}  Latency: 100ms, Bandwidth: 5mbit${NC}"
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" \
    --all \
    --latency 100ms \
    --bandwidth 5mbit
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" --show-status
echo ""
read -p "Press Enter to continue..."

# Step 8: Clear conditions
echo ""
echo -e "${BLUE}Step 8: Clearing all network conditions...${NC}"
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" --all --clear
echo ""
python3 "$SCRIPT_DIR/fl_network_monitor.py" --show-status
echo ""

echo ""
echo "========================================================================"
echo -e "${GREEN}  Demo Complete!${NC}"
echo "========================================================================"
echo ""
echo "What you learned:"
echo "  ✓ System verification"
echo "  ✓ Client listing"
echo "  ✓ Status checking"
echo "  ✓ Applying conditions to specific client"
echo "  ✓ Changing single parameter"
echo "  ✓ Applying to all clients"
echo "  ✓ Clearing conditions"
echo ""
echo "Next steps:"
echo "  • Start monitoring: python3 fl_training_dashboard.py"
echo "  • Interactive mode: python3 fl_network_monitor.py --monitor"
echo "  • Read guide: cat FL_NETWORK_CONTROL_GUIDE.md"
echo ""
