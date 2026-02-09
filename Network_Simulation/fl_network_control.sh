#!/bin/bash
# FL Network Control - Quick Commands
# Helper script for common network control operations during FL training

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Function to show usage
show_usage() {
    echo ""
    echo "FL Network Control - Quick Commands"
    echo "===================================="
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  monitor              Start FL training dashboard"
    echo "  control              Open interactive network controller"
    echo "  status               Show current network conditions"
    echo "  apply-client <id>    Apply conditions to specific client"
    echo "  apply-all            Apply conditions to all clients"
    echo "  clear-client <id>    Clear conditions from specific client"
    echo "  clear-all            Clear all network conditions"
    echo "  list-clients         List all running clients"
    echo "  quick-change <id>    Quick parameter change for client"
    echo ""
    echo "Examples:"
    echo "  $0 monitor                          # Start monitoring dashboard"
    echo "  $0 control                          # Interactive mode"
    echo "  $0 status                           # Show current conditions"
    echo "  $0 apply-client 1                   # Apply to client 1"
    echo "  $0 quick-change 2                   # Quick change for client 2"
    echo ""
}

# Function to list clients
list_clients() {
    print_info "Listing FL client containers..."
    docker ps --filter "name=client" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Function to show status
show_status() {
    print_info "Current network conditions:"
    python3 "$SCRIPT_DIR/fl_network_monitor.py" --show-status
}

# Function to start monitor
start_monitor() {
    print_info "Starting FL Training Dashboard..."
    echo ""
    python3 "$SCRIPT_DIR/fl_training_dashboard.py"
}

# Function to start interactive control
start_control() {
    print_info "Starting Interactive Network Controller..."
    echo ""
    python3 "$SCRIPT_DIR/fl_network_monitor.py" --monitor
}

# Function to apply conditions to specific client
apply_to_client() {
    local client_id=$1
    
    if [ -z "$client_id" ]; then
        print_error "Client ID required"
        echo "Usage: $0 apply-client <client_id>"
        return 1
    fi
    
    print_info "Applying conditions to client $client_id"
    echo ""
    
    read -p "Latency (e.g., 200ms) [Enter to skip]: " latency
    read -p "Bandwidth (e.g., 1mbit) [Enter to skip]: " bandwidth
    read -p "Packet loss % (e.g., 2) [Enter to skip]: " loss
    read -p "Jitter (e.g., 10ms) [Enter to skip]: " jitter
    
    cmd="python3 $SCRIPT_DIR/fl_network_monitor.py --client-id $client_id"
    
    [ -n "$latency" ] && cmd="$cmd --latency $latency"
    [ -n "$bandwidth" ] && cmd="$cmd --bandwidth $bandwidth"
    [ -n "$loss" ] && cmd="$cmd --loss $loss"
    [ -n "$jitter" ] && cmd="$cmd --jitter $jitter"
    
    echo ""
    print_info "Executing: $cmd"
    eval $cmd
}

# Function to apply to all clients
apply_to_all() {
    print_info "Applying conditions to ALL clients"
    echo ""
    
    read -p "Latency (e.g., 200ms) [Enter to skip]: " latency
    read -p "Bandwidth (e.g., 1mbit) [Enter to skip]: " bandwidth
    read -p "Packet loss % (e.g., 2) [Enter to skip]: " loss
    read -p "Jitter (e.g., 10ms) [Enter to skip]: " jitter
    
    cmd="python3 $SCRIPT_DIR/fl_network_monitor.py --all"
    
    [ -n "$latency" ] && cmd="$cmd --latency $latency"
    [ -n "$bandwidth" ] && cmd="$cmd --bandwidth $bandwidth"
    [ -n "$loss" ] && cmd="$cmd --loss $loss"
    [ -n "$jitter" ] && cmd="$cmd --jitter $jitter"
    
    echo ""
    print_info "Executing: $cmd"
    eval $cmd
}

# Function to clear conditions from client
clear_client() {
    local client_id=$1
    
    if [ -z "$client_id" ]; then
        print_error "Client ID required"
        echo "Usage: $0 clear-client <client_id>"
        return 1
    fi
    
    print_warning "Clearing network conditions from client $client_id"
    python3 "$SCRIPT_DIR/fl_network_monitor.py" --client-id "$client_id" --clear
}

# Function to clear all conditions
clear_all() {
    print_warning "Clearing network conditions from ALL clients"
    read -p "Are you sure? (y/n): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        python3 "$SCRIPT_DIR/fl_network_monitor.py" --all --clear
    else
        print_info "Operation cancelled"
    fi
}

# Function for quick parameter change
quick_change() {
    local client_id=$1
    
    if [ -z "$client_id" ]; then
        print_error "Client ID required"
        echo "Usage: $0 quick-change <client_id>"
        return 1
    fi
    
    print_info "Quick parameter change for client $client_id"
    echo ""
    echo "Select parameter to change:"
    echo "  1. Latency"
    echo "  2. Bandwidth"
    echo "  3. Packet Loss"
    echo "  4. Jitter"
    echo ""
    
    read -p "Choice (1-4): " choice
    
    case $choice in
        1)
            read -p "New latency (e.g., 200ms, 300ms): " value
            [ -n "$value" ] && python3 "$SCRIPT_DIR/fl_network_monitor.py" --client-id "$client_id" --latency "$value"
            ;;
        2)
            read -p "New bandwidth (e.g., 1mbit, 500kbit): " value
            [ -n "$value" ] && python3 "$SCRIPT_DIR/fl_network_monitor.py" --client-id "$client_id" --bandwidth "$value"
            ;;
        3)
            read -p "New packet loss % (e.g., 2, 5): " value
            [ -n "$value" ] && python3 "$SCRIPT_DIR/fl_network_monitor.py" --client-id "$client_id" --loss "$value"
            ;;
        4)
            read -p "New jitter (e.g., 10ms, 30ms): " value
            [ -n "$value" ] && python3 "$SCRIPT_DIR/fl_network_monitor.py" --client-id "$client_id" --jitter "$value"
            ;;
        *)
            print_error "Invalid choice"
            ;;
    esac
}

# Main script logic
case "${1:-}" in
    monitor)
        start_monitor
        ;;
    control)
        start_control
        ;;
    status)
        show_status
        ;;
    list-clients)
        list_clients
        ;;
    apply-client)
        apply_to_client "$2"
        ;;
    apply-all)
        apply_to_all
        ;;
    clear-client)
        clear_client "$2"
        ;;
    clear-all)
        clear_all
        ;;
    quick-change)
        quick_change "$2"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
