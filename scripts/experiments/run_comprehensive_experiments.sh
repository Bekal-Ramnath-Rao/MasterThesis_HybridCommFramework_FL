#!/bin/bash

################################################################################
# Comprehensive Federated Learning Experiment Runner
# 
# Runs experiments for all use cases (emotion, mental state, temperature)
# across all protocols (MQTT, AMQP, gRPC, QUIC, DDS) and scenarios
# 
# Scenario 1: All 9 network scenarios WITHOUT quantization
# Scenario 2: All 9 network scenarios WITH quantization
#
# Usage:
#   ./run_comprehensive_experiments.sh [options]
#
# Options:
#   --use-case [emotion|mentalstate|temperature]  - Run specific use case
#   --scenario [1|2]                              - Run specific scenario
#   --protocol [mqtt|amqp|grpc|quic|dds]          - Run specific protocol
#   --rounds N                                     - Number of training rounds (default: 100)
#   --single                                       - Run single protocol per scenario
#   --skip-build                                   - Skip Docker image build
#   --skip-consolidate                             - Skip result consolidation
#   --help                                         - Show this help message
#
# Examples:
#   ./run_comprehensive_experiments.sh                              # Run all experiments
#   ./run_comprehensive_experiments.sh --use-case emotion           # Emotion only
#   ./run_comprehensive_experiments.sh --scenario 1                 # Scenario 1 only
#   ./run_comprehensive_experiments.sh --rounds 50                  # Custom rounds
#
################################################################################

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/experiments_${TIMESTAMP}"
RESULTS_DIR="experiment_results"
ROUNDS=${ROUNDS:-100}
ENABLE_GPU=${ENABLE_GPU:-true}
SKIP_BUILD=${SKIP_BUILD:-false}
SKIP_CONSOLIDATE=${SKIP_CONSOLIDATE:-false}

# Arrays
USE_CASES=(emotion mentalstate temperature)
PROTOCOLS=(mqtt amqp grpc quic dds)
NETWORK_SCENARIOS=(excellent good moderate poor very_poor satellite congested_light congested_moderate congested_heavy)
QUANTIZATION_MODES=(disabled enabled)

# Parsed options
SPECIFIC_USE_CASE=""
SPECIFIC_SCENARIO=""
SPECIFIC_PROTOCOL=""
SPECIFIC_NETWORK_SCENARIO=""
RUN_SINGLE=false

# Functions
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

show_help() {
    head -n 35 "$0" | tail -n 33
}

verify_environment() {
    print_header "Verifying Environment"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    print_success "Docker found: $(docker --version)"
    
    # Check Docker Compose V2
    if ! docker compose version &> /dev/null; then
        print_error "Docker Compose V2 not found. Please install Docker Compose V2."
        exit 1
    fi
    print_success "Docker Compose V2 found: $(docker compose version | head -1)"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found. Please install Python3."
        exit 1
    fi
    print_success "Python3 found: $(python3 --version)"
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        print_success "NVIDIA GPUs found: $GPU_COUNT"
        nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/  - /'
    else
        print_warning "NVIDIA GPU tools not found. Running on CPU mode."
        ENABLE_GPU=false
    fi
    
    # Check required files
    if [ ! -d "Docker" ]; then
        print_error "Docker directory not found."
        exit 1
    fi
    print_success "Project structure verified"
    
    echo ""
}

create_log_directory() {
    mkdir -p "$LOG_DIR"
    print_success "Created log directory: $LOG_DIR"
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --use-case)
                SPECIFIC_USE_CASE="$2"
                shift 2
                ;;
            --scenario)
                SPECIFIC_SCENARIO="$2"
                shift 2
                ;;
            --protocol)
                SPECIFIC_PROTOCOL="$2"
                shift 2
                ;;
            --network-scenario)
                SPECIFIC_NETWORK_SCENARIO="$2"
                shift 2
                ;;
            --rounds)
                ROUNDS="$2"
                shift 2
                ;;
            --single)
                RUN_SINGLE=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-consolidate)
                SKIP_CONSOLIDATE=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

run_experiment() {
    local use_case="$1"
    local scenario_num="$2"
    local network_scenario="$3"
    local quantization="$4"
    local protocol="$5"
    
    local log_file="$LOG_DIR/${use_case}_scenario${scenario_num}_${network_scenario}_${quantization}_${protocol}.log"
    
    # Build command
    local cmd="python3 Network_Simulation/run_network_experiments.py"
    cmd="$cmd --use-case $use_case"
    cmd="$cmd --single"
    cmd="$cmd --protocol $protocol"
    cmd="$cmd --scenario $network_scenario"
    cmd="$cmd --rounds $ROUNDS"
    
    if [ "$ENABLE_GPU" == "true" ]; then
        cmd="$cmd --enable-gpu"
    fi
    
    if [ "$quantization" == "enabled" ]; then
        cmd="$cmd --use-quantization"
    fi
    
    print_info "Running: $use_case | Scenario $scenario_num | $network_scenario | $quantization | $protocol"
    
    # Run command and log output
    if eval "$cmd" > "$log_file" 2>&1; then
        print_success "Completed: $use_case | Scenario $scenario_num | $network_scenario | $quantization | $protocol"
        return 0
    else
        print_error "Failed: $use_case | Scenario $scenario_num | $network_scenario | $quantization | $protocol"
        print_info "Log file: $log_file"
        return 1
    fi
}

run_scenario_1() {
    local use_case="$1"
    
    print_header "Running Scenario 1: All network scenarios WITHOUT quantization for $use_case"
    
    local total=0
    local passed=0
    local failed=0
    
    for network_scenario in "${NETWORK_SCENARIOS[@]}"; do
        # If a specific network scenario is requested, skip others
        if [ -n "$SPECIFIC_NETWORK_SCENARIO" ] && [ "$network_scenario" != "$SPECIFIC_NETWORK_SCENARIO" ]; then
            continue
        fi
        for protocol in "${PROTOCOLS[@]}"; do
            # If a specific protocol is requested, skip others
            if [ -n "$SPECIFIC_PROTOCOL" ] && [ "$protocol" != "$SPECIFIC_PROTOCOL" ]; then
                continue
            fi
            total=$((total + 1))
            
            if run_experiment "$use_case" "1" "$network_scenario" "disabled" "$protocol"; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
            
            # Sleep between experiments to allow GPU memory cleanup
            sleep 5
        done
    done
    
    print_info "Scenario 1 Summary for $use_case: $passed/$total passed, $failed failed"
    echo ""
    return $failed
}

run_scenario_2() {
    local use_case="$1"
    
    print_header "Running Scenario 2: All network scenarios WITH quantization for $use_case"
    
    local total=0
    local passed=0
    local failed=0
    
    for network_scenario in "${NETWORK_SCENARIOS[@]}"; do
        # If a specific network scenario is requested, skip others
        if [ -n "$SPECIFIC_NETWORK_SCENARIO" ] && [ "$network_scenario" != "$SPECIFIC_NETWORK_SCENARIO" ]; then
            continue
        fi
        for protocol in "${PROTOCOLS[@]}"; do
            # If a specific protocol is requested, skip others
            if [ -n "$SPECIFIC_PROTOCOL" ] && [ "$protocol" != "$SPECIFIC_PROTOCOL" ]; then
                continue
            fi
            total=$((total + 1))
            
            if run_experiment "$use_case" "2" "$network_scenario" "enabled" "$protocol"; then
                passed=$((passed + 1))
            else
                failed=$((failed + 1))
            fi
            
            # Sleep between experiments
            sleep 5
        done
    done
    
    print_info "Scenario 2 Summary for $use_case: $passed/$total passed, $failed failed"
    echo ""
    return $failed
}

consolidate_results() {
    local use_case="$1"
    
    print_info "Consolidating results for $use_case..."
    
    # Find the latest experiment folder for this use case
    local latest_folder=$(ls -td "$RESULTS_DIR"/${use_case}_*/ 2>/dev/null | head -1)
    
    if [ -z "$latest_folder" ]; then
        print_warning "No results found for $use_case"
        return
    fi
    
    # Extract folder name without path
    local folder_name=$(basename "$latest_folder")
    
    # Run consolidation
    if python3 Network_Simulation/consolidate_results.py \
        --use-case "$use_case" \
        --experiment-folder "$folder_name" \
        >> "$LOG_DIR/consolidate_${use_case}.log" 2>&1; then
        print_success "Results consolidated for $use_case"
    else
        print_warning "Failed to consolidate results for $use_case"
    fi
}

build_docker_images() {
    if [ "$SKIP_BUILD" == "true" ]; then
        print_warning "Skipping Docker image build (--skip-build flag set)"
        return
    fi
    
    print_header "Building Docker Images"
    
    for use_case in "${USE_CASES[@]}"; do
        print_info "Building for $use_case..."
        local compose_file="Docker/docker-compose-${use_case}.gpu-isolated.yml"
        
        if [ ! -f "$compose_file" ]; then
            print_warning "Docker compose file not found: $compose_file"
            continue
        fi
        
        if docker compose -f "$compose_file" build --no-cache >> "$LOG_DIR/build_${use_case}.log" 2>&1; then
            print_success "Built Docker images for $use_case"
        else
            print_error "Failed to build Docker images for $use_case"
            print_info "Log file: $LOG_DIR/build_${use_case}.log"
        fi
    done
    
    echo ""
}

print_experiment_summary() {
    print_header "Experiment Summary"
    
    echo "Timestamp: $TIMESTAMP"
    echo "Rounds per experiment: $ROUNDS"
    echo "GPU enabled: $ENABLE_GPU"
    echo ""
    
    echo "Configuration:"
    echo "  Use cases: ${USE_CASES[@]}"
    echo "  Protocols: ${PROTOCOLS[@]}"
    echo "  Network scenarios: ${NETWORK_SCENARIOS[@]}"
    echo "  Quantization modes: ${QUANTIZATION_MODES[@]}"
    echo ""
    
    echo "Total combinations:"
    local total_combos=$((${#USE_CASES[@]} * ${#NETWORK_SCENARIOS[@]} * ${#PROTOCOLS[@]} * ${#QUANTIZATION_MODES[@]}))
    echo "  $total_combos experiments (${#USE_CASES[@]} use cases × ${#NETWORK_SCENARIOS[@]} scenarios × ${#PROTOCOLS[@]} protocols × ${#QUANTIZATION_MODES[@]} quantization modes)"
    echo ""
    
    echo "Estimated time:"
    local est_time=$((total_combos * ROUNDS / 100))  # Rough estimate
    echo "  ~$((est_time / 60)) hours (based on $ROUNDS rounds per experiment)"
    echo ""
    
    echo "Log directory: $LOG_DIR"
    echo "Results directory: $RESULTS_DIR"
    echo ""
}

main() {
    parse_arguments "$@"
    
    print_header "Federated Learning Comprehensive Experiment Runner"
    
    verify_environment
    create_log_directory
    print_experiment_summary
    
    # Build Docker images
    build_docker_images
    
    # Determine which use cases to run
    local use_cases_to_run=("${USE_CASES[@]}")
    if [ -n "$SPECIFIC_USE_CASE" ]; then
        use_cases_to_run=("$SPECIFIC_USE_CASE")
    fi
    
    # Run experiments
    local total_failed=0
    
    for use_case in "${use_cases_to_run[@]}"; do
        print_header "Processing Use Case: $use_case"
        
        # Determine which scenarios to run
        local scenarios_to_run=(1 2)
        if [ -n "$SPECIFIC_SCENARIO" ]; then
            scenarios_to_run=("$SPECIFIC_SCENARIO")
        fi
        
        for scenario in "${scenarios_to_run[@]}"; do
            if [ "$scenario" == "1" ]; then
                run_scenario_1 "$use_case" || total_failed=$((total_failed + $?))
            elif [ "$scenario" == "2" ]; then
                run_scenario_2 "$use_case" || total_failed=$((total_failed + $?))
            fi
        done
        
        # Consolidate results for this use case
        if [ "$SKIP_CONSOLIDATE" != "true" ]; then
            consolidate_results "$use_case"
        fi
    done
    
    # Final summary
    print_header "Experiment Run Complete"
    
    if [ $total_failed -eq 0 ]; then
        print_success "All experiments completed successfully!"
    else
        print_warning "$total_failed experiments failed. Check logs for details."
    fi
    
    echo ""
    print_info "Full logs available in: $LOG_DIR"
    echo ""
}

# Run main function with all arguments
main "$@"
