# Scripts Directory

This directory contains all utility scripts organized by purpose.

## 📁 Directory Structure

### `/experiments/`
Scripts for running experiments and evaluations.
- `run_experiments.sh` - Main experiment runner
- `run_comprehensive_experiments.sh` - Run comprehensive experiments across all protocols
- `run_emotion_recognition_gpu.sh` - Run emotion recognition experiments with GPU
- `run_temperature_rl_demo.sh` - Run temperature regulation RL demo
- `launch_experiment_gui.sh` - Launch experiment GUI
- `EXPERIMENT_QUICK_COMMANDS.sh` - Quick commands for common experiments

### `/fixes/`
Scripts for fixing issues and applying patches.
- `fix_compose_env.py` - Fix docker-compose environment variables
- `fix_generic_broadcast.py` - Fix generic broadcast issues
- `fix_late_joining.py` - Fix late-joining client issues
- `update_dynamic_client_support.py` - Update dynamic client support
- `update_servers_dynamic_clients.py` - Update servers for dynamic clients

### `/integration/`
Scripts for integrating features and components.
- `integrate_quantization.py` - Integrate quantization into the system
- `integrate_grpc_servers.py` - Integrate gRPC servers
- `integrate_all_protocols_quantization.py` - Integrate quantization across all protocols
- `complete_server_integration.py` - Complete server integration tasks
- `add_compression_logic.py` - Add compression logic to components
- `add_quantization_init.py` - Add quantization initialization

### `/setup/`
Installation and setup scripts.
- `install_cyclonedds.sh` - Install CycloneDDS dependencies
- `rebuild_docker_with_dds.sh` - Rebuild Docker images with DDS support
- `REBUILD_IMAGES.sh` - Rebuild all Docker images
- `build_and_test.sh` - Build and test the system

### `/testing/`
Test and verification scripts.

**Python Tests:**
- `test_amqp_direct.py` - Direct AMQP connection tests
- `test_packet_logger.py` - Packet logger tests
- `test_pruning.py` - Pruning implementation tests
- `test_quantization.py` - Quantization implementation tests

**Shell Tests:**
- `test_dds_configs.sh` - Test DDS configurations
- `test_dds_poor_network_fix.sh` - Test DDS poor network fix
- `test_gpu_oom_fix.sh` - Test GPU OOM fix
- `test_network_conditions.sh` - Test network conditions

**Verification Scripts:**
- `verify_cyclonedds_unicast.sh` - Verify CycloneDDS unicast configuration
- `verify_dds_poor_network_config.sh` - Verify DDS poor network config
- `verify_fair_config.sh` - Verify fair protocol configuration
- `verify_gpu_setup.sh` - Verify GPU setup

### `/utilities/`
General utility scripts for common tasks.
- `display_packet_logs.py` - Display packet logs from database
- `packet_logger.py` - Packet logging utility
- `network_condition_manager.py` - Network condition management
- `demo_gui.sh` - Launch demo GUI
- `quickstart.sh` - Quick start script
- `start_packet_logging.sh` - Start packet logging service

## 🚀 Quick Usage

### Running Experiments
```bash
# Run all experiments
./scripts/experiments/run_experiments.sh

# Run GPU experiments
./scripts/experiments/run_emotion_recognition_gpu.sh

# Launch experiment GUI
./scripts/experiments/launch_experiment_gui.sh
```

### Setup
```bash
# Install dependencies
./scripts/setup/install_cyclonedds.sh

# Rebuild Docker images
./scripts/setup/REBUILD_IMAGES.sh
```

### Testing
```bash
# Test network conditions
./scripts/testing/test_network_conditions.sh

# Verify GPU setup
./scripts/testing/verify_gpu_setup.sh

# Test quantization
python scripts/testing/test_quantization.py
```

### Utilities
```bash
# Start packet logging
./scripts/utilities/start_packet_logging.sh

# Display packet logs
python scripts/utilities/display_packet_logs.py

# Quick start
./scripts/utilities/quickstart.sh
```

## 📝 Script Categories

- **Setup**: Install dependencies, rebuild images
- **Experiments**: Run FL experiments and evaluations
- **Testing**: Test features and verify configurations
- **Fixes**: Apply patches and fixes
- **Integration**: Integrate new features
- **Utilities**: Common tools and helpers

## ⚠️ Important Notes

1. Most shell scripts should be run from the project root directory
2. Python scripts may require virtual environment activation
3. Docker scripts require Docker and Docker Compose to be installed
4. GPU scripts require NVIDIA Docker runtime
5. Check script permissions: `chmod +x script.sh` if needed
