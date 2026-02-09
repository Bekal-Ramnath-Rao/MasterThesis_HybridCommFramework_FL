# Documentation Index

This directory contains all project documentation organized by topic.

## 📁 Directory Structure

### `/architecture/`
System architecture, design decisions, and implementation status reports.
- `IMPLEMENTATION_STATUS_REPORT.md` - Overall project implementation status
- `UNIFIED_CLIENT_ARCHITECTURE.md` - Unified client architecture documentation
- `COMMUNICATION_FLOW.md` - Communication flow between components
- `PROTOCOL_IMPLEMENTATION_DETAILS.md` - Protocol-specific implementation details

### `/compression/`
Documentation related to model compression techniques (quantization and pruning).
- `COMPRESSION_EVALUATION_GUIDE.md` - Guide for evaluating compression techniques
- `QUANTIZATION_QUICK_REF.md` - Quick reference for quantization
- `PRUNING_QUICK_REF.md` - Quick reference for pruning
- `QUANTIZATION_COMPLETE.md` - Quantization implementation details

### `/distributed/`
Distributed Federated Learning setup and client management.
- `DISTRIBUTED_FL_README.md` - Overview of distributed FL setup
- `DISTRIBUTED_CLIENT_QUICK_START.md` - Quick start guide for distributed clients
- `DISTRIBUTED_CLIENT_SETUP.md` - Detailed setup instructions

### `/experiments/`
Guides for running experiments and evaluations.
- `EXPERIMENT_RUNNER_GUIDE.md` - Comprehensive guide for running experiments
- `COMPREHENSIVE_EXPERIMENT_SETUP.md` - Detailed experiment setup instructions

### `/fixes/`
Bug fixes, issue resolutions, and troubleshooting documentation.
- Protocol-specific fixes (AMQP, DDS, QUIC)
- GPU and dependency fixes
- Docker compose fixes
- Network and timeout fixes

### `/gpu/`
GPU setup, configuration, and troubleshooting.
- `GPU_README.md` - Main GPU documentation
- `GPU_QUICK_START.md` - Quick start guide for GPU setup
- `GPU_DOCKER_SETUP.md` - Docker setup for GPU support
- `GPU_EXPERIMENTS_QUICKSTART.md` - Running GPU experiments

### `/gui/`
GUI-related documentation and user guides.
- `GUI_README.md` - Main GUI documentation
- `GUI_USER_GUIDE.md` - User guide for the GUI
- `GUI_ARCHITECTURE.md` - GUI architecture details
- `GUI_ENHANCED_FEATURES.md` - Enhanced features documentation

### `/guides/`
User guides, quick starts, and how-to documentation.
- Quick start guides for various components
- Integration guides
- Docker guides
- RL protocol selection guides

### `/network/`
Network simulation, conditions, and dynamic network control.
- `README_DYNAMIC_NETWORK.md` - Dynamic network control documentation
- `FL_NETWORK_CONTROL_README.md` - FL network control overview
- `NETWORK_CONDITIONS_USAGE.md` - Network conditions usage guide

### `/protocols/`
Protocol-specific implementation and configuration.
- DDS chunking implementation
- CycloneDDS configuration
- AMQP implementation
- Protocol completion status

## 🔍 Finding Documentation

- **Getting Started**: See `/guides/` for quick start guides
- **Setup Issues**: Check `/fixes/` for troubleshooting
- **Architecture**: See `/architecture/` for system design
- **Experiments**: See `/experiments/` for running evaluations
- **GPU Setup**: See `/gpu/` for GPU configuration

## 📚 Main Documentation Entry Points

1. **Start Here**: `README.md` (project root)
2. **Implementation Status**: `architecture/IMPLEMENTATION_STATUS_REPORT.md`
3. **Quick Start**: `guides/UNIFIED_QUICK_START.md`
4. **Experiments**: `experiments/EXPERIMENT_RUNNER_GUIDE.md`
5. **GPU Setup**: `gpu/GPU_QUICK_START.md`
