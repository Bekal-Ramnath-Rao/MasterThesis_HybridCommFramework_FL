# Project Structure

This document provides a comprehensive overview of the project's directory structure after reorganization.

## 📂 Root Level Structure

```
MasterThesis_HybridCommFramework_FL/
├── README.md                          # Main project README
├── QUANTIZATION_CONFIG.py             # Quantization configuration
├── requirements.txt                   # Python dependencies
├── PROJECT_STRUCTURE.md               # This file
│
├── archives/                          # Archived files and packages
├── certs/                            # SSL/TLS certificates
├── Client/                           # Client implementations
├── config/                           # Configuration files
├── data/                             # Runtime data and databases
├── Docker/                           # Docker configurations
├── docs/                             # 📚 All documentation (see below)
├── experiment_results/               # Experiment output data
├── experiment_results_baseline/      # Baseline experiment results
├── GUI/                              # GUI components
├── Images/                           # Project images/diagrams
├── logs/                             # Log files
├── Miscellaneous/                    # Miscellaneous utilities
├── mqtt-config/                      # MQTT broker configuration
├── Network_Simulation/               # Network simulation tools
├── Plant_UML/                        # PlantUML diagrams
├── Protocol_References/              # Protocol reference docs
├── Protocols/                        # Protocol definitions
├── scripts/                          # 🔧 All scripts (see below)
├── Server/                           # Server implementations
└── shared_data/                      # Shared runtime data
```

## 📚 Documentation (`docs/`)

Organized by topic for easy navigation:

```
docs/
├── README.md                         # Documentation index
│
├── architecture/                     # System design & architecture
│   ├── IMPLEMENTATION_STATUS_REPORT.md
│   ├── UNIFIED_CLIENT_ARCHITECTURE.md
│   ├── COMMUNICATION_FLOW.md
│   └── ... (13 files total)
│
├── compression/                      # Quantization & pruning
│   ├── COMPRESSION_EVALUATION_GUIDE.md
│   ├── QUANTIZATION_QUICK_REF.md
│   ├── PRUNING_QUICK_REF.md
│   └── ... (10 files total)
│
├── distributed/                      # Distributed FL setup
│   ├── DISTRIBUTED_FL_README.md
│   ├── DISTRIBUTED_CLIENT_QUICK_START.md
│   └── ... (4 files total)
│
├── experiments/                      # Experiment guides
│   ├── EXPERIMENT_RUNNER_GUIDE.md
│   └── COMPREHENSIVE_EXPERIMENT_SETUP.md
│
├── fixes/                           # Bug fixes & troubleshooting
│   ├── AMQP_FIX_QUICK_REF.md
│   ├── DDS_POOR_NETWORK_FIX.md
│   ├── GPU_OOM_FIX_COMPLETE.md
│   └── ... (24 files total)
│
├── gpu/                             # GPU setup & configuration
│   ├── GPU_README.md
│   ├── GPU_QUICK_START.md
│   ├── GPU_DOCKER_SETUP.md
│   └── ... (6 files total)
│
├── gui/                             # GUI documentation
│   ├── GUI_README.md
│   ├── GUI_USER_GUIDE.md
│   ├── GUI_ARCHITECTURE.md
│   └── ... (12 files total)
│
├── guides/                          # User guides & quick starts
│   ├── UNIFIED_QUICK_START.md
│   ├── QUICK_START_RL_SYSTEM.md
│   ├── FL_BASELINE_GUIDE.md
│   └── ... (14 files total)
│
├── network/                         # Network simulation & control
│   ├── README_DYNAMIC_NETWORK.md
│   ├── FL_NETWORK_CONTROL_README.md
│   └── ... (5 files total)
│
└── protocols/                       # Protocol implementations
    ├── DDS_CHUNKING_IMPLEMENTATION_COMPLETE.md
    ├── CYCLONEDDS_UNICAST_CONFIG.md
    └── ... (8 files total)
```

## 🔧 Scripts (`scripts/`)

Organized by purpose for efficient workflow:

```
scripts/
├── README.md                         # Scripts index & usage
│
├── experiments/                      # Run experiments
│   ├── run_experiments.sh
│   ├── run_comprehensive_experiments.sh
│   ├── run_emotion_recognition_gpu.sh
│   ├── run_temperature_rl_demo.sh
│   ├── launch_experiment_gui.sh
│   └── EXPERIMENT_QUICK_COMMANDS.sh
│
├── fixes/                           # Fix scripts
│   ├── fix_compose_env.py
│   ├── fix_generic_broadcast.py
│   ├── fix_late_joining.py
│   ├── update_dynamic_client_support.py
│   └── update_servers_dynamic_clients.py
│
├── integration/                     # Integration scripts
│   ├── integrate_quantization.py
│   ├── integrate_grpc_servers.py
│   ├── integrate_all_protocols_quantization.py
│   ├── complete_server_integration.py
│   ├── add_compression_logic.py
│   └── add_quantization_init.py
│
├── setup/                           # Setup & installation
│   ├── install_cyclonedds.sh
│   ├── rebuild_docker_with_dds.sh
│   ├── REBUILD_IMAGES.sh
│   └── build_and_test.sh
│
├── testing/                         # Tests & verification
│   ├── test_amqp_direct.py
│   ├── test_packet_logger.py
│   ├── test_pruning.py
│   ├── test_quantization.py
│   ├── test_dds_configs.sh
│   ├── test_gpu_oom_fix.sh
│   ├── verify_cyclonedds_unicast.sh
│   └── ... (12 files total)
│
└── utilities/                       # Utility scripts
    ├── display_packet_logs.py
    ├── packet_logger.py
    ├── network_condition_manager.py
    ├── demo_gui.sh
    ├── quickstart.sh
    └── start_packet_logging.sh
```

## 🗂️ Other Key Directories

### `config/`
Configuration files for system components:
- CycloneDDS XML configurations
- MQTT broker configuration
- Docker Compose files

### `data/`
Runtime data and databases:
- `packet_logs.db` - Packet transmission logs

### `logs/`
Log files from experiments and debugging:
- Comprehensive experiment logs
- NVIDIA GPU bug reports

### `archives/`
Archived packages and compressed files:
- Protocol buffer compiler
- Other archived tools

### `Client/`
Client implementations for different use cases:
- Emotion Recognition
- Mental State Recognition
- Temperature Regulation
- RL-based protocol selection

### `Server/`
Server implementations for different use cases:
- Emotion Recognition
- Mental State Recognition
- Temperature Regulation
- Compression techniques

### `Network_Simulation/`
Network simulation and evaluation tools:
- Network simulator
- Dynamic network controller
- Experiment GUI
- Evaluation scripts

### `Protocol_References/`
Protocol-specific reference documentation:
- gRPC, MQTT, AMQP, DDS, QUIC
- Troubleshooting guides

## 🎯 Quick Navigation

### Getting Started
1. **First Time**: `README.md` → `docs/guides/UNIFIED_QUICK_START.md`
2. **Setup**: `scripts/setup/` → `docs/guides/`
3. **Run Experiments**: `scripts/experiments/run_experiments.sh`

### Troubleshooting
1. **Find Issue**: `docs/fixes/`
2. **Apply Fix**: `scripts/fixes/`
3. **Verify**: `scripts/testing/`

### Development
1. **Architecture**: `docs/architecture/`
2. **Integration**: `scripts/integration/`
3. **Testing**: `scripts/testing/`

## 📊 File Count Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Documentation | ~90 | All .md files organized by topic |
| Scripts | ~35 | Utilities, tests, experiments |
| Config Files | ~8 | System configurations |
| Source Code | ~100+ | Client/Server implementations |

## 🔍 Finding Files

**By Purpose:**
- Setup/Installation → `scripts/setup/`, `docs/guides/`
- Running Experiments → `scripts/experiments/`, `docs/experiments/`
- Troubleshooting → `docs/fixes/`, `scripts/testing/`
- Configuration → `config/`, `docs/protocols/`

**By Topic:**
- GPU → `docs/gpu/`
- GUI → `docs/gui/`, `GUI/`
- Network → `docs/network/`, `Network_Simulation/`
- Compression → `docs/compression/`
- Protocols → `docs/protocols/`, `Protocol_References/`

## ✅ Benefits of This Structure

1. **Organized**: Files grouped by purpose and topic
2. **Discoverable**: Clear directory names and README files
3. **Maintainable**: Easy to add new files in the right place
4. **Navigable**: Quick access to docs and scripts
5. **Professional**: Clean root directory

## 📝 Maintenance Notes

- Keep root-level files minimal (README, requirements, config)
- Add new documentation to appropriate `docs/` subdirectory
- Add new scripts to appropriate `scripts/` subdirectory
- Update this document when adding new top-level directories
- Keep README files up-to-date in each directory
