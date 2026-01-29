# 🖥️ FL Experiment GUI - User Guide

## Overview

The **Federated Learning Experiment GUI** is a beautiful, comprehensive interface for configuring and running distributed FL experiments with network simulation capabilities. It provides an intuitive way to manage complex experiment configurations without dealing with command-line parameters.

## Features

### ✨ Main Capabilities

- **🎯 Multiple Use Cases**: Mental State Recognition, Emotion Recognition, Temperature Regulation
- **📡 5 Communication Protocols**: MQTT, AMQP, gRPC, QUIC, DDS
- **🌐 9 Network Scenarios**: From excellent to satellite and congestion scenarios
- **🖥️ GPU Acceleration**: Enable/disable GPU with configurable GPU count
- **🔢 Model Quantization**: 8/16/32-bit quantization with multiple strategies
- **📦 Model Compression**: gzip, lz4, zstd, snappy algorithms
- **✂️ Model Pruning**: Configurable pruning ratios
- **🎛️ Dynamic Network Control**: Real-time sliders for latency, bandwidth, jitter, packet loss
- **📊 Live Output Monitoring**: Real-time experiment output in console
- **⚙️ Advanced Options**: TensorBoard, profiling, verbose logging, checkpoints

---

## Installation

### Quick Install

```bash
# Install PyQt5 dependency
pip install -r Network_Simulation/gui_requirements.txt
```

### Launch the GUI

```bash
# Using the launcher script (recommended)
./launch_experiment_gui.sh

# Or directly
python3 Network_Simulation/experiment_gui.py
```

---

## User Interface Guide

### 📑 Tab 1: Basic Configuration

#### 🎯 Use Case
Select the federated learning use case:
- **Mental State Recognition**: Multi-class mental state classification
- **Emotion Recognition**: Emotion detection from sensor data
- **Temperature Regulation**: IoT temperature control system

#### 📡 Communication Protocols
Select one or more protocols to test:
- **MQTT**: Lightweight pub/sub messaging
- **AMQP**: Advanced message queuing
- **gRPC**: High-performance RPC framework
- **QUIC**: Modern transport protocol
- **DDS**: Data Distribution Service

💡 *Tip: Select multiple protocols to compare their performance*

#### 🌐 Network Scenarios
Choose network conditions to simulate:
- **Excellent**: No latency, no loss (ideal conditions)
- **Good**: 20ms latency, 0.1% loss
- **Moderate**: 50ms latency, 0.5% loss
- **Poor**: 100ms latency, 1% loss
- **Very Poor**: 200ms latency, 2% loss
- **Satellite**: 600ms latency, 0.5% loss
- **Light/Moderate/Heavy Congestion**: Traffic-based scenarios

💡 *Tip: Select multiple scenarios to test protocol robustness*

#### 🖥️ GPU Configuration
- **Enable GPU Acceleration**: Use GPU for faster training
- **GPU Count**: Number of GPUs to use (0-8)

#### 🎓 Training Configuration
- **Number of Rounds**: FL training rounds (1-1000)
- **Batch Size**: Training batch size (1-512)
- **Learning Rate**: Model learning rate (e.g., 0.001)
- **Min Clients**: Minimum clients required per round

---

### 📑 Tab 2: Network Control

#### 🎛️ Dynamic Network Control

Real-time network parameter control with sliders:

**Latency (0-1000 ms)**
- Controls network delay
- Simulates WAN/satellite conditions
- Real-time adjustment during experiments

**Bandwidth (1-1000 Mbps)**
- Limits available bandwidth
- Tests protocol efficiency
- Simulates constrained networks

**Jitter (0-100 ms)**
- Adds variability to latency
- Simulates unstable networks
- Tests protocol resilience

**Packet Loss (0-10%)**
- Simulates lossy networks
- Tests retransmission mechanisms
- Validates error handling

#### 🚦 Traffic Congestion

- **Enable Congestion**: Use traffic generator for realistic congestion
- **Congestion Level**: Light/Moderate/Heavy

---

### 📑 Tab 3: Advanced Options

#### 🔢 Model Quantization

Reduce model size and communication overhead:

- **Enable Quantization**: Toggle quantization on/off
- **Quantization Bits**: 8, 16, or 32 bits
  - 8-bit: Maximum compression, slight accuracy loss
  - 16-bit: Balanced compression and accuracy
  - 32-bit: Minimal compression, full precision
- **Strategy**: 
  - `full_quantization`: Quantize all parameters
  - `parameter_quantization`: Only weights
  - `activation_quantization`: Only activations
- **Symmetric**: Use symmetric quantization
- **Per-Channel**: Apply per-channel quantization

#### 📦 Model Compression

Additional compression for parameter transmission:

- **Algorithm**: gzip, lz4, zstd, snappy
  - `gzip`: Best compression ratio
  - `lz4`: Fast compression
  - `zstd`: Balanced
  - `snappy`: Fastest
- **Level**: Compression level (1-9)

#### ✂️ Model Pruning

Remove less important model parameters:

- **Pruning Ratio**: 0-90% of parameters to prune
- Higher ratios = smaller models but potential accuracy loss

#### ⚙️ Other Options

- **Save Model Checkpoints**: Save intermediate models
- **Verbose Logging**: Detailed debug output
- **Enable TensorBoard**: Real-time training visualization
- **Profile Performance**: Collect performance metrics

---

## Running Experiments

### Step-by-Step Guide

1. **Select Use Case**
   - Choose Mental State, Emotion, or Temperature

2. **Choose Protocols**
   - Select one or more protocols to test

3. **Select Network Scenarios**
   - Pick scenarios that match your testing goals

4. **Configure GPU**
   - Enable GPU if available
   - Set GPU count

5. **Set Training Parameters**
   - Rounds: Typically 10-100
   - Batch size: 16-64 for most cases
   - Learning rate: 0.001 is a good default

6. **Adjust Network Controls** (Optional)
   - Use sliders for custom network conditions
   - Override scenario defaults if needed
   - Click "🌐 Apply Network Changes" to apply dynamically during experiments

7. **Enable Advanced Features** (Optional)
   - Quantization for reduced communication
   - Compression for bandwidth savings
   - Pruning for smaller models

8. **Click "▶️ Start Experiment"**
   - Review confirmation dialog
   - Monitor progress in output consoles

9. **Monitor Execution**
   - **Experiment Output Tab**: Main experiment logs
   - **FL Training Monitor Tab**: Real-time training metrics with baseline comparison
   - **Server Logs Tab**: FL server container logs
   - **Client Logs Tab**: FL client container logs
   - Check GPU usage: `watch -n 1 nvidia-smi`
   - Use "⏹️ Stop Experiment" if needed

10. **Review Results**
    - Results saved in `experiment_results/`
    - JSON files with detailed metrics
    - Use consolidation scripts for analysis

---

## Monitoring Features

### 📈 FL Training Monitor (vs Baseline)

The GUI now integrates **fl_training_dashboard.py** to provide real-time training monitoring with baseline comparison:

- **Real-time Metrics**: See accuracy, loss, and RTT per round
- **Baseline Comparison**: Compare current experiment with baseline results
- **Performance Tracking**: Monitor if experiment is performing better/worse than baseline
- **Auto-refresh**: Updates every 5 seconds

**What You'll See:**
```
Round 5/10 - Current: Acc=0.82, Loss=0.35, RTT=12.3s
Baseline:            Acc=0.78, Loss=0.42, RTT=15.1s
Status: ✅ Outperforming baseline!
```

### 🖥️ Server & Client Logs

Live container log streaming from Docker:

- **Server Logs**: FL server activities, model aggregation, round progress
- **Client Logs**: Client training, parameter updates, communication status
- **Auto-scroll**: Follows latest log entries
- **Color-coded**: Different colors for server (green) and client (yellow) logs

### 🌐 Dynamic Network Control

Apply network conditions **during** running experiments:

1. Adjust sliders in Network Control tab
2. Click "🌐 Apply Network Changes"
3. Conditions applied via **fl_network_monitor.py** to all clients
4. See immediate impact on training performance

**Use Cases:**
- Test protocol resilience to sudden network changes
- Simulate mobile handoffs (good → poor network)
- Study adaptive behavior under varying conditions

---

## Example Configurations

### 🎯 Quick Test (Single Protocol, Single Scenario)

```
Use Case: Emotion Recognition
Protocols: [✓] MQTT
Scenarios: [✓] Excellent
GPU: Enabled
Rounds: 3
```

**Purpose**: Quick validation that everything works

---

### 🔬 Protocol Comparison (Multiple Protocols, Good Network)

```
Use Case: Mental State Recognition
Protocols: [✓] MQTT [✓] AMQP [✓] gRPC [✓] QUIC [✓] DDS
Scenarios: [✓] Good
GPU: Enabled
Rounds: 10
Quantization: Disabled
```

**Purpose**: Compare all protocols under favorable conditions

---

### 🌐 Resilience Test (Single Protocol, All Scenarios)

```
Use Case: Emotion Recognition
Protocols: [✓] MQTT
Scenarios: [✓] Excellent [✓] Good [✓] Moderate [✓] Poor [✓] Very Poor [✓] Satellite
GPU: Enabled
Rounds: 10
```

**Purpose**: Test how one protocol handles varying network conditions

---

### 🔢 Quantization Evaluation

```
Use Case: Temperature Regulation
Protocols: [✓] MQTT [✓] gRPC
Scenarios: [✓] Moderate [✓] Poor
GPU: Enabled
Rounds: 20
Quantization: Enabled
  - Bits: 8
  - Strategy: full_quantization
  - Symmetric: Yes
```

**Purpose**: Evaluate communication savings with quantization

---

### 🚦 Congestion Testing

```
Use Case: Mental State
Protocols: [✓] MQTT [✓] QUIC
Scenarios: [✓] Congested Light [✓] Congested Moderate [✓] Congested Heavy
GPU: Enabled
Traffic Congestion: Enabled
  - Level: Moderate
```

**Purpose**: Test protocol behavior under traffic congestion

---

### 📊 Comprehensive Evaluation (Long Run)

```
Use Case: Emotion Recognition
Protocols: [✓] All
Scenarios: [✓] All
GPU: Enabled (2 GPUs)
Rounds: 50
Quantization: Enabled (8-bit)
Compression: Enabled (gzip, level 6)
Save Checkpoints: Yes
Enable TensorBoard: Yes
```

**Purpose**: Full evaluation across all combinations (135 experiments)
**Estimated Time**: 6-12 hours

---

## Advanced Features

### 🎛️ Dynamic Network Adjustment

The sliders allow you to override scenario defaults:

```
Example: Custom Satellite-like Conditions
- Latency: 650 ms
- Bandwidth: 10 Mbps
- Jitter: 50 ms
- Packet Loss: 1%
```

These values will be applied to all selected scenarios.

### 📈 Real-Time Monitoring

**GPU Usage:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

**Container Logs:**
```bash
docker logs -f fl-server-mqtt-emotion
```

**Network Traffic:**
```bash
docker stats
```

### 🛑 Stopping Experiments

- Click "⏹️ Stop Experiment" button
- Confirms before stopping
- Gracefully terminates processes
- Partial results may still be saved

---

## Results Analysis

### Results Location

```
experiment_results/
├── emotion_20260129_143022/
│   ├── mqtt_excellent_results.json
│   ├── mqtt_moderate_results.json
│   ├── grpc_excellent_results.json
│   └── ...
└── mentalstate_20260129_150145/
    └── ...
```

### Consolidate Results

```bash
# Get latest experiment folder
LATEST=$(ls -td experiment_results/*/ | head -1 | xargs basename)

# Consolidate results
python3 Network_Simulation/consolidate_results.py \
    --use-case emotion \
    --experiment-folder $LATEST
```

### Metrics Collected

Each experiment produces:
- **Accuracy**: Model performance
- **Loss**: Training/validation loss
- **Communication Time**: Protocol overhead
- **Data Size**: Bytes transmitted
- **Latency**: Round-trip times
- **Throughput**: Messages/second
- **GPU Metrics**: Memory, utilization

---

## Troubleshooting

### GUI Won't Launch

**Error: PyQt5 not found**
```bash
pip install PyQt5
```

**Error: Display not found (headless server)**
```bash
# Use X11 forwarding or VNC
ssh -X user@server
./launch_experiment_gui.sh
```

### Experiment Fails to Start

**Check Docker:**
```bash
docker ps
docker info | grep Runtimes
```

**Check GPU:**
```bash
nvidia-smi
docker run --rm --runtime=nvidia nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Check Paths:**
- Ensure you're in the correct working directory
- Verify `run_network_experiments.py` exists

### No Output in Console

- Check if experiment actually started
- Look for error messages
- Verify network connectivity
- Check Docker logs

### Slow Performance

- Reduce number of rounds
- Limit protocols/scenarios
- Disable quantization/compression if not needed
- Use fewer GPUs if memory constrained
- Check system resources

---

## Command-Line Equivalent

The GUI builds commands like this:

```bash
# GUI Configuration:
# - Use Case: emotion
# - Protocols: mqtt, grpc
# - Scenarios: excellent, moderate
# - GPU: enabled
# - Rounds: 10
# - Quantization: 8-bit

# Equivalent Command:
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --rounds 10 \
    --protocols mqtt grpc \
    --scenarios excellent moderate \
    --use-quantization \
    --quantization-bits 8 \
    --quantization-strategy full_quantization
```

---

## Performance Tips

### For Fast Testing
- Use 3-5 rounds
- Select 1-2 protocols
- Select 1-2 scenarios
- Disable advanced features

### For Comprehensive Testing
- Use 50+ rounds
- Select all protocols
- Select all scenarios
- Enable quantization/compression
- Run overnight

### For GPU Optimization
- Enable GPU acceleration
- Use batch size 32-64
- Monitor GPU memory
- Adjust based on available VRAM

### For Network Testing
- Focus on specific scenarios
- Use dynamic sliders for fine-tuning
- Enable congestion for realism
- Test edge cases (satellite, very poor)

---

## Best Practices

1. **Start Small**: Test with single protocol/scenario first
2. **Monitor Resources**: Watch GPU/CPU/memory usage
3. **Save Configurations**: Document your test matrix
4. **Review Results**: Consolidate after each major run
5. **Iterative Testing**: Refine parameters based on results
6. **Use Checkpoints**: Enable for long experiments
7. **Version Control**: Track which configurations were tested
8. **Document Findings**: Note observations in output

---

## Keyboard Shortcuts

- **Ctrl+C**: Stop experiment (in terminal)
- **Ctrl+W**: Close GUI window
- **Tab**: Navigate between controls
- **Space**: Toggle checkboxes/buttons
- **Enter**: Activate focused button

---

## Support & Documentation

- **Main README**: `/README.md`
- **Experiment Guide**: `/EXPERIMENT_RUNNER_GUIDE.md`
- **Network Control**: `/NETWORK_CONDITIONS_USAGE.md`
- **Quantization**: `/QUANTIZATION_QUICK_REF.md`
- **GPU Setup**: `/GPU_QUICK_START.md`

---

## Version History

**v1.0** (2026-01-29)
- Initial release
- All basic features
- PyQt5 implementation
- Real-time monitoring
- Advanced configuration options

---

## Credits

Built for the Master Thesis: Hybrid Communication Framework for Federated Learning

🚀 Happy Experimenting! 🚀
