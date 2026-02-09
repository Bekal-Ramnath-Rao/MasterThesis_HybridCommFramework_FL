# 🎨 NEW: Graphical User Interface (GUI)

## Quick Launch

```bash
./launch_experiment_gui.sh
```

---

## Overview

A beautiful, comprehensive GUI has been created for configuring and running FL experiments without dealing with command-line parameters. The GUI provides:

- ✨ **Intuitive Interface**: Organized in 3 tabs for easy navigation
- 🎛️ **30+ Configuration Options**: All experiment parameters in one place
- 📊 **Real-time Monitoring**: Live output console with experiment progress
- 🖥️ **Professional Design**: Modern gradient theme with purple-blue colors
- 🧵 **Background Execution**: UI stays responsive during experiments
- ✅ **Validation**: Automatic checks before starting experiments

---

## What You Can Configure

### 🎯 Basic Settings (Tab 1)
- **Use Cases**: Mental State, Emotion Recognition, Temperature
- **Protocols**: MQTT, AMQP, gRPC, QUIC, DDS (multi-select)
- **Scenarios**: 9 network conditions from excellent to satellite
- **GPU**: Enable/disable with GPU count selector
- **Training**: Rounds, batch size, learning rate, min clients

### 🌐 Network Control (Tab 2)
- **Dynamic Sliders**: Latency (0-1000ms), Bandwidth (1-1000 Mbps)
- **Jitter & Loss**: Fine-tune network conditions
- **Congestion**: Enable traffic generator with level control

### 🔧 Advanced Options (Tab 3)
- **Quantization**: 8/16/32-bit with multiple strategies
- **Compression**: gzip, lz4, zstd, snappy algorithms
- **Pruning**: Adjustable ratio (0-90%)
- **Other**: TensorBoard, profiling, verbose logging, checkpoints

---

## Screenshots

![GUI Main Window](docs/screenshots/gui_main.png)

The GUI features:
- **Header**: Gradient purple-blue with title and subtitle
- **3 Tabs**: Basic, Network, Advanced configuration
- **Control Panel**: Start (green), Stop (red), Clear buttons
- **Output Console**: Dark terminal theme with real-time updates
- **Status Bar**: Current experiment status

---

## Installation

### Quick Install
```bash
pip install PyQt5
```

### Verify Installation
```bash
python3 Network_Simulation/test_gui.py
```

### Launch GUI
```bash
# Using launcher (recommended)
./launch_experiment_gui.sh

# Or directly
python3 Network_Simulation/experiment_gui.py
```

---

## Quick Start Example

1. **Launch**: `./launch_experiment_gui.sh`

2. **Configure** (3 clicks):
   - Select "Emotion Recognition"
   - Check "MQTT" protocol
   - Check "Excellent" scenario

3. **Click**: "▶️ Start Experiment"

4. **Monitor**: Watch real-time output

5. **Results**: Find in `experiment_results/`

---

## Files Created

| File | Purpose |
|------|---------|
| `Network_Simulation/experiment_gui.py` | Main GUI application (1100+ lines) |
| `launch_experiment_gui.sh` | Launcher script with auto-install |
| `Network_Simulation/gui_requirements.txt` | Python dependencies |
| `Network_Simulation/test_gui.py` | Test suite for verification |
| `GUI_USER_GUIDE.md` | Complete user manual (600+ lines) |
| `GUI_QUICK_REF.md` | Quick reference card |
| `GUI_INSTALLATION.md` | Installation guide |
| `GUI_ARCHITECTURE.md` | Technical documentation |
| `GUI_SUMMARY.md` | Overview and summary |

**Total: 9 files, 2800+ lines of documentation**

---

## Documentation

- **For Quick Start**: Read [GUI_QUICK_REF.md](GUI_QUICK_REF.md)
- **For Complete Guide**: Read [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md)
- **For Installation Help**: Read [GUI_INSTALLATION.md](GUI_INSTALLATION.md)
- **For Developers**: Read [GUI_ARCHITECTURE.md](GUI_ARCHITECTURE.md)
- **For Overview**: Read [GUI_SUMMARY.md](GUI_SUMMARY.md)

---

## Example Configurations

### Quick Test (1 min)
```
✓ Emotion Recognition
✓ MQTT
✓ Excellent
✓ GPU enabled
✓ 3 rounds
```

### Protocol Comparison (30 min)
```
✓ Mental State
✓ All 5 protocols
✓ Good scenario
✓ GPU enabled
✓ 10 rounds
```

### Full Evaluation (8-12 hours)
```
✓ Any use case
✓ All protocols
✓ All scenarios
✓ GPU enabled
✓ 50 rounds
✓ Quantization (8-bit)
✓ Compression (gzip)
```

---

## Advantages Over CLI

| Feature | CLI | GUI |
|---------|-----|-----|
| Configuration | Remember flags | Visual controls |
| Validation | Manual | Automatic |
| Visualization | None | Real-time sliders |
| Monitoring | Separate terminal | Integrated console |
| Learning curve | Steep | Gentle |
| Ease of use | Complex | Simple |

---

## System Requirements

- **OS**: Linux (Ubuntu/Debian/Fedora), macOS, Windows
- **Python**: 3.7+
- **Display**: X11 server (or VNC for headless)
- **Memory**: ~50-100 MB for GUI
- **Dependencies**: PyQt5

---

## Troubleshooting

### GUI won't start
```bash
# Install PyQt5
pip install PyQt5

# Check display
echo $DISPLAY

# Test
python3 Network_Simulation/test_gui.py
```

### Headless server
```bash
# Option 1: X11 forwarding
ssh -X user@server
./launch_experiment_gui.sh

# Option 2: Use CLI instead
python3 Network_Simulation/run_network_experiments.py --help
```

---

## Future Enhancements

Potential additions:
- Results viewer with graphs
- Configuration presets (save/load)
- Batch experiment queue
- Live metrics visualization
- Web-based alternative
- Dark mode toggle

---

## Credits

Built with:
- **PyQt5**: Cross-platform GUI framework
- **Python Threading**: Background execution
- **Modern Design**: Gradient themes and professional UX

---

## Getting Started

1. ✅ Install PyQt5: `pip install PyQt5`
2. ✅ Test: `python3 Network_Simulation/test_gui.py`
3. ✅ Launch: `./launch_experiment_gui.sh`
4. ✅ Read: [GUI_QUICK_REF.md](GUI_QUICK_REF.md)
5. ✅ Experiment!

---

**The GUI is production-ready and fully documented! Enjoy! 🎉**

For command-line usage, see [commands.txt](Network_Simulation/commands.txt)
