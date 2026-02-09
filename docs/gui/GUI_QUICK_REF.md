# 🚀 FL Experiment GUI - Quick Reference

## Launch GUI

```bash
./launch_experiment_gui.sh
```

## Quick Start (3 Steps)

1. **Select Configuration** (Tab 1: Basic Configuration)
   - Use Case: Choose one (emotion/mentalstate/temperature)
   - Protocols: Check boxes for desired protocols
   - Scenarios: Check boxes for network conditions
   - GPU: Enable if available

2. **Click "▶️ Start Experiment"**
   - Review confirmation dialog
   - Click "Yes" to start

3. **Monitor Progress**
   - Watch output console
   - Wait for completion
   - Find results in `experiment_results/`

---

## Common Configurations

### Quick Test (1 min)
```
✓ Emotion Recognition
✓ MQTT only
✓ Excellent scenario
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
✓ All use cases (run separately)
✓ All protocols
✓ All scenarios
✓ GPU enabled
✓ 50 rounds
```

---

## Key Features

| Feature | Location | Description |
|---------|----------|-------------|
| Protocol Selection | Tab 1 | Choose MQTT, AMQP, gRPC, QUIC, DDS |
| Network Scenarios | Tab 1 | 9 scenarios from excellent to satellite |
| GPU Control | Tab 1 | Enable/disable GPU acceleration |
| Latency Slider | Tab 2 | 0-1000ms network delay |
| Bandwidth Slider | Tab 2 | 1-1000 Mbps limit |
| Jitter Slider | Tab 2 | 0-100ms variability |
| Packet Loss | Tab 2 | 0-10% loss rate |
| **Apply Network** | **Tab 2** | **🆕 Apply changes during experiment** |
| Quantization | Tab 3 | 8/16/32-bit model compression |
| Compression | Tab 3 | gzip/lz4/zstd/snappy |
| Pruning | Tab 3 | 0-90% parameter reduction |
| **Experiment Output** | **Output** | **Main experiment logs** |
| **FL Training Monitor** | **Output** | **🆕 Real-time metrics vs baseline** |
| **Server Logs** | **Output** | **🆕 FL server container logs** |
| **Client Logs** | **Output** | **🆕 FL client container logs** |

---

## New Features (v1.1) 🆕

### 📈 FL Training Monitor
- Real-time training metrics
- **Baseline comparison** - see Δ from baseline
- Performance indicators (✅/⚠️/❌)
- Auto-refresh every 5 seconds

### 🖥️ Container Logs  
- Live server logs (green theme)
- Live client logs (yellow theme)
- Auto-scrolling
- Separate tabs for clarity

### 🌐 Dynamic Network Control
- Adjust sliders during experiments
- Click "Apply Network Changes"
- Immediate effect on all clients
- Test protocol resilience

---

## Sliders Quick Guide

**Latency** → Network delay (satellite = 600ms)  
**Bandwidth** → Speed limit (3G = ~5 Mbps)  
**Jitter** → Delay variation (unstable = 20-50ms)  
**Packet Loss** → Dropped packets (WiFi = 1-2%)

---

## Results

**Location**: `experiment_results/`  
**Format**: `{usecase}_{timestamp}/`  
**Files**: `{protocol}_{scenario}_results.json`

**Consolidate**:
```bash
python3 Network_Simulation/consolidate_results.py \
    --use-case emotion \
    --experiment-folder emotion_20260129_143022
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GUI won't start | `pip install PyQt5` |
| No GPU detected | `docker context use default` |
| Experiment hangs | Click "⏹️ Stop Experiment" |
| No output | Check Docker: `docker ps` |

---

## Monitor GPU
```bash
watch -n 1 nvidia-smi
```

## Stop Experiment
Click "⏹️ Stop Experiment" button

## Clear Output
Click "🗑️ Clear Output" button

---

## Tips

✅ **DO**:
- Start with quick test first
- Enable GPU for faster results
- Select multiple scenarios to compare
- Monitor GPU usage during experiments
- Save experiment configurations

❌ **DON'T**:
- Run all combinations at once initially
- Forget to check GPU availability
- Close GUI during experiment
- Mix incompatible options

---

## Example Workflows

### Workflow 1: Protocol Evaluation
```
1. Tab 1: Select all protocols
2. Tab 1: Select "Good" scenario
3. Tab 1: Enable GPU, 10 rounds
4. Start → Wait → Analyze results
```

### Workflow 2: Network Resilience
```
1. Tab 1: Select MQTT only
2. Tab 1: Select all scenarios
3. Tab 1: Enable GPU, 20 rounds
4. Start → Compare scenarios
```

### Workflow 3: Optimization Test
```
1. Tab 1: Select MQTT + gRPC
2. Tab 1: Select Moderate scenario
3. Tab 3: Enable quantization (8-bit)
4. Tab 3: Enable compression (gzip)
5. Start → Measure savings
```

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Tab | Navigate controls |
| Space | Toggle checkbox |
| Enter | Activate button |
| Ctrl+W | Close GUI |

---

For detailed documentation, see: [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md)
