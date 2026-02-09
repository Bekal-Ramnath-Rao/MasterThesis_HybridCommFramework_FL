# 🖥️ GUI Installation & Launch Guide

## Prerequisites

- Python 3.7+
- PyQt5 (automatically installed)
- Linux with X11 display (GUI support)

---

## Installation

### Method 1: Automatic (Recommended)

The launcher script will automatically install dependencies:

```bash
./launch_experiment_gui.sh
```

### Method 2: Manual

```bash
# Install PyQt5
pip install -r Network_Simulation/gui_requirements.txt

# Launch GUI
python3 Network_Simulation/experiment_gui.py
```

### Method 3: Using Package Manager

```bash
# Debian/Ubuntu
sudo apt-get install python3-pyqt5

# Fedora
sudo dnf install python3-qt5

# macOS (Homebrew)
brew install pyqt5

# Then launch
python3 Network_Simulation/experiment_gui.py
```

---

## First Launch

1. **Navigate to project root**:
   ```bash
   cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL
   ```

2. **Launch GUI**:
   ```bash
   ./launch_experiment_gui.sh
   ```

3. **GUI should appear** with three tabs:
   - ⚙️ Basic Configuration
   - 🌐 Network Control
   - 🔧 Advanced Options

---

## Headless Server Setup

If running on a headless server (no display), you have options:

### Option 1: X11 Forwarding (SSH)

```bash
# On your local machine
ssh -X user@server-ip

# Then on server
./launch_experiment_gui.sh
```

### Option 2: VNC Server

```bash
# Install VNC server
sudo apt-get install tightvncserver

# Start VNC
vncserver :1 -geometry 1920x1080 -depth 24

# Connect from local machine
# Use VNC client to connect to server-ip:5901
```

### Option 3: Xvfb (Virtual Display)

```bash
# Install Xvfb
sudo apt-get install xvfb

# Launch with virtual display
xvfb-run python3 Network_Simulation/experiment_gui.py
```

### Option 4: Use Command Line

If GUI is not feasible, use the command-line interface:

```bash
# See commands.txt for all command-line options
python3 Network_Simulation/run_network_experiments.py \
    --use-case emotion \
    --enable-gpu \
    --rounds 10
```

---

## Verifying Installation

### Check PyQt5

```bash
python3 -c "import PyQt5; print('PyQt5 version:', PyQt5.QtCore.QT_VERSION_STR)"
```

Expected output:
```
PyQt5 version: 5.15.x
```

### Check Display

```bash
echo $DISPLAY
```

Should show something like `:0` or `:1`

### Test GUI

```bash
python3 -c "from PyQt5.QtWidgets import QApplication; app = QApplication([]); print('GUI works!')"
```

---

## Common Installation Issues

### Issue 1: PyQt5 Installation Fails

**Error**: `Could not find a version that satisfies the requirement PyQt5`

**Solution**:
```bash
# Update pip
pip install --upgrade pip

# Try again
pip install PyQt5
```

### Issue 2: Display Connection Error

**Error**: `qt.qpa.xcb: could not connect to display`

**Solution**:
```bash
# Check DISPLAY variable
echo $DISPLAY

# If empty, set it
export DISPLAY=:0

# Or use X11 forwarding
ssh -X user@server
```

### Issue 3: Missing Dependencies

**Error**: `ImportError: libxcb-xinerama.so.0`

**Solution (Ubuntu/Debian)**:
```bash
sudo apt-get install libxcb-xinerama0
```

**Solution (Fedora)**:
```bash
sudo dnf install libxcb
```

### Issue 4: Permission Denied

**Error**: `Permission denied: './launch_experiment_gui.sh'`

**Solution**:
```bash
chmod +x launch_experiment_gui.sh
./launch_experiment_gui.sh
```

### Issue 5: Qt Platform Plugin Error

**Error**: `Could not load the Qt platform plugin "xcb"`

**Solution**:
```bash
# Install Qt platform plugins
sudo apt-get install qt5-default

# Or set platform to offscreen
export QT_QPA_PLATFORM=offscreen
```

---

## Alternative Frontends

If PyQt5 GUI doesn't work for your environment:

### Web-Based Interface (Future)
A web-based dashboard could be created using:
- Flask/FastAPI backend
- React/Vue.js frontend
- Access via browser at `http://localhost:5000`

### TUI (Terminal UI)
A text-based interface using:
- `curses` library
- Works in any terminal
- No display server needed

### Jupyter Notebook
Use Jupyter widgets for interactive configuration:
```bash
jupyter notebook Network_Simulation/experiment_notebook.ipynb
```

---

## File Locations

| File | Path | Purpose |
|------|------|---------|
| Main GUI | `Network_Simulation/experiment_gui.py` | Main application |
| Launcher | `launch_experiment_gui.sh` | Startup script |
| Requirements | `Network_Simulation/gui_requirements.txt` | Dependencies |
| User Guide | `GUI_USER_GUIDE.md` | Full documentation |
| Quick Ref | `GUI_QUICK_REF.md` | Quick reference |

---

## Updating the GUI

To get latest version:

```bash
# Pull updates
git pull origin main

# Reinstall dependencies if needed
pip install -r Network_Simulation/gui_requirements.txt
```

---

## Uninstallation

To remove PyQt5:

```bash
pip uninstall PyQt5 PyQt5-sip
```

The GUI is self-contained and can be removed by deleting:
- `Network_Simulation/experiment_gui.py`
- `launch_experiment_gui.sh`
- `Network_Simulation/gui_requirements.txt`

---

## Performance Considerations

### GUI Performance
- GUI runs on main thread
- Experiments run in background thread
- No impact on experiment performance
- Safe to run on experiment server

### Resource Usage
- GUI: ~50-100 MB RAM
- Minimal CPU usage
- No GPU usage (experiments use GPU)

---

## Screenshots

### Main Dashboard
![Main Dashboard](docs/screenshots/gui_main.png)

### Basic Configuration Tab
![Basic Config](docs/screenshots/gui_basic_config.png)

### Network Control Tab
![Network Control](docs/screenshots/gui_network_control.png)

### Advanced Options Tab
![Advanced Options](docs/screenshots/gui_advanced_options.png)

### Running Experiment
![Running Experiment](docs/screenshots/gui_running.png)

---

## Getting Help

If you encounter issues:

1. **Check User Guide**: `GUI_USER_GUIDE.md`
2. **Check Quick Reference**: `GUI_QUICK_REF.md`
3. **Check Logs**: Console output in GUI
4. **Check Docker**: `docker ps` and `docker logs`
5. **Check System**: `nvidia-smi` for GPU

---

## Next Steps

After installation:

1. ✅ Read the [Quick Reference](GUI_QUICK_REF.md)
2. ✅ Run a quick test experiment
3. ✅ Review the [User Guide](GUI_USER_GUIDE.md)
4. ✅ Plan your experiment matrix
5. ✅ Start comprehensive testing!

---

## Support

For questions or issues:
- Refer to main documentation in `/docs`
- Check experiment guides in root directory
- Review command-line alternatives in `commands.txt`

---

**Happy Experimenting! 🚀**
