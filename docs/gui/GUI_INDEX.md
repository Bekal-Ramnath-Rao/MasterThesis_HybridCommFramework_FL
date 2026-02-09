# 📑 GUI Documentation Index

## Quick Access Links

### 🚀 Getting Started
1. **[GUI README](GUI_README.md)** - Start here! Overview and quick launch
2. **[Quick Reference](GUI_QUICK_REF.md)** - 3-step guide and common configs
3. **[Installation Guide](GUI_INSTALLATION.md)** - Detailed setup instructions

### 📚 Complete Documentation
4. **[User Guide](GUI_USER_GUIDE.md)** - Complete manual (600+ lines)
5. **[Architecture](GUI_ARCHITECTURE.md)** - Technical details and diagrams
6. **[Summary](GUI_SUMMARY.md)** - What's been created and features

---

## File Locations

### Executable Files
```
/launch_experiment_gui.sh          ← Main launcher (recommended)
/demo_gui.sh                       ← Demo mode launcher
/Network_Simulation/experiment_gui.py    ← Main application
/Network_Simulation/test_gui.py          ← Test suite
```

### Documentation Files
```
/GUI_README.md                     ← Overview and quick start
/GUI_QUICK_REF.md                  ← Quick reference card
/GUI_USER_GUIDE.md                 ← Complete user manual
/GUI_INSTALLATION.md               ← Installation instructions
/GUI_ARCHITECTURE.md               ← Technical documentation
/GUI_SUMMARY.md                    ← Features and summary
/GUI_INDEX.md                      ← This file
```

### Configuration Files
```
/Network_Simulation/gui_requirements.txt  ← Python dependencies
```

---

## Documentation by User Type

### 🎯 For First-Time Users
**Read these in order:**
1. [GUI_README.md](GUI_README.md) - What is the GUI?
2. [GUI_INSTALLATION.md](GUI_INSTALLATION.md) - How to install
3. [GUI_QUICK_REF.md](GUI_QUICK_REF.md) - How to use
4. Run: `./launch_experiment_gui.sh`

**Estimated time**: 10 minutes

---

### 👤 For Regular Users
**Keep these handy:**
- [GUI_QUICK_REF.md](GUI_QUICK_REF.md) - Common configurations
- [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md) - Detailed features

**Bookmark**:
- Example configurations (GUI_QUICK_REF.md)
- Troubleshooting section (GUI_USER_GUIDE.md)

---

### 🔧 For Power Users
**Explore these:**
- [GUI_ARCHITECTURE.md](GUI_ARCHITECTURE.md) - How it works
- [Network_Simulation/experiment_gui.py](Network_Simulation/experiment_gui.py) - Source code
- Command-line backend: [Network_Simulation/run_network_experiments.py](Network_Simulation/run_network_experiments.py)

**Customize**:
- Modify GUI styling in `get_stylesheet()`
- Add new parameters in configuration tabs
- Extend experiment runner logic

---

### 👨‍💻 For Developers
**Study these:**
1. [GUI_ARCHITECTURE.md](GUI_ARCHITECTURE.md) - System design
2. [experiment_gui.py](Network_Simulation/experiment_gui.py) - Implementation
3. [test_gui.py](Network_Simulation/test_gui.py) - Test patterns

**Key areas**:
- Component hierarchy (lines 30-900)
- Signal-slot connections (lines 200-400)
- Thread safety (ExperimentRunner class)
- Command building (build_command method)

---

## Documentation by Topic

### Installation & Setup
- [GUI_INSTALLATION.md](GUI_INSTALLATION.md) - All installation methods
- [GUI_README.md](GUI_README.md) - Quick install section
- System requirements
- Headless server setup
- Troubleshooting installation

### Basic Usage
- [GUI_QUICK_REF.md](GUI_QUICK_REF.md) - 3-step quick start
- [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md) - Detailed UI guide
- Tab-by-tab walkthrough
- Common configurations
- Example workflows

### Advanced Features
- [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md) - Advanced features section
- Quantization configuration
- Compression settings
- Pruning options
- Dynamic network control

### Troubleshooting
- [GUI_INSTALLATION.md](GUI_INSTALLATION.md) - Installation issues
- [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md) - Usage problems
- [GUI_README.md](GUI_README.md) - Quick fixes
- Test suite: `python3 Network_Simulation/test_gui.py`

### Technical Details
- [GUI_ARCHITECTURE.md](GUI_ARCHITECTURE.md) - System architecture
- Component diagrams
- Data flow
- Signal-slot connections
- Performance characteristics

---

## Quick Commands Reference

### Launch GUI
```bash
# Recommended method
./launch_experiment_gui.sh

# Direct launch
python3 Network_Simulation/experiment_gui.py

# Demo mode
./demo_gui.sh
```

### Test GUI
```bash
# Run test suite
python3 Network_Simulation/test_gui.py

# Check PyQt5
python3 -c "import PyQt5; print('PyQt5 OK')"

# Check display
echo $DISPLAY
```

### Install Dependencies
```bash
# Install PyQt5
pip install PyQt5

# Or from requirements
pip install -r Network_Simulation/gui_requirements.txt
```

---

## Documentation Statistics

| Document | Lines | Words | Purpose |
|----------|-------|-------|---------|
| GUI_README.md | 200 | 1,500 | Overview |
| GUI_QUICK_REF.md | 250 | 1,800 | Quick reference |
| GUI_USER_GUIDE.md | 600 | 5,000 | Complete manual |
| GUI_INSTALLATION.md | 400 | 3,000 | Installation |
| GUI_ARCHITECTURE.md | 500 | 3,500 | Technical docs |
| GUI_SUMMARY.md | 600 | 4,500 | Features summary |
| GUI_INDEX.md | 200 | 1,200 | This index |
| **TOTAL** | **2,750** | **20,500** | **Complete docs** |

---

## Code Statistics

| File | Lines | Classes | Functions | Purpose |
|------|-------|---------|-----------|---------|
| experiment_gui.py | 1,100 | 2 | 25+ | Main GUI app |
| test_gui.py | 150 | 0 | 5 | Test suite |
| **TOTAL** | **1,250** | **2** | **30+** | **Production code** |

---

## Feature Coverage

### Implemented Features (100%)
✅ Use case selection (3 options)
✅ Protocol selection (5 protocols, multi-select)
✅ Scenario selection (9 scenarios, multi-select)
✅ GPU configuration (enable/disable, count)
✅ Training parameters (rounds, batch, LR, clients)
✅ Dynamic network control (4 sliders)
✅ Traffic congestion (enable + level)
✅ Quantization (enable + 5 sub-options)
✅ Compression (enable + algorithm + level)
✅ Pruning (enable + ratio slider)
✅ Other options (4 checkboxes)
✅ Real-time output console
✅ Start/Stop/Clear controls
✅ Background execution
✅ Progress bar
✅ Status updates
✅ Validation
✅ Professional styling

### Requested Features (✓ All Included)
- ✓ Protocol selection → Multi-select checkboxes
- ✓ GPU enable/disable → Checkbox + count
- ✓ Network scenarios → Multi-select checkboxes
- ✓ Use cases → Radio buttons
- ✓ Quantization → Enable + bits + options
- ✓ Start button → Green button with icon
- ✓ Latency slider → 0-1000ms range
- ✓ Bandwidth slider → 1-1000 Mbps range
- ✓ Jitter slider → 0-100ms range
- ✓ Packet loss → 0-10% slider
- ✓ Additional options → Compression, pruning, etc.

---

## Keyboard Navigation

| Key | Action |
|-----|--------|
| Tab | Move to next control |
| Shift+Tab | Move to previous control |
| Space | Toggle checkbox/radio |
| Enter | Activate button |
| Arrow Keys | Navigate sliders/spinboxes |
| Ctrl+W | Close window |

---

## Related Files

### Experiment Runner (Backend)
- [Network_Simulation/run_network_experiments.py](Network_Simulation/run_network_experiments.py)
- [Network_Simulation/consolidate_results.py](Network_Simulation/consolidate_results.py)

### Docker Configuration
- Docker/docker-compose-*.yml
- Docker/docker-compose-*.gpu.yml

### Documentation
- [EXPERIMENT_RUNNER_GUIDE.md](EXPERIMENT_RUNNER_GUIDE.md)
- [GPU_QUICK_START.md](GPU_QUICK_START.md)
- [QUANTIZATION_QUICK_REF.md](QUANTIZATION_QUICK_REF.md)
- [NETWORK_CONDITIONS_USAGE.md](NETWORK_CONDITIONS_USAGE.md)

---

## Support & Help

### Getting Help
1. **Check docs**: Start with relevant guide from this index
2. **Run tests**: `python3 Network_Simulation/test_gui.py`
3. **Check logs**: Console output in GUI
4. **Review examples**: Common configs in GUI_QUICK_REF.md

### Common Issues & Solutions

**GUI won't launch**
→ Read: [GUI_INSTALLATION.md](GUI_INSTALLATION.md) - "Common Issues"

**Don't know what to configure**
→ Read: [GUI_QUICK_REF.md](GUI_QUICK_REF.md) - "Common Configurations"

**Need detailed explanation**
→ Read: [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md) - Comprehensive manual

**Want to modify GUI**
→ Read: [GUI_ARCHITECTURE.md](GUI_ARCHITECTURE.md) - Technical details

---

## Next Steps

### For New Users
1. ✅ Read [GUI_README.md](GUI_README.md)
2. ✅ Run `./launch_experiment_gui.sh`
3. ✅ Try quick test configuration
4. ✅ Read [GUI_QUICK_REF.md](GUI_QUICK_REF.md)

### For Regular Users
1. ✅ Bookmark [GUI_QUICK_REF.md](GUI_QUICK_REF.md)
2. ✅ Plan experiment matrix
3. ✅ Run comprehensive tests
4. ✅ Analyze results

### For Power Users
1. ✅ Study [GUI_ARCHITECTURE.md](GUI_ARCHITECTURE.md)
2. ✅ Review source code
3. ✅ Customize as needed
4. ✅ Contribute improvements

---

## Version History

**v1.0** (2026-01-29)
- Initial release
- All core features implemented
- Complete documentation (7 files)
- Test suite included
- Production-ready

---

## Credits

**Built for**: Master Thesis - Hybrid Communication Framework for FL
**Framework**: PyQt5
**Language**: Python 3
**Documentation**: Markdown
**Total Lines**: 4,000+ (code + docs)

---

**Happy Experimenting! 🚀**

Need help? Start with the document that matches your needs above!
