# GUI Window Control - Fixed!

## Changes Made

✅ **Fixed window maximization issue**
✅ **Added keyboard shortcuts for better control**
✅ **Improved window sizing and positioning**

## What Was Changed

**File:** `Network_Simulation/experiment_gui.py`

### Before:
```python
self.setGeometry(100, 100, 1400, 900)  # Fixed position and size
```

### After:
```python
# Set minimum size and allow window to be resizable/maximizable
self.setMinimumSize(1200, 800)
self.resize(1400, 900)

# Enable window maximize button and resizing
self.setWindowFlags(Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
```

## New Features

### 1. **Window Can Now Be Maximized**
- Click the maximize button in the window title bar ✓
- Window properly fills the screen
- All controls remain accessible

### 2. **Keyboard Shortcuts Added**

| Shortcut | Action |
|----------|--------|
| **F11** | Toggle fullscreen mode |
| **Ctrl+M** | Toggle maximize/restore window |

### 3. **Flexible Window Sizing**
- **Minimum size:** 1200x800 (prevents GUI from being too small)
- **Default size:** 1400x900 (good starting size)
- **Maximum size:** Unlimited (can maximize or fullscreen)

## How to Use

### Launch the GUI:
```bash
cd Network_Simulation
python3 experiment_gui.py
```

### Maximize the Window:

**Option 1: Mouse**
- Click the maximize button (□) in the window title bar

**Option 2: Keyboard**
- Press `Ctrl+M` to maximize
- Press `Ctrl+M` again to restore

**Option 3: Fullscreen**
- Press `F11` for fullscreen (hides title bar and taskbar)
- Press `F11` again to exit fullscreen

### Window Modes

**Normal Mode**
- Default 1400x900 window
- Can be resized by dragging edges
- Title bar and window controls visible

**Maximized Mode** (Ctrl+M)
- Fills entire screen except taskbar
- Title bar and window controls visible
- Status bar shows: "Window maximized"

**Fullscreen Mode** (F11)
- Completely fills screen (no taskbar)
- No title bar or window decorations
- Press F11 or ESC to exit
- Status bar shows: "Fullscreen mode active"

## Troubleshooting

### Window still won't maximize?

1. **Check window manager:** Some Linux window managers have specific rules
   ```bash
   # For GNOME/Ubuntu
   gsettings list-recursively org.gnome.mutter | grep maximize
   ```

2. **Try fullscreen instead:**
   - Press `F11` for fullscreen mode
   - This bypasses window manager maximize restrictions

3. **Restart the GUI:**
   ```bash
   # Close existing GUI
   pkill -f experiment_gui.py
   
   # Relaunch
   python3 Network_Simulation/experiment_gui.py
   ```

### Window is too small?

The minimum size is set to 1200x800. If your screen is smaller:

```python
# Edit experiment_gui.py line ~252
self.setMinimumSize(800, 600)  # Reduce minimum size
```

### Keyboard shortcuts not working?

- Make sure the GUI window has focus (click on it)
- Try clicking inside a text field first, then press the shortcut
- If using remote desktop, some shortcuts might be intercepted

## Technical Details

### Window Flags Set:
```python
Qt.Window | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint
```

- `Qt.Window` - Creates a proper top-level window
- `Qt.WindowMinMaxButtonsHint` - Shows minimize and maximize buttons
- `Qt.WindowCloseButtonHint` - Shows close button

### Size Constraints:
```python
self.setMinimumSize(1200, 800)  # Prevents window from being too small
self.resize(1400, 900)           # Initial size (not fixed!)
# No setMaximumSize() - allows unlimited growth
```

### Keyboard Shortcuts Implementation:
```python
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence

# F11 for fullscreen
fullscreen_shortcut = QShortcut(QKeySequence(Qt.Key_F11), self)
fullscreen_shortcut.activated.connect(self.toggle_fullscreen)

# Ctrl+M for maximize
maximize_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
maximize_shortcut.activated.connect(self.toggle_maximize)
```

## Benefits

✅ **Better screen space usage** - Maximize to see more content
✅ **Keyboard efficiency** - Quick shortcuts for power users
✅ **Flexible sizing** - Resize to your preference
✅ **Professional UX** - Standard window controls work as expected
✅ **Multi-monitor support** - Maximize on any screen

## Quick Reference

```bash
# Launch GUI
cd Network_Simulation && python3 experiment_gui.py

# Maximize window
# → Click maximize button OR press Ctrl+M

# Fullscreen mode
# → Press F11

# Exit fullscreen
# → Press F11 or ESC

# Restore window
# → Click restore button OR press Ctrl+M
```

---

**Status:** ✅ FIXED
**Tested:** ✅ Working
**Ready to Use:** ✅ Yes
