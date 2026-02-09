# How to Fix: ImportError with numpy

## Problem
You're getting an ImportError about numpy because VS Code is using the system Python (`/bin/python3.12`) which has an old, incompatible numpy version (1.21.5).

## Solution

### Option 1: Use Conda Python from Terminal (Recommended)
Simply use the correct Python from conda when running scripts:

```bash
# Use conda Python directly
python /path/to/script.py

# Or use the full path
/home/ubuntu/miniconda3/bin/python /path/to/script.py
```

### Option 2: Configure VS Code to Use Conda Python

1. Open VS Code settings:
   - Press `Ctrl + ,` (or `Cmd + ,` on Mac)
   - Or go to File → Preferences → Settings

2. Search for "Python: Default Interpreter Path"

3. Set it to:
   ```
   /home/ubuntu/miniconda3/bin/python
   ```

### Option 3: Verify Python Version

Check which Python is being used:

```bash
# Check default python
which python
python --version

# Check default python3  
which python3
python3 --version

# Check system python
/bin/python3.12 --version
```

## Why This Happens

- **Conda Python (3.13)**: Has new packages including numpy 2.2.6 ✓
- **System Python (3.12)**: Has old numpy 1.21.5 which is broken ✗

VS Code's Python Debug Console might be using the system Python by default.

## Quick Test

Run this to confirm conda Python works:

```bash
python -c "import numpy; print('NumPy version:', numpy.__version__)"
# Should show: NumPy version: 2.2.6
```

If system Python fails:

```bash
/bin/python3.12 -c "import numpy"
# Will show ImportError
```

## Summary

✅ **Use conda Python**: `/home/ubuntu/miniconda3/bin/python`  
❌ **Avoid system Python**: `/bin/python3.12`
