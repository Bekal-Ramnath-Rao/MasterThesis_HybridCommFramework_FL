# Code Compatibility Improvements

## Overview
This document outlines the enhancements made to improve code flexibility and compatibility with modern Python environments, particularly for Python 3.13.9 and CUDA 12.x support.

## Changes Made

### 1. **Flexible GPU Device Configuration**

**Files Modified:**
- `Client/Emotion_Recognition/FL_Client_Unified.py`
- `Server/Emotion_Recognition/FL_Server_Unified.py`

**Changes:**
- **Before:** GPU device was hardcoded to `CUDA_VISIBLE_DEVICES='0'`
- **After:** 
  - Reads `GPU_DEVICE_ID` environment variable first
  - Falls back to `CUDA_VISIBLE_DEVICES` if `GPU_DEVICE_ID` is not set
  - Defaults to `'0'` if neither is set
  
**Benefits:**
- Allows users to easily switch GPU devices using environment variables
- Prevents conflicts when running multiple experiments on different GPUs
- Supports multi-GPU systems more flexibly

**Usage:**
```bash
# Use specific GPU
GPU_DEVICE_ID=1 python Client/Emotion_Recognition/FL_Client_Unified.py

# Or use traditional method
CUDA_VISIBLE_DEVICES=1 python Client/Emotion_Recognition/FL_Client_Unified.py
```

### 2. **pip-Installed CUDA Compatibility**

**Files Modified:**
- `Client/Emotion_Recognition/FL_Client_Unified.py`
- `Server/Emotion_Recognition/FL_Server_Unified.py`
- `Network_Simulation/run_native_experiments.py`

**Changes:**
- Automatically detects and adds pip-installed CUDA 12 binaries to `PATH`
- Looks for `nvidia/cuda_nvcc/bin` in the Python site-packages directory
- Ensures the correct version of `ptxas` (PTX assembler) is used

**Benefits:**
- Resolves "ptxas is too old" errors when using pip-installed CUDA
- Eliminates conflicts between system CUDA and pip-installed CUDA
- Works with modern Python virtual environments that have pip-installed CUDA

**Technical Detail:**
```python
_nvcc_bin = os.path.join(sys.prefix, 'lib', 
                        f'python{sys.version_info[0]}.{sys.version_info[1]}',
                        'site-packages', 'nvidia', 'cuda_nvcc', 'bin')
```

### 3. **Stale CUDA Library Path Cleanup**

**Files Modified:**
- `Client/Emotion_Recognition/FL_Client_Unified.py`
- `Server/Emotion_Recognition/FL_Server_Unified.py`
- `Network_Simulation/run_native_experiments.py`

**Changes:**
- Removes obsolete CUDA 10.x paths from `LD_LIBRARY_PATH`
- Prevents library version conflicts that can cause runtime errors
- Filters out any path containing 'cuda-10' substring

**Benefits:**
- Eliminates "libcudart.so.10.x not found" errors
- Prevents version mismatch issues between old and new CUDA versions
- Improves reliability when upgrading from older CUDA versions

### 4. **Forward Compatibility for Type Hints**

**Files Modified:**
- `Network_Simulation/network_delay_model.py`

**Changes:**
- Added `from __future__ import annotations` at the module level
- Enables PEP 563 - Postponed Evaluation of Annotations

**Benefits:**
- Allows use of modern type hints syntax on older Python versions
- Prepares codebase for future Python versions
- Enables string-based annotations for forward references

### 5. **Local Privileged Environment Configuration**

**Files Modified:**
- `Network_Simulation/run_native_experiments.py`

**Changes:**
- Added `_load_local_privileged_env()` function to load local environment configuration
- Supports `.privileged_ops.env` or `privileged_ops.env` files in `Network_Simulation/` directory
- Validates environment variables and skips placeholder values
- Called at the start of `main()` function

**Benefits:**
- Enables local privileged operation configuration without modifying code
- Supports multi-user experiments without security concerns
- Flexible credential and environment management for native experiments

### 6. **Enhanced GPU Configuration for Native Experiments**

**Files Modified:**
- `Network_Simulation/run_native_experiments.py`

**Changes for Server:**
- Pin server to GPU 0
- Apply pip-installed CUDA 12 ptxas to PATH
- Enhanced logging for GPU mode selection

**Changes for Clients:**
- Pin clients to GPU 1 (if available) or GPU 0 (fallback)
- Apply pip-installed CUDA 12 ptxas to PATH
- Improved distinction between `CUDA_VISIBLE_DEVICES` and `GPU_DEVICE_ID`

**Benefits:**
- Prevents GPU contention when running server and clients on the same machine
- Optimizes GPU utilization in multi-GPU systems
- Better error handling for single-GPU systems

## Environment Variables Supported

| Variable | Purpose | Default |
|----------|---------|---------|
| `GPU_DEVICE_ID` | Primary GPU device selector | `'0'` |
| `CUDA_VISIBLE_DEVICES` | Alternative GPU device selector | `'0'` |
| `TF_CPP_MIN_LOG_LEVEL` | TensorFlow verbosity level | `'3'` |
| `PATH` | Contains CUDA binaries | Auto-detected |
| `LD_LIBRARY_PATH` | Contains CUDA libraries (cleaned) | Auto-cleaned |

## Testing the Changes

### Verify GPU Configuration
```bash
GPU_DEVICE_ID=0 python Client/Emotion_Recognition/FL_Client_Unified.py
```

### Check for CUDA Path Issues
```bash
python -c "import os; print('CUDA path in PATH:', any('cuda_nvcc' in p for p in os.environ.get('PATH', '').split(':')))"
```

### Validate Syntax
```bash
python -m py_compile Client/Emotion_Recognition/FL_Client_Unified.py
python -m py_compile Server/Emotion_Recognition/FL_Server_Unified.py
python -m py_compile Network_Simulation/network_delay_model.py
python -m py_compile Network_Simulation/run_native_experiments.py
```

## Compatibility Matrix

| Python Version | TensorFlow | NumPy | CUDA | Status |
|---|---|---|---|---|
| 3.13.9 | 2.20.0 | 2.2.6 | 12.x | ✓ Tested |
| 3.12.x | 2.20.0 | 2.2.x | 12.x | ✓ Expected |
| 3.11.x | 2.19.0 | 2.1.x | 12.x | ✓ Expected |

## Troubleshooting

### Issue: "ptxas is too old"
**Solution:** Ensure pip-installed CUDA is accessible
```bash
python -c "import sys; print(sys.prefix)"
# Check if cuda_nvcc/bin exists in the output path
```

### Issue: "libcudart.so.10.x not found"
**Solution:** Verify CUDA 10.x is removed from libraries
```bash
echo $LD_LIBRARY_PATH | grep -i cuda-10
# Should be empty if successfully cleaned
```

### Issue: GPU not detected
**Solution:** Set GPU device explicitly
```bash
GPU_DEVICE_ID=0 python script.py
```

## Notes

- All modifications maintain backward compatibility
- Hardcoded values are preserved as defaults
- No breaking changes to existing APIs
- Code is Python 3.11+ compatible
