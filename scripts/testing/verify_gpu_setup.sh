#!/bin/bash
# GPU Setup and Verification Script for Emotion Recognition FL

echo "================================"
echo "GPU Setup Verification Script"
echo "================================"
echo ""

# Check NVIDIA GPU
echo "1. Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "❌ NVIDIA GPU driver not found. Please install NVIDIA drivers."
fi

echo ""
echo "2. Checking CUDA..."
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "❌ CUDA toolkit not found. Please install CUDA Toolkit."
fi

echo ""
echo "3. Checking TensorFlow..."
python3 -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✅ GPUs found: {len(gpus)}')
    for gpu in gpus:
        print(f'   - {gpu}')
else:
    print('❌ No GPUs found')
" 2>/dev/null || echo "❌ TensorFlow not installed or error occurred"

echo ""
echo "4. System Information..."
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo ""
echo "================================"
echo "Installation Requirements:"
echo "================================"
echo ""
echo "If any checks failed, run:"
echo ""
echo "# Install TensorFlow with GPU support"
echo "pip install tensorflow[and-cuda]"
echo ""
echo "# Or reinstall if already installed"
echo "pip uninstall tensorflow tensorflow-gpu -y"
echo "pip install tensorflow[and-cuda]==2.13.0"
echo ""
