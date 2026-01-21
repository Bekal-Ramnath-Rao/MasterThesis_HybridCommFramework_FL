# CycloneDDS Installation Guide

## Problem
CycloneDDS Python package requires the CycloneDDS C library to be installed first. The error you're seeing (`dds/ddsi/q_radmin.h: No such file or directory`) means the C library headers are missing.

## ⚠️ Recommendation: Skip DDS Protocol

**You already have 4 working protocols (80% coverage):**
- ✅ AMQP (pika)
- ✅ gRPC (grpcio)
- ✅ MQTT (paho-mqtt)
- ✅ QUIC (aioquic)

**Just use one of these instead!** DDS is optional and not critical for your federated learning system.

---

## Option 1: Use Alternative Protocols (Easiest)

Simply use AMQP, gRPC, MQTT, or QUIC for your experiments:

```bash
# Run with AMQP
./run_emotion_recognition_gpu.sh 0 amqp

# Run with MQTT
./run_emotion_recognition_gpu.sh 0 mqtt

# Run with gRPC
./run_emotion_recognition_gpu.sh 0 grpc

# Run with QUIC
./run_emotion_recognition_gpu.sh 0 quic
```

---

## Option 2: Install CycloneDDS (Advanced - 15-20 minutes)

If you absolutely need DDS protocol, follow these steps:

### Method A: Using Installation Script (Automated)

```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL
chmod +x install_cyclonedds.sh
sudo bash install_cyclonedds.sh
```

### Method B: Manual Installation

```bash
# 1. Install build tools
sudo apt-get update
sudo apt-get install -y cmake git build-essential

# 2. Clone CycloneDDS
cd /tmp
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds

# 3. Build and install
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install
sudo ldconfig

# 4. Install Python bindings
pip install cyclonedds

# 5. Verify installation
python -c "import cyclonedds; print('CycloneDDS version:', cyclonedds.__version__)"
```

---

## Verification

After installation, verify cyclonedds works:

```bash
python3 << 'EOF'
try:
    import cyclonedds
    from cyclonedds.domain import DomainParticipant
    print("✅ CycloneDDS installed successfully!")
    print(f"Version: {cyclonedds.__version__}")
except ImportError as e:
    print(f"❌ CycloneDDS not available: {e}")
EOF
```

---

## Why Installation Fails

The `pip install cyclonedds` command tries to compile the Python bindings from source, but it needs:

1. **CycloneDDS C library** - Core DDS implementation
2. **Development headers** - Header files (*.h) for compilation
3. **CMake** - Build system
4. **Compiler** - GCC/Clang

Without the C library installed, pip can't find the header files, causing the compilation to fail.

---

## Summary

| Option | Difficulty | Time | Recommended |
|--------|-----------|------|-------------|
| **Skip DDS** | Easy | 0 min | ✅ **Yes** |
| **Use script** | Medium | 15-20 min | If needed |
| **Manual install** | Hard | 20-30 min | Advanced users |

**My recommendation**: Skip DDS and use AMQP, gRPC, MQTT, or QUIC. They work perfectly and are sufficient for your federated learning research!
