#!/bin/bash
# Install CycloneDDS C Library and Python Bindings

set -e

echo "============================================"
echo "CycloneDDS Installation Script"
echo "============================================"
echo ""

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo "This script needs sudo privileges to install system packages."
    echo "Please run with: sudo bash install_cyclonedds.sh"
    exit 1
fi

echo "Step 1: Installing build dependencies..."
apt-get update
apt-get install -y cmake git build-essential

echo ""
echo "Step 2: Cloning CycloneDDS repository..."
cd /tmp
if [ -d "cyclonedds" ]; then
    rm -rf cyclonedds
fi
git clone https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds

echo ""
echo "Step 3: Building CycloneDDS..."
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)

echo ""
echo "Step 4: Installing CycloneDDS..."
make install
ldconfig

echo ""
echo "Step 5: Installing Python cyclonedds package..."
# Switch back to regular user
SUDO_USER_HOME=$(eval echo ~$SUDO_USER)
su - $SUDO_USER -c "pip install cyclonedds"

echo ""
echo "============================================"
echo "✅ CycloneDDS Installation Complete!"
echo "============================================"
echo ""
echo "Verify installation:"
echo "  python -c 'import cyclonedds; print(cyclonedds.__version__)'"
echo ""
