#!/bin/bash

#######################################################################
# Rebuild Docker Images with CycloneDDS Support
#######################################################################

echo "=========================================================================="
echo "Rebuilding Docker Images with CycloneDDS Support"
echo "=========================================================================="
echo ""

cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL

# Stop any running containers
echo "[Step 1/4] Stopping existing containers..."
docker-compose -f Docker/docker-compose-temperature.yml down 2>/dev/null
docker-compose -f Docker/docker-compose-emotion.yml down 2>/dev/null
docker-compose -f Docker/docker-compose-mentalstate.yml down 2>/dev/null

# Remove old images (optional - comment out if you want to keep them)
echo ""
echo "[Step 2/4] Removing old images (optional)..."
docker rmi masterthesissybridcommframework_fl-fl-client-temperature -f 2>/dev/null
docker rmi masterthesissybridcommframework_fl-fl-server-temperature -f 2>/dev/null

# Rebuild client image
echo ""
echo "[Step 3/4] Building Client image with CycloneDDS..."
docker build --no-cache -t fl-client-with-dds -f Client/Dockerfile .

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Client image build FAILED!"
    echo "Check the error messages above."
    exit 1
fi

# Rebuild server image
echo ""
echo "[Step 4/4] Building Server image with CycloneDDS..."
docker build --no-cache -t fl-server-with-dds -f Server/Dockerfile .

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Server image build FAILED!"
    echo "Check the error messages above."
    exit 1
fi

echo ""
echo "=========================================================================="
echo "✅ Docker Images Rebuilt Successfully with CycloneDDS!"
echo "=========================================================================="
echo ""
echo "Verification:"
echo "  docker run --rm fl-client-with-dds python -c 'import cyclonedds; print(\"CycloneDDS available\")'"
echo ""
echo "Now you can run DDS experiments:"
echo "  python Network_Simulation/run_network_experiments.py --use-case temperature --protocols dds --scenarios excellent --rounds 10"
echo ""
