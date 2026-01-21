# Docker DDS Support - Setup & Testing Guide

## What Was Fixed

**Problem:** CycloneDDS was not installed in Docker containers, causing DDS protocol to fail.

**Solution Applied:**
1. ✅ Enabled `cyclonedds>=0.10.0` in [requirements.txt](requirements.txt)
2. ✅ Added CycloneDDS C library build to [Client/Dockerfile](Client/Dockerfile)
3. ✅ Added CycloneDDS C library build to [Server/Dockerfile](Server/Dockerfile)

## Rebuild Docker Images

### Step 1: Rebuild (5-10 minutes)
```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL

# Option A: Use the rebuild script
./rebuild_docker_with_dds.sh

# Option B: Manual rebuild
docker build --no-cache -t fl-client-with-dds -f Client/Dockerfile .
docker build --no-cache -t fl-server-with-dds -f Server/Dockerfile .
```

### Step 2: Verify CycloneDDS Installation
```bash
# Test client image
docker run --rm fl-client-with-dds python -c "import cyclonedds; print('✅ CycloneDDS installed in client')"

# Test server image
docker run --rm fl-server-with-dds python -c "import cyclonedds; print('✅ CycloneDDS installed in server')"
```

## Run DDS Experiments

### Single Protocol Test (DDS only)
```bash
python Network_Simulation/run_network_experiments.py \
  --use-case temperature \
  --protocols dds \
  --scenarios excellent \
  --rounds 10
```

### Multiple Scenarios with DDS
```bash
python Network_Simulation/run_network_experiments.py \
  --use-case temperature \
  --protocols dds \
  --scenarios excellent good moderate poor \
  --rounds 100
```

### Compare All Protocols (Including DDS)
```bash
python Network_Simulation/run_network_experiments.py \
  --use-case temperature \
  --protocols mqtt amqp grpc quic dds \
  --scenarios excellent moderate poor \
  --rounds 50
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cyclonedds'"

**Cause:** Using old Docker images without CycloneDDS.

**Fix:**
```bash
# Force rebuild without cache
docker build --no-cache -f Client/Dockerfile -t fl-client-with-dds .
docker build --no-cache -f Server/Dockerfile -t fl-server-with-dds .

# Remove old images
docker images | grep "fl-client\|fl-server" | awk '{print $3}' | xargs docker rmi -f
```

### Issue: Build fails during CycloneDDS compilation

**Check:** Ensure you have enough disk space and memory.

**Alternative:** Use pre-built wheels (Python 3.9 or 3.10 only):
```dockerfile
# In Dockerfile, replace the git clone section with:
RUN pip install --no-cache-dir cyclonedds==0.10.5
```

### Issue: Docker compose fails to start DDS containers

**Check logs:**
```bash
docker-compose -f Docker/docker-compose-temperature.yml logs fl-server-dds-temp
docker-compose -f Docker/docker-compose-temperature.yml logs fl-client-dds-temp-1
```

**Common fixes:**
- Ensure DDS_DOMAIN_ID is consistent across server/clients
- Check network connectivity between containers
- Verify ports aren't blocked

## Technical Details

### CycloneDDS Build in Dockerfile

```dockerfile
# Install CycloneDDS C library from source
RUN git clone --depth 1 --branch 0.10.5 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds && \
    cd /tmp/cyclonedds && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF .. && \
    cmake --build . && \
    cmake --install . && \
    ldconfig && \
    cd / && rm -rf /tmp/cyclonedds
```

**What it does:**
1. Clones CycloneDDS v0.10.5 from GitHub
2. Builds using CMake (installs to /usr/local)
3. Runs ldconfig to register shared libraries
4. Cleans up build files to keep image small

### Python Package Installation

After the C library is built, the Python binding installs via requirements.txt:
```
cyclonedds>=0.10.0
```

This will compile against the installed C library headers.

## Expected Results

After successful rebuild and testing:

```
✅ Client Image: fl-client-with-dds
✅ Server Image: fl-server-with-dds
✅ CycloneDDS Python module available
✅ DDS protocol functional in Docker
```

**Sample output from DDS experiment:**
```
[Protocol: DDS] Starting experiment...
[DDS] Domain ID: 0
[DDS] Topic: FederatedLearning_Temperature
[Server] Waiting for 2 clients...
[Client 1] Connected via DDS
[Client 2] Connected via DDS
[Round 1/10] Training...
```

## Next Steps

1. Wait for rebuild to complete (~10 minutes)
2. Verify installation with test commands above
3. Run DDS experiments with `run_network_experiments.py`
4. Compare DDS performance against other protocols

## Files Modified

- [requirements.txt](requirements.txt#L26) - Enabled cyclonedds
- [Client/Dockerfile](Client/Dockerfile#L8-L19) - Added CycloneDDS build
- [Server/Dockerfile](Server/Dockerfile#L8-L19) - Added CycloneDDS build
- [rebuild_docker_with_dds.sh](rebuild_docker_with_dds.sh) - Rebuild automation script
