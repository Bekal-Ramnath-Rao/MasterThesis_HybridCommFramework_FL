# GPU-Enabled Network Experiments - Quick Start Guide

## ✅ Setup Complete

All three use cases now have GPU support enabled:
- ✅ **Emotion Recognition** - docker-compose-emotion.gpu.yml
- ✅ **Mental State Recognition** - docker-compose-mentalstate.gpu.yml  
- ✅ **Temperature Regulation** - docker-compose-temperature.gpu.yml

## 🚀 Main Command (All Use Cases, All Scenarios, All Protocols with GPU)

```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL

# Run complete experiments for all three use cases with GPU
for use_case in emotion mentalstate temperature; do
    echo "=========================================="
    echo "Starting GPU experiments for: $use_case"
    echo "=========================================="
    python3 Network_Simulation/run_network_experiments.py --use-case $use_case --enable-gpu --rounds 10
    echo "Completed: $use_case"
done
```

This will run:
- **3 use cases** (emotion, mentalstate, temperature)
- **5 protocols** (MQTT, AMQP, gRPC, QUIC, DDS)
- **9 network scenarios** (excellent, good, moderate, poor, very_poor, satellite, congested_light, congested_moderate, congested_heavy)
- **Total: 135 experiments**

## Individual Use Case Commands

```bash
# Emotion Recognition with GPU
python3 Network_Simulation/run_network_experiments.py --use-case emotion --enable-gpu --rounds 10

# Mental State Recognition with GPU
python3 Network_Simulation/run_network_experiments.py --use-case mentalstate --enable-gpu --rounds 10

# Temperature Regulation with GPU
python3 Network_Simulation/run_network_experiments.py --use-case temperature --enable-gpu --rounds 10
```

## Test First (Recommended)

Quick test to verify GPU is working:

```bash
# Single quick test (2 rounds)
python3 Network_Simulation/run_network_experiments.py --use-case emotion --enable-gpu --single --protocol mqtt --scenario excellent --rounds 2

# Monitor GPU during test (in another terminal)
watch -n 1 nvidia-smi
```

## All Commands

See [Network_Simulation/commands.txt](Network_Simulation/commands.txt) for comprehensive list of all available commands including:
- Specific protocol testing
- Specific scenario testing  
- Congestion experiments
- Quantization + GPU
- Single experiments
- Troubleshooting commands

## GPU Files Created

1. `Docker/docker-compose-emotion.gpu.yml`
2. `Docker/docker-compose-mentalstate.gpu.yml`
3. `Docker/docker-compose-temperature.gpu.yml`
4. `Network_Simulation/run_network_experiments.py` (updated with `--enable-gpu` flag)
5. `Network_Simulation/commands.txt` (GPU commands reference)
6. `GPU_DOCKER_SETUP.md` (GPU configuration documentation)

## Results

Results will be saved in:
```
experiment_results/{use_case}_{YYYYMMDD_HHMMSS}/
```

## ⚠️ Important Reminders

1. **Always use native Docker (NOT Docker Desktop):**
   ```bash
   docker context use default
   docker info | grep nvidia  # Should show nvidia runtime
   ```

2. **Both RTX 3080 GPUs are available** to all containers simultaneously

3. **Estimated time for full experiments:** 6-12 hours depending on rounds

4. **Monitor GPU memory:**
   ```bash
   watch -n 1 nvidia-smi
   ```

## Verify Setup

```bash
# Check Docker context
docker context ls  # Should show "default *"

# Check nvidia runtime is available
docker info | grep Runtimes  # Should include: nvidia

# Test GPU container
docker run --rm --runtime=nvidia nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```
