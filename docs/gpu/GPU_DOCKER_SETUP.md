# GPU Docker Configuration Guide

## ✅ Successfully Configured

Your system now has GPU support working in Docker containers!

## System Configuration

### Hardware
- **GPUs**: 2x NVIDIA GeForce RTX 3080 (10GB each)
- **Driver**: NVIDIA 535.288.01
- **CUDA Version**: 12.2

### Docker Setup
- **Docker Engine**: CE (Community Edition) 29.1.5 - **Use this, NOT Docker Desktop**
- **NVIDIA Container Toolkit**: 1.18.1-1
- **nvidia-docker2**: 2.13.0-1
- **Runtime**: `nvidia` runtime is registered in native Docker

## ⚠️ CRITICAL: Use Native Docker, Not Docker Desktop

**Docker Desktop runs containers in a VM which breaks GPU passthrough!**

### Always Use Native Docker Context
```bash
# Check current context (must be "default" for GPU support)
docker context ls

# If not on default, switch to it:
docker context use default

# Verify nvidia runtime is available
docker info | grep Runtimes
# Should show: Runtimes: io.containerd.runc.v2 nvidia runc
```

### Docker Desktop is Disabled
- Auto-start has been disabled via `systemctl --user disable docker-desktop`
- If Docker Desktop starts accidentally, stop it:
  ```bash
  systemctl --user stop docker-desktop
  docker context use default
  ```

## Running GPU-Enabled Federated Learning

### Using GPU Overlay Files

For **Emotion Recognition** experiments with GPU:
```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL/Docker

# MQTT with GPU
docker compose -f docker-compose-emotion.yml -f docker-compose-emotion.gpu.yml up

# Or run individual service
docker compose -f docker-compose-emotion.yml -f docker-compose-emotion.gpu.yml run --rm fl-server-mqtt-emotion python Server/Emotion_Recognition/FL_Server_MQTT.py
```

### Available GPU Overlay Files
- `docker-compose-emotion.gpu.yml` - Uses nvidia runtime with `driver: nvidia` (PREFERRED)
- `docker-compose-emotion.gpu-alt.yml` - Fallback using privileged mode (not needed now)

### Test GPU Access

**Quick GPU test:**
```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

**Test TensorFlow GPU detection:**
```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all tensorflow/tensorflow:2.13.0-gpu \
  python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Expected output:** 2 GPUs detected

## GPU Configuration in docker-compose-emotion.gpu.yml

Each service has GPU reservation:
```yaml
services:
  fl-server-mqtt-emotion:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              driver: nvidia
              count: all  # Uses all available GPUs
```

To limit GPU usage per service:
- Change `count: all` to `count: 1` for single GPU
- Add `device_ids: ['0']` or `device_ids: ['1']` to specify which GPU

## Troubleshooting

### If GPUs not detected:
1. **Check you're using native Docker, not Desktop:**
   ```bash
   docker context ls  # Should show "default *"
   docker info | grep Runtimes  # Must include "nvidia"
   ```

2. **Verify NVIDIA driver on host:**
   ```bash
   nvidia-smi  # Should show both RTX 3080 GPUs
   ```

3. **Check nvidia runtime registration:**
   ```bash
   cat /etc/docker/daemon.json  # Should have nvidia runtime entry
   sudo systemctl restart docker
   docker info | grep Runtimes
   ```

### Port Conflicts
MQTT broker uses ports **31883** (MQTT) and **39001** (WebSocket) to avoid conflicts with host Mosquitto on port 1883.

## Performance Notes

- **GPU Memory**: Each RTX 3080 has 10GB VRAM
- **Concurrent Training**: Can run multiple FL clients simultaneously across 2 GPUs
- **Memory Management**: TensorFlow uses dynamic memory allocation by default
- To limit GPU memory per container:
  ```python
  # Add to training code:
  gpus = tf.config.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      # Or set hard limit:
      # tf.config.set_logical_device_configuration(
      #     gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # MB
      # )
  ```

## Architecture Notes

- **Base Image**: tensorflow/tensorflow:2.13.0-gpu (includes CUDA 11.8)
- **Host Driver**: 535.288.01 (compatible with CUDA 12.2)
- **Compatibility**: Driver 535.x supports CUDA 11.8 (container) and CUDA 12.2 (host)

## Additional GPU Commands

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Check GPU processes
nvidia-smi pmon

# GPU utilization stats
nvidia-smi dmon
```

## Migration Notes

**From Docker Desktop to Native Docker:**
- ✅ Switched context from `desktop-linux` to `default`
- ✅ Docker Desktop auto-start disabled
- ✅ Verified nvidia runtime in native daemon
- ✅ Tested GPU passthrough successfully

## References

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
