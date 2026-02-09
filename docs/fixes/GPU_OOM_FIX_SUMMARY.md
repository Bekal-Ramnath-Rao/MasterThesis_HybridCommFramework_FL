# GPU Out-of-Memory (OOM) Issue - Resolution Guide

## Issue Summary
Client 1 encountered a CUDA Out-of-Memory error while training with the moderate network scenario:
```
OOM when allocating tensor with shape[64,32,46,46] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
```

## Root Causes Identified

### 1. **GPU Memory Contention**
   - **Problem**: The original docker-compose file used `count: all`, allowing both clients to access ALL GPUs simultaneously
   - **Impact**: Both client containers could run on the same GPU (GPU:0), competing for the 10GB VRAM
   - **Result**: When both clients tried to allocate large tensors, GPU memory was exhausted

### 2. **Large Batch Sizes**
   - **Training batch size**: 32 (each tensor shape: [32, channels, height, width])
   - **Data augmentation batch size**: 64 (each tensor shape: [64, channels, height, width])
   - **Memory requirement**: Each batch * model parameters consumed significant GPU memory
   - **During error**: Trying to allocate [64,32,46,46] float tensor = ~200MB per batch

### 3. **Unbounded GPU Memory Growth**
   - TensorFlow was set to grow GPU memory allocation dynamically
   - No upper limit was set, allowing one container to consume all available VRAM
   - Other containers couldn't allocate memory when needed

## Solutions Implemented

### 1. **Batch Size Reduction** ✓
**File Modified**: `Client/Emotion_Recognition/FL_Client_MQTT.py`

Changed batch sizes from 32→16 and 64→16:
```python
# Before
self.training_config = {"batch_size": 32, "local_epochs": 20}  # Line 74
train_generator batch_size=64  # Line ~485
validation_generator batch_size=64  # Line ~495

# After
self.training_config = {"batch_size": 16, "local_epochs": 20}
train_generator batch_size=16
validation_generator batch_size=16
```

**Benefit**: Reduces peak memory usage by ~50% per batch

### 2. **GPU Device Isolation** ✓
**New File Created**: `Docker/docker-compose-emotion.gpu-isolated.yml`

Configuration changes:
```yaml
# OLD: GPU shared between all containers
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
          driver: nvidia
          count: all  # ❌ Both clients compete for same GPU

# NEW: GPU isolated per client
fl-client-mqtt-emotion-1:
  environment:
    - CUDA_VISIBLE_DEVICES=0  # Exclusive access to GPU 0
    - GPU_DEVICE_ID=0
  deploy:
    devices:
      - device_ids: ['0']  # Only GPU 0 visible

fl-client-mqtt-emotion-2:
  environment:
    - CUDA_VISIBLE_DEVICES=1  # Exclusive access to GPU 1
    - GPU_DEVICE_ID=1
  deploy:
    devices:
      - device_ids: ['1']  # Only GPU 1 visible
```

**Benefit**: Each client has exclusive access to one GPU, eliminating contention

### 3. **GPU Memory Limits** ✓
**File Modified**: `Client/Emotion_Recognition/FL_Client_MQTT.py`

Added TensorFlow memory management:
```python
# GPU Configuration - Must be done BEFORE TensorFlow import
os.environ["GPU_DEVICE_ID"] = os.environ.get("GPU_DEVICE_ID", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device  # Isolate to specific GPU
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # Dedicated thread pool

# Set GPU memory limit per process
for gpu in gpus:
    tf.config.set_logical_device_configuration(
        gpu,
        [tf.config.LogicalDeviceConfiguration(memory_limit=8000)]  # 8GB per GPU
    )
```

**Benefit**: Prevents one container from consuming all GPU memory (10GB RTX 3080 reserves 8GB per client, leaving 2GB buffer)

## Usage Instructions

### To Use the New GPU-Isolated Docker Compose:

```bash
# Replace the old GPU compose with the new isolated version for emotion experiments
docker-compose -f Docker/docker-compose-emotion.gpu-isolated.yml up

# Or update the run_network_experiments.py to use this file for GPU experiments
# (Update will be done in next phase)
```

### For Other Use Cases (Mental State, Temperature):

Same configuration pattern should be applied:
1. Reduce batch size from 32→16 (and data aug batch from 64→16)
2. Create `docker-compose-{usecase}.gpu-isolated.yml` with device isolation
3. Set environment variables:
   - `CUDA_VISIBLE_DEVICES` (0 for client 1, 1 for client 2)
   - `GPU_DEVICE_ID` (0 or 1)
   - `TF_GPU_MEMORY_FRACTION=0.9`

## Memory Allocation Summary

**With 2x RTX 3080 (10GB each):**

| Component | GPU 0 | GPU 1 |
|-----------|-------|-------|
| Client 1 limit | 8GB | - |
| Client 2 limit | - | 8GB |
| Server (if used) | 4GB (40%) | 4GB (40%) |
| TensorFlow buffer | 2GB | 2GB |
| **Total** | 10GB | 10GB |

## Expected Improvements

✅ **Eliminates GPU contention** - Each client has dedicated GPU
✅ **Reduces OOM errors** - Batch size reduced + memory limits enforced
✅ **Enables stable training** - Prevents one container from starving others
✅ **Better experiment reproducibility** - Consistent memory availability per run

## Verification Steps

1. **Rebuild Docker images** (includes batch size changes):
   ```bash
   docker-compose -f Docker/docker-compose-emotion.gpu-isolated.yml build
   ```

2. **Run test experiment with moderate scenario**:
   ```bash
   python3 Network_Simulation/run_network_experiments.py --use-case emotion --enable-gpu --single --protocol mqtt --scenario moderate --rounds 3
   ```

3. **Monitor GPU memory** during training:
   ```bash
   nvidia-smi -l 1  # Update every 1 second
   ```

   Expected output:
   ```
   GPU 0: Client 1 using ~5-7GB
   GPU 1: Client 2 using ~5-7GB
   Both stable, no OOM errors
   ```

## Files Modified

1. ✅ `Client/Emotion_Recognition/FL_Client_MQTT.py`
   - Reduced batch sizes (32→16, 64→16)
   - Added GPU memory limit (8GB per client)
   - Added dynamic GPU device selection

2. ✅ `Docker/docker-compose-emotion.gpu-isolated.yml` (NEW)
   - GPU device isolation per client
   - Memory fraction limits for server
   - Proper `CUDA_VISIBLE_DEVICES` configuration

## Next Steps

1. Rebuild Docker images with batch size changes
2. Run test experiment with the moderate scenario
3. Monitor for OOM errors
4. Apply same pattern to `FL_Client_AMQP.py`, `FL_Client_gRPC.py`, etc.
5. Apply same pattern to other use cases (mentalstate, temperature)
6. Update `run_network_experiments.py` to use gpu-isolated compose files

