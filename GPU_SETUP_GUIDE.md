# GPU Setup Guide for Emotion Recognition Federated Learning

## Overview
All emotion recognition client implementations (AMQP, gRPC, MQTT, DDS, QUIC) have been configured to use GPU acceleration for training and inference.

## GPU Configuration Applied

### TensorFlow GPU Settings
The following GPU optimizations have been added to all client files:

```python
# GPU Configuration - Must be done BEFORE TensorFlow import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow gradual GPU memory growth

# Verify GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Training with GPU
The `train_local_model()` method now uses GPU acceleration:

```python
with tf.device('/GPU:0'):
    history = self.model.fit(
        self.train_generator,
        epochs=epochs,
        validation_data=self.validation_generator,
        verbose=2
    )
```

### Evaluation with GPU
The `evaluate_model()` method now uses GPU acceleration:

```python
with tf.device('/GPU:0'):
    loss, accuracy = self.model.evaluate(
        self.validation_generator, 
        verbose=0
    )
```

## Prerequisites

### 1. NVIDIA GPU Driver
```bash
# Check GPU availability
nvidia-smi

# Output should show your GPU details
```

### 2. CUDA Toolkit
- **Required Version**: 11.x or 12.x
- **Download**: https://developer.nvidia.com/cuda-downloads

### 3. cuDNN Library
- **Required Version**: Compatible with your CUDA version
- **Download**: https://developer.nvidia.com/cudnn

### 4. TensorFlow with GPU Support
```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Or upgrade existing installation
pip install --upgrade tensorflow[and-cuda]
```

### 5. Verify TensorFlow GPU Setup
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Files Updated with GPU Support

### Emotion Recognition Clients:
- `Client/Emotion_Recognition/FL_Client_AMQP.py`
- `Client/Emotion_Recognition/FL_Client_gRPC.py`
- `Client/Emotion_Recognition/FL_Client_MQTT.py`
- `Client/Emotion_Recognition/FL_Client_DDS.py`
- `Client/Emotion_Recognition/FL_Client_QUIC.py`

## Performance Considerations

### GPU Memory Management
- **Dynamic Memory Growth**: Enabled to prevent OOM errors
- **Memory Growth**: TensorFlow will only allocate GPU memory as needed
- **Max Memory**: Adjust if needed with environment variables

### Multi-GPU Setup (Optional)
For systems with multiple GPUs, modify the configuration:

```python
# Use specific GPU (e.g., GPU 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Or use multiple GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

### Batch Size Optimization
Current batch size: `32` (in training_config)

**For GPU Acceleration:**
- Increase batch size to 64-128 for better GPU utilization
- Monitor GPU memory usage with `nvidia-smi -l 1`

## Running Emotion Recognition with GPU

### 1. Start RabbitMQ/Message Broker
```bash
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

### 2. Start FL Server
```bash
export AMQP_HOST=localhost
export AMQP_PORT=5672
cd Server/Emotion_Recognition
python FL_Server_AMQP.py
```

### 3. Start FL Clients (with GPU)
```bash
export CLIENT_ID=0
export AMQP_HOST=localhost
cd Client/Emotion_Recognition
python FL_Client_AMQP.py
```

**Monitor GPU Usage During Training:**
```bash
# In another terminal
nvidia-smi -l 1  # Updates every 1 second
```

## Troubleshooting

### Issue: "No GPUs found. Running on CPU."

**Solution 1: Verify NVIDIA GPU Driver**
```bash
nvidia-smi
# If command not found, install NVIDIA drivers
```

**Solution 2: Check CUDA Installation**
```bash
nvcc --version
# If not found, install CUDA Toolkit
```

**Solution 3: Reinstall TensorFlow GPU**
```bash
pip uninstall tensorflow tensorflow-gpu -y
pip install tensorflow[and-cuda]==2.13.0
```

### Issue: CUDA Out of Memory

**Solution:**
```python
# Reduce batch size in training_config
training_config = {"batch_size": 16, "local_epochs": 20}

# Or set memory growth with less aggressive allocation
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
```

### Issue: GPU Not Fully Utilized

**Solution:**
```bash
# Check GPU utilization
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory --format=csv,noheader -l 1

# Increase batch size or number of workers
```

## Expected Performance Improvements

### Training Time Reduction
- **CPU**: ~60-120 seconds per epoch
- **GPU (NVIDIA)**: ~10-30 seconds per epoch
- **Speedup**: 3-6x faster with GPU

### Inference Speed
- **CPU**: ~100-200ms per batch
- **GPU**: ~20-50ms per batch
- **Speedup**: 2-5x faster with GPU

## Advanced GPU Optimization

### Mixed Precision Training (Optional)
```python
from tensorflow.keras.mixed_precision import Policy

# Use mixed precision for better performance
policy = Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### Data Pipeline Optimization
```python
# Pre-fetch and parallelize data loading
train_generator = train_generator.prefetch(tf.data.AUTOTUNE)
validation_generator = validation_generator.prefetch(tf.data.AUTOTUNE)
```

## Monitoring GPU Performance

### Real-time GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### TensorFlow GPU Profiling
```python
import tensorflow as tf

# Enable TensorFlow profiling
tf.profiler.experimental.start('logdir')
# ... run training ...
tf.profiler.experimental.stop()
```

## References
- [TensorFlow GPU Setup](https://www.tensorflow.org/install/gpu)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
