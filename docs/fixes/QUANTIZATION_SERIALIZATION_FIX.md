# Quantization Serialization Fix

## Issue
When quantization was enabled, the **QUIC protocol** crashed with:
```
TypeError: Object of type ndarray is not JSON serializable
```

The server was trying to send quantized model weights (which are numpy arrays inside a dictionary) directly through JSON, which doesn't support numpy arrays.

## Root Cause
The `compress_global_model()` method returns a dictionary containing:
- `compressed_data`: List of quantized numpy arrays
- `quantization_params`: Dictionary with scale, zero_point, etc.

This dictionary contains numpy arrays that cannot be directly serialized to JSON.

## Solution
Different protocols handle binary data differently:

### ✅ MQTT (Already Correct)
Uses **pickle + base64** encoding:
```python
serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
```

### ✅ AMQP (Already Correct)  
Uses **pickle + base64** encoding:
```python
serialized = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
```

### ✅ gRPC (Already Correct)
Uses **pickle.dumps** directly (protobuf bytes field):
```python
serialized_weights = pickle.dumps(compressed_data)
```

### ❌ QUIC (FIXED)
**Problem**: Was sending raw dictionary with numpy arrays to JSON
**Solution**: Added pickle + base64 encoding like MQTT:
```python
weights_data = base64.b64encode(pickle.dumps(compressed_data)).decode('utf-8')
```

**Fixed in 2 locations:**
1. `distribute_initial_model()` - Initial model distribution
2. `aggregate_and_distribute()` - Round updates

### ❌ DDS (FIXED)
**Problem**: Was assigning raw dictionary directly to DDS sequence<octet>
**Solution**: Added pickle + list conversion:
```python
serialized_weights = list(pickle.dumps(compressed_data))
```

**Fixed in 2 locations:**
1. `distribute_initial_model()` - Initial model distribution  
2. `aggregate_model_updates()` - Round updates

## Client-Side Changes

### QUIC Client (Fixed)
Updated to deserialize base64+pickle encoded data:
```python
if 'quantized_data' in message:
    compressed_data = pickle.loads(base64.b64decode(message['quantized_data']))
    weights = self.quantizer.decompress(compressed_data)
```

### DDS Client (No Change Needed)
Already correctly deserializes with `pickle.loads(bytes(serialized_weights))`

## Files Modified
1. **Server/Emotion_Recognition/FL_Server_QUIC.py** (2 locations)
   - Line ~351: Initial model distribution
   - Line ~410: Round updates

2. **Server/Emotion_Recognition/FL_Server_DDS.py** (2 locations)
   - Line ~407: Initial model distribution
   - Line ~553: Round updates

3. **Client/Emotion_Recognition/FL_Client_QUIC.py** (1 location)
   - Line ~229: Global model deserialization

## Testing
To verify the fix works:
```bash
cd Docker
USE_QUANTIZATION=true docker-compose -f docker-compose-emotion.yml up fl-server-quic-emotion fl-client-quic-emotion-1 fl-client-quic-emotion-2
```

Expected output:
- ✓ Compressed initial global model - Ratio: 2.00x (for 16-bit quantization)
- ✓ Client 1: Received and decompressed quantized global model
- ✓ Client 2: Received and decompressed quantized global model
- NO JSON serialization errors

## Summary
All 5 protocols now correctly handle quantized model transmission:
- **MQTT**: pickle + base64 ✅
- **AMQP**: pickle + base64 ✅
- **gRPC**: pickle (protobuf bytes) ✅
- **QUIC**: pickle + base64 ✅ (FIXED)
- **DDS**: pickle + list conversion ✅ (FIXED)

Quantization is now fully functional across all communication protocols!
