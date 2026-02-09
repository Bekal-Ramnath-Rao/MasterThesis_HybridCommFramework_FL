# Quantization Quick Reference

## 🚀 Quick Start

```powershell
# Enable 8-bit quantization (4x compression)
$env:USE_QUANTIZATION="true"
$env:QUANTIZATION_BITS="8"

# Run any server/client
python Server/Emotion_Recognition/FL_Server_MQTT.py
python Client/Emotion_Recognition/FL_Client_MQTT.py
```

## ✅ What's Integrated

**ALL 30 FILES:**
- ✅ MQTT (6 files): Emotion, MentalState, Temperature
- ✅ AMQP (6 files): Emotion, MentalState, Temperature
- ✅ gRPC (6 files): Emotion, MentalState, Temperature
- ✅ QUIC (6 files): Emotion, MentalState, Temperature
- ✅ DDS (6 files): Emotion, MentalState, Temperature

## 📊 Compression Ratios

| Bits | Ratio | Original | Compressed | Use Case |
|------|-------|----------|------------|----------|
| 8 | 4x | 25 MB | 6.25 MB | Maximum savings |
| 16 | 2x | 25 MB | 12.5 MB | Balanced |
| 32 | 1x | 25 MB | 25 MB | No compression |

## ⚙️ Configuration

```powershell
# Strategy
$env:QUANTIZATION_STRATEGY="parameter_quantization"  # default, fastest
$env:QUANTIZATION_STRATEGY="qat"                    # training-aware
$env:QUANTIZATION_STRATEGY="ptq"                    # post-training

# Bit depth
$env:QUANTIZATION_BITS="8"   # 4x compression
$env:QUANTIZATION_BITS="16"  # 2x compression
$env:QUANTIZATION_BITS="32"  # no compression

# Mode
$env:QUANTIZATION_SYMMETRIC="true"    # symmetric quantization
$env:QUANTIZATION_PER_CHANNEL="true"  # per-channel granularity

# Disable
$env:USE_QUANTIZATION="false"
```

## 📁 Files

**Core:**
- `Client/Compression_Technique/quantization_client.py`
- `Server/Compression_Technique/quantization_server.py`

**Docs:**
- `README_QUANTIZATION.md` - Full guide
- `QUANTIZATION_STATUS.md` - Current status
- `QUANTIZATION_COMPLETE.md` - Implementation summary
- `QUANTIZATION_CONFIG.py` - Configuration reference

## 🔍 Verify It's Working

**Look for these logs:**

```
Client 0: Quantization enabled
Client 0: Compressed weights - Ratio: 4.00x, Size: 6.34MB
Server: Quantization enabled
Server: Compressed global model - Ratio: 4.00x
Received and decompressed update from client 0
```

## 📞 Need Help?

1. Check `README_QUANTIZATION.md` for details
2. See `QUANTIZATION_STATUS.md` for status
3. Review `QUANTIZATION_COMPLETE.md` for full summary

---

**Status:** ✅ COMPLETE  
**Coverage:** 30/30 files (100%)  
**Protocols:** MQTT, AMQP, gRPC, QUIC, DDS  
**Ready:** YES ✅
