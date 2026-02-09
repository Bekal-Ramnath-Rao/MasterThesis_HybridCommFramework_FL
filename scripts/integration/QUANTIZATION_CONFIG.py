"""
Quantization Configuration and Setup Guide
============================================

This module provides configuration options for the three quantization strategies:
1. Quantization-Aware Training (QAT)
2. Post-Training Quantization (PTQ)
3. Model Parameter Quantization

Environment Variables:
---------------------
USE_QUANTIZATION=true|false          # Enable/disable quantization (default: true)
QUANTIZATION_STRATEGY=qat|ptq|param  # Choose strategy (default: parameter_quantization)
QUANTIZATION_BITS=8|16|32            # Quantization bits (default: 8)
QUANTIZATION_SYMMETRIC=true|false    # Symmetric quantization (default: true)
QUANTIZATION_PER_CHANNEL=true|false  # Per-channel quantization (default: false)

Quantization Strategies:
------------------------

1. QUANTIZATION-AWARE TRAINING (QAT)
   - Best for: Highest accuracy retention
   - Trade-off: Requires model retraining
   - Use when: You can afford training time and want best results
   - Set: QUANTIZATION_STRATEGY=quantization_aware_training
   
   Example:
   ```bash
   $env:USE_QUANTIZATION="true"
   $env:QUANTIZATION_STRATEGY="quantization_aware_training"
   $env:QUANTIZATION_BITS="8"
   ```

2. POST-TRAINING QUANTIZATION (PTQ)
   - Best for: Quick deployment without retraining
   - Trade-off: Slight accuracy loss possible
   - Use when: Model is already trained and you want fast compression
   - Set: QUANTIZATION_STRATEGY=post_training_quantization
   
   Example:
   ```bash
   $env:USE_QUANTIZATION="true"
   $env:QUANTIZATION_STRATEGY="post_training_quantization"
   $env:QUANTIZATION_BITS="8"
   ```

3. MODEL PARAMETER QUANTIZATION (Default)
   - Best for: Balance of speed and accuracy
   - Trade-off: Good compression, minimal setup
   - Use when: You want simple, effective compression
   - Set: QUANTIZATION_STRATEGY=parameter_quantization (or leave unset)
   
   Example:
   ```bash
   $env:USE_QUANTIZATION="true"
   $env:QUANTIZATION_STRATEGY="parameter_quantization"
   $env:QUANTIZATION_BITS="8"
   $env:QUANTIZATION_SYMMETRIC="true"
   $env:QUANTIZATION_PER_CHANNEL="false"
   ```

Quantization Bits:
-----------------
- 8-bit: Best compression (4x), good accuracy
- 16-bit: Moderate compression (2x), better accuracy
- 32-bit: Minimal compression, highest accuracy

Symmetric vs Asymmetric:
------------------------
- Symmetric (default): Simpler, faster, good for most cases
- Asymmetric: More flexible, better for asymmetric distributions

Per-Channel Quantization:
-------------------------
- False (default): Quantize entire tensor with same parameters
- True: Quantize each channel separately for better accuracy

Quick Start Examples:
--------------------

# Example 1: Use 8-bit parameter quantization (recommended default)
$env:USE_QUANTIZATION="true"
$env:QUANTIZATION_STRATEGY="parameter_quantization"
$env:QUANTIZATION_BITS="8"

# Example 2: Use QAT for best accuracy
$env:USE_QUANTIZATION="true"
$env:QUANTIZATION_STRATEGY="quantization_aware_training"
$env:QUANTIZATION_BITS="8"

# Example 3: Use PTQ for fastest setup
$env:USE_QUANTIZATION="true"
$env:QUANTIZATION_STRATEGY="post_training_quantization"

# Example 4: Disable quantization
$env:USE_QUANTIZATION="false"

# Example 5: High precision quantization
$env:USE_QUANTIZATION="true"
$env:QUANTIZATION_BITS="16"
$env:QUANTIZATION_PER_CHANNEL="true"

Running with Quantization:
--------------------------

For Emotion Recognition:
```powershell
# Set quantization config
$env:USE_QUANTIZATION="true"
$env:QUANTIZATION_STRATEGY="parameter_quantization"
$env:QUANTIZATION_BITS="8"
$env:USE_CASE="emotion"

# Start server
python Server/Emotion_Recognition/FL_Server_MQTT.py

# In another terminal, start clients
$env:CLIENT_ID="0"
python Client/Emotion_Recognition/FL_Client_MQTT.py
```

For Mental State Recognition:
```powershell
$env:USE_QUANTIZATION="true"
$env:USE_CASE="mentalstate"
python Server/MentalState_Recognition/FL_Server_MQTT.py
```

For Temperature Regulation:
```powershell
$env:USE_QUANTIZATION="true"
$env:USE_CASE="temperature"
python Server/Temperature_Regulation/FL_Server_MQTT.py
```

Testing Different Strategies:
-----------------------------

# Test all three strategies
$strategies = @("parameter_quantization", "post_training_quantization", "quantization_aware_training")
foreach ($strategy in $strategies) {
    Write-Host "Testing $strategy..."
    $env:QUANTIZATION_STRATEGY = $strategy
    # Run your experiment here
}

Performance Monitoring:
----------------------

The quantization module automatically prints compression statistics:
- Original size (MB)
- Compressed size (MB)
- Compression ratio
- Size reduction percentage

Monitor these metrics to evaluate different strategies.

Troubleshooting:
---------------

1. If QAT fails:
   - Install tensorflow-model-optimization: pip install tensorflow-model-optimization
   - Or fall back to parameter quantization

2. If PTQ fails:
   - System will automatically fall back to parameter quantization
   - Check TensorFlow version compatibility

3. Import errors:
   - Ensure quantization_client.py is in Client/Compression_Technique/
   - Ensure path is added in client/server files

4. Memory issues with large models:
   - Reduce QUANTIZATION_BITS to 8
   - Enable PER_CHANNEL quantization for better memory usage

Best Practices:
--------------

1. Start with parameter quantization (default) - it works well for most cases
2. Use 8-bit quantization for best compression
3. Enable symmetric quantization for faster computation
4. Use per-channel for larger models or when accuracy is critical
5. Monitor compression stats and model accuracy to find optimal settings
6. Test different strategies on validation set before production

Integration Checklist:
---------------------

✓ Client-side:
  - quantization.py in Client/Compression_Technique/
  - Import and initialize Quantization in FL_Client_*.py
  - Compress weights before sending
  - Decompress received global model

✓ Server-side:
  - quantization.py in Server/Compression_Technique/
  - Import ServerQuantizationHandler in FL_Server_*.py
  - Decompress client updates
  - Aggregate decompressed weights
  - Optionally compress global model before distribution

✓ Configuration:
  - Set environment variables
  - Test with different strategies
  - Monitor compression metrics

For more details, see the quantization.py implementation.
"""

import os
from typing import Dict, Any


class QuickConfig:
    """Quick configuration presets for quantization"""
    
    @staticmethod
    def use_default():
        """8-bit parameter quantization (recommended)"""
        return {
            "USE_QUANTIZATION": "true",
            "QUANTIZATION_STRATEGY": "parameter_quantization",
            "QUANTIZATION_BITS": "8",
            "QUANTIZATION_SYMMETRIC": "true",
            "QUANTIZATION_PER_CHANNEL": "false"
        }
    
    @staticmethod
    def use_qat():
        """Quantization-Aware Training for best accuracy"""
        return {
            "USE_QUANTIZATION": "true",
            "QUANTIZATION_STRATEGY": "quantization_aware_training",
            "QUANTIZATION_BITS": "8",
            "QUANTIZATION_SYMMETRIC": "true",
            "QUANTIZATION_PER_CHANNEL": "false"
        }
    
    @staticmethod
    def use_ptq():
        """Post-Training Quantization for quick deployment"""
        return {
            "USE_QUANTIZATION": "true",
            "QUANTIZATION_STRATEGY": "post_training_quantization",
            "QUANTIZATION_BITS": "8",
            "QUANTIZATION_SYMMETRIC": "true",
            "QUANTIZATION_PER_CHANNEL": "false"
        }
    
    @staticmethod
    def use_high_precision():
        """16-bit quantization with per-channel for accuracy"""
        return {
            "USE_QUANTIZATION": "true",
            "QUANTIZATION_STRATEGY": "parameter_quantization",
            "QUANTIZATION_BITS": "16",
            "QUANTIZATION_SYMMETRIC": "true",
            "QUANTIZATION_PER_CHANNEL": "true"
        }
    
    @staticmethod
    def use_max_compression():
        """8-bit for maximum compression"""
        return {
            "USE_QUANTIZATION": "true",
            "QUANTIZATION_STRATEGY": "parameter_quantization",
            "QUANTIZATION_BITS": "8",
            "QUANTIZATION_SYMMETRIC": "true",
            "QUANTIZATION_PER_CHANNEL": "false"
        }
    
    @staticmethod
    def disable_quantization():
        """Disable quantization"""
        return {
            "USE_QUANTIZATION": "false"
        }
    
    @staticmethod
    def apply_config(config: Dict[str, str]):
        """Apply configuration to environment"""
        for key, value in config.items():
            os.environ[key] = value
        print(f"Applied quantization config: {config}")


def print_current_config():
    """Print current quantization configuration"""
    print("\n" + "="*70)
    print("Current Quantization Configuration")
    print("="*70)
    print(f"USE_QUANTIZATION:        {os.getenv('USE_QUANTIZATION', 'true')}")
    print(f"QUANTIZATION_STRATEGY:   {os.getenv('QUANTIZATION_STRATEGY', 'parameter_quantization')}")
    print(f"QUANTIZATION_BITS:       {os.getenv('QUANTIZATION_BITS', '8')}")
    print(f"QUANTIZATION_SYMMETRIC:  {os.getenv('QUANTIZATION_SYMMETRIC', 'true')}")
    print(f"QUANTIZATION_PER_CHANNEL: {os.getenv('QUANTIZATION_PER_CHANNEL', 'false')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    print(__doc__)
    print_current_config()
