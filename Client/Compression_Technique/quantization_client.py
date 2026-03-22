"""
Quantization Compression Techniques for Federated Learning
Implements three quantization strategies:
1. Quantization-Aware Training (QAT) - Simulates quantization during training
2. Post-Training Quantization (PTQ) - Quantizes trained model weights
3. Model Parameter Quantization - Direct weight/gradient quantization

Supported bit widths: 4, 8, 16, 32
- 4-bit: Uses nibble packing (2 values per byte) for 2x size reduction vs 8-bit
- 8-bit: Standard uint8 quantization
- 16-bit: uint16 quantization for higher precision
- 32-bit: int32 quantization for maximum precision
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Any
import pickle
import json
import os
from enum import Enum

try:
    from pruning_client import ModelPruning, PruningConfig
    PRUNING_MODULE_AVAILABLE = True
except Exception:
    ModelPruning = None
    PruningConfig = None
    PRUNING_MODULE_AVAILABLE = False


class QuantizationStrategy(Enum):
    """Quantization strategy types"""
    QAT = "quantization_aware_training"  # Quantization-Aware Training
    PTQ = "post_training_quantization"   # Post-Training Quantization
    PARAM = "parameter_quantization"     # Direct parameter quantization


class QuantizationConfig:
    """Configuration for quantization"""
    def __init__(
        self,
        strategy: str = "parameter_quantization",
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = False,
        use_gradient_quantization: bool = True
    ):
        self.strategy = strategy
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.use_gradient_quantization = use_gradient_quantization
        
        # Load from environment if available
        self.strategy = os.getenv("QUANTIZATION_STRATEGY", strategy)
        self.bits = int(os.getenv("QUANTIZATION_BITS", str(bits)))
        self.symmetric = os.getenv("QUANTIZATION_SYMMETRIC", str(symmetric)).lower() == "true"
        self.per_channel = os.getenv("QUANTIZATION_PER_CHANNEL", str(per_channel)).lower() == "true"
        
    def to_dict(self):
        return {
            "strategy": self.strategy,
            "bits": self.bits,
            "symmetric": self.symmetric,
            "per_channel": self.per_channel,
            "use_gradient_quantization": self.use_gradient_quantization
        }


class Quantization:
    """
    Comprehensive Quantization Implementation
    Supports QAT, PTQ, and Parameter Quantization
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self.qat_model = None  # For QAT strategy
        self.quantization_params = {}  # Store quantization parameters
        self.pruner = None
        self.pruning_step = 0
        
        print(f"\n{'='*70}")
        print(f"Quantization Module Initialized")
        print(f"{'='*70}")
        print(f"Strategy: {self.config.strategy}")
        print(f"Bits: {self.config.bits}")
        print(f"Symmetric: {self.config.symmetric}")
        print(f"Per-channel: {self.config.per_channel}")
        print(f"{'='*70}\n")

        use_pruning = os.getenv("USE_PRUNING", "false").lower() in ("true", "1", "yes", "y")
        if use_pruning and PRUNING_MODULE_AVAILABLE:
            self.pruner = ModelPruning(PruningConfig())
            print("Quantization module: pruning enabled (prune -> quantize pipeline)")
        elif use_pruning and not PRUNING_MODULE_AVAILABLE:
            print("Quantization module: pruning requested but pruning module unavailable")

    def _maybe_prune_before_quantization(self, data: Any, data_type: str) -> Any:
        """Apply optional pruning before quantization to support global prune->quantize flow."""
        if self.pruner is None or data_type not in ("weights", "gradients"):
            return data

        weights = None
        if isinstance(data, tf.keras.Model):
            weights = data.get_weights()
        elif isinstance(data, list):
            weights = data

        if weights is None:
            return data

        pruned_weights = self.pruner.prune_weights(weights, step=self.pruning_step)
        self.pruning_step += 1

        try:
            pruning_stats = self.pruner.get_pruning_statistics(pruned_weights)
            print(
                f"Pruning before quantization - "
                f"Sparsity: {pruning_stats['overall_sparsity']:.2%}, "
                f"Compression: {pruning_stats['compression_ratio']:.2f}x"
            )
        except Exception:
            pass

        return pruned_weights
    
    # ==================== Strategy 1: Quantization-Aware Training (QAT) ====================
    
    def prepare_qat_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Prepare model for Quantization-Aware Training
        Inserts fake quantization nodes into the model
        """
        try:
            import tensorflow_model_optimization as tfmot
            
            # Apply quantization-aware training
            quantize_model = tfmot.quantization.keras.quantize_model
            
            # Clone model to avoid modifying original
            qat_model = tf.keras.models.clone_model(model)
            qat_model.set_weights(model.get_weights())
            
            # Apply quantization
            self.qat_model = quantize_model(qat_model)
            
            print(f"✓ Model prepared for Quantization-Aware Training")
            return self.qat_model
            
        except ImportError:
            print("Warning: tensorflow_model_optimization not available. Falling back to parameter quantization.")
            print("Install with: pip install tensorflow-model-optimization")
            return model
        except Exception as e:
            print(f"Warning: QAT preparation failed: {e}. Using original model.")
            return model
    
    def train_with_qat(
        self,
        model: tf.keras.Model,
        train_data,
        validation_data=None,
        epochs: int = 10,
        batch_size: int = 32
    ) -> tf.keras.Model:
        """
        Train model with quantization awareness
        """
        if self.qat_model is None:
            self.qat_model = self.prepare_qat_model(model)
        
        # Compile QAT model
        self.qat_model.compile(
            optimizer=model.optimizer,
            loss=model.loss,
            metrics=model.metrics
        )
        
        # Train with fake quantization
        history = self.qat_model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        return self.qat_model
    
    def export_qat_model(self) -> tf.keras.Model:
        """Export quantized model from QAT"""
        if self.qat_model is None:
            raise ValueError("No QAT model available. Train with QAT first.")
        
        try:
            import tensorflow_model_optimization as tfmot
            
            # Convert to actual quantized model
            converter = tf.lite.TFLiteConverter.from_keras_model(self.qat_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            quantized_tflite_model = converter.convert()
            
            return quantized_tflite_model
        except Exception as e:
            print(f"Warning: QAT export failed: {e}")
            return self.qat_model.get_weights()
    
    # ==================== Strategy 2: Post-Training Quantization (PTQ) ====================
    
    def post_training_quantize_model(
        self,
        model: tf.keras.Model,
        representative_dataset=None
    ) -> bytes:
        """
        Apply post-training quantization to a trained model
        Returns quantized TFLite model
        """
        try:
            # Convert model to TFLite with quantization
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Full integer quantization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            if representative_dataset is not None:
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            quantized_model = converter.convert()
            
            print(f"✓ Post-training quantization completed")
            print(f"  Original model size: ~{self._estimate_model_size(model.get_weights())} MB")
            print(f"  Quantized model size: {len(quantized_model) / (1024*1024):.2f} MB")
            
            return quantized_model
            
        except Exception as e:
            print(f"Warning: PTQ failed: {e}. Falling back to parameter quantization.")
            return None
    
    def _estimate_model_size(self, weights: List[np.ndarray]) -> float:
        """Estimate model size in MB"""
        total_bytes = sum(w.nbytes for w in weights)
        return total_bytes / (1024 * 1024)
    
    # ==================== Strategy 3: Model Parameter Quantization ====================
    
    def quantize_weights(
        self,
        weights: List[np.ndarray],
        store_params: bool = True
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Quantize model weights to lower precision
        
        Args:
            weights: List of weight arrays
            store_params: Whether to store quantization parameters
            
        Returns:
            Tuple of (quantized_weights, quantization_params)
        """
        quantized_weights = []
        quant_params = {}
        
        for idx, weight in enumerate(weights):
            if weight.size == 0:
                quantized_weights.append(weight)
                continue
            
            # Calculate quantization parameters
            if self.config.per_channel and len(weight.shape) > 1:
                # Per-channel quantization (along first axis)
                params = self._calculate_per_channel_params(weight)
            else:
                # Per-tensor quantization
                params = self._calculate_quantization_params(weight)
            
            # Quantize
            q_weight = self._quantize_array(weight, params)
            quantized_weights.append(q_weight)
            
            if store_params:
                quant_params[f"layer_{idx}"] = params
        
        if store_params:
            self.quantization_params = quant_params
        
        # Calculate compression ratio
        original_size = sum(w.nbytes for w in weights)
        quantized_size = sum(w.nbytes for w in quantized_weights)
        compression_ratio = original_size / quantized_size if quantized_size > 0 else 1.0
        
        print(f"✓ Quantized {len(weights)} weight tensors")
        print(f"  Original size: {original_size / (1024*1024):.2f} MB")
        print(f"  Quantized size: {quantized_size / (1024*1024):.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return quantized_weights, quant_params
    
    def dequantize_weights(
        self,
        quantized_weights: List[np.ndarray],
        quant_params: Optional[Dict] = None
    ) -> List[np.ndarray]:
        """
        Dequantize weights back to original precision
        
        Args:
            quantized_weights: List of quantized weight arrays
            quant_params: Quantization parameters (uses stored if None)
            
        Returns:
            List of dequantized weights
        """
        if quant_params is None:
            quant_params = self.quantization_params
        
        dequantized_weights = []
        
        for idx, q_weight in enumerate(quantized_weights):
            if q_weight.size == 0:
                dequantized_weights.append(q_weight)
                continue
            
            params = quant_params.get(f"layer_{idx}")
            if params is None:
                print(f"Warning: No quantization params for layer {idx}, using quantized values")
                dequantized_weights.append(q_weight.astype(np.float32))
                continue
            
            # Dequantize
            deq_weight = self._dequantize_array(q_weight, params)
            dequantized_weights.append(deq_weight)
        
        return dequantized_weights
    
    def _calculate_quantization_params(self, array: np.ndarray) -> Dict:
        """Calculate quantization parameters for an array"""
        min_val = float(np.min(array))
        max_val = float(np.max(array))
        
        # Calculate scale and zero point
        qmin = 0
        qmax = 2 ** self.config.bits - 1
        
        if self.config.symmetric:
            # Symmetric quantization
            abs_max = max(abs(min_val), abs(max_val))
            scale = 2 * abs_max / (qmax - qmin) if abs_max > 0 else 1.0
            zero_point = (qmax + qmin) // 2
        else:
            # Asymmetric quantization
            scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
            zero_point = int(qmin - min_val / scale) if scale > 0 else 0
            zero_point = np.clip(zero_point, qmin, qmax)
        
        params = {
            "scale": scale,
            "zero_point": zero_point,
            "qmin": qmin,
            "qmax": qmax,
            "original_dtype": str(array.dtype),
            "original_shape": array.shape,
            "bits": self.config.bits
        }
        
        # Store original element count for 4-bit unpacking
        if self.config.bits == 4:
            params["original_numel"] = array.size
        
        return params
    
    def _calculate_per_channel_params(self, array: np.ndarray) -> Dict:
        """Calculate per-channel quantization parameters"""
        num_channels = array.shape[0]
        scales = []
        zero_points = []
        
        qmin = 0
        qmax = 2 ** self.config.bits - 1
        
        for i in range(num_channels):
            channel_data = array[i]
            min_val = float(np.min(channel_data))
            max_val = float(np.max(channel_data))
            
            if self.config.symmetric:
                abs_max = max(abs(min_val), abs(max_val))
                scale = 2 * abs_max / (qmax - qmin) if abs_max > 0 else 1.0
                zero_point = (qmax + qmin) // 2
            else:
                scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
                zero_point = int(qmin - min_val / scale) if scale > 0 else 0
                zero_point = np.clip(zero_point, qmin, qmax)
            
            scales.append(scale)
            zero_points.append(zero_point)
        
        params = {
            "scales": scales,
            "zero_points": zero_points,
            "qmin": qmin,
            "qmax": qmax,
            "per_channel": True,
            "original_dtype": str(array.dtype),
            "original_shape": array.shape,
            "bits": self.config.bits
        }
        
        # Store original element count for 4-bit unpacking
        if self.config.bits == 4:
            params["original_numel"] = array.size
        
        return params
    
    def _quantize_array(self, array: np.ndarray, params: Dict) -> np.ndarray:
        """Quantize array using parameters"""
        if params.get("per_channel", False):
            return self._quantize_per_channel(array, params)
        
        scale = params["scale"]
        zero_point = params["zero_point"]
        qmin = params["qmin"]
        qmax = params["qmax"]
        
        # Quantize: q = clip(round(x / scale) + zero_point, qmin, qmax)
        quantized = np.round(array / scale) + zero_point
        quantized = np.clip(quantized, qmin, qmax)
        
        # Use appropriate dtype based on bits
        if self.config.bits == 4:
            # 4-bit quantization: pack two values per byte for actual size reduction
            quantized = quantized.astype(np.uint8)
            return self._pack_4bit(quantized)
        elif self.config.bits <= 8:
            dtype = np.uint8
        elif self.config.bits <= 16:
            dtype = np.uint16
        else:
            dtype = np.int32
        
        return quantized.astype(dtype)
    
    def _pack_4bit(self, array: np.ndarray) -> np.ndarray:
        """Pack 4-bit values into bytes (two values per byte)"""
        flat = array.flatten()
        original_size = flat.shape[0]
        
        # Pad to even length if needed
        if original_size % 2 != 0:
            flat = np.concatenate([flat, np.array([0], dtype=np.uint8)])
        
        # Pack two 4-bit values per byte: high nibble | low nibble
        packed = (flat[0::2] << 4) | (flat[1::2] & 0x0F)
        return packed.astype(np.uint8)
    
    def _unpack_4bit(self, packed: np.ndarray, original_size: int) -> np.ndarray:
        """Unpack 4-bit values from bytes"""
        flat = packed.flatten()
        
        # Extract high and low nibbles
        high = (flat >> 4) & 0x0F
        low = flat & 0x0F
        
        # Interleave back to original order
        unpacked = np.empty(len(flat) * 2, dtype=np.uint8)
        unpacked[0::2] = high
        unpacked[1::2] = low
        
        # Trim to original size
        return unpacked[:original_size]
    
    def _quantize_per_channel(self, array: np.ndarray, params: Dict) -> np.ndarray:
        """Quantize array with per-channel parameters"""
        quantized = np.zeros_like(array)
        scales = params["scales"]
        zero_points = params["zero_points"]
        qmin = params["qmin"]
        qmax = params["qmax"]
        
        for i in range(array.shape[0]):
            channel_data = array[i]
            q_channel = np.round(channel_data / scales[i]) + zero_points[i]
            quantized[i] = np.clip(q_channel, qmin, qmax)
        
        if self.config.bits == 4:
            # 4-bit quantization: pack two values per byte
            quantized = quantized.astype(np.uint8)
            return self._pack_4bit(quantized)
        elif self.config.bits <= 8:
            dtype = np.uint8
        elif self.config.bits <= 16:
            dtype = np.uint16
        else:
            dtype = np.int32
        
        return quantized.astype(dtype)
    
    def _dequantize_array(self, quantized: np.ndarray, params: Dict) -> np.ndarray:
        """Dequantize array using parameters"""
        if params.get("per_channel", False):
            return self._dequantize_per_channel(quantized, params)
        
        scale = params["scale"]
        zero_point = params["zero_point"]
        original_shape = params.get("original_shape")
        bits = params.get("bits", self.config.bits)
        
        # Handle 4-bit unpacking
        if bits == 4:
            original_numel = params.get("original_numel", np.prod(original_shape) if original_shape else quantized.size * 2)
            unpacked = self._unpack_4bit(quantized, original_numel)
            # Dequantize: x = (q - zero_point) * scale
            dequantized = (unpacked.astype(np.float32) - zero_point) * scale
            if original_shape is not None:
                dequantized = dequantized.reshape(original_shape)
            return dequantized.astype(np.float32)
        
        # Dequantize: x = (q - zero_point) * scale
        dequantized = (quantized.astype(np.float32) - zero_point) * scale
        
        return dequantized.astype(np.float32)
    
    def _dequantize_per_channel(self, quantized: np.ndarray, params: Dict) -> np.ndarray:
        """Dequantize array with per-channel parameters"""
        original_shape = params["original_shape"]
        scales = params["scales"]
        zero_points = params["zero_points"]
        bits = params.get("bits", self.config.bits)
        
        # Handle 4-bit unpacking
        if bits == 4:
            original_numel = params.get("original_numel", np.prod(original_shape))
            unpacked = self._unpack_4bit(quantized, original_numel)
            unpacked = unpacked.reshape(original_shape)
            
            dequantized = np.zeros(original_shape, dtype=np.float32)
            for i in range(unpacked.shape[0]):
                q_channel = unpacked[i]
                dequantized[i] = (q_channel.astype(np.float32) - zero_points[i]) * scales[i]
            return dequantized
        
        dequantized = np.zeros(original_shape, dtype=np.float32)
        for i in range(quantized.shape[0]):
            q_channel = quantized[i]
            dequantized[i] = (q_channel.astype(np.float32) - zero_points[i]) * scales[i]
        
        return dequantized
    
    # ==================== Main Compression Interface ====================
    
    def compress(
        self,
        data: Any,
        data_type: str = "weights"
    ) -> Dict:
        """
        Compress model weights or gradients
        
        Args:
            data: Model weights (list of arrays) or model object
            data_type: Type of data ("weights", "gradients", "model")
            
        Returns:
            Dictionary with compressed data and metadata
        """
        input_data = self._maybe_prune_before_quantization(data, data_type)

        if self.config.strategy == QuantizationStrategy.QAT.value:
            return self._compress_qat(input_data, data_type)
        elif self.config.strategy == QuantizationStrategy.PTQ.value:
            return self._compress_ptq(input_data, data_type)
        else:  # Parameter quantization
            return self._compress_param(input_data, data_type)
    
    def _compress_qat(self, data: Any, data_type: str) -> Dict:
        """Compress using QAT strategy"""
        if isinstance(data, tf.keras.Model):
            # Train with QAT and export
            if self.qat_model is None:
                self.qat_model = self.prepare_qat_model(data)
            weights = self.qat_model.get_weights()
        elif isinstance(data, list):
            weights = data
        else:
            raise ValueError(f"Unsupported data type for QAT: {type(data)}")
        
        # Quantize the weights
        q_weights, params = self.quantize_weights(weights)
        
        return {
            "compressed_data": q_weights,
            "quantization_params": params,
            "strategy": self.config.strategy,
            "config": self.config.to_dict(),
            "data_type": data_type
        }
    
    def _compress_ptq(self, data: Any, data_type: str) -> Dict:
        """Compress using PTQ strategy"""
        if isinstance(data, tf.keras.Model):
            # Apply post-training quantization
            quantized_model = self.post_training_quantize_model(data)
            if quantized_model is None:
                # Fallback to parameter quantization
                weights = data.get_weights()
                q_weights, params = self.quantize_weights(weights)
                return {
                    "compressed_data": q_weights,
                    "quantization_params": params,
                    "strategy": "parameter_quantization",
                    "config": self.config.to_dict(),
                    "data_type": data_type
                }
            else:
                return {
                    "compressed_data": quantized_model,
                    "quantization_params": {},
                    "strategy": self.config.strategy,
                    "config": self.config.to_dict(),
                    "data_type": "tflite_model"
                }
        elif isinstance(data, list):
            # Quantize weights directly
            q_weights, params = self.quantize_weights(data)
            return {
                "compressed_data": q_weights,
                "quantization_params": params,
                "strategy": self.config.strategy,
                "config": self.config.to_dict(),
                "data_type": data_type
            }
        else:
            raise ValueError(f"Unsupported data type for PTQ: {type(data)}")
    
    def _compress_param(self, data: Any, data_type: str) -> Dict:
        """Compress using parameter quantization"""
        if isinstance(data, tf.keras.Model):
            weights = data.get_weights()
        elif isinstance(data, list):
            weights = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Quantize weights
        q_weights, params = self.quantize_weights(weights)
        
        return {
            "compressed_data": q_weights,
            "quantization_params": params,
            "strategy": self.config.strategy,
            "config": self.config.to_dict(),
            "data_type": data_type
        }
    
    def decompress(
        self,
        compressed_data: Dict
    ) -> List[np.ndarray]:
        """
        Decompress quantized data back to original format
        
        Args:
            compressed_data: Dictionary from compress()
            
        Returns:
            Decompressed weights as list of arrays
        """
        strategy = compressed_data.get("strategy", self.config.strategy)
        data_type = compressed_data.get("data_type", "weights")
        
        if data_type == "tflite_model":
            # TFLite model - return as-is (would need interpreter for inference)
            print("Warning: TFLite model decompression not fully supported. Returning compressed model.")
            return compressed_data["compressed_data"]
        
        # Dequantize weights
        q_weights = compressed_data["compressed_data"]
        params = compressed_data["quantization_params"]
        
        weights = self.dequantize_weights(q_weights, params)
        
        return weights

    def as_training_weights(self, compressed_data: Dict) -> List[np.ndarray]:
        """
        Return weights suitable for local training WITHOUT dequantizing.

        This intentionally does NOT apply quantization_params (no scale/zero-point restoration).
        It only casts the quantized tensors to float32 so TensorFlow can load them into a model.
        """
        if not isinstance(compressed_data, dict):
            raise ValueError(f"compressed_data must be a dict, got {type(compressed_data)}")

        if compressed_data.get("data_type") == "tflite_model":
            raise ValueError("tflite_model cannot be used as training weights in this pipeline")

        q_weights = compressed_data.get("compressed_data")
        if q_weights is None:
            raise ValueError("compressed_data missing 'compressed_data'")

        if not isinstance(q_weights, list):
            # In this codebase, weight quantization uses list[np.ndarray]. Keep this strict.
            raise ValueError(f"Expected quantized weights as list, got {type(q_weights)}")

        return [np.asarray(w, dtype=np.float32) for w in q_weights]
    
    # ==================== Utility Methods ====================
    
    def get_compression_stats(
        self,
        original_weights: List[np.ndarray],
        compressed_data: Dict
    ) -> Dict:
        """Calculate compression statistics"""
        original_size = sum(w.nbytes for w in original_weights)
        
        if compressed_data.get("data_type") == "tflite_model":
            compressed_size = len(compressed_data["compressed_data"])
        else:
            compressed_size = sum(w.nbytes for w in compressed_data["compressed_data"])
        
        return {
            "original_size_mb": original_size / (1024 * 1024),
            "compressed_size_mb": compressed_size / (1024 * 1024),
            "compression_ratio": original_size / compressed_size if compressed_size > 0 else 1.0,
            "size_reduction_percent": ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0.0,
            "strategy": compressed_data["strategy"],
            "bits": compressed_data["config"]["bits"]
        }


# Backward compatibility - keep old class name
class quantization(Quantization):
    """Alias for backward compatibility"""
    pass
