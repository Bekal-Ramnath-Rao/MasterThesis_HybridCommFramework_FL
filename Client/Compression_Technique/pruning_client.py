"""
Model Pruning Compression for Federated Learning
Implements magnitude-based pruning for all model types:
- CNN models (Emotion Recognition)
- CNN+BiLSTM+MHA models (Mental State Recognition)
- LSTM models (Temperature Regulation)

Supports both structured and unstructured pruning
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Any
import os


class PruningConfig:
    """Configuration for model pruning"""
    def __init__(
        self,
        target_sparsity: float = 0.5,
        pruning_schedule: str = "polynomial",  # polynomial, constant
        begin_step: int = 0,
        end_step: int = 1000,
        frequency: int = 100,
        prune_bias: bool = False,
        structured: bool = False,  # True for structured, False for unstructured
        block_size: Tuple[int, int] = (1, 1)
    ):
        self.target_sparsity = target_sparsity
        self.pruning_schedule = pruning_schedule
        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency
        self.prune_bias = prune_bias
        self.structured = structured
        self.block_size = block_size
        
        # Load from environment if available
        self.target_sparsity = float(os.getenv("PRUNING_SPARSITY", str(target_sparsity)))
        self.structured = os.getenv("PRUNING_STRUCTURED", str(structured)).lower() == "true"
        
    def to_dict(self):
        return {
            "target_sparsity": self.target_sparsity,
            "pruning_schedule": self.pruning_schedule,
            "begin_step": self.begin_step,
            "end_step": self.end_step,
            "frequency": self.frequency,
            "prune_bias": self.prune_bias,
            "structured": self.structured,
            "block_size": self.block_size
        }


class ModelPruning:
    """
    Comprehensive Model Pruning Implementation
    Works with CNN, LSTM, and hybrid architectures
    """
    
    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        self.pruned_model = None
        self.pruning_masks = {}
        self.original_weights = {}
        self.current_step = 0
        
        print(f"\n{'='*70}")
        print(f"Model Pruning Module Initialized")
        print(f"{'='*70}")
        print(f"Target Sparsity: {self.config.target_sparsity * 100:.1f}%")
        print(f"Pruning Type: {'Structured' if self.config.structured else 'Unstructured'}")
        print(f"Schedule: {self.config.pruning_schedule}")
        print(f"{'='*70}\n")
    
    def compute_sparsity_schedule(self, step: int) -> float:
        """Compute current sparsity based on schedule"""
        if step < self.config.begin_step:
            return 0.0
        if step >= self.config.end_step:
            return self.config.target_sparsity
        
        if self.config.pruning_schedule == "constant":
            return self.config.target_sparsity
        
        # Polynomial decay (cubic by default)
        progress = (step - self.config.begin_step) / (self.config.end_step - self.config.begin_step)
        sparsity = self.config.target_sparsity * (1 - (1 - progress) ** 3)
        return sparsity
    
    def create_pruning_mask(self, weights: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Create pruning mask based on magnitude
        Returns binary mask (1 = keep, 0 = prune)
        """
        if sparsity == 0:
            return np.ones_like(weights)
        
        # Flatten weights to find threshold
        flat_weights = np.abs(weights.flatten())
        threshold_index = int(len(flat_weights) * sparsity)
        
        if threshold_index >= len(flat_weights):
            return np.zeros_like(weights)
        
        # Sort and find threshold
        sorted_weights = np.sort(flat_weights)
        threshold = sorted_weights[threshold_index]
        
        # Create mask
        mask = (np.abs(weights) > threshold).astype(np.float32)
        
        return mask
    
    def create_structured_mask(
        self,
        weights: np.ndarray,
        sparsity: float,
        axis: int = -1
    ) -> np.ndarray:
        """
        Create structured pruning mask (prunes entire channels/filters)
        """
        if sparsity == 0:
            return np.ones_like(weights)
        
        # Compute L2 norm along specified axis
        if len(weights.shape) == 4:  # Conv2D: (height, width, in_channels, out_channels)
            # Prune output filters
            filter_norms = np.sum(weights ** 2, axis=(0, 1, 2))
            num_filters = len(filter_norms)
            num_to_prune = int(num_filters * sparsity)
            
            # Find filters to prune
            threshold_index = num_to_prune
            sorted_indices = np.argsort(filter_norms)
            prune_indices = sorted_indices[:threshold_index]
            
            # Create mask
            mask = np.ones_like(weights)
            mask[:, :, :, prune_indices] = 0
            
        elif len(weights.shape) == 3:  # Conv1D: (kernel_size, in_channels, out_channels)
            # Prune output filters
            filter_norms = np.sum(weights ** 2, axis=(0, 1))
            num_filters = len(filter_norms)
            num_to_prune = int(num_filters * sparsity)
            
            threshold_index = num_to_prune
            sorted_indices = np.argsort(filter_norms)
            prune_indices = sorted_indices[:threshold_index]
            
            mask = np.ones_like(weights)
            mask[:, :, prune_indices] = 0
            
        elif len(weights.shape) == 2:  # Dense: (in_features, out_features)
            # Prune output neurons
            neuron_norms = np.sum(weights ** 2, axis=0)
            num_neurons = len(neuron_norms)
            num_to_prune = int(num_neurons * sparsity)
            
            threshold_index = num_to_prune
            sorted_indices = np.argsort(neuron_norms)
            prune_indices = sorted_indices[:threshold_index]
            
            mask = np.ones_like(weights)
            mask[:, prune_indices] = 0
        else:
            # Fallback to unstructured
            mask = self.create_pruning_mask(weights, sparsity)
        
        return mask
    
    def prune_weights(self, weights: List[np.ndarray], step: int = None) -> List[np.ndarray]:
        """
        Prune model weights based on current schedule
        """
        if step is not None:
            self.current_step = step
        
        current_sparsity = self.compute_sparsity_schedule(self.current_step)
        
        pruned_weights = []
        total_params = 0
        pruned_params = 0
        
        for i, weight in enumerate(weights):
            # Skip bias terms if configured
            if not self.config.prune_bias and len(weight.shape) == 1:
                pruned_weights.append(weight)
                continue
            
            # Skip very small layers
            if weight.size < 10:
                pruned_weights.append(weight)
                continue
            
            # Create or retrieve mask
            if i not in self.pruning_masks:
                if self.config.structured:
                    mask = self.create_structured_mask(weight, current_sparsity)
                else:
                    mask = self.create_pruning_mask(weight, current_sparsity)
                self.pruning_masks[i] = mask
            else:
                # Update existing mask
                if self.config.structured:
                    mask = self.create_structured_mask(weight, current_sparsity)
                else:
                    mask = self.create_pruning_mask(weight, current_sparsity)
                self.pruning_masks[i] = mask
            
            # Apply mask
            pruned_weight = weight * self.pruning_masks[i]
            pruned_weights.append(pruned_weight)
            
            # Track statistics
            total_params += weight.size
            pruned_params += np.sum(self.pruning_masks[i] == 0)
        
        actual_sparsity = pruned_params / total_params if total_params > 0 else 0
        
        if self.current_step % 100 == 0:
            print(f"[Pruning] Step {self.current_step}: "
                  f"Target sparsity: {current_sparsity:.2%}, "
                  f"Actual sparsity: {actual_sparsity:.2%}")
        
        self.current_step += 1
        return pruned_weights
    
    def apply_pruning_to_model(self, model: tf.keras.Model, step: int = None) -> tf.keras.Model:
        """
        Apply pruning directly to a Keras model
        """
        if step is not None:
            self.current_step = step
        
        # Get current weights
        weights = model.get_weights()
        
        # Prune weights
        pruned_weights = self.prune_weights(weights, step)
        
        # Set pruned weights back to model
        model.set_weights(pruned_weights)
        
        return model
    
    def get_pruning_statistics(self, weights: List[np.ndarray]) -> Dict[str, Any]:
        """
        Compute pruning statistics
        """
        total_params = 0
        zero_params = 0
        layer_stats = []
        
        for i, weight in enumerate(weights):
            layer_total = weight.size
            layer_zeros = np.sum(weight == 0)
            layer_sparsity = layer_zeros / layer_total if layer_total > 0 else 0
            
            total_params += layer_total
            zero_params += layer_zeros
            
            layer_stats.append({
                "layer": i,
                "shape": weight.shape,
                "total_params": layer_total,
                "zero_params": int(layer_zeros),
                "sparsity": layer_sparsity
            })
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0
        
        return {
            "total_params": total_params,
            "zero_params": int(zero_params),
            "non_zero_params": int(total_params - zero_params),
            "overall_sparsity": overall_sparsity,
            "compression_ratio": 1.0 / (1.0 - overall_sparsity) if overall_sparsity < 1.0 else float('inf'),
            "layer_stats": layer_stats
        }
    
    def compress_pruned_weights(self, weights: List[np.ndarray]) -> Tuple[bytes, Dict]:
        """
        Compress pruned weights using sparse representation
        Only store non-zero values and their indices
        """
        import pickle
        
        compressed_data = []
        metadata = {"shapes": [], "dtypes": []}
        
        for weight in weights:
            # Get non-zero elements
            if len(weight.shape) > 1:
                # Use COO (Coordinate) format for multi-dimensional arrays
                non_zero_indices = np.nonzero(weight)
                non_zero_values = weight[non_zero_indices]
                
                compressed_data.append({
                    "indices": non_zero_indices,
                    "values": non_zero_values,
                    "format": "coo"
                })
            else:
                # For 1D arrays (biases), just store non-zero values and indices
                non_zero_mask = weight != 0
                compressed_data.append({
                    "indices": np.where(non_zero_mask)[0],
                    "values": weight[non_zero_mask],
                    "format": "sparse"
                })
            
            metadata["shapes"].append(weight.shape)
            metadata["dtypes"].append(str(weight.dtype))
        
        # Serialize
        serialized = pickle.dumps({"data": compressed_data, "metadata": metadata})
        
        return serialized, metadata
    
    def decompress_pruned_weights(self, compressed_data: bytes) -> List[np.ndarray]:
        """
        Decompress sparse weights back to dense format
        """
        import pickle
        
        data_dict = pickle.loads(compressed_data)
        compressed_list = data_dict["data"]
        metadata = data_dict["metadata"]
        
        weights = []
        for i, compressed_weight in enumerate(compressed_list):
            shape = metadata["shapes"][i]
            dtype = metadata["dtypes"][i]
            
            # Reconstruct dense array
            dense_weight = np.zeros(shape, dtype=dtype)
            
            if compressed_weight["format"] == "coo":
                indices = compressed_weight["indices"]
                values = compressed_weight["values"]
                dense_weight[indices] = values
            else:  # sparse format for 1D
                indices = compressed_weight["indices"]
                values = compressed_weight["values"]
                dense_weight[indices] = values
            
            weights.append(dense_weight)
        
        return weights
    
    def fine_tune_pruned_model(
        self,
        model: tf.keras.Model,
        train_data,
        validation_data=None,
        epochs: int = 5,
        batch_size: int = 32
    ) -> tf.keras.Model:
        """
        Fine-tune pruned model to recover accuracy
        Maintains pruning masks during training
        """
        # Store current weights
        current_weights = model.get_weights()
        
        # Create custom training loop that maintains masks
        for epoch in range(epochs):
            print(f"\nFine-tuning epoch {epoch + 1}/{epochs}")
            
            # Train for one epoch
            history = model.fit(
                train_data,
                validation_data=validation_data,
                epochs=1,
                batch_size=batch_size,
                verbose=1
            )
            
            # Re-apply pruning masks after each epoch
            updated_weights = model.get_weights()
            masked_weights = []
            
            for i, weight in enumerate(updated_weights):
                if i in self.pruning_masks:
                    masked_weight = weight * self.pruning_masks[i]
                    masked_weights.append(masked_weight)
                else:
                    masked_weights.append(weight)
            
            model.set_weights(masked_weights)
        
        return model


def create_pruning_callback(pruning_config: PruningConfig):
    """
    Create a Keras callback for gradual pruning during training
    """
    try:
        import tensorflow_model_optimization as tfmot
        
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=pruning_config.target_sparsity,
                begin_step=pruning_config.begin_step,
                end_step=pruning_config.end_step,
                frequency=pruning_config.frequency
            )
        }
        
        return tfmot.sparsity.keras.UpdatePruningStep(), tfmot.sparsity.keras.PruningSummaries(log_dir='/tmp/pruning')
    
    except ImportError:
        print("Warning: tensorflow_model_optimization not available for pruning callback")
        return None


# Example usage for different model types
if __name__ == "__main__":
    print("Model Pruning Module - Supports all FL model architectures")
    print("=" * 70)
    print("Supported Models:")
    print("  1. CNN (Emotion Recognition) - Conv2D layers")
    print("  2. CNN+BiLSTM+MHA (Mental State) - Conv1D, LSTM, Dense layers")
    print("  3. LSTM (Temperature) - LSTM, Dense layers")
    print("=" * 70)
