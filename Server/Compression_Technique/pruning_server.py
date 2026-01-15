"""
Server-Side Pruning for Federated Learning
Coordinates pruning across global model and client updates
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Optional, Any, Tuple
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from Client.Compression_Technique.pruning_client import PruningConfig, ModelPruning


class ServerPruning:
    """
    Server-side pruning coordinator for FL
    Manages global model pruning and aggregation with pruned client updates
    """
    
    def __init__(self, config: Optional[PruningConfig] = None):
        self.config = config or PruningConfig()
        self.pruning_engine = ModelPruning(self.config)
        self.global_pruning_masks = {}
        self.round_number = 0
        
        print(f"\n{'='*70}")
        print(f"Server-Side Pruning Coordinator Initialized")
        print(f"{'='*70}")
        print(f"Target Sparsity: {self.config.target_sparsity * 100:.1f}%")
        print(f"Pruning Type: {'Structured' if self.config.structured else 'Unstructured'}")
        print(f"{'='*70}\n")
    
    def prune_global_model(
        self,
        model: tf.keras.Model,
        round_num: int
    ) -> tf.keras.Model:
        """
        Apply pruning to global model after aggregation
        """
        self.round_number = round_num
        
        # Apply pruning
        pruned_model = self.pruning_engine.apply_pruning_to_model(
            model,
            step=round_num
        )
        
        # Store global masks
        self.global_pruning_masks = self.pruning_engine.pruning_masks.copy()
        
        # Log statistics
        if round_num % 10 == 0:
            stats = self.pruning_engine.get_pruning_statistics(pruned_model.get_weights())
            print(f"\n[Server Pruning] Round {round_num}")
            print(f"  Overall Sparsity: {stats['overall_sparsity']:.2%}")
            print(f"  Compression Ratio: {stats['compression_ratio']:.2f}x")
            print(f"  Non-zero Params: {stats['non_zero_params']:,}/{stats['total_params']:,}")
        
        return pruned_model
    
    def aggregate_pruned_updates(
        self,
        client_weights_list: List[List[np.ndarray]],
        num_samples_list: List[int]
    ) -> List[np.ndarray]:
        """
        Aggregate client updates with pruning awareness
        Weighted average that respects pruning masks
        """
        total_samples = sum(num_samples_list)
        
        # Initialize aggregated weights
        aggregated_weights = [np.zeros_like(w) for w in client_weights_list[0]]
        
        # Weighted aggregation
        for client_weights, num_samples in zip(client_weights_list, num_samples_list):
            weight = num_samples / total_samples
            
            for i, client_weight in enumerate(client_weights):
                aggregated_weights[i] += weight * client_weight
        
        # Apply global pruning masks if they exist
        if self.global_pruning_masks:
            for i, mask_idx in enumerate(self.global_pruning_masks.keys()):
                if mask_idx < len(aggregated_weights):
                    aggregated_weights[mask_idx] *= self.global_pruning_masks[mask_idx]
        
        return aggregated_weights
    
    def compress_for_broadcast(
        self,
        weights: List[np.ndarray]
    ) -> Tuple[bytes, Dict]:
        """
        Compress pruned global model for efficient broadcast to clients
        """
        return self.pruning_engine.compress_pruned_weights(weights)
    
    def decompress_client_update(
        self,
        compressed_data: bytes
    ) -> List[np.ndarray]:
        """
        Decompress pruned client update
        """
        return self.pruning_engine.decompress_pruned_weights(compressed_data)
    
    def get_compression_stats(
        self,
        weights: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Get detailed compression statistics for logging
        """
        stats = self.pruning_engine.get_pruning_statistics(weights)
        
        # Calculate communication savings
        original_size = sum(w.nbytes for w in weights)
        compressed_data, _ = self.compress_for_broadcast(weights)
        compressed_size = len(compressed_data)
        
        stats['communication_savings'] = {
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
            'size_reduction_percent': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        }
        
        return stats
    
    def should_prune_this_round(self, round_num: int) -> bool:
        """
        Determine if pruning should be applied this round
        """
        # Prune according to frequency schedule
        if round_num < self.config.begin_step:
            return False
        
        if round_num > self.config.end_step:
            return True  # Always apply final pruning after end_step
        
        # Apply at specified frequency
        return (round_num - self.config.begin_step) % self.config.frequency == 0
    
    def save_pruning_state(self, filepath: str):
        """
        Save pruning masks and configuration
        """
        import pickle
        
        state = {
            'config': self.config.to_dict(),
            'global_masks': self.global_pruning_masks,
            'round_number': self.round_number
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[Server Pruning] State saved to {filepath}")
    
    def load_pruning_state(self, filepath: str):
        """
        Load pruning masks and configuration
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.global_pruning_masks = state['global_masks']
        self.round_number = state['round_number']
        self.pruning_engine.pruning_masks = self.global_pruning_masks.copy()
        
        print(f"[Server Pruning] State loaded from {filepath}")
        print(f"  Resumed at round {self.round_number}")


class PruningMetricsLogger:
    """
    Logger for tracking pruning metrics across FL rounds
    """
    
    def __init__(self, log_dir: str = "pruning_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = []
    
    def log_round(
        self,
        round_num: int,
        stats: Dict[str, Any],
        model_accuracy: Optional[float] = None
    ):
        """
        Log metrics for a single round
        """
        entry = {
            'round': round_num,
            'sparsity': stats['overall_sparsity'],
            'compression_ratio': stats['compression_ratio'],
            'non_zero_params': stats['non_zero_params'],
            'total_params': stats['total_params']
        }
        
        if model_accuracy is not None:
            entry['accuracy'] = model_accuracy
        
        if 'communication_savings' in stats:
            entry.update({
                'original_size_mb': stats['communication_savings']['original_size_bytes'] / (1024 * 1024),
                'compressed_size_mb': stats['communication_savings']['compressed_size_bytes'] / (1024 * 1024),
                'comm_compression_ratio': stats['communication_savings']['compression_ratio'],
                'size_reduction_percent': stats['communication_savings']['size_reduction_percent']
            })
        
        self.metrics.append(entry)
    
    def save_metrics(self, filename: str = "pruning_metrics.json"):
        """
        Save all metrics to JSON file
        """
        import json
        
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"[Pruning Metrics] Saved to {filepath}")
    
    def print_summary(self):
        """
        Print summary of pruning progression
        """
        if not self.metrics:
            print("No metrics logged yet")
            return
        
        print(f"\n{'='*70}")
        print("Pruning Summary")
        print(f"{'='*70}")
        print(f"Total Rounds: {len(self.metrics)}")
        
        final = self.metrics[-1]
        print(f"\nFinal Model Statistics:")
        print(f"  Sparsity: {final['sparsity']:.2%}")
        print(f"  Compression Ratio: {final['compression_ratio']:.2f}x")
        print(f"  Active Parameters: {final['non_zero_params']:,}/{final['total_params']:,}")
        
        if 'accuracy' in final:
            print(f"  Model Accuracy: {final['accuracy']:.2%}")
        
        if 'size_reduction_percent' in final:
            print(f"\nCommunication Efficiency:")
            print(f"  Size Reduction: {final['size_reduction_percent']:.1f}%")
            print(f"  Original Size: {final['original_size_mb']:.2f} MB")
            print(f"  Compressed Size: {final['compressed_size_mb']:.2f} MB")
        
        print(f"{'='*70}\n")


# Example usage
if __name__ == "__main__":
    print("Server-Side Pruning Module for Federated Learning")
    print("Coordinates pruning across all model architectures:")
    print("  - CNN (Emotion Recognition)")
    print("  - CNN+BiLSTM+MHA (Mental State Recognition)")
    print("  - LSTM (Temperature Regulation)")
