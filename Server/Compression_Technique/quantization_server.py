"""
Server-side Quantization Handler for Federated Learning
Handles decompression and aggregation of quantized model updates
"""

import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple, Optional, Any
import os
import sys
import importlib.util

# Import client quantization module dynamically to avoid circular import
client_quantization_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
    'Client', 
    'Compression_Technique', 
    'quantization_client.py'
)

# Check if file exists before loading
if not os.path.exists(client_quantization_path):
    raise FileNotFoundError(
        f"quantization_client.py not found at {client_quantization_path}. "
        f"Please ensure the Client/Compression_Technique directory is properly mounted in Docker."
    )

spec = importlib.util.spec_from_file_location("client_quantization", client_quantization_path)
client_quantization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client_quantization)

Quantization = client_quantization.Quantization
QuantizationConfig = client_quantization.QuantizationConfig
QuantizationStrategy = client_quantization.QuantizationStrategy


class ServerQuantizationHandler:
    """
    Server-side handler for quantized model updates
    Manages decompression and aggregation of quantized weights
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self.quantizer = Quantization(self.config)
        self.client_quantization_params = {}  # Store per-client quantization params
        self.client_quantizers = {}  # Optional per-client Quantization instances
        
        print(f"\n{'='*70}")
        print(f"Server Quantization Handler Initialized")
        print(f"{'='*70}")
        print(f"Strategy: {self.config.strategy}")
        print(f"Bits: {self.config.bits}")
        print(f"{'='*70}\n")
    
    def decompress_client_update(
        self,
        client_id: int,
        compressed_data: Dict
    ) -> List[np.ndarray]:
        """
        Decompress quantized weights from a client
        
        Args:
            client_id: Client identifier
            compressed_data: Compressed data dictionary from client
            
        Returns:
            Decompressed weights
        """
        # Store client's quantization parameters
        params = compressed_data.get("quantization_params", {}) or {}
        self.client_quantization_params[client_id] = params

        # If client provided params that differ from server default, create or update per-client quantizer
        if params:
            # Build a QuantizationConfig from params
            try:
                client_config = self._config_from_params(params)
                # If we don't have a client quantizer or config changed, create a new one
                existing = self.client_quantizers.get(client_id)
                if existing is None or getattr(existing, 'config', None) != client_config:
                    self.client_quantizers[client_id] = Quantization(client_config)
                    print(f"Server: Created per-client quantizer for client {client_id} -> {params}")
                quant = self.client_quantizers[client_id]
            except Exception as e:
                print(f"Server: Failed to build client quantizer for client {client_id}, falling back to default - {e}")
                quant = self.quantizer
        else:
            quant = self.quantizer

        # Decompress using the selected quantizer
        weights = quant.decompress(compressed_data)

        return weights

    def _config_from_params(self, params: Dict[str, Any]) -> QuantizationConfig:
        """Create a QuantizationConfig from a client's quantization_params dict.

        Expected keys in params (optional): 'strategy', 'bits', 'symmetric', 'per_channel'
        """
        cfg = QuantizationConfig()
        # Map simple keys; be tolerant of strings
        strategy = params.get('strategy')
        if strategy:
            try:
                cfg.strategy = QuantizationStrategy(strategy)
            except Exception:
                # allow passing enum member or string name
                try:
                    cfg.strategy = QuantizationStrategy[strategy]
                except Exception:
                    pass

        bits = params.get('bits')
        if bits is not None:
            try:
                cfg.bits = int(bits)
            except Exception:
                pass

        sym = params.get('symmetric')
        if sym is not None:
            if isinstance(sym, str):
                cfg.symmetric = sym.lower() in ('1', 'true', 'yes')
            else:
                cfg.symmetric = bool(sym)

        per_ch = params.get('per_channel')
        if per_ch is not None:
            if isinstance(per_ch, str):
                cfg.per_channel = per_ch.lower() in ('1', 'true', 'yes')
            else:
                cfg.per_channel = bool(per_ch)

        return cfg
    
    def aggregate_quantized_updates(
        self,
        client_updates: Dict[int, Dict],
        aggregation_method: str = "fedavg"
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        Aggregate quantized updates from multiple clients
        
        Args:
            client_updates: Dictionary mapping client_id to update data
                          Each update should contain:
                          - 'compressed_data': compressed weights
                          - 'num_samples': number of training samples
            aggregation_method: Aggregation method ("fedavg", "weighted_avg")
            
        Returns:
            Tuple of (aggregated_weights, aggregation_stats)
        """
        print(f"\n{'='*70}")
        print(f"Aggregating Quantized Updates from {len(client_updates)} Clients")
        print(f"{'='*70}")
        
        # Decompress all client updates
        decompressed_updates = {}
        total_samples = 0
        
        for client_id, update in client_updates.items():
            weights = self.decompress_client_update(client_id, update['compressed_data'])
            num_samples = update.get('num_samples', 1)
            
            decompressed_updates[client_id] = {
                'weights': weights,
                'num_samples': num_samples
            }
            total_samples += num_samples
            
            print(f"  Client {client_id}: Decompressed {len(weights)} layers, {num_samples} samples")
        
        # Aggregate based on method
        if aggregation_method == "fedavg" or aggregation_method == "weighted_avg":
            aggregated = self._federated_averaging(decompressed_updates, total_samples)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        stats = {
            "num_clients": len(client_updates),
            "total_samples": total_samples,
            "aggregation_method": aggregation_method,
            "num_layers": len(aggregated)
        }
        
        print(f"✓ Aggregation complete: {len(aggregated)} layers")
        print(f"{'='*70}\n")
        
        return aggregated, stats
    
    def _federated_averaging(
        self,
        client_updates: Dict[int, Dict],
        total_samples: int
    ) -> List[np.ndarray]:
        """
        Perform federated averaging on decompressed weights
        
        Args:
            client_updates: Decompressed client updates
            total_samples: Total number of samples across all clients
            
        Returns:
            Averaged weights
        """
        # Get first client's weights to determine structure
        first_client = list(client_updates.values())[0]
        num_layers = len(first_client['weights'])
        
        # Initialize aggregated weights
        aggregated_weights = [np.zeros_like(w) for w in first_client['weights']]
        
        # Weighted averaging
        for client_id, update in client_updates.items():
            weights = update['weights']
            num_samples = update['num_samples']
            weight_factor = num_samples / total_samples
            
            for i, layer_weights in enumerate(weights):
                aggregated_weights[i] += layer_weights * weight_factor
        
        return aggregated_weights
    
    def compress_global_model(
        self,
        weights: List[np.ndarray]
    ) -> Dict:
        """
        Compress global model before sending to clients
        
        Args:
            weights: Global model weights
            
        Returns:
            Compressed data dictionary
        """
        compressed = self.quantizer.compress(weights, data_type="weights")
        
        return compressed
    
    def should_use_quantization(self) -> bool:
        """Check if quantization should be applied"""
        return os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    
    def get_compression_stats(
        self,
        original_weights: List[np.ndarray],
        compressed_data: Dict
    ) -> Dict:
        """Get compression statistics"""
        return self.quantizer.get_compression_stats(original_weights, compressed_data)


# Backward compatibility
class quantization(ServerQuantizationHandler):
    """Alias for backward compatibility"""
    pass
