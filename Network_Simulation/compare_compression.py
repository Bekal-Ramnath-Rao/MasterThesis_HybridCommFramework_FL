#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Compression Comparison Tool
Compares FL experiments with different compression techniques:
- Baseline (no compression)
- Quantization only
- Pruning only
- Quantization + Pruning (combined)
"""

import argparse
import sys
from pathlib import Path
from compare_quantization_results import QuantizationComparator
from compare_pruning_results import PruningComparator

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def main():
    parser = argparse.ArgumentParser(
        description="Compare FL experiments with different compression techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  # Compare quantization results
  python compare_compression.py --mode quantization \\
    --baseline experiment_results/temperature_baseline \\
    --compressed experiment_results/temperature_quantized_8bit
  
  # Compare pruning results
  python compare_compression.py --mode pruning \\
    --baseline experiment_results/temperature_baseline \\
    --compressed experiment_results/temperature_pruned_50pct
  
  # Compare both techniques
  python compare_compression.py --mode both \\
    --baseline experiment_results/temperature_baseline \\
    --quantized experiment_results/temperature_quantized_8bit \\
    --pruned experiment_results/temperature_pruned_50pct
        """
    )
    
    parser.add_argument("--mode", "-m", required=True,
                       choices=["quantization", "pruning", "both"],
                       help="Comparison mode")
    parser.add_argument("--baseline", "-b", required=True,
                       help="Path to baseline (no compression) experiment directory")
    parser.add_argument("--compressed", "-c",
                       help="Path to compressed experiment (for quantization or pruning mode)")
    parser.add_argument("--quantized", "-q",
                       help="Path to quantized experiment (for 'both' mode)")
    parser.add_argument("--pruned", "-p",
                       help="Path to pruned experiment (for 'both' mode)")
    parser.add_argument("--output", "-o", default=None,
                       help="Output directory for comparison results")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "quantization":
        if not args.compressed:
            parser.error("--compressed is required for quantization mode")
        print("\n" + "="*80)
        print("QUANTIZATION COMPARISON MODE")
        print("="*80)
        
        comparator = QuantizationComparator(args.baseline, args.compressed, args.output)
        df = comparator.compare_experiments()
        
        if len(df) == 0:
            print("\n[ERROR] No matching experiments found!")
            return 1
        
        comparator.save_results(df)
        comparator.plot_comparison_charts(df)
        
    elif args.mode == "pruning":
        if not args.compressed:
            parser.error("--compressed is required for pruning mode")
        print("\n" + "="*80)
        print("PRUNING COMPARISON MODE")
        print("="*80)
        
        comparator = PruningComparator(args.baseline, args.compressed, args.output)
        df = comparator.compare_experiments()
        
        if len(df) == 0:
            print("\n[ERROR] No matching experiments found!")
            return 1
        
        comparator.save_results(df)
        comparator.plot_comparison_charts(df)
        
    elif args.mode == "both":
        if not args.quantized or not args.pruned:
            parser.error("--quantized and --pruned are required for 'both' mode")
        
        print("\n" + "="*80)
        print("COMBINED COMPARISON MODE")
        print("="*80)
        print("Comparing: Baseline vs Quantization vs Pruning")
        print("="*80)
        
        # Run quantization comparison
        print("\n[1/2] Running quantization comparison...")
        output_quant = f"{args.output}/quantization" if args.output else None
        quant_comparator = QuantizationComparator(args.baseline, args.quantized, output_quant)
        quant_df = quant_comparator.compare_experiments()
        
        if len(quant_df) > 0:
            quant_comparator.save_results(quant_df)
            quant_comparator.plot_comparison_charts(quant_df)
        
        # Run pruning comparison
        print("\n[2/2] Running pruning comparison...")
        output_prune = f"{args.output}/pruning" if args.output else None
        prune_comparator = PruningComparator(args.baseline, args.pruned, output_prune)
        prune_df = prune_comparator.compare_experiments()
        
        if len(prune_df) > 0:
            prune_comparator.save_results(prune_df)
            prune_comparator.plot_comparison_charts(prune_df)
        
        # Generate combined summary
        print("\n" + "="*80)
        print("COMBINED SUMMARY")
        print("="*80)
        
        if len(quant_df) > 0:
            avg_quant_improvement = quant_df['loss_improvement_pct'].mean()
            print(f"Quantization - Average Loss Improvement: {avg_quant_improvement:.2f}%")
        
        if len(prune_df) > 0:
            avg_prune_change = prune_df['loss_change_pct'].mean()
            avg_sparsity = prune_df['sparsity'].mean()
            print(f"Pruning - Average Loss Change: {avg_prune_change:.2f}%")
            print(f"Pruning - Average Sparsity: {avg_sparsity:.2f}%")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
