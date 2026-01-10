#!/usr/bin/env python3
"""
Master Evaluation Pipeline
Runs the complete evaluation pipeline: consolidate -> compare -> comprehensive analysis
"""

import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed successfully")
    else:
        print(f"\n✗ {description} failed with return code {result.returncode}")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Master Evaluation Pipeline for Network Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the complete evaluation pipeline:
1. Consolidate results from experiment logs
2. Compare protocols for each scenario
3. Generate comprehensive cross-scenario analysis
4. Export CSV data

Example usage:
  # Evaluate all scenarios for temperature use case
  python evaluate_all.py --use-case temperature

  # Evaluate specific scenarios
  python evaluate_all.py --use-case temperature --scenarios excellent good moderate

  # Skip comparison plots (faster)
  python evaluate_all.py --use-case temperature --skip-comparison
        """
    )
    
    parser.add_argument("--use-case", "-u",
                       choices=["emotion", "mentalstate", "temperature"],
                       default="temperature",
                       help="Use case to evaluate (default: temperature)")
    parser.add_argument("--scenarios", "-s",
                       nargs="+",
                       choices=["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                               "congested_light", "congested_moderate", "congested_heavy"],
                       help="Specific scenarios to evaluate (default: all available)")
    parser.add_argument("--output-dir", "-o",
                       default="comparison_results",
                       help="Directory to save evaluation results (default: comparison_results)")
    parser.add_argument("--skip-consolidate", action="store_true",
                       help="Skip the consolidation step (if already done)")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Skip individual scenario comparisons")
    parser.add_argument("--export-csv", action="store_true",
                       help="Export results to CSV format")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("MASTER EVALUATION PIPELINE")
    print("="*80)
    print(f"Use Case: {args.use_case.title()}")
    print(f"Scenarios: {', '.join(args.scenarios) if args.scenarios else 'All available'}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    total_steps = 3 if not args.skip_consolidate else 2
    if not args.skip_comparison:
        total_steps += 1
    
    # Step 1: Consolidate results
    if not args.skip_consolidate:
        cmd = ["python", "Network_Simulation/consolidate_results.py", "--use-case", args.use_case]
        if args.scenarios:
            cmd.extend(["--scenarios"] + args.scenarios)
        
        if run_command(cmd, "STEP 1: Consolidating experiment results"):
            success_count += 1
    else:
        print("\n⊳ Skipping consolidation step")
        success_count += 1
    
    # Step 2: Compare protocols for each scenario (optional)
    if not args.skip_comparison:
        scenarios = args.scenarios if args.scenarios else ["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                                                          "congested_light", "congested_moderate", "congested_heavy"]
        
        for scenario in scenarios:
            cmd = [
                "python", "Network_Simulation/compare_protocols.py",
                "--use-case", args.use_case,
                "--scenarios", scenario,
                "--output-dir", args.output_dir
            ]
            
            run_command(cmd, f"STEP 2.{scenarios.index(scenario)+1}: Comparing protocols for {scenario} scenario")
        
        success_count += 1
    else:
        print("\n⊳ Skipping individual scenario comparisons")
    
    # Step 3: Comprehensive network scenario evaluation
    cmd = [
        "python", "Network_Simulation/evaluate_network_scenarios.py",
        "--use-case", args.use_case,
        "--output-dir", args.output_dir
    ]
    
    if args.export_csv:
        cmd.append("--export-csv")
    
    if run_command(cmd, "STEP 3: Comprehensive cross-scenario analysis"):
        success_count += 1
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION PIPELINE SUMMARY")
    print("="*80)
    print(f"Steps completed: {success_count}/{total_steps}")
    print(f"Results directory: {args.output_dir}/")
    
    # List generated files
    if os.path.exists(args.output_dir):
        files = list(Path(args.output_dir).glob("*"))
        if files:
            print(f"\nGenerated files ({len(files)}):")
            for f in sorted(files)[:10]:  # Show first 10 files
                print(f"  - {f.name}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
    
    print("="*80 + "\n")
    
    if success_count == total_steps:
        print("✓ All evaluation steps completed successfully!\n")
        return 0
    else:
        print("⚠ Some evaluation steps failed. Check the output above for details.\n")
        return 1


if __name__ == "__main__":
    exit(main())
