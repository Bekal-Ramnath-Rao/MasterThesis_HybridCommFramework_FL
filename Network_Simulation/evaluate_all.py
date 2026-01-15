#!/usr/bin/env python3
"""
Master Evaluation Pipeline
Runs the complete evaluation pipeline: consolidate -> compare -> comprehensive analysis
"""

import subprocess
import argparse
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


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
  # Evaluate specific experiment folder
  python evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711

  # Evaluate with congestion experiments
  python evaluate_all.py --experiment-folder temperature_quantized_8bit_congestion_20260111_100707

  # Evaluate specific scenarios from a folder
  python evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711 --scenarios excellent good moderate

  # Skip comparison plots (faster)
  python evaluate_all.py --experiment-folder temperature_quantized_8bit_20260110_195711 --skip-comparison
        """
    )
    
    parser.add_argument("--experiment-folder", "-e",
                       required=True,
                       help="Name of the experiment folder in experiment_results/ to evaluate")
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
                       default=None,
                       help="Directory to save evaluation results (default: comparison_results/<experiment-folder>)")
    parser.add_argument("--skip-consolidate", action="store_true",
                       help="Skip the consolidation step (if already done)")
    parser.add_argument("--skip-comparison", action="store_true",
                       help="Skip individual scenario comparisons")
    parser.add_argument("--export-csv", action="store_true",
                       help="Export results to CSV format")
    
    args = parser.parse_args()
    
    # Validate experiment folder exists
    exp_folder_path = Path("experiment_results") / args.experiment_folder
    if not exp_folder_path.exists():
        print(f"\n✗ Error: Experiment folder not found: {exp_folder_path}")
        print(f"\nAvailable folders in experiment_results/:")
        exp_base = Path("experiment_results")
        if exp_base.exists():
            for folder in sorted(exp_base.iterdir()):
                if folder.is_dir():
                    print(f"  - {folder.name}")
        return 1
    
    # Set output directory to match experiment folder name if not specified
    if args.output_dir is None:
        args.output_dir = os.path.join("comparison_results", args.experiment_folder)
    
    print("\n" + "="*80)
    print("MASTER EVALUATION PIPELINE")
    print("="*80)
    print(f"Experiment Folder: {args.experiment_folder}")
    print(f"Use Case: {args.use_case.title()}")
    print(f"Scenarios: {', '.join(args.scenarios) if args.scenarios else 'All available'}")
    print(f"Output Directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    # Calculate total steps: consolidate (0 or 1) + comparison (0 or 1) + comprehensive (1)
    total_steps = 1  # Comprehensive analysis always runs
    if not args.skip_consolidate:
        total_steps += 1
    if not args.skip_comparison:
        total_steps += 1
    
    # Step 1: Consolidate results
    if not args.skip_consolidate:
        cmd = [
            "python", "Network_Simulation/consolidate_results.py", 
            "--use-case", args.use_case,
            "--experiment-folder", args.experiment_folder
        ]
        if args.scenarios:
            cmd.extend(["--scenarios"] + args.scenarios)
        
        if run_command(cmd, "STEP 1: Consolidating experiment results"):
            success_count += 1
    else:
        print("\n⊳ Skipping consolidation step")
    
    # Step 2: Compare protocols for each scenario (optional)
    if not args.skip_comparison:
        scenarios = args.scenarios if args.scenarios else ["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                                                          "congested_light", "congested_moderate", "congested_heavy"]
        
        for scenario in scenarios:
            cmd = [
                "python", "Network_Simulation/compare_protocols.py",
                "--use-case", args.use_case,
                "--scenarios", scenario,
                "--output-dir", args.output_dir,
                "--experiment-folder", args.experiment_folder
            ]
            
            run_command(cmd, f"STEP 2.{scenarios.index(scenario)+1}: Comparing protocols for {scenario} scenario")
        
        success_count += 1
    else:
        print("\n⊳ Skipping individual scenario comparisons")
    
    # Step 3: Comprehensive network scenario evaluation
    cmd = [
        "python", "Network_Simulation/evaluate_network_scenarios.py",
        "--use-case", args.use_case,
        "--output-dir", args.output_dir,
        "--experiment-folder", args.experiment_folder
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
