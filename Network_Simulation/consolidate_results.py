#!/usr/bin/env python3
"""
Extract and consolidate training results from experiment logs
"""
import json
import re
from pathlib import Path

def extract_results_from_logs(log_file):
    """Extract training results from server logs"""
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # Extract rounds and metrics
    rounds = []
    mse = []
    mae = []
    mape = []
    loss = []
    accuracy = []
    
    # Pattern for round metrics (handles both formats: "Round X -" and "Round X ")
    # Old format: Loss: X MSE: X MAE: X MAPE: X
    pattern = r'Round (\d+)\s*-?\s*Aggregated Metrics:\s+Loss: ([\d.]+)\s+MSE:\s+([\d.]+)\s+MAE:\s+([\d.]+)\s+MAPE:\s+([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        round_num, loss_val, mse_val, mae_val, mape_val = match
        rounds.append(int(round_num))
        loss.append(float(loss_val))
        mse.append(float(mse_val))
        mae.append(float(mae_val))
        mape.append(float(mape_val))
    
    # If old format not found, try newer format with Accuracy
    if not rounds:
        pattern_new = r'Round (\d+)\s*-?\s*Aggregated Metrics:\s+Loss:\s+([\d.]+)\s+Accuracy:\s+([\d.]+)%?'
        matches = re.findall(pattern_new, content, re.MULTILINE | re.DOTALL)
        for match in matches:
            round_num, loss_val, acc_val = match
            rounds.append(int(round_num))
            loss.append(float(loss_val))
            accuracy.append(float(acc_val) / 100.0 if float(acc_val) > 1 else float(acc_val))
    
    # Extract convergence time
    time_pattern = r'Time to Convergence: ([\d.]+) seconds \(([\d.]+) minutes\)'
    time_match = re.search(time_pattern, content)
    
    if time_match:
        conv_time_sec = float(time_match.group(1))
        conv_time_min = float(time_match.group(2))
    else:
        conv_time_sec = 0
        conv_time_min = 0
    
    # Extract compression technique metrics
    result = {
        "rounds": rounds,
        "loss": loss,
        "convergence_time_seconds": conv_time_sec,
        "convergence_time_minutes": conv_time_min,
        "total_rounds": len(rounds),
        "num_clients": 2
    }
    
    # Add optional metrics if found
    if mse:
        result["mse"] = mse
    if mae:
        result["mae"] = mae
    if mape:
        result["mape"] = mape
    if accuracy:
        result["accuracy"] = accuracy
    
    # Detect quantization metrics
    quantization_enabled = 'USE_QUANTIZATION' in content or 'Quantization' in content
    if quantization_enabled:
        # Extract quantization bits
        bits_pattern = r'QUANTIZATION_BITS[=:]\s*(\d+)'
        bits_match = re.search(bits_pattern, content)
        if bits_match:
            result["quantization_bits"] = int(bits_match.group(1))
        
        # Extract compression ratio
        compression_pattern = r'Compression\s+[Rr]atio[:\s]+([\d.]+)'
        compression_matches = re.findall(compression_pattern, content)
        if compression_matches:
            result["avg_quantization_compression"] = sum(float(c) for c in compression_matches) / len(compression_matches)
    
    # Detect pruning metrics
    pruning_enabled = 'USE_PRUNING' in content or 'Pruning' in content or 'sparsity' in content.lower()
    if pruning_enabled:
        # Extract sparsity
        sparsity_pattern = r'(?:Overall\s+)?Sparsity[:\s]+([\d.]+)%'
        sparsity_matches = re.findall(sparsity_pattern, content, re.IGNORECASE)
        if sparsity_matches:
            result["avg_sparsity_pct"] = sum(float(s) for s in sparsity_matches) / len(sparsity_matches)
        
        # Extract compression ratio from pruning
        pruning_compression_pattern = r'Compression\s+[Rr]atio[:\s]+([\d.]+)x?'
        pruning_compression_matches = re.findall(pruning_compression_pattern, content)
        if pruning_compression_matches:
            result["avg_pruning_compression"] = sum(float(c) for c in pruning_compression_matches) / len(pruning_compression_matches)
        
        # Extract communication savings
        comm_savings_pattern = r'(?:Comm.*?reduction|Size\s+reduction|Communication\s+[Ss]avings)[:\s]+([\d.]+)%'
        comm_savings_matches = re.findall(comm_savings_pattern, content, re.IGNORECASE)
        if comm_savings_matches:
            result["communication_savings_pct"] = sum(float(c) for c in comm_savings_matches) / len(comm_savings_matches)
    
    # Mark which compression techniques are enabled
    if quantization_enabled:
        result["uses_quantization"] = True
    if pruning_enabled:
        result["uses_pruning"] = True
    
    if not rounds:
        return None
    
    return result

def consolidate_results(use_case="temperature", scenarios=None, experiment_folder=None):
    """Consolidate all experiment results across all network scenarios"""
    if experiment_folder:
        exp_base = Path("experiment_results") / experiment_folder
    else:
        exp_base = Path("experiment_results")
    
    # Save results back to the experiment directory, not to Server directory
    # This prevents cross-contamination between experiments
    if experiment_folder:
        results_dir = Path("experiment_results") / experiment_folder
    else:
        results_dir = Path(f"Server/{use_case.title()}_Regulation/results")
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # All available network scenarios
    all_scenarios = ["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                    "congested_light", "congested_moderate", "congested_heavy"]
    if scenarios is None:
        scenarios = all_scenarios
    
    # Find all experiment directories
    all_results = {}  # {scenario: {protocol: results}}
    
    # If experiment_folder is specified, use it directly; otherwise search for matching folders
    if experiment_folder:
        exp_dirs = [exp_base] if exp_base.exists() else []
    else:
        exp_dirs = list(exp_base.glob(f"{use_case}_*"))
    
    for exp_dir in exp_dirs:
        # Find protocol subdirectories for all scenarios
        for scenario in scenarios:
            if scenario not in all_results:
                all_results[scenario] = {}
            
            # Match both standard format and congestion format
            # Standard: mqtt_excellent
            # Congestion: mqtt_excellent_congestion_moderate
            for proto_dir in exp_dir.glob(f"*_{scenario}*"):
                dir_name = proto_dir.name
                
                # Extract protocol name (first part before first underscore)
                protocol = dir_name.split('_')[0]
                
                # Check if this is a congestion experiment
                congestion_level = None
                if '_congestion_' in dir_name:
                    # Extract congestion level
                    parts = dir_name.split('_congestion_')
                    if len(parts) > 1:
                        congestion_level = parts[1]
                        # Create scenario key with congestion info
                        scenario_key = f"{scenario}_congestion_{congestion_level}"
                        if scenario_key not in all_results:
                            all_results[scenario_key] = {}
                        target_dict = all_results[scenario_key]
                    else:
                        target_dict = all_results[scenario]
                else:
                    # Standard scenario without congestion
                    target_dict = all_results[scenario]
                
                log_file = proto_dir / "server_logs.txt"
                
                if log_file.exists() and protocol not in target_dict:
                    scenario_label = scenario_key if congestion_level else scenario
                    print(f"Extracting {protocol.upper()} results from {proto_dir}...")
                    results = extract_results_from_logs(log_file)
                    
                    if results:
                        # Add scenario and congestion info to results
                        results["scenario"] = scenario
                        if congestion_level:
                            results["congestion_level"] = congestion_level
                        target_dict[protocol] = results
                        
                        # Save JSON directly in the protocol directory (not in a separate results folder)
                        # Use scenario-specific filename
                        if congestion_level:
                            output_file = proto_dir / f"{protocol}_{scenario}_congestion_{congestion_level}_training_results.json"
                        else:
                            output_file = proto_dir / f"{protocol}_{scenario}_training_results.json"
                        
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        print(f"  ✓ Saved to {output_file}")
                    else:
                        print(f"  ✗ No results found in logs")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Consolidated Results Summary")
    print(f"{'='*80}")
    
    # Group scenarios by base scenario and congestion
    for base_scenario in ["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                         "congested_light", "congested_moderate", "congested_heavy"]:
        # Check base scenario
        if base_scenario in all_results and all_results[base_scenario]:
            print(f"\n{base_scenario.upper().replace('_', ' ')} Network Scenario:")
            for protocol in sorted(all_results[base_scenario].keys()):
                results = all_results[base_scenario][protocol]
                congestion_info = f" (no congestion)" if "congestion_level" not in results else ""
                
                # Build compression info string
                compression_info = []
                if results.get("uses_quantization"):
                    bits = results.get("quantization_bits", "?")
                    compression_info.append(f"Q:{bits}bit")
                if results.get("uses_pruning"):
                    sparsity = results.get("avg_sparsity_pct", 0)
                    compression_info.append(f"P:{sparsity:.1f}%")
                
                compression_str = f" [{', '.join(compression_info)}]" if compression_info else ""
                
                print(f"  - {protocol.upper()}: {results['total_rounds']} rounds, "
                      f"{results['convergence_time_seconds']:.2f}s{congestion_info}{compression_str}")
        
        # Check for congestion variants
        congestion_keys = [k for k in all_results.keys() if k.startswith(f"{base_scenario}_congestion_")]
        for scenario_key in sorted(congestion_keys):
            if all_results[scenario_key]:
                congestion_level = scenario_key.split('_congestion_')[1]
                print(f"  └─ WITH {congestion_level.upper()} CONGESTION:")
                for protocol in sorted(all_results[scenario_key].keys()):
                    results = all_results[scenario_key][protocol]
                    
                    # Build compression info string
                    compression_info = []
                    if results.get("uses_quantization"):
                        bits = results.get("quantization_bits", "?")
                        compression_info.append(f"Q:{bits}bit")
                    if results.get("uses_pruning"):
                        sparsity = results.get("avg_sparsity_pct", 0)
                        compression_info.append(f"P:{sparsity:.1f}%")
                    
                    compression_str = f" [{', '.join(compression_info)}]" if compression_info else ""
                    
                    print(f"     - {protocol.upper()}: {results['total_rounds']} rounds, "
                          f"{results['convergence_time_seconds']:.2f}s{compression_str}")
    
    print(f"\n{'='*80}")
    
    return all_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Consolidate experiment results")
    parser.add_argument("--use-case", default="temperature", 
                       choices=["temperature", "emotion", "mentalstate"])
    parser.add_argument("--scenarios", nargs="+",
                       choices=["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                               "congested_light", "congested_moderate", "congested_heavy"],
                       help="Specific scenarios to consolidate (default: all)")
    parser.add_argument("--experiment-folder", 
                       help="Name of specific experiment folder in experiment_results/ to process")
    args = parser.parse_args()
    
    consolidate_results(args.use_case, args.scenarios, args.experiment_folder)
