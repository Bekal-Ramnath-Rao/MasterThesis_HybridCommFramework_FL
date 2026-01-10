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
    
    # Pattern for round metrics (handles both formats: "Round X -" and "Round X ")
    pattern = r'Round (\d+)\s*-?\s*Aggregated Metrics:\s+Loss: ([\d.]+)\s+MSE:\s+([\d.]+)\s+MAE:\s+([\d.]+)\s+MAPE:\s+([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
    
    for match in matches:
        round_num, loss_val, mse_val, mae_val, mape_val = match
        rounds.append(int(round_num))
        loss.append(float(loss_val))
        mse.append(float(mse_val))
        mae.append(float(mae_val))
        mape.append(float(mape_val))
    
    # Extract convergence time
    time_pattern = r'Time to Convergence: ([\d.]+) seconds \(([\d.]+) minutes\)'
    time_match = re.search(time_pattern, content)
    
    if time_match:
        conv_time_sec = float(time_match.group(1))
        conv_time_min = float(time_match.group(2))
    else:
        conv_time_sec = 0
        conv_time_min = 0
    
    if not rounds:
        return None
    
    return {
        "rounds": rounds,
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "loss": loss,
        "convergence_time_seconds": conv_time_sec,
        "convergence_time_minutes": conv_time_min,
        "total_rounds": len(rounds),
        "num_clients": 2
    }

def consolidate_results(use_case="temperature", scenarios=None):
    """Consolidate all experiment results across all network scenarios"""
    exp_base = Path("experiment_results")
    results_dir = Path(f"Server/{use_case.title()}_Regulation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # All available network scenarios
    all_scenarios = ["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                    "congested_light", "congested_moderate", "congested_heavy"]
    if scenarios is None:
        scenarios = all_scenarios
    
    # Find all experiment directories
    all_results = {}  # {scenario: {protocol: results}}
    
    for exp_dir in exp_base.glob(f"{use_case}_*"):
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
                        
                        # Save to results directory with scenario and congestion in filename
                        if congestion_level:
                            output_file = results_dir / f"{protocol}_{scenario}_congestion_{congestion_level}_training_results.json"
                        else:
                            output_file = results_dir / f"{protocol}_{scenario}_training_results.json"
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
                print(f"  - {protocol.upper()}: {results['total_rounds']} rounds, "
                      f"{results['convergence_time_seconds']:.2f}s{congestion_info}")
        
        # Check for congestion variants
        congestion_keys = [k for k in all_results.keys() if k.startswith(f"{base_scenario}_congestion_")]
        for scenario_key in sorted(congestion_keys):
            if all_results[scenario_key]:
                congestion_level = scenario_key.split('_congestion_')[1]
                print(f"  └─ WITH {congestion_level.upper()} CONGESTION:")
                for protocol in sorted(all_results[scenario_key].keys()):
                    results = all_results[scenario_key][protocol]
                    print(f"     - {protocol.upper()}: {results['total_rounds']} rounds, "
                          f"{results['convergence_time_seconds']:.2f}s")
    
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
    args = parser.parse_args()
    
    consolidate_results(args.use_case, args.scenarios)
