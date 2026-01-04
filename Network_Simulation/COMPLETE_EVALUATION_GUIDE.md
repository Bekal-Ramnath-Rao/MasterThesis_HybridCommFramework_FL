# Complete Protocol Evaluation Guide with Network Simulation

This guide walks you through the complete process of evaluating all FL protocols across different network conditions.

## 🎯 Overview

You will:
1. Prepare Docker environment for network simulation
2. Run automated experiments across all protocols and network conditions
3. Collect and analyze results

## 📝 Step-by-Step Setup

### Step 1: Update Dockerfiles (Already Done! ✓)

The Server and Client Dockerfiles now include `iproute2` package for traffic control.

### Step 2: Add NET_ADMIN Capability to Docker Compose Files

You need to add `cap_add: [NET_ADMIN]` to all FL services. Choose one method:

#### Method A: Manual Edit (Recommended for precision)

Follow the guide in [HOWTO_ADD_NET_ADMIN.md](HOWTO_ADD_NET_ADMIN.md)

For each FL service in all three docker-compose files, add:
```yaml
cap_add:
  - NET_ADMIN
```

#### Method B: PowerShell Script (Quick but verify after)

```powershell
# Run from project root
$files = @(
    "docker-compose-emotion.yml",
    "docker-compose-mentalstate.yml",
    "docker-compose-temperature.yml"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        $lines = Get-Content $file
        $newLines = @()
        
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $newLines += $lines[$i]
            
            # If line contains "container_name: fl-" and next line isn't cap_add
            if ($lines[$i] -match "container_name: fl-" -and 
                $i+1 -lt $lines.Count -and 
                $lines[$i+1] -notmatch "cap_add:") {
                $newLines += "    cap_add:"
                $newLines += "      - NET_ADMIN"
            }
        }
        
        Set-Content $file $newLines
        Write-Host "✓ Updated $file"
    }
}
```

### Step 3: Rebuild Docker Images

```powershell
# Rebuild with new iproute2 package
docker-compose -f docker-compose-emotion.yml build
docker-compose -f docker-compose-mentalstate.yml build  
docker-compose -f docker-compose-temperature.yml build
```

This may take 5-10 minutes.

### Step 4: Verify Setup

Test that network simulation works:

```powershell
# Start a test service
docker-compose -f docker-compose-emotion.yml up -d fl-server-mqtt-emotion fl-client-mqtt-emotion-1

# Wait a few seconds
Start-Sleep -Seconds 5

# Try to apply network conditions
python network_simulator.py --scenario moderate --pattern fl-client-mqtt

# Check if it worked
docker exec fl-client-mqtt-emotion-1 tc qdisc show dev eth0

# Should show something like: "qdisc netem ... delay 50.0ms"

# Clean up
docker-compose -f docker-compose-emotion.yml down
```

## 🚀 Running Experiments

### Quick Test (Single Experiment)

Test one protocol under one network condition:

```powershell
# Test MQTT under poor network conditions (Emotion Recognition)
python run_network_experiments.py `
    --single `
    --protocol mqtt `
    --scenario poor `
    --use-case emotion `
    --rounds 5
```

This takes ~10-30 minutes depending on your data and model.

### Full Evaluation (All Protocols × All Networks)

Run comprehensive evaluation:

```powershell
# All protocols × All network conditions for Emotion Recognition
python run_network_experiments.py `
    --use-case emotion `
    --rounds 10

# Expected duration: 5-10 hours (5 protocols × 6 scenarios × ~10-30 min each)
```

### Subset Evaluation

Test specific protocols or scenarios:

```powershell
# Test only MQTT and gRPC
python run_network_experiments.py `
    --protocols mqtt grpc `
    --use-case emotion `
    --rounds 10

# Test only challenging network conditions
python run_network_experiments.py `
    --scenarios poor very_poor satellite `
    --use-case emotion `
    --rounds 10

# Combination
python run_network_experiments.py `
    --protocols mqtt amqp grpc `
    --scenarios moderate poor `
    --use-case emotion `
    --rounds 5
```

## 📊 Understanding Results

### Results Directory Structure

```
experiment_results/
└── emotion_20250130_143022/
    ├── mqtt_excellent/
    │   ├── metadata.json           # Experiment configuration
    │   ├── server_logs.txt         # Complete server logs
    │   └── mqtt_training_results.json  # Performance metrics
    ├── mqtt_good/
    │   └── ...
    ├── mqtt_moderate/
    │   └── ...
    ├── amqp_excellent/
    │   └── ...
    └── ...
```

### Key Metrics in Results Files

```json
{
  "protocol": "mqtt",
  "total_rounds": 10,
  "total_time": 245.67,          // seconds
  "avg_round_time": 24.57,       // seconds per round
  "final_accuracy": 0.85,        // model performance
  "convergence_round": 8,        // round where model converged
  "total_bytes_sent": 15728640,  // communication overhead
  "messages_sent": 156
}
```

### Analyzing Results

Create a simple analysis script:

```python
# analyze_results.py
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_experiments(results_dir):
    results = []
    
    for exp_dir in Path(results_dir).iterdir():
        if exp_dir.is_dir():
            metadata_file = exp_dir / "metadata.json"
            results_file = list(exp_dir.glob("*_training_results.json"))
            
            if metadata_file.exists() and results_file:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                
                with open(results_file[0]) as f:
                    results_data = json.load(f)
                
                results.append({
                    "Protocol": metadata["protocol"],
                    "Network": metadata["scenario"],
                    "Avg Round Time (s)": results_data.get("avg_round_time", 0),
                    "Total Time (s)": results_data.get("total_time", 0),
                    "Final Accuracy": results_data.get("final_accuracy", 0),
                    "Total Bytes": results_data.get("total_bytes_sent", 0)
                })
    
    df = pd.DataFrame(results)
    
    # Pivot table for comparison
    pivot = df.pivot_table(
        values="Avg Round Time (s)",
        index="Protocol",
        columns="Network"
    )
    
    print("\nAverage Round Time by Protocol and Network Condition:")
    print(pivot)
    
    # Plot
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title("Protocol Performance Across Network Conditions")
    plt.ylabel("Avg Round Time (seconds)")
    plt.xlabel("Protocol")
    plt.xticks(rotation=45)
    plt.legend(title="Network Scenario")
    plt.tight_layout()
    plt.savefig(Path(results_dir) / "analysis.png")
    print(f"\nPlot saved to: {Path(results_dir) / 'analysis.png'}")
    
    return df

# Usage
df = analyze_experiments("experiment_results/emotion_20250130_143022")
```

## 🔍 Manual Network Simulation

For more control, manually apply network conditions:

```powershell
# Start containers
docker-compose -f docker-compose-emotion.yml up -d mqtt-broker fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2

# Apply specific network scenario
python network_simulator.py --scenario poor --pattern fl-

# Monitor progress
docker logs -f fl-server-mqtt-emotion

# When done, collect results
docker cp fl-server-mqtt-emotion:/app/Server/Emotion_Recognition/results ./manual_results/

# Clean up
python network_simulator.py --reset
docker-compose -f docker-compose-emotion.yml down
```

### Custom Network Conditions

```powershell
# Apply custom conditions
python network_simulator.py `
    --custom `
    --latency 75ms `
    --jitter 25ms `
    --bandwidth 5mbit `
    --loss 2% `
    --pattern fl-client
```

### Different Conditions for Different Clients (Asymmetric)

```powershell
# Client 1: Good network
python network_simulator.py --scenario good --pattern fl-client-mqtt-emotion-1

# Client 2: Poor network
python network_simulator.py --scenario poor --pattern fl-client-mqtt-emotion-2
```

## 📈 Experiment Design Recommendations

### Minimal Evaluation (Quick)
- Protocols: MQTT, gRPC, QUIC
- Scenarios: good, moderate, poor
- Rounds: 5
- **Time: ~3-4 hours**

### Standard Evaluation (Recommended)
- Protocols: All 5 (MQTT, AMQP, gRPC, QUIC, DDS)
- Scenarios: excellent, good, moderate, poor
- Rounds: 10
- **Time: ~8-12 hours**

### Comprehensive Evaluation (Research)
- Protocols: All 5
- Scenarios: All 6
- Rounds: 20-50 (until convergence)
- **Time: 20-40 hours**

### Running Overnight

```powershell
# Start comprehensive evaluation before leaving
python run_network_experiments.py `
    --use-case emotion `
    --rounds 20 `
    > experiment_log.txt 2>&1 &

# Or use Windows Task Scheduler / nohup equivalent
```

## 🛠️ Troubleshooting

### "Permission denied" when applying network conditions

**Issue**: Container doesn't have NET_ADMIN capability

**Solution**: 
1. Verify `cap_add: [NET_ADMIN]` in docker-compose file
2. Rebuild and restart containers

### "tc command not found"

**Issue**: iproute2 not installed in container

**Solution**:
1. Verify Dockerfiles include `iproute2`
2. Rebuild images: `docker-compose build`

### Experiments timing out

**Issue**: Network conditions too severe or model not converging

**Solution**:
- Reduce packet loss: `--loss 1%` instead of `--loss 5%`
- Increase bandwidth: `--bandwidth 5mbit` instead of `--bandwidth 1mbit`
- Increase timeout in `run_network_experiments.py` (edit timeout parameter)

### Containers crash during simulation

**Issue**: Out of memory or network too degraded

**Solution**:
- Check Docker resource limits
- Use less severe network scenarios
- Reduce number of parallel clients

## 📋 Checklist

Before running full experiments:

- [ ] Updated both Dockerfiles with iproute2
- [ ] Added cap_add: NET_ADMIN to all FL services in docker-compose files
- [ ] Rebuilt all Docker images
- [ ] Verified network simulation works (test command succeeded)
- [ ] Tested one single experiment successfully
- [ ] Have adequate disk space for results (~1-5 GB)
- [ ] Planned for experiment duration (hours to days)

## 🎓 What You'll Learn

From these experiments, you can determine:

1. **Protocol Efficiency**: Which protocol has lowest overhead?
2. **Latency Tolerance**: Which handles high latency best?
3. **Packet Loss Resilience**: Which recovers from losses?
4. **Bandwidth Requirements**: Minimum bandwidth for each protocol?
5. **Convergence Impact**: Do network conditions affect model accuracy?
6. **Trade-offs**: Speed vs reliability vs resource usage

## 📚 Next Steps

1. Run experiments as outlined above
2. Analyze results using the provided scripts
3. Create visualizations comparing protocols
4. Write up findings in your thesis
5. Consider publishing results!

Good luck with your Master's Thesis! 🎓
