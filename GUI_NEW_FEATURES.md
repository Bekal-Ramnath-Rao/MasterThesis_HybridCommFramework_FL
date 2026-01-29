# FL Experiment GUI - New Features Guide

## 🎯 Three Major Enhancements

### 1. **Baseline Mode Selection** 

#### What is it?
A special experiment mode that creates reference models with optimal network conditions for comparison purposes.

#### Location
**Basic Configuration Tab** → Top section

#### Features
- 🎯 **Baseline Mode Checkbox**: "Create Baseline Model (Excellent Network, GPU Required)"
- When enabled:
  - **GPU**: Automatically enabled and locked (required for baselines)
  - **Network Controls**: Entire Network Control tab disabled
  - **Network Scenario**: Not required (uses excellent network automatically)
  - **Output Directory**: Saves to `experiment_results_baseline/`

#### Why Use It?
Baseline models serve as reference points for comparing:
- Different network conditions
- Different protocols under stress
- Impact of quantization/compression

#### Usage Example
```
1. Check "Create Baseline Model" checkbox
2. Notice:
   - GPU checkbox becomes checked and locked
   - Network Control tab becomes disabled (grayed out)
3. Select protocol(s) - e.g., MQTT, gRPC
4. Select use case - e.g., Emotion Recognition
5. Start experiment
6. Results saved to: experiment_results_baseline/emotion/mqtt_baseline/
```

#### Validation
- GPU requirement enforced (error if unchecked)
- Network scenarios not required
- Confirmation dialog before starting
- Clear visual feedback in output console

---

### 2. **Multi-Client Log Viewer**

#### What is it?
Individual log viewing for each federated learning client container.

#### Location
**Client Logs Tab** → Top toolbar

#### Features
- 🖥️ **Client Selector Dropdown**: "Select Client:"
  - Auto-populated with detected clients
  - Format: "🖥️ Client 1", "🖥️ Client 2", etc.
- 🔄 **Refresh Button**: Updates client list dynamically
- **Real-time Log Streaming**: Each client's logs in separate view

#### How It Works
1. **Auto-Detection**: After experiment starts, GUI detects running client containers
2. **Dynamic Population**: Dropdown fills with available clients
3. **Switch Clients**: Select different client from dropdown to view their logs
4. **Separate Log Monitors**: Each client has independent log stream

#### Usage Example
```
1. Start experiment with 3 clients
2. Wait 5-7 seconds for containers to start
3. Go to "Client Logs" tab
4. Click "Refresh" button
5. Dropdown now shows:
   - 🖥️ Client 1
   - 🖥️ Client 2  
   - 🖥️ Client 3
6. Select "Client 2" to view its specific logs
7. Switch to "Client 1" to compare
```

#### Auto-Refresh
- Clients detected automatically 7 seconds after experiment start
- Manual refresh available anytime via "🔄 Refresh" button
- Works with any number of clients (1-10+)

---

### 3. **Per-Client/Server Network Control**

#### What is it?
Granular network condition control targeting specific clients or server.

#### Location
**Network Control Tab** → Top section (new "Network Control Target" group)

#### Features
- 🎯 **Target Selector Dropdown**: "Apply Network Conditions To:"
  - **All Clients**: Apply to all federated learning clients
  - **Server**: Apply only to FL server
  - **Client 1, 2, 3...**: Apply to specific individual client
- 🔄 **Refresh Targets Button**: Updates available targets
- **Independent Controls**: Each target gets separate network conditions

#### How It Works
1. **Target Selection**: Choose which component to affect
2. **Apply Conditions**: Latency, bandwidth, jitter, packet loss
3. **fl_network_monitor Integration**: Uses `--client-id`, `--server`, or `--all` flags
4. **Simulates Real Scenarios**: 
   - Client 1 in excellent network
   - Client 2 with high latency
   - Server with limited bandwidth

#### Usage Example

**Scenario 1: Test Client-Specific Poor Network**
```
1. Start experiment
2. Network Control Tab → Select "Client 2"
3. Set:
   - Latency: 200ms
   - Packet Loss: 5%
4. Click "Apply Network Conditions"
5. Only Client 2 affected, others remain normal
```

**Scenario 2: Bottleneck at Server**
```
1. Select target: "Server"
2. Set:
   - Bandwidth: 10 Mbps (limited)
   - Latency: 100ms
3. Apply conditions
4. Server becomes bottleneck, clients unaffected
```

**Scenario 3: Heterogeneous Network**
```
1. Apply different conditions to each client:
   - Client 1: Excellent (no changes)
   - Client 2: Moderate latency (100ms)
   - Client 3: Poor network (500ms, 10% loss)
2. Simulate real-world diverse edge devices
```

#### Auto-Detection
- Client containers detected from Docker
- Updates 7 seconds after experiment start
- Manual refresh available anytime
- Shows only running containers

---

## 🔄 Integration Between Features

### Baseline Mode + Network Control
- **Baseline mode DISABLES network control tab**
- Rationale: Baselines must have excellent network
- If you try to apply network changes in baseline mode:
  ```
  ⚠️ Warning: "Network conditions cannot be changed in baseline mode."
  ```

### Multi-Client Logs + Network Control
- **Use together for debugging**
- Example workflow:
  1. Apply poor network to Client 2
  2. Switch to Client 2 logs
  3. Observe timeout/retry messages
  4. Compare with Client 1 (excellent network) logs

### Baseline + Multi-Client
- Baselines typically use fewer clients (1-2)
- Client logs still work
- Useful for verifying GPU usage

---

## 🎨 Visual Indicators

### Baseline Mode Active
```
Output Console:
⚠️ BASELINE MODE ENABLED:
  • GPU: FORCED ON (required for baseline)
  • Network Controls: DISABLED (excellent network conditions)
  • Baseline models will be stored in experiment_results_baseline/
```

### Network Target Applied
```
Output Console:
🌐 Applying network conditions to Client 2:
  Latency: 200ms
  Bandwidth: 50Mbps
  Jitter: 10ms
  Packet Loss: 5%

✅ Network conditions applied to client_2
```

### Client Detection
```
Output Console:
✅ Detected 3 client container(s)
✅ Detected 3 client(s) for log viewing
```

---

## 🚀 Quick Start Examples

### Example 1: Create Baseline Then Compare
```
Step 1: Create Baseline
- Check "Create Baseline Model"
- Protocol: MQTT
- Use Case: Emotion
- Start → Creates baseline

Step 2: Run Network Experiment
- Uncheck "Create Baseline Model"
- Same protocol/use case
- Network Scenario: Poor Network
- Start → Compares against baseline in FL Training Monitor tab
```

### Example 2: Simulate Edge Device Heterogeneity
```
Step 1: Start Experiment (3 clients)
Step 2: Apply Different Conditions
- Client 1: No changes (edge device with good connection)
- Client 2: 100ms latency (mobile device)
- Client 3: 500ms latency, 10% loss (IoT device on satellite)

Step 3: Monitor Each Client
- Switch between client logs
- Observe different training speeds
- Compare aggregation performance
```

### Example 3: Bottleneck Testing
```
Test 1: Server Bottleneck
- Target: Server
- Bandwidth: 10 Mbps
- Result: All clients wait for slow server

Test 2: Client Bottleneck
- Target: Client 1
- Bandwidth: 5 Mbps
- Result: One slow client delays aggregation
```

---

## 🛠️ Technical Details

### Baseline Mode Implementation
- Uses `run_baseline_experiments.py` instead of `run_network_experiments.py`
- Skips network scenario selection
- Forces GPU enable
- Disables Network Control tab (index 1)

### Multi-Client Implementation
- Docker container detection: `docker ps --filter name=client`
- Separate `LogMonitor` thread per client
- Dynamic switching via `switch_client_log()` method
- Auto-refresh 7 seconds after experiment start

### Network Target Implementation
- Extended `NetworkController` with `target` parameter
- Command building:
  - `--all`: All clients
  - `--server`: Server only
  - `--client-id N`: Specific client
- Container name parsing for client ID extraction

---

## 📝 Notes & Best Practices

### Baseline Mode
✅ **Do:**
- Create baselines first before network experiments
- Use same hyperparameters for fair comparison
- Store baselines for each use case separately

❌ **Don't:**
- Try to apply network conditions in baseline mode
- Disable GPU in baseline mode
- Delete baseline results (needed for comparison)

### Multi-Client Logs
✅ **Do:**
- Refresh client list if containers were restarted
- Check logs when debugging client-specific issues
- Use with network control for cause-effect analysis

❌ **Don't:**
- Expect logs before containers start (wait 5-7 seconds)
- Assume client numbers match container IDs (use dropdown names)

### Network Control Targeting
✅ **Do:**
- Test one client at a time first
- Use "All Clients" for uniform conditions
- Refresh targets if containers change
- Combine with client logs for verification

❌ **Don't:**
- Apply extreme conditions without testing
- Forget to refresh after restarting containers
- Apply conditions in baseline mode

---

## 🐛 Troubleshooting

### "No clients detected"
**Cause**: Containers not yet started or stopped  
**Solution**: 
1. Wait 5-7 seconds after starting experiment
2. Click "🔄 Refresh" button
3. Verify containers running: `docker ps --filter name=client`

### Network conditions not applied
**Cause**: Baseline mode active  
**Solution**: Uncheck "Create Baseline Model" checkbox

### Client logs empty
**Cause**: Wrong client selected or container stopped  
**Solution**:
1. Check client selector dropdown
2. Refresh client list
3. Verify container exists: `docker logs <container_name>`

### GPU locked in baseline mode
**Cause**: Intended behavior - baselines require GPU  
**Solution**: This is correct. Uncheck baseline mode if you don't want forced GPU.

---

## 📊 File Structure Impact

### Baseline Mode Output
```
experiment_results_baseline/
├── emotion/
│   ├── mqtt_baseline/
│   │   ├── model.pth
│   │   └── metrics.json
│   └── grpc_baseline/
│       └── ...
├── mentalstate/
│   └── ...
└── temperature/
    └── ...
```

### Normal Experiment Output
```
experiment_results/
├── emotion/
│   ├── mqtt_excellent/
│   ├── mqtt_poor/
│   └── ...
```

---

## 🎓 Learning Resources

### Understanding Baselines
- Baselines = optimal performance reference
- Used for comparison in FL Training Monitor tab
- Critical for evaluating impact of network conditions

### Network Control Strategies
1. **Uniform**: All clients same conditions (testing protocol robustness)
2. **Heterogeneous**: Different per client (realistic edge scenarios)
3. **Server Bottleneck**: Test aggregation performance
4. **Client Bottleneck**: Test stragglers impact

### Monitoring Strategy
1. Start experiment
2. Wait for auto-detection (7 sec)
3. Check FL Training Monitor for aggregation progress
4. Switch between client logs for individual behavior
5. Apply targeted network changes if needed
6. Observe impact in real-time

---

**Version**: 2.0  
**Last Updated**: 2024  
**Compatibility**: PyQt5 5.15+, Docker 20.10+, Python 3.7+
