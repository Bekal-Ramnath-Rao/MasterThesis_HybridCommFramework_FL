# CycloneDDS Unicast Discovery Configuration

## Overview

This document describes the configuration changes made to switch CycloneDDS from **multicast discovery** (default) to **unicast discovery** to improve performance in poor and very poor network conditions.

## Problem Statement

CycloneDDS uses multicast discovery by default for discovering DDS participants on the network. While this works well in good network conditions, multicast can cause significant delays in degraded network scenarios:

- **Poor Network**: High latency, packet loss can cause multicast discovery timeouts
- **Very Poor Network**: Multicast packets may be dropped or delayed extensively
- **Network Isolation**: Docker bridge networks may not properly propagate multicast traffic

## Solution

Switched to **unicast discovery** with static peer configuration:

1. **Disabled Multicast**: Set `AllowMulticast=false` in CycloneDDS configuration
2. **Static Peer List**: Configured known peers (server and clients) for direct unicast communication
3. **Container Integration**: All DDS services now mount and use the unicast configuration

## Configuration File

**Location**: `cyclonedds-unicast.xml`

**Key Settings**:
```xml
<General>
    <AllowMulticast>false</AllowMulticast>
    <NetworkInterfaceAddress>auto</NetworkInterfaceAddress>
</General>

<Discovery>
    <ParticipantIndex>auto</ParticipantIndex>
    <Peers>
        <Peer address="fl-server-dds-emotion"/>
        <Peer address="fl-client-dds-emotion-1"/>
        <Peer address="fl-client-dds-emotion-2"/>
        <!-- ... additional peers for temperature and mentalstate -->
    </Peers>
</Discovery>
```

## Docker-Compose Changes

All DDS services in the following files have been updated:
- `Docker/docker-compose-emotion.yml`
- `Docker/docker-compose-temperature.yml`
- `Docker/docker-compose-mentalstate.yml`

**Changes Applied to Each DDS Service**:

1. **Environment Variable**: Added `CYCLONEDDS_URI=file:///app/cyclonedds-unicast.xml`
2. **Volume Mount**: Added `../cyclonedds-unicast.xml:/app/cyclonedds-unicast.xml:ro`

**Example**:
```yaml
fl-server-dds-emotion:
  environment:
    - DDS_DOMAIN_ID=0
    - NUM_CLIENTS=2
    - NUM_ROUNDS=1000
    - CYCLONEDDS_URI=file:///app/cyclonedds-unicast.xml  # NEW
  volumes:
    - ../Server/Emotion_Recognition/results:/app/Server/Emotion_Recognition/results
    - ../cyclonedds-unicast.xml:/app/cyclonedds-unicast.xml:ro  # NEW
```

## Benefits

### 1. **Reduced Discovery Time**
- Direct unicast communication to known peers
- No waiting for multicast discovery timeouts
- Faster participant detection in poor network conditions

### 2. **Better Performance in Degraded Networks**
- Unicast is more reliable than multicast in lossy networks
- Direct peer-to-peer communication reduces hops
- Better suited for Docker bridge networks

### 3. **Predictable Behavior**
- No dependency on multicast routing
- Static peer list eliminates discovery uncertainty
- Consistent performance across network conditions

### 4. **Network Isolation Compatibility**
- Works better in containerized environments
- No multicast routing requirements
- Explicit peer addressing

## Expected Impact on Network Scenarios

| Network Scenario | Before (Multicast) | After (Unicast) |
|-----------------|-------------------|-----------------|
| **Good** | Fast discovery | Similar performance |
| **Average** | Moderate delays | Improved consistency |
| **Poor** | Significant delays | Much faster discovery |
| **Very Poor** | Frequent timeouts | Reliable discovery |

## Verification

### Check Configuration File
```bash
cat cyclonedds-unicast.xml | grep AllowMulticast
# Should show: <AllowMulticast>false</AllowMulticast>
```

### Check Docker-Compose Files
```bash
./verify_cyclonedds_unicast.sh
```

### Verify in Running Container
```bash
# Check environment variable
docker exec fl-server-dds-emotion env | grep CYCLONEDDS_URI

# Check mounted config file
docker exec fl-server-dds-emotion cat /app/cyclonedds-unicast.xml
```

### Test Discovery Performance
Run experiments with poor/very_poor network scenarios and compare:
- Discovery time (time until all participants are detected)
- Initial round latency
- Overall communication reliability

## Rollback Instructions

To revert to multicast discovery:

1. **Remove environment variable** from docker-compose files:
   ```yaml
   # Remove or comment out:
   - CYCLONEDDS_URI=file:///app/cyclonedds-unicast.xml
   ```

2. **Remove volume mount**:
   ```yaml
   # Remove or comment out:
   - ../cyclonedds-unicast.xml:/app/cyclonedds-unicast.xml:ro
   ```

3. **Rebuild containers**:
   ```bash
   docker-compose -f Docker/docker-compose-emotion.yml down
   docker-compose -f Docker/docker-compose-emotion.yml up --build
   ```

## Additional Configuration Options

### Adjust Discovery Timeout
If discovery is still slow, you can adjust timeout settings:
```xml
<Discovery>
    <SPDPInterval>100ms</SPDPInterval>
    <MaxAutoParticipantIndex>100</MaxAutoParticipantIndex>
</Discovery>
```

### Enable Debug Logging
To troubleshoot discovery issues:
```xml
<Tracing>
    <Verbosity>trace</Verbosity>
    <OutputFile>stderr</OutputFile>
</Tracing>
```

### Network-Specific Peer Configuration
For different network setups, you can create scenario-specific peer lists:
```xml
<Discovery>
    <Peers>
        <!-- Only include peers for the specific experiment -->
        <Peer address="fl-server-dds-emotion"/>
        <Peer address="fl-client-dds-emotion-1"/>
        <Peer address="fl-client-dds-emotion-2"/>
    </Peers>
</Discovery>
```

## References

- [CycloneDDS Configuration Guide](https://github.com/eclipse-cyclonedds/cyclonedds)
- [DDS Discovery Mechanisms](https://www.omg.org/spec/DDSI-RTPS/)
- [Docker Network Multicast Limitations](https://docs.docker.com/network/bridge/)

## Testing Recommendations

1. **Baseline Test**: Run experiments in good network conditions with both configs
2. **Poor Network Test**: Compare discovery times in poor network scenarios
3. **Very Poor Network Test**: Measure reliability and timeout rates
4. **Mixed Scenarios**: Test across all network conditions to verify improvement

## Notes

- The static peer list includes all possible DDS participants
- Each deployment only uses the peers relevant to its domain ID
- Unicast discovery is generally recommended for containerized deployments
- This configuration is compatible with all CycloneDDS QoS settings
