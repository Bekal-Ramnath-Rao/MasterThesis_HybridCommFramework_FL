# Troubleshooting DDS (CycloneDDS) Connection Issues

## Quick Diagnostics Checklist

- [ ] CycloneDDS library installed correctly
- [ ] DLL path configured in code
- [ ] All participants using same Domain ID
- [ ] Multicast enabled on network
- [ ] Firewall allows UDP ports 7400-7500
- [ ] Peers are on same subnet
- [ ] QoS policies match between readers/writers

---

## Step 1: Verify CycloneDDS Installation

### Check Python Bindings

```powershell
python -c "from cyclonedds.domain import DomainParticipant; print('✓ CycloneDDS Python bindings OK')"
```

**If Error:**
```
ModuleNotFoundError: No module named 'cyclonedds'
```

**Solution:**
```powershell
pip install cyclonedds
```

### Check Native DLL (Windows)

```powershell
# Default vcpkg location
$dllPath = "C:\Masters_Infotech\Semester_5\MT_SW_Addons\vcpkg\buildtrees\cyclonedds\x64-windows-rel\bin"

# Check if DLL exists
Test-Path "$dllPath\ddsc.dll"

# Should return: True
```

**If DLL Not Found:**

1. **Find your installation:**
   ```powershell
   Get-ChildItem -Path C:\ -Filter "ddsc.dll" -Recurse -ErrorAction SilentlyContinue | Select-Object FullName
   ```

2. **Update paths in code:**
   ```python
   # In FL_Server_DDS.py and FL_Client_DDS.py
   cyclone_path = r"C:\YOUR\PATH\TO\cyclonedds\bin"
   ```

### Verify Installation

```powershell
# Should work without errors
python -c "from cyclonedds.domain import DomainParticipant; dp = DomainParticipant(0); print('✓ DDS Working'); dp = None"
```

---

## Step 2: Common DDS Errors

### 1. CycloneDDSLoaderException

**Error:**
```
CycloneDDSLoaderException: Could not load DLL: ddsc.dll
```

**Causes:**
- DLL not in PATH
- Wrong DLL architecture (x86 vs x64)
- Missing dependencies

**Solutions:**

1. **Add to PATH manually:**
   ```powershell
   $env:PATH = "C:\path\to\cyclonedds\bin;" + $env:PATH
   python FL_Server_DDS.py
   ```

2. **Check code has PATH setup:**
   ```python
   # At top of FL_Server_DDS.py and FL_Client_DDS.py
   import os
   cyclone_path = r"C:\path\to\cyclonedds\bin"
   if cyclone_path not in os.environ.get('PATH', ''):
       os.environ['PATH'] = cyclone_path + os.pathsep + os.environ.get('PATH', '')
   ```

3. **Install system-wide (alternative):**
   - Add DLL path to Windows System PATH
   - Restart terminal

### 2. No Peer Discovery

**Symptom:**
```
Server: Waiting for clients to register...
Client: Waiting for server status... (never connects)
```

**Debug:**

```python
# Add to both server and client after setup_dds()
print(f"Domain ID: {DDS_DOMAIN_ID}")
print(f"Participants: {self.participant.guid}")
time.sleep(5)  # Wait for discovery
```

**Causes and Solutions:**

#### A. Different Domain IDs

```powershell
# Verify both use same domain
# Server
$env:DDS_DOMAIN_ID="0"

# Client
$env:DDS_DOMAIN_ID="0"  # Must match!
```

#### B. Multicast Disabled/Blocked

**Check if multicast works:**
```powershell
# On one PC
ddsperf pub

# On another PC
ddsperf sub

# Should show data transfer
```

**Enable multicast routing:**
```powershell
# Windows - add multicast route
route add 239.255.0.1 mask 255.255.255.255 <your_local_ip> -p
```

**Check network adapter settings:**
- Ensure "IP Multicast" is enabled
- Check if VPN/virtual adapters interfere

#### C. Firewall Blocking UDP

```powershell
# Run as Administrator

# Allow CycloneDDS discovery (UDP multicast)
New-NetFirewallRule -DisplayName "DDS Discovery" `
    -Direction Inbound `
    -Protocol UDP `
    -LocalPort 7400-7500 `
    -Action Allow

New-NetFirewallRule -DisplayName "DDS Discovery Out" `
    -Direction Outbound `
    -Protocol UDP `
    -RemotePort 7400-7500 `
    -Action Allow

# Allow multicast address
New-NetFirewallRule -DisplayName "DDS Multicast" `
    -Direction Inbound `
    -Protocol UDP `
    -RemoteAddress 239.255.0.1 `
    -Action Allow
```

#### D. Different Subnets

**Problem:** PCs on different subnets can't discover each other via multicast.

**Check IPs:**
```powershell
ipconfig

# Example:
# PC1: 192.168.1.100
# PC2: 192.168.2.100  <- Different subnet!
```

**Solutions:**

1. **Use same subnet:**
   - Move to same network
   - Configure router to allow multicast between subnets

2. **Use unicast discovery (advanced):**
   Create `cyclonedds.xml`:
   ```xml
   <CycloneDDS>
     <Domain>
       <Discovery>
         <Peers>
           <Peer address="192.168.1.100"/>
           <Peer address="192.168.2.100"/>
         </Peers>
       </Discovery>
     </Domain>
   </CycloneDDS>
   ```
   
   Set environment:
   ```powershell
   $env:CYCLONEDDS_URI="file://path/to/cyclonedds.xml"
   ```

### 3. Type Errors

**Error:**
```
TypeError: <class 'bytes'> is not an idl type
```

**Cause:** Using wrong type for binary data.

**Solution:** Use `sequence[int]` for serialized weights:

```python
from cyclonedds.idl.types import sequence

@dataclass
class GlobalModel(IdlStruct):
    round: int
    weights: sequence[int]  # NOT bytes, NOT List[int]
```

**Convert bytes to sequence[int]:**
```python
# Serialize
serialized = pickle.dumps(weights)
int_list = list(serialized)  # Convert to list of ints

# Deserialize
bytes_data = bytes(int_list)
weights = pickle.loads(bytes_data)
```

### 4. IdlStruct Base Class Missing

**Error:**
```
TypeError: ... is not an idl type
```

**Cause:** Dataclass doesn't inherit from `IdlStruct`.

**Solution:**
```python
from cyclonedds.idl import IdlStruct
from dataclasses import dataclass

# Correct
@dataclass
class GlobalModel(IdlStruct):
    round: int
    weights: sequence[int]

# Wrong - no IdlStruct
@dataclass
class GlobalModel:
    round: int
    weights: bytes
```

### 5. QoS Mismatch

**Symptom:** Writer publishes but reader never receives.

**Cause:** QoS policies don't match.

**Solution:** Ensure both reader and writer use compatible QoS:

```python
from cyclonedds.core import Qos, Policy
from cyclonedds.util import duration

reliable_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
    Policy.History.KeepAll,
    Policy.Durability.TransientLocal
)

# Use same QoS for both
writer = DataWriter(participant, topic, qos=reliable_qos)
reader = DataReader(participant, topic, qos=reliable_qos)
```

### 6. Messages Lost

**Symptom:** Client sends but server doesn't receive.

**Causes:**
- BEST_EFFORT reliability (default)
- History KeepLast(1) loses old messages
- No durability for late joiners

**Solution:** Use RELIABLE QoS (already in code):

```python
reliable_qos = Qos(
    Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),  # Guaranteed delivery
    Policy.History.KeepAll,  # Keep all messages
    Policy.Durability.TransientLocal  # Late joiners get history
)
```

### 7. AttributeError: 'DomainParticipant' has no attribute 'close'

**Error:**
```
AttributeError: 'DomainParticipant' object has no attribute 'close'
```

**Cause:** Old code trying to call non-existent method.

**Solution:**
```python
def cleanup(self):
    """Cleanup DDS resources"""
    if self.participant:
        # Don't call close() - it doesn't exist
        self.participant = None  # Let Python GC handle it
    print("DDS resources cleaned up")
```

---

## Step 3: Network Configuration

### Check Network Interfaces

```powershell
# List all network adapters
Get-NetAdapter | Where-Object Status -eq 'Up' | Select-Object Name, Status, LinkSpeed

# Check IP configuration
ipconfig /all
```

**Multiple NICs?** DDS might bind to wrong interface.

**Solution:** Specify interface in config.

### Configure Network Interface (Advanced)

Create `cyclonedds.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS>
  <Domain id="any">
    <General>
      <!-- Bind to specific interface -->
      <NetworkInterfaceAddress>192.168.0.101</NetworkInterfaceAddress>
      
      <!-- Or by adapter name (Windows) -->
      <!-- <NetworkInterfaceAddress>Ethernet</NetworkInterfaceAddress> -->
    </General>
  </Domain>
</CycloneDDS>
```

**Set environment:**
```powershell
$env:CYCLONEDDS_URI="file://C:/path/to/cyclonedds.xml"
python FL_Server_DDS.py
```

### Disable IPv6 (if causing issues)

```xml
<CycloneDDS>
  <Domain>
    <General>
      <EnableIPv6>false</EnableIPv6>
    </General>
  </Domain>
</CycloneDDS>
```

---

## Step 4: Debug DDS Communication

### Enable CycloneDDS Logging

```powershell
# Set trace level
$env:CYCLONEDDS_TRACE="trace,discovery,data"
$env:CYCLONEDDS_VERBOSITY="finest"

# Run server/client
python FL_Server_DDS.py
```

**Log Levels:**
- `fatal`, `error`, `warning`, `info`, `config`, `fine`, `finer`, `finest`

**Trace Categories:**
- `discovery` - Peer discovery
- `data` - Data transfer
- `tcp` - TCP transport
- `udp` - UDP transport
- `trace` - Everything

### Create Logging Config

`cyclonedds.xml`:
```xml
<CycloneDDS>
  <Domain>
    <Tracing>
      <Verbosity>finest</Verbosity>
      <Category>discovery,data</Category>
      <OutputFile>dds_trace.log</OutputFile>
    </Tracing>
  </Domain>
</CycloneDDS>
```

**Use it:**
```powershell
$env:CYCLONEDDS_URI="file://cyclonedds.xml"
python FL_Server_DDS.py
```

Check `dds_trace.log` for details.

### Test Discovery Manually

**Simple Test Script:**

```python
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.idl import IdlStruct
from dataclasses import dataclass
import time

@dataclass
class TestMessage(IdlStruct):
    text: str

# Create participant
dp = DomainParticipant(0)
print(f"Created participant: {dp.guid}")

# Create topic
topic = Topic(dp, "TestTopic", TestMessage)
print(f"Created topic: {topic.name}")

# Create writer
writer = DataWriter(dp, topic)
print("Created writer")

# Create reader
reader = DataReader(dp, topic)
print("Created reader")

# Wait for discovery
time.sleep(2)

# Send message
msg = TestMessage(text="Hello DDS!")
writer.write(msg)
print("Sent message")

# Receive message
time.sleep(1)
samples = reader.take()
if samples:
    print(f"Received: {samples[0].text}")
else:
    print("No message received!")

# Cleanup
dp = None
```

**Run on both machines:**
- If same machine: Should work
- If different machines: Tests discovery

---

## Step 5: Performance Issues

### Slow Message Delivery

**Problem:** High latency between send and receive.

**Solutions:**

1. **Use Reliable QoS** (already enabled in code)

2. **Increase Transport Priority:**
   ```xml
   <CycloneDDS>
     <Domain>
       <Internal>
         <TransportPriority>1</TransportPriority>
       </Internal>
     </Domain>
   </CycloneDDS>
   ```

3. **Disable Nagle Algorithm:**
   ```xml
   <CycloneDDS>
     <Domain>
       <Internal>
         <TCP_NODELAY>true</TCP_NODELAY>
       </Internal>
     </Domain>
   </CycloneDDS>
   ```

### High CPU Usage

**Problem:** DDS using too much CPU.

**Solutions:**

1. **Reduce Polling:**
   ```python
   # In client main loop
   time.sleep(0.1)  # Add small delay
   ```

2. **Use WaitSets (Advanced):**
   ```python
   from cyclonedds.core import WaitSet, ReadCondition
   
   waitset = WaitSet(self.participant)
   condition = ReadCondition(reader, lambda: True)
   waitset.attach(condition)
   
   # Wait for data instead of polling
   waitset.wait(duration(seconds=1))
   ```

### Memory Usage

**Problem:** Memory grows over time.

**Cause:** History.KeepAll accumulates messages.

**Solution:**

1. **Use KeepLast:**
   ```python
   qos = Qos(
       Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
       Policy.History.KeepLast(10),  # Only keep last 10 messages
       Policy.Durability.TransientLocal
   )
   ```

2. **Set Resource Limits:**
   ```xml
   <CycloneDDS>
     <Domain>
       <ResourceLimits>
         <MaxSamples>1000</MaxSamples>
         <MaxInstances>100</MaxInstances>
         <MaxSamplesPerInstance>10</MaxSamplesPerInstance>
       </ResourceLimits>
     </Domain>
   </CycloneDDS>
   ```

---

## Step 6: Distributed Setup Issues

### Works Locally, Fails Remotely

**Checklist:**

1. **Same Domain ID:**
   ```powershell
   # Verify on both machines
   $env:DDS_DOMAIN_ID="0"
   ```

2. **Same Subnet:**
   ```powershell
   # PC1
   ipconfig | Select-String IPv4
   
   # PC2
   ipconfig | Select-String IPv4
   
   # Should be like:
   # 192.168.0.101 and 192.168.0.102
   ```

3. **Firewall Rules:**
   ```powershell
   # On both machines
   New-NetFirewallRule -DisplayName "DDS Ports" -Direction Inbound -Protocol UDP -LocalPort 7400-7500 -Action Allow
   ```

4. **Test Multicast:**
   ```powershell
   # PC1: Send multicast
   $socket = New-Object System.Net.Sockets.UdpClient
   $multicast = [System.Net.IPAddress]::Parse("239.255.0.1")
   $socket.JoinMulticastGroup($multicast)
   $bytes = [System.Text.Encoding]::ASCII.GetBytes("test")
   $socket.Send($bytes, $bytes.Length, "239.255.0.1", 7400)
   ```

5. **Router Configuration:**
   - Ensure router forwards multicast
   - Check IGMP snooping settings
   - Some routers block multicast by default

### VPN/Virtual Adapters

**Problem:** VPN or VMware adapters interfere.

**Solution:**

1. **Disable virtual adapters:**
   ```powershell
   Disable-NetAdapter -Name "VMware*" -Confirm:$false
   ```

2. **Or specify physical interface:**
   ```xml
   <CycloneDDS>
     <Domain>
       <General>
         <NetworkInterfaceAddress>Ethernet</NetworkInterfaceAddress>
       </General>
     </Domain>
   </CycloneDDS>
   ```

---

## Step 7: Alternative Discovery Methods

### Unicast Discovery (No Multicast)

**When to use:**
- Network doesn't support multicast
- Firewall blocks multicast
- Different subnets

**Configuration:**

```xml
<CycloneDDS>
  <Domain id="0">
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <Peers>
        <Peer address="192.168.0.101"/>
        <Peer address="192.168.0.102"/>
        <!-- Add all peer IPs -->
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
```

**Set on all machines:**
```powershell
$env:CYCLONEDDS_URI="file://C:/path/to/cyclonedds.xml"
```

### Static Discovery

**For deterministic deployments:**

```xml
<CycloneDDS>
  <Domain>
    <Discovery>
      <SPDPInterval>30s</SPDPInterval>
      <LeaseDuration>60s</LeaseDuration>
    </Discovery>
  </Domain>
</CycloneDDS>
```

---

## Step 8: Common Windows-Specific Issues

### Windows Defender

**Problem:** Windows Defender blocks DDS communication.

**Solution:**
```powershell
# Add firewall rules (see Step 3)
# Or temporarily disable for testing
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Re-enable after testing!
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True
```

### Hyper-V Virtual Switch

**Problem:** Hyper-V creates virtual switches that interfere.

**Solution:**
1. Disable unused virtual switches
2. Or specify physical interface in config

### Windows Service

**Problem:** Running as Windows Service without network access.

**Solution:** Service must run under account with network permissions.

---

## Diagnostics Tools

### ddsperf (DDS Performance Test)

**Install:**
```bash
# Usually comes with CycloneDDS
ddsperf --help
```

**Test pub/sub:**
```powershell
# Terminal 1
ddsperf pub

# Terminal 2  
ddsperf sub

# Should show throughput stats
```

### Wireshark

**Capture DDS traffic:**

1. Install Wireshark
2. Start capture on network interface
3. Filter: `udp.port >= 7400 and udp.port <= 7500`
4. Should see DDS discovery and data packets

**Multicast filter:**
```
ip.dst == 239.255.0.1
```

---

## Quick Reference

### Environment Variables

```powershell
# Domain ID
$env:DDS_DOMAIN_ID="0"

# Config file
$env:CYCLONEDDS_URI="file://cyclonedds.xml"

# Debug logging
$env:CYCLONEDDS_TRACE="discovery,data"
$env:CYCLONEDDS_VERBOSITY="finest"
```

### Ports Used

- **7400** - Default discovery port
- **7401-7500** - Dynamic data ports
- **UDP** - All DDS communication

### Common Issues Summary

| Issue | Solution |
|-------|----------|
| DLL not found | Add to PATH or update cyclone_path |
| No peer discovery | Check domain ID, firewall, multicast |
| Type errors | Use IdlStruct, sequence[int] |
| Messages lost | Use Reliable QoS |
| Different subnets | Use unicast discovery |
| Firewall | Allow UDP 7400-7500 |

---

## When All Else Fails

1. **Simplest Test:**
   ```python
   from cyclonedds.domain import DomainParticipant
   dp = DomainParticipant(0)
   print(f"GUID: {dp.guid}")
   input("Press Enter...")
   ```
   
   Run on both machines - should see different GUIDs.

2. **Check Dependencies:**
   ```powershell
   pip list | Select-String cyclone
   # Should show: cyclonedds
   ```

3. **Reinstall:**
   ```powershell
   pip uninstall cyclonedds
   pip install cyclonedds
   ```

4. **Use Docker:**
   ```dockerfile
   FROM python:3.9
   RUN apt-get update && apt-get install -y cyclonedds
   RUN pip install cyclonedds
   WORKDIR /app
   COPY . .
   CMD ["python", "FL_Server_DDS.py"]
   ```

5. **Alternative:** Use pre-built binaries from https://github.com/eclipse-cyclonedds/cyclonedds/releases

---

## Resources

- **CycloneDDS GitHub:** https://github.com/eclipse-cyclonedds/cyclonedds
- **Python Bindings:** https://github.com/eclipse-cyclonedds/cyclonedds-python
- **Documentation:** https://cyclonedds.io/docs/
- **DDS Specification:** https://www.omg.org/spec/DDS/
- **Community:** https://gitter.im/eclipse-cyclonedds/community
