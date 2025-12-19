# Troubleshooting gRPC Connection Issues

## Quick Diagnostics Checklist

- [ ] gRPC server is running and listening
- [ ] Port 50051 is open in firewall
- [ ] Client has correct server address
- [ ] Protocol buffers compiled correctly
- [ ] Network connectivity between client and server
- [ ] No port conflicts

---

## Step 1: Verify Server is Running

### Check if Server Started Successfully

**Expected Output:**
```
======================================================================
Starting Federated Learning Server (gRPC)
Server Address: localhost:50051
Number of Clients: 2
Number of Rounds: 5
======================================================================
gRPC Server started on localhost:50051
Waiting for clients to register...
```

**If Server Doesn't Start:**

1. **Check if port is already in use:**
   ```powershell
   # Find what's using port 50051
   netstat -ano | findstr :50051
   
   # Kill the process if needed
   taskkill /PID <PID> /F
   ```

2. **Try a different port:**
   ```powershell
   $env:GRPC_PORT="50052"
   python FL_Server_gRPC.py
   ```

---

## Step 2: Verify Protocol Buffers Compilation

### Check if Generated Files Exist

```powershell
ls Protocols\federated_learning_pb2.py
ls Protocols\federated_learning_pb2_grpc.py
```

**If Files Don't Exist:**

```powershell
cd Protocols
python compile_proto.py
```

### Manual Compilation

```powershell
python -m grpc_tools.protoc `
    -I. `
    --python_out=. `
    --grpc_python_out=. `
    federated_learning.proto
```

### Verify Compilation

```powershell
python -c "import sys; sys.path.insert(0, 'Protocols'); import federated_learning_pb2; print('✓ Proto compiled successfully')"
```

---

## Step 3: Common Connection Errors

### 1. Connection Refused

**Error:**
```
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.UNAVAILABLE
        details = "failed to connect to all addresses"
```

**Causes:**
- Server is not running
- Wrong host/port
- Network issue

**Solutions:**

1. **Verify Server is Running:**
   ```powershell
   # Check if process is running
   Get-Process python | Where-Object {$_.CommandLine -like "*FL_Server_gRPC*"}
   ```

2. **Check Connection:**
   ```powershell
   Test-NetConnection -ComputerName localhost -Port 50051
   ```

3. **Use Correct Address:**
   ```python
   # For local testing
   GRPC_HOST = "localhost"  # or "127.0.0.1"
   
   # For remote testing
   GRPC_HOST = "192.168.0.101"  # Server's IP
   ```

### 2. Deadline Exceeded

**Error:**
```
grpc._channel._InactiveRpcError: <_InactiveRpcError of RPC that terminated with:
        status = StatusCode.DEADLINE_EXCEEDED
        details = "Deadline Exceeded"
```

**Causes:**
- Server is slow to respond
- Network latency
- Server is processing large data

**Solutions:**

1. **Increase Timeout:**
   ```python
   # In client code
   response = stub.GetGlobalModel(request, timeout=30)  # 30 seconds
   ```

2. **Check Server Load:**
   - Is server CPU/memory saturated?
   - Are there too many clients?

### 3. Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'federated_learning_pb2'
```

**Solutions:**

1. **Check Python Path:**
   ```python
   import sys
   sys.path.insert(0, 'Protocols')  # Add before imports
   ```

2. **Recompile Proto:**
   ```powershell
   cd Protocols
   python compile_proto.py
   ```

3. **Check File Names:**
   - `federated_learning.proto` (input)
   - `federated_learning_pb2.py` (generated)
   - `federated_learning_pb2_grpc.py` (generated)

### 4. Stub Import Errors

**Error:**
```
AttributeError: module 'federated_learning_pb2_grpc' has no attribute 'FederatedLearningServiceStub'
```

**Causes:**
- Proto file not compiled with grpc
- Wrong grpcio-tools version
- Corrupted generated files

**Solutions:**

1. **Reinstall grpcio-tools:**
   ```powershell
   pip uninstall grpcio-tools
   pip install grpcio-tools
   ```

2. **Delete and Recompile:**
   ```powershell
   cd Protocols
   Remove-Item federated_learning_pb2.py -ErrorAction SilentlyContinue
   Remove-Item federated_learning_pb2_grpc.py -ErrorAction SilentlyContinue
   python compile_proto.py
   ```

3. **Check Service Name in Proto:**
   ```protobuf
   service FederatedLearningService {  // Must match
       rpc RegisterClient(...) returns (...);
   }
   ```

### 5. Channel Credentials Error

**Error:**
```
grpc._channel._InactiveRpcError: SSL handshake failed
```

**Cause:** Using secure channel without proper certificates.

**Solution:**

For testing, use **insecure channel**:
```python
# Correct for testing
channel = grpc.insecure_channel(f'{GRPC_HOST}:{GRPC_PORT}')

# Don't use this for testing:
# channel = grpc.secure_channel(...)  # Requires SSL certificates
```

---

## Step 4: Configure Windows Firewall

```powershell
# Run as Administrator

# Allow gRPC port (default 50051)
New-NetFirewallRule -DisplayName "gRPC FL Server" `
    -Direction Inbound `
    -LocalPort 50051 `
    -Protocol TCP `
    -Action Allow

# Allow outbound (for responses)
New-NetFirewallRule -DisplayName "gRPC FL Client" `
    -Direction Outbound `
    -RemotePort 50051 `
    -Protocol TCP `
    -Action Allow
```

### Verify Firewall Rules

```powershell
Get-NetFirewallRule -DisplayName "*gRPC*" | Select-Object DisplayName, Enabled, Direction
```

---

## Step 5: Network Connectivity Tests

### Test from Client Machine

```powershell
# Test TCP connection
Test-NetConnection -ComputerName 192.168.0.101 -Port 50051

# Expected output:
# TcpTestSucceeded : True
```

### Test with Telnet

```powershell
# Enable telnet client if not installed
dism /online /Enable-Feature /FeatureName:TelnetClient

# Test connection
telnet 192.168.0.101 50051

# If connection succeeds, server is listening
```

### Verify Server is Listening

```powershell
# On server machine
netstat -an | Select-String ":50051"

# Should show:
# TCP    0.0.0.0:50051          0.0.0.0:0              LISTENING
# or
# TCP    [::]:50051             [::]:0                 LISTENING
```

---

## Step 6: Debug gRPC Communication

### Enable gRPC Logging

**Server:**
```python
import logging
import grpc

# At the top of FL_Server_gRPC.py
logging.basicConfig(level=logging.DEBUG)
grpc_logger = logging.getLogger('grpc')
grpc_logger.setLevel(logging.DEBUG)
```

**Client:**
```python
import logging
import grpc

# At the top of FL_Client_gRPC.py
logging.basicConfig(level=logging.DEBUG)
grpc_logger = logging.getLogger('grpc')
grpc_logger.setLevel(logging.DEBUG)
```

### Enable gRPC Environment Variables

```powershell
# Verbose logging
$env:GRPC_VERBOSITY="DEBUG"
$env:GRPC_TRACE="all"

# Run server/client
python FL_Server_gRPC.py
```

---

## Step 7: Test gRPC Connection

### Simple Test Script

**test_grpc_connection.py:**
```python
import grpc
import sys
sys.path.insert(0, 'Protocols')
import federated_learning_pb2 as fl_pb2
import federated_learning_pb2_grpc as fl_pb2_grpc

def test_connection(host, port):
    try:
        # Create channel
        channel = grpc.insecure_channel(f'{host}:{port}')
        
        # Wait for channel to be ready (5 second timeout)
        grpc.channel_ready_future(channel).result(timeout=5)
        
        print(f"✓ Successfully connected to {host}:{port}")
        
        # Try to create stub
        stub = fl_pb2_grpc.FederatedLearningServiceStub(channel)
        print("✓ Stub created successfully")
        
        channel.close()
        return True
        
    except grpc.FutureTimeoutError:
        print(f"✗ Connection timeout to {host}:{port}")
        print("  Server might not be running or network issue")
        return False
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False

if __name__ == "__main__":
    host = input("Enter server host (default: localhost): ") or "localhost"
    port = input("Enter server port (default: 50051): ") or "50051"
    
    test_connection(host, port)
```

**Run Test:**
```powershell
python test_grpc_connection.py
```

---

## Common Configuration Issues

### 1. Server Listening on Wrong Interface

**Problem:** Server only accessible from localhost, not remotely.

**Check Server Binding:**
```python
# In FL_Server_gRPC.py
server.add_insecure_port(f'{GRPC_HOST}:{GRPC_PORT}')
```

**For Remote Access:**
```python
# Listen on all interfaces
GRPC_HOST = "0.0.0.0"

# Or set via environment variable
$env:GRPC_HOST="0.0.0.0"
```

**For Local Only:**
```python
GRPC_HOST = "localhost"  # or "127.0.0.1"
```

### 2. Client Using Wrong Address

**Problem:** Client can't find server.

**For Same Machine:**
```python
GRPC_HOST = "localhost"
```

**For Remote Server:**
```python
GRPC_HOST = "192.168.0.101"  # Server's actual IP
```

**Set via Environment:**
```powershell
# Client machine
$env:GRPC_HOST="192.168.0.101"
$env:GRPC_PORT="50051"

cd Client
python FL_Client_gRPC.py
```

### 3. Port Conflicts

**Error:**
```
OSError: [WinError 10048] Only one usage of each socket address (protocol/network address/port) is normally permitted
```

**Solution:**

1. **Find and kill process:**
   ```powershell
   $port = 50051
   $proc = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
   if ($proc) {
       Stop-Process -Id $proc.OwningProcess -Force
   }
   ```

2. **Use different port:**
   ```powershell
   $env:GRPC_PORT="50052"
   ```

---

## Debugging Workflow

### Step-by-Step Debugging

1. **Verify Proto Compilation:**
   ```powershell
   python -c "import sys; sys.path.insert(0, 'Protocols'); import federated_learning_pb2, federated_learning_pb2_grpc; print('OK')"
   ```

2. **Start Server with Debug:**
   ```powershell
   $env:GRPC_VERBOSITY="DEBUG"
   python Server/FL_Server_gRPC.py
   ```

3. **Check Server Listening:**
   ```powershell
   netstat -an | Select-String ":50051"
   ```

4. **Test Connection:**
   ```powershell
   Test-NetConnection -ComputerName localhost -Port 50051
   ```

5. **Start Client with Debug:**
   ```powershell
   $env:CLIENT_ID="0"
   $env:GRPC_VERBOSITY="DEBUG"
   python Client/FL_Client_gRPC.py
   ```

6. **Monitor Both Terminals:**
   - Server should show: "Client 0 registered"
   - Client should show: "Successfully registered with server"

---

## Performance Issues

### 1. Slow Model Transfer

**Problem:** Large model weights take too long to send.

**Solutions:**

1. **Enable Compression:**
   ```python
   # Server
   server = grpc.server(
       futures.ThreadPoolExecutor(max_workers=10),
       compression=grpc.Compression.Gzip
   )
   
   # Client
   channel = grpc.insecure_channel(
       f'{GRPC_HOST}:{GRPC_PORT}',
       options=[('grpc.default_compression_algorithm', grpc.Compression.Gzip)]
   )
   ```

2. **Increase Message Size Limits:**
   ```python
   # Server
   server = grpc.server(
       futures.ThreadPoolExecutor(max_workers=10),
       options=[
           ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
           ('grpc.max_receive_message_length', 100 * 1024 * 1024)
       ]
   )
   
   # Client
   channel = grpc.insecure_channel(
       f'{GRPC_HOST}:{GRPC_PORT}',
       options=[
           ('grpc.max_send_message_length', 100 * 1024 * 1024),
           ('grpc.max_receive_message_length', 100 * 1024 * 1024)
       ]
   )
   ```

### 2. High Polling Overhead

**Problem:** Client polls every 0.5 seconds, causing CPU usage.

**Solution:**

1. **Increase Polling Interval:**
   ```python
   # In FL_Client_gRPC.py main loop
   time.sleep(1.0)  # Poll every 1 second instead of 0.5
   ```

2. **Use Server Streaming (Advanced):**
   ```protobuf
   // In federated_learning.proto
   service FederatedLearningService {
       rpc StreamTrainingStatus(ClientInfo) returns (stream TrainingStatus);
   }
   ```

### 3. Timeout Issues

**Problem:** Operations timeout on slow networks.

**Solutions:**

1. **Increase RPC Timeout:**
   ```python
   response = stub.GetGlobalModel(request, timeout=60)  # 60 seconds
   ```

2. **Set Keepalive:**
   ```python
   options = [
       ('grpc.keepalive_time_ms', 10000),  # 10 seconds
       ('grpc.keepalive_timeout_ms', 5000),  # 5 seconds
       ('grpc.keepalive_permit_without_calls', True)
   ]
   channel = grpc.insecure_channel(f'{GRPC_HOST}:{GRPC_PORT}', options=options)
   ```

---

## Security Configuration (Production)

### Enable TLS/SSL

**1. Generate Certificates:**

```powershell
# Generate private key
openssl genrsa -out server.key 2048

# Generate certificate signing request
openssl req -new -key server.key -out server.csr

# Generate self-signed certificate (for testing)
openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt
```

**2. Update Server:**

```python
import grpc

# Load credentials
with open('server.key', 'rb') as f:
    private_key = f.read()
with open('server.crt', 'rb') as f:
    certificate_chain = f.read()

server_credentials = grpc.ssl_server_credentials([(private_key, certificate_chain)])

# Use secure port
server.add_secure_port(f'{GRPC_HOST}:{GRPC_PORT}', server_credentials)
```

**3. Update Client:**

```python
import grpc

# Load CA certificate
with open('server.crt', 'rb') as f:
    trusted_certs = f.read()

credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)

# Use secure channel
channel = grpc.secure_channel(f'{GRPC_HOST}:{GRPC_PORT}', credentials)
```

---

## VS Code Debugging

### Debug Configuration Issues

**Problem:** Can't debug in VS Code.

**Check launch.json:**

```json
{
    "name": "FL Server (gRPC) - Local",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/Server/FL_Server_gRPC.py",
    "console": "integratedTerminal",
    "justMyCode": true,
    "env": {
        "GRPC_HOST": "localhost",
        "GRPC_PORT": "50051"
    }
}
```

**Set Breakpoints:**
- In `RegisterClient` method
- In client's `register_with_server()` method
- Step through to find issues

---

## When All Else Fails

1. **Restart Everything:**
   ```powershell
   # Kill all Python processes
   Get-Process python | Stop-Process -Force
   
   # Restart server
   python Server/FL_Server_gRPC.py
   ```

2. **Check Dependencies:**
   ```powershell
   pip list | Select-String grpc
   # Should show: grpcio, grpcio-tools, protobuf
   ```

3. **Reinstall gRPC:**
   ```powershell
   pip uninstall grpcio grpcio-tools protobuf
   pip install grpcio grpcio-tools protobuf
   ```

4. **Check Python Version:**
   ```powershell
   python --version
   # Should be Python 3.7+
   ```

5. **Use Docker (Alternative):**
   ```dockerfile
   FROM python:3.9
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   CMD ["python", "Server/FL_Server_gRPC.py"]
   ```

---

## Quick Reference Commands

```powershell
# Compile Proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. federated_learning.proto

# Check Port
netstat -an | Select-String ":50051"

# Test Connection
Test-NetConnection -ComputerName localhost -Port 50051

# Kill Process on Port
Get-NetTCPConnection -LocalPort 50051 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

# Enable Debug
$env:GRPC_VERBOSITY="DEBUG"
$env:GRPC_TRACE="all"

# Set Server Address
$env:GRPC_HOST="192.168.0.101"
$env:GRPC_PORT="50051"
```

---

## Resources

- **gRPC Documentation:** https://grpc.io/docs/
- **gRPC Python:** https://grpc.io/docs/languages/python/
- **Protocol Buffers:** https://developers.google.com/protocol-buffers
- **GitHub Issues:** https://github.com/grpc/grpc/issues
