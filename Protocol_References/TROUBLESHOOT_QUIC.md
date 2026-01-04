# Troubleshooting QUIC Federated Learning

This guide helps resolve common issues with the QUIC-based federated learning implementation.

## Installation Issues

### Issue: aioquic not installing

**Error:**
```
ERROR: Could not find a version that satisfies the requirement aioquic
```

**Solutions:**

1. **Update pip:**
   ```powershell
   python -m pip install --upgrade pip
   ```

2. **Install build dependencies:**
   ```powershell
   # Install Visual Studio Build Tools or use conda
   conda install -c conda-forge aioquic
   ```

3. **Install from source:**
   ```powershell
   pip install git+https://github.com/aiortc/aioquic.git
   ```

4. **Check Python version (requires 3.7+):**
   ```powershell
   python --version
   ```

### Issue: OpenSSL command not found

**Error:**
```
'openssl' is not recognized as an internal or external command
```

**Solutions:**

1. **Install OpenSSL for Windows:**
   - Download from: https://slproweb.com/products/Win32OpenSSL.html
   - Add to PATH: `C:\Program Files\OpenSSL-Win64\bin`

2. **Alternative - Use Git Bash:**
   ```bash
   # In Git Bash
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   ```

3. **Use Python to generate certificates:**
   ```python
   from cryptography import x509
   from cryptography.x509.oid import NameOID
   from cryptography.hazmat.primitives import hashes
   from cryptography.hazmat.backends import default_backend
   from cryptography.hazmat.primitives.asymmetric import rsa
   from cryptography.hazmat.primitives import serialization
   import datetime
   
   # Generate key
   key = rsa.generate_private_key(
       public_exponent=65537,
       key_size=4096,
       backend=default_backend()
   )
   
   # Generate certificate
   subject = issuer = x509.Name([
       x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
   ])
   
   cert = x509.CertificateBuilder().subject_name(
       subject
   ).issuer_name(
       issuer
   ).public_key(
       key.public_key()
   ).serial_number(
       x509.random_serial_number()
   ).not_valid_before(
       datetime.datetime.utcnow()
   ).not_valid_after(
       datetime.datetime.utcnow() + datetime.timedelta(days=365)
   ).sign(key, hashes.SHA256(), default_backend())
   
   # Write key
   with open("key.pem", "wb") as f:
       f.write(key.private_bytes(
           encoding=serialization.Encoding.PEM,
           format=serialization.PrivateFormat.TraditionalOpenSSL,
           encryption_algorithm=serialization.NoEncryption()
       ))
   
   # Write certificate
   with open("cert.pem", "wb") as f:
       f.write(cert.public_bytes(serialization.Encoding.PEM))
   ```

## Connection Issues

### Issue: Client cannot connect to server

**Error:**
```
Client 0 failed to connect to server
Connection refused
```

**Solutions:**

1. **Check server is running:**
   ```powershell
   # Server should show "Waiting for clients to connect..."
   ```

2. **Verify host/port configuration:**
   ```powershell
   # Ensure both use same settings
   echo $env:QUIC_HOST   # Should be "localhost" or IP
   echo $env:QUIC_PORT   # Should be 4433 (or custom)
   ```

3. **Check firewall settings:**
   ```powershell
   # Allow UDP traffic on port 4433
   netsh advfirewall firewall add rule name="QUIC FL Server" dir=in action=allow protocol=UDP localport=4433
   ```

4. **Try different port:**
   ```powershell
   $env:QUIC_PORT="5433"
   ```

5. **Check if port is in use:**
   ```powershell
   netstat -an | findstr "4433"
   ```

### Issue: Certificate verification failed

**Error:**
```
SSL/TLS certificate verification failed
```

**Solutions:**

1. **Verify certificates exist:**
   ```powershell
   ls Server/cert.pem
   ls Server/key.pem
   ```

2. **Regenerate certificates:**
   ```powershell
   cd Server
   del cert.pem key.pem
   # Regenerate using OpenSSL or Python method above
   ```

3. **Check client configuration:**
   - Client code has `configuration.verify_mode = False` for testing
   - For production, use proper CA-signed certificates

### Issue: UDP packets blocked

**Error:**
```
Timeout waiting for server response
Connection attempt failed
```

**Solutions:**

1. **Check UDP is not blocked:**
   ```powershell
   # Test UDP connectivity
   Test-NetConnection -ComputerName localhost -Port 4433 -InformationLevel Detailed
   ```

2. **Disable firewall temporarily (for testing):**
   ```powershell
   # As Administrator
   Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
   # Remember to re-enable after testing!
   ```

3. **Add firewall exception:**
   ```powershell
   New-NetFirewallRule -DisplayName "QUIC Server" -Direction Inbound -Protocol UDP -LocalPort 4433 -Action Allow
   ```

## Runtime Issues

### Issue: AsyncIO event loop errors

**Error:**
```
RuntimeError: Event loop is closed
Cannot run asyncio from within asyncio
```

**Solutions:**

1. **Don't nest asyncio.run():**
   ```python
   # Wrong
   asyncio.run(asyncio.run(main()))
   
   # Correct
   asyncio.run(main())
   ```

2. **Use proper async/await:**
   ```python
   # Ensure all blocking operations use await
   await asyncio.sleep(1)
   await client.send_message(msg)
   ```

3. **Run in executor for CPU-bound tasks:**
   ```python
   loop = asyncio.get_event_loop()
   result = await loop.run_in_executor(None, cpu_intensive_function)
   ```

### Issue: Model training blocks event loop

**Error:**
```
Event loop blocked during model training
No response from client
```

**Solutions:**

1. **Training already uses executor:**
   ```python
   # This is correct (already in code)
   history = await loop.run_in_executor(
       None,
       lambda: self.model.fit(...)
   )
   ```

2. **Check TensorFlow threading:**
   ```python
   import tensorflow as tf
   tf.config.threading.set_intra_op_parallelism_threads(1)
   tf.config.threading.set_inter_op_parallelism_threads(1)
   ```

### Issue: Clients not receiving messages

**Error:**
```
Client waiting for global model... (indefinitely)
No training signal received
```

**Solutions:**

1. **Check stream IDs are correct:**
   - Server broadcasts to all registered client streams
   - Verify registration completed

2. **Add debug logging:**
   ```python
   print(f"Sending message type: {message['type']}")
   print(f"Stream ID: {stream_id}")
   ```

3. **Verify message encoding:**
   ```python
   # Check JSON serialization works
   test_msg = json.dumps(message)
   decoded = json.loads(test_msg)
   ```

4. **Check for exceptions in event handler:**
   ```python
   async def handle_message(self, message):
       try:
           # ... handling code
       except Exception as e:
           print(f"ERROR in handle_message: {e}")
           import traceback
           traceback.print_exc()
   ```

### Issue: Training doesn't start

**Error:**
```
All clients registered
Server doesn't distribute initial model
```

**Solutions:**

1. **Check NUM_CLIENTS matches actual clients:**
   ```powershell
   # Server
   echo $env:NUM_CLIENTS  # Should be 2
   
   # Client 0
   echo $env:NUM_CLIENTS  # Should be 2
   
   # Client 1
   echo $env:NUM_CLIENTS  # Should be 2
   ```

2. **Verify all clients registered:**
   ```
   # Server output should show:
   Client 0 registered (1/2)
   Client 1 registered (2/2)
   All clients registered...
   ```

3. **Check for async deadlocks:**
   - Add timeouts to await calls
   - Use `asyncio.wait_for()` with timeout

## Performance Issues

### Issue: High latency

**Symptoms:**
- Slow model updates
- Long round times

**Solutions:**

1. **Reduce model size:**
   - Use smaller batch sizes
   - Fewer local epochs

2. **Check network:**
   ```powershell
   ping localhost
   # Should be < 1ms for local testing
   ```

3. **Optimize serialization:**
   - Model weights are already using pickle + base64
   - Consider compression if very large models

4. **Monitor system resources:**
   ```powershell
   # CPU usage
   Get-Process python | Select-Object CPU, WorkingSet
   ```

### Issue: Memory errors

**Error:**
```
MemoryError: Unable to allocate array
Out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   self.training_config = {
       "batch_size": 16,  # Reduce from 32
       "local_epochs": 20
   }
   ```

2. **Use smaller dataset:**
   ```python
   # Sample dataset
   dataframe = dataframe.sample(frac=0.5)
   ```

3. **Enable GPU memory growth (if using GPU):**
   ```python
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

## Data Issues

### Issue: Dataset not found

**Error:**
```
FileNotFoundError: Dataset/base_data_baseline_unique.csv
```

**Solutions:**

1. **Check working directory:**
   ```powershell
   pwd  # Should be in Client directory
   ```

2. **Verify dataset exists:**
   ```powershell
   ls Dataset/base_data_baseline_unique.csv
   ```

3. **Use absolute path:**
   ```python
   dataset_path = Path(__file__).parent / "Dataset" / "base_data_baseline_unique.csv"
   dataframe = pd.read_csv(dataset_path)
   ```

### Issue: Data partition errors

**Error:**
```
IndexError: index out of bounds
Empty training data
```

**Solutions:**

1. **Check NUM_CLIENTS vs dataset size:**
   ```python
   print(f"Dataset size: {len(dataframe)}")
   print(f"Partition size: {len(dataframe) // NUM_CLIENTS}")
   # Should be > 0
   ```

2. **Ensure enough data per client:**
   - Need at least 100+ samples per client
   - Current dataset should have 1000+ total samples

## Debugging Tips

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Monitor QUIC Connection

```python
# Add to server/client
import aioquic.asyncio
aioquic.asyncio.logger.setLevel(logging.DEBUG)
```

### Check Protocol State

```python
# In protocol class
print(f"Connection state: {self._quic.get_state()}")
print(f"Available streams: {self._quic.get_next_available_stream_id()}")
```

### Test Components Separately

1. **Test certificate generation:**
   ```powershell
   openssl x509 -in cert.pem -text -noout
   ```

2. **Test QUIC connection:**
   ```python
   # Simple QUIC echo server/client test
   # (use aioquic examples)
   ```

3. **Test model training:**
   ```python
   # Run single client without server
   # Test data loading and model training
   ```

## Getting Help

If issues persist:

1. **Check aioquic issues:**
   - https://github.com/aiortc/aioquic/issues

2. **Review QUIC protocol specs:**
   - RFC 9000: https://www.rfc-editor.org/rfc/rfc9000

3. **Enable debug mode and share logs:**
   ```python
   logging.basicConfig(
       level=logging.DEBUG,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

4. **Check system compatibility:**
   - Python 3.7+ required
   - Windows 10+ / Linux / macOS supported
   - UDP protocol support required

## Common Environment Variable Issues

```powershell
# View all environment variables
Get-ChildItem Env:

# Set temporarily (current session only)
$env:QUIC_PORT="4433"
$env:CLIENT_ID="0"

# Set permanently (requires restart)
[System.Environment]::SetEnvironmentVariable("QUIC_PORT", "4433", "User")
```

## Port Already in Use

```powershell
# Find process using port 4433
netstat -ano | findstr "4433"

# Kill process by PID
taskkill /PID <PID> /F

# Or use different port
$env:QUIC_PORT="5433"
```
