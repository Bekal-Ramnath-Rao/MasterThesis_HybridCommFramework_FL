# Troubleshooting AMQP/RabbitMQ Connection Issues

## Quick Diagnostics Checklist

- [ ] RabbitMQ service is running
- [ ] Port 5672 is open in firewall
- [ ] User has correct permissions
- [ ] Client has correct connection credentials
- [ ] Virtual host (vhost) is configured correctly
- [ ] Network connectivity between client and broker

---

## Step 1: Verify RabbitMQ is Running

### Windows

```powershell
# Check RabbitMQ service status
Get-Service -Name "RabbitMQ"

# If not running, start it
Start-Service -Name "RabbitMQ"

# Check if management plugin is enabled
& "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.x.x\sbin\rabbitmq-plugins.bat" list
```

### Linux

```bash
# Check service status
sudo systemctl status rabbitmq-server

# Start if not running
sudo systemctl start rabbitmq-server

# Enable on boot
sudo systemctl enable rabbitmq-server
```

### Docker

```bash
# Check if container is running
docker ps | grep rabbitmq

# Check logs
docker logs rabbitmq

# Restart if needed
docker restart rabbitmq
```

---

## Step 2: Configure RabbitMQ for Remote Access

### Problem: `ACCESS_REFUSED` Error

**Error Message:**
```
pika.exceptions.ProbableAccessDeniedError: (403, 'ACCESS_REFUSED')
```

**Cause:** The default `guest` user can only connect from localhost.

**Solution:** Create a new user with remote access:

```powershell
# Windows PowerShell (run as Administrator)
cd "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.x.x\sbin"

# Create new user
.\rabbitmqctl.bat add_user fluser flpassword

# Set as administrator
.\rabbitmqctl.bat set_user_tags fluser administrator

# Grant permissions on default vhost
.\rabbitmqctl.bat set_permissions -p / fluser ".*" ".*" ".*"

# List users to verify
.\rabbitmqctl.bat list_users
```

**Linux:**
```bash
sudo rabbitmqctl add_user fluser flpassword
sudo rabbitmqctl set_user_tags fluser administrator
sudo rabbitmqctl set_permissions -p / fluser ".*" ".*" ".*"
sudo rabbitmqctl list_users
```

### Update Your Code

```python
# In both FL_Server_AMQP.py and FL_Client_AMQP.py
AMQP_HOST = "192.168.0.101"
AMQP_PORT = 5672
AMQP_USER = "fluser"        # NOT "guest"
AMQP_PASSWORD = "flpassword"
```

---

## Step 3: Configure Windows Firewall

```powershell
# Run as Administrator

# Allow AMQP port (5672)
New-NetFirewallRule -DisplayName "RabbitMQ AMQP" -Direction Inbound -LocalPort 5672 -Protocol TCP -Action Allow

# Allow Management UI (15672) - optional but useful
New-NetFirewallRule -DisplayName "RabbitMQ Management" -Direction Inbound -LocalPort 15672 -Protocol TCP -Action Allow

# Allow Erlang Port Mapper (4369) - for clustering
New-NetFirewallRule -DisplayName "RabbitMQ EPMD" -Direction Inbound -LocalPort 4369 -Protocol TCP -Action Allow

# Allow inter-node communication (25672) - for clustering
New-NetFirewallRule -DisplayName "RabbitMQ Clustering" -Direction Inbound -LocalPort 25672 -Protocol TCP -Action Allow
```

### Verify Firewall Rules

```powershell
Get-NetFirewallRule -DisplayName "*RabbitMQ*" | Select-Object DisplayName, Enabled, Direction, Action
```

---

## Step 4: Verify Port Connectivity

### From Broker Machine

```powershell
# Check if RabbitMQ is listening on all interfaces
netstat -an | Select-String ":5672"

# Should show:
# TCP    0.0.0.0:5672           0.0.0.0:0              LISTENING
```

If it shows `127.0.0.1:5672`, RabbitMQ is only listening locally!

**Fix:** Edit RabbitMQ config to listen on all interfaces.

### From Client Machine

```powershell
# Test TCP connectivity
Test-NetConnection -ComputerName 192.168.0.101 -Port 5672

# Should show: TcpTestSucceeded : True
```

---

## Step 5: Access RabbitMQ Management UI

Open browser: `http://192.168.0.101:15672`

**Login:**
- Username: `fluser` (or `guest` if localhost)
- Password: `flpassword` (or `guest`)

### Enable Management Plugin (if not enabled)

```powershell
# Windows
& "C:\Program Files\RabbitMQ Server\rabbitmq_server-3.x.x\sbin\rabbitmq-plugins.bat" enable rabbitmq_management

# Linux
sudo rabbitmq-plugins enable rabbitmq_management

# Restart RabbitMQ after enabling
```

### What to Check in Management UI

1. **Overview Tab:**
   - Check if server is running
   - View connections, channels, queues

2. **Connections Tab:**
   - Should see your server/client connections
   - Check connection state

3. **Queues Tab:**
   - `fl.client.0.broadcast`, `fl.client.1.broadcast` should exist
   - Check message rates

4. **Exchanges Tab:**
   - `fl_broadcast` (fanout)
   - `fl_client_updates` (direct)

---

## Common Errors and Solutions

### 1. CONNECTION_FORCED Error

**Error:**
```
pika.exceptions.ConnectionClosedByBroker: (320, 'CONNECTION_FORCED')
```

**Causes:**
- Node was restarted while client was connected
- Network interruption
- RabbitMQ crash or restart

**Solution:**
- Implement reconnection logic in code
- Check RabbitMQ logs for crash reasons
- Increase heartbeat timeout

```python
parameters = pika.ConnectionParameters(
    host=AMQP_HOST,
    port=AMQP_PORT,
    credentials=credentials,
    heartbeat=600,  # Increase to 10 minutes
    blocked_connection_timeout=300
)
```

### 2. NOT_FOUND - No Queue Error

**Error:**
```
pika.exceptions.ChannelClosedByBroker: (404, "NOT_FOUND - no queue...")
```

**Causes:**
- Queue doesn't exist
- Queue was deleted
- Wrong queue name

**Solution:**
- Ensure `queue_declare` is called before consuming
- Check queue name spelling
- Use `durable=True` for persistent queues

### 3. PRECONDITION_FAILED Error

**Error:**
```
pika.exceptions.ChannelClosedByBroker: (406, "PRECONDITION_FAILED...")
```

**Causes:**
- Queue already exists with different parameters
- Exchange already exists with different type

**Solution:**
- Delete the queue/exchange in Management UI
- Or use `passive=True` to check existence without creating

```python
# Delete queue via Management UI or:
channel.queue_delete(queue='fl.client.0.broadcast')
```

### 4. Message Not Routing

**Problem:** Messages published but clients don't receive them.

**Debug Steps:**

1. **Check Exchange Type:**
   ```python
   # Fanout: broadcasts to all bound queues (use for global model)
   channel.exchange_declare('fl_broadcast', exchange_type='fanout')
   
   # Direct: routes by routing_key (use for client updates)
   channel.exchange_declare('fl_client_updates', exchange_type='direct')
   ```

2. **Check Queue Binding:**
   ```python
   # Fanout - no routing key needed
   channel.queue_bind(exchange='fl_broadcast', queue='client_queue')
   
   # Direct - routing key must match
   channel.queue_bind(
       exchange='fl_client_updates',
       queue='server_queue',
       routing_key='client.update'
   )
   ```

3. **Check Message Type Field:**
   Ensure `message_type` field matches in callbacks:
   ```python
   if message.get('message_type') == 'global_model':
       # Handle global model
   ```

### 5. Memory/Disk Alarms

**Error:**
```
pika.exceptions.ConnectionClosedByBroker: (320, 'CONNECTION_FORCED - broker forced connection closure with reason 'shutdown'')
```

**Check Alarms:**
```powershell
.\rabbitmqctl.bat status
.\rabbitmqctl.bat list_alarms
```

**Solution:**
- Free up disk space (RabbitMQ needs 50MB free minimum)
- Free up memory
- Clear old queues/messages

```powershell
# Clear all messages from a queue
.\rabbitmqctl.bat purge_queue fl.client.0.broadcast
```

### 6. Slow Performance

**Symptoms:**
- High latency
- Messages backing up in queues

**Solutions:**

1. **Enable Publisher Confirms:**
   ```python
   channel.confirm_delivery()
   ```

2. **Use QoS Prefetch:**
   ```python
   channel.basic_qos(prefetch_count=1)
   ```

3. **Check Network Latency:**
   ```powershell
   ping 192.168.0.101
   ```

4. **Monitor in Management UI:**
   - Check message rates
   - Check consumer utilization
   - Check connection/channel counts

---

## Step 6: Test with RabbitMQ Tools

### Publish Test Message

```powershell
# Windows (if rabbitmqadmin is available)
python rabbitmqadmin.py publish exchange=fl_broadcast routing_key="" payload="test message"
```

### Python Test Script

```python
import pika

credentials = pika.PlainCredentials('fluser', 'flpassword')
parameters = pika.ConnectionParameters(
    host='192.168.0.101',
    port=5672,
    credentials=credentials
)

try:
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    print("✓ Connected to RabbitMQ successfully!")
    
    # Test exchange
    channel.exchange_declare('test_exchange', exchange_type='fanout')
    print("✓ Exchange created successfully!")
    
    connection.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

---

## Advanced Configuration

### Enable TLS/SSL

1. **Generate Certificates:**
   ```bash
   # Use RabbitMQ's test certificate generation script
   git clone https://github.com/rabbitmq/tls-gen.git
   cd tls-gen/basic
   make
   ```

2. **Configure RabbitMQ:**
   Edit `rabbitmq.conf`:
   ```conf
   listeners.ssl.default = 5671
   ssl_options.cacertfile = /path/to/ca_certificate.pem
   ssl_options.certfile   = /path/to/server_certificate.pem
   ssl_options.keyfile    = /path/to/server_key.pem
   ssl_options.verify     = verify_peer
   ssl_options.fail_if_no_peer_cert = false
   ```

3. **Update Python Code:**
   ```python
   import ssl
   
   context = ssl.create_default_context(cafile='ca_certificate.pem')
   context.load_cert_chain('client_certificate.pem', 'client_key.pem')
   
   ssl_options = pika.SSLOptions(context, 'broker_hostname')
   parameters = pika.ConnectionParameters(
       host='192.168.0.101',
       port=5671,
       ssl_options=ssl_options,
       credentials=credentials
   )
   ```

### Clustering (Multiple Broker Nodes)

```powershell
# On node 1
.\rabbitmqctl.bat stop_app
.\rabbitmqctl.bat reset
.\rabbitmqctl.bat start_app

# On node 2
.\rabbitmqctl.bat stop_app
.\rabbitmqctl.bat reset
.\rabbitmqctl.bat join_cluster rabbit@node1
.\rabbitmqctl.bat start_app

# Verify cluster status
.\rabbitmqctl.bat cluster_status
```

---

## Debugging Tips

### 1. Enable Verbose Logging

**RabbitMQ Side:**
Edit `rabbitmq.conf`:
```conf
log.console.level = debug
log.file.level = debug
```

**Python Side:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Monitor Logs

**Windows:**
```
C:\Users\<username>\AppData\Roaming\RabbitMQ\log\
```

**Linux:**
```bash
sudo tail -f /var/log/rabbitmq/rabbit@hostname.log
```

### 3. Use `rabbitmqctl` Commands

```powershell
# List queues with message counts
.\rabbitmqctl.bat list_queues name messages consumers

# List exchanges
.\rabbitmqctl.bat list_exchanges name type

# List bindings
.\rabbitmqctl.bat list_bindings

# Check node health
.\rabbitmqctl.bat node_health_check

# Environment details
.\rabbitmqctl.bat environment
```

### 4. Check pika Connection State

```python
print(f"Connection open: {connection.is_open}")
print(f"Channel open: {channel.is_open}")
```

---

## Network Configuration for Remote Access

### RabbitMQ Config File Location

**Windows:**
```
C:\Users\<username>\AppData\Roaming\RabbitMQ\rabbitmq.conf
```
Or:
```
C:\Program Files\RabbitMQ Server\rabbitmq_server-3.x.x\etc\rabbitmq.conf
```

**Linux:**
```
/etc/rabbitmq/rabbitmq.conf
```

### Minimal rabbitmq.conf for Remote Access

```conf
# Listen on all interfaces
listeners.tcp.default = 5672

# Management plugin
management.tcp.port = 15672
management.tcp.ip = 0.0.0.0

# Logging
log.console.level = info
log.file.level = info

# Defaults
default_vhost = /
default_user = fluser
default_pass = flpassword
default_permissions.configure = .*
default_permissions.read = .*
default_permissions.write = .*
```

---

## Performance Tuning

### Increase Connection Limit

```conf
# rabbitmq.conf
# Default is 50,000
vm_memory_high_watermark.relative = 0.6
disk_free_limit.absolute = 50GB
```

### Optimize for Low Latency

```python
# Disable Nagle's algorithm
parameters = pika.ConnectionParameters(
    host=AMQP_HOST,
    tcp_options={'TCP_NODELAY': 1}
)
```

### Use Lazy Queues for Large Messages

```python
args = {'x-queue-mode': 'lazy'}
channel.queue_declare(
    queue='large_message_queue',
    durable=True,
    arguments=args
)
```

---

## When All Else Fails

1. **Completely Reset RabbitMQ:**
   ```powershell
   .\rabbitmqctl.bat stop_app
   .\rabbitmqctl.bat reset
   .\rabbitmqctl.bat start_app
   ```

2. **Reinstall RabbitMQ:**
   - Uninstall RabbitMQ
   - Delete data directory
   - Reinstall fresh

3. **Use Docker Instead:**
   ```bash
   docker run -d --name rabbitmq \
       -p 5672:5672 \
       -p 15672:15672 \
       -e RABBITMQ_DEFAULT_USER=fluser \
       -e RABBITMQ_DEFAULT_PASS=flpassword \
       rabbitmq:3-management
   ```

4. **Check GitHub Issues:**
   - [RabbitMQ Server Issues](https://github.com/rabbitmq/rabbitmq-server/issues)
   - [Pika Issues](https://github.com/pika/pika/issues)

---

## Quick Reference Commands

```powershell
# Service Management
Get-Service RabbitMQ
Start-Service RabbitMQ
Stop-Service RabbitMQ
Restart-Service RabbitMQ

# User Management
.\rabbitmqctl.bat list_users
.\rabbitmqctl.bat add_user <username> <password>
.\rabbitmqctl.bat delete_user <username>
.\rabbitmqctl.bat change_password <username> <newpassword>
.\rabbitmqctl.bat set_user_tags <username> administrator

# Permissions
.\rabbitmqctl.bat list_permissions
.\rabbitmqctl.bat set_permissions -p / <username> ".*" ".*" ".*"
.\rabbitmqctl.bat clear_permissions -p / <username>

# Queue Management
.\rabbitmqctl.bat list_queues
.\rabbitmqctl.bat purge_queue <queue_name>
.\rabbitmqctl.bat delete_queue <queue_name>

# Status and Health
.\rabbitmqctl.bat status
.\rabbitmqctl.bat node_health_check
.\rabbitmqctl.bat list_alarms

# Plugins
.\rabbitmq-plugins.bat list
.\rabbitmq-plugins.bat enable rabbitmq_management
.\rabbitmq-plugins.bat disable rabbitmq_management
```

---

## Contact and Support

- **RabbitMQ Documentation:** https://www.rabbitmq.com/documentation.html
- **RabbitMQ Community:** https://groups.google.com/forum/#!forum/rabbitmq-users
- **Pika Documentation:** https://pika.readthedocs.io/
