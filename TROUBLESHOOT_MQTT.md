# Troubleshooting MQTT Broker Connection

## Step 1: Verify Mosquitto is Running

On the broker machine (192.168.0.101):

```powershell
# Check if Mosquitto service is running
Get-Service -Name "Mosquitto Broker" | Select-Object Name, Status

# If not running, start it
net start mosquitto
```

## Step 2: Configure Mosquitto for External Connections

Mosquitto by default only listens on localhost. You need to configure it to listen on all interfaces.

### Find Mosquitto Config File

Typical locations:
- Windows: `C:\Program Files\mosquitto\mosquitto.conf`
- Or: `C:\ProgramData\mosquitto\mosquitto.conf`

### Edit mosquitto.conf

Open as Administrator and add/modify these lines:

```conf
# Listen on all network interfaces
listener 1883 0.0.0.0

# Allow anonymous connections (for testing - disable in production)
allow_anonymous true

# Enable logging for debugging
log_dest file C:/mosquitto/mosquitto.log
log_type all
```

**Important:** If the file has `listener 1883 127.0.0.1`, change it to `0.0.0.0`

### Restart Mosquitto

```powershell
# Stop and start to reload config
net stop mosquitto
net start mosquitto
```

## Step 3: Configure Windows Firewall

On the broker machine (192.168.0.101):

```powershell
# Run PowerShell as Administrator
New-NetFirewallRule -DisplayName "MQTT Broker Inbound" -Direction Inbound -LocalPort 1883 -Protocol TCP -Action Allow

New-NetFirewallRule -DisplayName "MQTT Broker Outbound" -Direction Outbound -LocalPort 1883 -Protocol TCP -Action Allow
```

## Step 4: Verify Mosquitto is Listening

On broker machine:

```powershell
# Check if port 1883 is listening
netstat -an | Select-String ":1883"

# Should show something like:
# TCP    0.0.0.0:1883           0.0.0.0:0              LISTENING
```

If it shows `127.0.0.1:1883` instead of `0.0.0.0:1883`, Mosquitto is only listening on localhost!

## Step 5: Test Connection from Client Machine

From the server/client machine trying to connect:

```powershell
# Test if port 1883 is reachable
Test-NetConnection -ComputerName 192.168.0.101 -Port 1883

# Should show: TcpTestSucceeded : True
```

If it fails, the issue is network/firewall.

## Step 6: Test with Mosquitto Clients

On the client/server machine:

```powershell
# Install mosquitto clients if needed
choco install mosquitto

# Try to subscribe
mosquitto_sub -h 192.168.0.101 -p 1883 -t test/topic -v
```

If this works, your FL client/server should also work!

## Quick Fix for Testing

If you want to test quickly on the same machine first:

**Option A: Use localhost for all**
- Keep broker at `localhost` or `127.0.0.1`
- Run server and clients on same machine
- Set `MQTT_BROKER=localhost` for all

**Option B: Use actual IP even on same machine**
- Set `MQTT_BROKER=192.168.0.101` 
- Mosquitto must listen on `0.0.0.0`
- Firewall must allow connections

## Common Issues

### Issue: "Bind failed: Address already in use"
Another process is using port 1883. Kill it or use a different port.

### Issue: Mosquitto won't start after config change
Check syntax in mosquitto.conf. Remove any duplicate `listener` lines.

### Issue: Connection times out (not refused)
- Firewall is blocking
- Network issue
- Wrong IP address

### Issue: Connection refused immediately
- Mosquitto not running
- Mosquitto listening on wrong interface (127.0.0.1 instead of 0.0.0.0)

## Recommended mosquitto.conf for Testing

Create/edit `C:\Program Files\mosquitto\mosquitto.conf`:

```conf
# Persistence settings
persistence true
persistence_location C:/mosquitto/data/

# Logging
log_dest file C:/mosquitto/mosquitto.log
log_type error
log_type warning
log_type notice
log_type information

# Network
listener 1883 0.0.0.0
allow_anonymous true
max_connections 100

# Message size (increase for large models)
message_size_limit 10485760
```

## After Making Changes

Always restart Mosquitto:

```powershell
net stop mosquitto
net start mosquitto

# Verify it's running
Get-Service mosquitto
```

## Alternative: Use Docker

If you keep having issues:

```powershell
# Run Mosquitto in Docker (easier configuration)
docker run -d -p 1883:1883 -p 9001:9001 --name mosquitto eclipse-mosquitto
```
