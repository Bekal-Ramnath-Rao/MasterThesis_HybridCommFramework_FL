# Running Federated Learning on Different Nodes (Distributed Setup)

## Network Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Node 1        │         │   Node 2        │         │   Node 3        │
│   (Server)      │────────▶│ MQTT Broker     │◀────────│   (Client)      │
│  192.168.1.10   │         │  192.168.1.100  │         │  192.168.1.20   │
└─────────────────┘         └─────────────────┘         └─────────────────┘
                                    ▲
                                    │
                            ┌───────┴────────┐
                            │   Node 4       │
                            │   (Client)     │
                            │  192.168.1.30  │
                            └────────────────┘
```

## Setup Steps

### Step 1: Choose MQTT Broker Location

You need to run an MQTT broker on one node that all clients and server can reach.

**Option A: Dedicated Broker Node**
- Run Mosquitto on a separate machine (e.g., `192.168.1.100`)
- Most scalable for production

**Option B: Broker on Server Node**
- Run Mosquitto on the same machine as FL Server
- Simple for testing

**Option C: Cloud MQTT Broker**
- Use services like HiveMQ Cloud, CloudMQTT, etc.
- Best for geographically distributed nodes

### Step 2: Install MQTT Broker on Broker Node

**On the broker machine:**

```bash
# Windows
choco install mosquitto

# Linux
sudo apt-get update
sudo apt-get install mosquitto mosquitto-clients
```

**Configure Mosquitto to accept external connections:**

Edit `mosquitto.conf` (typically in `C:\Program Files\mosquitto\` or `/etc/mosquitto/`):

```conf
# Listen on all network interfaces
listener 1883 0.0.0.0

# Allow anonymous connections (for testing)
allow_anonymous true

# Optional: Enable logging
log_dest file C:/mosquitto/mosquitto.log
log_type all
```

**Start Mosquitto:**
```bash
# Windows (as Administrator)
net stop mosquitto
net start mosquitto

# Linux
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto
```

**Verify broker is accessible:**
```bash
# From another machine
mosquitto_sub -h 192.168.1.100 -t test/topic -v
```

### Step 3: Configure Firewall

**On the broker node, allow port 1883:**

```powershell
# Windows Firewall
New-NetFirewallRule -DisplayName "MQTT Broker" -Direction Inbound -LocalPort 1883 -Protocol TCP -Action Allow

# Linux (ufw)
sudo ufw allow 1883/tcp
```

### Step 4: Update IP Addresses in launch.json

Edit [.vscode/launch.json](.vscode/launch.json) and replace `192.168.1.100` with your actual broker IP address:

```json
"env": {
    "MQTT_BROKER": "192.168.1.100",  // ← Change this
    "MQTT_PORT": "1883"
}
```

### Step 5: Run on Different Nodes

#### **Node 1: FL Server**

```bash
# Set environment variables
$env:MQTT_BROKER="192.168.1.100"
$env:MQTT_PORT="1883"
$env:NUM_CLIENTS="2"
$env:NUM_ROUNDS="5"

# Run server
cd Server
python FL_Server_MQTT.py
```

Or use VS Code launch configuration: **"FL Server (MQTT) - Remote Broker"**

#### **Node 2: FL Client 0**

```bash
# Set environment variables
$env:MQTT_BROKER="192.168.1.100"
$env:MQTT_PORT="1883"
$env:CLIENT_ID="0"
$env:NUM_CLIENTS="2"

# Run client
cd Client
python FL_Client_MQTT.py
```

Or use VS Code launch configuration: **"FL Client 0 (MQTT) - Remote Broker"**

#### **Node 3: FL Client 1**

```bash
# Set environment variables
$env:MQTT_BROKER="192.168.1.100"
$env:MQTT_PORT="1883"
$env:CLIENT_ID="1"
$env:NUM_CLIENTS="2"

# Run client
cd Client
python FL_Client_MQTT.py
```

Or use VS Code launch configuration: **"FL Client 1 (MQTT) - Remote Broker"**

## Configuration Options

### Environment Variables

Both server and client support these environment variables:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MQTT_BROKER` | MQTT broker IP/hostname | `localhost` | `192.168.1.100` |
| `MQTT_PORT` | MQTT broker port | `1883` | `1883` |
| `CLIENT_ID` | Client identifier (0, 1, 2...) | `0` | `1` |
| `NUM_CLIENTS` | Total number of clients | `2` | `3` |
| `NUM_ROUNDS` | Training rounds | `5` | `10` |

### Network Discovery

To find your machine's IP address:

**Windows:**
```powershell
ipconfig
# Look for IPv4 Address under your active network adapter
```

**Linux:**
```bash
ip addr show
# or
hostname -I
```

### Testing Network Connectivity

**Test MQTT broker connectivity from client/server nodes:**

```bash
# Subscribe to test topic
mosquitto_sub -h 192.168.1.100 -p 1883 -t test/topic

# In another terminal, publish to test topic
mosquitto_pub -h 192.168.1.100 -p 1883 -t test/topic -m "Hello"
```

## Security Considerations

### For Production Deployment:

1. **Enable Authentication:**
   
   Create password file:
   ```bash
   mosquitto_passwd -c /etc/mosquitto/passwd username
   ```
   
   Update `mosquitto.conf`:
   ```conf
   allow_anonymous false
   password_file /etc/mosquitto/passwd
   ```

2. **Enable TLS/SSL:**
   
   Generate certificates and update code:
   ```python
   mqtt_client.tls_set(ca_certs="ca.crt", 
                       certfile="client.crt",
                       keyfile="client.key")
   ```

3. **Use Firewall Rules:**
   - Only allow specific IP addresses
   - Use VPN for internet-distributed nodes

## Troubleshooting

### Connection Refused
- Check firewall settings on broker node
- Verify broker is listening on `0.0.0.0` not `127.0.0.1`
- Test with `telnet 192.168.1.100 1883`

### Clients Not Registering
- Verify all nodes use same `MQTT_BROKER` address
- Check network connectivity: `ping 192.168.1.100`
- Review Mosquitto logs

### Large Model Transfer Issues
- Increase MQTT message size limit in `mosquitto.conf`:
  ```conf
  message_size_limit 10485760  # 10 MB
  ```

### Network Latency
- Monitor round times
- Consider reducing model size
- Use compression for weight serialization

## Example: 3-Node Setup

**Node 1 (192.168.1.10) - Server:**
```powershell
$env:MQTT_BROKER="192.168.1.10"  # Broker runs here
cd Server
python FL_Server_MQTT.py
```

**Node 2 (192.168.1.20) - Client 0:**
```powershell
$env:MQTT_BROKER="192.168.1.10"
$env:CLIENT_ID="0"
cd Client
python FL_Client_MQTT.py
```

**Node 3 (192.168.1.30) - Client 1:**
```powershell
$env:MQTT_BROKER="192.168.1.10"
$env:CLIENT_ID="1"
cd Client
python FL_Client_MQTT.py
```

## Cloud MQTT Broker Example

Using HiveMQ Cloud (free tier):

```powershell
$env:MQTT_BROKER="your-cluster.hivemq.cloud"
$env:MQTT_PORT="8883"  # TLS port
```

Update code to use TLS and authentication for cloud brokers.
