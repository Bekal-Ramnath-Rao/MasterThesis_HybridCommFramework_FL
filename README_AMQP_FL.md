# Federated Learning with AMQP/RabbitMQ

This implementation uses AMQP (Advanced Message Queuing Protocol) with RabbitMQ instead of MQTT for federated learning communication.

## MQTT vs AMQP

| Feature | MQTT (Mosquitto) | AMQP (RabbitMQ) |
|---------|------------------|-----------------|
| Protocol | Lightweight pub/sub | Enterprise messaging |
| Broker | Mosquitto | RabbitMQ |
| Port | 1883 | 5672 |
| Message Pattern | Topics | Exchanges & Queues |
| Routing | Topic wildcards | Routing keys |
| Delivery | At most once/At least once/Exactly once | Guaranteed delivery |
| Persistence | Optional | Built-in durable queues |
| Management | CLI | Web UI (port 15672) |

## Architecture

### AMQP Components

```
┌────────────────────────────────────────────────────────────────┐
│                        RabbitMQ Broker                         │
│                                                                │
│  ┌─────────────────────┐         ┌──────────────────────┐    │
│  │ fl_broadcast        │         │ fl_client_updates    │    │
│  │ (fanout exchange)   │         │ (direct exchange)    │    │
│  └──────┬──────────────┘         └──────┬───────────────┘    │
│         │                                │                     │
│    ┌────┴────┬────────┬─────────┐  ┌───┴────┬──────────┐    │
│    ▼         ▼        ▼         ▼  ▼        ▼          ▼    │
│  Queue    Queue    Queue    Queue  Queue   Queue    Queue    │
│  global   config   train    eval   regist  update  metrics   │
│  model                                                         │
└────────────────────────────────────────────────────────────────┘
     ▲          ▲          ▲          ▲          │        │        │
     │          │          │          │          │        │        │
  Server     Clients    Clients    Clients    Clients  Clients  Clients
```

### Exchanges

1. **fl_broadcast (fanout)**: Server broadcasts to all clients
   - Global model
   - Training config
   - Start training signal
   - Start evaluation signal

2. **fl_client_updates (direct)**: Clients send to server with routing keys
   - `client.register`: Client registration
   - `client.update`: Model weight updates
   - `client.metrics`: Evaluation metrics

## Setup

### 1. Install RabbitMQ

**Windows:**
```powershell
# Using Chocolatey
choco install rabbitmq

# Or download from: https://www.rabbitmq.com/download.html
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install rabbitmq-server
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server
```

**Docker (easiest):**
```powershell
docker run -d --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management
```

### 2. Enable RabbitMQ Management Plugin

```powershell
# Windows
"C:\Program Files\RabbitMQ Server\rabbitmq_server-X.X.X\sbin\rabbitmq-plugins.bat" enable rabbitmq_management

# Linux
sudo rabbitmq-plugins enable rabbitmq_management
```

Access web UI at: http://localhost:15672 (username: guest, password: guest)

### 3. Install Python Dependencies

```powershell
pip install pika
```

### 4. Verify RabbitMQ is Running

```powershell
# Check service status
Get-Service -Name RabbitMQ

# Or check port
Test-NetConnection -ComputerName localhost -Port 5672
```

## Running Federated Learning

### Local Testing

**Terminal 1 - Start Server:**
```powershell
cd Server
python FL_Server_AMQP.py
```

**Terminal 2 - Start Client 0:**
```powershell
$env:CLIENT_ID="0"
cd Client
python FL_Client_AMQP.py
```

**Terminal 3 - Start Client 1:**
```powershell
$env:CLIENT_ID="1"
cd Client
python FL_Client_AMQP.py
```

### Distributed Setup (Different PCs)

#### **PC 1 (RabbitMQ Broker - 192.168.0.101):**
Just keep RabbitMQ running.

#### **PC 2 (Server):**
```powershell
$env:AMQP_HOST="192.168.0.101"
$env:AMQP_PORT="5672"
$env:AMQP_USER="guest"
$env:AMQP_PASSWORD="guest"
$env:NUM_CLIENTS="2"
$env:NUM_ROUNDS="5"

cd Server
python FL_Server_AMQP.py
```

#### **PC 3 (Client 0):**
```powershell
$env:AMQP_HOST="192.168.0.101"
$env:CLIENT_ID="0"
$env:NUM_CLIENTS="2"

cd Client
python FL_Client_AMQP.py
```

#### **PC 4 (Client 1):**
```powershell
$env:AMQP_HOST="192.168.0.101"
$env:CLIENT_ID="1"
$env:NUM_CLIENTS="2"

cd Client
python FL_Client_AMQP.py
```

## Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `AMQP_HOST` | RabbitMQ server address | `localhost` | `192.168.0.101` |
| `AMQP_PORT` | RabbitMQ port | `5672` | `5672` |
| `AMQP_USER` | RabbitMQ username | `guest` | `admin` |
| `AMQP_PASSWORD` | RabbitMQ password | `guest` | `password` |
| `CLIENT_ID` | Client identifier | `0` | `1` |
| `NUM_CLIENTS` | Total clients | `2` | `3` |
| `NUM_ROUNDS` | Training rounds | `5` | `10` |

## Key Differences from MQTT Implementation

### 1. Message Routing

**MQTT:**
```python
# Topic-based
mqtt_client.publish("fl/global_model", message)
mqtt_client.subscribe("fl/client/+/update")
```

**AMQP:**
```python
# Exchange-based with routing keys
channel.basic_publish(exchange='fl_broadcast', routing_key='', body=message)
channel.basic_publish(exchange='fl_client_updates', routing_key='client.update', body=message)
```

### 2. Message Persistence

**MQTT:** Optional, depends on QoS level

**AMQP:** Built-in with durable queues and exchanges
```python
channel.exchange_declare(exchange='fl_broadcast', exchange_type='fanout', durable=True)
properties=pika.BasicProperties(delivery_mode=2)  # Persistent
```

### 3. Broadcasting

**MQTT:** Uses topic pattern matching

**AMQP:** Uses fanout exchange - automatically routes to all bound queues

### 4. Acknowledgments

**MQTT:** QoS-based

**AMQP:** Explicit ack/nack with `auto_ack=True/False`

## Monitoring with RabbitMQ Management UI

Open browser: http://localhost:15672

**Features:**
- View queues and message counts
- Monitor exchange routing
- See connection statistics
- Track message rates
- Debug routing issues

## Troubleshooting

### Connection Refused

```powershell
# Check if RabbitMQ is running
Get-Service RabbitMQ

# Start RabbitMQ
net start RabbitMQ

# Check port
netstat -an | Select-String ":5672"
```

### Authentication Failed

Default credentials: `guest/guest` (only works on localhost)

Create new user for remote access:
```bash
rabbitmqctl add_user myuser mypassword
rabbitmqctl set_permissions -p / myuser ".*" ".*" ".*"
rabbitmqctl set_user_tags myuser administrator
```

Then use:
```powershell
$env:AMQP_USER="myuser"
$env:AMQP_PASSWORD="mypassword"
```

### Firewall Issues

Allow RabbitMQ ports:
```powershell
# AMQP port
New-NetFirewallRule -DisplayName "RabbitMQ AMQP" -Direction Inbound -LocalPort 5672 -Protocol TCP -Action Allow

# Management UI port
New-NetFirewallRule -DisplayName "RabbitMQ Management" -Direction Inbound -LocalPort 15672 -Protocol TCP -Action Allow
```

### Messages Not Routing

Check in Management UI:
1. Exchanges exist and are bound
2. Queues exist and have consumers
3. Messages are being published
4. Routing keys match

### Memory Issues

RabbitMQ can use significant memory. Configure limits in `rabbitmq.conf`:
```conf
vm_memory_high_watermark.relative = 0.6
```

## Performance Comparison

### MQTT (Mosquitto)
- ✅ Lightweight, low overhead
- ✅ Fast for simple pub/sub
- ✅ Easy to set up
- ⚠️ Basic routing
- ⚠️ Limited management

### AMQP (RabbitMQ)
- ✅ Enterprise-grade reliability
- ✅ Advanced routing capabilities
- ✅ Excellent monitoring/management
- ✅ Built-in message persistence
- ⚠️ Higher resource usage
- ⚠️ More complex setup

## Advanced Features

### Message TTL (Time To Live)

```python
# Set message expiration
channel.basic_publish(
    exchange='fl_broadcast',
    routing_key='',
    body=message,
    properties=pika.BasicProperties(
        delivery_mode=2,
        expiration='60000'  # 60 seconds
    )
)
```

### Priority Queues

```python
# Declare queue with priority
channel.queue_declare(
    queue='fl.client.update',
    arguments={'x-max-priority': 10}
)

# Send with priority
channel.basic_publish(
    exchange='',
    routing_key='fl.client.update',
    body=message,
    properties=pika.BasicProperties(priority=5)
)
```

### Dead Letter Exchange

Handle failed messages:
```python
channel.queue_declare(
    queue='fl.client.update',
    arguments={
        'x-dead-letter-exchange': 'fl_dlx',
        'x-message-ttl': 60000
    }
)
```

## Files

- [Client/FL_Client_AMQP.py](Client/FL_Client_AMQP.py) - AMQP client implementation
- [Server/FL_Server_AMQP.py](Server/FL_Server_AMQP.py) - AMQP server implementation

## Output

- Results plot: `fl_amqp_results.png`
- Results JSON: `fl_amqp_results.json`
