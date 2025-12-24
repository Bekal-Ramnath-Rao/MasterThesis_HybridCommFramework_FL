# Federated Learning with QUIC Protocol

This implementation uses the QUIC protocol (HTTP/3 transport) for federated learning communication.

## Overview

QUIC (Quick UDP Internet Connections) is a modern transport protocol that provides:
- Low-latency connection establishment
- Built-in encryption (TLS 1.3)
- Multiplexed streams without head-of-line blocking
- Connection migration support
- Better performance over unreliable networks

## Prerequisites

### Install Required Packages

```bash
pip install aioquic tensorflow scikit-learn pandas numpy matplotlib
```

### Generate SSL Certificates (Required for QUIC)

QUIC requires TLS encryption. Generate self-signed certificates for testing:

```bash
# Navigate to Server directory
cd Server

# Generate private key and certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

## Configuration

Configure the following environment variables:

### Server Configuration
- `QUIC_HOST`: Server hostname (default: "localhost")
- `QUIC_PORT`: Server port (default: 4433)
- `NUM_CLIENTS`: Number of clients (default: 2)
- `NUM_ROUNDS`: Maximum training rounds (default: 1000)
- `CONVERGENCE_THRESHOLD`: Loss improvement threshold (default: 0.001)
- `CONVERGENCE_PATIENCE`: Rounds to wait for improvement (default: 2)
- `MIN_ROUNDS`: Minimum rounds before checking convergence (default: 3)

### Client Configuration
- `QUIC_HOST`: Server hostname (default: "localhost")
- `QUIC_PORT`: Server port (default: 4433)
- `CLIENT_ID`: Client identifier (0, 1, 2, ...)
- `NUM_CLIENTS`: Total number of clients (default: 2)

## Running the System

### Step 1: Start the Server

```powershell
# In PowerShell terminal 1
cd Server
python FL_Server_QUIC.py
```

The server will:
1. Initialize the global LSTM model
2. Wait for all clients to connect
3. Distribute the initial global model
4. Coordinate training rounds
5. Aggregate client updates using FedAvg
6. Monitor convergence
7. Save and display results

### Step 2: Start Clients

Open separate terminals for each client:

```powershell
# Terminal 2 - Client 0
cd Client
$env:CLIENT_ID="0"
python FL_Client_QUIC.py
```

```powershell
# Terminal 3 - Client 1
cd Client
$env:CLIENT_ID="1"
python FL_Client_QUIC.py
```

## Training Flow

1. **Initialization**
   - Server initializes global LSTM model
   - Clients load and partition data
   - Clients connect to server via QUIC

2. **Registration**
   - Clients register with server
   - Server waits for all clients
   - Server distributes initial model and config

3. **Training Loop** (for each round):
   - Server sends global model to clients
   - Server signals start of training
   - Clients train on local data (with random delay)
   - Clients send model updates to server
   - Server aggregates updates (FedAvg)
   - Server signals start of evaluation
   - Clients evaluate and send metrics
   - Server aggregates metrics
   - Server checks convergence

4. **Completion**
   - Training stops on convergence or max rounds
   - Server saves results and displays plots
   - Server notifies clients
   - Clients disconnect

## Features

### QUIC-Specific Advantages
- **Fast Connection Setup**: 0-RTT or 1-RTT handshake
- **Built-in Encryption**: TLS 1.3 integrated
- **Stream Multiplexing**: Multiple concurrent streams without blocking
- **Connection Migration**: Survives IP address changes
- **Improved Loss Recovery**: Better than TCP

### Federated Learning Features
- FedAvg aggregation with weighted averaging
- Convergence-based early stopping
- Random client update delays (0.5-3.0 seconds)
- Comprehensive metrics tracking (MSE, MAE, MAPE, Loss)
- Results visualization and JSON export
- Blocking plot display (closes when you close the window)

## Output

### Console Output
- Connection status and client registration
- Training progress per round
- Aggregated metrics after each round
- Convergence detection
- Training completion summary

### Generated Files
- `Server/results/quic_training_metrics.png` - Training metrics visualization
- `Server/results/quic_training_results.json` - Detailed results in JSON format

### Results JSON Structure
```json
{
  "rounds": [1, 2, 3, ...],
  "mse": [...],
  "mae": [...],
  "mape": [...],
  "loss": [...],
  "convergence_time_seconds": 123.45,
  "convergence_time_minutes": 2.06,
  "total_rounds": 5,
  "num_clients": 2
}
```

## Architecture

### QUIC Communication Pattern

```
Server (FL_Server_QUIC.py)
    ↕ QUIC Streams (Multiplexed, Encrypted)
Clients (FL_Client_QUIC.py)
```

### Message Types

1. **register**: Client → Server
   - Client ID registration

2. **training_config**: Server → Clients
   - Batch size, epochs configuration

3. **global_model**: Server → Clients
   - Serialized model weights
   - Round number

4. **start_training**: Server → Clients
   - Signal to begin local training

5. **model_update**: Client → Server
   - Updated model weights
   - Training metrics
   - Number of samples

6. **start_evaluation**: Server → Clients
   - Signal to evaluate model

7. **metrics**: Client → Server
   - Evaluation metrics
   - Number of test samples

8. **training_complete**: Server → Clients
   - Training finished notification

## Model Architecture

LSTM-based regression model:
- Input: 4 features (Ambient_Temp, Cabin_Temp, Relative_Humidity, Solar_Load)
- LSTM layer: 50 units, ReLU activation
- Dense output layer: 1 unit
- Loss: Mean Squared Error
- Optimizer: Adam
- Metrics: MSE, MAE, MAPE

## Data Partitioning

- Data is partitioned equally among clients
- Each client uses 80% for training, 20% for testing
- Ensures no data overlap between clients (privacy-preserving)

## Performance Considerations

### QUIC Benefits
- Lower latency than TCP-based protocols
- Better performance over lossy networks
- Efficient multiplexing reduces delays
- 0-RTT resumption for repeat connections

### Potential Issues
- Requires certificate management
- UDP may be blocked by some firewalls
- Library support still evolving

## Security

- Built-in TLS 1.3 encryption
- For production: Use proper CA-signed certificates
- Current setup: Self-signed certificates for testing
- Connection authentication via TLS

## Troubleshooting

See `TROUBLESHOOT_QUIC.md` for common issues and solutions.

## Comparison with Other Protocols

| Feature | QUIC | MQTT | AMQP | gRPC | DDS |
|---------|------|------|------|------|-----|
| Transport | UDP | TCP | TCP | HTTP/2 | UDP/TCP |
| Encryption | Built-in TLS 1.3 | Optional | Optional | Optional | Optional |
| Latency | Very Low | Low | Medium | Low | Very Low |
| Overhead | Low | Very Low | Medium | Low | Low |
| Stream Mux | Yes | No | No | Yes | Yes |
| Setup | 0-1 RTT | 1-2 RTT | 2-3 RTT | 1-2 RTT | 1-2 RTT |

## References

- QUIC Protocol: https://www.chromium.org/quic
- aioquic Library: https://github.com/aiortc/aioquic
- RFC 9000: QUIC Transport Protocol
- Federated Learning: https://federated.withgoogle.com/
