# Federated Learning with MQTT

This implementation replaces Flower's gRPC-based communication with MQTT for federated learning.

## Architecture

### Communication Flow
1. **Client Registration**: Clients connect to MQTT broker and register with server
2. **Training Configuration**: Server broadcasts training hyperparameters
3. **Round 1 - Initial Training**: 
   - Clients train on local data with synchronized initial weights (same random seed)
   - Clients send model weights (parameters) to server via MQTT
4. **Aggregation**: Server aggregates weights using FedAvg algorithm
5. **Global Model Distribution**: Server sends aggregated weights back to clients
6. **Subsequent Rounds**:
   - Clients receive global model and update their local models
   - Clients train on local data
   - Clients send updated weights to server
7. **Evaluation**: Clients evaluate global model and send metrics
8. **Repeat**: Process continues for configured number of rounds

**Note**: All clients use the same random seed (7) to ensure they start with the same initial model weights in Round 1.

### MQTT Topics
- `fl/global_model` - Server publishes aggregated model weights
- `fl/client_{id}/update` - Clients publish local model updates
- `fl/client_{id}/metrics` - Clients publish evaluation metrics
- `fl/training_config` - Server publishes training configuration
- `fl/start_training` - Server signals start of training round
- `fl/start_evaluation` - Server signals start of evaluation
- `fl/client_register` - Clients register with server

## Setup

### 1. Install MQTT Broker
You need an MQTT broker running. Install Mosquitto:

**Windows:**
```bash
# Download from https://mosquitto.org/download/
# Or using Chocolatey:
choco install mosquitto
```

**Linux:**
```bash
sudo apt-get install mosquitto mosquitto-clients
sudo systemctl start mosquitto
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Dataset
Ensure dataset is available at:
```
Client/Dataset/base_data_baseline_unique.csv
```

## Running Federated Learning

### Step 1: Start MQTT Broker
```bash
mosquitto -v
```

Or if installed as service, it should already be running.

### Step 2: Start Server
```bash
cd Server
python FL_Server_MQTT.py
```

### Step 3: Start Clients
Open separate terminals for each client:

**Client 0:**
```bash
cd Client
python FL_Client_MQTT.py
```

**Client 1:**
Edit `FL_Client_MQTT.py` and change `CLIENT_ID = 1`, then:
```bash
cd Client
python FL_Client_MQTT.py
```

## Configuration

### Server Configuration (FL_Server_MQTT.py)
```python
MQTT_BROKER = "localhost"  # MQTT broker address
MQTT_PORT = 1883          # MQTT broker port
NUM_CLIENTS = 2           # Number of clients
NUM_ROUNDS = 5            # Number of training rounds
```

### Client Configuration (FL_Client_MQTT.py)
```python
MQTT_BROKER = "localhost"  # MQTT broker address
MQTT_PORT = 1883          # MQTT broker port
CLIENT_ID = 0             # Unique client ID (0, 1, 2, ...)
NUM_CLIENTS = 2           # Total number of clients
```

### Training Configuration
Modify in server file:
```python
training_config = {
    "batch_size": 32,
    "local_epochs": 20
}
```

## Output

### Server Output
- Aggregated metrics per round (MSE, MAE, MAPE, Loss)
- Plots saved as `fl_mqtt_results.png`
- Results saved as `fl_mqtt_results.json`

### Client Output
- Training progress per round
- Local evaluation metrics

## Key Features

### 1. Model Weight Serialization
- Weights are serialized using pickle
- Base64 encoded for MQTT transmission
- Efficiently handles numpy arrays

### 2. FedAvg Aggregation
- Weighted average based on number of samples
- Preserves model architecture across aggregation

### 3. Asynchronous Communication
- MQTT callbacks handle messages asynchronously
- Non-blocking communication between server and clients

### 4. Metrics Tracking
- Training metrics: loss, MSE, MAE, MAPE, validation metrics
- Evaluation metrics: loss, MSE, MAE, MAPE
- Aggregated across all clients using weighted average

## Differences from Flower Framework

| Aspect | Flower (gRPC) | MQTT Implementation |
|--------|---------------|---------------------|
| Communication | gRPC | MQTT pub/sub |
| Model Transfer | Built-in serialization | Custom pickle + base64 |
| Aggregation | Strategy classes | Custom FedAvg |
| Synchronization | Built-in | Manual via topics |
| Client Management | Automatic | Manual registration |
| Scalability | High | Broker-dependent |

## Troubleshooting

### MQTT Broker Connection Failed
- Ensure Mosquitto is running: `mosquitto -v`
- Check firewall settings
- Verify MQTT_BROKER and MQTT_PORT

### Clients Not Registering
- Ensure all clients have unique CLIENT_ID
- Check MQTT broker logs
- Verify network connectivity

### Memory Issues with Large Models
- Reduce model size (fewer LSTM units)
- Use compression for weight transfer
- Implement chunked transfer for very large models

### Dataset Not Found
- Verify dataset path in FL_Client_MQTT.py
- Ensure CSV file exists in Client/Dataset/

## Advanced Configuration

### Remote MQTT Broker
To use a remote broker:
```python
MQTT_BROKER = "mqtt.example.com"
MQTT_PORT = 1883
```

### TLS/SSL for MQTT
Add security:
```python
mqtt_client.tls_set(ca_certs="ca.crt", 
                    certfile="client.crt",
                    keyfile="client.key")
```

### More Clients
1. Update NUM_CLIENTS in both server and client files
2. Start each client with unique CLIENT_ID

### More Rounds
```python
NUM_ROUNDS = 10  # Increase training rounds
```

## Performance Metrics

The implementation tracks:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **Loss** (Training loss)

All metrics are aggregated using weighted average based on number of samples.
