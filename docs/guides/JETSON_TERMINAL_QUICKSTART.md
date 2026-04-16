# NVIDIA Jetson — Terminal-Only Experiment Guide

> No GUI required. All experiments are launched from the terminal using
> `run_network_experiments.py` (main machine) and `docker run` (Jetson).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites — Jetson Setup](#2-prerequisites--jetson-setup)
3. [Main Machine Setup](#3-main-machine-setup)
4. [Run Normal (Single-Protocol) Experiments from Main Machine](#4-run-normal-single-protocol-experiments-from-main-machine)
5. [Run Unified (RL-Based) Experiments from Main Machine](#5-run-unified-rl-based-experiments-from-main-machine)
6. [Run a Single Client on Jetson](#6-run-a-single-client-on-jetson)
   - [6a. Normal Use Case Client](#6a-normal-use-case-client)
   - [6b. Unified Use Case Client](#6b-unified-use-case-client)
   - [6c. DDS Client (extra env vars)](#6c-dds-client-extra-env-vars)
7. [Monitoring](#7-monitoring)
8. [Firewall Port Reference](#8-firewall-port-reference)
9. [Environment Variable Reference](#9-environment-variable-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────┐
│           Main Machine (x86)             │
│                                          │
│  • FL Server                             │
│  • MQTT Broker (port 1883)               │
│  • AMQP / RabbitMQ Broker (port 5672)    │
│  • FL Client 1  (local)                  │
└───────────────────┬──────────────────────┘
                    │  LAN / Wi-Fi
                    │
          ┌─────────▼─────────┐
          │   NVIDIA Jetson    │
          │                   │
          │  FL Client 2      │
          │  (this machine)   │
          └───────────────────┘
```

- The main machine starts the server, brokers, and **Client 1**.
- The Jetson runs **Client 2** (or any higher-numbered client).
- Both machines must be on the **same LAN** (or reachable over the network).

---

## 2. Prerequisites — Jetson Setup

### 2.1 Verify NVIDIA runtime

```bash
nvidia-smi
# OR on older Jetson JetPack:
tegrastats
```

### 2.2 Verify Docker with NVIDIA runtime

```bash
docker run --rm --runtime nvidia nvidia/cuda:11.4.0-base-ubuntu20.04 nvidia-smi
```

If this fails, install the NVIDIA container runtime:

```bash
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2.3 Identify your JetPack version

```bash
cat /etc/nv_tegra_release
# OR
dpkg -l | grep jetpack
```

| JetPack | L4T release | Devices | L4T_TAG for build |
|---------|-------------|---------|-------------------|
| 5.1.x   | r35.x.x     | Orin, Xavier AGX/NX | `r35.4.1-tf2.11-py3` *(default)* |
| 4.6.x   | r32.7.x     | Nano, Xavier, TX2   | `r32.7.1-tf2.7-py3` |

### 2.4 Clone the project on Jetson

```bash
git clone <your-repo-url>
cd MasterThesis_HybridCommFramework_FL
```

Or copy from the main machine:

```bash
# Run on the main machine — replace with your Jetson's username and IP
rsync -avz --exclude '.git' \
  /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL/ \
  jetson@<JETSON_IP>:~/MasterThesis_HybridCommFramework_FL/
```

### 2.5 Build the Jetson client image (arm64)

> **Why a separate Dockerfile?**
> The standard `Client/Dockerfile` uses `tensorflow/tensorflow:2.13.0-gpu` which is
> `linux/amd64` only and causes an **"exec format error"** on Jetson.
> `Client/Dockerfile.jetson` uses NVIDIA's official L4T TensorFlow image
> (`nvcr.io/nvidia/l4t-tensorflow`) which is built for `linux/arm64`.

```bash
cd ~/MasterThesis_HybridCommFramework_FL

# JetPack 5.x (Orin / Xavier — default)
docker build \
  -f Client/Dockerfile.jetson \
  -t fl-client-emotion:jetson \
  .

# JetPack 4.6 (Nano / older Xavier)
docker build \
  -f Client/Dockerfile.jetson \
  --build-arg L4T_TAG=r32.7.1-tf2.7-py3 \
  -t fl-client-emotion:jetson \
  .
```

> **Note:** The build pulls ~5 GB from nvcr.io on first run. Make sure you
> have a stable network connection and sufficient disk space (~10 GB free).

---

## 3. Main Machine Setup

### 3.1 Find the main machine's LAN IP

```bash
hostname -I
# Example output: 192.168.1.100
```

Keep this IP handy — it is `SERVER_IP` in all commands below.

### 3.2 Open firewall ports on the main machine

```bash
sudo ufw allow 1883/tcp    # MQTT
sudo ufw allow 5672/tcp    # AMQP / RabbitMQ
sudo ufw allow 50051/tcp   # gRPC
sudo ufw allow 4433/udp    # QUIC
sudo ufw allow 4434/udp    # HTTP/3
# DDS (CycloneDDS RTPS) — replace subnet with your LAN
sudo ufw allow from 192.168.1.0/24 to any port 7400:7500 proto udp
```

### 3.3 Navigate to the project root

```bash
cd /home/ubuntu/Desktop/MT_Ramnath/MasterThesis_HybridCommFramework_FL
```

---

## 4. Run Normal (Single-Protocol) Experiments from Main Machine

Use `--local-clients 1 --min-clients 2` so the server starts **one local client**
and **waits for the Jetson client** before beginning training.

### Single protocol, single scenario

```bash
python3 Network_Simulation/run_network_experiments.py \
  --use-case emotion \
  --single \
  --protocol mqtt \
  --scenario excellent \
  --rounds 50 \
  --enable-gpu \
  --network-mode host \
  --local-clients 1 \
  --min-clients 2
```

### Replace `--protocol` with any of:

| Value  | Description  |
|--------|--------------|
| `mqtt`  | MQTT via Mosquitto |
| `amqp`  | AMQP via RabbitMQ |
| `grpc`  | gRPC (TCP) |
| `quic`  | QUIC (UDP) |
| `http3` | HTTP/3 over QUIC |
| `dds`   | DDS / CycloneDDS |

### Replace `--scenario` with any of:

| Value | Latency | Bandwidth |
|-------|---------|-----------|
| `excellent` | 5 ms | 100 Mbps |
| `good` | 20 ms | 50 Mbps |
| `moderate` | 50 ms | 20 Mbps |
| `poor` | 100 ms | 5 Mbps |
| `very_poor` | 200 ms | 1 Mbps |
| `satellite` | 600 ms | 10 Mbps |
| `dynamic` | shuffle-bag of above | — |

### Termination modes

```bash
# Stop when accuracy converges (default)
--termination-mode client_convergence

# Always run exactly N rounds
--termination-mode fixed_rounds
```

---

## 5. Run Unified (RL-Based) Experiments from Main Machine

```bash
python3 Network_Simulation/run_network_experiments.py \
  --use-case emotion \
  --single \
  --protocol rl_unified \
  --scenario excellent \
  --rounds 100 \
  --enable-gpu \
  --network-mode host \
  --local-clients 1 \
  --min-clients 2
```

### RL training variants

```bash
# RL inference only — greedy, uses existing Q-table (no exploration)
python3 Network_Simulation/run_network_experiments.py \
  --use-case emotion \
  --single --protocol rl_unified --scenario excellent \
  --rounds 50 \
  --enable-gpu --network-mode host \
  --local-clients 1 --min-clients 2 \
  --rl-inference-only

# Resume interrupted training (keeps epsilon, does not reset to 1.0)
python3 Network_Simulation/run_network_experiments.py \
  --use-case emotion \
  --single --protocol rl_unified --scenario dynamic \
  --rounds 200 \
  --enable-gpu --network-mode host \
  --local-clients 1 --min-clients 2 \
  --no-reset-epsilon
```

---

## 6. Run a Single Client on Jetson

> Run these commands **on the Jetson terminal** after the main machine's
> server and Client 1 are already started (Step 4 or 5).

Set your variables once at the top of the terminal session:

```bash
export SERVER_IP=192.168.1.100   # ← replace with main machine's LAN IP
export JETSON_IP=192.168.1.50    # ← replace with Jetson's own LAN IP
export CLIENT_ID=2               # ← must be unique; 2 if main machine runs client 1
export NUM_CLIENTS=2             # ← total clients server expects
export NUM_ROUNDS=50             # ← match what you set on the main machine
export PROJECT_DIR=~/MasterThesis_HybridCommFramework_FL
```

---

### 6a. Normal Use Case Client

```bash
docker run --rm \
  --name fl-client-emotion-jetson \
  --network host \
  --cap-add NET_ADMIN \
  --runtime nvidia \
  -e NODE_TYPE=client \
  -e CLIENT_ID=${CLIENT_ID} \
  -e NUM_CLIENTS=${NUM_CLIENTS} \
  -e NUM_ROUNDS=${NUM_ROUNDS} \
  -e STOP_ON_CLIENT_CONVERGENCE=true \
  -e MQTT_BROKER=${SERVER_IP} \
  -e AMQP_BROKER=${SERVER_IP} \
  -e AMQP_HOST=${SERVER_IP} \
  -e GRPC_HOST=${SERVER_IP} \
  -e GRPC_PORT=50051 \
  -e QUIC_HOST=${SERVER_IP} \
  -e QUIC_PORT=4433 \
  -e HTTP3_HOST=${SERVER_IP} \
  -e HTTP3_PORT=4434 \
  -v ${PROJECT_DIR}/Client/Emotion_Recognition:/app/Client/Emotion_Recognition \
  -v ${PROJECT_DIR}/shared_data:/shared_data \
  -v ${PROJECT_DIR}/experiment_results:/app/results \
  -v ${PROJECT_DIR}/certs:/app/certs:ro \
  fl-client-emotion:jetson \
  python3 Client/Emotion_Recognition/FL_Client_MQTT.py
```

**To use a different protocol**, change only the last line:

| Protocol | Last line |
|----------|-----------|
| MQTT     | `python3 Client/Emotion_Recognition/FL_Client_MQTT.py` |
| AMQP     | `python3 Client/Emotion_Recognition/FL_Client_AMQP.py` |
| gRPC     | `python3 Client/Emotion_Recognition/FL_Client_gRPC.py` |
| QUIC     | `python3 Client/Emotion_Recognition/FL_Client_QUIC.py` |
| HTTP/3   | `python3 Client/Emotion_Recognition/FL_Client_HTTP3.py` |

---

### 6b. Unified Use Case Client

```bash
docker run --rm \
  --name fl-client-unified-jetson \
  --network host \
  --cap-add NET_ADMIN \
  --runtime nvidia \
  -e NODE_TYPE=client \
  -e CLIENT_ID=${CLIENT_ID} \
  -e NUM_CLIENTS=${NUM_CLIENTS} \
  -e NUM_ROUNDS=${NUM_ROUNDS} \
  -e STOP_ON_CLIENT_CONVERGENCE=true \
  -e USE_RL_SELECTION=true \
  -e USE_COMMUNICATION_MODEL_REWARD=true \
  -e RL_BOUNDARY_PIPELINE=true \
  -e RL_PHASE0_ROUNDS=20 \
  -e DEFAULT_PROTOCOL=mqtt \
  -e MQTT_BROKER=${SERVER_IP} \
  -e AMQP_BROKER=${SERVER_IP} \
  -e AMQP_HOST=${SERVER_IP} \
  -e GRPC_HOST=${SERVER_IP} \
  -e GRPC_PORT=50051 \
  -e QUIC_HOST=${SERVER_IP} \
  -e QUIC_PORT=4433 \
  -e HTTP3_HOST=${SERVER_IP} \
  -e HTTP3_PORT=4434 \
  -v ${PROJECT_DIR}/Client/Emotion_Recognition:/app/Client/Emotion_Recognition \
  -v ${PROJECT_DIR}/Network_Simulation:/app/Network_Simulation:ro \
  -v ${PROJECT_DIR}/shared_data:/shared_data \
  -v ${PROJECT_DIR}/experiment_results:/app/results \
  -v ${PROJECT_DIR}/certs:/app/certs:ro \
  fl-client-emotion:jetson \
  python3 Client/Emotion_Recognition/FL_Client_Unified.py
```

**Inference-only mode** (use trained Q-table, no RL exploration):

```bash
# Add this env var to the docker run command above:
  -e USE_RL_EXPLORATION=false \
```

---

### 6c. DDS Client (extra env vars)

DDS requires real LAN IPs — **not** `127.0.0.1`:

```bash
docker run --rm \
  --name fl-client-dds-jetson \
  --network host \
  --cap-add NET_ADMIN \
  --runtime nvidia \
  -e NODE_TYPE=client \
  -e CLIENT_ID=${CLIENT_ID} \
  -e NUM_CLIENTS=${NUM_CLIENTS} \
  -e NUM_ROUNDS=${NUM_ROUNDS} \
  -e STOP_ON_CLIENT_CONVERGENCE=true \
  -e DDS_PEER_SERVER=${SERVER_IP} \
  -e DDS_PEER_CLIENT1=${SERVER_IP} \
  -e DDS_PEER_CLIENT2=${JETSON_IP} \
  -e DDS_SPDP_SERVER=${SERVER_IP} \
  -e DDS_SPDP_CLIENT1=${SERVER_IP} \
  -e DDS_SPDP_CLIENT2=${JETSON_IP} \
  -e DDS_DOMAIN_ID=0 \
  -v ${PROJECT_DIR}/Client/Emotion_Recognition:/app/Client/Emotion_Recognition \
  -v ${PROJECT_DIR}/shared_data:/shared_data \
  -v ${PROJECT_DIR}/experiment_results:/app/results \
  -v ${PROJECT_DIR}/certs:/app/certs:ro \
  fl-client-emotion:jetson \
  python3 Client/Emotion_Recognition/FL_Client_DDS.py
```

> **Important for DDS on Jetson:**
> - `DDS_PEER_CLIENT2` and `DDS_SPDP_CLIENT2` must be set to the **Jetson's own LAN IP**, not `127.0.0.1`.
> - The main machine must also export the same `DDS_PEER_*` and `DDS_SPDP_*` vars.
> - Make sure UDP ports `7400–7500` are open on both machines.

---

## 7. Monitoring

### On the Jetson

```bash
# Live client container logs
docker logs -f fl-client-unified-jetson

# GPU and system usage (Jetson native)
tegrastats

# OR standard nvidia-smi (if available on your JetPack version)
watch -n 2 nvidia-smi
```

### On the main machine

```bash
# Watch all FL container logs together
docker compose -f Docker/docker-compose-unified-emotion.yml logs -f

# Or single container
docker logs -f fl-server-unified-emotion

# GPU usage
watch -n 2 nvidia-smi

# Running containers and resource stats
docker stats
```

---

## 8. Firewall Port Reference

| Protocol | Port  | Transport | Where to open |
|----------|-------|-----------|---------------|
| MQTT     | 1883  | TCP       | Main machine  |
| AMQP     | 5672  | TCP       | Main machine  |
| gRPC     | 50051 | TCP       | Main machine  |
| QUIC     | 4433  | UDP       | Main machine  |
| HTTP/3   | 4434  | UDP       | Main machine  |
| DDS RTPS | 7400–7500 | UDP   | Both machines |

---

## 9. Environment Variable Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `CLIENT_ID` | Unique ID for this client (no duplicates across machines) | `2` |
| `NUM_CLIENTS` | Total clients the FL server waits for | `2` |
| `NUM_ROUNDS` | Number of FL training rounds | `50` |
| `STOP_ON_CLIENT_CONVERGENCE` | `true` = stop early on convergence, `false` = run all rounds | `true` |
| `MQTT_BROKER` | IP of the MQTT broker (main machine) | `192.168.1.100` |
| `AMQP_BROKER` / `AMQP_HOST` | IP of RabbitMQ (main machine) | `192.168.1.100` |
| `GRPC_HOST` | IP of the gRPC server (main machine) | `192.168.1.100` |
| `QUIC_HOST` | IP of the QUIC server (main machine) | `192.168.1.100` |
| `HTTP3_HOST` | IP of the HTTP/3 server (main machine) | `192.168.1.100` |
| `USE_RL_SELECTION` | Enable RL-based protocol switching (unified only) | `true` |
| `USE_COMMUNICATION_MODEL_REWARD` | Include communication-model cost in RL reward | `true` |
| `RL_BOUNDARY_PIPELINE` | Enable phase-0 exploration boundary pipeline | `true` |
| `RL_PHASE0_ROUNDS` | Number of rounds in RL phase 0 (exploration) | `20` |
| `USE_RL_EXPLORATION` | `false` = greedy inference, no epsilon exploration | `false` |
| `DDS_PEER_SERVER` | LAN IP of the DDS server host | `192.168.1.100` |
| `DDS_PEER_CLIENT1` | LAN IP of client 1 host | `192.168.1.100` |
| `DDS_PEER_CLIENT2` | LAN IP of client 2 host (Jetson) | `192.168.1.50` |
| `DDS_SPDP_SERVER` | SPDP peer IP for server | `192.168.1.100` |
| `DDS_SPDP_CLIENT1` | SPDP peer IP for client 1 | `192.168.1.100` |
| `DDS_SPDP_CLIENT2` | SPDP peer IP for client 2 (Jetson) | `192.168.1.50` |
| `DDS_DOMAIN_ID` | CycloneDDS domain | `0` |

---

## 10. Troubleshooting

### Client cannot reach the server

```bash
# Test connectivity from Jetson
ping 192.168.1.100

# Test individual ports
nc -zv 192.168.1.100 1883    # MQTT
nc -zv 192.168.1.100 5672    # AMQP
nc -zv 192.168.1.100 50051   # gRPC
nc -zvu 192.168.1.100 4433   # QUIC (UDP)
```

### "exec format error" / wrong architecture (amd64 vs arm64)

This is the most common issue on Jetson. The standard `Client/Dockerfile` uses
`tensorflow/tensorflow:2.13.0-gpu` which only exists for `linux/amd64`.
Jetson is `linux/arm64` (aarch64). Use the dedicated Jetson Dockerfile:

```bash
cd ~/MasterThesis_HybridCommFramework_FL

# JetPack 5.x (Orin / Xavier — default)
docker build -f Client/Dockerfile.jetson -t fl-client-emotion:jetson .

# JetPack 4.6 (Nano / older Xavier)
docker build \
  -f Client/Dockerfile.jetson \
  --build-arg L4T_TAG=r32.7.1-tf2.7-py3 \
  -t fl-client-emotion:jetson .
```

Confirm the correct architecture after the build:

```bash
docker inspect fl-client-emotion:jetson | grep Architecture
# Expected: "Architecture": "arm64"
```

### Docker image not found on Jetson

```bash
docker images | grep fl-client

# Rebuild using the Jetson-specific Dockerfile (NOT the standard one)
cd ~/MasterThesis_HybridCommFramework_FL
docker build -f Client/Dockerfile.jetson -t fl-client-emotion:jetson .
```

### Container name already in use

```bash
docker ps -a | grep jetson
docker rm fl-client-unified-jetson
```

### nvidia runtime not found

```bash
# Check installed runtimes
docker info | grep -i runtime

# If nvidia is missing
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### DDS participants not discovering each other

- Confirm `DDS_PEER_CLIENT2` = Jetson's **actual LAN IP** (not `127.0.0.1`)
- Confirm the same DDS env vars are set on **both** the main machine and Jetson
- Confirm UDP `7400–7500` is open on both machines
- Confirm you are **not** running both clients on loopback; use `--network host`

### Server starts training before Jetson client joins

- On the main machine, make sure `--min-clients 2` is passed to `run_network_experiments.py`
- Start the Jetson client promptly after the main machine is ready
- Check server logs: `docker logs -f fl-server-unified-emotion`

### GPU not detected inside the container

```bash
# Verify inside a test container
docker run --rm --runtime nvidia fl-client-emotion:jetson \
  python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected output: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

---

*Last updated: April 2026*
