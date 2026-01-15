# Docker Setup for Federated Learning Framework

This document provides comprehensive instructions for running the Federated Learning framework using Docker containers.

> **� NEW: Quantization Compression Support**  
> All protocols now support quantization compression (up to 4x reduction in model size) via simple environment variables! See **[DOCKER_QUANTIZATION_GUIDE.md](../DOCKER_QUANTIZATION_GUIDE.md)** for complete guide.

> **�🌐 NEW: Network Simulation Support**  
> You can now simulate various network conditions (latency, bandwidth, packet loss) to evaluate protocol performance under different scenarios. See **[README_NETWORK_SIMULATION.md](README_NETWORK_SIMULATION.md)** for details!

## 📋 Prerequisites

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 1.29 or higher)
- **OpenSSL** (for generating SSL certificates)
- At least **8GB RAM** available for Docker
- At least **10GB disk space**

## 🏗️ Architecture Overview

The Docker setup includes:

- **Server Containers**: One per protocol (MQTT, AMQP, gRPC, QUIC, DDS)
- **Client Containers**: Multiple clients per protocol (default: 2 clients)
- **Broker Containers**: MQTT (Mosquitto), AMQP (RabbitMQ)
- **Isolated Networks**: Each protocol runs in its own Docker network

## � Use Case Specific Docker Compose Files

The framework includes separate docker-compose files for each use case to avoid path conflicts:

- **[docker-compose.yml](docker-compose.yml)** - Emotion Recognition (default)
- **[docker-compose-mentalstate.yml](docker-compose-mentalstate.yml)** - Mental State Recognition
- **[docker-compose-temperature.yml](docker-compose-temperature.yml)** - Temperature Regulation

Each file has isolated networks, different ports, and appropriate DDS domain IDs to prevent conflicts.
## 🔥 Quantization-Enabled Docker Compose Files

**NEW:** Pre-configured files with quantization compression enabled:

- **[docker-compose-quantized.yml](docker-compose-quantized.yml)** - MQTT with 8-bit quantization (4x compression)
- **[docker-compose-all-protocols-quantized.yml](docker-compose-all-protocols-quantized.yml)** - Multi-protocol with quantization

See **[DOCKER_QUANTIZATION_GUIDE.md](../DOCKER_QUANTIZATION_GUIDE.md)** for complete quantization setup guide.
## �🚀 Quick Start

### 1. Generate SSL Certificates (Required for QUIC/gRPC)

```bash
# Run the certificate generation script
python generate_certs.py
```

This creates self-signed certificates in the `certs/` directory.

### 2. Build Docker Images

```bash
# Build both server and client images
docker-compose build
```

### 3. Run a Specific Protocol

Choose your use case first, then select a protocol:

#### Emotion Recognition (docker-compose.yml)

**MQTT Protocol**
```bash
docker-compose up mqtt-broker fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2
```

**AMQP Protocol**
```bash
docker-compose up rabbitmq fl-server-amqp-emotion fl-client-amqp-emotion-1 fl-client-amqp-emotion-2
```

**gRPC Protocol**
```bash
docker-compose up fl-server-grpc-emotion fl-client-grpc-emotion-1 fl-client-grpc-emotion-2
```

**QUIC Protocol**
```bash
docker-compose up fl-server-quic-emotion fl-client-quic-emotion-1 fl-client-quic-emotion-2
```

**DDS Protocol**
```bash
docker-compose up fl-server-dds-emotion fl-client-dds-emotion-1 fl-client-dds-emotion-2
```

#### Mental State Recognition (docker-compose-mentalstate.yml)

**MQTT Protocol**
```bash
docker-compose -f docker-compose-mentalstate.yml up mqtt-broker-mental fl-server-mqtt-mental fl-client-mqtt-mental-1 fl-client-mqtt-mental-2
```

**AMQP Protocol**
```bash
docker-compose -f docker-compose-mentalstate.yml up rabbitmq-mental fl-server-amqp-mental fl-client-amqp-mental-1 fl-client-amqp-mental-2
```

**gRPC Protocol**
```bash
docker-compose -f docker-compose-mentalstate.yml up fl-server-grpc-mental fl-client-grpc-mental-1 fl-client-grpc-mental-2
```

**QUIC Protocol**
```bash
docker-compose -f docker-compose-mentalstate.yml up fl-server-quic-mental fl-client-quic-mental-1 fl-client-quic-mental-2
```

**DDS Protocol**
```bash
docker-compose -f docker-compose-mentalstate.yml up fl-server-dds-mental fl-client-dds-mental-1 fl-client-dds-mental-2
```

#### Temperature Regulation (docker-compose-temperature.yml)

**MQTT Protocol**
```bash
docker-compose -f docker-compose-temperature.yml up mqtt-broker-temp fl-server-mqtt-temp fl-client-mqtt-temp-1 fl-client-mqtt-temp-2
```

**AMQP Protocol**
```bash
docker-compose -f docker-compose-temperature.yml up rabbitmq-temp fl-server-amqp-temp fl-client-amqp-temp-1 fl-client-amqp-temp-2
```

**gRPC Protocol**
```bash
docker-compose -f docker-compose-temperature.yml up fl-server-grpc-temp fl-client-grpc-temp-1 fl-client-grpc-temp-2
```

**QUIC Protocol**
```bash
docker-compose -f docker-compose-temperature.yml up fl-server-quic-temp fl-client-quic-temp-1 fl-client-quic-temp-2
```

**DDS Protocol**
```bash
docker-compose -f docker-compose-temperature.yml up fl-server-dds-temp fl-client-dds-temp-1 fl-client-dds-temp-2
```

### 4. View Logs

```bash
# View all logs (for specific use case)
docker-compose logs -f

# For Mental State Recognition
docker-compose -f docker-compose-mentalstate.yml logs -f

# For Temperature Regulation
docker-compose -f docker-compose-temperature.yml logs -f

# View specific service logs
docker-compose logs -f fl-server-mqtt-emotion
docker-compose logs -f fl-client-mqtt-emotion-1
```

### 5. Stop Containers

```bash
# Stop Emotion Recognition containers
docker-compose down

# Stop Mental State Recognition containers
docker-compose -f docker-compose-mentalstate.yml down

# Stop Temperature Regulation containers
docker-compose -f docker-compose-temperature.yml down

# Stop specific services
docker-compose stop fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2
```

## 📊 Accessing Results

Results are stored in mounted volumes and persist on your host machine:

```
Server/Emotion_Recognition/results/       # Emotion Recognition results
Server/MentalState_Recognition/results/   # Mental State Recognition results
Server/Temperature_Regulation/results/    # Temperature Regulation results
```

Each use case writes to its own results directory, preventing conflicts.

## 🔧 Configuration

### Port Mappings

Each use case uses different host ports to avoid conflicts:

| Use Case | MQTT | AMQP | RabbitMQ UI | gRPC | QUIC | DDS Domain |
|----------|------|------|-------------|------|------|------------|
| Emotion Recognition | 1883 | 5672 | 15672 | 50051 | 4433/udp | 0 |
| Mental State | 1884 | 5673 | 15673 | 50052 | 4434/udp | 1 |
| Temperature | 1885 | 5674 | 15674 | 50053 | 4435/udp | 2 |

### Environment Variables

You can customize the behavior by modifying environment variables in `docker-compose.yml`:

- `NUM_CLIENTS`: Number of federated learning clients (default: 2)
- `NUM_ROUNDS`: Maximum number of training rounds (default: 1000)
- `CONVERGENCE_THRESHOLD`: Convergence threshold (default: 0.001)
- `CLIENT_ID`: Unique identifier for each client

### Scaling Clients

To add more clients, modify the appropriate docker-compose file (e.g., `docker-compose.yml` for Emotion Recognition):

```yaml
fl-client-mqtt-emotion-3:
  build:
    context: .
    dockerfile: Client/Dockerfile
  container_name: fl-client-mqtt-emotion-3
  depends_on:
    - fl-server-mqtt-emotion
  environment:
    - MQTT_BROKER=mqtt-broker
    - MQTT_PORT=1883
    - CLIENT_ID=3
    - NUM_CLIENTS=3  # Update this on server and all clients
  command: python -u Client/Emotion_Recognition/FL_Client_MQTT.py
  networks:
    - fl-mqtt-network
```

Don't forget to update `NUM_CLIENTS` on the server and all existing clients.

## 🎯 Certificate Management

All QUIC and gRPC services use certificates from the centralized `certs/` directory:

- **Server certificates**: `certs/server-cert.pem`, `certs/server-key.pem`
- **Generated by**: `python generate_certs.py`
- **Mounted in containers**: `/app/certs/`
- **Used by**: All QUIC servers and clients

The certificate paths are now standardized across all use cases, eliminating path conflicts.

## 🐛 Troubleshooting

### Issue: Containers can't connect to broker

**Solution**: Ensure the broker is fully started before clients connect.

```bash
# Start broker first
docker-compose up -d mqtt-broker
# Wait 5 seconds
sleep 5
# Start server and clients
docker-compose up fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2
```

### Issue: Port already in use

**Solution**: Stop conflicting services or change port mappings in `docker-compose.yml`.

```bash
# Check what's using the port
# Windows
netstat -ano | findstr :1883

# Linux/Mac
lsof -i :1883
```

### Issue: Out of memory

**Solution**: Increase Docker's memory allocation:
- Docker Desktop: Settings → Resources → Memory → Increase to at least 8GB

### Issue: Permission denied on certificate files

**Solution**: Ensure proper permissions on certificate files:

```bash
chmod 600 certs/server-key.pem
chmod 644 certs/server-cert.pem
```

### Issue: QUIC connection fails

**Solution**: QUIC uses UDP. Ensure UDP port 4433 is open:

```bash
# Check Docker network
docker network inspect masterthesis_hybridcommframework_fl_fl-quic-network
```

## 📦 Docker Commands Reference

### Building

```bash
# Build all images (works for all compose files)
docker-compose build
docker-compose -f docker-compose-mentalstate.yml build
docker-compose -f docker-compose-temperature.yml build

# Build specific service
docker-compose build fl-server-mqtt-emotion

# Build without cache
docker-compose build --no-cache
```

### Running

```bash
# Run Emotion Recognition in foreground
docker-compose up

# Run Mental State Recognition in background
docker-compose -f docker-compose-mentalstate.yml up -d

# Run Temperature Regulation with specific services
docker-compose -f docker-compose-temperature.yml up mqtt-broker-temp fl-server-mqtt-temp
```

### Monitoring

```bash
# List running containers
docker-compose ps
docker ps

# View logs
docker-compose logs -f
docker-compose -f docker-compose-mentalstate.yml logs -f fl-server-mqtt-mental

# Execute command in container
docker-compose exec fl-server-mqtt-emotion bash

# Check resource usage
docker stats
```

### Cleanup

```bash
# Stop and remove containers for specific use case
docker-compose down
docker-compose -f docker-compose-mentalstate.yml down
docker-compose -f docker-compose-temperature.yml down

# Remove containers, networks, and volumes
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Complete cleanup for all use cases
docker-compose down -v --rmi all
docker-compose -f docker-compose-mentalstate.yml down -v --rmi all
docker-compose -f docker-compose-temperature.yml down -v --rmi all
```

## 🌐 Network Architecture

Each use case and protocol has isolated networks to prevent interference:

**Emotion Recognition (docker-compose.yml)**
- `fl-mqtt-network`: MQTT broker, server, and clients
- `fl-amqp-network`: RabbitMQ, AMQP server, and clients
- `fl-grpc-network`: gRPC server and clients
- `fl-quic-network`: QUIC server and clients
- `fl-dds-network`: DDS server and clients (Domain ID: 0)

**Mental State Recognition (docker-compose-mentalstate.yml)**
- `fl-mqtt-mental-network`: MQTT components
- `fl-amqp-mental-network`: AMQP components
- `fl-grpc-mental-network`: gRPC components
- `fl-quic-mental-network`: QUIC components
- `fl-dds-mental-network`: DDS components (Domain ID: 1)

**Temperature Regulation (docker-compose-temperature.yml)**
- `fl-mqtt-temp-network`: MQTT components
- `fl-amqp-temp-network`: AMQP components
- `fl-grpc-temp-network`: gRPC components
- `fl-quic-temp-network`: QUIC components
- `fl-dds-temp-network`: DDS components (Domain ID: 2)

This isolation ensures no interference between different protocol tests or use cases.

## 🎯 Best Practices

1. **Always generate certificates before running QUIC/gRPC**: `python generate_certs.py`
2. **Start brokers before servers** (for MQTT/AMQP)
3. **Use separate docker-compose files** for different use cases to avoid conflicts
4. **Monitor resource usage** with `docker stats`
5. **Clean up regularly**: `docker-compose down -v` after each experiment
6. **Use specific service names** instead of running all services at once
7. **Check logs** for debugging: `docker-compose logs -f [service-name]`
8. **Different DDS Domain IDs** prevent DDS interference between use cases

## 📈 Running Multiple Use Cases

You can run different use cases simultaneously since they use different ports and networks:

```bash
# Terminal 1: Emotion Recognition - MQTT
docker-compose up mqtt-broker fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2

# Terminal 2: Mental State - gRPC
docker-compose -f docker-compose-mentalstate.yml up fl-server-grpc-mental fl-client-grpc-mental-1 fl-client-grpc-mental-2

# Terminal 3: Temperature - QUIC
docker-compose -f docker-compose-temperature.yml up fl-server-quic-temp fl-client-quic-temp-1 fl-client-quic-temp-2
```

⚠️ **Resource Requirements**: Running multiple use cases simultaneously requires 16GB+ RAM.

For sequential testing (recommended):

```bash
# Test Emotion Recognition - MQTT
docker-compose up mqtt-broker fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2
docker-compose down

# Test Mental State - AMQP
docker-compose -f docker-compose-mentalstate.yml up rabbitmq-mental fl-server-amqp-mental fl-client-amqp-mental-1 fl-client-amqp-mental-2
docker-compose -f docker-compose-mentalstate.yml down

# Test Temperature - gRPC
docker-compose -f docker-compose-temperature.yml up fl-server-grpc-temp fl-client-grpc-temp-1 fl-client-grpc-temp-2
docker-compose -f docker-compose-temperature.yml down
```

## 🔒 Security Notes

- The provided certificates are **self-signed** and for **development/testing only**
- For production deployments, use certificates from a trusted Certificate Authority (CA)
- The default RabbitMQ credentials are `guest/guest` - change these for production
- Consider using Docker secrets for sensitive data in production

## 📚 Additional Resources

- [MQTT Troubleshooting](TROUBLESHOOT_MQTT.md)
- [AMQP Troubleshooting](TROUBLESHOOT_AMQP.md)
- [gRPC Troubleshooting](TROUBLESHOOT_gRPC.md)
- [QUIC Troubleshooting](TROUBLESHOOT_QUIC.md)
- [DDS Troubleshooting](TROUBLESHOOT_DDS.md)

## 💡 Tips for Thesis Work

1. **Document each run**: Save logs and results with timestamps
2. **Compare protocols**: Run same scenario across all protocols
3. **Resource monitoring**: Use `docker stats` to compare resource usage
4. **Network analysis**: Use tools like Wireshark to analyze protocol behavior
5. **Reproducibility**: Docker ensures your experiments are reproducible

## 🤝 Support

If you encounter issues not covered in this guide:

1. Check the protocol-specific troubleshooting guides
2. Review Docker logs: `docker-compose logs -f [service-name]`
3. Verify Docker and Docker Compose versions
4. Ensure sufficient system resources are available

---

## 🎓 Docker Basics & Core Concepts

### Understanding Docker Components

#### 📝 Dockerfile
- **What**: A text file with instructions to build an image
- **Purpose**: Defines what goes into your application container
- **Location in project**: `Client/Dockerfile`, `Server/Dockerfile`
- **Example**:
  ```dockerfile
  FROM python:3.10-slim
  COPY Client/ ./Client/
  ENV CLIENT_ID=1
  CMD ["python", "-u", "Client/Emotion_Recognition/FL_Client_MQTT.py"]
  ```

#### 📦 Image
- **What**: A read-only template/blueprint created from a Dockerfile
- **Purpose**: Package containing your application and all dependencies
- **Analogy**: Like a .exe installer or a class definition
- **How to create**: `docker build` or `docker-compose build`
- **Example**: `masterthesis_fl-client` image

#### 🏃 Container
- **What**: A running instance of an image
- **Purpose**: The actual executing application
- **Analogy**: A running program or an object created from a class
- **How to create**: `docker run` or `docker-compose up`
- **Example**: `fl-client-mqtt-emotion-1` container

#### 📋 docker-compose.yml
- **What**: Configuration file that defines multiple containers
- **Purpose**: Orchestrates multiple containers to work together
- **Example**: Defines broker, server, and 2 clients

### The Docker Flow

```
1. Write Dockerfile (instructions)
        ↓
2. Build Image (docker build)
        ↓
3. Run Container(s) (docker run)
```

**Real Example from Your FL Project:**
```
Client/Dockerfile (1 file)
    ↓ docker build
masterthesis_fl-client (1 IMAGE)
    ↓ docker run
├─ fl-client-mqtt-emotion-1 (CONTAINER 1, CLIENT_ID=1)
└─ fl-client-mqtt-emotion-2 (CONTAINER 2, CLIENT_ID=2)
```

### Key Relationships

| Concept | Count | Creates |
|---------|-------|---------|
| 1 Dockerfile | → | 1 Image |
| 1 Image | → | Many Containers |
| 1 docker-compose.yml | → | Multiple Containers from Multiple Images |

**In Your FL Project:**
- 2 Dockerfiles → 2 Images → 4+ Containers
- Client Dockerfile → Client Image → 2 Client Containers
- Server Dockerfile → Server Image → 1 Server Container

### Your Project Structure

```
Project Root
├── Client/
│   └── Dockerfile (→ fl-client image)
├── Server/
│   └── Dockerfile (→ fl-server image)
├── docker-compose-emotion.yml (Emotion Recognition)
├── docker-compose-mentalstate.yml (Mental State)
└── docker-compose-temperature.yml (Temperature)
```

**Image Count: 3 Images**
1. **Broker Image**: `eclipse-mosquitto:2` (pre-built from Docker Hub)
2. **Server Image**: `masterthesis_fl-server` (built from Server/Dockerfile)
3. **Client Image**: `masterthesis_fl-client` (built from Client/Dockerfile)

**Container Count: 4 Containers per Protocol**

For MQTT Emotion Recognition:
1. `mqtt-broker` (from mosquitto image)
2. `fl-server-mqtt-emotion` (from server image)
3. `fl-client-mqtt-emotion-1` (from client image, CLIENT_ID=1)
4. `fl-client-mqtt-emotion-2` (from client image, CLIENT_ID=2)

---

## 📖 Docker Command Reference

### Building Images

```bash
# Build all images
docker-compose -f docker-compose-emotion.yml build

# Build specific service
docker-compose -f docker-compose-emotion.yml build fl-client-mqtt-emotion-1

# Build without cache (fresh build)
docker-compose -f docker-compose-emotion.yml build --no-cache

# Build and run together
docker-compose -f docker-compose-emotion.yml up --build
```

### Viewing Images & Containers

```bash
# List all images
docker images

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Inspect image details
docker inspect masterthesis_fl-client

# Inspect container details
docker inspect fl-client-mqtt-emotion-1
```

### Monitoring & Debugging

```bash
# Real-time resource usage (CPU, RAM, Network)
docker stats

# Stats for specific container
docker stats fl-client-mqtt-emotion-1

# View logs for all services
docker-compose -f docker-compose-emotion.yml logs

# Follow logs in real-time
docker-compose -f docker-compose-emotion.yml logs -f

# Logs for specific service
docker-compose -f docker-compose-emotion.yml logs -f fl-server-mqtt-emotion

# Last 100 lines
docker-compose -f docker-compose-emotion.yml logs --tail=100

# Save logs to file
docker-compose -f docker-compose-emotion.yml logs > experiment-logs.txt
```

### Executing Commands in Containers

```bash
# Open bash shell in running container
docker exec -it fl-client-mqtt-emotion-1 bash

# Run single command
docker exec fl-client-mqtt-emotion-1 python --version

# Check environment variables
docker exec fl-client-mqtt-emotion-1 env

# List files
docker exec fl-client-mqtt-emotion-1 ls -la /app
```

### Copying Files

```bash
# Copy from container to host
docker cp fl-server-mqtt-emotion:/app/Server/Emotion_Recognition/results/metrics.json ./

# Copy from host to container
docker cp ./test.py fl-client-mqtt-emotion-1:/app/
```

### Restarting Containers

```bash
# Restart specific container
docker restart fl-client-mqtt-emotion-1

# Restart all services
docker-compose -f docker-compose-emotion.yml restart
```

### Advanced Cleanup

```bash
# Remove all stopped containers
docker container prune

# Remove all unused images
docker image prune

# Remove all unused volumes
docker volume prune

# Remove all unused networks
docker network prune

# Nuclear option - remove EVERYTHING (⚠️ use carefully!)
docker system prune -a --volumes

# Check disk usage
docker system df

# See detailed space usage
docker system df -v
```

### Network Commands

```bash
# List networks
docker network ls

# Inspect specific network
docker network inspect masterthesis_fl-mqtt-network

# See which containers are on a network
docker network inspect masterthesis_fl-mqtt-network --format '{{json .Containers}}'
```

### Volume Commands

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect masterthesis_mqtt-data

# Remove specific volume
docker volume rm masterthesis_mqtt-data
```

### Typical Workflow for Experiments

```bash
# 1. Generate certificates (one-time)
python generate_certs.py

# 2. Build images
docker-compose -f docker-compose-emotion.yml build

# 3. Run containers
docker-compose -f docker-compose-emotion.yml up mqtt-broker fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2

# 4. Monitor in another terminal
docker stats

# 5. View logs
docker-compose -f docker-compose-emotion.yml logs -f

# 6. Stop when done
docker-compose -f docker-compose-emotion.yml down
```

### Command Comparison Table

| Task | Command | Description |
|------|---------|-------------|
| **Build** | `docker-compose build` | Create images from Dockerfiles |
| **Run** | `docker-compose up` | Start containers |
| **Run (background)** | `docker-compose up -d` | Start containers in detached mode |
| **Stop** | `docker-compose down` | Stop and remove containers |
| **Logs** | `docker-compose logs -f` | View container output |
| **List containers** | `docker ps` | Show running containers |
| **List images** | `docker images` | List all images |
| **Stats** | `docker stats` | Monitor resource usage |
| **Exec** | `docker exec -it <container> bash` | Access container shell |
| **Clean** | `docker system prune -a` | Remove unused resources |
| **Inspect** | `docker inspect <container>` | View detailed configuration |
| **Copy** | `docker cp` | Copy files to/from container |

### Quick Troubleshooting Commands

```bash
# Container won't start
docker-compose -f docker-compose-emotion.yml logs fl-client-mqtt-emotion-1

# Check if containers are running
docker ps

# See why container stopped
docker ps -a
docker logs fl-client-mqtt-emotion-1

# Rebuild from scratch
docker-compose -f docker-compose-emotion.yml down --rmi all
docker-compose -f docker-compose-emotion.yml build --no-cache
docker-compose -f docker-compose-emotion.yml up

# Check resource limits
docker stats

# Test network connectivity
docker exec fl-client-mqtt-emotion-1 ping mqtt-broker

# View configuration that will be used
docker-compose -f docker-compose-emotion.yml config
```

### Getting Help

```bash
# Docker help
docker --help
docker-compose --help

# Command-specific help
docker build --help
docker run --help

# Check Docker version
docker --version
docker-compose --version

# Test Docker installation
docker run hello-world
```

---

**Happy Experimenting! 🚀**
