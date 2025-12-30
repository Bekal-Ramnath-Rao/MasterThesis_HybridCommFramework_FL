# Docker Setup for Federated Learning Framework

This document provides comprehensive instructions for running the Federated Learning framework using Docker containers.

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

## 🚀 Quick Start

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

#### MQTT Protocol
```bash
docker-compose up mqtt-broker fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2
```

#### AMQP Protocol
```bash
docker-compose up rabbitmq fl-server-amqp-emotion fl-client-amqp-emotion-1 fl-client-amqp-emotion-2
```

#### gRPC Protocol
```bash
docker-compose up fl-server-grpc-emotion fl-client-grpc-emotion-1 fl-client-grpc-emotion-2
```

#### QUIC Protocol
```bash
docker-compose up fl-server-quic-emotion fl-client-quic-emotion-1 fl-client-quic-emotion-2
```

#### DDS Protocol
```bash
docker-compose up fl-server-dds-emotion fl-client-dds-emotion-1 fl-client-dds-emotion-2
```

### 4. View Logs

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f fl-server-mqtt-emotion
docker-compose logs -f fl-client-mqtt-emotion-1
```

### 5. Stop Containers

```bash
# Stop all containers
docker-compose down

# Stop specific protocol
docker-compose stop fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2
```

## 📊 Accessing Results

Results are stored in mounted volumes and persist on your host machine:

```
Server/Emotion_Recognition/results/
Server/MentalState_Recognition/results/
Server/Temperature_Regulation/results/
```

## 🔧 Configuration

### Environment Variables

You can customize the behavior by modifying environment variables in `docker-compose.yml`:

- `NUM_CLIENTS`: Number of federated learning clients (default: 2)
- `NUM_ROUNDS`: Maximum number of training rounds (default: 1000)
- `CONVERGENCE_THRESHOLD`: Convergence threshold (default: 0.001)
- `CLIENT_ID`: Unique identifier for each client

### Scaling Clients

To add more clients, modify `docker-compose.yml`:

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
# Build all images
docker-compose build

# Build specific service
docker-compose build fl-server-mqtt-emotion

# Build without cache
docker-compose build --no-cache
```

### Running

```bash
# Run in foreground (see logs)
docker-compose up

# Run in background (detached)
docker-compose up -d

# Run specific services
docker-compose up mqtt-broker fl-server-mqtt-emotion
```

### Monitoring

```bash
# List running containers
docker-compose ps

# View logs
docker-compose logs -f

# Execute command in container
docker-compose exec fl-server-mqtt-emotion bash

# Check resource usage
docker stats
```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove containers, networks, and volumes
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Remove everything (including volumes)
docker-compose down -v --rmi all
```

## 🌐 Network Architecture

Each protocol has its own isolated network:

- `fl-mqtt-network`: MQTT broker, server, and clients
- `fl-amqp-network`: RabbitMQ, AMQP server, and clients
- `fl-grpc-network`: gRPC server and clients
- `fl-quic-network`: QUIC server and clients
- `fl-dds-network`: DDS server and clients

This isolation ensures no interference between different protocol tests.

## 🎯 Best Practices

1. **Always generate certificates before running QUIC/gRPC**
2. **Start brokers before servers** (for MQTT/AMQP)
3. **Monitor resource usage** with `docker stats`
4. **Clean up regularly** with `docker-compose down -v`
5. **Use specific service names** instead of running all services at once
6. **Check logs** for debugging: `docker-compose logs -f [service-name]`

## 📈 Running All Protocols Simultaneously

⚠️ **Warning**: Running all protocols simultaneously requires significant resources (16GB+ RAM recommended)

```bash
# Start all protocols (not recommended on low-resource machines)
docker-compose up
```

For better resource management, run one protocol at a time:

```bash
# Test MQTT
docker-compose up mqtt-broker fl-server-mqtt-emotion fl-client-mqtt-emotion-1 fl-client-mqtt-emotion-2
docker-compose down

# Test AMQP
docker-compose up rabbitmq fl-server-amqp-emotion fl-client-amqp-emotion-1 fl-client-amqp-emotion-2
docker-compose down

# ... and so on
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

**Happy Experimenting! 🚀**
