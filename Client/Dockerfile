# Client Dockerfile for Federated Learning Framework
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install QUIC support
RUN pip install --no-cache-dir aioquic

# Copy the Protocols directory for gRPC and DDS
COPY Protocols/ ./Protocols/

# Copy the Client directory
COPY Client/ ./Client/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# Default client configuration (can be overridden)
ENV CLIENT_ID=1
ENV NUM_CLIENTS=2
ENV MQTT_BROKER=mqtt-broker
ENV AMQP_HOST=rabbitmq
ENV GRPC_HOST=fl-server
ENV QUIC_HOST=fl-server
ENV DDS_DOMAIN_ID=0

# Default command - can be overridden in docker-compose
CMD ["python", "-u", "../Client/Emotion_Recognition/FL_Client_MQTT.py"]
