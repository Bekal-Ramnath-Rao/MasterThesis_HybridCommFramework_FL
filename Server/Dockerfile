# Server Dockerfile for Federated Learning Framework
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

# Copy the Server directory
COPY Server/ ./Server/

# Create results directory
RUN mkdir -p /app/Server/results \
    /app/Server/Emotion_Recognition/results \
    /app/Server/MentalState_Recognition/results \
    /app/Server/Temperature_Regulation/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# Expose ports for different protocols
# MQTT: 1883
# AMQP: 5672
# gRPC: 50051
# QUIC: 4433
# DDS: 7400 (default)
EXPOSE 1883 5672 50051 4433 7400

# Default command - can be overridden in docker-compose
CMD ["python", "-u", "../Server/Emotion_Recognition/FL_Server_MQTT.py"]
