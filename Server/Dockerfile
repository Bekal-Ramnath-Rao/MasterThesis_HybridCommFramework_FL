# Server Dockerfile for Federated Learning Framework (GPU-enabled)
# Use TensorFlow GPU image that includes CUDA/cuDNN
FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies including CycloneDDS
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    iproute2 \
    cmake \
    git \
    tcpdump \
    && rm -rf /var/lib/apt/lists/*

# Install CycloneDDS C library from source
RUN git clone --depth 1 --branch 0.10.5 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds && \
    cd /tmp/cyclonedds && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF .. && \
    cmake --build . && \
    cmake --install . && \
    ldconfig && \
    cd / && rm -rf /tmp/cyclonedds

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (TensorFlow already provided by base image)
RUN pip install --no-cache-dir -r requirements.txt

# Install QUIC support
RUN pip install --no-cache-dir aioquic

# Copy the Protocols directory for gRPC and DDS
COPY Protocols/ ./Protocols/

# Compile gRPC protobuf files
RUN cd Protocols && \
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. federated_learning.proto && \
    cd ..

# Copy the Server directory
COPY scripts/utilities/packet_logger.py ./
COPY Server/ ./Server/

# Copy Client/Compression_Technique for quantization support (needed by Server)
COPY Client/Compression_Technique/ ./Client/Compression_Technique/

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
CMD ["python", "-u", "Server/Emotion_Recognition/FL_Server_MQTT.py"]
