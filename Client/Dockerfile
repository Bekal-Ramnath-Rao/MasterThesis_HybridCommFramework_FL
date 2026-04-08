# Client Dockerfile for Federated Learning Framework (GPU-enabled)
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

COPY scripts/utilities/packet_logger.py ./
COPY scripts/utilities/q_learning_logger.py ./
COPY scripts/utilities/client_fl_metrics_log.py ./
# DDS: LAN multicast + runtime static unicast generator (DDS_PEER_*)
COPY config/cyclonedds-multicast-lan.xml /app/config/cyclonedds-multicast-lan.xml
COPY config/dds_distributed_unicast.py /app/config/dds_distributed_unicast.py
COPY config/dds_distributed_env.sh /app/config/dds_distributed_env.sh
# Copy the Client directory
COPY Client/ ./Client/
# T_calc / R_tcalc in Q-learning rewards (communication_model, scenario tables)
COPY Network_Simulation/ ./Network_Simulation/

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
# Emotion DDS static unicast (same three on server + both clients; override at runtime if needed)
ENV DDS_PEER_SERVER=129.69.102.245
ENV DDS_PEER_CLIENT1=129.69.102.245
ENV DDS_PEER_CLIENT2=129.69.102.173

# Default command - can be overridden in docker-compose
CMD ["python", "-u", "Client/Emotion_Recognition/FL_Client_MQTT.py"]
