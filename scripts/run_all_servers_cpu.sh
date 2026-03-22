#!/usr/bin/env bash
# Run FL servers for all protocols on CPU (Emotion_Recognition use case).
# Use CUDA_VISIBLE_DEVICES="" so TensorFlow uses CPU only.
#
# Prerequisites:
#   - gRPC, DDS: none (just Python).
#   - AMQP: RabbitMQ running (e.g. sudo systemctl start rabbitmq-server).
#   - MQTT: Mosquitto running (e.g. sudo systemctl start mosquitto).
#   - QUIC/HTTP3: certs in project certs/ (server-cert.pem, server-key.pem).
#
# Ports: gRPC 50051, QUIC 4433, HTTP3 4434. AMQP/MQTT connect to brokers (5672, 1883).

set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=""
SERVER_DIR="$PROJECT_ROOT/Server/Emotion_Recognition"

echo "Project root: $PROJECT_ROOT"
echo "Starting FL servers (CPU) for Emotion_Recognition..."
echo ""

# gRPC
echo "[1/6] Starting gRPC server (port 50051)..."
python3 "$SERVER_DIR/FL_Server_gRPC.py" &
GRPC_PID=$!

# QUIC
echo "[2/6] Starting QUIC server (port 4433)..."
python3 "$SERVER_DIR/FL_Server_QUIC.py" &
QUIC_PID=$!

# HTTP3
echo "[3/6] Starting HTTP/3 server (port 4434)..."
python3 "$SERVER_DIR/FL_Server_HTTP3.py" &
HTTP3_PID=$!

# DDS
echo "[4/6] Starting DDS server..."
python3 "$SERVER_DIR/FL_Server_DDS.py" &
DDS_PID=$!

# AMQP (requires RabbitMQ on localhost:5672)
echo "[5/6] Starting AMQP server (broker localhost:5672)..."
python3 "$SERVER_DIR/FL_Server_AMQP.py" &
AMQP_PID=$!

# MQTT (requires Mosquitto on localhost:1883)
echo "[6/6] Starting MQTT server (broker localhost:1883)..."
python3 "$SERVER_DIR/FL_Server_MQTT.py" &
MQTT_PID=$!

echo ""
echo "Servers started (PIDs: gRPC=$GRPC_PID QUIC=$QUIC_PID HTTP3=$HTTP3_PID DDS=$DDS_PID AMQP=$AMQP_PID MQTT=$MQTT_PID)"
echo "Stop with: kill $GRPC_PID $QUIC_PID $HTTP3_PID $DDS_PID $AMQP_PID $MQTT_PID"
echo "Or: pkill -f 'FL_Server_'"

wait
