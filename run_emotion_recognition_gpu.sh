#!/bin/bash
# Quick Start Script for GPU-Enabled Emotion Recognition FL

set -e

echo "================================"
echo "GPU-Enabled Emotion Recognition FL"
echo "================================"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLIENT_ID=${1:-0}
PROTOCOL=${2:-amqp}
AMQP_HOST=${AMQP_HOST:-localhost}
AMQP_PORT=${AMQP_PORT:-5672}

echo -e "${BLUE}Configuration:${NC}"
echo "  Client ID: $CLIENT_ID"
echo "  Protocol: $PROTOCOL"
echo "  AMQP Host: $AMQP_HOST"
echo "  AMQP Port: $AMQP_PORT"
echo ""

# Verify GPU
echo -e "${BLUE}Verifying GPU Setup...${NC}"
python3 << 'EOF'
import tensorflow as tf
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU memory growth enabled")
    except RuntimeError as e:
        print(f"⚠ Warning: {e}")
else:
    print("⚠ No GPUs found - running on CPU")
    print("  This will be slower. Check GPU_SETUP_GUIDE.md for GPU setup instructions.")
EOF

echo ""
echo -e "${BLUE}Starting FL Client (Protocol: $PROTOCOL)...${NC}"
echo ""

# Set environment variables
export CLIENT_ID=$CLIENT_ID
export AMQP_HOST=$AMQP_HOST
export AMQP_PORT=$AMQP_PORT
export NUM_CLIENTS=${NUM_CLIENTS:-2}
export USE_QUANTIZATION=${USE_QUANTIZATION:-true}
export TF_CPP_MIN_LOG_LEVEL=2

# Change to client directory
cd Client/Emotion_Recognition

# Run appropriate client based on protocol
case $PROTOCOL in
    amqp)
        echo -e "${GREEN}Running AMQP Client...${NC}"
        python3 FL_Client_AMQP.py
        ;;
    grpc|grpc)
        echo -e "${GREEN}Running gRPC Client...${NC}"
        python3 FL_Client_gRPC.py
        ;;
    mqtt)
        echo -e "${GREEN}Running MQTT Client...${NC}"
        python3 FL_Client_MQTT.py
        ;;
    dds)
        echo -e "${GREEN}Running DDS Client...${NC}"
        python3 FL_Client_DDS.py
        ;;
    quic)
        echo -e "${GREEN}Running QUIC Client...${NC}"
        python3 FL_Client_QUIC.py
        ;;
    *)
        echo "Unknown protocol: $PROTOCOL"
        echo "Supported: amqp, grpc, mqtt, dds, quic"
        exit 1
        ;;
esac
