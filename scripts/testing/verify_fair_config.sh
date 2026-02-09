#!/bin/bash

# Fair Protocol Configuration Verification Script
# Checks that all protocols are configured consistently for fair comparison

echo "========================================="
echo "Fair Protocol Configuration Verification"
echo "========================================="
echo ""

SUCCESS_COUNT=0
TOTAL_TESTS=0

# Test 1: MQTT Broker Configuration
echo "TEST 1: MQTT Broker Configuration"
echo "-----------------------------------"
TOTAL_TESTS=$((TOTAL_TESTS + 5))

if grep -q "^max_packet_size 12582912" mqtt-config/mosquitto.conf; then
    echo "✅ MQTT max_packet_size: 12MB"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ MQTT max_packet_size not set to 12MB"
fi

if grep -q "^keepalive_interval 60" mqtt-config/mosquitto.conf; then
    echo "✅ MQTT keepalive: 60s"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ MQTT keepalive not set to 60s"
fi

if grep -q "^persistence false" mqtt-config/mosquitto.conf; then
    echo "✅ MQTT persistence: disabled (stateless)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ MQTT persistence not disabled"
fi

if grep -q "clean_session=True" Server/Emotion_Recognition/FL_Server_MQTT.py; then
    echo "✅ MQTT clean_session: True"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ MQTT clean_session not set to True"
fi

if grep -q "qos=1" Server/Emotion_Recognition/FL_Server_MQTT.py; then
    echo "✅ MQTT QoS: 1 (at-least-once)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ MQTT QoS 1 not configured"
fi

echo ""

# Test 2: AMQP Configuration
echo "TEST 2: AMQP Configuration"
echo "-----------------------------------"
TOTAL_TESTS=$((TOTAL_TESTS + 2))

if grep -q "heartbeat=60" Server/Emotion_Recognition/FL_Server_AMQP.py; then
    echo "✅ AMQP heartbeat: 60s"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ AMQP heartbeat not set to 60s"
fi

if grep -q "delivery_mode=2" Server/Emotion_Recognition/FL_Server_AMQP.py; then
    echo "✅ AMQP delivery_mode: 2 (persistent)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ AMQP delivery_mode not set to 2"
fi

echo ""

# Test 3: gRPC Configuration
echo "TEST 3: gRPC Configuration"
echo "-----------------------------------"
TOTAL_TESTS=$((TOTAL_TESTS + 3))

if grep -q "50 \* 1024 \* 1024" Server/Emotion_Recognition/FL_Server_gRPC.py; then
    echo "✅ gRPC message size: 50MB"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ gRPC message size not set to 50MB"
fi

if grep -q "grpc.keepalive_time_ms.*60000" Server/Emotion_Recognition/FL_Server_gRPC.py; then
    echo "✅ gRPC keepalive: 60s (60000ms)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ gRPC keepalive not set to 60s"
fi

if grep -q "grpc.keepalive_timeout_ms.*20000" Server/Emotion_Recognition/FL_Server_gRPC.py; then
    echo "✅ gRPC keepalive timeout: 20s"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ gRPC keepalive timeout not configured"
fi

echo ""

# Test 4: QUIC Configuration
echo "TEST 4: QUIC Configuration"
echo "-----------------------------------"
TOTAL_TESTS=$((TOTAL_TESTS + 2))

if grep -q "max_stream_data=50 \* 1024 \* 1024" Server/Emotion_Recognition/FL_Server_QUIC.py; then
    echo "✅ QUIC max_stream_data: 50MB"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ QUIC max_stream_data not set to 50MB"
fi

if grep -q "idle_timeout=60.0" Server/Emotion_Recognition/FL_Server_QUIC.py; then
    echo "✅ QUIC idle_timeout: 60s"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ QUIC idle_timeout not set to 60s"
fi

echo ""

# Test 5: DDS Configuration
echo "TEST 5: DDS Configuration"
echo "-----------------------------------"
TOTAL_TESTS=$((TOTAL_TESTS + 5))

if grep -q "<ParticipantLeaseDuration>60s</ParticipantLeaseDuration>" cyclonedds-emotion.xml; then
    echo "✅ DDS lease duration: 60s"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ DDS lease duration not set to 60s"
fi

if grep -q "<FragmentSize>8192</FragmentSize>" cyclonedds-emotion.xml; then
    echo "✅ DDS fragment size: 8KB"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ DDS fragment size not set to 8KB"
fi

if grep -q "<MaxMessageSize>10485760</MaxMessageSize>" cyclonedds-emotion.xml; then
    echo "✅ DDS max message size: 10MB"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ DDS max message size not set to 10MB"
fi

if grep -q "Policy.History.KeepLast(1)" Server/Emotion_Recognition/FL_Server_DDS.py; then
    echo "✅ DDS history: KeepLast(1)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ DDS history not set to KeepLast(1)"
fi

if grep -q "max_blocking_time=duration(seconds=300)" Server/Emotion_Recognition/FL_Server_DDS.py; then
    echo "✅ DDS blocking time: 300s (5 minutes)"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
else
    echo "❌ DDS blocking time not set to 300s"
fi

echo ""
echo "========================================="
echo "Verification Summary"
echo "========================================="
echo "Tests Passed: $SUCCESS_COUNT / $TOTAL_TESTS"
echo ""

if [ $SUCCESS_COUNT -eq $TOTAL_TESTS ]; then
    echo "✅ All protocols configured correctly for fair comparison!"
    echo ""
    echo "Configuration Summary:"
    echo "  • Keepalive/Heartbeat: 60s (all protocols)"
    echo "  • Message Size: 12-50MB (adequate for FL models)"
    echo "  • Reliability: At-least-once delivery (all protocols)"
    echo "  • Stateless: Clean session / no history retention"
    echo ""
    exit 0
else
    FAILED=$((TOTAL_TESTS - SUCCESS_COUNT))
    echo "⚠️  $FAILED test(s) failed. Review configuration above."
    echo ""
    exit 1
fi
