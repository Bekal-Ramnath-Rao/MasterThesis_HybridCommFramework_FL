#!/bin/bash
# Verification script for DDS poor network configuration

echo "=========================================="
echo "DDS Poor Network Configuration Verification"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_count=0
fail_count=0

check_xml_setting() {
    local file=$1
    local setting=$2
    local expected=$3
    local description=$4
    
    if grep -q "$setting.*$expected" "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $description: $expected"
        ((pass_count++))
    else
        echo -e "${RED}✗${NC} $description: Expected $expected"
        ((fail_count++))
    fi
}

echo "Checking cyclonedds-emotion.xml..."
echo "-----------------------------------"

# Check lease duration
check_xml_setting "cyclonedds-emotion.xml" "ParticipantLeaseDuration" "180s" "Lease Duration"

# Check fragment size
check_xml_setting "cyclonedds-emotion.xml" "FragmentSize" "16384" "Fragment Size"

# Check shared memory disabled
check_xml_setting "cyclonedds-emotion.xml" "EnableSharedMemory" "false" "Shared Memory Disabled"

# Check multicast disabled
check_xml_setting "cyclonedds-emotion.xml" "AllowMulticast" "false" "Multicast Disabled"

# Check socket buffers
check_xml_setting "cyclonedds-emotion.xml" "SocketReceiveBufferSize" "10MB" "Socket Receive Buffer"
check_xml_setting "cyclonedds-emotion.xml" "SocketSendBufferSize" "10MB" "Socket Send Buffer"

# Check retransmission tuning
check_xml_setting "cyclonedds-emotion.xml" "HeartbeatResponseDelay" "5s" "Heartbeat Response Delay"
check_xml_setting "cyclonedds-emotion.xml" "NackResponseDelay" "1s" "NACK Response Delay"
check_xml_setting "cyclonedds-emotion.xml" "NackDelay" "2s" "NACK Delay"

# Check defrag limits
check_xml_setting "cyclonedds-emotion.xml" "DefragReliableMaxSamples" "1000" "Defrag Reliable Max Samples"
check_xml_setting "cyclonedds-emotion.xml" "DefragUnreliableMaxSamples" "1000" "Defrag Unreliable Max Samples"

# Check SPDP interval
check_xml_setting "cyclonedds-emotion.xml" "SPDPInterval" "60s" "SPDP Interval"

echo ""
echo "Checking cyclonedds-temperature.xml..."
echo "---------------------------------------"

check_xml_setting "cyclonedds-temperature.xml" "ParticipantLeaseDuration" "180s" "Lease Duration"
check_xml_setting "cyclonedds-temperature.xml" "FragmentSize" "16384" "Fragment Size"
check_xml_setting "cyclonedds-temperature.xml" "EnableSharedMemory" "false" "Shared Memory Disabled"

echo ""
echo "Checking cyclonedds-mentalstate.xml..."
echo "---------------------------------------"

check_xml_setting "cyclonedds-mentalstate.xml" "ParticipantLeaseDuration" "180s" "Lease Duration"
check_xml_setting "cyclonedds-mentalstate.xml" "FragmentSize" "16384" "Fragment Size"
check_xml_setting "cyclonedds-mentalstate.xml" "EnableSharedMemory" "false" "Shared Memory Disabled"

echo ""
echo "Checking FL_Server_DDS.py QoS settings..."
echo "------------------------------------------"

# Check max_blocking_time
if grep -q "max_blocking_time=duration(seconds=600)" "Server/Emotion_Recognition/FL_Server_DDS.py" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} QoS max_blocking_time: 600s"
    ((pass_count++))
else
    echo -e "${RED}✗${NC} QoS max_blocking_time: Expected 600s"
    ((fail_count++))
fi

# Check KeepLast(1)
if grep -q "Policy.History.KeepLast(1)" "Server/Emotion_Recognition/FL_Server_DDS.py" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} History Policy: KeepLast(1)"
    ((pass_count++))
else
    echo -e "${RED}✗${NC} History Policy: Expected KeepLast(1)"
    ((fail_count++))
fi

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo -e "${GREEN}Passed:${NC} $pass_count"
echo -e "${RED}Failed:${NC} $fail_count"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}All checks passed! ✓${NC}"
    echo ""
    echo "Configuration optimized for poor networks:"
    echo "  - Lease: 180s (handles 400s transfers)"
    echo "  - Fragments: 16KB (750 vs 1500)"
    echo "  - Blocking: 600s (10 min timeout)"
    echo "  - Buffers: 10MB (handles 12MB models)"
    echo "  - Retransmission: Tuned for congestion avoidance"
    echo ""
    echo "Ready to test in very_poor network conditions!"
    exit 0
else
    echo -e "${RED}Some checks failed. Please review configuration.${NC}"
    exit 1
fi
