#!/bin/bash

echo "================================================"
echo "Testing CycloneDDS Configuration Files"
echo "================================================"
echo ""

# Test 1: Check each config has only relevant peers
echo "Test 1: Verifying scenario-specific peer lists"
echo "-----------------------------------------------"

echo "Emotion config peers:"
grep "Peer address" cyclonedds-emotion.xml || echo "  No peers found"
echo ""

echo "Temperature config peers:"
grep "Peer address" cyclonedds-temperature.xml || echo "  No peers found"
echo ""

echo "Mentalstate config peers:"
grep "Peer address" cyclonedds-mentalstate.xml || echo "  No peers found"
echo ""

# Test 2: Check no deprecated elements
echo "Test 2: Checking for deprecated elements"
echo "-----------------------------------------------"
for file in cyclonedds-*.xml; do
    if grep -q "NetworkInterfaceAddress" "$file"; then
        echo "❌ $file contains deprecated NetworkInterfaceAddress"
    else
        echo "✅ $file - no deprecated elements"
    fi
done
echo ""

# Test 3: Check AllowMulticast is false
echo "Test 3: Verifying unicast configuration"
echo "-----------------------------------------------"
for file in cyclonedds-*.xml; do
    if grep -q "<AllowMulticast>false</AllowMulticast>" "$file"; then
        echo "✅ $file - unicast enabled (AllowMulticast=false)"
    else
        echo "❌ $file - multicast may be enabled"
    fi
done
echo ""

# Test 4: Check docker-compose files
echo "Test 4: Verifying docker-compose configurations"
echo "-----------------------------------------------"
if grep -q "cyclonedds-emotion.xml" Docker/docker-compose-emotion.yml; then
    echo "✅ docker-compose-emotion.yml uses emotion-specific config"
else
    echo "❌ docker-compose-emotion.yml config issue"
fi

if grep -q "cyclonedds-temperature.xml" Docker/docker-compose-temperature.yml; then
    echo "✅ docker-compose-temperature.yml uses temperature-specific config"
else
    echo "❌ docker-compose-temperature.yml config issue"
fi

if grep -q "cyclonedds-mentalstate.xml" Docker/docker-compose-mentalstate.yml; then
    echo "✅ docker-compose-mentalstate.yml uses mentalstate-specific config"
else
    echo "❌ docker-compose-mentalstate.yml config issue"
fi

echo ""
echo "================================================"
echo "Configuration Test Complete!"
echo "================================================"
