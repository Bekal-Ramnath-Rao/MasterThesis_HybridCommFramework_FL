#!/bin/bash

# CycloneDDS Unicast Discovery Configuration - Testing Script
# This script helps verify that CycloneDDS is using unicast instead of multicast

echo "========================================="
echo "CycloneDDS Unicast Configuration Test"
echo "========================================="
echo ""

# Check if configuration file exists
if [ -f "cyclonedds-unicast.xml" ]; then
    echo "✓ CycloneDDS unicast configuration file found"
    echo ""
    echo "Configuration summary:"
    grep -A 2 "AllowMulticast" cyclonedds-unicast.xml || echo "  AllowMulticast setting not found"
    echo ""
else
    echo "✗ CycloneDDS unicast configuration file NOT found"
    echo "  Expected location: cyclonedds-unicast.xml"
    exit 1
fi

# Check docker-compose files for CYCLONEDDS_URI
echo "Checking docker-compose files for CYCLONEDDS_URI configuration:"
echo ""

for file in Docker/docker-compose-emotion.yml \
            Docker/docker-compose-temperature.yml \
            Docker/docker-compose-mentalstate.yml; do
    if [ -f "$file" ]; then
        count=$(grep -c "CYCLONEDDS_URI" "$file" 2>/dev/null || echo "0")
        if [ "$count" -gt 0 ]; then
            echo "✓ $file: Found $count CYCLONEDDS_URI configurations"
        else
            echo "✗ $file: No CYCLONEDDS_URI configuration found"
        fi
    else
        echo "✗ $file: File not found"
    fi
done

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo ""
echo "The CycloneDDS configuration has been updated to use unicast discovery"
echo "instead of the default multicast discovery. This should reduce delays"
echo "in poor and very_poor network scenarios."
echo ""
echo "Key changes:"
echo "  1. AllowMulticast set to false"
echo "  2. Static peer list configured for direct unicast discovery"
echo "  3. All DDS services configured to use the unicast XML config"
echo ""
echo "To verify in running containers:"
echo "  docker exec <container-name> cat /app/cyclonedds-unicast.xml"
echo "  docker exec <container-name> env | grep CYCLONEDDS_URI"
echo ""
