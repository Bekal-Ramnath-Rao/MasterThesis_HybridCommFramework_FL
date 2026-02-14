#!/usr/bin/env python3
"""
Test script to verify packet_logger.py creates databases correctly.
This simulates what happens inside Docker containers.
"""

import os
import sys

# Resolve project root and packet_logger path (Docker: /app, local: from this file)
_this_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists('/app'):
    _project_root = '/app'
else:
    _project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
_utilities_path = os.path.join(_project_root, 'scripts', 'utilities')
if _utilities_path not in sys.path:
    sys.path.insert(0, _utilities_path)

# Test 1: Simulate server environment
print("=" * 60)
print("TEST 1: Simulating Server Environment")
print("=" * 60)
os.environ['NODE_TYPE'] = 'server'
os.environ.pop('CLIENT_ID', None)

# Reload packet_logger module
if 'packet_logger' in sys.modules:
    del sys.modules['packet_logger']

import packet_logger

print(f"✓ Database path: {packet_logger.DB_PATH}")
print(f"✓ Database exists: {os.path.exists(packet_logger.DB_PATH)}")

# Initialize database
packet_logger.init_db()
print(f"✓ Database initialized: {os.path.exists(packet_logger.DB_PATH)}")

# Log a test packet
packet_logger.log_sent_packet(
    protocol="MQTT",
    message_type="test_message",
    size_bytes=1024,
    destination="test/topic"
)
print("✓ Logged test packet to sent_packets table")

# Test 2: Simulate client environment
print("\n" + "=" * 60)
print("TEST 2: Simulating Client Environment (CLIENT_ID=1)")
print("=" * 60)
os.environ['NODE_TYPE'] = 'client'
os.environ['CLIENT_ID'] = '1'

# Reload packet_logger module
if 'packet_logger' in sys.modules:
    del sys.modules['packet_logger']

import packet_logger as packet_logger_client

print(f"✓ Database path: {packet_logger_client.DB_PATH}")
print(f"✓ Database exists: {os.path.exists(packet_logger_client.DB_PATH)}")

# Initialize database
packet_logger_client.init_db()
print(f"✓ Database initialized: {os.path.exists(packet_logger_client.DB_PATH)}")

# Log a test packet
packet_logger_client.log_received_packet(
    protocol="gRPC",
    message_type="model_update",
    size_bytes=5242880,
    source="server"
)
print("✓ Logged test packet to received_packets table")

# Test 3: Check shared_data directory
print("\n" + "=" * 60)
print("TEST 3: Checking shared_data Directory")
print("=" * 60)

shared_data_dir = os.path.join(os.path.dirname(__file__), 'shared_data')
print(f"Expected directory: {shared_data_dir}")
print(f"Directory exists: {os.path.exists(shared_data_dir)}")

if os.path.exists(shared_data_dir):
    files = os.listdir(shared_data_dir)
    print(f"\nFiles in shared_data/:")
    if files:
        for f in files:
            full_path = os.path.join(shared_data_dir, f)
            size = os.path.getsize(full_path) if os.path.isfile(full_path) else 0
            print(f"  - {f} ({size} bytes)")
    else:
        print("  (empty)")
else:
    print("⚠️  shared_data directory does not exist!")
    print("    Run: mkdir -p shared_data")

print("\n" + "=" * 60)
print("IMPORTANT NOTES")
print("=" * 60)
print("""
1. The .db files are ONLY created when the code runs inside Docker containers.
   
2. When you start Docker containers with docker-compose, the Python code
   inside the containers will:
   - Detect NODE_TYPE environment variable (server or client)
   - Create the appropriate database file in /shared_data
   - The /shared_data inside container is mounted to ./shared_data on host
   
3. To create the databases, you MUST run the Docker containers:
   
   cd Docker
   docker-compose -f docker-compose-unified-emotion.yml up
   
4. After containers start and begin FL training, check:
   
   ls -lh shared_data/
   
   You should see:
   - packet_logs_server.db
   - packet_logs_client_1.db
   - packet_logs_client_2.db
   
5. The databases will remain empty UNTIL actual FL communication happens.
   Wait for training to start (clients register, model updates sent, etc.)

6. Use the GUI's "Packet Logs" tab to view the packets in real-time!
""")
