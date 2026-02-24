#!/usr/bin/env python3
"""
Minimal DDS Discovery Server for CycloneDDS (unicast discovery).
Creates a single DomainParticipant with ParticipantIndex 0 (listens on port 7410).
Other participants use this as peer (127.0.0.1:7410) with network_mode: host.
Run with: CYCLONEDDS_URI=file:///app/config/cyclonedds-discovery-server.xml python scripts/dds_discovery_server.py
"""
import os
import sys
import time
import signal

# Set CycloneDDS config before any cyclonedds import (native lib may read at load time)
if not os.environ.get("CYCLONEDDS_URI") and os.path.exists("/app/config/cyclonedds-discovery-server.xml"):
    os.environ["CYCLONEDDS_URI"] = "file:///app/config/cyclonedds-discovery-server.xml"

# Ensure config path exists when run from Docker
if os.path.exists("/app"):
    sys.path.insert(0, "/app")

def main():
    domain_id = int(os.getenv("DDS_DOMAIN_ID", "0"))
    uri = os.getenv("CYCLONEDDS_URI", "")
    if not uri and os.path.exists("/app/config/cyclonedds-discovery-server.xml"):
        uri = "file:///app/config/cyclonedds-discovery-server.xml"
        os.environ["CYCLONEDDS_URI"] = uri
    if not uri or "cyclonedds-discovery-server" not in uri:
        print("Set CYCLONEDDS_URI to file:///path/to/cyclonedds-discovery-server.xml", file=sys.stderr)
    print(f"[DDS] CYCLONEDDS_URI={uri or '(not set)'}")
    print(f"DDS Discovery Server starting on domain {domain_id} (participant index 0, port 7410)...")
    from cyclonedds.domain import DomainParticipant
    participant = DomainParticipant(domain_id)
    print("Discovery Server participant created. Peers can use 127.0.0.1:7410 as peer.")
    def shutdown(*_):
        print("Shutting down Discovery Server...")
        participant = None
        sys.exit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    while True:
        time.sleep(3600)

if __name__ == "__main__":
    main()
