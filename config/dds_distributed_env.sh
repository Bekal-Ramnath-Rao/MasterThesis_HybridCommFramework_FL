#!/usr/bin/env bash
# Source this file on the host before running Emotion FL_Server_DDS / FL_Client_DDS with static unicast:
#   source config/dds_distributed_env.sh
#   python -u Server/Emotion_Recognition/FL_Server_DDS.py
# Same values are set as ENV in Server/Dockerfile and Client/Dockerfile for container runs.
#
# For multicast LAN instead, do NOT source this; set e.g.:
#   export CYCLONEDDS_URI=file://$PWD/config/cyclonedds-multicast-lan.xml
# (CYCLONEDDS_URI takes precedence over DDS_PEER_* in the Python entrypoints.)

export DDS_PEER_SERVER=129.69.102.245
export DDS_PEER_CLIENT1=129.69.102.245
export DDS_PEER_CLIENT2=129.69.102.173

# Optional bind/interface for CycloneDDS (uncomment if needed):
# export DDS_NETWORK_INTERFACE=eth0
