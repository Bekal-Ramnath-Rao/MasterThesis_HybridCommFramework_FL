#!/bin/bash
# Stop all native FL and broker processes (server, clients, mosquitto, AMQP broker, run_native_experiments).
# Run with: bash stop_native_experiment.sh   or   ./stop_native_experiment.sh
# Use when Ctrl+C does not stop the experiment (e.g. processes in background or multiple requests blocking).

set -e
echo "Stopping all FL and broker processes..."

# Force kill (SIGKILL) so processes stop even if they are busy handling requests
sudo pkill -9 -f "FL_Server_MQTT.py" 2>/dev/null || true
sudo pkill -9 -f "FL_Client_MQTT.py" 2>/dev/null || true
sudo pkill -9 -f "run_native_experiments.py" 2>/dev/null || true
sudo pkill -9 -f "mosquitto_native.conf" 2>/dev/null || true
sudo pkill -9 -f "amqp_proxy.py" 2>/dev/null || true

# Also stop other protocol servers if running
sudo pkill -9 -f "FL_Server_gRPC.py" 2>/dev/null || true
sudo pkill -9 -f "FL_Server_DDS.py" 2>/dev/null || true
sudo pkill -9 -f "FL_Server_HTTP3.py" 2>/dev/null || true
sudo pkill -9 -f "FL_Server_QUIC.py" 2>/dev/null || true

sleep 2
if pgrep -f "FL_Server_MQTT|FL_Client_MQTT|run_native_experiments|mosquitto_native" >/dev/null 2>&1; then
  echo "Warning: Some processes may still be running. Try running this script again or: sudo pkill -9 -f FL_Client_MQTT; sudo pkill -9 -f FL_Server_MQTT; sudo pkill -9 -f mosquitto_native"
else
  echo "All native FL and broker processes stopped."
fi
