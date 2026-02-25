# Configuration Files

This directory contains configuration files for various components of the system.

## Files

### CycloneDDS Configuration Files
- `cyclonedds-unicast.xml` - CycloneDDS unicast configuration (general)
- `cyclonedds-emotion.xml` - CycloneDDS configuration for emotion recognition
- `cyclonedds-mentalstate.xml` - CycloneDDS configuration for mental state recognition
- `cyclonedds-temperature.xml` - CycloneDDS configuration for temperature regulation
- **DDS MACVLAN (static peers, no multicast):** `cyclonedds-server.xml`, `cyclonedds-client1.xml`, `cyclonedds-client2.xml` — used by `Docker/docker-compose-emotion.macvlan.yml`. **Replace placeholders with macvlan IPs** before starting:
  - Get IPs: `docker inspect fl-server-dds-emotion | grep IPAddress`, same for `fl-client-dds-emotion-1`, `fl-client-dds-emotion-2` (after first `up` or from network plan).
  - In `cyclonedds-server.xml`: replace `CLIENT1_IP`, `CLIENT2_IP` with client container IPs.
  - In `cyclonedds-client1.xml` and `cyclonedds-client2.xml`: replace `SERVER_IP` with server container IP.
  - Then: `docker-compose down && docker-compose up`, wait 10s, then verify with `docker exec fl-server-dds-emotion netstat -ulnp | grep 74` and client registration in logs.

### MQTT Configuration
- `mosquitto.conf` - Mosquitto MQTT broker configuration

### Docker Compose
- `docker-compose-unified.yml` - Unified Docker Compose configuration

## Usage

### CycloneDDS Configuration
To use a specific CycloneDDS configuration, set the environment variable:
```bash
export CYCLONEDDS_URI=file://$PWD/config/cyclonedds-unicast.xml
```

### MQTT Broker
To start the MQTT broker with the configuration:
```bash
mosquitto -c config/mosquitto.conf
```

### Docker Compose
To use the unified Docker Compose configuration:
```bash
docker-compose -f config/docker-compose-unified.yml up
```

## Notes

- CycloneDDS XML files configure DDS networking parameters, QoS settings, and discovery mechanisms
- Mosquitto configuration includes broker settings, ports, and authentication
- Docker Compose files define service configurations for containerized deployments
