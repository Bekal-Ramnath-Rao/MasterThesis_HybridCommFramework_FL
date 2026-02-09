# Configuration Files

This directory contains configuration files for various components of the system.

## Files

### CycloneDDS Configuration Files
- `cyclonedds-unicast.xml` - CycloneDDS unicast configuration (general)
- `cyclonedds-emotion.xml` - CycloneDDS configuration for emotion recognition
- `cyclonedds-mentalstate.xml` - CycloneDDS configuration for mental state recognition
- `cyclonedds-temperature.xml` - CycloneDDS configuration for temperature regulation

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
