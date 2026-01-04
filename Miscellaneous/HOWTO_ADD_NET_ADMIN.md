# Example: Adding cap_add: NET_ADMIN to Docker Compose Services

## What to Add

For each FL server and client service in your docker-compose files, add:

```yaml
cap_add:
  - NET_ADMIN
```

## Where to Add It

Add this capability to all services that start with `fl-` (servers and clients), but **NOT** to brokers (mqtt-broker, rabbitmq).

## Example - Before and After

### BEFORE:
```yaml
fl-client-mqtt-emotion-1:
  build:
    context: .
    dockerfile: Client/Dockerfile
  container_name: fl-client-mqtt-emotion-1
  depends_on:
    - fl-server-mqtt-emotion
  environment:
    - MQTT_BROKER=mqtt-broker
    - MQTT_PORT=1883
    - CLIENT_ID=1
  command: python -u Client/Emotion_Recognition/FL_Client_MQTT.py
  networks:
    - fl-mqtt-network
```

### AFTER:
```yaml
fl-client-mqtt-emotion-1:
  build:
    context: .
    dockerfile: Client/Dockerfile
  container_name: fl-client-mqtt-emotion-1
  cap_add:
    - NET_ADMIN
  depends_on:
    - fl-server-mqtt-emotion
  environment:
    - MQTT_BROKER=mqtt-broker
    - MQTT_PORT=1883
    - CLIENT_ID=1
  command: python -u Client/Emotion_Recognition/FL_Client_MQTT.py
  networks:
    - fl-mqtt-network
```

## Services to Update

### In docker-compose-emotion.yml:
- `fl-server-mqtt-emotion`
- `fl-client-mqtt-emotion-1`
- `fl-client-mqtt-emotion-2`
- `fl-server-amqp-emotion`
- `fl-client-amqp-emotion-1`
- `fl-client-amqp-emotion-2`
- `fl-server-grpc-emotion`
- `fl-client-grpc-emotion-1`
- `fl-client-grpc-emotion-2`
- `fl-server-quic-emotion`
- `fl-client-quic-emotion-1`
- `fl-client-quic-emotion-2`
- `fl-server-dds-emotion`
- `fl-client-dds-emotion-1`
- `fl-client-dds-emotion-2`

### In docker-compose-mentalstate.yml:
- All services starting with `fl-server-` and `fl-client-`

### In docker-compose-temperature.yml:
- All services starting with `fl-server-` and `fl-client-`

## Quick Edit Instructions

1. Open each docker-compose file
2. Search for `fl-server-` or `fl-client-`
3. After the `container_name:` line, add:
   ```yaml
   cap_add:
     - NET_ADMIN
   ```
4. Repeat for all FL services

## Automated Method

Run this PowerShell script to add automatically:

```powershell
# Add NET_ADMIN to all FL services in docker-compose files
$files = @(
    "docker-compose-emotion.yml",
    "docker-compose-mentalstate.yml", 
    "docker-compose-temperature.yml"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw
        
        # Add cap_add after container_name for fl- services
        $content = $content -replace `
            '(  fl-[^\s]+:\s+.*?container_name: [^\n]+\n)(?!.*cap_add)', `
            '$1    cap_add:\n      - NET_ADMIN\n'
        
        Set-Content $file $content
        Write-Host "Updated $file"
    }
}
```

## Verification

After updating, verify with:

```bash
# Check if NET_ADMIN is present
grep -A 1 "container_name: fl-" docker-compose-emotion.yml
```

You should see `cap_add:` and `- NET_ADMIN` appearing after each FL service's container_name.
