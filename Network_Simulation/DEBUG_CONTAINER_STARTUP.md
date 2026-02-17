# Debugging Container Startup Issues

## Common Reasons Containers Don't Start

### 1. **Service Name Mismatch**
The experiment runner expects specific service names. Check if they match:

**Expected for HTTP/3 Emotion:**
- `fl-server-http3-emotion`
- `fl-client-http3-emotion-1`
- `fl-client-http3-emotion-2`

**Check with:**
```bash
docker compose -f Docker/docker-compose-emotion.gpu-isolated.yml config --services | grep http3
```

### 2. **Containers Start Then Exit Immediately**
Check container logs:
```bash
docker logs fl-server-http3-emotion
docker logs fl-client-http3-emotion-1
docker logs fl-client-http3-emotion-2
```

Common causes:
- Missing dependencies
- Configuration errors
- Port conflicts
- Certificate issues (for HTTP/3/QUIC)

### 3. **Docker Compose File Errors**
Validate the compose file:
```bash
docker compose -f Docker/docker-compose-emotion.gpu-isolated.yml config
```

### 4. **Images Not Built**
Containers need images to be built first:
```bash
docker compose -f Docker/docker-compose-emotion.gpu-isolated.yml build
```

### 5. **Check Container Status**
```bash
# See all containers (running and stopped)
docker ps -a --filter "name=http3"

# Check specific container status
docker inspect fl-server-http3-emotion --format='{{.State.Status}}'
docker inspect fl-server-http3-emotion --format='{{.State.Error}}'
```

### 6. **Check Experiment Runner Output**
The experiment runner prints errors when containers fail to start. Look for:
- `[ERROR] Failed to start server`
- `[ERROR] Failed to start clients`
- Check the `stdout` and `stderr` output

### 7. **Manual Container Start Test**
Try starting containers manually to see detailed errors:
```bash
docker compose -f Docker/docker-compose-emotion.gpu-isolated.yml up -d fl-server-http3-emotion
docker compose -f Docker/docker-compose-emotion.gpu-isolated.yml up -d fl-client-http3-emotion-1
docker compose -f Docker/docker-compose-emotion.gpu-isolated.yml up -d fl-client-http3-emotion-2
```

### 8. **Check Dependencies**
HTTP/3 containers depend on:
- Certificates in `/app/certs/` (server-cert.pem, server-key.pem)
- Network connectivity
- GPU availability (if GPU-isolated compose file)

### 9. **Check Experiment Runner Logs**
The experiment runner logs all commands. Look for:
- The exact `docker compose` command being run
- Any error messages in `result.stderr`
- Container startup stage messages

## Quick Diagnostic Commands

```bash
# 1. Check if services exist in compose file
docker compose -f Docker/docker-compose-emotion.gpu-isolated.yml config --services | grep http3

# 2. Check container status
docker ps -a | grep http3

# 3. Check logs for errors
docker logs fl-server-http3-emotion 2>&1 | tail -50
docker logs fl-client-http3-emotion-1 2>&1 | tail -50

# 4. Check if images exist
docker images | grep fl-

# 5. Validate compose file syntax
docker compose -f Docker/docker-compose-emotion.gpu-isolated.yml config > /dev/null && echo "Syntax OK" || echo "Syntax Error"
```

## What to Look For in Experiment Output

When running experiments, watch for:
1. `[Stage X/4] Starting server/client...` messages
2. `[ERROR] Failed to start...` messages
3. Container names being printed
4. Any exception traces

The experiment runner should print detailed error messages if containers fail to start.
