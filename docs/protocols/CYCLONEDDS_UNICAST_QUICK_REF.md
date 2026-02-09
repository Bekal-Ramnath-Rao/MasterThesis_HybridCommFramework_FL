# CycloneDDS Unicast Configuration - Quick Reference

## What Changed?

**Before**: CycloneDDS used multicast discovery (default) ❌
**After**: CycloneDDS uses unicast discovery with static peers ✅

## Why?

Multicast discovery causes delays in poor/very_poor network scenarios. Unicast is:
- ✅ Faster in degraded networks
- ✅ More reliable in Docker environments
- ✅ No multicast timeout issues

## Files Changed

1. **New Config File**: `cyclonedds-unicast.xml`
   - Sets `AllowMulticast=false`
   - Defines static peer list

2. **Updated Docker-Compose Files**:
   - `Docker/docker-compose-emotion.yml`
   - `Docker/docker-compose-temperature.yml`
   - `Docker/docker-compose-mentalstate.yml`

## Quick Test

```bash
# Verify configuration
./verify_cyclonedds_unicast.sh

# Test with poor network
cd Docker
docker-compose -f docker-compose-emotion.yml up fl-server-dds-emotion fl-client-dds-emotion-1 fl-client-dds-emotion-2

# Check environment in running container
docker exec fl-server-dds-emotion env | grep CYCLONEDDS_URI
```

## Expected Results

| Network Quality | Multicast (Old) | Unicast (New) |
|----------------|----------------|---------------|
| Good           | ~1-2s          | ~1-2s         |
| Poor           | ~5-10s         | ~2-3s         |
| Very Poor      | 10s+ / timeout | ~3-5s         |

## Verification Checklist

- [x] Config file created: `cyclonedds-unicast.xml`
- [x] `AllowMulticast=false` in config
- [x] Static peers defined in config
- [x] All DDS services have `CYCLONEDDS_URI` env var
- [x] All DDS services mount the config file
- [x] Verification script created and tested

## Next Steps

1. **Rebuild Images** (if needed):
   ```bash
   docker-compose -f Docker/docker-compose-emotion.yml build
   ```

2. **Run Experiments**:
   ```bash
   # Test with poor network scenario
   ./run_comprehensive_experiments.sh
   ```

3. **Compare Results**:
   - Check discovery time in logs
   - Compare initial round latency
   - Verify no timeout errors

## Troubleshooting

**Problem**: Discovery still slow
- Check if config file is mounted: `docker exec <container> ls -la /app/cyclonedds-unicast.xml`
- Verify env var: `docker exec <container> env | grep CYCLONEDDS`

**Problem**: Peers not found
- Check container names match peer addresses in config
- Ensure all containers are on same Docker network

**Problem**: Want to revert
- Remove `CYCLONEDDS_URI` env var from docker-compose
- Remove config volume mount
- Rebuild containers

---

**Status**: ✅ Configuration Complete - Ready for Testing
