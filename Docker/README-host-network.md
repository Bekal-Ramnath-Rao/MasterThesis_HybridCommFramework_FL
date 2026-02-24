# Host-Network Docker Setup (High-Performance)

Use `docker-compose-emotion.host-network.yml` to run the Emotion FL stack with `network_mode: host`, avoiding Docker bridge latency (helps with DDS and HTTP/3 QUIC latency deviations).

## Port assignment (no conflicts on host)

| Role        | MQTT | AMQP | gRPC | QUIC | HTTP/3 | DDS (SPDP)   |
|------------|------|------|------|------|--------|--------------|
| Broker/Discovery | 1883, 9001 | 5672, 15672 | - | - | - | 7410 (discovery) |
| Server     | -    | -    | 50051 | 4433 | 4434 | 7412         |
| Client 1   | -    | -    | -    | -    | -    | 7414         |
| Client 2   | -    | -    | -    | -    | -    | 7416         |

All services connect to brokers/servers via **127.0.0.1** when using host network.

## HTTP/3 (QUIC) – kernel buffers and GSO

For better throughput and lower latency with QUIC/HTTP3:

1. **Larger UDP buffers** (run on host before starting containers):
   ```bash
   sudo sysctl -w net.core.rmem_max=7340032
   sudo sysctl -w net.core.wmem_max=7340032
   sudo sysctl -w net.core.rmem_default=2097152
   sudo sysctl -w net.core.wmem_default=2097152
   ```
   To make persistent, add to `/etc/sysctl.conf` or a file under `/etc/sysctl.d/`.

2. **Generic Segmentation Offload (GSO)**  
   If your kernel (4.18+) and NIC support GSO, it is typically enabled by default. To check:
   ```bash
   ethtool -k <interface> | grep generic-segmentation
   ```

Compose passes `QUIC_RECV_BUFFER` and `QUIC_SEND_BUFFER` (default 2MB) as env for any app-level tuning; aioquic uses the kernel’s socket buffers, so raising `rmem_max`/`wmem_max` is what matters most.

## DDS Discovery Server

DDS uses **unicast discovery** via a dedicated Discovery Server container (`dds-discovery-emotion`) listening on **127.0.0.1:7410**. The FL server and clients use CycloneDDS configs that point to this peer and use fixed participant indices (1, 2, 3) so each has a unique port (7412, 7414, 7416). Start the discovery service before the DDS server and clients (compose `depends_on` handles this).

## Run

From the repo root:

```bash
cd Docker
docker compose -f docker-compose-emotion.host-network.yml up --build
```

To run only one protocol (e.g. DDS):

```bash
docker compose -f docker-compose-emotion.host-network.yml up dds-discovery-emotion fl-server-dds-emotion fl-client-dds-emotion-1 fl-client-dds-emotion-2
```
