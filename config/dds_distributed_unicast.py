"""
Static unicast SPDP peers for Emotion FL across hosts (no multicast).

Set these environment variables (all three required) before starting server/clients:
  DDS_PEER_SERVER   — IP or hostname of the machine running FL_Server_DDS
  DDS_PEER_CLIENT1  — IP or hostname of the machine running CLIENT_ID=1
  DDS_PEER_CLIENT2  — IP or hostname of the machine running CLIENT_ID=2

Participant indices match config/cyclonedds-emotion-server.xml and client1/2:
  server: 1, client1: 2, client2: 3
SPDP unicast port for domain 0 (CycloneDDS 0.10.x): 7410 + 2 * ParticipantIndex

Optional SPDP address overrides (same logical deployment, different reachability):
  DDS_SPDP_SERVER   — hostname/IP Cyclone uses in <Peer> for participant 1 (default: DDS_PEER_SERVER)
  DDS_SPDP_CLIENT1  — for participant 2 (default: DDS_PEER_CLIENT1)
  DDS_SPDP_CLIENT2  — for participant 3 (default: DDS_PEER_CLIENT2)

Use Docker Compose service names for server/client1 when they share a bridge network, e.g.:
  DDS_SPDP_SERVER=fl-server-dds-emotion DDS_SPDP_CLIENT1=fl-client-dds-emotion-1
  DDS_SPDP_CLIENT2=129.69.102.173
Remote machines keep defaults from DDS_PEER_* (LAN IPs).

Optional:
  DDS_NETWORK_INTERFACE — pin Cyclone to an interface name (e.g. enp68s0) or IPv4 on that NIC.
  On multi-NIC hosts (Wi‑Fi + Ethernet), "auto" may pick the wrong interface; set this explicitly
  or rely on auto-detection below.

  DDS_AUTO_NETWORK_INTERFACE — default "1": on Linux, set NetworkInterfaceAddress from
  `ip -4 route get` to DDS_PEER_SERVER (clients) or to a remote peer (server). Set "0" to disable.

  Do NOT set DDS_NETWORK_INTERFACE to a LAN IP inside a normal Docker bridge container unless
  that IP exists in the container's network namespace.

Open UDP between all three hosts for the ports Cyclone uses (typically ~7400–7500 and
per-participant SPDP ports above).

Docker bridge — two valid setups:
  A) Cross-host (recommended parity with docker-compose-emotion DDS): set DDS_PEER_* and DDS_SPDP_*
     to real LAN IPs for all three participants. Publish host UDP 7412 / 7414 / 7416 to the containers.
  B) All containers on one bridge only: you may set DDS_SPDP_SERVER / DDS_SPDP_CLIENT1 to compose
     service names (e.g. fl-server-unified-emotion) so SPDP targets container IPs. Do not use
     service names on a machine that is not on that Docker network (e.g. client 2 on another PC);
     those names will not resolve and DDS discovery will fail silently from the remote host.
Alternatively use Docker/docker-compose-unified-emotion.host-network.yml for Cyclone on the host NIC.

Server in Docker bridge: DDS_SERVER_ADVERTISE_AS_PEER=1 (default) sets ExternalNetworkAddress = DDS_PEER_SERVER
so **remote** LAN peers reach user-data on a routable address. That same setting often breaks **co-located**
containers on one Docker bridge: writers send to host_IP:ephemeral_UDP, which is not published (only SPDP
ports like 7412/7414 are), so the server never receives ModelUpdateChunk. For server + client1 on the same
bridge, set DDS_DISABLE_EXTERNAL_ADVERTISE=1 on the server (container-to-container locators). Remote DDS
then needs host/macvlan networking or a different topology; see Docker/docker-compose-unified-emotion.host-network.yml.
  DDS_EXTERNAL_NETWORK_ADDRESS / DDS_EXTERNAL_NETWORK_MASK — optional explicit advertise address + mask.

Do not set CYCLONEDDS_URI when using this mode; it is set automatically to a temp file.
"""

from __future__ import annotations

import os
import subprocess
import tempfile


def spdp_port_domain0(participant_index: int) -> int:
    return 7410 + 2 * int(participant_index)


def _peer(host: str, participant_index: int) -> str:
    h = host.strip()
    if not h:
        raise ValueError("peer host must be non-empty")
    return f"{h}:{spdp_port_domain0(participant_index)}"


def _general_xml(
    network_iface: str | None,
    external_address: str | None = None,
    external_mask: str | None = None,
) -> str:
    """Optional ExternalNetworkAddress: advertise LAN IP when bound to Docker-internal eth0."""
    lines = [
        "        <General>",
        "            <AllowMulticast>false</AllowMulticast>",
    ]
    iface = (network_iface or "").strip()
    if iface:
        lines.append(f"            <NetworkInterfaceAddress>{iface}</NetworkInterfaceAddress>")
    ext = (external_address or "").strip()
    if ext:
        lines.append(f"            <ExternalNetworkAddress>{ext}</ExternalNetworkAddress>")
    mask = (external_mask or "").strip()
    if mask and ext:
        lines.append(f"            <ExternalNetworkMask>{mask}</ExternalNetworkMask>")
    lines.append("        </General>")
    return "\n".join(lines)


def _document(
    participant_index: int,
    peer_addrs: list[str],
    network_iface: str | None = None,
    external_address: str | None = None,
    external_mask: str | None = None,
) -> str:
    lines = "\n".join(f'                <Peer address="{a}"/>' for a in peer_addrs)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain id="any">
{_general_xml(network_iface, external_address, external_mask)}
        <Discovery>
            <ParticipantIndex>{participant_index}</ParticipantIndex>
            <Peers>
{lines}
            </Peers>
        </Discovery>
        <Internal>
            <HeartbeatInterval>500ms</HeartbeatInterval>
        </Internal>
        <Tracing>
            <Verbosity>warning</Verbosity>
            <OutputFile>stderr</OutputFile>
        </Tracing>
    </Domain>
</CycloneDDS>
"""


def _write_uri(xml: str) -> str:
    fd, path = tempfile.mkstemp(prefix="cyclonedds-distributed-", suffix=".xml")
    try:
        os.write(fd, xml.encode("utf-8"))
    finally:
        os.close(fd)
    os.environ["CYCLONEDDS_URI"] = "file://" + os.path.abspath(path)
    return path


def _peers_from_env() -> tuple[str, str, str] | None:
    s = os.environ.get("DDS_PEER_SERVER", "").strip()
    c1 = os.environ.get("DDS_PEER_CLIENT1", "").strip()
    c2 = os.environ.get("DDS_PEER_CLIENT2", "").strip()
    if not (s and c1 and c2):
        return None
    return (s, c1, c2)


def _auto_iface_enabled() -> bool:
    return os.environ.get("DDS_AUTO_NETWORK_INTERFACE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _iface_for_route_to(dst: str) -> str | None:
    """Linux: return egress interface for IPv4 route to dst (skip loopback)."""
    if not dst or not str(dst).strip():
        return None
    try:
        out = subprocess.check_output(
            ["ip", "-4", "route", "get", dst.strip()],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    parts = out.split()
    if "dev" not in parts:
        return None
    i = parts.index("dev")
    if i + 1 >= len(parts):
        return None
    iface = parts[i + 1]
    if iface == "lo":
        return None
    return iface


def _bind_iface_server(s: str, c1: str, c2: str) -> str | None:
    explicit = os.environ.get("DDS_NETWORK_INTERFACE", "").strip()
    if explicit:
        return explicit
    if not _auto_iface_enabled():
        return None
    # Prefer route toward a peer on another host (reveals LAN-facing NIC)
    for dst in (c2, c1):
        if dst and dst != s:
            hit = _iface_for_route_to(dst)
            if hit:
                return hit
    return None


def _bind_iface_client(s: str, _c1: str, _c2: str, _cid: int) -> str | None:
    explicit = os.environ.get("DDS_NETWORK_INTERFACE", "").strip()
    if explicit:
        return explicit
    if not _auto_iface_enabled():
        return None
    return _iface_for_route_to(s)


def _spdp_host_for_participant(pi: int, s: str, c1: str, c2: str) -> str:
    """Host part of <Peer> for the given remote participant index (1=server, 2=client1, 3=client2)."""
    if pi == 1:
        return os.environ.get("DDS_SPDP_SERVER", "").strip() or s
    if pi == 2:
        return os.environ.get("DDS_SPDP_CLIENT1", "").strip() or c1
    if pi == 3:
        return os.environ.get("DDS_SPDP_CLIENT2", "").strip() or c2
    raise ValueError(f"invalid participant index {pi}")


def _external_and_mask_for_server(peer_server_ip: str) -> tuple[str | None, str | None]:
    """LAN address to advertise when the process binds to a Docker-internal interface."""
    if os.environ.get("DDS_DISABLE_EXTERNAL_ADVERTISE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None, None
    explicit = os.environ.get("DDS_EXTERNAL_NETWORK_ADDRESS", "").strip()
    mask = os.environ.get("DDS_EXTERNAL_NETWORK_MASK", "").strip()
    if explicit:
        return explicit, (mask or None)
    if os.environ.get("DDS_SERVER_ADVERTISE_AS_PEER", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None, None
    host = (peer_server_ip or "").strip()
    return (host or None), (mask or None)


def _external_and_mask_for_client() -> tuple[str | None, str | None]:
    """Remote clients behind NAT can set DDS_EXTERNAL_NETWORK_ADDRESS to their WAN/LAN IP."""
    explicit = os.environ.get("DDS_EXTERNAL_NETWORK_ADDRESS", "").strip()
    if not explicit:
        return None, None
    mask = os.environ.get("DDS_EXTERNAL_NETWORK_MASK", "").strip()
    return explicit, (mask or None)


def try_apply_server_uri() -> bool:
    """If DDS_PEER_* are set, generate XML for server (ParticipantIndex 1) and set CYCLONEDDS_URI."""
    if os.environ.get("CYCLONEDDS_URI"):
        return False
    t = _peers_from_env()
    if not t:
        return False
    s, c1, c2 = t
    # Other participants: client1 (PI=2), client2 (PI=3)
    peers = [
        _peer(_spdp_host_for_participant(2, s, c1, c2), 2),
        _peer(_spdp_host_for_participant(3, s, c1, c2), 3),
    ]
    bind = _bind_iface_server(s, c1, c2)
    ext, ext_mask = _external_and_mask_for_server(s)
    xml = _document(1, peers, bind, ext, ext_mask)
    _write_uri(xml)
    extra = f" NetworkInterfaceAddress={bind}" if bind else " (interface: auto)"
    ext_extra = f" ExternalNetworkAddress={ext}" if ext else ""
    mask_extra = f" ExternalNetworkMask={ext_mask}" if ext_mask else ""
    print(
        "[DDS] Static unicast peers (server): "
        f"DDS_PEER_SERVER={s} DDS_PEER_CLIENT1={c1} DDS_PEER_CLIENT2={c2}{extra}{ext_extra}{mask_extra} "
        f"peers={peers} "
        f"-> CYCLONEDDS_URI={os.environ['CYCLONEDDS_URI']}"
    )
    return True


def try_apply_client_uri() -> bool:
    """If DDS_PEER_* are set, generate XML for this client (PI 2 or 3) and set CYCLONEDDS_URI."""
    if os.environ.get("CYCLONEDDS_URI"):
        return False
    t = _peers_from_env()
    if not t:
        return False
    s, c1, c2 = t
    try:
        cid = int(os.environ.get("CLIENT_ID", "1"))
    except ValueError:
        cid = 1
    if cid == 1:
        pi = 2
        peers = [
            _peer(_spdp_host_for_participant(1, s, c1, c2), 1),
            _peer(_spdp_host_for_participant(3, s, c1, c2), 3),
        ]
    elif cid == 2:
        pi = 3
        peers = [
            _peer(_spdp_host_for_participant(1, s, c1, c2), 1),
            _peer(_spdp_host_for_participant(2, s, c1, c2), 2),
        ]
    else:
        print(
            f"[DDS] DDS_PEER_* set but CLIENT_ID must be 1 or 2 for static unicast (got {cid}); "
            "falling back to CYCLONEDDS_URI / multicast."
        )
        return False
    bind = _bind_iface_client(s, c1, c2, cid)
    ext, ext_mask = _external_and_mask_for_client()
    xml = _document(pi, peers, bind, ext, ext_mask)
    _write_uri(xml)
    extra = f" NetworkInterfaceAddress={bind}" if bind else " (interface: auto)"
    ext_extra = f" ExternalNetworkAddress={ext}" if ext else ""
    mask_extra = f" ExternalNetworkMask={ext_mask}" if ext_mask else ""
    print(
        f"[DDS] Static unicast peers (client {cid}): "
        f"DDS_PEER_SERVER={s} DDS_PEER_CLIENT1={c1} DDS_PEER_CLIENT2={c2}{extra}{ext_extra}{mask_extra} "
        f"peers={peers} "
        f"-> CYCLONEDDS_URI={os.environ['CYCLONEDDS_URI']}"
    )
    return True
