"""
Static unicast SPDP peers for Emotion FL across hosts (no multicast).

Set these environment variables (all three required) before starting server/clients:
  DDS_PEER_SERVER   — IP or hostname of the machine running FL_Server_DDS
  DDS_PEER_CLIENT1  — IP or hostname of the machine running CLIENT_ID=1
  DDS_PEER_CLIENT2  — IP or hostname of the machine running CLIENT_ID=2

Participant indices match config/cyclonedds-emotion-server.xml and client1/2:
  server: 1, client1: 2, client2: 3
SPDP unicast port for domain 0 (CycloneDDS 0.10.x): 7410 + 2 * ParticipantIndex

Optional:
  DDS_NETWORK_INTERFACE — only when you need to pin Cyclone to one NIC (e.g. eth0, enp3s0).
  Do NOT set this to a host LAN IP (e.g. 129.x) inside a normal Docker bridge container:
  that address is not assigned to any interface there and Cyclone will fail with
  "does not match an available interface". For multi-host DDS use docker --network host,
  macvlan, or leave DDS_NETWORK_INTERFACE unset and let Cyclone choose.

Open UDP between all three hosts for the ports Cyclone uses (typically ~7400–7500 and
per-participant SPDP ports above).

Do not set CYCLONEDDS_URI when using this mode; it is set automatically to a temp file.
"""

from __future__ import annotations

import os
import tempfile


def spdp_port_domain0(participant_index: int) -> int:
    return 7410 + 2 * int(participant_index)


def _peer(host: str, participant_index: int) -> str:
    h = host.strip()
    if not h:
        raise ValueError("peer host must be non-empty")
    return f"{h}:{spdp_port_domain0(participant_index)}"


def _general_xml(network_iface: str | None) -> str:
    """network_iface: only if DDS_NETWORK_INTERFACE is set (interface name recommended)."""
    iface = (network_iface or "").strip()
    if iface:
        return f"""        <General>
            <AllowMulticast>false</AllowMulticast>
            <NetworkInterfaceAddress>{iface}</NetworkInterfaceAddress>
        </General>"""
    return """        <General>
            <AllowMulticast>false</AllowMulticast>
        </General>"""


def _document(participant_index: int, peer_addrs: list[str], network_iface: str | None = None) -> str:
    lines = "\n".join(f'                <Peer address="{a}"/>' for a in peer_addrs)
    return f"""<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain id="any">
{_general_xml(network_iface)}
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


def try_apply_server_uri() -> bool:
    """If DDS_PEER_* are set, generate XML for server (ParticipantIndex 1) and set CYCLONEDDS_URI."""
    if os.environ.get("CYCLONEDDS_URI"):
        return False
    t = _peers_from_env()
    if not t:
        return False
    s, c1, c2 = t
    # Other participants: client1 (PI=2), client2 (PI=3)
    peers = [_peer(c1, 2), _peer(c2, 3)]
    bind = os.environ.get("DDS_NETWORK_INTERFACE", "").strip() or None
    xml = _document(1, peers, bind)
    _write_uri(xml)
    extra = f" NetworkInterfaceAddress={bind}" if bind else " (interface: auto)"
    print(
        "[DDS] Static unicast peers (server): "
        f"DDS_PEER_SERVER={s} DDS_PEER_CLIENT1={c1} DDS_PEER_CLIENT2={c2}{extra} "
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
        peers = [_peer(s, 1), _peer(c2, 3)]
    elif cid == 2:
        pi = 3
        peers = [_peer(s, 1), _peer(c1, 2)]
    else:
        print(
            f"[DDS] DDS_PEER_* set but CLIENT_ID must be 1 or 2 for static unicast (got {cid}); "
            "falling back to CYCLONEDDS_URI / multicast."
        )
        return False
    bind = os.environ.get("DDS_NETWORK_INTERFACE", "").strip() or None
    xml = _document(pi, peers, bind)
    _write_uri(xml)
    extra = f" NetworkInterfaceAddress={bind}" if bind else " (interface: auto)"
    print(
        f"[DDS] Static unicast peers (client {cid}): "
        f"DDS_PEER_SERVER={s} DDS_PEER_CLIENT1={c1} DDS_PEER_CLIENT2={c2}{extra} "
        f"-> CYCLONEDDS_URI={os.environ['CYCLONEDDS_URI']}"
    )
    return True
