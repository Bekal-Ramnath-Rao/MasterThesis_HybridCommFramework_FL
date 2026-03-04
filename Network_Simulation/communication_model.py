#!/usr/bin/env python3
"""
Communication model: collect network parameters using iperf3 (client-side).

Used for T_calc in diagnostic pipeline and for RL reward impact.
- Docker: run iperf3 inside the client container (client → server).
- Native: run iperf3 from the client script / client namespace.

Output is stored in JSON under shared_data (iperf3_network_params.json).
"""

import json
import math
import re
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Shared path for α_proto JSON (kept in shared_data, updated by diagnostic pipeline)
ALPHA_PATH = PROJECT_ROOT / "shared_data" / "alpha_proto.json"

# Default iperf3 server port
IPERF3_PORT = int(os.environ.get("IPERF3_PORT", "5201"))

# T_calc constants (aligned with diagnostic_pipeline)
MSS_BITS = 11680
T_REPAIR = 0.005
O_TLS_HANDSHAKE_S = float(os.environ.get("O_TLS_HANDSHAKE_S", "0.003"))
K_TLS_SEC_PER_BYTE = float(os.environ.get("K_TLS_SEC_PER_BYTE", "2e-8"))

# Protocol-specific scaling factors α_proto for analytical network term
ALPHA_PROTO = {
    "MQTT": 1.8,
    "AMQP": 2.1,
    "GRPC": 1.7,
    "QUIC": 1.0,
    "HTTP3": 4.5,
    "DDS": 8.2,
}


def load_alpha_proto(path: Optional[Path] = None) -> Dict[str, float]:
    """Load α_proto from JSON if available."""
    p = Path(path) if path is not None else ALPHA_PATH
    if not p.exists():
        return {}
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return {str(k).upper(): float(v) for k, v in data.items()}


def init_alpha_proto_from_disk() -> None:
    """
    Initialize ALPHA_PROTO from the persisted alpha_proto.json produced
    by the diagnostic pipeline, if present.
    """
    global ALPHA_PROTO
    loaded = load_alpha_proto(ALPHA_PATH)
    if loaded:
        ALPHA_PROTO.update(loaded)


def _run_iperf3_cmd(cmd: list, timeout_sec: int = 30, cwd: Optional[str] = None) -> Tuple[bool, str, str]:
    """Run a command; return (success, stdout, stderr)."""
    cwd = cwd or str(PROJECT_ROOT)
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            cwd=cwd,
        )
        return r.returncode == 0, (r.stdout or ""), (r.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        return False, "", str(e)


def _parse_iperf3_json(stdout: str) -> Dict:
    """
    Parse iperf3 JSON output. Handles both TCP and UDP.
    Returns dict with: bandwidth_bps, jitter_ms, loss_fraction, (optional) latency_ms.
    """
    out = {}
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return out

    # TCP: end.sum_sent or end.sum_received
    end = data.get("end") or {}
    sum_sent = end.get("sum_sent") or {}
    sum_recv = end.get("sum_received") or end.get("sum") or {}
    # Prefer received for bandwidth (what server got)
    bps = sum_recv.get("bits_per_second") or sum_sent.get("bits_per_second")
    if bps is not None:
        out["bandwidth_bps"] = float(bps)

    # UDP: jitter_ms, lost_packets, packets, optional latency_ms
    jitter_ms = sum_recv.get("jitter_ms") or sum_sent.get("jitter_ms")
    if jitter_ms is not None:
        out["jitter_ms"] = float(jitter_ms)
    packets = sum_recv.get("packets") or sum_sent.get("packets")
    lost = sum_recv.get("lost_packets") or sum_sent.get("lost_packets")
    if packets is not None and packets > 0 and lost is not None:
        out["loss_fraction"] = float(lost) / float(packets)
    elif "lost_percent" in (sum_recv or sum_sent):
        pct = (sum_recv or sum_sent).get("lost_percent")
        if pct is not None:
            out["loss_fraction"] = float(pct) / 100.0

    # Some iperf3 UDP builds expose latency_ms per-stream; take max or first available
    try:
        streams = (end.get("streams") or []) if isinstance(end, dict) else []
        latencies = []
        for s in streams:
            udp_section = s.get("udp") or s.get("sum", {})
            if isinstance(udp_section, dict) and "latency_ms" in udp_section:
                latencies.append(float(udp_section.get("latency_ms")))
        if latencies:
            out["latency_ms"] = max(latencies)
    except Exception:
        # Best-effort; latency may not be available in all iperf3 versions
        pass

    return out


def run_iperf3_native(
    server_host: str,
    server_port: int = IPERF3_PORT,
    duration_sec: int = 5,
    use_udp: bool = True,
    output_json_path: Optional[Path] = None,
) -> Dict:
    """
    Run iperf3 from current process (client script / native client side).
    Client sends to server; measures bandwidth, and for UDP: jitter and loss.
    Returns parsed metrics dict and writes to output_json_path if given.
    """
    out_path = output_json_path or (PROJECT_ROOT / "shared_data" / "iperf3_network_params.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Run TCP first for bandwidth, then UDP for jitter/loss (or single UDP for all)
    result = {}
    if use_udp:
        cmd = [
            "iperf3", "-c", server_host, "-p", str(server_port),
            "-u", "-b", "10M", "-t", str(duration_sec), "-J"
        ]
        ok, stdout, stderr = _run_iperf3_cmd(cmd, timeout_sec=duration_sec + 15)
        if ok and stdout.strip():
            result = _parse_iperf3_json(stdout)
        # If UDP failed or no bandwidth, try TCP
        if not result.get("bandwidth_bps"):
            cmd_tcp = [
                "iperf3", "-c", server_host, "-p", str(server_port),
                "-t", str(min(3, duration_sec)), "-J"
            ]
            ok_tcp, stdout_tcp, _ = _run_iperf3_cmd(cmd_tcp, timeout_sec=20)
            if ok_tcp and stdout_tcp.strip():
                tcp_parsed = _parse_iperf3_json(stdout_tcp)
                result.setdefault("bandwidth_bps", tcp_parsed.get("bandwidth_bps"))
    else:
        cmd = [
            "iperf3", "-c", server_host, "-p", str(server_port),
            "-t", str(duration_sec), "-J"
        ]
        ok, stdout, stderr = _run_iperf3_cmd(cmd, timeout_sec=duration_sec + 15)
        if ok and stdout.strip():
            result = _parse_iperf3_json(stdout)

    result["source"] = "native"
    result["server_host"] = server_host
    result["server_port"] = server_port
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass
    return result


def run_iperf3_docker(
    client_container: str,
    server_host: str,
    server_port: int = IPERF3_PORT,
    duration_sec: int = 5,
    use_udp: bool = True,
    output_json_path: Optional[Path] = None,
) -> Dict:
    """
    Run iperf3 inside the client container (client → server).
    Ensures network parameters are measured from client egress (tc applied on client).
    """
    out_path = output_json_path or (PROJECT_ROOT / "shared_data" / "iperf3_network_params.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {}
    if use_udp:
        cmd = [
            "docker", "exec", client_container,
            "iperf3", "-c", server_host, "-p", str(server_port),
            "-u", "-b", "10M", "-t", str(duration_sec), "-J"
        ]
        ok, stdout, stderr = _run_iperf3_cmd(cmd, timeout_sec=duration_sec + 20)
        if ok and stdout.strip():
            result = _parse_iperf3_json(stdout)
    if not result.get("bandwidth_bps"):
        cmd_tcp = [
            "docker", "exec", client_container,
            "iperf3", "-c", server_host, "-p", str(server_port),
            "-t", str(min(3, duration_sec)), "-J"
        ]
        ok_tcp, stdout_tcp, _ = _run_iperf3_cmd(cmd_tcp, timeout_sec=20)
        if ok_tcp and stdout_tcp.strip():
            tcp_parsed = _parse_iperf3_json(stdout_tcp)
            result.setdefault("bandwidth_bps", tcp_parsed.get("bandwidth_bps"))

    result["source"] = "docker"
    result["client_container"] = client_container
    result["server_host"] = server_host
    result["server_port"] = server_port
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass
    return result


def load_network_params(json_path: Optional[Path] = None) -> Dict:
    """Load last iperf3 result from JSON. Returns dict with bandwidth_bps, jitter_ms, loss_fraction."""
    path = json_path or (PROJECT_ROOT / "shared_data" / "iperf3_network_params.json")
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def network_params_to_t_calc_input(params: Dict, scenario_fallback: Optional[Dict] = None) -> Tuple[float, float, float, float]:
    """
    Convert iperf3 params + optional scenario fallback to (B_bps, D_tc_sec, J_sec, p).
    Used by diagnostic pipeline and by T_calc computation.
    """
    B = float(params.get("bandwidth_bps") or 0)
    if B <= 0 or not math.isfinite(B):
        B = float("inf")
    jitter_ms = params.get("jitter_ms")
    J = (float(jitter_ms) / 1000.0) if jitter_ms is not None else 0.0
    p = float(params.get("loss_fraction") or 0.0)
    # Delay: prefer iperf3 latency_ms when available, else scenario fallback
    D_tc = 0.0
    latency_ms = params.get("latency_ms")
    if latency_ms is not None:
        try:
            D_tc = float(latency_ms) / 1000.0
        except (TypeError, ValueError):
            D_tc = 0.0
    if D_tc == 0.0 and scenario_fallback:
        lat = (scenario_fallback.get("latency") or "0").strip().lower()
        if lat:
            import re
            m = re.match(r"(\d+(?:\.\d+)?)\s*(\w*)", lat)
            if m:
                val, unit = float(m.group(1)), (m.group(2) or "").lower()
                D_tc = val / 1000.0 if "ms" in unit else val
    return B, D_tc, J, p


def compute_t_calc(
    protocol: str,
    S_bits: int,
    O_app: float,
    B_bps: float,
    D_tc_sec: float,
    J_sec: float,
    p: float,
) -> float:
    """
    Compute analytical transfer time T_calc (same formula as diagnostic_pipeline).
    protocol: mqtt, amqp, grpc, quic, http3, dds
    """
    RTT_eff = D_tc_sec + J_sec

    def transfer_time_bits_bps(S: float, B: float) -> float:
        if B is None or B <= 0 or not math.isfinite(B):
            return 0.0
        return S / B

    if B_bps <= 0 or not math.isfinite(B_bps):
        B_eff = float("inf")
    else:
        B_eff = B_bps
    transfer_s = transfer_time_bits_bps(float(S_bits), B_eff)

    proto_key = protocol.upper()

    # QUIC/HTTP3 TLS overhead (not scaled by α)
    O_TLS = 0.0
    if protocol in ("quic", "http3"):
        O_TLS = O_TLS_HANDSHAKE_S + (K_TLS_SEC_PER_BYTE * (S_bits / 8.0))

    # Protocol-specific network term without α scaling; α is applied uniformly below
    if protocol in ("mqtt", "amqp", "grpc"):
        if p > 0 and RTT_eff > 0 and math.isfinite(B_bps):
            B_eff = min(B_bps, MSS_BITS / (RTT_eff * math.sqrt(p)))
            transfer_s = transfer_time_bits_bps(float(S_bits), B_eff)
        if protocol == "mqtt":
            network_term = (2.0 * RTT_eff) + transfer_s
        else:
            network_term = (1.5 * RTT_eff) + transfer_s
    elif protocol == "quic":
        loss_term = p * (S_bits / MSS_BITS) * RTT_eff
        network_term = (1.0 * RTT_eff) + transfer_time_bits_bps(S_bits, B_bps) + loss_term
    elif protocol == "http3":
        loss_term = p * (S_bits / MSS_BITS) * RTT_eff
        network_term = (1.0 * RTT_eff) + transfer_time_bits_bps(S_bits, B_bps) + loss_term
    else:
        # dds or other
        loss_term = p * (S_bits / MSS_BITS) * (RTT_eff + T_REPAIR)
        network_term = transfer_s + loss_term

    alpha = ALPHA_PROTO.get(proto_key, 1.0)
    return alpha * network_term + O_app + O_TLS


def scenario_conditions_to_t_calc_input(conditions: Dict[str, str]) -> Tuple[float, float, float, float]:
    """
    Convert a scenario conditions dict (latency, jitter, bandwidth, loss as strings)
    to (B_bps, D_tc_sec, J_sec, p) for compute_t_calc.
    Same format as network_simulator.NETWORK_SCENARIOS (e.g. "20ms", "5ms", "50mbit", "0.1%").
    """
    def _bandwidth_bps(s: str) -> float:
        s = (s or "").strip().lower()
        if not s:
            return float("inf")
        m = re.match(r"(\d+(?:\.\d+)?)\s*(\w*)", s)
        if not m:
            return float("inf")
        val, unit = float(m.group(1)), (m.group(2) or "").lower()
        if "gbit" in unit or "gibit" in unit:
            return val * 1e9
        if "mbit" in unit or "mibit" in unit:
            return val * 1e6
        if "kbit" in unit or "kibit" in unit:
            return val * 1e3
        return val

    def _loss_decimal() -> float:
        s = (conditions.get("loss") or "0").strip().replace("%", "")
        try:
            return float(s) / 100.0
        except ValueError:
            return 0.0

    def _delay_sec(key: str) -> float:
        s = (conditions.get(key) or "0").strip().lower()
        m = re.match(r"(\d+(?:\.\d+)?)\s*(\w*)", s)
        if not m:
            return 0.0
        val, unit = float(m.group(1)), (m.group(2) or "").lower()
        return val / 1000.0 if "ms" in unit else val

    B = _bandwidth_bps(conditions.get("bandwidth") or "")
    if B <= 0 or not math.isfinite(B):
        B = float("inf")
    D_tc = _delay_sec("latency")
    J = _delay_sec("jitter")
    p = _loss_decimal()
    return B, D_tc, J, p


def get_t_calc_for_scenario(
    protocol: str,
    payload_bytes: int,
    scenario_name: str,
    O_app_estimate: float = 0.01,
) -> Optional[float]:
    """
    Compute T_calc for a named network scenario (e.g. good, moderate, poor) without
    using live iperf3 params. Used when training the RL agent in excellent conditions
    but rewarding as if the round ran in the target scenario (RL_REWARD_SCENARIO).
    """
    init_alpha_proto_from_disk()
    try:
        from network_simulator import NetworkSimulator
        conditions = NetworkSimulator.get_scenario_conditions(scenario_name)
    except Exception:
        return None
    B, D_tc, J, p = scenario_conditions_to_t_calc_input(conditions)
    S_bits = payload_bytes * 8
    return compute_t_calc(protocol, S_bits, O_app_estimate, B, D_tc, J, p)


def get_t_calc_for_reward(
    protocol: str,
    payload_bytes: int,
    json_path: Optional[Path] = None,
    O_app_estimate: float = 0.01,
) -> Optional[float]:
    """
    Load network params from iperf3 JSON and compute T_calc for the given protocol.
    Used by RL client to pass t_calc into calculate_reward (high T_calc -> more negative reward).
    Ensures that the latest calibrated α_proto values from alpha_proto.json are used.
    Returns None if params unavailable.
    """
    # Ensure α_proto is initialized from disk before computing T_calc so RL
    # rewards reflect the latest tuned analytical model.
    init_alpha_proto_from_disk()

    params = load_network_params(json_path)
    if not params:
        return None
    B, D_tc, J, p = network_params_to_t_calc_input(params, scenario_fallback=None)
    S_bits = payload_bytes * 8
    return compute_t_calc(protocol, S_bits, O_app_estimate, B, D_tc, J, p)


if __name__ == "__main__":
    # Ensure α_proto reflects the latest calibration if available on disk
    init_alpha_proto_from_disk()

    # Quick test: run native iperf3 if server given
    server = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    print(f"Running iperf3 (native) to {server}...")
    r = run_iperf3_native(server, duration_sec=3)
    print(json.dumps(r, indent=2))
