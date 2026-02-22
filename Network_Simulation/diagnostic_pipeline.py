#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Pipeline for Single-Protocol FL: Empirical Overhead, Network Extraction, Analytical Model.
Runs separately from normal single-protocol implementation.
Use: Select one protocol + use case (and optional baseline) from GUI, then "Run Diagnostic Pipeline".
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Constants
MSS_BITS = 11680
T_REPAIR = 0.005  # seconds
B_BRIDGE_BPS = 10_000_000_000  # 10 Gbps

# Protocol -> (sender containers list, receiver container) per use_case – both clients for FL rounds
SERVICE_PATTERNS = {
    "emotion": {
        "mqtt": (["fl-client-mqtt-emotion-1", "fl-client-mqtt-emotion-2"], "fl-server-mqtt-emotion"),
        "amqp": (["fl-client-amqp-emotion-1", "fl-client-amqp-emotion-2"], "fl-server-amqp-emotion"),
        "grpc": (["fl-client-grpc-emotion-1", "fl-client-grpc-emotion-2"], "fl-server-grpc-emotion"),
        "quic": (["fl-client-quic-emotion-1", "fl-client-quic-emotion-2"], "fl-server-quic-emotion"),
        "http3": (["fl-client-http3-emotion-1", "fl-client-http3-emotion-2"], "fl-server-http3-emotion"),
        "dds": (["fl-client-dds-emotion-1", "fl-client-dds-emotion-2"], "fl-server-dds-emotion"),
    },
    "mentalstate": {
        "mqtt": (["fl-client-mqtt-mental-1", "fl-client-mqtt-mental-2"], "fl-server-mqtt-mental"),
        "amqp": (["fl-client-amqp-mental-1", "fl-client-amqp-mental-2"], "fl-server-amqp-mental"),
        "grpc": (["fl-client-grpc-mental-1", "fl-client-grpc-mental-2"], "fl-server-grpc-mental"),
        "quic": (["fl-client-quic-mental-1", "fl-client-quic-mental-2"], "fl-server-quic-mental"),
        "http3": (["fl-client-http3-mental-1", "fl-client-http3-mental-2"], "fl-server-http3-mental"),
        "dds": (["fl-client-dds-mental-1", "fl-client-dds-mental-2"], "fl-server-dds-mental"),
    },
    "temperature": {
        "mqtt": (["fl-client-mqtt-temp-1", "fl-client-mqtt-temp-2"], "fl-server-mqtt-temp"),
        "amqp": (["fl-client-amqp-temp-1", "fl-client-amqp-temp-2"], "fl-server-amqp-temp"),
        "grpc": (["fl-client-grpc-temp-1", "fl-client-grpc-temp-2"], "fl-server-grpc-temp"),
        "quic": (["fl-client-quic-temp-1", "fl-client-quic-temp-2"], "fl-server-quic-temp"),
        "http3": (["fl-client-http3-temp-1", "fl-client-http3-temp-2"], "fl-server-http3-temp"),
        "dds": (["fl-client-dds-temp-1", "fl-client-dds-temp-2"], "fl-server-dds-temp"),
    },
}


def run_cmd(cmd, check=True, env=None, cwd=None):
    env = env or os.environ.copy()
    cwd = str(cwd or PROJECT_ROOT)
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=check, env=env, cwd=cwd
    )


def extract_tc_params(sender_container: str) -> dict:
    """Phase 2: Run docker exec <container> tc qdisc show dev eth0 and extract B, p, D_tc, J.
    B must be in bits per second. If no bandwidth limit is found (e.g. unconstrained Docker bridge),
    B is strictly float('inf') so that S/B = 0. Never return B=0 or B=1 (would explode S/B).
    """
    result = run_cmd(
        ["docker", "exec", sender_container, "tc", "qdisc", "show", "dev", "eth0"],
        check=False,
    )
    out = (result.stdout or "") + (result.stderr or "")
    B_candidates = []  # collect all rates; use max so main link rate wins over burst/small rates
    p = 0.0
    D_tc = 0.0
    J = 0.0

    # tc output: rate 100Mbit -> 100e6 bps, rate 1000mbit -> 1e9 bps (regex is case-insensitive)
    # netem: delay 50.0ms 10.0ms loss 1%
    # tbf: rate 20mbit burst 32kbit -> we only match "rate <num><unit>"; take max so link rate wins
    for line in out.splitlines():
        m = re.search(r"rate\s+(\d+(?:\.\d+)?)\s*(\w*)", line, re.I)
        if m:
            val = float(m.group(1))
            unit = (m.group(2) or "").strip().lower()
            # 100Mbit / 100mbit -> 100 * 1e6 = 100000000 bps
            if "gbit" in unit or "gibit" in unit:
                bps = val * 1e9
            elif "mbit" in unit or "mibit" in unit:
                bps = val * 1e6
            elif "kbit" in unit or "kibit" in unit:
                bps = val * 1e3
            elif "bit" in unit or not unit:
                bps = val  # raw bps if "bit" or no unit
            else:
                bps = val
            B_candidates.append(bps)
            continue
        # delay 500.0ms 10.0ms or delay 500000us 10000us -> D_tc and J in seconds
        # tc output e.g.: "delay 500.0ms 10.0ms loss 1% rate 100Mbit"
        m = re.search(r"delay\s+([\d.]+)\s*(\w*)", line, re.I)
        if m:
            raw_val, raw_unit = (m.group(1) or "").strip(), (m.group(2) or "").strip().lower()
            if raw_val:
                D_val = float(raw_val)
                # Convert to seconds: us/usec -> *0.000001, ms -> *0.001, s -> *1
                if "usec" in raw_unit or ("us" in raw_unit and "ms" not in raw_unit):
                    D_tc = D_val * 0.000001
                elif "ms" in raw_unit:
                    D_tc = D_val * 0.001
                elif "s" in raw_unit and "ms" not in raw_unit and "us" not in raw_unit:
                    D_tc = D_val
                elif not raw_unit and D_val > 1000:
                    D_tc = D_val * 0.000001  # no unit + large value: assume microseconds
                else:
                    D_tc = D_val * 0.001  # assume ms if unknown
            # Jitter: second number on same line (e.g. delay 500.0ms 10.0ms)
            m2 = re.search(r"delay\s+[\d.]+\s*\S+\s+([\d.]+)\s*(\w*)", line, re.I)
            if m2:
                j_raw, j_unit = (m2.group(1) or "").strip(), (m2.group(2) or "").strip().lower()
                if j_raw:
                    J_val = float(j_raw)
                    if "usec" in j_unit or ("us" in j_unit and "ms" not in j_unit):
                        J = J_val * 0.000001
                    elif "ms" in j_unit:
                        J = J_val * 0.001
                    elif "s" in j_unit and "ms" not in j_unit and "us" not in j_unit:
                        J = J_val
                    elif not j_unit and J_val > 1000:
                        J = J_val * 0.000001
                    else:
                        J = J_val * 0.001
            continue
        m = re.search(r"loss\s+(\d+(?:\.\d+)?)\s*%?", line, re.I)
        if m:
            p = float(m.group(1)) / 100.0
            continue

    # No rate found (e.g. no tc applied) -> B must be inf so S/B = 0. Never 0 or 1.
    B_bps = max(B_candidates) if B_candidates else float("inf")
    if B_bps <= 0 or not math.isfinite(B_bps):
        B_bps = float("inf")
    return {"B": B_bps, "p": p, "D_tc": D_tc, "J": J}


def _scenario_bandwidth_bps(conditions: dict) -> float:
    """Convert scenario bandwidth string (e.g. '20mbit') to bps."""
    s = (conditions.get("bandwidth") or "").strip().lower()
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


def _scenario_loss_decimal(conditions: dict) -> float:
    """Convert scenario loss string (e.g. '1%') to decimal (0.01)."""
    s = (conditions.get("loss") or "0").strip().replace("%", "")
    try:
        return float(s) / 100.0
    except ValueError:
        return 0.0


def _scenario_delay_jitter_sec(conditions: dict) -> tuple:
    """Convert scenario latency and jitter (e.g. '50ms', '10ms') to seconds. Returns (D_tc, J)."""
    def to_sec(x):
        s = (x or "0").strip().lower()
        m = re.match(r"(\d+(?:\.\d+)?)\s*(\w*)", s)
        if not m:
            return 0.0
        val, unit = float(m.group(1)), (m.group(2) or "").lower()
        if "ms" in unit:
            return val / 1000.0
        return val
    D_tc = to_sec(conditions.get("latency"))
    J = to_sec(conditions.get("jitter"))
    return (D_tc, J)


def parse_fl_diag_from_logs(sender_container: str, receiver_container: str, round_index: int) -> dict:
    """Parse FL_DIAG lines from docker logs. round_index 0 = first round (calibration), 1 = second (lossy)."""
    client_log = run_cmd(["docker", "logs", sender_container], check=False)
    server_log = run_cmd(["docker", "logs", receiver_container], check=False)
    client_out = (client_log.stdout or "") + (client_log.stderr or "")
    server_out = (server_log.stdout or "") + (server_log.stderr or "")

    def parse_line(prefix, text):
        vals = {}
        for line in text.splitlines():
            if "FL_DIAG" not in line:
                continue
            # FL_DIAG O_send=0.001 payload_bytes=12345 send_start_ts=...
            m = re.search(r"O_send=([\d.]+)", line)
            if m:
                vals["O_send"] = float(m.group(1))
            m = re.search(r"payload_bytes=(\d+)", line)
            if m:
                vals["payload_bytes"] = int(m.group(1))
            m = re.search(r"send_start_ts=([\d.]+)", line)
            if m:
                vals["send_start_ts"] = float(m.group(1))
        return vals

    def parse_server_line(text):
        vals = {}
        for line in text.splitlines():
            if "FL_DIAG" not in line:
                continue
            m = re.search(r"O_recv=([\d.]+)", line)
            if m:
                vals["O_recv"] = float(m.group(1))
            m = re.search(r"recv_end_ts=([\d.]+)", line)
            if m:
                vals["recv_end_ts"] = float(m.group(1))
            m = re.search(r"send_start_ts=([\d.]+)", line)
            if m:
                vals["send_start_ts"] = float(m.group(1))
        return vals

    client_vals_list = []
    for line in client_out.splitlines():
        if "FL_DIAG" in line and "O_send=" in line:
            v = {}
            m = re.search(r"O_send=([\d.]+)", line)
            if m:
                v["O_send"] = float(m.group(1))
            m = re.search(r"payload_bytes=(\d+)", line)
            if m:
                v["payload_bytes"] = int(m.group(1))
            m = re.search(r"send_start_ts=([\d.]+)", line)
            if m:
                v["send_start_ts"] = float(m.group(1))
            if v:
                client_vals_list.append(v)

    server_vals_list = []
    for line in server_out.splitlines():
        if "FL_DIAG" in line and "O_recv=" in line:
            v = {}
            m = re.search(r"O_recv=([\d.]+)", line)
            if m:
                v["O_recv"] = float(m.group(1))
            m = re.search(r"recv_end_ts=([\d.]+)", line)
            if m:
                v["recv_end_ts"] = float(m.group(1))
            m = re.search(r"send_start_ts=([\d.]+)", line)
            if m:
                v["send_start_ts"] = float(m.group(1))
            if v:
                server_vals_list.append(v)

    idx = min(round_index, len(client_vals_list) - 1, len(server_vals_list) - 1)
    if idx < 0:
        return {}
    c = client_vals_list[idx]
    s = server_vals_list[idx]
    payload_bytes = c.get("payload_bytes", 0)
    O_send = c.get("O_send", 0.0)
    O_recv = s.get("O_recv", 0.0)
    send_start_ts = c.get("send_start_ts") or s.get("send_start_ts")
    recv_end_ts = s.get("recv_end_ts")
    T_total = (recv_end_ts - send_start_ts) if (recv_end_ts and send_start_ts) else 0.0
    return {
        "O_send": O_send,
        "O_recv": O_recv,
        "T_total": T_total,
        "payload_bytes": payload_bytes,
        "S_bits": payload_bytes * 8,
    }


def parse_fl_diag_from_logs_multi(
    sender_containers: list,
    receiver_container: str,
    round_index: int,
) -> list:
    """Parse FL_DIAG for all clients; return list of dicts one per client (client_id 1, 2, ...)."""
    # Server logs: FL_DIAG client_id=N O_recv=... recv_end_ts=... send_start_ts=...
    server_log = run_cmd(["docker", "logs", receiver_container], check=False)
    server_out = (server_log.stdout or "") + (server_log.stderr or "")
    server_lines_per_client = {}  # client_id -> list of {O_recv, recv_end_ts, send_start_ts}
    for line in server_out.splitlines():
        if "FL_DIAG" not in line or "O_recv=" not in line:
            continue
        m = re.search(r"client_id=(\d+)", line)
        client_id = int(m.group(1)) if m else None
        if client_id is None:
            continue
        v = {}
        for pat, key in [(r"O_recv=([\d.]+)", "O_recv"), (r"recv_end_ts=([\d.]+)", "recv_end_ts"), (r"send_start_ts=([\d.]+)", "send_start_ts")]:
            m = re.search(pat, line)
            if m:
                v[key] = float(m.group(1))
        if v:
            server_lines_per_client.setdefault(client_id, []).append(v)
    # Client logs: one container per client; order = sender_containers order (client 1, client 2)
    results = []
    for idx, sender_container in enumerate(sender_containers):
        client_id = idx + 1  # 1-based
        client_log = run_cmd(["docker", "logs", sender_container], check=False)
        client_out = (client_log.stdout or "") + (client_log.stderr or "")
        client_vals_list = []
        for line in client_out.splitlines():
            if "FL_DIAG" in line and "O_send=" in line:
                v = {}
                m = re.search(r"O_send=([\d.]+)", line)
                if m:
                    v["O_send"] = float(m.group(1))
                m = re.search(r"payload_bytes=(\d+)", line)
                if m:
                    v["payload_bytes"] = int(m.group(1))
                m = re.search(r"send_start_ts=([\d.]+)", line)
                if m:
                    v["send_start_ts"] = float(m.group(1))
                if v:
                    client_vals_list.append(v)
        s_list = server_lines_per_client.get(client_id, [])
        r_idx = min(round_index, len(client_vals_list) - 1, len(s_list) - 1)
        if r_idx < 0:
            results.append({"client_id": client_id, "O_send": 0.0, "O_recv": 0.0, "T_total": 0.0, "payload_bytes": 0, "S_bits": int(0)})
            continue
        c = client_vals_list[r_idx]
        s = s_list[r_idx]
        payload_bytes = c.get("payload_bytes", 0)
        O_send = c.get("O_send", 0.0)
        O_recv = s.get("O_recv", 0.0)
        send_start_ts = c.get("send_start_ts") or s.get("send_start_ts")
        recv_end_ts = s.get("recv_end_ts")
        T_total = (recv_end_ts - send_start_ts) if (recv_end_ts and send_start_ts) else 0.0
        results.append({
            "client_id": client_id,
            "O_send": O_send,
            "O_recv": O_recv,
            "T_total": T_total,
            "payload_bytes": int(payload_bytes),
            "S_bits": int(payload_bytes) * 8,  # bits; ensure integer
        })
    return results


def run_pipeline(
    protocol: str,
    use_case: str,
    sender_container: str,
    receiver_container: str,
    scenario: str = "excellent",
    enable_gpu: bool = False,
    payload_file: str = None,
    compose_file: str = None,
    services: list = None,
) -> dict:
    """Execute full diagnostic pipeline and return summary dict.
    Phase 1: Calibration with NO channel losses (protocol/broker overhead only).
    Phases 2-4: Use the user-selected network scenario from the GUI.
    When enable_gpu is True, use GPU compose if available; otherwise fall back to CPU (same as single-protocol).
    """
    use_case = use_case.lower()
    protocol = protocol.lower()
    scenario = (scenario or "excellent").lower()
    if protocol not in ("mqtt", "amqp", "grpc", "quic", "http3", "dds"):
        raise ValueError("Protocol must be one of: mqtt, amqp, grpc, quic, http3, dds")

    # Import here to avoid circular deps and to use same scenarios as GUI/network_simulator
    sys.path.insert(0, str(SCRIPT_DIR))
    from network_simulator import NetworkSimulator
    sim = NetworkSimulator(verbose=True)
    if scenario not in sim.NETWORK_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from: {list(sim.NETWORK_SCENARIOS.keys())}")

    docker_dir = PROJECT_ROOT / "Docker"
    if compose_file is None:
        if enable_gpu:
            compose_file = str(docker_dir / f"docker-compose-{use_case}.gpu-isolated.yml")
        else:
            compose_file = str(docker_dir / f"docker-compose-{use_case}.yml")
    if services is None:
        pat = SERVICE_PATTERNS.get(use_case, {}).get(protocol)
        if pat:
            sender_containers = list(pat[0])  # both clients
            receiver_container = pat[1]
        else:
            sender_containers = [sender_container] if isinstance(sender_container, str) else list(sender_container or [])
        services = [receiver_container] + sender_containers
        if protocol in ("mqtt", "amqp"):
            # GPU compose uses "amqp-broker" for AMQP; CPU compose uses "rabbitmq" (or -mental/-temp)
            if protocol == "mqtt":
                broker = "mqtt-broker" if use_case == "emotion" else ("mqtt-broker-mental" if use_case == "mentalstate" else "mqtt-broker-temp")
            else:
                if enable_gpu:
                    broker = "amqp-broker"  # GPU compose service name for all use cases
                else:
                    broker = "rabbitmq" if use_case == "emotion" else ("rabbitmq-mental" if use_case == "mentalstate" else "rabbitmq-temp")
            services = [broker, receiver_container] + sender_containers

    env = os.environ.copy()
    env["NUM_ROUNDS"] = "2"
    env["FL_DIAGNOSTIC_PIPELINE"] = "1"

    # Phase 1: Calibration – NO channel losses. Protocol and broker overhead only (perfect/unshaped channel).
    print("[Phase 1] Calibration: measuring protocol & broker overhead with NO channel losses (clean channel)...")
    print("          Starting both clients for FL rounds (GPU={})...".format("yes" if enable_gpu else "CPU"))
    result = run_cmd(
        ["docker", "compose", "-f", compose_file, "up", "-d"] + list(services),
        env=env,
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0 and enable_gpu:
        print("          GPU compose failed (nvidia-docker or GPU not available). Falling back to CPU...")
        run_cmd(["docker", "compose", "-f", compose_file, "down"], check=False, cwd=PROJECT_ROOT)
        compose_file = str(docker_dir / f"docker-compose-{use_case}.yml")
        # CPU compose uses different broker service names (e.g. rabbitmq vs amqp-broker)
        if protocol in ("mqtt", "amqp"):
            broker_cpu = "mqtt-broker" if protocol == "mqtt" else "rabbitmq"
            if use_case == "mentalstate":
                broker_cpu = "mqtt-broker-mental" if protocol == "mqtt" else "rabbitmq-mental"
            elif use_case == "temperature":
                broker_cpu = "mqtt-broker-temp" if protocol == "mqtt" else "rabbitmq-temp"
            services = [broker_cpu, receiver_container] + sender_containers
        result = run_cmd(
            ["docker", "compose", "-f", compose_file, "up", "-d"] + list(services),
            env=env,
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Failed to start containers after GPU fallback. Check docker compose and network.")
    elif result.returncode != 0:
        raise RuntimeError("Failed to start containers. Check docker compose and network.")
    time.sleep(15)
    # Ensure all senders have no tc so calibration round sees zero loss/delay
    for sc in sender_containers:
        try:
            sim.reset_container_network(sc)
        except Exception:
            pass
    print("          Sender tc cleared for calibration (no losses).")

    # Wait for first round to complete (calibration – no losses); need both clients' timings
    n_expected = len(sender_containers)
    for _ in range(120):
        time.sleep(2)
        cal_list = parse_fl_diag_from_logs_multi(sender_containers, receiver_container, 0)
        n_ready = sum(1 for c in (cal_list or []) if c.get("T_total", 0) > 0)
        if cal_list and len(cal_list) >= n_expected and n_ready >= n_expected:
            break
    else:
        print("[WARNING] No calibration timing for all clients in logs; using available or defaults.")
        cal_list = [
            {"client_id": 1, "O_send": 0.0, "O_recv": 0.0, "T_total": 0.0, "payload_bytes": 0, "S_bits": 0},
            {"client_id": 2, "O_send": 0.0, "O_recv": 0.0, "T_total": 0.0, "payload_bytes": 0, "S_bits": 0},
        ]

    # O_broker once (from first client) for broker-based protocols
    cal0 = cal_list[0] if cal_list else {}
    S_bits = int(cal0.get("S_bits", 0))  # bits (payload_bytes * 8 from FL_DIAG)
    if payload_file and os.path.isfile(payload_file):
        S_bits = int(8 * os.path.getsize(payload_file))
    if protocol in ("grpc", "quic", "http3", "dds"):
        O_broker = 0.0
    else:
        B_bridge = B_BRIDGE_BPS
        T_total_baseline0 = cal0.get("T_total", 0)
        O_send0 = cal0.get("O_send", 0)
        O_recv0 = cal0.get("O_recv", 0)
        O_broker = T_total_baseline0 - (O_send0 + O_recv0 + (S_bits / B_bridge))
        O_broker = max(0.0, O_broker)

    # Phase 2: Apply user-selected network scenario to both sender containers
    print(f"[Phase 2] Applying user-selected network scenario: {scenario}...")
    conditions = sim.NETWORK_SCENARIOS[scenario]
    for sc in sender_containers:
        try:
            sim.apply_network_conditions(sc, conditions)
        except Exception as e:
            print(f"          WARNING: Could not apply scenario to {sc}: {e}")
    print(f"          Applied to both clients: {conditions.get('name', scenario)}")
    print("[Phase 2] Extracting tc parameters from sender container...")
    tc = extract_tc_params(sender_containers[0])
    B, p, D_tc, J = tc["B"], tc["p"], tc["D_tc"], tc["J"]
    # No bandwidth in tc (unconstrained Docker bridge) -> use scenario B so S/B is well-defined
    if B == float("inf"):
        B = _scenario_bandwidth_bps(conditions)
        if p == 0 and D_tc == 0 and J == 0:
            p = _scenario_loss_decimal(conditions)
            D_tc, J = _scenario_delay_jitter_sec(conditions)
        print(f"          No tc rate found; using scenario-derived B={B} bps, p={p}, D_tc={D_tc}, J={J}")
    elif B != float("inf") and (B < 1e6 or B <= 0) and conditions.get("bandwidth"):
        # Extracted B too small or invalid (e.g. burst 32kbit); use scenario bandwidth
        B = _scenario_bandwidth_bps(conditions)
        print(f"          Using scenario bandwidth (extracted B was {tc['B']:.0f} bps): B={B}")
    RTT_eff = D_tc + J
    time.sleep(2)

    # Phase 3: Lossy round – get T_actual per client (cap 90*2s so we don't block long after FL finished)
    print("[Phase 3] Waiting for lossy round (round 2) under selected scenario...")
    lossy_list = []
    phase3_max_wait = 90
    for iteration in range(phase3_max_wait):
        time.sleep(2)
        lossy_list = parse_fl_diag_from_logs_multi(sender_containers, receiver_container, 1)
        n_ready = sum(1 for c in (lossy_list or []) if c.get("T_total", 0) > 0)
        if lossy_list and len(lossy_list) >= n_expected and n_ready >= n_expected:
            break
        if iteration == phase3_max_wait - 1:
            print(f"          Proceeding after {phase3_max_wait * 2}s (training may have finished; using available data).")
    if not lossy_list or len(lossy_list) < len(cal_list):
        lossy_list = [{"client_id": i + 1, "T_total": cal_list[i].get("T_total", 0)} if i < len(cal_list) else {"client_id": i + 1, "T_total": 0} for i in range(max(2, len(cal_list)))]

    # Safe transfer time: S_bits/B in seconds; S/inf = 0, avoid division by zero
    def transfer_time_bits_bps(S_bits_val, B_val):
        if B_val is None or B_val <= 0 or not math.isfinite(B_val):
            return 0.0
        return float(S_bits_val) / float(B_val)

    # Per-client summaries
    summaries = []
    for i, cal in enumerate(cal_list):
        cid = cal.get("client_id", i + 1)
        O_send = cal.get("O_send", 0.0)
        O_recv = cal.get("O_recv", 0.0)
        O_app = O_send + O_recv + O_broker
        # S in bits: from FL_DIAG payload_bytes * 8 (ensure int)
        S_bits_c = int(cal.get("S_bits", S_bits))
        lossy_c = next((x for x in lossy_list if x.get("client_id") == cid), lossy_list[i] if i < len(lossy_list) else {})
        T_actual = lossy_c.get("T_total", 0.0)
        if protocol in ("mqtt", "amqp", "grpc"):
            B_eff = B if p == 0 else min(B, MSS_BITS / (RTT_eff * math.sqrt(p))) if RTT_eff and p else B
        else:
            B_eff = B
        # DEBUG: values used in T_calc (user-requested)
        print(f"DEBUG -> S: {S_bits_c} bits | B: {B} bps | RTT_eff: {RTT_eff} s | Loss (p): {p} | O_app: {O_app} s")
        transfer_s = transfer_time_bits_bps(S_bits_c, B_eff)
        if protocol in ("mqtt", "amqp", "grpc"):
            T_calc = (1.5 * RTT_eff) + transfer_s + O_app
        elif protocol in ("quic", "http3"):
            # HTTP/3 is QUIC-based; same analytical formula
            T_calc = (1 * RTT_eff) + transfer_time_bits_bps(S_bits_c, B) + (p * (S_bits_c / MSS_BITS) * RTT_eff) + O_app
        else:
            T_calc = transfer_time_bits_bps(S_bits_c, B) + (p * (S_bits_c / MSS_BITS) * (RTT_eff + T_REPAIR)) + O_app
        error_pct = 100.0 * (T_actual - T_calc) / T_actual if T_actual else 0.0
        summaries.append({
            "client_id": cid,
            "protocol": protocol.upper(),
            "scenario": scenario,
            "O_app": O_app,
            "O_broker": O_broker,
            "p": p,
            "T_actual": T_actual,
            "T_calc": T_calc,
            "error_pct": error_pct,
        })

    run_cmd(["docker", "compose", "-f", compose_file, "down"], check=False, cwd=PROJECT_ROOT)
    return summaries


def print_table(summaries: list):
    """Phase 4: Print formatted terminal table (one row per client), including protocol and network scenario."""
    print()
    sep = "+--------+----------+-------------+----------------+----------------+----------+----------------+----------------+----------+"
    print(sep)
    print("| Client | Protocol | Scenario    | O_app (s)      | O_broker       | Loss (p) | T_actual (s)    | T_calc (s)     | Error %  |")
    print(sep)
    for r in summaries:
        scen = (r.get("scenario", "") or "")[:12]
        print("| {:^6} | {:^8} | {:^11} | {:>14.6f} | {:>14.6f} | {:>8.4f} | {:>14.4f} | {:>14.4f} | {:>7} |".format(
            r.get("client_id", 0), r["protocol"], scen,
            r["O_app"], r["O_broker"], r["p"],
            r["T_actual"], r["T_calc"], f"{r['error_pct']:.2f}%"))
    print(sep)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic pipeline: Empirical overhead, network extraction, analytical model (single protocol).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--protocol", "-p", required=True, choices=["mqtt", "amqp", "grpc", "quic", "http3", "dds"])
    parser.add_argument("--use-case", "-u", default="emotion", choices=["emotion", "mentalstate", "temperature"])
    parser.add_argument(
        "--scenario", "-s",
        default="excellent",
        choices=["excellent", "good", "moderate", "poor", "very_poor", "satellite",
                 "congested_light", "congested_moderate", "congested_heavy"],
        help="Network scenario from GUI for Phases 2–4 (Phase 1 always uses no losses)",
    )
    parser.add_argument("--enable-gpu", "-g", action="store_true",
                        help="Use GPU for clients if available (same as single-protocol); fall back to CPU if GPU unavailable")
    parser.add_argument("--sender-container", help="Override sender (client) container name")
    parser.add_argument("--receiver-container", help="Override receiver (server) container name")
    parser.add_argument("--payload-file", help="Path to payload file for S (bits); else from calibration run")
    args = parser.parse_args()

    pat = SERVICE_PATTERNS.get(args.use_case, {}).get(args.protocol)
    sender = args.sender_container or (pat[0] if pat else None)
    receiver = args.receiver_container or (pat[1] if pat else None)
    if not sender or not receiver:
        print("Missing sender/receiver container. Use --sender-container and --receiver-container or --use-case.", file=sys.stderr)
        sys.exit(1)

    summary = run_pipeline(
        protocol=args.protocol,
        use_case=args.use_case,
        sender_container=sender,
        receiver_container=receiver,
        scenario=args.scenario,
        enable_gpu=args.enable_gpu,
        payload_file=args.payload_file,
    )
    print_table(summary)
    # Emit JSON line for GUI to display in Diagnostic Results tab (flush so it is not buffered)
    print("FL_DIAG_TABLE_JSON|" + json.dumps(summary), flush=True)
    # Also write to file so GUI can load reliably when experiment completes (fallback if stdout is lost)
    out_dir = PROJECT_ROOT / "shared_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "diagnostic_results_latest.json"
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write {out_file}: {e}", flush=True)


if __name__ == "__main__":
    main()
