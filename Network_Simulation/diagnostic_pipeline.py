#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Pipeline for FL: Empirical Overhead, Network Extraction, Analytical Model.
Runs separately from normal single-protocol implementation.
Supports single or multiple protocols and network scenarios:
  - Single: --protocol X --scenario Y (one experiment).
  - Multiple: --protocols A B --scenarios X Y (runs all combinations A+X, A+Y, B+X, B+Y).
Use: Select one or more protocols and scenarios from GUI, then "Run Diagnostic Pipeline".
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

import numpy as np
import pandas as pd

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Paths for diagnostics and α_proto persistence
DIAGNOSTIC_PATH = PROJECT_ROOT / "shared_data" / "Diagnostic_Pipeline_Untuned.xlsx"
ALPHA_PATH = PROJECT_ROOT / "shared_data" / "alpha_proto.json"

# Use iperf3 for network params (client-side) when set; else use tc extraction
USE_IPERF3 = os.environ.get("USE_IPERF3", "1").strip().lower() in ("1", "true", "yes")
IPERF3_DURATION = int(os.environ.get("IPERF3_DURATION", "5"))

# Constants
MSS_BITS = 11680
T_REPAIR = 0.005  # seconds
B_BRIDGE_BPS = 10_000_000_000  # 10 Gbps

# Target maximum absolute percentage error for tuned T_calc; used for
# iterative α tuning per (protocol, scenario) during the diagnostic pipeline.
ALPHA_TUNE_ERROR_THRESHOLD = float(os.environ.get("ALPHA_TUNE_ERROR_THRESHOLD", "5.0"))
ALPHA_TUNE_MAX_ITERS = max(1, int(os.environ.get("ALPHA_TUNE_MAX_ITERS", "5")))

# Protocol-specific scaling factors α_proto (fitted from diagnostics; used to scale network term)
ALPHA_PROTO = {
    "MQTT": 1.8,
    "AMQP": 2.1,
    "GRPC": 1.7,
    "QUIC": 1.0,
    "HTTP3": 4.5,
    "DDS": 8.2,
}


def load_diagnostics(path: Path = DIAGNOSTIC_PATH) -> pd.DataFrame:
    """Load diagnostics from Excel/CSV into a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def fit_alpha_proto(protocol: str, df: pd.DataFrame) -> float:
    """
    Fit α_proto so that T_calc_tuned = α * network_term + O_app ≈ T_actual.
    Hence α = (T_actual - O_app) / network_term, with network_term = T_calc_raw - O_app
    (for protocols without TLS; for QUIC/HTTP3, T_calc_raw already includes O_TLS in the raw term).
    """
    proto = protocol.upper()
    if "T_calc_raw" not in df.columns or "O_app" not in df.columns:
        return float(ALPHA_PROTO.get(proto, 1.0))
    df_proto = df[df["protocol"].str.upper() == proto].copy()
    if df_proto.empty:
        return float(ALPHA_PROTO.get(proto, 1.0))
    network_term = df_proto["T_calc_raw"] - df_proto["O_app"]
    mask = network_term > 1e-9  # avoid div by zero
    if not mask.any():
        return float(ALPHA_PROTO.get(proto, 1.0))
    # α such that α * network_term + O_app = T_actual  =>  α = (T_actual - O_app) / network_term
    ratios = (df_proto.loc[mask, "T_actual"] - df_proto.loc[mask, "O_app"]) / network_term[mask]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if len(ratios) == 0:
        return float(ALPHA_PROTO.get(proto, 1.0))
    return float(np.median(ratios))


def calibrate_alpha_from_diagnostics(df: pd.DataFrame) -> dict:
    """Fit α for all protocols present in diagnostics DataFrame."""
    alphas: dict[str, float] = {}
    if "protocol" not in df.columns:
        return alphas
    for proto in df["protocol"].dropna().astype(str).str.upper().unique():
        alphas[proto] = fit_alpha_proto(proto, df)
    return alphas


def load_alpha_proto(path: Path = ALPHA_PATH) -> dict:
    """Load α_proto from JSON."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    # Normalize keys to uppercase protocol names
    return {str(k).upper(): float(v) for k, v in data.items()}


def save_alpha_proto(alphas: dict, path: Path = ALPHA_PATH, merge: bool = True) -> dict:
    """
    Save α_proto to JSON.

    If merge is True and a file already exists, merge new values into the existing
    dict so protocols not present in this run keep their previous α.
    """
    path = Path(path)
    existing = load_alpha_proto(path) if merge and path.exists() else {}
    new_vals = {str(k).upper(): float(v) for k, v in (alphas or {}).items()}
    merged = {**existing, **new_vals}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return merged


def init_alpha_proto():
    """
    Load α_proto from last run, or fit from available diagnostics.

    This should be called once at the beginning of the diagnostic pipeline.
    """
    global ALPHA_PROTO

    alphas_loaded = load_alpha_proto(ALPHA_PATH)
    if alphas_loaded:
        print("🔄 Using α_proto from previous run:")
        for proto, alpha in sorted(alphas_loaded.items()):
            print(f"  {proto}: α = {alpha:.3f}")
        ALPHA_PROTO.update(alphas_loaded)
        return

    print("📊 No alpha_proto.json found, fitting from historical diagnostics (if available)...")
    try:
        df_diag = load_diagnostics(DIAGNOSTIC_PATH)
    except FileNotFoundError:
        print(f"  No diagnostics file found at {DIAGNOSTIC_PATH}. Using default α_proto values.")
        return

    alphas = calibrate_alpha_from_diagnostics(df_diag)
    if not alphas:
        print("  Diagnostics did not contain usable data for α_proto tuning. Using defaults.")
        return
    merged = save_alpha_proto(alphas, ALPHA_PATH, merge=False)
    ALPHA_PROTO.update(merged)
    print("  Fitted α_proto from existing diagnostics and saved to alpha_proto.json.")


def save_results_to_excel(df_results: pd.DataFrame, path: Path = DIAGNOSTIC_PATH) -> None:
    """Save diagnostic results DataFrame to Excel (and ensure directory exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_excel(path, index=False)

# TLS 1.3 (QUIC/HTTP3) encryption overhead – CPU-bound
# One-time handshake crypto (key exchange + cert verify); typical 1–5 ms on modern CPU
O_TLS_HANDSHAKE_S = float(os.environ.get("O_TLS_HANDSHAKE_S", "0.003"))
# Per-byte crypto time for encrypt+decrypt (AES-GCM); typical ~10–50 ns/byte → 1e-8–5e-8 s/byte
K_TLS_SEC_PER_BYTE = float(os.environ.get("K_TLS_SEC_PER_BYTE", "2e-8"))

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


# Pattern to match FL / Framework processes that may hold GPU memory (e.g. path contains Framework_FL)
GPU_CLEANUP_PATTERN = "Framework_FL"
if os.environ.get("FL_GPU_CLEANUP_PATTERN", "").strip():
    GPU_CLEANUP_PATTERN = os.environ.get("FL_GPU_CLEANUP_PATTERN", "").strip()


def kill_gpu_processes_matching_pattern(pattern: str = None, verbose: bool = True) -> None:
    """
    After each experiment: run nvidia-smi, find processes using GPU whose command line
    contains the given pattern (e.g. Framework_FL), and sudo kill them to free GPU memory.
    """
    pattern = (pattern or GPU_CLEANUP_PATTERN).strip()
    if not pattern:
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(PROJECT_ROOT),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        if verbose:
            print("[GPU cleanup] nvidia-smi not available or failed; skipping.", flush=True)
        return
    if result.returncode != 0 or not (result.stdout or "").strip():
        return
    pids = []
    for line in (result.stdout or "").strip().splitlines():
        line = line.strip().strip('"')
        if line.isdigit():
            pids.append(int(line))
    my_pid = os.getpid()
    to_kill = []
    for pid in pids:
        if pid == my_pid:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "r", encoding="utf-8", errors="replace") as f:
                cmdline = f.read().replace("\x00", " ")
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if pattern.lower() in cmdline.lower():
            to_kill.append(pid)
    if not to_kill:
        return
    sudo_pw = os.environ.get("FL_SUDO_PASSWORD", "").strip()
    for pid in to_kill:
        try:
            if sudo_pw:
                subprocess.run(
                    ["sudo", "-S", "kill", str(pid)],
                    input=(sudo_pw + "\n"),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(PROJECT_ROOT),
                )
            else:
                subprocess.run(
                    ["sudo", "kill", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=str(PROJECT_ROOT),
                )
            if verbose:
                print(f"[GPU cleanup] Killed PID {pid} (matched *{pattern}*)", flush=True)
        except Exception as e:
            if verbose:
                print(f"[GPU cleanup] Failed to kill PID {pid}: {e}", flush=True)


def _parse_rate_from_line(line: str):
    """Parse 'rate 100Mbit' from a line; return bps or None."""
    m = re.search(r"rate\s+(\d+(?:\.\d+)?)\s*(\w*)", line, re.I)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "").strip().lower()
    if "gbit" in unit or "gibit" in unit:
        return val * 1e9
    if "mbit" in unit or "mibit" in unit:
        return val * 1e6
    if "kbit" in unit or "kibit" in unit:
        return val * 1e3
    if "bit" in unit or not unit:
        return val
    return val


def extract_tc_params(sender_container: str) -> dict:
    """Phase 2: Run tc qdisc/class show on sender (client) container and extract B, p, D_tc, J. Tc is clients-only."""
    result = run_cmd(
        ["docker", "exec", sender_container, "tc", "qdisc", "show", "dev", "eth0"],
        check=False,
    )
    out = (result.stdout or "") + (result.stderr or "")
    if "htb" in out:
        class_result = run_cmd(
            ["docker", "exec", sender_container, "tc", "class", "show", "dev", "eth0"],
            check=False,
        )
        out += "\n" + (class_result.stdout or "") + (class_result.stderr or "")
    B_candidates = []
    p = 0.0
    D_tc = 0.0
    J = 0.0

    # tc output: rate 100Mbit -> 100e6 bps, rate 1000mbit -> 1e9 bps (regex is case-insensitive)
    # netem: delay 50.0ms 10.0ms loss 1%
    # tbf: rate 20mbit burst 32kbit -> we only match "rate <num><unit>"; take max so link rate wins
    for line in out.splitlines():
        bps = _parse_rate_from_line(line)
        if bps is not None:
            B_candidates.append(bps)
        # delay 10.0ms 2.0ms loss 0.1% – parse delay, jitter, and loss on same line
        m = re.search(r"delay\s+([\d.]+)\s*(\w*)", line, re.I)
        if m:
            raw_val, raw_unit = (m.group(1) or "").strip(), (m.group(2) or "").strip().lower()
            if raw_val:
                D_val = float(raw_val)
                if "usec" in raw_unit or ("us" in raw_unit and "ms" not in raw_unit):
                    D_tc = D_val * 0.000001
                elif "ms" in raw_unit:
                    D_tc = D_val * 0.001
                elif "s" in raw_unit and "ms" not in raw_unit and "us" not in raw_unit:
                    D_tc = D_val
                elif not raw_unit and D_val > 1000:
                    D_tc = D_val * 0.000001
                else:
                    D_tc = D_val * 0.001
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
            loss_m = re.search(r"loss\s+(\d+(?:\.\d+)?)\s*%?", line, re.I)
            if loss_m:
                p = float(loss_m.group(1)) / 100.0
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
    """Parse FL_DIAG lines from docker logs. round_index 0 = Round 1 (calibration), 2 = Round 2 (T_actual with tc)."""
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


def _parse_fl_diag_multi_from_strings(server_out: str, client_outs: list, round_index: int) -> list:
    """Parse FL_DIAG from pre-fetched log strings. client_outs = [client_1_text, client_2_text, ...]. Returns list of dicts per client."""
    server_lines_per_client = {}
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
    results = []
    for idx, client_out in enumerate(client_outs):
        client_id = idx + 1
        client_vals_list = []
        for line in (client_out or "").splitlines():
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
            "S_bits": int(payload_bytes) * 8,
        })
    return results


def parse_fl_diag_from_logs_multi(
    sender_containers: list,
    receiver_container: str,
    round_index: int,
) -> list:
    """Parse FL_DIAG for all clients from Docker logs; return list of dicts one per client (client_id 1, 2, ...)."""
    server_log = run_cmd(["docker", "logs", receiver_container], check=False)
    server_out = (server_log.stdout or "") + (server_log.stderr or "")
    client_outs = []
    for sender_container in sender_containers:
        client_log = run_cmd(["docker", "logs", sender_container], check=False)
        client_outs.append((client_log.stdout or "") + (client_log.stderr or ""))
    return _parse_fl_diag_multi_from_strings(server_out, client_outs, round_index)


def parse_fl_diag_from_logs_native(log_dir: Path, round_index: int, num_clients: int = 2) -> list:
    """Parse FL_DIAG from native run log files (server.log, client_1.log, client_2.log, ...). Same return shape as parse_fl_diag_from_logs_multi."""
    log_dir = Path(log_dir)
    server_path = log_dir / "server.log"
    server_out = server_path.read_text(encoding="utf-8", errors="replace") if server_path.exists() else ""
    client_outs = []
    for i in range(1, num_clients + 1):
        p = log_dir / f"client_{i}.log"
        client_outs.append(p.read_text(encoding="utf-8", errors="replace") if p.exists() else "")
    return _parse_fl_diag_multi_from_strings(server_out, client_outs, round_index)


def count_docker_fl_diag_rounds(sender_containers: list) -> list:
    """Return number of FL_DIAG (O_send=) lines per client from Docker logs. [n1, n2, ...]."""
    counts = []
    for sender_container in sender_containers:
        client_log = run_cmd(["docker", "logs", sender_container], check=False)
        text = (client_log.stdout or "") + (client_log.stderr or "")
        n = sum(1 for line in text.splitlines() if "FL_DIAG" in line and "O_send=" in line)
        counts.append(n)
    return counts


def count_native_fl_diag_rounds(log_dir: Path, num_clients: int = 2) -> list:
    """Return number of FL_DIAG (O_send=) lines per client, i.e. how many rounds have sent an update. [n1, n2, ...]."""
    log_dir = Path(log_dir)
    counts = []
    for i in range(1, num_clients + 1):
        p = log_dir / f"client_{i}.log"
        text = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        n = sum(1 for line in text.splitlines() if "FL_DIAG" in line and "O_send=" in line)
        counts.append(n)
    return counts


def run_pipeline(
    protocol: str,
    use_case: str,
    sender_container: str,
    receiver_container: str,
    scenario: str = "excellent",
    enable_gpu: bool = False,
    network_mode: str = "gpu",
    payload_file: str = None,
    compose_file: str = None,
    services: list = None,
    num_clients: int = 1,
) -> dict:
    """Execute full diagnostic pipeline and return summary dict.

    Flow:
      Phase 1 – Round 1 (round index 0): No tc on clients. Server→Global model→Client; Client→Local model→Server.
        Used for application overhead (O_send, O_recv, O_broker).
      Phase 2 – Apply tc at client egress only (server never shaped).
      Phase 3 – Round 2 (round index 2): Tc on client egress. T_actual = recv_end_ts − send_start_ts for round index 2.

    NUM_ROUNDS=3 so we have round indices 0, 1, 2. We use index 0 for cal and index 2 for T_actual.
    network_mode: gpu (Docker bridge), host, host_macvlan.
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

    network_mode = (network_mode or "gpu").lower().strip()
    if network_mode not in ("gpu", "host", "host_macvlan"):
        network_mode = "gpu"
    docker_dir = PROJECT_ROOT / "Docker"
    if compose_file is None:
        if network_mode == "host":
            compose_file = str(docker_dir / f"docker-compose-{use_case}.host-network.yml")
            if not os.path.isfile(compose_file):
                raise FileNotFoundError(f"Host-network compose not found: {compose_file}")
        elif network_mode == "host_macvlan":
            compose_file = str(docker_dir / f"docker-compose-{use_case}.macvlan.yml")
            if not os.path.isfile(compose_file):
                raise FileNotFoundError(f"Macvlan compose not found: {compose_file}")
            if not sim.ensure_macvlan_network():
                raise RuntimeError("Failed to create macvlan network for host_macvlan mode")
        elif enable_gpu:
            compose_file = str(docker_dir / f"docker-compose-{use_case}.gpu-isolated.yml")
        else:
            compose_file = str(docker_dir / f"docker-compose-{use_case}.yml")
    if services is None:
        pat = SERVICE_PATTERNS.get(use_case, {}).get(protocol)
        if pat:
            all_senders = list(pat[0])
            if num_clients is None or num_clients <= 0:
                num_to_use = len(all_senders)
            else:
                num_to_use = min(num_clients, len(all_senders))
            sender_containers = all_senders[:num_to_use]
            receiver_container = pat[1]
        else:
            sender_containers = [sender_container] if isinstance(sender_container, str) else list(sender_container or [])
        services = [receiver_container] + sender_containers
        if protocol in ("mqtt", "amqp"):
            # GPU compose uses "amqp-broker" for AMQP; CPU compose uses "rabbitmq" (or -mental/-temp)
            if protocol == "mqtt":
                broker = "mqtt-broker" if use_case == "emotion" else ("mqtt-broker-mental" if use_case == "mentalstate" else "mqtt-broker-temp")
            else:
                if enable_gpu or network_mode in ("host", "host_macvlan"):
                    broker = "amqp-broker"
                else:
                    broker = "rabbitmq" if use_case == "emotion" else ("rabbitmq-mental" if use_case == "mentalstate" else "rabbitmq-temp")
            services = [broker, receiver_container] + sender_containers

    n_expected = len(sender_containers)

    env = os.environ.copy()
    # 3 rounds: Round 1 (index 0) = calibration, Round 2 (index 2) = T_actual with tc on client egress
    env["NUM_ROUNDS"] = "3"
    env["FL_DIAGNOSTIC_PIPELINE"] = "1"
    env["MIN_CLIENTS"] = str(n_expected)
    env["MAX_CLIENTS"] = str(n_expected)

    # Phase 1: Round 1 – Application overhead (no tc on client egress).
    # Server → Global model → Client; Client → Local model → Server. Use round index 0 for O_send, O_recv, O_broker.
    print("[Phase 1] Round 1 – Application overhead (no tc on clients)...")
    net_label = {"gpu": "bridge", "host": "host (tc on host)", "host_macvlan": "macvlan (per-container tc)"}.get(network_mode, "bridge")
    print("          Starting {} client(s) for FL rounds (GPU={}, network={})...".format(
        n_expected, "yes" if enable_gpu else "CPU", net_label))
    compose_cmd = ["docker", "compose", "-f", compose_file, "up", "-d"] + list(services)
    result = run_cmd(
        compose_cmd,
        env=env,
        cwd=PROJECT_ROOT,
        check=False,
    )
    if result.returncode != 0 and enable_gpu and network_mode == "gpu":
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
    # Tc is applied to CLIENTS ONLY (never to server). Ensure server has no tc so all shaping is on client egress.
    try:
        sim.reset_container_network(receiver_container)
    except Exception:
        pass
    # Ensure all senders (clients) have no tc so calibration round sees zero loss/delay
    for sc in sender_containers:
        try:
            sim.reset_container_network(sc)
        except Exception:
            pass
    print("          Server tc cleared (tc applied to clients only). Client tc cleared for calibration (no losses).")

    # Wait for Round 1 (round index 0) to complete – calibration for application overhead.
    # For DDS specifically, also ensure that the server has finished aggregating round 1
    # before any tc shaping is applied (round 1 must be fully complete end‑to‑end).
    for _ in range(120):
        time.sleep(2)
        cal_list = parse_fl_diag_from_logs_multi(sender_containers, receiver_container, 0)  # round index 0 = Round 1
        n_ready = sum(1 for c in (cal_list or []) if c.get("T_total", 0) > 0)
        if cal_list and len(cal_list) >= n_expected and n_ready >= n_expected:
            if protocol == "dds":
                # Extra guard for DDS: wait until the DDS server log shows that
                # round 1 has been fully aggregated before we move on to tc.
                try:
                    srv_log = run_cmd(["docker", "logs", receiver_container], check=False)
                    srv_text = (srv_log.stdout or "") + (srv_log.stderr or "")
                except Exception:
                    srv_text = ""
                if (
                    "Round 1 Aggregated Metrics" not in srv_text
                    and "Aggregated global model from round 1" not in srv_text
                ):
                    # Calibration timings are present but round 1 is still in progress
                    # on the DDS server – keep waiting so that tc is not applied mid‑round.
                    continue
            break
    else:
        print("[WARNING] No calibration timing for all clients in logs; using available or defaults.")
        cal_list = [
            {
                "client_id": i + 1,
                "O_send": 0.0,
                "O_recv": 0.0,
                "T_total": 0.0,
                "payload_bytes": 0,
                "S_bits": 0,
            }
            for i in range(max(n_expected, 1))
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

    # Phase 2: Apply tc at egress of clients only. Then Round 2 runs: Server → Global model → Client; Client → Local model → Server.
    # T_actual = recv_end_ts − send_start_ts for round index 2 (second round, client send to server receive).
    print(f"[Phase 2] Applying tc at client egress only: {scenario}. Round 2 will run with tc...")
    try:
        conditions = sim.get_scenario_conditions(scenario)
    except (KeyError, ValueError) as e:
        print(f"[ERROR] Cannot apply scenario: {e}")
        conditions = {}
    host_mode = network_mode == "host"  # only apply tc on host when using host-network compose
    if host_mode:
        print(f"          Containers use network_mode: host; applying scenario to host interface.")
        if sim.apply_network_conditions_host(conditions):
            print(f"          Host tc applied; T_actual will reflect scenario (delay/loss/bandwidth).")
        else:
            print(f"          WARNING: Could not apply host tc (sudo may be required). T_actual will be unshaped.")
        # In host mode we use scenario parameters for T_calc (no per-container tc to read)
        B = _scenario_bandwidth_bps(conditions)
        p = _scenario_loss_decimal(conditions)
        D_tc, J = _scenario_delay_jitter_sec(conditions)
        print(f"          Using scenario-derived B={B} bps, p={p}, D_tc={D_tc}, J={J} (host mode).")
    else:
        for sc in sender_containers:
            try:
                sim.apply_network_conditions(sc, conditions)
            except Exception as e:
                print(f"          WARNING: Could not apply scenario to {sc}: {e}")
        print(f"          Applied to both clients: {conditions.get('name', scenario)}")
        B, p, D_tc, J = None, None, None, None
        params = {}
        if USE_IPERF3:
            # Collect network parameters using iperf3 from client container (client egress = tc applied)
            sys.path.insert(0, str(SCRIPT_DIR))
            try:
                from communication_model import (
                    run_iperf3_docker,
                    load_network_params,
                    network_params_to_t_calc_input,
                )
                iperf3_out = PROJECT_ROOT / "shared_data" / "iperf3_network_params.json"
                iperf3_out.parent.mkdir(parents=True, exist_ok=True)
                # Start iperf3 server on receiver so client can measure to it
                run_cmd(
                    ["docker", "exec", "-d", receiver_container, "iperf3", "-s"],
                    check=False,
                )
                time.sleep(2)
                print("[Phase 2] Running iperf3 from client container to get network params...")
                run_iperf3_docker(
                    sender_containers[0],
                    server_host=receiver_container,
                    server_port=5201,
                    duration_sec=min(IPERF3_DURATION, 10),
                    use_udp=True,
                    output_json_path=iperf3_out,
                )
                run_cmd(
                    ["docker", "exec", receiver_container, "pkill", "-x", "iperf3"],
                    check=False,
                )
                params = load_network_params(iperf3_out)
                if params:
                    B, D_tc, J, p = network_params_to_t_calc_input(params, scenario_fallback=conditions)
                    print(f"          iperf3: B={B} bps, p={p}, D_tc={D_tc}, J={J} (D from scenario)")
            except Exception as e:
                print(f"          iperf3 failed ({e}); falling back to tc.")
        if B is None or p is None:
            print("[Phase 2] Extracting tc parameters from sender container...")
            tc = extract_tc_params(sender_containers[0])
            B, p, D_tc, J = tc["B"], tc["p"], tc["D_tc"], tc["J"]
        # No bandwidth (unconstrained Docker bridge) -> use scenario B so S/B is well-defined
        if B == float("inf"):
            B = _scenario_bandwidth_bps(conditions)
            if p == 0 and D_tc == 0 and J == 0:
                p = _scenario_loss_decimal(conditions)
                D_tc, J = _scenario_delay_jitter_sec(conditions)
            if not (USE_IPERF3 and params):
                print(f"          No tc rate found; using scenario-derived B={B} bps, p={p}, D_tc={D_tc}, J={J}")
        elif B != float("inf") and (B < 1e6 or B <= 0) and conditions.get("bandwidth"):
            B = _scenario_bandwidth_bps(conditions)
            print(f"          Using scenario bandwidth: B={B}")
        if p == 0 and D_tc == 0 and J == 0 and (conditions.get("loss") or conditions.get("latency")):
            p = _scenario_loss_decimal(conditions)
            D_tc, J = _scenario_delay_jitter_sec(conditions)
            if not (USE_IPERF3 and params):
                print(f"          Tc had no loss/delay; using scenario p={p}, D_tc={D_tc}, J={J} for analytical model.")
    RTT_eff = D_tc + J
    time.sleep(2)

    # Phase 3: Wait for Round 2 to fully complete: client finishes training → sends update → server receives.
    # T_actual = recv_end_ts − send_start_ts (client send to server receive for round index 2). Do not tear down until we have this.
    # FL_DIAG_PHASE3_WAIT=0 or negative = no timeout (wait indefinitely). Positive = max iterations (each 2s).
    try:
        phase3_max_wait = int(os.environ.get("FL_DIAG_PHASE3_WAIT", "600"))  # 600*2s = 20 min default; 0 = no limit
    except (ValueError, TypeError):
        phase3_max_wait = 600
    no_timeout = phase3_max_wait <= 0
    if no_timeout:
        phase3_max_wait = None
        print("[Phase 3] Waiting for Round 2 to complete (no timeout; set FL_DIAG_PHASE3_WAIT>0 for a limit).")
    else:
        phase3_max_wait = max(phase3_max_wait, 90)
        print(f"[Phase 3] Waiting for Round 2 to complete. Will wait up to {phase3_max_wait * 2}s (FL_DIAG_PHASE3_WAIT=0 for no limit)...")
    lossy_list = []
    iteration = 0
    while no_timeout or iteration < phase3_max_wait:
        time.sleep(2)
        lossy_list = parse_fl_diag_from_logs_multi(sender_containers, receiver_container, 2)  # round index 2 = third round
        round_counts = count_docker_fl_diag_rounds(sender_containers)
        n_ready = sum(1 for c in (lossy_list or []) if c.get("T_total", 0) > 0)
        has_round2_data = round_counts and min(round_counts) >= 3
        if lossy_list and len(lossy_list) >= n_expected and n_ready >= n_expected and has_round2_data:
            print(f"          Round 2 complete for all {n_expected} clients (train→send→receive) after {(iteration + 1) * 2}s.")
            break
        if not no_timeout and iteration == phase3_max_wait - 1:
            print(f"          Timeout after {phase3_max_wait * 2}s. Set FL_DIAG_PHASE3_WAIT=0 for no timeout, or higher (e.g. 900) for 30 min.")
        iteration += 1
    if not lossy_list or len(lossy_list) < len(cal_list):
        count = max(n_expected, len(cal_list))
        lossy_list = [
            {
                "client_id": i + 1,
                "T_total": cal_list[i].get("T_total", 0) if i < len(cal_list) else 0,
            }
            for i in range(count)
        ]
    # Require round index 2 for T_actual; otherwise parser used an earlier round
    docker_round_counts = count_docker_fl_diag_rounds(sender_containers)
    if docker_round_counts and any(c < 3 for c in docker_round_counts):
        print(
            "[WARNING] Not all clients have 3 FL_DIAG rounds (have {}). T_actual must use round index 2; clients with < 3 will show T_actual=0.".format(
                docker_round_counts
            )
        )

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
        lossy_c = next(
            (x for x in lossy_list if x.get("client_id") == cid),
            lossy_list[i] if i < len(lossy_list) else {},
        )
        T_actual = lossy_c.get("T_total", 0.0)
        if i < len(docker_round_counts) and docker_round_counts[i] < 3:
            T_actual = 0.0
        if protocol in ("mqtt", "amqp", "grpc"):
            B_eff = B if p == 0 else min(B, MSS_BITS / (RTT_eff * math.sqrt(p))) if RTT_eff and p else B
        else:
            B_eff = B
        # TLS 1.3 (QUIC/HTTP3 only): handshake + per-byte encrypt/decrypt CPU time
        O_TLS = 0.0
        if protocol in ("quic", "http3"):
            O_TLS = O_TLS_HANDSHAKE_S + (K_TLS_SEC_PER_BYTE * (S_bits_c / 8.0))
        # DEBUG: values used in T_calc (user-requested)
        debug_extra = f" | O_TLS: {O_TLS:.6f} s" if protocol in ("quic", "http3") else ""
        print(
            f"DEBUG -> S: {S_bits_c} bits | B: {B} bps | RTT_eff: {RTT_eff} s | "
            f"Loss (p): {p} | O_app: {O_app} s{debug_extra}"
        )
        transfer_s = transfer_time_bits_bps(S_bits_c, B_eff)
        proto_key = protocol.upper()

        # Protocol-specific network term (without α scaling; α is applied uniformly below)
        if protocol in ("mqtt", "amqp", "grpc"):
            if protocol == "mqtt":
                network_term = (2.0 * RTT_eff) + transfer_s
            else:
                network_term = (1.5 * RTT_eff) + transfer_s
        elif protocol == "quic":
            loss_term = p * (S_bits_c / MSS_BITS) * RTT_eff
            network_term = (1.0 * RTT_eff) + transfer_time_bits_bps(S_bits_c, B) + loss_term
        elif protocol == "http3":
            loss_term = p * (S_bits_c / MSS_BITS) * RTT_eff
            network_term = (1.0 * RTT_eff) + transfer_time_bits_bps(S_bits_c, B) + loss_term
        else:
            # DDS-style: chunking/repair term
            loss_term = p * (S_bits_c / MSS_BITS) * (RTT_eff + T_REPAIR)
            network_term = transfer_time_bits_bps(S_bits_c, B) + loss_term

        alpha = ALPHA_PROTO.get(proto_key, 1.0)
        T_calc_raw = network_term + O_app + O_TLS
        T_calc_tuned = alpha * network_term + O_app + O_TLS
        error_pct_raw = 100.0 * abs(T_actual - T_calc_raw) / T_actual if T_actual else 0.0
        error_pct_tuned = 100.0 * abs(T_actual - T_calc_tuned) / T_actual if T_actual else 0.0

        summaries.append(
            {
                "client_id": cid,
                "protocol": protocol.upper(),
                "scenario": scenario,
                "O_app": O_app,
                "O_broker": O_broker,
                "p": p,
                "T_actual": T_actual,
                "T_calc_raw": T_calc_raw,
                # Alpha used for this protocol in this run
                "alpha_proto": alpha,
                "alpha": alpha,
                "T_calc_tuned": T_calc_tuned,
                # Backwards-compatible keys:
                "T_calc": T_calc_tuned,
                "error_pct_raw": error_pct_raw,
                "error_pct_tuned": error_pct_tuned,
                "error_pct": error_pct_tuned,
            }
        )

    # If we had applied tc on the host (legacy host mode), clear it. With macvlan we use per-container tc only.
    if host_mode:
        try:
            sim.reset_host_network()
        except Exception as e:
            print(f"          WARNING: Could not reset host tc: {e}")
    run_cmd(["docker", "compose", "-f", compose_file, "down"], check=False, cwd=PROJECT_ROOT)
    return summaries


def run_pipeline_native(
    protocol: str,
    use_case: str,
    scenario: str = "excellent",
    enable_gpu: bool = False,
    num_clients: int = 1,
    payload_file: str = None,
) -> list:
    """Execute diagnostic pipeline using native namespaces (no Docker).

    Single run with 2 rounds:
      Phase 1 (round 1): No tc on clients → O_app from round index 0.
      Phase 2 (round 2): Tc applied before round 2; client receive then send (send affected by tc) → T_actual from round index 1.
      Phase 3: O_app and T_actual calculated from logs.
    """
    use_case = use_case.lower()
    protocol = protocol.lower()
    scenario = (scenario or "excellent").lower()
    if protocol not in ("mqtt", "amqp", "grpc", "quic", "http3", "dds"):
        raise ValueError("Protocol must be one of: mqtt, amqp, grpc, quic, http3, dds")

    sys.path.insert(0, str(SCRIPT_DIR))
    from network_simulator import NetworkSimulator
    from run_native_experiments import NativeExperimentRunner

    sim = NetworkSimulator(verbose=True)
    if scenario not in sim.NETWORK_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Choose from: {list(sim.NETWORK_SCENARIOS.keys())}")

    log_dir = SCRIPT_DIR / "native_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Native] Logs: server={log_dir / 'server.log'}  clients={[str(log_dir / f'client_{i}.log') for i in range(1, num_clients + 1)]}")
    orig_env_num_rounds = os.environ.get("NUM_ROUNDS")
    orig_env_diag = os.environ.get("FL_DIAGNOSTIC_PIPELINE")
    os.environ["NUM_ROUNDS"] = "2"
    os.environ["FL_DIAGNOSTIC_PIPELINE"] = "1"

    n_expected = num_clients

    try:
        # Single run: round 1 without tc (O_app), then tc applied before round 2 (T_actual from round 2 send).
        print("[Phase 1] Native: Round 1 – no tc on clients (for O_app)...")
        print("[Phase 2] Native: Before round 2, tc will be applied at client egress; round 2 send → T_actual.")
        conditions = sim.NETWORK_SCENARIOS[scenario]
        B = _scenario_bandwidth_bps(conditions)
        p = _scenario_loss_decimal(conditions)
        D_tc, J = _scenario_delay_jitter_sec(conditions)
        RTT_eff = D_tc + J
        print(f"          Scenario for round 2: B={B} bps, p={p}, D_tc={D_tc}, J={J}")

        runner = NativeExperimentRunner(
            use_case=use_case,
            protocol=protocol,
            scenario=scenario,
            num_rounds=2,
            num_clients=num_clients,
            enable_gpu=enable_gpu,
            apply_tc_after_round_1=scenario,
        )
        code = runner.run()
        if code != 0:
            print(f"[WARNING] Run exited with code {code}; parsing available logs.")

        # Native: load iperf3 params from client (written during run when FL_DIAGNOSTIC_PIPELINE=1)
        if USE_IPERF3:
            try:
                from communication_model import load_network_params, network_params_to_t_calc_input
                iperf3_path = PROJECT_ROOT / "shared_data" / "iperf3_network_params.json"
                params = load_network_params(iperf3_path)
                if params:
                    B, D_tc, J, p = network_params_to_t_calc_input(params, scenario_fallback=conditions)
                    RTT_eff = D_tc + J
                    print(f"          Native: using iperf3 params from client: B={B} bps, p={p}, D_tc={D_tc}, J={J}")
            except Exception as e:
                print(f"          Native: iperf3 params not available ({e}); using scenario.")

        # Phase 3: O_app from round index 0 (round 1, no tc), T_actual from round index 1 (round 2, with tc).
        cal_list = parse_fl_diag_from_logs_native(log_dir, 0, num_clients)
        if not cal_list or len(cal_list) < n_expected:
            cal_list = [
                {
                    "client_id": i + 1,
                    "O_send": 0.0,
                    "O_recv": 0.0,
                    "T_total": 0.0,
                    "payload_bytes": 0,
                    "S_bits": 0,
                }
                for i in range(max(n_expected, 1))
            ]
        n_ready = sum(1 for c in cal_list if c.get("T_total", 0) > 0)
        if n_ready < n_expected:
            print("[WARNING] No round-1 timing for all clients; using available or defaults.")

        cal0 = cal_list[0] if cal_list else {}
        S_bits = int(cal0.get("S_bits", 0))
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

        lossy_list = parse_fl_diag_from_logs_native(log_dir, 1, num_clients)
        if not lossy_list or len(lossy_list) < len(cal_list):
            count = max(n_expected, len(cal_list))
            lossy_list = [
                {
                    "client_id": i + 1,
                    "T_total": cal_list[i].get("T_total", 0) if i < len(cal_list) else 0,
                }
                for i in range(count)
            ]
        round_counts = count_native_fl_diag_rounds(log_dir, num_clients)
        if any(c < 2 for c in round_counts):
            print(
                "[WARNING] Not all clients completed 2 rounds (have {} FL_DIAG O_send= lines). "
                "T_actual uses round index 1 (round 2 with tc); clients with < 2 rounds will show T_actual=0.".format(
                    round_counts
                )
            )

        def transfer_time_bits_bps(S_bits_val, B_val):
            if B_val is None or B_val <= 0 or not math.isfinite(B_val):
                return 0.0
            return float(S_bits_val) / float(B_val)

        summaries = []
        for i, cal in enumerate(cal_list):
            cid = cal.get("client_id", i + 1)
            O_send = cal.get("O_send", 0.0)
            O_recv = cal.get("O_recv", 0.0)
            O_app = O_send + O_recv + O_broker
            S_bits_c = int(cal.get("S_bits", S_bits))
            lossy_c = next(
                (x for x in lossy_list if x.get("client_id") == cid),
                lossy_list[i] if i < len(lossy_list) else {},
            )
            T_actual = lossy_c.get("T_total", 0.0)
            # Require round index 1 for T_actual (round 2: client send affected by tc)
            if i < len(round_counts) and round_counts[i] < 2:
                T_actual = 0.0
            if protocol in ("mqtt", "amqp", "grpc"):
                B_eff = B if p == 0 else min(B, MSS_BITS / (RTT_eff * math.sqrt(p))) if RTT_eff and p else B
            else:
                B_eff = B
            O_TLS = 0.0
            if protocol in ("quic", "http3"):
                O_TLS = O_TLS_HANDSHAKE_S + (K_TLS_SEC_PER_BYTE * (S_bits_c / 8.0))
            debug_extra = f" | O_TLS: {O_TLS:.6f} s" if protocol in ("quic", "http3") else ""
            print(
                f"DEBUG -> S: {S_bits_c} bits | B: {B} bps | RTT_eff: {RTT_eff} s | "
                f"Loss (p): {p} | O_app: {O_app} s{debug_extra}"
            )
            transfer_s = transfer_time_bits_bps(S_bits_c, B_eff)
            proto_key = protocol.upper()

            # Protocol-specific network term (without α scaling; α is applied uniformly below)
            if protocol in ("mqtt", "amqp", "grpc"):
                if protocol == "mqtt":
                    network_term = (2.0 * RTT_eff) + transfer_s
                else:
                    network_term = (1.5 * RTT_eff) + transfer_s
            elif protocol == "quic":
                loss_term = p * (S_bits_c / MSS_BITS) * RTT_eff
                network_term = (1.0 * RTT_eff) + transfer_time_bits_bps(S_bits_c, B) + loss_term
            elif protocol == "http3":
                loss_term = p * (S_bits_c / MSS_BITS) * RTT_eff
                network_term = (1.0 * RTT_eff) + transfer_time_bits_bps(S_bits_c, B) + loss_term
            else:
                loss_term = p * (S_bits_c / MSS_BITS) * (RTT_eff + T_REPAIR)
                network_term = transfer_time_bits_bps(S_bits_c, B) + loss_term

            alpha = ALPHA_PROTO.get(proto_key, 1.0)
            T_calc_raw = network_term + O_app + O_TLS
            T_calc_tuned = alpha * network_term + O_app + O_TLS
            error_pct_raw = 100.0 * abs(T_actual - T_calc_raw) / T_actual if T_actual else 0.0
            error_pct_tuned = 100.0 * abs(T_actual - T_calc_tuned) / T_actual if T_actual else 0.0

            summaries.append(
                {
                    "client_id": cid,
                    "protocol": protocol.upper(),
                    "scenario": scenario,
                    "O_app": O_app,
                    "O_broker": O_broker,
                    "p": p,
                    "T_actual": T_actual,
                    "T_calc_raw": T_calc_raw,
                    # Alpha used for this protocol in this run
                    "alpha_proto": alpha,
                    "alpha": alpha,
                    "T_calc_tuned": T_calc_tuned,
                    # Backwards-compatible keys:
                    "T_calc": T_calc_tuned,
                    "error_pct_raw": error_pct_raw,
                    "error_pct_tuned": error_pct_tuned,
                    "error_pct": error_pct_tuned,
                }
            )
        return summaries
    finally:
        if orig_env_num_rounds is not None:
            os.environ["NUM_ROUNDS"] = orig_env_num_rounds
        elif "NUM_ROUNDS" in os.environ:
            del os.environ["NUM_ROUNDS"]
        if orig_env_diag is not None:
            os.environ["FL_DIAGNOSTIC_PIPELINE"] = orig_env_diag
        elif "FL_DIAGNOSTIC_PIPELINE" in os.environ:
            del os.environ["FL_DIAGNOSTIC_PIPELINE"]


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


def _all_scenario_choices():
    """Return list of scenario names for validation (must match network_simulator.NETWORK_SCENARIOS)."""
    return ["excellent", "good", "moderate", "poor", "very_poor", "satellite",
            "congested_light", "congested_moderate", "congested_heavy"]


def main():
    # Load or initialize α_proto BEFORE running any diagnostics so all T_calc
    # computations for this run use the latest calibrated values.
    init_alpha_proto()

    parser = argparse.ArgumentParser(
        description="Diagnostic pipeline: Empirical overhead, network extraction, analytical model. Supports multiple protocols and scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--protocol", "-p", choices=["mqtt", "amqp", "grpc", "quic", "http3", "dds"],
                        help="Single protocol (ignored if --protocols is used)")
    parser.add_argument("--protocols", nargs="*", default=None,
                        help="Multiple protocols (e.g. --protocols mqtt amqp quic). Runs pipeline for each protocol × scenario.")
    parser.add_argument("--use-case", "-u", default="emotion", choices=["emotion", "mentalstate", "temperature"])
    parser.add_argument(
        "--scenario", "-s",
        default="excellent",
        choices=_all_scenario_choices(),
        help="Single network scenario for Phases 2–4 (ignored if --scenarios is used)",
    )
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Multiple scenarios (e.g. --scenarios excellent poor). Runs pipeline for each protocol × scenario.")
    parser.add_argument("--enable-gpu", "-g", action="store_true",
                        help="Use GPU for clients. Servers use CPU only. Tune TF_GPU_MEMORY_LIMIT_MB (default 4000) and BATCH_SIZE (default 16) if OOM.")
    parser.add_argument("--network-mode", choices=["gpu", "host", "host_macvlan"], default="gpu",
                        help="gpu=Docker bridge (per-container tc), host=host network (tc on host), host_macvlan=macvlan (per-container tc)")
    parser.add_argument("--native", action="store_true",
                        help="Run pipeline in native mode (namespaces, no Docker). Uses run_native_experiments with 2 rounds for calibration and lossy.")
    parser.add_argument("--sender-container", help="Override sender (client) container name")
    parser.add_argument("--receiver-container", help="Override receiver (server) container name")
    parser.add_argument("--payload-file", help="Path to payload file for S (bits); else from calibration run")
    parser.add_argument(
        "--num-clients",
        type=int,
        default=1,
        help="Number of FL clients to include in the diagnostic pipeline (default: 1).",
    )
    args = parser.parse_args()

    # Resolve protocols list: --protocols takes precedence, else single --protocol
    if args.protocols is not None and len(args.protocols) > 0:
        protocols = [p.lower() for p in args.protocols]
    elif args.protocol:
        protocols = [args.protocol.lower()]
    else:
        print("Missing protocol. Use --protocol <p> or --protocols <p1> [p2 ...].", file=sys.stderr)
        sys.exit(1)
    valid_protocols = {"mqtt", "amqp", "grpc", "quic", "http3", "dds"}
    for p in protocols:
        if p not in valid_protocols:
            print(f"Invalid protocol: {p}. Choose from: {sorted(valid_protocols)}", file=sys.stderr)
            sys.exit(1)

    # Resolve scenarios list: --scenarios takes precedence, else single --scenario
    scenario_choices = _all_scenario_choices()
    if args.scenarios is not None and len(args.scenarios) > 0:
        scenarios = [s.lower() for s in args.scenarios]
        for s in scenarios:
            if s not in scenario_choices:
                print(f"Unknown scenario: {s}. Choose from: {scenario_choices}", file=sys.stderr)
                sys.exit(1)
    else:
        scenarios = [(args.scenario or "excellent").lower()]

    # Build (protocol, scenario) pairs and run pipeline for each
    all_summaries = []
    pairs = [(p, s) for p in protocols for s in scenarios]
    out_dir = PROJECT_ROOT / "shared_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "diagnostic_results_latest.json"

    for idx, (protocol, scenario) in enumerate(pairs):
        print("\n" + "=" * 60)
        print(f"Diagnostic run {idx + 1}/{len(pairs)}: protocol={protocol.upper()}, scenario={scenario}")
        print("=" * 60)

        # For each (protocol, scenario) pair, iteratively refine α_proto using
        # the just-completed run until the tuned T_calc error is below the
        # target threshold, or ALPHA_TUNE_MAX_ITERS is reached.
        proto_key = protocol.upper()
        pair_summaries = []
        for iter_idx in range(ALPHA_TUNE_MAX_ITERS):
            print(f"\n[α-tune] Iteration {iter_idx + 1}/{ALPHA_TUNE_MAX_ITERS} for {proto_key} / {scenario}")
            try:
                if args.native:
                    summary = run_pipeline_native(
                        protocol=protocol,
                        use_case=args.use_case,
                        scenario=scenario,
                        enable_gpu=args.enable_gpu,
                        num_clients=args.num_clients,
                        payload_file=args.payload_file,
                    )
                else:
                    pat = SERVICE_PATTERNS.get(args.use_case, {}).get(protocol)
                    sender = args.sender_container or (pat[0] if pat else None)
                    receiver = args.receiver_container or (pat[1] if pat else None)
                    if not sender or not receiver:
                        print(
                            f"Missing sender/receiver for protocol {protocol}. Use --sender-container and --receiver-container or --use-case.",
                            file=sys.stderr,
                        )
                        break
                    # sender can be list (both clients) or single; run_pipeline expects receiver as str
                    # and sender as first client for single-container override
                    sender_list = list(sender) if isinstance(sender, (list, tuple)) else [sender]
                    summary = run_pipeline(
                        protocol=protocol,
                        use_case=args.use_case,
                        sender_container=sender_list[0] if sender_list else sender,
                        receiver_container=receiver,
                        scenario=scenario,
                        enable_gpu=args.enable_gpu,
                        network_mode=args.network_mode,
                        payload_file=args.payload_file,
                        num_clients=args.num_clients,
                    )
                pair_summaries.extend(summary)
                all_summaries.extend(summary)
                # Emit after each experiment so GUI Diagnostic Results tab updates incrementally
                print("FL_DIAG_TABLE_JSON|" + json.dumps(summary), flush=True)
                try:
                    with open(out_file, "w", encoding="utf-8") as f:
                        json.dump(all_summaries, f, indent=2)
                except Exception as e:
                    print(f"Warning: could not write {out_file}: {e}", flush=True)
                # Free GPU memory: kill processes matching *Framework_FL* using GPU
                kill_gpu_processes_matching_pattern(verbose=True)
            except Exception as e:
                print(f"[ERROR] Pipeline failed for protocol={protocol}, scenario={scenario}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                kill_gpu_processes_matching_pattern(verbose=True)
                break

            # Compute median absolute tuned error % for this (protocol, scenario)
            try:
                df_pair = pd.DataFrame(summary)
                if not df_pair.empty and "T_actual" in df_pair.columns:
                    # Prefer error_pct_tuned (new key), fall back to error_pct (legacy)
                    if "error_pct_tuned" in df_pair.columns:
                        errs = df_pair["error_pct_tuned"]
                    elif "error_pct" in df_pair.columns:
                        errs = df_pair["error_pct"]
                    else:
                        errs = None
                    if errs is not None:
                        errs_values = [abs(float(x)) for x in errs if x is not None]
                        current_err = float(np.median(errs_values)) if errs_values else 0.0
                    else:
                        current_err = 0.0
                else:
                    current_err = 0.0
            except Exception as e:
                print(f"Warning: could not compute error statistics for α tuning: {e}", flush=True)
                current_err = 0.0

            print(f"[α-tune] {proto_key} / {scenario}: median |error_pct_tuned| = {current_err:.2f}% "
                  f"(target ≤ {ALPHA_TUNE_ERROR_THRESHOLD:.2f}%)")

            # Stop if target error threshold reached
            if current_err <= ALPHA_TUNE_ERROR_THRESHOLD:
                print(f"[α-tune] Target error reached for {proto_key} / {scenario}; stopping iterations.")
                break

            # Otherwise, refit α from all data for this (protocol, scenario) so far for stability
            try:
                pair_df = pd.DataFrame(pair_summaries) if pair_summaries else df_pair
                new_alpha = fit_alpha_proto(proto_key, pair_df)
                old_alpha = ALPHA_PROTO.get(proto_key, 1.0)
                ALPHA_PROTO[proto_key] = new_alpha
                save_alpha_proto({proto_key: new_alpha}, ALPHA_PATH, merge=True)
                print(
                    f"[α-tune] Updating α_proto[{proto_key}] from {old_alpha:.3f} to {new_alpha:.3f} "
                    f"(fitted from {len(pair_summaries)} runs for this pair)."
                )
            except Exception as e:
                print(f"Warning: could not refit α_proto for {proto_key}: {e}", flush=True)
                break

    summary = all_summaries
    if not summary:
        print("No diagnostic results (all runs failed or no protocol/scenario).", file=sys.stderr)
        sys.exit(1)

    # Build DataFrame, save diagnostics, and perform mandatory α_proto refit
    try:
        df_results = pd.DataFrame(summary)
    except Exception as e:
        df_results = None
        print(f"Warning: could not create DataFrame from diagnostic results: {e}", flush=True)

    if df_results is not None and not df_results.empty:
        try:
            save_results_to_excel(df_results, DIAGNOSTIC_PATH)
            print(f"\nDiagnostics saved to Excel at {DIAGNOSTIC_PATH}")
        except Exception as e:
            print(f"Warning: could not write diagnostics Excel to {DIAGNOSTIC_PATH}: {e}", flush=True)

        # NEW: Mandatory α_proto refit and save based on this run
        print("\n=== MANDATORY: Updating α_proto from this run's diagnostics ===")
        try:
            df_diag = load_diagnostics(DIAGNOSTIC_PATH)
            alphas = calibrate_alpha_from_diagnostics(df_diag)
            if alphas:
                merged = save_alpha_proto(alphas, ALPHA_PATH, merge=True)
                # Update in-memory ALPHA_PROTO for any subsequent use in this process
                ALPHA_PROTO.update(merged)
                print("✅ α_proto updated and saved to alpha_proto.json")
                print("Next run will use these values:")
                for proto, alpha in sorted(merged.items()):
                    print(f"  {proto}: α = {float(alpha):.3f}")
            else:
                print("No valid diagnostics rows found for α_proto calibration; keeping existing values.")
        except FileNotFoundError:
            print(f"Diagnostics file {DIAGNOSTIC_PATH} not found; skipping α_proto update.", flush=True)
        except Exception as e:
            print(f"Warning: could not calibrate α_proto from diagnostics: {e}", flush=True)
    print("\n" + "=" * 60)
    print("Combined results (all protocols × scenarios)")
    print("=" * 60)
    print_table(summary)
    # If any scenario is degraded but T_actual is similar across clients and close to calibration, network may not be shaped
    scenarios_in_results = {str(r.get("scenario", "")).strip() for r in summary}
    if summary and scenarios_in_results - {"excellent"}:
        t_actuals = [r["T_actual"] for r in summary if r.get("T_actual")]
        if t_actuals and max(t_actuals) - min(t_actuals) < 0.5 and t_actuals[0] < 15:
            print(
                "NOTE: T_actual shows little variation and is relatively low for a degraded scenario.\n"
                "  Possible causes: (1) Host network + loopback — tc on eth0 does not shape 127.0.0.1 traffic;\n"
                "  (2) tc not applied (e.g. host mode without sudo, or per-container tc skipped).\n"
                "  Run with --network-mode gpu or host_macvlan to see scenario effects, or apply tc on the interface used by FL traffic.",
                flush=True,
            )
    # Final emit: full combined results (GUI merge will replace/append so tab shows complete table)
    print("FL_DIAG_TABLE_JSON|" + json.dumps(summary), flush=True)
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write {out_file}: {e}", flush=True)
    # Final GPU cleanup after all experiments
    kill_gpu_processes_matching_pattern(verbose=True)
    print("\n[DIAGNOSTIC] Experiment completed. All protocol runs finished; clients and servers will be disconnected.", flush=True)
    # Ensure GUI receives all output and file is on disk before process exits
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    if "--kill-gpu-only" in sys.argv:
        sys.argv = [a for a in sys.argv if a != "--kill-gpu-only"]
        kill_gpu_processes_matching_pattern(verbose=True)
        if not sys.argv[1:]:  # no other args
            sys.exit(0)
    main()
