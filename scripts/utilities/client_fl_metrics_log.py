"""
Per-client FL metrics (JSONL) for multi-machine experiments.

Writes one JSON object per line to:
  ``{CLIENT_METRICS_LOG_DIR or /shared_data or cwd}/client_fl_metrics_{use_case}_client{id}.jsonl``

Environment:
  CLIENT_METRICS_LOG — set to 0/false/no to disable
  CLIENT_METRICS_LOG_DIR — output directory
  CLIENT_USE_CASE — overrides default use_case (e.g. emotion, temperature, mental_state)
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional


def use_case_from_env(default: str = "emotion") -> str:
    v = os.environ.get("CLIENT_USE_CASE", default).strip()
    return v if v else default


def _sanitize_use_case(name: str) -> str:
    s = (name or "emotion").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    return s or "emotion"


def append_client_fl_metrics_record(
    client_id: int,
    record: Dict[str, Any],
    *,
    use_case: str = "emotion",
    protocol: Optional[str] = None,
) -> None:
    """Append a single JSON line with FL metrics (loss, accuracy or regression metrics, times, battery, etc.)."""
    if os.environ.get("CLIENT_METRICS_LOG", "true").strip().lower() in ("0", "false", "no"):
        return
    uc = _sanitize_use_case(use_case_from_env(use_case))
    log_dir = os.environ.get("CLIENT_METRICS_LOG_DIR", "").strip()
    if not log_dir:
        log_dir = "/shared_data" if os.path.exists("/shared_data") else os.getcwd()
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = os.getcwd()
    path = os.path.join(log_dir, f"client_fl_metrics_{uc}_client{client_id}.jsonl")
    out = dict(record)
    if protocol:
        out.setdefault("protocol", protocol)
    if "ts" not in out:
        out["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[Client {client_id}] Appended FL metrics: {path}")
    except Exception as e:
        print(f"[Client {client_id}] WARNING: could not write client FL metrics log: {e}")
