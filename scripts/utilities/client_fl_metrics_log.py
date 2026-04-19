"""
Per-client FL metrics (JSONL) for multi-machine experiments.

Writes one JSON object per line to:
  ``{CLIENT_METRICS_LOG_DIR or /shared_data or cwd}/client_fl_metrics_{use_case}_client{id}.jsonl``

Every record is normalized to include:
  - loss, accuracy (classification; regression may map MSE into ``accuracy`` when callers send it)
  - battery_energy_joules, cumulative_battery_energy_joules, battery_soc_after,
    battery_consumption_joules_round / _cumulative
  - total_fl_wall_time_sec (this round), total_fl_cumulative_wall_time_sec (train+comm, summed),
    total_fl_cumulative_training_time_sec and total_fl_training_time_sec (local training only, summed)
  - cpu_percent (host CPU % sampled at log time via psutil, or caller-supplied)
  - memory_percent (host RAM % sampled at log time via psutil, or caller-supplied)

Environment:
  CLIENT_METRICS_LOG — set to 0/false/no to disable
  CLIENT_METRICS_LOG_DIR — output directory (distributed_client_gui.py sets this to /shared_data when shared_data is mounted)
  CLIENT_USE_CASE — overrides default use_case (e.g. emotion, temperature, mental_state)
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

# Per (use_case, client_id) running totals in the client process.
_cumulative_wall_sec: Dict[Tuple[str, int], float] = {}
_cumulative_training_sec: Dict[Tuple[str, int], float] = {}
_cumulative_battery_joules: Dict[Tuple[str, int], float] = {}
_psutil_cpu_warmed = False


def use_case_from_env(default: str = "emotion") -> str:
    v = os.environ.get("CLIENT_USE_CASE", default).strip()
    return v if v else default


def _sanitize_use_case(name: str) -> str:
    s = (name or "emotion").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    return s or "emotion"


def _infer_total_fl_wall_time_sec(record: Dict[str, Any]) -> float:
    v = record.get("total_fl_wall_time_sec")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    t = 0.0
    for k in (
        "training_time_sec",
        "uplink_model_comm_sec",
        "uplink_metrics_comm_sec",
        "pre_uplink_delay_sec",
    ):
        x = record.get(k)
        if x is not None:
            try:
                t += float(x)
            except (TypeError, ValueError):
                pass
    return t


def _coerce_accuracy(record: Dict[str, Any]) -> float:
    if record.get("accuracy") is not None:
        try:
            return float(record["accuracy"])
        except (TypeError, ValueError):
            pass
    for k in ("val_accuracy", "accuracy_proxy", "top2_accuracy"):
        if record.get(k) is not None:
            try:
                return float(record[k])
            except (TypeError, ValueError):
                pass
    if record.get("mse") is not None:
        try:
            return float(record["mse"])
        except (TypeError, ValueError):
            pass
    return 0.0


def _coerce_loss(record: Dict[str, Any]) -> float:
    if record.get("loss") is not None:
        try:
            return float(record["loss"])
        except (TypeError, ValueError):
            pass
    return 0.0


def _normalize_metrics_record(
    client_id: int,
    record: Dict[str, Any],
    *,
    use_case: str,
) -> Dict[str, Any]:
    uc = _sanitize_use_case(use_case)
    key = (uc, int(client_id))

    out = dict(record)
    train_sec = out.get("training_time_sec")
    try:
        train_sec_f = float(train_sec) if train_sec is not None else 0.0
    except (TypeError, ValueError):
        train_sec_f = 0.0
    wall_sec = _infer_total_fl_wall_time_sec(out)
    out["total_fl_wall_time_sec"] = wall_sec
    out["training_time_sec"] = train_sec_f

    _cumulative_wall_sec[key] = _cumulative_wall_sec.get(key, 0.0) + wall_sec
    _cumulative_training_sec[key] = _cumulative_training_sec.get(key, 0.0) + train_sec_f
    out["total_fl_cumulative_wall_time_sec"] = float(_cumulative_wall_sec[key])
    out["total_fl_cumulative_training_time_sec"] = float(_cumulative_training_sec[key])
    # Total local training time accumulated across rounds (excludes comm); wall-clock FL is *_wall_*.
    out["total_fl_training_time_sec"] = float(_cumulative_training_sec[key])

    out["loss"] = _coerce_loss(out)
    out["accuracy"] = _coerce_accuracy(out)

    ej = out.get("battery_energy_joules")
    try:
        ej_f = float(ej) if ej is not None else 0.0
    except (TypeError, ValueError):
        ej_f = 0.0
    out["battery_energy_joules"] = ej_f
    out["battery_consumption_joules_round"] = ej_f

    cum_in = out.get("cumulative_battery_energy_joules")
    if cum_in is not None:
        try:
            cum_b = float(cum_in)
        except (TypeError, ValueError):
            cum_b = _cumulative_battery_joules.get(key, 0.0) + ej_f
    else:
        cum_b = _cumulative_battery_joules.get(key, 0.0) + ej_f
    _cumulative_battery_joules[key] = cum_b
    out["cumulative_battery_energy_joules"] = float(cum_b)
    out["battery_consumption_joules_cumulative"] = float(cum_b)

    soc = out.get("battery_soc_after")
    if soc is None:
        soc = out.get("battery_soc")
    try:
        out["battery_soc_after"] = float(soc) if soc is not None else 1.0
    except (TypeError, ValueError):
        out["battery_soc_after"] = 1.0

    if out.get("battery_soc_before") is not None:
        try:
            out["battery_soc_before"] = float(out["battery_soc_before"])
        except (TypeError, ValueError):
            out.pop("battery_soc_before", None)

    # CPU and memory: use caller-supplied values if present, otherwise sample now via psutil.
    # interval=0.0 returns the non-blocking cached reading — fast enough for per-round logging.
    if out.get("cpu_percent") is None and _HAS_PSUTIL:
        try:
            out["cpu_percent"] = float(_psutil.cpu_percent(interval=0.0))
        except Exception:
            pass
    if out.get("memory_percent") is None and _HAS_PSUTIL:
        try:
            out["memory_percent"] = float(_psutil.virtual_memory().percent)
        except Exception:
            pass

    return out


def append_client_fl_metrics_record(
    client_id: int,
    record: Dict[str, Any],
    *,
    use_case: str = "emotion",
    protocol: Optional[str] = None,
) -> None:
    """Append a single JSON line with FL metrics (loss, accuracy, times, battery, etc.)."""
    global _psutil_cpu_warmed
    if _HAS_PSUTIL and not _psutil_cpu_warmed:
        try:
            _psutil.cpu_percent(interval=0.05)
        except Exception:
            pass
        _psutil_cpu_warmed = True
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

    out = _normalize_metrics_record(int(client_id), record, use_case=uc)
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
