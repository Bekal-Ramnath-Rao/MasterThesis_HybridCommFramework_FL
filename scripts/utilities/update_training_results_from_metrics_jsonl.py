#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def select_run_rows(rows: Iterable[dict[str, Any]], run_key: str) -> list[dict[str, Any]]:
    """
    run_key:
      - http3
      - amqp
      - grpc
      - mqtt
      - unified  (preferred: protocol=="unified"; legacy: protocol=="mqtt" plus a unified marker)
    """
    out: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for r in rows:
        all_rows.append(r)
        proto = r.get("protocol")
        if run_key in {"http3", "amqp", "grpc"}:
            if proto == run_key:
                out.append(r)
            continue

        if run_key == "mqtt":
            if proto == "mqtt" and "battery_soc_before" not in r:
                out.append(r)
            continue

        if run_key == "unified":
            # New format: unified runs explicitly tag the run, while still recording the
            # selected transport in e.g. "selected_protocol".
            if proto == "unified" or r.get("run_key") == "unified" or r.get("is_unified") is True:
                out.append(r)
                continue
            # Legacy fallback used by some older clients: stored as mqtt but with an extra field.
            if proto == "mqtt" and "battery_soc_before" in r:
                out.append(r)
            continue

        raise ValueError(f"Unknown run_key: {run_key}")

    # Safety fallback: some runs log unified rounds in a dedicated JSONL file but do not
    # tag them explicitly. In that case, selecting "unified" should still work.
    if run_key == "unified" and not out:
        for r in all_rows:
            if r.get("total_fl_wall_time_sec") is None:
                continue
            if r.get("round") is None:
                continue
            out.append(r)
    return out


def aggregate_per_round(rows: Iterable[dict[str, Any]]) -> dict[str, list[float]]:
    buckets: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in rows:
        rnd = r.get("round")
        try:
            rnd_i = int(rnd)
        except (TypeError, ValueError):
            continue

        for key in [
            "loss",
            "accuracy",
            "total_fl_wall_time_sec",
            "battery_consumption_joules_round",
            "cpu_percent",
            "memory_percent",
        ]:
            val = _safe_float(r.get(key))
            if val is not None:
                buckets[rnd_i][key].append(val)

    rounds_sorted = sorted(buckets.keys())
    out: dict[str, list[float]] = {"rounds": [float(x) for x in rounds_sorted]}
    for key in [
        "loss",
        "accuracy",
        "total_fl_wall_time_sec",
        "battery_consumption_joules_round",
        "cpu_percent",
        "memory_percent",
    ]:
        series: list[float] = []
        for rnd in rounds_sorted:
            vals = buckets[rnd].get(key, [])
            if not vals:
                series.append(float("nan"))
            else:
                series.append(sum(vals) / len(vals))
        out[key] = series
    return out


def _strip_nan_pairs(rounds: list[int], *series: list[float]) -> tuple[list[int], list[list[float]]]:
    keep_idx: list[int] = []
    for i in range(len(rounds)):
        ok = True
        for s in series:
            v = s[i]
            if v != v:  # NaN
                ok = False
                break
        if ok:
            keep_idx.append(i)
    new_rounds = [rounds[i] for i in keep_idx]
    new_series = [[s[i] for i in keep_idx] for s in series]
    return new_rounds, new_series


def update_training_results_json(training_json: Path, agg: dict[str, list[float]], run_key: str) -> None:
    data = json.loads(training_json.read_text(encoding="utf-8"))

    rounds = [int(x) for x in agg["rounds"]]
    loss = agg["loss"]
    acc = agg["accuracy"]
    wall = agg["total_fl_wall_time_sec"]
    batt = agg["battery_consumption_joules_round"]
    cpu = agg["cpu_percent"]
    mem = agg["memory_percent"]

    rounds, [loss, acc, wall, batt, cpu, mem] = _strip_nan_pairs(
        rounds, loss, acc, wall, batt, cpu, mem
    )

    # Canonical keys (keep legacy keys too when present)
    data["rounds"] = rounds
    data["loss"] = loss
    data["accuracy"] = acc
    data["total_fl_wall_time_sec"] = wall
    data["battery_consumption_joules_round"] = batt
    data["avg_cpu_percent"] = cpu
    data["avg_memory_percent"] = mem
    data["avg_cpu_percent_source"] = "client_metrics_jsonl"
    data["avg_memory_percent_source"] = "client_metrics_jsonl"
    data["total_fl_wall_time_sec_source"] = "client_metrics_jsonl"
    data["battery_consumption_joules_round_source"] = "client_metrics_jsonl"
    data["updated_from_metrics_jsonl_run_key"] = run_key

    # Back-compat: if older plotting expects round_times_seconds / battery_consumption, update them too
    data["round_times_seconds"] = wall
    data["battery_consumption"] = batt

    data["total_rounds"] = len(rounds)
    if rounds:
        data["final_accuracy"] = acc[-1]
        data["final_loss"] = loss[-1]

    training_json.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Update *training_results.json from client metrics JSONL (avg per round).")
    ap.add_argument("--metrics-jsonl", required=True, type=Path)
    ap.add_argument("--run", required=True, choices=["http3", "amqp", "grpc", "mqtt", "unified"])
    ap.add_argument("--training-json", required=True, type=Path)
    args = ap.parse_args()

    rows = read_jsonl(args.metrics_jsonl)
    run_rows = select_run_rows(rows, args.run)
    agg = aggregate_per_round(run_rows)
    update_training_results_json(args.training_json, agg, args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

