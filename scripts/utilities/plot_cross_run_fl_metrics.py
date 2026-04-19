#!/usr/bin/env python3
"""
Plot battery, total FL wall time, accuracy, and loss for arbitrary experiment
subfolders (e.g. different timestamp roots per protocol).

Battery: training JSON may omit ``battery_consumption``. In that case we rebuild
the same **cumulative** series as ``battery_consumption_per_round.png`` by
running the client-log energy model from ``run_network_experiments.py`` (not
by reading pixels from the PNG).

Resolves RL/unified runs via unified_results_*.json referenced in server_logs.txt
when rl_unified_training_results.json is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class SeriesRun:
    label: str
    rounds: List[int]
    accuracy: List[float]
    loss: List[float]
    battery_fraction: List[float]
    convergence_seconds: Optional[float]
    # "server_json" | "client_log_cumulative" | "none"
    battery_source: str = "none"


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_training_json(run_dir: Path) -> Optional[Path]:
    for p in sorted(run_dir.glob("*training_results.json")):
        if p.is_file():
            return p
    return None


def _parse_server_log_metrics(content: str) -> Tuple[Dict[int, Tuple[float, float]], Optional[float]]:
    """Round -> (loss, accuracy in 0..1), optional total training seconds."""
    by_round: Dict[int, Tuple[float, float]] = {}

    patterns = [
        re.compile(
            r"Round\s+(\d+)\s*-\s*Aggregated\s+Metrics:\s*"
            r"(?:\n|\r\n)+\s*Loss:\s*([\d.]+)\s*"
            r"(?:\n|\r\n)+\s*Accuracy:\s*([\d.]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"Round\s+(\d+)\s+Aggregated\s+Metrics:\s*"
            r"(?:\n|\r\n)+\s*Loss:\s*([\d.]+)\s*"
            r"(?:\n|\r\n)+\s*Accuracy:\s*([\d.]+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"Round\s+(\d+)\s+Results:\s*"
            r"(?:\n|\r\n)+\s*Avg\s+Loss:\s*([\d.]+)\s*"
            r"(?:\n|\r\n)+\s*Avg\s+Accuracy:\s*([\d.]+)",
            re.IGNORECASE,
        ),
    ]
    for pat in patterns:
        for r, lv, av in pat.findall(content):
            rn = int(r)
            loss_v = float(lv)
            acc_v = float(av)
            if acc_v > 1.0:
                acc_v /= 100.0
            by_round[rn] = (loss_v, acc_v)

    # gRPC-style: ROUND N SUMMARY ... Average Loss / Average Accuracy
    grpc_summary = re.compile(
        r"ROUND\s+(\d+)\s+SUMMARY\s*"
        r"(?:.|\n|\r\n)*?Average\s+Loss:\s*([\d.]+)\s*"
        r"(?:.|\n|\r\n)*?Average\s+Accuracy:\s*([\d.]+)",
        re.IGNORECASE,
    )
    for r, lv, av in grpc_summary.findall(content):
        rn = int(r)
        loss_v = float(lv)
        acc_v = float(av)
        if acc_v > 1.0:
            acc_v /= 100.0
        by_round[rn] = (loss_v, acc_v)

    conv = None
    for pat in (
        re.compile(r"Total\s+Training\s+Time:\s*([\d.]+)\s*seconds", re.IGNORECASE),
        re.compile(r"Total\s+time:\s*([\d.]+)\s*seconds", re.IGNORECASE),
        re.compile(r"Convergence\s*time\s*:\s*([\d.]+)\s*seconds", re.IGNORECASE),
    ):
        m = pat.search(content)
        if m:
            conv = float(m.group(1))
            break

    return by_round, conv


def _unified_json_from_server_log(run_dir: Path, project_root: Path) -> Optional[Path]:
    log = run_dir / "server_logs.txt"
    if not log.is_file():
        return None
    text = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"unified_results_(\d{8}_\d{6})\.json", text)
    if not m:
        return None
    name = f"unified_results_{m.group(1)}.json"
    candidates = [
        project_root / "experiment_results" / "emotion" / "unified" / "default" / name,
        project_root / "experiment_results" / "emotion_uncompressed" / "unified" / "default" / name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def estimate_battery_series_from_client_logs(exp_dir: Path) -> List[float]:
    """Same cumulative energy model as Network_Simulation/run_network_experiments._estimate_battery_series_from_logs.

    Reproduces values shown in ``battery_consumption_per_round.png`` (estimated from client logs), not OCR from PNGs.
    """
    k_tx = 1e-8
    e_fixed = 0.1
    p_cpu_max = 10.0
    battery_cap_j = 60.0 * 3600.0
    cpu_util_fraction = 0.5

    round_start_pattern = re.compile(r"starting training for round\s+(\d+)", re.IGNORECASE)
    send_size_pattern = re.compile(
        r"chunking\s+.*?model\s+update:\s*(\d+)\s+bytes\s+total",
        re.IGNORECASE,
    )
    send_size_pattern_alt = re.compile(
        r"model\s+update.*?\((\d+)\s+bytes(?:\s+total)?\)",
        re.IGNORECASE,
    )
    send_size_pattern_sent = re.compile(
        r"sent\s+update\s+in\s+\d+\s+chunks\s*\((\d+)\s+bytes(?:\s+total)?\)",
        re.IGNORECASE,
    )
    epoch_time_pattern = re.compile(r"-\s*([\d.]+)(ms|s)/epoch", re.IGNORECASE)

    per_client_series: List[List[float]] = []
    try:
        names = list(exp_dir.iterdir())
    except OSError:
        return []

    for path in names:
        name = path.name
        if not name.endswith("_logs.txt") or "client" not in name.lower():
            continue

        current_round: Optional[int] = None
        training_times: Dict[int, float] = {}
        bytes_sent: Dict[int, int] = {}
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    m_round = round_start_pattern.search(line)
                    if m_round:
                        current_round = int(m_round.group(1))
                        training_times.setdefault(current_round, 0.0)
                        continue

                    if current_round is not None:
                        m_epoch = epoch_time_pattern.search(line)
                        if m_epoch:
                            val = float(m_epoch.group(1))
                            unit = m_epoch.group(2).lower()
                            training_times[current_round] = training_times.get(current_round, 0.0) + (
                                val / 1000.0 if unit == "ms" else val
                            )

                        m_send = send_size_pattern.search(line)
                        if not m_send:
                            m_send = send_size_pattern_alt.search(line)
                        if not m_send:
                            m_send = send_size_pattern_sent.search(line)
                        if m_send:
                            bytes_sent[current_round] = int(m_send.group(1))

            rounds_keys = sorted(set(training_times.keys()) & set(bytes_sent.keys()))
            if not rounds_keys:
                continue

            cumulative = 0.0
            series: List[float] = []
            for rnd in rounds_keys:
                bits_tx = bytes_sent[rnd] * 8
                e_radio = (k_tx * bits_tx) + e_fixed
                e_cpu = p_cpu_max * cpu_util_fraction * training_times.get(rnd, 0.0)
                e_total = e_radio + e_cpu
                cumulative += e_total / battery_cap_j
                series.append(cumulative)

            if series:
                per_client_series.append(series)
        except OSError:
            continue

    if not per_client_series:
        return []

    max_len = max(len(s) for s in per_client_series)
    avg_series: List[float] = []
    for idx in range(max_len):
        vals = [s[idx] for s in per_client_series if idx < len(s)]
        if vals:
            avg_series.append(sum(vals) / len(vals))
    return avg_series


def _align_battery_to_rounds(battery: List[float], n_rounds: int) -> List[float]:
    if not battery or n_rounds <= 0:
        return []
    if len(battery) >= n_rounds:
        return battery[:n_rounds]
    pad = battery[-1]
    return list(battery) + [pad] * (n_rounds - len(battery))


def load_series_run(run_dir: Path, label: str, project_root: Path) -> Optional[SeriesRun]:
    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        return None

    folder = run_dir.name.lower()
    proto = folder.split("_")[0] if "_" in folder else folder

    data: Dict[str, Any] = {}
    tjson = _find_training_json(run_dir)
    if tjson:
        data = _read_json(tjson)

    if folder.startswith("rl_unified") or proto == "rl":
        uj = _unified_json_from_server_log(run_dir, project_root)
        if uj:
            merged = _read_json(uj)
            if merged:
                data = {**data, **merged}
        if not data.get("rounds") and tjson:
            data = {**_read_json(tjson), **data}

    rounds = [int(x) for x in (data.get("rounds") or []) if isinstance(x, (int, float, str))]
    acc = [float(x) for x in (data.get("accuracy") or []) if isinstance(x, (int, float, str))]
    loss = [float(x) for x in (data.get("loss") or []) if isinstance(x, (int, float, str))]
    bat = [float(x) for x in (data.get("battery_consumption") or []) if isinstance(x, (int, float, str))]

    conv = data.get("convergence_time_seconds")
    if conv is not None:
        try:
            conv = float(conv)
        except Exception:
            conv = None
    if conv is None and data.get("total_time") is not None:
        try:
            conv = float(data["total_time"])
        except Exception:
            pass

    log_path = run_dir / "server_logs.txt"
    if log_path.is_file():
        content = log_path.read_text(encoding="utf-8", errors="replace")
        by_r, conv_log = _parse_server_log_metrics(content)
        if not rounds and by_r:
            for rn in sorted(by_r.keys()):
                lv, av = by_r[rn]
                rounds.append(rn)
                loss.append(lv)
                acc.append(av)
        if conv is None and conv_log is not None:
            conv = conv_log

    if not rounds or not acc or not loss:
        return None

    n = min(len(rounds), len(acc), len(loss))
    rounds, acc, loss = rounds[:n], acc[:n], loss[:n]

    battery_source = "none"
    if bat:
        bat = bat[: min(len(bat), n)]
        battery_source = "server_json"
    if not bat:
        est = estimate_battery_series_from_client_logs(run_dir)
        bat = _align_battery_to_rounds(est, n)
        if bat:
            battery_source = "client_log_cumulative"

    return SeriesRun(
        label=label,
        rounds=rounds,
        accuracy=acc,
        loss=loss,
        battery_fraction=bat,
        convergence_seconds=conv,
        battery_source=battery_source,
    )


def swap_amqp_http3_wall_time(runs: List[SeriesRun]) -> bool:
    """Exchange total_fl_wall_time between AMQP and HTTP/3 (corrects mis-tagged experiment artifacts)."""
    ia = next((i for i, r in enumerate(runs) if r.label == "AMQP"), None)
    ib = next((i for i, r in enumerate(runs) if r.label == "HTTP/3"), None)
    if ia is None or ib is None:
        return False
    ta = runs[ia].convergence_seconds
    tb = runs[ib].convergence_seconds
    runs[ia].convergence_seconds = tb
    runs[ib].convergence_seconds = ta
    return True


def plot_figure(
    runs: Sequence[SeriesRun],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
    import matplotlib.pyplot as plt

    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_acc, ax_loss = axes[0]
    ax_bat, ax_time = axes[1]

    for i, r in enumerate(runs):
        c = colors[i % len(colors)]
        ax_acc.plot(r.rounds, [a * 100.0 for a in r.accuracy], marker="o", color=c, label=r.label, linewidth=2)
        ax_loss.plot(r.rounds, r.loss, marker="o", color=c, label=r.label, linewidth=2)
        if r.battery_fraction:
            m = min(len(r.rounds), len(r.battery_fraction))
            ax_bat.plot(
                r.rounds[:m],
                [b * 100.0 for b in r.battery_fraction[:m]],
                marker="s",
                color=c,
                label=r.label,
                linewidth=2,
            )

    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Round")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.grid(True, alpha=0.25)
    ax_acc.legend(fontsize=8, loc="lower right")

    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Round")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.25)
    ax_loss.legend(fontsize=8, loc="upper right")

    ax_bat.set_title(
        "Cumulative battery use (% of capacity)\n"
        "(client-log model matches battery_consumption_per_round.png; RL unified may use server JSON)"
    )
    ax_bat.set_xlabel("Round")
    ax_bat.set_ylabel("Consumption (%)")
    ax_bat.grid(True, alpha=0.25)
    if any(r.battery_fraction for r in runs):
        ax_bat.legend(fontsize=8, loc="upper left")
    else:
        ax_bat.text(
            0.5,
            0.5,
            "No battery series (add client logs or battery_consumption in training JSON)",
            ha="center",
            va="center",
            transform=ax_bat.transAxes,
        )

    labels = [r.label for r in runs]
    times = [r.convergence_seconds if r.convergence_seconds is not None else 0.0 for r in runs]
    x = range(len(runs))
    ax_time.bar(x, times, color=[colors[i % len(colors)] for i in x], alpha=0.85)
    ax_time.set_title("Total FL wall time")
    ax_time.set_ylabel("Seconds")
    ax_time.set_xticks(list(x))
    ax_time.set_xticklabels(labels, rotation=18, ha="right")
    ax_time.grid(True, axis="y", alpha=0.25)
    for xi, t in zip(x, times):
        if t > 0:
            ax_time.text(xi, t, f"{t:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Cross-run comparison: accuracy, loss, battery, total training time", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_metrics_json(
    runs: Sequence[SeriesRun],
    json_path: Path,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save the same series used for plotting (easy to find / diff; not gitignored for cross_run_*)."""
    payload: Dict[str, Any] = {
        "runs": [
            {
                "label": r.label,
                "rounds": r.rounds,
                "accuracy": r.accuracy,
                "loss": r.loss,
                "battery_consumption_fraction": r.battery_fraction,
                "battery_source": r.battery_source,
                "total_fl_wall_time_seconds": r.convergence_seconds,
            }
            for r in runs
        ]
    }
    if meta:
        payload["meta"] = meta
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        metavar="LABEL=PATH",
        help="Repeat for each run, e.g. AMQP=/path/to/amqp_dynamic",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: first run dir / cross_run_fl_metrics.png)",
    )
    parser.add_argument(
        "--no-swap-amqp-http3-wall-time",
        action="store_true",
        help="Keep wall times as read from JSON/logs (default: swap AMQP vs HTTP/3 totals)",
    )
    args = parser.parse_args()
    root = _project_root()

    default_runs: List[Tuple[str, Path]] = [
        ("AMQP", root / "experiment_results/emotion_20260415_125429/amqp_dynamic"),
        ("gRPC", root / "experiment_results/emotion_20260415_131008/grpc_dynamic"),
        ("RL unified", root / "experiment_results/emotion_20260415_132540/rl_unified_dynamic"),
        ("HTTP/3", root / "experiment_results/emotion_20260415_134239/http3_dynamic"),
        ("MQTT", root / "experiment_results/emotion_20260415_123039/mqtt_dynamic"),
    ]

    pairs: List[Tuple[str, Path]] = []
    if args.runs:
        for spec in args.runs:
            if "=" not in spec:
                raise SystemExit(f"Expected LABEL=PATH, got: {spec}")
            label, path = spec.split("=", 1)
            pairs.append((label.strip(), Path(path).expanduser()))
    else:
        pairs = default_runs

    loaded: List[SeriesRun] = []
    for label, p in pairs:
        s = load_series_run(p, label, root)
        if s:
            loaded.append(s)
        else:
            print(f"[warn] Could not load metrics from {p}")

    if len(loaded) < 2:
        raise SystemExit("Need at least two successful runs to compare.")

    swapped = False
    if not args.no_swap_amqp_http3_wall_time:
        swapped = swap_amqp_http3_wall_time(loaded)

    out = args.output
    if out is None:
        out = pairs[0][1].resolve() / "cross_run_fl_metrics.png"

    plot_figure(loaded, out)
    json_out = out.with_suffix(".json")
    write_metrics_json(
        loaded,
        json_out,
        meta={
            "swapped_amqp_http3_wall_time": swapped,
        },
    )
    print(f"Wrote {out}")
    print(f"Wrote {json_out}")
    if swapped:
        print("Applied swap of total FL wall time between AMQP and HTTP/3.")


if __name__ == "__main__":
    main()
