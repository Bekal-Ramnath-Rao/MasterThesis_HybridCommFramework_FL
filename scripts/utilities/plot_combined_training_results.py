#!/usr/bin/env python3
"""
Overlay training curves from *training_results.json files (HTTP/3, gRPC, MQTT,
AMQP, RL unified, etc.).

Default input paths and all metrics are defined below; run with no arguments
from the repo root, or pass optional paths to override. Use ``--no-split`` to
skip per-metric PNGs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Repository root (this file lives in scripts/utilities/)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Default experiment JSONs — edit here to point at your runs.
DEFAULT_TRAINING_JSON_PATHS: List[Path] = [
    REPO_ROOT
    / "experiment_results/emotion_20260418_183419/http3_excellent/http3_training_results.json",
    REPO_ROOT
    / "experiment_results/emotion_20260418_123535/rl_unified_dynamic/rl_unified_training_results.json",
    REPO_ROOT
    / "experiment_results/emotion_20260418_111428/grpc_dynamic/grpc_training_results.json",
    REPO_ROOT
    / "experiment_results/emotion_20260417_200957/amqp_dynamic/amqp_training_results.json",
    REPO_ROOT
    / "experiment_results/emotion_20260417_191832/mqtt_dynamic/mqtt_training_results.json",
]

# All series plotted (order = subplot order). Derived metrics computed when absent in JSON.
ALL_METRICS: List[str] = [
    "accuracy",
    "loss",
    "round_times_seconds",
    "cumulative_wall_time_seconds",
    "battery_consumption",
    "battery_model_consumption",
]

DEFAULT_OUTPUT = REPO_ROOT / "experiment_results/combined_training_results.png"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _default_label(path: Path) -> str:
    stem = path.stem
    m = re.match(r"^(.+)_training_results$", stem, re.I)
    base = m.group(1) if m else stem
    base = base.replace("_", " ").strip()
    # Short friendly names
    aliases = {
        "rl unified": "RL unified (dynamic)",
        "http3": "HTTP/3",
        "grpc": "gRPC",
        "mqtt": "MQTT",
        "amqp": "AMQP",
        "quic": "QUIC",
    }
    key = base.lower()
    return aliases.get(key, base.title())


def _merge_duplicate_rounds(
    rounds: Sequence[int], *series: Sequence[float]
) -> Tuple[List[int], List[Tuple[float, ...]]]:
    """If the same round index appears multiple times, keep the last sample."""
    if not rounds:
        return [], [tuple() for _ in series]
    n = min(len(rounds), *(len(s) for s in series if s))
    if n == 0:
        return [], [tuple() for _ in series]
    merged: Dict[int, Tuple[float, ...]] = {}
    for i in range(n):
        key = int(rounds[i])
        vals = tuple(float(series[j][i]) for j in range(len(series)))
        merged[key] = vals
    order = sorted(merged.keys())
    out_series: List[List[float]] = [[] for _ in series]
    for k in order:
        for j, v in enumerate(merged[k]):
            out_series[j].append(v)
    return order, [tuple(out_series[j]) for j in range(len(series))]


def _extract_metric(data: Dict[str, Any], name: str) -> List[float]:
    if name not in data:
        return []
    v = data[name]
    if not isinstance(v, list):
        return []
    return [float(x) for x in v]


def _derived_metric_series(data: Dict[str, Any], name: str) -> Optional[List[float]]:
    """Synthetic series aligned with ``rounds`` (same length as round_times_seconds)."""
    if name == "cumulative_wall_time_seconds":
        rts = data.get("round_times_seconds")
        if not isinstance(rts, list) or not rts:
            return None
        total = 0.0
        out: List[float] = []
        for x in rts:
            total += float(x)
            out.append(total)
        return out
    return None


def _prepare_single_metric(
    data: Dict[str, Any], metric: str
) -> Tuple[List[int], List[float]]:
    """One metric at a time so a missing optional field does not skip the whole run."""
    rounds = data.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return [], []
    raw = _extract_metric(data, metric)
    if raw:
        series = raw
    else:
        derived = _derived_metric_series(data, metric)
        if not derived:
            return [], []
        series = derived
    n = min(len(rounds), len(series))
    if n == 0:
        return [], []
    rounds = rounds[:n]
    series = series[:n]
    r_merged, merged_tuples = _merge_duplicate_rounds(rounds, series)
    y = [merged_tuples[0][i] for i in range(len(r_merged))]
    return r_merged, y


def _plot(
    paths: List[Path],
    labels: List[str],
    metrics: List[str],
    output: Path | None,
    show: bool,
    title: str | None,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    n_metrics = len(metrics)
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(9, 3.2 * n_metrics),
        sharex=True,
        squeeze=False,
    )
    ax_list = axes.flatten()

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(colors) < max(1, len(paths)):
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    for pi, path in enumerate(paths):
        data = _read_json(path)
        label = labels[pi] if pi < len(labels) else _default_label(path)
        color = colors[pi % len(colors)]
        for mi, metric in enumerate(metrics):
            rounds, y = _prepare_single_metric(data, metric)
            if not rounds:
                continue
            ax = ax_list[mi]
            ax.plot(rounds, y, marker="o", markersize=3, linewidth=1.5, label=label, color=color)

    metric_titles = {
        "accuracy": "Accuracy",
        "loss": "Loss",
        "round_times_seconds": "FL round wall time (s)",
        "cumulative_wall_time_seconds": "Cumulative FL wall time (s)",
        "battery_consumption": "Battery consumption (fraction)",
        "battery_model_consumption": "Battery model consumption (fraction)",
    }
    for mi, metric in enumerate(metrics):
        ax = ax_list[mi]
        ax.set_ylabel(metric_titles.get(metric, metric.replace("_", " ").title()))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    ax_list[-1].set_xlabel("Round")
    fig.suptitle(title or "Federated learning — training results comparison", fontsize=12, y=1.02)
    fig.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Combine and plot all metrics from *training_results.json files "
        "(defaults: paths and metrics are set in this script)."
    )
    p.add_argument(
        "json_files",
        nargs="*",
        type=Path,
        metavar="PATH",
        default=[],
        help="Optional paths to training_results.json files (default: DEFAULT_TRAINING_JSON_PATHS in script)",
    )
    p.add_argument(
        "--no-split",
        action="store_true",
        help="Do not write per-metric PNGs ({stem}_{metric}.png). By default they are written when -o is set.",
    )
    p.add_argument(
        "--no-combined",
        action="store_true",
        help="With --split, do not write the multi-panel combined image (only per-metric files).",
    )
    p.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=None,
        help="Series label for each file, in the same order as PATH. Repeatable.",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Save figure to this path (e.g. combined_training.png)",
    )
    p.add_argument("--title", type=str, default=None, help="Figure title")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true", help="Show interactive window")
    args = p.parse_args()

    raw_paths = list(args.json_files) if args.json_files else DEFAULT_TRAINING_JSON_PATHS
    paths = [Path(f).resolve() for f in raw_paths]
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Not a file: {path}")

    metrics = list(ALL_METRICS)
    split = not args.no_split

    labels: List[str]
    if args.labels:
        if len(args.labels) != len(paths):
            raise SystemExit(
                f"Expected {len(paths)} --label values (one per file), got {len(args.labels)}"
            )
        labels = args.labels
    else:
        labels = [_default_label(path) for path in paths]

    out = args.output
    if split and args.no_combined and not out:
        raise SystemExit("--no-combined with split output requires -o PATH (basename for split files).")
    if out is None and not args.show:
        out = DEFAULT_OUTPUT

    # Combined multi-panel: save to -o; with split, show interactively only at the end.
    if not args.no_combined:
        show_combined_now = bool(args.show and not split)
        _plot(paths, labels, metrics, out, show_combined_now, args.title, args.dpi)
        if out:
            print(f"Wrote {out}")
    elif args.show:
        _plot(paths, labels, metrics, None, True, args.title, args.dpi)

    if split:
        if out is None:
            raise SystemExit("Per-metric output requires -o PATH (or use default output path)")
        stem = out.stem
        suffix = out.suffix if out.suffix else ".png"
        parent = out.parent
        for metric in metrics:
            split_path = parent / f"{stem}_{metric}{suffix}"
            _plot(paths, labels, [metric], split_path, False, args.title, args.dpi)
            print(f"Wrote {split_path}")

    if split and args.show and not args.no_combined:
        _plot(paths, labels, metrics, None, True, args.title, args.dpi)


if __name__ == "__main__":
    main()
