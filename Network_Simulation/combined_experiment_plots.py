#!/usr/bin/env python3
"""
Combine plots across protocols for a single experiment timestamp folder.

Expected folder layout (already used by the GUI scanner):
  experiment_results/<experiment_folder>/<protocol>_<scenario>/...
Where each protocol_scenario subfolder typically contains:
  - <proto>_training_results.json (preferred, if present)
  - metadata.json (optional)
  - server_logs.txt (optional, used as fallback for model size / payload bytes)

Outputs:
  - combined_metrics_grid.png (single figure with subplots)
  - combined_fl_rounds_bar.png (bar chart)
  - combined_convergence_time_bar.png (bar chart)
  - combined_model_size_bar.png (bar chart, if available)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RunMetrics:
    protocol: str
    scenario: str
    label: str  # used for x-axis ticks
    rounds: List[int]
    loss: List[float]
    accuracy: List[float]
    battery_consumption: List[float]  # fraction [0..1] per round, or empty
    total_rounds: Optional[int]
    convergence_time_seconds: Optional[float]
    model_size_bytes: Optional[float]  # best-effort estimate
    source_dir: Path


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_protocol_and_scenario(folder_name: str) -> Tuple[str, str]:
    parts = folder_name.split("_")
    if len(parts) < 2:
        return folder_name.lower(), "unknown"
    protocol = parts[0].lower()
    scenario = "_".join(parts[1:]).split("_congestion_")[0] if "_congestion_" in folder_name else "_".join(parts[1:])
    return protocol, scenario.lower()


def _find_training_results_json(run_dir: Path, protocol: str) -> Optional[Path]:
    # Common names: mqtt_training_results.json, grpc_training_results.json, unified_training_results.json, ...
    preferred = run_dir / f"{protocol}_training_results.json"
    if preferred.is_file():
        return preferred
    # Fallback: any *training_results.json in folder
    for p in run_dir.glob("*training_results.json"):
        if p.is_file():
            return p
    return None


def _parse_model_size_bytes_from_server_logs(server_log: Path) -> Optional[float]:
    """
    Best-effort model size estimate from logs.
    Priority:
      - explicit "model payload bytes" lines
      - "Sending global model ... (XXXX bytes total)" lines
      - "Sent global model ... (XXXX bytes)" lines
    Returns median bytes if multiple.
    """
    try:
        text = server_log.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    candidates: List[int] = []

    for m in re.finditer(r"model payload bytes:\s*(\d+)", text, re.IGNORECASE):
        candidates.append(int(m.group(1)))
    for m in re.finditer(r"\((\d+)\s*bytes total\)", text, re.IGNORECASE):
        candidates.append(int(m.group(1)))
    for m in re.finditer(r"\((\d+)\s*bytes\)", text, re.IGNORECASE):
        candidates.append(int(m.group(1)))

    if not candidates:
        return None
    candidates.sort()
    mid = len(candidates) // 2
    return float(candidates[mid])


def load_runs(experiment_folder: Path) -> List[RunMetrics]:
    runs: List[RunMetrics] = []
    if not experiment_folder.is_dir():
        return runs

    for child in sorted([p for p in experiment_folder.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        protocol, scenario = _extract_protocol_and_scenario(child.name)
        training_json = _find_training_results_json(child, protocol)
        data = _safe_read_json(training_json) if training_json else {}
        meta = _safe_read_json(child / "metadata.json")

        rounds = [int(x) for x in (data.get("rounds") or []) if isinstance(x, (int, float, str))]
        loss = [float(x) for x in (data.get("loss") or []) if isinstance(x, (int, float, str))]
        accuracy = [float(x) for x in (data.get("accuracy") or []) if isinstance(x, (int, float, str))]
        battery = [float(x) for x in (data.get("battery_consumption") or []) if isinstance(x, (int, float, str))]

        total_rounds = data.get("total_rounds")
        if total_rounds is None and rounds:
            total_rounds = len(rounds)

        conv_sec = data.get("convergence_time_seconds")
        if conv_sec is not None:
            try:
                conv_sec = float(conv_sec)
            except Exception:
                conv_sec = None

        model_size_bytes = None
        if "model_size_bytes" in data:
            try:
                model_size_bytes = float(data["model_size_bytes"])
            except Exception:
                model_size_bytes = None
        if model_size_bytes is None and "model_payload_bytes" in data:
            try:
                model_size_bytes = float(data["model_payload_bytes"])
            except Exception:
                model_size_bytes = None
        if model_size_bytes is None:
            server_log = child / "server_logs.txt"
            if server_log.is_file():
                model_size_bytes = _parse_model_size_bytes_from_server_logs(server_log)

        # If metadata says unified / protocol override, keep folder-derived protocol but label includes unified.
        unified_flag = bool(meta.get("unified", False) or meta.get("protocol", "").lower() == "unified" or protocol == "unified")
        base_label = f"{protocol.upper()} ({scenario})"
        label = f"{base_label} + UNIFIED" if (unified_flag and protocol != "unified") else base_label

        runs.append(
            RunMetrics(
                protocol=protocol.upper(),
                scenario=scenario,
                label=label,
                rounds=rounds,
                loss=loss,
                accuracy=accuracy,
                battery_consumption=battery,
                total_rounds=int(total_rounds) if isinstance(total_rounds, (int, float)) else (len(rounds) if rounds else None),
                convergence_time_seconds=conv_sec,
                model_size_bytes=model_size_bytes,
                source_dir=child,
            )
        )

    return runs


def _ensure_matplotlib():
    import matplotlib

    # Safe default for scripts called from GUI; let GUI set backend if needed.
    if os.environ.get("MPLBACKEND"):
        return
    matplotlib.use("Qt5Agg")


def plot_combined(
    experiment_folder: Path,
    *,
    single_figure: bool = True,
    show: bool = True,
    save: bool = True,
) -> Dict[str, Path]:
    """
    Create combined plots for the given experiment folder.
    Returns dict of generated file paths (may be empty if save=False).
    """
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    out_files: Dict[str, Path] = {}
    runs = load_runs(experiment_folder)
    if not runs:
        raise FileNotFoundError(f"No protocol scenario runs found under: {experiment_folder}")

    # Sort: protocol name, scenario
    runs.sort(key=lambda r: (r.protocol, r.scenario, r.label))

    labels = [r.label for r in runs]
    x = list(range(len(runs)))

    fl_rounds = [r.total_rounds if r.total_rounds is not None else 0 for r in runs]
    conv_secs = [r.convergence_time_seconds if r.convergence_time_seconds is not None else 0.0 for r in runs]
    model_sizes = [r.model_size_bytes for r in runs]
    have_model_sizes = any(v is not None for v in model_sizes)

    # --- Required separate bar plots ---
    fig_r, ax_r = plt.subplots(figsize=(max(8, len(runs) * 1.2), 4.5))
    ax_r.bar(x, fl_rounds, color="#4C78A8", alpha=0.9)
    ax_r.set_title("FL Rounds (Total rounds to convergence/end)")
    ax_r.set_ylabel("FL rounds")
    ax_r.set_xticks(x)
    ax_r.set_xticklabels(labels, rotation=25, ha="right")
    ax_r.grid(True, axis="y", alpha=0.25)
    fig_r.tight_layout()
    if save:
        p = experiment_folder / "combined_fl_rounds_bar.png"
        fig_r.savefig(p, dpi=300, bbox_inches="tight")
        out_files["fl_rounds_bar"] = p
    if not show:
        plt.close(fig_r)

    fig_c, ax_c = plt.subplots(figsize=(max(8, len(runs) * 1.2), 4.5))
    ax_c.bar(x, conv_secs, color="#F58518", alpha=0.9)
    ax_c.set_title("Convergence Time")
    ax_c.set_ylabel("Time (seconds)")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels, rotation=25, ha="right")
    ax_c.grid(True, axis="y", alpha=0.25)
    fig_c.tight_layout()
    if save:
        p = experiment_folder / "combined_convergence_time_bar.png"
        fig_c.savefig(p, dpi=300, bbox_inches="tight")
        out_files["convergence_time_bar"] = p
    if not show:
        plt.close(fig_c)

    if have_model_sizes:
        fig_m, ax_m = plt.subplots(figsize=(max(8, len(runs) * 1.2), 4.5))
        sizes_mb = [(v / (1024 * 1024)) if v is not None else 0.0 for v in model_sizes]
        ax_m.bar(x, sizes_mb, color="#54A24B", alpha=0.9)
        ax_m.set_title("Model Size (estimated payload)")
        ax_m.set_ylabel("Size (MB)")
        ax_m.set_xticks(x)
        ax_m.set_xticklabels(labels, rotation=25, ha="right")
        ax_m.grid(True, axis="y", alpha=0.25)
        fig_m.tight_layout()
        if save:
            p = experiment_folder / "combined_model_size_bar.png"
            fig_m.savefig(p, dpi=300, bbox_inches="tight")
            out_files["model_size_bar"] = p
        if not show:
            plt.close(fig_m)

    # --- Optional: all-in-one figure ---
    if single_figure:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1.1, 1.0])

        ax_bat = fig.add_subplot(gs[0, 0])
        ax_acc = fig.add_subplot(gs[0, 1])
        ax_loss = fig.add_subplot(gs[1, 0])
        ax_size = fig.add_subplot(gs[1, 1])
        ax_rounds = fig.add_subplot(gs[2, 0])
        ax_conv = fig.add_subplot(gs[2, 1])

        # battery (use final consumption per round if available)
        any_battery = any(r.battery_consumption for r in runs)
        if any_battery:
            for r in runs:
                if not r.rounds or not r.battery_consumption:
                    continue
                n = min(len(r.rounds), len(r.battery_consumption))
                ax_bat.plot(r.rounds[:n], [v * 100 for v in r.battery_consumption[:n]], marker="o", linewidth=2, label=r.label)
            ax_bat.set_title("Battery consumption over rounds")
            ax_bat.set_xlabel("Round")
            ax_bat.set_ylabel("Battery consumption (%)")
            ax_bat.grid(True, alpha=0.25)
            ax_bat.legend(fontsize=8)
        else:
            ax_bat.set_title("Battery consumption over rounds (not available)")
            ax_bat.axis("off")

        # accuracy
        for r in runs:
            if not r.rounds or not r.accuracy:
                continue
            n = min(len(r.rounds), len(r.accuracy))
            ax_acc.plot(r.rounds[:n], [v * 100 for v in r.accuracy[:n]], marker="s", linewidth=2, label=r.label)
        ax_acc.set_title("Accuracy over rounds")
        ax_acc.set_xlabel("Round")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.grid(True, alpha=0.25)
        ax_acc.legend(fontsize=8)

        # loss
        for r in runs:
            if not r.rounds or not r.loss:
                continue
            n = min(len(r.rounds), len(r.loss))
            ax_loss.plot(r.rounds[:n], r.loss[:n], marker="o", linewidth=2, label=r.label)
        ax_loss.set_title("Loss over rounds")
        ax_loss.set_xlabel("Round")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(True, alpha=0.25)
        ax_loss.legend(fontsize=8)

        # model size (bar)
        if have_model_sizes:
            sizes_mb = [(v / (1024 * 1024)) if v is not None else 0.0 for v in model_sizes]
            ax_size.bar(x, sizes_mb, color="#54A24B", alpha=0.9)
            ax_size.set_title("Model size (MB, estimated)")
            ax_size.set_ylabel("Size (MB)")
            ax_size.set_xticks(x)
            ax_size.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax_size.grid(True, axis="y", alpha=0.25)
        else:
            ax_size.set_title("Model size (not available)")
            ax_size.axis("off")

        # rounds bar
        ax_rounds.bar(x, fl_rounds, color="#4C78A8", alpha=0.9)
        ax_rounds.set_title("FL rounds")
        ax_rounds.set_ylabel("Rounds")
        ax_rounds.set_xticks(x)
        ax_rounds.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax_rounds.grid(True, axis="y", alpha=0.25)

        # convergence time bar
        ax_conv.bar(x, conv_secs, color="#F58518", alpha=0.9)
        ax_conv.set_title("Convergence time")
        ax_conv.set_ylabel("Seconds")
        ax_conv.set_xticks(x)
        ax_conv.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax_conv.grid(True, axis="y", alpha=0.25)

        fig.suptitle(f"Combined Results: {experiment_folder.name}", fontsize=14, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        if save:
            p = experiment_folder / "combined_metrics_grid.png"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            out_files["combined_grid"] = p
        if not show:
            plt.close(fig)

    if show:
        plt.show(block=False)

    return out_files


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def list_experiment_folders() -> List[Path]:
    """Return experiment_results/* folders sorted (newest first)."""
    root = _project_root() / "experiment_results"
    if not root.is_dir():
        return []
    folders = [p for p in root.iterdir() if p.is_dir()]
    # Sort by mtime desc, then name desc
    folders.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return folders


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine plots for a timestamped experiment folder.")
    parser.add_argument("--experiment-folder", required=True, help="Folder name under experiment_results/, e.g. emotion_pruned_5pct_20260316_110414")
    parser.add_argument("--single-figure", action="store_true", help="Also generate a combined grid figure with all plots")
    parser.add_argument("--no-show", action="store_true", help="Do not open plot windows; only save PNGs")
    args = parser.parse_args()

    exp = _project_root() / "experiment_results" / args.experiment_folder
    files = plot_combined(exp, single_figure=args.single_figure, show=(not args.no_show), save=True)
    if files:
        print("Generated:")
        for k, v in files.items():
            print(f"  - {k}: {v}")
