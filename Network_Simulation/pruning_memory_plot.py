#!/usr/bin/env python3
"""
Plot model memory (and sparsity) per round when pruning is enabled.
Parses server logs or pruning_metrics.json and generates a graph at experiment end.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional matplotlib for plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _parse_pruning_from_log(log_path: Path) -> List[Dict[str, Any]]:
    """
    Parse server log for [Server Pruning] Round N and following stats.
    Returns list of dicts: round, sparsity (0-1), non_zero_params, total_params, model_size_kb.
    """
    if not log_path.exists():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    metrics = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Match: [Server Pruning] Round 10  or  Round 10  (with optional suffix)
        m = re.search(r"\[?Server Pruning\]?\s*Round\s+(\d+)", line, re.IGNORECASE)
        if not m:
            i += 1
            continue
        round_num = int(m.group(1))
        sparsity = None
        non_zero_params = None
        total_params = None
        original_kb = None
        compressed_kb = None
        # Next few lines: Overall Sparsity, Compression Ratio, Non-zero Params, (optional) Original/Compressed
        for j in range(i + 1, min(i + 12, len(lines))):
            l = lines[j]
            if re.match(r"^\s*Overall Sparsity:\s*([\d.]+)%", l):
                sp = re.match(r"^\s*Overall Sparsity:\s*([\d.]+)%", l)
                if sp:
                    sparsity = float(sp.group(1)) / 100.0
            if "Non-zero Params:" in l or "Non-zero params:" in l:
                np_match = re.search(r"Non-zero Params?:\s*([\d,]+)\s*/\s*([\d,]+)", l, re.IGNORECASE)
                if np_match:
                    non_zero_params = int(np_match.group(1).replace(",", ""))
                    total_params = int(np_match.group(2).replace(",", ""))
            if "Original:" in l and "KB" in l:
                orig = re.search(r"Original:\s*([\d.]+)\s*KB", l)
                if orig:
                    original_kb = float(orig.group(1))
            if "Compressed:" in l and "KB" in l:
                comp = re.search(r"Compressed:\s*([\d.]+)\s*KB", l)
                if comp:
                    compressed_kb = float(comp.group(1))
            # Stop at next [Server Pruning] or next section
            if j > i + 1 and re.search(r"\[?Server Pruning\]?\s*Round", l, re.IGNORECASE):
                break
        # Model memory: prefer compressed size (effective pruned size), else non_zero_params * 4 bytes
        if compressed_kb is not None:
            model_size_kb = compressed_kb
        elif original_kb is not None and sparsity is not None:
            model_size_kb = original_kb * (1.0 - sparsity)  # rough pruned size
        elif non_zero_params is not None:
            model_size_kb = (non_zero_params * 4) / 1024.0  # float32
        else:
            model_size_kb = None
        if sparsity is not None or non_zero_params is not None or model_size_kb is not None:
            entry = {"round": round_num}
            if sparsity is not None:
                entry["sparsity"] = sparsity
            if non_zero_params is not None:
                entry["non_zero_params"] = non_zero_params
            if total_params is not None:
                entry["total_params"] = total_params
            if model_size_kb is not None:
                entry["model_size_kb"] = model_size_kb
            metrics.append(entry)
        i += 1
    return metrics


def _parse_pruning_from_json(json_path: Path) -> List[Dict[str, Any]]:
    """Load pruning_metrics.json (from PruningMetricsLogger.save_metrics())."""
    if not json_path.exists():
        return []
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for entry in data:
        r = entry.get("round")
        if r is None:
            continue
        sparsity = entry.get("sparsity")
        non_zero = entry.get("non_zero_params")
        total = entry.get("total_params")
        # model_size_kb: from compressed_size_mb or non_zero * 4 bytes
        model_size_kb = None
        if "compressed_size_mb" in entry:
            model_size_kb = entry["compressed_size_mb"] * 1024.0
        elif non_zero is not None:
            model_size_kb = (non_zero * 4) / 1024.0
        out.append({
            "round": r,
            "sparsity": sparsity,
            "non_zero_params": non_zero,
            "total_params": total,
            "model_size_kb": model_size_kb,
        })
    return out


def collect_pruning_metrics(exp_dir: Path, server_log_name: str = "server_logs.txt") -> List[Dict[str, Any]]:
    """
    Collect per-round pruning metrics from an experiment directory.
    Tries pruning_metrics.json first, then parses server_logs.txt.
    exp_dir can be a path to a single protocol_scenario folder (Docker) or to a dir containing server.log (native).
    """
    exp_dir = Path(exp_dir)
    # Try JSON first (written by server when using PruningMetricsLogger)
    json_path = exp_dir / "pruning_metrics.json"
    metrics = _parse_pruning_from_json(json_path)
    if metrics:
        return sorted(metrics, key=lambda x: x["round"])
    # Parse server log (Docker: server_logs.txt; native: server.log)
    for log_name in (server_log_name, "server.log"):
        log_path = exp_dir / log_name
        if log_path.exists():
            metrics = _parse_pruning_from_log(log_path)
            if metrics:
                return sorted(metrics, key=lambda x: x["round"])
    return []


def plot_pruning_memory(
    metrics: List[Dict[str, Any]],
    output_path: Path,
    title: Optional[str] = "Model memory vs round (pruning enabled)",
) -> bool:
    """
    Plot model memory (KB) and optionally sparsity (%) vs round.
    Saves figure to output_path. Returns True if plot was written.
    """
    if not _HAS_MATPLOTLIB or not metrics:
        return False
    rounds = [m["round"] for m in metrics]
    has_size = any(m.get("model_size_kb") is not None for m in metrics)
    has_sparsity = any(m.get("sparsity") is not None for m in metrics)
    if not has_size and not has_sparsity:
        return False

    fig, ax1 = plt.subplots(figsize=(10, 5))
    if has_size:
        size_kb = [m.get("model_size_kb") or 0 for m in metrics]
        ax1.plot(rounds, size_kb, "b-o", markersize=4, linewidth=1, label="Model memory (KB)")
        ax1.set_ylabel("Model memory (KB)", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.set_xlabel("Round")
        ax1.grid(True, alpha=0.3)
    if has_sparsity:
        ax2 = ax1.twinx()
        sparsity_pct = [(m.get("sparsity") or 0) * 100 for m in metrics]
        ax2.plot(rounds, sparsity_pct, "g-s", markersize=4, linewidth=1, label="Sparsity (%)")
        ax2.set_ylabel("Sparsity (%)", color="g")
        ax2.tick_params(axis="y", labelcolor="g")
    ax1.set_title(title or "Model memory vs round (pruning enabled)")
    fig.tight_layout()
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return True
    except Exception:
        plt.close(fig)
        return False


def plot_pruning_memory_from_experiment(
    exp_dir: Path,
    output_filename: str = "pruning_memory_by_round.png",
    server_log_name: str = "server_logs.txt",
) -> Optional[Path]:
    """
    Collect pruning metrics from exp_dir, plot if any, and return path to saved figure or None.
    Only generates plot when pruning data is present.
    """
    exp_dir = Path(exp_dir)
    metrics = collect_pruning_metrics(exp_dir, server_log_name=server_log_name)
    if not metrics:
        return None
    out_path = exp_dir / output_filename
    if plot_pruning_memory(metrics, out_path):
        return out_path
    return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pruning_memory_plot.py <exp_dir> [server_log_name]")
        sys.exit(1)
    exp_dir = Path(sys.argv[1])
    server_log_name = sys.argv[2] if len(sys.argv) > 2 else "server_logs.txt"
    path = plot_pruning_memory_from_experiment(exp_dir, server_log_name=server_log_name)
    if path:
        print(f"Plot saved: {path}")
    else:
        print("No pruning metrics found or plot failed.")
