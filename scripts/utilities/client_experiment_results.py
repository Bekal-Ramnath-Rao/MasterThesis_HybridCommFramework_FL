"""
Per-client training summary (JSON + plots) under the same tree as server results:

  {EXPERIMENT_RESULTS_ROOT or <project>/experiment_results}/{use_case}/{protocol}/{NETWORK_SCENARIO}/

Example (Docker compose): host ``experiment_results/`` is usually mounted at ``/app/results``, so paths look like
``/app/results/emotion/mqtt/default/``. The distributed GUI may mount the same folder at ``/app/experiment_results``.

Filenames are prefixed with ``client{client_id}_`` so they do not overwrite
``mqtt_training_results.json`` and plots from the server.

Environment:
  CLIENT_EXPERIMENT_ARTIFACTS — set to 0/false/no to disable saving
  EXPERIMENT_RESULTS_ROOT — optional; see ``experiment_results_path.get_experiment_results_base``
  NETWORK_SCENARIO — scenario subfolder (default ``default``)
  CLIENT_EXPERIMENT_CHECKPOINT_EACH_ROUND — if true, save after every round; if false, never; if unset,
    true when NODE_TYPE=client (distributed GUI) so artifacts exist without training_complete.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt

from experiment_results_path import get_experiment_results_dir


def _artifacts_enabled() -> bool:
    v = os.environ.get("CLIENT_EXPERIMENT_ARTIFACTS", "true").strip().lower()
    return v not in ("0", "false", "no")


def checkpoint_client_artifacts_each_round_enabled() -> bool:
    """Save experiment_results JSON/plots after each eval round (for remote clients without training_complete)."""
    if not _artifacts_enabled():
        return False
    v = os.environ.get("CLIENT_EXPERIMENT_CHECKPOINT_EACH_ROUND", "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    if v in ("1", "true", "yes"):
        return True
    return os.environ.get("NODE_TYPE", "").strip().lower() == "client"


def maybe_checkpoint_client_training_artifacts(
    client_id: int,
    *,
    use_case: str,
    protocol: str,
    round_history: List[Dict[str, Any]],
    total_elapsed_sec: Optional[float] = None,
) -> None:
    if not checkpoint_client_artifacts_each_round_enabled():
        return
    if not round_history:
        return
    save_client_training_artifacts(
        client_id,
        use_case=use_case,
        protocol=protocol,
        round_history=list(round_history),
        total_elapsed_sec=total_elapsed_sec,
        quiet=True,
    )


def _sanitize_token(name: str) -> str:
    s = (name or "mqtt").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    return s or "mqtt"


def save_client_training_artifacts(
    client_id: int,
    *,
    use_case: str,
    protocol: str,
    round_history: List[Dict[str, Any]],
    total_elapsed_sec: Optional[float] = None,
    quiet: bool = False,
) -> Optional[Path]:
    """
    Build JSON + three plots from per-round history (one entry per completed eval round).

    Each history row should include: round, loss, accuracy, and either
    ``total_fl_wall_time_sec`` or ``round_time_sec``, and optionally
    ``battery_soc_after`` or ``battery_soc``.
    """
    if not _artifacts_enabled():
        return None
    if not round_history:
        if not quiet:
            print(f"[Client {client_id}] No round history; skipping client experiment artifacts")
        return None

    uc = _sanitize_token(use_case)
    proto = _sanitize_token(protocol)

    rows = sorted(round_history, key=lambda r: int(r.get("round", 0)))
    rounds: List[int] = []
    losses: List[float] = []
    accuracies: List[float] = []
    round_times: List[float] = []
    battery_cons: List[float] = []

    for r in rows:
        try:
            rv = int(r["round"])
        except (KeyError, TypeError, ValueError):
            continue
        rounds.append(rv)
        try:
            losses.append(float(r.get("loss", 0.0)))
        except (TypeError, ValueError):
            losses.append(0.0)
        try:
            accuracies.append(float(r.get("accuracy", 0.0)))
        except (TypeError, ValueError):
            accuracies.append(0.0)
        rt = r.get("total_fl_wall_time_sec")
        if rt is None:
            rt = r.get("round_time_sec")
        try:
            round_times.append(float(rt) if rt is not None else 0.0)
        except (TypeError, ValueError):
            round_times.append(0.0)
        soc = r.get("battery_soc_after")
        if soc is None:
            soc = r.get("battery_soc")
        try:
            soc_f = float(soc) if soc is not None else 1.0
        except (TypeError, ValueError):
            soc_f = 1.0
        battery_cons.append(max(0.0, min(1.0, 1.0 - soc_f)))

    if not rounds:
        if not quiet:
            print(f"[Client {client_id}] Round history had no valid rows; skipping artifacts")
        return None

    results_dir = get_experiment_results_dir(uc, proto)
    prefix = f"client{int(client_id)}_{proto}"

    results = {
        "client_id": int(client_id),
        "use_case": uc,
        "protocol": proto,
        "rounds": rounds,
        "loss": losses,
        "accuracy": accuracies,
        "round_times_seconds": round_times,
        "battery_consumption": battery_cons,
        "total_elapsed_seconds": total_elapsed_sec,
        "total_rounds": len(rounds),
        "final_accuracy": accuracies[-1] if accuracies else None,
        "final_loss": losses[-1] if losses else None,
    }
    json_path = results_dir / f"{prefix}_training_results.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if not quiet:
            print(f"[Client {client_id}] Client results saved to {json_path}")
    except Exception as e:
        print(f"[Client {client_id}] WARNING: could not write client results JSON: {e}")
        json_path = None

    n = len(rounds)
    conv_time = float(total_elapsed_sec) if total_elapsed_sec is not None else float(sum(round_times))

    try:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        bc = battery_cons if len(battery_cons) == n else (battery_cons + [0.0] * (n - len(battery_cons)))[:n]
        if bc:
            ax1.plot(rounds, [c * 100 for c in bc], marker="o", linewidth=2, markersize=6, color="#2e86ab")
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("Battery consumption (1 − SoC, %)", fontsize=12)
        ax1.set_title("Client battery drain over FL rounds", fontsize=14)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        p1 = results_dir / f"{prefix}_battery_consumption.png"
        fig1.savefig(p1, dpi=300, bbox_inches="tight")
        plt.close(fig1)
        if not quiet:
            print(f"[Client {client_id}] Battery plot saved to {p1}")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        rt = round_times if len(round_times) == n else (round_times + [0.0] * (n - len(round_times)))[:n]
        if rt:
            ax2.bar(rounds, rt, color="#a23b72", alpha=0.8, label="Time per round (s)")
        ax2.axhline(
            y=conv_time,
            color="#f18f01",
            linestyle="--",
            linewidth=2,
            label=f"Total elapsed: {conv_time:.1f} s",
        )
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Time (s)", fontsize=12)
        ax2.set_title("Client time per round and total elapsed", fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        p2 = results_dir / f"{prefix}_round_and_elapsed_time.png"
        fig2.savefig(p2, dpi=300, bbox_inches="tight")
        plt.close(fig2)
        if not quiet:
            print(f"[Client {client_id}] Time plot saved to {p2}")

        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
        ax3a.plot(rounds, losses, marker="o", linewidth=2, markersize=8, color="red")
        ax3a.set_xlabel("Round", fontsize=12)
        ax3a.set_ylabel("Loss (eval)", fontsize=12)
        ax3a.set_title("Client loss over rounds", fontsize=14)
        ax3a.grid(True, alpha=0.3)
        ax3b.plot(rounds, [acc * 100 for acc in accuracies], marker="s", linewidth=2, markersize=8, color="green")
        ax3b.set_xlabel("Round", fontsize=12)
        ax3b.set_ylabel("Accuracy (%)", fontsize=12)
        ax3b.set_title("Client accuracy over rounds", fontsize=14)
        ax3b.grid(True, alpha=0.3)
        fig3.tight_layout()
        p3 = results_dir / f"{prefix}_training_metrics.png"
        fig3.savefig(p3, dpi=300, bbox_inches="tight")
        plt.close(fig3)
        if not quiet:
            print(f"[Client {client_id}] Metrics plot saved to {p3}")
    except Exception as e:
        print(f"[Client {client_id}] WARNING: could not save client plots: {e}")

    if quiet and json_path:
        print(
            f"[Client {client_id}] Checkpoint: client artifacts updated "
            f"({len(rounds)} rounds) under {results_dir}"
        )

    return results_dir if json_path else None
