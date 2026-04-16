#!/usr/bin/env python3
"""
Build one **downlink** Q-table (3 × 12 × 5 per scenario in flat view) from archived training pickles.

``shared_data/q_table_emotion_downlink_trained.pkl`` may be all zeros after a fresh reset; archives
``q_table_emotion_downlink_trained_archive_*.pkl`` retain partial Q-learning updates.
For each coarse scenario slice (excellent / moderate / poor), the archive with the largest max|Q|
on that slice is chosen and merged into a single (3, 3, 2, 2, 5) array.

Outputs (default: this directory):

- ``q_table_emotion_downlink_MERGED_from_training.md``
- ``q_table_emotion_downlink_merged_from_training_archives.pkl``
- ``q_table_emotion_downlink_merged_{excellent,moderate,poor}.csv``

If archives are not readable, fix permissions on ``shared_data/*.pkl`` or run with ``sudo``.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np

SCENARIOS = ["excellent", "moderate", "poor"]
PROTOCOLS = ["mqtt", "amqp", "grpc", "http3", "dds"]
COMM = ["low", "mid", "high"]
RES = ["high", "low"]
BAT = ["high", "low"]


def _load(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    with open(path, "rb") as f:
        d = pickle.load(f)
    return np.asarray(d["q_table"], dtype=float), d


def _best_archive_per_slice(
    archive_paths: List[str],
    script_name: str,
) -> Tuple[List[str], np.ndarray]:
    """Return path with max |Q| per scenario index 0,1,2."""
    best_paths = ["" for _ in range(3)]
    best_vals = [-1.0, -1.0, -1.0]
    merged = np.zeros((3, 3, 2, 2, 5), dtype=float)
    loaded = 0
    for p in archive_paths:
        try:
            q, _ = _load(p)
        except (OSError, PermissionError) as e:
            print(f"[skip unreadable] {p}: {e}")
            continue
        loaded += 1
        for si in range(3):
            m = float(np.max(np.abs(q[si])))
            if m > best_vals[si]:
                best_vals[si] = m
                best_paths[si] = p
                merged[si] = q[si]
    if loaded == 0:
        raise SystemExit(
            "No archive pickle could be read (permission denied?). "
            "Fix permissions on shared_data/*.pkl or run: "
            f"sudo python3 scripts/utilities/{script_name}"
        )
    return best_paths, merged


def _state_label(ci: int, ri: int, bi: int) -> str:
    return f"{COMM[ci]}/{RES[ri]}/{BAT[bi]}"


def export_markdown(
    merged: np.ndarray,
    best_paths: List[str],
    source_maps: Dict[str, Any],
    out_md: str,
    out_pkl: str,
    csv_prefix: str,
) -> None:
    lines: List[str] = [
        "# Emotion downlink Q-table — merged from training archives\n\n",
        "Each block is **12 states × 5 actions** (comm × resource × battery × protocol). "
        "Values come from Q-learning; **(s,a) pairs that were never updated stay 0**.\n\n",
        "## Source per scenario slice (auto-picked: max |Q| on that slice)\n\n",
        "| Scenario | Archive |\n|---|---|\n",
    ]
    for si, sn in enumerate(SCENARIOS):
        lines.append(f"| {sn} | `{best_paths[si]}` |\n")
    lines.append(f"\n**Pickle:** `{out_pkl}`\n\n")
    lines.append(f"**converged_protocol_by_scenario (from sources):** `{source_maps}`\n\n")

    for si, sn in enumerate(SCENARIOS):
        lines.append(f"## {sn}\n\n")
        lines.append("| row | state | " + " | ".join(PROTOCOLS) + " |\n")
        lines.append("|---:|---|" + "|".join(["---:" for _ in PROTOCOLS]) + "|\n")
        row_id = 0
        block: List[np.ndarray] = []
        for ci in range(3):
            for ri in range(2):
                for bi in range(2):
                    vec = merged[si, ci, ri, bi, :]
                    lines.append(
                        f"| {row_id} | {_state_label(ci, ri, bi)} | "
                        + " | ".join(f"{float(x):.6f}" for x in vec)
                        + " |\n"
                    )
                    block.append(vec)
                    row_id += 1
        arr12 = np.stack(block, axis=0)
        csv_path = f"{csv_prefix}_{sn}.csv"
        hdr = "row," + ",".join(PROTOCOLS)
        np.savetxt(
            csv_path,
            np.column_stack([np.arange(12), arr12]),
            delimiter=",",
            fmt=["%d"] + ["%.8f"] * 5,
            header=hdr,
            comments="",
        )

    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(lines)

    payload = {
        "q_table": merged,
        "network_scenario_levels": list(SCENARIOS),
        "merge_best_paths": best_paths,
        "source_converged_maps": source_maps,
        "link_direction": "downlink",
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)


def main() -> None:
    script_name = os.path.basename(__file__)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--shared-data",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "shared_data",
        ),
        help="Directory containing q_table_emotion_downlink_trained_archive_*.pkl",
    )
    ap.add_argument(
        "--out-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Output directory for .md, .csv, merged .pkl",
    )
    args = ap.parse_args()
    pattern = os.path.join(args.shared_data, "q_table_emotion_downlink_trained_archive_*.pkl")
    archives = sorted(glob.glob(pattern))
    if not archives:
        raise SystemExit(f"No archives matching {pattern!r}")

    best_paths, merged = _best_archive_per_slice(archives, script_name)
    source_maps = {}
    for p in set(best_paths):
        if not p:
            continue
        _, d = _load(p)
        source_maps[os.path.basename(p)] = d.get("converged_protocol_by_scenario")

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_md = os.path.join(out_dir, "q_table_emotion_downlink_MERGED_from_training.md")
    out_pkl = os.path.join(out_dir, "q_table_emotion_downlink_merged_from_training_archives.pkl")
    csv_prefix = os.path.join(out_dir, "q_table_emotion_downlink_merged")
    export_markdown(merged, best_paths, source_maps, out_md, out_pkl, csv_prefix)
    print("Wrote:", out_md)
    print("Wrote:", out_pkl)
    print("CSVs:", csv_prefix + "_*.csv")
    print("Slices from:", best_paths)


if __name__ == "__main__":
    main()
