"""Export Q-table pickle(s) to CSV for all network scenarios."""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

NETWORK = ["excellent", "moderate", "poor"]
COMM = ["low", "mid", "high"]
RESOURCE = ["high", "low"]
BATTERY = ["high", "low"]
PROTOCOLS = ["mqtt", "amqp", "grpc", "http3", "dds"]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "shared_data"
    for name in ("uplink", "downlink"):
        pkl = out_dir / f"q_table_emotion_{name}_trained.pkl"
        if not pkl.exists():
            print(f"Skip (missing): {pkl}", file=sys.stderr)
            continue
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        q = np.asarray(data["q_table"], dtype=float)
        if q.ndim != 5 or q.shape != (3, 3, 2, 2, 5):
            print(f"Unexpected shape {q.shape} in {pkl}", file=sys.stderr)
            continue
        out_csv = out_dir / f"q_table_emotion_{name}_trained_export.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(
                [
                    "network_scenario",
                    "comm_level",
                    "resource",
                    "battery_level",
                    *["Q_" + p for p in PROTOCOLS],
                ]
            )
            for si, ns in enumerate(NETWORK):
                for ci, cl in enumerate(COMM):
                    for ri, res in enumerate(RESOURCE):
                        for bi, batt in enumerate(BATTERY):
                            row = q[si, ci, ri, bi, :].tolist()
                            w.writerow([ns, cl, res, batt, *[f"{v:.6f}" for v in row]])
        meta_csv = out_dir / f"q_table_emotion_{name}_trained_meta.csv"
        with open(meta_csv, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            for k in (
                "episode_count",
                "epsilon",
                "comm_t_low",
                "comm_t_high",
                "resource_load_threshold",
                "battery_soc_threshold",
                "data_network_scenario",
                "detected_network_scenario",
            ):
                w.writerow([k, data.get(k, "")])
        print(f"Wrote {out_csv}")
        print(f"Wrote {meta_csv}")


if __name__ == "__main__":
    main()
