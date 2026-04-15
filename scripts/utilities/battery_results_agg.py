"""
Aggregate per-round battery model drain for training_results.json.

Matches Client/battery_model.BatteryModel cumulative drain:
``cumulative_energy_j / BATTERY_CAP_J``, or ``1 - battery_soc`` when energy is not sent.

A duplicate of this module lives at ``Server/Emotion_Recognition/battery_results_agg.py``
for Docker images that omit ``scripts/utilities/``; keep them identical.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

# Keep in sync with Client/battery_model.py
BATTERY_CAP_J = 60.0 * 3600.0


def avg_battery_model_drain_fraction(client_metrics: Mapping[Any, Dict[str, Any]]) -> float:
    """
    Average clients' cumulative energy drain fraction after this FL round.

    Prefers ``cumulative_energy_j`` from client metrics (BatteryModel), otherwise
    ``1 - battery_soc`` from the same BatteryModel state.
    """
    drifts: list[float] = []
    for m in client_metrics.values():
        if not isinstance(m, dict):
            continue
        cum = m.get("cumulative_energy_j")
        metrics = m.get("metrics")
        if cum is None and isinstance(metrics, dict):
            cum = metrics.get("cumulative_energy_j")
        if cum is not None:
            try:
                drifts.append(float(cum) / BATTERY_CAP_J)
            except (TypeError, ValueError):
                pass
    if drifts and any(x > 1e-18 for x in drifts):
        return sum(drifts) / len(drifts)

    socs: list[float] = []
    for m in client_metrics.values():
        if not isinstance(m, dict):
            continue
        soc = m.get("battery_soc")
        metrics = m.get("metrics")
        if soc is None and isinstance(metrics, dict):
            soc = metrics.get("battery_soc")
        if soc is not None:
            try:
                socs.append(float(soc))
            except (TypeError, ValueError):
                pass
    if socs:
        return 1.0 - (sum(socs) / len(socs))
    return 0.0
