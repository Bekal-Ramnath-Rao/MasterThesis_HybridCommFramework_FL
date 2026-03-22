"""
Experiment results directory path: experiment_results/{use_case}/{protocol}/{network_scenario}/
Used by FL servers to save plots and JSON/CSV under a consistent folder structure.
"""
import os
from pathlib import Path


def get_project_root():
    """Project root: /app in Docker, else parent of Server/."""
    if os.path.exists("/app"):
        return Path("/app")
    # Assume called from Server/<UseCase>/FL_Server_*.py -> go up to repo root
    return Path(__file__).resolve().parent.parent.parent


def get_experiment_results_dir(use_case: str, protocol: str, scenario: str = None) -> Path:
    """
    Return experiment_results/{use_case}/{protocol}/{network_scenario}/.
    Creates the directory if needed.

    use_case: "emotion" | "mental_state" | "temperature"
    protocol: "mqtt" | "amqp" | "grpc" | "dds" | "quic" | "http3" | "unified"
    scenario: network scenario (e.g. excellent, good, moderate, poor, very_poor).
             If None, uses env NETWORK_SCENARIO or "default".
    """
    if scenario is None:
        scenario = os.getenv("NETWORK_SCENARIO", "default").strip() or "default"
    root = get_project_root()
    path = root / "experiment_results" / use_case / protocol / scenario
    path.mkdir(parents=True, exist_ok=True)
    return path
