"""
Experiment results directory:

 {base}/{use_case}/{protocol}/{network_scenario}/

where ``base`` is (first match wins):

- ``EXPERIMENT_RESULTS_ROOT`` if set (absolute path to the parent of the use_case folders), or
- In Docker: ``/app/results`` if that directory exists (compose bind-mounts host ``experiment_results`` here), else ``/app/experiment_results`` if that directory exists (distributed GUI mount), or
- ``{project_root}/experiment_results`` otherwise (project_root is /app in Docker, else repo root).

Used by FL servers and client-side artifact writers (plots, JSON) under one tree.

Environment:
  EXPERIMENT_RESULTS_ROOT — optional; overrides the default ``.../experiment_results`` base.
  NETWORK_SCENARIO — subfolder name (default ``default``) when scenario is not passed explicitly.
"""
import os
from pathlib import Path


def get_project_root():
    """Project root: /app in Docker, else parent of Server/."""
    if os.path.exists("/app"):
        return Path("/app")
    # Assume called from Server/<UseCase>/FL_Server_*.py -> go up to repo root
    return Path(__file__).resolve().parent.parent.parent


def get_experiment_results_base() -> Path:
    """Parent directory containing ``emotion/``, ``mental_state/``, ``temperature/``, etc."""
    override = os.getenv("EXPERIMENT_RESULTS_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    # Docker compose (e.g. Docker/docker-compose-unified-emotion.yml) mounts host experiment_results → /app/results.
    # Distributed client GUI mounts the same host folder → /app/experiment_results.
    if os.path.exists("/app"):
        results_legacy = Path("/app/results")
        if results_legacy.is_dir():
            return results_legacy.resolve()
        results_alt = Path("/app/experiment_results")
        if results_alt.is_dir():
            return results_alt.resolve()
    return get_project_root() / "experiment_results"


def get_experiment_results_dir(use_case: str, protocol: str, scenario: str = None) -> Path:
    """
    Return {base}/{use_case}/{protocol}/{network_scenario}/.
    Creates the directory if needed.

    use_case: "emotion" | "mental_state" | "temperature"
    protocol: "mqtt" | "amqp" | "grpc" | "dds" | "quic" | "http3" | "unified"
    scenario: network scenario (e.g. excellent, good, moderate, poor, very_poor).
             If None, uses env NETWORK_SCENARIO or "default".
    """
    if scenario is None:
        scenario = os.getenv("NETWORK_SCENARIO", "default").strip() or "default"
    path = get_experiment_results_base() / use_case / protocol / scenario
    path.mkdir(parents=True, exist_ok=True)
    return path
