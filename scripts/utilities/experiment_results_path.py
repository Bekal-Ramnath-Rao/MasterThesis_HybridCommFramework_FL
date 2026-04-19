"""
Experiment results directory layout (default, new):

 {base}/{run_segment}/{protocol}/{network_scenario}/

``run_segment`` is (first match):

- ``EXPERIMENT_RESULTS_RUN_SEGMENT`` if set (e.g. ``emotion_20260418_183419`` from the network runner
  so host-collected paths match in-container writes), else
- ``{use_case}_{EXPERIMENT_SESSION_ID}`` if ``EXPERIMENT_SESSION_ID`` is set, else
- ``{use_case}_{YYYYMMDD_HHMMSS}`` frozen on first call in this process.

Legacy layout (no timestamp segment):

 Set ``EXPERIMENT_LEGACY_RESULTS_LAYOUT=1`` → ``{base}/{use_case}/{protocol}/{network_scenario}/``

``base`` is (first match):

- ``EXPERIMENT_RESULTS_ROOT`` if set (absolute path to the parent of the run folders), or
- In Docker: ``/app/results`` if that directory exists (compose bind-mounts host ``experiment_results`` here), else ``/app/experiment_results`` if that directory exists (distributed GUI mount), or
- ``{project_root}/experiment_results`` otherwise (project_root is /app in Docker, else repo root).

Environment:
  EXPERIMENT_RESULTS_ROOT — optional; overrides the default ``.../experiment_results`` base.
  NETWORK_SCENARIO — subfolder name (default ``default``) when scenario is not passed explicitly.
  EXPERIMENT_RESULTS_RUN_SEGMENT — optional; full first path segment under ``base`` (use_case + timestamp etc.).
  EXPERIMENT_SESSION_ID — optional; timestamp token combined with use_case when RUN_SEGMENT unset.
  EXPERIMENT_LEGACY_RESULTS_LAYOUT — set to 1/true for old ``use_case/protocol/scenario`` tree.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

_SESSION_TS: Optional[str] = None


def get_project_root():
    """Project root: /app in Docker, else parent of Server/."""
    if os.path.exists("/app"):
        return Path("/app")
    # Assume called from Server/<UseCase>/FL_Server_*.py -> go up to repo root
    return Path(__file__).resolve().parent.parent.parent


def get_experiment_results_base() -> Path:
    """Parent directory containing run folders (timestamp+use_case, or legacy use_case)."""
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


def _frozen_session_ts() -> str:
    global _SESSION_TS
    if _SESSION_TS is None:
        env = os.getenv("EXPERIMENT_SESSION_ID", "").strip()
        _SESSION_TS = env if env else datetime.now().strftime("%Y%m%d_%H%M%S")
    return _SESSION_TS


def _results_run_segment(use_case: str) -> str:
    if os.getenv("EXPERIMENT_LEGACY_RESULTS_LAYOUT", "").strip().lower() in ("1", "true", "yes"):
        return (use_case or "emotion").strip().lower()
    seg = os.getenv("EXPERIMENT_RESULTS_RUN_SEGMENT", "").strip()
    if seg:
        return seg
    uc = (use_case or "emotion").strip().lower()
    return f"{uc}_{_frozen_session_ts()}"


def get_experiment_results_dir(use_case: str, protocol: str, scenario: str = None) -> Path:
    """
    Return the directory for JSON/plots. Creates the directory if needed.

    use_case: "emotion" | "mental_state" | "temperature"
    protocol: "mqtt" | "amqp" | "grpc" | "dds" | "quic" | "http3" | "unified"
    scenario: network scenario (e.g. excellent, good). If None, uses env NETWORK_SCENARIO or "default".
    """
    if scenario is None:
        scenario = os.getenv("NETWORK_SCENARIO", "default").strip() or "default"
    base = get_experiment_results_base()
    run_seg = _results_run_segment(use_case)
    path = base / run_seg / protocol / scenario
    path.mkdir(parents=True, exist_ok=True)
    return path
