"""FL termination flags — shared by all Emotion client entrypoints (Docker + native)."""
import os


def stop_on_client_convergence() -> bool:
    """
    If False, clients keep participating until the server finishes NUM_ROUNDS (user-defined rounds).
    TRAINING_TERMINATION_MODE (fixed_rounds | client_convergence) is set by experiment runners and wins
    over STOP_ON_CLIENT_CONVERGENCE when present.
    """
    mode = (os.getenv("TRAINING_TERMINATION_MODE") or "").strip().lower()
    if mode == "fixed_rounds":
        return False
    if mode == "client_convergence":
        return True
    v = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").strip().lower()
    return v in ("1", "true", "yes")
