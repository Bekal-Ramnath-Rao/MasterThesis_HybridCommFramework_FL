"""FL termination flags — shared by all MentalState server entrypoints (Docker + native)."""
import os


def stop_on_client_convergence() -> bool:
    """
    If False, server keeps training until NUM_ROUNDS (user-defined rounds).
    TRAINING_TERMINATION_MODE (fixed_rounds | client_convergence) is set by experiment runners and wins
    over STOP_ON_CLIENT_CONVERGENCE when present.
    
    AUTO-DETECTION: When USE_QL_CONVERGENCE=True (RL training mode), defaults to fixed_rounds
    to allow full RL pipeline (Phase 1: boundary collection → Phase 2: computation → Phase 3: Q-learning)
    """
    mode = (os.getenv("TRAINING_TERMINATION_MODE") or "").strip().lower()
    if mode == "fixed_rounds":
        return False
    if mode == "client_convergence":
        return True
    
    # Auto-detect RL mode: if Q-learning convergence is enabled, use fixed_rounds by default
    use_ql = os.getenv("USE_QL_CONVERGENCE", "").strip().lower() in ("1", "true", "yes")
    if use_ql:
        print("[Server] Auto-detected USE_QL_CONVERGENCE=True → using fixed_rounds mode (no early stopping)")
        return False
    
    v = os.getenv("STOP_ON_CLIENT_CONVERGENCE", "true").strip().lower()
    return v in ("1", "true", "yes")
