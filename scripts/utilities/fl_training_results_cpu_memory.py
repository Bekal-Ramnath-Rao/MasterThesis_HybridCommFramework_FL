"""
Merge per-round CPU / RAM (from client ``client_fl_metrics_*.jsonl``) into training results JSON
and plot average utilisation vs round.

Servers call ``merge_cpu_memory_into_results`` before writing ``{protocol}_training_results.json``,
and ``plot_cpu_memory_training_results`` from ``plot_results`` after metrics JSON exists.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional plotting (servers always have matplotlib)
try:
    import matplotlib

    matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def _sanitize_use_case_token(name: str) -> str:
    s = (name or "emotion").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    return s or "emotion"


def _jsonl_search_roots() -> List[Path]:
    roots: List[Path] = []
    for p in (
        os.environ.get("CLIENT_METRICS_LOG_DIR", "").strip(),
        "/shared_data",
    ):
        if p and Path(p).is_dir():
            roots.append(Path(p))
    if os.path.exists("/app"):
        roots.append(Path("/app") / "shared_data")
    try:
        here = Path(__file__).resolve().parent.parent.parent
        roots.append(here / "shared_data")
        # Also search the project root itself — clients running locally write
        # JSONL to their CWD which is typically the project root.
        roots.append(here)
    except Exception:
        pass
    # Also try current working directory
    try:
        cwd = Path(os.getcwd())
        roots.append(cwd)
        roots.append(cwd / "shared_data")
    except Exception:
        pass
    # de-dupe, keep order
    out: List[Path] = []
    for r in roots:
        if r not in out and r.is_dir():
            out.append(r)
    return out


def _per_round_averages_from_jsonl(use_case: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Average ``cpu_percent`` / ``memory_percent`` across clients per round (from JSONL)."""
    uc = _sanitize_use_case_token(use_case)
    prefix = f"client_fl_metrics_{uc}_client"
    cpu_by: Dict[int, List[float]] = {}
    mem_by: Dict[int, List[float]] = {}

    for root in _jsonl_search_roots():
        if not root.is_dir():
            continue
        for fp in sorted(root.glob(f"{prefix}*.jsonl")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rnd = int(rec.get("round") or 0)
                        if rnd <= 0:
                            continue
                        c = rec.get("cpu_percent")
                        if c is not None:
                            try:
                                cpu_by.setdefault(rnd, []).append(float(c))
                            except (TypeError, ValueError):
                                pass
                        m = rec.get("memory_percent")
                        if m is not None:
                            try:
                                mem_by.setdefault(rnd, []).append(float(m))
                            except (TypeError, ValueError):
                                pass
            except OSError:
                continue

    def _avg(d: Dict[int, List[float]]) -> Dict[int, float]:
        return {k: sum(v) / len(v) for k, v in d.items() if v}

    return _avg(cpu_by), _avg(mem_by)


def _align_to_rounds(
    rounds: List[Any], cpu_avg: Dict[int, float], mem_avg: Dict[int, float]
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    out_c: List[Optional[float]] = []
    out_m: List[Optional[float]] = []
    for r in rounds:
        try:
            ri = int(r)
        except (TypeError, ValueError):
            out_c.append(None)
            out_m.append(None)
            continue
        out_c.append(cpu_avg.get(ri))
        out_m.append(mem_avg.get(ri))
    return out_c, out_m


def merge_cpu_memory_into_results(results: Dict[str, Any], use_case: str) -> Dict[str, Any]:
    """
    Add ``avg_cpu_percent``, ``avg_memory_percent`` (aligned to ``results['rounds']``),
    and ``cpu_memory_source`` metadata. Missing rounds use JSON ``null``.
    """
    rounds = results.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        results.setdefault("avg_cpu_percent", [])
        results.setdefault("avg_memory_percent", [])
        results["cpu_memory_source"] = "none"
        return results

    cpu_avg, mem_avg = _per_round_averages_from_jsonl(use_case)
    ac, am = _align_to_rounds(rounds, cpu_avg, mem_avg)
    results["avg_cpu_percent"] = ac
    results["avg_memory_percent"] = am
    has = any(x is not None for x in ac) or any(x is not None for x in am)
    results["cpu_memory_source"] = "client_fl_metrics_jsonl" if has else "none"
    return results


def plot_cpu_memory_training_results(
    results_dir: Path,
    image_filename: str,
    results: Dict[str, Any],
    *,
    title: str,
) -> None:
    """Save ``{protocol}_cpu_memory_per_round.png`` if any CPU/RAM samples exist."""
    if plt is None:
        return
    rounds = results.get("rounds") or []
    cpu = results.get("avg_cpu_percent") or []
    mem = results.get("avg_memory_percent") or []
    if not rounds or not (cpu or mem):
        return

    lim = min(len(rounds), len(cpu) if cpu else len(rounds), len(mem) if mem else len(rounds))
    xs: List[Any] = []
    ycpu: List[float] = []
    ymem: List[float] = []
    for i in range(lim):
        r = rounds[i]
        c = cpu[i] if cpu and i < len(cpu) else None
        m = mem[i] if mem and i < len(mem) else None
        if c is None and m is None:
            continue
        try:
            cf = float(c) if c is not None else float("nan")
            mf = float(m) if m is not None else float("nan")
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cf) and not math.isfinite(mf):
            continue
        xs.append(r)
        ycpu.append(cf)
        ymem.append(mf)
    if not xs:
        return

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("CPU (%)", color="#c73e1d", fontsize=12)
    ax1.plot(xs, ycpu, color="#c73e1d", marker="o", linewidth=2, label="Avg CPU %")
    ax1.tick_params(axis="y", labelcolor="#c73e1d")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Memory (%)", color="#3a7ca5", fontsize=12)
    ax2.plot(xs, ymem, color="#3a7ca5", marker="s", linewidth=2, label="Avg RAM %")
    ax2.tick_params(axis="y", labelcolor="#3a7ca5")

    ax1.set_title(title, fontsize=13)
    fig.tight_layout()
    out = results_dir / image_filename
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"CPU/RAM plot saved to {out}")
    except Exception as e:
        print(f"[WARN] Could not save CPU/RAM plot: {e}")
    finally:
        plt.close(fig)


def load_training_results_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def plot_cpu_memory_for_server_rounds(
    results_dir: Path,
    image_filename: str,
    rounds: List[Any],
    use_case: str,
    *,
    title: str,
) -> None:
    """Merge JSONL CPU/RAM for ``rounds`` and save plot (works whether JSON was saved before or after plots)."""
    if not rounds:
        return
    payload: Dict[str, Any] = {"rounds": list(rounds)}
    merge_cpu_memory_into_results(payload, use_case)
    plot_cpu_memory_training_results(results_dir, image_filename, payload, title=title)
