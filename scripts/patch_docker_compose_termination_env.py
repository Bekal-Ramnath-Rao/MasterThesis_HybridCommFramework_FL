#!/usr/bin/env python3
"""Inject FL termination env + host NUM_ROUNDS into Docker compose FL server/client services."""
from pathlib import Path

DOCKER = Path(__file__).resolve().parent.parent / "Docker"

STOP_BLOCK = (
    "      - STOP_ON_CLIENT_CONVERGENCE=${STOP_ON_CLIENT_CONVERGENCE:-true}\n"
    "      - TRAINING_TERMINATION_MODE=${TRAINING_TERMINATION_MODE:-}\n"
)


def patch_text(text: str):
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        # Normalize hardcoded round caps so experiment runner NUM_ROUNDS applies
        if line.strip() == "- NUM_ROUNDS=1000":
            line = "      - NUM_ROUNDS=${NUM_ROUNDS:-1000}\n"
            changed = True
        out.append(line)
        st = line.lstrip()
        if st.startswith("- NUM_ROUNDS="):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if "STOP_ON_CLIENT_CONVERGENCE" not in nxt and "TRAINING_TERMINATION_MODE" not in nxt:
                out.append(STOP_BLOCK)
                changed = True
        elif st.startswith("- NUM_CLIENTS=") and ("NUM_CLIENTS=2" in line or "NUM_CLIENTS=1" in line):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if ("FL_DIAGNOSTIC" in nxt or "PROTOCOL=" in nxt) and "STOP_ON_CLIENT_CONVERGENCE" not in nxt:
                out.append(STOP_BLOCK)
                changed = True
        i += 1
    return ("".join(out), changed)


def main() -> None:
    patterns = (
        "docker-compose-emotion*.yml",
        "docker-compose-mentalstate*.yml",
        "docker-compose-temperature*.yml",
    )
    for pat in patterns:
        for path in sorted(DOCKER.glob(pat)):
            if ".bak" in path.name:
                continue
            raw = path.read_text(encoding="utf-8")
            new, changed = patch_text(raw)
            if changed:
                path.write_text(new, encoding="utf-8")
                print(f"Patched {path.name}")


if __name__ == "__main__":
    main()
