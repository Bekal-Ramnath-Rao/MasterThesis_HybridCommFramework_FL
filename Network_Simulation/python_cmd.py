"""
Resolve which Python executable to use for subprocess calls (Windows + Linux).

- If the environment variable PYTHON_CMD is set, it is used (full path or command name).
- Otherwise sys.executable is used so child processes share the same interpreter and venv
  as the running GUI or script.
"""
from __future__ import annotations

import os
import sys


def get_python_executable() -> str:
    override = os.environ.get("PYTHON_CMD", "").strip()
    if override:
        return override
    return sys.executable
