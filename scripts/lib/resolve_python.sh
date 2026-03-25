#!/usr/bin/env bash
# Sets PYTHON to the Python 3 interpreter command for the current OS.
# - Linux/macOS and typical Unix: prefer python3, then python
# - Windows (Git Bash / MSYS / Cygwin): prefer python (python3 is often missing), then python3
#
# Override anytime: export PYTHON_CMD="/path/to/python3" or PYTHON_CMD="python"

if [ -n "${PYTHON_CMD:-}" ]; then
  PYTHON="$PYTHON_CMD"
elif case "$(uname -s 2>/dev/null)" in MINGW*|MSYS*|CYGWIN*) true ;; *) false ;; esac; then
  if command -v python >/dev/null 2>&1; then
    PYTHON=python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "python not found in PATH (Windows environment)" >&2
    return 1 2>/dev/null || exit 1
  fi
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  elif command -v python >/dev/null 2>&1; then
    PYTHON=python
  else
    echo "python3/python not found in PATH" >&2
    return 1 2>/dev/null || exit 1
  fi
fi
