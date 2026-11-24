#!/usr/bin/env bash
# ======================================================================
# SIDIK-ML: Explainable Non-ANN ML Extension for SIDIK
# Cross-Platform Launcher for Linux and macOS
# ======================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
cd "$SCRIPT_DIR"

if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY_CMD="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/../SIDIK/.venv/bin/python" ]; then
    PY_CMD="$SCRIPT_DIR/../SIDIK/.venv/bin/python"
elif command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo "Error: Python 3 is required but was not found in PATH."
    exit 1
fi

if [ "$#" -eq 0 ]; then
    exec "$PY_CMD" -m sidik_ml --gui
else
    exec "$PY_CMD" -m sidik_ml "$@"
fi
