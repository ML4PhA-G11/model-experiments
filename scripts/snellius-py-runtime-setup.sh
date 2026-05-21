#!/usr/bin/env bash
# Set up the Python runtime needed by run-all-tensorflow.py on Snellius.
#
# Run this ONCE on a Snellius login node (e.g. snellius.surf.nl). It does NOT
# train anything — training must be submitted through Slurm (sbatch), never on
# the login node. After this script finishes, the project's .venv is ready
# and `uv run python run-all-tensorflow.py` will work inside a Slurm job.
#
# What it does:
#   1. Loads a Snellius Python module that matches pyproject.toml
#      (requires-python >=3.10,<3.14).
#   2. Installs `uv` to ~/.local/bin if it is not already on PATH.
#   3. Runs `uv sync` in the project root to create .venv and install every
#      pinned dependency from uv.lock (tensorflow, keras, numba, ...).
#   4. Smoke-tests the venv by importing tensorflow and keras.
#
# Usage:
#   bash scripts/snellius-py-runtime-setup.sh
#
# Re-running is safe: module load is idempotent, uv install is skipped if
# present, and `uv sync` only changes the venv when uv.lock has changed.

set -euo pipefail

############################################
# 0. sanity: are we on a Snellius login node?
############################################
HOSTNAME_SHORT="$(hostname -s)"
case "$HOSTNAME_SHORT" in
    int*|login*|tcn*|gcn*|hcn*|fcn*)
        echo "[setup] Host: $HOSTNAME_SHORT — looks like a Snellius node."
        ;;
    *)
        echo "[setup] Warning: host '$HOSTNAME_SHORT' does not match a typical"
        echo "        Snellius login pattern (int*/login*). Continue only if"
        echo "        you are sure this is the right machine."
        ;;
esac

if [[ "$HOSTNAME_SHORT" == tcn* || "$HOSTNAME_SHORT" == gcn* ]]; then
    echo "[setup] You appear to be on a COMPUTE node. This setup script is"
    echo "        meant for the LOGIN node. Submit it via Slurm only if you"
    echo "        know what you are doing."
fi

############################################
# 1. project root (script lives in scripts/)
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[setup] Project root: $PROJECT_ROOT"

if [[ ! -f pyproject.toml ]]; then
    echo "[setup] ERROR: pyproject.toml not found in $PROJECT_ROOT" >&2
    exit 1
fi

############################################
# 2. load Snellius modules (Lmod)
############################################
# Snellius exposes year-based software stacks. 2024 ships Python 3.12 built
# with GCC 13.3 — within our >=3.10,<3.14 range and recent enough for the
# manylinux tensorflow 2.21 wheel on PyPI.
echo "[setup] Loading modules from the 2024 Snellius stack ..."

# `module` is a shell function defined by /etc/profile.d/lmod.sh; make sure
# it is available even when this script is run with a non-interactive shell.
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/lmod.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    else
        echo "[setup] ERROR: 'module' command not found and Lmod init script"
        echo "        not in expected location. Are you on Snellius?" >&2
        exit 1
    fi
fi

module purge
module load 2024
# Python 3.12 from the 2024 stack. If this exact name ever shifts, run
# `module spider Python` on the login node and update the line below.
module load Python/3.12.3-GCCcore-13.3.0

echo "[setup] Loaded modules:"
module list 2>&1 | sed 's/^/        /'

PYTHON_BIN="$(command -v python3)"
echo "[setup] python3 -> $PYTHON_BIN ($(python3 --version 2>&1))"

############################################
# 3. install uv (user-local, no sudo)
############################################
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] 'uv' not found — installing to ~/.local/bin via the"
    echo "        official installer (no sudo, no system changes)."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "[setup] uv already installed: $(uv --version)"
fi

# Re-check after install.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] ERROR: uv installation appears to have failed." >&2
    exit 1
fi

############################################
# 4. create .venv and install deps from uv.lock
############################################
# UV_PROJECT_ENVIRONMENT defaults to .venv in the project root, which matches
# the CLAUDE.md convention. We pin uv to the module-provided Python so the
# venv lives on the same interpreter the cluster ships, instead of letting
# uv download its own.

# Add tensorflow[and-cuda] to the venv, which pulls in the CUDA runtime and cuDNN wheels so GPU jobs can actually use the GPU.
echo "[setup] Adding tensorflow[and-cuda] to the venv (pulls in CUDA runtime + cuDNN) ..."
uv add --python "$PYTHON_BIN" "tensorflow[and-cuda]>=2.21.0"
echo "[setup] Running 'uv sync' (creates .venv, installs locked deps) ..."
uv sync --python "$PYTHON_BIN"

############################################
# 5. smoke test
############################################
echo "[setup] Verifying that tensorflow and keras import cleanly ..."
uv run --python "$PYTHON_BIN" python - <<'PY'
import sys
print(f"python: {sys.version.split()[0]}")
import numpy, tensorflow as tf, keras
print(f"numpy:      {numpy.__version__}")
print(f"tensorflow: {tf.__version__}")
print(f"keras:      {keras.__version__}")
PY

echo
echo "[setup] Done. Next steps:"
echo "  - To use the venv interactively:  source .venv/bin/activate"
echo "  - To run a one-off command:       uv run python <script>.py"
echo "  - Do NOT train on the login node. Submit a Slurm job (sbatch)."
