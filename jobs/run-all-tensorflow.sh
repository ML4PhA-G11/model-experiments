#!/usr/bin/env bash
#SBATCH --job-name=lbm-tf
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=jobs/logs/%x-%j.out
#SBATCH --error=jobs/logs/%x-%j.err
#
# Submit with:  sbatch jobs/run-all-tensorflow.sh
# (Run this from the project root, NOT from inside jobs/.)
#
# Inspecting TensorBoard during or after the run (run on the login node):
#   uv run tensorboard --logdir artifacts-run-all-tensorflow --port 6006
# Then on your machine:
#   ssh -NL 6006:localhost:6006 <user>@snellius.surf.nl
#
# Switching to GPU (requires `uv add tensorflow[and-cuda]` first):
#   #SBATCH --partition=gpu_a100
#   #SBATCH --gpus=1
#   #SBATCH --cpus-per-task=18      # gpu_a100 allows ~18 CPUs / GPU
#
# Logs go to jobs/logs/<jobname>-<jobid>.{out,err}; the directory is created
# below if it does not yet exist.

set -euo pipefail

############################################
# 0. timing — capture the wall clock at job start
############################################
JOB_START_EPOCH=$(date +%s)
JOB_START_HUMAN=$(date -Is)
echo "[job] Started at ${JOB_START_HUMAN}"
echo "[job] SLURM_JOB_ID=${SLURM_JOB_ID:-<not-in-slurm>} on $(hostname -s)"

############################################
# 1. project root + log dir
############################################
# Slurm copies the batch script into a spool directory and runs it from
# there, so $BASH_SOURCE[0] cannot locate the project. Use $SLURM_SUBMIT_DIR
# (the directory `sbatch` was invoked from) and fall back to the script path
# for direct/local execution. SBATCH --output paths are also relative to
# $SLURM_SUBMIT_DIR, so always submit this from the project root.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$PROJECT_ROOT"
mkdir -p jobs/logs

echo "[job] Project root: ${PROJECT_ROOT}"

############################################
# 1b. per-job artifacts dir (avoid races between concurrent jobs)
############################################
# Every job in the sweep otherwise writes to the same
# artifacts-run-all-tensorflow/ tree (dataset, weights, plots, TensorBoard
# logs), so parallel jobs clobber each other's outputs. Give each job its own
# subdirectory keyed by job name + id (falls back to PID for local runs) and
# hand it to run-all-tensorflow.py via RUN_ALL_TF_ARTIFACTS_DIR.
RUN_TAG="${SLURM_JOB_NAME:-local}-${SLURM_JOB_ID:-$$}"
export RUN_ALL_TF_ARTIFACTS_DIR="${PROJECT_ROOT}/artifacts-run-all-tensorflow/${RUN_TAG}"
mkdir -p "$RUN_ALL_TF_ARTIFACTS_DIR"
echo "[job] Artifacts dir: ${RUN_ALL_TF_ARTIFACTS_DIR}"

############################################
# 2. environment — same modules as the setup script
############################################
if ! command -v module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/lmod.sh ]]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    fi
fi

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
# CUDA runtime so TensorFlow can dlopen the GPU/driver libraries. Without this
# (and the LD_LIBRARY_PATH below) TF logs "Cannot dlopen some GPU libraries",
# reports 0 visible GPUs, and silently runs on CPU even on a GPU node. Safe to
# load on CPU partitions too — it just goes unused there.
module load CUDA/12.6.0 || echo "[job] WARNING: CUDA/12.6.0 module not available"

export PATH="$HOME/.local/bin:$PATH"  # for uv

# tensorflow[and-cuda] ships its own CUDA/cuDNN/cuBLAS wheels under
# .venv/.../site-packages/nvidia/*/lib. Put them ahead on LD_LIBRARY_PATH so TF
# loads the exact versions it was built against (matches the CUDA module above).
NV_LIB_DIRS="$(ls -d "$PROJECT_ROOT"/.venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)"
if [[ -n "$NV_LIB_DIRS" ]]; then
    export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[job] Added venv NVIDIA libs to LD_LIBRARY_PATH."
fi

echo "[job] Loaded modules:"
module list 2>&1 | sed 's/^/         /'

############################################
# 3. point TF / numba at the allocated cores
############################################
NCPU="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="${NCPU}"
export TF_NUM_INTRAOP_THREADS="${NCPU}"
export TF_NUM_INTEROP_THREADS=2
export NUMBA_NUM_THREADS="${NCPU}"
echo "[job] CPUs allocated: ${NCPU}"

############################################
# 4. run training + simulation
############################################
# `uv run` activates .venv automatically and uses uv.lock for reproducibility.
# -u flushes stdout so the .out file streams instead of buffering.
echo "[job] Launching run-all-tensorflow.py ..."
uv run python -u run-all-tensorflow.py

RUN_RC=$?

############################################
# 5. timing — print wall clock at job end
############################################
JOB_END_EPOCH=$(date +%s)
JOB_END_HUMAN=$(date -Is)
ELAPSED_SECONDS=$(( JOB_END_EPOCH - JOB_START_EPOCH ))
ELAPSED_HMS=$(printf '%02d:%02d:%02d' \
    $(( ELAPSED_SECONDS / 3600 )) \
    $(( (ELAPSED_SECONDS % 3600) / 60 )) \
    $(( ELAPSED_SECONDS % 60 )))

echo "[job] Finished at ${JOB_END_HUMAN}"
echo "[job] Exit code: ${RUN_RC}"
echo "[job] Total wall time: ${ELAPSED_SECONDS}s (${ELAPSED_HMS})"

# Slurm also records this; print it after the job lands so it's in the same
# log next to our own timing.
if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v sacct >/dev/null 2>&1; then
    echo "[job] sacct view (Elapsed / MaxRSS / State):"
    sacct -j "${SLURM_JOB_ID}" \
        --format=JobID,JobName,Partition,Elapsed,MaxRSS,State \
        2>&1 | sed 's/^/         /'
fi

exit "${RUN_RC}"
