#!/usr/bin/env bash
# Diagnose whether the project's TensorFlow can detect and use a GPU in the
# CURRENT shell environment. Run this on whatever node you intend to train on
# (Snellius compute node inside salloc/sbatch, local workstation, ...).
#
# What it checks, in order:
#   1. Shell env: host, CUDA_VISIBLE_DEVICES, loaded modules, LD_LIBRARY_PATH.
#   2. nvidia-smi: does the OS see a GPU at all?
#   3. Venv bundled libs: is tensorflow[and-cuda] really installed with the
#      cudnn/cublas/cuda_runtime wheels under .venv/.../nvidia/*/lib?
#   4. LD_LIBRARY_PATH setup: prepends those wheel lib dirs (same logic as
#      jobs/run-all-tensorflow.sh) so TF can dlopen libcudnn/libcublas.
#   5. TF import + tf.config.list_physical_devices('GPU').
#   6. tf.sysconfig.get_build_info() — what CUDA/cuDNN versions TF was built
#      against (compare with what's on disk to spot mismatches).
#   7. Tiny matmul on /GPU:0 with soft_device_placement=False — fails loudly
#      if no GPU is actually usable, instead of silently falling back to CPU.
#
# Exit code: 0 if a GPU is visible AND the matmul ran on it; non-zero otherwise.
#
# Usage:
#   bash scripts/cuda-gpu-tensorflow-enabled.sh

set -uo pipefail   # NOT -e: we want to keep printing diagnostics after a failure

PASS="\033[32m[ OK ]\033[0m"
FAIL="\033[31m[FAIL]\033[0m"
WARN="\033[33m[WARN]\033[0m"
INFO="\033[36m[INFO]\033[0m"

OVERALL_RC=0
fail() { echo -e "$FAIL $*"; OVERALL_RC=1; }
ok()   { echo -e "$PASS $*"; }
warn() { echo -e "$WARN $*"; }
info() { echo -e "$INFO $*"; }

############################################
# 0. project root
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo " TensorFlow GPU diagnostic"
echo "============================================================"
info "Project root: $PROJECT_ROOT"

############################################
# 1. shell environment
############################################
echo
echo "--- 1. shell environment ---"
info "Host:                $(hostname -s)"
info "Date:                $(date -Is)"
info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
info "SLURM_JOB_ID=${SLURM_JOB_ID:-<not in slurm>}"

if command -v module >/dev/null 2>&1; then
    LOADED_MODULES="$(module list 2>&1 | tail -n +2 | tr '\n' ' ' || true)"
    info "Loaded modules:      ${LOADED_MODULES:-<none>}"
    if echo "$LOADED_MODULES" | grep -qiE '(^|[ /])CUDA([/ ]|$)'; then
        warn "A CUDA module is loaded. tensorflow[and-cuda] ships its own CUDA"
        warn "runtime; loading a system CUDA module often produces version mismatches"
        warn "with the bundled cuDNN wheel. Consider 'module unload CUDA'."
    fi
    if echo "$LOADED_MODULES" | grep -qiE 'cuDNN'; then
        warn "A cuDNN module is loaded — same caveat as above."
    fi
else
    info "Lmod not available (no 'module' command)."
fi

info "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

############################################
# 2. nvidia-smi — does the OS see a GPU?
############################################
echo
echo "--- 2. nvidia-smi ---"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L >/dev/null 2>&1; then
        nvidia-smi -L | sed 's/^/        /'
        ok "nvidia-smi reports at least one GPU."
        DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
        info "Driver version:      ${DRIVER_VER}"
    else
        fail "nvidia-smi exists but reports no GPUs (allocated?). Are you on a GPU node?"
    fi
else
    fail "nvidia-smi not found on PATH. This shell does not have GPU access."
fi

############################################
# 3. venv presence + bundled NVIDIA wheels
############################################
echo
echo "--- 3. venv + tensorflow[and-cuda] wheels ---"
if [[ ! -d .venv ]]; then
    fail ".venv not found in $PROJECT_ROOT. Run scripts/snellius-py-runtime-setup.sh first."
fi

NV_DIRS=( "$PROJECT_ROOT"/.venv/lib/python*/site-packages/nvidia/*/lib )
if [[ -d "${NV_DIRS[0]:-/nonexistent}" ]]; then
    ok "Found bundled NVIDIA libs:"
    for d in "${NV_DIRS[@]}"; do
        # show the package (cudnn, cublas, cuda_runtime, ...) and its top .so
        pkg="$(basename "$(dirname "$d")")"
        so="$(ls "$d"/*.so* 2>/dev/null | head -1 | xargs -r basename)"
        printf "        %-20s %s\n" "$pkg" "${so:-<empty>}"
    done
else
    fail "No bundled NVIDIA libs under .venv/.../site-packages/nvidia/*/lib."
    fail "Re-run: uv add 'tensorflow[and-cuda]>=2.21.0' && uv sync"
fi

############################################
# 4. expose wheel libs via LD_LIBRARY_PATH
############################################
echo
echo "--- 4. configure LD_LIBRARY_PATH for the wheels ---"
NV_LIB_DIRS="$(ls -d "$PROJECT_ROOT"/.venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd: -)"
if [[ -n "$NV_LIB_DIRS" ]]; then
    export LD_LIBRARY_PATH="${NV_LIB_DIRS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ok "Prepended $(echo "$NV_LIB_DIRS" | tr ':' '\n' | wc -l) wheel lib dirs to LD_LIBRARY_PATH."
else
    warn "No wheel lib dirs to add — TF may fail to find libcudnn / libcublas."
fi

############################################
# 5–7. TF import, GPU detection, build info, matmul on /GPU:0
############################################
echo
echo "--- 5–7. TensorFlow import, GPU enumeration, matmul on GPU ---"
if ! command -v uv >/dev/null 2>&1; then
    fail "'uv' not on PATH. Add ~/.local/bin to PATH (the snellius-py-runtime-setup script does this)."
    echo
    echo "Overall exit: $OVERALL_RC"
    exit $OVERALL_RC
fi

# Run the Python diagnostics inside the project's uv-managed venv.
# Exit codes:
#   0 = GPU detected AND matmul ran on /GPU:0
#   2 = TF imported but no GPU detected
#   3 = TF imported, GPU detected, but matmul failed on GPU
#   4 = TF import error
TF_RC=0
uv run python - <<'PY' || TF_RC=$?
import os
import sys
import traceback

# Surface TF's own log lines so cuDNN/CUDA messages aren't swallowed by the
# default WARNING filter. 1 = INFO+, 0 = all.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

try:
    import tensorflow as tf
except Exception:
    print("TF import failed:")
    traceback.print_exc()
    sys.exit(4)

print(f"tensorflow version : {tf.__version__}")
print(f"keras version      : {tf.keras.__version__}")

build = tf.sysconfig.get_build_info()
print(f"built with CUDA    : {build.get('cuda_version', '?')}")
print(f"built with cuDNN   : {build.get('cudnn_version', '?')}")
print(f"is_cuda_build      : {build.get('is_cuda_build', '?')}")

gpus = tf.config.list_physical_devices("GPU")
print(f"physical GPUs      : {gpus}")
if not gpus:
    print("No GPU visible to TensorFlow.")
    sys.exit(2)

# Force the matmul onto the GPU. soft_device_placement=False makes TF raise
# instead of silently falling back to CPU if the op can't run on GPU.
tf.config.set_soft_device_placement(False)
try:
    with tf.device("/GPU:0"):
        a = tf.random.normal((1024, 1024), dtype=tf.float32)
        b = tf.random.normal((1024, 1024), dtype=tf.float32)
        c = tf.matmul(a, b)
        # .numpy() forces synchronous execution so any error surfaces here.
        s = float(tf.reduce_sum(c).numpy())
    print(f"matmul on /GPU:0   : OK (sum={s:.3e}, device={c.device})")
except Exception:
    print("matmul on /GPU:0 failed:")
    traceback.print_exc()
    sys.exit(3)

# Bonus: log a fp64 matmul too — the project trains in float64 (K.set_floatx).
try:
    with tf.device("/GPU:0"):
        a = tf.random.normal((512, 512), dtype=tf.float64)
        b = tf.random.normal((512, 512), dtype=tf.float64)
        c = tf.matmul(a, b)
        _ = float(tf.reduce_sum(c).numpy())
    print(f"fp64 matmul on GPU : OK (device={c.device})")
except Exception:
    print("fp64 matmul on /GPU:0 failed (CPU-only op, or unsupported):")
    traceback.print_exc()
    # not fatal — fp64 ops may legitimately fall back; flag but don't fail.

sys.exit(0)
PY

case $TF_RC in
    0) ok   "TF imported, GPU detected, fp32 matmul ran on /GPU:0." ;;
    2) fail "TF imported but reports no GPU. See section 1–4 above for clues (likely libcudnn dlopen)." ;;
    3) fail "TF saw a GPU but failed to actually run on it (driver/cuDNN mismatch or OOM)." ;;
    4) fail "TF failed to import. Check 'uv sync' completed without errors." ;;
    *) fail "Unexpected python exit code: $TF_RC." ;;
esac
[[ $TF_RC -ne 0 ]] && OVERALL_RC=1

echo
echo "============================================================"
if [[ $OVERALL_RC -eq 0 ]]; then
    ok "TensorFlow can detect and USE the GPU in this environment."
else
    fail "TensorFlow cannot use the GPU in this environment. See messages above."
    echo
    echo "Common fixes:"
    echo "  1. Run from a node with a GPU (salloc/sbatch on a gpu_* partition)."
    echo "  2. 'module purge' any CUDA/cuDNN modules — tensorflow[and-cuda] is self-contained."
    echo "  3. Reinstall the wheels: uv add 'tensorflow[and-cuda]>=2.21.0' && uv sync"
fi
echo "============================================================"
exit $OVERALL_RC
