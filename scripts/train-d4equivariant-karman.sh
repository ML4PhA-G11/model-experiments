#!/usr/bin/env bash
# Train the D4-equivariant collision operator on REAL Karman-vortex-street
# simulator output (per-step fpre_*.npy / fpost_*.npy pairs), as opposed to the
# synthetic dataset used by train-d4equivariant.sh.
#
# The data lives on Snellius only; override DATA_DIR to point elsewhere.
set -euo pipefail

model="d4equivariant"
batch_size=8192
n_epochs=4000
patience=12800

# Directory of simulator output produced by:
#   python lbm_karman-ng.py --save-every 1
DATA_DIR="${DATA_DIR:-/gpfs/scratch1/shared/scur0076/output-lbm-data-02-30000steps-data.per.step-npy}"

# A single 550x102 snapshot already yields ~56k samples, so sub-sample nodes
# per step and stride over timesteps to keep the training set tractable.
# Override via the environment as needed.
samples_per_step="${SAMPLES_PER_STEP:-2000}"
step_stride="${STEP_STRIDE:-10}"
max_steps="${MAX_STEPS:-}"   # empty = no cap

extra=()
[[ -n "${max_steps}" ]] && extra+=(--max-steps "${max_steps}")

# --skip-simulate: applying the trained model back to the Karman simulator is
# done in the separate Apply-NN-KarmanVortexStreet repo, not by the built-in
# Taylor-Green sanity simulation here.
time uv run python -u run_all.py \
    --model "${model}" \
    --data-dir "${DATA_DIR}" \
    --samples-per-step "${samples_per_step}" \
    --step-stride "${step_stride}" \
    "${extra[@]}" \
    --batch-size "${batch_size}" \
    --n-epochs "${n_epochs}" \
    --patience "${patience}" \
    --skip-simulate \
    --run-name "${model}-karman-${batch_size}-${n_epochs}"
