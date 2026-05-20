#!/usr/bin/env bash
# Verify two things about Snellius GPUs that drive the sweep/helper scripts:
#
#   1. Whether an H100 MIG profile exists (it did NOT as of 2026-05-20 — the
#      only MIG gres cluster-wide is a100_3g.20gb; gpu_h100 is full-GPU only).
#   2. The SBU/hour billing rate per partition, derived from each partition's
#      TRESBillingWeights together with PriorityFlags (MAX_TRES => the job is
#      billed the MAX of its weighted TRES, not the sum).
#
# These are the exact commands used to fill in the RATE table in
# 01-exp-sweep.node.with.without.gpu.execution.time.sh and to conclude the
# job-gpu-mig-h100-* helpers can't schedule. Re-run any time to re-check.
#
# Usage:
#   # from your laptop (runs the queries over ssh):
#   bash scripts/snellius-verify-gpu-mig-and-billing.sh
#   bash scripts/snellius-verify-gpu-mig-and-billing.sh <ssh-host>
#
#   # already on a Snellius login node (run the slurm queries locally):
#   bash scripts/snellius-verify-gpu-mig-and-billing.sh --local

set -euo pipefail

SSH_HOST="snellius-from-maibalai"
LOCAL=0
case "${1:-}" in
    --local) LOCAL=1 ;;
    "")      ;;
    *)       SSH_HOST="$1" ;;
esac

# run <cmd...> — execute on Snellius (locally or over ssh depending on mode)
run() {
    if [[ $LOCAL -eq 1 ]]; then
        bash -c "$1"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=25 "$SSH_HOST" "$1"
    fi
}

if [[ $LOCAL -eq 0 ]]; then
    echo "## Querying Snellius via ssh '$SSH_HOST' (pass --local if already on a login node)"
else
    echo "## Running slurm queries locally on this Snellius node"
fi
echo

###########################################################################
echo "============================================================"
echo " 1. GPU gres types per partition (what hardware/MIG exists)"
echo "============================================================"
# %G is the partition's generic-resource (gres) spec. The MIG profile shows
# up here as e.g. 'gpu:a100_3g.20gb:8'. If no line mentions an h100 MIG
# profile (like h100_3g.40gb), H100 MIG simply does not exist on the cluster.
echo "\$ sinfo -o '%P %G' -h | sort -u"
run "sinfo -o '%P %G' -h | sort -u"
echo

echo "------------------------------------------------------------"
echo " Distinct GPU gres types cluster-wide (the decisive check)"
echo "------------------------------------------------------------"
echo "\$ sinfo -h -N -o '%G' | tr ',' '\\n' | sed -E 's/\\(.*//' | grep '^gpu:' | sort -u"
run "sinfo -h -N -o '%G' | tr ',' '\n' | sed -E 's/\(.*//' | grep '^gpu:' | sort -u"
echo
echo " >> If the only 'a100_3g' / Ng.Ngb entry is a100_3g.20gb, there is no"
echo "    H100 MIG, so --gpus=h100_3g.40gb jobs can never be scheduled."
echo

###########################################################################
echo "============================================================"
echo " 2. SBU billing rate per partition"
echo "============================================================"
# Snellius bills using Slurm's TRESBillingWeights. Whether the weighted TRES
# are summed or maxed depends on PriorityFlags: MAX_TRES => bill = max over
# TRES of (weight * allocated count).
echo "\$ scontrol show config | grep -i PriorityFlags"
run "scontrol show config | grep -iE 'PriorityFlags'"
echo
echo " >> 'MAX_TRES' present => job SBU/h = MAX(weighted TRES), not the sum."
echo

echo "------------------------------------------------------------"
echo " TRESBillingWeights per partition"
echo "------------------------------------------------------------"
echo "\$ scontrol show partition <p> | grep -o 'TRESBillingWeights=...'"
run '
for p in rome gpu_mig gpu_a100 gpu_h100; do
  w=$(scontrol show partition "$p" | grep -oE "TRESBillingWeights=[^ ]*")
  printf "%-9s %s\n" "$p" "${w:-<none>}"
done
'
echo
cat <<'EXPLAIN'
 >> Worked example (MAX_TRES) for the job configs the sweep submits:
      rome      16 cpu            : max(16*1.0)              =  16 SBU/h
      gpu_mig   1 mig + 9  cpu    : max(1*64,  9*7.11112=64) =  64 SBU/h
      gpu_a100  1 gpu + 18 cpu    : max(1*128, 18*7.11112=128)= 128 SBU/h
      gpu_h100  1 gpu + 16 cpu    : max(1*192, 16*12.0=192)  = 192 SBU/h
    Each partition's CPU weight is tuned so a fair-share job (1 GPU + its
    CPU quota) costs exactly the GPU weight. These are the RATE values in
    01-exp-sweep.node.with.without.gpu.execution.time.sh.

 (Account-level budget/usage, not per-partition rates, comes from `accinfo`.)
EXPLAIN
