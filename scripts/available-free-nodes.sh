#!/usr/bin/env bash
# Show per-node free-GPU and free-CPU counts for a Snellius partition.
#
# Usage:
#   bash scripts/available-free-nodes.sh                # default: gpu_mig
#   bash scripts/available-free-nodes.sh gpu_h100
#   bash scripts/available-free-nodes.sh gpu_a100
#   bash scripts/available-free-nodes.sh rome
#
# Output line per node:
#   <node>  state=<state>  free_gpu=<a>/<b>  free_cpu=<c>/<d>  free_mem=<MB>
# Left number is "free right now", right number is "configured".
# Rows that can immediately host a 1-GPU job (>=1 free slot AND enough
# free CPU for the partition's typical per-GPU share) are printed in
# green when the output is a TTY.

set -euo pipefail

PART="${1:-gpu_mig}"

# typical CPU-per-GPU per partition (informational; gates the green row)
case "$PART" in
    gpu_mig)  CPU_PER_GPU=9  ;;
    gpu_a100) CPU_PER_GPU=18 ;;
    gpu_h100) CPU_PER_GPU=16 ;;
    *)        CPU_PER_GPU=0  ;;   # CPU partitions: don't gate on GPU
esac

if [[ -t 1 ]]; then GRN=$'\033[32m'; RST=$'\033[0m'; else GRN=""; RST=""; fi

printf '%-7s  state=%-10s  free_gpu=%-7s  free_cpu=%-9s  free_mem=%s\n' \
    NODE STATE GPU CPU MB

sinfo -p "$PART" -h -N -o '%N %G' \
| awk '!seen[$1]++' \
| while read -r N GRES; do
    # cfg_gpu = number after the LAST ":" in the "gpu:" token of GRES
    cfg_gpu=$(awk -F, '{for(i=1;i<=NF;i++) if($i ~ /^gpu:/){n=split($i,a,":"); print a[n]; exit}}' <<<"$GRES")
    cfg_gpu=${cfg_gpu:-0}

    SC=$(scontrol show node "$N" 2>/dev/null)
    # Full State= field, which includes flags like "MIXED+RESERVED" that
    # sinfo's short `%t` code (e.g. "mix") collapses away.
    full_state=$(sed -n 's/.*State=\([A-Z+]*\).*/\1/p' <<<"$SC")
    full_state=${full_state:-UNKNOWN}
    rsv_name=$(sed -n 's/.*ReservationName=\([^ ]*\).*/\1/p' <<<"$SC")
    alloc_gpu=$(sed -n 's/.*AllocTRES=.*gres\/gpu=\([0-9]*\).*/\1/p' <<<"$SC")
    alloc_gpu=${alloc_gpu:-0}
    cpu_tot=$(sed -n 's/.*CPUTot=\([0-9]*\).*/\1/p' <<<"$SC")
    cpu_alloc=$(sed -n 's/.*CPUAlloc=\([0-9]*\).*/\1/p' <<<"$SC")
    cpu_tot=${cpu_tot:-0}; cpu_alloc=${cpu_alloc:-0}
    free_mem=$(sed -n 's/.*FreeMem=\([0-9]*\).*/\1/p' <<<"$SC")
    free_mem=${free_mem:-?}

    free_gpu=$(( cfg_gpu - alloc_gpu ))
    free_cpu=$(( cpu_tot - cpu_alloc ))

    # Compose a state string that exposes a reservation if present, so the
    # caller can see that "MIXED" alone is misleading when a reservation
    # is holding the remaining slots.
    state_disp="$full_state"
    if [[ -n "$rsv_name" ]]; then
        state_disp="${state_disp}(rsv:${rsv_name})"
    fi

    line=$(printf '%-7s  state=%-26s  free_gpu=%d/%-5d  free_cpu=%d/%-7d  free_mem=%s' \
                  "$N" "$state_disp" "$free_gpu" "$cfg_gpu" "$free_cpu" "$cpu_tot" "$free_mem")
    # Green only when the slot is *actually* takeable: no reservation, not
    # DOWN/DRAIN, and the free counts meet the partition's per-GPU CPU share.
    if [[ "$CPU_PER_GPU" -gt 0 && "$free_gpu" -ge 1 && "$free_cpu" -ge "$CPU_PER_GPU" \
          && "$full_state" != *RESERVED* && "$full_state" != *DOWN* && "$full_state" != *DRAIN* ]]; then
        printf '%s%s%s\n' "$GRN" "$line" "$RST"
    else
        printf '%s\n' "$line"
    fi
done
