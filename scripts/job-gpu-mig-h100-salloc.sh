#!/usr/bin/env bash
# H100 MIG parity of job-gpu-mig-salloc.sh (3/7 slice, 9 CPUs, 45-min backfill).
#
# NOT AVAILABLE ON SNELLIUS (checked 2026-05-20): there is no H100 MIG.
# The only MIG gres cluster-wide is a100_3g.20gb (gpu_mig); gpu_h100 exposes
# only full gpu:h100:4. Kept ready for if an h100 MIG profile is ever enabled
# on gpu_mig — verify the exact profile name first:
#   sinfo -p gpu_mig -N -o '%N %G'
echo "$(basename "$0"): H100 MIG not available on Snellius (no h100 MIG gres). Aborting." >&2
exit 1

# salloc --partition=gpu_mig \
#   --nodes=1 --ntasks=1 \
#   --gpus=h100_3g.40gb:1 \
#   --cpus-per-task=9 \
#   --time=00:45:00
