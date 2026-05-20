#!/usr/bin/env bash
# Sweep run-all-tensorflow.py across CPU + 3 GPU partitions on Snellius and
# build a cost/speed comparison table.
#
# What it does:
#   1. (Once) Runs `uv sync` so the shared .venv matches uv.lock (which now
#      includes tensorflow[and-cuda]) before any GPU job starts, instead of
#      each job racing to install CUDA extras.
#   2. sbatch-submits 4 jobs with overrides on top of jobs/run-all-tensorflow.sh:
#        cpu_rome   rome     gpus=0  cpus=16
#        gpu_mig    gpu_mig  gpus=1  cpus=9
#        gpu_a100   gpu_a100 gpus=1  cpus=18
#        gpu_h100   gpu_h100 gpus=1  cpus=16
#   3. Waits for all four to finish (poll squeue every 30s).
#   4. Parses each .out log for the `[job] Total wall time: ...` line, the
#      `[gpu] gpu_used=...` marker (to verify GPU jobs really used the GPU),
#      plus sacct Elapsed/State and writes a markdown comparison to:
#        artifacts-run-all-tensorflow/node-execution-time-comparison.md
#      Also keeps the job-id map at:
#        artifacts-run-all-tensorflow/sweep-jobs.tsv
#
# Usage (run on the Snellius login node):
#   bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh
#   bash scripts/01-exp-sweep.node.with.without.gpu.execution.time.sh --no-wait
#       just submits and exits; re-run with no flags to summarize an
#       existing sweep (uses the saved job IDs from sweep-jobs.tsv).
#
# Re-runs: rerunning without --no-wait after a sweep already completed will
# simply rebuild the markdown from the cached job IDs, which is cheap.

set -euo pipefail

############################################
# args
############################################
WAIT=1
SUBMIT=1
for arg in "$@"; do
    case "$arg" in
        --no-wait)     WAIT=0 ;;
        --summarize)   SUBMIT=0 ;;  # rebuild markdown only
        *)             echo "Unknown arg: $arg" >&2; exit 2 ;;
    esac
done

############################################
# locate project
############################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="$PROJECT_ROOT/artifacts-run-all-tensorflow"
mkdir -p "$OUT_DIR" jobs/logs
MAP_FILE="$OUT_DIR/sweep-jobs.tsv"
SUMMARY="$OUT_DIR/node-execution-time-comparison.md"

############################################
# environment for `uv` (login-node side)
############################################
if ! command -v module >/dev/null 2>&1; then
    [[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
fi
module purge >/dev/null 2>&1 || true
module load 2024 >/dev/null 2>&1 || true
module load Python/3.12.3-GCCcore-13.3.0 >/dev/null 2>&1 || true
export PATH="$HOME/.local/bin:$PATH"

############################################
# 1. pre-install CUDA extras (idempotent, shared across jobs)
############################################
if [[ $SUBMIT -eq 1 ]]; then
    if [[ ! -f uv.lock ]]; then
        echo "[sweep] ERROR: uv.lock not found — run scripts/snellius-py-runtime-setup.sh first." >&2
        exit 1
    fi
    # uv.lock pins tensorflow[and-cuda], so a single `uv sync` materialises the
    # CUDA extras (~2-3 GB) into the shared .venv once. Idempotent: a no-op when
    # the venv already matches the lock.
    echo "[sweep] Syncing .venv to uv.lock (ensures tensorflow[and-cuda]) ..."
    uv sync --quiet
    if ! uv pip show nvidia-cudnn-cu12 >/dev/null 2>&1; then
        echo "[sweep] ERROR: CUDA extras still missing after uv sync — check uv.lock." >&2
        exit 1
    fi
    echo "[sweep] CUDA extras present in .venv."
fi

############################################
# 2. submit (unless --summarize)
############################################
# label  partition  gpus  cpus
CONFIGS=(
    "cpu_rome rome     0  16"
    "gpu_mig  gpu_mig  1  9"
    "gpu_a100 gpu_a100 1  18"
    "gpu_h100 gpu_h100 1  16"
)

if [[ $SUBMIT -eq 1 ]]; then
    printf 'label\tpartition\tgpus\tcpus\tjobid\n' > "$MAP_FILE"
    declare -a JOB_IDS=()
    for cfg in "${CONFIGS[@]}"; do
        read -r LABEL PART GPUS CPUS <<<"$cfg"
        NAME="lbm-tf-${LABEL}"

        # Per-partition tweaks to improve scheduling. For gpu_mig we ask
        # for the exact MIG profile Snellius exposes (a100_3g.20gb, 32
        # slices across gcn[2-5]) and a 45-min walltime so the job is
        # eligible for backfill.
        # Per-partition tweaks. For gpu_mig we pin the exact MIG profile
        # Snellius exposes (a100_3g.20gb, 32 slices total across gcn[2-5])
        # so the scheduler matches us against any of them, and we request
        # 45 min instead of 1 h so the job is eligible for backfill into
        # gaps between longer jobs.
        TIME_ARG="--time=01:00:00"
        case "$PART" in
            gpu_mig)
                # Typed --gpus form cleanly overrides the default --gpus=1
                # in jobs/run-all-tensorflow.sh; --gres would conflict with
                # it ("with and without type identification" sbatch error).
                GPU_FLAG=(--gpus="a100_3g.20gb:${GPUS}")
                TIME_ARG="--time=00:45:00"
                ;;
            *)
                GPU_FLAG=(--gpus="$GPUS")
                ;;
        esac

        JID=$(sbatch --parsable \
                --job-name="$NAME" \
                --partition="$PART" \
                "${GPU_FLAG[@]}" \
                --cpus-per-task="$CPUS" \
                "$TIME_ARG" \
                --output="jobs/logs/${NAME}-%j.out" \
                --error="jobs/logs/${NAME}-%j.err" \
                jobs/run-all-tensorflow.sh)
        echo "[sweep] $LABEL on $PART  gpus=$GPUS cpus=$CPUS  -> jobid $JID"
        printf '%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$PART" "$GPUS" "$CPUS" "$JID" >> "$MAP_FILE"
        JOB_IDS+=("$JID")
    done
else
    [[ -f "$MAP_FILE" ]] || { echo "[sweep] $MAP_FILE missing, nothing to summarize." >&2; exit 1; }
    JOB_IDS=()
    while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
        [[ "$LABEL" == "label" ]] && continue
        JOB_IDS+=("$JID")
    done < "$MAP_FILE"
fi

if [[ $WAIT -eq 0 ]]; then
    echo "[sweep] --no-wait: skipping wait/summary; job IDs in $MAP_FILE"
    exit 0
fi

############################################
# 3. wait for jobs to leave the queue
############################################
echo "[sweep] Waiting for jobs to finish ..."
JIDS_CSV="$(IFS=,; echo "${JOB_IDS[*]}")"
while :; do
    IN_QUEUE=$(squeue -h -j "$JIDS_CSV" -o "%i %T %M" 2>/dev/null || true)
    [[ -z "$IN_QUEUE" ]] && break
    echo "[sweep] $(date +%H:%M:%S) — still active:"
    echo "$IN_QUEUE" | sed 's/^/         /'
    sleep 30
done
echo "[sweep] All jobs have left the queue."

############################################
# 4. parse logs + build markdown
############################################
{
    echo "# Snellius node/GPU sweep — \`run-all-tensorflow.py\`"
    echo
    echo "_Generated: $(date -Is)_"
    echo
    echo "Each row launches the same \`run-all-tensorflow.py\` via"
    echo "\`jobs/run-all-tensorflow.sh\` with sbatch overrides for partition,"
    echo "GPU count, and CPU count."
    echo
    echo "## Execution times"
    echo
    echo "| Label | Partition | GPUs req | CPUs | Job ID | GPUs visible | GPU used | Wall time | Slurm Elapsed | State | Speedup vs CPU |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|"
} > "$SUMMARY"

declare -A WALL_SEC WALL_HMS ELAPSED STATE GPU_VIS GPU_USED

while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
    [[ "$LABEL" == "label" ]] && continue
    OUT="jobs/logs/lbm-tf-${LABEL}-${JID}.out"
    sec="N/A"; hms="N/A"
    gpu_vis="N/A"; gpu_used="N/A"
    if [[ -f "$OUT" ]]; then
        line=$(grep -E '^\[job\] Total wall time:' "$OUT" | tail -1 || true)
        if [[ -n "$line" ]]; then
            sec=$(awk '{for(i=1;i<=NF;i++) if($i ~ /^[0-9]+s$/){gsub(/s/,"",$i); print $i; exit}}' <<<"$line")
            hms=$(grep -oE '\([0-9:]+\)' <<<"$line" | tr -d '()')
        fi
        # GPU diagnostics emitted by run-all-tensorflow.py's report_gpu():
        #   [gpu] visible_gpu_count=N
        #   [gpu] gpu_used=yes|no
        gv=$(grep -E '^\[gpu\] visible_gpu_count=' "$OUT" | tail -1 | sed 's/.*=//' || true)
        [[ -n "$gv" ]] && gpu_vis="$gv"
        gu=$(grep -E '^\[gpu\] gpu_used=' "$OUT" | tail -1 | sed 's/.*=//' || true)
        [[ -n "$gu" ]] && gpu_used="$gu"
    fi
    WALL_SEC[$LABEL]="$sec"
    WALL_HMS[$LABEL]="$hms"
    GPU_VIS[$LABEL]="$gpu_vis"
    GPU_USED[$LABEL]="$gpu_used"

    sac=$(sacct -j "$JID" -X --noheader -P --format=Elapsed,State 2>/dev/null | head -1 || true)
    ELAPSED[$LABEL]="$(cut -d'|' -f1 <<<"$sac")"
    STATE[$LABEL]="$(cut -d'|' -f2 <<<"$sac")"
done < "$MAP_FILE"

CPU_SEC="${WALL_SEC[cpu_rome]:-N/A}"

while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
    [[ "$LABEL" == "label" ]] && continue
    sec="${WALL_SEC[$LABEL]}"
    hms="${WALL_HMS[$LABEL]}"
    elapsed="${ELAPSED[$LABEL]:-?}"
    state="${STATE[$LABEL]:-?}"
    if [[ "$sec" != "N/A" && "$CPU_SEC" != "N/A" && "$sec" =~ ^[0-9]+$ && "$CPU_SEC" =~ ^[0-9]+$ ]]; then
        speedup=$(awk -v a="$CPU_SEC" -v b="$sec" 'BEGIN{ if(b==0) print "—"; else printf "%.2fx", a/b }')
    else
        speedup="—"
    fi
    wall_str="N/A"
    [[ "$sec" != "N/A" ]] && wall_str="${sec}s (${hms})"
    gpu_vis="${GPU_VIS[$LABEL]:-N/A}"
    gpu_used="${GPU_USED[$LABEL]:-N/A}"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$LABEL" "$PART" "$GPUS" "$CPUS" "$JID" \
        "$gpu_vis" "$gpu_used" \
        "$wall_str" "$elapsed" "$state" "$speedup" \
        >> "$SUMMARY"
done < "$MAP_FILE"

############################################
# 4b. GPU usage verification
############################################
# For each job decide whether the GPU outcome matches what was requested:
#   GPUs req >= 1  -> expect gpu_used=yes  (otherwise the GPU sat idle / the
#                     CUDA build of TF was missing and TF fell back to CPU)
#   GPUs req == 0  -> expect gpu_used=no
{
    echo
    echo "## GPU usage verification"
    echo
    echo "Parsed from the \`[gpu] gpu_used=...\` marker that"
    echo "\`run-all-tensorflow.py\` prints after probing TensorFlow with a real"
    echo "matmul on \`/GPU:0\`. A GPU job that reports \`no\` did **not** actually"
    echo "use the GPU (e.g. \`tensorflow[and-cuda]\` not installed)."
    echo
    echo "| Label | Partition | GPUs req | GPUs visible | GPU used | Verdict |"
    echo "|---|---|---|---|---|---|"
} >> "$SUMMARY"

GPU_FAIL=0
while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
    [[ "$LABEL" == "label" ]] && continue
    gpu_vis="${GPU_VIS[$LABEL]:-N/A}"
    gpu_used="${GPU_USED[$LABEL]:-N/A}"
    if [[ "$GPUS" -ge 1 ]]; then
        if [[ "$gpu_used" == "yes" ]]; then
            verdict="✅ GPU used"
        elif [[ "$gpu_used" == "no" ]]; then
            verdict="❌ GPU requested but NOT used"
            GPU_FAIL=1
        else
            verdict="⚠️ unknown (no marker in log)"
            GPU_FAIL=1
        fi
    else
        if [[ "$gpu_used" == "no" ]]; then
            verdict="✅ CPU-only as expected"
        elif [[ "$gpu_used" == "yes" ]]; then
            verdict="⚠️ unexpected GPU use"
        else
            verdict="⚠️ unknown (no marker in log)"
        fi
    fi
    printf '| %s | %s | %s | %s | %s | %s |\n' \
        "$LABEL" "$PART" "$GPUS" "$gpu_vis" "$gpu_used" "$verdict" >> "$SUMMARY"
    echo "[sweep] GPU check — $LABEL ($PART): req=$GPUS visible=$gpu_vis used=$gpu_used -> $verdict"
done < "$MAP_FILE"

if [[ $GPU_FAIL -ne 0 ]]; then
    echo "[sweep] WARNING: at least one GPU job did not actually use its GPU." >&2
    echo "[sweep]          Check that 'tensorflow[and-cuda]' is installed in .venv" >&2
    echo "[sweep]          and inspect the job's [gpu] lines in jobs/logs/." >&2
fi

{
    echo
    echo "## Approximate cost (SBU = Service Billing Unit, Snellius)"
    echo
    echo "Rates are estimates from public Snellius documentation; the"
    echo "authoritative numbers live in \`accinfo\` / \`mybudget\`."
    echo
    echo "| Label | Approx. SBU rate | Wall time (h) | Approx. SBUs |"
    echo "|---|---|---|---|"
} >> "$SUMMARY"

# rough SBU rates (per allocated unit per hour):
#   rome:     1 SBU per core-hour                                 -> 16 cpu ≈ 16/h
#   gpu_mig:  Snellius slices each A100 into a100_3g.20gb (3/7 of
#             the GPU). 3/7 × 128 ≈ 55 SBU/h per slice.
#   gpu_a100: ~128 SBU per A100-hour
#   gpu_h100: ~192 SBU per H100-hour (newer; check `accinfo`)
declare -A RATE=(
    [cpu_rome]=16
    [gpu_mig]=55
    [gpu_a100]=128
    [gpu_h100]=192
)

while IFS=$'\t' read -r LABEL PART GPUS CPUS JID; do
    [[ "$LABEL" == "label" ]] && continue
    sec="${WALL_SEC[$LABEL]}"
    rate="${RATE[$LABEL]:-?}"
    if [[ "$sec" =~ ^[0-9]+$ ]]; then
        sbus=$(awk -v r="$rate" -v s="$sec" 'BEGIN{printf "%.1f", r*s/3600.0}')
        hours=$(awk -v s="$sec" 'BEGIN{printf "%.3f", s/3600.0}')
    else
        sbus="—"; hours="—"
    fi
    printf '| %s | %s | %s | %s |\n' \
        "$LABEL" "$rate" "$hours" "$sbus" \
        >> "$SUMMARY"
done < "$MAP_FILE"

{
    echo
    echo "## Notes"
    echo
    echo "- The model is tiny (17,702 params, batch_size=32). GPUs are"
    echo "  under-utilised at this batch size; CPU partitions are often"
    echo "  the best cost/speed pick **without** code changes."
    echo "- To make GPUs pay off, raise \`batch_size\` (e.g. 1024-4096) in"
    echo "  \`run-all-tensorflow.py\` and verify loss still converges."
    echo "- \`gpu_mig\` is the cheapest GPU on Snellius — useful for"
    echo "  GPU smoke tests before committing to A100/H100 budget."
} >> "$SUMMARY"

echo "[sweep] Summary written to $SUMMARY"
