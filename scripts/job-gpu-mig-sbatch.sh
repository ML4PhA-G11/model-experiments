#!/usr/bin/env bash
sbatch --job-name=lbm-tf-gpu_mig \
  --partition=gpu_mig \
  --gpus=a100_3g.20gb:1 \
  --cpus-per-task=9 \
  --time=00:45:00 \
  --output=jobs/logs/lbm-tf-gpu_mig-%j.out \
  --error=jobs/logs/lbm-tf-gpu_mig-%j.err \
  jobs/run-all-tensorflow.sh
