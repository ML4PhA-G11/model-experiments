#!/usr/bin/env bash
sbatch --job-name=lbm-tf-gpu_a100 \
  --partition=gpu_a100 \
  --gpus=1 \
  --cpus-per-task=18 \
  --time=01:00:00 \
  --output=jobs/logs/lbm-tf-gpu_a100-%j.out \
  --error=jobs/logs/lbm-tf-gpu_a100-%j.err \
  jobs/run-all-tensorflow.sh
