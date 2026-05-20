#!/usr/bin/env bash
sbatch --job-name=lbm-tf-gpu_h100 \
  --partition=gpu_h100 \
  --gpus=1 \
  --cpus-per-task=16 \
  --time=01:00:00 \
  --output=jobs/logs/lbm-tf-gpu_h100-%j.out \
  --error=jobs/logs/lbm-tf-gpu_h100-%j.err \
  jobs/run-all-tensorflow.sh
