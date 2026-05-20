#!/usr/bin/env bash
sbatch --job-name=lbm-tf-cpu_rome \
  --partition=rome \
  --gpus=0 \
  --cpus-per-task=16 \
  --time=01:00:00 \
  --output=jobs/logs/lbm-tf-cpu_rome-%j.out \
  --error=jobs/logs/lbm-tf-cpu_rome-%j.err \
  jobs/run-all-tensorflow.sh
