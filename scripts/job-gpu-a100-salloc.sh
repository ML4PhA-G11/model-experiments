#!/usr/bin/env bash
salloc --partition=gpu_a100 \
  --nodes=1 --ntasks=1 \
  --gpus=1 \
  --cpus-per-task=18 \
  --time=01:00:00
