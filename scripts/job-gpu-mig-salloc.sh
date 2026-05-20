#!/usr/bin/env bash
salloc --partition=gpu_mig \
  --nodes=1 --ntasks=1 \
  --gpus=a100_3g.20gb:1 \
  --cpus-per-task=9 \
  --time=00:45:00
