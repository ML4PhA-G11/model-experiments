#!/usr/bin/env bash
salloc --partition=gpu_h100 \
  --nodes=1 --ntasks=1 \
  --gpus=1 \
  --cpus-per-task=16 \
  --time=12:00:00
