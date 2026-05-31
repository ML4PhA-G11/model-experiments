#!/usr/bin/env bash
salloc --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=16 \
  --time=24:00:00
