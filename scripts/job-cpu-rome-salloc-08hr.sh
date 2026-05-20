#!/usr/bin/env bash
salloc --partition=rome \
  --nodes=1 --ntasks=1 \
  --gpus=0 \
  --cpus-per-task=16 \
  --time=08:00:00
