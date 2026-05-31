#!/usr/bin/env bash
model="d4equivariant"
batch_size=32
n_epochs=4000
patience=12800
time uv run python -u run_all.py --model ${model} --batch-size ${batch_size} --n-epochs ${n_epochs}  --patience ${patience} --run-name ${model}-${batch_size}-${n_epochs}

