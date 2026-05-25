"""Reconstruct a portable model.keras from a weights checkpoint.

``ModelCheckpoint`` saves only weights; this script re-instantiates the
architecture and produces a self-contained model.keras loadable via
``keras.models.load_model`` without the original source code.

Output defaults to model.keras in the same directory as the weights.

Usage::

    python rebuild.py weights.keras
    python rebuild.py weights.keras model.keras --model resnet --learning-rate 5e-4
"""

import argparse
from pathlib import Path

import keras
from keras import backend as K
from lbm_ml.model.network import MODEL_REGISTRY
from lbm_ml import rmsre

p = argparse.ArgumentParser(description="Rebuild a saved model from weights alone.")
p.add_argument("weights_path", help="Path to weights.keras")
p.add_argument("output_path", nargs="?", default=None, help="Path to write model.keras (default: same dir as weights)")
p.add_argument("--model", default="lenn", choices=list(MODEL_REGISTRY))
p.add_argument("--learning-rate", type=float, default=1e-3)
args = p.parse_args()

weights_path = Path(args.weights_path)
output_path = Path(args.output_path) if args.output_path else weights_path.parent / "model.keras"

if output_path.exists():
    answer = input(f"{output_path} already exists. Overwrite? [y/N] ")
    if answer.strip().lower() != "y":
        print("Aborted.")
        raise SystemExit(0)

K.set_floatx("float64")

model = MODEL_REGISTRY[args.model](
    loss=rmsre,
    optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
    ll_activation="softmax",
)
model.load_weights(weights_path)
model.save(output_path)
print("Saved:", output_path)
