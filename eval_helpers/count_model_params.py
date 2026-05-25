#!/usr/bin/env python3
"""Print free (trainable) and expanded parameter counts for each model.

Both metrics are derived purely from the Keras model — no class imports needed.

  Free params   — model.count_params(): unique trainable scalars stored in memory.

  Expanded params — parameters accessed per forward pass, counting reuse:
    * layer._inbound_nodes gives how many times a layer is called within the
      model (e.g. 8 for a GAVG sub-network shared across all D4 transforms).
    * layer.expanded_params() reports per-call weight usage when a layer
      reconstructs a larger tensor from its compact free parameters (e.g. LENN
      gather-expands A_tilde into A_full on every call).
    * For all other layers, per-call usage == stored params.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras

keras.backend.set_floatx("float64")

from lbm_ml.model.network import MODEL_REGISTRY


def _per_call_params(layer: keras.layers.Layer) -> int:
    """Weight values accessed in a single call to this layer."""
    if hasattr(layer, "expanded_params"):
        return layer.expanded_params()
    return sum(w.numpy().size for w in layer.trainable_weights)


def _call_count(layer: keras.layers.Layer) -> int:
    """Times this layer is called when the model runs once."""
    return max(len(layer._inbound_nodes), 1)


def expanded_params(model: keras.Model) -> int:
    return sum(_per_call_params(l) * _call_count(l) for l in model.layers)


def main():
    common_kwargs = dict(loss="mse", optimizer="adam", ll_activation="softmax")

    header = f"{'Model':<18} {'Free params':>14} {'Expanded params':>16} {'Ratio':>8}"
    print(header)
    print("-" * len(header))

    for name, factory in MODEL_REGISTRY.items():
        model = factory(**common_kwargs)
        free = model.count_params()
        exp = expanded_params(model)
        ratio = exp / free if free else float("nan")
        print(f"{name:<18} {free:>14,} {exp:>16,} {ratio:>8.1f}x")


if __name__ == "__main__":
    main()
