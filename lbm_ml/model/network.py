from collections.abc import Callable

import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense

from lbm_ml.lattice.symmetry import D4Symmetry, D4AntiSymmetry, AlgReconstruction
from lbm_ml.model.losses import rmsre

# ---------------------------------------------------------------------------
# Inner sub-networks
# ---------------------------------------------------------------------------


def sequential_model(Q=9, n_hidden_layers=2, n_per_layer=50, activation="relu", ll_activation="linear", bias=False):
    """Plain feed-forward inner network (no skip connections)."""
    model = Sequential(
        [
            keras.Input(shape=(Q,)),
            Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform"),
        ]
    )
    for _ in range(n_hidden_layers):
        model.add(Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform"))
    model.add(Dense(Q, activation=ll_activation, use_bias=bias, kernel_initializer="he_uniform"))
    return model


# ---------------------------------------------------------------------------
# D4-equivariant wrappers
# ---------------------------------------------------------------------------


def _wrap_d4(
    sub_model_fn, loss, optimizer, Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias
) -> keras.Model:
    """Wrap any inner sub-network factory in the D4-equivariant lift/pool pattern."""
    the_input = keras.Input(shape=(Q,))

    sub = sub_model_fn(Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias)

    input_lst = D4Symmetry()(the_input)
    output_lst = [sub(x) for x in input_lst]
    output_lst = [AlgReconstruction()(input_lst[k], x) for k, x in enumerate(output_lst)]
    output_lst = D4AntiSymmetry()(output_lst)

    the_output = layers.Average()(output_lst)
    model = keras.Model(inputs=the_input, outputs=the_output)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def create_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    n_hidden_layers: int = 2,
    n_per_layer: int = 50,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = False,
) -> keras.Model:
    """D4-equivariant network with a plain feed-forward inner sub-network.

    Architecture:
      1. Lift input to all 8 D4-transformed copies (D4Symmetry).
      2. Pass each copy through the same shared-weight sequential sub-network.
      3. Enforce conservation laws (AlgReconstruction) on each branch output.
      4. Undo each transform (D4AntiSymmetry) then average.
    """
    return _wrap_d4(sequential_model, loss, optimizer, Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias)


# ---------------------------------------------------------------------------
# Model registry — maps name → factory function
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, Callable] = {
    "d4equivariant": create_model,
}
