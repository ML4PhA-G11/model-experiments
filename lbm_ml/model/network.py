from collections.abc import Callable
from functools import partial
from typing import cast
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense

from lbm_ml.lattice.symmetry import (
    D4Symmetry,
    D4AntiSymmetry,
    AlgReconstruction,
    compute_d2q9_orbit_indices,
    compute_d2q9_bias_orbit_indices,
)
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


def resnet_sequential_model(
    Q=9, n_hidden_layers=2, n_per_layer=50, activation="relu", ll_activation="linear", bias=False
):
    """Residual inner network: project → residual blocks → project back.

    Each residual block is a two-layer bottleneck:
        x_new = W₂(activation(W₁·x)) + x
    The second Dense (W₂) has no activation, so its output can be any sign.
    This lets the skip connection genuinely correct in either direction —
    unlike a single-layer relu block where relu(W·x) ≥ 0 forces the hidden
    state to only grow, crippling the network's expressiveness.
    """
    inp = keras.Input(shape=(Q,))

    # Project input to hidden dimension
    x = Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform")(inp)

    # Two-layer residual blocks: activate → linear projection → add skip
    for _ in range(n_hidden_layers):
        residual = x
        x = Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform")(x)
        # No activation here: output can be negative, so skip can correct either way
        x = Dense(n_per_layer, activation=None, use_bias=bias, kernel_initializer="he_uniform")(x)
        x = layers.Add()([x, residual])

    # Project back to Q populations
    out = Dense(Q, activation=ll_activation, use_bias=bias, kernel_initializer="he_uniform")(x)

    return keras.Model(inputs=inp, outputs=out)


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
    model.compile(loss=loss, optimizer=optimizer, jit_compile=cast(str, False))
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


def create_resnet_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    n_hidden_layers: int = 2,
    n_per_layer: int = 50,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = False,
) -> keras.Model:
    """D4-equivariant network with a residual inner sub-network.

    Identical equivariant wrapper as create_model; the inner sub-network uses
    skip connections (ResNet-style) instead of a plain sequential stack.
    """
    return _wrap_d4(
        resnet_sequential_model, loss, optimizer, Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias
    )


# ---------------------------------------------------------------------------
# Plain network (no physics constraints) — Corbetta et al. 2023, "NN Naive"
# ---------------------------------------------------------------------------


def plain_sequential(
    Q: int = 9,
    depth: int = 2,
    n_per_layer: int = 50,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = False,
) -> keras.Model:
    """Plain feed-forward stack: `depth` Dense(n_per_layer) hidden layers + Dense(Q) output.

    Unlike `sequential_model`, `depth` here is the literal hidden-layer count
    (no off-by-one), so `depth=2` reproduces the paper's NN Naive (Table 1).
    """
    model = Sequential([keras.Input(shape=(Q,))])
    for _ in range(depth):
        model.add(Dense(n_per_layer, activation=activation, use_bias=bias, kernel_initializer="he_uniform"))
    model.add(Dense(Q, activation=ll_activation, use_bias=bias, kernel_initializer="he_uniform"))
    return model


def create_plain_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    depth: int = 2,
    n_per_layer: int = 50,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = False,
) -> keras.Model:
    """Plain network with NO physics constraints (no D4 GAVG, no conservation reconstruction).

    Reproduces the "NN Naive" baseline of Corbetta et al. 2023 (Eur. Phys. J. E 46:10,
    Table 1) when depth=2, n_per_layer=50, activation="relu", bias=False. Bias-less
    ReLU layers are degree-1 homogeneous, so scale equivariance (P1) is the only
    physics property satisfied; D8 symmetry (P2) and conservation (P3) are not.
    """
    model = plain_sequential(Q, depth, n_per_layer, activation, ll_activation, bias)
    model.compile(loss=loss, optimizer=optimizer)
    return model


# ---------------------------------------------------------------------------
# LENN: lattice-equivariant neural network (Ortali et al. 2025)
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="lbm")
class LENNLayer(keras.layers.Layer):
    """Single lattice-equivariant layer for D2Q9 under D4 symmetry.

    Maps (batch, Q, C_in) → (batch, Q, C_out) via an equivariant affine
    transform followed by a pointwise activation.

    The weight matrix A ∈ R^{Q×Q×C_in×C_out} satisfies AP = PA for every
    generator P of D4, leaving only 15 free weight scalars (instead of Q²=81)
    per (C_in, C_out) channel pair.  The bias has 3 free scalars (one per
    population orbit: rest / axis-aligned / diagonal).

    Implementation: free parameters Ã ∈ R^{15×C_in×C_out} are stored; the
    full matrix is reconstructed on-the-fly as A_full = gather(Ã, orbit_idx),
    then contracted with the input via einsum 'ijca,bjc→bia'.

    Reference: Ortali et al., AIAA J. 63(2), 716-731 (2025).
               https://doi.org/10.2514/1.J064453
    """

    def __init__(self, channels_out: int, activation: str = "relu", use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.channels_out = channels_out
        self.activation_fn = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shape):
        C_in = int(input_shape[-1])

        orbit_idx = compute_d2q9_orbit_indices()  # (9, 9)
        n_orbits = int(orbit_idx.max()) + 1  # = 15 for D2Q9
        self._orbit_idx = orbit_idx.astype("int32")

        self.A_tilde = self.add_weight(
            name="A_tilde",
            shape=(n_orbits, C_in, self.channels_out),
            initializer="glorot_uniform",
        )

        if self.use_bias:
            bias_idx = compute_d2q9_bias_orbit_indices()  # (9,)
            n_bias = int(bias_idx.max()) + 1  # = 3 for D2Q9
            self._bias_idx = bias_idx.astype("int32")
            self.b_tilde = self.add_weight(
                name="b_tilde",
                shape=(n_bias, self.channels_out),
                initializer="zeros",
            )

        super().build(input_shape)

    def call(self, x):
        # Reconstruct full (9, 9, C_in, C_out) weight tensor via orbit indexing
        A_full = keras.ops.take(self.A_tilde, self._orbit_idx, axis=0)
        # out[b,i,a] = Σ_{j,c} A_full[i,j,c,a] * x[b,j,c]
        out = keras.ops.einsum("ijca,bjc->bia", A_full, x)
        if self.use_bias:
            b_full = keras.ops.take(self.b_tilde, self._bias_idx, axis=0)  # (9, C_out)
            out = out + b_full[None]
        return self.activation_fn(out)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels_out": self.channels_out,
                "activation": keras.activations.serialize(self.activation_fn),
                "use_bias": self.use_bias,
            }
        )
        return config


def create_lenn_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    channels: tuple[int, ...] = (1, 8, 8, 10),
    activation: str = "relu",
    ll_activation: str = "linear",
    use_bias: bool = True,
    **_kwargs,
) -> keras.Model:
    """Lattice-equivariant neural network (LENN) collision operator surrogate.

    Architecture:
      f_pre → reshape (Q,1) → LENN hidden layers → LENN output layer (1 ch)
            → reshape (Q,) → ll_activation → AlgReconstruction → f_post

    Each LENNLayer is equivariant under D4 by construction — no group averaging
    needed, so inference cost is the same as a plain MLP.  AlgReconstruction
    enforces mass and momentum conservation algebraically.

    Args:
        channels: C_out for each hidden LENN layer (default matches paper Table 1).
        use_bias: whether to include learnable bias in each LENN layer.
        **_kwargs: silently absorbs unused MLP-style kwargs (n_hidden_layers, etc.)
    """
    inp = keras.Input(shape=(Q,))

    x = keras.layers.Reshape((Q, 1))(inp)  # (batch, Q, 1)
    for c in channels:
        x = LENNLayer(c, activation=activation, use_bias=use_bias)(x)
    x = LENNLayer(1, activation="linear", use_bias=use_bias)(x)  # output ch
    x = keras.layers.Reshape((Q,))(x)  # (batch, Q)
    x = keras.layers.Activation(ll_activation)(x)

    out = AlgReconstruction()(inp, x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(loss=loss, optimizer=optimizer, jit_compile=cast(str, False))
    return model


# ---------------------------------------------------------------------------
# Model registry — maps name → factory function
# ---------------------------------------------------------------------------


def create_lenn_resnet_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    channels: tuple[int, ...] = (8, 8, 8),
    activation: str = "relu",
    ll_activation: str = "linear",
    use_bias: bool = True,
    **_kwargs,
) -> keras.Model:
    """LENN with residual blocks (LENN+ResNet) collision operator surrogate.

    Architecture:
      f_pre -> reshape (Q,1) -> entry LENNLayer -> residual blocks
            -> exit LENNLayer (1 ch) -> reshape (Q,) -> ll_activation
            -> AlgReconstruction -> f_post

    Each residual block follows the two-layer pattern:
        x_new = LENNLayer_linear(LENNLayer_activate(x)) + x
    When consecutive channel counts differ a linear LENN projection is used
    for the shortcut so dimensions always match before the Add.

    Args:
        channels: hidden channel count for each residual block.  All equal
                  gives a pure ResNet; varying counts add projection shortcuts.
        use_bias: whether to include learnable bias in each LENN layer.
        **_kwargs: silently absorbs unused MLP-style kwargs (n_hidden_layers, etc.)
    """
    inp = keras.Input(shape=(Q,))

    x = keras.layers.Reshape((Q, 1))(inp)  # (batch, Q, 1)

    # Entry projection: 1 -> channels[0]
    x = LENNLayer(channels[0], activation=activation, use_bias=use_bias)(x)

    prev_c = channels[0]
    for c in channels:
        residual = x
        x = LENNLayer(c, activation=activation, use_bias=use_bias)(x)
        # No activation: output can be negative so skip corrects in either direction
        x = LENNLayer(c, activation="linear", use_bias=use_bias)(x)
        if prev_c != c:
            residual = LENNLayer(c, activation="linear", use_bias=use_bias)(residual)
        x = layers.Add()([x, residual])
        prev_c = c

    # Exit projection: channels[-1] -> 1
    x = LENNLayer(1, activation="linear", use_bias=use_bias)(x)
    x = keras.layers.Reshape((Q,))(x)  # (batch, Q)
    x = keras.layers.Activation(ll_activation)(x)

    out = AlgReconstruction()(inp, x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(loss=loss, optimizer=optimizer, jit_compile=cast(str, False))
    return model


MODEL_REGISTRY: dict[str, Callable] = {
    "d4equivariant": create_model,
    "resnet": create_resnet_model,
    "plain_2": partial(create_plain_model, depth=2),
    "plain_10": partial(create_plain_model, depth=10),
    "plain_20": partial(create_plain_model, depth=20),
    "lenn": create_lenn_model,
    "lenn_resnet": create_lenn_resnet_model,
}
