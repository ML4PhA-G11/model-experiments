from collections.abc import Callable
from enum import Enum
from functools import partial
from typing import cast
import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

from lbm_ml.lattice.symmetry import (
    D4Symmetry,
    D4AntiSymmetry,
    AlgReconstruction,
    SymmetricAlgReconstruction,
    PositivitySafeAlgReconstruction,
    MultiplicativeAlgReconstruction,
    compute_d2q9_orbit_indices,
    compute_d2q9_bias_orbit_indices,
)


class Reconstruction(str, Enum):
    """Conservation reconstruction layer applied after the NN output.

    Pass None to any factory to skip reconstruction entirely (use with a
    softmax last layer — the normalize/denormalize trick conserves mass and
    softmax guarantees positivity).
    """

    ALG = "alg"                        # AlgReconstruction — fixes indices 2,5,8 (legacy, breaks D4 equivariance)
    SYMMETRIC = "symmetric"            # SymmetricAlgReconstruction — D4-equivariant minimum-norm correction
    POSITIVITY_SAFE = "positivity_safe"  # PositivitySafeAlgReconstruction — scales correction to keep f_i ≥ 0
    MULTIPLICATIVE = "multiplicative"  # MultiplicativeAlgReconstruction — multiplicative, anchored to fpred

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


def _pick_recon(reconstruction: Reconstruction | None) -> keras.layers.Layer | None:
    """Instantiate the requested reconstruction layer, or None to skip."""
    match reconstruction:
        case None:
            return None
        case Reconstruction.MULTIPLICATIVE:
            return MultiplicativeAlgReconstruction()
        case Reconstruction.POSITIVITY_SAFE:
            return PositivitySafeAlgReconstruction()
        case Reconstruction.ALG:
            return AlgReconstruction()
        case _:
            return SymmetricAlgReconstruction()


def _wrap_d4(
    sub_model_fn,
    loss,
    optimizer,
    Q,
    n_hidden_layers,
    n_per_layer,
    activation,
    ll_activation,
    bias,
    steps_per_execution: int = 1,
    reconstruction: Reconstruction | None = Reconstruction.ALG,
) -> keras.Model:
    """Wrap any inner sub-network factory in the D4-equivariant lift/pool pattern.

    reconstruction=None skips conservation enforcement entirely (use with softmax).
    See Reconstruction enum for available options.
    """
    the_input = keras.Input(shape=(Q,))

    sub = sub_model_fn(Q, n_hidden_layers, n_per_layer, activation, ll_activation, bias)

    input_lst = D4Symmetry()(the_input)
    output_lst = [sub(x) for x in input_lst]
    recon = _pick_recon(reconstruction)
    if recon is not None:
        output_lst = [recon(input_lst[k], x) for k, x in enumerate(output_lst)]
    output_lst = D4AntiSymmetry()(output_lst)

    the_output = layers.Average()(output_lst)
    model = keras.Model(inputs=the_input, outputs=the_output)
    model.compile(loss=loss, optimizer=optimizer, jit_compile=cast(str, True), steps_per_execution=steps_per_execution)
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
    steps_per_execution: int = 1,
    reconstruction: Reconstruction | None = Reconstruction.ALG,
) -> keras.Model:
    """D4-equivariant network with a plain feed-forward inner sub-network.

    Architecture:
      1. Lift input to all 8 D4-transformed copies (D4Symmetry).
      2. Pass each copy through the same shared-weight sequential sub-network.
      3. Apply reconstruction layer (see Reconstruction enum); None skips it.
      4. Undo each transform (D4AntiSymmetry) then average.
    """
    return _wrap_d4(
        sequential_model,
        loss,
        optimizer,
        Q,
        n_hidden_layers,
        n_per_layer,
        activation,
        ll_activation,
        bias,
        steps_per_execution,
        reconstruction,
    )


def create_resnet_model(
    loss: str | Callable = "mape",
    optimizer: str = "adam",
    Q: int = 9,
    n_hidden_layers: int = 2,
    n_per_layer: int = 50,
    activation: str = "relu",
    ll_activation: str = "linear",
    bias: bool = False,
    steps_per_execution: int = 1,
    reconstruction: Reconstruction | None = Reconstruction.ALG,
) -> keras.Model:
    """D4-equivariant network with a residual inner sub-network.

    Identical equivariant wrapper as create_model; the inner sub-network uses
    skip connections (ResNet-style) instead of a plain sequential stack.
    """
    return _wrap_d4(
        resnet_sequential_model,
        loss,
        optimizer,
        Q,
        n_hidden_layers,
        n_per_layer,
        activation,
        ll_activation,
        bias,
        steps_per_execution,
        reconstruction,
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
    steps_per_execution: int = 1,
    reconstruction: Reconstruction | None = None,
    **_kwargs,
) -> keras.Model:
    """Plain network with NO physics constraints (no D4 GAVG, no conservation reconstruction).

    Reproduces the "NN Naive" baseline of Corbetta et al. 2023 (Eur. Phys. J. E 46:10,
    Table 1) when depth=2, n_per_layer=50, activation="relu", bias=False. Bias-less
    ReLU layers are degree-1 homogeneous, so scale equivariance (P1) is the only
    physics property satisfied; D8 symmetry (P2) and conservation (P3) are not.

    reconstruction is accepted but ignored — the plain model has no reconstruction layer.
    """
    model = plain_sequential(Q, depth, n_per_layer, activation, ll_activation, bias)
    model.compile(loss=loss, optimizer=optimizer, steps_per_execution=steps_per_execution)
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

    def display_weights(self) -> None:
        """Print the full A and bias matrices for inspection."""

        np.set_printoptions(precision=4, suppress=True, linewidth=100)
        A_full = self.A_tilde.numpy()[self._orbit_idx]  # (9, 9, C_in, C_out)
        for ci in range(A_full.shape[2]):
            for co in range(A_full.shape[3]):
                print(f"A_full [C_in={ci}, C_out={co}]:\n{A_full[:, :, ci, co]}\n")
        if self.use_bias:
            b_full = self.b_tilde.numpy()[self._bias_idx]  # (9, C_out)
            print(f"b_full (9 × C_out):\n{b_full}\n")

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
    steps_per_execution: int = 1,
    reconstruction: Reconstruction | None = Reconstruction.SYMMETRIC,
    **_kwargs,
) -> keras.Model:
    """Lattice-equivariant neural network (LENN) collision operator surrogate.

    Architecture:
      f_pre → reshape (Q,1) → LENN hidden layers → LENN output layer (1 ch)
            → reshape (Q,) → ll_activation → [reconstruction layer] → f_post

    reconstruction=None skips conservation enforcement (use with softmax).
    See Reconstruction enum for available options.

    Args:
        channels: C_out for each hidden LENN layer (default matches paper Table 1).
        use_bias: whether to include learnable bias in each LENN layer.
        reconstruction: conservation layer to apply after the NN output.
        **_kwargs: silently absorbs unused MLP-style kwargs (n_hidden_layers, etc.)
    """
    inp = keras.Input(shape=(Q,))

    x = keras.layers.Reshape((Q, 1))(inp)  # (batch, Q, 1)
    for c in channels:
        x = LENNLayer(c, activation=activation, use_bias=use_bias)(x)
    x = LENNLayer(1, activation="linear", use_bias=use_bias)(x)  # output ch
    x = keras.layers.Reshape((Q,))(x)  # (batch, Q)
    x = keras.layers.Activation(ll_activation)(x)

    recon = _pick_recon(reconstruction)
    out = recon(inp, x) if recon is not None else x

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(loss=loss, optimizer=optimizer, jit_compile=cast(str, True), steps_per_execution=steps_per_execution)
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
    steps_per_execution: int = 1,
    reconstruction: Reconstruction | None = Reconstruction.SYMMETRIC,
    **_kwargs,
) -> keras.Model:
    """LENN with residual blocks (LENN+ResNet) collision operator surrogate.

    Architecture:
      f_pre -> reshape (Q,1) -> entry LENNLayer -> residual blocks
            -> exit LENNLayer (1 ch) -> reshape (Q,) -> ll_activation
            -> [reconstruction layer] -> f_post

    Each residual block follows the two-layer pattern:
        x_new = LENNLayer_linear(LENNLayer_activate(x)) + x
    When consecutive channel counts differ a linear LENN projection is used
    for the shortcut so dimensions always match before the Add.

    Args:
        channels: hidden channel count for each residual block.  All equal
                  gives a pure ResNet; varying counts add projection shortcuts.
        use_bias: whether to include learnable bias in each LENN layer.
        reconstruction: conservation layer to apply after the NN output.
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

    recon = _pick_recon(reconstruction)
    out = recon(inp, x) if recon is not None else x

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(loss=loss, optimizer=optimizer, jit_compile=cast(str, True), steps_per_execution=steps_per_execution)
    return model


MODEL_REGISTRY: dict[str, Callable[..., keras.Model]] = {
    # Conservation-enforced variants (AlgReconstruction / SymmetricAlgReconstruction)
    "d4equivariant": create_model,
    "d4equivariant_10K_wide": partial(create_model, n_per_layer=69),
    "resnet": create_resnet_model,
    "plain_2": partial(create_plain_model, depth=2),
    "plain_10": partial(create_plain_model, depth=10),
    "plain_20": partial(create_plain_model, depth=20),
    "lenn": partial(create_lenn_model, channels=(1, 8, 8, 10), use_bias=True),
    "lenn_18_18_18": partial(create_lenn_model, channels=(18, 18, 18), use_bias=True),
    "lenn_18_18_18_safe": partial(
        create_lenn_model, channels=(18, 18, 18), use_bias=True, reconstruction=Reconstruction.POSITIVITY_SAFE
    ),
    "lenn_resnet": partial(create_lenn_resnet_model, channels=(8, 8, 8), use_bias=True),
    "lenn_resnet_11_11_11": partial(create_lenn_resnet_model, channels=(11, 11, 11), use_bias=True),
    "lenn_resnet_18_18_18": partial(create_lenn_resnet_model, channels=(18, 18, 18), use_bias=True),
    "lenn_resnet_18_18_18_safe": partial(
        create_lenn_resnet_model, channels=(18, 18, 18), use_bias=True, reconstruction=Reconstruction.POSITIVITY_SAFE
    ),
    # Softmax only — reconstruction=None, positivity via softmax + normalize/denormalize trick.
    "d4equivariant_softmax": partial(create_model, ll_activation="softmax", reconstruction=None),
    "d4equivariant_10K_wide_softmax": partial(create_model, n_per_layer=69, ll_activation="softmax", reconstruction=None),
    "resnet_softmax": partial(create_resnet_model, ll_activation="softmax", reconstruction=None),
    "plain_2_softmax": partial(create_plain_model, depth=2, ll_activation="softmax"),
    "lenn_softmax": partial(create_lenn_model, channels=(1, 8, 8, 10), use_bias=True, ll_activation="softmax", reconstruction=None),
    "lenn_18_18_18_softmax": partial(create_lenn_model, channels=(18, 18, 18), use_bias=True, ll_activation="softmax", reconstruction=None),
    "lenn_resnet_softmax": partial(create_lenn_resnet_model, channels=(8, 8, 8), use_bias=True, ll_activation="softmax", reconstruction=None),
    "lenn_resnet_18_18_18_softmax": partial(create_lenn_resnet_model, channels=(18, 18, 18), use_bias=True, ll_activation="softmax", reconstruction=None),
    # Softmax + SYMMETRIC reconstruction (paper's approach, Sec. 3.1).
    "d4equivariant_softmax_cons": partial(create_model, ll_activation="softmax", reconstruction=Reconstruction.SYMMETRIC),
    "d4equivariant_10K_wide_softmax_cons": partial(create_model, n_per_layer=69, ll_activation="softmax", reconstruction=Reconstruction.SYMMETRIC),
    "resnet_softmax_cons": partial(create_resnet_model, ll_activation="softmax", reconstruction=Reconstruction.SYMMETRIC),
    "plain_2_softmax_cons": partial(create_plain_model, depth=2, ll_activation="softmax"),
    "lenn_softmax_cons": partial(create_lenn_model, channels=(1, 8, 8, 10), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.SYMMETRIC),
    "lenn_18_18_18_softmax_cons": partial(create_lenn_model, channels=(18, 18, 18), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.SYMMETRIC),
    "lenn_resnet_softmax_cons": partial(create_lenn_resnet_model, channels=(8, 8, 8), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.SYMMETRIC),
    "lenn_resnet_18_18_18_softmax_cons": partial(create_lenn_resnet_model, channels=(18, 18, 18), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.SYMMETRIC),
    # Softmax + POSITIVITY_SAFE reconstruction (Option B).
    "d4equivariant_softmax_safe": partial(create_model, ll_activation="softmax", reconstruction=Reconstruction.POSITIVITY_SAFE),
    "d4equivariant_10K_wide_softmax_safe": partial(create_model, n_per_layer=69, ll_activation="softmax", reconstruction=Reconstruction.POSITIVITY_SAFE),
    "resnet_softmax_safe": partial(create_resnet_model, ll_activation="softmax", reconstruction=Reconstruction.POSITIVITY_SAFE),
    "plain_2_softmax_safe": partial(create_plain_model, depth=2, ll_activation="softmax"),
    "lenn_softmax_safe": partial(create_lenn_model, channels=(1, 8, 8, 10), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.POSITIVITY_SAFE),
    "lenn_18_18_18_softmax_safe": partial(create_lenn_model, channels=(18, 18, 18), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.POSITIVITY_SAFE),
    "lenn_resnet_softmax_safe": partial(create_lenn_resnet_model, channels=(8, 8, 8), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.POSITIVITY_SAFE),
    "lenn_resnet_18_18_18_softmax_safe": partial(create_lenn_resnet_model, channels=(18, 18, 18), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.POSITIVITY_SAFE),
    # Softmax + MULTIPLICATIVE reconstruction (Option C).
    "d4equivariant_softmax_multcons": partial(create_model, ll_activation="softmax", reconstruction=Reconstruction.MULTIPLICATIVE),
    "d4equivariant_10K_wide_softmax_multcons": partial(create_model, n_per_layer=69, ll_activation="softmax", reconstruction=Reconstruction.MULTIPLICATIVE),
    "resnet_softmax_multcons": partial(create_resnet_model, ll_activation="softmax", reconstruction=Reconstruction.MULTIPLICATIVE),
    "plain_2_softmax_multcons": partial(create_plain_model, depth=2, ll_activation="softmax"),
    "lenn_softmax_multcons": partial(create_lenn_model, channels=(1, 8, 8, 10), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.MULTIPLICATIVE),
    "lenn_18_18_18_softmax_multcons": partial(create_lenn_model, channels=(18, 18, 18), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.MULTIPLICATIVE),
    "lenn_resnet_softmax_multcons": partial(create_lenn_resnet_model, channels=(8, 8, 8), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.MULTIPLICATIVE),
    "lenn_resnet_18_18_18_softmax_multcons": partial(create_lenn_resnet_model, channels=(18, 18, 18), use_bias=True, ll_activation="softmax", reconstruction=Reconstruction.MULTIPLICATIVE),
}
