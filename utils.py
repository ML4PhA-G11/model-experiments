import numpy as np
import tensorflow as tf
import keras
from numba import jit


# Root Mean Square Relative Error — used as the training loss/metric.
# Divides by (y_true + eps) to make the error scale-invariant, so the network
# is penalised equally for errors in high- and low-density regions.
def rmsre(y_true, y_pred, eps=1e-8):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square((y_true - y_pred) / (y_true + eps)), axis=-1))


def LB_stencil():
    """Return the D2Q9 lattice stencil and the equilibrium distribution function.

    Lattice Boltzmann Method (LBM) background
    ------------------------------------------
    Instead of tracking individual fluid particles, LBM tracks f_i(x, t): the
    probability density (or "population") of particles at lattice node x that
    are travelling in discrete direction i at time t.

    D2Q9 means: 2 spatial Dimensions, 9 discrete velocity directions Q.
    The nine directions are:
        0 — rest (stationary)
        1–4 — axis-aligned (East, North, West, South)
        5–8 — diagonal (NE, NW, SW, SE)

    Returns
    -------
    c        : (Q, 2) int array  — lattice velocity vectors, one per direction
    w        : (Q,)  float array — quadrature weights for each direction
    cs2      : float            — lattice speed of sound squared (= 1/3)
    compute_feq : callable      — evaluates the Maxwell-Boltzmann equilibrium f^eq
    """

    Q = 9
    c = np.zeros((Q, 2), dtype=np.int32)  # velocity vectors: c[i] = (cx, cy)
    w = np.zeros(Q)  # quadrature weights summing to 1

    # cs2 is the lattice speed of sound squared.  In standard D2Q9 units it is
    # always 1/3; it appears in the equilibrium distribution below.
    cs2 = 1.0 / 3.0
    qorder = 2  # order of the Gauss–Hermite quadrature (kept for reference)

    # Rest particle (direction 0) — largest weight because most particles are
    # stationary on average.
    c[0, 0] = 0
    c[0, 1] = 0
    w[0] = 4.0 / 9.0

    # Axis-aligned neighbours (directions 1–4): East, North, West, South
    c[1, 0] = 1
    c[1, 1] = 0
    w[1] = 1.0 / 9.0
    c[2, 0] = 0
    c[2, 1] = 1
    w[2] = 1.0 / 9.0
    c[3, 0] = -1
    c[3, 1] = 0
    w[3] = 1.0 / 9.0
    c[4, 0] = 0
    c[4, 1] = -1
    w[4] = 1.0 / 9.0

    # Diagonal neighbours (directions 5–8): NE, NW, SW, SE — smaller weight
    # because the effective speed (√2) is higher, so fewer particles travel there.
    c[5, 0] = 1
    c[5, 1] = 1
    w[5] = 1.0 / 36.0
    c[6, 0] = -1
    c[6, 1] = 1
    w[6] = 1.0 / 36.0
    c[7, 0] = -1
    c[7, 1] = -1
    w[7] = 1.0 / 36.0
    c[8, 0] = 1
    c[8, 1] = -1
    w[8] = 1.0 / 36.0

    # ------------------------------------------------------------------
    # Equilibrium distribution f^eq
    # ------------------------------------------------------------------
    # After a collision, populations relax towards a local Maxwell-Boltzmann
    # equilibrium.  For low Mach-number flows the second-order expansion is:
    #
    #   f^eq_i = w_i * rho * [1  +  (c_i·u)/cs²
    #                              +  (c_i·u)²/(2 cs⁴)
    #                              -  u²/(2 cs²)]
    #
    # where rho is the local density and u=(ux,uy) is the local velocity.
    # The @jit decorator (Numba) compiles this inner loop to native machine
    # code for speed, since it is called at every lattice node every timestep.
    @jit
    def compute_feq(feq, rho, ux, uy, c, w):
        # u² / cs²  — magnitude term, same for all directions
        uu = (ux**2 + uy**2) * (1.0 / cs2)

        for ip in range(Q):
            # (c_i · u) / cs²  — directional projection
            cu = (c[ip, 0] * ux[:, :] + c[ip, 1] * uy[:, :]) * (1.0 / cs2)

            # Full second-order equilibrium population
            feq[:, :, ip] = w[ip] * rho * (1.0 + cu + 0.5 * (cu * cu - uu))

        return feq

    return c, w, cs2, compute_feq


# ---------------------------------------------------------------------------
# D4 symmetry helpers
# ---------------------------------------------------------------------------
# The square lattice has the dihedral symmetry group D4: 4 rotations (0°, 90°,
# 180°, 270°) and 4 reflections.  A physically correct collision operator must
# be equivariant under these 8 transforms — if you rotate the input populations
# by 90°, the output should rotate by 90° too.
#
# The functions below implement these transforms directly on the 9-component
# population vector.  The rest population (index 0) is invariant under all
# symmetries.  The axis-aligned populations (indices 1–4) form one orbit, and
# the diagonal populations (indices 5–8) form another; a cyclic roll on each
# group independently implements a 90° lattice rotation.


def LBrot90(f, k=1):
    """Rotate the D2Q9 population vector by k×90° counter-clockwise.

    f : tensor of shape (batch, 9)
    k : number of 90° rotation steps (positive = CCW)
    """
    # Index 0 (rest) is unchanged.
    # Indices 1–4 (axis-aligned) and 5–8 (diagonal) each cycle as a group.
    return tf.concat(
        [f[:, 0, None], tf.roll(f[:, 1:5], k, axis=-1), tf.roll(f[:, 5:], k, axis=-1)],
        axis=-1,
    )


def LBmirror(f):
    """Reflect the D2Q9 population vector across the x-axis (swap North↔South).

    This swaps direction indices so that populations moving in the +y direction
    are exchanged with their -y counterparts:
        2 (N) ↔ 4 (S),  5 (NE) ↔ 8 (SE),  6 (NW) ↔ 7 (SW)
    """
    return tf.concat(
        [
            f[:, 0, None],  # rest — unchanged
            f[:, 1, None],  # East — unchanged (on mirror axis)
            f[:, 4, None],  # was South → now North
            f[:, 3, None],  # West — unchanged (on mirror axis)
            f[:, 2, None],  # was North → now South
            f[:, 8, None],  # was SE → now NE
            f[:, 7, None],  # was SW → now NW
            f[:, 6, None],  # was NW → now SW
            f[:, 5, None],  # was NE → now SE
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# D4Symmetry / D4AntiSymmetry Keras layers
# ---------------------------------------------------------------------------
# These two layers are used together to enforce equivariance in the neural
# network without hard-coding the symmetry into the architecture.
#
# Pattern (group-equivariant lift/pool):
#   1. D4Symmetry  — "lift": given one input, produce all 8 group-transformed
#      copies so the network sees every orientation.
#   2. Process each copy through the same (shared-weight) sub-network.
#   3. D4AntiSymmetry — "project": undo the transform on each output and
#      average, so the final result is invariant (or equivariant) by construction.


@keras.saving.register_keras_serializable(package="lbm")
class D4Symmetry(keras.layers.Layer):
    """Lift a single population vector to all 8 D4-transformed copies.

    Input  : tensor of shape (batch, 9)
    Output : list of 8 tensors, each of shape (batch, 9), corresponding to
             0°, 90°, 180°, 270° rotations and their x-axis mirror images.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        y = [
            x,  # identity (0°)
            LBrot90(x, k=1),  # 90° CCW
            LBrot90(x, k=2),  # 180°
            LBrot90(x, k=3),  # 270° CCW
            LBmirror(x),  # mirror
            LBmirror(LBrot90(x, k=1)),  # mirror ∘ 90°
            LBmirror(LBrot90(x, k=2)),  # mirror ∘ 180°
            LBmirror(LBrot90(x, k=3)),  # mirror ∘ 270°
        ]
        return y


@keras.saving.register_keras_serializable(package="lbm")
class D4AntiSymmetry(keras.layers.Layer):
    """Undo each D4 transform on the corresponding processed output.

    This is the inverse of D4Symmetry: it maps the 8 transformed outputs back
    to the original orientation so they can be meaningfully averaged.

    Input  : list of 8 tensors (one per group element), shape (batch, 9) each
    Output : list of 8 tensors in the canonical (identity) orientation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x):
        y = [
            x[0],  # identity — no transform needed
            LBrot90(x[1], k=-1),  # undo 90° CCW → rotate 90° CW
            LBrot90(x[2], k=-2),  # undo 180°
            LBrot90(x[3], k=-3),  # undo 270° CCW
            LBmirror(x[4]),  # mirror is its own inverse
            LBrot90(LBmirror(x[5]), k=-1),
            LBrot90(LBmirror(x[6]), k=-2),
            LBrot90(LBmirror(x[7]), k=-3),
        ]
        return y


# ---------------------------------------------------------------------------
# AlgReconstruction
# ---------------------------------------------------------------------------


@keras.saving.register_keras_serializable(package="lbm")
class AlgReconstruction(keras.layers.Layer):
    """Recover the full 9-component population from a symmetry-reduced prediction.

    Background
    ----------
    The D4 symmetry of the square lattice means that some of the 9 populations
    are not independent: once 6 of the 9 are known, the remaining 3 can be
    derived from the conservation laws (mass and two momentum components):
        Σ_i f_i         = rho   (mass)
        Σ_i f_i c_{ix}  = rho*ux (x-momentum)
        Σ_i f_i c_{iy}  = rho*uy (y-momentum)

    The network therefore only predicts a reduced set of populations (fpred).
    This layer uses the three conservation constraints to algebraically solve
    for the three missing components (indices 2, 5, 8) and reconstructs the
    full post-collision population.

    Parameters
    ----------
    fpre  : pre-collision populations  (batch, 9) — provides the reference values
    fpred : network output             (batch, 9) — the predicted correction

    Returns
    -------
    Tensor of shape (batch, 9) — the physically consistent post-collision populations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, fpre, fpred):
        # Difference between predicted and pre-collision populations
        df = fpred - fpre

        # Solve for the three constrained directions (2, 5, 8) so that
        # mass and momentum are conserved exactly.
        # These linear expressions come from substituting the stencil weights
        # and velocity vectors into the three conservation equations and solving
        # for df[2], df[5], df[8] given the other six df values.
        df2 = -(df[:, 0] + 2 * df[:, 3] + df[:, 4] + 2 * df[:, 6] + 2 * df[:, 7])
        df5 = 0.5 * (df[:, 0] + 3 * df[:, 3] + 2 * df[:, 4] + 2 * df[:, 6] + 4 * df[:, 7] - df[:, 1])
        df8 = -0.5 * (df[:, 0] + df[:, 1] + df[:, 3] + 2 * df[:, 4] + 2 * df[:, 7])

        # Reassemble the full correction vector with the reconstructed directions
        df = tf.concat(
            [
                df[:, 0, None],
                df[:, 1, None],
                df2[:, None],  # reconstructed
                df[:, 3, None],
                df[:, 4, None],
                df5[:, None],  # reconstructed
                df[:, 6, None],
                df[:, 7, None],
                df8[:, None],  # reconstructed
            ],
            axis=-1,
        )

        # Add the correction back to the pre-collision state
        return fpre + df
