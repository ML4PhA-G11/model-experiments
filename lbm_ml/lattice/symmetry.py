import numpy as np
import keras


def compute_d2q9_orbit_indices() -> np.ndarray:
    """Return a (9, 9) int array mapping each weight-matrix entry to its D4 orbit.

    Under the D4 symmetry group of the square lattice, the equivariance constraint
    AP = PA forces A[i,j] = A[π(i), π(j)] for every group generator permutation π.
    This function finds all such orbits of pairs (i,j) using BFS over the two
    generators R (90° CCW rotation) and S (x-axis mirror), yielding 15 orbits
    for D2Q9 — the 15 free weight parameters per (C_in, C_out) channel pair.

    The permutation arrays are derived directly from LBrot90(k=1) and LBmirror:
      new[i] = old[perm[i]], so the equivariance condition is A[i,j] = A[π(i),π(j)].
    """
    Q = 9
    perm_R = np.array([0, 4, 1, 2, 3, 8, 5, 6, 7])  # LBrot90(k=1)
    perm_S = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5])  # LBmirror

    orbit_map = np.full((Q, Q), -1, dtype=np.int32)
    orbit_id = 0

    for si in range(Q):
        for sj in range(Q):
            if orbit_map[si, sj] != -1:
                continue
            queue = [(si, sj)]
            orbit_map[si, sj] = orbit_id
            while queue:
                i, j = queue.pop()
                for perm in (perm_R, perm_S):
                    ni, nj = int(perm[i]), int(perm[j])
                    if orbit_map[ni, nj] == -1:
                        orbit_map[ni, nj] = orbit_id
                        queue.append((ni, nj))
            orbit_id += 1

    return orbit_map  # values in [0, 14], 15 distinct orbits


def compute_d2q9_bias_orbit_indices() -> np.ndarray:
    """Return a (9,) int array mapping each bias component to its D4 orbit.

    The three orbits are: {0} (rest), {1,2,3,4} (axis-aligned), {5,6,7,8} (diagonal).
    """
    Q = 9
    perm_R = np.array([0, 4, 1, 2, 3, 8, 5, 6, 7])
    perm_S = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5])

    orbit_map = np.full(Q, -1, dtype=np.int32)
    orbit_id = 0

    for s in range(Q):
        if orbit_map[s] != -1:
            continue
        queue = [s]
        orbit_map[s] = orbit_id
        while queue:
            i = queue.pop()
            for perm in (perm_R, perm_S):
                ni = int(perm[i])
                if orbit_map[ni] == -1:
                    orbit_map[ni] = orbit_id
                    queue.append(ni)
        orbit_id += 1

    return orbit_map  # values in [0, 2], 3 distinct orbits


def LBrot90(f, k=1):
    """Rotate the D2Q9 population vector by k×90° counter-clockwise.

    f : tensor of shape (batch, 9)
    k : number of 90° rotation steps (positive = CCW)
    """
    # Index 0 (rest) is unchanged.
    # Indices 1–4 (axis-aligned) and 5–8 (diagonal) each cycle as a group.
    return keras.ops.concatenate(
        [f[:, 0, None], keras.ops.roll(f[:, 1:5], k, axis=-1), keras.ops.roll(f[:, 5:], k, axis=-1)],
        axis=-1,
    )


def LBmirror(f):
    """Reflect the D2Q9 population vector across the x-axis (swap North↔South).

    This swaps direction indices so that populations moving in the +y direction
    are exchanged with their -y counterparts:
        2 (N) ↔ 4 (S),  5 (NE) ↔ 8 (SE),  6 (NW) ↔ 7 (SW)
    """
    return keras.ops.concatenate(
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
# D4 symmetry helpers
# ---------------------------------------------------------------------------
# The square lattice has the dihedral symmetry group D4: 4 rotations (0°, 90°,
# 180°, 270°) and 4 reflections.  A physically correct collision operator must
# be equivariant under these 8 transforms — if you rotate the input populations
# by 90°, the output should rotate by 90° too.
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
        return [
            x,  # identity (0°)
            LBrot90(x, k=1),  # 90° CCW
            LBrot90(x, k=2),  # 180°
            LBrot90(x, k=3),  # 270° CCW
            LBmirror(x),  # mirror
            LBmirror(LBrot90(x, k=1)),  # mirror ∘ 90°
            LBmirror(LBrot90(x, k=2)),  # mirror ∘ 180°
            LBmirror(LBrot90(x, k=3)),  # mirror ∘ 270°
        ]


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
        return [
            x[0],  # identity — no transform needed
            LBrot90(x[1], k=-1),  # undo 90° CCW → rotate 90° CW
            LBrot90(x[2], k=-2),  # undo 180°
            LBrot90(x[3], k=-3),  # undo 270° CCW
            LBmirror(x[4]),  # mirror is its own inverse
            LBrot90(LBmirror(x[5]), k=-1),
            LBrot90(LBmirror(x[6]), k=-2),
            LBrot90(LBmirror(x[7]), k=-3),
        ]


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
        df = keras.ops.concatenate(
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


@keras.saving.register_keras_serializable(package="lbm")
class SymmetricAlgReconstruction(keras.layers.Layer):
    """Updated version of AlgReconstruction that projects onto the conservation manifold in a D4-equivariant way.

    Project the network prediction onto the conservation manifold.

    Enforces mass and momentum conservation by applying the minimum-norm
    (pseudo-inverse) correction to the predicted correction vector df = fpred − fpre:

        df_corrected[i] = df[i] − Δmass/9 − cx[i]·Δpx/6 − cy[i]·Δpy/6

    where Δmass = Σ df_j, Δpx = Σ df_j cx_j, Δpy = Σ df_j cy_j, and the
    denominators come from C Cᵀ = diag(9, 6, 6) for the D2Q9 stencil.

    This projection is D4-equivariant because it depends only on the conservation
    defects (scalars/vectors under D4) and the velocity components, not on any
    privileged choice of population indices.  The previous implementation fixed
    indices 2, 5, 8 (North, NE, SE), which broke equivariance.

    Parameters
    ----------
    fpre  : pre-collision populations  (batch, 9) — provides the reference values
    fpred : network output             (batch, 9) — the predicted correction

    Returns
    -------
    Tensor of shape (batch, 9) — the physically consistent post-collision populations.
    """

    # D2Q9 velocity components, shape (9,)
    _CX = keras.ops.cast([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype="float64")
    _CY = keras.ops.cast([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype="float64")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, fpre, fpred):
        cx = keras.ops.cast(self._CX, fpred.dtype)
        cy = keras.ops.cast(self._CY, fpred.dtype)

        # Difference between predicted and pre-collision populations
        df = fpred - fpre

        # Conservation defects: how much df violates each law
        d_mass = keras.ops.sum(df, axis=-1, keepdims=True)  # (batch, 1)
        d_px = keras.ops.sum(df * cx, axis=-1, keepdims=True)  # (batch, 1)
        d_py = keras.ops.sum(df * cy, axis=-1, keepdims=True)  # (batch, 1)

        # Subtract the minimum-norm correction: C⁺ (C df)
        # C⁺[i] = [1/9, cx[i]/6, cy[i]/6]  from (C Cᵀ)⁻¹ = diag(1/9, 1/6, 1/6)
        df_corrected = df - d_mass / 9 - d_px * cx / 6 - d_py * cy / 6

        return self._apply(fpre, df_corrected)

    def _apply(self, fpre, df_corrected):
        return fpre + df_corrected


@keras.saving.register_keras_serializable(package="lbm")
class PositivitySafeAlgReconstruction(SymmetricAlgReconstruction):
    """Conservation projection with a positivity-safe scaling fallback.

    Identical to SymmetricAlgReconstruction except the correction is scaled
    by the largest α ∈ [0, 1] that keeps all f_i ≥ epsilon.  When the full
    correction is already safe, α = 1 and the result is identical.

    Parameters
    ----------
    epsilon : float
        Minimum allowed population value (default 0.0).
    """

    def __init__(self, epsilon: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _apply(self, fpre, df_corrected):
        eps = keras.ops.cast(self.epsilon, df_corrected.dtype)
        # α_i = (fpre[i] - ε) / (-df_corrected[i]) for directions where correction < 0
        safe_denom = keras.ops.where(df_corrected < 0, -df_corrected, keras.ops.ones_like(df_corrected))
        alpha_i = keras.ops.where(
            df_corrected < 0,
            (fpre - eps) / safe_denom,
            keras.ops.ones_like(df_corrected),
        )
        alpha = keras.ops.clip(keras.ops.min(alpha_i, axis=-1, keepdims=True), 0.0, 1.0)
        return fpre + alpha * df_corrected

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon})
        return config


@keras.saving.register_keras_serializable(package="lbm")
class MultiplicativeAlgReconstruction(PositivitySafeAlgReconstruction):
    """Momentum-conserving correction via multiplicative scaling of the NN output.

    Unlike SymmetricAlgReconstruction (additive correction anchored to fpre),
    this layer starts from fpred (assumed positive, e.g. softmax output) and
    applies a multiplicative weight:

        f_post[i] = fpred[i] · (1 + α·(λ₁·cx[i] + λ₂·cy[i]))

    λ₁, λ₂ are solved from a 2×2 linear system that enforces momentum
    conservation (mass is already conserved via the normalize/denormalize trick
    with softmax).  α ∈ [0,1] is then scaled to keep (1 + α·δ[i]) ≥ epsilon.

    Because positivity depends only on fpred (not fpre), it is guaranteed even
    when fpre itself contains negatives from upstream simulation drift — the key
    advantage over PositivitySafeAlgReconstruction.

    Inherits _CX, _CY, epsilon, and get_config from PositivitySafeAlgReconstruction.
    """

    def call(self, fpre, fpred):
        cx = keras.ops.cast(self._CX, fpred.dtype)
        cy = keras.ops.cast(self._CY, fpred.dtype)
        eps = keras.ops.cast(self.epsilon, fpred.dtype)

        # Momentum defects: how much fpred differs from fpre
        d_px = keras.ops.sum((fpre - fpred) * cx, axis=-1, keepdims=True)  # (batch, 1)
        d_py = keras.ops.sum((fpre - fpred) * cy, axis=-1, keepdims=True)  # (batch, 1)

        # 2×2 system A·[λ₁, λ₂]ᵀ = [d_px, d_py]ᵀ,  A[i,j] = Σ_k fpred[k]·c_i[k]·c_j[k]
        A00 = keras.ops.sum(fpred * cx * cx, axis=-1, keepdims=True)
        A01 = keras.ops.sum(fpred * cx * cy, axis=-1, keepdims=True)
        A11 = keras.ops.sum(fpred * cy * cy, axis=-1, keepdims=True)

        # Analytic 2×2 inverse; guard against near-singular (e.g. uniform flow)
        det = A00 * A11 - A01 * A01
        det_safe = keras.ops.where(keras.ops.abs(det) > 1e-12, det, keras.ops.ones_like(det))
        lam1 = (A11 * d_px - A01 * d_py) / det_safe
        lam2 = (A00 * d_py - A01 * d_px) / det_safe
        singular = keras.ops.abs(det) <= 1e-12
        lam1 = keras.ops.where(singular, keras.ops.zeros_like(lam1), lam1)
        lam2 = keras.ops.where(singular, keras.ops.zeros_like(lam2), lam2)

        delta = lam1 * cx + lam2 * cy  # (batch, 9)

        # Largest α ∈ [0,1] such that (1 + α·δ[i]) ≥ epsilon for all i
        safe_denom = keras.ops.where(delta < 0, -delta, keras.ops.ones_like(delta))
        alpha_i = keras.ops.where(
            delta < 0,
            (1.0 - eps) / safe_denom,
            keras.ops.ones_like(delta),
        )
        alpha = keras.ops.clip(keras.ops.min(alpha_i, axis=-1, keepdims=True), 0.0, 1.0)

        return fpred * (1.0 + alpha * delta)
