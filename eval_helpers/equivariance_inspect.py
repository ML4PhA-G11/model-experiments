"""Stage-by-stage equivariance diagnostics for LENN models.

Isolates where D4 equivariance violations enter the pipeline by testing:
  Stage 1 — LENN backbone (before AlgReconstruction)
  Stage 2 — AlgReconstruction alone, given perfectly equivariant input
  Stage 3 — Full model (backbone + AlgReconstruction)

Usage:
    from eval_helpers.equivariance_inspect import inspect_lenn_equivariance
    from keras import models

    model = models.load_model("path/model.keras", ...)
    fpre  = ...  # (N, 9) normalised float64 array
    inspect_lenn_equivariance(model, fpre)
"""

from typing import cast

import numpy as np
import keras

from lbm_ml.lattice.symmetry import AlgReconstruction

# D4 permutations on Q=9 D2Q9 populations
# perm[i] = old index whose value moves to new index i after the transform
_PERM_R = np.array([0, 4, 1, 2, 3, 8, 5, 6, 7])  # 90° CCW rotation
_PERM_S = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5])  # x-axis mirror

D4_PERMS: dict[str, np.ndarray] = {
    "identity": np.arange(9),
    "R90": _PERM_R,
    "R180": _PERM_R[_PERM_R],
    "R270": _PERM_R[_PERM_R[_PERM_R]],
    "mirror": _PERM_S,
    "mirror∘R90": _PERM_S[_PERM_R],
    "mirror∘R180": _PERM_S[_PERM_R[_PERM_R]],
    "mirror∘R270": _PERM_S[_PERM_R[_PERM_R[_PERM_R]]],
}


def _perm(f: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Apply a D4 permutation along the Q axis. Handles (N,Q) and (N,Q,C)."""
    return f[:, p] if f.ndim == 2 else f[:, p, :]


def _row(name: str, g_name: str, residual: float, tol: float = 1e-12) -> None:
    tag = "PASS" if residual <= tol else "FAIL"
    print(f"  [{tag}]  {name:<35s} {g_name:<15s}  max={residual:.2e}")


def _check_equivariance(
    stage_name: str,
    fn,
    fpre: np.ndarray,
    tol: float = 1e-12,
) -> dict[str, float]:
    """Check model(g·f) == g·model(f) for all 8 D4 elements.

    fn must accept (N, 9) float64 and return an array with Q as axis 1.
    """
    out_f = np.array(fn(fpre))
    results: dict[str, float] = {}
    for gname, perm in D4_PERMS.items():
        out_gf = np.array(fn(_perm(fpre, perm)))  # model(g·f)
        g_out_f = _perm(out_f, perm)  # g·model(f)
        residual = float(np.abs(out_gf - g_out_f).max())
        results[gname] = residual
        _row(stage_name, gname, residual, tol)
    return results


def _backbone_model(model: keras.Model) -> keras.Model:
    """Return a sub-model that outputs just before AlgReconstruction.

    Walks the layer list and returns the output of the last layer before
    the AlgReconstruction node.
    """
    alg_layer = next((l for l in model.layers if isinstance(l, AlgReconstruction)), None)
    if alg_layer is None:
        raise ValueError("No AlgReconstruction layer found in model")

    # In Keras functional models the layer records its inbound call nodes.
    # The second argument to AlgReconstruction()(fpre, fpred) is fpred.
    try:
        fpred_tensor = alg_layer._inbound_nodes[0].input_tensors[1]
        return keras.Model(inputs=model.input, outputs=fpred_tensor)
    except Exception as e:
        raise RuntimeError(f"Could not extract backbone output tensor: {e}. " "Try Keras >= 3.0 functional API.") from e


def inspect_lenn_equivariance(
    model: keras.Model,
    fpre: np.ndarray,
    batch_size: int = 512,
    tol_backbone: float = 1e-12,
    tol_full: float = 1e-12,
) -> None:
    """Print a per-stage equivariance breakdown for a LENN model.

    Args:
        model:        Compiled LENN keras.Model.
        fpre:         (N, 9) float64 normalised pre-collision populations.
        batch_size:   Prediction batch size.
        tol_backbone: Pass threshold for the backbone stage.
        tol_full:     Pass threshold for the full model stage.
    """
    keras.backend.set_floatx("float64")

    def predict(m, f):
        return m.predict(f, verbose=cast(str, 0), batch_size=batch_size)

    # ── Stage 1: LENN backbone ─────────────────────────────────────────────
    print("── Stage 1: LENN backbone (before AlgReconstruction) ─────────────")
    try:
        backbone = _backbone_model(model)
        r1 = _check_equivariance("backbone", lambda f: predict(backbone, f), fpre, tol_backbone)
        worst1 = max(r1.values())
        print(f"  → worst residual: {worst1:.2e}\n")
    except Exception as e:
        print(f"  [SKIP] {e}\n")
        worst1 = None

    # ── Stage 2: AlgReconstruction in isolation ────────────────────────────
    print("── Stage 2: AlgReconstruction alone (equivariant input from backbone) ─")
    try:
        backbone = _backbone_model(model)
        fpred = np.array(predict(backbone, fpre), dtype=np.float64)

        alg_layer = next(l for l in model.layers if isinstance(l, AlgReconstruction))

        # Manual loop: test AlgReconstruction(g·fpre, g·fpred) == g·AlgReconstruction(fpre, fpred)
        out_f = np.array(
            alg_layer(
                keras.ops.convert_to_tensor(fpre, dtype="float64"),
                keras.ops.convert_to_tensor(fpred, dtype="float64"),
            )
        )
        r2: dict[str, float] = {}
        for gname, perm in D4_PERMS.items():
            g_fpre = _perm(fpre, perm)
            g_fpred = _perm(fpred, perm)
            out_gf = np.array(
                alg_layer(
                    keras.ops.convert_to_tensor(g_fpre, dtype="float64"),
                    keras.ops.convert_to_tensor(g_fpred, dtype="float64"),
                )
            )
            g_out_f = _perm(np.array(out_f), perm)
            residual = float(np.abs(out_gf - g_out_f).max())
            r2[gname] = residual
            _row("AlgReconstruction", gname, residual, tol_full)
        worst2 = max(r2.values())
        print(f"  → worst residual: {worst2:.2e}\n")
    except Exception as e:
        print(f"  [SKIP] {e}\n")
        worst2 = None

    # ── Stage 3: Full model ────────────────────────────────────────────────
    print("── Stage 3: Full model (backbone + AlgReconstruction) ────────────")
    r3 = _check_equivariance("full model", lambda f: predict(model, f), fpre, tol_full)
    worst3 = max(r3.values())
    print(f"  → worst residual: {worst3:.2e}\n")

    # ── Summary ───────────────────────────────────────────────────────────
    print("── Summary ────────────────────────────────────────────────────────")
    if worst1 is not None:
        print(f"  Backbone (LENN layers only):        {worst1:.2e}")
    if worst2 is not None:
        print(f"  AlgReconstruction (isolated):       {worst2:.2e}")
    print(f"  Full model:                         {worst3:.2e}")

    if worst1 is not None and worst2 is not None:
        if worst3 <= 1e-12:
            print("\n  → All stages at machine precision. ✓")
        elif worst1 < 1e-12 and worst2 > 1e-10:
            print("\n  → Violation originates in AlgReconstruction (not D4-equivariant).")
        elif worst1 > 1e-10:
            print("\n  → Violation already present in the LENN backbone.")
