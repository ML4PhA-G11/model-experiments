#!/usr/bin/env python3
"""Verify a trained model's adherence to D4 symmetry and LBM conservation constraints.

Usage:
    python constraints_check.py                         # latest d4equivariant run
    python constraints_check.py --model lenn            # specify model name
    python constraints_check.py --run-dir path/         # explicit run directory
    python constraints_check.py --dataset data.npz      # reuse an existing dataset
    python constraints_check.py --n-samples 2000        # sample count (default: 1000)
    python constraints_check.py --tol-conservation 1e-9 --tol-symmetry 1e-5
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, cast

from keras import Model, models
import numpy as np

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts-run-all-tensorflow"

# D2Q9 velocity vectors — must match lbm_ml/lattice/stencil.py
C = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Numpy D4 group operations (mirror the keras ops in symmetry.py)
# ---------------------------------------------------------------------------


def rot90_np(f: np.ndarray, k: int = 1) -> np.ndarray:
    """Rotate D2Q9 population vector by k×90° CCW.  f: (N, 9)."""
    k = k % 4
    return np.concatenate(
        [f[:, :1], np.roll(f[:, 1:5], k, axis=-1), np.roll(f[:, 5:], k, axis=-1)],
        axis=-1,
    )


def mirror_np(f: np.ndarray) -> np.ndarray:
    """Reflect D2Q9 population across the x-axis (N↔S swap).  f: (N, 9)."""
    return f[:, [0, 1, 4, 3, 2, 8, 7, 6, 5]]


# 8 D4 elements as (name, forward_transform) pairs — forward is applied to both
# the input and the reference output to form the equivariance residual.
D4_ELEMENTS: list[tuple[str, Callable]] = [
    ("identity", lambda f: f),
    ("R90", lambda f: rot90_np(f, 1)),
    ("R180", lambda f: rot90_np(f, 2)),
    ("R270", lambda f: rot90_np(f, 3)),
    ("mirror", lambda f: mirror_np(f)),
    ("mirror∘R90", lambda f: mirror_np(rot90_np(f, 1))),
    ("mirror∘R180", lambda f: mirror_np(rot90_np(f, 2))),
    ("mirror∘R270", lambda f: mirror_np(rot90_np(f, 3))),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _latest_run_dir(model_name: str) -> Path:
    matches = sorted(ARTIFACTS_DIR.glob(f"{model_name}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No runs for '{model_name}' in {ARTIFACTS_DIR}")
    return matches[-1]


def _load_model(run_dir: Path) -> Model:
    from lbm_ml.model.losses import rmsre

    model_path = run_dir / "model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"model.keras not found in {run_dir}")
    return cast(Model, models.load_model(str(model_path), custom_objects={"rmsre": rmsre}))


def _get_samples(dataset_path: Path | None, n: int) -> np.ndarray:
    """Return density-normalised fpre (N, 9) from file or freshly generated."""
    if dataset_path is not None:
        data = np.load(dataset_path, allow_pickle=True)
        fpre = data["f_pre"][:n].astype(np.float64)
    else:
        from lbm_ml.data.generation import generate_samples

        print(f"Generating {n} test samples …")
        _feq, fpre, _fpost = generate_samples(n)
        fpre = fpre.astype(np.float64)
    # Normalise exactly as done in training / simulation
    fpre /= fpre.sum(axis=1, keepdims=True)
    return fpre


def _row(name: str, errors: np.ndarray, tol: float) -> bool:
    passed = float(np.max(np.abs(errors))) <= tol
    tag = "PASS" if passed else "FAIL"
    print(
        f"  [{tag}]  {name:<30s}"
        f"  max={np.max(np.abs(errors)):.2e}"
        f"  mean={np.mean(np.abs(errors)):.2e}"
        f"  (tol {tol:.0e})"
    )
    return passed


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def run_checks(
    model_name: str = "d4equivariant",
    run_dir: Path | None = None,
    dataset_path: Path | None = None,
    n_samples: int = 1000,
    tol_conservation: float = 1e-9,
    tol_symmetry: float = 1e-5,
    batch_size: int = 512,
) -> bool:
    from keras import backend as K

    K.set_floatx("float64")

    if run_dir is None:
        run_dir = _latest_run_dir(model_name)
    print(f"Run dir  : {run_dir.name}")

    model = _load_model(run_dir)

    fpre = _get_samples(dataset_path, n_samples)
    n = fpre.shape[0]
    print(f"Samples  : {n}\n")

    fpost = np.array(model.predict(fpre, verbose=cast(str, 0), batch_size=batch_size), dtype=np.float64)

    all_passed = True

    # ── 1. Conservation ────────────────────────────────────────────────────
    print("── Conservation ──────────────────────────────────────────────────────")

    # Mass and both momentum components must be identical pre- and post-collision.
    # AlgReconstruction enforces these algebraically, so expect near-machine-epsilon errors.
    all_passed &= _row("mass  Σf_post = Σf_pre", fpost.sum(axis=1) - fpre.sum(axis=1), tol_conservation)
    all_passed &= _row("x-momentum  Σf·cx conserved", fpost @ C[:, 0] - fpre @ C[:, 0], tol_conservation)
    all_passed &= _row("y-momentum  Σf·cy conserved", fpost @ C[:, 1] - fpre @ C[:, 1], tol_conservation)

    # ── 2. D4 equivariance ─────────────────────────────────────────────────
    print("\n── D4 Equivariance  |model(g·f) − g·model(f)|  (per-sample L∞) ──────")

    # Equivariance condition: model(g(f)) == g(model(f)).
    # identity is trivially exact; real signal is in the remaining 7 elements.
    for name, g in D4_ELEMENTS:
        pred_gf = np.array(model.predict(g(fpre), verbose=cast(str, 0), batch_size=batch_size), dtype=np.float64)
        residual = np.abs(pred_gf - g(fpost)).max(axis=1)
        all_passed &= _row(name, residual, tol_symmetry)

    # ── Summary ────────────────────────────────────────────────────────────
    print()
    verdict = "ALL PASSED" if all_passed else "SOME CHECKS FAILED"
    print(f"══ Result: {verdict}")
    return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model",
        default="d4equivariant",
        help="Model name used to locate the latest run directory (default: d4equivariant)",
    )
    p.add_argument("--run-dir", default=None, help="Explicit run directory containing model.keras (overrides --model)")
    p.add_argument("--dataset", default=None, help=".npz dataset file; omit to generate fresh samples")
    p.add_argument("--n-samples", type=int, default=1000, help="Number of test samples (default: 1000)")
    p.add_argument(
        "--tol-conservation",
        type=float,
        default=1e-9,
        help="Absolute tolerance for mass / momentum checks (default: 1e-9)",
    )
    p.add_argument(
        "--tol-symmetry", type=float, default=1e-5, help="Absolute tolerance for D4 equivariance checks (default: 1e-5)"
    )
    p.add_argument("--batch-size", type=int, default=512, help="Prediction batch size (default: 512)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = run_checks(
        model_name=args.model,
        run_dir=Path(args.run_dir) if args.run_dir else None,
        dataset_path=Path(args.dataset) if args.dataset else None,
        n_samples=args.n_samples,
        tol_conservation=args.tol_conservation,
        tol_symmetry=args.tol_symmetry,
        batch_size=args.batch_size,
    )
    sys.exit(0 if ok else 1)
