"""Load training data produced by the Karman-vortex-street LBM *simulator*.

This is the counterpart to :mod:`lbm_ml.data.generation`, which fabricates
*synthetic* BGK collision pairs (the Taylor-Green-style dataset shipped with the
paper).  Here we instead consume the **real** distribution functions dumped by
the simulator ``lbm_karman-ng.py`` when run with ``--save-every N``.

Simulator output format (the assumption this module is built on)
----------------------------------------------------------------
For every saved timestep ``s`` the simulator writes *two* NumPy ``.npy`` files
into its output directory, right after the BGK collision step and before
bounce-back / streaming::

    fpre_<s:06d>.npy   # array f      — pre-collision  populations
    fpost_<s:06d>.npy  # array f_out  — post-collision populations  (= f - (f - feq)/tau)

Each array has shape ``(Nx, Ny, Q)`` with ``Q = 9`` (D2Q9), dtype float64.
For the default Karman geometry (2.2m x 0.41m at 250 lu/m) that is
``(550, 102, 9)``.  The last axis is the population vector in the **same**
channel order the network expects::

    [0 rest, 1 E, 2 N, 3 W, 4 S, 5 NE, 6 NW, 7 SW, 8 SE]

so no channel re-mapping is needed — critical, because the D4-equivariant
layers in :mod:`lbm_ml.lattice.symmetry` assume exactly this ordering.

What the trainer expects (the gap this module bridges)
------------------------------------------------------
``run_all.py`` / :func:`lbm_ml.data.generation.load_data` expect a flat list of
collision pairs: arrays of shape ``(n_samples, Q)`` for ``f_pre``, ``f_post``
and ``f_eq``.  This module flattens the per-step ``(Nx, Ny, Q)`` snapshots into
``(Nx*Ny, Q)`` rows, stacks them across timesteps, optionally sub-samples
(the full per-step dataset is enormous), drops unphysical negative samples, and
reconstructs ``f_eq`` (which the simulator does not save) from the macroscopic
moments of ``f_pre``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from lbm_ml.lattice.stencil import LB_stencil

_c, _w, _cs2, _compute_feq = LB_stencil()

# Matches the simulator's "fpre_000123.npy" naming and captures the step number.
_FPRE_RE = re.compile(r"^fpre_(\d+)\.npy$")


def _equilibrium_from_populations(f: np.ndarray) -> np.ndarray:
    """Reconstruct the D2Q9 equilibrium f_eq from a batch of populations.

    The simulator only saves f_pre / f_post, not f_eq.  We recover it from the
    conserved moments of f_pre (density and momentum) using the same
    second-order equilibrium the rest of the codebase uses.

    Parameters
    ----------
    f : (N, 9) array of populations.

    Returns
    -------
    (N, 9) array of equilibrium populations.
    """
    rho = np.sum(f, axis=1)
    # Guard against division by zero on (effectively) empty nodes.
    safe_rho = np.where(rho == 0.0, 1.0, rho)
    ux = np.einsum("nq,q->n", f, _c[:, 0]) / safe_rho
    uy = np.einsum("nq,q->n", f, _c[:, 1]) / safe_rho

    feq_buf = np.zeros((f.shape[0], 1, 9))
    feq = _compute_feq(feq_buf, rho[:, None], ux[:, None], uy[:, None], _c, _w)
    return feq[:, 0, :]


def _discover_steps(data_dir: Path) -> list[tuple[int, Path, Path]]:
    """Return sorted ``(step, fpre_path, fpost_path)`` triples found in *data_dir*.

    A step is only included when *both* its fpre and fpost files exist.
    """
    triples: list[tuple[int, Path, Path]] = []
    for fpre_path in sorted(data_dir.glob("fpre_*.npy")):
        m = _FPRE_RE.match(fpre_path.name)
        if not m:
            continue
        step = int(m.group(1))
        fpost_path = fpre_path.with_name(fpre_path.name.replace("fpre_", "fpost_", 1))
        if fpost_path.exists():
            triples.append((step, fpre_path, fpost_path))
    return triples


def load_simulation_pairs(
    data_dir: str | Path,
    samples_per_step: int | None = None,
    step_stride: int = 1,
    max_steps: int | None = None,
    drop_negative: bool = True,
    seed: int | None = 0,
):
    """Load simulator collision pairs as flat ``(N, 9)`` training arrays.

    Parameters
    ----------
    data_dir
        Directory containing ``fpre_*.npy`` / ``fpost_*.npy`` pairs written by
        ``lbm_karman-ng.py --save-every N``.
    samples_per_step
        If given, randomly sub-sample this many lattice nodes from each loaded
        snapshot (without replacement).  ``None`` keeps every node — be aware a
        single 550x102 snapshot already yields 56,100 samples.
    step_stride
        Use every ``step_stride``-th saved timestep (after sorting by step).
    max_steps
        Cap the number of timesteps loaded (applied after striding).
    drop_negative
        Drop samples where any population in f_pre or f_post is negative
        (unphysical), mirroring ``generation.delete_negative_samples``.
    seed
        Seed for the per-step sub-sampling RNG (``None`` = nondeterministic).

    Returns
    -------
    (feq, fpre, fpost) : tuple of three ``(N, 9)`` float arrays.
        Ordered to match :func:`lbm_ml.data.generation.load_data`.  ``feq`` is
        reconstructed from ``fpre`` since the simulator does not save it.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Simulation data directory not found: {data_dir}")

    steps = _discover_steps(data_dir)
    if not steps:
        raise FileNotFoundError(
            f"No 'fpre_*.npy' / 'fpost_*.npy' pairs found in {data_dir}. " "Was the simulator run with --save-every N?"
        )

    steps = steps[::step_stride]
    if max_steps is not None:
        steps = steps[:max_steps]

    rng = np.random.default_rng(seed)
    fpre_chunks: list[np.ndarray] = []
    fpost_chunks: list[np.ndarray] = []
    for _step, fpre_path, fpost_path in tqdm(steps, desc="Loading simulation steps", unit="step"):
        fpre = np.load(fpre_path)
        fpost = np.load(fpost_path)
        if fpre.shape != fpost.shape or fpre.shape[-1] != 9:
            raise ValueError(
                f"Unexpected shapes for step {_step}: fpre={fpre.shape}, "
                f"fpost={fpost.shape} (expected matching (..., 9))."
            )

        # (Nx, Ny, 9) -> (Nx*Ny, 9)
        fpre = fpre.reshape(-1, 9)
        fpost = fpost.reshape(-1, 9)

        if samples_per_step is not None and samples_per_step < fpre.shape[0]:
            idx = rng.choice(fpre.shape[0], size=samples_per_step, replace=False)
            fpre = fpre[idx]
            fpost = fpost[idx]

        fpre_chunks.append(fpre)
        fpost_chunks.append(fpost)

    logger.info("Concatenating %d chunks...", len(fpre_chunks))
    fpre = np.concatenate(fpre_chunks, axis=0)
    fpost = np.concatenate(fpost_chunks, axis=0)
    logger.info("  -> %d total samples", fpre.shape[0])

    if drop_negative:
        logger.info("Filtering negative populations...")
        keep = ~((fpre < 0).any(axis=1) | (fpost < 0).any(axis=1))
        fpre, fpost = fpre[keep], fpost[keep]
        logger.info("  -> %d samples after filtering", fpre.shape[0])

    logger.info("Computing equilibrium distributions...")
    feq = _equilibrium_from_populations(fpre)
    logger.info("  -> Done.")

    logger.info(
        "Loaded simulator data from %s: %d steps -> %d collision pairs "
        "(samples_per_step=%s, step_stride=%d, drop_negative=%s)",
        data_dir, len(steps), fpre.shape[0], samples_per_step, step_stride, drop_negative,
    )
    return feq, fpre, fpost


def consolidate_to_npz(
    data_dir: str | Path,
    out_npz: str | Path,
    **load_kwargs,
) -> Path:
    """Convert a directory of simulator npy pairs into a single ``.npz`` cache.

    The resulting file uses the keys ``f_eq`` / ``f_pre`` / ``f_post`` and is
    therefore directly loadable by :func:`lbm_ml.data.generation.load_data`.
    Useful to avoid re-scanning thousands of per-step files on every run.
    """
    feq, fpre, fpost = load_simulation_pairs(data_dir, **load_kwargs)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, f_eq=feq, f_pre=fpre, f_post=fpost)
    logger.info("  Wrote consolidated dataset -> %s (%d samples)", out_npz, fpre.shape[0])
    return out_npz
