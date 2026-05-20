from pathlib import Path

import numpy as np

from lbm_ml.lattice.stencil import LB_stencil

# D2Q9 stencil — used for projecting non-equilibrium populations
_c, _w, _cs2, _compute_feq = LB_stencil()


def compute_rho_u(num_samples, rho_min=0.95, rho_max=1.05, u_abs_min=0.0, u_abs_max=0.01):
    """Sample random macroscopic density and velocity fields."""
    rho   = np.random.uniform(rho_min, rho_max, size=num_samples)
    u_abs = np.random.uniform(u_abs_min, u_abs_max, size=num_samples)
    theta = np.random.uniform(0, 2 * np.pi, size=num_samples)
    ux = u_abs * np.cos(theta)
    uy = u_abs * np.sin(theta)
    u  = np.array([ux, uy]).transpose()
    return rho, u


def compute_f_rand(num_samples, sigma_min, sigma_max):
    """Generate random non-equilibrium perturbations with zero conserved moments."""
    Q  = 9
    K0 = 1 / 9.
    K1 = 1 / 6.

    f_rand = np.zeros((num_samples, Q))

    if sigma_min == sigma_max:
        sigma = sigma_min * np.ones(num_samples)
    else:
        sigma = np.random.uniform(sigma_min, sigma_max, size=num_samples)

    for i in range(num_samples):
        f_rand[i, :] = np.random.normal(0, sigma[i], size=(1, Q))

        rho_hat = np.sum(f_rand[i, :])
        ux_hat  = np.sum(f_rand[i, :] * _c[:, 0])
        uy_hat  = np.sum(f_rand[i, :] * _c[:, 1])

        # Project out conserved moments so f_neq has zero mass and momentum
        f_rand[i, :] = f_rand[i, :] - K0 * rho_hat - K1 * ux_hat * _c[:, 0] - K1 * uy_hat * _c[:, 1]

    return f_rand


def compute_f_pre_f_post(f_eq, f_neq, tau_min=1, tau_max=1):
    """Apply BGK relaxation: f_post = f_pre + (1/tau) * (f_eq - f_pre)."""
    tau    = np.random.uniform(tau_min, tau_max, size=f_eq.shape[0])
    f_pre  = f_eq + f_neq
    f_post = f_pre + 1 / tau[:, None] * (f_eq - f_pre)
    return tau, f_pre, f_post


def delete_negative_samples(n_samples, f_eq, f_pre, f_post):
    """Remove samples where any population is negative (unphysical)."""
    i_neg_f_eq   = np.where(np.sum(f_eq   < 0, axis=1) > 0)[0]
    i_neg_f_pre  = np.where(np.sum(f_pre  < 0, axis=1) > 0)[0]
    i_neg_f_post = np.where(np.sum(f_post < 0, axis=1) > 0)[0]
    i_neg_f = np.concatenate((i_neg_f_pre, i_neg_f_post, i_neg_f_eq))
    f_eq   = np.delete(np.copy(f_eq),   i_neg_f, 0)
    f_pre  = np.delete(np.copy(f_pre),  i_neg_f, 0)
    f_post = np.delete(np.copy(f_post), i_neg_f, 0)
    return f_eq, f_pre, f_post


def load_data(fname):
    """Load a training dataset from an .npz file."""
    data  = np.load(fname, allow_pickle=True)
    feq   = data['f_eq']
    fpre  = data['f_pre']
    fpost = data['f_post']
    return feq, fpre, fpost


def generate_dataset(
    dataset_path: Path,
    n_samples: int = 100_000,
    rho_min: float = 0.95,
    rho_max: float = 1.05,
    u_abs_min: float = 1e-15,
    u_abs_max: float = 0.01,
    sigma_min: float = 1e-15,
    sigma_max: float = 5e-4,
    tau_min: float = 1.0,
    tau_max: float = 1.0,
) -> Path:
    """Generate and save a BGK collision dataset to *dataset_path*.

    Loops until *n_samples* physically valid (non-negative) population vectors
    have been collected, then writes an .npz file with keys f_pre, f_post, f_eq.

    Returns
    -------
    Path to the saved .npz file.
    """
    Q = 9
    c, w, cs2, compute_feq = LB_stencil()

    fPreLst  = np.empty((n_samples, Q))
    fPostLst = np.empty((n_samples, Q))
    fEqLst   = np.empty((n_samples, Q))

    idx = 0
    while idx < n_samples:
        rho, u = compute_rho_u(n_samples, rho_min, rho_max, u_abs_min, u_abs_max)

        rho = rho[:, np.newaxis]
        ux  = u[:, 0][:, np.newaxis]
        uy  = u[:, 1][:, np.newaxis]

        f_eq  = np.zeros((n_samples, 1, Q))
        f_eq  = compute_feq(f_eq, rho, ux, uy, c, w)[:, 0, :]

        f_neq = compute_f_rand(n_samples, sigma_min, sigma_max)

        _tau, f_pre, f_post = compute_f_pre_f_post(f_eq, f_neq, tau_min, tau_max)

        f_eq, f_pre, f_post = delete_negative_samples(n_samples, f_eq, f_pre, f_post)

        non_negatives = f_pre.shape[0]
        idx1        = min(idx + non_negatives, n_samples)
        to_be_added = min(n_samples - idx, non_negatives)

        fPreLst[ idx:idx1] = f_pre[ :to_be_added]
        fPostLst[idx:idx1] = f_post[:to_be_added]
        fEqLst[  idx:idx1] = f_eq[  :to_be_added]

        idx += non_negatives

    np.savez(dataset_path, f_pre=fPreLst, f_post=fPostLst, f_eq=fEqLst)
    return dataset_path
