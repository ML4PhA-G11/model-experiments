#!/usr/bin/env python3
"""
evaluate_nn.py
==============
Evaluates the trained neural network collision operator against
f_pre / f_post pairs saved from the Kármán vortex street simulation.

Usage:
    python evaluate_nn.py --model-path artifacts-run-all-tensorflow/example_network.keras \
                          --data-dir output \
                          --out-dir eval_results

The script expects pairs of files named:
    fpre_XXXXXX.npy   (pre-collision distribution)
    fpost_XXXXXX.npy  (post-collision distribution, BGK ground truth)
"""

import os
import glob
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import backend as K

from utils import D4Symmetry, AlgReconstruction, D4AntiSymmetry

# ──────────────────────────────────────────────
# Helpers — must match training definitions
# ──────────────────────────────────────────────

def rmsre(y_true, y_pred):
    """Root Mean Squared Relative Error — same loss used during training."""
    return keras.backend.sqrt(
        keras.backend.mean(
            keras.backend.square((y_true - y_pred) / (y_true + keras.backend.epsilon()))
        )
    )


def normalize(f):
    """Divide each sample by its total density (same as training pipeline)."""
    norm = np.sum(f, axis=1, keepdims=True)
    return f / norm, norm


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate NN collision operator on Kármán vortex data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", type=str,
                   default="artifacts-run-all-tensorflow/example_network.keras",
                   help="Path to the saved Keras model (.keras file)")
    p.add_argument("--data-dir", type=str, default="output",
                   help="Directory containing fpre_*.npy / fpost_*.npy files")
    p.add_argument("--out-dir", type=str, default="eval_results",
                   help="Directory where evaluation plots and CSV are saved")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size for model.predict()")
    return p.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────────────────
    K.set_floatx('float64')
    print(f"Loading model from: {args.model_path}")
    model = keras.models.load_model(
        args.model_path,
        custom_objects={'rmsre': rmsre}
    )
    model.summary()

    # ── 2. Discover data files ─────────────────────────────────────────────
    fpre_files = sorted(glob.glob(os.path.join(args.data_dir, "fpre_*.npy")))

    if len(fpre_files) == 0:
        raise FileNotFoundError(
            f"No fpre_*.npy files found in '{args.data_dir}'.\n"
            "Run lbm_karman-ng.py with --save-every > 0 first."
        )

    print(f"\nFound {len(fpre_files)} snapshot pair(s) in '{args.data_dir}'")

    # ── 3. Evaluate each snapshot ──────────────────────────────────────────
    steps        = []
    rmsre_scores = []
    mae_scores   = []
    max_err      = []

    for fpre_path in fpre_files:
        # Derive matching fpost path
        fname = os.path.basename(fpre_path)
        step_str = fname.replace("fpre_", "").replace(".npy", "")
        fpost_path = os.path.join(args.data_dir, f"fpost_{step_str}.npy")

        if not os.path.exists(fpost_path):
            print(f"  [SKIP] No matching fpost for {fname}")
            continue

        step = int(step_str)
        steps.append(step)

        # Load and flatten spatial dimensions → (N_cells, 9)
        fpre_raw  = np.load(fpre_path)   # shape: (Nx, Ny, 9)
        fpost_raw = np.load(fpost_path)  # shape: (Nx, Ny, 9)

        Nx, Ny, Q = fpre_raw.shape
        fpre_flat  = fpre_raw.reshape(-1, Q)   # (Nx*Ny, 9)
        fpost_flat = fpost_raw.reshape(-1, Q)

        # Normalize — same as training pipeline
        fpre_norm,  norm = normalize(fpre_flat)
        fpost_norm, _    = normalize(fpost_flat)

        # NN prediction
        fpost_pred_norm = model.predict(fpre_norm, batch_size=args.batch_size, verbose=0)

        # ── Per-snapshot metrics ───────────────────────────────────────────
        eps = 1e-15

        # RMSRE
        rel_sq = ((fpost_norm - fpost_pred_norm) / (fpost_norm + eps)) ** 2
        rmsre_val = np.sqrt(np.mean(rel_sq))

        # MAE
        mae_val = np.mean(np.abs(fpost_norm - fpost_pred_norm))

        # Max absolute error
        max_val = np.max(np.abs(fpost_norm - fpost_pred_norm))

        rmsre_scores.append(rmsre_val)
        mae_scores.append(mae_val)
        max_err.append(max_val)

        print(f"  Step {step:>6d} | RMSRE={rmsre_val:.4e}  MAE={mae_val:.4e}  MaxErr={max_val:.4e}")

    steps        = np.array(steps)
    rmsre_scores = np.array(rmsre_scores)
    mae_scores   = np.array(mae_scores)
    max_err      = np.array(max_err)

    # ── 4. Save metrics to CSV ─────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "eval_metrics.csv")
    header = "step,rmsre,mae,max_abs_error"
    data   = np.column_stack([steps, rmsre_scores, mae_scores, max_err])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"\nMetrics saved to: {csv_path}")

    # ── 5. Plot metrics over simulation time ───────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].semilogy(steps, rmsre_scores, 'o-', color='steelblue', lw=1.5, ms=4)
    axes[0].set_ylabel("RMSRE")
    axes[0].set_title("Neural Network Evaluation on Kármán Vortex Data")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(steps, mae_scores, 'o-', color='darkorange', lw=1.5, ms=4)
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(steps, max_err, 'o-', color='crimson', lw=1.5, ms=4)
    axes[2].set_ylabel("Max Abs Error")
    axes[2].set_xlabel("Simulation Step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, "eval_metrics.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to:    {plot_path}")

    # ── 6. Distribution plot — best vs worst snapshot ─────────────────────
    # Reload best and worst snapshots for a per-direction error breakdown
    best_idx  = np.argmin(rmsre_scores)
    worst_idx = np.argmax(rmsre_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, idx, label in zip(axes,
                               [best_idx, worst_idx],
                               ["Best snapshot", "Worst snapshot"]):
        step_str  = f"{steps[idx]:06d}"
        fpre_raw  = np.load(os.path.join(args.data_dir, f"fpre_{step_str}.npy"))
        fpost_raw = np.load(os.path.join(args.data_dir, f"fpost_{step_str}.npy"))

        Nx, Ny, Q = fpre_raw.shape
        fpre_norm,  _ = normalize(fpre_raw.reshape(-1, Q))
        fpost_norm, _ = normalize(fpost_raw.reshape(-1, Q))

        fpost_pred = model.predict(fpre_norm, batch_size=args.batch_size, verbose=0)

        eps = 1e-15
        rel_err = np.abs((fpost_norm - fpost_pred) / (fpost_norm + eps))

        # Mean relative error per velocity direction
        mean_per_dir = rel_err.mean(axis=0)
        ax.bar(range(Q), mean_per_dir, color='steelblue', edgecolor='k', linewidth=0.5)
        ax.set_xticks(range(Q))
        ax.set_xticklabels([f"f{i}" for i in range(Q)])
        ax.set_ylabel("Mean relative error")
        ax.set_title(f"{label} (step {steps[idx]})\nRMSRE={rmsre_scores[idx]:.4e}")
        ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    dist_path = os.path.join(args.out_dir, "per_direction_error.png")
    fig.savefig(dist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-direction plot: {dist_path}")

    # ── 7. Summary ─────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────")
    print(f"  Snapshots evaluated : {len(steps)}")
    print(f"  RMSRE  — mean: {rmsre_scores.mean():.4e}  "
          f"min: {rmsre_scores.min():.4e}  max: {rmsre_scores.max():.4e}")
    print(f"  MAE    — mean: {mae_scores.mean():.4e}  "
          f"min: {mae_scores.min():.4e}  max: {mae_scores.max():.4e}")
    print(f"  MaxErr — mean: {max_err.mean():.4e}  "
          f"min: {max_err.min():.4e}  max: {max_err.max():.4e}")
    print("──────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()