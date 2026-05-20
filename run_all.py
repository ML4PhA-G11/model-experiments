#!/usr/bin/env python3
"""End-to-end pipeline: dataset generation → training → ML-collision simulation.

Select the model with --model (or by editing MODEL_NAME below):
  python run_all.py --model d4equivariant   # default
  python run_all.py --model resnet
"""
import argparse
import atexit
import datetime
import subprocess
import sys
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

from lbm_ml.data.generation import generate_dataset, load_data
from lbm_ml.lattice.stencil import LB_stencil
from lbm_ml.model.losses import rmsre
from lbm_ml.model.network import MODEL_REGISTRY


# ---------------------------------------------------------------------------
# Runtime model selection — change this or pass --model on the CLI
# ---------------------------------------------------------------------------
MODEL_NAME: str = "d4equivariant"  # "d4equivariant" | "resnet"


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts-run-all-tensorflow"
DATASET_PATH  = ARTIFACTS_DIR / "example_dataset.npz"


def _make_run_dir(model_name: str, run_name: str | None, timestamp: str) -> Path:
    """Create and return artifacts-run-all-tensorflow/<model>[_<name>]_<timestamp>/."""
    stem = f"{model_name}_{run_name}_{timestamp}" if run_name else f"{model_name}_{timestamp}"
    d = ARTIFACTS_DIR / stem
    d.mkdir(parents=True, exist_ok=True)
    (d / "velocity_fields").mkdir(exist_ok=True)
    return d


def _run_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "weights":    run_dir / "weights.keras",
        "model":      run_dir / "model.keras",
        "loss_plot":  run_dir / "training_loss.png",
        "tb_log":     run_dir / "tensorboard_logs",
        "decay_plot": run_dir / "velocity_decay.png",
        "fields_dir": run_dir / "velocity_fields",
    }


def _latest_run_dir(model_name: str) -> Path:
    """Return the most-recently modified run directory for a given model."""
    matches = sorted(ARTIFACTS_DIR.glob(f"{model_name}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(
            f"No runs found for model '{model_name}' in {ARTIFACTS_DIR}"
        )
    return matches[-1]


# ---------------------------------------------------------------------------
# 1. Dataset generation
# ---------------------------------------------------------------------------

def generate(n_samples: int = 100_000) -> None:
    """Generate BGK collision pairs and save to DATASET_PATH."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n_samples} training samples ...")
    generate_dataset(
        DATASET_PATH,
        n_samples=n_samples,
        u_abs_min=1e-15, u_abs_max=0.01,
        sigma_min=1e-15, sigma_max=5e-4,
    )
    print(f"  Saved -> {DATASET_PATH}")


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def _start_tensorboard(log_dir: Path, port: int = 6006) -> None:
    tb = Path(sys.executable).parent / "tensorboard"
    try:
        proc = subprocess.Popen(
            [str(tb), "--logdir", str(log_dir), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(proc.terminate)
        print(f"  TensorBoard   -> http://localhost:{port}")
    except FileNotFoundError:
        print(f"  TensorBoard not found; run manually: tensorboard --logdir {log_dir} --port {port}")


def train(
    model_name: str = MODEL_NAME,
    batch_size: int = 32,
    n_epochs: int = 200,
    patience: int = 50,
    learning_rate: float = 1e-3,
    tensorboard: bool = False,
    run_name: str | None = None,
) -> keras.Model:
    """Load the dataset, train the selected network, and save artifacts under a timestamped run dir."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")

    K.set_floatx('float64')

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = _make_run_dir(model_name, run_name, timestamp)
    paths = _run_paths(run_dir)
    print(f"Run dir: {run_dir}")

    feq, fpre, fpost = load_data(DATASET_PATH)

    # Normalise on density so all inputs/outputs sum to 1
    feq   = feq   / np.sum(feq,   axis=1)[:, np.newaxis]
    fpre  = fpre  / np.sum(fpre,  axis=1)[:, np.newaxis]
    fpost = fpost / np.sum(fpost, axis=1)[:, np.newaxis]

    fpre_train, fpre_test, fpost_train, fpost_test = train_test_split(
        fpre, fpost, test_size=0.3, shuffle=True
    )

    print(f"Training model: {model_name}")
    if tensorboard:
        _start_tensorboard(paths["tb_log"])
    model = MODEL_REGISTRY[model_name](
        loss=rmsre,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        ll_activation="softmax",
    )

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience // 3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(paths["weights"]), monitor="val_loss", save_best_only=True),
        TensorBoard(log_dir=str(paths["tb_log"]), histogram_freq=1),
    ]

    hist = model.fit(
        fpre_train, fpost_train,
        epochs=n_epochs, verbose=1, callbacks=callbacks,  # pyright: ignore[reportArgumentType]
        validation_data=(fpre_test, fpost_test),
        batch_size=batch_size,
    )

    epochs_run = len(hist.history["loss"])
    print(f"  Trained {epochs_run}/{n_epochs} epochs (patience={patience})")

    model.load_weights(str(paths["weights"]))
    model.save(str(paths["model"]))
    model.evaluate(fpre_test, fpost_test)

    plt.figure()
    plt.semilogy(hist.history['loss'],     lw=3, label='Training')
    plt.semilogy(hist.history['val_loss'], lw=3, label='Validation')
    plt.legend(loc='best', frameon=False)
    plt.savefig(paths["loss_plot"], dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Loss plot  -> {paths['loss_plot']}")

    return model


# ---------------------------------------------------------------------------
# 3. Simulation
# ---------------------------------------------------------------------------

def _analytic_decay(t, L, F0, nu):
    """Analytic velocity decay for the Taylor-Green vortex."""
    return F0 * np.exp(-2 * nu * t / (L / (2 * np.pi)) ** 2)


def simulate(
    model_name: str = MODEL_NAME,
    run_dir: Path | None = None,
    nx: int = 32,
    ny: int = 32,
    niter: int = 1000,
    dumpit: int = 100,
    tau: float = 1.0,
    u0: float = 0.01,
) -> None:
    """Run the Taylor-Green decay simulation using the trained ML collision operator.

    If run_dir is None, the most recently modified run for model_name is used.
    """
    if run_dir is None:
        run_dir = _latest_run_dir(model_name)
        print(f"Simulating from latest run: {run_dir.name}")
    paths = _run_paths(run_dir)
    K.set_floatx('float64')

    Q = 9
    c, w, cs2, compute_feq = LB_stencil()

    model: keras.Model = keras.models.load_model(
        str(paths["model"]), custom_objects={'rmsre': rmsre}
    )  # pyright: ignore[reportAssignmentType]
    model.summary()

    # -- Initial conditions --
    ix, iy = np.meshgrid(range(nx), range(ny), indexing='ij')
    x = 2.0 * np.pi * (ix / nx)
    y = 2.0 * np.pi * (iy / ny)
    ux  =  u0 * np.sin(x) * np.cos(y)
    uy  = -u0 * np.cos(x) * np.sin(y)
    rho = np.ones((nx, ny))

    feq = np.zeros((nx, ny, Q))
    feq = compute_feq(feq, rho, ux, uy, c, w)
    f1 = np.copy(feq)
    f2 = np.copy(feq)

    # -- Data collection buffer --
    ndumps   = int(niter // dumpit)
    dumpfile = np.zeros((ndumps * nx * ny, 4))

    def collect(t, ux, uy, rho):
        it   = t // dumpit
        idx0 =  it      * (nx * ny)
        idx1 = (it + 1) * (nx * ny)
        dumpfile[idx0:idx1, 0] = t
        dumpfile[idx0:idx1, 1] = rho.reshape(nx * ny)
        dumpfile[idx0:idx1, 2] = ux.reshape(nx * ny)
        dumpfile[idx0:idx1, 3] = uy.reshape(nx * ny)

    collect(0, ux, uy, rho)
    m_initial = np.sum(f1)

    # -- Time loop --
    for t in range(1, niter):
        # Streaming
        for ip in range(Q):
            f1[:, :, ip] = np.roll(np.roll(f2[:, :, ip], c[ip, 0], axis=0), c[ip, 1], axis=1)

        rho = np.sum(f1, axis=2)
        ux  = (1. / rho) * np.einsum('ijk,k', f1, c[:, 0])
        uy  = (1. / rho) * np.einsum('ijk,k', f1, c[:, 1])

        # ML collision step
        fpre = f1.reshape((nx * ny, Q))
        norm = np.sum(fpre, axis=1)[:, np.newaxis]
        fpre = fpre / norm
        f2   = model.predict(fpre, verbose=0)  # pyright: ignore[reportArgumentType]
        f2   = (norm * f2).reshape((nx, ny, Q))

        if t % dumpit == 0:
            collect(t, ux, uy, rho)

    m_final = np.sum(f2)
    print(f'Sim ended. Mass err: {np.abs(m_initial - m_final) / m_initial:.2e}')

    _plot_results(dumpfile, niter, dumpit, nx, ny, tau, cs2,
                  paths["decay_plot"], paths["fields_dir"])


def _plot_results(dumpfile, niter, dumpit, nx, ny, tau, cs2, decay_plot, fields_dir):
    tLst = np.arange(0, niter, dumpit)
    nu   = (tau - 0.5) * cs2
    w_fig, h_fig = 3.46 * 3, 2.14 * 3

    # Velocity decay
    fig, ax = plt.subplots(figsize=(w_fig, h_fig))
    F0 = None
    for i, t in enumerate(tLst):
        ux = dumpfile[dumpfile[:, 0] == t, 2]
        uy = dumpfile[dumpfile[:, 0] == t, 3]
        Ft = np.average((ux ** 2 + uy ** 2) ** 0.5)
        if i == 0:
            F0 = Ft
            ax.semilogy(t, Ft, 'ob', label='lbm')
        else:
            ax.semilogy(t, Ft, 'ob')

    ax.semilogy(tLst, _analytic_decay(tLst, nx, F0, nu),
                linewidth=2.0, linestyle='--', color='r', label='analytic')
    ax.set_xlabel(r'$t~\rm{[L.U.]}$', fontsize=16)
    ax.set_ylabel(r'$\langle |u| \rangle$', fontsize=16, rotation=90, labelpad=0)
    ax.legend(loc='best', frameon=False, prop={'size': 16})
    ax.tick_params(which="both", direction="in", top="on", right="on", labelsize=14)
    fig.savefig(decay_plot, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Decay plot -> {decay_plot}")

    # Velocity field snapshots
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    for t in tLst:
        ux = dumpfile[dumpfile[:, 0] == t, 2].reshape((nx, ny))
        uy = dumpfile[dumpfile[:, 0] == t, 3].reshape((nx, ny))
        u  = (ux ** 2 + uy ** 2) ** 0.5

        fig, ax = plt.subplots(figsize=(w_fig, h_fig))
        im = ax.imshow(u)
        ax.streamplot(X, Y, ux, uy, density=0.5, color='w')
        fig.colorbar(im, ax=ax, orientation='vertical', pad=0, shrink=0.69)
        ax.set_title(f"Iteration {int(t)}", size=16)
        field_path = fields_dir / f"velocity_field_t{int(t):05d}.png"
        fig.savefig(field_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Field plot -> {field_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL_NAME, choices=list(MODEL_REGISTRY),
                   help="Which model architecture to train and simulate with")
    p.add_argument("--n-epochs", type=int, default=200,
                   help="Maximum number of training epochs")
    p.add_argument("--patience", type=int, default=50,
                   help="EarlyStopping patience (epochs without val_loss improvement)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Training batch size")
    p.add_argument("--learning-rate", type=float, default=1e-3,
                   help="Initial Adam learning rate (default: 1e-3)")
    p.add_argument("--skip-generate", action="store_true",
                   help="Skip dataset generation (reuse existing file)")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training (load existing saved model)")
    p.add_argument("--skip-simulate", action="store_true",
                   help="Skip simulation")
    p.add_argument("--tensorboard", action="store_true",
                   help="Open TensorBoard in browser during training (logs always saved)")
    p.add_argument("--run-name", default=None,
                   help="Optional label added to the run directory name (default: timestamp only)")
    p.add_argument("--run-dir", default=None,
                   help="Explicit run directory to simulate from (default: latest run for --model)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.skip_generate:
        generate()
    if not args.skip_train:
        train(model_name=args.model, batch_size=args.batch_size,
              n_epochs=args.n_epochs, patience=args.patience,
              learning_rate=args.learning_rate,
              tensorboard=args.tensorboard, run_name=args.run_name)
    if not args.skip_simulate:
        run_dir = Path(args.run_dir) if args.run_dir else None
        simulate(model_name=args.model, run_dir=run_dir)
