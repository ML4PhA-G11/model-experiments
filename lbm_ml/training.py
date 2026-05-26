"""Shared training utilities used by run_all.py and continue_training.py."""

import atexit
import logging
import subprocess
import sys
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

from lbm_ml.data.generation import load_data
from lbm_ml.data.simulation import load_simulation_pairs

logger = logging.getLogger(__name__)


def load_training_data(
    data_dir: Path | None = None,
    dataset_path: Path | None = None,
    samples_per_step: int | None = None,
    step_stride: int = 1,
    max_steps: int | None = None,
    test_size: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load, normalise, and split collision pairs into train/test sets.

    Parameters
    ----------
    data_dir
        Directory of real simulator output (fpre_*.npy / fpost_*.npy pairs).
        When set, real data is used; otherwise loads from dataset_path.
    dataset_path
        Path to a synthetic .npz dataset produced by generate_dataset.
    samples_per_step, step_stride, max_steps
        Passed through to load_simulation_pairs when data_dir is set.
    test_size
        Fraction held out for validation.

    Returns
    -------
    (fpre_train, fpre_test, fpost_train, fpost_test) : normalised arrays of shape (N, 9).
    """
    if data_dir is not None:
        logger.info("Loading training data from simulator output in %s ...", data_dir)
        _, fpre, fpost = load_simulation_pairs(
            data_dir,
            samples_per_step=samples_per_step,
            step_stride=step_stride,
            max_steps=max_steps,
        )
        logger.info("Loaded %d samples", fpre.shape[0])
    else:
        _, fpre, fpost = load_data(dataset_path)

    # Normalise on density so all inputs/outputs sum to 1
    logger.info("Normalising samples...")
    fpre = fpre / np.sum(fpre, axis=1)[:, np.newaxis]
    fpost = fpost / np.sum(fpost, axis=1)[:, np.newaxis]
    logger.info("  -> Done.")

    logger.info("Splitting train/test...")
    fpre_train, fpre_test, fpost_train, fpost_test = train_test_split(fpre, fpost, test_size=test_size, shuffle=True)
    logger.info("  -> %d train / %d test samples", len(fpre_train), len(fpre_test))

    return fpre_train, fpre_test, fpost_train, fpost_test


def fit_model(
    model: keras.Model,
    fpre_train: np.ndarray,
    fpost_train: np.ndarray,
    fpre_test: np.ndarray,
    fpost_test: np.ndarray,
    paths: dict[str, Path],
    n_epochs: int = 200,
    patience: int = 50,
    batch_size: int = 32,
    tensorboard: bool = False,
    verbose: bool = True,
    initial_epoch: int = 0,
    tb_log_dir: Path | None = None,
) -> keras.Model:
    """Run the Keras training loop and save weights, model, and loss plot.

    Parameters
    ----------
    model
        Compiled Keras model. May already have weights loaded (e.g. for resuming).
    paths
        Artifact path dict with keys: 'weights', 'model', 'loss_plot', 'tb_log'.
    n_epochs
        Maximum number of epochs (absolute, not additional — consistent with Keras).
    patience
        EarlyStopping patience (epochs without val_loss improvement).
    initial_epoch
        Epoch to start counting from. Pass the number of epochs already trained
        so TensorBoard shows a continuous x-axis across runs.
    tb_log_dir
        Override for the TensorBoard log directory. Defaults to paths['tb_log'].
        Pass the previous run's log dir to append events to the same chart.

    Returns
    -------
    Trained model with best weights restored.
    """
    tb_log = tb_log_dir if tb_log_dir is not None else paths["tb_log"]

    if tensorboard:
        _start_tensorboard(tb_log)

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=patience // 3, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(paths["weights"]), monitor="val_loss", save_best_only=True),
        TensorBoard(log_dir=str(tb_log), histogram_freq=1),
    ]

    hist = model.fit(
        fpre_train,
        fpost_train,
        initial_epoch=initial_epoch,
        epochs=initial_epoch + n_epochs,
        verbose=cast(str, verbose),
        callbacks=callbacks,  # pyright: ignore[reportArgumentType]
        validation_data=(fpre_test, fpost_test),
        batch_size=batch_size,
    )

    epochs_run = len(hist.history["loss"])
    logger.info("  Trained %d/%d epochs (patience=%d)", epochs_run, n_epochs, patience)

    model.load_weights(str(paths["weights"]))
    model.save(str(paths["model"]))
    model.evaluate(fpre_test, fpost_test)

    plt.figure()
    plt.semilogy(hist.history["loss"], lw=3, label="Training")
    plt.semilogy(hist.history["val_loss"], lw=3, label="Validation")
    plt.legend(loc="best", frameon=False)
    plt.savefig(paths["loss_plot"], dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Loss plot  -> %s", paths["loss_plot"])

    return model


def _start_tensorboard(log_dir: Path, port: int = 6006) -> None:
    tb = Path(sys.executable).parent / "tensorboard"
    try:
        proc = subprocess.Popen(
            [str(tb), "--logdir", str(log_dir), "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(proc.terminate)
        logger.info("  TensorBoard   -> http://localhost:%d", port)
    except FileNotFoundError:
        logger.warning(
            "  TensorBoard not found; run manually: tensorboard --logdir %s --port %d",
            log_dir,
            port,
        )
