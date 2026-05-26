#!/usr/bin/env python3
"""Resume training from a saved checkpoint in a new run directory.

Two ways to specify the starting checkpoint:

  From a full model.keras (preserves optimizer state, e.g. Adam moments and
  learning-rate schedule — the closest thing to a true resume):

      python continue_training.py path/to/model.keras --data-dir ../data/...

  From weights.keras only (resets optimizer state; use when model.keras is not
  available, e.g. after a crash that only left the ModelCheckpoint output):

      python continue_training.py path/to/weights.keras --model lenn --data-dir ../data/...

A fresh timestamped run directory is always created under artifacts-run-all-tensorflow/
so the original checkpoint is never overwritten.

Behavior
--------
- Previous TensorBoard logs are copied into the new run dir so the loss curve
  is continuous across runs. Open TensorBoard on the new dir to see the full history.
- The epoch counter starts from the last epoch of the previous run (auto-detected
  from the TB events, or set manually with --initial-epoch).

Arguments
---------
checkpoint               Path to model.keras or weights.keras from a previous run.
--model MODEL            Architecture name (required with weights.keras).
--data-dir DIR           Directory of simulator output (fpre_*.npy / fpost_*.npy).
--n-epochs N             Number of additional epochs to train (default: 200).
--initial-epoch N        Override the auto-detected starting epoch.
--patience N             EarlyStopping patience (default: 50).
--batch-size N           Training batch size (default: 32).
--learning-rate LR       Adam LR — only applies when loading weights.keras (default: 1e-3).
--samples-per-step N     Sub-sample N lattice nodes per saved timestep.
--step-stride N          Use every Nth saved timestep (default: 1).
--max-steps N            Cap the number of timesteps loaded.
--run-name NAME          Label appended to the new run directory name (default: "continued").
--tensorboard            Launch TensorBoard during training.
--quiet                  Suppress INFO logging.
"""

import argparse
import datetime
import logging
import shutil
from pathlib import Path

import keras
from keras import backend as K

from lbm_ml.model.losses import rmsre
from lbm_ml.model.network import MODEL_REGISTRY
from lbm_ml.training import fit_model, load_training_data
from run_all import _make_run_dir, _run_paths, setup_logging

logger = logging.getLogger(__name__)


def _count_tb_epochs(tb_log_dir: Path) -> int:
    """Count completed epochs by reading the highest step in TensorBoard event files."""
    if not tb_log_dir.exists():
        return 0
    try:
        import tensorflow as tf

        event_files = sorted(tb_log_dir.rglob("events.out.tfevents.*"))
        if not event_files:
            return 0
        max_step = -1
        for ef in event_files:
            for event in tf.compat.v1.train.summary_iterator(str(ef)):
                for v in event.summary.value:
                    if v.tag in ("epoch_loss", "loss"):
                        max_step = max(max_step, event.step)
        return max_step + 1  # steps are 0-indexed epochs
    except Exception:
        return 0


def _load_model(checkpoint: Path, model_name: str | None, learning_rate: float) -> keras.Model:
    """Load a model from either a model.keras or a weights.keras checkpoint.

    Parameters
    ----------
    checkpoint
        Path to model.keras (full model) or weights.keras (weights only).
    model_name
        Required when checkpoint is a weights file; selects the architecture
        from MODEL_REGISTRY to instantiate before loading weights.
    learning_rate
        Learning rate for a fresh Adam optimizer (only used with weights.keras).

    Returns
    -------
    Compiled Keras model ready for training.
    """
    name = checkpoint.name.lower()
    if "model" in name or checkpoint.suffix == ".keras" and "weight" not in name:
        # Heuristic: treat as full model if filename contains "model" or
        # does not contain "weight". The user can always be explicit.
        try:
            logger.info("Loading full model from %s ...", checkpoint)
            model = keras.models.load_model(str(checkpoint), custom_objects={"rmsre": rmsre})
            logger.info("  -> Optimizer state restored (true resume).")
            return model  # pyright: ignore[reportReturnType]
        except Exception:
            pass  # fall through to weights path

    # weights.keras path
    if model_name is None:
        raise ValueError(
            "Cannot load weights without knowing the architecture. "
            "Pass --model to specify which model to instantiate."
        )
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_REGISTRY)}")

    logger.info("Instantiating %s and loading weights from %s ...", model_name, checkpoint)
    model = MODEL_REGISTRY[model_name](
        loss=rmsre,
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        ll_activation="softmax",
    )
    model.load_weights(str(checkpoint))
    logger.info("  -> Weights loaded (optimizer state reset).")
    return model


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="Path to model.keras (full resume) or weights.keras (weights only)")
    p.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_REGISTRY),
        help="Architecture to instantiate (required when checkpoint is weights.keras)",
    )
    p.add_argument("--n-epochs", type=int, default=200, help="Number of additional epochs to train")
    p.add_argument(
        "--initial-epoch",
        type=int,
        default=None,
        help="Epoch offset for TensorBoard x-axis and Keras epoch counter "
        "(default: auto-detected from previous run's TB log, or 0)",
    )
    p.add_argument("--patience", type=int, default=50, help="EarlyStopping patience")
    p.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    p.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate (weights.keras only)")
    p.add_argument(
        "--data-dir",
        required=True,
        help="Directory of simulator output (fpre_*.npy / fpost_*.npy pairs)",
    )
    p.add_argument("--samples-per-step", type=int, default=None)
    p.add_argument("--step-stride", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--run-name", default="continued", help="Label added to the new run directory name")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--quiet", action="store_true", help="Suppress INFO logging")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    setup_logging(verbose=not args.quiet)

    K.set_floatx("float64")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    prev_run_dir = checkpoint.parent
    prev_tb_log = prev_run_dir / "tensorboard_logs"

    # Auto-detect how many epochs the previous run completed from its TB events.
    initial_epoch = args.initial_epoch
    if initial_epoch is None:
        initial_epoch = _count_tb_epochs(prev_tb_log)
        if initial_epoch > 0:
            logger.info("Detected %d completed epochs from previous TB log.", initial_epoch)
        else:
            logger.info("Could not detect previous epoch count; starting from epoch 0.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_label = args.model or prev_run_dir.name
    run_dir = _make_run_dir(model_label, args.run_name, timestamp)
    paths = _run_paths(run_dir)
    logger.info("Run dir: %s", run_dir)

    if prev_tb_log.exists():
        shutil.copytree(prev_tb_log, paths["tb_log"], dirs_exist_ok=True)
        logger.info("Copied previous TB logs -> %s (epoch offset: %d)", paths["tb_log"], initial_epoch)
    else:
        logger.info("TensorBoard log: %s (epoch offset: %d)", paths["tb_log"], initial_epoch)

    model = _load_model(checkpoint, args.model, args.learning_rate)

    fpre_train, fpre_test, fpost_train, fpost_test = load_training_data(
        data_dir=Path(args.data_dir),
        samples_per_step=args.samples_per_step,
        step_stride=args.step_stride,
        max_steps=args.max_steps,
    )

    fit_model(
        model,
        fpre_train,
        fpost_train,
        fpre_test,
        fpost_test,
        paths=paths,
        n_epochs=args.n_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        tensorboard=args.tensorboard,
        verbose=not args.quiet,
        initial_epoch=initial_epoch,
    )
