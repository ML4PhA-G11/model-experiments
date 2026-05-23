#!/usr/bin/env python3
"""Evaluate RMSRE for one or more saved models on a freshly generated test set.

Usage:
  python fetch_rmsre.py path/to/model1.keras path/to/model2.keras ...
"""

import argparse

import numpy as np
from keras import backend as K
import keras

from lbm_ml.data.generation import generate_samples
from lbm_ml.model.losses import rmsre


def _generate_test_data(n_samples: int = 10_000) -> tuple:
    _, fpre, fpost = generate_samples(
        n_samples=n_samples,
        u_abs_min=1e-15,
        u_abs_max=0.01,
        sigma_min=1e-15,
        sigma_max=5e-4,
    )
    fpre = fpre / np.sum(fpre, axis=1)[:, np.newaxis]
    fpost = fpost / np.sum(fpost, axis=1)[:, np.newaxis]
    return fpre, fpost


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="Paths to .keras model files")
    p.add_argument("--n-samples", type=int, default=10_000, help="Test set size (default: 10 000)")
    args = p.parse_args()

    K.set_floatx("float64")

    print(f"Generating {args.n_samples} test samples ...")
    fpre, fpost = _generate_test_data(args.n_samples)

    for model_path in args.models:
        model: keras.Model = keras.models.load_model(
            model_path, custom_objects={"rmsre": rmsre}
        )  # pyright: ignore[reportAssignmentType]
        fpred = model.predict(fpre, verbose=0)  # pyright: ignore[reportArgumentType]
        per_sample: np.ndarray = rmsre(fpost, fpred).numpy()  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
        mean = float(np.mean(per_sample))
        stderr = float(np.std(per_sample) / np.sqrt(len(per_sample)))
        print(f"{model_path}: RMSRE = {mean:.6e} ± {stderr:.6e}")


if __name__ == "__main__":
    main()
