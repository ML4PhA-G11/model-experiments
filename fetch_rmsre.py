#!/usr/bin/env python3
"""Evaluate RMSRE for one or more trained models on a freshly generated test set.

Each argument can be either:
  - a direct path to a model.keras file, or
  - a folder whose immediate subfolders each contain a model.keras file.

The subfolder name is used as the model title when discovering models from a folder.

Usage
-----
Evaluate specific model files:
  python fetch_rmsre.py artifacts/run_a/model.keras artifacts/run_b/model.keras

Evaluate all models under a folder:
  python fetch_rmsre.py artifacts/

Mix both forms:
  python fetch_rmsre.py artifacts/ extra_runs/my_model/model.keras

Options:
  --n-samples INT   Number of test samples to generate (default: 10 000)
"""

import argparse
import math
from pathlib import Path

import numpy as np
from keras import backend as K
import keras

from lbm_ml.data.generation import generate_samples
from lbm_ml.model.losses import rmsre


def _sci_fmt(mean: float, stderr: float) -> str:
    """Format as (m.mmmm ± s.ssss) × 10^exp, both values sharing one exponent."""
    exp = math.floor(math.log10(abs(mean)))
    scale = 10.0 ** exp
    return f"({mean / scale:.4f} ± {stderr / scale:.4f}) × 10^{exp}"


def _generate_test_data(n_samples: int = 10_000) -> tuple:
    _, fpre, fpost = generate_samples(
        n_samples=n_samples,
        u_abs_min=1e-15,
        u_abs_max=0.01,
        sigma_min=1e-15,
        sigma_max=5e-4,
    )
    fpre  = fpre  / np.sum(fpre,  axis=1)[:, np.newaxis]
    fpost = fpost / np.sum(fpost, axis=1)[:, np.newaxis]
    return fpre, fpost


def _resolve_models(inputs: list[str]) -> list[tuple[str, Path]]:
    """Return (title, path) pairs from a mix of model files and search folders."""
    entries: list[tuple[str, Path]] = []
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.glob("*/model.keras"))
            if not found:
                print(f"Warning: no model.keras files found under {p}")
            for model_path in found:
                entries.append((model_path.parent.name, model_path))
        else:
            entries.append((p.parent.name if p.name == "model.keras" else p.stem, p))
    return entries


def _eval_model(model_path: Path, fpre: np.ndarray, fpost: np.ndarray) -> tuple[float, float]:
    model: keras.Model = keras.models.load_model(
        str(model_path), custom_objects={"rmsre": rmsre}
    )  # pyright: ignore[reportAssignmentType]
    fpred = model.predict(fpre, verbose=0)  # pyright: ignore[reportArgumentType]
    per_sample: np.ndarray = rmsre(fpost, fpred).numpy()  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
    mean   = float(np.mean(per_sample))
    stderr = float(np.std(per_sample) / np.sqrt(len(per_sample)))
    return mean, stderr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="Paths to model.keras files or folders containing submodel directories")
    p.add_argument("--n-samples", type=int, default=10_000, help="Test set size (default: 10 000)")
    args = p.parse_args()

    K.set_floatx("float64")

    entries = _resolve_models(args.models)
    if not entries:
        print("No models found.")
        return

    print(f"Generating {args.n_samples} test samples ...")
    fpre, fpost = _generate_test_data(args.n_samples)

    results: list[tuple[str, float, float]] = []
    for title, model_path in entries:
        print(f"Evaluating {title} ...")
        mean, stderr = _eval_model(model_path, fpre, fpost)
        results.append((title, mean, stderr))

    val_col = max(len(_sci_fmt(m, s)) for _, m, s in results)

    if len(results) == 1:
        title, mean, stderr = results[0]
        print(f"\n{title}: RMSRE = {_sci_fmt(mean, stderr)}")
        return

    col = max(len(t) for t, *_ in results)
    print(f"\n{'Model':<{col}}   {'RMSRE (mean ± stderr)':<{val_col}}")
    print("-" * (col + 3 + val_col))
    for title, mean, stderr in results:
        print(f"{title:<{col}}   {_sci_fmt(mean, stderr):<{val_col}}")

    best = min(results, key=lambda r: r[1])
    print(f"\nBest: {best[0]}  {_sci_fmt(best[1], best[2])}")


if __name__ == "__main__":
    main()
