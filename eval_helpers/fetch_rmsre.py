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
  --save PATH       Save results to PATH.md and PATH.csv (e.g. --save summary)
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from keras import backend as K
import keras

from lbm_ml.data.generation import generate_samples
from lbm_ml.model.losses import rmsre
from lbm_ml.training import load_training_data


def _sci_fmt(mean: float, stderr: float) -> str:
    """Format as (m.mmmm ± s.ssss) × 10^exp, both values sharing one exponent."""
    exp = math.floor(math.log10(abs(mean)))
    scale = 10.0**exp
    return f"({mean / scale:.4f} ± {stderr / scale:.4f}) × 10^{exp}"



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


def _extract_meta(title: str, model: keras.Model) -> dict[str, str]:
    """Extract training hyperparameters from optimizer config and folder name."""
    meta: dict[str, str] = {"#params": f"{model.count_params():,}"}
    try:
        lr = model.optimizer.get_config().get("learning_rate")
        if lr is not None:
            meta["lr"] = f"{float(lr):.0e}"
    except Exception:
        pass
    for key, pattern in [("bs", r"bs(\d+)"), ("ep", r"ep(\d+)")]:
        m = re.search(pattern, title)
        if m:
            meta[key] = m.group(1)
    return meta


def _eval_model(
    title: str, model_path: Path, fpre: np.ndarray, fpost: np.ndarray
) -> tuple[float, float, dict[str, str]]:
    model: keras.Model = keras.models.load_model(
        str(model_path), custom_objects={"rmsre": rmsre}
    )  # pyright: ignore[reportAssignmentType]
    fpred = model.predict(fpre, verbose=0)  # pyright: ignore[reportArgumentType]
    per_sample: np.ndarray = rmsre(
        fpost, fpred
    ).numpy()  # pyright: ignore[reportAttributeAccessIssue,reportAssignmentType]
    mean = float(np.mean(per_sample))
    stderr = float(np.std(per_sample) / np.sqrt(len(per_sample)))
    return mean, stderr, _extract_meta(title, model)


def _build_df(results: list[tuple[str, float, float, dict[str, str]]]) -> pd.DataFrame:
    """Build a DataFrame from evaluated results, expanding meta into columns."""
    rows = []
    for title, mean, stderr, meta in results:
        row = {"model": title, "RMSRE": _sci_fmt(mean, stderr), "rmsre_mean": mean, "rmsre_stderr": stderr}
        row.update(meta)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="Paths to model.keras files or folders containing submodel directories")
    p.add_argument("--data", metavar="PATH", help=".npz dataset or simulation directory (skips generation)")
    p.add_argument("--n-samples", type=int, default=10_000, help="Number of samples to evaluate on (default: 10 000)")
    p.add_argument("--sort", choices=["asc", "desc"], help="Sort results by RMSRE (asc: best first, desc: worst first)")
    p.add_argument("--save", metavar="PATH", help="Save results to PATH.md and PATH.csv")
    args = p.parse_args()

    K.set_floatx("float64")

    entries = _resolve_models(args.models)
    if not entries:
        print("No models found.")
        return

    if args.data:
        data_path = Path(args.data)
        print(f"Loading test data from {data_path} ...")
        fpre_train, fpre_test, fpost_train, fpost_test = load_training_data(
            data_dir=data_path if data_path.is_dir() else None,
            dataset_path=data_path if data_path.is_file() else None,
        )
        fpre = np.concatenate([fpre_train, fpre_test])
        fpost = np.concatenate([fpost_train, fpost_test])
        if args.n_samples < len(fpre):
            idx = np.random.choice(len(fpre), args.n_samples, replace=False)
            fpre, fpost = fpre[idx], fpost[idx]
    else:
        print(f"Generating {args.n_samples} test samples ...")
        _, fpre, fpost = generate_samples(
            n_samples=args.n_samples,
            u_abs_min=1e-15,
            u_abs_max=0.01,
            sigma_min=1e-15,
            sigma_max=5e-4,
        )
        fpre = fpre / np.sum(fpre, axis=1)[:, np.newaxis]
        fpost = fpost / np.sum(fpost, axis=1)[:, np.newaxis]

    results: list[tuple[str, float, float, dict[str, str]]] = []
    for title, model_path in entries:
        print(f"Evaluating {title} ...")
        mean, stderr, meta = _eval_model(title, model_path, fpre, fpost)
        results.append((title, mean, stderr, meta))

    if args.sort:
        results.sort(key=lambda r: r[1], reverse=(args.sort == "desc"))

    df = _build_df(results)
    meta_cols = [k for k in ("#params", "bs", "ep", "lr") if k in df.columns]
    display_cols = ["model", "RMSRE"] + meta_cols
    csv_cols = ["model", "rmsre_mean", "rmsre_stderr"] + meta_cols

    if len(results) == 1:
        title, mean, stderr, meta = results[0]
        meta_str = "  " + "  ".join(f"{k}={meta[k]}" for k in meta_cols if k in meta)
        print(f"\n{title}: RMSRE = {_sci_fmt(mean, stderr)}{meta_str}")
        return

    print()
    print(df[display_cols].to_string(index=False))

    best = df.loc[df["rmsre_mean"].idxmin()]
    print(f"\nBest: {best['model']}  {best['RMSRE']}")

    if args.save:
        save_path = Path(args.save)
        df[csv_cols].to_csv(save_path.with_suffix(".csv"), index=False)
        df[display_cols].to_markdown(save_path.with_suffix(".md"), index=False)
        print(f"Saved {save_path.with_suffix('.md')} and {save_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
