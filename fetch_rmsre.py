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
import csv
import math
import re
from pathlib import Path

import numpy as np
from keras import backend as K
import keras

from lbm_ml.data.generation import generate_samples
from lbm_ml.model.losses import rmsre


def _sci_fmt(mean: float, stderr: float) -> str:
    """Format as (m.mmmm ± s.ssss) × 10^exp, both values sharing one exponent."""
    exp = math.floor(math.log10(abs(mean)))
    scale = 10.0**exp
    return f"({mean / scale:.4f} ± {stderr / scale:.4f}) × 10^{exp}"


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
    meta: dict[str, str] = {}
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


def _save_md(path: Path, results: list, meta_keys: list[str], best: tuple) -> None:
    lines = ["| Model | RMSRE (mean ± stderr) |" + "".join(f" {k} |" for k in meta_keys)]
    lines.append("|---|---|" + "".join("---|" for _ in meta_keys))
    for title, mean, stderr, meta in results:
        row = f"| {title} | {_sci_fmt(mean, stderr)} |"
        row += "".join(f" {meta.get(k, '')} |" for k in meta_keys)
        lines.append(row)
    lines.append(f"\n**Best:** {best[0]} — {_sci_fmt(best[1], best[2])}")
    path.write_text("\n".join(lines) + "\n")


def _save_csv(path: Path, results: list, meta_keys: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "rmsre_mean", "rmsre_stderr"] + meta_keys)
        for title, mean, stderr, meta in results:
            writer.writerow([title, mean, stderr] + [meta.get(k, "") for k in meta_keys])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models", nargs="+", help="Paths to model.keras files or folders containing submodel directories")
    p.add_argument("--n-samples", type=int, default=10_000, help="Test set size (default: 10 000)")
    p.add_argument("--sort", choices=["asc", "desc"], help="Sort results by RMSRE (asc: best first, desc: worst first)")
    p.add_argument("--save", metavar="PATH", help="Save results to PATH.md and PATH.csv")
    args = p.parse_args()

    K.set_floatx("float64")

    entries = _resolve_models(args.models)
    if not entries:
        print("No models found.")
        return

    print(f"Generating {args.n_samples} test samples ...")
    fpre, fpost = _generate_test_data(args.n_samples)

    results: list[tuple[str, float, float, dict[str, str]]] = []
    for title, model_path in entries:
        print(f"Evaluating {title} ...")
        mean, stderr, meta = _eval_model(title, model_path, fpre, fpost)
        results.append((title, mean, stderr, meta))

    if args.sort:
        results.sort(key=lambda r: r[1], reverse=(args.sort == "desc"))

    val_col = max(len(_sci_fmt(m, s)) for _, m, s, _ in results)
    meta_keys = [k for k in ("bs", "ep", "lr") if any(k in r[3] for r in results)]

    if len(results) == 1:
        title, mean, stderr, meta = results[0]
        meta_str = "  " + "  ".join(f"{k}={meta[k]}" for k in meta_keys if k in meta) if meta_keys else ""
        print(f"\n{title}: RMSRE = {_sci_fmt(mean, stderr)}{meta_str}")
        return

    col = max(len(t) for t, *_ in results)
    meta_cols = {k: max(len(k), max(len(r[3].get(k, "")) for r in results)) for k in meta_keys}
    header = f"{'Model':<{col}}   {'RMSRE (mean ± stderr)':<{val_col}}"
    for k, w in meta_cols.items():
        header += f"   {k:>{w}}"
    print(f"\n{header}")
    print("-" * len(header))
    for title, mean, stderr, meta in results:
        row = f"{title:<{col}}   {_sci_fmt(mean, stderr):<{val_col}}"
        for k, w in meta_cols.items():
            row += f"   {meta.get(k, ''):>{w}}"
        print(row)

    best = min(results, key=lambda r: r[1])
    print(f"\nBest: {best[0]}  {_sci_fmt(best[1], best[2])}")

    if args.save:
        save_path = Path(args.save)
        _save_md(save_path.with_suffix(".md"), results, meta_keys, best)
        _save_csv(save_path.with_suffix(".csv"), results, meta_keys)
        print(f"Saved {save_path.with_suffix('.md')} and {save_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
