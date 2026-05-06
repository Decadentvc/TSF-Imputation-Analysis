from __future__ import annotations

"""Compute dataset-level bias correlation from bias_pairs.csv.

For a single model bias_pairs file, this computes correlation on:
- history_bias vs prediction_bias_mean
Grouped by: dataset + metric
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if d.shape[0] < 3:
        return np.nan
    if d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return np.nan
    if method == "spearman":
        return float(d["x"].rank(method="average").corr(d["y"].rank(method="average"), method="pearson"))
    return float(d["x"].corr(d["y"], method="pearson"))


def _strength(v: float, n: int) -> str:
    if n < 3 or pd.isna(v):
        return "insufficient"
    a = abs(float(v))
    if a >= 0.7:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "none"


def run(input_csv: Path, output_csv: Path, min_samples: int) -> None:
    df = pd.read_csv(input_csv)
    required = {"model", "dataset", "metric", "history_bias", "prediction_bias_mean"}
    miss = sorted(list(required - set(df.columns)))
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    rows: List[Dict[str, object]] = []
    for (model, dataset, metric), g in df.groupby(["model", "dataset", "metric"], dropna=False):
        x = pd.to_numeric(g["history_bias"], errors="coerce")
        y = pd.to_numeric(g["prediction_bias_mean"], errors="coerce")
        valid = pd.DataFrame({"x": x, "y": y}).dropna()
        n = int(valid.shape[0])
        if n < min_samples:
            continue

        pearson = _safe_corr(valid["x"], valid["y"], "pearson")
        spearman = _safe_corr(valid["x"], valid["y"], "spearman")
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "metric": metric,
                "n": n,
                "pearson": pearson,
                "spearman": spearman,
                "pearson_abs": float(abs(pearson)) if pd.notna(pearson) else np.nan,
                "spearman_abs": float(abs(spearman)) if pd.notna(spearman) else np.nan,
                "pearson_strength": _strength(pearson, n),
                "spearman_strength": _strength(spearman, n),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["metric", "dataset"], ascending=[True, True]) if not out.empty else out
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dataset-level correlation from bias pairs")
    p.add_argument("--input", required=True, help="Path to bias_pairs.csv")
    p.add_argument("--output", required=True, help="Path to output correlation csv")
    p.add_argument("--min-samples", type=int, default=30)
    args = p.parse_args()

    run(Path(args.input), Path(args.output), args.min_samples)
    print(f"[OK] dataset-level correlation -> {args.output}")
