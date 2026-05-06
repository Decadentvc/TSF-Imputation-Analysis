from __future__ import annotations

"""Cross-feature relation analysis on bias_pairs.

For each model (or any bias_pairs.csv), compute correlations between:
- history_bias of metric_i
- prediction_bias of metric_j

Outputs:
- cross_feature_corr_long.csv
- cross_feature_corr_pearson.csv (matrix)
- cross_feature_corr_spearman.csv (matrix)
- top_cross_feature_links.csv
"""

import argparse
from pathlib import Path
from typing import List

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


def _metric_order(metrics: List[str]) -> List[str]:
    default = [
        "trend_strength",
        "seasonal_strength",
        "seasonal_correlation",
        "residual_autocorr_lag1",
        "trend_linearity",
        "spectral_entropy",
    ]
    seen = set(metrics)
    ordered = [m for m in default if m in seen]
    ordered.extend(sorted([m for m in metrics if m not in ordered]))
    return ordered


def analyze_one(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)
    needed = {
        "dataset",
        "term",
        "ratio",
        "method",
        "window_idx",
        "metric",
        "history_bias",
        "prediction_bias_mean",
    }
    miss = sorted(list(needed - set(df.columns)))
    if miss:
        raise ValueError(f"Missing columns in {input_csv}: {miss}")

    metrics = _metric_order(df["metric"].dropna().astype(str).unique().tolist())

    key_cols = ["dataset", "term", "ratio", "method", "window_idx"]
    wide_h = df.pivot_table(index=key_cols, columns="metric", values="history_bias", aggfunc="first")
    wide_p = df.pivot_table(index=key_cols, columns="metric", values="prediction_bias_mean", aggfunc="first")

    rows = []
    for hm in metrics:
        for pm in metrics:
            if hm not in wide_h.columns or pm not in wide_p.columns:
                continue
            combo = pd.DataFrame({"h": wide_h[hm], "p": wide_p[pm]}).dropna()
            n = int(combo.shape[0])
            pearson = _safe_corr(combo["h"], combo["p"], "pearson")
            spearman = _safe_corr(combo["h"], combo["p"], "spearman")
            rows.append(
                {
                    "history_metric": hm,
                    "prediction_metric": pm,
                    "n": n,
                    "pearson": pearson,
                    "spearman": spearman,
                    "pearson_abs": float(abs(pearson)) if pd.notna(pearson) else np.nan,
                    "spearman_abs": float(abs(spearman)) if pd.notna(spearman) else np.nan,
                }
            )

    out = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "cross_feature_corr_long.csv", index=False)

    pearson_m = out.pivot(index="history_metric", columns="prediction_metric", values="pearson")
    spearman_m = out.pivot(index="history_metric", columns="prediction_metric", values="spearman")

    pearson_m = pearson_m.reindex(index=metrics, columns=metrics)
    spearman_m = spearman_m.reindex(index=metrics, columns=metrics)

    pearson_m.to_csv(output_dir / "cross_feature_corr_pearson.csv")
    spearman_m.to_csv(output_dir / "cross_feature_corr_spearman.csv")

    top = out.sort_values(["pearson_abs", "spearman_abs"], ascending=False).head(20)
    top.to_csv(output_dir / "top_cross_feature_links.csv", index=False)



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-feature relation analysis")
    p.add_argument("--input", default="results_analysis/bias_analysis/bias_pairs.csv", help="Path to bias_pairs.csv")
    p.add_argument("--output-dir", default=None, help="Output directory (default: input file folder)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inp = Path(args.input)
    out_dir = Path(args.output_dir) if args.output_dir else inp.parent
    analyze_one(inp, out_dir)
    print(f"[OK] cross-feature outputs -> {out_dir}")
