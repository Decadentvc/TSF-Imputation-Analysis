from __future__ import annotations

"""Plot dataset-level correlation bars for one model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_ORDER = [
    "trend_strength",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "trend_linearity",
    "spectral_entropy",
]


def metric_name(m: str) -> str:
    mp = {
        "trend_strength": "Trend Strength",
        "seasonal_strength": "Seasonal Strength",
        "seasonal_correlation": "Seasonal Correlation",
        "residual_autocorr_lag1": "Residual ACF(1)",
        "trend_linearity": "Trend Linearity",
        "spectral_entropy": "Spectral Entropy",
    }
    return mp.get(m, m)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot dataset-level correlations")
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    if df.empty:
        print("[WARN] Empty input")
        raise SystemExit(0)

    model_vals = df["model"].dropna().astype(str).unique().tolist()
    model_name = model_vals[0] if len(model_vals) == 1 else "model"

    metrics = [m for m in METRIC_ORDER if m in set(df["metric"].astype(str).unique().tolist())]
    if not metrics:
        metrics = sorted(df["metric"].astype(str).unique().tolist())

    ncols = 2
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = df[df["metric"] == metric].copy().sort_values("spearman_abs", ascending=False)
        if sub.empty:
            ax.set_axis_off()
            continue

        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub["pearson_abs"], width=0.36, label="|Pearson|", color="#3b82f6")
        ax.bar(x + 0.18, sub["spearman_abs"], width=0.36, label="|Spearman|", color="#10b981")
        ax.axhline(0.2, linestyle="--", linewidth=1, color="#6b7280")
        ax.axhline(0.4, linestyle="--", linewidth=1, color="#6b7280")
        ax.axhline(0.7, linestyle="--", linewidth=1, color="#6b7280")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["dataset"].tolist(), rotation=40, ha="right", fontsize=8)
        ax.set_title(metric_name(metric))
        ax.grid(axis="y", alpha=0.25)

    for j in range(len(metrics), len(axes)):
        axes[j].set_axis_off()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"Dataset-level Correlation by Metric ({model_name})", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    out_path = out_dir / "dataset_level_correlation.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"[OK] Saved: {out_path}")
