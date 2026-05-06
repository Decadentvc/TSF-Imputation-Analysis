from __future__ import annotations

"""Plot clean model comparison across models using mixed scales.

- 6 STL metrics: share ratio vs total across models (per metric)
- sMAPE: raw value
"""

import argparse
from pathlib import Path
from typing import List

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
    "smape",
]


def metric_name(m: str) -> str:
    mp = {
        "trend_strength": "Trend Strength",
        "seasonal_strength": "Seasonal Strength",
        "seasonal_correlation": "Seasonal Correlation",
        "residual_autocorr_lag1": "Residual ACF(1)",
        "trend_linearity": "Trend Linearity",
        "spectral_entropy": "Spectral Entropy",
        "smape": "sMAPE",
    }
    return mp.get(m, m)


def load_all(by_model_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for p in sorted(by_model_dir.glob("*/clean_model_bias_summary.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {"model", "metric", "abs_bias_mean"}.issubset(df.columns):
            continue
        rows.append(df[["model", "metric", "abs_bias_mean"]].copy())
    if not rows:
        return pd.DataFrame(columns=["model", "metric", "abs_bias_mean"])
    return pd.concat(rows, ignore_index=True)


def load_smape(results_dir: Path, term: str) -> pd.DataFrame:
    rows: List[dict] = []
    for p in sorted(results_dir.glob("*/clean/*_results.csv")):
        model = p.parent.parent.name
        name = p.name
        mark = f"_clean_{term}_results.csv"
        if not name.endswith(mark):
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if not {"metric", "value"}.issubset(df.columns):
            continue
        smape = df.loc[df["metric"] == "sMAPE[0.5]", "value"]
        if smape.empty:
            continue
        try:
            v = float(smape.iloc[0])
        except Exception:
            continue
        rows.append({"model": model, "metric": "smape", "abs_bias_mean": abs(v)})
    if not rows:
        return pd.DataFrame(columns=["model", "metric", "abs_bias_mean"])
    out = pd.DataFrame(rows)
    out = out.groupby(["model", "metric"], as_index=False)["abs_bias_mean"].mean()
    return out


def to_share_vs_total(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.dropna(subset=["abs_bias_mean"])
    out["value_kind"] = np.where(out["metric"] == "smape", "raw", "ratio")
    out["plot_value"] = np.nan

    ratio_mask = out["metric"] != "smape"
    ratio_df = out[ratio_mask].copy()
    if not ratio_df.empty:
        totals = ratio_df.groupby("metric")["abs_bias_mean"].transform("sum")
        out.loc[ratio_mask, "plot_value"] = ratio_df["abs_bias_mean"] / totals.replace(0, np.nan)

    out.loc[~ratio_mask, "plot_value"] = out.loc[~ratio_mask, "abs_bias_mean"]
    return out


def plot_grouped_bar(df: pd.DataFrame, out_path: Path) -> None:
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    metrics = [m for m in METRIC_ORDER if m in set(df["metric"].astype(str).unique().tolist())]

    piv = df.pivot_table(index="metric", columns="model", values="plot_value", aggfunc="mean")
    piv = piv.reindex(index=metrics, columns=models)

    n_metrics = len(metrics)
    n_models = len(models)
    x = np.arange(n_metrics)
    width = 0.82 / max(1, n_models)

    fig, ax = plt.subplots(figsize=(14, 6.5))
    for i, model in enumerate(models):
        vals = piv[model].values.astype(float)
        ax.bar(x - 0.41 + i * width + width / 2, vals, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_name(m) for m in metrics], rotation=20, ha="right")
    ax.set_ylabel("Value (6 metrics: share ratio, sMAPE: raw)")
    ax.set_title("Clean Prediction Comparison (6 metrics: share vs total, sMAPE: raw)")
    ax.text(
        0.01,
        0.98,
        "6 metrics use share ratio: 0.20 = 20% of total (per metric, across models)\nsMAPE uses raw metric value",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#374151",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    metrics = [m for m in METRIC_ORDER if m in set(df["metric"].astype(str).unique().tolist())]

    piv = df.pivot_table(index="model", columns="metric", values="plot_value", aggfunc="mean")
    piv = piv.reindex(index=models, columns=metrics)

    vals = piv.values.astype(float)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    im = ax.imshow(vals, cmap="GnBu", vmin=np.nanmin(vals), vmax=np.nanmax(vals))
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([metric_name(m) for m in metrics], rotation=25, ha="right")
    ax.set_yticklabels(models)
    ax.set_title("Clean Prediction Heatmap (6 metrics: share ratio, sMAPE: raw)")

    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    rng = max(1e-12, vmax - vmin)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isnan(v):
                txt = ""
            else:
                txt = f"{v:.3f}" if metrics[j] == "smape" else f"{v*100:.1f}%"
            if np.isnan(v):
                color = "#111827"
            else:
                t = (v - vmin) / rng
                color = "#f9fafb" if t > 0.55 else "#111827"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mixed scale: share ratio for 6 metrics, raw for sMAPE")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot clean model comparison across models")
    p.add_argument("--by-model-dir", default="results_analysis/bias_analysis/by_model")
    p.add_argument("--output-dir", default="results_analysis/bias_analysis/figures")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--term", default="long")
    args = p.parse_args()

    by_model_dir = Path(args.by_model_dir)
    out_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_all(by_model_dir)
    smape_df = load_smape(results_dir, term=args.term)
    if not smape_df.empty:
        all_df = pd.concat([all_df, smape_df], ignore_index=True)

    if all_df.empty:
        raise SystemExit("[WARN] No clean_model_bias_summary.csv found")

    all_df = to_share_vs_total(all_df)

    out1 = out_dir / "clean_model_bias_rank_all_models.png"
    out2 = out_dir / "clean_model_bias_heatmap_all_models.png"
    plot_grouped_bar(all_df, out1)
    plot_heatmap(all_df, out2)

    print(f"[OK] Saved: {out1}")
    print(f"[OK] Saved: {out2}")
