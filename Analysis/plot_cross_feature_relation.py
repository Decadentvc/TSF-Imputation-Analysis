from __future__ import annotations

"""Plot cross-feature correlation heatmaps and top links."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def _plot_heatmap(df: pd.DataFrame, title: str, out_path: Path) -> None:
    vals = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(vals, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_xticklabels([f"P:{c}" for c in df.columns.tolist()], rotation=30, ha="right")
    ax.set_yticklabels([f"H:{r}" for r in df.index.tolist()])
    ax.set_title(title)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            v = vals[i, j]
            text = "" if np.isnan(v) else f"{v:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_top_links(long_df: pd.DataFrame, out_path: Path, top_k: int = 12) -> None:
    d = long_df.copy()
    score_col = "pearson_abs"
    d = d.dropna(subset=[score_col])
    if d.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid links", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    d = d.sort_values(score_col, ascending=False).head(top_k)
    labels = [f"H:{h}\nP:{p}" for h, p in zip(d["history_metric"], d["prediction_metric"])]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(len(d)), d[score_col], color="#2563eb")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("|Pearson|")
    ax.set_title("Top Cross-Feature Links")
    ax.axvline(0.2, color="#6b7280", linestyle="--", linewidth=1)
    ax.axvline(0.4, color="#6b7280", linestyle="--", linewidth=1)
    ax.axvline(0.7, color="#6b7280", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot cross-feature correlation")
    p.add_argument("--input-dir", default="results_analysis/bias_analysis")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else inp / "figures"
    out.mkdir(parents=True, exist_ok=True)

    pearson = _read_matrix(inp / "cross_feature_corr_pearson.csv")
    spearman = _read_matrix(inp / "cross_feature_corr_spearman.csv")
    long_df = pd.read_csv(inp / "cross_feature_corr_long.csv")

    _plot_heatmap(pearson, "Cross-Feature Correlation (Pearson)  [Y=H, X=P]", out / "cross_feature_pearson_heatmap.png")
    _plot_heatmap(spearman, "Cross-Feature Correlation (Spearman) [Y=H, X=P]", out / "cross_feature_spearman_heatmap.png")
    _plot_top_links(long_df, out / "cross_feature_top_links.png")

    print(f"[OK] Saved: {out / 'cross_feature_pearson_heatmap.png'}")
    print(f"[OK] Saved: {out / 'cross_feature_spearman_heatmap.png'}")
    print(f"[OK] Saved: {out / 'cross_feature_top_links.png'}")
