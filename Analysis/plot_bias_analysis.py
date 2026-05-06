from __future__ import annotations

"""可视化偏差关联与模型差异结果。"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_ORDER: Tuple[str, ...] = (
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "spectral_entropy",
)

THRESHOLDS = (0.2, 0.4, 0.7)
THRESHOLD_LABELS = ("weak", "moderate", "strong")


def metric_display_name(metric: str) -> str:
    return {
        "trend_strength": "Trend Strength",
        "trend_linearity": "Trend Linearity",
        "seasonal_strength": "Seasonal Strength",
        "seasonal_correlation": "Seasonal Correlation",
        "residual_autocorr_lag1": "Residual ACF(1)",
        "spectral_entropy": "Spectral Entropy",
    }.get(metric, metric)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def _sorted_metric_df(df: pd.DataFrame, metric_col: str = "metric") -> pd.DataFrame:
    order_map = {m: i for i, m in enumerate(METRIC_ORDER)}
    out = df.copy()
    out["_order"] = out[metric_col].map(order_map).fillna(999)
    out = out.sort_values("_order").drop(columns=["_order"])
    return out


def metrics_in_data(*dfs: pd.DataFrame) -> List[str]:
    present = set()
    for df in dfs:
        if "metric" in df.columns:
            present.update(df["metric"].dropna().astype(str).unique().tolist())
    ordered = [m for m in METRIC_ORDER if m in present]
    return ordered


def _single_model_name(df: pd.DataFrame) -> str:
    if "group_value" in df.columns:
        vals = df["group_value"].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            return vals[0]
    if "model" in df.columns:
        vals = df["model"].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            return vals[0]
    return "model"


def plot_model_corr_bar(corr_df: pd.DataFrame, out_path: Path, metrics: List[str]) -> None:
    model_df = corr_df[corr_df["group_name"] == "model"].copy()
    model_df = _sorted_metric_df(model_df)
    model_df = model_df[model_df["metric"].isin(metrics)]

    metrics = model_df["metric"].tolist()
    x = np.arange(len(metrics))
    pearson_abs = pd.to_numeric(model_df["pearson_abs"], errors="coerce").fillna(0.0).values
    spearman_abs = pd.to_numeric(model_df["spearman_abs"], errors="coerce").fillna(0.0).values
    model_name = _single_model_name(model_df)

    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, pearson_abs, width, label="|Pearson|", color="#3b82f6")
    ax.bar(x + width / 2, spearman_abs, width, label="|Spearman|", color="#10b981")

    for thr, lbl in zip(THRESHOLDS, THRESHOLD_LABELS):
        ax.axhline(thr, color="#6b7280", linestyle="--", linewidth=1)
        ax.text(len(metrics) - 0.35, thr + 0.01, f"{lbl}={thr}", color="#4b5563", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([metric_display_name(m) for m in metrics], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Absolute correlation")
    ax.set_title(f"History Bias vs Prediction Bias Correlation ({model_name})")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_metric_relation_scatter(
    bias_pairs_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    out_path: Path,
    metrics: List[str],
) -> None:
    n = max(1, len(metrics))
    ncols = 2 if n <= 4 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.ravel()

    corr_model = corr_df[corr_df["group_name"] == "model"].set_index("metric")
    model_name = _single_model_name(corr_df[corr_df["group_name"] == "model"])

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sub = bias_pairs_df[bias_pairs_df["metric"] == metric].copy()
        if sub.empty:
            ax.set_axis_off()
            continue

        x = pd.to_numeric(sub["history_bias"], errors="coerce")
        y = pd.to_numeric(sub["prediction_bias_mean"], errors="coerce")
        valid = pd.DataFrame({"x": x, "y": y}).dropna()

        ax.scatter(valid["x"], valid["y"], s=18, alpha=0.45, color="#2563eb", edgecolors="none")

        if valid.shape[0] >= 2:
            coef = np.polyfit(valid["x"], valid["y"], 1)
            xx = np.linspace(valid["x"].min(), valid["x"].max(), 100)
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, color="#dc2626", linewidth=1.5)

        p = corr_model.loc[metric, "pearson"] if metric in corr_model.index else np.nan
        s = corr_model.loc[metric, "spearman"] if metric in corr_model.index else np.nan
        n = corr_model.loc[metric, "n"] if metric in corr_model.index else 0
        ax.set_title(f"{metric_display_name(metric)}\nN={int(n)}, P={p:.3f}, S={s:.3f}")
        ax.set_xlabel("History bias")
        ax.set_ylabel("Prediction bias mean")
        ax.grid(alpha=0.25)

    for j in range(len(metrics), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle(f"Relation Scatter: History Bias vs Prediction Bias ({model_name})", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_clean_model_bias_rank(clean_summary_df: pd.DataFrame, out_path: Path, metrics: List[str]) -> None:
    sub = clean_summary_df[clean_summary_df["metric"].isin(metrics)].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        return

    sub = _sorted_metric_df(sub)
    labels = [metric_display_name(m) for m in sub["metric"].tolist()]
    values = pd.to_numeric(sub["abs_bias_mean"], errors="coerce").values
    model_name = _single_model_name(sub)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(np.arange(len(labels)), values, color="#0ea5e9")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Abs bias mean")
    ax.set_title(f"Clean Prediction Bias by Metric ({model_name})")
    ax.grid(axis="y", alpha=0.25)

    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_readme_text(corr_df: pd.DataFrame, metrics: List[str]) -> List[str]:
    model_df = corr_df[corr_df["group_name"] == "model"].copy()
    model_df = _sorted_metric_df(model_df)

    lines: List[str] = []
    lines.append("Correlation threshold rule:")
    lines.append("- |r| < 0.2: none")
    lines.append("- 0.2 <= |r| < 0.4: weak")
    lines.append("- 0.4 <= |r| < 0.7: moderate")
    lines.append("- |r| >= 0.7: strong")
    lines.append("")
    model_name = _single_model_name(model_df)
    lines.append(f"Model-level correlation by metric ({model_name}):")

    model_df = model_df[model_df["metric"].isin(metrics)]

    for _, row in model_df.iterrows():
        metric = metric_display_name(str(row["metric"]))
        n = int(row["n"])
        p = float(row["pearson"]) if pd.notna(row["pearson"]) else np.nan
        s = float(row["spearman"]) if pd.notna(row["spearman"]) else np.nan
        ps = str(row.get("pearson_strength", ""))
        ss = str(row.get("spearman_strength", ""))
        lines.append(f"- {metric}: n={n}, Pearson={p:.3f} ({ps}), Spearman={s:.3f} ({ss})")

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot bias correlation analysis figures")
    parser.add_argument("--input-dir", default="results_analysis/bias_analysis")
    parser.add_argument("--output-dir", default="results_analysis/bias_analysis/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corr_df = load_csv(input_dir / "history_prediction_correlation.csv")
    bias_pairs_df = load_csv(input_dir / "bias_pairs.csv")
    clean_summary_df = load_csv(input_dir / "clean_model_bias_summary.csv")
    metrics = metrics_in_data(corr_df, bias_pairs_df, clean_summary_df)

    plot_model_corr_bar(corr_df, output_dir / "corr_model_bar.png", metrics=metrics)
    plot_metric_relation_scatter(
        bias_pairs_df=bias_pairs_df,
        corr_df=corr_df,
        out_path=output_dir / "relation_scatter_by_metric.png",
        metrics=metrics,
    )
    plot_clean_model_bias_rank(
        clean_summary_df=clean_summary_df,
        out_path=output_dir / "clean_model_bias_rank.png",
        metrics=metrics,
    )

    report_lines = build_readme_text(corr_df, metrics=metrics)
    (output_dir / "quick_interpretation.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[OK] Saved: {output_dir / 'corr_model_bar.png'}")
    print(f"[OK] Saved: {output_dir / 'relation_scatter_by_metric.png'}")
    print(f"[OK] Saved: {output_dir / 'clean_model_bias_rank.png'}")
    print(f"[OK] Saved: {output_dir / 'quick_interpretation.txt'}")


if __name__ == "__main__":
    main()
