"""Generate paper-ready candidate figures for Section 4.2.

The figures are designed around the seven findings currently stated in the
manuscript section. They use the main 4.2 record table rather than only the
pre-aggregated summaries, so most panels retain record-level or combination-level
detail.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "draw" / "paper_figures_4_2"

MODEL_ORDER = [
    "VisionTS++",
    "Kairos-50M",
    "Kairos-23M",
    "TimesFM-2.5",
    "Sundial",
    "Chronos-2",
    "TimesFM-2.0",
]
MODEL_LABELS = {
    "visiontspp": "VisionTS++",
    "kairos50m": "Kairos-50M",
    "kairos23m": "Kairos-23M",
    "timesfm2p5": "TimesFM-2.5",
    "sundial": "Sundial",
    "chronos2": "Chronos-2",
    "timesfm2p0": "TimesFM-2.0",
}
METHOD_MAP = {
    "/": "clean",
    "均值": "mean",
    "前项": "forward",
    "后向": "backward",
    "线性": "linear",
    "kalman_struct": "kalman_struct",
    "kalman_arima": "kalman_arima",
    "gp_rbf": "gp_rbf",
    "saits": "saits",
}
METHOD_ORDER = [
    "saits",
    "gp_rbf",
    "kalman_arima",
    "linear",
    "kalman_struct",
    "backward",
    "forward",
    "mean",
]
METHOD_LABELS = {
    "mean": "Mean",
    "forward": "Forward",
    "backward": "Backward",
    "linear": "Linear",
    "gp_rbf": "GP-RBF",
    "saits": "SAITS",
    "kalman_arima": "Kalman-ARIMA",
    "kalman_struct": "Kalman-Struct",
}
METHOD_COLORS = {
    "mean": "#0077BB",
    "forward": "#33BBEE",
    "backward": "#BBBBBB",
    "linear": "#EE7733",
    "gp_rbf": "#EE3377",
    "saits": "#009988",
    "kalman_arima": "#CC3311",
    "kalman_struct": "#000000",
}
METHOD_MARKERS = {
    "mean": "o",
    "forward": "s",
    "backward": "D",
    "linear": "^",
    "gp_rbf": "P",
    "saits": "X",
    "kalman_arima": "v",
    "kalman_struct": "*",
}
STRUCTURE_METRICS = [
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr",
    "spectral_entropy",
]
STRUCTURE_LABELS = {
    "trend_strength": "Trend strength",
    "trend_linearity": "Trend linearity",
    "seasonal_strength": "Seasonal strength",
    "seasonal_correlation": "Seasonal corr.",
    "residual_autocorr": "Residual ACF",
    "spectral_entropy": "Spectral entropy",
}


def configure_matplotlib() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "lines.linewidth": 2.4,
            "lines.markersize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "data": output_root / "data",
        "model": output_root / "model_robustness",
        "reconstruction": output_root / "reconstruction_vs_forecast",
        "method": output_root / "method_preference",
        "ratio": output_root / "missing_ratio",
        "dataset": output_root / "dataset_sensitivity",
        "structure": output_root / "structure_explanation",
        "contraction": output_root / "prediction_structure",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def save_figure(fig: plt.Figure, output_path: Path) -> list[Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    png = output_path.with_suffix(".png")
    svg = output_path.with_suffix(".svg")
    fig.savefig(png)
    fig.savefig(svg)
    plt.close(fig)
    return [png, svg]


def find_default_input() -> Path:
    matches = sorted((ROOT / "results_analysis").glob("*0509.csv"))
    if not matches:
        raise FileNotFoundError("No Section 4.2 source CSV matching *0509.csv")
    return matches[0]


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def parse_main_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], encoding="utf-8-sig")
    df.columns = [
        "model_raw",
        "dataset",
        "missing_ratio",
        "method_raw",
        "imputation_smape",
        "forecast_mse",
        "forecast_smape",
        "history_trend_strength",
        "history_trend_linearity",
        "history_seasonal_strength",
        "history_seasonal_correlation",
        "history_residual_autocorr",
        "history_spectral_entropy",
        "prediction_trend_strength",
        "prediction_trend_linearity",
        "prediction_seasonal_strength",
        "prediction_seasonal_correlation",
        "prediction_residual_autocorr",
        "prediction_spectral_entropy",
    ]
    for col in ["model_raw", "dataset", "missing_ratio"]:
        df[col] = df[col].replace("", pd.NA).ffill()
    df["model"] = df["model_raw"].map(MODEL_LABELS).fillna(df["model_raw"])
    df["method"] = df["method_raw"].astype(str).str.strip().map(METHOD_MAP).fillna(df["method_raw"].astype(str).str.strip())
    numeric_cols = [c for c in df.columns if c not in {"model_raw", "dataset", "method_raw", "model", "method"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].replace("/", pd.NA), errors="coerce")
    return df


def attach_baseline(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[(df["missing_ratio"].eq(0)) & (df["method"].eq("clean"))].copy()
    baseline_cols = ["forecast_mse", "forecast_smape", *(f"history_{m}" for m in STRUCTURE_METRICS)]
    clean = clean.drop_duplicates(["model", "dataset"])[["model", "dataset", *baseline_cols]].rename(
        columns={col: f"clean_{col}" for col in baseline_cols}
    )

    records = df[df["missing_ratio"].gt(0)].merge(clean, on=["model", "dataset"], how="left")
    records["relative_mse_change"] = (
        records["forecast_mse"] - records["clean_forecast_mse"]
    ) / records["clean_forecast_mse"].abs()
    records["relative_smape_change"] = (
        records["forecast_smape"] - records["clean_forecast_smape"]
    ) / records["clean_forecast_smape"].abs()
    records["mse_better_than_clean"] = records["relative_mse_change"] < 0

    for metric in STRUCTURE_METRICS:
        records[f"delta_{metric}"] = (
            records[f"history_{metric}"] - records[f"clean_history_{metric}"]
        ).abs()
        records[f"prediction_history_shift_{metric}"] = (
            records[f"prediction_{metric}"] - records[f"history_{metric}"]
        )

    denominators = {}
    for metric in STRUCTURE_METRICS:
        values = records[f"delta_{metric}"].dropna().abs()
        denom = float(values.quantile(0.95)) if not values.empty else 1.0
        denominators[metric] = denom if denom else 1.0
        records[f"norm_delta_{metric}"] = records[f"delta_{metric}"] / denominators[metric]
    records["history_structure_drift"] = records[[f"norm_delta_{m}" for m in STRUCTURE_METRICS]].mean(axis=1)
    return records


def prepare_records(input_path: Path, data_dir: Path) -> pd.DataFrame:
    raw = parse_main_table(input_path)
    records = attach_baseline(raw)
    raw.to_csv(data_dir / "section_4_2_raw_records_flat.csv", index=False, encoding="utf-8-sig")
    records.to_csv(data_dir / "section_4_2_imputed_records_with_baseline.csv", index=False, encoding="utf-8-sig")
    return records


def clip(values: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return values.clip(values.quantile(lo), values.quantile(hi))


def ordered(values: Iterable[str], order: list[str]) -> list[str]:
    present = set(str(v) for v in values)
    return [item for item in order if item in present] + sorted(present - set(order))


def plot_model_robustness(records: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    df = records.copy()
    df["mse_change_pct"] = df["relative_mse_change"] * 100
    df["mse_change_clip"] = clip(df["mse_change_pct"])
    models = ordered(df["model"], MODEL_ORDER)
    data = [df.loc[df["model"] == model, "mse_change_clip"].dropna().to_numpy() for model in models]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    box = ax.boxplot(data, vert=False, patch_artist=True, showfliers=False, widths=0.62)
    for patch in box["boxes"]:
        patch.set_facecolor("#A6CEE3")
        patch.set_alpha(0.62)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(1.2)
    for key in ["whiskers", "caps", "medians"]:
        for item in box[key]:
            item.set_color("#222222")
            item.set_linewidth(1.4 if key != "medians" else 2.2)

    summary_rows = []
    for y, model in enumerate(models, start=1):
        sub = df[df["model"] == model]
        p90 = sub["mse_change_pct"].quantile(0.90)
        median = sub["mse_change_pct"].median()
        better = sub["mse_better_than_clean"].mean()
        ax.scatter(
            p90,
            y,
            marker="^",
            s=78,
            color="#CC3311",
            edgecolors="#111111",
            linewidths=0.45,
            zorder=3,
            label="90th percentile" if y == 1 else None,
        )
        summary_rows.append({"model": model, "median_pct": median, "p90_pct": p90, "better_than_clean": better})
    ax.axvline(0, color="#222222", linewidth=1.2)
    ax.set_yticks(range(1, len(models) + 1))
    ax.set_yticklabels(models)
    ax.set_xlabel("MSE change vs clean (%)")
    ax.set_title("Model robustness differs under imputed block-missing inputs")
    ax.legend(frameon=False, loc="lower right")

    pd.DataFrame(summary_rows).to_csv(data_dir / "model_robustness_summary.csv", index=False, encoding="utf-8-sig")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_2_1_model_robustness_distribution")


def plot_reconstruction_vs_forecast(records: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    df = records.copy()
    df["mse_change_pct"] = df["relative_mse_change"] * 100
    df["mse_change_clip"] = clip(df["mse_change_pct"], 0.005, 0.99)
    df["imputation_smape_clip"] = df["imputation_smape"].clip(upper=df["imputation_smape"].quantile(0.995))

    fig, ax = plt.subplots(figsize=(7.1, 4.7))
    for method in METHOD_ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ax.scatter(
            sub["imputation_smape_clip"],
            sub["mse_change_clip"],
            s=30,
            alpha=0.42,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            edgecolors="white",
            linewidths=0.25,
            label=method_label(method),
        )
    bins = pd.qcut(df["imputation_smape_clip"], q=12, duplicates="drop")
    trend = (
        df.assign(bin=bins)
        .groupby("bin", observed=True)
        .agg(x=("imputation_smape_clip", "median"), y=("mse_change_pct", "median"))
        .dropna()
    )
    ax.plot(trend["x"], trend["y"], color="#111111", linewidth=2.8, marker="o", markersize=6, label="Binned median")
    ax.axhline(0, color="#222222", linewidth=1.2)
    ax.set_xlabel("Point-level imputation sMAPE")
    ax.set_ylabel("MSE change vs clean (%)")
    ax.set_title("Point reconstruction error is informative but insufficient")
    ax.legend(ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.22), loc="upper center")
    df.to_csv(data_dir / "reconstruction_vs_forecast_points.csv", index=False, encoding="utf-8-sig")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_2_2_reconstruction_vs_forecast_scatter")


def best_method_counts(records: pd.DataFrame) -> pd.DataFrame:
    idx = records.groupby(["model", "dataset", "missing_ratio"])["relative_mse_change"].idxmin()
    best = records.loc[idx, ["model", "dataset", "missing_ratio", "method", "relative_mse_change"]].copy()
    return best.groupby("method", as_index=False).size().rename(columns={"size": "best_count"})


def plot_method_preference(records: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    matrix = (
        records.groupby(["model", "method"])["relative_mse_change"]
        .median()
        .mul(100)
        .unstack("method")
        .reindex(index=MODEL_ORDER, columns=METHOD_ORDER)
    )
    counts = best_method_counts(records).set_index("method").reindex(METHOD_ORDER).fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), gridspec_kw={"width_ratios": [3.1, 1.0]})
    ax = axes[0]
    values = matrix.to_numpy(dtype=float)
    limit = np.nanmax(np.abs(values))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    im = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels([method_label(m) for m in METHOD_ORDER], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER)
    ax.set_title("Median MSE change by model and imputation")
    for i in range(values.shape[0]):
        best_j = int(np.nanargmin(values[i, :]))
        for j in range(values.shape[1]):
            color = "white" if abs(values[i, j]) > limit * 0.52 else "black"
            weight = "bold" if j == best_j else "normal"
            ax.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", fontsize=8.4, color=color, fontweight=weight)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Median MSE change (%)")

    ax = axes[1]
    bar_colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    ax.barh([method_label(m) for m in METHOD_ORDER], counts["best_count"], color=bar_colors, alpha=0.86)
    ax.invert_yaxis()
    ax.set_xlabel("Best-count")
    ax.set_title("Best method frequency")
    for y, value in enumerate(counts["best_count"]):
        ax.text(value, y, f" {int(value)}", va="center", fontsize=9)
    matrix.to_csv(data_dir / "model_method_median_matrix.csv", encoding="utf-8-sig")
    counts.reset_index().to_csv(data_dir / "method_best_count.csv", index=False, encoding="utf-8-sig")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_2_3_model_method_preference")


def classify_ratio_trajectory(values: dict[float, float]) -> str:
    a, b, c = values[0.1], values[0.2], values[0.3]
    if a < b < c:
        return "strictly increasing"
    if a > b > c:
        return "strictly decreasing"
    return "non-monotonic"


def plot_missing_ratio_trajectories(records: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    rows = []
    for key, group in records.groupby(["model", "dataset", "method"]):
        values = group.set_index("missing_ratio")["relative_mse_change"].to_dict()
        if not all(r in values for r in [0.1, 0.2, 0.3]):
            continue
        cls = classify_ratio_trajectory(values)
        for ratio, value in values.items():
            rows.append({"model": key[0], "dataset": key[1], "method": key[2], "missing_ratio": ratio, "mse_change_pct": value * 100, "class": cls})
    traj = pd.DataFrame(rows)
    class_order = ["strictly increasing", "non-monotonic", "strictly decreasing"]
    class_colors = {
        "strictly increasing": "#CC3311",
        "non-monotonic": "#777777",
        "strictly decreasing": "#0077BB",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for _, group in traj.groupby(["model", "dataset", "method"]):
        cls = group["class"].iloc[0]
        ax.plot(
            group.sort_values("missing_ratio")["missing_ratio"] * 100,
            group.sort_values("missing_ratio")["mse_change_pct"],
            color=class_colors[cls],
            linewidth=1.1,
            alpha=0.10 if cls == "non-monotonic" else 0.18,
        )
    median_line = traj.groupby("missing_ratio", as_index=False)["mse_change_pct"].median()
    ax.plot(
        median_line["missing_ratio"] * 100,
        median_line["mse_change_pct"],
        color="#000000",
        marker="o",
        markersize=8,
        linewidth=3.0,
        label="Median",
    )
    counts = traj.drop_duplicates(["model", "dataset", "method"])["class"].value_counts().reindex(class_order).fillna(0)
    for cls in class_order:
        ax.plot([], [], color=class_colors[cls], linewidth=3.0, label=f"{cls}: {int(counts[cls])}")
    ax.axhline(0, color="#222222", linewidth=1.2)
    ax.set_xlabel("Missing ratio (%)")
    ax.set_ylabel("MSE change vs clean (%)")
    ax.set_title("Missing ratio amplifies risk but many local trajectories are not monotonic")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    traj.to_csv(data_dir / "missing_ratio_trajectories.csv", index=False, encoding="utf-8-sig")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_2_4_missing_ratio_trajectories")


def plot_dataset_sensitivity(records: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    summary = (
        records.groupby("dataset")["relative_mse_change"]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), count="size")
        .mul({"median": 100, "q25": 100, "q75": 100, "count": 1})
        .sort_values("median")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    y = np.arange(len(summary))
    colors = np.where(summary["median"] >= 0, "#EE7733", "#0077BB")
    ax.hlines(y, summary["q25"], summary["q75"], color="#555555", linewidth=2.0, alpha=0.75)
    ax.scatter(summary["median"], y, s=64, color=colors, edgecolors="#111111", linewidths=0.45, zorder=3)
    ax.axvline(0, color="#222222", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(summary["dataset"])
    ax.set_xlabel("MSE change vs clean (%)")
    ax.set_title("Dataset sensitivity varies substantially across domains")
    summary.to_csv(data_dir / "dataset_sensitivity_all_datasets.csv", index=False, encoding="utf-8-sig")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_2_5_dataset_sensitivity_ranked")


def spearman_by_group(records: pd.DataFrame, x_col: str) -> tuple[float, float]:
    rhos = []
    for _, group in records.groupby(["model", "dataset", "missing_ratio"]):
        if group[x_col].nunique(dropna=True) < 2 or group["relative_mse_change"].nunique(dropna=True) < 2:
            continue
        pair = group[[x_col, "relative_mse_change"]].dropna()
        if len(pair) < 3:
            continue
        x_rank = pair[x_col].rank(method="average")
        y_rank = pair["relative_mse_change"].rank(method="average")
        rho = x_rank.corr(y_rank)
        if pd.notna(rho):
            rhos.append(float(rho))
    return float(np.median(rhos)), float(np.mean(np.array(rhos) > 0))


def plot_structure_explanation(records: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    df = records.copy()
    df["mse_change_pct"] = df["relative_mse_change"] * 100
    df["mse_change_clip"] = clip(df["mse_change_pct"], 0.005, 0.99)

    specs = [
        ("imputation_smape", "Point-level imputation sMAPE", "#0077BB"),
        ("history_structure_drift", "History-structure drift", "#CC3311"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), sharey=True)
    stat_rows = []
    for ax, (col, label, color) in zip(axes, specs):
        plot_df = df[[col, "mse_change_clip", "mse_change_pct", "relative_mse_change"]].dropna()
        if len(plot_df) > 1400:
            scatter_df = plot_df.sample(1400, random_state=42)
        else:
            scatter_df = plot_df
        ax.scatter(
            scatter_df[col],
            scatter_df["mse_change_clip"],
            s=22,
            alpha=0.20,
            color=color,
            edgecolors="none",
        )
        bins = pd.qcut(plot_df[col], q=12, duplicates="drop")
        trend = (
            plot_df.assign(bin=bins)
            .groupby("bin", observed=True)
            .agg(x=(col, "median"), y=("mse_change_pct", "median"), q25=("mse_change_pct", lambda s: s.quantile(0.25)), q75=("mse_change_pct", lambda s: s.quantile(0.75)))
            .dropna()
        )
        ax.plot(trend["x"], trend["y"], color="#111111", marker="o", linewidth=2.8, markersize=7)
        ax.fill_between(trend["x"], trend["q25"], trend["q75"], color=color, alpha=0.16)
        median_rho, positive_ratio = spearman_by_group(df, col)
        ax.text(
            0.03,
            0.95,
            f"Median group Spearman = {median_rho:.3f}\nPositive groups = {positive_ratio * 100:.1f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.92},
        )
        ax.axhline(0, color="#222222", linewidth=1.2)
        ax.set_xlabel(label)
        ax.set_title(label)
        stat_rows.append({"x": col, "median_group_spearman": median_rho, "positive_group_ratio": positive_ratio})
    axes[0].set_ylabel("MSE change vs clean (%)")
    fig.suptitle("Structural drift provides a more stable link to downstream error", y=1.03)
    pd.DataFrame(stat_rows).to_csv(data_dir / "structure_explanation_spearman_summary.csv", index=False, encoding="utf-8-sig")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_2_6_structure_vs_reconstruction_explanation")


def plot_prediction_structure(records: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    rows = []
    for metric in STRUCTURE_METRICS:
        shift = records[f"prediction_history_shift_{metric}"].dropna()
        rows.append(
            {
                "metric": metric,
                "label": STRUCTURE_LABELS[metric],
                "median_shift": float(shift.median()),
                "q25": float(shift.quantile(0.25)),
                "q75": float(shift.quantile(0.75)),
                "positive_ratio": float((shift > 0).mean()),
            }
        )
    summary = pd.DataFrame(rows)

    non_entropy = summary[summary["metric"] != "spectral_entropy"].copy()
    entropy = summary[summary["metric"] == "spectral_entropy"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [2.4, 1.0]})

    ax = axes[0]
    y = np.arange(len(non_entropy))
    colors = np.where(non_entropy["median_shift"] >= 0, "#009988", "#CC3311")
    ax.hlines(y, non_entropy["q25"], non_entropy["q75"], color="#555555", linewidth=2.2)
    ax.scatter(non_entropy["median_shift"], y, s=78, color=colors, marker="o", edgecolors="#111111", linewidths=0.45)
    ax.axvline(0, color="#222222", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(non_entropy["label"])
    ax.set_xlabel("Prediction-history structural shift")
    ax.set_title("Shape metrics")
    for yy, ratio in zip(y, non_entropy["positive_ratio"]):
        ax.text(ax.get_xlim()[1], yy, f" {ratio * 100:.1f}% > 0", va="center", fontsize=8)

    ax = axes[1]
    ax.barh(entropy["label"], entropy["median_shift"], color="#0077BB", alpha=0.84)
    ax.axvline(0, color="#222222", linewidth=1.2)
    ax.set_xlabel("Shift")
    ax.set_title("Spectral entropy")
    if not entropy.empty:
        ax.text(entropy["median_shift"].iloc[0], 0, f" {entropy['median_shift'].iloc[0]:.0f}", va="center", fontsize=9)

    summary.to_csv(data_dir / "prediction_history_structure_shift_summary.csv", index=False, encoding="utf-8-sig")
    fig.suptitle("Forecast outputs show a stable structural shift relative to imputed histories", y=1.03)
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_2_7_prediction_structure_shift")


def write_manifest(paths: list[Path], output_root: Path) -> None:
    manifest = output_root / "figure_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "kind"])
        for path in sorted(paths):
            writer.writerow([str(path.relative_to(ROOT)).replace("\\", "/"), path.suffix.lstrip(".")])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Section 4.2 candidate figures.")
    parser.add_argument("--input_csv", default=None, help="Main Section 4.2 result CSV. Default: results_analysis/*0509.csv")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT), help="Output root under draw/")
    return parser


def main() -> None:
    configure_matplotlib()
    args = build_parser().parse_args()
    input_path = Path(args.input_csv) if args.input_csv else find_default_input()
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    dirs = ensure_dirs(output_root)

    records = prepare_records(input_path, dirs["data"])
    written: list[Path] = []
    written += plot_model_robustness(records, dirs["model"], dirs["data"])
    written += plot_reconstruction_vs_forecast(records, dirs["reconstruction"], dirs["data"])
    written += plot_method_preference(records, dirs["method"], dirs["data"])
    written += plot_missing_ratio_trajectories(records, dirs["ratio"], dirs["data"])
    written += plot_dataset_sensitivity(records, dirs["dataset"], dirs["data"])
    written += plot_structure_explanation(records, dirs["structure"], dirs["data"])
    written += plot_prediction_structure(records, dirs["contraction"], dirs["data"])

    write_manifest(written, output_root)
    for path in written:
        print(path.relative_to(ROOT))
    print(f"manifest: {output_root.relative_to(ROOT) / 'figure_manifest.csv'}")


if __name__ == "__main__":
    main()
