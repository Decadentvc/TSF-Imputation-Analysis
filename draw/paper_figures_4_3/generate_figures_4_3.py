"""Generate candidate figures for Section 4.3.

The script reads existing experiment outputs and writes publication-oriented
figures under ``draw/paper_figures_4_3``. It intentionally combines aggregate
ablation summaries with window-level and record-level data so the figures show
more than table summaries.
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
DEFAULT_OUTPUT = ROOT / "draw" / "paper_figures_4_3"

CORE_MODELS = ["chronos2", "sundial", "timesfm2p0", "timesfm2p5", "visiontspp"]
MODEL_LABELS = {
    "chronos2": "Chronos-2",
    "sundial": "Sundial",
    "timesfm2p0": "TimesFM-2.0",
    "timesfm2p5": "TimesFM-2.5",
    "visiontspp": "VisionTS++",
    "kairos23m": "Kairos-23M",
    "kairos50m": "Kairos-50M",
}
METHOD_ORDER = ["mean", "forward", "linear", "gp_rbf", "saits"]
WINDOW_METHOD_ORDER = ["mean", "forward", "linear", "backward"]
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
MODEL_MARKERS = {
    "chronos2": "o",
    "sundial": "s",
    "timesfm2p0": "^",
    "timesfm2p5": "D",
    "visiontspp": "P",
    "kairos23m": "X",
    "kairos50m": "v",
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
            "lines.markersize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "data": output_root / "data",
        "aggregate": output_root / "aggregate_overview",
        "window": output_root / "window_landscape",
        "distribution": output_root / "error_distributions",
        "case": output_root / "forecast_cases",
        "structure": output_root / "structure_conditions",
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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def ordered_models(values: Iterable[str], include_extra: bool = False) -> list[str]:
    present = set(str(v) for v in values)
    order = list(CORE_MODELS)
    if include_extra:
        order += ["kairos23m", "kairos50m"]
    return [m for m in order if m in present]


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def clip_series(values: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lower = values.quantile(lower_q)
    upper = values.quantile(upper_q)
    return values.clip(lower=lower, upper=upper)


def plot_missing_ratio_sensitivity(ablation_dir: Path, output_dir: Path) -> list[Path]:
    ratio = read_csv(ablation_dir / "ablation_ratio_summary.csv")
    ratio = ratio[ratio["model"].isin(CORE_MODELS)].copy()

    model_curve = (
        ratio.groupby(["model", "missing_ratio"], as_index=False)["NRMSE[mean]"]
        .mean()
        .sort_values(["model", "missing_ratio"])
    )
    method_curve = (
        ratio.groupby(["imputer", "missing_ratio"], as_index=False)["mean_weighted_sum_quantile_loss"]
        .mean()
        .sort_values(["imputer", "missing_ratio"])
    )
    method_curve = method_curve[method_curve["imputer"].isin(METHOD_ORDER)]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=False)
    ax = axes[0]
    for model in ordered_models(model_curve["model"]):
        sub = model_curve[model_curve["model"] == model]
        ax.plot(
            sub["missing_ratio"] * 100,
            sub["NRMSE[mean]"],
            marker=MODEL_MARKERS.get(model, "o"),
            linewidth=2.5,
            markersize=7,
            markeredgecolor="#111111",
            markeredgewidth=0.6,
            label=model_label(model),
        )
    ax.set_title("Model sensitivity to missing ratio")
    ax.set_xlabel("Missing ratio (%)")
    ax.set_ylabel("Mean NRMSE")
    ax.legend(ncol=1, frameon=False)

    ax = axes[1]
    for method in METHOD_ORDER:
        sub = method_curve[method_curve["imputer"] == method]
        if sub.empty:
            continue
        ax.plot(
            sub["missing_ratio"] * 100,
            sub["mean_weighted_sum_quantile_loss"],
            marker=METHOD_MARKERS.get(method, "o"),
            linewidth=2.5,
            markersize=7,
            markeredgecolor="#111111",
            markeredgewidth=0.6,
            color=METHOD_COLORS[method],
            label=method_label(method),
        )
    ax.set_title("Imputation method loss across missing ratios")
    ax.set_xlabel("Missing ratio (%)")
    ax.set_ylabel("Mean weighted quantile loss")
    ax.legend(ncol=1, frameon=False)

    fig.suptitle("Missing-ratio ablation: aggregate trends across models and methods", y=1.04)
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_3_1_missing_ratio_sensitivity")


def compute_method_gap(df: pd.DataFrame, axis_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby(["model", "dataset", axis_col, "imputer"], as_index=False)["NRMSE[mean]"]
        .mean()
        .dropna()
    )
    rows = []
    for keys, sub in grouped.groupby(["model", "dataset", axis_col]):
        if sub["imputer"].nunique() < 2:
            continue
        rows.append(
            {
                "model": keys[0],
                "dataset": keys[1],
                axis_col: keys[2],
                "gap": sub["NRMSE[mean]"].max() - sub["NRMSE[mean]"].min(),
            }
        )
    if not rows:
        return pd.DataFrame(columns=[axis_col, "gap"])
    out = pd.DataFrame(rows)
    return out.groupby(axis_col, as_index=False)["gap"].mean().sort_values(axis_col)


def plot_method_gap_by_difficulty(ablation_dir: Path, output_dir: Path) -> list[Path]:
    ratio = read_csv(ablation_dir / "ablation_ratio_summary.csv")
    length = read_csv(ablation_dir / "ablation_length_summary.csv")
    position = read_csv(ablation_dir / "ablation_position_summary.csv")
    horizon = read_csv(ablation_dir / "ablation_horizon_summary.csv")
    context = read_csv(ablation_dir / "ablation_context_summary.csv")

    ratio = ratio[ratio["model"].isin(CORE_MODELS)]
    length = length[length["model"].isin(CORE_MODELS)]
    position = position[position["model"].isin(CORE_MODELS)]
    horizon = horizon[horizon["model"].isin(CORE_MODELS)]
    context = context[context["model"].isin(CORE_MODELS)]

    panels = [
        ("Missing ratio", "missing_ratio", compute_method_gap(ratio, "missing_ratio"), lambda x: x * 100, "%"),
        ("Block length", "block_length", compute_method_gap(length, "block_length"), lambda x: x, ""),
        ("Forecast horizon", "horizon", compute_method_gap(horizon, "horizon"), lambda x: x, ""),
        ("Context length", "context", compute_method_gap(context, "context"), lambda x: x, ""),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.1))
    for ax, (title, col, data, transform, suffix) in zip(axes.flat, panels):
        x = data[col].map(transform)
        ax.plot(
            x,
            data["gap"],
            color="#0077BB",
            marker="o",
            linewidth=2.6,
            markersize=7.5,
            markeredgecolor="#111111",
            markeredgewidth=0.6,
        )
        ax.set_title(title)
        ax.set_xlabel(title + (f" ({suffix})" if suffix else ""))
        ax.set_ylabel("Mean within-pair NRMSE gap")
        if title == "Context length":
            ax.set_xscale("log", base=2)
            ax.set_xticks(sorted(x.unique()))
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if title == "Missing ratio":
            ax.axvline(30, color="#777777", linestyle="--", linewidth=1.0, alpha=0.6)
    fig.suptitle("Imputation-choice gap across controlled difficulty factors", y=1.02)
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_3_2_method_gap_by_difficulty")


def load_window_points(window_root: Path, include_extra_models: bool) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(window_root.glob("*/*.csv")):
        try:
            df = pd.read_csv(
                csv_path,
                usecols=[
                    "dataset",
                    "term",
                    "ratio_code",
                    "prediction_mode",
                    "window_idx",
                    "distribution_gap",
                    "smape_diff_vs_clean",
                    "y_value",
                ],
            )
        except Exception:
            continue
        df["model"] = csv_path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    allowed_models = ordered_models(out["model"], include_extra=include_extra_models)
    out = out[out["model"].isin(allowed_models)].copy()
    out = out[out["prediction_mode"].isin(WINDOW_METHOD_ORDER)].copy()
    out["smape_diff"] = pd.to_numeric(out["smape_diff_vs_clean"], errors="coerce")
    out["distribution_gap"] = pd.to_numeric(out["distribution_gap"], errors="coerce")
    out = out.dropna(subset=["smape_diff", "distribution_gap"])
    return out


def plot_window_error_landscape(window_df: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    if window_df.empty:
        return []

    plot_df = window_df.copy()
    plot_df["smape_diff_clip"] = clip_series(plot_df["smape_diff"], 0.01, 0.99)
    x_upper = plot_df["distribution_gap"].quantile(0.995)
    plot_df = plot_df[plot_df["distribution_gap"] <= x_upper].copy()

    model_order = ordered_models(plot_df["model"])
    n_cols = 3
    n_rows = math.ceil(len(model_order) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.0, 3.2 * n_rows), sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()

    for ax, model in zip(axes_arr, model_order):
        sub_model = plot_df[plot_df["model"] == model]
        for method in WINDOW_METHOD_ORDER:
            sub = sub_model[sub_model["prediction_mode"] == method]
            if sub.empty:
                continue
            if len(sub) > 700:
                sub = sub.sample(n=700, random_state=17)
            ax.scatter(
                sub["distribution_gap"],
                sub["smape_diff_clip"],
                s=28,
                alpha=0.42,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS.get(method, "o"),
                label=method_label(method),
                edgecolors="white",
                linewidths=0.25,
            )
        ax.axhline(0, color="#222222", linewidth=1.2, alpha=0.75)
        ax.set_title(f"{model_label(model)} (n={len(sub_model)})")
        ax.set_xlabel("History-prediction distribution gap")
        ax.set_ylabel("sMAPE difference vs clean")
    for ax in axes_arr[len(model_order) :]:
        ax.axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Window-level error landscape: each point is one prediction window", y=1.06)
    fig.tight_layout()

    data_out = data_dir / "window_error_landscape_points.csv"
    plot_df.to_csv(data_out, index=False, encoding="utf-8-sig")
    return save_figure(fig, output_dir / "fig4_3_3_window_error_landscape")


def plot_window_error_distribution(window_df: pd.DataFrame, output_dir: Path, data_dir: Path) -> list[Path]:
    if window_df.empty:
        return []

    plot_df = window_df.copy()
    plot_df["smape_diff_clip"] = clip_series(plot_df["smape_diff"], 0.01, 0.99)
    arrays = [
        plot_df.loc[plot_df["prediction_mode"] == method, "smape_diff_clip"].to_numpy()
        for method in WINDOW_METHOD_ORDER
    ]
    positions = np.arange(1, len(WINDOW_METHOD_ORDER) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    parts = ax.violinplot(arrays, positions=positions, widths=0.78, showmeans=False, showmedians=False)
    for body, method in zip(parts["bodies"], WINDOW_METHOD_ORDER):
        body.set_facecolor(METHOD_COLORS[method])
        body.set_edgecolor("#333333")
        body.set_alpha(0.35)
    for key in ["cbars", "cmins", "cmaxes"]:
        parts[key].set_edgecolor("#555555")
        parts[key].set_linewidth(0.8)

    for i, (method, values) in enumerate(zip(WINDOW_METHOD_ORDER, arrays), start=1):
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.plot([i - 0.28, i + 0.28], [median, median], color="#111111", linewidth=1.8)
        ax.plot([i, i], [q25, q75], color="#111111", linewidth=4.0, solid_capstyle="butt")
        raw = plot_df[plot_df["prediction_mode"] == method]
        if len(raw) > 450:
            raw = raw.sample(n=450, random_state=31)
        jitter = np.random.default_rng(31 + i).normal(0, 0.035, size=len(raw))
        ax.scatter(
            np.full(len(raw), i) + jitter,
            raw["smape_diff_clip"],
            s=18,
            alpha=0.20,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS.get(method, "o"),
            edgecolors="white",
            linewidths=0.25,
        )

    ax.axhline(0, color="#222222", linewidth=1.2, alpha=0.75)
    ax.set_xticks(positions)
    ax.set_xticklabels([method_label(m) for m in WINDOW_METHOD_ORDER])
    ax.set_ylabel("sMAPE difference vs clean")
    ax.set_title("Window-level error distribution by imputation method")

    summary = (
        plot_df.groupby("prediction_mode")["smape_diff"]
        .agg(["count", "median", "mean", "std"])
        .reindex(WINDOW_METHOD_ORDER)
        .reset_index()
    )
    summary.to_csv(data_dir / "window_error_distribution_summary.csv", index=False, encoding="utf-8-sig")

    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_3_4_window_error_distribution")


def plot_forecast_case(sample_dir: Path, output_dir: Path, data_dir: Path) -> list[Path]:
    points_path = sample_dir / "sample_forward_linear_kalman_forecast_points.csv"
    metrics_path = sample_dir / "sample_forward_linear_kalman_forecast_metrics.csv"
    if not points_path.exists() or not metrics_path.exists():
        return []

    points = pd.read_csv(points_path)
    metrics = pd.read_csv(metrics_path)
    points["timestamp"] = pd.to_datetime(points["timestamp"])

    series = [
        ("ground_truth", "Ground truth", "#000000", 2.9),
        ("clean_input", "Clean input", "#777777", 2.2),
        ("forward", "Forward", METHOD_COLORS["forward"], 2.2),
        ("linear", "Linear", METHOD_COLORS["linear"], 2.2),
        ("kalman_arima", "Kalman-ARIMA", METHOD_COLORS["kalman_arima"], 2.2),
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.6, 5.7),
        gridspec_kw={"height_ratios": [3.1, 1.2]},
        constrained_layout=True,
    )
    ax = axes[0]
    for col, label, color, width in series:
        if col not in points.columns:
            continue
        ax.plot(points["timestamp"], points[col], label=label, color=color, linewidth=width)
    ax.set_title("Representative forecast trajectory after different imputations")
    ax.set_ylabel("Value")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    ax = axes[1]
    metrics = metrics.copy()
    metrics["series_key"] = metrics["series_key"].astype(str)
    metrics = metrics[metrics["series_key"].isin(["clean", "forward", "linear", "kalman_arima"])]
    metrics = metrics.sort_values("sMAPE")
    colors = [
        "#777777" if key == "clean" else METHOD_COLORS.get(key, "#0077BB")
        for key in metrics["series_key"]
    ]
    ax.barh(metrics["series"], metrics["sMAPE"], color=colors, alpha=0.82)
    ax.set_xlabel("Forecast sMAPE")
    ax.set_title("Same window, different downstream errors")
    for y, value in enumerate(metrics["sMAPE"]):
        ax.text(value, y, f" {value:.3f}", va="center", fontsize=8)

    points.to_csv(data_dir / "representative_forecast_case_points.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(data_dir / "representative_forecast_case_metrics.csv", index=False, encoding="utf-8-sig")
    return save_figure(fig, output_dir / "fig4_3_5_representative_forecast_case")


def normalize_structure_methods(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["imputation_method_norm"] = out["imputation_method_norm"].astype(str).replace({"后向": "backward"})
    return out


def plot_structure_condition_gain(structure_path: Path, output_dir: Path, data_dir: Path) -> list[Path]:
    usecols = [
        "model",
        "dataset",
        "missing_ratio",
        "imputation_method_norm",
        "relative_smape_gain",
        "score_full",
    ]
    records = read_csv(structure_path)
    records = records[usecols].copy()
    records = normalize_structure_methods(records)
    records = records[records["imputation_method_norm"].isin(METHOD_ORDER)].copy()
    records = records.replace([np.inf, -np.inf], np.nan).dropna(subset=["relative_smape_gain", "score_full"])

    records["score_decile"] = pd.qcut(records["score_full"], q=10, labels=False, duplicates="drop") + 1
    grouped = (
        records.groupby(["imputation_method_norm", "score_decile"], as_index=False)
        .agg(
            score_full=("score_full", "median"),
            gain_median=("relative_smape_gain", "median"),
            gain_q25=("relative_smape_gain", lambda s: s.quantile(0.25)),
            gain_q75=("relative_smape_gain", lambda s: s.quantile(0.75)),
            count=("relative_smape_gain", "size"),
        )
        .sort_values(["imputation_method_norm", "score_decile"])
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for method in METHOD_ORDER:
        sub = grouped[grouped["imputation_method_norm"] == method]
        if sub.empty:
            continue
        x = sub["score_decile"].to_numpy()
        y = sub["gain_median"].to_numpy()
        lo = sub["gain_q25"].to_numpy()
        hi = sub["gain_q75"].to_numpy()
        ax.plot(
            x,
            y,
            linewidth=2.4,
            markersize=7,
            marker=METHOD_MARKERS.get(method, "o"),
            markeredgecolor="#111111",
            markeredgewidth=0.55,
            color=METHOD_COLORS[method],
            label=method_label(method),
        )
        ax.fill_between(x, lo, hi, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
    ax.axhline(0, color="#222222", linewidth=1.2, alpha=0.75)
    ax.set_xlabel("Full structural-difference score decile")
    ax.set_ylabel("Relative sMAPE change vs clean")
    ax.set_title("Structural conditions modulate imputation impact")
    ax.legend(ncol=3, frameon=False)
    ax.set_xticks(sorted(grouped["score_decile"].unique()))

    records.to_csv(data_dir / "structure_condition_records_used.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(data_dir / "structure_condition_gain_by_decile.csv", index=False, encoding="utf-8-sig")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_3_6_structure_condition_gain")


def plot_structure_channel_heatmap(structure_summary_path: Path, output_dir: Path, data_dir: Path) -> list[Path]:
    summary = read_csv(structure_summary_path)
    summary = summary[summary["target"] == "relative_smape_gain"].copy()
    order = [
        "trend_only",
        "seasonal_only",
        "residual_only",
        "frequency_only",
        "drop_trend",
        "drop_seasonal",
        "drop_residual",
        "drop_frequency",
        "full",
    ]
    summary["combination"] = pd.Categorical(summary["combination"], categories=order, ordered=True)
    summary = summary.sort_values("combination")
    values = summary[["median_spearman", "overall_spearman", "overall_linear_r2"]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    im = ax.imshow(values, cmap="viridis", aspect="auto", vmin=0, vmax=max(0.25, np.nanmax(values)))
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary["combination"].astype(str).str.replace("_", " "))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Median Spearman", "Overall Spearman", "Linear R2"])
    ax.set_title("Structure-channel explanatory strength for sMAPE gain")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Statistic value")

    summary.to_csv(data_dir / "structure_channel_summary_used.csv", index=False, encoding="utf-8-sig")
    fig.tight_layout()
    return save_figure(fig, output_dir / "fig4_3_7_structure_channel_heatmap")


def write_manifest(paths: list[Path], output_root: Path) -> None:
    manifest = output_root / "figure_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "kind"])
        for path in sorted(paths):
            writer.writerow([str(path.relative_to(ROOT)).replace("\\", "/"), path.suffix.lstrip(".")])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Section 4.3 candidate figures.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT), help="Output root under draw/")
    parser.add_argument("--ablation_dir", default="results_analysis/ablation")
    parser.add_argument("--window_dir", default="draw/outputs_by_dataset")
    parser.add_argument("--sample_dir", default="tools/Sample")
    parser.add_argument(
        "--structure_records",
        default="results_analysis/structure_metric_ablation/structure_metric_ablation_records.csv",
    )
    parser.add_argument(
        "--structure_summary",
        default="results_analysis/structure_metric_ablation/structure_metric_ablation_summary.csv",
    )
    parser.add_argument(
        "--include_extra_models",
        action="store_true",
        help="Include Kairos models in window-level figures when available.",
    )
    return parser


def main() -> None:
    configure_matplotlib()
    args = build_parser().parse_args()

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    dirs = ensure_dirs(output_root)

    written: list[Path] = []
    ablation_dir = ROOT / args.ablation_dir
    window_root = ROOT / args.window_dir
    sample_dir = ROOT / args.sample_dir
    structure_records = ROOT / args.structure_records
    structure_summary = ROOT / args.structure_summary

    written += plot_missing_ratio_sensitivity(ablation_dir, dirs["aggregate"])
    written += plot_method_gap_by_difficulty(ablation_dir, dirs["aggregate"])

    window_df = load_window_points(window_root, include_extra_models=args.include_extra_models)
    if not window_df.empty:
        window_df.to_csv(dirs["data"] / "window_points_used.csv", index=False, encoding="utf-8-sig")
    written += plot_window_error_landscape(window_df, dirs["window"], dirs["data"])
    written += plot_window_error_distribution(window_df, dirs["distribution"], dirs["data"])

    written += plot_forecast_case(sample_dir, dirs["case"], dirs["data"])
    written += plot_structure_condition_gain(structure_records, dirs["structure"], dirs["data"])
    written += plot_structure_channel_heatmap(structure_summary, dirs["structure"], dirs["data"])

    write_manifest(written, output_root)
    for path in written:
        print(path.relative_to(ROOT))
    print(f"manifest: {output_root.relative_to(ROOT) / 'figure_manifest.csv'}")


if __name__ == "__main__":
    main()
