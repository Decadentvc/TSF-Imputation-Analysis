"""Ablate history-window structure channels against relative forecast error."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


METRIC_LABELS = {
    "趋势强度": "trend_strength",
    "趋势线性度": "trend_linearity",
    "季节强度": "seasonal_strength",
    "季节相关性": "seasonal_correlation",
    "残差自相关性": "residual_autocorr_lag1",
    "谱熵": "spectral_entropy",
}

HISTORY_METRICS = [
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "spectral_entropy",
]

CHANNELS = {
    "trend": ["trend_strength", "trend_linearity"],
    "seasonal": ["seasonal_strength", "seasonal_correlation"],
    "residual": ["residual_autocorr_lag1"],
    "frequency": ["spectral_entropy"],
}

COMBINATIONS = {
    "trend_only": CHANNELS["trend"],
    "seasonal_only": CHANNELS["seasonal"],
    "residual_only": CHANNELS["residual"],
    "frequency_only": CHANNELS["frequency"],
    "drop_trend": CHANNELS["seasonal"] + CHANNELS["residual"] + CHANNELS["frequency"],
    "drop_seasonal": CHANNELS["trend"] + CHANNELS["residual"] + CHANNELS["frequency"],
    "drop_residual": CHANNELS["trend"] + CHANNELS["seasonal"] + CHANNELS["frequency"],
    "drop_frequency": CHANNELS["trend"] + CHANNELS["seasonal"] + CHANNELS["residual"],
    "full": HISTORY_METRICS,
}

METHOD_MAP = {
    "/": "clean",
    "均值": "mean",
    "前项": "forward",
    "后项": "backward",
    "线性": "linear",
    "GP-RBF": "gp_rbf",
    "SAITS": "saits",
    "Kalman-Struct": "kalman_struct",
    "Kalman-ARIMA": "kalman_arima",
}


def _flatten_columns(columns: pd.MultiIndex) -> List[str]:
    names: List[str] = []
    for idx, (top, bottom) in enumerate(columns):
        top = str(top).strip()
        bottom = str(bottom).strip()
        if idx == 0:
            names.append("model")
        elif idx == 1:
            names.append("dataset")
        elif idx == 2:
            names.append("missing_ratio")
        elif idx == 3:
            names.append("imputation_method")
        elif idx == 4:
            names.append("imputation_smape")
        elif idx == 5:
            names.append("forecast_mse")
        elif idx == 6:
            names.append("forecast_smape")
        elif 7 <= idx <= 12:
            names.append(f"history_{METRIC_LABELS.get(bottom, bottom)}")
        elif 13 <= idx <= 18:
            names.append(f"prediction_{METRIC_LABELS.get(bottom, bottom)}")
        else:
            names.append(bottom if bottom and not bottom.startswith("Unnamed") else top)
    return names


def load_results_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], encoding="utf-8-sig")
    df.columns = _flatten_columns(df.columns)

    for col in ["model", "dataset", "missing_ratio"]:
        df[col] = df[col].replace("", np.nan).ffill()

    df["imputation_method"] = df["imputation_method"].replace("", np.nan)
    df["imputation_method_norm"] = (
        df["imputation_method"].astype(str).str.strip().map(METHOD_MAP).fillna(df["imputation_method"].astype(str).str.strip())
    )

    numeric_cols = [
        "missing_ratio",
        "imputation_smape",
        "forecast_mse",
        "forecast_smape",
        *(f"history_{metric}" for metric in HISTORY_METRICS),
        *(f"prediction_{metric}" for metric in HISTORY_METRICS),
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def attach_clean_baseline(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["missing_ratio"].fillna(-1).eq(0)].copy()
    if clean.empty:
        raise ValueError("No clean baseline rows found")

    baseline_cols = ["forecast_mse", "forecast_smape", *(f"history_{metric}" for metric in HISTORY_METRICS)]
    clean = clean.sort_values(["model", "dataset"]).drop_duplicates(["model", "dataset"])
    clean = clean[["model", "dataset", *baseline_cols]].rename(
        columns={col: f"clean_{col}" for col in baseline_cols}
    )

    merged = df.merge(clean, on=["model", "dataset"], how="left")
    merged = merged[merged["missing_ratio"].fillna(-1).gt(0)].copy()

    merged["relative_mse_gain"] = (
        (merged["forecast_mse"] - merged["clean_forecast_mse"]) / merged["clean_forecast_mse"].abs()
    )
    merged["relative_smape_gain"] = (
        (merged["forecast_smape"] - merged["clean_forecast_smape"]) / merged["clean_forecast_smape"].abs()
    )

    for metric in HISTORY_METRICS:
        merged[f"delta_{metric}"] = (
            merged[f"history_{metric}"] - merged[f"clean_history_{metric}"]
        ).abs()
    return merged


def add_normalized_deltas(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    for metric in HISTORY_METRICS:
        col = f"delta_{metric}"
        norm_col = f"norm_{metric}"
        values = out[col].astype(float)
        if mode == "none":
            denom = 1.0
            center = 0.0
        elif mode == "mean":
            denom = float(values.mean(skipna=True)) or 1.0
            center = 0.0
        elif mode == "zscore":
            denom = float(values.std(skipna=True, ddof=0)) or 1.0
            center = float(values.mean(skipna=True)) if values.notna().any() else 0.0
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        out[norm_col] = (values - center).abs() / denom

    for name, metrics in COMBINATIONS.items():
        cols = [f"norm_{metric}" for metric in metrics]
        out[f"score_{name}"] = out[cols].mean(axis=1)
    return out


def _linear_r2(x: pd.DataFrame, y: pd.Series) -> float:
    data = pd.concat([x, y.rename("target")], axis=1).dropna()
    if len(data) < 3:
        return np.nan
    y_arr = data["target"].to_numpy(dtype=float)
    if np.nanstd(y_arr) == 0:
        return np.nan
    x_arr = data.drop(columns=["target"]).to_numpy(dtype=float)
    x_arr = np.column_stack([np.ones(len(x_arr)), x_arr])
    coef, *_ = np.linalg.lstsq(x_arr, y_arr, rcond=None)
    pred = x_arr @ coef
    ss_res = float(np.sum((y_arr - pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else np.nan


def _spearman_details(df: pd.DataFrame, target: str, min_group_size: int) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["model", "dataset", "missing_ratio"], dropna=False)
    for (model, dataset, ratio), group in grouped:
        if len(group) < min_group_size:
            continue
        if group[target].nunique(dropna=True) < 2:
            continue
        for combo in COMBINATIONS:
            score_col = f"score_{combo}"
            if group[score_col].nunique(dropna=True) < 2:
                continue
            rho = group[score_col].corr(group[target], method="spearman")
            if pd.isna(rho):
                continue
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "missing_ratio": ratio,
                    "target": target,
                    "combination": combo,
                    "n_methods": len(group),
                    "spearman": float(rho),
                }
            )
    return pd.DataFrame(rows)


def summarize_details(details: pd.DataFrame, df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for combo, metrics in COMBINATIONS.items():
        subset = details[(details["target"] == target) & (details["combination"] == combo)]
        x_cols = [f"norm_{metric}" for metric in metrics]
        rows.append(
            {
                "target": target,
                "combination": combo,
                "metrics": ",".join(metrics),
                "group_count": int(len(subset)),
                "median_spearman": float(subset["spearman"].median()) if not subset.empty else np.nan,
                "mean_spearman": float(subset["spearman"].mean()) if not subset.empty else np.nan,
                "positive_ratio": float((subset["spearman"] > 0).mean()) if not subset.empty else np.nan,
                "overall_spearman": float(df[f"score_{combo}"].corr(df[target], method="spearman")),
                "overall_linear_r2": _linear_r2(df[x_cols], df[target]),
            }
        )
    return pd.DataFrame(rows)


def run_analysis(
    input_path: Path,
    output_dir: Path,
    normalization: str,
    min_group_size: int,
    targets: Sequence[str],
) -> None:
    raw = load_results_table(input_path)
    prepared = attach_clean_baseline(raw)
    prepared = add_normalized_deltas(prepared, normalization)

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(output_dir / "structure_metric_ablation_records.csv", index=False)

    all_details = []
    all_summary = []
    for target in targets:
        details = _spearman_details(prepared, target=target, min_group_size=min_group_size)
        summary = summarize_details(details, prepared, target=target)
        all_details.append(details)
        all_summary.append(summary)

    details_df = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()

    details_df.to_csv(output_dir / "structure_metric_ablation_group_spearman.csv", index=False)
    summary_df.to_csv(output_dir / "structure_metric_ablation_summary.csv", index=False)

    print(f"Records: {len(prepared)}")
    print(f"Group correlations: {len(details_df)}")
    print(f"Output: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run structure-channel ablation statistics")
    parser.add_argument(
        "--input",
        default="results_analysis/块状缺失对长序列预测影响-实验结果统计-0509.csv",
        help="Main experiment summary CSV",
    )
    parser.add_argument(
        "--output_dir",
        default="results_analysis/structure_metric_ablation",
        help="Output directory",
    )
    parser.add_argument(
        "--normalization",
        choices=["zscore", "mean", "none"],
        default="zscore",
        help="Normalization applied to absolute history-metric drift",
    )
    parser.add_argument("--min_group_size", type=int, default=3)
    parser.add_argument(
        "--targets",
        default="relative_mse_gain,relative_smape_gain",
        help="Comma-separated target columns",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    run_analysis(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        normalization=args.normalization,
        min_group_size=args.min_group_size,
        targets=targets,
    )


if __name__ == "__main__":
    main()
