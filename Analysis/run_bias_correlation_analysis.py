from __future__ import annotations

"""
偏差关联与模型差异分析脚本

目标：
1) 计算历史窗口指标偏差 与 预测窗口指标偏差 的关联性（Pearson/Spearman）
2) 比较各模型在“干净历史 -> 预测窗口”场景下的指标偏差差异

输入数据：
- results_analysis/{model}/history/*.csv
- results_analysis/{model}/prediction/*.csv
- results_analysis/clean_prediction_windows/*_prediction_gt.csv
- （可选）results/{model}/impute/*_results.csv，用于拼接平均预测准确度指标

输出：
- bias_pairs.csv
- history_prediction_correlation.csv
- clean_model_bias.csv
- clean_model_bias_summary.csv

说明：
- prediction bias 默认采用 `imputed_prediction - clean_prediction`（同模型同数据同窗口）
- 可通过参数切换为 `imputed_prediction - ground_truth`
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ALL_METRIC_KEYS: Tuple[str, ...] = (
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "spectral_entropy",
)

# 暂时不比较的指标：保留接口，默认不参与统计
PENDING_METRIC_KEYS: Tuple[str, ...] = (
    "trend_linearity",
    "spectral_entropy",
)

ACTIVE_METRIC_KEYS: Tuple[str, ...] = tuple(
    m for m in ALL_METRIC_KEYS if m not in PENDING_METRIC_KEYS
)

META_COLS = {
    "window_idx",
    "history_start",
    "history_end",
    "history_length",
    "forecast_start",
    "forecast_end",
    "prediction_length",
    "n_series",
}

ACC_KEYS = (
    "MSE[mean]",
    "MAE[0.5]",
    "RMSE[mean]",
    "mean_weighted_sum_quantile_loss",
)


def split_csv(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return vals or None


def in_filters(value: str, filters: Optional[List[str]]) -> bool:
    return True if filters is None else value in filters


def model_dirs(results_analysis_dir: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(results_analysis_dir.iterdir()):
        if not p.is_dir() or p.name == "clean_prediction_windows":
            continue
        if (p / "history").exists() and (p / "prediction").exists():
            out.append(p)
    return out


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def metric_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c in ALL_METRIC_KEYS]
    if cols:
        return cols
    dynamic = [c for c in df.columns if c not in META_COLS]
    return dynamic


def metric_mean_row(df: pd.DataFrame) -> Dict[str, float]:
    cols = metric_cols(df)
    out: Dict[str, float] = {}
    for c in cols:
        series = pd.to_numeric(df[c], errors="coerce").dropna()
        if not series.empty:
            out[c] = float(series.mean())
    return out


def extract_numeric_metrics_kv(results_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    df = read_csv(results_csv)
    if df is None or "metric" not in df.columns or "value" not in df.columns:
        return out

    for _, row in df.iterrows():
        k = str(row["metric"])
        v = row["value"]
        try:
            f = float(v)
            out[k] = f
        except Exception:
            continue
    return out


def load_accuracy_map(results_dir: Path) -> Dict[Tuple[str, str, str, str, str], Dict[str, float]]:
    """
    key = (model, dataset, term, ratio, method)
    value = {acc_metric: value}
    """
    acc_map: Dict[Tuple[str, str, str, str, str], Dict[str, float]] = {}
    pat = re.compile(
        r"^([A-Za-z0-9]+)_(.+)_BM_length\d+_(\d{3})_(short|medium|long)_results\.csv$"
    )

    if not results_dir.exists():
        return acc_map

    for fp in results_dir.glob("*/impute/*_results.csv"):
        model = fp.parent.parent.name
        m = pat.match(fp.name)
        if not m:
            continue
        method, dataset, ratio, term = m.group(1), m.group(2), m.group(3), m.group(4)
        kv = extract_numeric_metrics_kv(fp)
        if not kv:
            continue
        acc_map[(model, dataset, term, ratio, method)] = kv
    return acc_map


def build_bias_pairs(
    results_analysis_dir: Path,
    results_dir: Path,
    model_filters: Optional[List[str]],
    dataset_filters: Optional[List[str]],
    term_filters: Optional[List[str]],
    ratio_filters: Optional[List[str]],
    method_filters: Optional[List[str]],
    prediction_bias_base: str,
    metric_keys: Tuple[str, ...],
) -> pd.DataFrame:
    history_pat = re.compile(
        r"^(.+)_BM_(\d{3})_(short|medium|long)_([A-Za-z0-9]+)_history\.csv$"
    )

    gt_dir = results_analysis_dir / "clean_prediction_windows"
    acc_map = load_accuracy_map(results_dir)

    rows: List[Dict[str, object]] = []

    for mdir in model_dirs(results_analysis_dir):
        model = mdir.name
        if not in_filters(model, model_filters):
            continue

        hist_dir = mdir / "history"
        pred_dir = mdir / "prediction"

        for hfp in sorted(hist_dir.glob("*_BM_*_history.csv")):
            m = history_pat.match(hfp.name)
            if not m:
                continue

            dataset, ratio, term, method = m.group(1), m.group(2), m.group(3), m.group(4)
            if not in_filters(dataset, dataset_filters):
                continue
            if not in_filters(term, term_filters):
                continue
            if not in_filters(ratio, ratio_filters):
                continue
            if not in_filters(method, method_filters):
                continue

            pfp = pred_dir / f"{dataset}_BM_{ratio}_{term}_{method}_prediction.csv"
            cpfp = pred_dir / f"{dataset}_clean_{term}_prediction.csv"
            chfp = hist_dir / f"{dataset}_clean_{term}_history.csv"
            gtfp = gt_dir / f"{dataset}_clean_{term}_prediction_gt.csv"

            hdf = read_csv(hfp)
            pdf = read_csv(pfp)
            cpdf = read_csv(cpfp)
            chdf = read_csv(chfp)
            gtdf = read_csv(gtfp)
            if hdf is None or pdf is None or chdf is None or gtdf is None:
                continue
            if "window_idx" not in pdf.columns or "window_idx" not in gtdf.columns:
                continue

            merged = pdf.merge(gtdf, on="window_idx", suffixes=("_pred", "_gt"))
            if merged.empty:
                continue

            acc = acc_map.get((model, dataset, term, ratio, method), {})

            for metric in metric_keys:
                pred_col = f"{metric}_pred"
                gt_col = f"{metric}_gt"
                if pred_col not in merged.columns or gt_col not in merged.columns:
                    continue

                if (
                    "window_idx" not in hdf.columns
                    or "window_idx" not in chdf.columns
                    or metric not in hdf.columns
                    or metric not in chdf.columns
                ):
                    continue

                hist_pair = hdf[["window_idx", metric]].merge(
                    chdf[["window_idx", metric]],
                    on="window_idx",
                    suffixes=("_hist", "_clean_hist"),
                )
                if hist_pair.empty:
                    continue

                combo = merged[["window_idx", pred_col, gt_col]].merge(
                    hist_pair,
                    on="window_idx",
                    how="inner",
                )
                if combo.empty:
                    continue

                combo["history_bias"] = pd.to_numeric(
                    combo[f"{metric}_hist"], errors="coerce"
                ) - pd.to_numeric(combo[f"{metric}_clean_hist"], errors="coerce")
                combo["prediction_bias"] = pd.to_numeric(
                    combo[pred_col], errors="coerce"
                ) - pd.to_numeric(combo[gt_col], errors="coerce")

                if prediction_bias_base == "clean_pred":
                    if cpdf is None or "window_idx" not in cpdf.columns or metric not in cpdf.columns:
                        continue
                    clean_pred_pair = cpdf[["window_idx", metric]].rename(
                        columns={metric: f"{metric}_clean_pred"}
                    )
                    combo = combo.merge(clean_pred_pair, on="window_idx", how="inner")
                    if combo.empty:
                        continue
                    combo["prediction_bias"] = pd.to_numeric(
                        combo[pred_col], errors="coerce"
                    ) - pd.to_numeric(combo[f"{metric}_clean_pred"], errors="coerce")

                combo = combo[["window_idx", "history_bias", "prediction_bias"]].dropna()
                if combo.empty:
                    continue

                matched_windows = int(combo.shape[0])

                for _, crow in combo.iterrows():
                    row = {
                        "model": model,
                        "dataset": dataset,
                        "term": term,
                        "ratio": ratio,
                        "method": method,
                        "metric": metric,
                        "window_idx": int(crow["window_idx"]),
                        "n_windows": matched_windows,
                        "history_bias": float(crow["history_bias"]),
                        "prediction_bias_mean": float(crow["prediction_bias"]),
                        "prediction_bias_abs_mean": float(abs(crow["prediction_bias"])),
                        "prediction_bias_std": 0.0,
                        "prediction_bias_base": prediction_bias_base,
                    }

                    for k in ACC_KEYS:
                        if k in acc:
                            row[f"acc_{k}"] = float(acc[k])

                    rows.append(row)

    return pd.DataFrame(rows)


def correlation_table(bias_pairs: pd.DataFrame, metric_keys: Tuple[str, ...]) -> pd.DataFrame:
    out: List[Dict[str, object]] = []

    def corr_strength_label(value: float, n: int) -> str:
        if n < 3 or pd.isna(value):
            return "insufficient"
        a = abs(float(value))
        if a >= 0.7:
            return "strong"
        if a >= 0.4:
            return "moderate"
        if a >= 0.2:
            return "weak"
        return "none"

    def spearman_fallback(x: pd.Series, y: pd.Series) -> float:
        if x.shape[0] < 2 or y.shape[0] < 2:
            return np.nan
        rx = x.rank(method="average")
        ry = y.rank(method="average")
        if rx.nunique(dropna=True) < 2 or ry.nunique(dropna=True) < 2:
            return np.nan
        return float(rx.corr(ry, method="pearson"))

    def calc(sub: pd.DataFrame, group_name: str, group_value: str, metric: str) -> None:
        x = pd.to_numeric(sub["history_bias"], errors="coerce")
        y = pd.to_numeric(sub["prediction_bias_mean"], errors="coerce")
        valid = pd.DataFrame({"x": x, "y": y}).dropna()
        n = valid.shape[0]
        pearson = np.nan
        spearman = np.nan
        if n >= 2:
            pearson = float(valid["x"].corr(valid["y"], method="pearson"))
            spearman = spearman_fallback(valid["x"], valid["y"])
        pearson_strength = corr_strength_label(pearson, int(n))
        spearman_strength = corr_strength_label(spearman, int(n))
        out.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "metric": metric,
                "n": int(n),
                "pearson": pearson,
                "spearman": spearman,
                "pearson_abs": float(abs(pearson)) if not pd.isna(pearson) else np.nan,
                "spearman_abs": float(abs(spearman)) if not pd.isna(spearman) else np.nan,
                "pearson_strength": pearson_strength,
                "spearman_strength": spearman_strength,
            }
        )

    ordered_metrics = [m for m in metric_keys if m in set(bias_pairs["metric"].unique())]
    for metric in ordered_metrics:
        metric_df = bias_pairs[bias_pairs["metric"] == metric]
        for model, gdf in metric_df.groupby("model"):
            calc(gdf, "model", str(model), metric)

    return pd.DataFrame(out)


def build_clean_model_bias(
    results_analysis_dir: Path,
    model_filters: Optional[List[str]],
    dataset_filters: Optional[List[str]],
    term_filters: Optional[List[str]],
    metric_keys: Tuple[str, ...],
) -> pd.DataFrame:
    clean_pred_pat = re.compile(r"^(.+)_clean_(short|medium|long)_prediction\.csv$")
    gt_dir = results_analysis_dir / "clean_prediction_windows"

    rows: List[Dict[str, object]] = []

    for mdir in model_dirs(results_analysis_dir):
        model = mdir.name
        if not in_filters(model, model_filters):
            continue

        pred_dir = mdir / "prediction"

        for pfp in sorted(pred_dir.glob("*_clean_*_prediction.csv")):
            m = clean_pred_pat.match(pfp.name)
            if not m:
                continue
            dataset, term = m.group(1), m.group(2)

            if not in_filters(dataset, dataset_filters):
                continue
            if not in_filters(term, term_filters):
                continue

            gtfp = gt_dir / f"{dataset}_clean_{term}_prediction_gt.csv"
            pdf = read_csv(pfp)
            gtdf = read_csv(gtfp)
            if pdf is None or gtdf is None:
                continue
            if "window_idx" not in pdf.columns or "window_idx" not in gtdf.columns:
                continue

            merged = pdf.merge(gtdf, on="window_idx", suffixes=("_pred", "_gt"))
            if merged.empty:
                continue

            for metric in metric_keys:
                pred_col = f"{metric}_pred"
                gt_col = f"{metric}_gt"
                if pred_col not in merged.columns or gt_col not in merged.columns:
                    continue

                sub = merged[["window_idx", pred_col, gt_col]].copy()
                sub["prediction_bias"] = pd.to_numeric(sub[pred_col], errors="coerce") - pd.to_numeric(
                    sub[gt_col], errors="coerce"
                )
                sub = sub[["window_idx", "prediction_bias"]].dropna()
                if sub.empty:
                    continue

                for _, r in sub.iterrows():
                    rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "term": term,
                            "metric": metric,
                            "window_idx": int(r["window_idx"]),
                            "n_windows": 1,
                            "prediction_bias_mean": float(r["prediction_bias"]),
                            "prediction_bias_abs_mean": float(abs(r["prediction_bias"])),
                            "prediction_bias_std": 0.0,
                        }
                    )

    return pd.DataFrame(rows)


def summarize_clean_model_bias(clean_bias: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        clean_bias.groupby(["model", "metric"], as_index=False)
        .agg(
            n_samples=("prediction_bias_mean", "count"),
            bias_mean=("prediction_bias_mean", "mean"),
            bias_std=("prediction_bias_mean", "std"),
            abs_bias_mean=("prediction_bias_abs_mean", "mean"),
            abs_bias_std=("prediction_bias_abs_mean", "std"),
        )
        .sort_values(["metric", "abs_bias_mean", "model"], ascending=[True, True, True])
    )
    grouped["rank_by_abs_bias"] = grouped.groupby("metric")["abs_bias_mean"].rank(
        method="min", ascending=True
    )
    return grouped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze history/prediction bias association")
    p.add_argument("--results-analysis-dir", default="results_analysis")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--output-dir", default="results_analysis/bias_analysis")
    p.add_argument("--models", default=None, help="comma-separated model filters")
    p.add_argument("--datasets", default=None, help="comma-separated dataset filters")
    p.add_argument("--terms", default=None, help="comma-separated term filters")
    p.add_argument("--ratios", default=None, help="comma-separated ratio filters, e.g. 010,020")
    p.add_argument("--methods", default=None, help="comma-separated imputation method filters")
    p.add_argument(
        "--prediction-bias-base",
        default="clean_pred",
        choices=["clean_pred", "gt"],
        help=(
            "Prediction bias baseline: clean_pred => imputed-clean_prediction; "
            "gt => imputed-ground_truth"
        ),
    )
    p.add_argument(
        "--include-pending-metrics",
        action="store_true",
        help=(
            "Also include pending metrics (trend_linearity, spectral_entropy). "
            "Default only compares active metrics."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    results_analysis_dir = Path(args.results_analysis_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_filters = split_csv(args.models)
    dataset_filters = split_csv(args.datasets)
    term_filters = split_csv(args.terms)
    ratio_filters = split_csv(args.ratios)
    method_filters = split_csv(args.methods)
    metric_keys = ALL_METRIC_KEYS if args.include_pending_metrics else ACTIVE_METRIC_KEYS

    print(f"[INFO] active metrics: {list(metric_keys)}")
    if not args.include_pending_metrics:
        print(f"[INFO] pending metrics skipped: {list(PENDING_METRIC_KEYS)}")

    bias_pairs = build_bias_pairs(
        results_analysis_dir=results_analysis_dir,
        results_dir=results_dir,
        model_filters=model_filters,
        dataset_filters=dataset_filters,
        term_filters=term_filters,
        ratio_filters=ratio_filters,
        method_filters=method_filters,
        prediction_bias_base=args.prediction_bias_base,
        metric_keys=metric_keys,
    )

    if bias_pairs.empty:
        print("[WARN] No bias pairs found. Check filters and input files.")
    else:
        pairs_path = output_dir / "bias_pairs.csv"
        bias_pairs.to_csv(pairs_path, index=False)
        print(f"[OK] bias pairs -> {pairs_path}")

        corr = correlation_table(bias_pairs, metric_keys=metric_keys)
        corr_path = output_dir / "history_prediction_correlation.csv"
        corr.to_csv(corr_path, index=False)
        print(f"[OK] correlation -> {corr_path}")

    clean_bias = build_clean_model_bias(
        results_analysis_dir=results_analysis_dir,
        model_filters=model_filters,
        dataset_filters=dataset_filters,
        term_filters=term_filters,
        metric_keys=metric_keys,
    )

    if clean_bias.empty:
        print("[WARN] No clean model bias found. Check filters and input files.")
    else:
        clean_path = output_dir / "clean_model_bias.csv"
        clean_bias.to_csv(clean_path, index=False)
        print(f"[OK] clean model bias -> {clean_path}")

        summary = summarize_clean_model_bias(clean_bias)
        summary_path = output_dir / "clean_model_bias_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[OK] clean model bias summary -> {summary_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
