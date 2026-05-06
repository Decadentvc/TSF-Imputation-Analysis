"""
偏差关联与模型差异分析脚本。

目标：
1) 计算历史窗口指标偏差 与 预测窗口指标偏差 的关联性。
2) 对比各模型在 clean 条件下的历史/预测偏差差异。

数据来源：
- 历史窗口特征：results_analysis/{model}/history/*.csv
- 预测窗口特征：results_analysis/{model}/prediction/*.csv
- 干净预测窗口特征(GT)：results_analysis/clean_prediction_windows/*_prediction_gt.csv
- 预测精度均值：results/{model}/clean/*.csv, results/{model}/impute/*.csv

说明：
- 历史偏差定义：imputed_history_metric - clean_history_metric
- 预测偏差定义：imputed_prediction_metric - clean_prediction_gt_metric
- clean 模型对比中：
  - history_bias_to_gt = clean_history_metric - clean_prediction_gt_metric
  - prediction_bias_to_gt = clean_model_prediction_metric - clean_prediction_gt_metric
  - pred_hist_gap = clean_model_prediction_metric - clean_history_metric
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd


METRIC_KEYS = (
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "spectral_entropy",
)

TERMS = {"short", "medium", "long"}


def split_multi_values(raw_values: Optional[List[str]]) -> Optional[List[str]]:
    if not raw_values:
        return None
    values: List[str] = []
    for chunk in raw_values:
        for part in chunk.split(","):
            item = part.strip()
            if item:
                values.append(item)
    return values if values else None


def parse_feature_stem(stem: str, suffix: str) -> Optional[Dict[str, str]]:
    """解析 history/prediction 文件名（去掉 .csv）。"""
    if not stem.endswith("_" + suffix):
        return None

    base = stem[: -(len(suffix) + 1)]
    parts = base.split("_")

    # clean: {dataset}_clean_{term}_{suffix}
    if len(parts) >= 3 and parts[-2] == "clean" and parts[-1] in TERMS:
        return {
            "dataset": "_".join(parts[:-2]),
            "method": "clean",
            "ratio": "",
            "term": parts[-1],
            "impute_method": "",
            "is_clean": "1",
        }

    # imputed: {dataset}_{method}_{ratio}_{term}_{impute_method}_{suffix}
    if len(parts) >= 5 and parts[-2] in TERMS:
        return {
            "dataset": "_".join(parts[:-4]),
            "method": parts[-4],
            "ratio": parts[-3],
            "term": parts[-2],
            "impute_method": parts[-1],
            "is_clean": "0",
        }

    return None


def parse_clean_gt_stem(stem: str) -> Optional[Tuple[str, str]]:
    # {dataset}_clean_{term}_prediction_gt
    if not stem.endswith("_prediction_gt"):
        return None
    base = stem[: -len("_prediction_gt")]
    parts = base.split("_")
    if len(parts) < 3 or parts[-2] != "clean" or parts[-1] not in TERMS:
        return None
    dataset = "_".join(parts[:-2])
    term = parts[-1]
    return dataset, term


def read_feature_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if "window_idx" not in df.columns:
        return None

    required = ["window_idx", *METRIC_KEYS]
    if any(c not in df.columns for c in required):
        return None

    return df[required].copy()


def load_feature_maps(
    model_dir: Path,
    suffix: str,
) -> Tuple[Dict[Tuple[str, str], Path], Dict[Tuple[str, str, str, str, str], Path]]:
    clean_map: Dict[Tuple[str, str], Path] = {}
    imputed_map: Dict[Tuple[str, str, str, str, str], Path] = {}

    if not model_dir.exists():
        return clean_map, imputed_map

    for path in sorted(model_dir.glob("*.csv")):
        if path.stem.endswith("_summary"):
            continue
        info = parse_feature_stem(path.stem, suffix)
        if not info:
            continue

        if info["is_clean"] == "1":
            clean_map[(info["dataset"], info["term"])] = path
        else:
            key = (
                info["dataset"],
                info["method"],
                info["ratio"],
                info["term"],
                info["impute_method"],
            )
            imputed_map[key] = path

    return clean_map, imputed_map


def load_clean_gt_map(clean_gt_dir: Path) -> Dict[Tuple[str, str], Path]:
    result: Dict[Tuple[str, str], Path] = {}
    if not clean_gt_dir.exists():
        return result

    for path in sorted(clean_gt_dir.glob("*.csv")):
        parsed = parse_clean_gt_stem(path.stem)
        if not parsed:
            continue
        result[parsed] = path

    return result


def read_result_metric_csv(path: Path) -> Dict[str, Any]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "metric" not in df.columns or "value" not in df.columns:
        return {}

    payload: Dict[str, Any] = {}
    for _, row in df.iterrows():
        key = str(row["metric"])
        val = row["value"]
        if isinstance(val, str):
            v = val.strip()
            if v == "":
                payload[key] = np.nan
                continue
            try:
                payload[key] = float(v)
            except ValueError:
                payload[key] = v
        else:
            payload[key] = val
    return payload


def load_accuracy_maps(
    results_model_dir: Path,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[Tuple[str, str, str, str, str], Dict[str, Any]]]:
    clean_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    imputed_map: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}

    clean_dir = results_model_dir / "clean"
    impute_dir = results_model_dir / "impute"

    if clean_dir.exists():
        for path in sorted(clean_dir.glob("*_results.csv")):
            stem = path.stem[: -len("_results")]
            parts = stem.split("_")
            if len(parts) < 3 or parts[-2] != "clean" or parts[-1] not in TERMS:
                continue
            dataset = "_".join(parts[:-2])
            term = parts[-1]
            clean_map[(dataset, term)] = read_result_metric_csv(path)

    if impute_dir.exists():
        for path in sorted(impute_dir.glob("*_results.csv")):
            stem = path.stem[: -len("_results")]
            parts = stem.split("_")
            # {impute_method}_{dataset}_{method}_length{n}_{ratio}_{term}
            if len(parts) < 6:
                continue
            term = parts[-1]
            ratio = parts[-2]
            length_token = parts[-3]
            method = parts[-4]
            impute_method = parts[0]
            dataset = "_".join(parts[1:-4])

            if term not in TERMS or not length_token.startswith("length"):
                continue

            key = (dataset, method, ratio, term, impute_method)
            imputed_map[key] = read_result_metric_csv(path)

    return clean_map, imputed_map


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    data = pd.concat([x, y], axis=1).dropna()
    if len(data) < 2:
        return np.nan
    xs = data.iloc[:, 0]
    ys = data.iloc[:, 1]
    if float(xs.std(ddof=0)) == 0.0 or float(ys.std(ddof=0)) == 0.0:
        return np.nan
    if method == "spearman":
        return float(xs.rank().corr(ys.rank(), method="pearson"))
    return float(xs.corr(ys, method="pearson"))


def summarize_bias_correlation(
    df: pd.DataFrame,
    group_cols: List[str],
    min_samples: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(rows)

    for keys, g in df.groupby(group_cols, dropna=False):
        n = len(g)
        if n < min_samples:
            continue
        row: Dict[str, Any] = {}
        if isinstance(keys, tuple):
            for i, c in enumerate(group_cols):
                row[c] = keys[i]
        else:
            row[group_cols[0]] = keys

        row["samples"] = n
        row["pearson_corr"] = safe_corr(g["history_bias"], g["prediction_bias"], "pearson")
        row["spearman_corr"] = safe_corr(g["history_bias"], g["prediction_bias"], "spearman")
        row["history_bias_mean"] = float(g["history_bias"].mean())
        row["prediction_bias_mean"] = float(g["prediction_bias"].mean())
        row["history_bias_abs_mean"] = float(g["history_bias"].abs().mean())
        row["prediction_bias_abs_mean"] = float(g["prediction_bias"].abs().mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and "pearson_corr" in out.columns:
        out = out.sort_values(by=["pearson_corr", "samples"], ascending=[False, False])
    return out


def summarize_clean_model_diff(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(rows)

    for (model, metric), g in df.groupby(["model", "metric"], dropna=False):
        rows.append(
            {
                "model": model,
                "metric": metric,
                "samples": len(g),
                "history_bias_to_gt_mean": float(g["history_bias_to_gt"].mean()),
                "history_bias_to_gt_abs_mean": float(g["history_bias_to_gt"].abs().mean()),
                "prediction_bias_to_gt_mean": float(g["prediction_bias_to_gt"].mean()),
                "prediction_bias_to_gt_abs_mean": float(g["prediction_bias_to_gt"].abs().mean()),
                "pred_hist_gap_mean": float(g["pred_hist_gap"].mean()),
                "pred_hist_gap_abs_mean": float(g["pred_hist_gap"].abs().mean()),
                "pred_vs_hist_corr": safe_corr(
                    g["prediction_bias_to_gt"], g["history_bias_to_gt"], "pearson"
                ),
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by=["metric", "prediction_bias_to_gt_abs_mean", "model"])
    return out


def analyze_model(
    model: str,
    results_analysis_dir: Path,
    clean_gt_map: Dict[Tuple[str, str], Path],
    accuracy_clean_map: Dict[Tuple[str, str], Dict[str, Any]],
    accuracy_imputed_map: Dict[Tuple[str, str, str, str, str], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    history_dir = results_analysis_dir / model / "history"
    prediction_dir = results_analysis_dir / model / "prediction"

    history_clean, history_imputed = load_feature_maps(history_dir, "history")
    pred_clean, pred_imputed = load_feature_maps(prediction_dir, "prediction")

    imputed_records: List[Dict[str, Any]] = []
    clean_records: List[Dict[str, Any]] = []

    # 1) 历史偏差 vs 预测偏差（imputed 条件）
    shared_imputed_keys = sorted(set(history_imputed.keys()) & set(pred_imputed.keys()))
    for key in shared_imputed_keys:
        dataset, method, ratio, term, impute_method = key
        if (dataset, term) not in history_clean or (dataset, term) not in clean_gt_map:
            continue

        hist_i = read_feature_csv(history_imputed[key])
        hist_c = read_feature_csv(history_clean[(dataset, term)])
        pred_i = read_feature_csv(pred_imputed[key])
        pred_gt = read_feature_csv(clean_gt_map[(dataset, term)])
        if hist_i is None or hist_c is None or pred_i is None or pred_gt is None:
            continue

        merged = (
            hist_i.merge(hist_c, on="window_idx", suffixes=("_hist_imp", "_hist_clean"))
            .merge(pred_i, on="window_idx")
            .merge(pred_gt, on="window_idx", suffixes=("_pred_imp", "_pred_gt"))
        )
        if merged.empty:
            continue

        acc = accuracy_imputed_map.get(key, {})
        for _, row in merged.iterrows():
            for metric in METRIC_KEYS:
                rec: Dict[str, Any] = {
                    "model": model,
                    "dataset": dataset,
                    "method": method,
                    "ratio": ratio,
                    "term": term,
                    "impute_method": impute_method,
                    "window_idx": int(row["window_idx"]),
                    "metric": metric,
                    "history_bias": float(row[f"{metric}_hist_imp"] - row[f"{metric}_hist_clean"]),
                    "prediction_bias": float(row[f"{metric}_pred_imp"] - row[f"{metric}_pred_gt"]),
                }
                if "mean_weighted_sum_quantile_loss" in acc:
                    rec["mean_weighted_sum_quantile_loss"] = acc[
                        "mean_weighted_sum_quantile_loss"
                    ]
                if "RMSE[mean]" in acc:
                    rec["RMSE_mean"] = acc["RMSE[mean]"]
                imputed_records.append(rec)

    # 2) clean 条件下的模型差异
    shared_clean_keys = sorted(set(history_clean.keys()) & set(pred_clean.keys()) & set(clean_gt_map.keys()))
    for dataset, term in shared_clean_keys:
        hist_c = read_feature_csv(history_clean[(dataset, term)])
        pred_c = read_feature_csv(pred_clean[(dataset, term)])
        pred_gt = read_feature_csv(clean_gt_map[(dataset, term)])
        if hist_c is None or pred_c is None or pred_gt is None:
            continue

        merged = (
            hist_c.merge(pred_c, on="window_idx", suffixes=("_hist_clean", "_pred_clean"))
            .merge(pred_gt, on="window_idx", suffixes=("", "_pred_gt"))
        )
        if merged.empty:
            continue

        acc = accuracy_clean_map.get((dataset, term), {})
        for _, row in merged.iterrows():
            for metric in METRIC_KEYS:
                hist_val = float(row[f"{metric}_hist_clean"])
                pred_val = float(row[f"{metric}_pred_clean"])
                gt_val = float(row[f"{metric}"])
                rec = {
                    "model": model,
                    "dataset": dataset,
                    "term": term,
                    "window_idx": int(row["window_idx"]),
                    "metric": metric,
                    "history_bias_to_gt": hist_val - gt_val,
                    "prediction_bias_to_gt": pred_val - gt_val,
                    "pred_hist_gap": pred_val - hist_val,
                }
                if "mean_weighted_sum_quantile_loss" in acc:
                    rec["mean_weighted_sum_quantile_loss"] = acc[
                        "mean_weighted_sum_quantile_loss"
                    ]
                if "RMSE[mean]" in acc:
                    rec["RMSE_mean"] = acc["RMSE[mean]"]
                clean_records.append(rec)

    return imputed_records, clean_records


def main() -> None:
    parser = argparse.ArgumentParser(description="偏差关联与模型差异分析")
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="模型名，支持空格或逗号分隔；默认自动扫描 results_analysis 下模型目录",
    )
    parser.add_argument(
        "--results_analysis_dir",
        type=str,
        default="results_analysis",
        help="窗口特征结果根目录",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="预测精度结果根目录",
    )
    parser.add_argument(
        "--clean_gt_dir",
        type=str,
        default="results_analysis/clean_prediction_windows",
        help="干净预测窗口特征目录（GT）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results_analysis/bias_relation",
        help="输出目录",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="相关性统计最小样本数",
    )
    args = parser.parse_args()

    results_analysis_dir = Path(args.results_analysis_dir)
    results_dir = Path(args.results_dir)
    clean_gt_dir = Path(args.clean_gt_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_models = split_multi_values(args.models)
    if raw_models:
        models = sorted({m.strip().lower() for m in raw_models if m.strip()})
    else:
        models = sorted(
            [
                p.name.lower()
                for p in results_analysis_dir.iterdir()
                if p.is_dir() and (p / "history").exists() and (p / "prediction").exists()
            ]
        )

    clean_gt_map = load_clean_gt_map(clean_gt_dir)
    if not models:
        print("[错误] 未找到可分析的模型目录")
        return
    if not clean_gt_map:
        print(f"[错误] 未找到 clean prediction GT 文件: {clean_gt_dir}")
        return

    all_imputed_records: List[Dict[str, Any]] = []
    all_clean_records: List[Dict[str, Any]] = []

    for model in models:
        acc_clean_map, acc_imputed_map = load_accuracy_maps(results_dir / model)
        imputed_records, clean_records = analyze_model(
            model=model,
            results_analysis_dir=results_analysis_dir,
            clean_gt_map=clean_gt_map,
            accuracy_clean_map=acc_clean_map,
            accuracy_imputed_map=acc_imputed_map,
        )
        all_imputed_records.extend(imputed_records)
        all_clean_records.extend(clean_records)

    imputed_df = pd.DataFrame(all_imputed_records)
    clean_df = pd.DataFrame(all_clean_records)

    # 输出明细
    imputed_path = output_dir / "imputed_bias_records.csv"
    clean_path = output_dir / "clean_model_bias_records.csv"
    if not imputed_df.empty:
        imputed_df.to_csv(imputed_path, index=False)
    if not clean_df.empty:
        clean_df.to_csv(clean_path, index=False)

    # 任务1：历史偏差 vs 预测偏差关联
    corr_by_model_metric = summarize_bias_correlation(
        imputed_df,
        group_cols=["model", "metric"],
        min_samples=args.min_samples,
    )
    corr_by_model_dataset_term_metric = summarize_bias_correlation(
        imputed_df,
        group_cols=["model", "dataset", "term", "metric"],
        min_samples=args.min_samples,
    )

    corr_mm_path = output_dir / "history_prediction_bias_correlation_by_model_metric.csv"
    corr_mdtm_path = (
        output_dir / "history_prediction_bias_correlation_by_model_dataset_term_metric.csv"
    )
    if not corr_by_model_metric.empty:
        corr_by_model_metric.to_csv(corr_mm_path, index=False)
    if not corr_by_model_dataset_term_metric.empty:
        corr_by_model_dataset_term_metric.to_csv(corr_mdtm_path, index=False)

    # 任务2：clean 条件模型差异
    clean_compare = summarize_clean_model_diff(clean_df)
    clean_compare_path = output_dir / "clean_model_bias_compare.csv"
    if not clean_compare.empty:
        clean_compare.to_csv(clean_compare_path, index=False)

    print("=" * 88)
    print("偏差关联分析完成")
    print("=" * 88)
    print(f"模型数量: {len(models)}")
    print(f"imputed 偏差样本数: {len(imputed_df)}")
    print(f"clean 偏差样本数: {len(clean_df)}")
    print(f"输出目录: {output_dir}")
    if imputed_path.exists():
        print(f"- 明细(任务1): {imputed_path}")
    if corr_mm_path.exists():
        print(f"- 关联统计(模型-指标): {corr_mm_path}")
    if corr_mdtm_path.exists():
        print(f"- 关联统计(模型-数据集-term-指标): {corr_mdtm_path}")
    if clean_path.exists():
        print(f"- 明细(任务2): {clean_path}")
    if clean_compare_path.exists():
        print(f"- 模型差异汇总: {clean_compare_path}")


if __name__ == "__main__":
    main()
