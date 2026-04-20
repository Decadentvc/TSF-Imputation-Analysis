"""
干净预测窗口批量特征分析脚本

干净预测窗口：在干净数据集（data/datasets/ori/{dataset}.csv）尾部，
按 inject_range_utils 的切分规则（prediction_length × windows）划出的
真实值窗口，对应模型输出的预测窗口，可直接用作 Ground-Truth 做比较。

对每个 (dataset, term) 组合：
1. 通过 get_injection_range 取得 prediction_length 与 windows 数量
2. 从干净数据尾部按 window_idx 逆序切出每个预测窗口
3. 对每个窗口的所有变量做 STL 分解并计算 6 个特征指标
4. 输出按窗口聚合的 CSV 与 summary JSON

示例：
python Analysis/run_clean_prediction_window_analysis.py
python Analysis/run_clean_prediction_window_analysis.py --dataset ETTh1 --terms short,medium,long
python Analysis/run_clean_prediction_window_analysis.py --terms short --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "tools" / "Missing_Value_Injection"))

from Analysis.metrics import calculate_all_metrics, get_period  # noqa: E402
from inject_range_utils import get_injection_range  # noqa: E402
from statsmodels.tsa.seasonal import STL  # noqa: E402


DEFAULT_DATA_PATH = "data/datasets"
DEFAULT_PROPERTIES_PATH = "data/datasets/dataset_properties.json"
DEFAULT_OUTPUT_DIR = "results_analysis/clean_prediction_windows"
# max_context 仅影响 injection 起点，不影响预测窗口切分，固定任意值即可
DEFAULT_PLACEHOLDER_MAX_CONTEXT = 2880
TIME_COLS = {"date", "time", "timestamp", "datetime", "index"}
METRIC_KEYS = (
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "spectral_entropy",
)


def load_dataset_properties(properties_path: str) -> Dict[str, Any]:
    props_path = Path(properties_path)
    if not props_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {props_path}")
    with open(props_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_allowed_terms(dataset_name: str, properties: Dict[str, Any]) -> List[str]:
    if dataset_name not in properties:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    term_type = properties[dataset_name].get("term", "med_long")
    return ["short"] if term_type == "short" else ["short", "medium", "long"]


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


def dedupe_lower(values: List[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for v in values:
        key = v.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def compute_window_metrics(
    window_data: pd.DataFrame,
    period: int,
) -> Optional[Dict[str, float]]:
    """对单个预测窗口的所有变量做 STL 分解并计算平均指标。"""
    value_cols = [c for c in window_data.columns if c.lower() not in TIME_COLS]
    if not value_cols:
        return None

    collected: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}

    for col in value_cols:
        series = window_data[col].dropna().values
        if len(series) < 2 * period:
            continue
        try:
            stl = STL(series, period=period, robust=True)
            result = stl.fit()
            metrics = calculate_all_metrics(
                series, result.trend, result.seasonal, result.resid, period
            )
            for key, value in metrics.items():
                collected[key].append(value)
        except Exception:
            continue

    if not collected["trend_strength"]:
        return None

    return {key: float(np.mean(values)) for key, values in collected.items()}


def analyze_clean_prediction_windows(
    dataset_name: str,
    term: str,
    data_path: str = DEFAULT_DATA_PATH,
    properties_path: str = DEFAULT_PROPERTIES_PATH,
) -> Dict[str, Any]:
    """分析单个 (dataset, term) 下所有干净预测窗口的特征。"""
    clean_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    if not clean_path.exists():
        return {"success": False, "error": f"Clean data not found: {clean_path}"}

    inject_range = get_injection_range(
        dataset_name=dataset_name,
        term=term,
        data_path=data_path,
        max_context=DEFAULT_PLACEHOLDER_MAX_CONTEXT,
    )
    prediction_length = inject_range["prediction_length"]
    n_windows = inject_range["windows"]
    total_length = inject_range["total_length"]

    df = pd.read_csv(clean_path)
    period = get_period(dataset_name, properties_path)

    window_results: List[Dict[str, Any]] = []

    for window_idx in range(n_windows):
        forecast_end = total_length - prediction_length * window_idx
        forecast_start = forecast_end - prediction_length
        if forecast_start < 0:
            continue

        window_df = df.iloc[forecast_start:forecast_end]
        metrics = compute_window_metrics(window_df, period)
        if metrics is None:
            continue

        window_results.append(
            {
                "window_idx": window_idx,
                "forecast_start": forecast_start,
                "forecast_end": forecast_end,
                "prediction_length": forecast_end - forecast_start,
                "metrics": metrics,
                "n_series": len(
                    [c for c in window_df.columns if c.lower() not in TIME_COLS]
                ),
            }
        )

    if not window_results:
        return {
            "success": False,
            "error": (
                f"No valid windows for {dataset_name}/{term} "
                f"(need >= {2 * period} points per series, prediction_length={prediction_length})"
            ),
            "inject_range": inject_range,
            "required_length": 2 * period,
        }

    aggregated: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
    for wr in window_results:
        for key, value in wr["metrics"].items():
            aggregated[key].append(value)

    summary = {
        "mean": {key: float(np.mean(vals)) for key, vals in aggregated.items()},
        "std": {key: float(np.std(vals)) for key, vals in aggregated.items()},
        "min": {key: float(np.min(vals)) for key, vals in aggregated.items()},
        "max": {key: float(np.max(vals)) for key, vals in aggregated.items()},
    }

    return {
        "success": True,
        "dataset": dataset_name,
        "term": term,
        "period": period,
        "prediction_length": prediction_length,
        "inject_range": inject_range,
        "n_windows": len(window_results),
        "summary": summary,
        "window_results": window_results,
    }


def save_clean_prediction_result(
    result: Dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = result["dataset"]
    term = result["term"]
    base = f"{dataset}_clean_{term}_prediction_gt"
    csv_path = output_dir / f"{base}.csv"
    json_path = output_dir / f"{base}_summary.json"

    rows = []
    for wr in result["window_results"]:
        row = {
            "window_idx": wr["window_idx"],
            "forecast_start": wr["forecast_start"],
            "forecast_end": wr["forecast_end"],
            "prediction_length": wr["prediction_length"],
            "n_series": wr["n_series"],
        }
        row.update(wr["metrics"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    summary_payload = {
        "dataset": dataset,
        "term": term,
        "period": result["period"],
        "prediction_length": result["prediction_length"],
        "n_windows": result["n_windows"],
        "inject_range": result["inject_range"],
        "summary": result["summary"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    return csv_path


def generate_overall_summary(
    results_list: List[Dict[str, Any]],
    output_dir: Path,
) -> Optional[Path]:
    if not results_list:
        return None

    aggregated: Dict[str, List[float]] = {key: [] for key in METRIC_KEYS}
    per_term: Dict[str, Dict[str, List[float]]] = {}

    for result in results_list:
        term = result["term"]
        per_term.setdefault(term, {key: [] for key in METRIC_KEYS})
        for key in METRIC_KEYS:
            mean_val = result["summary"]["mean"].get(key)
            if mean_val is None:
                continue
            aggregated[key].append(mean_val)
            per_term[term][key].append(mean_val)

    if not any(aggregated.values()):
        return None

    def _stats(values: Dict[str, List[float]]) -> Dict[str, Dict[str, Optional[float]]]:
        return {
            "mean": {k: (float(np.mean(v)) if v else None) for k, v in values.items()},
            "std": {k: (float(np.std(v)) if v else None) for k, v in values.items()},
            "min": {k: (float(np.min(v)) if v else None) for k, v in values.items()},
            "max": {k: (float(np.max(v)) if v else None) for k, v in values.items()},
            "count": {k: len(v) for k, v in values.items()},
        }

    payload = {
        "window_type": "clean_prediction",
        "total_combinations": len(results_list),
        "overall_summary": _stats(aggregated),
        "per_term_summary": {term: _stats(vals) for term, vals in per_term.items()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "overall_clean_prediction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return summary_path


def discover_datasets(data_path: str) -> List[str]:
    ori_dir = Path(data_path) / "ori"
    if not ori_dir.exists():
        return []
    return sorted(p.stem for p in ori_dir.glob("*.csv"))


def build_tasks(
    datasets: List[str],
    requested_terms: Optional[List[str]],
    properties: Dict[str, Any],
) -> List[Tuple[str, str]]:
    tasks: List[Tuple[str, str]] = []
    for dataset in datasets:
        try:
            allowed = get_allowed_terms(dataset, properties)
        except ValueError:
            continue
        terms = (
            [t for t in requested_terms if t in allowed]
            if requested_terms
            else allowed
        )
        for term in terms:
            tasks.append((dataset, term))
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch analyze clean prediction windows (ground-truth slices) with 6 STL metrics",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=None,
        help="One or more dataset names. Omit to analyze all datasets under data/ori/.",
    )
    parser.add_argument(
        "--terms",
        nargs="+",
        default=None,
        help="Terms to analyze, e.g. short,medium,long. Omit to use each dataset's allowed terms.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help=f"Dataset root directory (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--properties_path",
        type=str,
        default=DEFAULT_PROPERTIES_PATH,
        help=f"Dataset properties JSON path (default: {DEFAULT_PROPERTIES_PATH})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when the result file already exists",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    properties = load_dataset_properties(args.properties_path)

    user_datasets = split_multi_values(args.dataset)
    if user_datasets:
        datasets = user_datasets
    else:
        datasets = discover_datasets(args.data_path)

    requested_terms_raw = split_multi_values(args.terms)
    requested_terms = dedupe_lower(requested_terms_raw) if requested_terms_raw else None

    tasks = build_tasks(datasets, requested_terms, properties)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("Clean Prediction Window Analysis")
    print("=" * 80)
    print(f"Datasets:    {datasets}")
    print(f"Terms:       {requested_terms if requested_terms else 'all allowed per dataset'}")
    print(f"Data path:   {args.data_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Tasks:       {len(tasks)}")
    print("=" * 80)

    total = len(tasks)
    succeeded = 0
    skipped = 0
    insufficient = 0
    failed = 0
    all_results: List[Dict[str, Any]] = []

    for dataset, term in tqdm(tasks, desc="Clean prediction windows"):
        csv_path = output_dir / f"{dataset}_clean_{term}_prediction_gt.csv"
        if csv_path.exists() and not args.force:
            skipped += 1
            continue

        try:
            result = analyze_clean_prediction_windows(
                dataset_name=dataset,
                term=term,
                data_path=args.data_path,
                properties_path=args.properties_path,
            )
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] Error: {dataset} {term} - {exc}")
            continue

        if result["success"]:
            saved = save_clean_prediction_result(result, output_dir)
            succeeded += 1
            all_results.append(result)
            print(f"  [OK] Saved: {saved}")
        else:
            err = result.get("error", "Unknown error")
            if "No valid windows" in err or "series too short" in err.lower():
                insufficient += 1
                print(f"  [SKIP-DATA] Insufficient: {dataset} {term} - {err}")
            else:
                failed += 1
                print(f"  [FAIL] {dataset} {term} - {err}")

    if all_results:
        summary_path = generate_overall_summary(all_results, output_dir)
        if summary_path:
            print(f"\n  [OK] Overall summary: {summary_path}")

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Total:        {total}")
    print(f"Succeeded:    {succeeded}")
    print(f"Skipped:      {skipped}")
    print(f"Insufficient: {insufficient} (STL needs >= 2*period per series)")
    print(f"Failed:       {failed}")


if __name__ == "__main__":
    main()
