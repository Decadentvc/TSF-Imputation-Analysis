"""
批量窗口特征分析脚本

针对一个或多个模型，找到对应的干净与填补窗口：
1. 分析预测窗口（Prediction Windows）的6个STL特征指标
2. 分析历史窗口（History/Injection Windows）的6个STL特征指标

示例：
python Analysis/run_batch_analysis.py --model sundial --terms short,medium,long
python Analysis/run_batch_analysis.py --model sundial --dataset ETTh1 --terms short
python Analysis/run_batch_analysis.py --model sundial chronos2 --terms short --imputation_methods linear,mean
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from Analysis.metrics import calculate_all_metrics, get_period
from Analysis.window_analysis import (
    get_model_max_context,
    parse_prediction_dirname,
    is_imputation_method_dir,
)
from statsmodels.tsa.seasonal import STL


DEFAULT_DATA_PATH = "data/datasets"
DEFAULT_PROPERTIES_PATH = "data/datasets/dataset_properties.json"
DEFAULT_MODEL_PROPERTIES_PATH = "Eval/model_properties.json"
DEFAULT_INTERMEDIATE_DIR = "data/Intermediate_Predictions"
DEFAULT_OUTPUT_DIR = "results_analysis"


def load_dataset_properties(properties_path: str = DEFAULT_PROPERTIES_PATH) -> Dict[str, Any]:
    props_path = Path(properties_path)
    if not props_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {props_path}")
    with open(props_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_properties(model_properties_path: str = DEFAULT_MODEL_PROPERTIES_PATH) -> Dict[str, Any]:
    props_path = Path(model_properties_path)
    if not props_path.exists():
        raise FileNotFoundError(f"Model properties not found: {props_path}")
    with open(props_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_allowed_terms(dataset_name: str, properties_path: str) -> List[str]:
    props = load_dataset_properties(properties_path)
    if dataset_name not in props:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    term_type = props[dataset_name].get("term", "med_long")
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


def analyze_single_window(
    data: pd.DataFrame,
    period: int,
    dataset: str = "",
    window_idx: int = 0,
) -> Dict[str, Any]:
    value_cols = [col for col in data.columns if col.lower() not in ['date', 'time', 'timestamp', 'datetime', 'index']]
    
    if not value_cols:
        return {
            "success": False,
            "error": "No value columns found",
            "window_idx": window_idx,
        }
    
    all_metrics = {
        "trend_strength": [],
        "trend_linearity": [],
        "seasonal_strength": [],
        "seasonal_correlation": [],
        "residual_autocorr_lag1": [],
        "spectral_entropy": [],
    }
    
    for col in value_cols:
        series = data[col].dropna().values
        
        if len(series) < 2 * period:
            continue
        
        try:
            stl = STL(series, period=period, robust=True)
            result = stl.fit()
            
            trend = result.trend
            seasonal = result.seasonal
            residual = result.resid
            
            metrics = calculate_all_metrics(series, trend, seasonal, residual, period)
            
            for key, value in metrics.items():
                all_metrics[key].append(value)
                
        except Exception as e:
            continue
    
    if not all_metrics["trend_strength"]:
        return {
            "success": False,
            "error": "No valid series for STL decomposition",
            "window_idx": window_idx,
        }
    
    avg_metrics = {key: float(np.mean(values)) for key, values in all_metrics.items()}
    
    return {
        "success": True,
        "window_idx": window_idx,
        "dataset": dataset,
        "metrics": avg_metrics,
        "n_series": len(all_metrics["trend_strength"]),
    }


def analyze_prediction_windows(
    prediction_dir: Path,
    properties_path: str,
    model: str,
) -> Dict[str, Any]:
    if not prediction_dir.exists():
        return {
            "success": False,
            "error": f"Prediction directory not found: {prediction_dir}",
        }
    
    imputation_method = None
    dir_info = None
    
    if is_imputation_method_dir(prediction_dir.name):
        imputation_method = prediction_dir.name.lower()
        parent_dir = prediction_dir.parent
        dir_info = parse_prediction_dirname(parent_dir.name)
        if not dir_info:
            return {
                "success": False,
                "error": f"Cannot parse parent directory name: {parent_dir.name}",
            }
    else:
        dir_info = parse_prediction_dirname(prediction_dir.name)
        if not dir_info:
            return {
                "success": False,
                "error": f"Cannot parse prediction directory name: {prediction_dir.name}",
            }
    
    dataset = dir_info['dataset']
    period = get_period(dataset, properties_path)
    
    csv_files = sorted(prediction_dir.glob("*.csv"))
    
    window_results = []
    
    for csv_file in csv_files:
        pattern = r'_(\d+)\.csv$'
        match = re.search(pattern, csv_file.name)
        if not match:
            continue
        
        window_idx = int(match.group(1))
        
        try:
            df = pd.read_csv(csv_file)
            result = analyze_single_window(
                data=df,
                period=period,
                dataset=dataset,
                window_idx=window_idx,
            )
            
            if result["success"]:
                window_results.append(result)
                
        except Exception as e:
            continue
    
    if not window_results:
        return {
            "success": False,
            "error": "No valid windows analyzed",
            "dir_info": dir_info,
            "imputation_method": imputation_method,
        }
    
    all_metrics = {
        "trend_strength": [],
        "trend_linearity": [],
        "seasonal_strength": [],
        "seasonal_correlation": [],
        "residual_autocorr_lag1": [],
        "spectral_entropy": [],
    }
    
    for result in window_results:
        for key, value in result["metrics"].items():
            all_metrics[key].append(value)
    
    summary = {
        "mean": {key: float(np.mean(values)) for key, values in all_metrics.items()},
        "std": {key: float(np.std(values)) for key, values in all_metrics.items()},
        "min": {key: float(np.min(values)) for key, values in all_metrics.items()},
        "max": {key: float(np.max(values)) for key, values in all_metrics.items()},
    }
    
    return {
        "success": True,
        "model": model,
        "dir_info": dir_info,
        "imputation_method": imputation_method,
        "n_windows": len(window_results),
        "period": period,
        "summary": summary,
        "window_results": window_results,
    }


def analyze_history_windows(
    dataset_name: str,
    model: str,
    term: str,
    method: str = "BM",
    ratio: float = 0.1,
    impute_method: Optional[str] = None,
    data_path: str = DEFAULT_DATA_PATH,
    properties_path: str = DEFAULT_PROPERTIES_PATH,
    model_properties_path: str = DEFAULT_MODEL_PROPERTIES_PATH,
) -> Dict[str, Any]:
    max_context = get_model_max_context(model, model_properties_path)
    
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "Missing_Value_Injection"))
    from inject_range_utils import get_injection_range
    
    inject_range = get_injection_range(
        dataset_name=dataset_name,
        term=term,
        data_path=data_path,
        max_context=max_context,
    )
    
    n_windows = inject_range['windows']
    prediction_length = inject_range['prediction_length']
    total_length = inject_range['total_length']
    
    if impute_method:
        ratio_str = f"{int(ratio * 100):03d}"
        imputed_path = Path(data_path) / "Imputed" / method / f"{method}_{ratio_str}" / f"{dataset_name}_{method}_{ratio_str}_{term}_{impute_method}.csv"
        
        if not imputed_path.exists():
            return {
                "success": False,
                "error": f"Imputed data not found: {imputed_path}",
                "inject_range": inject_range,
            }
        
        df = pd.read_csv(imputed_path)
    else:
        clean_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
        if not clean_path.exists():
            return {
                "success": False,
                "error": f"Clean data not found: {clean_path}",
                "inject_range": inject_range,
            }
        df = pd.read_csv(clean_path)
    
    time_cols = ['date', 'time', 'timestamp', 'datetime', 'index']
    data_cols = [col for col in df.columns if col.lower() not in time_cols]
    
    period = get_period(dataset_name, properties_path)
    
    window_results = []
    
    for window_idx in range(n_windows):
        forecast_end = total_length - prediction_length * window_idx
        forecast_start = forecast_end - prediction_length
        history_end = forecast_start
        history_start = max(0, history_end - max_context)
        
        history_data = df.iloc[history_start:history_end]
        
        all_metrics = {
            "trend_strength": [],
            "trend_linearity": [],
            "seasonal_strength": [],
            "seasonal_correlation": [],
            "residual_autocorr_lag1": [],
            "spectral_entropy": [],
        }
        
        for col in data_cols:
            series = history_data[col].dropna().values
            
            if len(series) < 2 * period:
                continue
            
            try:
                stl = STL(series, period=period, robust=True)
                result = stl.fit()
                
                trend = result.trend
                seasonal = result.seasonal
                residual = result.resid
                
                metrics = calculate_all_metrics(series, trend, seasonal, residual, period)
                
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                    
            except Exception as e:
                continue
        
        if all_metrics["trend_strength"]:
            avg_metrics = {key: float(np.mean(values)) for key, values in all_metrics.items()}
            window_results.append({
                "window_idx": window_idx,
                "history_start": history_start,
                "history_end": history_end,
                "history_length": history_end - history_start,
                "metrics": avg_metrics,
                "n_series": len(all_metrics["trend_strength"]),
            })
    
    if not window_results:
        return {
            "success": False,
            "error": "No valid windows analyzed",
            "inject_range": inject_range,
        }
    
    all_metrics = {
        "trend_strength": [],
        "trend_linearity": [],
        "seasonal_strength": [],
        "seasonal_correlation": [],
        "residual_autocorr_lag1": [],
        "spectral_entropy": [],
    }
    
    for result in window_results:
        for key, value in result["metrics"].items():
            all_metrics[key].append(value)
    
    summary = {
        "mean": {key: float(np.mean(values)) for key, values in all_metrics.items()},
        "std": {key: float(np.std(values)) for key, values in all_metrics.items()},
        "min": {key: float(np.min(values)) for key, values in all_metrics.items()},
        "max": {key: float(np.max(values)) for key, values in all_metrics.items()},
    }
    
    return {
        "success": True,
        "model": model,
        "dataset": dataset_name,
        "method": method if impute_method else "clean",
        "ratio": ratio if impute_method else None,
        "term": term,
        "imputation_method": impute_method,
        "inject_range": inject_range,
        "period": period,
        "n_windows": len(window_results),
        "summary": summary,
        "window_results": window_results,
    }


def find_prediction_dirs(
    intermediate_dir: Path,
    model: str,
    terms: List[str],
    imputation_methods: Optional[List[str]] = None,
) -> List[Tuple[Path, str, Optional[str]]]:
    model_dir = intermediate_dir / model.lower()
    if not model_dir.exists():
        return []
    
    results = []
    
    for pred_dir in sorted(model_dir.iterdir()):
        if not pred_dir.is_dir():
            continue
        
        dir_info = parse_prediction_dirname(pred_dir.name)
        if not dir_info:
            continue
        
        if dir_info['term'] not in terms:
            continue
        
        if dir_info['method'] == 'clean':
            results.append((pred_dir, 'clean', None))
        elif imputation_methods:
            for imp_method_dir in sorted(pred_dir.iterdir()):
                if imp_method_dir.is_dir() and is_imputation_method_dir(imp_method_dir.name):
                    if imp_method_dir.name.lower() in imputation_methods:
                        results.append((imp_method_dir, 'imputed', imp_method_dir.name.lower()))
    
    return results


def find_available_datasets(
    intermediate_dir: Path,
    model: str,
    terms: List[str],
) -> List[Tuple[str, str, Optional[float]]]:
    model_dir = intermediate_dir / model.lower()
    if not model_dir.exists():
        return []
    
    datasets_info = set()
    
    for pred_dir in sorted(model_dir.iterdir()):
        if not pred_dir.is_dir():
            continue
        
        dir_info = parse_prediction_dirname(pred_dir.name)
        if not dir_info:
            continue
        
        if dir_info['term'] not in terms:
            continue
        
        if dir_info['method'] == 'clean':
            datasets_info.add((dir_info['dataset'], 'clean', None))
        else:
            datasets_info.add((dir_info['dataset'], dir_info['method'], dir_info['ratio']))
    
    return sorted(datasets_info)


def save_prediction_analysis_result(
    result: Dict[str, Any],
    output_dir: Path,
    model: str,
) -> Path:
    model_output_dir = output_dir / model.lower() / "prediction"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    dir_info = result['dir_info']
    imputation_method = result.get('imputation_method')
    
    if dir_info['method'] == 'clean':
        filename = f"{dir_info['dataset']}_clean_{dir_info['term']}_prediction.csv"
    else:
        ratio_str = f"{int(dir_info['ratio'] * 100):03d}"
        if imputation_method:
            filename = f"{dir_info['dataset']}_{dir_info['method']}_{ratio_str}_{dir_info['term']}_{imputation_method}_prediction.csv"
        else:
            filename = f"{dir_info['dataset']}_{dir_info['method']}_{ratio_str}_{dir_info['term']}_prediction.csv"
    
    output_path = model_output_dir / filename
    
    window_results = result.get('window_results', [])
    rows = []
    for wr in window_results:
        row = {
            "window_idx": wr['window_idx'],
            "n_series": wr['n_series'],
        }
        row.update(wr['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_data = {
        "model": result.get('model'),
        "dataset": dir_info['dataset'],
        "method": dir_info['method'],
        "ratio": dir_info.get('ratio'),
        "term": dir_info['term'],
        "imputation_method": imputation_method,
        "n_windows": result.get('n_windows'),
        "period": result.get('period'),
        "summary": result.get('summary'),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def save_history_analysis_result(
    result: Dict[str, Any],
    output_dir: Path,
    model: str,
) -> Path:
    model_output_dir = output_dir / model.lower() / "history"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = result['dataset']
    method = result['method']
    ratio = result.get('ratio')
    term = result['term']
    impute_method = result.get('imputation_method')
    
    if method == 'clean':
        filename = f"{dataset}_clean_{term}_history.csv"
    else:
        ratio_str = f"{int(ratio * 100):03d}"
        if impute_method:
            filename = f"{dataset}_{method}_{ratio_str}_{term}_{impute_method}_history.csv"
        else:
            filename = f"{dataset}_{method}_{ratio_str}_{term}_history.csv"
    
    output_path = model_output_dir / filename
    
    window_results = result.get('window_results', [])
    rows = []
    for wr in window_results:
        row = {
            "window_idx": wr['window_idx'],
            "history_start": wr['history_start'],
            "history_end": wr['history_end'],
            "history_length": wr['history_length'],
            "n_series": wr['n_series'],
        }
        row.update(wr['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_data = {
        "model": result.get('model'),
        "dataset": dataset,
        "method": method,
        "ratio": ratio,
        "term": term,
        "imputation_method": impute_method,
        "n_windows": result.get('n_windows'),
        "period": result.get('period'),
        "inject_range": result.get('inject_range'),
        "summary": result.get('summary'),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    return output_path


def generate_overall_summary(
    results_list: List[Dict[str, Any]],
    output_dir: Path,
    model: str,
    window_type: str,
) -> Path:
    if not results_list:
        return None
    
    all_metrics = {
        "trend_strength": [],
        "trend_linearity": [],
        "seasonal_strength": [],
        "seasonal_correlation": [],
        "residual_autocorr_lag1": [],
        "spectral_entropy": [],
    }
    
    for result in results_list:
        if 'summary' in result:
            for metric_name in all_metrics.keys():
                if metric_name in result['summary'].get('mean', {}):
                    all_metrics[metric_name].append(result['summary']['mean'][metric_name])
    
    if not any(all_metrics.values()):
        return None
    
    overall_summary = {
        "mean": {key: float(np.mean(values)) if values else None for key, values in all_metrics.items()},
        "std": {key: float(np.std(values)) if values else None for key, values in all_metrics.items()},
        "min": {key: float(np.min(values)) if values else None for key, values in all_metrics.items()},
        "max": {key: float(np.max(values)) if values else None for key, values in all_metrics.items()},
        "count": {key: len(values) for key, values in all_metrics.items()},
    }
    
    clean_results = [r for r in results_list if r.get('method') == 'clean' or (r.get('dir_info') and r['dir_info'].get('method') == 'clean')]
    imputed_results = [r for r in results_list if r not in clean_results]
    
    clean_metrics = {key: [] for key in all_metrics.keys()}
    for result in clean_results:
        if 'summary' in result:
            for metric_name in clean_metrics.keys():
                if metric_name in result['summary'].get('mean', {}):
                    clean_metrics[metric_name].append(result['summary']['mean'][metric_name])
    
    imputed_metrics = {key: [] for key in all_metrics.keys()}
    for result in imputed_results:
        if 'summary' in result:
            for metric_name in imputed_metrics.keys():
                if metric_name in result['summary'].get('mean', {}):
                    imputed_metrics[metric_name].append(result['summary']['mean'][metric_name])
    
    summary_data = {
        "model": model,
        "window_type": window_type,
        "total_combinations": len(results_list),
        "clean_combinations": len(clean_results),
        "imputed_combinations": len(imputed_results),
        "overall_summary": overall_summary,
    }
    
    if clean_metrics and any(clean_metrics.values()):
        summary_data["clean_summary"] = {
            "mean": {key: float(np.mean(values)) if values else None for key, values in clean_metrics.items()},
            "std": {key: float(np.std(values)) if values else None for key, values in clean_metrics.items()},
            "min": {key: float(np.min(values)) if values else None for key, values in clean_metrics.items()},
            "max": {key: float(np.max(values)) if values else None for key, values in clean_metrics.items()},
            "count": {key: len(values) for key, values in clean_metrics.items()},
        }
    
    if imputed_metrics and any(imputed_metrics.values()):
        summary_data["imputed_summary"] = {
            "mean": {key: float(np.mean(values)) if values else None for key, values in imputed_metrics.items()},
            "std": {key: float(np.std(values)) if values else None for key, values in imputed_metrics.items()},
            "min": {key: float(np.min(values)) if values else None for key, values in imputed_metrics.items()},
            "max": {key: float(np.max(values)) if values else None for key, values in imputed_metrics.items()},
            "count": {key: len(values) for key, values in imputed_metrics.items()},
        }
    
    model_output_dir = output_dir / model.lower()
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = model_output_dir / f"overall_{window_type}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch analyze prediction and history windows with 6 STL-based metrics"
    )
    
    parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        choices=["sundial", "chronos2", "timesfm2p5"],
        help="One or more model names",
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name. If omitted, analyze all available datasets",
    )
    
    parser.add_argument(
        "--terms",
        nargs="+",
        default=None,
        help="One or more terms, e.g. short,medium,long",
    )
    
    parser.add_argument(
        "--imputation_methods",
        nargs="+",
        default=["linear", "mean", "forward", "backward"],
        help="Imputation methods to analyze",
    )
    
    parser.add_argument(
        "--intermediate_dir",
        type=str,
        default=DEFAULT_INTERMEDIATE_DIR,
        help="Intermediate predictions directory",
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for analysis results",
    )
    
    parser.add_argument(
        "--properties_path",
        type=str,
        default=DEFAULT_PROPERTIES_PATH,
        help="Dataset properties file path",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when result file already exists",
    )
    
    parser.add_argument(
        "--prediction_only",
        action="store_true",
        help="Only analyze prediction windows",
    )
    
    parser.add_argument(
        "--history_only",
        action="store_true",
        help="Only analyze history windows",
    )
    
    parser.add_argument(
        "--clean_only",
        action="store_true",
        help="Only analyze clean prediction windows",
    )
    
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    models = dedupe_lower(args.model)
    user_terms = split_multi_values(args.terms)
    normalized_terms = dedupe_lower(user_terms) if user_terms else None
    
    imputation_methods = dedupe_lower(args.imputation_methods) if args.imputation_methods else []
    
    intermediate_dir = Path(args.intermediate_dir)
    output_dir = Path(args.output_dir)
    properties_path = args.properties_path
    
    do_prediction = not args.history_only
    do_history = not args.prediction_only
    
    if not intermediate_dir.exists():
        raise FileNotFoundError(f"Intermediate directory not found: {intermediate_dir}")
    
    print("=" * 80)
    print("Batch Window Analysis")
    print("=" * 80)
    print(f"Models: {models}")
    print(f"Terms (requested): {normalized_terms if normalized_terms else 'all available'}")
    print(f"Imputation methods: {imputation_methods}")
    print(f"Clean only: {args.clean_only}")
    print(f"Analyze prediction: {do_prediction}")
    print(f"Analyze history: {do_history}")
    print(f"Intermediate dir: {intermediate_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 80)
    
    total_pred = 0
    skipped_pred = 0
    failed_pred = 0
    succeeded_pred = 0
    
    total_hist = 0
    skipped_hist = 0
    failed_hist = 0
    succeeded_hist = 0
    
    all_prediction_results = []
    all_history_results = []
    
    for model in models:
        print(f"\n{'='*80}")
        print(f"Model: {model}")
        print(f"{'='*80}")
        
        model_dir = intermediate_dir / model.lower()
        if not model_dir.exists():
            print(f"⚠ Model directory not found: {model_dir}")
            continue
        
        if args.dataset:
            dataset_filter = args.dataset
        else:
            dataset_filter = None
        
        available_terms = normalized_terms if normalized_terms else ["long", "medium", "short"]
        
        datasets_info = find_available_datasets(
            intermediate_dir=intermediate_dir,
            model=model,
            terms=available_terms,
        )
        
        if dataset_filter:
            datasets_info = [(d, m, r) for d, m, r in datasets_info if d == dataset_filter]
        
        model_prediction_results = []
        model_history_results = []
        
        if do_prediction:
            print(f"\n--- Analyzing Prediction Windows ---")
            pred_dirs = find_prediction_dirs(
                intermediate_dir=intermediate_dir,
                model=model,
                terms=available_terms,
                imputation_methods=[] if args.clean_only else imputation_methods,
            )
            
            if dataset_filter:
                pred_dirs = [
                    (p, t, m) for p, t, m in pred_dirs
                    if parse_prediction_dirname(p.parent.name if m else p.name)['dataset'] == dataset_filter
                ]
            
            for pred_dir, pred_type, imp_method in tqdm(pred_dirs, desc=f"Prediction {model}"):
                total_pred += 1
                
                dir_info = parse_prediction_dirname(pred_dir.parent.name if imp_method else pred_dir.name)
                
                result_filename = f"{dir_info['dataset']}_{dir_info['method']}"
                if dir_info['ratio']:
                    result_filename += f"_{int(dir_info['ratio'] * 100):03d}"
                result_filename += f"_{dir_info['term']}"
                if imp_method:
                    result_filename += f"_{imp_method}"
                result_filename += "_prediction.csv"
                
                result_path = output_dir / model.lower() / "prediction" / result_filename
                
                if result_path.exists() and not args.force:
                    skipped_pred += 1
                    continue
                
                try:
                    result = analyze_prediction_windows(
                        prediction_dir=pred_dir,
                        properties_path=properties_path,
                        model=model,
                    )
                    
                    if result['success']:
                        output_path = save_prediction_analysis_result(
                            result=result,
                            output_dir=output_dir,
                            model=model,
                        )
                        model_prediction_results.append(result)
                        succeeded_pred += 1
                        print(f"  ✓ Saved: {output_path}")
                    else:
                        failed_pred += 1
                        print(f"  ✗ Failed: {pred_dir} - {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    failed_pred += 1
                    print(f"  ✗ Error: {pred_dir} - {e}")
        
        if do_history:
            print(f"\n--- Analyzing History Windows ---")
            
            for dataset, method, ratio in tqdm(datasets_info, desc=f"History {model}"):
                if method == 'clean':
                    for term in available_terms:
                        total_hist += 1
                        
                        result_filename = f"{dataset}_clean_{term}_history.csv"
                        result_path = output_dir / model.lower() / "history" / result_filename
                        
                        if result_path.exists() and not args.force:
                            skipped_hist += 1
                            continue
                        
                        try:
                            result = analyze_history_windows(
                                dataset_name=dataset,
                                model=model,
                                term=term,
                                method="BM",
                                ratio=0.1,
                                impute_method=None,
                                data_path=DEFAULT_DATA_PATH,
                                properties_path=properties_path,
                                model_properties_path=DEFAULT_MODEL_PROPERTIES_PATH,
                            )
                            
                            if result['success']:
                                output_path = save_history_analysis_result(
                                    result=result,
                                    output_dir=output_dir,
                                    model=model,
                                )
                                model_history_results.append(result)
                                succeeded_hist += 1
                                print(f"  ✓ Saved: {output_path}")
                            else:
                                failed_hist += 1
                                print(f"  ✗ Failed: {dataset} clean {term} - {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            failed_hist += 1
                            print(f"  ✗ Error: {dataset} clean {term} - {e}")
                else:
                    for term in available_terms:
                        for impute_method in imputation_methods:
                            total_hist += 1
                            
                            ratio_str = f"{int(ratio * 100):03d}" if ratio else "010"
                            result_filename = f"{dataset}_{method}_{ratio_str}_{term}_{impute_method}_history.csv"
                            result_path = output_dir / model.lower() / "history" / result_filename
                            
                            if result_path.exists() and not args.force:
                                skipped_hist += 1
                                continue
                            
                            try:
                                result = analyze_history_windows(
                                    dataset_name=dataset,
                                    model=model,
                                    term=term,
                                    method=method,
                                    ratio=ratio if ratio else 0.1,
                                    impute_method=impute_method,
                                    data_path=DEFAULT_DATA_PATH,
                                    properties_path=properties_path,
                                    model_properties_path=DEFAULT_MODEL_PROPERTIES_PATH,
                                )
                                
                                if result['success']:
                                    output_path = save_history_analysis_result(
                                        result=result,
                                        output_dir=output_dir,
                                        model=model,
                                    )
                                    model_history_results.append(result)
                                    succeeded_hist += 1
                                    print(f"  ✓ Saved: {output_path}")
                                else:
                                    failed_hist += 1
                                    print(f"  ✗ Failed: {dataset} {method} {ratio} {term} {impute_method} - {result.get('error', 'Unknown error')}")
                                    
                            except Exception as e:
                                failed_hist += 1
                                print(f"  ✗ Error: {dataset} {method} {ratio} {term} {impute_method} - {e}")
        
        if do_prediction and model_prediction_results:
            overall_pred_path = generate_overall_summary(
                results_list=model_prediction_results,
                output_dir=output_dir,
                model=model,
                window_type="prediction",
            )
            if overall_pred_path:
                all_prediction_results.extend(model_prediction_results)
                print(f"\n  ✓ Overall prediction summary: {overall_pred_path}")
        
        if do_history and model_history_results:
            overall_hist_path = generate_overall_summary(
                results_list=model_history_results,
                output_dir=output_dir,
                model=model,
                window_type="history",
            )
            if overall_hist_path:
                all_history_results.extend(model_history_results)
                print(f"  ✓ Overall history summary: {overall_hist_path}")
    
    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    if do_prediction:
        print(f"Prediction Analysis:")
        print(f"  Total:    {total_pred}")
        print(f"  Succeeded:{succeeded_pred}")
        print(f"  Skipped:  {skipped_pred}")
        print(f"  Failed:   {failed_pred}")
    if do_history:
        print(f"History Analysis:")
        print(f"  Total:    {total_hist}")
        print(f"  Succeeded:{succeeded_hist}")
        print(f"  Skipped:  {skipped_hist}")
        print(f"  Failed:   {failed_hist}")


if __name__ == "__main__":
    main()
