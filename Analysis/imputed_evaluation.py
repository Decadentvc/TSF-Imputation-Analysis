"""
填补数据评估模块

用于计算填补数据与原始数据在各窗口上的差异
使用预测评估指标：MSE、MAE、MAPE、RMSE、NRMSE、ND 等
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_dataset_properties() -> Dict[str, Any]:
    """读取数据集属性配置"""
    props_path = Path(__file__).parent.parent / "datasets" / "dataset_properties.json"
    with open(props_path, 'r', encoding='utf-8') as f:
        return json.load(f)


TEST_SPLIT = 0.6
MAX_WINDOW = 20

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}


def parse_imputed_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    解析填补数据文件名
    
    格式: {dataset}_{method}_{ratio}_{term}_{imputation_method}.csv
    例如: ETTh1_BM_010_short_linear.csv
    
    Returns:
        解析后的字典，包含 dataset, method, ratio, term, imputation_method
    """
    pattern = r'^([A-Za-z0-9\-_]+?)_(MCAR|BM|TM|TVMR)_(\d{3})_(short|medium|long)_(zero|mean|forward|backward|linear|nearest|spline|seasonal)\.csv$'
    match = re.match(pattern, filename)
    if match:
        return {
            "dataset": match.group(1),
            "method": match.group(2),
            "ratio": int(match.group(3)) / 100.0,
            "term": match.group(4),
            "imputation_method": match.group(5),
        }
    return None


def compute_prediction_length(freq: str, term: str) -> int:
    """
    根据频率和 term 计算 prediction length
    
    Args:
        freq: 数据频率（如 'H', 'D', 'M' 等）
        term: short/medium/long
    
    Returns:
        prediction length
    """
    from gluonts.time_feature import norm_freq_str
    from pandas.tseries.frequencies import to_offset
    
    term_multiplier = {
        "short": 1,
        "medium": 10,
        "long": 15,
    }
    
    freq_normalized = norm_freq_str(to_offset(freq).name)
    deprecated_map = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }
    if freq_normalized in deprecated_map:
        freq_normalized = deprecated_map[freq_normalized]
    
    base_pred_len = PRED_LENGTH_MAP.get(freq_normalized, 48)
    return term_multiplier.get(term, 1) * base_pred_len


def calculate_window_metrics(
    imputed_window: np.ndarray,
    original_window: np.ndarray,
) -> Dict[str, float]:
    """
    计算单个窗口的评估指标
    
    Args:
        imputed_window: 填补数据窗口
        original_window: 原始数据窗口
    
    Returns:
        评估指标字典
    """
    if imputed_window.shape != original_window.shape:
        raise ValueError(f"Window shapes don't match: {imputed_window.shape} vs {original_window.shape}")
    
    diff = imputed_window - original_window
    
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))
    
    epsilon = 1e-10
    mape = np.mean(np.abs(diff) / (np.abs(original_window) + epsilon)) * 100
    
    original_range = np.max(original_window) - np.min(original_window)
    if original_range > epsilon:
        nrmse = rmse / original_range * 100
    else:
        nrmse = 0.0
    
    original_sum = np.sum(np.abs(original_window))
    if original_sum > epsilon:
        nd = np.sum(np.abs(diff)) / original_sum
    else:
        nd = 0.0
    
    denominator = (np.abs(original_window) + np.abs(imputed_window)) / 2 + epsilon
    smape = np.mean(np.abs(diff) / denominator) * 100
    
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "NRMSE": float(nrmse),
        "ND": float(nd),
        "sMAPE": float(smape),
    }


def evaluate_imputed_data(
    imputed_path: str,
    original_path: str,
    freq: str,
    term: str,
    prediction_length: Optional[int] = None,
) -> Dict[str, Any]:
    """
    评估填补数据与原始数据的差异
    
    Args:
        imputed_path: 填补数据文件路径
        original_path: 原始数据文件路径
        freq: 数据频率
        term: short/medium/long
        prediction_length: 窗口长度
    
    Returns:
        包含所有评估结果的字典
    """
    imputed_df = pd.read_csv(imputed_path)
    original_df = pd.read_csv(original_path)
    
    time_col = None
    for c in ['date', 'time', 'timestamp']:
        if c in imputed_df.columns:
            time_col = c
            break
    
    if time_col:
        imputed_df[time_col] = pd.to_datetime(imputed_df[time_col])
        imputed_df = imputed_df.set_index(time_col)
        original_df[time_col] = pd.to_datetime(original_df[time_col])
        original_df = original_df.set_index(time_col)
    
    if len(imputed_df) != len(original_df):
        raise ValueError(f"Imputed and original datasets must have the same length. "
                        f"Imputed: {len(imputed_df)}, Original: {len(original_df)}")
    
    if prediction_length is None:
        prediction_length = compute_prediction_length(freq, term)
    
    min_series_length = min(len(imputed_df.iloc[:, i]) for i in range(len(imputed_df.columns)))
    w = math.ceil(TEST_SPLIT * min_series_length / prediction_length)
    windows = min(max(1, w), MAX_WINDOW)
    
    total_length = prediction_length * windows
    start_idx = len(imputed_df) - total_length
    
    result = {
        "imputed_path": imputed_path,
        "original_path": original_path,
        "freq": freq,
        "term": term,
        "prediction_length": prediction_length,
        "windows": windows,
        "total_samples": len(imputed_df),
        "eval_samples": total_length,
        "window_results": [],
        "summary": {},
    }
    
    all_metrics = {key: [] for key in ["MSE", "RMSE", "MAE", "MAPE", "NRMSE", "ND", "sMAPE"]}
    
    data_cols = list(imputed_df.columns)
    
    for win_idx in range(windows):
        win_start = start_idx + win_idx * prediction_length
        win_end = win_start + prediction_length
        
        window_metrics = {
            "window_index": win_idx,
            "start_idx": win_start,
            "end_idx": win_end,
            "columns": {},
        }
        
        for col in data_cols:
            imputed_window = imputed_df[col].iloc[win_start:win_end].values.astype(np.float32)
            original_window = original_df[col].iloc[win_start:win_end].values.astype(np.float32)
            
            valid_mask = ~(np.isnan(imputed_window) | np.isnan(original_window))
            if valid_mask.sum() < 2:
                continue
            
            imputed_valid = imputed_window[valid_mask]
            original_valid = original_window[valid_mask]
            
            metrics = calculate_window_metrics(imputed_valid, original_valid)
            window_metrics["columns"][col] = metrics
            
            for key in metrics:
                all_metrics[key].append(metrics[key])
        
        if window_metrics["columns"]:
            col_metrics = {}
            for key in ["MSE", "RMSE", "MAE", "MAPE", "NRMSE", "ND", "sMAPE"]:
                values = [m[key] for m in window_metrics["columns"].values()]
                col_metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            window_metrics["summary"] = col_metrics
            result["window_results"].append(window_metrics)
    
    for key in all_metrics:
        if all_metrics[key]:
            result["summary"][key] = {
                "mean": float(np.mean(all_metrics[key])),
                "std": float(np.std(all_metrics[key])),
                "min": float(np.min(all_metrics[key])),
                "max": float(np.max(all_metrics[key])),
            }
    
    return result


def get_all_imputed_files(
    base_dir: str = "datasets/Imputed",
) -> List[Dict[str, Any]]:
    """
    获取所有填补数据文件信息
    
    Returns:
        包含文件信息的字典列表
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    files = []
    for method_dir in base_path.iterdir():
        if not method_dir.is_dir():
            continue
        
        method_name = method_dir.name
        
        for ratio_dir in method_dir.iterdir():
            if not ratio_dir.is_dir():
                continue
            
            ratio_name = ratio_dir.name
            
            for file in ratio_dir.iterdir():
                if not file.is_file() or not file.name.endswith('.csv'):
                    continue
                
                info = parse_imputed_filename(file.name)
                if info:
                    info["file_path"] = str(file)
                    info["method_dir"] = method_name
                    info["ratio_dir"] = ratio_name
                    files.append(info)
    
    return sorted(files, key=lambda x: (x["dataset"], x["method"], x["ratio"], x["term"], x["imputation_method"]))


def run_batch_evaluation(
    output_dir: str = "results/imputed_evaluation",
    datasets: Optional[List[str]] = None,
    terms: Optional[List[str]] = None,
    impute_methods: Optional[List[str]] = None,
    missing_methods: Optional[List[str]] = None,
    ratios: Optional[List[float]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    批量评估填补数据
    
    Args:
        output_dir: 输出目录
        datasets: 指定数据集列表
        terms: 指定 term 列表
        impute_methods: 指定填补方法列表
        missing_methods: 指定缺失注入方法列表
        ratios: 指定缺失比例列表
        overwrite: 是否覆盖已有结果
    
    Returns:
        汇总结果
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"填补数据批量评估")
    print(f"{'='*80}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*80}")
    
    imputed_files = get_all_imputed_files()
    print(f"  找到 {len(imputed_files)} 个填补数据文件")
    
    dataset_props = json.loads(
        (Path(__file__).parent.parent / "datasets" / "dataset_properties.json").read_text()
    )
    
    results = {
        "total_files": len(imputed_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": [],
    }
    
    for file_info in tqdm(imputed_files, desc="评估进度"):
        dataset = file_info["dataset"]
        term = file_info["term"]
        impute_method = file_info["imputation_method"]
        missing_method = file_info["method"]
        ratio = file_info["ratio"]
        
        if datasets and dataset not in datasets:
            continue
        if terms and term not in terms:
            continue
        if impute_methods and impute_method not in impute_methods:
            continue
        if missing_methods and missing_method not in missing_methods:
            continue
        if ratios and ratio not in ratios:
            continue
        
        key = f"{dataset}_{missing_method}_{int(ratio*100):03d}_{term}_{impute_method}"
        json_output = output_path / f"{key}.json"
        
        if json_output.exists() and not overwrite:
            results["skipped"] += 1
            continue
        
        if dataset not in dataset_props:
            print(f"  [跳过] 数据集 {dataset} 不在 dataset_properties.json 中")
            results["skipped"] += 1
            continue
        
        freq = dataset_props[dataset]["frequency"]
        original_path = Path(__file__).parent.parent / "datasets" / "ori" / f"{dataset}.csv"
        
        if not original_path.exists():
            print(f"  [跳过] 原始数据文件不存在: {original_path}")
            results["skipped"] += 1
            continue
        
        try:
            eval_result = evaluate_imputed_data(
                imputed_path=file_info["file_path"],
                original_path=str(original_path),
                freq=freq,
                term=term,
            )
            
            eval_result["dataset"] = dataset
            eval_result["missing_method"] = missing_method
            eval_result["ratio"] = ratio
            eval_result["imputation_method"] = impute_method
            
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(eval_result, f, indent=2, ensure_ascii=False)
            
            results["processed"] += 1
            
        except Exception as e:
            print(f"  [错误] {key}: {e}")
            results["failed"] += 1
            results["errors"].append({"file": key, "error": str(e)})
    
    summary_output = output_path / "summary.json"
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"评估完成")
    print(f"  处理: {results['processed']}")
    print(f"  跳过: {results['skipped']}")
    print(f"  失败: {results['failed']}")
    print(f"{'='*80}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="填补数据评估工具")
    parser.add_argument("--output_dir", type=str, default="results/imputed_evaluation",
                       help="输出目录")
    parser.add_argument("--datasets", type=str, default=None,
                       help="指定数据集，逗号分隔")
    parser.add_argument("--terms", type=str, default=None,
                       help="指定 term，逗号分隔")
    parser.add_argument("--impute_methods", type=str, default=None,
                       help="指定填补方法，逗号分隔")
    parser.add_argument("--missing_methods", type=str, default=None,
                       help="指定缺失注入方法，逗号分隔")
    parser.add_argument("--ratios", type=str, default=None,
                       help="指定缺失比例，逗号分隔")
    parser.add_argument("--overwrite", action="store_true",
                       help="覆盖已有结果")
    
    args = parser.parse_args()
    
    datasets = args.datasets.split(",") if args.datasets else None
    terms = args.terms.split(",") if args.terms else None
    impute_methods = args.impute_methods.split(",") if args.impute_methods else None
    missing_methods = args.missing_methods.split(",") if args.missing_methods else None
    ratios = [float(r) for r in args.ratios.split(",")] if args.ratios else None
    
    run_batch_evaluation(
        output_dir=args.output_dir,
        datasets=datasets,
        terms=terms,
        impute_methods=impute_methods,
        missing_methods=missing_methods,
        ratios=ratios,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
