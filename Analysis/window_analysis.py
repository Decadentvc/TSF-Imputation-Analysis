"""
窗口特征分析模块

用于分析：
1. Intermediate_Predictions 下的预测窗口特征
2. 原始数据中历史窗口（注入区间）的特征
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from statsmodels.tsa.seasonal import STL
import sys
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "Missing_Value_Injection" / "for_sundial"))
from inject_range_utils import get_injection_range, Term

from Analysis.metrics import (
    calculate_all_metrics,
    get_period,
)


def parse_prediction_dirname(dirname: str) -> Optional[Dict[str, Any]]:
    """
    解析预测目录名
    
    格式: {dataset}_{method}_length{length}_{ratio}_{term}_prediction
    例如: ETTh1_BM_length50_010_long_prediction
    
    Returns:
        解析后的字典，包含 dataset, method, length, ratio, term
    """
    pattern = r'^([A-Za-z0-9\-_]+?)_(MCAR|BM|TM|TVMR)(?:_length(\d+))?_(\d{3})_(short|medium|long)_prediction$'
    match = re.match(pattern, dirname)
    if match:
        return {
            "dataset": match.group(1),
            "method": match.group(2),
            "block_length": int(match.group(3)) if match.group(3) else None,
            "ratio": int(match.group(4)) / 100.0,
            "term": match.group(5),
        }
    clean_pattern = r'^([A-Za-z0-9\-_]+?)_clean_(short|medium|long)_prediction$'
    clean_match = re.match(clean_pattern, dirname)
    if clean_match:
        return {
            "dataset": clean_match.group(1),
            "method": "clean",
            "block_length": None,
            "ratio": None,
            "term": clean_match.group(2),
        }
    return None


def is_imputation_method_dir(dirname: str) -> bool:
    """
    判断是否为填补方法子目录
    
    填补方法名称: zero, mean, forward, backward, linear, nearest, spline, seasonal
    """
    imputation_methods = ['zero', 'mean', 'forward', 'backward', 'linear', 'nearest', 'spline', 'seasonal']
    return dirname.lower() in imputation_methods


def get_imputation_method_from_path(path: Path) -> Optional[str]:
    """
    从路径中提取填补方法名称
    
    新目录结构:
    datasets/Intermediate_Predictions/{eval_data_name}_prediction/{imputation_method}/
    """
    if path.name.lower() in ['zero', 'mean', 'forward', 'backward', 'linear', 'nearest', 'spline', 'seasonal']:
        return path.name.lower()
    
    parent = path.parent
    if parent.name.endswith('_prediction'):
        return None
    
    return None


def parse_prediction_filename(filename: str) -> Optional[int]:
    """
    解析预测文件名，返回窗口索引
    
    格式: {prefix}_{window_idx}.csv
    例如: ETTh1_BM_length50_010_long_prediction_0.csv -> 0
    """
    pattern = r'_(\d+)\.csv$'
    match = re.search(pattern, filename)
    
    if match:
        return int(match.group(1))
    return None


def analyze_single_window(
    data: pd.DataFrame,
    period: int,
    dataset: str = "",
    window_idx: int = 0,
) -> Dict[str, Any]:
    """
    分析单个窗口的 6 个特征指标
    
    Args:
        data: 窗口数据（包含 date 和 prediction/value 列）
        period: STL 分解周期
        dataset: 数据集名称
        window_idx: 窗口索引
        
    Returns:
        分析结果字典
    """
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
            print(f"    [警告] 窗口 {window_idx} 列 {col} STL 分解失败: {e}")
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
    prediction_dir: str,
    properties_path: str = "datasets/dataset_properties.json",
) -> Dict[str, Any]:
    """
    分析预测目录下所有窗口的特征指标
    
    支持两种目录结构：
    1. 干净数据: datasets/Intermediate_Predictions/{dataset}_clean_{term}_prediction/
    2. 填补数据: datasets/Intermediate_Predictions/{eval_data_name}_prediction/{imputation_method}/
    
    Args:
        prediction_dir: 预测窗口目录路径
        properties_path: 数据集属性文件路径
        
    Returns:
        汇总结果字典
    """
    pred_path = Path(prediction_dir)
    
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_path}")
    
    imputation_method = None
    dir_info = None
    
    if is_imputation_method_dir(pred_path.name):
        imputation_method = pred_path.name.lower()
        parent_dir = pred_path.parent
        dir_info = parse_prediction_dirname(parent_dir.name)
        if not dir_info:
            raise ValueError(f"Cannot parse parent directory name: {parent_dir.name}")
    else:
        dir_info = parse_prediction_dirname(pred_path.name)
        if not dir_info:
            raise ValueError(f"Cannot parse prediction directory name: {pred_path.name}")
    
    print(f"\n{'='*80}")
    print(f"分析预测窗口特征")
    print(f"{'='*80}")
    print(f"  数据集: {dir_info['dataset']}")
    print(f"  方法: {dir_info['method']}")
    if dir_info['ratio'] is not None:
        print(f"  缺失比例: {dir_info['ratio']}")
    print(f"  Term: {dir_info['term']}")
    if dir_info['block_length']:
        print(f"  块长度: {dir_info['block_length']}")
    if imputation_method:
        print(f"  填补方法: {imputation_method}")
    print(f"{'='*80}")
    
    period = get_period(dir_info['dataset'], properties_path) if dir_info['dataset'] else None
    if period is None:
        raise ValueError(f"Dataset name missing for directory: {pred_path.name}")
    print(f"  周期: {period}")
    
    csv_files = sorted(pred_path.glob("*.csv"))
    print(f"  找到 {len(csv_files)} 个预测窗口文件")
    
    window_results = []
    
    for csv_file in csv_files:
        window_idx = parse_prediction_filename(csv_file.name)
        if window_idx is None:
            print(f"    [跳过] 无法解析文件名: {csv_file.name}")
            continue
        
        try:
            df = pd.read_csv(csv_file)
            result = analyze_single_window(
                data=df,
                period=period,
                dataset=dir_info['dataset'],
                window_idx=window_idx,
            )
            
            if result["success"]:
                window_results.append(result)
                
        except Exception as e:
            print(f"    [错误] 处理窗口 {window_idx} 失败: {e}")
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
    
    print(f"\n  窗口特征汇总 (共 {len(window_results)} 个窗口):")
    print(f"  {'指标':<25} {'均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12}")
    print(f"  {'-'*73}")
    for metric_name in all_metrics.keys():
        print(f"  {metric_name:<25} {summary['mean'][metric_name]:>12.4f} {summary['std'][metric_name]:>12.4f} {summary['min'][metric_name]:>12.4f} {summary['max'][metric_name]:>12.4f}")
    
    return {
        "success": True,
        "dir_info": dir_info,
        "imputation_method": imputation_method,
        "n_windows": len(window_results),
        "period": period,
        "summary": summary,
        "window_results": window_results,
    }


def analyze_history_windows(
    dataset_name: str,
    term: str = "short",
    data_path: str = "datasets",
    properties_path: str = "datasets/dataset_properties.json",
) -> Dict[str, Any]:
    """
    分析原始数据中每个历史窗口的特征指标
    
    根据 inject_range_utils 的计算，每个预测窗口对应一个历史窗口，
    历史窗口长度为 max_context (2880)，紧邻预测窗口。
    
    Args:
        dataset_name: 数据集名称
        term: 预测周期
        data_path: 数据集目录
        properties_path: 数据集属性文件路径
        
    Returns:
        分析结果字典
    """
    print(f"\n{'='*80}")
    print(f"分析历史窗口特征")
    print(f"{'='*80}")
    print(f"  数据集: {dataset_name}")
    print(f"  Term: {term}")
    print(f"{'='*80}")
    
    inject_range = get_injection_range(
        dataset_name=dataset_name,
        term=term,
        data_path=data_path,
    )
    
    n_windows = inject_range['windows']
    prediction_length = inject_range['prediction_length']
    max_context = inject_range['max_context']
    total_length = inject_range['total_length']
    
    print(f"\n  窗口配置:")
    print(f"    窗口数: {n_windows}")
    print(f"    预测长度: {prediction_length}")
    print(f"    历史窗口长度: {max_context}")
    
    clean_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {clean_path}")
    
    df = pd.read_csv(clean_path)
    
    time_cols = ['date', 'time', 'timestamp', 'datetime', 'index']
    data_cols = [col for col in df.columns if col.lower() not in time_cols]
    
    period = get_period(dataset_name, properties_path)
    print(f"  周期: {period}")
    
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
    
    print(f"\n  历史窗口特征汇总 (共 {len(window_results)} 个窗口):")
    print(f"  {'指标':<25} {'均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12}")
    print(f"  {'-'*73}")
    for metric_name in all_metrics.keys():
        print(f"  {metric_name:<25} {summary['mean'][metric_name]:>12.4f} {summary['std'][metric_name]:>12.4f} {summary['min'][metric_name]:>12.4f} {summary['max'][metric_name]:>12.4f}")
    
    return {
        "success": True,
        "dataset": dataset_name,
        "term": term,
        "inject_range": inject_range,
        "period": period,
        "n_windows": len(window_results),
        "summary": summary,
        "window_results": window_results,
    }


def parse_imputed_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    解析填补数据文件名
    
    格式: {dataset}_{method}_{ratio}_{term}_{impute_method}.csv
    例如: ETTh1_BM_010_short_backward.csv
    
    Returns:
        解析后的字典，包含 dataset, method, ratio, term, impute_method
    """
    pattern = r'^([A-Za-z0-9_]+?)_(MCAR|BM|TM|TVMR)_(\d{3})_(short|medium|long)_(\w+)\.csv$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    return {
        "dataset": match.group(1),
        "method": match.group(2),
        "ratio": int(match.group(3)) / 100.0,
        "term": match.group(4),
        "impute_method": match.group(5),
    }


def analyze_imputed_history_windows(
    dataset_name: str,
    method: str = "BM",
    ratio: float = 0.1,
    term: str = "short",
    impute_method: str = "linear",
    data_path: str = "datasets",
    properties_path: str = "datasets/dataset_properties.json",
) -> Dict[str, Any]:
    """
    分析填补数据中每个历史窗口的特征指标
    
    Args:
        dataset_name: 数据集名称
        method: 缺失注入方法
        ratio: 缺失比例
        term: 预测周期
        impute_method: 填补方法
        data_path: 数据集目录
        properties_path: 数据集属性文件路径
        
    Returns:
        分析结果字典
    """
    print(f"\n{'='*80}")
    print(f"分析填补数据历史窗口特征")
    print(f"{'='*80}")
    print(f"  数据集: {dataset_name}")
    print(f"  缺失方法: {method}")
    print(f"  缺失比例: {ratio}")
    print(f"  Term: {term}")
    print(f"  填补方法: {impute_method}")
    print(f"{'='*80}")
    
    inject_range = get_injection_range(
        dataset_name=dataset_name,
        term=term,
        data_path=data_path,
    )
    
    n_windows = inject_range['windows']
    prediction_length = inject_range['prediction_length']
    max_context = inject_range['max_context']
    total_length = inject_range['total_length']
    
    print(f"\n  窗口配置:")
    print(f"    窗口数: {n_windows}")
    print(f"    预测长度: {prediction_length}")
    print(f"    历史窗口长度: {max_context}")
    
    ratio_str = f"{int(ratio * 100):03d}"
    imputed_path = Path(data_path) / "Imputed" / method / f"{method}_{ratio_str}" / f"{dataset_name}_{method}_{ratio_str}_{term}_{impute_method}.csv"
    
    if not imputed_path.exists():
        raise FileNotFoundError(f"Imputed data not found: {imputed_path}")
    
    df = pd.read_csv(imputed_path)
    
    time_cols = ['date', 'time', 'timestamp', 'datetime', 'index']
    data_cols = [col for col in df.columns if col.lower() not in time_cols]
    
    period = get_period(dataset_name, properties_path)
    print(f"  周期: {period}")
    
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
    
    print(f"\n  填补数据历史窗口特征汇总 (共 {len(window_results)} 个窗口):")
    print(f"  {'指标':<25} {'均值':>12} {'标准差':>12} {'最小值':>12} {'最大值':>12}")
    print(f"  {'-'*73}")
    for metric_name in all_metrics.keys():
        print(f"  {metric_name:<25} {summary['mean'][metric_name]:>12.4f} {summary['std'][metric_name]:>12.4f} {summary['min'][metric_name]:>12.4f} {summary['max'][metric_name]:>12.4f}")
    
    return {
        "success": True,
        "dataset": dataset_name,
        "method": method,
        "ratio": ratio,
        "term": term,
        "impute_method": impute_method,
        "inject_range": inject_range,
        "period": period,
        "n_windows": len(window_results),
        "summary": summary,
        "window_results": window_results,
    }


def get_available_impute_methods(
    dataset_name: str,
    method: str = "BM",
    ratio: float = 0.1,
    term: str = "short",
    data_path: str = "datasets",
) -> List[str]:
    """
    获取可用的填补方法列表
    
    Args:
        dataset_name: 数据集名称
        method: 缺失注入方法
        ratio: 缺失比例
        term: 预测周期
        data_path: 数据集目录
        
    Returns:
        可用的填补方法列表
    """
    ratio_str = f"{int(ratio * 100):03d}"
    imputed_dir = Path(data_path) / "Imputed" / method / f"{method}_{ratio_str}"
    
    if not imputed_dir.exists():
        return []
    
    impute_methods = []
    pattern = f"^{dataset_name}_{method}_{ratio_str}_{term}_(\\w+)\\.csv$"
    
    for file in imputed_dir.glob("*.csv"):
        match = re.match(pattern, file.name)
        if match:
            impute_methods.append(match.group(1))
    
    return sorted(impute_methods)


def save_results(results: Dict[str, Any], output_path: str):
    """保存分析结果到 JSON 文件"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n  结果已保存: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="窗口特征分析工具")
    subparsers = parser.add_subparsers(dest="mode", help="分析模式")
    
    parser_prediction = subparsers.add_parser("prediction", help="分析预测窗口特征")
    parser_prediction.add_argument(
        "--prediction_dir",
        type=str,
        required=True,
        help="预测窗口目录路径",
    )
    parser_prediction.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径",
    )
    
    parser_history = subparsers.add_parser("history", help="分析历史窗口特征")
    parser_history.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称",
    )
    parser_history.add_argument(
        "--term",
        type=str,
        default="short",
        choices=["short", "medium", "long"],
        help="预测周期",
    )
    parser_history.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径",
    )
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        sys.exit(0)
    
    try:
        results = None
        if args.mode == "prediction":
            results = analyze_prediction_windows(args.prediction_dir)
        elif args.mode == "history":
            results = analyze_history_windows(args.dataset, args.term)
        if args.output and results is not None:
            save_results(results, args.output)
            
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
