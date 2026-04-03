"""
插入缺失值模块

职责：读取干净数据 -> 分窗 -> 注入缺失值（复用现有 MCAR.py）-> 保存

使用示例：
    python Missing_Value_Injection/inject_missing.py --dataset exchange_rate --term short --missing_ratio 0.1
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

for_sundial_dir = Path(__file__).parent / "for_sundial"
sys.path.insert(0, str(for_sundial_dir))

from Missing_Value_Injection.for_sundial.inject_range_utils import (
    load_dataset_properties,
    compute_prediction_length,
    get_injection_range,
    Term,
    MAX_CONTEXT_SUNDIAL,
    TEST_SPLIT,
    MAX_WINDOW,
)
from Missing_Value_Injection.for_sundial.MCAR import inject_mcar


def compute_window_boundaries(
    dataset_length: int,
    prediction_length: int,
    window_index: int,
    n_windows: int,
) -> Dict:
    """
    计算单个窗口的边界
    
    Args:
        dataset_length: 数据集总长度
        prediction_length: 预测长度
        window_index: 窗口索引（从 0 开始）
        n_windows: 总窗口数
        
    Returns:
        包含窗口边界信息的字典
    """
    split_point = dataset_length - prediction_length * n_windows
    
    context_start = max(0, split_point + window_index * prediction_length - MAX_CONTEXT_SUNDIAL)
    context_end = split_point + window_index * prediction_length
    forecast_start = context_end
    forecast_end = forecast_start + prediction_length
    
    return {
        "window_index": window_index,
        "context_start": context_start,
        "context_end": context_end,
        "forecast_start": forecast_start,
        "forecast_end": forecast_end,
        "context_length": context_end - context_start,
        "forecast_length": prediction_length,
    }


def generate_windows_with_missing(
    dataset_name: str,
    term: str,
    missing_ratio: float,
    clean_data_path: Optional[str] = None,
    data_path: str = "datasets",
    seed: int = 42,
) -> Tuple[List[Tuple[pd.DataFrame, Dict]], Dict]:
    """
    生成带缺失值的窗口数据
    
    Args:
        dataset_name: 数据集名称
        term: short/medium/long
        missing_ratio: 缺失比例
        clean_data_path: 干净数据路径（可选）
        data_path: 数据集目录
        seed: 随机种子
        
    Returns:
        (windows, global_meta)
    """
    props = load_dataset_properties(data_path)
    ds_props = props[dataset_name]
    freq = ds_props["frequency"]
    
    if clean_data_path is None:
        clean_data_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    else:
        clean_data_path = Path(clean_data_path)
    
    df_clean = pd.read_csv(clean_data_path)
    
    time_col = None
    for c in ['date', 'time', 'timestamp']:
        if c in df_clean.columns:
            time_col = c
            break
    
    if time_col:
        df_clean[time_col] = pd.to_datetime(df_clean[time_col])
        df_clean = df_clean.set_index(time_col)
    
    data_cols = list(df_clean.columns)
    dataset_length = len(df_clean)
    
    term_enum = Term(term)
    prediction_length = compute_prediction_length(freq, term_enum)
    
    w = math.ceil(TEST_SPLIT * dataset_length / prediction_length)
    n_windows = min(max(1, w), MAX_WINDOW)
    
    print(f"\n{'='*80}")
    print(f"生成带缺失值的窗口数据")
    print(f"{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"Term: {term}")
    print(f"频率: {freq}")
    print(f"数据集长度: {dataset_length}")
    print(f"预测长度: {prediction_length}")
    print(f"窗口数: {n_windows}")
    print(f"缺失比例: {missing_ratio}")
    print(f"{'='*80}")
    
    windows = []
    window_metas = []
    
    for i in range(n_windows):
        boundaries = compute_window_boundaries(
            dataset_length, prediction_length, i, n_windows
        )
        
        ctx_start = boundaries["context_start"]
        ctx_end = boundaries["context_end"]
        fc_start = boundaries["forecast_start"]
        fc_end = boundaries["forecast_end"]
        
        df_window = df_clean.iloc[ctx_start:fc_end].copy()
        
        if missing_ratio > 0:
            injection_range = {
                "start_index": ctx_start,
                "end_index": ctx_end,
                "data_path": data_path,
            }
            
            df_full, injection_info = inject_mcar(
                dataset_name=dataset_name,
                injection_range=injection_range,
                missing_ratio=missing_ratio,
                term=term,
                seed=seed + i,
            )
            
            time_col_temp = None
            for c in ['date', 'time', 'timestamp']:
                if c in df_full.columns:
                    time_col_temp = c
                    break
            
            if time_col_temp:
                df_full[time_col_temp] = pd.to_datetime(df_full[time_col_temp])
                df_full = df_full.set_index(time_col_temp)
            
            for col in data_cols:
                df_window.loc[df_window.index[:ctx_end - ctx_start], col] = df_full.iloc[ctx_start:ctx_end][col].values
            
            window_meta = {
                **boundaries,
                "injection_info": {
                    "total_cells": injection_info["total_cells"],
                    "injected_missing": injection_info["injected_missing"],
                    "actual_missing_ratio": injection_info["actual_missing_ratio"],
                    "seed": seed + i,
                },
            }
        else:
            window_meta = {
                **boundaries,
                "injection_info": {
                    "total_cells": 0,
                    "injected_missing": 0,
                    "actual_missing_ratio": 0.0,
                    "seed": seed + i,
                },
            }
        
        windows.append((df_window, window_meta))
        window_metas.append(window_meta)
        
        print(f"\n窗口 {i}:")
        print(f"  Context: [{ctx_start}:{ctx_end}) (长度={ctx_end - ctx_start})")
        print(f"  Forecast: [{fc_start}:{fc_end}) (长度={fc_end - fc_start})")
        if missing_ratio > 0:
            print(f"  注入缺失值: {window_meta['injection_info']['injected_missing']} / {window_meta['injection_info']['total_cells']} "
                  f"({window_meta['injection_info']['actual_missing_ratio']:.2%})")
    
    global_meta = {
        "dataset_name": dataset_name,
        "term": term,
        "frequency": freq,
        "missing_pattern": "MCAR",
        "missing_ratio": missing_ratio,
        "prediction_length": prediction_length,
        "n_windows": n_windows,
        "dataset_length": dataset_length,
        "data_cols": data_cols,
        "time_col": time_col,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
        "clean_data_path": str(clean_data_path),
        "data_start_time": str(df_clean.index[0]) if time_col else None,
    }
    
    print(f"\n{'='*80}")
    print(f"完成！共生成 {n_windows} 个窗口")
    print(f"{'='*80}")
    
    return windows, global_meta


def save_windows(
    windows: List[Tuple[pd.DataFrame, Dict]],
    global_meta: Dict,
    output_dir: str,
) -> Path:
    """
    保存窗口数据
    
    Args:
        windows: 窗口数据列表
        global_meta: 全局元数据
        output_dir: 输出目录
        
    Returns:
        输出目录路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存窗口到: {output_path}")
    
    for df_window, window_meta in windows:
        window_idx = window_meta["window_index"]
        filename = f"window_{window_idx:03d}.csv"
        filepath = output_path / filename
        df_window.to_csv(filepath)
        print(f"  保存: {filename}")
    
    meta_path = output_path / "meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(global_meta, f, indent=2, ensure_ascii=False)
    print(f"  保存: meta.json")
    
    return output_path


def run_inject_missing(
    dataset_name: str,
    term: str,
    missing_ratio: float = 0.0,
    clean_data_path: Optional[str] = None,
    data_path: str = "datasets",
    output_base_dir: str = "datasets/window_data",
    seed: int = 42,
) -> Path:
    """
    运行缺失值注入流程
    
    Args:
        dataset_name: 数据集名称
        term: short/medium/long
        missing_ratio: 缺失比例
        clean_data_path: 干净数据路径
        data_path: 数据集目录
        output_base_dir: 输出基础目录
        seed: 随机种子
        
    Returns:
        输出目录路径
    """
    windows, global_meta = generate_windows_with_missing(
        dataset_name=dataset_name,
        term=term,
        missing_ratio=missing_ratio,
        clean_data_path=clean_data_path,
        data_path=data_path,
        seed=seed,
    )
    
    ratio_str = f"{int(missing_ratio * 100):03d}"
    output_dir = Path(output_base_dir) / dataset_name / term / "MCAR" / ratio_str / "missing"
    
    save_windows(windows, global_meta, str(output_dir))
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="插入缺失值模块")
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集名称")
    parser.add_argument("--term", type=str, choices=["short", "medium", "long"], required=True,
                        help="预测 term")
    parser.add_argument("--missing_ratio", type=float, default=0.0,
                        help="缺失比例 (默认: 0.0)")
    parser.add_argument("--clean_data_path", type=str, default=None,
                        help="干净数据路径 (可选)")
    parser.add_argument("--data_path", type=str, default="datasets",
                        help="数据集目录 (默认: datasets)")
    parser.add_argument("--output_base_dir", type=str, default="datasets/window_data",
                        help="输出基础目录 (默认: datasets/window_data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    output_dir = run_inject_missing(
        dataset_name=args.dataset,
        term=args.term,
        missing_ratio=args.missing_ratio,
        clean_data_path=args.clean_data_path,
        data_path=args.data_path,
        output_base_dir=args.output_base_dir,
        seed=args.seed,
    )
    
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
