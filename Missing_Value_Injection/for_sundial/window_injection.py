"""
窗口级缺失值注入工具

基于现有的 inject_mcar() 函数，为每个预测窗口注入缺失值
确保与原有注错方法使用相同的底层函数
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict

from inject_range_utils import (
    load_dataset_properties,
    Term,
    compute_prediction_length,
)
from MCAR import inject_mcar


def compute_window_boundaries(
    dataset_length: int,
    prediction_length: int,
    window_index: int,
    max_context: int = 2880,
) -> Tuple[int, int, int, int]:
    """
    计算单个窗口的边界
    
    Args:
        dataset_length: 数据集总长度
        prediction_length: 预测长度
        window_index: 窗口索引（从 0 开始）
        max_context: 最大回顾窗口（Sundial 固定为 2880）
        
    Returns:
        (context_start, context_end, forecast_start, forecast_end)
    """
    forecast_end = dataset_length - (window_index * prediction_length)
    forecast_start = forecast_end - prediction_length
    context_end = forecast_start
    context_start = max(0, forecast_start - max_context)
    
    return context_start, context_end, forecast_start, forecast_end


def get_window_injection_range(
    dataset_name: str,
    term: str,
    window_index: int,
    data_path: str = "datasets",
) -> Dict:
    """
    获取单个窗口的注错区间（使用现有注错区间工具）
    
    注错区间：[context_start, forecast_start)
    只在上下文窗口注入缺失值，不在预测区间注入
    
    Args:
        dataset_name: 数据集名称
        term: short/medium/long
        window_index: 窗口索引
        data_path: 数据集路径
        
    Returns:
        注错区间信息字典
    """
    props = load_dataset_properties(data_path)
    ds_props = props[dataset_name]
    freq = ds_props["frequency"]
    
    csv_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    dataset_length = len(df)
    
    term_enum = Term(term)
    prediction_length = compute_prediction_length(freq, term_enum)
    
    ctx_start, ctx_end, fc_start, fc_end = compute_window_boundaries(
        dataset_length, prediction_length, window_index
    )
    
    return {
        "dataset_name": dataset_name,
        "term": term,
        "window_index": window_index,
        "start_index": ctx_start,
        "end_index": fc_start,
        "context_start": ctx_start,
        "context_end": ctx_end,
        "forecast_start": fc_start,
        "forecast_end": fc_end,
        "data_path": data_path,
    }


def inject_missing_to_window(
    dataset_name: str,
    term: str,
    window_index: int,
    missing_pattern: str,
    missing_ratio: float,
    seed: int = 42,
    data_path: str = "datasets",
    save_missing_file: bool = False,
    output_base_dir: str = "datasets/window_imputation",
) -> Tuple[pd.DataFrame, Dict]:
    """
    对单个窗口注入缺失值（使用现有的 inject_mcar() 函数）
    
    Args:
        dataset_name: 数据集名称
        term: short/medium/long
        window_index: 窗口索引
        missing_pattern: 缺失模式（MCAR, BM, etc.）
        missing_ratio: 缺失比例
        seed: 随机种子
        data_path: 数据集路径
        save_missing_file: 是否保存缺失文件（注入缺失值后但未填补）
        output_base_dir: 缺失文件输出基础目录
        
    Returns:
        (df_with_missing, injection_info)
    """
    injection_range = get_window_injection_range(
        dataset_name, term, window_index, data_path
    )
    
    df_with_missing, info = inject_mcar(
        dataset_name=dataset_name,
        injection_range=injection_range,
        missing_ratio=missing_ratio,
        term=term,
        seed=seed + window_index,
    )
    
    ctx_start = injection_range["context_start"]
    ctx_end = injection_range["context_end"]
    fc_start = injection_range["forecast_start"]
    fc_end = injection_range["forecast_end"]
    
    # 只保留历史部分（context），不包含预测部分（forecast）
    # 这样设计的好处：
    # 1. 避免保存冗余数据（预测部分没有缺失值，不需要填补）
    # 2. 当预测长度变化时，历史部分可以重复使用，无需重新生成
    # 3. 评估时从干净数据集中提取对应窗口的预测部分作为标签
    df_window = df_with_missing.iloc[ctx_start:fc_start].copy()
    
    simplified_info = {
        "dataset_name": dataset_name,
        "term": term,
        "window_index": window_index,
        "missing_pattern": missing_pattern,
        "missing_ratio": missing_ratio,
        "context_start": ctx_start,
        "context_end": ctx_end,
        "forecast_start": fc_start,
        "forecast_end": fc_end,
        "injected_missing": info["injected_missing"],
        "total_cells": info["total_cells"],
        "seed": seed + window_index,
    }
    
    # 保存缺失文件（如果需要）
    if save_missing_file:
        missing_file_path = save_missing_window(
            df_window=df_window,
            dataset_name=dataset_name,
            term=term,
            window_index=window_index,
            missing_pattern=missing_pattern,
            missing_ratio=missing_ratio,
            seed=seed + window_index,
            output_base_dir=output_base_dir,
        )
        simplified_info["missing_file_path"] = str(missing_file_path)
    
    return df_window, simplified_info


def save_missing_window(
    df_window: pd.DataFrame,
    dataset_name: str,
    term: str,
    window_index: int,
    missing_pattern: str,
    missing_ratio: float,
    seed: int,
    output_base_dir: str = "datasets/window_imputation",
) -> Path:
    """
    保存注入缺失值后的窗口数据（未填补）
    
    Args:
        df_window: 注入缺失值后的窗口数据
        dataset_name: 数据集名称
        term: short/medium/long
        window_index: 窗口索引
        missing_pattern: 缺失模式
        missing_ratio: 缺失比例
        seed: 随机种子
        output_base_dir: 输出基础目录
        
    Returns:
        保存的文件路径
    """
    ratio_str = f"{int(missing_ratio * 100):03d}"
    output_dir = (
        Path(output_base_dir)
        / dataset_name
        / term
        / missing_pattern
        / ratio_str
        / "_missing_files"
        / f"window_{window_index:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存缺失数据
    missing_file_path = output_dir / "missing_data.csv"
    df_window.to_csv(missing_file_path, index=False)
    
    # 保存元数据
    meta = {
        "dataset_name": dataset_name,
        "term": term,
        "window_index": window_index,
        "missing_pattern": missing_pattern,
        "missing_ratio": missing_ratio,
        "seed": seed,
        "file_type": "missing_data",
        "generated_at": datetime.now().isoformat(),
    }
    
    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] Missing file saved: {missing_file_path}")
    
    return missing_file_path


# 添加 datetime 和 json 导入
from datetime import datetime
import json
