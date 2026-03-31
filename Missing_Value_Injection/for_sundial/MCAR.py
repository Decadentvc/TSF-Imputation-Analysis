"""
MCAR（完全随机缺失）缺失值注入模式

在注错区间内按照给定比例完全随机注入缺失值
纯函数库版本，供 MI_all.py 调用
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Tuple

from inject_range_utils import (
    get_injection_range,
    load_dataset_properties,
    Term,
)


def get_available_terms(dataset_name: str, data_path: str = "datasets") -> List[str]:
    """
    根据数据集配置获取可用的 term 列表
    
    Args:
        dataset_name: 数据集名称
        data_path: 数据集目录
        
    Returns:
        term 列表，如 ["short"] 或 ["short", "medium", "long"]
    """
    props = load_dataset_properties(data_path)
    if dataset_name not in props:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    
    ds_term = props[dataset_name].get("term", "med_long")
    
    if ds_term == "short":
        return ["short"]
    else:  # med_long
        return ["short", "medium", "long"]


def inject_mcar(
    dataset_name: str,
    injection_range: dict,
    missing_ratio: float,
    term: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    在注错区间内按照给定比例随机注入缺失值（MCAR 机制）
    
    Args:
        dataset_name: 数据集名称
        injection_range: 注错区间信息字典（来自 inject_range_utils.get_injection_range）
            必须包含：start_index, end_index
        missing_ratio: 缺失比例（0.0-1.0）
        term: term 类型（"short", "medium", "long"）
        seed: 随机种子
        
    Returns:
        Tuple[pd.DataFrame, dict]: 
            - DataFrame: 注入缺失值后的数据集
            - dict: 注入信息字典，包含：
                - dataset_name: 数据集名称
                - term: term 类型
                - missing_ratio: 缺失比例
                - total_cells: 注错区间总单元格数
                - injected_missing: 注入的缺失值数量
                - actual_missing_ratio: 实际缺失比例
                - injection_range: 注错区间信息
    """
    # 1. 读取原始数据
    data_path = injection_range.get("data_path", "datasets")
    csv_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    
    # 2. 获取注错区间
    start_idx = injection_range["start_index"]
    end_idx = injection_range["end_index"]
    
    # 3. 识别时间列和非时序列列
    time_cols = ['date', 'time', 'timestamp']
    data_cols = [col for col in df.columns if col not in time_cols]
    
    # 4. 计算需要注入的缺失值数量（四舍五入）
    injection_area_size = (end_idx - start_idx) * len(data_cols)
    n_missing = round(injection_area_size * missing_ratio)
    
    # 5. 设置随机种子（每个组合使用不同的种子）
    seed_offset = hash(f"{dataset_name}_{term}_{missing_ratio}") % 10000
    np.random.seed(seed + seed_offset)
    
    # 6. 使用补偿机制注入缺失值
    # 先生成随机位置，如果有重复或遇到已有缺失值，则补偿注入
    missing_count = 0
    attempts = 0
    max_attempts = n_missing * 3  # 保守的补偿：最多尝试 3 倍次数
    
    while missing_count < n_missing and attempts < max_attempts:
        row_idx = np.random.randint(start_idx, end_idx)
        col_name = np.random.choice(data_cols)
        
        if not pd.isna(df.loc[row_idx, col_name]):
            df.loc[row_idx, col_name] = np.nan
            missing_count += 1
        
        attempts += 1
    
    # 7. 构建返回信息
    info = {
        "dataset_name": dataset_name,
        "term": term,
        "missing_ratio": missing_ratio,
        "total_cells": injection_area_size,
        "injected_missing": missing_count,
        "actual_missing_ratio": missing_count / injection_area_size if injection_area_size > 0 else 0,
        "injection_range": {
            "start_index": start_idx,
            "end_index": end_idx,
            "length": end_idx - start_idx,
        },
    }
    
    return df, info
