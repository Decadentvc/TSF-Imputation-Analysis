"""
MCAR（完全随机缺失）缺失值注入模式

在注错区间内按照给定比例完全随机注入缺失值
支持批量缺失率和自动 term 检测
"""

import os
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List

from inject_range_utils import (
    get_injection_range,
    load_dataset_properties,
    Term,
)


# ============================================================================
# 辅助函数
# ============================================================================

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


def parse_missing_ratios(ratio_str: str) -> List[float]:
    """
    解析缺失比例字符串
    
    Args:
        ratio_str: 字符串格式，如 "0.05,0.1,0.15" 或 "[0.05,0.1,0.15]"
        
    Returns:
        浮点数列表
    """
    # 去除方括号（如果有）
    ratio_str = ratio_str.strip().strip('[]')
    
    # 分割并转换为浮点数
    ratios = [float(r.strip()) for r in ratio_str.split(',')]
    
    # 验证
    for ratio in ratios:
        if not 0 <= ratio <= 1:
            raise ValueError(f"missing_ratio must be between 0 and 1, got {ratio}")
    
    return ratios


# ============================================================================
# MCAR 缺失值注入
# ============================================================================

def inject_mcar(
    dataset_name: str,
    missing_ratio: Union[float, List[float]],
    term: Optional[Union[str, Term]] = None,
    data_path: str = "datasets",
    output_path: Optional[str] = None,
    seed: int = 42,
    auto_term: bool = True,
) -> List[dict]:
    """
    在注错区间内按照给定比例随机注入缺失值（MCAR 机制）
    
    Args:
        dataset_name: 数据集名称
        missing_ratio: 缺失比例，可以是单个值（0.05）或列表（[0.05, 0.1, 0.15]）
        term: 预测 horizon 类型，如果 auto_term=True 则自动检测
        data_path: 原始数据集目录
        output_path: 输出目录（默认：datasets/MCAR/MCAR_{ratio}）
        seed: 随机种子
        auto_term: 是否自动根据数据集配置选择 term（默认 True）
        
    Returns:
        包含注入信息的字典列表
    """
    results = []
    
    # 1. 处理 missing_ratio 为列表
    if isinstance(missing_ratio, float):
        missing_ratios = [missing_ratio]
    else:
        missing_ratios = missing_ratio
    
    # 2. 确定 term 列表
    if auto_term:
        terms = get_available_terms(dataset_name, data_path)
        print(f"自动检测到数据集 '{dataset_name}' 的 term 配置：{terms}")
    else:
        if term is None:
            term = "short"
        terms = [term.value if isinstance(term, Term) else term]
    
    # 3. 对每个 term 和 missing_ratio 组合进行注入
    for t in terms:
        for ratio in missing_ratios:
            print(f"\n{'='*80}")
            print(f"开始注入：term={t}, missing_ratio={ratio:.2%}")
            print(f"{'='*80}\n")
            
            result = _inject_single_mcar(
                dataset_name=dataset_name,
                missing_ratio=ratio,
                term=t,
                data_path=data_path,
                output_path=output_path,
                seed=seed,
            )
            results.append(result)
            print_injection_info(result)
    
    return results


def _inject_single_mcar(
    dataset_name: str,
    missing_ratio: float,
    term: str,
    data_path: str,
    output_path: Optional[str],
    seed: int,
) -> dict:
    """
    执行单次 MCAR 注入
    
    Args:
        dataset_name: 数据集名称
        missing_ratio: 缺失比例
        term: term 类型
        data_path: 数据目录
        output_path: 输出目录
        seed: 随机种子
        
    Returns:
        注入信息字典
    """
    # 1. 获取注错区间
    range_info = get_injection_range(dataset_name, term, data_path)
    start_idx = range_info["start_index"]
    end_idx = range_info["end_index"]
    
    # 2. 读取原始数据
    csv_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    
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
    
    # 8. 确定输出路径
    ratio_str = f"{int(missing_ratio * 100):03d}"
    if output_path is None:
        output_dir = Path(data_path) / "MCAR" / f"MCAR_{ratio_str}"
    else:
        output_dir = Path(output_path)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 9. 保存新数据集（文件名包含 term 信息）
    if len(get_available_terms(dataset_name, data_path)) > 1:
        # 多个 term 时，文件名包含 term
        output_filename = f"{dataset_name}_MCAR_{ratio_str}_{term}.csv"
    else:
        # 单个 term 时，文件名不包含 term
        output_filename = f"{dataset_name}_MCAR_{ratio_str}.csv"
    
    output_path = output_dir / output_filename
    df.to_csv(output_path, index=False)
    
    # 10. 返回注入信息
    return {
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
        "output_path": str(output_path),
        "original_path": str(csv_path),
    }


def print_injection_info(info: dict) -> None:
    """打印缺失值注入信息"""
    print("=" * 80)
    print(f"MCAR 缺失值注入 - {info['dataset_name']} ({info['term']})")
    print("=" * 80)
    print(f"注入参数:")
    print(f"  缺失比例：{info['missing_ratio']:.2%}")
    print(f"  注错区间：[{info['injection_range']['start_index']}, {info['injection_range']['end_index']})")
    print(f"  注错区间长度：{info['injection_range']['length']}")
    print(f"\n注入结果:")
    print(f"  总单元格数：{info['total_cells']}")
    print(f"  注入缺失值数：{info['injected_missing']}")
    print(f"  实际缺失比例：{info['actual_missing_ratio']:.2%}")
    print(f"\n文件保存:")
    print(f"  原始文件：{info['original_path']}")
    print(f"  注入后文件：{info['output_path']}")
    print("=" * 80)


# ============================================================================
# 命令行工具
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="在注错区间内注入 MCAR 缺失值"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称（如 ETTh1, national_illness）"
    )
    parser.add_argument(
        "--missing_ratio",
        type=str,
        required=True,
        help="缺失比例，支持单个值或逗号分隔列表（如 0.05 或 0.05,0.1,0.15）"
    )
    parser.add_argument(
        "--term",
        type=str,
        default=None,
        choices=["short", "medium", "long"],
        help="预测 horizon 类型（默认：自动根据数据集配置）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets",
        help="数据集目录路径（默认：datasets）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）"
    )
    parser.add_argument(
        "--no_auto_term",
        action="store_true",
        help="禁用自动 term 检测，使用 --term 指定的值"
    )
    
    args = parser.parse_args()
    
    # 解析缺失比例
    missing_ratios = parse_missing_ratios(args.missing_ratio)
    
    if len(missing_ratios) == 1:
        print(f"缺失比例：{missing_ratios[0]:.2%}")
    else:
        print(f"缺失比例列表：{missing_ratios}")
    
    # 注入缺失值
    results = inject_mcar(
        dataset_name=args.dataset,
        missing_ratio=missing_ratios,
        term=args.term,
        data_path=args.data_path,
        seed=args.seed,
        auto_term=not args.no_auto_term,
    )
    
    # 打印汇总
    print(f"\n{'='*80}")
    print(f"批量注入完成！共生成 {len(results)} 个文件:")
    print(f"{'='*80}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['output_path']}")
