"""
BM（块缺失）缺失值注入模块

在注错区间内按照块缺失模式注入缺失值
特点：
1. 缺失值以连续块的形式出现
2. 每个块的长度固定（默认 50）
3. 块与块之间不重叠、不相邻
4. 只对非时序列注入缺失值
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List


def inject_bm(
    dataset_name: str,
    injection_range: dict,
    missing_ratio: float,
    term: str,
    block_length: int = 50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, dict]:
    """
    在注错区间内按照块缺失模式注入缺失值
    
    Args:
        dataset_name: 数据集名称
        injection_range: 注错区间信息字典（来自 inject_range_utils.get_injection_range）
            必须包含：start_index, end_index
        missing_ratio: 缺失比例（0.0-1.0）
        term: term 类型（"short", "medium", "long"）
        block_length: 单个缺失块的长度（默认：50）
        seed: 随机种子
        
    Returns:
        Tuple[pd.DataFrame, dict]: 
            - DataFrame: 注入缺失值后的数据集
            - dict: 注入信息字典，包含：
                - dataset_name: 数据集名称
                - term: term 类型
                - missing_ratio: 缺失比例
                - block_length: 块长度
                - n_blocks: 块数量
                - total_cells: 注错区间总单元格数
                - injected_missing: 注入的缺失值数量
                - actual_missing_ratio: 实际缺失比例
                - injection_range: 注错区间信息
                - block_positions: 各块的位置信息
    """
    # 1. 读取原始数据
    data_path = injection_range.get("data_path", "datasets")
    csv_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)
    
    # 2. 获取注错区间
    start_idx = injection_range["start_index"]
    end_idx = injection_range["end_index"]
    injection_length = end_idx - start_idx
    
    # 3. 识别时间列和非时序列列
    time_cols = ['date', 'time', 'timestamp']
    data_cols = [col for col in df.columns if col not in time_cols]
    
    # 4. 计算块数量
    # 总缺失数 = 数据集长度 × 缺失率 × 列数
    # 块数 = 总缺失数 / 块长度，向下取整
    total_missing = int(injection_length * len(data_cols) * missing_ratio)
    n_blocks = total_missing // block_length
    
    # 5. 设置随机种子
    seed_offset = hash(f"{dataset_name}_{term}_{missing_ratio}_BM") % 10000
    np.random.seed(seed + seed_offset)
    
    # 6. 对每一列注入块缺失
    block_positions = []
    injected_count = 0
    
    for col in data_cols:
        # 计算该列的块数量（按比例分配）
        col_n_blocks = n_blocks // len(data_cols)
        
        if col_n_blocks == 0:
            continue
        
        # 生成不重叠的块起始位置
        block_starts = generate_non_overlapping_blocks(
            start_idx=start_idx,
            end_idx=end_idx,
            block_length=block_length,
            n_blocks=col_n_blocks,
        )
        
        # 注入缺失值
        for block_start in block_starts:
            block_end = min(block_start + block_length, end_idx)
            df.loc[block_start:block_end-1, col] = np.nan
            injected_count += (block_end - block_start)
            
            block_positions.append({
                "column": col,
                "start": block_start,
                "end": block_end,
                "length": block_end - block_start,
            })
    
    # 7. 计算实际缺失比例
    injection_area_size = injection_length * len(data_cols)
    actual_missing_ratio = injected_count / injection_area_size if injection_area_size > 0 else 0
    
    # 8. 构建返回信息
    info = {
        "dataset_name": dataset_name,
        "term": term,
        "missing_ratio": missing_ratio,
        "block_length": block_length,
        "n_blocks": n_blocks,
        "total_cells": injection_area_size,
        "injected_missing": injected_count,
        "actual_missing_ratio": actual_missing_ratio,
        "injection_range": {
            "start_index": start_idx,
            "end_index": end_idx,
            "length": injection_length,
        },
        "block_positions": block_positions,
        "data_columns": data_cols,
    }
    
    return df, info


def generate_non_overlapping_blocks(
    start_idx: int,
    end_idx: int,
    block_length: int,
    n_blocks: int,
    min_gap: int = 1,
) -> List[int]:
    """
    生成不重叠、不相邻的块起始位置
    
    Args:
        start_idx: 起始索引
        end_idx: 结束索引
        block_length: 块长度
        n_blocks: 块数量
        min_gap: 块之间的最小间隔（默认：1，即不相邻）
        
    Returns:
        块起始位置列表
        
    Raises:
        ValueError: 如果无法在给定区间内放置指定数量的块
    """
    available_length = end_idx - start_idx
    required_length = n_blocks * (block_length + min_gap) - min_gap
    
    if required_length > available_length:
        raise ValueError(
            f"Cannot place {n_blocks} blocks of length {block_length} "
            f"in interval [{start_idx}, {end_idx}). "
            f"Required length: {required_length}, available: {available_length}"
        )
    
    # 使用贪心算法生成块位置
    block_starts = []
    current_pos = start_idx
    
    for i in range(n_blocks):
        # 计算剩余可用空间
        remaining_blocks = n_blocks - i
        remaining_space = end_idx - current_pos - remaining_blocks * (block_length + min_gap) + min_gap
        
        # 随机选择下一个块的起始位置
        if remaining_space > 0:
            offset = np.random.randint(0, remaining_space + 1)
            block_start = current_pos + offset
        else:
            block_start = current_pos
        
        block_starts.append(block_start)
        
        # 更新下一个块的起始位置（确保不重叠、不相邻）
        current_pos = block_start + block_length + min_gap
    
    return block_starts


def get_available_terms(dataset_name: str, data_path: str = "datasets") -> List[str]:
    """
    根据数据集配置获取可用的 term 列表
    
    Args:
        dataset_name: 数据集名称
        data_path: 数据集目录
        
    Returns:
        term 列表，如 ["short"] 或 ["short", "medium", "long"]
    """
    from inject_range_utils import load_dataset_properties
    
    props = load_dataset_properties(data_path)
    if dataset_name not in props:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    
    ds_term = props[dataset_name].get("term", "med_long")
    
    if ds_term == "short":
        return ["short"]
    else:  # med_long
        return ["short", "medium", "long"]


if __name__ == "__main__":
    import argparse
    from inject_range_utils import get_injection_range
    
    parser = argparse.ArgumentParser(description="BM（块缺失）缺失值注入")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--term", type=str, default="short", help="term 类型")
    parser.add_argument("--missing_ratio", type=float, required=True, help="缺失比例")
    parser.add_argument("--block_length", type=int, default=50, help="块长度（默认：50）")
    parser.add_argument("--data_path", type=str, default="datasets", help="数据集目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 获取注错区间
    injection_range = get_injection_range(args.dataset, args.term, args.data_path)
    injection_range["data_path"] = args.data_path
    
    # 注入缺失值
    df, info = inject_bm(
        dataset_name=args.dataset,
        injection_range=injection_range,
        missing_ratio=args.missing_ratio,
        term=args.term,
        block_length=args.block_length,
        seed=args.seed,
    )
    
    # 打印信息
    print(f"\n{'='*80}")
    print(f"BM 缺失值注入完成")
    print(f"{'='*80}")
    print(f"数据集: {info['dataset_name']}")
    print(f"Term: {info['term']}")
    print(f"缺失比例: {info['missing_ratio']:.2%}")
    print(f"块长度: {info['block_length']}")
    print(f"块数量: {info['n_blocks']}")
    print(f"总单元格数: {info['total_cells']}")
    print(f"注入缺失值数: {info['injected_missing']}")
    print(f"实际缺失比例: {info['actual_missing_ratio']:.2%}")
    print(f"{'='*80}\n")
