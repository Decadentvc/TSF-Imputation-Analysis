"""
缺失率检测工具

用于检测数据集中非时序列特征列（排除 item_id）的缺失率
支持全数据集检测或指定范围检测
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Tuple


def check_missing_ratio(
    dataset_path: str,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> Dict:
    """
    检测数据集中非时序列特征列（排除 item_id）的缺失率
    
    Args:
        dataset_path: 数据集 CSV 文件路径
        start_index: 检测起始索引（可选），为 None 时从开头开始
        end_index: 检测结束索引（可选），为 None 时到结尾结束
        
    Returns:
        dict: 缺失率检测结果，包含：
            - dataset_name: 数据集名称
            - total_rows: 总行数（检测范围内）
            - total_feature_cols: 非时序列特征列数量
            - total_cells: 总单元格数
            - missing_cells: 缺失单元格数
            - missing_ratio: 缺失率
            - per_column_stats: 每列的缺失统计
    """
    # 读取数据
    df = pd.read_csv(dataset_path)
    
    # 识别需要排除的列（时间列 + 标识列）
    excluded_cols = {'date', 'time', 'timestamp', 'item_id'}
    feature_cols = [col for col in df.columns if col.lower() not in excluded_cols]
    
    # 确定检测范围
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(df)
    
    # 确保索引有效
    start_index = max(0, start_index)
    end_index = min(len(df), end_index)
    
    if start_index >= end_index:
        raise ValueError(f"Invalid range: start_index={start_index} >= end_index={end_index}")
    
    # 提取检测范围的数据（只包含非时序列特征列）
    df_range = df.iloc[start_index:end_index][feature_cols]
    
    # 计算缺失统计
    total_rows = len(df_range)
    total_feature_cols = len(feature_cols)
    total_cells = total_rows * total_feature_cols
    
    # 计算每列的缺失情况
    per_column_stats = {}
    total_missing = 0
    
    for col in feature_cols:
        missing_count = df_range[col].isna().sum()
        per_column_stats[col] = {
            "missing_count": int(missing_count),
            "total_cells": total_rows,
            "missing_ratio": float(missing_count / total_rows) if total_rows > 0 else 0.0
        }
        total_missing += missing_count
    
    # 计算总体缺失率
    overall_missing_ratio = total_missing / total_cells if total_cells > 0 else 0.0
    
    # 构建结果
    result = {
        "dataset_name": Path(dataset_path).stem,
        "total_rows": total_rows,
        "total_feature_cols": total_feature_cols,
        "total_cells": total_cells,
        "missing_cells": int(total_missing),
        "missing_ratio": float(overall_missing_ratio),
        "per_column_stats": per_column_stats,
        "range": {
            "start_index": start_index,
            "end_index": end_index,
            "length": end_index - start_index
        }
    }
    
    return result


def print_missing_ratio_report(result: Dict) -> None:
    """
    打印缺失率检测报告
    
    Args:
        result: check_missing_ratio 返回的结果字典
    """
    print("=" * 70)
    print(f"缺失率检测报告 - {result['dataset_name']}")
    print("=" * 70)
    print(f"检测范围：行 {result['range']['start_index']} 到 {result['range']['end_index']} "
          f"(共 {result['range']['length']} 行)")
    print(f"非时序列特征列数：{result['total_feature_cols']}")
    print(f"总单元格数：{result['total_cells']}")
    print(f"缺失单元格数：{result['missing_cells']}")
    print(f"总体缺失率：{result['missing_ratio']:.4f} ({result['missing_ratio']*100:.2f}%)")
    print("-" * 70)
    print("各列缺失情况:")
    print(f"{'列名':<30} {'缺失数':>10} {'缺失率':>12}")
    print("-" * 70)
    
    for col, stats in result['per_column_stats'].items():
        print(f"{col:<30} {stats['missing_count']:>10} {stats['missing_ratio']*100:>11.2f}%")
    
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="检测数据集中非时序列特征列的缺失率"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="数据集 CSV 文件路径"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="检测起始索引（可选，不指定则从开头开始）"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="检测结束索引（可选，不指定则到结尾结束）"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只输出 JSON 结果，不打印详细报告"
    )
    
    args = parser.parse_args()
    
    # 执行检测
    result = check_missing_ratio(
        dataset_path=args.dataset_path,
        start_index=args.start_index,
        end_index=args.end_index
    )
    
    # 输出结果
    if args.quiet:
        import json
        print(json.dumps(result, indent=2))
    else:
        print_missing_ratio_report(result)
