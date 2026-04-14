"""BM（块缺失）缺失值注入模块。

包含两层能力：
1) 底层 API: ``inject_bm``
2) 命令行入口: 支持单/多缺失率、自动 term 检测、批量保存
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    from .inject_range_utils import get_injection_range, load_dataset_properties
except ImportError:
    from inject_range_utils import get_injection_range, load_dataset_properties


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
    
    # 3. 识别不注空列（时间列 + item_id）和可注空列
    excluded_cols = {'date', 'time', 'timestamp', 'item_id'}
    data_cols = [col for col in df.columns if col.lower() not in excluded_cols]
    
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


def parse_missing_ratios(ratio_str: str) -> List[float]:
    """解析缺失比例字符串。"""
    ratio_str = ratio_str.strip().strip("[]")
    ratios = [float(r.strip()) for r in ratio_str.split(",") if r.strip()]

    if not ratios:
        raise ValueError("missing_ratio list is empty")

    for ratio in ratios:
        if not 0 <= ratio <= 1:
            raise ValueError(f"missing_ratio must be between 0 and 1, got {ratio}")

    return ratios


def save_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    missing_ratio: float,
    term: str,
    output_base_dir: str = "data/datasets",
    block_length: int = 50,
) -> str:
    """保存 BM 注入后的数据集。"""
    ratio_str = f"{int(missing_ratio * 100):03d}"
    output_dir = Path(output_base_dir) / "BM" / f"BM_{ratio_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = f"{dataset_name}_BM_length{block_length}_{ratio_str}_{term}.csv"
    output_path = output_dir / output_filename
    df.to_csv(output_path, index=False)
    return str(output_path)


def run_bm_injection(
    dataset_name: str,
    missing_ratios: List[float],
    terms: List[str],
    data_path: str = "data/datasets",
    output_base_dir: str = "data/datasets",
    block_length: int = 50,
    max_context: int = 8192,
    seed: int = 42,
) -> List[dict]:
    """批量运行 BM（块缺失）注入。"""
    data_path = str(Path(data_path))
    results = []
    for term in terms:
        print(f"\n获取数据集 '{dataset_name}' ({term}) 的注错区间...")
        injection_range = get_injection_range(
            dataset_name=dataset_name,
            term=term,
            data_path=data_path,
            max_context=max_context,
        )
        injection_range["data_path"] = data_path

        print(f"  注错区间：[{injection_range['start_index']}, {injection_range['end_index']})")
        print(f"  注错区间长度：{injection_range['end_index'] - injection_range['start_index']}")

        for missing_ratio in missing_ratios:
            print(f"\n{'=' * 80}")
            print(f"注入：pattern=BM, term={term}, missing_ratio={missing_ratio:.2%}")
            print(f"  块长度：{block_length}")
            print(f"{'=' * 80}")

            df_injected, info = inject_bm(
                dataset_name=dataset_name,
                injection_range=injection_range,
                missing_ratio=missing_ratio,
                term=term,
                block_length=block_length,
                seed=seed,
            )

            output_path = save_dataset(
                df=df_injected,
                dataset_name=dataset_name,
                missing_ratio=missing_ratio,
                term=term,
                output_base_dir=output_base_dir,
                block_length=block_length,
            )
            info["output_path"] = output_path
            info["original_path"] = str(Path(data_path) / "ori" / f"{dataset_name}.csv")
            info["pattern"] = "BM"

            print("\n注入结果:")
            print(f"  总单元格数：{info['total_cells']}")
            print(f"  块长度：{info['block_length']}")
            print(f"  块数量：{info['n_blocks']}")
            print(f"  注入缺失值数：{info['injected_missing']}")
            print(f"  实际缺失比例：{info['actual_missing_ratio']:.2%}")
            print("\n文件保存:")
            print(f"  {output_path}")
            print("=" * 80)

            results.append(info)

    return results


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
    props = load_dataset_properties(data_path)
    if dataset_name not in props:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    
    ds_term = props[dataset_name].get("term", "med_long")
    
    if ds_term == "short":
        return ["short"]
    else:  # med_long
        return ["short", "medium", "long"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM（块缺失）缺失值注入")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称（如 ETTh1）")
    parser.add_argument(
        "--missing_ratio",
        type=str,
        required=True,
        help="缺失比例，支持单个值或逗号分隔列表（如 0.05 或 0.05,0.1）",
    )
    parser.add_argument(
        "--term",
        type=str,
        default=None,
        choices=["short", "medium", "long"],
        help="预测 horizon 类型（默认：自动根据数据集配置）",
    )
    parser.add_argument("--data_path", type=str, default="data/datasets", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="data/datasets", help="输出目录")
    parser.add_argument("--block_length", type=int, default=50, help="块长度（默认：50）")
    parser.add_argument(
        "--max_context",
        type=int,
        default=8192,
        help="最大回顾窗口长度（默认：8192）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认：42）")
    parser.add_argument(
        "--no_auto_term",
        action="store_true",
        help="禁用自动 term 检测，使用 --term 指定值",
    )

    args = parser.parse_args()

    if args.data_path == "datasets":
        default_data_path = Path(__file__).resolve().parents[2] / "data" / "datasets"
        args.data_path = str(default_data_path)

    missing_ratios = parse_missing_ratios(args.missing_ratio)
    if len(missing_ratios) == 1:
        print(f"缺失比例：{missing_ratios[0]:.2%}")
    else:
        print(f"缺失比例列表：{missing_ratios}")

    if args.no_auto_term:
        terms = [args.term] if args.term else ["short"]
        print(f"使用指定的 term: {terms}")
    else:
        terms = get_available_terms(args.dataset, args.data_path)
        print(f"自动检测到数据集 '{args.dataset}' 的 term 配置：{terms}")

    results = run_bm_injection(
        dataset_name=args.dataset,
        missing_ratios=missing_ratios,
        terms=terms,
        data_path=args.data_path,
        output_base_dir=args.output_dir,
        block_length=args.block_length,
        max_context=args.max_context,
        seed=args.seed,
    )

    print(f"\n{'=' * 80}")
    print(f"批量注入完成！共生成 {len(results)} 个文件:")
    print(f"{'=' * 80}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['output_path']}")
    print(f"{'=' * 80}\n")
