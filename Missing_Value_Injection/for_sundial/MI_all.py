"""
缺失值注入统一入口脚本

汇总调用所有具体的注错模式(MCAR, BM 等)
负责：
1. 解析命令行参数
2. 获取注错区间
3. 调用具体注错模式
4. 保存结果
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from inject_range_utils import (
    get_injection_range,
    load_dataset_properties,
    Term,
)
from MCAR import inject_mcar, get_available_terms
from BM import inject_bm


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


def save_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    missing_ratio: float,
    term: str,
    pattern: str = "MCAR",
    output_base_dir: str = "datasets/",
    block_length: int = None,
) -> str:
    """
    保存注入缺失值后的数据集
    
    Args:
        df: DataFrame
        dataset_name: 数据集名称
        missing_ratio: 缺失比例
        term: term 类型
        pattern: 缺失模式（MCAR, BM, MAR, MNAR 等）
        output_base_dir: 输出基础目录
        block_length: 块长度（仅用于 BM 模式）
        
    Returns:
        保存的文件路径
    """
    ratio_str = f"{int(missing_ratio * 100):03d}"
    output_dir = Path(output_base_dir) / f"{pattern}"/ f"{pattern}_{ratio_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文件名始终包含 pattern、ratio 和 term
    # 对于 BM 模式，添加块长度信息
    if pattern == "BM" and block_length is not None:
        output_filename = f"{dataset_name}_{pattern}_length{block_length}_{ratio_str}_{term}.csv"
    else:
        output_filename = f"{dataset_name}_{pattern}_{ratio_str}_{term}.csv"
    
    output_path = output_dir / output_filename
    df.to_csv(output_path, index=False)
    
    return str(output_path)


def run_mcar_injection(
    dataset_name: str,
    missing_ratios: List[float],
    terms: List[str],
    data_path: str = "datasets",
    output_base_dir: str = "datasets/MI",
    seed: int = 42,
) -> List[dict]:
    """
    运行 MCAR 缺失值注入
    
    Args:
        dataset_name: 数据集名称
        missing_ratios: 缺失比例列表
        terms: term 列表
        data_path: 数据集目录
        output_base_dir: 输出基础目录
        seed: 随机种子
        
    Returns:
        注入信息列表
    """
    results = []
    pattern = "MCAR"
    
    for term in terms:
        # 1. 获取注错区间
        print(f"\n获取数据集 '{dataset_name}' ({term}) 的注错区间...")
        injection_range = get_injection_range(dataset_name, term, data_path)
        injection_range["data_path"] = data_path
        
        print(f"  注错区间：[{injection_range['start_index']}, {injection_range['end_index']})")
        print(f"  注错区间长度：{injection_range['end_index'] - injection_range['start_index']}")
        
        for missing_ratio in missing_ratios:
            print(f"\n{'='*80}")
            print(f"注入：pattern={pattern}, term={term}, missing_ratio={missing_ratio:.2%}")
            print(f"{'='*80}")
            
            # 2. 调用 MCAR 注入
            df_injected, info = inject_mcar(
                dataset_name=dataset_name,
                injection_range=injection_range,
                missing_ratio=missing_ratio,
                term=term,
                seed=seed,
            )
            
            # 3. 保存结果
            output_path = save_dataset(
                df=df_injected,
                dataset_name=dataset_name,
                missing_ratio=missing_ratio,
                term=term,
                pattern=pattern,
                output_base_dir=output_base_dir,
                block_length=block_length,
            )
            info["output_path"] = output_path
            info["original_path"] = str(Path(data_path) / "ori" / f"{dataset_name}.csv")
            info["pattern"] = pattern
            
            # 4. 打印信息
            print(f"\n注入结果:")
            print(f"  总单元格数：{info['total_cells']}")
            print(f"  注入缺失值数：{info['injected_missing']}")
            print(f"  实际缺失比例：{info['actual_missing_ratio']:.2%}")
            print(f"\n文件保存:")
            print(f"  {output_path}")
            print("="*80)
            
            results.append(info)
    
    return results


def run_bm_injection(
    dataset_name: str,
    missing_ratios: List[float],
    terms: List[str],
    data_path: str = "datasets",
    output_base_dir: str = "datasets",
    block_length: int = 50,
    seed: int = 42,
) -> List[dict]:
    """
    运行 BM（块缺失）缺失值注入
    
    Args:
        dataset_name: 数据集名称
        missing_ratios: 缺失比例列表
        terms: term 列表
        data_path: 数据集目录
        output_base_dir: 输出基础目录
        block_length: 块长度（默认：50）
        seed: 随机种子
        
    Returns:
        注入信息列表
    """
    results = []
    pattern = "BM"
    
    for term in terms:
        # 1. 获取注错区间
        print(f"\n获取数据集 '{dataset_name}' ({term}) 的注错区间...")
        injection_range = get_injection_range(dataset_name, term, data_path)
        injection_range["data_path"] = data_path
        
        print(f"  注错区间：[{injection_range['start_index']}, {injection_range['end_index']})")
        print(f"  注错区间长度：{injection_range['end_index'] - injection_range['start_index']}")
        
        for missing_ratio in missing_ratios:
            print(f"\n{'='*80}")
            print(f"注入：pattern={pattern}, term={term}, missing_ratio={missing_ratio:.2%}")
            print(f"  块长度：{block_length}")
            print(f"{'='*80}")
            
            # 2. 调用 BM 注入
            df_injected, info = inject_bm(
                dataset_name=dataset_name,
                injection_range=injection_range,
                missing_ratio=missing_ratio,
                term=term,
                block_length=block_length,
                seed=seed,
            )
            
            # 3. 保存结果
            output_path = save_dataset(
                df=df_injected,
                dataset_name=dataset_name,
                missing_ratio=missing_ratio,
                term=term,
                pattern=pattern,
                output_base_dir=output_base_dir,
                block_length=block_length,
            )
            info["output_path"] = output_path
            info["original_path"] = str(Path(data_path) / "ori" / f"{dataset_name}.csv")
            info["pattern"] = pattern
            
            # 4. 打印信息
            print(f"\n注入结果:")
            print(f"  总单元格数：{info['total_cells']}")
            print(f"  块长度：{info['block_length']}")
            print(f"  块数量：{info['n_blocks']}")
            print(f"  注入缺失值数：{info['injected_missing']}")
            print(f"  实际缺失比例：{info['actual_missing_ratio']:.2%}")
            print(f"\n文件保存:")
            print(f"  {output_path}")
            print("="*80)
            
            results.append(info)
    
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="缺失值注入统一入口（支持多种注错模式）"
    )
    parser.add_argument(
        "--missing_pattern",
        type=str,
        required=True,
        choices=["MCAR", "BM", "MAR", "MNAR"],
        help="缺失模式（MCAR, BM, MAR, MNAR）"
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="输出目录基础路径（默认：datasets）"
    )
    parser.add_argument(
        "--block_length",
        type=int,
        default=50,
        help="块长度，仅用于 BM 模式（默认：50）"
    )
    
    args = parser.parse_args()
    
    # 1. 解析缺失比例
    missing_ratios = parse_missing_ratios(args.missing_ratio)
    
    if len(missing_ratios) == 1:
        print(f"缺失比例：{missing_ratios[0]:.2%}")
    else:
        print(f"缺失比例列表：{missing_ratios}")
    
    # 2. 确定 term 列表
    if args.no_auto_term:
        if args.term is None:
            terms = ["short"]
        else:
            terms = [args.term]
        print(f"使用指定的 term: {terms}")
    else:
        terms = get_available_terms(args.dataset, args.data_path)
        print(f"自动检测到数据集 '{args.dataset}' 的 term 配置：{terms}")
    
    # 3. 根据 missing_pattern 调用对应的注入函数
    if args.missing_pattern == "MCAR":
        results = run_mcar_injection(
            dataset_name=args.dataset,
            missing_ratios=missing_ratios,
            terms=terms,
            data_path=args.data_path,
            output_base_dir=args.output_dir,
            seed=args.seed,
        )
    elif args.missing_pattern == "BM":
        results = run_bm_injection(
            dataset_name=args.dataset,
            missing_ratios=missing_ratios,
            terms=terms,
            data_path=args.data_path,
            output_base_dir=args.output_dir,
            block_length=args.block_length,
            seed=args.seed,
        )
    else:
        raise NotImplementedError(f"Missing pattern '{args.missing_pattern}' is not implemented yet.")
    
    # 4. 打印汇总
    print(f"\n{'='*80}")
    print(f"批量注入完成！共生成 {len(results)} 个文件:")
    print(f"{'='*80}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['output_path']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
