"""
批量填补脚本

对 datasets/MCAR 下所有注入缺失值的数据进行填补
只对注入区间内的数据进行填补，避免数据泄露
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Imputation.imputation_methods import get_imputation_method

sys.path.insert(0, str(project_root / "Missing_Value_Injection" / "for_sundial"))
from inject_range_utils import get_injection_range


def get_data_cols(df: pd.DataFrame) -> List[str]:
    """获取数据列（排除时间列）"""
    time_cols = ['date', 'time', 'timestamp', 'datetime', 'index']
    data_cols = [col for col in df.columns if col.lower() not in time_cols]
    return data_cols


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    从文件名解析数据集信息
    
    Args:
        filename: 如 "ETTh1_MCAR_005_short.csv"
        
    Returns:
        (dataset_name, ratio, term)
    """
    basename = filename.replace(".csv", "")
    parts = basename.split("_")
    
    dataset_name = parts[0]
    ratio = parts[2]
    term = parts[-1]
    
    return dataset_name, ratio, term


def get_injection_indices(dataset_name: str, term: str, data_path: str = "datasets") -> Tuple[int, int]:
    """
    获取注入区间的起始和结束索引
    
    Args:
        dataset_name: 数据集名称
        term: term 类型
        data_path: 数据集目录
        
    Returns:
        (start_index, end_index)
    """
    try:
        result = get_injection_range(dataset_name, term, data_path)
        return result["start_index"], result["end_index"]
    except Exception as e:
        print(f"    [警告] 无法获取注入区间: {e}")
        return 0, -1


def impute_single_file(
    input_path: str,
    output_path: str,
    method: str,
    injection_range: Tuple[int, int] = None,
) -> dict:
    """
    对单个文件进行填补
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        method: 填补方法
        injection_range: 注入区间 (start, end)，如果提供则只对该区间填补
        
    Returns:
        填补信息字典
    """
    df = pd.read_csv(input_path)
    
    data_cols = get_data_cols(df)
    
    missing_before = df[data_cols].isna().sum().sum()
    
    impute_func = get_imputation_method(method)
    
    if injection_range is not None:
        start_idx, end_idx = injection_range
        
        if end_idx == -1:
            end_idx = len(df)
        
        df_imputed = df.copy()
        
        df_range = df.iloc[start_idx:end_idx].copy()
        
        if method == 'spline':
            df_range_imputed = impute_func(df_range, data_cols, order=3)
        elif method == 'polynomial':
            df_range_imputed = impute_func(df_range, data_cols, order=2)
        else:
            df_range_imputed = impute_func(df_range, data_cols)
        
        for col in data_cols:
            df_imputed.loc[df_imputed.index[start_idx:end_idx], col] = df_range_imputed[col].values
    else:
        if method == 'spline':
            df_imputed = impute_func(df, data_cols, order=3)
        elif method == 'polynomial':
            df_imputed = impute_func(df, data_cols, order=2)
        else:
            df_imputed = impute_func(df, data_cols)
    
    missing_after = df_imputed[data_cols].isna().sum().sum()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_imputed.to_csv(output_path, index=False)
    
    return {
        "input": input_path,
        "output": output_path,
        "method": method,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "filled": missing_before - missing_after,
        "injection_range": injection_range,
    }


def run_batch_imputation(
    input_base_dir: str = "datasets/MCAR",
    output_base_dir: str = "datasets/imputed",
    methods: List[str] = None,
    skip_existing: bool = True,
):
    """
    批量填补
    
    Args:
        input_base_dir: 输入基础目录
        output_base_dir: 输出基础目录
        methods: 填补方法列表
        skip_existing: 是否跳过已存在的文件
    """
    if methods is None:
        methods = ['zero', 'forward', 'backward', 'mean', 'linear']
    
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)
    
    print(f"\n{'='*80}")
    print(f"批量填补")
    print(f"{'='*80}")
    print(f"输入目录: {input_base}")
    print(f"输出目录: {output_base}")
    print(f"填补方法: {methods}")
    print(f"{'='*80}")
    
    mcar_dirs = sorted([d for d in input_base.iterdir() if d.is_dir() and d.name.startswith('MCAR_')])
    
    total_files = 0
    total_filled = 0
    
    for mcar_dir in mcar_dirs:
        ratio = mcar_dir.name
        print(f"\n处理目录: {ratio}")
        print("-" * 40)
        
        csv_files = sorted(mcar_dir.glob("*.csv"))
        
        for csv_file in csv_files:
            filename = csv_file.name
            
            dataset_name = filename.replace(f"_{ratio}_", "_").replace(".csv", "")
            parts = filename.replace(".csv", "").split("_")
            
            dataset = parts[0]
            term = parts[-1]
            
            for method in methods:
                output_dir = output_base / ratio / method
                output_file = output_dir / filename
                
                if skip_existing and output_file.exists():
                    print(f"  [跳过] {filename} -> {method}")
                    continue
                
                print(f"  [处理] {filename} -> {method}")
                
                try:
                    result = impute_single_file(
                        str(csv_file),
                        str(output_file),
                        method,
                    )
                    total_files += 1
                    total_filled += result["filled"]
                    print(f"    填补: {result['missing_before']} -> {result['missing_after']} ({result['filled']} 个值)")
                except Exception as e:
                    print(f"    [错误] {e}")
    
    print(f"\n{'='*80}")
    print(f"批量填补完成")
    print(f"{'='*80}")
    print(f"处理文件数: {total_files}")
    print(f"填补缺失值总数: {total_filled}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="批量填补 MCAR 数据")
    
    parser.add_argument("--input_dir", type=str, default="datasets/MCAR",
                        help="输入目录 (默认: datasets/MCAR)")
    parser.add_argument("--output_dir", type=str, default="datasets/imputed",
                        help="输出目录 (默认: datasets/imputed)")
    parser.add_argument("--methods", type=str, nargs='+',
                        default=['zero', 'forward', 'backward', 'mean', 'linear'],
                        help="填补方法列表 (默认: zero forward backward mean linear)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的文件 (默认: True)")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="不跳过已存在的文件，覆盖重写")
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing if args.no_skip_existing else args.skip_existing
    
    run_batch_imputation(
        input_base_dir=args.input_dir,
        output_base_dir=args.output_dir,
        methods=args.methods,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    main()
