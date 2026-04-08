"""
独立的填补模块：对评估数据集进行缺失值填补并保存结果
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from Imputation.imputation_methods import (
    zero_imputation,
    mean_imputation,
    forward_fill,
    backward_fill,
    linear_interpolation,
    nearest_interpolation,
    spline_interpolation,
    seasonal_decomposition_imputation,
    none_imputation,
)

IMPUTATION_METHODS = {
    'zero': zero_imputation,
    'mean': mean_imputation,
    'forward': forward_fill,
    'backward': backward_fill,
    'linear': linear_interpolation,
    'nearest': nearest_interpolation,
    'spline': spline_interpolation,
    'seasonal': seasonal_decomposition_imputation,
    'none': none_imputation,
}


def get_imputation_method(method_name: str):
    """
    获取填补方法函数
    
    Args:
        method_name: 填补方法名称
    
    Returns:
        填补方法函数
    """
    if method_name not in IMPUTATION_METHODS:
        raise ValueError(
            f"Unknown imputation method: {method_name}. "
            f"Available methods: {list(IMPUTATION_METHODS.keys())}"
        )
    return IMPUTATION_METHODS[method_name]


def generate_imputed_dataset_path(
    eval_data_path: str,
    imputation_method: str,
    base_output_dir: str = "datasets/Imputed",
) -> str:
    """
    生成填补后数据集的保存路径
    
    Args:
        eval_data_path: 原评估数据集路径
        imputation_method: 填补方法名称
        base_output_dir: 输出基础目录
    
    Returns:
        填补后数据集的保存路径
    """
    eval_path = Path(eval_data_path)
    original_filename = eval_path.name
    
    parts = original_filename.replace('.csv', '').split('_')
    
    if len(parts) >= 4:
        dataset_name = parts[0]
        method = parts[1]
        ratio = parts[-2]
        term = parts[-1]
        
        output_filename = f"{dataset_name}_{method}_{ratio}_{term}_{imputation_method}.csv"
        
        output_dir = Path(base_output_dir) / method / f"{method}_{ratio}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(output_dir / output_filename)
    else:
        output_filename = f"{original_filename.replace('.csv', '')}_{imputation_method}.csv"
        output_dir = Path(base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return str(output_dir / output_filename)


def impute_dataset(
    eval_data_path: str,
    imputation_method: str,
    output_path: Optional[str] = None,
    base_output_dir: str = "datasets/Imputed",
    save_result: bool = True,
) -> pd.DataFrame:
    """
    对评估数据集进行缺失值填补
    
    Args:
        eval_data_path: 评估数据集路径
        imputation_method: 填补方法名称
        output_path: 输出路径（可选，如果不指定则自动生成）
        base_output_dir: 输出基础目录
        save_result: 是否保存结果
    
    Returns:
        填补后的 DataFrame
    """
    if not Path(eval_data_path).exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_data_path}")
    
    print(f"\n{'='*80}")
    print(f"Imputing Dataset")
    print(f"{'='*80}")
    print(f"  Input: {eval_data_path}")
    print(f"  Method: {imputation_method}")
    
    df = pd.read_csv(eval_data_path)
    
    time_col = None
    for c in ['date', 'time', 'timestamp']:
        if c in df.columns:
            time_col = c
            break
    
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    
    data_cols = list(df.columns)
    missing_before = df.isna().sum().sum()
    
    if imputation_method.lower() != 'none':
        imputation_func = get_imputation_method(imputation_method)
        df = imputation_func(df, data_cols)
    
    missing_after = df.isna().sum().sum()
    
    print(f"  Missing values: {missing_before} -> {missing_after}")
    print(f"  Shape: {df.shape}")
    
    if save_result:
        if output_path is None:
            output_path = generate_imputed_dataset_path(
                eval_data_path=eval_data_path,
                imputation_method=imputation_method,
                base_output_dir=base_output_dir,
            )
        
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if time_col:
            df_to_save = df.reset_index()
            df_to_save.columns = [time_col] + data_cols
        else:
            df_to_save = df
        
        df_to_save.to_csv(output_path, index=False)
        print(f"  Output: {output_path}")
    
    print(f"{'='*80}\n")
    
    return df


def batch_impute(
    eval_data_paths: List[str],
    imputation_methods: List[str],
    base_output_dir: str = "datasets/Imputed",
) -> List[str]:
    """
    批量填补多个数据集
    
    Args:
        eval_data_paths: 评估数据集路径列表
        imputation_methods: 填补方法列表
        base_output_dir: 输出基础目录
    
    Returns:
        填补后数据集路径列表
    """
    imputed_paths = []
    
    for eval_path in eval_data_paths:
        for method in imputation_methods:
            output_path = generate_imputed_dataset_path(
                eval_data_path=eval_path,
                imputation_method=method,
                base_output_dir=base_output_dir,
            )
            
            if Path(output_path).exists():
                print(f"✓ Already exists: {output_path}")
                imputed_paths.append(output_path)
                continue
            
            impute_dataset(
                eval_data_path=eval_path,
                imputation_method=method,
                output_path=output_path,
                base_output_dir=base_output_dir,
                save_result=True,
            )
            imputed_paths.append(output_path)
    
    return imputed_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Impute missing values in datasets")
    
    parser.add_argument(
        "--eval_data_path",
        type=str,
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--imputation_method",
        type=str,
        help="Imputation method name",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for imputed dataset",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="datasets/Imputed",
        help="Base output directory",
    )
    
    args = parser.parse_args()
    
    impute_dataset(
        eval_data_path=args.eval_data_path,
        imputation_method=args.imputation_method,
        output_path=args.output_path,
        base_output_dir=args.base_output_dir,
        save_result=True,
    )
