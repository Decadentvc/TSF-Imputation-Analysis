"""
填补模块

职责：读取带缺失数据 -> 填补 -> 保存

使用示例：
    python Imputation/impute.py --input_dir datasets/window_data/ETTh1/short/MCAR/010/missing --method linear
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Imputation.imputation_methods import get_imputation_method


def load_windows_from_dir(input_dir: str) -> tuple:
    """
    从目录加载窗口数据和元数据
    
    Args:
        input_dir: 包含窗口 CSV 和 meta.json 的目录
        
    Returns:
        (windows, meta) - windows 是 (df, window_info) 的列表
    """
    input_path = Path(input_dir)
    
    meta_path = input_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {input_dir}")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    n_windows = meta["n_windows"]
    windows = []
    
    for i in range(n_windows):
        window_file = input_path / f"window_{i:03d}.csv"
        if not window_file.exists():
            raise FileNotFoundError(f"Window file not found: {window_file}")
        
        df = pd.read_csv(window_file)
        
        time_col = meta.get("time_col")
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        
        windows.append((df, {"window_index": i}))
    
    return windows, meta


def apply_imputation(
    df: pd.DataFrame,
    method: str,
    data_cols: List[str],
    freq: Optional[str] = None,
) -> pd.DataFrame:
    """
    对数据应用填补方法
    
    Args:
        df: 包含缺失值的数据框
        method: 填补方法名称
        data_cols: 需要填补的列名
        freq: 数据频率（某些方法需要）
        
    Returns:
        填补后的数据框
    """
    impute_func = get_imputation_method(method)
    
    if method == 'seasonal':
        if freq is None:
            raise ValueError("freq is required for seasonal imputation")
        return impute_func(df, data_cols, freq)
    elif method == 'spline':
        return impute_func(df, data_cols, order=3)
    elif method == 'polynomial':
        return impute_func(df, data_cols, order=2)
    else:
        return impute_func(df, data_cols)


def run_imputation(
    input_dir: str,
    method: str,
    output_dir: Optional[str] = None,
    freq: Optional[str] = None,
) -> Path:
    """
    运行填补流程
    
    Args:
        input_dir: 带缺失数据的目录
        method: 填补方法
        output_dir: 输出目录（可选，默认为 input_dir 同级的 imputed/{method}）
        freq: 数据频率
        
    Returns:
        输出目录路径
    """
    windows, meta = load_windows_from_dir(input_dir)
    
    data_cols = meta.get("data_cols", [])
    if not data_cols:
        time_col = meta.get("time_col")
        df_sample = windows[0][0]
        data_cols = [col for col in df_sample.columns if col != time_col]
    
    if freq is None:
        freq = meta.get("frequency", "H")
    
    prediction_length = meta.get("prediction_length", 0)
    
    print(f"\n{'='*80}")
    print(f"填补数据")
    print(f"{'='*80}")
    print(f"输入目录: {input_dir}")
    print(f"填补方法: {method}")
    print(f"数据列数: {len(data_cols)}")
    print(f"窗口数: {len(windows)}")
    print(f"预测长度: {prediction_length}")
    print(f"{'='*80}")
    
    imputed_windows = []
    
    for df, window_info in windows:
        window_idx = window_info["window_index"]
        
        if prediction_length > 0:
            context_length = len(df) - prediction_length
            df_context = df.iloc[:context_length].copy()
            df_forecast = df.iloc[context_length:].copy()
            
            df_context_imputed = apply_imputation(df_context, method, data_cols, freq)
            
            df_imputed = pd.concat([df_context_imputed, df_forecast])
        else:
            df_imputed = apply_imputation(df, method, data_cols, freq)
        
        imputed_windows.append((df_imputed, window_info))
        
        missing_count = df[data_cols].isna().sum().sum()
        print(f"\n窗口 {window_idx}:")
        print(f"  Context 长度: {context_length if prediction_length > 0 else len(df)}")
        print(f"  原始缺失值数: {missing_count}")
        print(f"  填补后缺失值数: {df_imputed[data_cols].isna().sum().sum()}")
    
    if output_dir is None:
        input_path = Path(input_dir)
        parent_dir = input_path.parent
        output_dir = parent_dir / "imputed" / method
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存填补数据到: {output_dir}")
    
    for df_imputed, window_info in imputed_windows:
        window_idx = window_info["window_index"]
        filename = f"window_{window_idx:03d}.csv"
        filepath = output_dir / filename
        df_imputed.to_csv(filepath)
        print(f"  保存: {filename}")
    
    imputed_meta = {
        **meta,
        "imputation_method": method,
        "imputed_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
    }
    
    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(imputed_meta, f, indent=2, ensure_ascii=False)
    print(f"  保存: meta.json")
    
    print(f"\n{'='*80}")
    print(f"完成！")
    print(f"{'='*80}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="填补模块")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="带缺失数据的目录路径")
    parser.add_argument("--method", type=str, required=True,
                        help="填补方法 (zero, mean, forward, backward, linear, nearest, polynomial, spline, seasonal, none)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (可选，默认为 input_dir 同级的 imputed/{method})")
    parser.add_argument("--freq", type=str, default=None,
                        help="数据频率 (某些方法需要，如 seasonal)")
    
    args = parser.parse_args()
    
    output_dir = run_imputation(
        input_dir=args.input_dir,
        method=args.method,
        output_dir=args.output_dir,
        freq=args.freq,
    )
    
    print(f"\n输出目录: {output_dir}")


if __name__ == "__main__":
    main()
