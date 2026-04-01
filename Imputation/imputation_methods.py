"""
时间序列缺失值填补方法库

提供多种填补算法，所有方法都是纯函数，接收 DataFrame 返回 DataFrame
"""

import numpy as np
import pandas as pd
from typing import Literal


def zero_imputation(df: pd.DataFrame, data_cols: list) -> pd.DataFrame:
    """零值填补"""
    df_imputed = df.copy()
    df_imputed[data_cols] = df_imputed[data_cols].fillna(0)
    return df_imputed


def mean_imputation(df: pd.DataFrame, data_cols: list) -> pd.DataFrame:
    """均值填补"""
    df_imputed = df.copy()
    df_imputed[data_cols] = df_imputed[data_cols].fillna(df_imputed[data_cols].mean())
    return df_imputed


def forward_fill(df: pd.DataFrame, data_cols: list) -> pd.DataFrame:
    """前向填补"""
    df_imputed = df.copy()
    df_imputed[data_cols] = df_imputed[data_cols].ffill()
    return df_imputed


def backward_fill(df: pd.DataFrame, data_cols: list) -> pd.DataFrame:
    """后向填补"""
    df_imputed = df.copy()
    df_imputed[data_cols] = df_imputed[data_cols].bfill()
    return df_imputed


def linear_interpolation(df: pd.DataFrame, data_cols: list) -> pd.DataFrame:
    """线性插值填补"""
    df_imputed = df.copy()
    df_imputed[data_cols] = df_imputed[data_cols].interpolate(method='linear')
    return df_imputed


def nearest_interpolation(df: pd.DataFrame, data_cols: list) -> pd.DataFrame:
    """最近邻插值填补"""
    df_imputed = df.copy()
    df_imputed[data_cols] = df_imputed[data_cols].interpolate(method='nearest')
    return df_imputed


def spline_interpolation(df: pd.DataFrame, data_cols: list, order: int = 3) -> pd.DataFrame:
    """样条插值填补"""
    df_imputed = df.copy()
    df_imputed[data_cols] = df_imputed[data_cols].interpolate(method='spline', order=order)
    return df_imputed


def seasonal_decomposition_imputation(
    df: pd.DataFrame, 
    data_cols: list, 
    freq: str,
    model: Literal['additive', 'multiplicative'] = 'additive'
) -> pd.DataFrame:
    """基于季节分解的填补"""
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    df_imputed = df.copy()
    
    for col in data_cols:
        series = df_imputed[col]
        missing_mask = series.isna()
        
        if not missing_mask.any():
            continue
        
        series_filled = series.interpolate(method='linear')
        
        try:
            decomposition = seasonal_decompose(
                series_filled, 
                model=model, 
                freq=freq,
                period=None
            )
            reconstructed = decomposition.trend + decomposition.seasonal + decomposition.resid
            df_imputed.loc[missing_mask, col] = reconstructed.loc[missing_mask]
        except Exception:
            df_imputed[col] = series.interpolate(method='linear')
    
    return df_imputed


def get_imputation_method(method_name: str):
    """获取填补方法函数"""
    method_map = {
        'zero': zero_imputation,
        'mean': mean_imputation,
        'forward': forward_fill,
        'backward': backward_fill,
        'linear': linear_interpolation,
        'nearest': nearest_interpolation,
        'spline': spline_interpolation,
        'seasonal': seasonal_decomposition_imputation,
        'none': none_imputation,  # 不填补
    }
    
    if method_name not in method_map:
        raise ValueError(
            f"Unknown imputation method: {method_name}. "
            f"Available methods: {list(method_map.keys())}"
        )
    
    return method_map[method_name]


def none_imputation(df: pd.DataFrame, data_cols: list) -> pd.DataFrame:
    """不进行任何填补，直接返回原始数据（保留缺失值）
    
    Args:
        df: 包含缺失值的数据框
        data_cols: 需要填补的列名列表
    
    Returns:
        原始数据框（保留缺失值）
    """
    return df.copy()
