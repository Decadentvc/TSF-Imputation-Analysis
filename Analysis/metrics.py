"""
时间序列特征指标计算模块

包含 6 个基于 STL 分解的时间序列特征指标：
1. 趋势强度 (trend_strength)
2. 趋势线性度 (trend_linearity)
3. 季节强度 (seasonal_strength)
4. 季节相关性 (seasonal_correlation)
5. 残差一阶自相关性 (residual_autocorr_lag1)
6. 谱熵 (spectral_entropy)

参考文献：基于STL分解的时间序列特征提取方法
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict


def load_dataset_properties(properties_path: str = "datasets/dataset_properties.json") -> dict:
    """加载数据集属性"""
    props_path = Path(properties_path)
    if not props_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {props_path}")
    
    with open(props_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_period(dataset_name: str, properties_path: str = "datasets/dataset_properties.json") -> int:
    """获取数据集的周期"""
    props = load_dataset_properties(properties_path)
    if dataset_name not in props:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    return props[dataset_name].get("period", 24)


def calculate_trend_strength(trend: np.ndarray, residual: np.ndarray) -> float:
    """
    趋势强度
    
    公式: F_TS = max(0, 1 - Var(R) / Var(T + R))
    含义: 量化趋势分量相较于残差的强度
    取值范围: [0, 1]
    """
    var_residual = np.var(residual)
    var_trend_residual = np.var(trend + residual)
    
    if var_trend_residual == 0:
        return 0.0
    
    strength = 1 - (var_residual / var_trend_residual)
    return max(0.0, strength)


def calculate_trend_linearity(trend: np.ndarray) -> float:
    """
    趋势线性度
    
    公式: 使用正交二次回归 T_t = β_0 + β_1 * P_1(t) + β_2 * P_2(t) + ε_t
    趋势线性度 = β_1
    含义: 捕获趋势内线性进展的总体方向和陡度
    取值范围: 实数，正值上升，负值下降
    """
    L = len(trend)
    t = np.arange(1, L + 1)
    
    P1 = t - np.mean(t)
    P2 = (t - np.mean(t))**2 - np.mean((t - np.mean(t))**2)
    
    dot_P1_P2 = np.dot(P1, P2)
    if dot_P1_P2 != 0:
        P2 = P2 - (dot_P1_P2 / np.dot(P1, P1)) * P1
    
    X = np.column_stack([np.ones(L), P1, P2])
    beta = np.linalg.lstsq(X, trend, rcond=None)[0]
    
    return float(beta[1])


def calculate_seasonal_strength(seasonal: np.ndarray, residual: np.ndarray) -> float:
    """
    季节强度
    
    公式: F_SS = max(0, 1 - Var(R) / Var(S + R))
    含义: 量化季节性分量相较于残差的强度
    取值范围: [0, 1]
    """
    var_residual = np.var(residual)
    var_seasonal_residual = np.var(seasonal + residual)
    
    if var_seasonal_residual == 0:
        return 0.0
    
    strength = 1 - (var_residual / var_seasonal_residual)
    return max(0.0, strength)


def calculate_seasonal_correlation(seasonal: np.ndarray, period: int) -> float:
    """
    季节相关性
    
    公式: F_SC = 2/(K(K-1)) * Σ_{i=1}^{K} Σ_{j=i+1}^{K} Corr(s_i, s_j)
    含义: 所有K个完整季节周期之间的平均皮尔逊相关系数
    取值范围: [-1, 1]
    """
    n = len(seasonal)
    K = n // period
    
    if K < 2:
        return 0.0
    
    truncated_len = K * period
    seasonal_truncated = seasonal[:truncated_len]
    seasons = seasonal_truncated.reshape(K, period)
    
    correlations = []
    for i in range(K):
        for j in range(i + 1, K):
            s_i = seasons[i]
            s_j = seasons[j]
            
            std_i = np.std(s_i)
            std_j = np.std(s_j)
            
            if std_i == 0 or std_j == 0:
                continue
            
            corr = np.corrcoef(s_i, s_j)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    if not correlations:
        return 0.0
    
    return float(np.mean(correlations))


def calculate_residual_autocorr_lag1(residual: np.ndarray) -> float:
    """
    残差一阶自相关性
    
    公式: F_RA = E[(R_t - R̄)(R_{t-1} - R̄)] / Var(R)
    含义: 残差序列的一阶自相关程度
    取值范围: [-1, 1]
    """
    n = len(residual)
    
    if n < 2:
        return 0.0
    
    mean_res = np.mean(residual)
    
    numerator = np.sum((residual[1:] - mean_res) * (residual[:-1] - mean_res))
    denominator = np.sum((residual - mean_res) ** 2)
    
    if denominator == 0:
        return 0.0
    
    return float(numerator / denominator)


def calculate_spectral_entropy(series: np.ndarray) -> float:
    """
    谱熵
    
    公式: I(f) = ∫_{-π}^{π} log f(ω) dω
    离散形式: I(f) = Σ log(PSD(ω_i))
    含义: 衡量时间序列的频谱分布的复杂度
    取值范围: 实数
    """
    series = series - np.mean(series)
    n = len(series)
    
    if n < 2:
        return 0.0
    
    fft_result = np.fft.fft(series)
    psd = np.abs(fft_result[:n//2]) ** 2
    psd = psd[psd > 1e-10]
    
    if len(psd) == 0:
        return 0.0
    
    return float(np.sum(np.log(psd)))


def calculate_all_metrics(
    original: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    residual: np.ndarray,
    period: int
) -> Dict[str, float]:
    """
    计算所有 6 个特征指标
    
    Args:
        original: 原始序列
        trend: 趋势分量
        seasonal: 季节分量
        residual: 残差分量
        period: 周期长度
        
    Returns:
        包含 6 个指标的字典
    """
    return {
        "trend_strength": calculate_trend_strength(trend, residual),
        "trend_linearity": calculate_trend_linearity(trend),
        "seasonal_strength": calculate_seasonal_strength(seasonal, residual),
        "seasonal_correlation": calculate_seasonal_correlation(seasonal, period),
        "residual_autocorr_lag1": calculate_residual_autocorr_lag1(residual),
        "spectral_entropy": calculate_spectral_entropy(original),
    }


if __name__ == "__main__":
    np.random.seed(42)
    
    n = 200
    period = 24
    t = np.arange(n)
    
    trend = 0.05 * t + 10
    seasonal = 5 * np.sin(2 * np.pi * t / period)
    residual = np.random.normal(0, 1, n)
    original = trend + seasonal + residual
    
    metrics = calculate_all_metrics(original, trend, seasonal, residual, period)
    
    print("=" * 60)
    print("时间序列特征指标计算结果")
    print("=" * 60)
    for name, value in metrics.items():
        print(f"{name:25s}: {value:.4f}")
    print("=" * 60)
