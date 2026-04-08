"""
时间序列分析模块

包含 STL 分解分析和特征指标计算
参考文献：基于STL分解的时间序列特征提取方法
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from statsmodels.tsa.seasonal import STL


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


def get_data_cols(df: pd.DataFrame) -> List[str]:
    """获取数据列（排除时间列）"""
    time_cols = ['date', 'time', 'timestamp', 'datetime', 'index']
    return [col for col in df.columns if col.lower() not in time_cols]


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
    """计算所有6个指标"""
    return {
        "trend_strength": calculate_trend_strength(trend, residual),
        "trend_linearity": calculate_trend_linearity(trend),
        "seasonal_strength": calculate_seasonal_strength(seasonal, residual),
        "seasonal_correlation": calculate_seasonal_correlation(seasonal, period),
        "residual_autocorr_lag1": calculate_residual_autocorr_lag1(residual),
        "spectral_entropy": calculate_spectral_entropy(original),
    }


class STLAnalyzer:
    """STL 分解分析器"""
    
    def __init__(self, period: int = 24):
        self.period = period
    
    def analyze(
        self,
        data: pd.DataFrame,
        dataset: str = "",
        data_type: str = "",
        method: str = "",
        save_series: bool = True,
    ) -> Dict[str, Any]:
        """
        执行 STL 分解分析，计算6个特征指标
        
        Args:
            data: 时间序列数据
            dataset: 数据集名称
            data_type: 数据类型
            method: 填补方法
            save_series: 是否保存序列数据
            
        Returns:
            分析结果字典
        """
        data_cols = get_data_cols(data)
        
        all_metrics = {
            "trend_strength": [],
            "trend_linearity": [],
            "seasonal_strength": [],
            "seasonal_correlation": [],
            "residual_autocorr_lag1": [],
            "spectral_entropy": [],
        }
        
        all_trend_series = []
        all_seasonal_series = []
        all_residual_series = []
        
        for col in data_cols:
            series = data[col].dropna()
            
            if len(series) < 2 * self.period:
                continue
            
            try:
                stl = STL(series, period=self.period, robust=True)
                result = stl.fit()
                
                trend = result.trend.values
                seasonal = result.seasonal.values
                residual = result.resid.values
                original = series.values
                
                metrics = calculate_all_metrics(
                    original, trend, seasonal, residual, self.period
                )
                
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                
                if save_series:
                    all_trend_series.append(trend)
                    all_seasonal_series.append(seasonal)
                    all_residual_series.append(residual)
                    
            except Exception as e:
                print(f"    [警告] STL 分解失败 {col}: {e}")
                continue
        
        if not all_metrics["trend_strength"]:
            return {
                "analyzer": "STL",
                "dataset": dataset,
                "data_type": data_type,
                "method": method,
                "metrics": {},
                "details": {},
                "success": False,
                "error": "No valid series for STL decomposition",
            }
        
        avg_metrics = {key: float(np.mean(values)) for key, values in all_metrics.items()}
        
        details = {
            "n_series": len(all_metrics["trend_strength"]),
            "period": self.period,
        }
        
        if save_series and all_trend_series:
            min_len = min(len(s) for s in all_trend_series)
            max_series_len = 500
            if min_len > max_series_len:
                step = min_len // max_series_len
                indices = np.arange(0, min_len, step)[:max_series_len]
            else:
                indices = np.arange(min_len)
            
            avg_trend = np.mean([s[:min_len] for s in all_trend_series], axis=0)[indices]
            avg_seasonal = np.mean([s[:min_len] for s in all_seasonal_series], axis=0)[indices]
            avg_residual = np.mean([s[:min_len] for s in all_residual_series], axis=0)[indices]
            
            details["series"] = {
                "trend": avg_trend.tolist(),
                "seasonal": avg_seasonal.tolist(),
                "residual": avg_residual.tolist(),
                "length": len(indices),
                "original_length": min_len,
            }
        
        return {
            "analyzer": "STL",
            "dataset": dataset,
            "data_type": data_type,
            "method": method,
            "metrics": avg_metrics,
            "details": details,
            "success": True,
        }


def compare_results(
    ori_result: Dict[str, Any],
    missing_result: Dict[str, Any],
    imputed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    比较分析结果
    
    Args:
        ori_result: 干净数据的分析结果
        missing_result: 缺失数据的分析结果
        imputed_results: 填补数据的分析结果列表
        
    Returns:
        比较结果字典
    """
    if not ori_result.get("success") or not ori_result.get("metrics"):
        return {"error": "Original data result is invalid"}
    
    ori_metrics = ori_result["metrics"]
    
    comparison = {
        "analyzer": ori_result["analyzer"],
        "dataset": ori_result["dataset"],
        "metrics": {},
    }
    
    for metric_name, ori_value in ori_metrics.items():
        metric_comparison = {
            "ori": ori_value,
            "missing": None,
            "imputed": {},
            "missing_diff_pct": None,
            "imputed_diff_pct": {},
            "missing_recovery_pct": None,
            "imputed_recovery_pct": {},
        }
        
        if missing_result.get("success") and metric_name in missing_result.get("metrics", {}):
            missing_value = missing_result["metrics"][metric_name]
            metric_comparison["missing"] = missing_value
            if ori_value != 0:
                metric_comparison["missing_diff_pct"] = (missing_value - ori_value) / abs(ori_value) * 100
                metric_comparison["missing_recovery_pct"] = 100 - abs(metric_comparison["missing_diff_pct"])
        
        for imputed_result in imputed_results:
            if imputed_result.get("success") and metric_name in imputed_result.get("metrics", {}):
                method = imputed_result.get("method", "unknown")
                imputed_value = imputed_result["metrics"][metric_name]
                metric_comparison["imputed"][method] = imputed_value
                
                if ori_value != 0:
                    diff_pct = (imputed_value - ori_value) / abs(ori_value) * 100
                    metric_comparison["imputed_diff_pct"][method] = diff_pct
                    metric_comparison["imputed_recovery_pct"][method] = 100 - abs(diff_pct)
        
        comparison["metrics"][metric_name] = metric_comparison
    
    return comparison


def print_comparison(comparison: Dict[str, Any], imputed_methods: List[str] = None):
    """打印比较结果"""
    if "error" in comparison:
        print(f"  [错误] {comparison['error']}")
        return
    
    analyzer = comparison["analyzer"]
    dataset = comparison["dataset"]
    metrics = comparison["metrics"]
    
    if imputed_methods is None:
        imputed_methods = list(set(
            method 
            for m in metrics.values() 
            for method in m.get("imputed", {}).keys()
        ))
    
    header = f"{dataset} - {analyzer} 分析结果比较"
    print(f"\n{'='*80}")
    print(header)
    print(f"{'='*80}")
    
    col_width = 12
    header_line = f"{'指标':<25} {'Ori':<{col_width}} {'Missing':<{col_width}}"
    for method in imputed_methods:
        header_line += f" {method:<{col_width}}"
    print(header_line)
    print("-" * 80)
    
    for metric_name, metric_data in metrics.items():
        ori_val = metric_data.get("ori", 0)
        missing_val = metric_data.get("missing")
        
        line = f"{metric_name:<25} {ori_val:<{col_width}.4f}"
        
        if missing_val is not None:
            line += f" {missing_val:<{col_width}.4f}"
        else:
            line += f" {'-':<{col_width}}"
        
        for method in imputed_methods:
            val = metric_data.get("imputed", {}).get(method)
            if val is not None:
                line += f" {val:<{col_width}.4f}"
            else:
                line += f" {'-':<{col_width}}"
        
        print(line)
    
    print("-" * 80)
    print("与Ori的差异 (%):")
    
    for metric_name, metric_data in metrics.items():
        missing_diff = metric_data.get("missing_diff_pct")
        
        line = f"{metric_name:<25} {'-':<{col_width}}"
        
        if missing_diff is not None:
            line += f" {missing_diff:>+{col_width-2}.1f}%"
        else:
            line += f" {'-':<{col_width}}"
        
        for method in imputed_methods:
            diff = metric_data.get("imputed_diff_pct", {}).get(method)
            if diff is not None:
                line += f" {diff:>+{col_width-2}.1f}%"
            else:
                line += f" {'-':<{col_width}}"
        
        print(line)
    
    print(f"{'='*80}")


def calculate_recovery_score(comparison: Dict[str, Any], method: str) -> float:
    """计算填补方法的恢复得分"""
    if "metrics" not in comparison:
        return 0.0
    
    recovery_scores = []
    for metric_data in comparison["metrics"].values():
        recovery = metric_data.get("imputed_recovery_pct", {}).get(method)
        if recovery is not None:
            recovery_scores.append(recovery)
    
    return np.mean(recovery_scores) if recovery_scores else 0.0


def get_best_method(comparison: Dict[str, Any], imputed_methods: List[str] = None) -> Tuple[str, float]:
    """获取最佳填补方法"""
    if imputed_methods is None:
        imputed_methods = list(set(
            method 
            for m in comparison.get("metrics", {}).values() 
            for method in m.get("imputed", {}).keys()
        ))
    
    if not imputed_methods:
        return None, 0.0
    
    best_method = None
    best_score = -float('inf')
    
    for method in imputed_methods:
        score = calculate_recovery_score(comparison, method)
        if score > best_score:
            best_score = score
            best_method = method
    
    return best_method, best_score


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
