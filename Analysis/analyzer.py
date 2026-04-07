"""
时间序列分析模块

包含 STL 分解、FFT 频域分析、ACF 自相关分析
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf


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
        执行 STL 分解分析
        
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
        
        all_trend_strength = []
        all_seasonal_strength = []
        all_residual_strength = []
        all_trend_change_rate = []
        
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
                
                trend = result.trend
                seasonal = result.seasonal
                residual = result.resid
                
                var_original = np.var(series)
                var_trend = np.var(trend)
                var_seasonal = np.var(seasonal)
                var_residual = np.var(residual)
                
                if var_original > 0:
                    all_trend_strength.append(var_trend / var_original)
                    all_seasonal_strength.append(var_seasonal / var_original)
                    all_residual_strength.append(var_residual / var_original)
                
                trend_diff = np.diff(trend)
                if len(trend_diff) > 0:
                    all_trend_change_rate.append(np.mean(np.abs(trend_diff)))
                
                if save_series:
                    all_trend_series.append(trend.values)
                    all_seasonal_series.append(seasonal.values)
                    all_residual_series.append(residual.values)
                    
            except Exception as e:
                print(f"    [警告] STL 分解失败 {col}: {e}")
                continue
        
        if not all_trend_strength:
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
        
        metrics = {
            "trend_strength": float(np.mean(all_trend_strength)),
            "seasonal_strength": float(np.mean(all_seasonal_strength)),
            "residual_strength": float(np.mean(all_residual_strength)),
            "trend_change_rate": float(np.mean(all_trend_change_rate)) if all_trend_change_rate else 0.0,
        }
        
        details = {
            "n_series": len(all_trend_strength),
            "period": self.period,
        }
        
        if save_series and all_trend_series:
            min_len = min(len(s) for s in all_trend_series)
            avg_trend = np.mean([s[:min_len] for s in all_trend_series], axis=0)
            avg_seasonal = np.mean([s[:min_len] for s in all_seasonal_series], axis=0)
            avg_residual = np.mean([s[:min_len] for s in all_residual_series], axis=0)
            
            details["series"] = {
                "trend": avg_trend.tolist(),
                "seasonal": avg_seasonal.tolist(),
                "residual": avg_residual.tolist(),
                "length": min_len,
            }
        
        return {
            "analyzer": "STL",
            "dataset": dataset,
            "data_type": data_type,
            "method": method,
            "metrics": metrics,
            "details": details,
            "success": True,
        }


class FFTAnalyzer:
    """FFT 频域分析器"""
    
    def __init__(self, low_freq_ratio: float = 0.1):
        self.low_freq_ratio = low_freq_ratio
    
    def analyze(
        self,
        data: pd.DataFrame,
        dataset: str = "",
        data_type: str = "",
        method: str = "",
        save_series: bool = True,
    ) -> Dict[str, Any]:
        """
        执行 FFT 频域分析
        
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
        
        all_dominant_freq = []
        all_low_freq_energy_ratio = []
        all_high_freq_energy_ratio = []
        all_spectral_entropy = []
        
        all_freqs = []
        all_magnitudes = []
        
        for col in data_cols:
            series = data[col].dropna().values
            
            if len(series) < 10:
                continue
            
            try:
                series = series - np.mean(series)
                
                fft_result = np.fft.fft(series)
                fft_magnitude = np.abs(fft_result)
                
                n = len(series)
                freqs = np.fft.fftfreq(n)
                
                positive_mask = freqs > 0
                positive_freqs = freqs[positive_mask]
                positive_magnitude = fft_magnitude[positive_mask]
                
                if len(positive_magnitude) == 0:
                    continue
                
                dominant_idx = np.argmax(positive_magnitude)
                dominant_freq = positive_freqs[dominant_idx]
                all_dominant_freq.append(dominant_freq)
                
                total_energy = np.sum(positive_magnitude ** 2)
                if total_energy > 0:
                    energy = positive_magnitude ** 2
                    
                    low_freq_threshold = self.low_freq_ratio * np.max(positive_freqs)
                    low_freq_mask = positive_freqs <= low_freq_threshold
                    
                    low_freq_energy = np.sum(energy[low_freq_mask])
                    high_freq_energy = np.sum(energy[~low_freq_mask])
                    
                    all_low_freq_energy_ratio.append(low_freq_energy / total_energy)
                    all_high_freq_energy_ratio.append(high_freq_energy / total_energy)
                    
                    energy_normalized = energy / total_energy
                    energy_normalized = energy_normalized[energy_normalized > 0]
                    entropy = -np.sum(energy_normalized * np.log2(energy_normalized))
                    all_spectral_entropy.append(entropy)
                    
                    if save_series:
                        all_freqs.append(positive_freqs)
                        all_magnitudes.append(positive_magnitude)
                    
            except Exception as e:
                print(f"    [警告] FFT 分析失败 {col}: {e}")
                continue
        
        if not all_dominant_freq:
            return {
                "analyzer": "FFT",
                "dataset": dataset,
                "data_type": data_type,
                "method": method,
                "metrics": {},
                "details": {},
                "success": False,
                "error": "No valid series for FFT analysis",
            }
        
        metrics = {
            "dominant_freq": float(np.mean(all_dominant_freq)),
            "low_freq_energy_ratio": float(np.mean(all_low_freq_energy_ratio)),
            "high_freq_energy_ratio": float(np.mean(all_high_freq_energy_ratio)),
            "spectral_entropy": float(np.mean(all_spectral_entropy)),
        }
        
        details = {
            "n_series": len(all_dominant_freq),
            "low_freq_threshold_ratio": self.low_freq_ratio,
        }
        
        if save_series and all_freqs:
            min_len = min(len(f) for f in all_freqs)
            avg_freqs = np.mean([f[:min_len] for f in all_freqs], axis=0)
            avg_magnitudes = np.mean([m[:min_len] for m in all_magnitudes], axis=0)
            
            details["series"] = {
                "freqs": avg_freqs.tolist(),
                "magnitude": avg_magnitudes.tolist(),
                "length": min_len,
            }
        
        return {
            "analyzer": "FFT",
            "dataset": dataset,
            "data_type": data_type,
            "method": method,
            "metrics": metrics,
            "details": details,
            "success": True,
        }


class ACFAnalyzer:
    """ACF 自相关分析器"""
    
    def __init__(self, max_lag: int = 40):
        self.max_lag = max_lag
    
    def analyze(
        self,
        data: pd.DataFrame,
        dataset: str = "",
        data_type: str = "",
        method: str = "",
        save_series: bool = True,
    ) -> Dict[str, Any]:
        """
        执行 ACF 自相关分析
        
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
        
        lags = [10, 20, 30, 40]
        all_acf_values = {lag: [] for lag in lags}
        all_decay_rates = []
        all_acf_series = []
        
        for col in data_cols:
            series = data[col].dropna().values
            
            if len(series) < self.max_lag + 1:
                continue
            
            try:
                acf_values = acf(series, nlags=self.max_lag, fft=True)
                
                for lag in lags:
                    if lag < len(acf_values):
                        all_acf_values[lag].append(acf_values[lag])
                
                positive_acf = acf_values[acf_values > 0]
                if len(positive_acf) > 1:
                    x = np.arange(len(positive_acf))
                    log_acf = np.log(positive_acf + 1e-10)
                    slope = np.polyfit(x, log_acf, 1)[0]
                    all_decay_rates.append(-slope)
                
                if save_series:
                    all_acf_series.append(acf_values)
                    
            except Exception as e:
                print(f"    [警告] ACF 分析失败 {col}: {e}")
                continue
        
        if not all_decay_rates:
            return {
                "analyzer": "ACF",
                "dataset": dataset,
                "data_type": data_type,
                "method": method,
                "metrics": {},
                "details": {},
                "success": False,
                "error": "No valid series for ACF analysis",
            }
        
        metrics = {}
        for lag in lags:
            if all_acf_values[lag]:
                metrics[f"acf_lag_{lag}"] = float(np.mean(all_acf_values[lag]))
        
        metrics["decay_rate"] = float(np.mean(all_decay_rates))
        
        details = {
            "n_series": len(all_decay_rates),
            "max_lag": self.max_lag,
        }
        
        if save_series and all_acf_series:
            min_len = min(len(s) for s in all_acf_series)
            avg_acf = np.mean([s[:min_len] for s in all_acf_series], axis=0)
            
            details["series"] = {
                "acf": avg_acf.tolist(),
                "lags": list(range(min_len)),
                "length": min_len,
            }
        
        return {
            "analyzer": "ACF",
            "dataset": dataset,
            "data_type": data_type,
            "method": method,
            "metrics": metrics,
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
    header_line = f"{'指标':<15} {'Ori':<{col_width}} {'Missing':<{col_width}}"
    for method in imputed_methods:
        header_line += f" {method:<{col_width}}"
    print(header_line)
    print("-" * 80)
    
    for metric_name, metric_data in metrics.items():
        ori_val = metric_data.get("ori", 0)
        missing_val = metric_data.get("missing", "-")
        
        line = f"{metric_name:<15} {ori_val:<{col_width}.4f}"
        
        if missing_val != "-":
            line += f" {missing_val:<{col_width}.4f}"
        else:
            line += f" {'-':<{col_width}}"
        
        for method in imputed_methods:
            val = metric_data.get("imputed", {}).get(method, "-")
            if val != "-":
                line += f" {val:<{col_width}.4f}"
            else:
                line += f" {'-':<{col_width}}"
        
        print(line)
    
    print("-" * 80)
    print("与Ori的差异 (%):")
    
    for metric_name, metric_data in metrics.items():
        ori_val = metric_data.get("ori", 0)
        missing_diff = metric_data.get("missing_diff_pct", "-")
        
        line = f"{metric_name:<15} {'-':<{col_width}}"
        
        if missing_diff != "-":
            line += f" {missing_diff:>+{col_width-2}.1f}%"
        else:
            line += f" {'-':<{col_width}}"
        
        for method in imputed_methods:
            diff = metric_data.get("imputed_diff_pct", {}).get(method, "-")
            if diff != "-":
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
