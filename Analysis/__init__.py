"""
时间序列分析模块

包含：
- metrics: 6 个时间序列特征指标计算
- window_analysis: 窗口特征分析（预测窗口 + 历史窗口）
- batch_window_analysis: 批量窗口特征分析
"""

from Analysis.metrics import (
    calculate_trend_strength,
    calculate_trend_linearity,
    calculate_seasonal_strength,
    calculate_seasonal_correlation,
    calculate_residual_autocorr_lag1,
    calculate_spectral_entropy,
    calculate_all_metrics,
    get_period,
    load_dataset_properties,
)

from Analysis.window_analysis import (
    analyze_prediction_windows,
    analyze_history_windows,
    analyze_imputed_history_windows,
    analyze_single_window,
    get_available_impute_methods,
    save_results,
)

__all__ = [
    "calculate_trend_strength",
    "calculate_trend_linearity",
    "calculate_seasonal_strength",
    "calculate_seasonal_correlation",
    "calculate_residual_autocorr_lag1",
    "calculate_spectral_entropy",
    "calculate_all_metrics",
    "get_period",
    "load_dataset_properties",
    "analyze_prediction_windows",
    "analyze_history_windows",
    "analyze_imputed_history_windows",
    "analyze_single_window",
    "get_available_impute_methods",
    "save_results",
    "run_batch_analysis",
    "get_all_prediction_dirs",
    "get_datasets_with_predictions",
]
