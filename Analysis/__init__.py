"""
时间序列分析模块

包含 STL 分解、FFT 频域分析、ACF 自相关分析
"""

from Analysis.analyzer import (
    STLAnalyzer,
    FFTAnalyzer,
    ACFAnalyzer,
    load_dataset_properties,
    get_period,
    compare_results,
    print_comparison,
    calculate_recovery_score,
    get_best_method,
)

from Analysis.run_analysis import (
    run_single_analysis,
    run_batch_analysis,
    generate_summary_report,
)

from Analysis.visualize import (
    visualize_single_result,
    visualize_all_results,
    plot_stl_comparison,
    plot_fft_comparison,
    plot_acf_comparison,
    plot_recovery_comparison,
)

__all__ = [
    "STLAnalyzer",
    "FFTAnalyzer",
    "ACFAnalyzer",
    "load_dataset_properties",
    "get_period",
    "compare_results",
    "print_comparison",
    "calculate_recovery_score",
    "get_best_method",
    "run_single_analysis",
    "run_batch_analysis",
    "generate_summary_report",
    "visualize_single_result",
    "visualize_all_results",
    "plot_stl_comparison",
    "plot_fft_comparison",
    "plot_acf_comparison",
    "plot_recovery_comparison",
]
