"""
分析结果可视化模块

读取 JSON 结果文件，生成对比图表
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
from matplotlib.font_manager import FontProperties

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Analysis.analyzer import (
    load_dataset_properties,
    get_period,
    get_data_cols,
)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_analysis_result(json_path: str) -> Dict[str, Any]:
    """加载分析结果 JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_stl_comparison(
    result: Dict[str, Any],
    output_path: str,
    figsize: tuple = (14, 10),
):
    """
    绘制 STL 分析对比图
    
    Args:
        result: 分析结果字典
        output_path: 输出路径
        figsize: 图表大小
    """
    if "comparison" not in result or "STL" not in result["comparison"]:
        print("  [跳过] 无 STL 分析结果")
        return
    
    comparison = result["comparison"]["STL"]
    metrics = comparison.get("metrics", {})
    
    if not metrics:
        print("  [跳过] 无 STL 指标数据")
        return
    
    has_series = False
    ori_series = result.get("ori", {}).get("STL", {}).get("details", {}).get("series")
    missing_series = result.get("missing", {}).get("STL", {}).get("details", {}).get("series")
    imputed_series = {}
    for method, method_result in result.get("imputed", {}).items():
        if "STL" in method_result and method_result["STL"].get("details", {}).get("series"):
            imputed_series[method] = method_result["STL"]["details"]["series"]
    
    if ori_series:
        has_series = True
    
    if has_series:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"STL 分解对比 - {result.get('dataset', '')} MCAR_{result.get('ratio', '')} {result.get('term', '')}", 
                     fontsize=14, fontweight='bold')
        
        colors = {'ori': '#2ecc71', 'missing': '#e74c3c'}
        method_colors = {'zero': '#3498db', 'forward': '#9b59b6', 'backward': '#e67e22', 
                         'mean': '#1abc9c', 'linear': '#f39c12'}
        
        components = ['trend', 'seasonal', 'residual']
        titles = ['趋势', '季节', '残差']
        
        for idx, (comp, title) in enumerate(zip(components, titles)):
            ax = axes[idx // 2, idx % 2]
            
            if ori_series and comp in ori_series:
                x = range(len(ori_series[comp]))
                ax.plot(x, ori_series[comp], color=colors['ori'], label='Ori', linewidth=2, alpha=0.8)
            
            if missing_series and comp in missing_series:
                x = range(len(missing_series[comp]))
                ax.plot(x, missing_series[comp], color=colors['missing'], label='Missing', linewidth=1.5, alpha=0.7)
            
            for method, series_data in imputed_series.items():
                if comp in series_data:
                    x = range(len(series_data[comp]))
                    ax.plot(x, series_data[comp], color=method_colors.get(method, '#95a5a6'), 
                           label=method.capitalize(), linewidth=1, alpha=0.6, linestyle='--')
            
            ax.set_title(title, fontsize=12)
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ax.axis('off')
        
        methods = list(set(
            method 
            for m in metrics.values() 
            for method in m.get("imputed", {}).keys()
        ))
        
        metric_names = ['trend_strength', 'seasonal_strength', 'residual_strength']
        metric_labels = ['趋势强度', '季节强度', '残差强度']
        
        y_pos = 0.9
        ax.text(0.1, y_pos, '指标对比:', fontsize=11, fontweight='bold')
        y_pos -= 0.08
        
        for metric_name, metric_label in zip(metric_names, metric_labels):
            if metric_name in metrics:
                metric_data = metrics[metric_name]
                ori_val = metric_data.get('ori', 0)
                text = f"{metric_label}: Ori={ori_val:.4f}"
                
                missing_val = metric_data.get('missing')
                if missing_val is not None:
                    text += f", Missing={missing_val:.4f}"
                
                ax.text(0.1, y_pos, text, fontsize=9)
                y_pos -= 0.06
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [保存] {output_path}")
    else:
        methods = list(set(
            method 
            for m in metrics.values() 
            for method in m.get("imputed", {}).keys()
        ))
        
        metric_names = list(metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"STL 分析对比 - {result.get('dataset', '')} MCAR_{result.get('ratio', '')} {result.get('term', '')}", 
                     fontsize=14, fontweight='bold')
        
        colors = {'ori': '#2ecc71', 'missing': '#e74c3c'}
        method_colors = {'zero': '#3498db', 'forward': '#9b59b6', 'backward': '#e67e22', 
                         'mean': '#1abc9c', 'linear': '#f39c12'}
        
        for idx, metric_name in enumerate(metric_names[:4]):
            ax = axes[idx // 2, idx % 2]
            
            metric_data = metrics[metric_name]
            
            labels = ['Ori', 'Missing'] + [m.capitalize() for m in methods]
            values = [metric_data.get('ori', 0), metric_data.get('missing', 0)]
            colors_list = [colors['ori'], colors['missing']]
            
            for method in methods:
                values.append(metric_data.get('imputed', {}).get(method, 0))
                colors_list.append(method_colors.get(method, '#95a5a6'))
            
            bars = ax.bar(labels, values, color=colors_list, edgecolor='black', linewidth=0.5)
            
            ax.set_title(metric_name, fontsize=12)
            ax.set_ylabel('Value', fontsize=10)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [保存] {output_path}")


def plot_fft_comparison(
    result: Dict[str, Any],
    output_path: str,
    figsize: tuple = (14, 10),
):
    """
    绘制 FFT 分析对比图
    
    Args:
        result: 分析结果字典
        output_path: 输出路径
        figsize: 图表大小
    """
    if "comparison" not in result or "FFT" not in result["comparison"]:
        print("  [跳过] 无 FFT 分析结果")
        return
    
    comparison = result["comparison"]["FFT"]
    metrics = comparison.get("metrics", {})
    
    if not metrics:
        print("  [跳过] 无 FFT 指标数据")
        return
    
    has_series = False
    ori_series = result.get("ori", {}).get("FFT", {}).get("details", {}).get("series")
    missing_series = result.get("missing", {}).get("FFT", {}).get("details", {}).get("series")
    imputed_series = {}
    for method, method_result in result.get("imputed", {}).items():
        if "FFT" in method_result and method_result["FFT"].get("details", {}).get("series"):
            imputed_series[method] = method_result["FFT"]["details"]["series"]
    
    if ori_series:
        has_series = True
    
    if has_series:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"FFT 频谱对比 - {result.get('dataset', '')} MCAR_{result.get('ratio', '')} {result.get('term', '')}", 
                     fontsize=14, fontweight='bold')
        
        colors = {'ori': '#2ecc71', 'missing': '#e74c3c'}
        method_colors = {'zero': '#3498db', 'forward': '#9b59b6', 'backward': '#e67e22', 
                         'mean': '#1abc9c', 'linear': '#f39c12'}
        
        ax = axes[0]
        
        if ori_series and 'freqs' in ori_series and 'magnitude' in ori_series:
            ax.plot(ori_series['freqs'], ori_series['magnitude'], color=colors['ori'], 
                   label='Ori', linewidth=2, alpha=0.8)
        
        if missing_series and 'freqs' in missing_series and 'magnitude' in missing_series:
            ax.plot(missing_series['freqs'], missing_series['magnitude'], color=colors['missing'], 
                   label='Missing', linewidth=1.5, alpha=0.7)
        
        for method, series_data in imputed_series.items():
            if 'freqs' in series_data and 'magnitude' in series_data:
                ax.plot(series_data['freqs'], series_data['magnitude'], 
                       color=method_colors.get(method, '#95a5a6'), 
                       label=method.capitalize(), linewidth=1, alpha=0.6, linestyle='--')
        
        ax.set_xlabel('Frequency', fontsize=11)
        ax.set_ylabel('Magnitude', fontsize=11)
        ax.set_title('频谱曲线对比', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        methods = list(set(
            method 
            for m in metrics.values() 
            for method in m.get("imputed", {}).keys()
        ))
        
        metric_names = ['dominant_freq', 'low_freq_energy_ratio', 'spectral_entropy']
        metric_labels = ['主频率', '低频能量比', '频谱熵']
        
        y_pos = 0.9
        ax.text(0.1, y_pos, '指标对比:', fontsize=11, fontweight='bold')
        y_pos -= 0.08
        
        for metric_name, metric_label in zip(metric_names, metric_labels):
            if metric_name in metrics:
                metric_data = metrics[metric_name]
                ori_val = metric_data.get('ori', 0)
                text = f"{metric_label}: Ori={ori_val:.4f}"
                
                missing_val = metric_data.get('missing')
                if missing_val is not None:
                    text += f", Missing={missing_val:.4f}"
                
                ax.text(0.1, y_pos, text, fontsize=9)
                y_pos -= 0.06
        
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [保存] {output_path}")
    else:
        methods = list(set(
            method 
            for m in metrics.values() 
            for method in m.get("imputed", {}).keys()
        ))
        
        metric_names = list(metrics.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"FFT 分析对比 - {result.get('dataset', '')} MCAR_{result.get('ratio', '')} {result.get('term', '')}", 
                     fontsize=14, fontweight='bold')
        
        colors = {'ori': '#2ecc71', 'missing': '#e74c3c'}
        method_colors = {'zero': '#3498db', 'forward': '#9b59b6', 'backward': '#e67e22', 
                         'mean': '#1abc9c', 'linear': '#f39c12'}
        
        for idx, metric_name in enumerate(metric_names[:4]):
            ax = axes[idx // 2, idx % 2]
            
            metric_data = metrics[metric_name]
            
            labels = ['Ori', 'Missing'] + [m.capitalize() for m in methods]
            values = [metric_data.get('ori', 0), metric_data.get('missing', 0)]
            colors_list = [colors['ori'], colors['missing']]
            
            for method in methods:
                values.append(metric_data.get('imputed', {}).get(method, 0))
                colors_list.append(method_colors.get(method, '#95a5a6'))
            
            bars = ax.bar(labels, values, color=colors_list, edgecolor='black', linewidth=0.5)
            
            ax.set_title(metric_name, fontsize=12)
            ax.set_ylabel('Value', fontsize=10)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.4f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [保存] {output_path}")


def plot_acf_comparison(
    result: Dict[str, Any],
    output_path: str,
    figsize: tuple = (14, 10),
):
    """
    绘制 ACF 分析对比图
    
    Args:
        result: 分析结果字典
        output_path: 输出路径
        figsize: 图表大小
    """
    if "comparison" not in result or "ACF" not in result["comparison"]:
        print("  [跳过] 无 ACF 分析结果")
        return
    
    comparison = result["comparison"]["ACF"]
    metrics = comparison.get("metrics", {})
    
    if not metrics:
        print("  [跳过] 无 ACF 指标数据")
        return
    
    has_series = False
    ori_series = result.get("ori", {}).get("ACF", {}).get("details", {}).get("series")
    missing_series = result.get("missing", {}).get("ACF", {}).get("details", {}).get("series")
    imputed_series = {}
    for method, method_result in result.get("imputed", {}).items():
        if "ACF" in method_result and method_result["ACF"].get("details", {}).get("series"):
            imputed_series[method] = method_result["ACF"]["details"]["series"]
    
    if ori_series:
        has_series = True
    
    colors = {'ori': '#2ecc71', 'missing': '#e74c3c'}
    method_colors = {'zero': '#3498db', 'forward': '#9b59b6', 'backward': '#e67e22', 
                     'mean': '#1abc9c', 'linear': '#f39c12'}
    
    if has_series:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"ACF 自相关对比 - {result.get('dataset', '')} MCAR_{result.get('ratio', '')} {result.get('term', '')}", 
                     fontsize=14, fontweight='bold')
        
        ax = axes[0]
        
        if ori_series and 'acf' in ori_series and 'lags' in ori_series:
            ax.plot(ori_series['lags'], ori_series['acf'], color=colors['ori'], 
                   label='Ori', linewidth=2, alpha=0.8)
        
        if missing_series and 'acf' in missing_series and 'lags' in missing_series:
            ax.plot(missing_series['lags'], missing_series['acf'], color=colors['missing'], 
                   label='Missing', linewidth=1.5, alpha=0.7)
        
        for method, series_data in imputed_series.items():
            if 'acf' in series_data and 'lags' in series_data:
                ax.plot(series_data['lags'], series_data['acf'], 
                       color=method_colors.get(method, '#95a5a6'), 
                       label=method.capitalize(), linewidth=1, alpha=0.6, linestyle='--')
        
        ax.set_xlabel('Lag', fontsize=11)
        ax.set_ylabel('ACF Value', fontsize=11)
        ax.set_title('自相关函数曲线', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        ax = axes[1]
        methods = list(set(
            method 
            for m in metrics.values() 
            for method in m.get("imputed", {}).keys()
        ))
        
        if 'decay_rate' in metrics:
            decay_data = metrics['decay_rate']
            
            labels = ['Ori', 'Missing'] + [m.capitalize() for m in methods]
            values = [decay_data.get('ori', 0), decay_data.get('missing', 0)]
            colors_list = [colors['ori'], colors['missing']]
            
            for method in methods:
                values.append(decay_data.get('imputed', {}).get(method, 0))
                colors_list.append(method_colors.get(method, '#95a5a6'))
            
            bars = ax.bar(labels, values, color=colors_list, edgecolor='black', linewidth=0.5)
            
            ax.set_title('衰减速率对比', fontsize=12)
            ax.set_ylabel('Decay Rate', fontsize=10)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [保存] {output_path}")
    else:
        methods = list(set(
            method 
            for m in metrics.values() 
            for method in m.get("imputed", {}).keys()
        ))
        
        acf_metrics = {k: v for k, v in metrics.items() if k.startswith('acf_lag_')}
        
        if not acf_metrics:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"ACF 分析对比 - {result.get('dataset', '')} MCAR_{result.get('ratio', '')} {result.get('term', '')}", 
                     fontsize=14, fontweight='bold')
        
        ax1 = axes[0]
        lags = sorted([int(k.replace('acf_lag_', '')) for k in acf_metrics.keys()])
        
        ori_values = [acf_metrics[f'acf_lag_{lag}']['ori'] for lag in lags]
        missing_values = [acf_metrics[f'acf_lag_{lag}']['missing'] for lag in lags]
        
        ax1.plot(lags, ori_values, 'o-', color=colors['ori'], label='Ori', linewidth=2, markersize=8)
        ax1.plot(lags, missing_values, 'o-', color=colors['missing'], label='Missing', linewidth=2, markersize=8)
        
        for method in methods:
            method_values = [acf_metrics[f'acf_lag_{lag}']['imputed'].get(method, 0) for lag in lags]
            ax1.plot(lags, method_values, 'o--', color=method_colors.get(method, '#95a5a6'), 
                    label=method.capitalize(), linewidth=1.5, markersize=6, alpha=0.8)
        
        ax1.set_xlabel('Lag', fontsize=11)
        ax1.set_ylabel('ACF Value', fontsize=11)
        ax1.set_title('自相关系数对比', fontsize=12)
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.1, 1.1)
        
        ax2 = axes[1]
        
        if 'decay_rate' in metrics:
            decay_data = metrics['decay_rate']
            
            labels = ['Ori', 'Missing'] + [m.capitalize() for m in methods]
            values = [decay_data.get('ori', 0), decay_data.get('missing', 0)]
            colors_list = [colors['ori'], colors['missing']]
            
            for method in methods:
                values.append(decay_data.get('imputed', {}).get(method, 0))
                colors_list.append(method_colors.get(method, '#95a5a6'))
            
            bars = ax2.bar(labels, values, color=colors_list, edgecolor='black', linewidth=0.5)
            
            ax2.set_title('衰减速率对比', fontsize=12)
            ax2.set_ylabel('Decay Rate', fontsize=10)
            
            for bar, val in zip(bars, values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [保存] {output_path}")


def plot_recovery_comparison(
    result: Dict[str, Any],
    output_path: str,
    figsize: tuple = (12, 8),
):
    """
    绘制恢复率对比图
    
    Args:
        result: 分析结果字典
        output_path: 输出路径
        figsize: 图表大小
    """
    if "comparison" not in result:
        print("  [跳过] 无比较结果")
        return
    
    methods = set()
    for analyzer in ["STL", "FFT", "ACF"]:
        if analyzer in result["comparison"]:
            comparison = result["comparison"][analyzer]
            for metric_data in comparison.get("metrics", {}).values():
                methods.update(metric_data.get("imputed", {}).keys())
    
    methods = sorted(list(methods))
    
    if not methods:
        print("  [跳过] 无填补方法数据")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"恢复率对比 - {result.get('dataset', '')} MCAR_{result.get('ratio', '')} {result.get('term', '')}", 
                 fontsize=14, fontweight='bold')
    
    method_colors = {'zero': '#3498db', 'forward': '#9b59b6', 'backward': '#e67e22', 
                     'mean': '#1abc9c', 'linear': '#f39c12'}
    
    for idx, analyzer in enumerate(["STL", "FFT", "ACF"]):
        ax = axes[idx]
        
        if analyzer not in result["comparison"]:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(analyzer, fontsize=12)
            continue
        
        comparison = result["comparison"][analyzer]
        
        recovery_scores = {}
        for method in methods:
            scores = []
            for metric_data in comparison.get("metrics", {}).values():
                recovery = metric_data.get("imputed_recovery_pct", {}).get(method)
                if recovery is not None:
                    scores.append(recovery)
            if scores:
                recovery_scores[method] = np.mean(scores)
        
        if recovery_scores:
            labels = [m.capitalize() for m in recovery_scores.keys()]
            values = list(recovery_scores.values())
            colors_list = [method_colors.get(m, '#95a5a6') for m in recovery_scores.keys()]
            
            bars = ax.bar(labels, values, color=colors_list, edgecolor='black', linewidth=0.5)
            
            ax.set_title(analyzer, fontsize=12)
            ax.set_ylabel('Recovery %', fontsize=10)
            ax.set_ylim(0, 105)
            ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Perfect')
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(analyzer, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [保存] {output_path}")


def visualize_single_result(
    json_path: str,
    output_dir: str = None,
):
    """
    对单个分析结果生成可视化
    
    Args:
        json_path: JSON 结果文件路径
        output_dir: 输出目录
    """
    result = load_analysis_result(json_path)
    
    if output_dir is None:
        output_dir = Path(json_path).parent
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    base_name = Path(json_path).stem
    
    print(f"\n生成可视化: {base_name}")
    
    plot_stl_comparison(result, str(output_path / f"{base_name}_stl.png"))
    plot_fft_comparison(result, str(output_path / f"{base_name}_fft.png"))
    plot_acf_comparison(result, str(output_path / f"{base_name}_acf.png"))
    plot_recovery_comparison(result, str(output_path / f"{base_name}_recovery.png"))


def visualize_all_results(
    input_dir: str = "results/analysis",
    output_dir: str = None,
):
    """
    对所有分析结果生成可视化
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_dir
    
    json_files = list(input_path.glob("*_analysis.json"))
    
    if not json_files:
        print(f"未找到分析结果文件: {input_path}")
        return
    
    print(f"找到 {len(json_files)} 个分析结果文件")
    
    for json_file in json_files:
        visualize_single_result(str(json_file), output_dir)


def main():
    parser = argparse.ArgumentParser(description="分析结果可视化")
    
    parser.add_argument("--input", type=str, default=None,
                        help="单个 JSON 文件路径")
    parser.add_argument("--input_dir", type=str, default="results/analysis",
                        help="输入目录 (默认: results/analysis)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: 与输入相同)")
    
    args = parser.parse_args()
    
    if args.input:
        visualize_single_result(args.input, args.output_dir)
    else:
        visualize_all_results(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
