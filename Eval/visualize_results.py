"""
评估结果可视化脚本

功能：
1. 读取指定数据集和注空模式的所有结果文件
2. 按 term 分组整合数据
3. 绘制指标随缺失率变化的折线图
"""

import os
import re
import csv
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 常量定义
# ============================================================================

# 原始数据集列表
ORIGINAL_DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "exchange_rate", "electricity", "traffic",
    "weather", "national_illness"
]

# 注空模式
MISSING_METHODS = ["MCAR", "BM", "TM", "TVMR"]

# 缺失率
MISSING_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MISSING_RATIO_CODES = ["005", "010", "015", "020", "025", "030"]

# term 类型
TERMS = ["short", "medium", "long"]

# 默认指标
DEFAULT_METRICS = ["MSE[mean]", "MAE[0.5]", "MASE[0.5]"]

# 结果目录
RESULTS_DIR = "results/sundial/sundial_Missing"
OUTPUT_DIR = "results/sundial/visualization"


# ============================================================================
# 工具函数
# ============================================================================

def parse_result_filename(filename: str) -> Optional[Tuple[str, str, str, str]]:
    """
    解析结果文件名
    
    Args:
        filename: 结果文件名，如 ETTh1_MCAR_005_short_results.csv
        
    Returns:
        (dataset, method, ratio, term) 元组，如果解析失败则返回 None
    """
    pattern = r'^([^.]+?)_(MCAR|BM|TM|TVMR)_(\d{3})_(short|medium|long)_results\.csv$'
    match = re.match(pattern, filename)
    
    if match:
        dataset = match.group(1)
        method = match.group(2)
        ratio_code = match.group(3)
        term = match.group(4)
        
        # 将缺失率编码转换为浮点数（005 -> 0.05, 010 -> 0.10, etc.）
        ratio = int(ratio_code) / 100.0
        
        return dataset, method, ratio, term
    
    return None


def read_result_file(file_path: str) -> Dict:
    """
    读取结果文件
    
    Args:
        file_path: 结果文件路径
        
    Returns:
        包含所有指标的字典
    """
    results = {}
    
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row['metric']
            value = row['value']
            
            # 尝试转换为浮点数
            try:
                results[metric] = float(value)
            except ValueError:
                results[metric] = value
    
    return results


def collect_result_files(
    dataset: str,
    method: str,
    results_dir: str = RESULTS_DIR
) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    收集指定数据集和注空模式的所有结果文件
    
    Args:
        dataset: 数据集名称
        method: 注空模式
        results_dir: 结果目录
        
    Returns:
        嵌套字典：{term: {ratio_code: file_path}}
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    # 按 term 和 ratio 组织文件
    organized = {term: {} for term in TERMS}
    
    for filename in os.listdir(results_path):
        if not filename.endswith('_results.csv'):
            continue
        
        parsed = parse_result_filename(filename)
        if parsed is None:
            continue
        
        ds, mth, ratio, term = parsed
        
        # 只收集匹配的文件
        if ds == dataset and mth == method:
            # 使用原始解析的 ratio 作为 key，而不是重新计算
            ratio_key = ratio
            file_path = results_path / filename
            organized[term][ratio_key] = str(file_path)
    
    return organized


def load_all_results(organized_files: Dict[str, Dict[float, str]]) -> Dict[str, Dict[float, Dict]]:
    """
    加载所有结果文件的数据
    
    Args:
        organized_files: 组织好的文件路径字典 {term: {ratio: file_path}}
        
    Returns:
        按 term 和 ratio 组织的数据 {term: {ratio: results_dict}}
    """
    data = {}
    
    for term, ratio_files in organized_files.items():
        data[term] = {}
        
        for ratio, file_path in ratio_files.items():
            try:
                results = read_result_file(file_path)
                data[term][ratio] = results
                print(f"  ✓ Loaded: {Path(file_path).name}")
            except Exception as e:
                print(f"  ✗ Error loading {file_path}: {e}")
    
    return data


# ============================================================================
# 可视化函数
# ============================================================================

def plot_metrics(
    data: Dict[str, Dict[float, Dict]],
    dataset: str,
    method: str,
    metrics: List[str] = DEFAULT_METRICS,
    output_dir: str = OUTPUT_DIR
):
    """
    绘制指标随缺失率变化的折线图
    
    Args:
        data: 按 term 和 ratio 组织的数据
        dataset: 数据集名称
        method: 注空模式
        metrics: 要绘制的指标列表
        output_dir: 输出目录
    """
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 为每个 term 创建一个子图
    n_terms = len([t for t in TERMS if t in data and data[t]])
    
    if n_terms == 0:
        print("  ⚠️  No data available for plotting")
        return
    
    fig, axes = plt.subplots(n_terms, len(metrics), figsize=(5*len(metrics), 4*n_terms))
    
    # 如果是单行或单列，需要特殊处理
    if n_terms == 1 and len(metrics) == 1:
        axes = [[axes]]
    elif n_terms == 1:
        axes = [axes]
    elif len(metrics) == 1:
        axes = [[ax] for ax in axes]
    
    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(MISSING_RATIOS)))
    
    for term_idx, term in enumerate(TERMS):
        if term not in data or not data[term]:
            continue
        
        term_data = data[term]
        ratios = sorted(term_data.keys())
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[term_idx][metric_idx]
            
            # 提取数据
            x = ratios
            y = [term_data[ratio].get(metric, None) for ratio in ratios]
            
            # 过滤掉 None 值
            valid_points = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
            if valid_points:
                x_valid, y_valid = zip(*valid_points)
                
                # 绘制折线图
                ax.plot(x_valid, y_valid, 'o-', linewidth=2, markersize=6, color='steelblue')
                
                # 添加数据标签
                for xi, yi in valid_points:
                    ax.annotate(f'{yi:.4f}', xy=(xi, yi), xytext=(0, 5),
                               textcoords='offset points', ha='center', fontsize=8)
            
            # 设置标签和标题
            ax.set_xlabel('Missing Ratio', fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f'{metric} ({term})', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(ratios)
            ax.set_xticklabels([f'{r:.0%}' for r in ratios], rotation=0)
            
            # 自动调整 y 轴范围，避免折线图挤在顶部或底部
            if valid_points:
                y_min = min(y_valid)
                y_max = max(y_valid)
                y_range = y_max - y_min
                
                # 如果数据范围很小，设置一个最小范围
                if y_range < 0.1:
                    y_range = 0.1
                
                # 设置 y 轴上下边距（各扩展 10% 的范围）
                margin = y_range * 0.1
                ax.set_ylim(bottom=max(0, y_min - margin), top=y_max + margin)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = output_path / f"{dataset}_{method}_metrics.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_file}")
    
    plt.close()


def plot_all_terms_combined(
    data: Dict[str, Dict[float, Dict]],
    dataset: str,
    method: str,
    metrics: List[str] = DEFAULT_METRICS,
    output_dir: str = OUTPUT_DIR
):
    """
    将所有 term 绘制在同一张图上（不同颜色）
    
    Args:
        data: 按 term 和 ratio 组织的数据
        dataset: 数据集名称
        method: 注空模式
        metrics: 要绘制的指标列表
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 为每个指标创建一张图
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 为每个 term 绘制折线
        colors = {'short': 'red', 'medium': 'green', 'long': 'blue'}
        markers = {'short': 'o', 'medium': 's', 'long': '^'}
        
        has_data = False
        for term in TERMS:
            if term not in data or not data[term]:
                continue
            
            term_data = data[term]
            ratios = sorted(term_data.keys())
            
            x = ratios
            y = [term_data[ratio].get(metric, None) for ratio in ratios]
            
            # 过滤掉 None 值
            valid_points = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
            if valid_points:
                x_valid, y_valid = zip(*valid_points)
                ax.plot(x_valid, y_valid, 
                       marker=markers.get(term, 'o'), 
                       linewidth=2, markersize=6,
                       color=colors.get(term, 'gray'),
                       label=term)
                has_data = True
        
        if has_data:
            ax.set_xlabel('Missing Ratio', fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f'{metric}\n{dataset} - {method}', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(title='Term', loc='best')
            ax.set_xticks(MISSING_RATIOS)
            ax.set_xticklabels([f'{r:.0%}' for r in MISSING_RATIOS])
            
            # 自动调整 y 轴范围，避免折线图挤在顶部或底部
            # 收集所有 term 的数据来计算合适的范围
            all_y_values = []
            for term in TERMS:
                if term not in data or not data[term]:
                    continue
                term_data = data[term]
                y_values = [term_data[ratio].get(metric, None) for ratio in sorted(term_data.keys())]
                valid_y = [y for y in y_values if y is not None]
                all_y_values.extend(valid_y)
            
            if all_y_values:
                y_min = min(all_y_values)
                y_max = max(all_y_values)
                y_range = y_max - y_min
                
                # 如果数据范围很小，设置一个最小范围
                if y_range < 0.1:
                    y_range = 0.1
                
                # 设置 y 轴上下边距（各扩展 10% 的范围）
                margin = y_range * 0.1
                ax.set_ylim(bottom=max(0, y_min - margin), top=y_max + margin)
            
            plt.tight_layout()
            
            output_file = output_path / f"{dataset}_{method}_{metric.replace('[', '_').replace(']', '_')}_all_terms.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {output_file}")
        
        plt.close()


def create_summary_table(
    data: Dict[str, Dict[float, Dict]],
    dataset: str,
    method: str,
    metrics: List[str] = DEFAULT_METRICS,
    output_dir: str = OUTPUT_DIR
):
    """
    创建汇总表格
    
    Args:
        data: 按 term 和 ratio 组织的数据
        dataset: 数据集名称
        method: 注空模式
        metrics: 要包含的指标
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建 CSV 文件
    output_file = output_path / f"{dataset}_{method}_summary.csv"
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 表头
        header = ['Term', 'Missing Ratio'] + metrics
        writer.writerow(header)
        
        # 数据行
        for term in TERMS:
            if term not in data or not data[term]:
                continue
            
            for ratio in sorted(data[term].keys()):
                # 将缺失率转换为百分比格式 (5%, 10%, 15% 等)
                ratio_pct = int(ratio * 100)
                row = [term, f'{ratio_pct}%']
                
                for metric in metrics:
                    value = data[term][ratio].get(metric, 'N/A')
                    if isinstance(value, float):
                        row.append(f'{value:.6f}')
                    else:
                        row.append(str(value))
                
                writer.writerow(row)
    
    print(f"✓ Summary table saved to: {output_file}")


# ============================================================================
# 主函数
# ============================================================================

def visualize(
    dataset: str,
    method: str,
    metrics: List[str] = DEFAULT_METRICS,
    results_dir: str = RESULTS_DIR,
    output_dir: str = OUTPUT_DIR,
    plot_mode: str = "both"
):
    """
    可视化评估结果
    
    Args:
        dataset: 数据集名称
        method: 注空模式
        metrics: 要绘制的指标列表
        results_dir: 结果目录
        output_dir: 输出目录
        plot_mode: 绘图模式 ("separate", "combined", "both")
    """
    print(f"\n{'='*80}")
    print(f"Visualization")
    print(f"{'='*80}")
    print(f"  Dataset: {dataset}")
    print(f"  Method: {method}")
    print(f"  Metrics: {metrics}")
    print(f"  Results dir: {results_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*80}")
    
    # Step 1: 收集结果文件
    print(f"\nStep 1: Collecting result files")
    organized_files = collect_result_files(dataset, method, results_dir)
    
    # 统计文件数量
    total_files = sum(len(files) for files in organized_files.values())
    print(f"  Found {total_files} result files")
    
    if total_files == 0:
        print(f"  ⚠️  No result files found for {dataset} + {method}")
        return
    
    # Step 2: 加载数据
    print(f"\nStep 2: Loading data")
    data = load_all_results(organized_files)
    
    # Step 3: 创建汇总表格
    print(f"\nStep 3: Creating summary table")
    create_summary_table(data, dataset, method, metrics, output_dir)
    
    # Step 4: 绘图
    print(f"\nStep 4: Plotting")
    
    if plot_mode in ["separate", "both"]:
        print(f"  Creating separate term plots...")
        plot_metrics(data, dataset, method, metrics, output_dir)
    
    if plot_mode in ["combined", "both"]:
        print(f"  Creating combined term plots...")
        plot_all_terms_combined(data, dataset, method, metrics, output_dir)
    
    print(f"\n{'='*80}")
    print(f"Visualization Complete")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (default metrics: MSE[mean], MAE[0.5], MASE[0.5])
  python Eval/visualize_results.py --dataset ETTh1 --method MCAR
  
  # Custom metrics
  python Eval/visualize_results.py --dataset ETTh1 --method MCAR \\
    --metrics "MSE[mean]" "MAE[0.5]" "MAPE[0.5]"
  
  # Only separate plots (one plot per term)
  python Eval/visualize_results.py --dataset ETTh1 --method MCAR --plot_mode separate
  
  # Only combined plots (all terms in one plot)
  python Eval/visualize_results.py --dataset ETTh1 --method MCAR --plot_mode combined
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=ORIGINAL_DATASETS,
        help="Dataset name"
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=MISSING_METHODS,
        help="Missing pattern method"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        default=DEFAULT_METRICS,
        help=f"Metrics to plot (default: {DEFAULT_METRICS})"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default=RESULTS_DIR,
        help="Results directory (default: results/sundial/sundial_Missing)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Output directory (default: results/sundial/visualization)"
    )
    
    parser.add_argument(
        "--plot_mode",
        type=str,
        choices=["separate", "combined", "both"],
        default="both",
        help="Plot mode: separate (one plot per term), combined (all terms in one plot), or both"
    )
    
    args = parser.parse_args()
    
    visualize(
        dataset=args.dataset,
        method=args.method,
        metrics=args.metrics,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        plot_mode=args.plot_mode
    )


if __name__ == "__main__":
    main()
