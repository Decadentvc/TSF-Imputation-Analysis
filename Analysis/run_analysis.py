"""
批量分析脚本

对干净数据、缺失数据和填补数据进行 STL、FFT、ACF 分析
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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

from Missing_Value_Injection.for_sundial.inject_range_utils import (
    get_injection_range as compute_inject_range,
)


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "exchange_rate", "national_illness", "traffic", "weather"]

DATASETS_SHORT_ONLY = ["national_illness"]

MISSING_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

IMPUTATION_METHODS = ["zero", "forward", "backward", "mean", "linear"]


def get_terms_for_dataset(dataset_name: str, properties_path: str = "datasets/dataset_properties.json") -> List[str]:
    """获取数据集支持的 term 列表"""
    props = load_dataset_properties(properties_path)
    if dataset_name not in props:
        return ["short", "medium", "long"]
    
    term_type = props[dataset_name].get("term", "med_long")
    if term_type == "short":
        return ["short"]
    return ["short", "medium", "long"]


def get_data_path(
    data_type: str,
    dataset: str,
    ratio: str = None,
    term: str = None,
    method: str = None,
    base_dir: str = "datasets",
) -> Optional[str]:
    """
    获取数据路径
    
    Args:
        data_type: "ori", "missing", "imputed", "inject_info"
        dataset: 数据集名称
        ratio: 缺失比例 (如 "005")
        term: term 类型
        method: 填补方法
        base_dir: 数据根目录
        
    Returns:
        数据文件路径
    """
    base_path = Path(base_dir)
    
    if data_type == "ori":
        path = base_path / "ori" / f"{dataset}.csv"
        if path.exists():
            return str(path)
    
    elif data_type == "missing":
        if ratio and term:
            path = base_path / "MCAR" / f"MCAR_{ratio}" / f"{dataset}_MCAR_{ratio}_{term}.csv"
            if path.exists():
                return str(path)
    
    elif data_type == "imputed":
        if ratio and term and method:
            path = base_path / "Impute" / f"MCAR_{ratio}" / method / f"{dataset}_MCAR_{ratio}_{term}.csv"
            if path.exists():
                return str(path)
    
    elif data_type == "inject_info":
        if ratio and term:
            path = base_path / "MCAR" / f"MCAR_{ratio}" / f"{dataset}_MCAR_{ratio}_{term}_inject_info.json"
            if path.exists():
                return str(path)
    
    return None


def get_inject_range(
    dataset: str,
    ratio: str,
    term: str,
    base_dir: str = "datasets",
) -> Optional[tuple]:
    """
    获取注入区间
    
    Args:
        dataset: 数据集名称
        ratio: 缺失比例
        term: term 类型
        base_dir: 数据根目录
        
    Returns:
        (inject_start, inject_end) 或 None
    """
    try:
        result = compute_inject_range(
            dataset_name=dataset,
            term=term,
            data_path=base_dir,
        )
        return result["start_index"], result["end_index"]
    except Exception as e:
        print(f"  [错误] 计算注入区间失败: {e}")
        return None


def run_single_analysis(
    data_path: str,
    dataset: str,
    data_type: str,
    method: str = "",
    period: int = 24,
    analyzers: List[str] = None,
    inject_start: int = None,
    inject_end: int = None,
    use_inject_range: bool = True,
) -> Dict[str, Any]:
    """
    对单个数据文件执行分析
    
    Args:
        data_path: 数据文件路径
        dataset: 数据集名称
        data_type: 数据类型
        method: 填补方法
        period: 周期
        analyzers: 要使用的分析器列表
        inject_start: 注入区间起点
        inject_end: 注入区间终点
        use_inject_range: 是否只分析注入区间
        
    Returns:
        分析结果字典
    """
    if analyzers is None:
        analyzers = ["STL", "FFT", "ACF"]
    
    data = pd.read_csv(data_path)
    
    original_len = len(data)
    
    if use_inject_range and inject_start is not None and inject_end is not None:
        data = data.iloc[inject_start:inject_end]
        range_info = f"区间 [{inject_start}:{inject_end}]"
    else:
        range_info = "完整数据"
    
    results = {
        "data_path": data_path,
        "dataset": dataset,
        "data_type": data_type,
        "method": method,
        "period": period,
        "use_inject_range": use_inject_range,
        "inject_start": inject_start if use_inject_range else None,
        "inject_end": inject_end if use_inject_range else None,
        "original_length": original_len,
        "analyzed_length": len(data),
        "range_info": range_info,
        "timestamp": datetime.now().isoformat(),
        "results": {},
    }
    
    if "STL" in analyzers:
        stl_analyzer = STLAnalyzer(period=period)
        results["results"]["STL"] = stl_analyzer.analyze(data, dataset, data_type, method)
    
    if "FFT" in analyzers:
        fft_analyzer = FFTAnalyzer()
        results["results"]["FFT"] = fft_analyzer.analyze(data, dataset, data_type, method)
    
    if "ACF" in analyzers:
        acf_analyzer = ACFAnalyzer()
        results["results"]["ACF"] = acf_analyzer.analyze(data, dataset, data_type, method)
    
    return results


def run_batch_analysis(
    datasets: List[str] = None,
    ratios: List[str] = None,
    terms: Dict[str, List[str]] = None,
    methods: List[str] = None,
    analyzers: List[str] = None,
    base_dir: str = "datasets",
    output_dir: str = "results/analysis",
    properties_path: str = "datasets/dataset_properties.json",
    skip_existing: bool = True,
    use_inject_range: bool = True,
) -> Dict[str, Any]:
    """
    批量分析
    
    Args:
        datasets: 数据集列表
        ratios: 缺失比例列表
        terms: 各数据集的 term 字典
        methods: 填补方法列表
        analyzers: 分析器列表
        base_dir: 数据根目录
        output_dir: 输出目录
        properties_path: 属性文件路径
        skip_existing: 是否跳过已存在的结果
        use_inject_range: 是否只分析注入区间 (默认: True)
        
    Returns:
        批量分析结果
    """
    if datasets is None:
        datasets = DATASETS
    if ratios is None:
        ratios = [f"{int(r*100):03d}" for r in MISSING_RATIOS]
    if methods is None:
        methods = IMPUTATION_METHODS
    if analyzers is None:
        analyzers = ["STL", "FFT", "ACF"]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"分析数据集: {dataset}")
        print(f"{'='*80}")
        
        period = get_period(dataset, properties_path)
        
        if terms and dataset in terms:
            dataset_terms = terms[dataset]
        else:
            dataset_terms = get_terms_for_dataset(dataset, properties_path)
        
        dataset_results = {}
        
        for ratio in ratios:
            for term in dataset_terms:
                print(f"\n处理: {dataset} - MCAR_{ratio} - {term}")
                print("-" * 40)
                
                key = f"{dataset}_MCAR_{ratio}_{term}"
                
                output_file = output_path / f"{key}_analysis.json"
                if skip_existing and output_file.exists():
                    print(f"  [跳过] 结果已存在")
                    with open(output_file, 'r', encoding='utf-8') as f:
                        dataset_results[key] = json.load(f)
                    continue
                
                ori_path = get_data_path("ori", dataset, base_dir=base_dir)
                missing_path = get_data_path("missing", dataset, ratio, term, base_dir=base_dir)
                
                if not ori_path:
                    print(f"  [跳过] 干净数据不存在")
                    continue
                if not missing_path:
                    print(f"  [跳过] 缺失数据不存在")
                    continue
                
                inject_start, inject_end = None, None
                if use_inject_range:
                    inject_range = get_inject_range(dataset, ratio, term, base_dir)
                    if inject_range is not None:
                        inject_start, inject_end = inject_range
                        print(f"  [区间] 注入区间: [{inject_start}:{inject_end}]")
                    else:
                        print(f"  [警告] 未找到注入区间信息，将分析完整数据")
                
                result = {
                    "dataset": dataset,
                    "ratio": ratio,
                    "term": term,
                    "period": period,
                    "use_inject_range": use_inject_range,
                    "inject_start": inject_start,
                    "inject_end": inject_end,
                    "ori": {},
                    "missing": {},
                    "imputed": {},
                    "comparison": {},
                }
                
                print(f"  [分析] 干净数据")
                ori_result = run_single_analysis(
                    ori_path, dataset, "ori", period=period, analyzers=analyzers,
                    inject_start=inject_start, inject_end=inject_end, use_inject_range=use_inject_range
                )
                result["ori"] = ori_result["results"]
                
                print(f"  [分析] 缺失数据")
                missing_result = run_single_analysis(
                    missing_path, dataset, "missing", period=period, analyzers=analyzers,
                    inject_start=inject_start, inject_end=inject_end, use_inject_range=use_inject_range
                )
                result["missing"] = missing_result["results"]
                
                imputed_results = {}
                for method in methods:
                    imputed_path = get_data_path("imputed", dataset, ratio, term, method, base_dir)
                    if imputed_path:
                        print(f"  [分析] 填补数据 - {method}")
                        imputed_result = run_single_analysis(
                            imputed_path, dataset, "imputed", method, period, analyzers,
                            inject_start=inject_start, inject_end=inject_end, use_inject_range=use_inject_range
                        )
                        imputed_results[method] = imputed_result["results"]
                
                result["imputed"] = imputed_results
                
                print(f"\n  [比较] 分析结果")
                for analyzer in analyzers:
                    ori_analyzer_result = result["ori"].get(analyzer, {})
                    missing_analyzer_result = result["missing"].get(analyzer, {})
                    imputed_analyzer_results = [r.get(analyzer, {}) for r in imputed_results.values()]
                    
                    comparison = compare_results(ori_analyzer_result, missing_analyzer_result, imputed_analyzer_results)
                    result["comparison"][analyzer] = comparison
                    
                    print_comparison(comparison, methods)
                
                dataset_results[key] = result
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False, default=str)
                print(f"\n  [保存] {output_file}")
        
        all_results[dataset] = dataset_results
    
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[完成] 汇总结果保存到: {summary_file}")
    
    return all_results


def generate_summary_report(all_results: Dict[str, Any], output_dir: str = "results/analysis"):
    """生成汇总报告"""
    output_path = Path(output_dir)
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("时间序列分析汇总报告")
    report_lines.append("=" * 100)
    
    for dataset, dataset_results in all_results.items():
        for key, result in dataset_results.items():
            if "comparison" not in result:
                continue
            
            report_lines.append(f"\n{key} - 综合分析报告")
            report_lines.append("#" * 100)
            
            methods = list(result.get("imputed", {}).keys())
            
            for analyzer in ["STL", "FFT", "ACF"]:
                comparison = result["comparison"].get(analyzer, {})
                if "error" in comparison:
                    continue
                
                best_method, best_score = get_best_method(comparison, methods)
                
                report_lines.append(f"\n【{analyzer} 分析】")
                if best_method:
                    report_lines.append(f"  最佳填补方法: {best_method} (恢复率: {best_score:.1f}%)")
                
                for metric_name, metric_data in comparison.get("metrics", {}).items():
                    ori_val = metric_data.get("ori", 0)
                    missing_recovery = metric_data.get("missing_recovery_pct")
                    
                    line = f"  {metric_name}: Ori={ori_val:.4f}"
                    if missing_recovery is not None:
                        line += f", Missing恢复={missing_recovery:.1f}%"
                    
                    for method in methods:
                        recovery = metric_data.get("imputed_recovery_pct", {}).get(method)
                        if recovery is not None:
                            line += f", {method}恢复={recovery:.1f}%"
                    
                    report_lines.append(line)
            
            method_scores = {}
            for method in methods:
                total_score = 0
                count = 0
                for analyzer in ["STL", "FFT", "ACF"]:
                    comparison = result["comparison"].get(analyzer, {})
                    score = calculate_recovery_score(comparison, method)
                    if score > 0:
                        total_score += score
                        count += 1
                if count > 0:
                    method_scores[method] = total_score / count
            
            if method_scores:
                report_lines.append(f"\n【综合排名】")
                sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
                for rank, (method, score) in enumerate(sorted_methods, 1):
                    report_lines.append(f"  {rank}. {method} (平均恢复率: {score:.1f}%)")
            
            report_lines.append("#" * 100)
    
    report_content = "\n".join(report_lines)
    
    report_file = output_path / "summary_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\n[完成] 汇总报告保存到: {report_file}")
    return report_content


def main():
    parser = argparse.ArgumentParser(description="时间序列分析")
    
    parser.add_argument("--datasets", type=str, nargs='+', default=None,
                        help=f"数据集列表 (默认: {DATASETS})")
    parser.add_argument("--ratios", type=str, nargs='+', default=None,
                        help="缺失比例列表 (默认: 005 010 015 020 025 030)")
    parser.add_argument("--methods", type=str, nargs='+', default=None,
                        help=f"填补方法列表 (默认: {IMPUTATION_METHODS})")
    parser.add_argument("--analyzers", type=str, nargs='+', default=["STL", "FFT", "ACF"],
                        help="分析器列表 (默认: STL FFT ACF)")
    parser.add_argument("--base_dir", type=str, default="datasets",
                        help="数据根目录 (默认: datasets)")
    parser.add_argument("--output_dir", type=str, default="results/analysis",
                        help="输出目录 (默认: results/analysis)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的结果 (默认: True)")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="不跳过已存在的结果")
    parser.add_argument("--full_data", action="store_true",
                        help="分析完整数据而非注入区间 (默认: 分析注入区间)")
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing if args.no_skip_existing else args.skip_existing
    use_inject_range = not args.full_data
    
    if use_inject_range:
        print("[模式] 分析注入区间")
    else:
        print("[模式] 分析完整数据")
    
    ratios = args.ratios
    if ratios is None:
        ratios = [f"{int(r*100):03d}" for r in MISSING_RATIOS]
    
    all_results = run_batch_analysis(
        datasets=args.datasets,
        ratios=ratios,
        methods=args.methods,
        analyzers=args.analyzers,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        skip_existing=skip_existing,
        use_inject_range=use_inject_range,
    )
    
    generate_summary_report(all_results, args.output_dir)


if __name__ == "__main__":
    main()
