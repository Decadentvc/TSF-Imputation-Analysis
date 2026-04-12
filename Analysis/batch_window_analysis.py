"""
批量窗口特征分析脚本

用于批量分析：
1. Intermediate_Predictions 下的预测窗口特征
2. 干净数据的历史窗口特征
3. 填补数据的历史窗口特征
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from Analysis.window_analysis import (
    analyze_prediction_windows,
    analyze_history_windows,
    analyze_imputed_history_windows,
    get_available_impute_methods,
    save_results,
    parse_prediction_dirname,
    is_imputation_method_dir,
)


def get_all_prediction_dirs(base_dir: str = "datasets/Intermediate_Predictions") -> List[str]:
    """
    获取所有预测窗口目录
    
    支持两种目录结构：
    1. 干净数据: {base_dir}/{dataset}_clean_{term}_prediction/
    2. 填补数据: {base_dir}/{eval_data_name}_prediction/{imputation_method}/
    
    Returns:
        预测窗口目录路径列表
    """
    pred_base = Path(base_dir)
    if not pred_base.exists():
        return []
    
    dirs = []
    for item in pred_base.iterdir():
        if item.is_dir() and item.name.endswith("_prediction"):
            subdirs = [d for d in item.iterdir() if d.is_dir() and is_imputation_method_dir(d.name)]
            if subdirs:
                for subdir in subdirs:
                    dirs.append(str(subdir))
            else:
                dirs.append(str(item))
    
    return sorted(dirs)


def get_datasets_with_predictions(base_dir: str = "datasets/Intermediate_Predictions") -> Dict[str, List[Dict]]:
    """
    获取有预测窗口的数据集信息
    
    支持两种目录结构：
    1. 干净数据: {base_dir}/{dataset}_clean_{term}_prediction/
    2. 填补数据: {base_dir}/{eval_data_name}_prediction/{imputation_method}/
    
    Returns:
        {dataset: [{method, block_length, ratio, term, imputation_method, dir_path}, ...]}
    """
    pred_dirs = get_all_prediction_dirs(base_dir)
    
    datasets = defaultdict(list)
    for dir_path in pred_dirs:
        path = Path(dir_path)
        imputation_method = None
        
        if is_imputation_method_dir(path.name):
            imputation_method = path.name.lower()
            dir_name = path.parent.name
        else:
            dir_name = path.name
        
        info = parse_prediction_dirname(dir_name)
        if info:
            datasets[info['dataset']].append({
                'method': info['method'],
                'block_length': info['block_length'],
                'ratio': info['ratio'],
                'term': info['term'],
                'imputation_method': imputation_method,
                'dir_path': dir_path,
            })
    
    return dict(datasets)


def get_available_terms(dataset: str, properties_path: str = "datasets/dataset_properties.json") -> List[str]:
    """获取数据集支持的 term"""
    props_path = Path(properties_path)
    if not props_path.exists():
        return ["short", "medium", "long"]
    
    with open(props_path, 'r', encoding='utf-8') as f:
        props = json.load(f)
    
    if dataset not in props:
        return ["short", "medium", "long"]
    
    return props[dataset].get("terms", ["short", "medium", "long"])


def run_batch_analysis(
    output_dir: str = "results/window_analysis",
    analyze_predictions: bool = True,
    analyze_clean_history: bool = True,
    analyze_imputed_history: bool = True,
    datasets: Optional[List[str]] = None,
    terms: Optional[List[str]] = None,
    impute_methods: Optional[List[str]] = None,
    overwrite: bool = False,
):
    """
    批量分析窗口特征
    
    Args:
        output_dir: 输出目录
        analyze_predictions: 是否分析预测窗口
        analyze_clean_history: 是否分析干净数据的历史窗口
        analyze_imputed_history: 是否分析填补数据的历史窗口
        datasets: 指定数据集列表，None 表示分析所有
        terms: 指定 term 列表，None 表示分析所有
        impute_methods: 指定填补方法列表，None 表示分析所有
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"批量窗口特征分析")
    print(f"{'='*80}")
    print(f"  输出目录: {output_dir}")
    print(f"  分析预测窗口: {analyze_predictions}")
    print(f"  分析干净历史窗口: {analyze_clean_history}")
    print(f"  分析填补历史窗口: {analyze_imputed_history}")
    print(f"{'='*80}")
    
    all_results = {
        "predictions": {},
        "clean_history": {},
        "imputed_history": {},
    }
    
    if analyze_predictions:
        print(f"\n{'='*80}")
        print(f"Step 1: 分析预测窗口")
        print(f"{'='*80}")
        
        pred_dirs = get_all_prediction_dirs()
        print(f"  找到 {len(pred_dirs)} 个预测窗口目录")
        
        for pred_dir in pred_dirs:
            path = Path(pred_dir)
            imputation_method = None
            
            if is_imputation_method_dir(path.name):
                imputation_method = path.name.lower()
                dir_name = path.parent.name
            else:
                dir_name = path.name
            
            info = parse_prediction_dirname(dir_name)
            
            if not info:
                print(f"  [跳过] 无法解析: {dir_name}")
                continue
            
            if datasets and info['dataset'] not in datasets:
                continue
            if terms and info['term'] not in terms:
                continue
            if impute_methods and imputation_method and imputation_method not in impute_methods:
                continue
            
            display_name = f"{dir_name}/{imputation_method}" if imputation_method else dir_name
            print(f"\n  处理: {display_name}")
            
            if imputation_method:
                key = f"{info['dataset']}_{info['method']}_{info['ratio']}_{info['term']}_{imputation_method}"
            else:
                key = f"{info['dataset']}_{info['method']}_{info['ratio']}_{info['term']}"
            json_output = output_path / "predictions" / f"{key}.json"
            if json_output.exists() and not overwrite:
                print(f"    [跳过] 结果已存在: {json_output}")
                try:
                    with open(json_output, 'r', encoding='utf-8') as f:
                        all_results["predictions"][key] = json.load(f)
                except Exception as e:
                    print(f"      [警告] 读取已有结果失败: {e}")
                continue
            try:
                result = analyze_prediction_windows(pred_dir)
                if result.get("success"):
                    all_results["predictions"][key] = result
                    save_results(result, str(json_output))
                else:
                    print(f"    [失败] {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"    [错误] {e}")
                continue
    
    if analyze_clean_history:
        print(f"\n{'='*80}")
        print(f"Step 2: 分析干净数据历史窗口")
        print(f"{'='*80}")
        
        pred_info = get_datasets_with_predictions()
        
        if datasets:
            pred_info = {k: v for k, v in pred_info.items() if k in datasets}
        
        for dataset, infos in pred_info.items():
            available_terms = set(info['term'] for info in infos)
            if terms:
                available_terms = available_terms & set(terms)
            
            for term in available_terms:
                print(f"\n  处理: {dataset} - {term}")
                
                key = f"{dataset}_{term}"
                json_output = output_path / "clean_history" / f"{key}.json"
                if json_output.exists() and not overwrite:
                    print(f"    [跳过] 结果已存在: {json_output}")
                    try:
                        with open(json_output, 'r', encoding='utf-8') as f:
                            all_results["clean_history"][key] = json.load(f)
                    except Exception as e:
                        print(f"      [警告] 读取已有结果失败: {e}")
                    continue
                try:
                    result = analyze_history_windows(
                        dataset_name=dataset,
                        term=term,
                    )
                    if result.get("success"):
                        all_results["clean_history"][key] = result
                        save_results(result, str(json_output))
                    else:
                        print(f"    [失败] {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"    [错误] {e}")
                    continue
    
    if analyze_imputed_history:
        print(f"\n{'='*80}")
        print(f"Step 3: 分析填补数据历史窗口")
        print(f"{'='*80}")
        
        pred_info = get_datasets_with_predictions()
        
        if datasets:
            pred_info = {k: v for k, v in pred_info.items() if k in datasets}
        
        for dataset, infos in pred_info.items():
            method_ratio_terms = defaultdict(set)
            for info in infos:
                # clean 预测窗口没有缺失注入配置，跳过填补历史分析
                if info['method'] == 'clean' or info['ratio'] is None:
                    continue
                key = (info['method'], info['ratio'], info['term'])
                method_ratio_terms[key].add(info['term'])
            
            for (method, ratio, term), _ in method_ratio_terms.items():
                if terms and term not in terms:
                    continue
                
                available_impute_methods = get_available_impute_methods(
                    dataset_name=dataset,
                    method=method,
                    ratio=ratio,
                    term=term,
                )
                
                if impute_methods:
                    available_impute_methods = [m for m in available_impute_methods if m in impute_methods]
                
                for impute_method in available_impute_methods:
                    print(f"\n  处理: {dataset} - {method} - {ratio} - {term} - {impute_method}")
                    
                    key = f"{dataset}_{method}_{ratio}_{term}_{impute_method}"
                    json_output = output_path / "imputed_history" / f"{key}.json"
                    if json_output.exists() and not overwrite:
                        print(f"    [跳过] 结果已存在: {json_output}")
                        try:
                            with open(json_output, 'r', encoding='utf-8') as f:
                                all_results["imputed_history"][key] = json.load(f)
                        except Exception as e:
                            print(f"      [警告] 读取已有结果失败: {e}")
                        continue
                    try:
                        result = analyze_imputed_history_windows(
                            dataset_name=dataset,
                            method=method,
                            ratio=ratio,
                            term=term,
                            impute_method=impute_method,
                        )
                        if result.get("success"):
                            all_results["imputed_history"][key] = result
                            save_results(result, str(json_output))
                        else:
                            print(f"    [失败] {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"    [错误] {e}")
                        continue
    
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        summary = {
            "n_predictions": len(all_results["predictions"]),
            "n_clean_history": len(all_results["clean_history"]),
            "n_imputed_history": len(all_results["imputed_history"]),
            "prediction_keys": list(all_results["predictions"].keys()),
            "clean_history_keys": list(all_results["clean_history"].keys()),
            "imputed_history_keys": list(all_results["imputed_history"].keys()),
        }
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"分析完成")
    print(f"{'='*80}")
    print(f"  预测窗口分析: {len(all_results['predictions'])} 个")
    print(f"  干净历史窗口分析: {len(all_results['clean_history'])} 个")
    print(f"  填补历史窗口分析: {len(all_results['imputed_history'])} 个")
    print(f"  结果保存于: {output_dir}")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="批量窗口特征分析")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/window_analysis",
        help="输出目录",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="指定数据集，逗号分隔 (如 ETTh1,ETTh2)",
    )
    parser.add_argument(
        "--terms",
        type=str,
        default=None,
        help="指定 term，逗号分隔 (如 short,medium)",
    )
    parser.add_argument(
        "--impute_methods",
        type=str,
        default=None,
        help="指定填补方法，逗号分隔 (如 linear,mean)",
    )
    parser.add_argument(
        "--skip_predictions",
        action="store_true",
        help="跳过预测窗口分析",
    )
    parser.add_argument(
        "--skip_clean_history",
        action="store_true",
        help="跳过干净数据历史窗口分析",
    )
    parser.add_argument(
        "--skip_imputed_history",
        action="store_true",
        help="跳过填补数据历史窗口分析",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有结果 (默认跳过已存在的 JSON)",
    )
    
    args = parser.parse_args()
    
    datasets = None
    if args.datasets:
        datasets = [x.strip() for x in args.datasets.split(",")]
    
    terms = None
    if args.terms:
        terms = [x.strip() for x in args.terms.split(",")]
    
    impute_methods = None
    if args.impute_methods:
        impute_methods = [x.strip() for x in args.impute_methods.split(",")]
    
    run_batch_analysis(
        output_dir=args.output_dir,
        analyze_predictions=not args.skip_predictions,
        analyze_clean_history=not args.skip_clean_history,
        analyze_imputed_history=not args.skip_imputed_history,
        datasets=datasets,
        terms=terms,
        impute_methods=impute_methods,
        overwrite=args.overwrite,
    )
