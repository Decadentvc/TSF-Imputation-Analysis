"""
运行脚本：解析命令行参数，调用 eval_sundial 模块进行评估，保存结果
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List

from eval_sundial import evaluate_sundial, save_results_to_csv
from impute_dataset import impute_dataset, generate_imputed_dataset_path


def parse_eval_dataset_name(eval_path: str) -> Tuple[str, str]:
    """
    从评估数据集路径解析出原始数据集名和 term 信息
    
    命名格式：[original_name]_method_ratio_term.csv
    例如：ETTh1_MCAR_005_long.csv, exchange_rate_MCAR_005_medium.csv
    method: MCAR, BM, TM, TVMR
    ratio: 005, 010, 015 等（缺失比例，如 005 表示 5%）
    
    Args:
        eval_path: 评估数据集路径
    
    Returns:
        (original_dataset_name, term) 元组
    """
    path = Path(eval_path)
    filename = path.stem  # 不带扩展名的文件名
    
    # 匹配格式：[original_name]_method_ratio_term
    # method: MCAR|BM|TM|TVMR
    # ratio: 数字（通常 3 位，如 005, 010）
    # term: short|medium|long
    pattern = r'^(.+?)_(MCAR|BM|TM|TVMR)_(\d+)_(short|medium|long)$'
    match = re.match(pattern, filename, re.IGNORECASE)
    
    if not match:
        raise ValueError(
            f"Invalid eval dataset filename: {filename}\n"
            f"Expected format: [original_name]_method_ratio_term.csv\n"
            f"  method: MCAR, BM, TM, TVMR\n"
            f"  ratio: 005, 010, 015, etc. (missing ratio, e.g., 005 = 5%)\n"
            f"  term: short, medium, long\n"
            f"Examples: ETTh1_MCAR_005_long.csv, exchange_rate_MCAR_005_medium.csv"
        )
    
    original_name = match.group(1)
    # method = match.group(2)  # 如果需要可以启用
    # ratio = match.group(3)   # 如果需要可以启用
    term = match.group(4)
    
    return original_name, term


def find_clean_dataset_path(
    original_name: str,
    base_data_dir: str = "datasets"
) -> str:
    """
    根据原始数据集名查找干净数据集路径
    
    Args:
        original_name: 原始数据集名（如 ETTh1, exchange_rate）
        base_data_dir: 数据集根目录
    
    Returns:
        干净数据集的完整路径
    """
    clean_dir = Path(base_data_dir) / "ori"
    clean_path = clean_dir / f"{original_name}.csv"
    
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean dataset not found at: {clean_path}\n"
            f"Expected clean dataset name: {original_name}.csv"
        )
    
    return str(clean_path)


def get_frequency_from_properties(
    dataset_name: str,
    properties_path: str = "datasets/dataset_properties.json"
) -> str:
    """
    从 dataset_properties.json 中查找数据集的频率
    
    Args:
        dataset_name: 数据集名称
        properties_path: 属性文件路径
    
    Returns:
        数据频率（如 H, D, M 等）
    """
    props_path = Path(properties_path)
    
    if not props_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {props_path}")
    
    with open(props_path, "r") as f:
        properties = json.load(f)
    
    if dataset_name not in properties:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    
    return properties[dataset_name].get("frequency", "H")


def get_allowed_terms(
    dataset_name: str,
    properties_path: str = "datasets/dataset_properties.json"
) -> list:
    """
    从 dataset_properties.json 中查找数据集允许的 term
    
    Args:
        dataset_name: 数据集名称
        properties_path: 属性文件路径
    
    Returns:
        term 列表，如 ["short", "medium", "long"] 或 ["short"]
    """
    props_path = Path(properties_path)
    
    if not props_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {props_path}")
    
    with open(props_path, "r") as f:
        properties = json.load(f)
    
    if dataset_name not in properties:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    
    term_type = properties[dataset_name].get("term", "med_long")
    
    if term_type == "short":
        return ["short"]
    else:  # med_long
        return ["short", "medium", "long"]


def check_and_impute_dataset(
    eval_data_path: str,
    imputation_method: str,
    base_output_dir: str = "datasets/Imputed",
) -> str:
    """
    检查填补后的数据集是否存在，如果不存在则进行填补
    
    Args:
        eval_data_path: 原评估数据集路径
        imputation_method: 填补方法名称
        base_output_dir: 填补数据集输出基础目录
    
    Returns:
        填补后的数据集路径
    """
    if imputation_method.lower() == 'none':
        return eval_data_path
    
    imputed_path = generate_imputed_dataset_path(
        eval_data_path=eval_data_path,
        imputation_method=imputation_method,
        base_output_dir=base_output_dir,
    )
    
    if Path(imputed_path).exists():
        print(f"  ✓ Imputed dataset exists: {imputed_path}")
        return imputed_path
    
    print(f"  ⚠ Imputed dataset not found, generating...")
    impute_dataset(
        eval_data_path=eval_data_path,
        imputation_method=imputation_method,
        output_path=imputed_path,
        base_output_dir=base_output_dir,
        save_result=True,
    )
    
    return imputed_path


def batch_check_and_impute(
    eval_data_paths: List[str],
    imputation_methods: List[str],
    base_output_dir: str = "datasets/Imputed",
) -> dict:
    """
    批量检查并填补数据集
    
    Args:
        eval_data_paths: 评估数据集路径列表
        imputation_methods: 填补方法列表
        base_output_dir: 填补数据集输出基础目录
    
    Returns:
        字典，键为 (eval_path, imputation_method)，值为填补后的数据集路径
    """
    imputed_paths_map = {}
    
    print(f"\n{'='*80}")
    print(f"Checking and Generating Imputed Datasets")
    print(f"{'='*80}")
    print(f"  Eval datasets: {len(eval_data_paths)}")
    print(f"  Imputation methods: {imputation_methods}")
    
    for eval_path in eval_data_paths:
        for method in imputation_methods:
            imputed_path = check_and_impute_dataset(
                eval_data_path=eval_path,
                imputation_method=method,
                base_output_dir=base_output_dir,
            )
            imputed_paths_map[(eval_path, method)] = imputed_path
    
    print(f"{'='*80}\n")
    
    return imputed_paths_map


def generate_eval_dataset_paths(
    dataset_name: str,
    method: str,
    missing_ratios: list = None,
    base_data_dir: str = "datasets",
    block_length: int = None,
) -> list:
    """
    生成一系列评估数据集的路径
    
    Args:
        dataset_name: 原始数据集名称
        method: 注空模式（MCAR, BM, TM, TVMR）
        missing_ratios: 缺失比例列表，默认为 [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        base_data_dir: 基础数据集目录
        block_length: 块长度（仅用于 BM 模式）
    
    Returns:
        评估数据集路径列表，每个元素为 (eval_path, term) 元组
    """
    if missing_ratios is None:
        missing_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    
    # 获取允许的 term
    allowed_terms = get_allowed_terms(dataset_name)
    
    eval_paths = []
    
    for ratio in missing_ratios:
        ratio_str = f"{int(ratio * 100):03d}"  # 3 位编码，如 005, 010
        
        for term in allowed_terms:
            # 构建评估数据集文件名
            # 对于 BM 模式，文件名中包含块长度信息
            if method == "BM" and block_length is not None:
                eval_filename = f"{dataset_name}_{method}_length{block_length}_{ratio_str}_{term}.csv"
            else:
                eval_filename = f"{dataset_name}_{method}_{ratio_str}_{term}.csv"
            
            # 构建完整路径：datasets/[method]/[method]_[ratio]/[filename]
            eval_path = Path(base_data_dir) / method / f"{method}_{ratio_str}" / eval_filename
            
            eval_paths.append((str(eval_path), term))
    
    return eval_paths
    
    with open(props_path, 'r') as f:
        properties = json.load(f)
    
    if dataset_name not in properties:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in {props_path}\n"
            f"Available datasets: {list(properties.keys())}"
        )
    
    freq = properties[dataset_name].get('frequency')
    if not freq:
        raise ValueError(
            f"Frequency not specified for dataset '{dataset_name}' in {props_path}"
        )
    
    return freq


def run_evaluation(
    eval_data_path: str,
    clean_data_path: Optional[str] = None,
    term: Optional[str] = None,
    base_data_dir: str = "datasets",
    properties_path: str = "datasets/dataset_properties.json",
    output_dir: str = "results/sundial/sundial_Missing",
    prediction_length: Optional[int] = None,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    imputation_method: str = "none",
):
    """
    执行完整评估流程
    
    Args:
        eval_data_path: 评估数据集路径
        clean_data_path: 干净数据集路径（可选，如果不指定则自动从 base_data_dir/ori 查找）
        term: 预测周期（可选，指定模式下必需，自动模式下从文件名解析）
        base_data_dir: 数据集根目录
        properties_path: 数据集属性文件路径
        output_dir: 结果输出目录
        prediction_length: 预测长度（可选，不指定则自动计算）
        num_samples: 采样数
        batch_size: 批次大小
        device: 设备
        imputation_method: 填补方法（none, zero, mean, forward, backward, linear, nearest, spline, seasonal）
    """
    
    print(f"\n{'='*80}")
    print(f"Sundial Evaluation Pipeline")
    print(f"{'='*80}")
    print(f"  Imputation method: {imputation_method}")
    
    eval_path = Path(eval_data_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_path}")
    
    print(f"\nStep 1: Parsing eval dataset name")
    print(f"  Input: {eval_path}")
    
    if clean_data_path is None:
        # 自动模式：从文件名解析 original_name 和 term
        original_name, parsed_term = parse_eval_dataset_name(str(eval_path))
        if term is not None:
            print(f"  Warning: --term ignored in auto mode, using term from filename: {parsed_term}")
        term = parsed_term
        print(f"  Original dataset name: {original_name}")
        print(f"  Term: {term} (from filename)")
    else:
        # 指定模式：term 必须从命令行读取
        if term is None:
            raise ValueError(
                "In specified mode (with --clean_data_path), --term is required.\n"
                "Please specify --term short|medium|long"
            )
        # 从文件名尝试提取 original_name（可选，仅用于显示）
        original_name = eval_path.stem
        print(f"  Mode: Specified")
        print(f"  Original dataset name: {original_name} (from filename)")
        print(f"  Term: {term} (from command line)")
    
    print(f"\nStep 2: Finding clean dataset")
    if clean_data_path is None:
        # 自动模式：从 base_data_dir/ori 查找
        clean_path = find_clean_dataset_path(original_name, base_data_dir)
        print(f"  Mode: Auto (from base_data_dir/ori)")
        print(f"  Clean dataset: {clean_path}")
    else:
        # 指定模式：使用用户提供的路径
        clean_path_obj = Path(clean_data_path)
        if not clean_path_obj.exists():
            raise FileNotFoundError(f"Specified clean dataset not found: {clean_data_path}")
        clean_path = clean_data_path
        print(f"  Mode: Specified")
        print(f"  Clean dataset: {clean_path}")
    
    print(f"\nStep 3: Getting frequency from properties")
    freq = get_frequency_from_properties(original_name, properties_path)
    print(f"  Frequency: {freq}")
    
    print(f"\nStep 4: Checking and generating imputed dataset")
    actual_eval_path = check_and_impute_dataset(
        eval_data_path=str(eval_path),
        imputation_method=imputation_method,
        base_output_dir="datasets/Imputed",
    )
    
    print(f"\nStep 5: Running evaluation")
    results = evaluate_sundial(
        eval_data_path=actual_eval_path,
        clean_data_path=clean_path,
        freq=freq,
        term=term,
        prediction_length=prediction_length,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        debug=False,
        debug_samples=5,
        imputation_method=None,  # 填补已经在 Step 4 完成
    )
    
    print(f"\nStep 6: Saving results")
    eval_name = eval_path.stem
    
    # 根据填补方法确定输出目录
    if imputation_method and imputation_method.lower() != "none":
        output_subdir = "results/sundial/sundial_Impute"
        # 添加填补方法前缀
        impute_prefix = imputation_method.lower()
        output_filename = f"{impute_prefix}_{eval_name}_{term}_results.csv"
    else:
        output_subdir = "results/sundial/sundial_Ori"
        output_filename = f"{eval_name}_{term}_results.csv"
    
    # 使用用户指定的输出目录覆盖默认值
    if output_dir != "results/sundial/sundial_Missing":
        output_subdir = output_dir
    
    output_path = Path(output_subdir) / output_filename
    save_results_to_csv(results, str(output_path))
    
    # 保存中间预测结果
    save_intermediate_predictions(
        results=results,
        dataset_name=original_name,
        eval_data_name=eval_name,
        base_output_dir="datasets/Intermediate_Predictions",
    )
    
    return results


def save_intermediate_predictions(
    results: dict,
    dataset_name: str,
    eval_data_name: str,
    base_output_dir: str = "datasets/Intermediate_Predictions",
):
    """
    保存中间预测结果，按窗口拆分保存
    
    Args:
        results: evaluate_sundial 返回的结果字典，包含 forecasts 等信息
        dataset_name: 数据集名称（如 ETTh1, exchange_rate）
        eval_data_name: 评估数据集名称（如 ETTh1_MCAR_005_short）
        base_output_dir: 输出基础目录
    """
    import pandas as pd
    import numpy as np
    
    # 检查是否有预测结果
    if 'forecasts' not in results:
        print(f"\nWarning: No forecasts found in results, skipping intermediate prediction saving")
        return
    
    forecasts = results['forecasts']
    prediction_length = results['prediction_length']
    freq = results['freq']
    
    print(f"\n{'='*80}")
    print(f"Saving Intermediate Predictions")
    print(f"{'='*80}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Eval data: {eval_data_name}")
    print(f"  Prediction length: {prediction_length}")
    print(f"  Frequency: {freq}")
    print(f"  Number of windows: {len(forecasts)}")
    
    # 创建输出目录：base_output_dir/[eval_data_name]_prediction/
    output_dir = Path(base_output_dir) / f"{eval_data_name}_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 遍历每个窗口的预测结果
    for window_idx, forecast in enumerate(forecasts):
        # 提取 mean 预测值
        mean_prediction = forecast.mean  # shape: (prediction_length,)
        
        # 生成时间序列
        start_date = forecast.start_date
        # 如果 start_date 是 Period 对象，转换为 Timestamp
        if hasattr(start_date, 'to_timestamp'):
            start_date = start_date.to_timestamp()
        date_range = pd.date_range(
            start=start_date,
            periods=prediction_length,
            freq=freq
        )
        
        # 创建 DataFrame：只包含时序列和 mean 预测值
        df = pd.DataFrame({
            'date': date_range,
            'prediction': mean_prediction,
        })
        
        # 文件命名：[eval_data_name]_prediction_[window_index].csv
        output_filename = f"{eval_data_name}_prediction_{window_idx}.csv"
        output_path = output_dir / output_filename
        df.to_csv(output_path, index=False)
        
        if window_idx == 0:
            print(f"\n  Sample output (window 0):")
            print(f"    File: {output_path}")
            print(f"    Shape: {df.shape}")
            print(f"    Mean range: [{mean_prediction.min():.4f}, {mean_prediction.max():.4f}]")
            print(f"    Date range: {date_range[0]} to {date_range[-1]}")
    
    print(f"\n  ✅ Saved {len(forecasts)} prediction windows to: {output_dir}")
    print(f"{'='*80}\n")


def batch_evaluate(
    dataset_name: str,
    method: str,
    missing_ratios: list = None,
    base_data_dir: str = "datasets",
    properties_path: str = "datasets/dataset_properties.json",
    output_dir: str = "results/sundial/sundial_Missing",
    prediction_length: Optional[int] = None,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    imputation_methods: list = None,
    block_length: int = None,
) -> list:
    """
    一键评估功能：批量评估所有缺失率和 term 组合
    
    Args:
        dataset_name: 原始数据集名称（如 ETTh1, exchange_rate）
        method: 注空模式（MCAR, BM, TM, TVMR）
        missing_ratios: 缺失比例列表，默认为 [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        base_data_dir: 基础数据集目录
        properties_path: 数据集属性文件路径
        output_dir: 输出目录
        prediction_length: 预测长度
        num_samples: 采样数
        batch_size: 批量大小
        device: 设备类型
        imputation_methods: 填补方法列表
        block_length: 块长度（仅用于 BM 模式），默认为 None（使用所有填补方法）
    
    Returns:
        评估结果列表，每个元素为 (eval_path, term, imputation_method, results) 元组
    """
    
    print(f"\n{'='*80}")
    print(f"One-Click Batch Evaluation")
    print(f"{'='*80}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Method: {method}")
    print(f"  Missing ratios: {missing_ratios if missing_ratios else '[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]'}")
    print(f"  Imputation methods: {imputation_methods if imputation_methods else 'all methods'}")
    print(f"  Device: {device}")
    print(f"{'='*80}")
    
    # 如果没有指定填补方法，使用所有填补方法（包括 none）
    if imputation_methods is None:
        imputation_methods = ["none", "zero", "mean", "forward", "backward", "linear", "nearest", "spline", "seasonal"]
    
    # Step 1: 生成所有评估数据集路径
    print(f"\nStep 1: Generating eval dataset paths")
    eval_paths = generate_eval_dataset_paths(
        dataset_name=dataset_name,
        method=method,
        missing_ratios=missing_ratios,
        base_data_dir=base_data_dir,
        block_length=block_length,
    )
    print(f"  Found {len(eval_paths)} eval datasets to evaluate")
    
    # Step 2: 查找干净数据集
    print(f"\nStep 2: Finding clean dataset")
    clean_path = find_clean_dataset_path(dataset_name, base_data_dir)
    print(f"  Clean dataset: {clean_path}")
    
    # Step 3: 获取频率
    print(f"\nStep 3: Getting frequency from properties")
    freq = get_frequency_from_properties(dataset_name, properties_path)
    print(f"  Frequency: {freq}")
    
    # Step 4: 批量检查并生成填补数据集
    eval_path_list = [eval_path for eval_path, _ in eval_paths]
    imputed_paths_map = batch_check_and_impute(
        eval_data_paths=eval_path_list,
        imputation_methods=imputation_methods,
        base_output_dir="datasets/Imputed",
    )
    
    # Step 5: 批量评估（对每个数据集使用不同的填补方法依次评估）
    print(f"\nStep 5: Running batch evaluation")
    all_results = []
    
    for idx, (eval_path, term) in enumerate(eval_paths, 1):
        print(f"\n{'='*80}")
        print(f"Evaluation Dataset {idx}/{len(eval_paths)}")
        print(f"  Eval: {eval_path}")
        print(f"  Term: {term}")
        print(f"{'='*80}")
        
        # 检查评估数据集是否存在
        if not Path(eval_path).exists():
            print(f"  ⚠️  Warning: Eval dataset not found, skipping: {eval_path}")
            continue
        
        # 对每个填补方法依次进行评估
        for impute_method in imputation_methods:
            print(f"\n  Applying imputation method: {impute_method}")
            
            try:
                # 获取填补后的数据集路径
                actual_eval_path = imputed_paths_map.get((eval_path, impute_method), eval_path)
                
                # 运行评估（使用填补后的数据集）
                results = evaluate_sundial(
                    eval_data_path=actual_eval_path,
                    clean_data_path=clean_path,
                    freq=freq,
                    term=term,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    device=device,
                    debug=False,
                    debug_samples=5,
                    imputation_method=None,  # 填补已经在 Step 4 完成
                )
                
                # 保存结果
                eval_name = Path(eval_path).stem
                # 检查文件名是否已经包含 term，如果已包含则不再重复添加
                if eval_name.endswith(f"_{term}"):
                    base_filename = f"{eval_name}"
                else:
                    base_filename = f"{eval_name}_{term}"
                
                # 根据填补方法确定输出目录和文件名
                if impute_method and impute_method.lower() != "none":
                    output_subdir = "results/sundial/sundial_Impute"
                    output_filename = f"{impute_method.lower()}_{base_filename}_results.csv"
                else:
                    output_subdir = "results/sundial/sundial_Ori"
                    output_filename = f"{base_filename}_results.csv"
                
                # 使用用户指定的输出目录覆盖默认值
                if output_dir != "results/sundial/sundial_Missing":
                    output_subdir = output_dir
                
                output_path = Path(output_subdir) / output_filename
                save_results_to_csv(results, str(output_path))
                print(f"    ✅ Results saved to: {output_path}")
                
                # 保存中间预测结果
                save_intermediate_predictions(
                    results=results,
                    dataset_name=dataset_name,
                    eval_data_name=eval_name,
                    base_output_dir="datasets/Intermediate_Predictions",
                )
                
                all_results.append((eval_path, term, impute_method, results))
                
            except Exception as e:
                print(f"    ❌ Error during evaluation with {impute_method}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Step 6: 汇总结果
    print(f"\n{'='*80}")
    print(f"Batch Evaluation Complete")
    print(f"  Total: {len(eval_paths)} datasets × {len(imputation_methods)} imputation methods = {len(eval_paths) * len(imputation_methods)} evaluations")
    print(f"  Successful: {len(all_results)}")
    print(f"{'='*80}")
    
    return all_results


def main():
    # 创建主解析器
    main_parser = argparse.ArgumentParser(
        description="Run Sundial evaluation on eval datasets"
    )
    
    # 创建子解析器（用于不同的评估模式）
    subparsers = main_parser.add_subparsers(dest="mode", help="Evaluation mode")
    
    # ========== 模式 1: 单个评估（原有功能）==========
    single_parser = subparsers.add_parser("single", help="Evaluate a single dataset")
    single_parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to the eval dataset CSV file (format: [name]_[suffix]_[short|medium|long].csv)",
    )
    single_parser.add_argument(
        "--clean_data_path",
        type=str,
        default=None,
        help="Path to the clean dataset CSV file (optional, if not specified, will be auto-detected from base_data_dir/ori)",
    )
    single_parser.add_argument(
        "--term",
        type=str,
        choices=["short", "medium", "long"],
        default=None,
        help="Prediction term (required in specified mode, optional in auto mode)",
    )
    single_parser.add_argument(
        "--base_data_dir",
        type=str,
        default="datasets",
        help="Base directory for datasets (default: datasets)",
    )
    single_parser.add_argument(
        "--properties_path",
        type=str,
        default="datasets/dataset_properties.json",
        help="Path to dataset_properties.json (default: datasets/dataset_properties.json)",
    )
    single_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sundial/sundial_Missing",
        help="Output directory for results (default: results/sundial/sundial_Missing)",
    )
    single_parser.add_argument(
        "--prediction_length",
        type=int,
        default=None,
        help="Prediction length (if not specified, will be computed automatically)",
    )
    single_parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for generation (default: 100)",
    )
    single_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    single_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model (cpu or cuda:X, default: cpu)",
    )
    single_parser.add_argument(
        "--imputation_method",
        type=str,
        default="none",
        help="Imputation method for missing values (none, zero, mean, forward, backward, linear, nearest, spline, seasonal). Default: none",
    )
    
    # ========== 模式 2: 一键批量评估（新增功能）==========
    batch_parser = subparsers.add_parser("batch", help="Batch evaluate all missing ratios and terms")
    batch_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Original dataset name (e.g., ETTh1, exchange_rate)",
    )
    batch_parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["MCAR", "BM", "TM", "TVMR"],
        help="Missing value injection method",
    )
    batch_parser.add_argument(
        "--missing_ratios",
        type=str,
        default=None,
        help="Comma-separated missing ratios (e.g., '0.05,0.10,0.15'). Default: [0.05,0.10,0.15,0.20,0.25,0.30]",
    )
    batch_parser.add_argument(
        "--base_data_dir",
        type=str,
        default="datasets",
        help="Base directory for datasets (default: datasets)",
    )
    batch_parser.add_argument(
        "--properties_path",
        type=str,
        default="datasets/dataset_properties.json",
        help="Path to dataset_properties.json (default: datasets/dataset_properties.json)",
    )
    batch_parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sundial/sundial_Missing",
        help="Output directory for results (default: results/sundial/sundial_Missing)",
    )
    batch_parser.add_argument(
        "--prediction_length",
        type=int,
        default=None,
        help="Prediction length (if not specified, will be computed automatically)",
    )
    batch_parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for generation (default: 100)",
    )
    batch_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    batch_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model (cpu or cuda:X, default: cpu)",
    )
    batch_parser.add_argument(
        "--imputation_methods",
        type=str,
        default=None,
        help="Comma-separated imputation methods (e.g., 'none,zero,mean'). Default: all methods (none,zero,mean,forward,backward,linear,nearest,spline,seasonal)",
    )
    batch_parser.add_argument(
        "--block_length",
        type=int,
        default=None,
        help="Block length for BM pattern (default: None, only used for BM method)",
    )
    
    args = main_parser.parse_args()
    
    # 如果没有指定模式，显示帮助
    if args.mode is None:
        main_parser.print_help()
        return
    
    try:
        if args.mode == "single":
            # 单个评估模式（原有功能）
            run_evaluation(
                eval_data_path=args.eval_data_path,
                clean_data_path=args.clean_data_path,
                term=args.term,
                base_data_dir=args.base_data_dir,
                properties_path=args.properties_path,
                output_dir=args.output_dir,
                prediction_length=args.prediction_length,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=args.device,
                imputation_method=args.imputation_method,
            )
        elif args.mode == "batch":
            # 一键批量评估模式（新增功能）
            # 解析缺失比例
            missing_ratios = None
            if args.missing_ratios:
                missing_ratios = [float(x.strip()) for x in args.missing_ratios.split(",")]
            
            # 解析填补方法列表
            imputation_methods = None
            if args.imputation_methods:
                imputation_methods = [x.strip().lower() for x in args.imputation_methods.split(",")]
            
            batch_evaluate(
                dataset_name=args.dataset,
                method=args.method,
                missing_ratios=missing_ratios,
                base_data_dir=args.base_data_dir,
                properties_path=args.properties_path,
                output_dir=args.output_dir,
                prediction_length=args.prediction_length,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=args.device,
                imputation_methods=imputation_methods,
                block_length=args.block_length,
            )
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
