"""
运行脚本：解析命令行参数，调用 eval_sundial 模块进行评估，保存结果
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from eval_sundial import evaluate_sundial, save_results_to_csv


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
    """
    
    print(f"\n{'='*80}")
    print(f"Sundial Evaluation Pipeline")
    print(f"{'='*80}")
    
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
    
    print(f"\nStep 4: Running evaluation")
    results = evaluate_sundial(
        eval_data_path=str(eval_path),
        clean_data_path=clean_path,
        freq=freq,
        term=term,
        prediction_length=prediction_length,
        num_samples=num_samples,
        batch_size=batch_size,
        device=device,
        debug=True,
        debug_samples=5,
    )
    
    print(f"\nStep 5: Saving results")
    eval_name = eval_path.stem
    output_path = Path(output_dir) / f"{eval_name}_results.csv"
    save_results_to_csv(results, str(output_path))
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Sundial evaluation on eval datasets"
    )
    
    parser.add_argument(
        "--eval_data_path",
        type=str,
        required=True,
        help="Path to the eval dataset CSV file (format: [name]_[suffix]_[short|medium|long].csv)",
    )
    parser.add_argument(
        "--clean_data_path",
        type=str,
        default=None,
        help="Path to the clean dataset CSV file (optional, if not specified, will be auto-detected from base_data_dir/ori)",
    )
    parser.add_argument(
        "--term",
        type=str,
        choices=["short", "medium", "long"],
        default=None,
        help="Prediction term (required in specified mode, optional in auto mode)",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default="datasets",
        help="Base directory for datasets (default: datasets)",
    )
    parser.add_argument(
        "--properties_path",
        type=str,
        default="datasets/dataset_properties.json",
        help="Path to dataset_properties.json (default: datasets/dataset_properties.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sundial/sundial_Missing",
        help="Output directory for results (default: results/sundial/sundial_Missing)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=None,
        help="Prediction length (if not specified, will be computed automatically)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for generation (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model (cpu or cuda:X, default: cpu)",
    )
    
    args = parser.parse_args()
    
    try:
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
        )
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
