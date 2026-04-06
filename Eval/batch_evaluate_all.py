"""
批量评估脚本

对干净数据和缺失数据进行评估
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "Eval"))

from run_sundial import batch_evaluate


DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "exchange_rate", "national_illness", "traffic", "weather"]

DATASETS_SHORT_ONLY = ["national_illness"]

MISSING_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def get_terms_for_dataset(dataset_name: str) -> List[str]:
    """获取数据集支持的 term 列表"""
    if dataset_name in DATASETS_SHORT_ONLY:
        return ["short"]
    return ["short", "medium", "long"]


def run_evaluate_ori(
    base_data_dir: str = "datasets",
    output_dir: str = "results/sundial/sundial_Ori",
    device: str = "cpu",
    num_samples: int = 100,
    batch_size: int = 32,
):
    """
    评估干净数据
    
    干净数据评估：使用干净数据作为评估数据和干净数据
    """
    print(f"\n{'='*80}")
    print(f"评估干净数据 (Ori)")
    print(f"{'='*80}")
    
    for dataset in DATASETS:
        terms = get_terms_for_dataset(dataset)
        
        for term in terms:
            print(f"\n处理: {dataset} - {term}")
            print("-" * 40)
            
            clean_path = Path(base_data_dir) / "ori" / f"{dataset}.csv"
            if not clean_path.exists():
                print(f"  [跳过] 干净数据不存在: {clean_path}")
                continue
            
            output_file = Path(output_dir) / f"{dataset}_{term}_results.csv"
            if output_file.exists():
                print(f"  [跳过] 结果已存在: {output_file}")
                continue
            
            try:
                from run_sundial import run_evaluation
                run_evaluation(
                    eval_data_path=str(clean_path),
                    clean_data_path=str(clean_path),
                    term=term,
                    base_data_dir=base_data_dir,
                    output_dir=output_dir,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    device=device,
                )
            except Exception as e:
                print(f"  [错误] {e}")


def run_evaluate_missing(
    base_data_dir: str = "datasets",
    output_dir: str = "results/sundial/sundial_Missing",
    device: str = "cpu",
    num_samples: int = 100,
    batch_size: int = 32,
    missing_ratios: List[float] = None,
):
    """
    评估缺失数据
    
    缺失数据评估：使用缺失数据作为评估数据，干净数据作为对比数据
    """
    if missing_ratios is None:
        missing_ratios = MISSING_RATIOS
    
    print(f"\n{'='*80}")
    print(f"评估缺失数据 (Missing)")
    print(f"{'='*80}")
    
    for dataset in DATASETS:
        terms = get_terms_for_dataset(dataset)
        
        for ratio in missing_ratios:
            ratio_str = f"{int(ratio * 100):03d}"
            
            for term in terms:
                eval_filename = f"{dataset}_MCAR_{ratio_str}_{term}.csv"
                eval_path = Path(base_data_dir) / "MCAR" / f"MCAR_{ratio_str}" / eval_filename
                
                if not eval_path.exists():
                    print(f"  [跳过] 缺失数据不存在: {eval_path}")
                    continue
                
                output_file = Path(output_dir) / f"{dataset}_MCAR_{ratio_str}_{term}_results.csv"
                if output_file.exists():
                    print(f"  [跳过] 结果已存在: {output_file}")
                    continue
                
                print(f"\n处理: {dataset} - MCAR_{ratio_str} - {term}")
                print("-" * 40)
                
                try:
                    from run_sundial import run_evaluation
                    run_evaluation(
                        eval_data_path=str(eval_path),
                        clean_data_path=None,
                        term=term,
                        base_data_dir=base_data_dir,
                        output_dir=output_dir,
                        num_samples=num_samples,
                        batch_size=batch_size,
                        device=device,
                    )
                except Exception as e:
                    print(f"  [错误] {e}")


def run_evaluate_all(
    base_data_dir: str = "datasets",
    output_dir_ori: str = "results/sundial/sundial_Ori",
    output_dir_missing: str = "results/sundial/sundial_Missing",
    device: str = "cpu",
    num_samples: int = 100,
    batch_size: int = 32,
    missing_ratios: List[float] = None,
    skip_ori: bool = False,
    skip_missing: bool = False,
):
    """
    批量评估所有数据
    """
    if not skip_ori:
        run_evaluate_ori(
            base_data_dir=base_data_dir,
            output_dir=output_dir_ori,
            device=device,
            num_samples=num_samples,
            batch_size=batch_size,
        )
    
    if not skip_missing:
        run_evaluate_missing(
            base_data_dir=base_data_dir,
            output_dir=output_dir_missing,
            device=device,
            num_samples=num_samples,
            batch_size=batch_size,
            missing_ratios=missing_ratios,
        )


def main():
    parser = argparse.ArgumentParser(description="批量评估干净数据和缺失数据")
    
    parser.add_argument("--base_data_dir", type=str, default="datasets",
                        help="数据集根目录 (默认: datasets)")
    parser.add_argument("--output_dir_ori", type=str, default="results/sundial/sundial_Ori",
                        help="干净数据结果输出目录 (默认: results/sundial/sundial_Ori)")
    parser.add_argument("--output_dir_missing", type=str, default="results/sundial/sundial_Missing",
                        help="缺失数据结果输出目录 (默认: results/sundial/sundial_Missing)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备 (默认: cuda)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="采样数 (默认: 100)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小 (默认: 32)")
    parser.add_argument("--missing_ratios", type=str, default=None,
                        help="缺失比例列表，逗号分隔 (默认: 0.05,0.10,0.15,0.20,0.25,0.30)")
    parser.add_argument("--skip_ori", action="store_true",
                        help="跳过干净数据评估")
    parser.add_argument("--skip_missing", action="store_true",
                        help="跳过缺失数据评估")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="不跳过已存在的结果，覆盖重写")
    
    args = parser.parse_args()
    
    missing_ratios = None
    if args.missing_ratios:
        missing_ratios = [float(x.strip()) for x in args.missing_ratios.split(",")]
    
    run_evaluate_all(
        base_data_dir=args.base_data_dir,
        output_dir_ori=args.output_dir_ori,
        output_dir_missing=args.output_dir_missing,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        missing_ratios=missing_ratios,
        skip_ori=args.skip_ori,
        skip_missing=args.skip_missing,
    )


if __name__ == "__main__":
    main()
