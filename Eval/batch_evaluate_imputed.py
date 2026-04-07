"""
批量评估填补数据

对 datasets/Impute 下所有填补后的数据进行评估
结果保存到 results/sundial/sundial_Impute
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "Eval"))

from run_sundial import run_evaluation


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    从文件名解析数据集信息
    
    Args:
        filename: 如 "ETTh1_MCAR_005_short.csv"
        
    Returns:
        (dataset_name, ratio, term)
    """
    basename = filename.replace(".csv", "")
    parts = basename.split("_")
    
    dataset_name = parts[0]
    ratio = parts[2]
    term = parts[-1]
    
    return dataset_name, ratio, term


def run_batch_evaluation(
    input_base_dir: str = "datasets/Impute",
    output_base_dir: str = "results/sundial/sundial_Impute",
    base_data_dir: str = "datasets",
    methods: List[str] = None,
    skip_existing: bool = True,
    device: str = "cpu",
    num_samples: int = 100,
    batch_size: int = 32,
):
    """
    批量评估填补数据
    
    Args:
        input_base_dir: 填补数据目录
        output_base_dir: 结果输出目录
        base_data_dir: 数据集根目录
        methods: 填补方法列表
        skip_existing: 是否跳过已存在的结果
        device: 设备
        num_samples: 采样数
        batch_size: 批次大小
    """
    if methods is None:
        methods = ['zero', 'forward', 'backward', 'mean', 'linear']
    
    input_base = Path(input_base_dir)
    output_base = Path(output_base_dir)
    
    print(f"\n{'='*80}")
    print(f"批量评估填补数据")
    print(f"{'='*80}")
    print(f"输入目录: {input_base}")
    print(f"输出目录: {output_base}")
    print(f"填补方法: {methods}")
    print(f"设备: {device}")
    print(f"{'='*80}")
    
    mcar_dirs = sorted([d for d in input_base.iterdir() if d.is_dir() and d.name.startswith('MCAR_')])
    
    total_evaluated = 0
    total_skipped = 0
    total_failed = 0
    
    for mcar_dir in mcar_dirs:
        ratio = mcar_dir.name
        print(f"\n处理目录: {ratio}")
        print("-" * 40)
        
        for method in methods:
            method_dir = mcar_dir / method
            if not method_dir.exists():
                print(f"  [跳过] 方法目录不存在: {method}")
                continue
            
            print(f"\n  方法: {method}")
            print("  " + "-" * 36)
            
            csv_files = sorted(method_dir.glob("*.csv"))
            
            for csv_file in csv_files:
                filename = csv_file.name
                dataset_name, _, term = parse_filename(filename)
                
                output_dir = output_base / ratio / method
                output_file = output_dir / f"{csv_file.stem}_results.csv"
                
                if skip_existing and output_file.exists():
                    print(f"    [跳过] {filename}")
                    total_skipped += 1
                    continue
                
                print(f"    [评估] {filename}")
                
                try:
                    run_evaluation(
                        eval_data_path=str(csv_file),
                        clean_data_path=None,
                        term=term,
                        base_data_dir=base_data_dir,
                        output_dir=str(output_dir),
                        num_samples=num_samples,
                        batch_size=batch_size,
                        device=device,
                    )
                    total_evaluated += 1
                    print(f"    [完成] 结果保存到: {output_file}")
                except Exception as e:
                    total_failed += 1
                    print(f"    [错误] {e}")
    
    print(f"\n{'='*80}")
    print(f"批量评估完成")
    print(f"{'='*80}")
    print(f"评估成功: {total_evaluated}")
    print(f"跳过: {total_skipped}")
    print(f"失败: {total_failed}")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="批量评估填补数据")
    
    parser.add_argument("--input_dir", type=str, default="datasets/Impute",
                        help="填补数据目录 (默认: datasets/Impute)")
    parser.add_argument("--output_dir", type=str, default="results/sundial/sundial_Impute",
                        help="结果输出目录 (默认: results/sundial/sundial_Impute)")
    parser.add_argument("--base_data_dir", type=str, default="datasets",
                        help="数据集根目录 (默认: datasets)")
    parser.add_argument("--methods", type=str, nargs='+',
                        default=['zero', 'forward', 'backward', 'mean', 'linear'],
                        help="填补方法列表 (默认: zero forward backward mean linear)")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="跳过已存在的结果 (默认: True)")
    parser.add_argument("--no_skip_existing", action="store_true",
                        help="不跳过已存在的结果，覆盖重写")
    parser.add_argument("--device", type=str, default="cpu",
                        help="设备 (默认: cpu)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="采样数 (默认: 100)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小 (默认: 32)")
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip_existing if args.no_skip_existing else args.skip_existing
    
    run_batch_evaluation(
        input_base_dir=args.input_dir,
        output_base_dir=args.output_dir,
        base_data_dir=args.base_data_dir,
        methods=args.methods,
        skip_existing=skip_existing,
        device=args.device,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
