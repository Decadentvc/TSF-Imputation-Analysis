"""按多个 max_context 计算注错区间缺失率。"""

import argparse
from pathlib import Path

from missing_ratio_checker import check_missing_ratio
from Missing_Value_Injection.inject_range_utils import get_injection_range


def parse_contexts(value: str) -> list[int]:
    chunks = [c.strip() for c in value.strip().strip("[]").split(",") if c.strip()]
    if not chunks:
        raise ValueError("contexts list is empty")
    contexts = sorted(set(int(c) for c in chunks))
    if any(c <= 0 for c in contexts):
        raise ValueError("contexts must be positive integers")
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="按 max_context 计算区间缺失率")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--term", type=str, default="long", choices=["short", "medium", "long"], help="term")
    parser.add_argument("--ratio", type=str, default="010", help="缺失率目录标识，如 010")
    parser.add_argument("--block_length", type=int, default=50, help="块长度")
    parser.add_argument("--data_path", type=str, default="data/datasets", help="数据目录")
    parser.add_argument("--file_path", type=str, default=None, help="缺失文件路径（可选）")
    parser.add_argument("--contexts", type=str, default="512,2048,2880,4096,8192", help="max_context 列表")
    args = parser.parse_args()

    contexts = parse_contexts(args.contexts)
    if args.file_path:
        file_path = Path(args.file_path)
    else:
        filename = f"{args.dataset}_BM_length{args.block_length}_{args.ratio}_{args.term}.csv"
        file_path = Path(args.data_path) / "BM" / f"BM_{args.ratio}" / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file not found: {file_path}")

    print("max_context,start_index,end_index,injection_length,missing_cells,total_cells,missing_ratio")
    for mc in contexts:
        inject_range = get_injection_range(
            dataset_name=args.dataset,
            term=args.term,
            data_path=args.data_path,
            max_context=mc,
        )
        result = check_missing_ratio(
            dataset_path=str(file_path),
            start_index=inject_range["start_index"],
            end_index=inject_range["end_index"],
        )
        print(
            f"{mc},{inject_range['start_index']},{inject_range['end_index']},{inject_range['injection_length']},"
            f"{result['missing_cells']},{result['total_cells']},{result['missing_ratio']:.6f}"
        )


if __name__ == "__main__":
    main()
