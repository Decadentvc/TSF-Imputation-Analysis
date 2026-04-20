from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Visualize.config import TERM_CHOICES
from Visualize.router import run_plot
from Visualize.utils import normalize_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("统一可视化入口：根据 plot_type 路由到不同对比方式。")
    )
    parser.add_argument(
        "--plot-type",
        required=True,
        choices=["method", "clean"],
        help="对比类型：method(多填补方法) 或 clean(纯干净窗口)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型目录名，如 timesfm2p5；仅 plot-type=method 需要",
    )
    parser.add_argument("--dataset", required=True, help="数据集名，如 ETTh1")
    parser.add_argument(
        "--term",
        required=True,
        choices=TERM_CHOICES,
        help="term 类型",
    )
    parser.add_argument(
        "--missing-ratio",
        default=None,
        help="缺失比例，仅 plot-type=method 时需要，可用 010/10/0.1",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="填补方法列表，逗号分隔；不传则自动发现，如 mean,linear,forward,backward",
    )
    parser.add_argument(
        "--layout",
        default="both",
        choices=["single", "panel", "both"],
        help="出图布局：single(单指标单图)、panel(6指标合并图)、both(两者都输出)",
    )
    parser.add_argument(
        "--results-analysis-dir",
        default="results_analysis",
        help="分析结果根目录",
    )
    parser.add_argument(
        "--results-pic-dir",
        default="results_pic",
        help="图片输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ratio = normalize_ratio(args.missing_ratio) if args.missing_ratio else None
    model = args.model.strip() if args.model else None
    outputs = run_plot(
        plot_type=args.plot_type,
        results_analysis_dir=Path(args.results_analysis_dir),
        results_pic_dir=Path(args.results_pic_dir),
        model=model,
        dataset=args.dataset.strip(),
        term=args.term.strip().lower(),
        ratio=ratio,
        methods=args.methods,
        layout=args.layout,
    )

    print(f"[DONE] Generated {len(outputs)} plot(s):")
    for p in outputs:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
