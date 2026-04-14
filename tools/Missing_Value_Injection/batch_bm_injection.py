"""批量对 datasets/ori 下的所有数据集注入 BM 缺失值。

功能：
1. 自动遍历 datasets/ori 目录下的所有 CSV 数据集；
2. 根据 dataset_properties.json 获取可用 term（short/medium/long）；
3. 对每个 term 注入 10%、20%、30% 的 BM（块缺失）缺失值，块长度 50；
4. 结果保存到 datasets/BM/BM_xxx/ 目录，若文件已存在则自动跳过。

可通过命令行参数自定义数据/输出目录、缺失率列表、块长度及随机种子。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FOR_SUNDIAL_DIR = PROJECT_ROOT / "Missing_Value_Injection" / "for_sundial"

sys.path.insert(0, str(FOR_SUNDIAL_DIR))

from inject_range_utils import get_injection_range  # type: ignore  # noqa: E402
from MCAR import get_available_terms  # type: ignore  # noqa: E402
from BM import inject_bm  # type: ignore  # noqa: E402


def parse_missing_ratios(ratio_str: str) -> List[float]:
    """将逗号分隔/方括号格式的缺失比例解析为浮点数列表。"""
    ratio_str = ratio_str.strip().strip("[]")
    ratios: List[float] = []
    for chunk in ratio_str.split(","):
        if chunk.strip():
            value = float(chunk.strip())
            if not 0 <= value <= 1:
                raise ValueError(f"missing ratio must be in [0,1], got {value}")
            ratios.append(value)
    if not ratios:
        raise ValueError("missing ratio list is empty")
    return ratios


def load_dataset_list(data_path: Path) -> List[str]:
    """列出 data_path/ori 下所有 CSV 数据集名称（不含后缀）。"""
    ori_dir = data_path / "ori"
    if not ori_dir.exists():
        raise FileNotFoundError(f"优要的原始数据目录不存在: {ori_dir}")
    dataset_names = sorted(p.stem for p in ori_dir.glob("*.csv"))
    if not dataset_names:
        raise RuntimeError(f"在 {ori_dir} 中未找到任何 CSV 数据集")
    return dataset_names


def load_dataset_properties(data_path: Path) -> Dict:
    props_path = data_path / "dataset_properties.json"
    if not props_path.exists():
        raise FileNotFoundError(f"缺少 dataset_properties.json: {props_path}")
    with open(props_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_output_path(
    output_base_dir: Path,
    dataset_name: str,
    pattern: str,
    missing_ratio: float,
    term: str,
    block_length: int,
) -> Path:
    ratio_str = f"{int(missing_ratio * 100):03d}"
    output_dir = output_base_dir / pattern / f"{pattern}_{ratio_str}"
    filename = f"{dataset_name}_{pattern}_length{block_length}_{ratio_str}_{term}.csv"
    return output_dir / filename


def main() -> None:
    parser = argparse.ArgumentParser(description="批量 BM 缺失值注入脚本")
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets",
        help="数据集根目录（默认：datasets）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets",
        help="输出根目录，结果会存到 <output_dir>/BM/BM_xxx 下（默认：datasets）",
    )
    parser.add_argument(
        "--missing_ratios",
        type=str,
        default="0.1,0.2,0.3",
        help="缺失比例列表，逗号分隔或方括号格式（默认：0.1,0.2,0.3）",
    )
    parser.add_argument(
        "--block_length",
        type=int,
        default=50,
        help="BM 缺失块大小（默认：50）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    missing_ratios = parse_missing_ratios(args.missing_ratios)
    block_length = args.block_length
    seed = args.seed

    print("========================================")
    print("批量 BM 缺失值注入-开始")
    print(f"数据目录: {data_path}")
    print(f"输出目录: {output_dir / 'BM'}")
    print(f"缺失比例: {missing_ratios}")
    print(f"块长度: {block_length}")
    print("========================================")

    dataset_props = load_dataset_properties(data_path)
    dataset_names = load_dataset_list(data_path)

    total_generated = 0
    skipped = 0
    for dataset_name in dataset_names:
        if dataset_name not in dataset_props:
            print(f"[Warning] {dataset_name} 未在 dataset_properties.json 中定义，跳过")
            continue

        try:
            terms = get_available_terms(dataset_name, str(data_path))
        except Exception as exc:  # pragma: no cover - logging purpose
            print(f"[Warning] 无法获取 {dataset_name} 的 term：{exc}")
            continue

        print(f"\n---- 数据集：{dataset_name} | terms={terms} ----")
        for term in terms:
            print(f"处理 term: {term}")
            injection_range = get_injection_range(dataset_name, term, str(data_path))
            injection_range["data_path"] = str(data_path)

            for ratio in missing_ratios:
                output_path = build_output_path(
                    output_base_dir=output_dir,
                    dataset_name=dataset_name,
                    pattern="BM",
                    missing_ratio=ratio,
                    term=term,
                    block_length=block_length,
                )

                if output_path.exists():
                    print(f"  [Skip] {output_path} 已存在，跳过")
                    skipped += 1
                    continue

                output_path.parent.mkdir(parents=True, exist_ok=True)
                df_injected, info = inject_bm(
                    dataset_name=dataset_name,
                    injection_range=injection_range,
                    missing_ratio=ratio,
                    term=term,
                    block_length=block_length,
                    seed=seed,
                )
                df_injected.to_csv(output_path, index=False)

                total_generated += 1
                print(
                    f"  [OK] 保存 {output_path.name} | 缺失: "
                    f"{info['injected_missing']}/{info['total_cells']} "
                    f"({info['actual_missing_ratio']:.2%})"
                )

    print("\n========================================")
    print("批量 BM 缺失值注入-完成")
    print(f"新生成文件数: {total_generated}")
    print(f"跳过已有文件: {skipped}")
    print("========================================")


if __name__ == "__main__":
    main()
