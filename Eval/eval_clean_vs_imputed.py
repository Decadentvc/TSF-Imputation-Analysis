"""统一脚本：评估干净数据与填补后数据的 Sundial 预测表现。

特性：
1. 自动先评估干净数据（per term）
2. 自动对指定缺失数据进行填补并评估预测
3. 默认使用 CUDA，加速推理
4. 已存在的结果自动跳过，支持 --force 重新计算
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

try:  # pragma: no cover - import guard for environments w/o torch
    torch = importlib.import_module("torch")
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore

# 将项目根目录加入 sys.path，以便导入 Eval 包
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Eval.run_sundial import (  # noqa: E402
    batch_check_and_impute,
    evaluate_clean,
    generate_eval_dataset_paths,
    get_allowed_terms,
    get_frequency_from_properties,
    save_intermediate_predictions,
    save_results_to_csv,
    find_clean_dataset_path,
)
from Eval.eval_sundial import evaluate_sundial  # noqa: E402


ALL_IMPUTATION_METHODS = [
    "none",
    "zero",
    "mean",
    "forward",
    "backward",
    "linear",
    "nearest",
    "spline",
    "seasonal",
]


def parse_ratio_list(raw: str | None) -> List[float] | None:
    """解析缺失率字符串，支持 `0.1`, `10`, `10%` 等格式。"""

    if not raw:
        return None

    ratios: List[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        if text.endswith("%"):
            value = float(text[:-1]) / 100.0
        else:
            value = float(text)
            if value > 1:
                value /= 100.0
        if not (0 < value < 1):
            raise ValueError(f"Missing ratio must be within (0, 1): {value}")
        ratios.append(round(value, 4))
    return ratios or None


def load_dataset_names(properties_path: str) -> List[str]:
    path = Path(properties_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset properties format in {path}")
    return sorted(data.keys())


def parse_terms(raw: str | None, allowed: Sequence[str]) -> List[str]:
    if not raw:
        return list(allowed)
    selected = [term.strip().lower() for term in raw.split(",") if term.strip()]
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported term(s): {invalid}. Allowed: {allowed}")
    # 保持输入顺序但去重
    seen = set()
    ordered: List[str] = []
    for term in selected:
        if term not in seen:
            seen.add(term)
            ordered.append(term)
    return ordered


def parse_imputation_methods(raw: str | None) -> List[str]:
    if not raw:
        return ["linear"]  # 默认至少进行一次真实填补
    if raw.strip().lower() == "all":
        return list(ALL_IMPUTATION_METHODS)
    methods = [m.strip().lower() for m in raw.split(",") if m.strip()]
    invalid = sorted(set(methods) - set(ALL_IMPUTATION_METHODS))
    if invalid:
        raise ValueError(
            f"Unsupported imputation method(s): {invalid}. Allowed: {ALL_IMPUTATION_METHODS}"
        )
    return methods or ["linear"]


def resolve_device(requested: str) -> str:
    req = requested.lower()
    if req.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，自动退回 CPU 设备")
            return "cpu"
    return requested


def filter_eval_paths(
    eval_paths: Sequence[Tuple[str, str]],
    allowed_terms: Iterable[str],
) -> List[Tuple[str, str]]:
    allowed_set = set(allowed_terms)
    return [item for item in eval_paths if item[1] in allowed_set]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def should_skip(result_path: Path, force: bool) -> bool:
    if result_path.exists() and not force:
        print(f"  ✓ Skip existing results: {result_path}")
        return True
    return False


def evaluate_clean_datasets(
    dataset_name: str,
    terms: Sequence[str],
    device: str,
    prediction_length: int | None,
    num_samples: int,
    batch_size: int,
    base_data_dir: str,
    properties_path: str,
    output_dir: Path,
    force: bool,
) -> None:
    print("\n========== Evaluating clean datasets ==========")
    for term in terms:
        result_path = output_dir / f"{dataset_name}_clean_{term}_results.csv"
        if should_skip(result_path, force):
            continue
        evaluate_clean(
            dataset_name=dataset_name,
            term=term,
            base_data_dir=base_data_dir,
            properties_path=properties_path,
            output_dir=str(output_dir),
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
        )


def evaluate_imputed_datasets(
    dataset_name: str,
    method: str,
    ratios: Sequence[float] | None,
    terms: Sequence[str],
    imputation_methods: Sequence[str],
    device: str,
    prediction_length: int | None,
    num_samples: int,
    batch_size: int,
    base_data_dir: str,
    properties_path: str,
    block_length: int | None,
    output_dir: Path,
    force: bool,
    debug: bool,
    debug_samples: int,
) -> None:
    print("\n========== Evaluating imputed datasets ==========")
    gen_kwargs: dict[str, Any] = {
        "dataset_name": dataset_name,
        "method": method,
        "base_data_dir": base_data_dir,
    }
    if ratios:
        gen_kwargs["missing_ratios"] = list(ratios)
    if block_length is not None:
        gen_kwargs["block_length"] = block_length

    eval_paths = generate_eval_dataset_paths(**gen_kwargs)  # type: ignore[arg-type]
    eval_paths = filter_eval_paths(eval_paths, terms)
    if not eval_paths:
        print("⚠️  No eval datasets found for the given configuration.")
        return

    clean_path = find_clean_dataset_path(dataset_name, base_data_dir)
    freq = get_frequency_from_properties(dataset_name, properties_path)

    eval_path_list = [eval_path for eval_path, _ in eval_paths]
    imputed_map = batch_check_and_impute(
        eval_data_paths=eval_path_list,
        imputation_methods=list(imputation_methods),
        base_output_dir="datasets/Imputed",
    )

    for idx, (eval_path, term) in enumerate(eval_paths, 1):
        print(f"\n---- Eval dataset {idx}/{len(eval_paths)} ----")
        print(f"Eval file: {eval_path}")
        print(f"Term: {term}")

        if not Path(eval_path).exists():
            print(f"  ⚠️  File missing, skip: {eval_path}")
            continue

        eval_name = Path(eval_path).stem
        base_filename = eval_name if eval_name.endswith(f"_{term}") else f"{eval_name}_{term}"

        for method_name in imputation_methods:
            method_label = method_name.lower()
            if method_label == "none":
                output_filename = f"{base_filename}_results.csv"
            else:
                output_filename = f"{method_label}_{base_filename}_results.csv"
            result_path = output_dir / output_filename

            if should_skip(result_path, force):
                continue

            actual_eval_path = imputed_map.get((eval_path, method_name), eval_path)
            try:
                results = evaluate_sundial(
                    eval_data_path=actual_eval_path,
                    clean_data_path=clean_path,
                    freq=freq,
                    term=term,
                    prediction_length=prediction_length,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    device=device,
                    debug=debug,
                    debug_samples=debug_samples,
                    imputation_method=method_name,
                )
            except Exception as exc:  # pragma: no cover - 运行期保障
                print(f"    ❌ Evaluation failed ({method_name}): {exc}")
                continue

            ensure_parent(result_path)
            save_results_to_csv(results, str(result_path))
            print(f"    ✅ Results saved to {result_path}")

            save_intermediate_predictions(
                results=results,
                dataset_name=dataset_name,
                eval_data_name=eval_name,
                base_output_dir="datasets/Intermediate_Predictions",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "评估干净数据和填补数据的 Sundial 预测表现；"
            "默认使用 CUDA，并自动跳过已存在的结果。"
        )
    )
    parser.add_argument("--dataset", type=str, default=None, help="指定单个数据集名称，例如 ETTh1")
    parser.add_argument("--all_datasets", action="store_true", help="遍历 dataset_properties.json 中的所有数据集")
    parser.add_argument("--method", required=True, help="缺失模式，如 MCAR/BM/TM/TVMR")
    parser.add_argument(
        "--missing_ratios",
        type=str,
        default=None,
        help="逗号分隔缺失率（0-1 或百分比），留空则使用默认配置",
    )
    parser.add_argument(
        "--terms",
        type=str,
        default=None,
        help="限定 term，例如 short,medium。默认依据数据集属性",
    )
    parser.add_argument(
        "--imputation_methods",
        type=str,
        default=None,
        help="填补方法列表，逗号分隔；使用 all 表示全部",
    )
    parser.add_argument("--device", type=str, default="cuda", help="推理设备，默认 cuda")
    parser.add_argument("--num_samples", type=int, default=100, help="采样数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=None,
        help="覆盖默认推断长度，如未指定则自动根据 term/freq 计算",
    )
    parser.add_argument("--base_data_dir", type=str, default="datasets", help="数据集根目录")
    parser.add_argument(
        "--properties_path",
        type=str,
        default="datasets/dataset_properties.json",
        help="dataset_properties.json 路径",
    )
    parser.add_argument(
        "--block_length",
        type=int,
        default=None,
        help="BM 模式下的块长度（可选）",
    )
    parser.add_argument(
        "--clean_output_dir",
        type=str,
        default="results/sundial/sundial_Clean",
        help="干净数据评估结果输出目录",
    )
    parser.add_argument(
        "--impute_output_dir",
        type=str,
        default="results/sundial/sundial_Impute",
        help="填补数据评估结果输出目录",
    )
    parser.add_argument("--force", action="store_true", help="忽略已有结果，强制重新计算")
    parser.add_argument("--debug", action="store_true", help="开启 debug 表格输出")
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=5,
        help="debug 表格展示样本数",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.all_datasets and not args.dataset:
        raise ValueError("必须指定 --dataset 或开启 --all_datasets")

    dataset_names = (
        load_dataset_names(args.properties_path)
        if args.all_datasets
        else [args.dataset]
    )

    ratios = parse_ratio_list(args.missing_ratios)
    impute_methods = parse_imputation_methods(args.imputation_methods)
    device = resolve_device(args.device)

    clean_output_dir = Path(args.clean_output_dir)
    impute_output_dir = Path(args.impute_output_dir)
    clean_output_dir.mkdir(parents=True, exist_ok=True)
    impute_output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Dataset evaluation: {dataset_name}")
        print("=" * 80)

        allowed_terms = get_allowed_terms(dataset_name, args.properties_path)
        term_list = parse_terms(args.terms, allowed_terms)

        evaluate_clean_datasets(
            dataset_name=dataset_name,
            terms=term_list,
            device=device,
            prediction_length=args.prediction_length,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            base_data_dir=args.base_data_dir,
            properties_path=args.properties_path,
            output_dir=clean_output_dir,
            force=args.force,
        )

        evaluate_imputed_datasets(
            dataset_name=dataset_name,
            method=args.method,
            ratios=ratios,
            terms=term_list,
            imputation_methods=impute_methods,
            device=device,
            prediction_length=args.prediction_length,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            base_data_dir=args.base_data_dir,
            properties_path=args.properties_path,
            block_length=args.block_length,
            output_dir=impute_output_dir,
            force=args.force,
            debug=args.debug,
            debug_samples=args.debug_samples,
        )


if __name__ == "__main__":
    main()
