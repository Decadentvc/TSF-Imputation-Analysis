"""
批量评估入口：在 Eval/run_eval.py 基础上增加“已生成结果自动跳过”。

示例：
python Eval/run_batch_eval.py \
  --model sundial \
  --dataset ETTh1 \
  --method BM \
  --terms short,medium \
  --imputation_methods linear,mean,forward \
  --missing_ratios 0.10,0.20,0.30
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

from run_eval import (
    find_clean_dataset_path,
    generate_eval_dataset_paths,
    get_allowed_terms,
    run_single_evaluation,
)


def _split_multi_values(raw_values: Optional[List[str]]) -> Optional[List[str]]:
    """支持空格分隔与逗号分隔混用。"""
    if not raw_values:
        return None

    values: List[str] = []
    for chunk in raw_values:
        for part in chunk.split(","):
            item = part.strip()
            if item:
                values.append(item)
    return values if values else None


def _parse_missing_ratios(raw_values: Optional[List[str]]) -> Optional[List[float]]:
    """
    允许输入格式：
    - 0.1 / 0.10
    - 10 / 20（视作百分比）
    - 010 / 020（视作百分比）
    """
    values = _split_multi_values(raw_values)
    if not values:
        return None

    ratios: List[float] = []
    for token in values:
        v = float(token)
        if v > 1:
            v = v / 100.0
        if v <= 0 or v >= 1:
            raise ValueError(f"Invalid missing ratio: {token}. Expected range (0, 1)")
        ratios.append(round(v, 6))

    # 去重并保持稳定顺序
    deduped: List[float] = []
    seen = set()
    for r in ratios:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def _dedupe_lower(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for v in values:
        key = v.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _default_impute_result_dir(model: str) -> Path:
    return Path("results") / model.lower() / "impute"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch evaluate one model with skip-existing support"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["sundial", "chronos2", "timesfm2p5"],
        help="Model type",
    )
    parser.add_argument("--model_name", type=str, default=None, help="Model checkpoint")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--method", type=str, default="BM", choices=["BM"])
    parser.add_argument("--block_length", type=int, default=None)

    parser.add_argument(
        "--terms",
        nargs="+",
        default=None,
        help="One or more terms, e.g. short,medium or short medium",
    )
    parser.add_argument(
        "--imputation_methods",
        nargs="+",
        default=["linear", "mean", "forward"],
        help="One or more imputation methods",
    )
    parser.add_argument(
        "--missing_ratios",
        nargs="+",
        default=None,
        help="One or more ratios, e.g. 0.1,0.2,0.3 or 10 20 30",
    )

    parser.add_argument("--base_data_dir", type=str, default="data/datasets")
    parser.add_argument(
        "--properties_path",
        type=str,
        default="data/datasets/dataset_properties.json",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--imputed_data_dir", type=str, default="data/datasets/Imputed")
    parser.add_argument(
        "--intermediate_dir", type=str, default="data/Intermediate_Predictions"
    )

    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--predict_batches_jointly", action="store_true")
    parser.add_argument("--torch_dtype", type=str, default=None)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even when result file already exists",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    allowed_terms = get_allowed_terms(args.dataset, args.properties_path)
    user_terms = _split_multi_values(args.terms)
    if user_terms is None:
        terms = allowed_terms
    else:
        normalized_terms = _dedupe_lower(user_terms)
        invalid_terms = [t for t in normalized_terms if t not in allowed_terms]
        if invalid_terms:
            raise ValueError(
                f"Invalid terms for {args.dataset}: {invalid_terms}, allowed: {allowed_terms}"
            )
        terms = normalized_terms

    imputation_methods = _split_multi_values(args.imputation_methods)
    if not imputation_methods:
        raise ValueError("--imputation_methods is empty")
    imputation_methods = _dedupe_lower(imputation_methods)
    imputation_methods = [m for m in imputation_methods if m != "none"]
    if not imputation_methods:
        raise ValueError("No valid imputation methods provided (none is not allowed)")

    missing_ratios = _parse_missing_ratios(args.missing_ratios)

    eval_paths_with_terms = generate_eval_dataset_paths(
        dataset_name=args.dataset,
        method=args.method,
        missing_ratios=missing_ratios,
        base_data_dir=args.base_data_dir,
        block_length=args.block_length,
        properties_path=args.properties_path,
    )
    eval_paths_with_terms = [
        (eval_path, term)
        for eval_path, term in eval_paths_with_terms
        if term in terms
    ]

    clean_data_path = find_clean_dataset_path(args.dataset, args.base_data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else _default_impute_result_dir(args.model)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    failed = 0
    succeeded = 0

    print("=" * 80)
    print("Batch Evaluation (skip existing enabled)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Terms: {terms}")
    print(f"Imputation methods: {imputation_methods}")
    print(f"Missing ratios: {missing_ratios if missing_ratios else 'default from run_eval.py'}")
    print(f"Output dir: {output_dir}")

    for eval_path, term in eval_paths_with_terms:
        eval_file = Path(eval_path)
        if not eval_file.exists():
            print(f"⚠ Missing eval dataset, skip file: {eval_path}")
            continue

        eval_name = eval_file.stem
        for method in imputation_methods:
            total += 1
            result_path = output_dir / f"{method}_{eval_name}_{term}_results.csv"
            if result_path.exists() and not args.force:
                skipped += 1
                print(f"✓ Skip existing result: {result_path}")
                continue

            print(f"\n▶ Running: {eval_name} | term={term} | impute={method}")
            try:
                run_single_evaluation(
                    model=args.model,
                    model_name=args.model_name,
                    eval_data_path=eval_path,
                    clean_data_path=clean_data_path,
                    term=term,
                    base_data_dir=args.base_data_dir,
                    properties_path=args.properties_path,
                    output_dir=str(output_dir),
                    prediction_length=args.prediction_length,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    imputation_method=method,
                    imputed_data_dir=args.imputed_data_dir,
                    intermediate_dir=args.intermediate_dir,
                    predict_batches_jointly=args.predict_batches_jointly,
                    torch_dtype=args.torch_dtype,
                )
                succeeded += 1
            except Exception as exc:
                failed += 1
                print(f"✗ Failed: {eval_name} / {method} -> {exc}")

    print("\n" + "=" * 80)
    print("Done")
    print("=" * 80)
    print(f"Total tasks: {total}")
    print(f"Succeeded:   {succeeded}")
    print(f"Skipped:     {skipped}")
    print(f"Failed:      {failed}")


if __name__ == "__main__":
    main()
