"""
通用评估脚本：支持 sundial / chronos2 / timesfm2p5。
旧脚本 run_sundial.py 暂保留，本文件为新入口。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from eval_pipeline import (
    evaluate_with_adapter,
    get_allowed_terms,
    get_frequency_from_properties,
    save_results_to_csv,
)
from impute_dataset import generate_imputed_dataset_path, impute_dataset
from model_registry import build_model_adapter


ALLOWED_MISSING_METHODS = {"BM"}
DEFAULT_MODEL_PROPERTIES_PATH = "Eval/model_properties.json"


def load_model_properties(
    model_properties_path: str = DEFAULT_MODEL_PROPERTIES_PATH,
) -> Dict[str, Any]:
    props_path = Path(model_properties_path)
    if not props_path.exists():
        raise FileNotFoundError(f"Model properties not found: {props_path}")
    with open(props_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_max_context(
    model: str,
    model_properties_path: str = DEFAULT_MODEL_PROPERTIES_PATH,
) -> int:
    model_props = load_model_properties(model_properties_path)
    model_key = model.lower()
    if model_key not in model_props:
        available = ", ".join(sorted(model_props.keys()))
        raise ValueError(
            f"Model '{model}' not found in model properties. Available: {available}"
        )

    max_context = model_props[model_key].get("max_context")
    if max_context is None:
        raise ValueError(f"Model '{model}' missing required field: max_context")
    if not isinstance(max_context, int) or max_context <= 0:
        raise ValueError(f"Model '{model}' has invalid max_context: {max_context}")
    return max_context


def parse_eval_dataset_name(eval_path: str) -> Tuple[str, str]:
    """从评估文件名解析 (original_dataset_name, term)。"""
    stem = Path(eval_path).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid eval dataset filename: {stem}")

    term = parts[-1].lower()
    if term not in {"short", "medium", "long"}:
        raise ValueError(f"Invalid term in filename: {stem}")

    method_idx = None
    for i, token in enumerate(parts):
        if token.upper() in ALLOWED_MISSING_METHODS:
            method_idx = i
            break

    if method_idx is None:
        # 若不是注空文件，退化为去掉末尾 term
        original_name = "_".join(parts[:-1])
    else:
        original_tokens = [
            p for p in parts[:method_idx] if not p.lower().startswith("length")
        ]
        original_name = "_".join(original_tokens)

    return original_name, term


def find_clean_dataset_path(
    original_name: str, base_data_dir: str = "data/datasets"
) -> str:
    clean_path = Path(base_data_dir) / "ori" / f"{original_name}.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {clean_path}")
    return str(clean_path)


def check_and_impute_dataset(
    eval_data_path: str,
    imputation_method: str,
    imputed_data_dir: str = "data/datasets/Imputed",
) -> str:
    imputed_path = generate_imputed_dataset_path(
        eval_data_path=eval_data_path,
        imputation_method=imputation_method,
        base_output_dir=imputed_data_dir,
    )
    if Path(imputed_path).exists():
        print(f"  ✓ Imputed dataset exists: {imputed_path}")
        return imputed_path

    print("  [WARN] Imputed dataset not found, generating...")
    impute_dataset(
        eval_data_path=eval_data_path,
        imputation_method=imputation_method,
        output_path=imputed_path,
        base_output_dir=imputed_data_dir,
        save_result=True,
    )
    return imputed_path


def batch_check_and_impute(
    eval_data_paths: List[str],
    imputation_methods: List[str],
    imputed_data_dir: str = "data/datasets/Imputed",
) -> Dict[Tuple[str, str], str]:
    imputed_paths_map: Dict[Tuple[str, str], str] = {}
    for eval_path in eval_data_paths:
        for method in imputation_methods:
            imputed_path = check_and_impute_dataset(eval_path, method, imputed_data_dir)
            imputed_paths_map[(eval_path, method)] = imputed_path
    return imputed_paths_map


def _resolve_method_root(base_data_dir: str, method: str) -> Path:
    # 新目录优先：data/datasets/Block_Missing/BM_010/...
    if method.upper() == "BM":
        bm_root = Path(base_data_dir) / "Block_Missing"
        if bm_root.exists():
            return bm_root

    # 兼容旧目录：data/datasets/BM/BM_010/...
    legacy_root = Path(base_data_dir) / method.upper()
    if legacy_root.exists():
        return legacy_root

    # 默认回到 Block_Missing（主要用于 BM）
    return Path(base_data_dir) / "Block_Missing"


def generate_eval_dataset_paths(
    dataset_name: str,
    method: str,
    missing_ratios: Optional[List[float]] = None,
    base_data_dir: str = "data/datasets",
    block_length: Optional[int] = None,
    properties_path: str = "data/datasets/dataset_properties.json",
) -> List[Tuple[str, str]]:
    if missing_ratios is None:
        missing_ratios = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    allowed_terms = get_allowed_terms(dataset_name, properties_path)
    eval_paths: List[Tuple[str, str]] = []

    root = _resolve_method_root(base_data_dir, method)
    method_key = method.upper()

    for ratio in missing_ratios:
        ratio_str = f"{int(ratio * 100):03d}"
        ratio_dir = root / f"{method_key}_{ratio_str}"

        for term in allowed_terms:
            eval_path = None
            if method_key == "BM":
                if block_length is not None:
                    eval_filename = f"{dataset_name}_{method_key}_length{block_length}_{ratio_str}_{term}.csv"
                    eval_path = ratio_dir / eval_filename
                else:
                    pattern = (
                        f"{dataset_name}_{method_key}_length*_{ratio_str}_{term}.csv"
                    )
                    matches = (
                        sorted(ratio_dir.glob(pattern)) if ratio_dir.exists() else []
                    )
                    if matches:
                        eval_path = matches[0]

            if eval_path is None:
                eval_filename = f"{dataset_name}_{method_key}_{ratio_str}_{term}.csv"
                eval_path = ratio_dir / eval_filename

            eval_paths.append((str(eval_path), term))

    return eval_paths


def _extract_point_prediction(forecast) -> Any:
    # SampleForecast 一般有 mean
    try:
        pred = forecast.mean
        if pred is not None:
            return pred
    except Exception:
        pass

    # QuantileForecast：使用 0.5 分位
    if hasattr(forecast, "quantile"):
        try:
            return forecast.quantile(0.5)
        except Exception:
            return forecast.quantile("0.5")

    raise ValueError("Unsupported forecast object: cannot extract point prediction")


def save_intermediate_predictions(
    results: Dict[str, Any],
    model: str,
    dataset_name: str,
    eval_data_name: str,
    intermediate_dir: str = "data/Intermediate_Predictions",
    imputation_method: Optional[str] = None,
):
    import numpy as np
    import pandas as pd

    forecasts = results.get("forecasts")
    if not forecasts:
        print("Warning: no forecasts in results, skip intermediate saving")
        return

    prediction_length = results["prediction_length"]
    freq = results["freq"]

    model_dir = Path(intermediate_dir) / model.lower()
    if imputation_method:
        output_dir = model_dir / f"{eval_data_name}_prediction" / imputation_method
    else:
        output_dir = model_dir / f"{eval_data_name}_prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    for window_idx, forecast in enumerate(forecasts):
        point_prediction = np.asarray(_extract_point_prediction(forecast))
        start_date = forecast.start_date
        if hasattr(start_date, "to_timestamp"):
            start_date = start_date.to_timestamp()

        date_range = pd.date_range(
            start=start_date, periods=prediction_length, freq=freq
        )
        df = pd.DataFrame({"date": date_range, "prediction": point_prediction})
        output_file = output_dir / f"{eval_data_name}_prediction_{window_idx}.csv"
        df.to_csv(output_file, index=False)

    print(f"  [OK] Intermediate predictions saved: {output_dir}")


def _default_result_subdir(model: str, mode: str) -> str:
    # 用户定义：仅 clean / impute 两类
    return str(
        Path("results") / model.lower() / ("clean" if mode == "clean" else "impute")
    )


def _build_impute_result_filename(method_name: str, eval_name: str, term: str) -> str:
    term_norm = term.lower()
    if eval_name.lower().endswith(f"_{term_norm}"):
        return f"{method_name.lower()}_{eval_name}_results.csv"
    return f"{method_name.lower()}_{eval_name}_{term_norm}_results.csv"


def run_single_evaluation(
    model: str,
    model_name: Optional[str],
    eval_data_path: str,
    clean_data_path: Optional[str] = None,
    term: Optional[str] = None,
    base_data_dir: str = "data/datasets",
    properties_path: str = "data/datasets/dataset_properties.json",
    output_dir: Optional[str] = None,
    prediction_length: Optional[int] = None,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    imputation_method: Optional[str] = None,
    imputed_data_dir: str = "data/datasets/Imputed",
    intermediate_dir: str = "data/Intermediate_Predictions",
    predict_batches_jointly: bool = False,
    torch_dtype: Optional[str] = None,
    model_properties_path: str = DEFAULT_MODEL_PROPERTIES_PATH,
):
    eval_path = Path(eval_data_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {eval_path}")

    if clean_data_path is None:
        original_name, parsed_term = parse_eval_dataset_name(str(eval_path))
        term = parsed_term if term is None else term
        clean_path = find_clean_dataset_path(original_name, base_data_dir)
    else:
        original_name = Path(clean_data_path).stem
        if term is None:
            raise ValueError("Specified mode requires --term")
        clean_path = clean_data_path

    freq = get_frequency_from_properties(original_name, properties_path)

    # clean 模式：直接 eval==clean，不做填补
    is_clean_mode = Path(eval_data_path).resolve() == Path(clean_path).resolve()

    actual_eval_path = str(eval_path)
    if not is_clean_mode:
        if not imputation_method or imputation_method.lower() == "none":
            raise ValueError(
                "Missing-data evaluation requires imputation_method (none not allowed)"
            )
        actual_eval_path = check_and_impute_dataset(
            eval_data_path=str(eval_path),
            imputation_method=imputation_method,
            imputed_data_dir=imputed_data_dir,
        )

    max_context = get_model_max_context(model, model_properties_path)

    adapter = build_model_adapter(
        model=model,
        model_name=model_name,
        prediction_length=prediction_length or 1,
        batch_size=batch_size,
        device=device,
        num_samples=num_samples,
        predict_batches_jointly=predict_batches_jointly,
        torch_dtype=torch_dtype,
        max_context=max_context,
    )

    # 若未显式指定 prediction_length，这里先用 pipeline 自动计算值再回写 adapter
    if prediction_length is None:
        from eval_pipeline import compute_prediction_length, Term

        computed = compute_prediction_length(freq, Term(term))
        adapter.prediction_length = computed

    results = evaluate_with_adapter(
        model_adapter=adapter,
        model_name=model_name or model,
        eval_data_path=actual_eval_path,
        clean_data_path=clean_path,
        freq=freq,
        term=term,
        prediction_length=prediction_length,
        batch_size=batch_size,
        debug=False,
        debug_samples=5,
    )

    eval_name = eval_path.stem
    mode = "clean" if is_clean_mode else "impute"
    out_dir = output_dir or _default_result_subdir(model, mode)
    if mode == "impute":
        method_name = imputation_method if imputation_method is not None else "none"
        filename = _build_impute_result_filename(method_name, eval_name, term)
    else:
        filename = f"{eval_name}_{term}_results.csv"

    output_path = Path(out_dir) / filename
    save_results_to_csv(results, str(output_path))

    save_intermediate_predictions(
        results=results,
        model=model,
        dataset_name=original_name,
        eval_data_name=eval_name,
        intermediate_dir=intermediate_dir,
        imputation_method=imputation_method if mode == "impute" else None,
    )
    return results


def batch_evaluate(
    model: str,
    model_name: Optional[str],
    dataset_name: str,
    method: str,
    missing_ratios: Optional[List[float]] = None,
    base_data_dir: str = "data/datasets",
    properties_path: str = "data/datasets/dataset_properties.json",
    output_dir: Optional[str] = None,
    prediction_length: Optional[int] = None,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    imputation_methods: Optional[List[str]] = None,
    block_length: Optional[int] = None,
    imputed_data_dir: str = "data/datasets/Imputed",
    intermediate_dir: str = "data/Intermediate_Predictions",
    predict_batches_jointly: bool = False,
    torch_dtype: Optional[str] = None,
    model_properties_path: str = DEFAULT_MODEL_PROPERTIES_PATH,
) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    if imputation_methods is None:
        imputation_methods = [
            "zero",
            "mean",
            "forward",
            "backward",
            "linear",
            "nearest",
            "spline",
            "seasonal",
        ]
    imputation_methods = [m.lower() for m in imputation_methods if m.lower() != "none"]
    if not imputation_methods:
        raise ValueError("No valid imputation methods provided")

    eval_paths = generate_eval_dataset_paths(
        dataset_name=dataset_name,
        method=method,
        missing_ratios=missing_ratios,
        base_data_dir=base_data_dir,
        block_length=block_length,
        properties_path=properties_path,
    )
    clean_path = find_clean_dataset_path(dataset_name, base_data_dir)
    freq = get_frequency_from_properties(dataset_name, properties_path)
    max_context = get_model_max_context(model, model_properties_path)

    eval_path_list = [p for p, _ in eval_paths]
    imputed_map = batch_check_and_impute(
        eval_path_list, imputation_methods, imputed_data_dir
    )

    all_results: List[Tuple[str, str, str, Dict[str, Any]]] = []
    for eval_path, term in eval_paths:
        if not Path(eval_path).exists():
            print(f"Warning: missing eval dataset, skip: {eval_path}")
            continue

        for impute_method in imputation_methods:
            try:
                actual_eval_path = imputed_map[(eval_path, impute_method)]
                adapter = build_model_adapter(
                    model=model,
                    model_name=model_name,
                    prediction_length=prediction_length or 1,
                    batch_size=batch_size,
                    device=device,
                    num_samples=num_samples,
                    predict_batches_jointly=predict_batches_jointly,
                    torch_dtype=torch_dtype,
                    max_context=max_context,
                )
                if prediction_length is None:
                    from eval_pipeline import compute_prediction_length, Term

                    adapter.prediction_length = compute_prediction_length(
                        freq, Term(term)
                    )

                results = evaluate_with_adapter(
                    model_adapter=adapter,
                    model_name=model_name or model,
                    eval_data_path=actual_eval_path,
                    clean_data_path=clean_path,
                    freq=freq,
                    term=term,
                    prediction_length=prediction_length,
                    batch_size=batch_size,
                    debug=False,
                    debug_samples=5,
                )

                eval_name = Path(eval_path).stem
                out_dir = output_dir or _default_result_subdir(model, "impute")
                filename = _build_impute_result_filename(impute_method, eval_name, term)
                save_results_to_csv(results, str(Path(out_dir) / filename))

                save_intermediate_predictions(
                    results=results,
                    model=model,
                    dataset_name=dataset_name,
                    eval_data_name=eval_name,
                    intermediate_dir=intermediate_dir,
                    imputation_method=impute_method,
                )
                all_results.append((eval_path, term, impute_method, results))
            except Exception as exc:
                print(f"Error with {eval_path} / {impute_method}: {exc}")
                continue
    return all_results


def evaluate_clean(
    model: str,
    model_name: Optional[str],
    dataset_name: str,
    term: str = "short",
    base_data_dir: str = "data/datasets",
    properties_path: str = "data/datasets/dataset_properties.json",
    output_dir: Optional[str] = None,
    prediction_length: Optional[int] = None,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    intermediate_dir: str = "data/Intermediate_Predictions",
    predict_batches_jointly: bool = False,
    torch_dtype: Optional[str] = None,
    model_properties_path: str = DEFAULT_MODEL_PROPERTIES_PATH,
) -> Dict[str, Any]:
    clean_path = Path(base_data_dir) / "ori" / f"{dataset_name}.csv"
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {clean_path}")

    freq = get_frequency_from_properties(dataset_name, properties_path)
    max_context = get_model_max_context(model, model_properties_path)
    adapter = build_model_adapter(
        model=model,
        model_name=model_name,
        prediction_length=prediction_length or 1,
        batch_size=batch_size,
        device=device,
        num_samples=num_samples,
        predict_batches_jointly=predict_batches_jointly,
        torch_dtype=torch_dtype,
        max_context=max_context,
    )
    if prediction_length is None:
        from eval_pipeline import compute_prediction_length, Term

        adapter.prediction_length = compute_prediction_length(freq, Term(term))

    results = evaluate_with_adapter(
        model_adapter=adapter,
        model_name=model_name or model,
        eval_data_path=str(clean_path),
        clean_data_path=str(clean_path),
        freq=freq,
        term=term,
        prediction_length=prediction_length,
        batch_size=batch_size,
        debug=False,
        debug_samples=5,
    )

    out_dir = output_dir or _default_result_subdir(model, "clean")
    output_file = Path(out_dir) / f"{dataset_name}_clean_{term}_results.csv"
    save_results_to_csv(results, str(output_file))

    save_intermediate_predictions(
        results=results,
        model=model,
        dataset_name=dataset_name,
        eval_data_name=f"{dataset_name}_clean_{term}",
        intermediate_dir=intermediate_dir,
        imputation_method=None,
    )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run generic TSF model evaluation")
    subparsers = parser.add_subparsers(dest="mode", help="Evaluation mode")

    def add_common_args(p):
        p.add_argument(
            "--model",
            type=str,
            default="sundial",
            choices=[
                "sundial",
                "chronos2",
                "timesfm2p5",
                "kairos23m",
                "kairos50m",
                "timesfm2p0",
                "visiontspp",
            ],
        )
        p.add_argument("--model_name", type=str, default=None)
        p.add_argument("--base_data_dir", type=str, default="data/datasets")
        p.add_argument(
            "--properties_path",
            type=str,
            default="data/datasets/dataset_properties.json",
        )
        p.add_argument("--output_dir", type=str, default=None)
        p.add_argument("--prediction_length", type=int, default=None)
        p.add_argument("--num_samples", type=int, default=100)
        p.add_argument("--batch_size", type=int, default=32)
        p.add_argument("--device", type=str, default="cpu")
        p.add_argument(
            "--intermediate_dir", type=str, default="data/Intermediate_Predictions"
        )
        p.add_argument("--predict_batches_jointly", action="store_true")
        p.add_argument("--torch_dtype", type=str, default=None)
        p.add_argument(
            "--model_properties_path",
            type=str,
            default=DEFAULT_MODEL_PROPERTIES_PATH,
        )

    single = subparsers.add_parser("single", help="Evaluate single missing dataset")
    single.add_argument("--eval_data_path", type=str, required=True)
    single.add_argument("--clean_data_path", type=str, default=None)
    single.add_argument(
        "--term", type=str, choices=["short", "medium", "long"], default=None
    )
    single.add_argument("--imputation_method", type=str, required=True)
    single.add_argument("--imputed_data_dir", type=str, default="data/datasets/Imputed")
    add_common_args(single)

    batch = subparsers.add_parser("batch", help="Batch evaluate missing datasets")
    batch.add_argument("--dataset", type=str, required=True)
    batch.add_argument("--method", type=str, required=True, choices=["BM"])
    batch.add_argument("--missing_ratios", type=str, default=None)
    batch.add_argument("--imputation_methods", type=str, default=None)
    batch.add_argument("--block_length", type=int, default=None)
    batch.add_argument("--imputed_data_dir", type=str, default="data/datasets/Imputed")
    add_common_args(batch)

    clean = subparsers.add_parser("clean", help="Evaluate clean dataset")
    clean.add_argument("--dataset", type=str, required=True)
    clean.add_argument(
        "--term", type=str, choices=["short", "medium", "long"], default="short"
    )
    add_common_args(clean)

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    try:
        if args.mode == "single":
            run_single_evaluation(
                model=args.model,
                model_name=args.model_name,
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
                imputed_data_dir=args.imputed_data_dir,
                intermediate_dir=args.intermediate_dir,
                predict_batches_jointly=args.predict_batches_jointly,
                torch_dtype=args.torch_dtype,
                model_properties_path=args.model_properties_path,
            )
        elif args.mode == "batch":
            missing_ratios = (
                [float(x.strip()) for x in args.missing_ratios.split(",")]
                if args.missing_ratios
                else None
            )
            imputation_methods = (
                [x.strip().lower() for x in args.imputation_methods.split(",")]
                if args.imputation_methods
                else None
            )
            batch_evaluate(
                model=args.model,
                model_name=args.model_name,
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
                imputed_data_dir=args.imputed_data_dir,
                intermediate_dir=args.intermediate_dir,
                predict_batches_jointly=args.predict_batches_jointly,
                torch_dtype=args.torch_dtype,
                model_properties_path=args.model_properties_path,
            )
        elif args.mode == "clean":
            evaluate_clean(
                model=args.model,
                model_name=args.model_name,
                dataset_name=args.dataset,
                term=args.term,
                base_data_dir=args.base_data_dir,
                properties_path=args.properties_path,
                output_dir=args.output_dir,
                prediction_length=args.prediction_length,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                device=args.device,
                intermediate_dir=args.intermediate_dir,
                predict_batches_jointly=args.predict_batches_jointly,
                torch_dtype=args.torch_dtype,
                model_properties_path=args.model_properties_path,
            )
    except Exception as e:
        print(f"\n[ERROR] Error during evaluation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
