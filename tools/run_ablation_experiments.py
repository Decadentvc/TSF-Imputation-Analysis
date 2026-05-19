"""Build or run command plans for the section 4.3 ablation experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


DEFAULT_RATIO_MODELS = ["timesfm2p0", "timesfm2p5", "visiontspp", "chronos2"]
DEFAULT_SHAPE_MODELS = ["timesfm2p0", "timesfm2p5", "visiontspp", "chronos2"]
DEFAULT_HORIZON_MODELS = ["timesfm2p0", "timesfm2p5", "sundial", "visiontspp"]
DEFAULT_CONTEXT_MODELS = ["chronos2", "timesfm2p0", "timesfm2p5", "visiontspp"]

DEFAULT_RATIO_DATASETS = [
    "electricity",
    "ETTh1",
    "weather",
    "traffic",
    "exchange_rate",
    "azure2019_U_5T",
]
DEFAULT_SHAPE_DATASETS = ["ETTh1", "weather", "traffic", "exchange_rate"]
DEFAULT_RATIO_VALUES = ["0.05", "0.10", "0.20", "0.30", "0.40", "0.50"]
DEFAULT_LENGTH_VALUES = [10, 25, 50, 100, 200]
DEFAULT_HORIZONS = [96, 192, 336, 720]
DEFAULT_CONTEXTS = [512, 1024, 2048, 4096]
DEFAULT_RATIO_IMPUTERS = ["mean", "forward", "linear", "gp_rbf", "saits"]
DEFAULT_SHAPE_IMPUTERS = ["mean", "linear", "gp_rbf", "saits"]
DEFAULT_HORIZON_IMPUTERS = ["linear", "mean", "saits"]


def _split_csv(value: Optional[str], default: Sequence[str]) -> List[str]:
    if not value:
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


def _list2cmdline(cmd: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


def _python_cmd(script: str) -> List[str]:
    return [sys.executable, script]


def _append_common_eval_args(cmd: List[str], args: argparse.Namespace) -> List[str]:
    cmd.extend(["--device", args.device])
    cmd.extend(["--batch_size", str(args.batch_size)])
    cmd.extend(["--num_samples", str(args.num_samples)])
    if args.torch_dtype:
        cmd.extend(["--torch_dtype", args.torch_dtype])
    if args.model_name:
        cmd.extend(["--model_name", args.model_name])
    return cmd


def _batch_eval_command(
    model: str,
    dataset: str,
    suite: str,
    imputation_methods: Sequence[str],
    ratios: Sequence[str],
    block_length: int,
    args: argparse.Namespace,
) -> List[str]:
    cmd = _python_cmd("Eval/run_batch_eval.py")
    cmd.extend(
        [
            "--model",
            model,
            "--dataset",
            dataset,
            "--method",
            "BM",
            "--terms",
            args.terms,
            "--missing_ratios",
            ",".join(ratios),
            "--block_length",
            str(block_length),
            "--imputation_methods",
            ",".join(imputation_methods),
            "--output_dir",
            str(Path(args.results_root) / suite / model / "impute"),
            "--imputed_data_dir",
            str(Path(args.imputed_root) / suite),
            "--intermediate_dir",
            str(Path(args.intermediate_root) / suite),
            "--preserve_imputed_stem",
        ]
    )
    if args.force:
        cmd.append("--force")
    _append_common_eval_args(cmd, args)
    return cmd


def _single_eval_command(
    model: str,
    eval_path: Path,
    suite: str,
    imputation_method: str,
    args: argparse.Namespace,
    prediction_length: Optional[int] = None,
    max_context: Optional[int] = None,
) -> List[str]:
    dataset = eval_path.stem.split("_BM_")[0]
    term = eval_path.stem.split("_")[-1]
    cmd = _python_cmd("Eval/run_eval.py")
    cmd.extend(
        [
            "single",
            "--model",
            model,
            "--eval_data_path",
            str(eval_path),
            "--clean_data_path",
            str(Path(args.base_data_dir) / "ori" / f"{dataset}.csv"),
            "--term",
            term,
            "--imputation_method",
            imputation_method,
            "--output_dir",
            str(Path(args.results_root) / suite / model / "impute"),
            "--imputed_data_dir",
            str(Path(args.imputed_root) / suite),
            "--intermediate_dir",
            str(Path(args.intermediate_root) / suite),
            "--preserve_imputed_stem",
        ]
    )
    if prediction_length is not None:
        cmd.extend(["--prediction_length", str(prediction_length)])
    if max_context is not None:
        cmd.extend(["--max_context", str(max_context)])
    _append_common_eval_args(cmd, args)
    return cmd


def _standard_file(dataset: str, ratio: str, term: str, block_length: int, base_data_dir: str) -> Path:
    ratio_text = f"{int(round(float(ratio) * 100)):03d}"
    return (
        Path(base_data_dir)
        / "BM"
        / f"BM_{ratio_text}"
        / f"{dataset}_BM_length{block_length}_{ratio_text}_{term}.csv"
    )


def _existing_variant_files(root: Path, term: str) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.glob(f"BM_030/*_{term}.csv"))


def build_commands(args: argparse.Namespace) -> List[List[str]]:
    suites = _split_csv(args.suites, ["ratio", "length", "position", "pattern", "horizon", "context"])
    terms = _split_csv(args.terms, ["long"])
    if len(terms) != 1:
        raise ValueError("Command planning currently expects one term at a time")
    term = terms[0]
    args.terms = term

    commands: List[List[str]] = []

    if "ratio" in suites:
        models = _split_csv(args.models, DEFAULT_RATIO_MODELS)
        datasets = _split_csv(args.datasets, DEFAULT_RATIO_DATASETS)
        imputers = _split_csv(args.imputation_methods, DEFAULT_RATIO_IMPUTERS)
        ratios = _split_csv(args.ratios, DEFAULT_RATIO_VALUES)
        for model in models:
            for dataset in datasets:
                commands.append(
                    _batch_eval_command(
                        model=model,
                        dataset=dataset,
                        suite="ratio",
                        imputation_methods=imputers,
                        ratios=ratios,
                        block_length=args.block_length,
                        args=args,
                    )
                )

    if "length" in suites:
        models = _split_csv(args.models, DEFAULT_SHAPE_MODELS)
        datasets = _split_csv(args.datasets, DEFAULT_SHAPE_DATASETS)
        imputers = _split_csv(args.imputation_methods, DEFAULT_SHAPE_IMPUTERS)
        lengths = [int(item) for item in _split_csv(args.block_lengths, [str(v) for v in DEFAULT_LENGTH_VALUES])]
        for model in models:
            for dataset in datasets:
                for block_length in lengths:
                    commands.append(
                        _batch_eval_command(
                            model=model,
                            dataset=dataset,
                            suite="length",
                            imputation_methods=imputers,
                            ratios=[str(args.ratio)],
                            block_length=block_length,
                            args=args,
                        )
                    )

    if "position" in suites:
        models = _split_csv(args.models, DEFAULT_SHAPE_MODELS)
        imputers = _split_csv(args.imputation_methods, DEFAULT_SHAPE_IMPUTERS)
        files = _existing_variant_files(Path(args.base_data_dir) / "BM_POSITION", term)
        for model in models:
            for eval_path in files:
                for imputer in imputers:
                    commands.append(_single_eval_command(model, eval_path, "position", imputer, args))

    if "pattern" in suites:
        models = _split_csv(args.models, DEFAULT_SHAPE_MODELS)
        imputers = _split_csv(args.imputation_methods, DEFAULT_SHAPE_IMPUTERS)
        files = _existing_variant_files(Path(args.base_data_dir) / "BM_PATTERN", term)
        for model in models:
            for eval_path in files:
                for imputer in imputers:
                    commands.append(_single_eval_command(model, eval_path, "pattern", imputer, args))

    if "horizon" in suites:
        models = _split_csv(args.models, DEFAULT_HORIZON_MODELS)
        datasets = _split_csv(args.datasets, DEFAULT_SHAPE_DATASETS)
        imputers = _split_csv(args.imputation_methods, DEFAULT_HORIZON_IMPUTERS)
        horizons = [int(item) for item in _split_csv(args.horizons, [str(v) for v in DEFAULT_HORIZONS])]
        for horizon in horizons:
            suite = f"horizon_H{horizon:03d}"
            for model in models:
                for dataset in datasets:
                    eval_path = _standard_file(dataset, str(args.ratio), term, args.block_length, args.base_data_dir)
                    if not eval_path.exists():
                        continue
                    for imputer in imputers:
                        commands.append(
                            _single_eval_command(
                                model,
                                eval_path,
                                suite,
                                imputer,
                                args,
                                prediction_length=horizon,
                                max_context=args.horizon_context,
                            )
                        )

    if "context" in suites:
        models = _split_csv(args.models, DEFAULT_CONTEXT_MODELS)
        datasets = _split_csv(args.datasets, DEFAULT_SHAPE_DATASETS)
        imputers = _split_csv(args.imputation_methods, DEFAULT_HORIZON_IMPUTERS)
        contexts = [int(item) for item in _split_csv(args.contexts, [str(v) for v in DEFAULT_CONTEXTS])]
        for context in contexts:
            suite = f"context_L{context:04d}"
            for model in models:
                for dataset in datasets:
                    eval_path = _standard_file(dataset, str(args.ratio), term, args.block_length, args.base_data_dir)
                    if not eval_path.exists():
                        continue
                    for imputer in imputers:
                        commands.append(
                            _single_eval_command(
                                model,
                                eval_path,
                                suite,
                                imputer,
                                args,
                                prediction_length=args.context_horizon,
                                max_context=context,
                            )
                        )

    if args.limit is not None:
        commands = commands[: args.limit]
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run section 4.3 ablation experiments")
    parser.add_argument("--suites", default="ratio,length,position,pattern,horizon,context")
    parser.add_argument("--models", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--terms", default="long")
    parser.add_argument("--imputation_methods", default=None)
    parser.add_argument("--ratios", default=None)
    parser.add_argument("--ratio", type=float, default=0.30)
    parser.add_argument("--block_length", type=int, default=50)
    parser.add_argument("--block_lengths", default=None)
    parser.add_argument("--horizons", default=None)
    parser.add_argument("--contexts", default=None)
    parser.add_argument("--horizon_context", type=int, default=2048)
    parser.add_argument("--context_horizon", type=int, default=720)
    parser.add_argument("--base_data_dir", default="data/datasets")
    parser.add_argument("--results_root", default="results_ablation")
    parser.add_argument("--imputed_root", default="data/datasets/Imputed_Ablation")
    parser.add_argument("--intermediate_root", default="data/Intermediate_Predictions_Ablation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--torch_dtype", default=None)
    parser.add_argument("--model_name", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument("--write_commands", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = build_commands(args)
    lines = [_list2cmdline(cmd) for cmd in commands]

    if args.write_commands:
        output_path = Path(args.write_commands)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"Wrote {len(lines)} commands to {output_path}")
    else:
        for line in lines:
            print(line)
        print(f"Total commands: {len(lines)}")

    if args.execute:
        for idx, cmd in enumerate(commands, 1):
            print(f"[{idx}/{len(commands)}] {_list2cmdline(cmd)}")
            completed = subprocess.run(cmd, check=False)
            if completed.returncode != 0 and not args.keep_going:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
