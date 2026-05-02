"""
驱动脚本：在每个模型现有 mean 结果覆盖范围上，补跑指定的若干填补方法。

用法（默认 4 个新填补方法、device=cuda、跳过已有结果）：

    python tools/run_new_imputers_driver.py --model sundial

    python tools/run_new_imputers_driver.py --model sundial \
        --imputation_methods kalman_struct,kalman_arima,gp_rbf,saits \
        --device cuda

可选 --models a,b,c 同时跑多个模型；脚本对每个模型的覆盖按
results/<model>/impute/mean_*_results.csv 推断 (dataset, ratio, term)，
确保新填补方法的覆盖与现有 4 个 imputer 完全对齐。
"""

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Eval"))

from run_eval import _build_impute_result_filename, run_single_evaluation  # noqa: E402

DEFAULT_MODELS = [
    "sundial",
    "chronos2",
    "timesfm2p5",
    "kairos23m",
    "kairos50m",
    "timesfm2p0",
    "visiontspp",
]
DEFAULT_IMPUTERS = ["kalman_struct", "kalman_arima", "gp_rbf", "saits"]

MEAN_RE = re.compile(
    r"^mean_(?P<dataset>.+?)_BM_length(?P<L>\d+)_(?P<ratio>\d{3})_(?P<term>short|medium|long)_results\.csv$"
)


def _parse_csv_arg(raw: str) -> List[str]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return parts


def discover_targets(
    model: str, repo_root: Path
) -> List[Tuple[str, str, int, str, int]]:
    """返回 [(dataset, term, ratio_int, eval_path, block_length), ...]."""

    impute_dir = repo_root / "results" / model / "impute"
    if not impute_dir.exists():
        print(f"  [WARN] {impute_dir} 不存在，跳过 {model}")
        return []

    targets: List[Tuple[str, str, int, str, int]] = []
    for f in sorted(impute_dir.glob("mean_*_results.csv")):
        m = MEAN_RE.match(f.name)
        if not m:
            continue
        ds = m["dataset"]
        ratio = int(m["ratio"])
        term = m["term"]
        L = int(m["L"])
        bm_path = (
            repo_root
            / "data"
            / "datasets"
            / "BM"
            / f"BM_{ratio:03d}"
            / f"{ds}_BM_length{L}_{ratio:03d}_{term}.csv"
        )
        if not bm_path.exists():
            print(f"    [SKIP] 缺失 BM 数据文件: {bm_path}")
            continue
        targets.append((ds, term, ratio, str(bm_path), L))
    return targets


def cleanup_runtime(device: str) -> None:
    gc.collect()
    if device.lower().startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def run_for_model(
    model: str,
    imputers: List[str],
    device: str,
    force: bool,
    num_samples: int,
    batch_size: int,
    torch_dtype: str,
    repo_root: Path,
) -> Tuple[int, int, int, int]:
    targets = discover_targets(model, repo_root)
    if not targets:
        print(f"  [WARN] {model}: 无可跑目标")
        return (0, 0, 0, 0)

    impute_dir = repo_root / "results" / model / "impute"
    total = len(targets) * len(imputers)
    succeeded = skipped = failed = 0
    print(
        f"\n=== Model: {model} | targets={len(targets)} | imputers={imputers} | total tasks={total} ==="
    )

    t_start = time.time()
    done = 0
    for imputer in imputers:
        for ds, term, ratio, bm_path, L in targets:
            done += 1
            eval_name = Path(bm_path).stem
            result_filename = _build_impute_result_filename(imputer, eval_name, term)
            result_path = impute_dir / result_filename
            if result_path.exists() and not force:
                skipped += 1
                continue

            elapsed = time.time() - t_start
            print(
                f"  [{done}/{total}] {model} | {ds} | {term} | r={ratio:03d} | {imputer} | "
                f"elapsed {elapsed:.0f}s"
            )
            try:
                run_single_evaluation(
                    model=model,
                    model_name=None,
                    eval_data_path=bm_path,
                    clean_data_path=None,
                    term=term,
                    base_data_dir=str(repo_root / "data" / "datasets"),
                    properties_path=str(
                        repo_root / "data" / "datasets" / "dataset_properties.json"
                    ),
                    output_dir=str(impute_dir),
                    prediction_length=None,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    device=device,
                    imputation_method=imputer,
                    imputed_data_dir=str(repo_root / "data" / "datasets" / "Imputed"),
                    intermediate_dir=str(repo_root / "data" / "Intermediate_Predictions"),
                    predict_batches_jointly=False,
                    torch_dtype=torch_dtype if torch_dtype else None,
                    model_properties_path=str(
                        repo_root / "Eval" / "model_properties.json"
                    ),
                    random_seed=42,
                )
                succeeded += 1
            except Exception as exc:
                failed += 1
                print(f"    [FAIL] {ds}/{term}/r={ratio:03d}/{imputer}: {exc}")
            finally:
                cleanup_runtime(device)

    elapsed = time.time() - t_start
    print(
        f"  -> {model} 完成: succeeded={succeeded} skipped={skipped} failed={failed} "
        f"elapsed={elapsed:.0f}s"
    )
    return (total, succeeded, skipped, failed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models",
        type=str,
        default=None,
        help="逗号分隔的模型列表；省略则全部 7 个模型",
    )
    ap.add_argument(
        "--model",
        type=str,
        default=None,
        help="单个模型；与 --models 互斥，等价于 --models <name>",
    )
    ap.add_argument(
        "--imputation_methods",
        type=str,
        default=",".join(DEFAULT_IMPUTERS),
        help="逗号分隔的填补方法列表",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--num_samples", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--torch_dtype", type=str, default=None)
    args = ap.parse_args()

    if args.model and args.models:
        raise SystemExit("只能指定 --model 或 --models 之一")
    if args.model:
        models = [args.model]
    elif args.models:
        models = _parse_csv_arg(args.models)
    else:
        models = DEFAULT_MODELS
    imputers = _parse_csv_arg(args.imputation_methods)

    repo_root = REPO_ROOT
    print(f"REPO_ROOT = {repo_root}")
    print(f"models    = {models}")
    print(f"imputers  = {imputers}")
    print(f"device    = {args.device}")
    print(f"force     = {args.force}")

    grand_total = grand_ok = grand_skip = grand_fail = 0
    for model in models:
        total, ok, skip, fail = run_for_model(
            model=model,
            imputers=imputers,
            device=args.device,
            force=args.force,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            torch_dtype=args.torch_dtype,
            repo_root=repo_root,
        )
        grand_total += total
        grand_ok += ok
        grand_skip += skip
        grand_fail += fail

    print("\n" + "=" * 70)
    print(
        f"GRAND TOTAL: tasks={grand_total} succeeded={grand_ok} skipped={grand_skip} failed={grand_fail}"
    )
    if grand_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
