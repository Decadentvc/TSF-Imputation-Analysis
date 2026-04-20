from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Visualize.router import run_plot


def split_csv(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return vals or None


def in_filters(value: str, filters: list[str] | None) -> bool:
    if filters is None:
        return True
    return value in filters


def model_dirs(results_analysis_dir: Path) -> list[Path]:
    dirs: list[Path] = []
    for p in sorted(results_analysis_dir.iterdir()):
        if not p.is_dir() or p.name == "clean_prediction_windows":
            continue
        if (p / "history").exists() and (p / "prediction").exists():
            dirs.append(p)
    return dirs


def discover_method_tasks(
    results_analysis_dir: Path,
    model_filters: list[str] | None,
    dataset_filters: list[str] | None,
    term_filters: list[str] | None,
    ratio_filters: list[str] | None,
) -> list[tuple[str, str, str, str]]:
    history_re = re.compile(
        r"^(.+)_BM_(\d{3})_(short|medium|long)_[A-Za-z0-9]+_history\.csv$"
    )
    pred_re = re.compile(
        r"^(.+)_BM_(\d{3})_(short|medium|long)_[A-Za-z0-9]+_prediction\.csv$"
    )

    clean_pred_dir = results_analysis_dir / "clean_prediction_windows"
    tasks: set[tuple[str, str, str, str]] = set()

    for mdir in model_dirs(results_analysis_dir):
        model = mdir.name
        if not in_filters(model, model_filters):
            continue

        history_dir = mdir / "history"
        prediction_dir = mdir / "prediction"

        history_keys = {
            (m.group(1), m.group(2), m.group(3))
            for p in history_dir.glob("*.csv")
            for m in [history_re.match(p.name)]
            if m
        }
        pred_keys = {
            (m.group(1), m.group(2), m.group(3))
            for p in prediction_dir.glob("*.csv")
            for m in [pred_re.match(p.name)]
            if m
        }

        for dataset, ratio, term in history_keys & pred_keys:
            if not in_filters(dataset, dataset_filters):
                continue
            if not in_filters(term, term_filters):
                continue
            if not in_filters(ratio, ratio_filters):
                continue

            if not (history_dir / f"{dataset}_clean_{term}_history.csv").exists():
                continue
            if not (prediction_dir / f"{dataset}_clean_{term}_prediction.csv").exists():
                continue
            if not (
                clean_pred_dir / f"{dataset}_clean_{term}_prediction_gt.csv"
            ).exists():
                continue

            tasks.add((model, dataset, term, ratio))

    return sorted(tasks)


def discover_clean_tasks(
    results_analysis_dir: Path,
    dataset_filters: list[str] | None,
    term_filters: list[str] | None,
) -> list[tuple[str, str]]:
    history_re = re.compile(r"^(.+)_clean_(short|medium|long)_history\.csv$")

    clean_pred_dir = results_analysis_dir / "clean_prediction_windows"
    tasks: set[tuple[str, str]] = set()

    for mdir in model_dirs(results_analysis_dir):
        history_dir = mdir / "history"

        history_keys = {
            (m.group(1), m.group(2))
            for p in history_dir.glob("*.csv")
            for m in [history_re.match(p.name)]
            if m
        }

        for dataset, term in history_keys:
            if not in_filters(dataset, dataset_filters):
                continue
            if not in_filters(term, term_filters):
                continue
            if not (
                clean_pred_dir / f"{dataset}_clean_{term}_prediction_gt.csv"
            ).exists():
                continue
            tasks.add((dataset, term))

    return sorted(tasks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch generate method/clean plots")
    p.add_argument(
        "--plot-types", default="method,clean", help="method,clean or subset"
    )
    p.add_argument("--models", default=None, help="Filter models, comma-separated")
    p.add_argument("--datasets", default=None, help="Filter datasets, comma-separated")
    p.add_argument("--terms", default=None, help="Filter terms, comma-separated")
    p.add_argument(
        "--missing-ratios",
        default=None,
        help="Filter BM ratios, comma-separated like 010,020",
    )
    p.add_argument(
        "--methods", default=None, help="Optional fixed methods for method mode"
    )
    p.add_argument(
        "--layout",
        default="both",
        choices=["single", "panel", "both"],
        help="Output layout: single, panel, or both",
    )
    p.add_argument("--results-analysis-dir", default="results_analysis")
    p.add_argument("--results-pic-dir", default="results_pic")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    plot_types = split_csv(args.plot_types) or ["method", "clean"]
    model_filters = split_csv(args.models)
    dataset_filters = split_csv(args.datasets)
    term_filters = split_csv(args.terms)
    ratio_filters = split_csv(args.missing_ratios)

    ra_dir = Path(args.results_analysis_dir)
    rp_dir = Path(args.results_pic_dir)

    total_jobs = 0
    ok_jobs = 0
    total_plots = 0

    if "method" in plot_types:
        method_tasks = discover_method_tasks(
            results_analysis_dir=ra_dir,
            model_filters=model_filters,
            dataset_filters=dataset_filters,
            term_filters=term_filters,
            ratio_filters=ratio_filters,
        )
        print(f"[INFO] method tasks: {len(method_tasks)}")
        for model, dataset, term, ratio in method_tasks:
            total_jobs += 1
            try:
                outputs = run_plot(
                    plot_type="method",
                    results_analysis_dir=ra_dir,
                    results_pic_dir=rp_dir,
                    model=model,
                    dataset=dataset,
                    term=term,
                    ratio=ratio,
                    methods=args.methods,
                    layout=args.layout,
                )
                ok_jobs += 1
                total_plots += len(outputs)
                print(
                    f"[OK] method {model}/{dataset}/{term}/BM_{ratio}: {len(outputs)}"
                )
            except Exception as exc:
                print(f"[FAIL] method {model}/{dataset}/{term}/BM_{ratio}: {exc}")

    if "clean" in plot_types:
        clean_tasks = discover_clean_tasks(
            results_analysis_dir=ra_dir,
            dataset_filters=dataset_filters,
            term_filters=term_filters,
        )
        print(f"[INFO] clean tasks: {len(clean_tasks)}")
        for dataset, term in clean_tasks:
            total_jobs += 1
            try:
                outputs = run_plot(
                    plot_type="clean",
                    results_analysis_dir=ra_dir,
                    results_pic_dir=rp_dir,
                    model=None,
                    dataset=dataset,
                    term=term,
                    ratio=None,
                    methods=None,
                    layout=args.layout,
                )
                ok_jobs += 1
                total_plots += len(outputs)
                print(f"[OK] clean {dataset}/{term}: {len(outputs)}")
            except Exception as exc:
                print(f"[FAIL] clean {dataset}/{term}: {exc}")

    print("-" * 60)
    print(f"[DONE] jobs={ok_jobs}/{total_jobs}, plots={total_plots}")


if __name__ == "__main__":
    main()
