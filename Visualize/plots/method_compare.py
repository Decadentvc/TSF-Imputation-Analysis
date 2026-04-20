from __future__ import annotations

from pathlib import Path
from typing import List

from Visualize.config import METRIC_COLUMNS
from Visualize.data_loader import (
    build_file_paths,
    discover_imputation_methods,
    load_metric_csv,
)
from Visualize.plotters import (
    plot_history_all_methods,
    plot_history_all_methods_panel,
    plot_prediction_all_methods,
    plot_prediction_all_methods_panel,
)


def run_method_compare(
    results_analysis_dir: Path,
    results_pic_dir: Path,
    model: str,
    dataset: str,
    term: str,
    ratio: str,
    methods_arg: str | None = None,
    layout: str = "both",
) -> List[Path]:
    history_dir = results_analysis_dir / model / "history"
    prediction_dir = results_analysis_dir / model / "prediction"

    if not history_dir.exists() or not prediction_dir.exists():
        raise FileNotFoundError(
            f"Model result folders not found: {history_dir} or {prediction_dir}"
        )

    if methods_arg:
        methods = [m.strip().lower() for m in methods_arg.split(",") if m.strip()]
    else:
        methods = discover_imputation_methods(
            history_dir=history_dir,
            prediction_dir=prediction_dir,
            dataset=dataset,
            term=term,
            ratio=ratio,
        )

    if not methods:
        raise ValueError(
            "No matching imputation methods found. "
            f"model={model}, dataset={dataset}, term={term}, BM={ratio}"
        )

    out_dir = results_pic_dir / "method" / model / dataset / term / f"BM_{ratio}"

    expected_outputs: List[Path] = []
    if layout in {"single", "both"}:
        expected_outputs.extend(
            [out_dir / "history" / f"history_{metric}.png" for metric in METRIC_COLUMNS]
        )
        expected_outputs.extend(
            [
                out_dir / "prediction" / f"prediction_{metric}.png"
                for metric in METRIC_COLUMNS
            ]
        )
    if layout in {"panel", "both"}:
        expected_outputs.append(out_dir / "history" / "history_panel.png")
        expected_outputs.append(out_dir / "prediction" / "prediction_panel.png")

    if expected_outputs and all(p.exists() for p in expected_outputs):
        print(
            f"[SKIP] method {model}/{dataset}/{term}/BM_{ratio} "
            f"({layout}) already generated"
        )
        return expected_outputs

    history_frames = {}
    prediction_frames = {}
    clean_history_df = None
    clean_prediction_df = None
    gt_prediction_df = None

    for method in methods:
        paths = build_file_paths(
            results_analysis_dir=results_analysis_dir,
            model=model,
            dataset=dataset,
            term=term,
            ratio=ratio,
            method=method,
        )

        history_frames[method] = load_metric_csv(paths["imputed_history"])
        prediction_frames[method] = load_metric_csv(paths["imputed_prediction"])

        if clean_history_df is None:
            clean_history_df = load_metric_csv(paths["clean_history"])
        if clean_prediction_df is None:
            clean_prediction_df = load_metric_csv(paths["clean_prediction"])
        if gt_prediction_df is None:
            gt_prediction_df = load_metric_csv(paths["gt_prediction"])

    assert clean_history_df is not None
    assert clean_prediction_df is not None
    assert gt_prediction_df is not None

    title_prefix = f"{model} | {dataset} | {term} | BM_{ratio}"

    outputs: List[Path] = []

    if layout in {"single", "both"}:
        history_outputs = plot_history_all_methods(
            method_frames=history_frames,
            clean_df=clean_history_df,
            save_dir=out_dir / "history",
            title_prefix=title_prefix,
        )
        prediction_outputs = plot_prediction_all_methods(
            method_frames=prediction_frames,
            clean_prediction_df=clean_prediction_df,
            gt_prediction_df=gt_prediction_df,
            save_dir=out_dir / "prediction",
            title_prefix=title_prefix,
        )
        outputs.extend(history_outputs)
        outputs.extend(prediction_outputs)

    if layout in {"panel", "both"}:
        outputs.append(
            plot_history_all_methods_panel(
                method_frames=history_frames,
                clean_df=clean_history_df,
                save_dir=out_dir / "history",
                title_prefix=title_prefix,
            )
        )
        outputs.append(
            plot_prediction_all_methods_panel(
                method_frames=prediction_frames,
                clean_prediction_df=clean_prediction_df,
                gt_prediction_df=gt_prediction_df,
                save_dir=out_dir / "prediction",
                title_prefix=title_prefix,
            )
        )

    return outputs
