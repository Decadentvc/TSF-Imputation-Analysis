from __future__ import annotations

from pathlib import Path
from typing import List

from Visualize.plots.clean_compare import run_clean_compare
from Visualize.plots.method_compare import run_method_compare


def run_plot(
    plot_type: str,
    results_analysis_dir: Path,
    results_pic_dir: Path,
    model: str | None,
    dataset: str,
    term: str,
    ratio: str | None,
    methods: str | None,
    layout: str = "both",
) -> List[Path]:
    if plot_type == "method":
        if ratio is None:
            raise ValueError("plot_type=method requires --missing-ratio")
        if not model:
            raise ValueError("plot_type=method requires --model")
        return run_method_compare(
            results_analysis_dir=results_analysis_dir,
            results_pic_dir=results_pic_dir,
            model=model,
            dataset=dataset,
            term=term,
            ratio=ratio,
            methods_arg=methods,
            layout=layout,
        )

    if plot_type == "clean":
        return run_clean_compare(
            results_analysis_dir=results_analysis_dir,
            results_pic_dir=results_pic_dir,
            dataset=dataset,
            term=term,
            layout=layout,
        )

    raise ValueError(f"Unsupported plot_type: {plot_type}")
