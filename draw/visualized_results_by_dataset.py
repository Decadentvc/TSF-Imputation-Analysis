"""
Visualize window-level distribution gap and prediction accuracy gap by dataset.

Compared with `visualized_results_by_mode.py`:
- one figure per dataset
- each figure contains points from multiple imputation methods
- same imputation method uses the same color across all figures
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from draw.visualized_results_by_mode import SUPPORTED_MODES, build_plot_dataframe
except ModuleNotFoundError:
    from visualized_results_by_mode import SUPPORTED_MODES, build_plot_dataframe


DEFAULT_METHODS = ("zero", "mean", "forward", "backward", "linear", "nearest", "spline", "seasonal")


def parse_mode_list(raw_modes: str) -> List[str]:
    modes = [m.strip().lower() for m in raw_modes.split(",") if m.strip()]
    if not modes:
        raise ValueError("No valid modes were provided.")

    unsupported = sorted({m for m in modes if m not in SUPPORTED_MODES})
    if unsupported:
        raise ValueError(f"Unsupported modes: {unsupported}. Supported: {sorted(SUPPORTED_MODES)}")
    return modes


def parse_dataset_filter(raw_datasets: Optional[str]) -> Optional[set[str]]:
    if not raw_datasets:
        return None
    datasets = {d.strip() for d in raw_datasets.split(",") if d.strip()}
    return datasets or None


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def build_combined_dataframe(
    model: str,
    modes: Sequence[str],
    results_analysis_root: Path,
    intermediate_root: Path,
    data_root: Path,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for mode in modes:
        mode_df = build_plot_dataframe(
            model=model,
            prediction_mode=mode,
            results_analysis_root=results_analysis_root,
            intermediate_root=intermediate_root,
            data_root=data_root,
        )
        if mode_df.empty:
            continue
        with_mode = mode_df.copy()
        with_mode["prediction_mode"] = mode
        frames.append(with_mode)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def draw_dataset_scatter(
    df: pd.DataFrame,
    model: str,
    dataset: str,
    modes: Sequence[str],
    color_map: Dict[str, Tuple[float, float, float, float]],
    output_path: Path,
    max_points: Optional[int] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    plot_df = df.copy()
    if max_points and len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=random_seed)

    y_values = np.where(
        plot_df["prediction_mode"].to_numpy() == "clean",
        0.0,
        plot_df["smape_diff_vs_clean"].to_numpy(dtype=float),
    )
    plot_df["y_value"] = y_values

    fig, ax = plt.subplots(figsize=(12, 8))
    used_modes = [m for m in modes if m in set(plot_df["prediction_mode"].unique())]
    for mode in used_modes:
        sub = plot_df[plot_df["prediction_mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(
            sub["distribution_gap"],
            sub["y_value"],
            s=20,
            alpha=0.72,
            color=color_map[mode],
            label=mode,
            edgecolors="none",
        )

    ax.set_xlabel("History vs Prediction Distribution Gap (Standardized Euclidean Distance)")
    ax.set_ylabel("Prediction Error Difference (sMAPE[0.5] - Clean sMAPE[0.5])")
    ax.set_title(f"{model} | dataset={dataset} | Window-level Gap vs Prediction Error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return plot_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize window-level distribution gap and prediction accuracy by dataset."
    )
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. sundial/chronos2/timesfm2p5")
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated modes to include. "
            "Default: zero,mean,forward,backward,linear,nearest,spline,seasonal"
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset filter. Default: include all matched datasets.",
    )
    parser.add_argument("--results_analysis_dir", type=str, default="results_analysis")
    parser.add_argument("--intermediate_dir", type=str, default="data/Intermediate_Predictions")
    parser.add_argument("--data_dir", type=str, default="data/datasets")
    parser.add_argument("--max_points", type=int, default=1200, help="Optional subsample size per dataset figure")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: draw/outputs_by_dataset/{model}",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model = args.model.strip().lower()
    modes = parse_mode_list(args.modes)
    dataset_filter = parse_dataset_filter(args.datasets)

    results_analysis_root = Path(args.results_analysis_dir)
    intermediate_root = Path(args.intermediate_dir)
    data_root = Path(args.data_dir)
    output_root = Path(args.output_dir) if args.output_dir else Path("draw/outputs_by_dataset") / model

    combined = build_combined_dataframe(
        model=model,
        modes=modes,
        results_analysis_root=results_analysis_root,
        intermediate_root=intermediate_root,
        data_root=data_root,
    )
    if combined.empty:
        raise RuntimeError("No valid windows were matched for the selected model/modes.")

    if dataset_filter is not None:
        combined = combined[combined["dataset"].isin(dataset_filter)].copy()
    if combined.empty:
        raise RuntimeError("No data left after dataset filtering.")

    datasets = sorted(combined["dataset"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(modes)))
    color_map = {mode: cmap(i) for i, mode in enumerate(modes)}

    total_points = 0
    for dataset in datasets:
        ds_df = combined[combined["dataset"] == dataset].copy()
        if ds_df.empty:
            continue

        file_stem = f"{sanitize_filename(dataset)}_window_gap_scatter_by_imputation"
        png_out = output_root / f"{file_stem}.png"
        csv_out = output_root / f"{file_stem}.csv"

        plotted = draw_dataset_scatter(
            df=ds_df,
            model=model,
            dataset=dataset,
            modes=modes,
            color_map=color_map,
            output_path=png_out,
            max_points=args.max_points,
            random_seed=args.random_seed,
        )
        plotted = plotted.sort_values(
            ["prediction_mode", "term", "ratio_code", "window_idx"],
            na_position="last",
        )
        plotted.to_csv(csv_out, index=False)
        total_points += len(plotted)

        print(f"[OK] plot saved: {png_out}")
        print(f"[OK] points saved: {csv_out}")
        print(f"[INFO] dataset={dataset}, points={len(plotted)}")

    print(f"[INFO] total datasets: {len(datasets)}")
    print(f"[INFO] total plotted points: {total_points}")


if __name__ == "__main__":
    main()
