from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from Visualize.config import METRIC_COLUMNS
from Visualize.data_loader import build_clean_file_paths, load_metric_csv
from Visualize.plotters import metric_title


def _apply_panel_layout(fig, title: str, handles, labels) -> None:
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.14, 1, 0.9])


def run_clean_compare(
    results_analysis_dir: Path,
    results_pic_dir: Path,
    dataset: str,
    term: str,
    layout: str = "both",
) -> List[Path]:
    out_dir = results_pic_dir / "clean" / dataset / term

    expected_outputs: List[Path] = []
    if layout in {"single", "both"}:
        expected_outputs.extend(
            [out_dir / f"clean_{metric}.png" for metric in METRIC_COLUMNS]
        )
    if layout in {"panel", "both"}:
        expected_outputs.append(out_dir / "clean_panel.png")

    if expected_outputs and all(p.exists() for p in expected_outputs):
        print(f"[SKIP] clean {dataset}/{term} ({layout}) already generated")
        return expected_outputs

    paths = build_clean_file_paths(
        results_analysis_dir=results_analysis_dir,
        dataset=dataset,
        term=term,
    )

    clean_history_df = load_metric_csv(paths["clean_history"])
    gt_prediction_df = load_metric_csv(paths["gt_prediction"])

    out_dir.mkdir(parents=True, exist_ok=True)
    title_prefix = f"{dataset} | {term} | clean-only"

    outputs: List[Path] = []

    if layout in {"single", "both"}:
        for metric in METRIC_COLUMNS:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.plot(
                clean_history_df["window_idx"],
                clean_history_df[metric],
                linewidth=2.0,
                marker="s",
                markersize=3,
                label="clean history",
            )
            ax.plot(
                gt_prediction_df["window_idx"],
                gt_prediction_df[metric],
                linewidth=2.0,
                marker="o",
                markersize=3,
                label="clean prediction ground truth",
                color="black",
            )

            ax.set_title(metric_title(metric), fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlabel("Window Index")
            ax.set_ylabel("Metric Value")
            ax.legend(loc="best", fontsize=9)

            fig.suptitle(
                f"{title_prefix} | {metric_title(metric)}",
                fontsize=13,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            save_path = out_dir / f"clean_{metric}.png"
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
            outputs.append(save_path)

    if layout in {"panel", "both"}:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
        flat_axes = axes.flatten()

        for idx, metric in enumerate(METRIC_COLUMNS):
            ax = flat_axes[idx]
            ax.plot(
                clean_history_df["window_idx"],
                clean_history_df[metric],
                linewidth=2.0,
                marker="s",
                markersize=2.5,
                label="clean history",
            )
            ax.plot(
                gt_prediction_df["window_idx"],
                gt_prediction_df[metric],
                linewidth=2.0,
                marker="o",
                markersize=2.5,
                label="clean prediction ground truth",
                color="black",
            )
            ax.set_title(metric_title(metric), fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlabel("Window Index")
            ax.set_ylabel("Metric Value")

        handles, labels = flat_axes[0].get_legend_handles_labels()
        _apply_panel_layout(fig, f"{title_prefix} | 6 Metrics Panel", handles, labels)

        panel_path = out_dir / "clean_panel.png"
        fig.savefig(panel_path, dpi=220)
        plt.close(fig)
        outputs.append(panel_path)

    return outputs
