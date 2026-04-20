from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from Visualize.config import METRIC_COLUMNS


METHOD_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def metric_title(metric_key: str) -> str:
    return metric_key.replace("_", " ").title()


def _style_axis(ax, metric: str) -> None:
    ax.set_title(metric_title(metric), fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Metric Value")


def _apply_panel_layout(fig, title: str, handles, labels) -> None:
    legend_ncol = min(max(2, len(labels) // 2), 5)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=legend_ncol,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0.14, 1, 0.9])


def plot_history_all_methods(
    method_frames: Dict[str, pd.DataFrame],
    clean_df: pd.DataFrame,
    save_dir: Path,
    title_prefix: str,
) -> List[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    for metric in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(11, 6))

        for idx, (method, df) in enumerate(sorted(method_frames.items())):
            ax.plot(
                df["window_idx"],
                df[metric],
                linewidth=1.6,
                marker="o",
                markersize=3,
                label=f"{method} (imputed history)",
                color=METHOD_COLORS[idx % len(METHOD_COLORS)],
            )

        ax.plot(
            clean_df["window_idx"],
            clean_df[metric],
            linewidth=2.2,
            marker="s",
            markersize=3,
            label="clean history",
            color="tab:purple",
            alpha=0.9,
        )

        _style_axis(ax, metric)
        ax.legend(loc="best", fontsize=9)
        fig.suptitle(f"{title_prefix} | History | {metric_title(metric)}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = save_dir / f"history_{metric}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        outputs.append(save_path)

    return outputs


def plot_history_all_methods_panel(
    method_frames: Dict[str, pd.DataFrame],
    clean_df: pd.DataFrame,
    save_dir: Path,
    title_prefix: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    flat_axes = axes.flatten()

    for metric_idx, metric in enumerate(METRIC_COLUMNS):
        ax = flat_axes[metric_idx]
        for idx, (method, df) in enumerate(sorted(method_frames.items())):
            ax.plot(
                df["window_idx"],
                df[metric],
                linewidth=1.4,
                marker="o",
                markersize=2.5,
                label=f"{method} (imputed history)",
                color=METHOD_COLORS[idx % len(METHOD_COLORS)],
            )
        ax.plot(
            clean_df["window_idx"],
            clean_df[metric],
            linewidth=2.0,
            marker="s",
            markersize=2.5,
            label="clean history",
            color="tab:purple",
            alpha=0.9,
        )
        _style_axis(ax, metric)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    _apply_panel_layout(
        fig,
        f"{title_prefix} | History | 6 Metrics Panel",
        handles,
        labels,
    )

    save_path = save_dir / "history_panel.png"
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    return save_path


def plot_prediction_all_methods(
    method_frames: Dict[str, pd.DataFrame],
    clean_prediction_df: pd.DataFrame,
    gt_prediction_df: pd.DataFrame,
    save_dir: Path,
    title_prefix: str,
) -> List[Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    for metric in METRIC_COLUMNS:
        fig, ax = plt.subplots(figsize=(11, 6))

        for idx, (method, df) in enumerate(sorted(method_frames.items())):
            ax.plot(
                df["window_idx"],
                df[metric],
                linewidth=1.6,
                marker="o",
                markersize=3,
                label=f"{method} (imputed prediction)",
                color=METHOD_COLORS[idx % len(METHOD_COLORS)],
            )

        ax.plot(
            clean_prediction_df["window_idx"],
            clean_prediction_df[metric],
            linewidth=2.2,
            marker="s",
            markersize=3,
            label="clean prediction",
            color="tab:orange",
            alpha=0.9,
        )
        ax.plot(
            gt_prediction_df["window_idx"],
            gt_prediction_df[metric],
            linewidth=2.2,
            marker="^",
            markersize=3,
            label="ground truth prediction",
            color="black",
            alpha=0.9,
        )

        _style_axis(ax, metric)
        ax.legend(loc="best", fontsize=9)
        fig.suptitle(
            f"{title_prefix} | Prediction | {metric_title(metric)}", fontsize=13
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        save_path = save_dir / f"prediction_{metric}.png"
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
        outputs.append(save_path)

    return outputs


def plot_prediction_all_methods_panel(
    method_frames: Dict[str, pd.DataFrame],
    clean_prediction_df: pd.DataFrame,
    gt_prediction_df: pd.DataFrame,
    save_dir: Path,
    title_prefix: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    flat_axes = axes.flatten()

    for metric_idx, metric in enumerate(METRIC_COLUMNS):
        ax = flat_axes[metric_idx]
        for idx, (method, df) in enumerate(sorted(method_frames.items())):
            ax.plot(
                df["window_idx"],
                df[metric],
                linewidth=1.4,
                marker="o",
                markersize=2.5,
                label=f"{method} (imputed prediction)",
                color=METHOD_COLORS[idx % len(METHOD_COLORS)],
            )
        ax.plot(
            clean_prediction_df["window_idx"],
            clean_prediction_df[metric],
            linewidth=2.0,
            marker="s",
            markersize=2.5,
            label="clean prediction",
            color="tab:orange",
            alpha=0.9,
        )
        ax.plot(
            gt_prediction_df["window_idx"],
            gt_prediction_df[metric],
            linewidth=2.0,
            marker="^",
            markersize=2.5,
            label="ground truth prediction",
            color="black",
            alpha=0.9,
        )
        _style_axis(ax, metric)

    handles, labels = flat_axes[0].get_legend_handles_labels()
    _apply_panel_layout(
        fig,
        f"{title_prefix} | Prediction | 6 Metrics Panel",
        handles,
        labels,
    )

    save_path = save_dir / "prediction_panel.png"
    fig.savefig(save_path, dpi=220)
    plt.close(fig)
    return save_path
