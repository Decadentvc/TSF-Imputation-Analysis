from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

from Visualize.config import METRIC_COLUMNS


def load_metric_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = ["window_idx", *METRIC_COLUMNS]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    return df.sort_values("window_idx").reset_index(drop=True)


def discover_imputation_methods(
    history_dir: Path,
    prediction_dir: Path,
    dataset: str,
    term: str,
    ratio: str,
) -> List[str]:
    history_pattern = re.compile(
        rf"^{re.escape(dataset)}_BM_{ratio}_{term}_([A-Za-z0-9]+)_history\.csv$"
    )
    prediction_pattern = re.compile(
        rf"^{re.escape(dataset)}_BM_{ratio}_{term}_([A-Za-z0-9]+)_prediction\.csv$"
    )

    history_methods = {
        m.group(1).lower()
        for p in history_dir.glob("*.csv")
        for m in [history_pattern.match(p.name)]
        if m
    }
    prediction_methods = {
        m.group(1).lower()
        for p in prediction_dir.glob("*.csv")
        for m in [prediction_pattern.match(p.name)]
        if m
    }
    return sorted(history_methods & prediction_methods)


def build_file_paths(
    results_analysis_dir: Path,
    model: str,
    dataset: str,
    term: str,
    ratio: str,
    method: str,
) -> Dict[str, Path]:
    model_dir = results_analysis_dir / model
    history_dir = model_dir / "history"
    prediction_dir = model_dir / "prediction"
    clean_pred_dir = results_analysis_dir / "clean_prediction_windows"

    return {
        "imputed_history": history_dir
        / f"{dataset}_BM_{ratio}_{term}_{method}_history.csv",
        "clean_history": history_dir / f"{dataset}_clean_{term}_history.csv",
        "imputed_prediction": prediction_dir
        / f"{dataset}_BM_{ratio}_{term}_{method}_prediction.csv",
        "clean_prediction": prediction_dir / f"{dataset}_clean_{term}_prediction.csv",
        "gt_prediction": clean_pred_dir / f"{dataset}_clean_{term}_prediction_gt.csv",
    }


def build_clean_file_paths(
    results_analysis_dir: Path,
    dataset: str,
    term: str,
) -> Dict[str, Path]:
    clean_history_name = f"{dataset}_clean_{term}_history.csv"
    gt_prediction = (
        results_analysis_dir
        / "clean_prediction_windows"
        / f"{dataset}_clean_{term}_prediction_gt.csv"
    )

    clean_history: Path | None = None
    for model_dir in sorted(results_analysis_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "clean_prediction_windows":
            continue
        history_path = model_dir / "history" / clean_history_name
        if history_path.exists():
            clean_history = history_path
            break

    if clean_history is None:
        raise FileNotFoundError(
            f"No clean history file found across models: {clean_history_name}"
        )

    return {
        "clean_history": clean_history,
        "gt_prediction": gt_prediction,
    }
