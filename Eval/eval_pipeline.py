"""
通用评估流程：数据加载、窗口生成、指标评估与结果保存。
"""

from __future__ import annotations

import csv
import json
import logging
import math
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality, norm_freq_str
from pandas.tseries.frequencies import to_offset


TEST_SPLIT = 0.6
MAX_WINDOW = 20


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        if self == Term.MEDIUM:
            return 10
        return 15


PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}


def maybe_reconvert_freq(freq: str) -> str:
    deprecated_map = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }
    return deprecated_map.get(freq, freq)


def compute_prediction_length(freq: str, term: Term = Term.SHORT) -> int:
    freq_normalized = norm_freq_str(to_offset(freq).name)
    freq_normalized = maybe_reconvert_freq(freq_normalized)
    base_pred_len = PRED_LENGTH_MAP.get(freq_normalized, 48)
    return term.multiplier * base_pred_len


def build_metrics():
    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        ),
    ]


class WarningFilter(logging.Filter):
    def __init__(self, text_to_filter):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


def load_datasets_for_evaluation(
    eval_data_path: str,
    clean_data_path: str,
    freq: str,
    term: str = "short",
    prediction_length: Optional[int] = None,
):
    if not Path(eval_data_path).exists():
        raise FileNotFoundError(f"Eval dataset file not found: {eval_data_path}")
    if not Path(clean_data_path).exists():
        raise FileNotFoundError(f"Clean dataset file not found: {clean_data_path}")

    eval_df = pd.read_csv(eval_data_path)
    clean_df = pd.read_csv(clean_data_path)

    time_col = None
    for c in ["date", "time", "timestamp"]:
        if c in eval_df.columns:
            time_col = c
            break

    if time_col:
        eval_df[time_col] = pd.to_datetime(eval_df[time_col])
        eval_df = eval_df.set_index(time_col)
        clean_df[time_col] = pd.to_datetime(clean_df[time_col])
        clean_df = clean_df.set_index(time_col)

    if len(eval_df) != len(clean_df):
        raise ValueError(
            f"Eval and clean datasets must have same length. Eval={len(eval_df)}, Clean={len(clean_df)}"
        )

    term_enum = Term(term)
    if prediction_length is None:
        prediction_length = compute_prediction_length(freq, term_enum)

    eval_test_data = []
    clean_test_data = []
    for i in range(len(eval_df.columns)):
        col_name = eval_df.columns[i]
        eval_test_data.append(
            {
                "target": eval_df.iloc[:, i].values.astype(np.float32),
                "start": pd.Period(eval_df.index[0], freq=freq.lower()),
                "item_id": f"eval_{col_name}",
            }
        )
        clean_test_data.append(
            {
                "target": clean_df.iloc[:, i].values.astype(np.float32),
                "start": pd.Period(clean_df.index[0], freq=freq.lower()),
                "item_id": f"clean_{col_name}",
            }
        )

    eval_list_dataset = ListDataset(eval_test_data, freq=freq.lower())
    clean_list_dataset = ListDataset(clean_test_data, freq=freq.lower())

    min_series_length = min(
        len(eval_df.iloc[:, i]) for i in range(len(eval_df.columns))
    )
    if "m4" in eval_data_path.lower():
        windows = 1
    else:
        w = math.ceil(TEST_SPLIT * min_series_length / prediction_length)
        windows = min(max(1, w), MAX_WINDOW)

    _, eval_template = split(eval_list_dataset, offset=-prediction_length * windows)
    eval_instances = eval_template.generate_instances(
        prediction_length=prediction_length,
        windows=windows,
        distance=prediction_length,
    )

    _, clean_template = split(clean_list_dataset, offset=-prediction_length * windows)
    clean_instances = clean_template.generate_instances(
        prediction_length=prediction_length,
        windows=windows,
        distance=prediction_length,
    )

    return eval_instances, clean_instances, prediction_length, windows


def print_debug_table(
    forecasts,
    eval_test_data,
    clean_test_data,
    pred_len: int,
    num_samples: int = 5,
):
    eval_data_list = list(eval_test_data)
    clean_data_list = list(clean_test_data)

    print(f"\n{'=' * 120}")
    print(
        f"{'Sample':<10} {'Time Step':<12} {'Prediction':<20} {'Eval Data':<20} {'Clean Data':<20}"
    )
    print(f"{'=' * 120}")

    for idx in range(min(num_samples, len(forecasts))):
        forecast = forecasts[idx]
        eval_entry = (
            eval_data_list[idx][0]
            if isinstance(eval_data_list[idx], tuple)
            else eval_data_list[idx]
        )
        clean_entry = (
            clean_data_list[idx][0]
            if isinstance(clean_data_list[idx], tuple)
            else clean_data_list[idx]
        )

        pred_mean = forecast.mean
        eval_start = len(eval_entry["target"]) - pred_len
        eval_values = eval_entry["target"][eval_start : eval_start + pred_len]
        clean_start = len(clean_entry["target"]) - pred_len
        clean_values = clean_entry["target"][clean_start : clean_start + pred_len]

        for t in range(min(pred_len, 10)):
            prefix = f"{idx:<10}" if t == 0 else f"{'':<10}"
            print(
                f"{prefix} {t:<12} {pred_mean[t]:<20.6f} {eval_values[t]:<20.6f} {clean_values[t]:<20.6f}"
            )
        if pred_len > 10:
            print(f"{'':<10} {'...':<12} {'...':<20} {'...':<20} {'...':<20}")
        print(f"{'-' * 120}")


def evaluate_with_adapter(
    model_adapter,
    model_name: str,
    eval_data_path: str,
    clean_data_path: str,
    freq: str,
    term: str,
    prediction_length: Optional[int] = None,
    batch_size: int = 32,
    debug: bool = False,
    debug_samples: int = 5,
) -> Dict[str, Any]:
    eval_test_data, clean_test_data, pred_len, windows = load_datasets_for_evaluation(
        eval_data_path=eval_data_path,
        clean_data_path=clean_data_path,
        freq=freq,
        term=term,
        prediction_length=prediction_length,
    )

    season_length = get_seasonality(freq)
    metrics = build_metrics()

    print(f"\nGenerating forecasts with model={model_name} ...")
    forecasts = model_adapter.predict(eval_test_data)

    if debug:
        print_debug_table(
            forecasts, eval_test_data, clean_test_data, pred_len, debug_samples
        )

    class SimplePredictor:
        def __init__(self, fcs):
            self.forecasts = fcs

        def predict(self, test_data):
            return self.forecasts

    evaluator = evaluate_model(
        SimplePredictor(forecasts),
        test_data=clean_test_data,
        metrics=metrics,
        batch_size=batch_size,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    results = {
        "model": model_name,
        "eval_data": eval_data_path,
        "clean_data": clean_data_path,
        "freq": freq,
        "term": term,
        "prediction_length": pred_len,
        "windows": windows,
        "forecasts": forecasts,
        "MSE[mean]": float(evaluator["MSE[mean]"].mean()),
        "MSE[0.5]": float(evaluator["MSE[0.5]"].mean()),
        "MAE[0.5]": float(evaluator["MAE[0.5]"].mean()),
        "MASE[0.5]": float(evaluator["MASE[0.5]"].mean()),
        "MAPE[0.5]": float(evaluator["MAPE[0.5]"].mean()),
        "sMAPE[0.5]": float(evaluator["sMAPE[0.5]"].mean()),
        "MSIS": float(evaluator["MSIS"].mean()),
        "RMSE[mean]": float(evaluator["RMSE[mean]"].mean()),
        "NRMSE[mean]": float(evaluator["NRMSE[mean]"].mean()),
        "ND[0.5]": float(evaluator["ND[0.5]"].mean()),
        "mean_weighted_sum_quantile_loss": float(
            evaluator["mean_weighted_sum_quantile_loss"].mean()
        ),
    }
    return results


def save_results_to_csv(results: Dict[str, Any], output_path: str):
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in results.items():
            if key == "forecasts":
                continue
            writer.writerow([key, value])

    print(f"\nResults saved to: {output_path}")


def get_frequency_from_properties(
    dataset_name: str,
    properties_path: str = "data/datasets/dataset_properties.json",
) -> str:
    props_path = Path(properties_path)
    if not props_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {props_path}")

    with open(props_path, "r", encoding="utf-8") as f:
        properties = json.load(f)
    if dataset_name not in properties:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    return properties[dataset_name].get("frequency", "H")


def get_allowed_terms(
    dataset_name: str,
    properties_path: str = "data/datasets/dataset_properties.json",
) -> List[str]:
    props_path = Path(properties_path)
    if not props_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {props_path}")

    with open(props_path, "r", encoding="utf-8") as f:
        properties = json.load(f)
    if dataset_name not in properties:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")

    term_type = properties[dataset_name].get("term", "med_long")
    return ["short"] if term_type == "short" else ["short", "medium", "long"]
