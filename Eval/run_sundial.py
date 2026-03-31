import os
import csv
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, set_seed
from tqdm.auto import tqdm
from gluonts.itertools import batcher
from gluonts.transform import LastValueImputation
from gluonts.model.forecast import SampleForecast
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality, norm_freq_str
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
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from pandas.tseries.frequencies import to_offset


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15


PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}


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


set_seed(1)


def maybe_reconvert_freq(freq: str) -> str:
    """if the freq is one of the newest pandas freqs, convert it to the old freq"""
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
    if freq in deprecated_map:
        return deprecated_map[freq]
    return freq


def compute_prediction_length(freq: str, term: Term = Term.SHORT) -> int:
    """
    根据频率和 term 计算 prediction length（非 M4 数据集）
    
    Args:
        freq: 数据频率（如 'H', 'D', 'M' 等）
        term: short/medium/long，决定 multiplier
    
    Returns:
        prediction length
    """
    freq_normalized = norm_freq_str(to_offset(freq).name)
    freq_normalized = maybe_reconvert_freq(freq_normalized)
    base_pred_len = PRED_LENGTH_MAP.get(freq_normalized, 48)
    return term.multiplier * base_pred_len


def load_dataset_properties(data_path: str) -> dict:
    """加载数据集属性配置文件"""
    config_path = Path(data_path) / "dataset_properties.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def get_available_datasets(data_path: str, term_filter: Optional[str] = None) -> List[str]:
    """
    获取可用的数据集列表
    
    Args:
        data_path: 数据集根目录
        term_filter: 如果需要过滤，可以是 'short' 或 'med_long'
    
    Returns:
        数据集名称列表
    """
    props = load_dataset_properties(data_path)
    
    if term_filter is None:
        return list(props.keys())
    
    filtered = []
    for ds_name, ds_props in props.items():
        ds_term = ds_props.get("term", "med_long")
        if term_filter == "short" and ds_term == "short":
            filtered.append(ds_name)
        elif term_filter == "med_long" and ds_term != "short":
            filtered.append(ds_name)
    
    return filtered


class SundialPredictor:
    def __init__(
        self,
        num_samples: int,
        prediction_length: int,
        device_map,
        batch_size: int = 1024,
    ):
        self.device = device_map
        self.model = AutoModelForCausalLM.from_pretrained(
            "thuml/sundial-base-128m", 
            trust_remote_code=True,
            local_files_only=True
        )
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.batch_size = batch_size

    def left_pad_and_stack_1D(self, tensors):
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 1
            padding = torch.full(
                size=(max_len - len(c),), fill_value=torch.nan, device=c.device
            )
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def prepare_and_validate_context(self, context):
        if isinstance(context, list):
            context = self.left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    def predict(
        self,
        test_data_input,
        batch_x_shape: int = 2880,
    ):
        forecast_outputs = []
        for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size)):
            context = [torch.tensor(entry["target"]) for entry in batch]
            batch_x = self.prepare_and_validate_context(context)
            if batch_x.shape[-1] > batch_x_shape:
                batch_x = batch_x[..., -batch_x_shape:]
            if torch.isnan(batch_x).any():
                batch_x = np.array(batch_x)
                imputed_rows = []
                for i in range(batch_x.shape[0]):
                    row = batch_x[i]
                    imputed_row = LastValueImputation()(row)
                    imputed_rows.append(imputed_row)
                batch_x = np.vstack(imputed_rows)
                batch_x = torch.tensor(batch_x)
            batch_x = batch_x.to(self.device)
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        batch_x,
                        max_new_tokens=self.prediction_length,
                        revin=True,
                        num_samples=self.num_samples,
                    )
            else:
                outputs = self.model.generate(
                    batch_x,
                    max_new_tokens=self.prediction_length,
                    revin=True,
                    num_samples=self.num_samples,
                )
            forecast_outputs.append(outputs.detach().cpu().numpy())
        forecast_outputs = np.concatenate(forecast_outputs)

        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        return forecasts


def load_dataset(
    data_path: str,
    dataset_name: str,
    term: str = "short",
    prediction_length: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
):
    """
    加载数据集
    
    Args:
        data_path: 数据集根目录
        dataset_name: 数据集名称
        term: short/medium/long
        prediction_length: 预测长度，如果不指定则自动计算
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        train_dataset, test_data, freq, prediction_length, dataset_props
    """
    props = load_dataset_properties(data_path)
    if dataset_name not in props:
        raise ValueError(f"Dataset {dataset_name} not found in properties")
    
    ds_props = props[dataset_name]
    freq = ds_props["frequency"]
    
    csv_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if 'date' in df.columns or 'time' in df.columns or 'timestamp' in df.columns:
        time_col = next(c for c in ['date', 'time', 'timestamp'] if c in df.columns)
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size : train_size + val_size]
    test_df = df.iloc[train_size + val_size :]

    print(f"Dataset: {dataset_name}")
    print(f"  Domain: {ds_props.get('domain', 'Unknown')}")
    print(f"  Variates: {ds_props.get('num_variates', 'Unknown')}")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    term_enum = Term(term)
    
    if prediction_length is None:
        prediction_length = compute_prediction_length(freq, term_enum)
    
    print(f"  Frequency: {freq}, Term: {term}, Prediction Length: {prediction_length}")

    train_data = []
    for i in range(len(train_df.columns)):
        col_name = train_df.columns[i]
        entry = {
            "target": train_df.iloc[:, i].values.astype(np.float32),
            "start": pd.Period(train_df.index[0], freq=freq.lower()),
            "item_id": f"{dataset_name}_{col_name}",
        }
        train_data.append(entry)

    test_data = []
    for i in range(len(test_df.columns)):
        col_name = test_df.columns[i]
        entry = {
            "target": test_df.iloc[:, i].values.astype(np.float32),
            "start": pd.Period(test_df.index[0], freq=freq.lower()),
            "item_id": f"{dataset_name}_{col_name}",
        }
        test_data.append(entry)

    train_dataset = ListDataset(train_data, freq=freq.lower())
    test_dataset = ListDataset(test_data, freq=freq.lower())

    _, test_template = split(
        test_dataset, offset=-prediction_length
    )
    test_data_instances = test_template.generate_instances(
        prediction_length=prediction_length,
        windows=1,
        distance=prediction_length,
    )

    return train_dataset, test_data_instances, freq, prediction_length, ds_props


def run_evaluation(
    dataset_name: str,
    term: str,
    data_path: str,
    output_dir: str,
    prediction_length: Optional[int] = None,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
):
    """运行单个数据集的评估"""
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset_name} ({term})")
    print(f"{'='*80}")
    
    train_data, test_data, freq, pred_len, ds_props = load_dataset(
        data_path=data_path,
        dataset_name=dataset_name,
        term=term,
        prediction_length=prediction_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    print(f"\nInitializing Sundial model...")
    print(f"  Model: thuml/sundial-base-128m")
    print(f"  Prediction length: {pred_len}")
    print(f"  Device: {device}")
    print(f"  Num samples: {num_samples}")
    print(f"  Batch size: {batch_size}")

    predictor = SundialPredictor(
        num_samples=num_samples,
        prediction_length=pred_len,
        device_map=device,
        batch_size=batch_size,
    )

    season_length = get_seasonality(freq)
    print(f"\nSeason length: {season_length}")

    metrics = [
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

    print(f"\nEvaluating model...")
    res = evaluate_model(
        predictor,
        test_data=test_data,
        metrics=metrics,
        batch_size=batch_size,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file_path = output_path / f"{dataset_name}_{term}_results.csv"
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "metric",
                "value",
            ]
        )
        writer.writerow(["dataset", dataset_name])
        writer.writerow(["freq", freq])
        writer.writerow(["term", term])
        writer.writerow(["prediction_length", pred_len])
        writer.writerow(["MSE[mean]", res["MSE[mean]"].item()])
        writer.writerow(["MSE[0.5]", res["MSE[0.5]"].item()])
        writer.writerow(["MAE[0.5]", res["MAE[0.5]"].item()])
        writer.writerow(["MASE[0.5]", res["MASE[0.5]"].item()])
        writer.writerow(["MAPE[0.5]", res["MAPE[0.5]"].item()])
        writer.writerow(["sMAPE[0.5]", res["sMAPE[0.5]"].item()])
        writer.writerow(["MSIS", res["MSIS"].item()])
        writer.writerow(["RMSE[mean]", res["RMSE[mean]"].item()])
        writer.writerow(["NRMSE[mean]", res["NRMSE[mean]"].item()])
        writer.writerow(["ND[0.5]", res["ND[0.5]"].item()])
        writer.writerow(
            ["mean_weighted_sum_quantile_loss", res["mean_weighted_sum_quantile_loss"].item()]
        )

    print(f"\nResults saved to: {csv_file_path}")
    print("\nResults summary:")
    print(f"  MSE[mean]: {res['MSE[mean]'].item():.6f}")
    print(f"  MSE[0.5]: {res['MSE[0.5]'].item():.6f}")
    print(f"  MAE[0.5]: {res['MAE[0.5]'].item():.6f}")
    print(f"  MASE[0.5]: {res['MASE[0.5]'].item():.6f}")
    print(f"  MAPE[0.5]: {res['MAPE[0.5]'].item():.6f}")
    print(f"  sMAPE[0.5]: {res['sMAPE[0.5]'].item():.6f}")
    print(f"  MSIS: {res['MSIS'].item():.6f}")
    print(f"  RMSE[mean]: {res['RMSE[mean]'].item():.6f}")
    print(f"  NRMSE[mean]: {res['NRMSE[mean]'].item():.6f}")
    print(f"  ND[0.5]: {res['ND[0.5]'].item():.6f}")
    print(
        f"  mean_weighted_sum_quantile_loss: {res['mean_weighted_sum_quantile_loss'].item():.6f}"
    )
    
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Run Sundial on TSF-Imputation-Analysis datasets"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets",
        help="Path to the datasets directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name to run (if not specified, will run all available datasets)",
    )
    parser.add_argument(
        "--term",
        type=str,
        default="auto",
        choices=["auto", "short", "medium", "long"],
        help="Forecast horizon term (auto=automatically determine based on dataset properties)",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=None,
        help="Prediction length (if not specified, will be computed automatically based on freq and term)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/sundial",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples for generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model (cpu or cuda:X)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--list_datasets",
        action="store_true",
        help="List all available datasets and exit",
    )

    args = parser.parse_args()

    if args.list_datasets:
        props = load_dataset_properties(args.data_path)
        print("\nAvailable datasets:")
        print(f"{'Dataset':<25} {'Freq':<6} {'Term':<12} {'Domain':<15} {'Variates'}")
        print("-" * 80)
        for ds_name, ds_props in sorted(props.items()):
            freq = ds_props.get('frequency', 'N/A')
            term = ds_props.get('term', 'med_long')
            domain = ds_props.get('domain', 'N/A')
            variates = ds_props.get('num_variates', 'N/A')
            print(f"{ds_name:<25} {freq:<6} {term:<12} {domain:<15} {variates}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = get_available_datasets(args.data_path)
        print(f"Found {len(datasets_to_run)} datasets to evaluate")

    for dataset_name in datasets_to_run:
        props = load_dataset_properties(args.data_path)
        ds_props = props[dataset_name]
        ds_term_type = ds_props.get("term", "med_long")
        
        if args.term == "auto":
            if ds_term_type == "short":
                terms_to_run = ["short"]
            else:
                terms_to_run = ["short", "medium", "long"]
        else:
            if ds_term_type == "short" and args.term in ["medium", "long"]:
                print(f"\n⚠️  Skipping {dataset_name} with term={args.term} (dataset only supports short term)")
                continue
            terms_to_run = [args.term]
        
        for term in terms_to_run:
            try:
                run_evaluation(
                    dataset_name=dataset_name,
                    term=term,
                    data_path=args.data_path,
                    output_dir=args.output_dir,
                    prediction_length=args.prediction_length,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                )
            except Exception as e:
                print(f"\n❌ Error evaluating {dataset_name} ({term}): {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
