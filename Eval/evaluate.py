"""
评估模块 - Sundial 预测评估

职责：读取窗口数据 -> 使用 Sundial 模型预测 -> 与干净数据对比 -> 保存结果

使用示例：
    python Eval/evaluate.py --data_dir datasets/window_data/ETTh1/short/MCAR/000/missing --clean_data_path datasets/ori/ETTh1.csv --device cpu
"""

import os
import sys
import json
import csv
import argparse
import logging
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import set_seed
from tqdm.auto import tqdm
from gluonts.transform import LastValueImputation
from gluonts.model.forecast import SampleForecast
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split


MAX_CONTEXT_SUNDIAL = 2880
TEST_SPLIT = 0.1
MAX_WINDOW = 20


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


def load_window_data(data_dir: str) -> Tuple[List[pd.DataFrame], Dict]:
    """
    加载窗口数据和元数据
    
    Args:
        data_dir: 数据目录
        
    Returns:
        (dataframes, meta)
    """
    data_path = Path(data_dir)
    
    meta_path = data_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {data_dir}")
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    n_windows = meta["n_windows"]
    dataframes = []
    
    for i in range(n_windows):
        window_file = data_path / f"window_{i:03d}.csv"
        if not window_file.exists():
            raise FileNotFoundError(f"Window file not found: {window_file}")
        
        df = pd.read_csv(window_file)
        
        time_col = meta.get("time_col")
        if time_col and time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        
        dataframes.append(df)
    
    return dataframes, meta


def load_clean_data(clean_data_path: str, time_col: Optional[str] = None) -> pd.DataFrame:
    """
    加载干净数据
    
    Args:
        clean_data_path: 干净数据路径
        time_col: 时间列名
        
    Returns:
        干净数据 DataFrame
    """
    df = pd.read_csv(clean_data_path)
    
    if time_col is None:
        for c in ['date', 'time', 'timestamp']:
            if c in df.columns:
                time_col = c
                break
    
    if time_col and time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    
    return df


class SundialPredictor:
    """Sundial 预测器"""
    
    def __init__(
        self,
        num_samples: int,
        prediction_length: int,
        device_map,
        batch_size: int = 1024,
    ):
        self.device = device_map
        from transformers import AutoModelForCausalLM
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

    def predict_batch(self, contexts: List[np.ndarray], batch_x_shape: int = 2880) -> List[np.ndarray]:
        """
        直接预测一批 context
        
        Args:
            contexts: context 列表，每个是 numpy array
            batch_x_shape: 最大 context 长度
            
        Returns:
            预测结果列表，每个是 numpy array (num_samples, prediction_length)
        """
        batch_x_list = []
        for ctx in contexts:
            tensor = torch.tensor(ctx)
            if len(tensor) > batch_x_shape:
                tensor = tensor[-batch_x_shape:]
            batch_x_list.append(tensor)
        
        batch_x = self.prepare_and_validate_context(batch_x_list)
        
        if torch.isnan(batch_x).any():
            batch_x_np = np.array(batch_x)
            imputed_rows = []
            for i in range(batch_x_np.shape[0]):
                row = batch_x_np[i]
                imputed_row = LastValueImputation()(row)
                imputed_rows.append(imputed_row)
            batch_x = torch.tensor(np.vstack(imputed_rows))
        
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
        
        return outputs.detach().cpu().numpy()


def evaluate_forecast(
    data_dir: str,
    clean_data_path: str,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> Dict:
    """
    预测评估：在窗口数据上进行预测，与干净数据对比
    
    Args:
        data_dir: 窗口数据目录
        clean_data_path: 干净数据路径
        num_samples: 采样数
        batch_size: 批次大小
        device: 设备
        output_dir: 结果输出目录
        
    Returns:
        评估结果字典
    """
    window_dfs, meta = load_window_data(data_dir)
    clean_df = load_clean_data(clean_data_path, meta.get("time_col"))
    
    data_cols = meta.get("data_cols", [])
    if not data_cols:
        data_cols = [col for col in window_dfs[0].columns if col not in ['date', 'time', 'timestamp']]
    
    prediction_length = meta["prediction_length"]
    n_windows = meta["n_windows"]
    n_windows = min(n_windows, MAX_WINDOW)
    freq = meta.get("frequency", "H")
    missing_ratio = meta.get("missing_ratio", 0.0)
    
    print(f"\n{'='*80}")
    print(f"预测评估（使用 Sundial 模型）")
    print(f"{'='*80}")
    print(f"数据集: {meta['dataset_name']}")
    print(f"Term: {meta['term']}")
    print(f"频率: {freq}")
    print(f"缺失比例: {missing_ratio}")
    print(f"填补方法: {meta.get('imputation_method', 'none')}")
    print(f"窗口数: {n_windows}")
    print(f"预测长度: {prediction_length}")
    print(f"设备: {device}")
    print(f"{'='*80}")
    
    print(f"\n初始化 Sundial 模型...")
    predictor = SundialPredictor(
        num_samples=num_samples,
        prediction_length=prediction_length,
        device_map=device,
        batch_size=batch_size,
    )
    
    n_ts = len(data_cols)
    
    all_contexts = []
    all_start_periods = []
    
    data_start_time = meta.get("data_start_time")
    if data_start_time:
        data_start_period = pd.Period(data_start_time, freq=freq.lower())
    else:
        data_start_period = pd.Period(clean_df.index[0], freq=freq.lower())
    
    dataset_length = meta.get("dataset_length", len(clean_df))
    split_point = dataset_length - prediction_length * n_windows
    
    for ts_idx in range(n_ts):
        col = data_cols[ts_idx]
        
        for win_idx in range(n_windows):
            df_window = window_dfs[win_idx]
            window_data = df_window[col].values.astype(np.float32)
            
            context_length = len(window_data) - prediction_length
            context = window_data[:context_length]
            
            context_end = split_point + win_idx * prediction_length
            
            all_contexts.append(context)
            all_start_periods.append(data_start_period + context_end)
    
    print(f"\n生成预测... (共 {len(all_contexts)} 个样本)")
    
    all_forecasts = []
    for i in tqdm(range(0, len(all_contexts), batch_size), desc="Generating forecasts"):
        batch_contexts = all_contexts[i:i + batch_size]
        batch_outputs = predictor.predict_batch(batch_contexts, batch_x_shape=MAX_CONTEXT_SUNDIAL)
        
        for j, outputs in enumerate(batch_outputs):
            start_period = all_start_periods[i + j]
            forecast = SampleForecast(samples=outputs, start_date=start_period)
            all_forecasts.append(forecast)
    
    print(f"\n{'='*80}")
    print(f"评估预测结果...")
    print(f"{'='*80}")
    
    class SimplePredictor:
        def __init__(self, forecasts):
            self.forecasts = forecasts
        
        def predict(self, test_data):
            return self.forecasts
    
    simple_predictor = SimplePredictor(all_forecasts)
    
    clean_test_data = []
    for col in data_cols:
        clean_entry = {
            "target": clean_df[col].values.astype(np.float32),
            "start": pd.Period(clean_df.index[0], freq=freq.lower()),
            "item_id": f"clean_{col}",
        }
        clean_test_data.append(clean_entry)
    
    clean_list_dataset = ListDataset(clean_test_data, freq=freq.lower())
    
    _, clean_test_template = split(
        clean_list_dataset, offset=-prediction_length * n_windows
    )
    
    clean_data_instances = clean_test_template.generate_instances(
        prediction_length=prediction_length,
        windows=n_windows,
        distance=prediction_length,
    )
    
    season_length = get_seasonality(freq)
    
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
    
    evaluator = evaluate_model(
        simple_predictor,
        test_data=clean_data_instances,
        metrics=metrics,
        batch_size=batch_size,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )
    
    results = {
        "dataset_name": meta["dataset_name"],
        "term": meta["term"],
        "missing_pattern": meta.get("missing_pattern", "N/A"),
        "missing_ratio": missing_ratio,
        "imputation_method": meta.get("imputation_method", "none"),
        "n_windows": n_windows,
        "prediction_length": prediction_length,
        "evaluated_at": datetime.now().isoformat(),
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
        "mean_weighted_sum_quantile_loss": float(evaluator["mean_weighted_sum_quantile_loss"].mean()),
    }
    
    print(f"\n结果汇总:")
    print(f"  MSE[mean]: {results['MSE[mean]']:.6f}")
    print(f"  MSE[0.5]: {results['MSE[0.5]']:.6f}")
    print(f"  MAE[0.5]: {results['MAE[0.5]']:.6f}")
    print(f"  MASE[0.5]: {results['MASE[0.5]']:.6f}")
    print(f"  MAPE[0.5]: {results['MAPE[0.5]']:.6f}")
    print(f"  sMAPE[0.5]: {results['sMAPE[0.5]']:.6f}")
    print(f"  MSIS: {results['MSIS']:.6f}")
    print(f"  RMSE[mean]: {results['RMSE[mean]']:.6f}")
    print(f"  NRMSE[mean]: {results['NRMSE[mean]']:.6f}")
    print(f"  ND[0.5]: {results['ND[0.5]']:.6f}")
    print(f"  mean_weighted_sum_quantile_loss: {results['mean_weighted_sum_quantile_loss']:.6f}")
    
    if output_dir is None:
        output_dir = Path("results") / meta["dataset_name"] / meta["term"] / meta.get("missing_pattern", "MCAR") / f"{int(missing_ratio * 100):03d}" / meta.get("imputation_method", "none")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file_path = output_dir / "forecast_results.csv"
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in results.items():
            writer.writerow([key, value])
    
    print(f"\n结果已保存到: {csv_file_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Sundial 预测评估模块")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="窗口数据目录路径")
    parser.add_argument("--clean_data_path", type=str, required=True,
                        help="干净数据路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="结果输出目录 (可选)")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="采样数 (默认: 100)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小 (默认: 32)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="设备 (默认: cpu)")
    
    args = parser.parse_args()
    
    results = evaluate_forecast(
        data_dir=args.data_dir,
        clean_data_path=args.clean_data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    print(f"\n评估完成！")


if __name__ == "__main__":
    main()
