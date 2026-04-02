"""
带缺失值填补的 Sundial 模型评估脚本

参考新的 eval_sundial.py 重构，使用交叉评估方式：
- 在填补后的数据上生成预测
- 与干净数据集对应区间对比计算指标
"""

import os
import sys
import csv
import json
import argparse
import math
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE,
    MeanWeightedSumQuantileLoss,
)
from gluonts.dataset.common import ListDataset
from gluonts.dataset.split import split
from pandas.tseries.frequencies import to_offset

from Imputation import WindowImputationProcessor


# 常量定义（与 eval_sundial.py 保持一致）
TEST_SPLIT = 0.1  # 测试集比例（与 eval_sundial.py 一致）
MAX_WINDOW = 20   # 最大窗口数


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        return {Term.SHORT: 1, Term.MEDIUM: 10, Term.LONG: 15}[self]


PRED_LENGTH_MAP = {
    "M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60,
}


def maybe_reconvert_freq(freq: str) -> str:
    """转换 pandas 新频率为旧频率"""
    deprecated_map = {
        "Y": "A", "YE": "A", "QE": "Q", "ME": "M",
        "h": "H", "min": "T", "s": "S", "us": "U",
    }
    return deprecated_map.get(freq, freq)


def compute_prediction_length(freq: str, term: Term = Term.SHORT) -> int:
    """根据频率和 term 计算 prediction length"""
    freq_normalized = norm_freq_str(to_offset(freq).name)
    freq_normalized = maybe_reconvert_freq(freq_normalized)
    base_pred_len = PRED_LENGTH_MAP.get(freq_normalized, 48)
    return term.multiplier * base_pred_len


def load_dataset_properties(data_path: str) -> dict:
    """加载数据集属性"""
    config_path = Path(data_path) / "dataset_properties.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def load_clean_dataset(clean_data_path: str, freq: str) -> ListDataset:
    """
    加载干净数据集
    
    Args:
        clean_data_path: 干净数据集 CSV 路径
        freq: 数据频率
    
    Returns:
        ListDataset 格式的干净数据集
    """
    clean_df = pd.read_csv(clean_data_path)
    
    # 识别时间列
    time_col = None
    for c in ['date', 'time', 'timestamp']:
        if c in clean_df.columns:
            time_col = c
            break
    
    if time_col:
        clean_df[time_col] = pd.to_datetime(clean_df[time_col])
        clean_df = clean_df.set_index(time_col)
    
    clean_test_data = []
    for i in range(len(clean_df.columns)):
        col_name = clean_df.columns[i]
        clean_entry = {
            "target": clean_df.iloc[:, i].values.astype(np.float32),
            "start": pd.Period(clean_df.index[0], freq=freq.lower()),
            "item_id": f"clean_{col_name}",
        }
        clean_test_data.append(clean_entry)
    
    return ListDataset(clean_test_data, freq=freq.lower())


class SundialPredictor:
    """Sundial 预测器（与 eval_sundial.py 保持一致）"""
    
    def __init__(
        self,
        num_samples: int,
        prediction_length: int,
        device_map,
        batch_size: int = 1024,  # 与 eval_sundial.py 一致
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

    def predict(self, test_data_input, batch_x_shape: int = 2880):
        """生成预测"""
        forecast_outputs = []
        input_metadata = []
        
        for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size), desc="Generating forecasts"):
            context = [torch.tensor(entry[0]["target"]) for entry in batch]
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
            
            for entry in batch:
                input_entry = entry[0]
                input_metadata.append({
                    "start": input_entry["start"],
                    "target_length": len(input_entry["target"]),
                    "item_id": input_entry.get("item_id", None),
                })
        
        forecast_outputs = np.concatenate(forecast_outputs)

        forecasts = []
        for item, meta in zip(forecast_outputs, input_metadata):
            forecast_start_date = meta["start"] + meta["target_length"]
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )

        return forecasts


def run_evaluation_with_imputation(
    dataset_name: str,
    term: str,
    missing_pattern: str,
    missing_ratio: float,
    imputation_method: str,
    data_path: str,
    clean_data_path: Optional[str] = None,
    output_dir: str = "results/sundial/imputation",
    prediction_length: Optional[int] = None,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 42,
    force_regenerate: bool = False,
):
    """
    运行带缺失值填补的评估（参考 eval_sundial.py 的交叉评估方式）
    
    流程：
    1. 使用 WindowImputationProcessor 生成填补后的窗口数据
    2. 在填补后的数据上生成预测
    3. 与干净数据集对应区间对比计算指标
    
    Args:
        dataset_name: 数据集名称
        term: short/medium/long
        missing_pattern: 缺失模式（MCAR, BM, TM, TVMR）
        missing_ratio: 缺失比例
        imputation_method: 填补方法
        data_path: 数据集路径
        clean_data_path: 干净数据集路径（可选，默认从 data_path/ori 查找）
        output_dir: 结果输出目录
        prediction_length: 预测长度
        num_samples: 采样数
        batch_size: 批次大小
        device: 设备
        seed: 随机种子
        force_regenerate: 是否强制重新生成窗口数据
    """
    
    # 设置随机种子
    set_seed(1)
    
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset_name} ({term})")
    print(f"Missing: {missing_pattern} {missing_ratio:.0%}")
    print(f"Imputation: {imputation_method}")
    print(f"{'='*80}")
    
    # 加载数据集属性
    props = load_dataset_properties(data_path)
    ds_props = props[dataset_name]
    freq = ds_props["frequency"]
    
    print(f"\nData path: {data_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Frequency: {freq}")
    
    # 计算 prediction_length
    if prediction_length is None:
        prediction_length = compute_prediction_length(freq, Term(term))
    
    # 查找干净数据集
    if clean_data_path is None:
        clean_data_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
        if not clean_data_path.exists():
            raise FileNotFoundError(f"Clean dataset not found: {clean_data_path}")
        clean_data_path = str(clean_data_path)
    
    print(f"Clean data path: {clean_data_path}")
    
    # 创建填补处理器
    print(f"\nCreating WindowImputationProcessor...")
    processor = WindowImputationProcessor(
        dataset_name=dataset_name,
        term=term,
        missing_pattern=missing_pattern,
        missing_ratio=missing_ratio,
        imputation_method=imputation_method,
        data_path=data_path,
        seed=seed,
    )
    
    # 获取窗口信息（使用 processor 计算的窗口数）
    print(f"Getting windows info...")
    windows_info = processor.get_all_windows_info()
    n_windows = len(windows_info)
    
    print(f"\nTotal windows: {n_windows}")
    print(f"Prediction length: {prediction_length}")
    
    # 加载干净数据集
    print(f"\nLoading clean dataset...")
    clean_dataset = load_clean_dataset(clean_data_path, freq)
    
    # 初始化 Sundial 模型
    print(f"\nInitializing Sundial model...")
    print(f"  - Device: {device}")
    print(f"  - Prediction length: {prediction_length}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - This may take a while (loading model weights)...")
    
    import time
    start_time = time.time()
    predictor = SundialPredictor(
        num_samples=num_samples,
        prediction_length=prediction_length,
        device_map=device,
        batch_size=batch_size,
    )
    load_time = time.time() - start_time
    print(f"[OK] Model loaded in {load_time:.2f}s")
    
    # 收集所有窗口的预测结果
    all_forecasts = []
    all_forecasts_by_ts = []  # 按时间序列分组的预测
    
    # 注意：本脚本使用固定历史长度（2880 点）进行窗口评估
    # 与 eval_sundial.py 的变长历史不同，这会导致 MSE 结果差异
    # eval_sundial.py MSE ≈ 6.27, 本脚本 MSE ≈ 6.44
    # 
    # 设计理念：
    # 1. 避免保存冗余数据（预测部分没有缺失值，不需要填补）
    # 2. 支持 prediction_length 变化时复用历史数据
    # 3. 实验控制更严格（所有窗口使用相同的历史长度）
    #
    # 因此，本脚本专注于验证缺失值对模型预测的影响（相对比），
    # 而非与 eval_sundial.py 的绝对结果对比。
    
    # 注意：window_processor 的窗口索引是从后往前（窗口 0 是数据集末尾）
    # 但 eval_sundial.py 的窗口索引是从前往后（窗口 0 是 split 点之后）
    # 因此需要反转窗口索引以匹配 eval_sundial.py
    for i in range(n_windows):
        window_idx = n_windows - 1 - i  # 反转索引
        
        print(f"\n{'-'*60}")
        print(f"Window {i}/{n_windows} (actual window index: {window_idx})")
        print(f"{'-'*60}")
        
        # 处理窗口数据（注入缺失 + 填补）
        print(f"-> Processing window data (injection + imputation)...")
        df_imputed, meta = processor.process_window(
            window_idx,
            force_regenerate=force_regenerate,
        )
        print(f"[OK] Window data ready: shape={df_imputed.shape}")
        
        # 构建测试数据集（填补后的数据，只包含历史部分）
        test_data = []
        for col in df_imputed.columns:
            if col in ['date', 'time', 'timestamp']:
                continue
            
            date_col = df_imputed['date'].iloc[0] if 'date' in df_imputed.columns else df_imputed.index[0]
            
            entry = {
                "target": df_imputed[col].values.astype(np.float32),
                "start": pd.Period(date_col, freq=freq.lower()),
                "item_id": f"{dataset_name}_{col}",
            }
            test_data.append(entry)
        
        # 创建 ListDataset 用于生成测试实例
        test_dataset = ListDataset(test_data, freq=freq.lower())
        
        # 使用 split 和 generate_instances 创建测试数据实例
        # 注意：只生成当前窗口的实例（windows=1），而不是所有窗口
        # 这样可以避免不必要的计算，每个窗口只需要 7 个预测
        _, test_template = split(test_dataset, offset=-prediction_length)
        test_data_instances = test_template.generate_instances(
            prediction_length=prediction_length,
            windows=1,  # 只生成当前窗口的实例
            distance=prediction_length,
        )
        
        # 生成预测
        print(f"-> Generating forecasts...")
        # 使用与 eval_sundial.py 一致的 batch_x_shape=2880
        # Sundial 模型的最大上下文长度是 2880
        forecasts = predictor.predict(test_data_instances, batch_x_shape=2880)
        
        # 按时间序列分组存储预测
        if i == 0:  # 使用循环索引 i 而不是 window_idx
            # 初始化时间序列分组
            all_forecasts_by_ts = [[] for _ in range(len(forecasts))]
        
        for j, forecast in enumerate(forecasts):
            all_forecasts_by_ts[j].append(forecast)
        
        # 显示当前窗口的 MSE
        if len(forecasts) > 0:
            # 简单计算当前窗口的 MSE（仅用于显示）
            forecast = forecasts[0]
            pred_mean = forecast.mean
            # 从干净数据集中提取对应窗口的预测部分
            # 注意：eval_sundial.py 的窗口是从 split 点开始，不是从数据集末尾开始
            # split_point = dataset_length - prediction_length * n_windows
            # 窗口 i 的预测区间：[split_point + i*pred_len : split_point + (i+1)*pred_len]
            clean_entry = clean_dataset[0]
            clean_target = clean_entry["target"]
            split_point = len(clean_target) - prediction_length * n_windows
            forecast_start = split_point + i * prediction_length
            forecast_end = forecast_start + prediction_length
            clean_values = clean_target[forecast_start:forecast_end]
            mse = np.mean((pred_mean - clean_values) ** 2)
            print(f"MSE[mean] (window {i}): {mse:.6f}")
    
    # 重新组织预测顺序：先按时间序列，再按窗口
    # 这样与 clean_data_instances 的顺序一致
    for ts_forecasts in all_forecasts_by_ts:
        all_forecasts.extend(ts_forecasts)
    
    print(f"\n{'='*80}")
    print(f"Evaluating against clean data...")
    print(f"{'='*80}")
    
    # 创建简单预测器包装器
    class SimplePredictor:
        def __init__(self, forecasts):
            self.forecasts = forecasts
        
        def predict(self, test_data):
            return self.forecasts
    
    simple_predictor = SimplePredictor(all_forecasts)
    
    # 准备干净数据集的窗口实例
    # 需要从干净数据集中提取与每个窗口对应的预测部分
    # eval_sundial.py 的窗口是从 split 点开始，不是从数据集末尾开始
    # split 点 = dataset_length - prediction_length * windows
    print(f"\nPreparing clean data instances for {n_windows} windows...")
    
    # 计算 split 点（与 eval_sundial.py 一致）
    split_point = len(clean_dataset[0]["target"]) - prediction_length * n_windows
    
    # 构建干净数据集的窗口实例（与 all_forecasts 的顺序一致）
    # 顺序：先按时间序列，再按窗口
    # 注意：窗口 i 对应 eval_sundial.py 的窗口 i，预测区间从 split_point 开始
    clean_data_list = []
    for ts_idx in range(len(clean_dataset)):
        clean_entry = clean_dataset[ts_idx]
        clean_target = clean_entry["target"]
        
        for window_idx in range(n_windows):
            # 计算当前窗口在干净数据集中的预测区间（从 split 点开始）
            # 与 eval_sundial.py 一致：窗口 i 的预测区间 = [split_point + i*pred_len, split_point + (i+1)*pred_len)
            forecast_start = split_point + window_idx * prediction_length
            forecast_end = forecast_start + prediction_length
            
            # 提取预测部分作为标签
            label = clean_target[forecast_start:forecast_end]
            
            # 创建一个伪 input（实际评估时不会用到，但需要符合 TestData 格式）
            dummy_input = np.full(prediction_length, np.nan, dtype=np.float32)
            
            instance = {
                "target": dummy_input,
                "start": pd.Period(clean_entry["start"] + forecast_start, freq=freq.lower()),
                "item_id": f"{clean_entry['item_id']}_window_{window_idx}",
            }
            clean_data_list.append((instance, {"target": label}))
    
    # 将干净数据转换为 ListDataset 格式，然后使用 split 生成 TestData
    # 因为 evaluate_model 需要 TestData 格式
    clean_data_for_eval = ListDataset(
        [{"target": item[0]["target"], "start": item[0]["start"], "item_id": item[0]["item_id"]} for item in clean_data_list],
        freq=freq.lower()
    )
    
    # 使用 split 创建 TestData
    # 我们需要创建一个特殊的 splitter 来提取 label
    # 但更简单的方法是直接使用 generate_instances
    _, clean_template = split(clean_data_for_eval, offset=0)
    clean_data_instances = clean_template.generate_instances(
        prediction_length=prediction_length,
        windows=1,
        distance=prediction_length,
    )
    
    # 现在需要替换 label 为我们从干净数据集中提取的真实标签
    # 但 evaluate_model 会使用 input 生成预测，然后与 label 比较
    # 我们的 input 是 dummy，所以需要特殊处理
    
    # 更简单的方法：手动计算指标，不使用 evaluate_model
    # 因为我们已经生成了所有预测，只需要与干净数据的标签比较
    print(f"[OK] Prepared {len(clean_data_list)} clean data instances")
    
    # 手动计算指标
    from gluonts.ev.metrics import MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND, MeanWeightedSumQuantileLoss
    from gluonts.time_feature import get_seasonality
    
    season_length = get_seasonality(freq)
    
    # 收集所有预测和标签
    all_predictions = []
    all_labels = []
    
    for forecast_idx, forecast in enumerate(all_forecasts):
        pred_mean = forecast.mean
        # 从 clean_data_list 中提取对应的标签
        label = clean_data_list[forecast_idx][1]["target"]
        all_predictions.append(pred_mean)
        all_labels.append(label)
    
    # 转换为 numpy 数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # 计算指标
    mse_mean = np.mean((all_predictions - all_labels) ** 2, axis=1)
    mae = np.mean(np.abs(all_predictions - all_labels), axis=1)
    rmse = np.sqrt(mse_mean)
    
    # 计算平均指标
    results = {
        "dataset_name": dataset_name,
        "term": term,
        "freq": freq,
        "missing_pattern": missing_pattern,
        "missing_ratio": missing_ratio,
        "imputation_method": imputation_method,
        "prediction_length": prediction_length,
        "n_windows": n_windows,
        "seed": seed,
        "MSE[mean]": float(np.mean(mse_mean)),
        "RMSE[mean]": float(np.mean(rmse)),
        "MAE[0.5]": float(np.mean(mae)),
    }
    
    print(f"\nResults summary:")
    print(f"  MSE[mean]: {results['MSE[mean]']:.6f}")
    print(f"  RMSE[mean]: {results['RMSE[mean]']:.6f}")
    print(f"  MAE[0.5]: {results['MAE[0.5]']:.6f}")
    
    # 跳过 evaluate_model 部分
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ratio_str = f"{int(missing_ratio * 100):03d}"
    csv_file_path = output_path / f"{dataset_name}_{term}_{missing_pattern}_{ratio_str}_{imputation_method}_results.csv"
    
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in results.items():
            writer.writerow([key, value])
    
    print(f"\nResults saved to: {csv_file_path}")
    
    return results
    
    # 计算指标
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
    
    # 计算平均指标
    results = {
        "dataset_name": dataset_name,
        "term": term,
        "freq": freq,
        "missing_pattern": missing_pattern,
        "missing_ratio": missing_ratio,
        "imputation_method": imputation_method,
        "prediction_length": prediction_length,
        "n_windows": n_windows,
        "seed": seed,
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
    
    # 打印结果
    print(f"\nResults summary:")
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
    
    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    ratio_str = f"{int(missing_ratio * 100):03d}"
    csv_file_path = output_path / f"{dataset_name}_{term}_{missing_pattern}_{ratio_str}_{imputation_method}_results.csv"
    
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in results.items():
            writer.writerow([key, value])
    
    print(f"\nResults saved to: {csv_file_path}")
    
    # 保存汇总 JSON
    summary_path = output_path / f"{dataset_name}_{term}_{missing_pattern}_{ratio_str}_{imputation_method}_summary.json"
    summary_data = {
        "dataset_name": dataset_name,
        "term": term,
        "missing_pattern": missing_pattern,
        "missing_ratio": missing_ratio,
        "imputation_method": imputation_method,
        "n_windows": n_windows,
        "prediction_length": prediction_length,
        "seed": seed,
        "metrics": {k: v for k, v in results.items() if k not in ["dataset_name", "term", "freq", "missing_pattern", "missing_ratio", "imputation_method", "prediction_length", "n_windows", "seed"]},
        "generated_at": datetime.now().isoformat(),
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Sundial with imputation methods (cross-evaluation)"
    )
    
    parser.add_argument("--data_path", type=str, default="datasets")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--term", type=str, default="auto", 
                       choices=["auto", "short", "medium", "long"])
    parser.add_argument("--missing_pattern", type=str, default="MCAR")
    parser.add_argument("--missing_ratio", type=float, default=0.05)
    parser.add_argument("--imputation_method", type=str, default="linear",
                       choices=["zero", "mean", "forward", "backward",
                               "linear", "nearest", "spline", "seasonal", "none"])
    parser.add_argument("--output_dir", type=str, default="results/sundial/imputation")
    parser.add_argument("--prediction_length", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument("--list_datasets", action="store_true")
    
    args = parser.parse_args()
    
    if args.list_datasets:
        props = load_dataset_properties(args.data_path)
        print("\nAvailable datasets:")
        print(f"{'Dataset':<25} {'Freq':<6} {'Term':<12} {'Domain':<15} {'Variates'}")
        print("-" * 80)
        for ds_name, ds_props in sorted(props.items()):
            print(f"{ds_name:<25} {ds_props.get('frequency'):<6} "
                  f"{ds_props.get('term'):<12} {ds_props.get('domain'):<15} "
                  f"{ds_props.get('num_variates')}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.dataset:
        datasets_to_run = [args.dataset]
    else:
        datasets_to_run = list(load_dataset_properties(args.data_path).keys())
    
    for dataset_name in datasets_to_run:
        props = load_dataset_properties(args.data_path)
        ds_term_type = props[dataset_name].get("term", "med_long")
        
        if args.term == "auto":
            terms_to_run = ["short"] if ds_term_type == "short" else ["short", "medium", "long"]
        else:
            if ds_term_type == "short" and args.term in ["medium", "long"]:
                print(f"⚠️  Skipping {dataset_name} with term={args.term}")
                continue
            terms_to_run = [args.term]
        
        for term in terms_to_run:
            try:
                run_evaluation_with_imputation(
                    dataset_name=dataset_name,
                    term=term,
                    missing_pattern=args.missing_pattern,
                    missing_ratio=args.missing_ratio,
                    imputation_method=args.imputation_method,
                    data_path=args.data_path,
                    output_dir=args.output_dir,
                    prediction_length=args.prediction_length,
                    num_samples=args.num_samples,
                    batch_size=args.batch_size,
                    device=args.device,
                    seed=args.seed,
                    force_regenerate=args.force_regenerate,
                )
            except Exception as e:
                print(f"\n[ERROR] Error: {dataset_name} ({term}): {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
