"""
评估模块：在脏数据集上滑动窗口进行预测，与干净数据集对应区间对比计算指标
支持缺失值填补后再评估
"""
import csv
import logging
import math
from pathlib import Path
from enum import Enum
from typing import Optional, Dict, Any, List, Callable

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

# 导入填补方法
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from Imputation.imputation_methods import (
    zero_imputation,
    mean_imputation,
    forward_fill,
    backward_fill,
    linear_interpolation,
    nearest_interpolation,
    spline_interpolation,
    seasonal_decomposition_imputation,
    none_imputation,
)

TEST_SPLIT = 0.1
MAX_WINDOW = 20

# 填补方法映射
IMPUTATION_METHODS = {
    'zero': zero_imputation,
    'mean': mean_imputation,
    'forward': forward_fill,
    'backward': backward_fill,
    'linear': linear_interpolation,
    'nearest': nearest_interpolation,
    'spline': spline_interpolation,
    'seasonal': seasonal_decomposition_imputation,
    'none': none_imputation,
}

def get_imputation_method(method_name: str) -> Callable:
    """获取填补方法函数"""
    if method_name not in IMPUTATION_METHODS:
        raise ValueError(
            f"Unknown imputation method: {method_name}. "
            f"Available methods: {list(IMPUTATION_METHODS.keys())}"
        )
    return IMPUTATION_METHODS[method_name]

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
        """
        生成预测
        
        Args:
            test_data_input: TestData 实例，包含 (input, label) 元组
            batch_x_shape: 输入上下文的最大长度
            
        Returns:
            预测结果列表
        """
        forecast_outputs = []
        input_metadata = []
        for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size)):
            # batch 是元组列表 [(input, label), ...]，我们需要 input 部分
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
            
            # 从 input 部分提取元数据
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


def load_datasets_for_evaluation(
    eval_data_path: str,
    clean_data_path: str,
    freq: str,
    term: str = "short",
    prediction_length: Optional[int] = None,
    imputation_method: Optional[str] = None,
):
    """
    从路径加载两个数据集：评估数据集和干净数据集
    可选择对评估数据集进行缺失值填补
    
    Args:
        eval_data_path: 评估数据集的 CSV 文件路径
        clean_data_path: 干净数据集的 CSV 文件路径
        freq: 数据频率
        term: short/medium/long
        prediction_length: 预测长度，如果不指定则自动计算
        imputation_method: 填补方法名称，None 表示不填补
    
    Returns:
        eval_test_data, clean_test_data, prediction_length
    """
    if not Path(eval_data_path).exists():
        raise FileNotFoundError(f"Eval dataset file not found: {eval_data_path}")
    if not Path(clean_data_path).exists():
        raise FileNotFoundError(f"Clean dataset file not found: {clean_data_path}")
    
    eval_df = pd.read_csv(eval_data_path)
    clean_df = pd.read_csv(clean_data_path)
    
    time_col = None
    for c in ['date', 'time', 'timestamp']:
        if c in eval_df.columns:
            time_col = c
            break
    
    if time_col:
        eval_df[time_col] = pd.to_datetime(eval_df[time_col])
        eval_df = eval_df.set_index(time_col)
        clean_df[time_col] = pd.to_datetime(clean_df[time_col])
        clean_df = clean_df.set_index(time_col)
    
    if len(eval_df) != len(clean_df):
        raise ValueError(f"Eval and clean datasets must have the same length. "
                        f"Eval: {len(eval_df)}, Clean: {len(clean_df)}")
    
    # 如果指定了填补方法，对评估数据集进行填补
    if imputation_method and imputation_method.lower() != 'none':
        print(f"  Applying imputation method: {imputation_method}")
        imputation_func = get_imputation_method(imputation_method)
        data_cols = list(eval_df.columns)
        missing_before = eval_df.isna().sum().sum()
        eval_df = imputation_func(eval_df, data_cols)
        missing_after = eval_df.isna().sum().sum()
        print(f"    Missing values: {missing_before} -> {missing_after}")
    
    term_enum = Term(term)
    
    if prediction_length is None:
        prediction_length = compute_prediction_length(freq, term_enum)
    
    print(f"Loaded datasets:")
    print(f"  Eval: {eval_data_path}")
    print(f"  Clean: {clean_data_path}")
    print(f"  Samples: {len(eval_df)}")
    print(f"  Frequency: {freq}, Term: {term}, Prediction Length: {prediction_length}")

    eval_test_data = []
    clean_test_data = []
    
    for i in range(len(eval_df.columns)):
        col_name = eval_df.columns[i]
        eval_entry = {
            "target": eval_df.iloc[:, i].values.astype(np.float32),
            "start": pd.Period(eval_df.index[0], freq=freq.lower()),
            "item_id": f"eval_{col_name}",
        }
        clean_entry = {
            "target": clean_df.iloc[:, i].values.astype(np.float32),
            "start": pd.Period(clean_df.index[0], freq=freq.lower()),
            "item_id": f"clean_{col_name}",
        }
        eval_test_data.append(eval_entry)
        clean_test_data.append(clean_entry)

    eval_list_dataset = ListDataset(eval_test_data, freq=freq.lower())
    clean_list_dataset = ListDataset(clean_test_data, freq=freq.lower())

    # 计算滑动窗口数量（与 sundial.ipynb 一致）
    min_series_length = min(len(eval_df.iloc[:, i]) for i in range(len(eval_df.columns)))
    if "m4" in eval_data_path.lower():
        windows = 1
    else:
        w = math.ceil(TEST_SPLIT * min_series_length / prediction_length)  # TEST_SPLIT = 0.2
        windows = min(max(1, w), MAX_WINDOW)  # MAX_WINDOW = 20
    
    print(f"  Windows: {windows}")
    
    _, eval_test_template = split(
        eval_list_dataset, offset=-prediction_length * windows
    )
    eval_test_data_instances = eval_test_template.generate_instances(
        prediction_length=prediction_length,
        windows=windows,
        distance=prediction_length,
    )
    
    _, clean_test_template = split(
        clean_list_dataset, offset=-prediction_length * windows
    )
    clean_test_data_instances = clean_test_template.generate_instances(
        prediction_length=prediction_length,
        windows=windows,
        distance=prediction_length,
    )

    return eval_test_data_instances, clean_test_data_instances, prediction_length, windows


def evaluate_sundial(
    eval_data_path: str,
    clean_data_path: str,
    freq: str,
    term: str,
    prediction_length: Optional[int] = None,
    num_samples: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
    debug: bool = True,
    debug_samples: int = 5,
    imputation_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    核心评估函数：从评估数据集读取历史数据，预测后与干净数据集对比
    可选择在预测前对评估数据集进行缺失值填补
    
    Args:
        eval_data_path: 评估数据集的 CSV 文件路径
        clean_data_path: 干净数据集的 CSV 文件路径
        freq: 数据频率
        term: short/medium/long
        prediction_length: 预测长度
        num_samples: 采样数
        batch_size: 批次大小
        device: 设备
        debug: 是否输出调试表格
        debug_samples: 调试表格显示的样本数
        imputation_method: 填补方法名称，None 表示不填补
    
    Returns:
        包含所有评估指标的字典
    """
    
    print(f"\n{'='*80}")
    if imputation_method and imputation_method.lower() != 'none':
        print(f"Cross-dataset Evaluation with Imputation")
        print(f"  Imputation: {imputation_method}")
    else:
        print(f"Cross-dataset Evaluation")
    print(f"  Eval: {eval_data_path}")
    print(f"  Clean: {clean_data_path}")
    print(f"{'='*80}")
    
    eval_test_data, clean_test_data, pred_len, windows = load_datasets_for_evaluation(
        eval_data_path=eval_data_path,
        clean_data_path=clean_data_path,
        freq=freq,
        term=term,
        prediction_length=prediction_length,
        imputation_method=imputation_method,
    )

    print(f"\nInitializing Sundial model...")
    print(f"  Model: thuml/sundial-base-128m")
    print(f"  Prediction length: {pred_len}")
    print(f"  Windows: {windows}")
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

    print(f"\nGenerating forecasts on eval data...")
    forecasts = predictor.predict(eval_test_data)
    
    if debug:
        print(f"\n{'='*80}")
        print(f"Debug: Prediction Comparison (first {debug_samples} samples)")
        print(f"{'='*80}")
        print_debug_table(forecasts, eval_test_data, clean_test_data, pred_len, debug_samples)
    
    # 创建一个简单的 Predictor 包装器，用于 evaluate_model
    class SimplePredictor:
        def __init__(self, forecasts):
            self.forecasts = forecasts
        
        def predict(self, test_data):
            # 返回预先生成的 forecasts
            return self.forecasts
    
    simple_predictor = SimplePredictor(forecasts)
    
    print(f"\nEvaluating against clean data...")
    evaluator = evaluate_model(
        simple_predictor,
        test_data=clean_test_data,
        metrics=metrics,
        batch_size=batch_size,
        axis=None,
        mask_invalid_label=True,
        allow_nan_forecast=False,
        seasonality=season_length,
    )

    # 对多个窗口的指标进行平均（与 sundial.ipynb 一致）
    # evaluate_model 返回的是每个窗口的指标数组，需要取平均
    results = {
        "eval_data": eval_data_path,
        "clean_data": clean_data_path,
        "freq": freq,
        "term": term,
        "prediction_length": pred_len,
        "windows": windows,
        "imputation_method": imputation_method if imputation_method else "none",
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
    
    return results


def print_debug_table(forecasts, eval_test_data, clean_test_data, pred_len: int, num_samples: int = 5):
    """
    打印调试表格，展示预测值、eval 数据集值、干净数据集值的对比
    
    Args:
        forecasts: 预测结果列表
        eval_test_data: 评估数据集 (TestData 对象)
        clean_test_data: 干净数据集 (TestData 对象)
        pred_len: 预测长度
        num_samples: 显示的样本数
    """
    # TestData 对象不支持索引，需要先转换为列表
    # TestData 中的每个元素是 (input, label) 元组
    eval_data_list = list(eval_test_data)
    clean_data_list = list(clean_test_data)
    
    print(f"\n{'='*120}")
    print(f"{'Sample':<10} {'Time Step':<12} {'Prediction':<20} {'Eval Data':<20} {'Clean Data':<20}")
    print(f"{'='*120}")
    
    for idx in range(min(num_samples, len(forecasts))):
        forecast = forecasts[idx]
        # eval_data_list[idx] 是 (input, label) 元组，我们需要 input 部分
        eval_entry = eval_data_list[idx][0]
        clean_entry = clean_data_list[idx][0]
        
        # forecast.mean 是属性（numpy array），不是方法
        pred_mean = forecast.mean
        
        eval_start = len(eval_entry["target"]) - pred_len
        eval_values = eval_entry["target"][eval_start:eval_start + pred_len]
        
        clean_start = len(clean_entry["target"]) - pred_len
        clean_values = clean_entry["target"][clean_start:clean_start + pred_len]
        
        for t in range(min(pred_len, 10)):
            if t == 0:
                print(f"{idx:<10} {t:<12} {pred_mean[t]:<20.6f} {eval_values[t]:<20.6f} {clean_values[t]:<20.6f}")
            else:
                print(f"{'':<10} {t:<12} {pred_mean[t]:<20.6f} {eval_values[t]:<20.6f} {clean_values[t]:<20.6f}")
        
        if pred_len > 10:
            print(f"{'':<10} {'...':<12} {'...':<20} {'...':<20} {'...':<20}")
        
        print(f"{'-'*120}")


def save_results_to_csv(results: Dict[str, Any], output_path: str):
    """
    将评估结果保存到 CSV 文件
    
    Args:
        results: 评估结果字典
        output_path: 输出文件路径
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["metric", "value"])
        for key, value in results.items():
            writer.writerow([key, value])
    
    print(f"\nResults saved to: {output_path}")

