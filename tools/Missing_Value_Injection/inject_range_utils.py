"""注错区间确定工具。"""

import json
import math
import warnings
import pandas as pd
from pathlib import Path
from typing import Union
from enum import Enum

from pandas.tseries.frequencies import to_offset
from gluonts.time_feature import norm_freq_str

# 抑制 pandas 频率单位弃用警告
warnings.filterwarnings('ignore', category=FutureWarning, message=".*'H' is deprecated.*")


# ============================================================================
# 常量定义
# ============================================================================

# 测试集比例
TEST_SPLIT = 0.6

# 最大窗口数
MAX_WINDOW = 20


class Term(Enum):
    """预测 horizon 类型"""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        return {
            Term.SHORT: 1,
            Term.MEDIUM: 10,
            Term.LONG: 15,
        }[self]


# 非 M4 数据集的 prediction length 映射表
PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}


# ============================================================================
# 工具函数
# ============================================================================

def maybe_reconvert_freq(freq: str) -> str:
    """将新版 pandas 频率别名转换为旧版"""
    deprecated_map = {
        "Y": "A", "YE": "A", "QE": "Q", "ME": "M",
        "h": "H", "min": "T", "s": "S", "us": "U",
    }
    return deprecated_map.get(freq, freq)


def load_dataset_properties(data_path: str) -> dict:
    """加载数据集属性配置文件"""
    config_path = Path(data_path) / "dataset_properties.json"
    if not config_path.exists():
        fallback = Path(__file__).resolve().parents[2] / "data" / "datasets" / "dataset_properties.json"
        if fallback.exists():
            config_path = fallback
        else:
            raise FileNotFoundError(f"Dataset properties not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_prediction_length(freq: str, term: Term = Term.SHORT) -> int:
    """根据频率和 term 计算 prediction length"""
    freq_normalized = norm_freq_str(to_offset(freq).name)
    freq_normalized = maybe_reconvert_freq(freq_normalized)
    base_pred_len = PRED_LENGTH_MAP.get(freq_normalized, 48)
    return term.multiplier * base_pred_len


def compute_windows(min_series_length: int, prediction_length: int) -> int:
    """计算滑动窗口数量"""
    w = math.ceil(TEST_SPLIT * min_series_length / prediction_length)
    return min(max(1, w), MAX_WINDOW)


# ============================================================================
# 核心函数：计算注错区间
# ============================================================================

def get_injection_range(
    dataset_name: str,
    term: Union[str, Term] = "short",
    data_path: str = "data/datasets",
    max_context: int = 8192,
) -> dict:
    """
    计算注错区间位置
    
    注错区间定义：
    - 起始位置：offset = -prediction_length × windows - max_context
    - 终止位置：offset = -prediction_length
    
    Args:
        dataset_name: 数据集名称（如 "ETTh1", "national_illness"）
        term: 预测 horizon 类型 ("short", "medium", "long")
        data_path: 数据集目录路径（默认："datasets"）
        
    Returns:
        包含以下字段的字典：
        - dataset_name: 数据集名称
        - term: term 类型
        - frequency: 数据频率
        - prediction_length: 预测长度
        - windows: 滑动窗口数
        - max_context: 最大回顾窗口
        - start_index: 在整个数据集中的起始索引（从 0 开始）
        - end_index: 在整个数据集中的结束索引（不包含）
        - injection_length: 注错区间长度
        - total_length: 数据集总长度
        
    Example:
        >>> result = get_injection_range("ETTh1", "short")
        >>> print(f"注错区间：[{result['start_index']}, {result['end_index']})")
    """
    if isinstance(term, str):
        term = Term(term)
    
    props = load_dataset_properties(data_path)
    if dataset_name not in props:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    
    ds_props = props[dataset_name]
    freq = ds_props["frequency"]
    
    csv_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    total_length = len(df)
    
    prediction_length = compute_prediction_length(freq, term)
    windows = compute_windows(total_length, prediction_length)
    
    start_offset_from_end = -prediction_length * windows - max_context
    end_offset_from_end = -prediction_length
    
    start_index = max(0, total_length + start_offset_from_end)
    end_index = min(total_length, total_length + end_offset_from_end)
    injection_length = end_index - start_index
    
    return {
        "dataset_name": dataset_name,
        "term": term.value,
        "frequency": freq,
        "prediction_length": prediction_length,
        "windows": windows,
        "max_context": max_context,
        "start_index": start_index,
        "end_index": end_index,
        "injection_length": injection_length,
        "total_length": total_length,
    }


def print_injection_range(result: dict) -> None:
    """打印注错区间信息的格式化输出"""
    print("=" * 80)
    print(f"注错区间计算结果 - {result['dataset_name']} ({result['term']})")
    print("=" * 80)
    print(f"数据集信息:")
    print(f"  频率：{result['frequency']}")
    print(f"  总长度：{result['total_length']}")
    print(f"\n评估参数:")
    print(f"  Prediction Length: {result['prediction_length']}")
    print(f"  Windows: {result['windows']}")
    print(f"  Max Context: {result['max_context']}")
    print(f"\n注错区间位置:")
    print(f"  Start Index: {result['start_index']} (从 0 开始)")
    print(f"  End Index: {result['end_index']} (不包含)")
    print(f"  注错区间长度：{result['injection_length']}")
    print(f"\n注错区间占比:")
    if result['total_length'] > 0:
        ratio = result['injection_length'] / result['total_length'] * 100
        print(f"  {ratio:.2f}% ({result['injection_length']}/{result['total_length']})")
    print("=" * 80)


# ============================================================================
# 命令行工具
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="计算注错区间位置"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称（如 ETTh1, national_illness）"
    )
    parser.add_argument(
        "--term",
        type=str,
        default="short",
        choices=["short", "medium", "long"],
        help="预测 horizon 类型（默认：short）"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/datasets",
        help="数据集目录路径（默认：data/datasets）"
    )
    parser.add_argument(
        "--max_context",
        type=int,
        default=8192,
        help="最大回顾窗口长度（默认：8192）",
    )
    
    args = parser.parse_args()
    
    result = get_injection_range(
        dataset_name=args.dataset,
        term=args.term,
        data_path=args.data_path,
        max_context=args.max_context,
    )
    
    print_injection_range(result)
