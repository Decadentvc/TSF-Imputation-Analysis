"""
窗口数据处理器

整合注错和填补逻辑，管理窗口数据的加载、生成和保存
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import pandas as pd

# 添加当前目录到路径，确保能导入同级模块
sys.path.insert(0, str(Path(__file__).parent))
from imputation_methods import get_imputation_method

# 添加注错模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / "Missing_Value_Injection" / "for_sundial"))
from window_injection import inject_missing_to_window, get_window_injection_range
from inject_range_utils import load_dataset_properties, Term, compute_prediction_length


class WindowImputationProcessor:
    """
    窗口填补处理器
    
    管理每个窗口的：
    1. 检查是否已存在处理后的数据
    2. 如果存在则加载
    3. 如果不存在则生成（注错 + 填补）并保存
    """
    
    def __init__(
        self,
        dataset_name: str,
        term: str,
        missing_pattern: str,
        missing_ratio: float,
        imputation_method: str,
        data_path: str = "datasets",
        output_base_dir: str = "datasets/window_imputation",
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.term = term
        self.missing_pattern = missing_pattern
        self.missing_ratio = missing_ratio
        self.imputation_method = imputation_method
        self.data_path = Path(data_path)
        self.output_base_dir = Path(output_base_dir)
        self.seed = seed
        
        self.time_cols = ['date', 'time', 'timestamp']
        self.data_cols = self._get_data_columns()
    
    def _get_data_columns(self) -> list:
        """获取数据列名"""
        csv_path = self.data_path / "ori" / f"{self.dataset_name}.csv"
        df = pd.read_csv(csv_path)
        return [col for col in df.columns if col not in self.time_cols]
    
    def _get_window_output_dir(self, window_index: int) -> Path:
        """获取窗口输出目录"""
        ratio_str = f"{int(self.missing_ratio * 100):03d}"
        return (
            self.output_base_dir 
            / self.dataset_name 
            / self.term
            / self.missing_pattern
            / ratio_str
            / self.imputation_method
            / f"window_{window_index:03d}"
        )
    
    def _get_data_path(self, window_index: int) -> Path:
        """获取数据文件路径"""
        return self._get_window_output_dir(window_index) / "data.csv"
    
    def _get_meta_path(self, window_index: int) -> Path:
        """获取元数据文件路径"""
        return self._get_window_output_dir(window_index) / "meta.json"
    
    def _window_exists(self, window_index: int) -> bool:
        """检查窗口数据是否已存在"""
        return (
            self._get_data_path(window_index).exists() and
            self._get_meta_path(window_index).exists()
        )
    
    def _load_window(self, window_index: int) -> Tuple[pd.DataFrame, Dict]:
        """加载窗口数据"""
        df = pd.read_csv(self._get_data_path(window_index))
        
        with open(self._get_meta_path(window_index), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        return df, meta
    
    def _save_window(self, window_index: int, df: pd.DataFrame, meta: Dict):
        """保存窗口数据"""
        output_dir = self._get_window_output_dir(window_index)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(self._get_data_path(window_index), index=False)
        
        with open(self._get_meta_path(window_index), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
    
    def _process_clean_window(self, window_index: int) -> Tuple[pd.DataFrame, Dict]:
        """
        快速路径：直接返回干净数据（不经过缺失值注入流程）
        
        当 missing_ratio=0 且 imputation_method='none' 时使用此方法
        可以避免注入流程引入的副作用
        
        Args:
            window_index: 窗口索引
            
        Returns:
            (df_clean, meta)
        """
        from inject_range_utils import load_dataset_properties
        from window_injection import get_window_injection_range
        
        print(f"[Fast path] Loading clean data for window {window_index}...")
        
        # 获取窗口信息
        # 注意：window_injection.py 的窗口索引是从后往前（窗口 0 是数据集末尾）
        # 但 eval_sundial.py 的窗口索引是从前往后（窗口 0 是 split 点之后）
        # 因此需要反转窗口索引以匹配 eval_sundial.py
        # 首先获取数据集长度来计算窗口总数
        from inject_range_utils import load_dataset_properties
        from window_injection import get_window_injection_range, compute_window_boundaries
        
        props = load_dataset_properties(str(self.data_path))
        freq = props[self.dataset_name]["frequency"]
        
        csv_path = Path(self.data_path) / "ori" / f"{self.dataset_name}.csv"
        df_clean = pd.read_csv(csv_path)
        dataset_length = len(df_clean)
        
        term_enum = Term(self.term)
        prediction_length = compute_prediction_length(freq, term_enum)
        n_windows = min(20, dataset_length // prediction_length)  # 与 run_sundial_imputation.py 一致
        
        reversed_window_index = n_windows - 1 - window_index
        
        injection_range = get_window_injection_range(
            self.dataset_name, 
            self.term, 
            reversed_window_index,  # 使用反转后的索引
            str(self.data_path)
        )
        
        ctx_start = injection_range["context_start"]
        fc_start = injection_range["forecast_start"]
        
        # 加载干净数据
        csv_path = Path(self.data_path) / "ori" / f"{self.dataset_name}.csv"
        df_clean = pd.read_csv(csv_path)
        
        # 提取窗口数据（从数据集开始到预测点，匹配 eval_sundial.py 的逻辑）
        # 但 Sundial 模型最多只读 2880 点历史数据，所以只需要保存最近的 2880 点
        # eval_sundial.py 的 batch_x_shape 默认是 2880
        # 注意：我们需要额外包含 prediction_length 个点作为 label，这样 split 才能正确提取
        max_context = 2880
        actual_start = max(0, fc_start - max_context)
        df_window = df_clean.iloc[actual_start:fc_start + prediction_length].copy()
        
        meta = {
            "dataset_name": self.dataset_name,
            "term": self.term,
            "window_index": window_index,
            "missing_pattern": self.missing_pattern,
            "missing_ratio": self.missing_ratio,
            "imputation_method": self.imputation_method,
            "seed": self.seed + window_index,
            "generated_at": datetime.now().isoformat(),
            "fast_path": True,  # 标记使用了快速路径
            "context_start": actual_start,
            "context_end": fc_start,
            "context_length": fc_start - actual_start,
        }
        
        # 保存数据（与正常路径一致）
        self._save_window(window_index, df_window, meta)
        print(f"[OK] Window {window_index} saved (fast path).")
        
        return df_window, meta
    
    def process_window(
        self,
        window_index: int,
        force_regenerate: bool = False,
        external_file: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        处理单个窗口
        
        Args:
            window_index: 窗口索引
            force_regenerate: 是否强制重新生成
            external_file: 外部插值文件路径（如果提供，则直接使用该文件）
            
        Returns:
            (df_imputed, meta)
        """
        # 检查是否提供外部文件
        if external_file is not None:
            return self._process_external_file(window_index, external_file)
        
        # 快速路径：当 missing_ratio=0 且 imputation_method='none' 时，直接返回干净数据
        # 这样可以避免注入流程引入的副作用
        if self.missing_ratio == 0 and self.imputation_method == 'none':
            return self._process_clean_window(window_index)
        
        # 检查是否已存在
        if not force_regenerate and self._window_exists(window_index):
            print(f"[OK] Loading window {window_index}...")
            return self._load_window(window_index)
        
        print(f"-> Generating window {window_index}...")
        
        df_with_missing, injection_info = inject_missing_to_window(
            dataset_name=self.dataset_name,
            term=self.term,
            window_index=window_index,
            missing_pattern=self.missing_pattern,
            missing_ratio=self.missing_ratio,
            seed=self.seed,
            data_path=str(self.data_path),
            save_missing_file=True,
            output_base_dir=str(self.output_base_dir),
        )
        
        imputation_func = get_imputation_method(self.imputation_method)
        
        if self.imputation_method == 'seasonal':
            from inject_range_utils import load_dataset_properties
            props = load_dataset_properties(str(self.data_path))
            freq = props[self.dataset_name]["frequency"]
            df_imputed = imputation_func(df_with_missing, self.data_cols, freq)
        elif self.imputation_method == 'none':
            # 不填补，直接返回原始数据
            df_imputed = imputation_func(df_with_missing, self.data_cols)
        else:
            df_imputed = imputation_func(df_with_missing, self.data_cols)
        
        meta = {
            "dataset_name": self.dataset_name,
            "term": self.term,
            "window_index": window_index,
            "missing_pattern": self.missing_pattern,
            "missing_ratio": self.missing_ratio,
            "imputation_method": self.imputation_method,
            "seed": self.seed + window_index,
            "generated_at": datetime.now().isoformat(),
        }
        
        self._save_window(window_index, df_imputed, meta)
        print(f"[OK] Window {window_index} saved.")
        
        return df_imputed, meta
    
    def _process_external_file(
        self,
        window_index: int,
        external_file: str,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        处理外部提供的插值文件
        
        Args:
            window_index: 窗口索引
            external_file: 外部插值文件路径
            
        Returns:
            (df_imputed, meta)
        """
        print(f"-> Loading external file for window {window_index}: {external_file}")
        
        # 读取外部文件
        df_external = pd.read_csv(external_file)
        
        # 构建元数据
        meta = {
            "dataset_name": self.dataset_name,
            "term": self.term,
            "window_index": window_index,
            "missing_pattern": self.missing_pattern,
            "missing_ratio": self.missing_ratio,
            "imputation_method": "external",
            "external_file": external_file,
            "seed": self.seed + window_index,
            "generated_at": datetime.now().isoformat(),
        }
        
        # 保存
        self._save_window(window_index, df_external, meta)
        print(f"[OK] Window {window_index} saved from external file.")
        
        return df_external, meta
    
    def get_all_windows_info(self) -> List[Dict]:
        """获取所有窗口信息"""
        import math
        
        windows_info = []
        window_idx = 0
        
        print(f"  -> Counting windows...")
        
        # 获取数据集信息
        props = load_dataset_properties(str(self.data_path))
        ds_props = props[self.dataset_name]
        freq = ds_props["frequency"]
        
        csv_path = Path(self.data_path) / "ori" / f"{self.dataset_name}.csv"
        df = pd.read_csv(csv_path)
        dataset_length = len(df)
        
        # 计算 prediction_length
        term_enum = Term(self.term)
        prediction_length = compute_prediction_length(freq, term_enum)
        
        # 计算窗口数量：与 eval_sundial.py 保持一致
        # 使用 math.ceil 和 TEST_SPLIT=0.1（与 eval_sundial.py 一致）
        TEST_SPLIT = 0.1
        MAX_WINDOW = 20
        w = math.ceil(TEST_SPLIT * dataset_length / prediction_length)
        n_windows = min(max(1, w), MAX_WINDOW)
        
        print(f"    Dataset length: {dataset_length}")
        print(f"    Prediction length: {prediction_length}")
        print(f"    Calculated windows (TEST_SPLIT={TEST_SPLIT}): {w}")
        print(f"    Final windows (max={MAX_WINDOW}): {n_windows}")
        
        # 生成窗口信息
        for window_idx in range(n_windows):
            info = get_window_injection_range(
                self.dataset_name, 
                self.term, 
                window_idx,
                str(self.data_path)
            )
            windows_info.append(info)
        
        print(f"  Total windows: {len(windows_info)}")
        return windows_info
