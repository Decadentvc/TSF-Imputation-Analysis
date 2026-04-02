"""
外部插值文件导入工具

提供便捷函数用于将外部插值文件导入到窗口数据目录
"""

import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import pandas as pd


def import_external_imputation(
    external_file: str,
    dataset_name: str,
    term: str,
    window_index: int,
    missing_pattern: str,
    missing_ratio: float,
    output_base_dir: str = "datasets/window_imputation",
    imputation_method: str = "external",
    seed: int = 42,
) -> str:
    """
    导入外部插值文件到窗口数据目录
    
    Args:
        external_file: 外部插值文件路径
        dataset_name: 数据集名称
        term: short/medium/long
        window_index: 窗口索引
        missing_pattern: 缺失模式
        missing_ratio: 缺失比例
        output_base_dir: 输出基础目录
        imputation_method: 填补方法名称（标记为 external 或自定义）
        seed: 随机种子
        
    Returns:
        保存后的文件路径
    """
    ratio_str = f"{int(missing_ratio * 100):03d}"
    output_dir = (
        Path(output_base_dir)
        / dataset_name
        / term
        / missing_pattern
        / ratio_str
        / imputation_method
        / f"window_{window_index:03d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取并验证外部文件
    df_external = pd.read_csv(external_file)
    
    # 保存数据
    data_path = output_dir / "data.csv"
    df_external.to_csv(data_path, index=False)
    
    # 保存元数据
    meta = {
        "dataset_name": dataset_name,
        "term": term,
        "window_index": window_index,
        "missing_pattern": missing_pattern,
        "missing_ratio": missing_ratio,
        "imputation_method": imputation_method,
        "external_file": str(Path(external_file).resolve()),
        "seed": seed + window_index,
        "generated_at": datetime.now().isoformat(),
        "imported_at": datetime.now().isoformat(),
    }
    
    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"✓ External file imported: {data_path}")
    print(f"  Source: {external_file}")
    print(f"  Window: {window_index}")
    
    return str(data_path)


def batch_import_external_files(
    files_config: List[dict],
    output_base_dir: str = "datasets/window_imputation",
) -> List[str]:
    """
    批量导入外部插值文件
    
    Args:
        files_config: 配置文件列表，每个包含：
            - external_file: 外部文件路径
            - dataset_name: 数据集名称
            - term: short/medium/long
            - window_index: 窗口索引
            - missing_pattern: 缺失模式
            - missing_ratio: 缺失比例
            - imputation_method: 填补方法（可选，默认 "external"）
            - seed: 随机种子（可选，默认 42）
        output_base_dir: 输出基础目录
        
    Returns:
        保存后的文件路径列表
    """
    saved_paths = []
    
    for config in files_config:
        try:
            path = import_external_imputation(
                external_file=config["external_file"],
                dataset_name=config["dataset_name"],
                term=config["term"],
                window_index=config["window_index"],
                missing_pattern=config["missing_pattern"],
                missing_ratio=config["missing_ratio"],
                output_base_dir=output_base_dir,
                imputation_method=config.get("imputation_method", "external"),
                seed=config.get("seed", 42),
            )
            saved_paths.append(path)
        except Exception as e:
            print(f"❌ Error importing {config['external_file']}: {e}")
    
    return saved_paths


def copy_missing_file_to_imputation(
    missing_file_path: str,
    imputation_method: str,
    output_base_dir: str = "datasets/window_imputation",
) -> str:
    """
    将缺失文件复制到指定填补方法目录（用于在外部插值后保存）
    
    Args:
        missing_file_path: 缺失文件路径（包含缺失值的数据）
        imputation_method: 填补方法名称
        output_base_dir: 输出基础目录
        
    Returns:
        复制后的文件路径
    """
    missing_path = Path(missing_file_path)
    
    # 从缺失文件路径解析信息
    # 格式：.../{dataset}/{term}/{pattern}/{ratio}/_missing_files/window_{idx}/missing_data.csv
    parts = missing_path.parts
    window_idx = None
    ratio_str = None
    
    for i, part in enumerate(parts):
        if part.startswith("window_"):
            window_idx = part
        if i > 0 and parts[i-1] in ["MCAR", "BM", "TM", "TVMR"]:
            ratio_str = part
    
    if window_idx is None or ratio_str is None:
        raise ValueError(f"Cannot parse window info from path: {missing_file_path}")
    
    # 读取缺失文件
    df_missing = pd.read_csv(missing_path)
    
    # 读取元数据
    meta_path = missing_path.parent / "meta.json"
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            missing_meta = json.load(f)
    else:
        missing_meta = {}
    
    # 构建输出路径
    output_dir = (
        Path(output_base_dir)
        / missing_meta.get("dataset_name", "unknown")
        / missing_meta.get("term", "short")
        / missing_meta.get("missing_pattern", "MCAR")
        / ratio_str
        / imputation_method
        / window_idx
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据
    data_path = output_dir / "data.csv"
    df_missing.to_csv(data_path, index=False)
    
    # 保存元数据
    meta = {
        "dataset_name": missing_meta.get("dataset_name", "unknown"),
        "term": missing_meta.get("term", "short"),
        "window_index": missing_meta.get("window_index", 0),
        "missing_pattern": missing_meta.get("missing_pattern", "MCAR"),
        "missing_ratio": missing_meta.get("missing_ratio", 0.05),
        "imputation_method": imputation_method,
        "source_missing_file": str(missing_path.resolve()),
        "seed": missing_meta.get("seed", 42),
        "generated_at": datetime.now().isoformat(),
        "imported_at": datetime.now().isoformat(),
    }
    
    meta_path = output_dir / "meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Copied missing file to imputation directory: {data_path}")
    print(f"  Imputation method: {imputation_method}")
    
    return str(data_path)


def get_missing_file_path(
    dataset_name: str,
    term: str,
    window_index: int,
    missing_pattern: str,
    missing_ratio: float,
    output_base_dir: str = "datasets/window_imputation",
) -> Path:
    """
    获取缺失文件的路径
    
    Args:
        dataset_name: 数据集名称
        term: short/medium/long
        window_index: 窗口索引
        missing_pattern: 缺失模式
        missing_ratio: 缺失比例
        output_base_dir: 输出基础目录
        
    Returns:
        缺失文件路径
    """
    ratio_str = f"{int(missing_ratio * 100):03d}"
    return (
        Path(output_base_dir)
        / dataset_name
        / term
        / missing_pattern
        / ratio_str
        / "_missing_files"
        / f"window_{window_index:03d}"
        / "missing_data.csv"
    )


def list_available_missing_files(
    dataset_name: Optional[str] = None,
    term: Optional[str] = None,
    missing_pattern: Optional[str] = None,
    missing_ratio: Optional[float] = None,
    output_base_dir: str = "datasets/window_imputation",
) -> List[Path]:
    """
    列出所有可用的缺失文件
    
    Args:
        dataset_name: 数据集名称过滤
        term: term 过滤
        missing_pattern: 缺失模式过滤
        missing_ratio: 缺失比例过滤
        output_base_dir: 输出基础目录
        
    Returns:
        缺失文件路径列表
    """
    base_path = Path(output_base_dir)
    if not base_path.exists():
        return []
    
    missing_files = []
    
    for missing_file in base_path.glob("**/_missing_files/*/missing_data.csv"):
        parts = missing_file.parts
        
        # 解析路径信息
        info = {}
        for i, part in enumerate(parts):
            if part in ["MCAR", "BM", "TM", "TVMR"]:
                info["missing_pattern"] = part
                if i + 1 < len(parts):
                    info["missing_ratio"] = parts[i + 1]
            if part in ["short", "medium", "long"]:
                info["term"] = part
            if part.startswith("window_"):
                info["window_index"] = part
        
        # 应用过滤
        if dataset_name and dataset_name not in str(missing_file):
            continue
        if term and info.get("term") != term:
            continue
        if missing_pattern and info.get("missing_pattern") != missing_pattern:
            continue
        if missing_ratio:
            ratio_str = f"{int(missing_ratio * 100):03d}"
            if info.get("missing_ratio") != ratio_str:
                continue
        
        missing_files.append(missing_file)
    
    return sorted(missing_files)
