"""
分量变化率 vs SMAPE 变化率分析

分析 seasonal_decompose 分解的分量（趋势、季节）和主频分量在填补前后的变化率，
与预测精度变化率 (DeltaSMAPE) 的相关性。

分量变化率定义:
    component_change_rate = ||comp_imputed - comp_clean||_2 / ||comp_clean||_2

三种分量:
    - trend_change_rate:   趋势分量的变化率 (seasonal_decompose)
    - seasonal_change_rate: 季节分量的变化率 (seasonal_decompose)
    - freq_change_rate:     FFT 主频幅值的变化率

两种关系:
    1. 历史窗口分量变化率 vs DeltaSMAPE  (历史畸变是否影响预测精度)
    2. 历史窗口分量变化率 vs 预测窗口分量变化率 (畸变是否传导到预测输出)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore", category=UserWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "tools" / "Missing_Value_Injection"))

from inject_range_utils import get_injection_range
from Analysis.window_analysis import get_model_max_context
from Analysis.metrics import get_period

# ======================================================
# Constants
# ======================================================

MODELS = [
    "chronos2", "kairos23m", "kairos50m", "sundial",
    "timesfm2p0", "timesfm2p5", "visiontspp",
]

MODEL_LABELS = {
    "chronos2": "Chronos2", "kairos23m": "Kairos23m", "kairos50m": "Kairos50m",
    "sundial": "SunDial", "timesfm2p0": "TimesFM2.0",
    "timesfm2p5": "TimesFM2.5", "visiontspp": "VisionTS++",
}

MODEL_COLORS = {
    "chronos2": "#1f77b4", "kairos23m": "#ff7f0e", "kairos50m": "#2ca02c",
    "sundial": "#d62728", "timesfm2p0": "#9467bd",
    "timesfm2p5": "#8c564b", "visiontspp": "#e377c2",
}

COMPONENTS = ["trend", "seasonal", "freq"]
WINDOW_TYPES = ["pred", "hist"]

TERMS = {"short", "medium", "long"}

INTERMEDIATE_DIR = ROOT_DIR / "data" / "Intermediate_Predictions"
DATA_DIR = ROOT_DIR / "data" / "datasets"
RESULTS_DIR = ROOT_DIR / "results"
BIAS_BY_MODEL_DIR = ROOT_DIR / "results_analysis" / "bias_analysis" / "by_model"
MODEL_PROPERTIES_PATH = ROOT_DIR / "Eval" / "model_properties.json"

# ======================================================
# Utility Functions
# ======================================================

def _norm_rmse(imputed: np.ndarray, clean: np.ndarray) -> float:
    """归一化 RMSE: ||imp - cln||_2 / ||cln||_2 (变化率)"""
    diff = imputed - clean
    denom = np.sqrt(np.sum(clean ** 2))
    if denom < 1e-10:
        return float("nan")
    return float(np.sqrt(np.sum(diff ** 2)) / denom)


def _decompose_seasonal(series: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """seasonal_decompose 分解，返回 (trend, seasonal)"""
    if len(series) < 2 * period:
        raise ValueError(f"Series too short: {len(series)} < 2*{period}")
    result = seasonal_decompose(series, model='additive', period=period,
                                extrapolate_trend='freq')
    trend = pd.Series(result.trend).interpolate(limit_direction='both').bfill().ffill().values
    seasonal = pd.Series(result.seasonal).fillna(0).values
    return trend, seasonal


def _compute_fft_mags(series: np.ndarray) -> np.ndarray:
    """FFT 幅值谱 (去除均值后)"""
    series = series - np.mean(series)
    return np.abs(np.fft.rfft(series))


def _find_main_freq_indices(mags: np.ndarray, n_main: int = 5) -> np.ndarray:
    """找到幅值最大的前 n_main 个频率索引 (排除 DC)"""
    if len(mags) <= 1:
        return np.array([], dtype=int)
    indices = np.argsort(mags[1:])[::-1][:n_main] + 1
    return indices[indices < len(mags)]


def _compute_component_crs(
    clean_vals: np.ndarray, imputed_vals: np.ndarray, period: int
) -> Dict[str, float]:
    """计算一个序列的 3 种分量变化率"""
    result: Dict[str, float] = {
        "trend_change_rate": float("nan"),
        "seasonal_change_rate": float("nan"),
        "freq_change_rate": float("nan"),
    }

    # --- FFT 频率分量变化率 (始终可算) ---
    clean_mags = _compute_fft_mags(clean_vals)
    imp_mags = _compute_fft_mags(imputed_vals)
    main_idx = _find_main_freq_indices(clean_mags, n_main=5)
    if len(main_idx) > 0 and len(clean_mags) == len(imp_mags):
        result["freq_change_rate"] = _norm_rmse(imp_mags[main_idx], clean_mags[main_idx])

    # --- seasonal_decompose 趋势 + 季节分量变化率 ---
    if len(clean_vals) >= 2 * period:
        try:
            trend_c, seas_c = _decompose_seasonal(clean_vals, period)
            trend_i, seas_i = _decompose_seasonal(imputed_vals, period)
            result["trend_change_rate"] = _norm_rmse(trend_i, trend_c)
            result["seasonal_change_rate"] = _norm_rmse(seas_i, seas_c)
        except Exception:
            pass

    return result


# ======================================================
# SMAPE Loading (复用 plot_metric_vs_smape.py 逻辑)
# ======================================================

def load_clean_smape(results_dir: Path) -> Dict[Tuple[str, str, str], float]:
    """{(model, dataset, term): smape}"""
    result: Dict[Tuple[str, str, str], float] = {}
    pat = re.compile(r"^(.+)_clean_(short|medium|long)_results\.csv$")
    for fp in sorted(results_dir.glob("*/clean/*_results.csv")):
        model = fp.parent.parent.name
        m = pat.match(fp.name)
        if not m:
            continue
        dataset, term = m.group(1), m.group(2)
        df = pd.read_csv(fp)
        row = df[df["metric"] == "sMAPE[0.5]"]
        if row.empty:
            continue
        result[(model, dataset, term)] = float(row["value"].iloc[0])
    return result


def load_impute_smape(results_dir: Path) -> Dict[Tuple[str, str, str, int, str], float]:
    """{(model, dataset, term, ratio, method): smape}"""
    result: Dict[Tuple[str, str, str, int, str], float] = {}
    pat = re.compile(
        r"^([A-Za-z0-9]+)_(.+)_BM_length\d+_(\d{3})_(short|medium|long)_results\.csv$"
    )
    for fp in sorted(results_dir.glob("*/impute/*_results.csv")):
        model = fp.parent.parent.name
        m = pat.match(fp.name)
        if not m:
            continue
        method, dataset, ratio_str, term = (
            m.group(1), m.group(2), m.group(3), m.group(4)
        )
        df = pd.read_csv(fp)
        row = df[df["metric"] == "sMAPE[0.5]"]
        if row.empty:
            continue
        result[(model, dataset, term, int(ratio_str), method)] = float(row["value"].iloc[0])
    return result


def _find_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["timestamp", "date", "time", "datetime"]:
        if c in df.columns:
            return c
    return df.columns[0] if len(df.columns) > 0 else None


def _compute_smape_from_pred_file(
    pred_file: Path, dataset: str, data_dir: Path, orig_cache: Dict[str, tuple]
) -> Optional[float]:
    if not pred_file.exists():
        return None
    try:
        pred_df = pd.read_csv(pred_file)
    except Exception:
        return None

    if dataset not in orig_cache:
        orig_path = data_dir / "ori" / f"{dataset}.csv"
        if not orig_path.exists():
            orig_cache[dataset] = None
            return None
        try:
            odf = pd.read_csv(orig_path)
            tc = _find_time_col(odf)
            orig_cache[dataset] = (tc, odf) if tc else None
        except Exception:
            orig_cache[dataset] = None
            return None

    if orig_cache[dataset] is None:
        return None
    time_col, orig_df = orig_cache[dataset]

    pred_dates = pd.to_datetime(pred_df["date"])
    pred_vals = pred_df["prediction"].values.astype(np.float64)
    orig_dates = pd.to_datetime(orig_df[time_col])

    mask = orig_dates.isin(pred_dates)
    if mask.sum() == 0:
        return None

    var_cols = [c for c in orig_df.columns if c != time_col]
    smapes = []
    for col in var_cols:
        orig_vals = orig_df.loc[mask, col].values.astype(np.float64)
        if len(orig_vals) != len(pred_vals):
            continue
        diff = np.abs(pred_vals - orig_vals)
        denom = (np.abs(pred_vals) + np.abs(orig_vals)) / 2.0
        denom = np.where(denom < 1e-10, 1e-10, denom)
        smapes.append(float(np.mean(diff / denom) * 100))

    return float(np.mean(smapes)) if smapes else None


def load_bias_pairs(bias_dir: Path, models: List[str]) -> pd.DataFrame:
    """加载 bias_pairs.csv 并提取唯一 (model,dataset,term,ratio,method,window_idx)"""
    pieces = []
    for model in models:
        fp = bias_dir / model / "bias_pairs.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            pieces.append(df)
    if not pieces:
        return pd.DataFrame()
    df = pd.concat(pieces, ignore_index=True)
    # window_idx 和 ratio 转为 int
    df["window_idx"] = df["window_idx"].astype(int)
    df["ratio"] = df["ratio"].astype(int)
    # 去重保留唯一组合
    uniq = df[["model", "dataset", "term", "ratio", "method", "window_idx"]].drop_duplicates()
    return uniq.reset_index(drop=True)


# ======================================================
# Prediction Window 的分量变化率
# ======================================================

def _load_pred_series(pred_file: Path) -> Optional[np.ndarray]:
    """读取预测文件, 返回 prediction 列的值数组"""
    if not pred_file.exists():
        return None
    try:
        df = pd.read_csv(pred_file)
        if "prediction" not in df.columns:
            return None
        return df["prediction"].values.astype(np.float64)
    except Exception:
        return None


def compute_prediction_crs(
    model: str,
    dataset: str,
    term: str,
    ratio: int,
    method: str,
    window_indices: List[int],
    period: int,
) -> List[Dict[str, Any]]:
    """计算预测窗口的分量变化率"""
    ratio_s = f"{ratio:03d}"
    pred_dir = INTERMEDIATE_DIR / model / f"{dataset}_BM_length50_{ratio_s}_{term}_prediction" / method
    clean_dir = INTERMEDIATE_DIR / model / f"{dataset}_clean_{term}_prediction"

    records: List[Dict[str, Any]] = []
    for w_idx in window_indices:
        imp_file = pred_dir / f"{dataset}_BM_length50_{ratio_s}_{term}_prediction_{w_idx}.csv"
        cln_file = clean_dir / f"{dataset}_clean_{term}_prediction_{w_idx}.csv"

        imp_vals = _load_pred_series(imp_file)
        cln_vals = _load_pred_series(cln_file)
        if imp_vals is None or cln_vals is None:
            continue
        if len(imp_vals) != len(cln_vals):
            continue

        cr = _compute_component_crs(cln_vals, imp_vals, period)
        records.append({
            "window_idx": w_idx,
            "trend_change_rate_pred": cr["trend_change_rate"],
            "seasonal_change_rate_pred": cr["seasonal_change_rate"],
            "freq_change_rate_pred": cr["freq_change_rate"],
        })
    return records


# ======================================================
# History Window 的分量变化率
# ======================================================

def _get_data_cols(df: pd.DataFrame) -> List[str]:
    """返回非时间列"""
    time_cols = {"date", "time", "timestamp", "datetime", "index"}
    return [c for c in df.columns if c.lower() not in time_cols]


def compute_history_crs(
    dataset: str,
    term: str,
    ratio: int,
    method: str,
    model: str,
    window_indices: List[int],
    period: int,
    data_dir: Path = DATA_DIR,
) -> List[Dict[str, Any]]:
    """计算历史窗口的分量变化率 (跨变量平均)"""
    # ---- 加载 clean 原始数据 ----
    clean_path = data_dir / "ori" / f"{dataset}.csv"
    if not clean_path.exists():
        return []
    try:
        clean_df = pd.read_csv(clean_path)
    except Exception:
        return []
    cln_cols = _get_data_cols(clean_df)
    if not cln_cols:
        return []

    # ---- 加载 imputed 数据 ----
    ratio_s = f"{ratio:03d}"
    imp_path = data_dir / "imputed" / "BM" / f"BM_{ratio_s}" / f"{dataset}_BM_{ratio_s}_{term}_{method}.csv"
    if not imp_path.exists():
        return []
    try:
        imp_df = pd.read_csv(imp_path)
    except Exception:
        return []

    n_total = min(len(clean_df), len(imp_df))

    # ---- 计算 injection_range 获取窗口边界 ----
    try:
        max_ctx = get_model_max_context(model, str(MODEL_PROPERTIES_PATH))
    except Exception:
        return []

    try:
        inj = get_injection_range(
            dataset_name=dataset, term=term,
            data_path=str(data_dir), max_context=max_ctx,
        )
    except Exception:
        return []

    pred_len = inj["prediction_length"]
    total_length = inj["total_length"]

    records: List[Dict[str, Any]] = []
    for w_idx in window_indices:
        forecast_end = total_length - pred_len * w_idx
        history_end = forecast_end - pred_len
        history_start = max(0, history_end - max_ctx)

        if history_start >= n_total or history_end > n_total:
            records.append({
                "window_idx": int(w_idx),
                "trend_change_rate_hist": float("nan"),
                "seasonal_change_rate_hist": float("nan"),
                "freq_change_rate_hist": float("nan"),
                "n_vars_hist": 0,
            })
            continue

        trend_crs: List[float] = []
        seas_crs: List[float] = []
        freq_crs: List[float] = []

        for col in cln_cols:
            cln_seg = clean_df[col].iloc[history_start:history_end].values.astype(np.float64)
            if col in imp_df.columns:
                imp_seg = imp_df[col].iloc[history_start:history_end].values.astype(np.float64)
            else:
                continue
            if len(cln_seg) < 2 or len(imp_seg) < 2:
                continue

            cr = _compute_component_crs(cln_seg, imp_seg, period)
            if not np.isnan(cr["trend_change_rate"]):
                trend_crs.append(cr["trend_change_rate"])
            if not np.isnan(cr["seasonal_change_rate"]):
                seas_crs.append(cr["seasonal_change_rate"])
            if not np.isnan(cr["freq_change_rate"]):
                freq_crs.append(cr["freq_change_rate"])

        n_vars = len(trend_crs)
        records.append({
            "window_idx": int(w_idx),
            "trend_change_rate_hist": float(np.nanmean(trend_crs)) if trend_crs else float("nan"),
            "seasonal_change_rate_hist": float(np.nanmean(seas_crs)) if seas_crs else float("nan"),
            "freq_change_rate_hist": float(np.nanmean(freq_crs)) if freq_crs else float("nan"),
            "n_vars_hist": n_vars,
        })
    return records


# ======================================================
# 窗口级 SMAPE 计算
# ======================================================

def compute_per_window_smape(
    combos: pd.DataFrame,
) -> Dict[Tuple[str, str, str, int, str, int], float]:
    """逐窗口计算填补预测的 SMAPE"""
    orig_cache: Dict[str, tuple] = {}
    out: Dict[Tuple[str, str, str, int, str, int], float] = {}
    data_dir = DATA_DIR
    intermediate_dir = INTERMEDIATE_DIR

    skipped = 0
    for _, row in combos.iterrows():
        model = str(row["model"])
        dataset = str(row["dataset"])
        term = str(row["term"])
        ratio = int(row["ratio"])
        method = str(row["method"])
        window_idx = int(row["window_idx"])

        ratio_s = f"{ratio:03d}"
        pred_file = (
            intermediate_dir
            / model
            / f"{dataset}_BM_length50_{ratio_s}_{term}_prediction"
            / method
            / f"{dataset}_BM_length50_{ratio_s}_{term}_prediction_{window_idx}.csv"
        )
        smape = _compute_smape_from_pred_file(pred_file, dataset, data_dir, orig_cache)
        if smape is not None:
            out[(model, dataset, term, ratio, method, window_idx)] = smape
        else:
            skipped += 1

    if skipped:
        print(f"  [提示] {skipped} 个填补预测文件无法读取")
    return out


def compute_clean_per_window_smape(
    combos: pd.DataFrame,
) -> Dict[Tuple[str, str, str, int], float]:
    """逐窗口计算干净预测的 SMAPE"""
    orig_cache: Dict[str, tuple] = {}
    out: Dict[Tuple[str, str, str, int], float] = {}
    data_dir = DATA_DIR
    intermediate_dir = INTERMEDIATE_DIR
    skipped = 0

    configs = combos[["model", "dataset", "term", "window_idx"]].drop_duplicates()
    for _, row in configs.iterrows():
        model = str(row["model"])
        dataset = str(row["dataset"])
        term = str(row["term"])
        window_idx = int(row["window_idx"])

        pred_file = (
            intermediate_dir
            / model
            / f"{dataset}_clean_{term}_prediction"
            / f"{dataset}_clean_{term}_prediction_{window_idx}.csv"
        )
        smape = _compute_smape_from_pred_file(pred_file, dataset, data_dir, orig_cache)
        if smape is not None:
            out[(model, dataset, term, window_idx)] = smape
        else:
            skipped += 1

    if skipped:
        print(f"  [提示] {skipped} 个干净预测文件无法读取")
    return out


# ======================================================
# Plotting
# ======================================================

def _scatter_component(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_key: str,
    xlabel: str,
    title: str,
):
    """在 ax 上绘制散点图 + 回归线"""
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes,
                fontsize=11)
        ax.set_title(title, fontsize=10, fontweight="bold")
        return

    # 按缺失比例着色
    ratio_colors = {10: "#1a9850", 20: "#fdae61", 30: "#d73027"}
    for ratio, grp in df.groupby("ratio"):
        x = pd.to_numeric(grp[x_key], errors="coerce").values
        y = pd.to_numeric(grp["delta_smape"], errors="coerce").values
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) == 0:
            continue
        ax.scatter(x, y, alpha=0.4, s=8, c=ratio_colors.get(int(ratio), "#888888"),
                   label=f"{int(ratio)}%", edgecolors="none", rasterized=True)

    # 全局回归线
    x_all = pd.to_numeric(df[x_key], errors="coerce").values
    y_all = pd.to_numeric(df["delta_smape"], errors="coerce").values
    mask = ~(np.isnan(x_all) | np.isnan(y_all))
    x_all, y_all = x_all[mask], y_all[mask]
    if len(x_all) > 5:
        corr = float(np.corrcoef(x_all, y_all)[0, 1])
        coeffs = np.polyfit(x_all, y_all, 1)
        x_line = np.linspace(np.nanmin(x_all), np.nanmax(x_all), 100)
        ax.plot(x_line, np.polyval(coeffs, x_line), "--", color="black", alpha=0.3, lw=1)
        ax.text(0.05, 0.95, f"r = {corr:.4f}", transform=ax.transAxes,
                va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    ax.axhline(y=0, color="gray", lw=0.6, alpha=0.3)
    ax.axvline(x=0, color="gray", lw=0.6, alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("DeltaSMAPE (impute - clean)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best", markerscale=0.8, framealpha=0.7, title="Ratio")
    ax.grid(True, alpha=0.2)


def plot_all_figures(
    full_df: pd.DataFrame,
    output_dir: Path,
):
    """生成 2x3 网格图 (pred/hist x trend/seasonal/freq)"""
    if full_df.empty:
        print("  [跳过] 无数据")
        return

    x_keys_pred = {
        "trend": "trend_change_rate_pred",
        "seasonal": "seasonal_change_rate_pred",
        "freq": "freq_change_rate_pred",
    }
    x_keys_hist = {
        "trend": "trend_change_rate_hist",
        "seasonal": "seasonal_change_rate_hist",
        "freq": "freq_change_rate_hist",
    }
    x_labels = {
        "trend": "Trend Change Rate",
        "seasonal": "Seasonal Change Rate",
        "freq": "Freq Change Rate",
    }

    def _render(fig_df: pd.DataFrame, path: Path, title: str):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=14, y=1.01)

        # 第一行: prediction
        for i, comp in enumerate(COMPONENTS):
            ax = axes[0][i]
            sub = fig_df.dropna(subset=[x_keys_pred[comp], "delta_smape"])
            _scatter_component(
                ax, sub, x_key=x_keys_pred[comp],
                xlabel=x_labels[comp],
                title=f"Prediction - {x_labels[comp]}",
            )

        # 第二行: history
        for i, comp in enumerate(COMPONENTS):
            ax = axes[1][i]
            sub = fig_df.dropna(subset=[x_keys_hist[comp], "delta_smape"])
            _scatter_component(
                ax, sub, x_key=x_keys_hist[comp],
                xlabel=x_labels[comp],
                title=f"History - {x_labels[comp]}",
            )

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [保存] {path}")

    # ---- All models ----
    _render(
        full_df, output_dir / "all_models_scatter.png",
        "All Models - Component Change Rate vs DeltaSMAPE\n(Per-window SMAPE)",
    )

    # ---- Per model ----
    for model in MODELS:
        sub = full_df[full_df["model"] == model]
        if sub.empty:
            continue
        _render(
            sub, output_dir / f"{model}_scatter.png",
            f"{MODEL_LABELS.get(model, model)} - Component Change Rate vs DeltaSMAPE",
        )


# ======================================================
# Main Pipeline
# ======================================================

def process_model_group(
    group: pd.DataFrame,
    period_map: Dict[str, int],
    smape_cache: Dict,
    clean_smape_cache: Dict,
    compute_window_smape: bool,
    skip_history: bool,
) -> pd.DataFrame:
    """处理一个 model 的所有数据"""
    model = str(group["model"].iloc[0])
    all_rows: List[Dict[str, Any]] = []

    # 获取该 model 的所有唯一 (dataset, term, ratio, method) 组合
    configs = group[["dataset", "term", "ratio", "method"]].drop_duplicates()
    n_configs = len(configs)
    t0 = time.time()

    print(f"  模型 {model}: {n_configs} 个配置, ", end="", flush=True)

    for idx, (_, cfg) in enumerate(configs.iterrows()):
        dataset = str(cfg["dataset"])
        term = str(cfg["term"])
        ratio = int(cfg["ratio"])
        method = str(cfg["method"])

        # 获取 period
        period = period_map.get(dataset, 24)
        if period is None or period < 2:
            continue

        # 该配置的所有窗口
        sub = group[
            (group["dataset"] == dataset) &
            (group["term"] == term) &
            (group["ratio"] == ratio) &
            (group["method"] == method)
        ]
        window_indices = sorted(sub["window_idx"].unique().tolist())

        # --- 预测窗分量变化率 ---
        pred_records = compute_prediction_crs(
            model, dataset, term, ratio, method, window_indices, period,
        )
        pred_map: Dict[int, Dict] = {r["window_idx"]: r for r in pred_records}

        # --- 历史窗分量变化率 ---
        hist_map: Dict[int, Dict] = {}
        if not skip_history:
            hist_records = compute_history_crs(
                dataset, term, ratio, method, model, window_indices, period,
            )
            hist_map = {r["window_idx"]: r for r in hist_records}

        # --- SMAPE ---
        ds_key = (model, dataset, term)
        imp_key = (model, dataset, term, ratio, method)
        clean_smape_avg = clean_smape_cache.get(ds_key)
        impute_smape_avg = smape_cache.get(imp_key)
        if clean_smape_avg is None or impute_smape_avg is None:
            continue

        for w_idx in window_indices:
            pr = pred_map.get(w_idx, {})
            hr = hist_map.get(w_idx, {})

            # DeltaSMAPE: 优先用窗口级
            wk = (model, dataset, term, ratio, method, w_idx)
            cwk = (model, dataset, term, w_idx)
            if compute_window_smape and wk in smape_cache and cwk in clean_smape_cache:
                w_ds = smape_cache[wk] - clean_smape_cache[cwk]
            elif wk in smape_cache:
                w_ds = smape_cache[wk] - clean_smape_avg
            else:
                w_ds = impute_smape_avg - clean_smape_avg

            all_rows.append({
                "model": model,
                "dataset": dataset,
                "term": term,
                "ratio": ratio,
                "method": method,
                "window_idx": w_idx,
                "trend_change_rate_pred": pr.get("trend_change_rate_pred", float("nan")),
                "seasonal_change_rate_pred": pr.get("seasonal_change_rate_pred", float("nan")),
                "freq_change_rate_pred": pr.get("freq_change_rate_pred", float("nan")),
                "trend_change_rate_hist": hr.get("trend_change_rate_hist", float("nan")),
                "seasonal_change_rate_hist": hr.get("seasonal_change_rate_hist", float("nan")),
                "freq_change_rate_hist": hr.get("freq_change_rate_hist", float("nan")),
                "n_vars_hist": hr.get("n_vars_hist", 0),
                "clean_smape": clean_smape_avg,
                "impute_smape": impute_smape_avg,
                "delta_smape": w_ds,
            })

    elapsed = time.time() - t0
    print(f"{len(all_rows)} 行, {elapsed:.1f}s")
    return pd.DataFrame(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="分量变化率 vs SMAPE 变化率分析")
    parser.add_argument("--model", type=str, default=None,
                        help="只处理指定模型 (逗号分隔)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="只处理指定数据集 (逗号分隔)")
    parser.add_argument("--output_dir", type=str,
                        default="results_analysis/component_vs_smape")
    parser.add_argument("--compute_window_smape", action="store_true", default=True,
                        help="计算窗口级 SMAPE (默认启用)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="并行进程数 (默认 1)")
    parser.add_argument("--skip_history", action="store_true",
                        help="跳过历史窗处理 (调试用)")
    parser.add_argument("--overwrite", action="store_true",
                        help="覆盖已有输出")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.model.split(",")] if args.model else MODELS
    datasets_filter = set(args.dataset.split(",")) if args.dataset else None

    print("=" * 60)
    print("分量变化率 vs SMAPE 变化率分析")
    print("=" * 60)

    # Step 1: 加载所有 bias_pairs 的唯一组合
    print("\n[1/5] 加载 bias_pairs 组合...")
    combos = load_bias_pairs(BIAS_BY_MODEL_DIR, models)
    if combos.empty:
        print("  [错误] 未找到 bias_pairs 数据")
        return
    if datasets_filter:
        combos = combos[combos["dataset"].isin(datasets_filter)]
    print(f"  共 {len(combos)} 个 (model,dataset,term,ratio,method,window_idx) 组合")

    # Step 2: 加载 SMAPE
    print("\n[2/5] 加载 SMAPE 数据...")
    clean_smape_avg = load_clean_smape(RESULTS_DIR)
    impute_smape_avg = load_impute_smape(RESULTS_DIR)
    print(f"  干净 SMAPE: {len(clean_smape_avg)} 条")
    print(f"  填补 SMAPE: {len(impute_smape_avg)} 条")

    # 构建 period 映射
    period_map: Dict[str, int] = {}
    for dataset in combos["dataset"].unique():
        try:
            period_map[dataset] = get_period(dataset)
        except Exception:
            period_map[dataset] = 24
    print(f"  Period 映射: {len(period_map)} 个数据集")

    # Step 3: 窗口级 SMAPE
    window_smape: Dict = {}
    clean_window_smape: Dict = {}
    if args.compute_window_smape:
        print("\n[3/5] 计算窗口级 SMAPE...")
        print("  填补窗口 SMAPE...")
        window_smape = compute_per_window_smape(combos)
        print(f"    {len(window_smape)} 个")
        # 仅计算预窗口级干净 SMAPE (去重配置)
        clean_configs = combos[["model", "dataset", "term", "window_idx"]].drop_duplicates()
        print(f"  干净窗口 SMAPE ({len(clean_configs)} 个配置)...")
        clean_window_smape = compute_clean_per_window_smape(combos)
        print(f"    {len(clean_window_smape)} 个")

    # 合并 SMAPE 缓存
    smape_cache: Dict = {}
    smape_cache.update(impute_smape_avg)
    smape_cache.update(window_smape)  # 窗口级覆盖数据集级

    clean_cache: Dict = {}
    clean_cache.update(clean_smape_avg)
    clean_cache.update(clean_window_smape)

    # Step 4: 按模型处理
    print("\n[4/5] 处理各模型...")
    all_results: List[pd.DataFrame] = []

    if args.n_jobs > 1 and len(models) > 1:
        # 并行模式
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = []
            for model in models:
                group = combos[combos["model"] == model].copy()
                if group.empty:
                    continue
                future = executor.submit(
                    process_model_group, group, period_map,
                    smape_cache, clean_cache,
                    args.compute_window_smape, args.skip_history,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    result_df = future.result()
                    if not result_df.empty:
                        all_results.append(result_df)
                        # 保存中间结果
                        model_name = result_df["model"].iloc[0]
                        tmp_path = output_dir / f"temp_{model_name}.csv"
                        result_df.to_csv(tmp_path, index=False)
                except Exception as e:
                    print(f"  [错误] 模型处理失败: {e}")
    else:
        # 串行模式
        for model in models:
            group = combos[combos["model"] == model].copy()
            if group.empty:
                print(f"  [跳过] {model}: 无数据")
                continue
            result_df = process_model_group(
                group, period_map, smape_cache, clean_cache,
                args.compute_window_smape, args.skip_history,
            )
            if not result_df.empty:
                all_results.append(result_df)
                tmp_path = output_dir / f"temp_{model}.csv"
                result_df.to_csv(tmp_path, index=False)
                print(f"    -> {tmp_path}")

    if not all_results:
        print("  [错误] 无有效结果")
        return

    full_df = pd.concat(all_results, ignore_index=True)

    # Step 5: 保存 & 出图
    print("\n[5/5] 保存结果 & 生成图表...")

    csv_path = output_dir / "component_change_rates.csv"
    full_df.to_csv(csv_path, index=False)
    print(f"  CSV -> {csv_path} ({len(full_df)} 行)")

    plot_all_figures(full_df, output_dir)

    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
