"""
Visualize window-level distribution gap and prediction accuracy gap.

Input:
- --model: forecasting model name, e.g. chronos2
- --prediction_mode: clean/backward/forward/linear/mean/...

Output:
- Scatter plot image where:
  x-axis: distribution gap between history and prediction windows
  y-axis: prediction accuracy (sMAPE[0.5]) or accuracy gap vs clean
  each point: one sliding window
  color: dataset identity
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


TIME_COLS = {"date", "time", "timestamp", "datetime", "index"}
METRIC_COLS = [
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "spectral_entropy",
]
MISSING_METHODS = ("BM", "MCAR", "TM", "TVMR")
SUPPORTED_MODES = {
    "clean",
    "zero",
    "mean",
    "forward",
    "backward",
    "linear",
    "nearest",
    "spline",
    "seasonal",
}

TEST_SPLIT = 0.6
MAX_WINDOW = 20
PRED_LENGTH_MAP = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}
TERM_MULTIPLIER = {"short": 1, "medium": 10, "long": 15}


@dataclass(frozen=True)
class FileMeta:
    dataset: str
    term: str
    kind: str
    missing_method: Optional[str] = None
    ratio_code: Optional[str] = None
    imputation_method: Optional[str] = None


def parse_analysis_filename(filename: str) -> Optional[FileMeta]:
    clean_pattern = re.compile(
        r"^(?P<dataset>.+)_clean_(?P<term>short|medium|long)_(?P<kind>history|prediction)\.csv$",
        flags=re.IGNORECASE,
    )
    m = clean_pattern.match(filename)
    if m:
        return FileMeta(
            dataset=m.group("dataset"),
            term=m.group("term").lower(),
            kind=m.group("kind").lower(),
            imputation_method="clean",
        )

    missing_methods = "|".join(MISSING_METHODS)
    impute_pattern = re.compile(
        rf"^(?P<dataset>.+)_(?P<method>{missing_methods})_(?P<ratio>\d{{3}})_(?P<term>short|medium|long)_(?P<impute>[A-Za-z0-9]+)_(?P<kind>history|prediction)\.csv$",
        flags=re.IGNORECASE,
    )
    m = impute_pattern.match(filename)
    if not m:
        return None

    return FileMeta(
        dataset=m.group("dataset"),
        missing_method=m.group("method").upper(),
        ratio_code=m.group("ratio"),
        term=m.group("term").lower(),
        imputation_method=m.group("impute").lower(),
        kind=m.group("kind").lower(),
    )


def read_analysis_dir(
    directory: Path,
    expected_kind: str,
    prediction_mode: str,
) -> Dict[Tuple[str, str, Optional[str], str], pd.DataFrame]:
    result: Dict[Tuple[str, str, Optional[str], str], pd.DataFrame] = {}
    if not directory.exists():
        return result

    for csv_path in directory.glob("*.csv"):
        meta = parse_analysis_filename(csv_path.name)
        if meta is None or meta.kind != expected_kind:
            continue
        if prediction_mode == "clean":
            if meta.imputation_method != "clean":
                continue
        else:
            if meta.imputation_method != prediction_mode:
                continue

        df = pd.read_csv(csv_path)
        if "window_idx" not in df.columns:
            continue

        keep_cols = ["window_idx"] + [c for c in METRIC_COLS if c in df.columns]
        df = df[keep_cols].copy()
        key = (meta.dataset, meta.term, meta.ratio_code, meta.imputation_method or "clean")
        result[key] = df

    return result


def select_target_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() not in TIME_COLS]
    if not candidates:
        raise ValueError("no numeric/value columns found")
    return candidates[0]


def normalize_freq(freq: str) -> str:
    try:
        name = to_offset(freq).name
    except Exception:
        name = freq
    s = str(name).lower()
    if "min" in s or "t" in s:
        return "T"
    if "h" in s:
        return "H"
    if "d" in s:
        return "D"
    if "w" in s:
        return "W"
    if "m" in s:
        return "M"
    if "s" in s:
        return "S"
    return "H"


def compute_pred_len_and_windows(
    dataset: str,
    term: str,
    total_len: int,
    data_path: str = "data/datasets",
) -> Tuple[int, int]:
    props_path = Path(data_path) / "dataset_properties.json"
    if not props_path.exists():
        raise FileNotFoundError(f"dataset_properties not found: {props_path}")
    with open(props_path, "r", encoding="utf-8") as f:
        props = json.load(f)
    if dataset not in props:
        raise ValueError(f"dataset not found in properties: {dataset}")

    freq = props[dataset].get("frequency", "H")
    freq_norm = normalize_freq(freq)
    pred_len = PRED_LENGTH_MAP.get(freq_norm, 48) * TERM_MULTIPLIER.get(term.lower(), 1)
    windows = min(max(1, math.ceil(TEST_SPLIT * total_len / pred_len)), MAX_WINDOW)
    return int(pred_len), int(windows)


def compute_window_smape_map(
    clean_dataset_path: Path,
    prediction_dir: Path,
    dataset: str,
    term: str,
    data_path: str = "data/datasets",
) -> Dict[int, float]:
    if not clean_dataset_path.exists() or not prediction_dir.exists():
        return {}

    clean_df = pd.read_csv(clean_dataset_path)
    target_col = select_target_column(clean_df)
    target = clean_df[target_col].to_numpy(dtype=float)

    total_len = int(len(clean_df))
    pred_len, _ = compute_pred_len_and_windows(dataset=dataset, term=term, total_len=total_len, data_path=data_path)

    smape_map: Dict[int, float] = {}
    file_pattern = re.compile(r"_(\d+)\.csv$")
    for pred_file in prediction_dir.glob("*.csv"):
        match = file_pattern.search(pred_file.name)
        if not match:
            continue
        window_idx = int(match.group(1))

        pred_df = pd.read_csv(pred_file)
        pred_col = select_target_column(pred_df)
        pred_arr = pred_df[pred_col].to_numpy(dtype=float)

        forecast_end = total_len - pred_len * window_idx
        forecast_start = forecast_end - pred_len
        if forecast_start < 0 or forecast_end > total_len:
            continue

        gt_arr = target[forecast_start:forecast_end]
        n = min(len(gt_arr), len(pred_arr))
        if n <= 0:
            continue

        y_true = gt_arr[:n]
        y_pred = pred_arr[:n]
        denom = np.abs(y_true) + np.abs(y_pred)
        ratio = np.zeros_like(y_true, dtype=float)
        valid = denom > 1e-12
        ratio[valid] = (2.0 * np.abs(y_pred[valid] - y_true[valid])) / denom[valid]
        smape = float(100.0 * np.mean(ratio))
        smape_map[window_idx] = smape

    return smape_map


def resolve_prediction_dir(
    model: str,
    dataset: str,
    term: str,
    ratio_code: Optional[str],
    prediction_mode: str,
    intermediate_root: Path,
) -> Optional[Path]:
    model_root = intermediate_root / model.lower()
    if prediction_mode == "clean":
        pred_dir = model_root / f"{dataset}_clean_{term}_prediction"
        return pred_dir if pred_dir.exists() else None

    if ratio_code is None:
        return None

    pattern = f"{dataset}_*_length*_{ratio_code}_{term}_prediction"
    for parent in model_root.glob(pattern):
        mode_dir = parent / prediction_mode
        if mode_dir.exists():
            return mode_dir
    return None


def build_plot_dataframe(
    model: str,
    prediction_mode: str,
    results_analysis_root: Path,
    intermediate_root: Path,
    data_root: Path,
) -> pd.DataFrame:
    history_map = read_analysis_dir(
        directory=results_analysis_root / model.lower() / "history",
        expected_kind="history",
        prediction_mode=prediction_mode,
    )
    prediction_map = read_analysis_dir(
        directory=results_analysis_root / model.lower() / "prediction",
        expected_kind="prediction",
        prediction_mode=prediction_mode,
    )

    common_keys = sorted(set(history_map.keys()) & set(prediction_map.keys()))
    if not common_keys:
        return pd.DataFrame()

    clean_smape_cache: Dict[Tuple[str, str], Dict[int, float]] = {}
    rows: List[Dict[str, object]] = []

    for key in common_keys:
        dataset, term, ratio_code, impute = key
        hist_df = history_map[key]
        pred_df = prediction_map[key]

        merged = hist_df.merge(
            pred_df,
            on="window_idx",
            how="inner",
            suffixes=("_history", "_prediction"),
        )
        if merged.empty:
            continue

        pred_dir = resolve_prediction_dir(
            model=model,
            dataset=dataset,
            term=term,
            ratio_code=ratio_code,
            prediction_mode=prediction_mode,
            intermediate_root=intermediate_root,
        )
        if pred_dir is None:
            continue

        clean_dataset_path = data_root / "ori" / f"{dataset}.csv"
        smape_map = compute_window_smape_map(
            clean_dataset_path=clean_dataset_path,
            prediction_dir=pred_dir,
            dataset=dataset,
            term=term,
            data_path=str(data_root),
        )
        if not smape_map:
            continue

        clean_key = (dataset, term)
        if clean_key not in clean_smape_cache:
            clean_dir = resolve_prediction_dir(
                model=model,
                dataset=dataset,
                term=term,
                ratio_code=None,
                prediction_mode="clean",
                intermediate_root=intermediate_root,
            )
            clean_smape_cache[clean_key] = (
                compute_window_smape_map(
                    clean_dataset_path=clean_dataset_path,
                    prediction_dir=clean_dir,
                    dataset=dataset,
                    term=term,
                    data_path=str(data_root),
                )
                if clean_dir is not None
                else {}
            )

        # TIME 论文中的分析强调跨指标可比性和模式向量比较。
        # 这里采用标准化欧氏距离：每个指标先按池化窗口标准差归一，再计算 6 维 RMS 距离。
        metric_scales: Dict[str, float] = {}
        for col in METRIC_COLS:
            hcol = f"{col}_history"
            pcol = f"{col}_prediction"
            if hcol in merged.columns and pcol in merged.columns:
                pooled = pd.concat([merged[hcol], merged[pcol]], ignore_index=True).astype(float)
                std = float(pooled.std(ddof=0))
                metric_scales[col] = std if std > 1e-12 else 1.0

        clean_smape_map = clean_smape_cache[clean_key]
        for _, row in merged.iterrows():
            w = int(row["window_idx"])
            if w not in smape_map:
                continue

            norm_diffs = []
            for col in METRIC_COLS:
                hcol = f"{col}_history"
                pcol = f"{col}_prediction"
                if hcol in merged.columns and pcol in merged.columns:
                    scale = metric_scales.get(col, 1.0)
                    norm_diffs.append((float(row[hcol]) - float(row[pcol])) / scale)
            if not norm_diffs:
                continue

            dist_gap = float(np.sqrt(np.mean(np.square(norm_diffs))))
            smape = float(smape_map[w])
            clean_smape = clean_smape_map.get(w)
            smape_diff = smape - clean_smape if clean_smape is not None else np.nan

            rows.append(
                {
                    "dataset": dataset,
                    "term": term,
                    "ratio_code": ratio_code,
                    "imputation_method": impute,
                    "window_idx": w,
                    "distribution_gap": dist_gap,
                    "smape": smape,
                    "smape_diff_vs_clean": smape_diff,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df


def draw_scatter(
    df: pd.DataFrame,
    model: str,
    prediction_mode: str,
    output_path: Path,
    max_points: Optional[int] = None,
    random_seed: int = 42,
) -> None:
    if df.empty:
        raise ValueError("no matched windows to plot")

    plot_df = df.copy()
    if max_points and len(plot_df) > max_points:
        plot_df = plot_df.sample(n=max_points, random_state=random_seed)

    y_col = "smape_diff_vs_clean" if prediction_mode != "clean" else "smape"
    y_label = (
        "Prediction Error Difference (sMAPE[0.5] - Clean sMAPE[0.5])"
        if prediction_mode != "clean"
        else "Prediction Error (sMAPE[0.5])"
    )

    datasets = sorted(plot_df["dataset"].unique())
    cmap = plt.get_cmap("tab20", max(1, len(datasets)))
    color_map = {ds: cmap(i) for i, ds in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(12, 8))
    for ds in datasets:
        sub = plot_df[plot_df["dataset"] == ds]
        ax.scatter(
            sub["distribution_gap"],
            sub[y_col],
            s=20,
            alpha=0.72,
            color=color_map[ds],
            label=ds,
            edgecolors="none",
        )

    ax.set_xlabel("History vs Prediction Distribution Gap (Standardized Euclidean Distance)")
    ax.set_ylabel(y_label)
    ax.set_title(f"{model} | mode={prediction_mode} | Window-level Gap vs Prediction Error")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize window-level distribution gap and prediction accuracy.")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. sundial/chronos2/timesfm2p5")
    parser.add_argument(
        "--prediction_mode",
        type=str,
        required=True,
        help="Prediction mode: clean/zero/mean/forward/backward/linear/nearest/spline/seasonal",
    )
    parser.add_argument("--results_analysis_dir", type=str, default="results_analysis")
    parser.add_argument("--intermediate_dir", type=str, default="data/Intermediate_Predictions")
    parser.add_argument("--data_dir", type=str, default="data/datasets")
    parser.add_argument("--max_points", type=int, default=1200, help="Optional subsample size for plotting")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Default: draw/outputs_by_model/{model}/{prediction_mode}_window_gap_scatter.png",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model = args.model.strip().lower()
    prediction_mode = args.prediction_mode.strip().lower()
    if prediction_mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported prediction_mode='{prediction_mode}'. Supported: {sorted(SUPPORTED_MODES)}"
        )

    results_analysis_root = Path(args.results_analysis_dir)
    intermediate_root = Path(args.intermediate_dir)
    data_root = Path(args.data_dir)
    default_out = Path("draw/outputs_by_model") / model / f"{prediction_mode}_window_gap_scatter.png"
    output_path = Path(args.output) if args.output else default_out

    df = build_plot_dataframe(
        model=model,
        prediction_mode=prediction_mode,
        results_analysis_root=results_analysis_root,
        intermediate_root=intermediate_root,
        data_root=data_root,
    )

    if df.empty:
        raise RuntimeError(
            "No valid windows were matched. Check model/mode inputs and whether corresponding analysis files exist."
        )

    draw_scatter(
        df=df,
        model=model,
        prediction_mode=prediction_mode,
        output_path=output_path,
        max_points=args.max_points,
        random_seed=args.random_seed,
    )

    csv_out = output_path.with_suffix(".csv")
    df.sort_values(["dataset", "term", "ratio_code", "window_idx"], na_position="last").to_csv(csv_out, index=False)

    print(f"[OK] plot saved: {output_path}")
    print(f"[OK] points saved: {csv_out}")
    print(f"[INFO] total points: {len(df)}")


if __name__ == "__main__":
    main()
