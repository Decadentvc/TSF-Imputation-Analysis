from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHODS = ("forward", "linear", "kalman_arima")
TIME_COLUMNS = ("date", "time", "timestamp")

METHOD_LABELS = {
    "forward": "Forward",
    "linear": "Linear",
    "kalman_arima": "Kalman-ARIMA",
}

METHOD_COLORS = {
    "forward": "#0072B2",
    "linear": "#E69F00",
    "kalman_arima": "#009E73",
}

METHOD_LINESTYLES = {
    "forward": (0, (5, 2)),
    "linear": "solid",
    "kalman_arima": "solid",
}

METHOD_MARKERS = {
    "forward": "s",
    "linear": "o",
    "kalman_arima": "^",
}

FORECAST_COLORS = {
    "ground_truth": "#111111",
    "clean": "#6A3D9A",
    **METHOD_COLORS,
}


@dataclass(frozen=True)
class SeriesData:
    path: Path
    timestamp: pd.Series
    values: pd.Series
    value_col: str


@dataclass(frozen=True)
class PredictionSet:
    timestamp: pd.Series
    prediction: pd.Series
    path: Path


@dataclass(frozen=True)
class Candidate:
    score: float
    model: str
    dataset: str
    ratio: str
    term: str
    block_length: int
    window_idx: int
    block_start: int
    block_end: int
    value_col: str
    prediction_start_idx: int
    imputation_errors: dict[str, float]
    forecast_errors: dict[str, float]
    clean_forecast_error: float
    imputation_ranks: dict[str, int]
    forecast_ranks: dict[str, int]
    imputation_spread: float
    forecast_spread: float
    rank_gap: int
    kendall_distance: int


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def find_time_col(df: pd.DataFrame) -> str:
    for col in TIME_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError("No timestamp column found")


def first_value_col(df: pd.DataFrame, time_col: str) -> str:
    candidates = [c for c in df.columns if c != time_col]
    if not candidates:
        raise ValueError("No value column found")
    if "OT" in candidates:
        return "OT"
    numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]
    return numeric[0] if numeric else candidates[0]


def load_series(path: Path) -> SeriesData:
    path = _repo_path(path)
    df = pd.read_csv(path)
    time_col = find_time_col(df)
    value_col = first_value_col(df, time_col)
    timestamp = pd.to_datetime(df[time_col], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"Cannot parse timestamps in {path}")
    values = pd.to_numeric(df[value_col], errors="coerce")
    return SeriesData(path=path, timestamp=timestamp, values=values, value_col=value_col)


def load_prediction(path: Path) -> PredictionSet:
    path = _repo_path(path)
    df = pd.read_csv(path)
    if "date" not in df.columns or "prediction" not in df.columns:
        raise ValueError(f"Unexpected prediction schema: {path}")
    timestamp = pd.to_datetime(df["date"], errors="coerce")
    if timestamp.isna().any():
        raise ValueError(f"Cannot parse prediction timestamps in {path}")
    prediction = pd.to_numeric(df["prediction"], errors="coerce")
    return PredictionSet(timestamp=timestamp, prediction=prediction, path=path)


def prediction_file_map(path: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not path.exists():
        return out
    for file_path in path.glob("*_prediction_*.csv"):
        match = re.search(r"_prediction_(\d+)\.csv$", file_path.name)
        if match:
            out[int(match.group(1))] = file_path
    return out


def parse_bm_prediction_dir(name: str) -> tuple[str, int, str, str] | None:
    match = re.fullmatch(
        r"(?P<dataset>.+)_BM_length(?P<block>\d+)_(?P<ratio>\d{3})_(?P<term>short|medium|long)_prediction",
        name,
    )
    if not match:
        return None
    return (
        match.group("dataset"),
        int(match.group("block")),
        match.group("ratio"),
        match.group("term"),
    )


def ranks(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    return {method: rank for rank, (method, _) in enumerate(ordered, start=1)}


def kendall_distance(rank_a: dict[str, int], rank_b: dict[str, int]) -> int:
    distance = 0
    methods = list(rank_a)
    for i, left in enumerate(methods):
        for right in methods[i + 1 :]:
            if (rank_a[left] - rank_a[right]) * (rank_b[left] - rank_b[right]) < 0:
                distance += 1
    return distance


def relative_spread(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    denom = max(float(np.mean(np.abs(arr))), 1e-12)
    return float((np.max(arr) - np.min(arr)) / denom)


def mae(predicted: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> float:
    pred = np.asarray(predicted, dtype=float)
    true = np.asarray(actual, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(true)
    if not valid.any():
        return math.inf
    return float(np.mean(np.abs(pred[valid] - true[valid])))


def error_metrics(predicted: np.ndarray | pd.Series, actual: np.ndarray | pd.Series) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=float)
    true = np.asarray(actual, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(true)
    if not valid.any():
        return {
            "MAE": math.inf,
            "RMSE": math.inf,
            "MAPE": math.inf,
            "sMAPE": math.inf,
            "Bias": math.inf,
        }

    pred = pred[valid]
    true = true[valid]
    error = pred - true
    abs_error = np.abs(error)
    abs_true = np.abs(true)
    mape_mask = abs_true > 1e-12
    smape_denom = np.abs(pred) + abs_true
    smape_mask = smape_denom > 1e-12
    return {
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(abs_error[mape_mask] / abs_true[mape_mask]))
        if mape_mask.any()
        else math.inf,
        "sMAPE": float(np.mean(2.0 * abs_error[smape_mask] / smape_denom[smape_mask]))
        if smape_mask.any()
        else math.inf,
        "Bias": float(np.mean(error)),
    }


def find_nan_blocks(values: pd.Series) -> list[tuple[int, int]]:
    mask = values.isna().to_numpy()
    blocks: list[tuple[int, int]] = []
    idx = 0
    while idx < len(mask):
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < len(mask) and mask[idx]:
            idx += 1
        blocks.append((start, idx))
    return blocks


def actual_values_for_prediction(
    actual: SeriesData, prediction: PredictionSet
) -> pd.Series:
    actual_by_time = pd.Series(actual.values.to_numpy(), index=actual.timestamp)
    values = actual_by_time.reindex(prediction.timestamp)
    if values.isna().any():
        missing = int(values.isna().sum())
        raise ValueError(f"{missing} prediction timestamps are absent from clean data")
    return values.reset_index(drop=True)


def choose_forecast_view_slice(
    actual_forecast: pd.Series,
    clean_pred: PredictionSet,
    method_preds: dict[str, PredictionSet],
    methods: tuple[str, ...],
    target_method: str,
    view_points: int,
) -> slice:
    total = len(actual_forecast)
    if view_points <= 0 or view_points >= total:
        return slice(0, total)

    width = max(1, min(view_points, total))
    best_start = 0
    best_score = -math.inf

    for start in range(0, total - width + 1):
        stop = start + width
        target_error = mae(method_preds[target_method].prediction.iloc[start:stop], actual_forecast.iloc[start:stop])
        method_errors = {
            method: mae(method_preds[method].prediction.iloc[start:stop], actual_forecast.iloc[start:stop])
            for method in methods
        }
        all_errors = {
            "clean": mae(clean_pred.prediction.iloc[start:stop], actual_forecast.iloc[start:stop]),
            **method_errors,
        }
        all_ranks = ranks(all_errors)
        method_ranks = ranks(method_errors)
        other_errors = [err for key, err in all_errors.items() if key != target_method]
        next_best = min(other_errors) if other_errors else target_error
        denom = max(float(np.mean(list(all_errors.values()))), 1e-12)
        target_margin = (next_best - target_error) / denom
        spread = relative_spread(method_errors.values())
        truth_range = float(actual_forecast.iloc[start:stop].max() - actual_forecast.iloc[start:stop].min())

        score = (
            (100.0 if all_ranks[target_method] == 1 else 0.0)
            + (30.0 if method_ranks[target_method] == 1 else 0.0)
            + 40.0 * target_margin
            + 5.0 * spread
            + truth_range
        )
        if score > best_score:
            best_score = score
            best_start = start

    return slice(best_start, best_start + width)


def timestamp_to_index(actual: SeriesData, timestamp: pd.Timestamp) -> int:
    matches = np.flatnonzero(actual.timestamp.to_numpy() == timestamp.to_datetime64())
    if len(matches):
        return int(matches[0])
    ordered = actual.timestamp.to_numpy()
    return int(np.searchsorted(ordered, timestamp.to_datetime64()))


def candidate_score(
    imputation_errors: dict[str, float],
    forecast_errors: dict[str, float],
) -> tuple[float, dict[str, int], dict[str, int], float, float, int, int]:
    imp_ranks = ranks(imputation_errors)
    pred_ranks = ranks(forecast_errors)
    gap = sum(abs(imp_ranks[m] - pred_ranks[m]) for m in imputation_errors)
    kd = kendall_distance(imp_ranks, pred_ranks)
    imp_spread = relative_spread(imputation_errors.values())
    pred_spread = relative_spread(forecast_errors.values())

    best_forecast = min(forecast_errors, key=forecast_errors.get)
    best_imputation = min(imputation_errors, key=imputation_errors.get)
    rank_cross = imp_ranks[best_forecast] + pred_ranks[best_imputation]
    strong_reversal = 0.0
    for method in imputation_errors:
        if imp_ranks[method] >= 3 and pred_ranks[method] == 1:
            strong_reversal += 4.0
        if imp_ranks[method] == 1 and pred_ranks[method] >= 3:
            strong_reversal += 3.0

    score = (
        gap * 2.0
        + kd * 3.0
        + rank_cross
        + strong_reversal
        + min(8.0, imp_spread * 4.0)
        + min(8.0, pred_spread * 5.0)
    )
    return score, imp_ranks, pred_ranks, imp_spread, pred_spread, gap, kd


def load_model_contexts() -> dict[str, int]:
    path = REPO_ROOT / "Eval" / "model_properties.json"
    with open(path, "r", encoding="utf-8") as f:
        props = json.load(f)
    return {model: int(meta["max_context"]) for model, meta in props.items()}


def iter_prediction_dirs(models: set[str] | None) -> Iterable[tuple[str, Path]]:
    root = REPO_ROOT / "data" / "Intermediate_Predictions"
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name.lower()
        if models and model not in models:
            continue
        for pred_dir in sorted(model_dir.glob("*_BM_length*_prediction")):
            yield model, pred_dir


def build_candidates(
    methods: tuple[str, ...],
    models: set[str] | None,
    min_forecast_points: int,
    max_forecast_points: int | None,
    target_method: str,
    require_target_worst_imputation: bool,
    require_target_best_forecast: bool,
) -> list[Candidate]:
    model_contexts = load_model_contexts()
    candidates: list[Candidate] = []

    for model, pred_dir in iter_prediction_dirs(models):
        parsed = parse_bm_prediction_dir(pred_dir.name)
        if parsed is None:
            continue
        dataset, block_length, ratio, term = parsed
        method_dirs = {method: pred_dir / method for method in methods}
        if not all(path.exists() for path in method_dirs.values()):
            continue

        clean_dir = pred_dir.parent / f"{dataset}_clean_{term}_prediction"
        clean_files = prediction_file_map(clean_dir)
        method_files = {method: prediction_file_map(path) for method, path in method_dirs.items()}
        common_windows = set(clean_files)
        for files in method_files.values():
            common_windows &= set(files)
        if not common_windows:
            continue

        clean_path = REPO_ROOT / "data" / "datasets" / "ori" / f"{dataset}.csv"
        bm_path = (
            REPO_ROOT
            / "data"
            / "datasets"
            / "BM"
            / f"BM_{ratio}"
            / f"{dataset}_BM_length{block_length}_{ratio}_{term}.csv"
        )
        imputed_paths = {
            method: REPO_ROOT
            / "data"
            / "datasets"
            / "Imputed"
            / "BM"
            / f"BM_{ratio}"
            / f"{dataset}_BM_{ratio}_{term}_{method}.csv"
            for method in methods
        }
        if not clean_path.exists() or not bm_path.exists():
            continue
        if not all(path.exists() for path in imputed_paths.values()):
            continue

        try:
            actual = load_series(clean_path)
            bm = load_series(bm_path)
            imputed = {method: load_series(path) for method, path in imputed_paths.items()}
        except Exception:
            continue

        if any(len(data.values) != len(actual.values) for data in [bm, *imputed.values()]):
            continue

        blocks = find_nan_blocks(bm.values)
        if not blocks:
            continue

        context_length = model_contexts.get(model, 8192)
        actual_index = pd.Index(actual.timestamp)

        for window_idx in sorted(common_windows):
            try:
                clean_pred = load_prediction(clean_files[window_idx])
                if len(clean_pred.prediction) < min_forecast_points:
                    continue
                if max_forecast_points is not None and len(clean_pred.prediction) > max_forecast_points:
                    continue
                method_preds = {
                    method: load_prediction(method_files[method][window_idx])
                    for method in methods
                }
                actual_forecast = actual_values_for_prediction(actual, clean_pred)
            except Exception:
                continue

            forecast_errors = {
                method: mae(pred.prediction, actual_forecast)
                for method, pred in method_preds.items()
            }
            if any(not np.isfinite(v) for v in forecast_errors.values()):
                continue
            clean_error = mae(clean_pred.prediction, actual_forecast)
            prediction_start_idx = timestamp_to_index(actual, clean_pred.timestamp.iloc[0])
            context_start = max(0, prediction_start_idx - context_length)
            context_end = prediction_start_idx

            for block_start, block_end in blocks:
                if block_end <= context_start or block_start >= context_end:
                    continue
                eval_start = max(block_start, context_start)
                eval_end = min(block_end, context_end)
                if eval_end - eval_start < max(5, min(block_length, 50) // 2):
                    continue

                actual_block = actual.values.iloc[eval_start:eval_end]
                imp_errors = {
                    method: mae(imputed[method].values.iloc[eval_start:eval_end], actual_block)
                    for method in methods
                }
                if any(not np.isfinite(v) for v in imp_errors.values()):
                    continue

                (
                    score,
                    imp_ranks,
                    pred_ranks,
                    imp_spread,
                    pred_spread,
                    gap,
                    kd,
                ) = candidate_score(imp_errors, forecast_errors)
                if require_target_best_forecast and pred_ranks.get(target_method) != 1:
                    continue
                if require_target_worst_imputation and imp_ranks.get(target_method) != len(methods):
                    continue

                candidates.append(
                    Candidate(
                        score=score,
                        model=model,
                        dataset=dataset,
                        ratio=ratio,
                        term=term,
                        block_length=block_length,
                        window_idx=window_idx,
                        block_start=block_start,
                        block_end=block_end,
                        value_col=actual.value_col,
                        prediction_start_idx=prediction_start_idx,
                        imputation_errors=imp_errors,
                        forecast_errors=forecast_errors,
                        clean_forecast_error=clean_error,
                        imputation_ranks=imp_ranks,
                        forecast_ranks=pred_ranks,
                        imputation_spread=imp_spread,
                        forecast_spread=pred_spread,
                        rank_gap=gap,
                        kendall_distance=kd,
                    )
                )

    return sorted(candidates, key=lambda c: c.score, reverse=True)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 450,
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "legend.fontsize": 14,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.linewidth": 1.8,
            "lines.linewidth": 3.2,
            "font.family": "DejaVu Sans",
        }
    )


def set_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.tick_params(axis="both", which="major", width=1.6, length=6, pad=4)


def plot_imputation(candidate: Candidate, methods: tuple[str, ...], output_path: Path) -> None:
    clean_path = REPO_ROOT / "data" / "datasets" / "ori" / f"{candidate.dataset}.csv"
    bm_path = (
        REPO_ROOT
        / "data"
        / "datasets"
        / "BM"
        / f"BM_{candidate.ratio}"
        / f"{candidate.dataset}_BM_length{candidate.block_length}_{candidate.ratio}_{candidate.term}.csv"
    )
    imputed_paths = {
        method: REPO_ROOT
        / "data"
        / "datasets"
        / "Imputed"
        / "BM"
        / f"BM_{candidate.ratio}"
        / f"{candidate.dataset}_BM_{candidate.ratio}_{candidate.term}_{method}.csv"
        for method in methods
    }
    actual = load_series(clean_path)
    bm = load_series(bm_path)
    imputed = {method: load_series(path) for method, path in imputed_paths.items()}

    block_len = candidate.block_end - candidate.block_start
    min_view_points = 115
    margin = max(block_len, math.ceil((min_view_points - block_len) / 2))
    view_start = max(0, candidate.block_start - margin)
    view_end = min(len(actual.values), candidate.block_end + margin)
    block_slice = slice(candidate.block_start, candidate.block_end)
    view_slice = slice(view_start, view_end)

    fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.34)
    ax.axvspan(
        actual.timestamp.iloc[candidate.block_start],
        actual.timestamp.iloc[candidate.block_end - 1],
        color="#F4A6A6",
        alpha=0.25,
        label="Missing block",
        zorder=0,
    )
    ax.plot(
        actual.timestamp.iloc[view_slice],
        actual.values.iloc[view_slice],
        color="#111111",
        linewidth=2.4,
        label="Ground truth",
        zorder=3,
    )

    obs = bm.values.iloc[view_slice].notna().to_numpy()
    obs_idx = np.flatnonzero(obs)
    obs_idx = obs_idx[::2] if len(obs_idx) > 45 else obs_idx
    ax.scatter(
        actual.timestamp.iloc[view_slice].iloc[obs_idx],
        bm.values.iloc[view_slice].iloc[obs_idx],
        s=58,
        color="#7A7A7A",
        alpha=0.75,
        label="Observed input",
        zorder=4,
    )

    for method in methods:
        ax.plot(
            actual.timestamp.iloc[block_slice],
            imputed[method].values.iloc[block_slice],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.0,
            marker=METHOD_MARKERS[method],
            markersize=7.0,
            markevery=max(1, block_len // 8),
            label=METHOD_LABELS[method],
            zorder=5,
        )

    ax.set_title("Imputation", pad=10)
    ax.set_xlabel("Timestamp", labelpad=10)
    ax.set_ylabel(candidate.value_col)
    ax.grid(False)
    set_date_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=True,
        framealpha=0.94,
        ncols=3,
        borderaxespad=0.0,
        handlelength=2.5,
        columnspacing=1.3,
        labelspacing=0.35,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_forecast(
    candidate: Candidate,
    methods: tuple[str, ...],
    output_path: Path,
    target_method: str,
    forecast_view_points: int,
) -> slice:
    pred_base = (
        REPO_ROOT
        / "data"
        / "Intermediate_Predictions"
        / candidate.model
        / f"{candidate.dataset}_BM_length{candidate.block_length}_{candidate.ratio}_{candidate.term}_prediction"
    )
    clean_pred_path = (
        REPO_ROOT
        / "data"
        / "Intermediate_Predictions"
        / candidate.model
        / f"{candidate.dataset}_clean_{candidate.term}_prediction"
        / f"{candidate.dataset}_clean_{candidate.term}_prediction_{candidate.window_idx}.csv"
    )
    method_pred_paths = {
        method: pred_base
        / method
        / f"{candidate.dataset}_BM_length{candidate.block_length}_{candidate.ratio}_{candidate.term}_prediction_{candidate.window_idx}.csv"
        for method in methods
    }
    actual = load_series(REPO_ROOT / "data" / "datasets" / "ori" / f"{candidate.dataset}.csv")
    clean_pred = load_prediction(clean_pred_path)
    actual_forecast = actual_values_for_prediction(actual, clean_pred)
    method_preds = {method: load_prediction(method_pred_paths[method]) for method in methods}
    view_slice = choose_forecast_view_slice(
        actual_forecast=actual_forecast,
        clean_pred=clean_pred,
        method_preds=method_preds,
        methods=methods,
        target_method=target_method,
        view_points=forecast_view_points,
    )
    shown_len = len(actual_forecast.iloc[view_slice])

    fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.34)
    markevery = max(1, shown_len // 14)
    ax.plot(
        clean_pred.timestamp.iloc[view_slice],
        actual_forecast.iloc[view_slice],
        color=FORECAST_COLORS["ground_truth"],
        linewidth=2.5,
        marker="o",
        markersize=6.8,
        markevery=markevery,
        label="Ground truth",
        zorder=5,
    )
    ax.plot(
        clean_pred.timestamp.iloc[view_slice],
        clean_pred.prediction.iloc[view_slice],
        color=FORECAST_COLORS["clean"],
        linewidth=2.1,
        linestyle=(0, (5, 2)),
        marker="s",
        markersize=6.5,
        markevery=markevery,
        label="Clean input",
        zorder=4,
    )

    for method in methods:
        pred = method_preds[method]
        ax.plot(
            pred.timestamp.iloc[view_slice],
            pred.prediction.iloc[view_slice],
            color=FORECAST_COLORS[method],
            linewidth=2.0,
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=6.6,
            markevery=markevery,
            label=METHOD_LABELS[method],
            zorder=3,
        )

    ax.set_title("Forecast", pad=10)
    ax.set_xlabel("Timestamp", labelpad=10)
    ax.set_ylabel(candidate.value_col)
    ax.grid(False)
    set_date_axis(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        frameon=True,
        framealpha=0.94,
        ncols=3,
        borderaxespad=0.0,
        handlelength=2.5,
        columnspacing=1.3,
        labelspacing=0.35,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return view_slice


def sample_paths(candidate: Candidate, methods: tuple[str, ...]) -> dict[str, Path | dict[str, Path]]:
    pred_base = (
        REPO_ROOT
        / "data"
        / "Intermediate_Predictions"
        / candidate.model
        / f"{candidate.dataset}_BM_length{candidate.block_length}_{candidate.ratio}_{candidate.term}_prediction"
    )
    return {
        "clean": REPO_ROOT / "data" / "datasets" / "ori" / f"{candidate.dataset}.csv",
        "bm": REPO_ROOT
        / "data"
        / "datasets"
        / "BM"
        / f"BM_{candidate.ratio}"
        / f"{candidate.dataset}_BM_length{candidate.block_length}_{candidate.ratio}_{candidate.term}.csv",
        "imputed": {
            method: REPO_ROOT
            / "data"
            / "datasets"
            / "Imputed"
            / "BM"
            / f"BM_{candidate.ratio}"
            / f"{candidate.dataset}_BM_{candidate.ratio}_{candidate.term}_{method}.csv"
            for method in methods
        },
        "clean_prediction": REPO_ROOT
        / "data"
        / "Intermediate_Predictions"
        / candidate.model
        / f"{candidate.dataset}_clean_{candidate.term}_prediction"
        / f"{candidate.dataset}_clean_{candidate.term}_prediction_{candidate.window_idx}.csv",
        "method_predictions": {
            method: pred_base
            / method
            / f"{candidate.dataset}_BM_length{candidate.block_length}_{candidate.ratio}_{candidate.term}_prediction_{candidate.window_idx}.csv"
            for method in methods
        },
    }


def write_metric_tables(
    candidate: Candidate,
    methods: tuple[str, ...],
    output_dir: Path,
    target_method: str,
    forecast_view_points: int,
) -> tuple[Path, Path, Path]:
    paths = sample_paths(candidate, methods)
    actual = load_series(paths["clean"])  # type: ignore[arg-type]

    imputed_paths = paths["imputed"]  # type: ignore[assignment]
    imputation_rows = []
    block_slice = slice(candidate.block_start, candidate.block_end)
    actual_block = actual.values.iloc[block_slice]
    for method in methods:
        imputed = load_series(imputed_paths[method])  # type: ignore[index]
        row = {
            "method": METHOD_LABELS[method],
            "method_key": method,
            "model": candidate.model,
            "dataset": candidate.dataset,
            "ratio": candidate.ratio,
            "term": candidate.term,
            "window_idx": candidate.window_idx,
            "block_start": candidate.block_start,
            "block_end": candidate.block_end,
            "block_start_timestamp": actual.timestamp.iloc[candidate.block_start],
            "block_end_timestamp": actual.timestamp.iloc[candidate.block_end - 1],
            **error_metrics(imputed.values.iloc[block_slice], actual_block),
        }
        imputation_rows.append(row)

    imputation_df = pd.DataFrame(imputation_rows)
    imputation_df["MAE_rank"] = imputation_df["MAE"].rank(method="min", ascending=True).astype(int)
    imputation_df = imputation_df.sort_values(["MAE_rank", "method"])

    clean_pred = load_prediction(paths["clean_prediction"])  # type: ignore[arg-type]
    actual_forecast = actual_values_for_prediction(actual, clean_pred)
    prediction_paths = paths["method_predictions"]  # type: ignore[assignment]
    method_predictions = {
        method: load_prediction(prediction_paths[method])  # type: ignore[index]
        for method in methods
    }
    view_slice = choose_forecast_view_slice(
        actual_forecast=actual_forecast,
        clean_pred=clean_pred,
        method_preds=method_predictions,
        methods=methods,
        target_method=target_method,
        view_points=forecast_view_points,
    )
    view_start = view_slice.start or 0
    view_stop = view_slice.stop or len(actual_forecast)
    shown_actual = actual_forecast.iloc[view_slice]
    forecast_rows = [
        {
            "series": "Clean input",
            "series_key": "clean",
            "model": candidate.model,
            "dataset": candidate.dataset,
            "ratio": candidate.ratio,
            "term": candidate.term,
            "window_idx": candidate.window_idx,
            "forecast_view_start_idx": view_start,
            "forecast_view_end_idx": view_stop,
            "forecast_start_timestamp": clean_pred.timestamp.iloc[view_start],
            "forecast_end_timestamp": clean_pred.timestamp.iloc[view_stop - 1],
            **error_metrics(clean_pred.prediction.iloc[view_slice], shown_actual),
        }
    ]
    point_df = pd.DataFrame(
        {
            "timestamp": clean_pred.timestamp.iloc[view_slice].reset_index(drop=True),
            "ground_truth": shown_actual.reset_index(drop=True),
            "clean_input": clean_pred.prediction.iloc[view_slice].reset_index(drop=True),
        }
    )

    for method in methods:
        pred = method_predictions[method]
        point_df[method] = pred.prediction.iloc[view_slice].reset_index(drop=True)
        forecast_rows.append(
            {
                "series": METHOD_LABELS[method],
                "series_key": method,
                "model": candidate.model,
                "dataset": candidate.dataset,
                "ratio": candidate.ratio,
                "term": candidate.term,
                "window_idx": candidate.window_idx,
                "forecast_view_start_idx": view_start,
                "forecast_view_end_idx": view_stop,
                "forecast_start_timestamp": pred.timestamp.iloc[view_start],
                "forecast_end_timestamp": pred.timestamp.iloc[view_stop - 1],
                **error_metrics(pred.prediction.iloc[view_slice], shown_actual),
            }
        )

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_df["MAE_rank"] = forecast_df["MAE"].rank(method="min", ascending=True).astype(int)
    forecast_df = forecast_df.sort_values(["MAE_rank", "series"])

    imputation_path = output_dir / "sample_forward_linear_kalman_compact70_thin_mask_metrics.csv"
    forecast_path = output_dir / "sample_forward_linear_kalman_compact70_thin_forecast_metrics.csv"
    points_path = output_dir / "sample_forward_linear_kalman_compact70_thin_forecast_points.csv"
    imputation_df.to_csv(imputation_path, index=False)
    forecast_df.to_csv(forecast_path, index=False)
    point_df.to_csv(points_path, index=False)
    return imputation_path, forecast_path, points_path


def format_table(df: pd.DataFrame, name_col: str) -> str:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            f"  {row[name_col]}: MAE={row['MAE']:.6g}, RMSE={row['RMSE']:.6g}, "
            f"MAPE={row['MAPE']:.6g}, sMAPE={row['sMAPE']:.6g}, "
            f"Bias={row['Bias']:.6g}, MAE rank={int(row['MAE_rank'])}"
        )
    return "\n".join(rows)


def format_metrics(candidate: Candidate, methods: tuple[str, ...]) -> str:
    lines = [
        f"Selected sample: model={candidate.model}, dataset={candidate.dataset}, "
        f"ratio={candidate.ratio}, term={candidate.term}, window={candidate.window_idx}, "
        f"block=[{candidate.block_start}, {candidate.block_end})",
        f"score={candidate.score:.3f}, rank_gap={candidate.rank_gap}, "
        f"kendall_distance={candidate.kendall_distance}, "
        f"imputation_spread={candidate.imputation_spread:.3f}, "
        f"forecast_spread={candidate.forecast_spread:.3f}",
        "Imputation MAE and ranks:",
    ]
    for method in methods:
        lines.append(
            f"  {METHOD_LABELS[method]}: "
            f"{candidate.imputation_errors[method]:.6g} "
            f"(rank {candidate.imputation_ranks[method]})"
        )
    lines.append("Forecast MAE and ranks:")
    lines.append(f"  Clean input: {candidate.clean_forecast_error:.6g}")
    for method in methods:
        lines.append(
            f"  {METHOD_LABELS[method]}: "
            f"{candidate.forecast_errors[method]:.6g} "
            f"(rank {candidate.forecast_ranks[method]})"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and plot one sample showing imputation and forecast ranking mismatch."
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated imputation methods to compare.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model names. By default all available models are scanned.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of selected candidates to print.",
    )
    parser.add_argument(
        "--max-forecast-points",
        type=int,
        default=360,
        help="Skip candidates whose forecast window has more points than this. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-forecast-points",
        type=int,
        default=180,
        help="Skip candidates whose forecast window has fewer points than this.",
    )
    parser.add_argument(
        "--forecast-view-points",
        type=int,
        default=70,
        help="Number of forecast points to display and use for exported forecast metrics.",
    )
    parser.add_argument(
        "--target-method",
        default="kalman_arima",
        help="Method expected to have poor imputation but best forecast accuracy.",
    )
    parser.add_argument(
        "--allow-target-not-worst-imputation",
        action="store_true",
        help="Allow the target method to be non-worst in mask-only imputation MAE.",
    )
    parser.add_argument(
        "--allow-target-not-best-forecast",
        action="store_true",
        help="Allow the target method to be non-best in forecast-window MAE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = tuple(m.strip().lower() for m in args.methods.split(",") if m.strip())
    unknown = [m for m in methods if m not in METHOD_LABELS]
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")

    models = None
    if args.models:
        models = {m.strip().lower() for m in args.models.split(",") if m.strip()}

    configure_matplotlib()
    if args.target_method not in methods:
        raise ValueError(f"target method must be one of the selected methods: {methods}")

    max_forecast_points = args.max_forecast_points if args.max_forecast_points > 0 else None
    candidates = build_candidates(
        methods=methods,
        models=models,
        min_forecast_points=max(0, args.min_forecast_points),
        max_forecast_points=max_forecast_points,
        target_method=args.target_method,
        require_target_worst_imputation=not args.allow_target_not_worst_imputation,
        require_target_best_forecast=not args.allow_target_not_best_forecast,
    )
    if not candidates:
        raise RuntimeError("No candidate sample found.")

    selected = candidates[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    imputation_path = args.output_dir / "sample_forward_linear_kalman_compact70_thin_impute.png"
    forecast_path = args.output_dir / "sample_forward_linear_kalman_compact70_thin_forecast.png"
    plot_imputation(selected, methods, imputation_path)
    plot_forecast(
        selected,
        methods,
        forecast_path,
        target_method=args.target_method,
        forecast_view_points=max(1, args.forecast_view_points),
    )
    mask_metrics_path, forecast_metrics_path, forecast_points_path = write_metric_tables(
        selected,
        methods,
        args.output_dir,
        target_method=args.target_method,
        forecast_view_points=max(1, args.forecast_view_points),
    )

    print(format_metrics(selected, methods))
    mask_df = pd.read_csv(mask_metrics_path)
    forecast_df = pd.read_csv(forecast_metrics_path)
    print("\nMask-only imputation metrics:")
    print(format_table(mask_df, "method"))
    print("\nDisplayed forecast-segment metrics:")
    print(format_table(forecast_df, "series"))
    print("\nTop candidates:")
    for idx, candidate in enumerate(candidates[: max(1, args.top_k)], start=1):
        print(
            f"{idx}. score={candidate.score:.3f}, model={candidate.model}, "
            f"dataset={candidate.dataset}, ratio={candidate.ratio}, term={candidate.term}, "
            f"window={candidate.window_idx}, block=[{candidate.block_start}, {candidate.block_end}), "
            f"rank_gap={candidate.rank_gap}, kendall={candidate.kendall_distance}"
        )
    print(f"\nSaved: {imputation_path}")
    print(f"Saved: {forecast_path}")
    print(f"Saved: {mask_metrics_path}")
    print(f"Saved: {forecast_metrics_path}")
    print(f"Saved: {forecast_points_path}")


if __name__ == "__main__":
    main()
