"""Aggregate ablation experiment CSV outputs into three summary tables (ratio / length / position)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Force UTF-8 on stdout/stderr so unicode glyphs don't crash on Windows GBK consoles
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


KNOWN_IMPUTERS = [
    "kalman_struct",
    "kalman_arima",
    "gp_rbf",
    "saits",
    "forward",
    "backward",
    "linear",
    "mean",
    "nearest",
    "polynomial",
    "spline",
    "seasonal",
    "stl_kalman",
    "zero",
]

CORE_METRICS = [
    "prediction_length",
    "windows",
    "MSE[mean]",
    "MSE[0.5]",
    "MAE[0.5]",
    "MASE[0.5]",
    "MAPE[0.5]",
    "sMAPE[0.5]",
    "MSIS",
    "RMSE[mean]",
    "NRMSE[mean]",
    "ND[0.5]",
    "mean_weighted_sum_quantile_loss",
]

RESULT_SUFFIX = "_long_results"


def _split_imputer(stem: str) -> Optional[tuple[str, str]]:
    for imp in KNOWN_IMPUTERS:
        prefix = f"{imp}_"
        if stem.startswith(prefix):
            return imp, stem[len(prefix):]
    return None


def _parse_filename(stem: str, suite: str) -> Optional[Dict[str, str]]:
    if not stem.endswith(RESULT_SUFFIX):
        return None
    body = stem[: -len(RESULT_SUFFIX)]

    parsed = _split_imputer(body)
    if parsed is None:
        return None
    imputer, rest = parsed

    if suite == "position":
        m = re.match(r"^(?P<dataset>.+)_BM_position_(?P<position>front|middle|back)_length(?P<block>\d+)_(?P<ratio>\d+)$", rest)
        if not m:
            return None
        return {
            "imputer": imputer,
            "dataset": m.group("dataset"),
            "position": m.group("position"),
            "block_length": int(m.group("block")),
            "missing_ratio": int(m.group("ratio")) / 100.0,
        }
    else:
        # ratio / length / horizon / context 套件均使用统一文件名模式：
        # {dataset}_BM_length{block}_{ratio}
        m = re.match(r"^(?P<dataset>.+)_BM_length(?P<block>\d+)_(?P<ratio>\d+)$", rest)
        if not m:
            return None
        return {
            "imputer": imputer,
            "dataset": m.group("dataset"),
            "block_length": int(m.group("block")),
            "missing_ratio": int(m.group("ratio")) / 100.0,
        }


def _read_result_csv(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path)
    if not {"metric", "value"}.issubset(df.columns):
        raise ValueError(f"Unexpected schema in {path}")
    return dict(zip(df["metric"].astype(str), df["value"].astype(str)))


def collect_suite(suite_root: Path, suite: str) -> pd.DataFrame:
    rows: List[Dict] = []
    if not suite_root.exists():
        return pd.DataFrame()

    for model_dir in sorted(suite_root.iterdir()):
        if not model_dir.is_dir():
            continue
        impute_dir = model_dir / "impute"
        if not impute_dir.is_dir():
            continue

        for csv_path in sorted(impute_dir.glob("*.csv")):
            parsed = _parse_filename(csv_path.stem, suite)
            if parsed is None:
                continue

            metrics = _read_result_csv(csv_path)
            row: Dict = {
                "model": model_dir.name,
                "suite": suite,
                **parsed,
                "source_csv": str(csv_path).replace("\\", "/"),
            }
            for key in CORE_METRICS:
                value = metrics.get(key)
                if value is None:
                    row[key] = pd.NA
                else:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = value
            rows.append(row)

    return pd.DataFrame(rows)


def collect_horizon_suite(root: Path) -> pd.DataFrame:
    """聚合 horizon_H{xxx} 多子目录到统一 DataFrame，加上 horizon 列。"""
    frames: List[pd.DataFrame] = []
    if not root.exists():
        return pd.DataFrame()
    for suite_dir in sorted(root.glob("horizon_H*")):
        if not suite_dir.is_dir():
            continue
        m = re.match(r"^horizon_H(?P<h>\d+)$", suite_dir.name)
        if not m:
            continue
        horizon_value = int(m.group("h"))
        sub = collect_suite(suite_dir, "horizon")
        if sub.empty:
            continue
        sub["horizon"] = horizon_value
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect_context_suite(root: Path) -> pd.DataFrame:
    """聚合 context_L{xxxx} 多子目录到统一 DataFrame，加上 context 列。"""
    frames: List[pd.DataFrame] = []
    if not root.exists():
        return pd.DataFrame()
    for suite_dir in sorted(root.glob("context_L*")):
        if not suite_dir.is_dir():
            continue
        m = re.match(r"^context_L(?P<c>\d+)$", suite_dir.name)
        if not m:
            continue
        context_value = int(m.group("c"))
        sub = collect_suite(suite_dir, "context")
        if sub.empty:
            continue
        sub["context"] = context_value
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_summary(df: pd.DataFrame, output_path: Path, sort_cols: List[str]) -> None:
    if df.empty:
        print(f"[warn] No rows for {output_path.name}; writing empty file")
    else:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df):>5d} rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation experiment CSV outputs")
    parser.add_argument("--root", default="results_ablation", help="Root dir of ablation results")
    parser.add_argument(
        "--output_dir",
        default="results_analysis/ablation",
        help="Output dir for aggregated CSV files",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)

    ratio_df = collect_suite(root / "ratio", "ratio")
    length_df = collect_suite(root / "length", "length")
    position_df = collect_suite(root / "position", "position")
    horizon_df = collect_horizon_suite(root)
    context_df = collect_context_suite(root)

    write_summary(
        ratio_df,
        output_dir / "ablation_ratio_summary.csv",
        ["model", "dataset", "missing_ratio", "imputer"],
    )
    write_summary(
        length_df,
        output_dir / "ablation_length_summary.csv",
        ["model", "dataset", "block_length", "imputer"],
    )
    write_summary(
        position_df,
        output_dir / "ablation_position_summary.csv",
        ["model", "dataset", "position", "imputer"],
    )
    if not horizon_df.empty:
        write_summary(
            horizon_df,
            output_dir / "ablation_horizon_summary.csv",
            ["model", "dataset", "horizon", "imputer"],
        )
    if not context_df.empty:
        write_summary(
            context_df,
            output_dir / "ablation_context_summary.csv",
            ["model", "dataset", "context", "imputer"],
        )

    print("---")
    print(f"ratio rows: {len(ratio_df)} (expected 5 models × 6 datasets × 7 ratios × 5 imputers = 1050)")
    print(f"length rows: {len(length_df)} (expected 5 models × 4 datasets × 5 lengths × 4 imputers = 400)")
    print(f"position rows: {len(position_df)} (expected 5 models × 4 datasets × 3 positions × 4 imputers = 240)")
    print(f"horizon rows: {len(horizon_df)} (expected 5 models × 6 datasets × 4 horizons × 3 imputers = 360)")
    print(f"context rows: {len(context_df)} (expected 5 models × 6 datasets × 4 contexts × 3 imputers = 360)")


if __name__ == "__main__":
    main()
