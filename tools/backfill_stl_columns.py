"""回填 results_analysis/块状缺失对长序列预测影响-实验结果统计-0501.csv 中
4 个新 imputer 行（kalman_struct/kalman_arima/gp_rbf/saits）的 STL 6 列。

原 append 脚本写入新 imputer 行时 history summary 尚未生成，col 9-14 留空。
此脚本扫描 CSV，根据当前 (model, dataset, ratio, imputer) 上下文，从
results_analysis/<model>/history/<dataset>_BM_<ratio_3d>_long_<imputer>_history_summary.json
读取 summary.mean.* 回填。

默认 dry-run；--write 真写。写前自动备份为 .bak2。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results_analysis" / "块状缺失对长序列预测影响-实验结果统计-0501.csv"

TARGET_MODELS = {
    "sundial", "chronos2", "timesfm2p5",
    "kairos23m", "kairos50m", "timesfm2p0", "visiontspp",
}
NEW_IMPUTERS = {"kalman_struct", "kalman_arima", "gp_rbf", "saits"}
RATIO_FLOAT_TO_3D = {0.1: "010", 0.2: "020", 0.3: "030"}
STL_KEYS = [
    "trend_strength", "trend_linearity",
    "seasonal_strength", "seasonal_correlation",
    "residual_autocorr_lag1", "spectral_entropy",
]


def parse_stl_summary(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    mean = data.get("summary", {}).get("mean", {})
    if not all(k in mean for k in STL_KEYS):
        return None
    return [str(mean[k]) for k in STL_KEYS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--csv", type=str, default=str(CSV_PATH))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    out_lines: List[str] = []
    cur_model: Optional[str] = None
    cur_dataset: Optional[str] = None
    cur_ratio: Optional[float] = None

    filled = 0
    already_filled = 0
    missing = 0
    by_model = {m: 0 for m in TARGET_MODELS}

    for line in lines:
        fields = line.rstrip("\r\n").split(",")

        # 块首行：model,dataset,0,/...
        if len(fields) >= 4 and fields[0] in TARGET_MODELS and fields[1] and fields[2] == "0":
            cur_model = fields[0]
            cur_dataset = fields[1]
            cur_ratio = None

        # ratio 起始行：第一个 imputer (均值)
        if (
            len(fields) >= 4
            and not fields[0]
            and not fields[1]
            and fields[2] not in ("", "0")
            and cur_model is not None
        ):
            try:
                cur_ratio = float(fields[2])
            except ValueError:
                cur_ratio = None

        # 新 imputer 行 → 回填 STL
        if (
            cur_model is not None
            and cur_dataset is not None
            and cur_ratio in RATIO_FLOAT_TO_3D
            and len(fields) >= 14
            and fields[3] in NEW_IMPUTERS
        ):
            stl_currently_empty = all(fields[8 + k] == "" for k in range(6))
            if not stl_currently_empty:
                already_filled += 1
            else:
                ratio_3d = RATIO_FLOAT_TO_3D[cur_ratio]
                summary_path = (
                    REPO_ROOT / "results_analysis" / cur_model / "history"
                    / f"{cur_dataset}_BM_{ratio_3d}_long_{fields[3]}_history_summary.json"
                )
                stl_values = parse_stl_summary(summary_path)
                if stl_values is not None:
                    for k in range(6):
                        fields[8 + k] = stl_values[k]
                    line = ",".join(fields) + "\n"
                    filled += 1
                    by_model[cur_model] += 1
                else:
                    missing += 1

        out_lines.append(line)

    print(f"回填 STL 行数: {filled}")
    print(f"已有 STL 数据（跳过）: {already_filled}")
    print(f"无 history summary（跳过）: {missing}")
    print("各模型回填数:")
    for m, n in sorted(by_model.items()):
        print(f"  {m}: {n}")

    if args.write:
        backup = csv_path.with_suffix(csv_path.suffix + ".bak2")
        shutil.copy2(csv_path, backup)
        print(f"已备份原文件: {backup}")
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            f.writelines(out_lines)
        print(f"已写回 CSV: {csv_path}")
    else:
        print("Dry-run（未写入）")


if __name__ == "__main__":
    main()
