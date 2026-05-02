"""
向 results_analysis/块状缺失对长序列预测影响-实验结果统计-0501.csv 追加
4 个新填补方法（kalman_struct / kalman_arima / gp_rbf / saits）的实验结果。

策略：
- 仅处理 chronos2 / timesfm2p5 / kairos23m / kairos50m / timesfm2p0 / visiontspp
  这 6 个模型区块（sundial 因 transformers 兼容性问题暂未跑出新 imputer 结果）。
- 仅写 long term 的结果（CSV 现有结构本就只记录 long）。
- 在每个 (model, dataset, ratio) 现有 4 个 imputer 行（均值 / 前项 / 后向 / 线性）之后，
  追加 4 行新 imputer。
- 新 imputer 行：
    col 1 (基础模型)、col 2 (数据集)、col 3 (缺失率) 均留空（沿用现有重复列省略风格）
    col 4 (填补方法) = imputer 名（保持英文原名以与 results 文件名一致）
    col 5 (填补误差 MSE) = 空
    col 6 (预测误差 MSE[0.5]) = 从 results/<model>/impute/<imp>_<eval>_<term>_results.csv 读取
    col 7 (预测误差 sMAPE[0.5]) = 同上
    col 8..13 (STL 6 项) = 从 results_analysis/<model>/history/<...>_history_summary.json
                             里 summary.mean.* 取值；若 summary 文件尚未生成（Analysis 仍在跑）则留空
    col 14 = 空（保持 trailing comma 风格）

支持 dry-run（默认）：只输出预览不修改文件。`--write` 才真写。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = (
    REPO_ROOT
    / "results_analysis"
    / "块状缺失对长序列预测影响-实验结果统计-0501.csv"
)

TARGET_MODELS = [
    "chronos2",
    "timesfm2p5",
    "kairos23m",
    "kairos50m",
    "timesfm2p0",
    "visiontspp",
]
NEW_IMPUTERS = ["kalman_struct", "kalman_arima", "gp_rbf", "saits"]
OLD_LAST_IMPUTER_LABEL = "线性"  # 现有 4 个 imputer 中最后一个，新行紧随其后插入

# CSV 中标签到结果文件名 imputer 字段的映射（仅用于解析旧行；新行直接用英文名）
LABEL_TO_IMPUTER = {
    "均值": "mean",
    "前项": "forward",
    "后向": "backward",
    "线性": "linear",
}

RATIO_FLOAT_TO_3D = {0.1: "010", 0.2: "020", 0.3: "030"}

STL_KEYS = [
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "spectral_entropy",
]


def split_csv_line(line: str) -> List[str]:
    return line.rstrip("\r\n").split(",")


def join_csv_line(fields: List[str]) -> str:
    return ",".join(fields) + "\n"


def parse_metrics_csv(path: Path) -> Tuple[Optional[str], Optional[str]]:
    if not path.exists():
        return None, None
    mse = smape = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(",")
            if len(parts) >= 2:
                key, value = parts[0], parts[1]
                if key == "MSE[0.5]":
                    mse = value
                elif key == "sMAPE[0.5]":
                    smape = value
    return mse, smape


def parse_stl_summary(path: Path) -> Optional[List[str]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mean = data.get("summary", {}).get("mean", {})
    if not all(k in mean for k in STL_KEYS):
        return None
    return [str(mean[k]) for k in STL_KEYS]


def find_block_length(eval_path_dir: Path, dataset: str, ratio: int, term: str) -> Optional[int]:
    """从 BM_<ratio> 目录中匹配 length50 等。"""

    pattern = f"{dataset}_BM_length*_{ratio:03d}_{term}.csv"
    matches = sorted(eval_path_dir.glob(pattern))
    if not matches:
        return None
    m = re.search(r"_BM_length(\d+)_", matches[0].name)
    if not m:
        return None
    return int(m.group(1))


def build_new_row(
    model: str,
    dataset: str,
    ratio_float: float,
    term: str,
    imputer: str,
    keep_dataset_col: bool,
    keep_ratio_col: bool,
    repo_root: Path,
    has_trailing_comma: bool,
) -> Optional[List[str]]:
    ratio_3d = RATIO_FLOAT_TO_3D[ratio_float]
    bm_dir = repo_root / "data" / "datasets" / "BM" / f"BM_{ratio_3d}"
    block_length = find_block_length(bm_dir, dataset, int(ratio_3d), term)
    if block_length is None:
        return None

    eval_name = f"{dataset}_BM_length{block_length}_{ratio_3d}_{term}"
    result_path = (
        repo_root
        / "results"
        / model
        / "impute"
        / f"{imputer}_{eval_name}_results.csv"
    )
    mse, smape = parse_metrics_csv(result_path)
    if mse is None or smape is None:
        return None

    history_path = (
        repo_root
        / "results_analysis"
        / model
        / "history"
        / f"{dataset}_BM_{ratio_3d}_{term}_{imputer}_history_summary.json"
    )
    stl_values = parse_stl_summary(history_path)
    if stl_values is None:
        stl_values = ["", "", "", "", "", ""]

    fields = [
        "",  # col 1 model（不重复）
        "",  # col 2 dataset（不重复，因为这些新 imputer 行不是块内首行）
        "",  # col 3 ratio（不重复）
        imputer,  # col 4 填补方法（英文）
        "",  # col 5 填补误差 MSE（保持空）
        mse,  # col 6 预测误差 MSE[0.5]
        smape,  # col 7 预测误差 sMAPE[0.5]
        *stl_values,  # col 8..13
    ]
    if has_trailing_comma:
        fields.append("")  # col 14 空（trailing comma）
    return fields


# 块内行结构识别
DATASET_LINE_RE = re.compile(rf"^({'|'.join(TARGET_MODELS)})," , re.IGNORECASE)


def detect_dataset_in_line(line: str) -> Optional[Tuple[str, str]]:
    """返回 (model, dataset) 如果该行是某 (model, dataset) 块的起始行。"""

    fields = split_csv_line(line)
    if len(fields) < 4:
        return None
    if fields[0] not in TARGET_MODELS:
        return None
    if not fields[1]:
        return None
    if fields[2] != "0":
        return None
    return fields[0], fields[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="实际写入 CSV，否则仅 dry-run")
    ap.add_argument(
        "--csv",
        type=str,
        default=str(CSV_PATH),
        help="目标 CSV（默认根 CSV）",
    )
    ap.add_argument(
        "--include-empty-stl",
        action="store_true",
        help="即使 history summary 不存在也写新行（STL 6 列为空）",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    repo_root = REPO_ROOT
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    out_lines: List[str] = []
    inserted_count = 0
    skipped_count = 0
    no_history_count = 0

    cur_model: Optional[str] = None
    cur_dataset: Optional[str] = None
    cur_ratio: Optional[float] = None
    has_trailing_comma_for_block: bool = True  # chronos2 等都带 trailing comma

    i = 0
    while i < len(lines):
        line = lines[i]
        out_lines.append(line)
        fields = split_csv_line(line)

        # 检测 (model, dataset) 块首行（ratio=0 的 clean 行）
        if (
            len(fields) >= 4
            and fields[0] in TARGET_MODELS
            and fields[1]
            and fields[2] == "0"
        ):
            cur_model = fields[0]
            cur_dataset = fields[1]
            cur_ratio = None
            has_trailing_comma_for_block = (
                len(fields) >= 14 and fields[-1] == ""
            )

        # 检测 ratio 起始行（imputer = mean 的第一行）
        if (
            len(fields) >= 4
            and not fields[0]
            and not fields[1]
            and fields[2] not in ("", "0")
            and cur_model is not None
            and cur_dataset is not None
        ):
            try:
                cur_ratio = float(fields[2])
            except ValueError:
                cur_ratio = None

        # 当前行是某 (model, dataset, ratio) 的"线性"行 → 接着插入 4 个新行
        if (
            cur_model is not None
            and cur_dataset is not None
            and cur_ratio in RATIO_FLOAT_TO_3D
            and len(fields) >= 4
            and fields[3].strip() == OLD_LAST_IMPUTER_LABEL
        ):
            for imputer in NEW_IMPUTERS:
                new_fields = build_new_row(
                    model=cur_model,
                    dataset=cur_dataset,
                    ratio_float=cur_ratio,
                    term="long",
                    imputer=imputer,
                    keep_dataset_col=False,
                    keep_ratio_col=False,
                    repo_root=repo_root,
                    has_trailing_comma=has_trailing_comma_for_block,
                )
                if new_fields is None:
                    skipped_count += 1
                    continue
                # 检查 STL 是否填充
                stl_filled = any(
                    new_fields[7 + k] not in ("", None) for k in range(6)
                )
                if not stl_filled:
                    no_history_count += 1
                if (not stl_filled) and (not args.include_empty_stl):
                    # STL 全空时，仍然写入 MSE/sMAPE，方便用户看到结果
                    pass
                out_lines.append(join_csv_line(new_fields))
                inserted_count += 1

        i += 1

    print(
        f"插入新行 {inserted_count} 条；跳过（无 results）{skipped_count} 条；"
        f"其中 STL 列暂为空（无 history summary）{no_history_count} 条。"
    )

    if args.write:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            f.writelines(out_lines)
        print(f"已写回 CSV: {csv_path}")
    else:
        print("Dry-run 模式：未写入。新增 CSV 总长 ->", len(out_lines), "行")


if __name__ == "__main__":
    main()
