"""BM（块缺失）缺失值注入模块。"""

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .inject_range_utils import get_injection_range, load_dataset_properties
except ImportError:
    from inject_range_utils import get_injection_range, load_dataset_properties


DEFAULT_BALANCED_CONTEXTS = [512, 2048, 2880, 4096, 8192]


def parse_missing_ratios(ratio_str: str) -> List[float]:
    ratio_str = ratio_str.strip().strip("[]")
    ratios = [float(r.strip()) for r in ratio_str.split(",") if r.strip()]
    if not ratios:
        raise ValueError("missing_ratio list is empty")
    for ratio in ratios:
        if not 0 <= ratio <= 1:
            raise ValueError(f"missing_ratio must be between 0 and 1, got {ratio}")
    return ratios


def parse_int_list(value: str) -> List[int]:
    chunks = [c.strip() for c in value.strip().strip("[]").split(",") if c.strip()]
    if not chunks:
        raise ValueError("balanced_contexts list is empty")
    contexts = [int(c) for c in chunks]
    if any(c <= 0 for c in contexts):
        raise ValueError("balanced_contexts must be positive integers")
    return sorted(set(contexts))


def _allocate_integer(total: int, weights: Sequence[float]) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        out = [0 for _ in weights]
        out[0] = total
        return out
    raw = [total * (w / weight_sum) for w in weights]
    base = [int(np.floor(v)) for v in raw]
    rem = total - sum(base)
    if rem > 0:
        order = sorted(range(len(raw)), key=lambda i: (raw[i] - base[i]), reverse=True)
        for i in order[:rem]:
            base[i] += 1
    return base


def _can_place(occupied: np.ndarray, start_rel: int, length: int, min_gap: int = 1) -> bool:
    end_rel = start_rel + length
    left = max(0, start_rel - min_gap)
    right = min(len(occupied), end_rel + min_gap)
    return not occupied[left:right].any()


def _place_block_random(
    occupied: np.ndarray,
    global_start_idx: int,
    layer_start: int,
    layer_end: int,
    block_length: int,
    rng: np.random.Generator,
    trials: int,
) -> int | None:
    max_start = layer_end - block_length
    if max_start < layer_start:
        return None

    n_candidates = max_start - layer_start + 1
    attempts = min(max(1, trials), n_candidates)
    for _ in range(attempts):
        start = int(rng.integers(layer_start, max_start + 1))
        start_rel = start - global_start_idx
        if _can_place(occupied, start_rel, block_length):
            occupied[start_rel:start_rel + block_length] = True
            return start

    if n_candidates <= 10000:
        starts = np.arange(layer_start, max_start + 1)
        rng.shuffle(starts)
        for start in starts.tolist():
            start_rel = int(start) - global_start_idx
            if _can_place(occupied, start_rel, block_length):
                occupied[start_rel:start_rel + block_length] = True
                return int(start)

    return None


def _build_stratified_ranges(
    dataset_name: str,
    term: str,
    data_path: str,
    start_idx: int,
    end_idx: int,
    balanced_contexts: Sequence[int],
) -> List[Tuple[int, int]]:
    starts: List[int] = []
    for mc in sorted(set(balanced_contexts)):
        r = get_injection_range(dataset_name=dataset_name, term=term, data_path=data_path, max_context=mc)
        s = max(start_idx, int(r["start_index"]))
        e = min(end_idx, int(r["end_index"]))
        if s < e:
            starts.append(s)

    starts = sorted(set(starts), reverse=True)
    if not starts:
        return [(start_idx, end_idx)]

    ranges: List[Tuple[int, int]] = []
    prev = end_idx
    for s in starts:
        if s < prev:
            ranges.append((s, prev))
            prev = s
    if start_idx < prev:
        ranges.append((start_idx, prev))
    return ranges


def _context_stats(
    df: pd.DataFrame,
    data_cols: Sequence[str],
    dataset_name: str,
    term: str,
    data_path: str,
    start_idx: int,
    end_idx: int,
    balanced_contexts: Sequence[int],
) -> List[dict]:
    stats = []
    for mc in sorted(set(balanced_contexts)):
        r = get_injection_range(dataset_name=dataset_name, term=term, data_path=data_path, max_context=mc)
        s = max(start_idx, int(r["start_index"]))
        e = min(end_idx, int(r["end_index"]))
        if s >= e:
            continue
        part = df.iloc[s:e][list(data_cols)]
        total_cells = len(part) * len(data_cols)
        missing_cells = int(part.isna().sum().sum())
        ratio = (missing_cells / total_cells) if total_cells > 0 else 0.0
        stats.append(
            {
                "max_context": mc,
                "start_index": s,
                "end_index": e,
                "total_cells": total_cells,
                "missing_cells": missing_cells,
                "missing_ratio": ratio,
            }
        )
    return stats


def _inject_for_column(
    df: pd.DataFrame,
    col: str,
    start_idx: int,
    end_idx: int,
    ranges: Sequence[Tuple[int, int]],
    block_length: int,
    n_blocks_target: int,
    rng: np.random.Generator,
    repair_steps: int,
) -> List[dict]:
    occupied = np.zeros(end_idx - start_idx, dtype=bool)
    positions: List[dict] = []
    range_lengths = [max(0, e - s) for s, e in ranges]
    blocks_per_range = _allocate_integer(n_blocks_target, range_lengths)

    for (r_start, r_end), target_blocks in zip(ranges, blocks_per_range):
        placed_blocks = 0
        while placed_blocks < target_blocks:
            start = _place_block_random(
                occupied=occupied,
                global_start_idx=start_idx,
                layer_start=r_start,
                layer_end=r_end,
                block_length=block_length,
                rng=rng,
                trials=max(32, repair_steps * 8),
            )
            if start is None:
                break
            end = start + block_length
            df.loc[start:end - 1, col] = np.nan
            positions.append({"column": col, "start": start, "end": end, "length": block_length})
            placed_blocks += 1
    return positions


def inject_bm(
    dataset_name: str,
    injection_range: dict,
    missing_ratio: float,
    term: str,
    block_length: int = 50,
    seed: int = 42,
    mode: str = "stratified",
    balanced_contexts: Sequence[int] = DEFAULT_BALANCED_CONTEXTS,
    ratio_tolerance: float = 0.1,
    repair_steps: int = 20,
) -> Tuple[pd.DataFrame, dict]:
    data_path = injection_range.get("data_path", "data/datasets")
    csv_path = Path(data_path) / "ori" / f"{dataset_name}.csv"
    df = pd.read_csv(csv_path)

    start_idx = int(injection_range["start_index"])
    end_idx = int(injection_range["end_index"])
    injection_length = end_idx - start_idx

    excluded_cols = {"date", "time", "timestamp", "item_id"}
    data_cols = [col for col in df.columns if col.lower() not in excluded_cols]
    if not data_cols:
        raise ValueError("No data columns available for BM injection")

    total_target_missing = int(round(injection_length * len(data_cols) * missing_ratio))
    total_target_blocks = max(0, total_target_missing // block_length)
    blocks_per_col = _allocate_integer(total_target_blocks, [1.0] * len(data_cols))

    seed_offset = hash(f"{dataset_name}_{term}_{missing_ratio}_BM_{mode}") % 10000
    rng = np.random.default_rng(seed + seed_offset)

    if mode == "stratified":
        ranges = _build_stratified_ranges(
            dataset_name=dataset_name,
            term=term,
            data_path=data_path,
            start_idx=start_idx,
            end_idx=end_idx,
            balanced_contexts=balanced_contexts,
        )
    else:
        ranges = [(start_idx, end_idx)]

    block_positions: List[dict] = []
    for col, n_blocks in zip(data_cols, blocks_per_col):
        if n_blocks <= 0:
            continue
        positions = _inject_for_column(
            df=df,
            col=col,
            start_idx=start_idx,
            end_idx=end_idx,
            ranges=ranges,
            block_length=block_length,
            n_blocks_target=n_blocks,
            rng=rng,
            repair_steps=repair_steps,
        )
        block_positions.extend(positions)

    injection_area_size = injection_length * len(data_cols)
    injected_count = int(df.iloc[start_idx:end_idx][data_cols].isna().sum().sum())
    actual_missing_ratio = injected_count / injection_area_size if injection_area_size > 0 else 0.0

    context_stats = _context_stats(
        df=df,
        data_cols=data_cols,
        dataset_name=dataset_name,
        term=term,
        data_path=data_path,
        start_idx=start_idx,
        end_idx=end_idx,
        balanced_contexts=balanced_contexts,
    )
    lower = missing_ratio * (1 - ratio_tolerance)
    upper = missing_ratio * (1 + ratio_tolerance)
    within_tolerance = all(lower <= s["missing_ratio"] <= upper for s in context_stats) if context_stats else True

    info = {
        "dataset_name": dataset_name,
        "term": term,
        "missing_ratio": missing_ratio,
        "block_length": block_length,
        "mode": mode,
        "n_blocks": len(block_positions),
        "total_cells": injection_area_size,
        "injected_missing": injected_count,
        "actual_missing_ratio": actual_missing_ratio,
        "injection_range": {"start_index": start_idx, "end_index": end_idx, "length": injection_length},
        "balanced_contexts": list(sorted(set(balanced_contexts))),
        "ratio_tolerance": ratio_tolerance,
        "repair_steps": repair_steps,
        "within_tolerance": within_tolerance,
        "context_stats": context_stats,
        "block_positions": block_positions,
        "data_columns": data_cols,
    }
    return df, info


def save_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    missing_ratio: float,
    term: str,
    output_base_dir: str = "data/datasets",
    block_length: int = 50,
) -> str:
    ratio_str = f"{int(missing_ratio * 100):03d}"
    output_dir = Path(output_base_dir) / "BM" / f"BM_{ratio_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{dataset_name}_BM_length{block_length}_{ratio_str}_{term}.csv"
    output_path = output_dir / output_filename
    df.to_csv(output_path, index=False)
    return str(output_path)


def get_available_terms(dataset_name: str, data_path: str = "data/datasets") -> List[str]:
    props = load_dataset_properties(data_path)
    if dataset_name not in props:
        raise ValueError(f"Dataset '{dataset_name}' not found in properties")
    ds_term = props[dataset_name].get("term", "med_long")
    return ["short"] if ds_term == "short" else ["short", "medium", "long"]


def run_bm_injection(
    dataset_name: str,
    missing_ratios: List[float],
    terms: List[str],
    data_path: str = "data/datasets",
    output_base_dir: str = "data/datasets",
    block_length: int = 50,
    max_context: int = 8192,
    seed: int = 42,
    mode: str = "stratified",
    balanced_contexts: Sequence[int] = DEFAULT_BALANCED_CONTEXTS,
    ratio_tolerance: float = 0.1,
    repair_steps: int = 20,
) -> List[dict]:
    data_path = str(Path(data_path))
    effective_max_context = max(max_context, max(balanced_contexts) if balanced_contexts else max_context)
    results = []
    for term in terms:
        print(f"\n获取数据集 '{dataset_name}' ({term}) 的注错区间...")
        injection_range = get_injection_range(dataset_name=dataset_name, term=term, data_path=data_path, max_context=effective_max_context)
        injection_range["data_path"] = data_path
        print(f"  注错区间：[{injection_range['start_index']}, {injection_range['end_index']})")
        print(f"  注错区间长度：{injection_range['end_index'] - injection_range['start_index']}")

        for missing_ratio in missing_ratios:
            print(f"\n{'=' * 80}")
            print(f"注入：pattern=BM, term={term}, missing_ratio={missing_ratio:.2%}, mode={mode}")
            print(f"  块长度：{block_length}")
            print(f"{'=' * 80}")

            df_injected, info = inject_bm(
                dataset_name=dataset_name,
                injection_range=injection_range,
                missing_ratio=missing_ratio,
                term=term,
                block_length=block_length,
                seed=seed,
                mode=mode,
                balanced_contexts=balanced_contexts,
                ratio_tolerance=ratio_tolerance,
                repair_steps=repair_steps,
            )

            output_path = save_dataset(
                df=df_injected,
                dataset_name=dataset_name,
                missing_ratio=missing_ratio,
                term=term,
                output_base_dir=output_base_dir,
                block_length=block_length,
            )
            info["output_path"] = output_path
            info["original_path"] = str(Path(data_path) / "ori" / f"{dataset_name}.csv")
            info["pattern"] = "BM"

            print("\n注入结果:")
            print(f"  总单元格数：{info['total_cells']}")
            print(f"  块数量：{info['n_blocks']}")
            print(f"  注入缺失值数：{info['injected_missing']}")
            print(f"  实际缺失比例：{info['actual_missing_ratio']:.2%}")
            print(f"  五区间约束达标：{info['within_tolerance']}")
            print("\n文件保存:")
            print(f"  {output_path}")
            print("=" * 80)
            results.append(info)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM（块缺失）缺失值注入")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称（如 ETTh1）")
    parser.add_argument("--missing_ratio", type=str, required=True, help="缺失比例，支持单个值或逗号分隔列表")
    parser.add_argument("--term", type=str, default=None, choices=["short", "medium", "long"], help="预测 horizon 类型")
    parser.add_argument("--data_path", type=str, default="data/datasets", help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="data/datasets", help="输出目录")
    parser.add_argument("--block_length", type=int, default=50, help="块长度")
    parser.add_argument("--max_context", type=int, default=8192, help="最大回顾窗口长度")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mode", type=str, default="stratified", choices=["stratified", "random"], help="注空模式")
    parser.add_argument("--balanced_contexts", type=str, default="512,2048,2880,4096,8192", help="stratified 模式的 context 列表")
    parser.add_argument("--ratio_tolerance", type=float, default=0.1, help="允许相对偏差（默认 0.1）")
    parser.add_argument("--repair_steps", type=int, default=20, help="随机搜索尝试系数（默认 20）")
    parser.add_argument("--no_auto_term", action="store_true", help="禁用自动 term 检测，使用 --term 指定值")
    args = parser.parse_args()

    if args.data_path == "datasets":
        args.data_path = str(Path(__file__).resolve().parents[2] / "data" / "datasets")
    if args.output_dir == "datasets":
        args.output_dir = str(Path(__file__).resolve().parents[2] / "data" / "datasets")

    missing_ratios = parse_missing_ratios(args.missing_ratio)
    balanced_contexts = parse_int_list(args.balanced_contexts)
    if args.no_auto_term:
        terms = [args.term] if args.term else ["short"]
        print(f"使用指定的 term: {terms}")
    else:
        terms = get_available_terms(args.dataset, args.data_path)
        print(f"自动检测到数据集 '{args.dataset}' 的 term 配置：{terms}")

    results = run_bm_injection(
        dataset_name=args.dataset,
        missing_ratios=missing_ratios,
        terms=terms,
        data_path=args.data_path,
        output_base_dir=args.output_dir,
        block_length=args.block_length,
        max_context=args.max_context,
        seed=args.seed,
        mode=args.mode,
        balanced_contexts=balanced_contexts,
        ratio_tolerance=args.ratio_tolerance,
        repair_steps=args.repair_steps,
    )

    print(f"\n{'=' * 80}")
    print(f"批量注入完成！共生成 {len(results)} 个文件:")
    print(f"{'=' * 80}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['output_path']}")
    print(f"{'=' * 80}\n")
