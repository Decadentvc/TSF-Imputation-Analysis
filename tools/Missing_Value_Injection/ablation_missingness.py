"""Generate missing-data inputs for the ablation experiments in section 4.3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BALANCED_CONTEXTS = [512, 2048, 2880, 4096, 8192]
DEFAULT_RATIO_DATASETS = [
    "electricity",
    "ETTh1",
    "weather",
    "traffic",
    "exchange_rate",
    "azure2019_U_5T",
]
DEFAULT_SHAPE_DATASETS = ["ETTh1", "weather", "traffic", "exchange_rate"]
DEFAULT_RATIO_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
DEFAULT_LENGTH_VALUES = [10, 25, 50, 100, 200]
DEFAULT_POSITIONS = ["front", "middle", "back"]
DEFAULT_PATTERNS = ["random_point", "single_block", "multi_block"]
TIME_COLUMNS = {"date", "time", "timestamp", "datetime", "index", "item_id"}


def _split_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return parts or None


def _parse_float_list(value: Optional[str], default: Sequence[float]) -> List[float]:
    parts = _split_csv(value)
    if not parts:
        return list(default)
    parsed = []
    for part in parts:
        item = float(part)
        if item > 1:
            item = item / 100.0
        if item <= 0 or item >= 1:
            raise ValueError(f"Ratio must be in (0, 1), got {part}")
        parsed.append(round(item, 6))
    return parsed


def _parse_int_values(value: Optional[str], default: Sequence[int]) -> List[int]:
    parts = _split_csv(value)
    if not parts:
        return list(default)
    parsed = [int(part) for part in parts]
    if any(item <= 0 for item in parsed):
        raise ValueError("Integer values must be positive")
    return parsed


def _parse_balanced_contexts(value: str) -> List[int]:
    contexts = _parse_int_values(value, DEFAULT_BALANCED_CONTEXTS)
    return sorted(set(contexts))


def _ratio_str(ratio: float) -> str:
    return f"{int(round(ratio * 100)):03d}"


def _stable_seed(seed: int, *parts: object) -> int:
    raw = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


def _load_terms(dataset: str, data_path: Path, requested_terms: Optional[List[str]]) -> List[str]:
    props_path = data_path / "dataset_properties.json"
    if not props_path.exists():
        raise FileNotFoundError(f"Dataset properties not found: {props_path}")
    with open(props_path, "r", encoding="utf-8") as f:
        props = json.load(f)
    if dataset not in props:
        raise ValueError(f"Dataset '{dataset}' not found in dataset properties")
    allowed = ["short"] if props[dataset].get("term", "med_long") == "short" else ["short", "medium", "long"]
    terms = requested_terms or ["long"]
    selected = [term for term in terms if term in allowed]
    if not selected:
        print(f"[Skip] {dataset}: requested terms {terms} are unavailable, allowed={allowed}")
    return selected


def _data_columns(df: pd.DataFrame) -> List[str]:
    cols = [col for col in df.columns if col.lower() not in TIME_COLUMNS]
    if not cols:
        raise ValueError("No value columns found for missing-data injection")
    return cols


def _position_span(start_idx: int, end_idx: int, position: str) -> Tuple[int, int]:
    if position == "any":
        return start_idx, end_idx
    length = end_idx - start_idx
    if length <= 0:
        raise ValueError("Injection range is empty")
    first = start_idx + length // 3
    second = start_idx + (2 * length) // 3
    if position == "front":
        return start_idx, first
    if position == "middle":
        return first, second
    if position == "back":
        return second, end_idx
    raise ValueError(f"Unknown position: {position}")


def _allocate_integer(total: int, weights: Sequence[float]) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        out = [0 for _ in weights]
        out[0] = total
        return out
    raw = [total * (weight / weight_sum) for weight in weights]
    base = [int(np.floor(value)) for value in raw]
    remainder = total - sum(base)
    if remainder > 0:
        order = sorted(range(len(raw)), key=lambda idx: raw[idx] - base[idx], reverse=True)
        for idx in order[:remainder]:
            base[idx] += 1
    return base


def _can_place(occupied: np.ndarray, start_rel: int, length: int, min_gap: int = 1) -> bool:
    end_rel = start_rel + length
    left = max(0, start_rel - min_gap)
    right = min(len(occupied), end_rel + min_gap)
    return not occupied[left:right].any()


def _place_block(
    occupied: np.ndarray,
    segment_start: int,
    segment_end: int,
    block_length: int,
    rng: np.random.Generator,
    trials: int,
) -> Optional[int]:
    max_start = segment_end - block_length
    if max_start < segment_start:
        return None

    candidate_count = max_start - segment_start + 1
    for _ in range(min(max(1, trials), candidate_count)):
        start = int(rng.integers(segment_start, max_start + 1))
        start_rel = start - segment_start
        if _can_place(occupied, start_rel, block_length):
            occupied[start_rel:start_rel + block_length] = True
            return start

    if candidate_count <= 10000:
        candidates = np.arange(segment_start, max_start + 1)
        rng.shuffle(candidates)
        for start in candidates.tolist():
            start_rel = int(start) - segment_start
            if _can_place(occupied, start_rel, block_length):
                occupied[start_rel:start_rel + block_length] = True
                return int(start)
    return None


def _inject_random_points(
    df: pd.DataFrame,
    data_cols: Sequence[str],
    target_missing: int,
    segment_start: int,
    segment_end: int,
    rng: np.random.Generator,
) -> List[dict]:
    rows = segment_end - segment_start
    available_cells = rows * len(data_cols)
    if target_missing > available_cells:
        raise ValueError(
            f"Target missing cells ({target_missing}) exceed selected segment capacity ({available_cells})"
        )
    if target_missing <= 0:
        return []

    flat_indices = rng.choice(available_cells, size=target_missing, replace=False)
    col_positions = [df.columns.get_loc(col) for col in data_cols]
    positions = []
    for flat in flat_indices.tolist():
        row_offset = flat // len(data_cols)
        col_offset = flat % len(data_cols)
        row_idx = segment_start + int(row_offset)
        col_name = data_cols[col_offset]
        df.iat[row_idx, col_positions[col_offset]] = np.nan
        positions.append({"column": col_name, "start": row_idx, "end": row_idx + 1, "length": 1})
    return positions


def _inject_single_blocks(
    df: pd.DataFrame,
    data_cols: Sequence[str],
    full_range_length: int,
    missing_ratio: float,
    segment_start: int,
    segment_end: int,
    rng: np.random.Generator,
) -> List[dict]:
    block_length = int(round(full_range_length * missing_ratio))
    segment_length = segment_end - segment_start
    if block_length <= 0:
        return []
    if block_length > segment_length:
        raise ValueError(
            f"Single block length ({block_length}) exceeds selected segment length ({segment_length})"
        )

    positions = []
    for col in data_cols:
        start = int(rng.integers(segment_start, segment_end - block_length + 1))
        end = start + block_length
        df.loc[start:end - 1, col] = np.nan
        positions.append({"column": col, "start": start, "end": end, "length": block_length})
    return positions


def _inject_multi_blocks(
    df: pd.DataFrame,
    data_cols: Sequence[str],
    full_range_length: int,
    missing_ratio: float,
    block_length: int,
    segment_start: int,
    segment_end: int,
    rng: np.random.Generator,
    repair_steps: int,
) -> List[dict]:
    total_target_missing = int(round(full_range_length * len(data_cols) * missing_ratio))
    total_blocks = max(0, total_target_missing // block_length)
    blocks_per_col = _allocate_integer(total_blocks, [1.0] * len(data_cols))

    positions = []
    segment_length = segment_end - segment_start
    for col, n_blocks in zip(data_cols, blocks_per_col):
        occupied = np.zeros(segment_length, dtype=bool)
        placed = 0
        while placed < n_blocks:
            start = _place_block(
                occupied=occupied,
                segment_start=segment_start,
                segment_end=segment_end,
                block_length=block_length,
                rng=rng,
                trials=max(32, repair_steps * 8),
            )
            if start is None:
                break
            end = start + block_length
            df.loc[start:end - 1, col] = np.nan
            positions.append({"column": col, "start": start, "end": end, "length": block_length})
            placed += 1
    return positions


def inject_pattern(
    dataset: str,
    term: str,
    data_path: Path,
    missing_ratio: float,
    block_length: int,
    pattern: str,
    position: str,
    seed: int,
    max_context: int,
    repair_steps: int,
) -> Tuple[pd.DataFrame, dict]:
    from inject_range_utils import get_injection_range

    csv_path = data_path / "ori" / f"{dataset}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    data_cols = _data_columns(df)

    injection_range = get_injection_range(
        dataset_name=dataset,
        term=term,
        data_path=str(data_path),
        max_context=max_context,
    )
    start_idx = int(injection_range["start_index"])
    end_idx = int(injection_range["end_index"])
    segment_start, segment_end = _position_span(start_idx, end_idx, position)
    full_range_length = end_idx - start_idx
    target_missing = int(round(full_range_length * len(data_cols) * missing_ratio))
    rng = np.random.default_rng(_stable_seed(seed, dataset, term, missing_ratio, block_length, pattern, position))

    if pattern == "random_point":
        block_positions = _inject_random_points(
            df=df,
            data_cols=data_cols,
            target_missing=target_missing,
            segment_start=segment_start,
            segment_end=segment_end,
            rng=rng,
        )
    elif pattern == "single_block":
        block_positions = _inject_single_blocks(
            df=df,
            data_cols=data_cols,
            full_range_length=full_range_length,
            missing_ratio=missing_ratio,
            segment_start=segment_start,
            segment_end=segment_end,
            rng=rng,
        )
    elif pattern == "multi_block":
        block_positions = _inject_multi_blocks(
            df=df,
            data_cols=data_cols,
            full_range_length=full_range_length,
            missing_ratio=missing_ratio,
            block_length=block_length,
            segment_start=segment_start,
            segment_end=segment_end,
            rng=rng,
            repair_steps=repair_steps,
        )
    else:
        raise ValueError(f"Unknown missing pattern: {pattern}")

    injected = int(df.iloc[start_idx:end_idx][list(data_cols)].isna().sum().sum())
    total_cells = full_range_length * len(data_cols)
    info = {
        "dataset_name": dataset,
        "term": term,
        "missing_ratio": missing_ratio,
        "block_length": block_length,
        "pattern": pattern,
        "position": position,
        "data_columns": list(data_cols),
        "total_cells": total_cells,
        "injected_missing": injected,
        "actual_missing_ratio": injected / total_cells if total_cells else 0.0,
        "injection_range": injection_range,
        "selected_segment": {"start_index": segment_start, "end_index": segment_end},
        "block_positions": block_positions,
    }
    return df, info


def _save_output(
    df: pd.DataFrame,
    info: dict,
    output_path: Path,
    force: bool,
    dry_run: bool,
) -> bool:
    if output_path.exists() and not force:
        print(f"[Skip] {output_path}")
        return False
    print(f"[Write] {output_path}")
    if dry_run:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    meta_path = output_path.with_name(output_path.stem + "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return True


def _standard_output_path(
    output_dir: Path,
    dataset: str,
    ratio: float,
    term: str,
    block_length: int,
) -> Path:
    ratio_text = _ratio_str(ratio)
    return output_dir / "BM" / f"BM_{ratio_text}" / f"{dataset}_BM_length{block_length}_{ratio_text}_{term}.csv"


def _variant_output_path(
    output_dir: Path,
    variant_root: str,
    dataset: str,
    ratio: float,
    term: str,
    block_length: int,
    marker: str,
) -> Path:
    ratio_text = _ratio_str(ratio)
    return (
        output_dir
        / variant_root
        / f"BM_{ratio_text}"
        / f"{dataset}_BM_{marker}_length{block_length}_{ratio_text}_{term}.csv"
    )


def _generate_standard(
    datasets: Sequence[str],
    terms: Optional[List[str]],
    ratios: Sequence[float],
    block_lengths: Sequence[int],
    data_path: Path,
    output_dir: Path,
    seed: int,
    max_context: int,
    mode: str,
    balanced_contexts: Sequence[int],
    ratio_tolerance: float,
    repair_steps: int,
    force: bool,
    dry_run: bool,
) -> int:
    generated = 0
    for dataset in datasets:
        for term in _load_terms(dataset, data_path, terms):
            for ratio in ratios:
                for block_length in block_lengths:
                    output_path = _standard_output_path(output_dir, dataset, ratio, term, block_length)
                    if output_path.exists() and not force:
                        print(f"[Skip] {output_path}")
                        continue
                    if dry_run:
                        print(f"[Write] {output_path}")
                        generated += 1
                        continue
                    from BM import inject_bm
                    from inject_range_utils import get_injection_range

                    injection_range = get_injection_range(
                        dataset_name=dataset,
                        term=term,
                        data_path=str(data_path),
                        max_context=max_context,
                    )
                    injection_range["data_path"] = str(data_path)
                    df, info = inject_bm(
                        dataset_name=dataset,
                        injection_range=injection_range,
                        missing_ratio=ratio,
                        term=term,
                        block_length=block_length,
                        seed=seed,
                        mode=mode,
                        balanced_contexts=balanced_contexts,
                        ratio_tolerance=ratio_tolerance,
                        repair_steps=repair_steps,
                    )
                    info["pattern"] = "multi_block"
                    info["position"] = "stratified"
                    if _save_output(df, info, output_path, force=force, dry_run=dry_run):
                        generated += 1
    return generated


def _generate_variants(
    datasets: Sequence[str],
    terms: Optional[List[str]],
    ratio: float,
    block_length: int,
    patterns: Sequence[str],
    positions: Sequence[str],
    data_path: Path,
    output_dir: Path,
    seed: int,
    max_context: int,
    repair_steps: int,
    force: bool,
    dry_run: bool,
) -> int:
    generated = 0
    for dataset in datasets:
        for term in _load_terms(dataset, data_path, terms):
            for pattern in patterns:
                for position in positions:
                    if pattern == "multi_block" and position == "any":
                        variant_root = "BM_PATTERN"
                        marker = "pattern_multi_block"
                    elif position == "any":
                        variant_root = "BM_PATTERN"
                        marker = f"pattern_{pattern}"
                    else:
                        variant_root = "BM_POSITION"
                        marker = f"position_{position}"
                    output_path = _variant_output_path(
                        output_dir=output_dir,
                        variant_root=variant_root,
                        dataset=dataset,
                        ratio=ratio,
                        term=term,
                        block_length=block_length,
                        marker=marker,
                    )
                    if output_path.exists() and not force:
                        print(f"[Skip] {output_path}")
                        continue
                    if dry_run:
                        print(f"[Write] {output_path}")
                        generated += 1
                        continue
                    df, info = inject_pattern(
                        dataset=dataset,
                        term=term,
                        data_path=data_path,
                        missing_ratio=ratio,
                        block_length=block_length,
                        pattern=pattern,
                        position=position,
                        seed=seed,
                        max_context=max_context,
                        repair_steps=repair_steps,
                    )
                    if _save_output(df, info, output_path, force=force, dry_run=dry_run):
                        generated += 1
    return generated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate planned missing-data inputs for ablation experiments"
    )
    parser.add_argument(
        "scenario",
        choices=["ratio", "length", "position", "pattern", "all"],
        help="Ablation input set to generate",
    )
    parser.add_argument("--data_path", default="data/datasets")
    parser.add_argument("--output_dir", default="data/datasets")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset names")
    parser.add_argument("--terms", default="long", help="Comma-separated terms")
    parser.add_argument("--ratios", default=None, help="Comma-separated ratios")
    parser.add_argument("--block_lengths", default=None, help="Comma-separated block lengths")
    parser.add_argument("--patterns", default=None, help="Comma-separated patterns")
    parser.add_argument("--positions", default=None, help="Comma-separated positions")
    parser.add_argument("--block_length", type=int, default=50)
    parser.add_argument("--ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_context", type=int, default=8192)
    parser.add_argument("--mode", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--balanced_contexts", default="512,2048,2880,4096,8192")
    parser.add_argument("--ratio_tolerance", type=float, default=0.1)
    parser.add_argument("--repair_steps", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    terms = _split_csv(args.terms)
    balanced_contexts = _parse_balanced_contexts(args.balanced_contexts)

    if args.scenario == "ratio":
        datasets = _split_csv(args.datasets) or DEFAULT_RATIO_DATASETS
        generated = _generate_standard(
            datasets=datasets,
            terms=terms,
            ratios=_parse_float_list(args.ratios, DEFAULT_RATIO_VALUES),
            block_lengths=[args.block_length],
            data_path=data_path,
            output_dir=output_dir,
            seed=args.seed,
            max_context=args.max_context,
            mode=args.mode,
            balanced_contexts=balanced_contexts,
            ratio_tolerance=args.ratio_tolerance,
            repair_steps=args.repair_steps,
            force=args.force,
            dry_run=args.dry_run,
        )
    elif args.scenario == "length":
        datasets = _split_csv(args.datasets) or DEFAULT_SHAPE_DATASETS
        generated = _generate_standard(
            datasets=datasets,
            terms=terms,
            ratios=[args.ratio],
            block_lengths=_parse_int_values(args.block_lengths, DEFAULT_LENGTH_VALUES),
            data_path=data_path,
            output_dir=output_dir,
            seed=args.seed,
            max_context=args.max_context,
            mode=args.mode,
            balanced_contexts=balanced_contexts,
            ratio_tolerance=args.ratio_tolerance,
            repair_steps=args.repair_steps,
            force=args.force,
            dry_run=args.dry_run,
        )
    elif args.scenario == "position":
        datasets = _split_csv(args.datasets) or DEFAULT_SHAPE_DATASETS
        generated = _generate_variants(
            datasets=datasets,
            terms=terms,
            ratio=args.ratio,
            block_length=args.block_length,
            patterns=["multi_block"],
            positions=_split_csv(args.positions) or DEFAULT_POSITIONS,
            data_path=data_path,
            output_dir=output_dir,
            seed=args.seed,
            max_context=args.max_context,
            repair_steps=args.repair_steps,
            force=args.force,
            dry_run=args.dry_run,
        )
    elif args.scenario == "pattern":
        datasets = _split_csv(args.datasets) or DEFAULT_SHAPE_DATASETS
        generated = _generate_variants(
            datasets=datasets,
            terms=terms,
            ratio=args.ratio,
            block_length=args.block_length,
            patterns=_split_csv(args.patterns) or DEFAULT_PATTERNS,
            positions=["any"],
            data_path=data_path,
            output_dir=output_dir,
            seed=args.seed,
            max_context=args.max_context,
            repair_steps=args.repair_steps,
            force=args.force,
            dry_run=args.dry_run,
        )
    else:
        ratio_count = _generate_standard(
            datasets=DEFAULT_RATIO_DATASETS,
            terms=terms,
            ratios=DEFAULT_RATIO_VALUES,
            block_lengths=[args.block_length],
            data_path=data_path,
            output_dir=output_dir,
            seed=args.seed,
            max_context=args.max_context,
            mode=args.mode,
            balanced_contexts=balanced_contexts,
            ratio_tolerance=args.ratio_tolerance,
            repair_steps=args.repair_steps,
            force=args.force,
            dry_run=args.dry_run,
        )
        length_count = _generate_standard(
            datasets=DEFAULT_SHAPE_DATASETS,
            terms=terms,
            ratios=[args.ratio],
            block_lengths=DEFAULT_LENGTH_VALUES,
            data_path=data_path,
            output_dir=output_dir,
            seed=args.seed,
            max_context=args.max_context,
            mode=args.mode,
            balanced_contexts=balanced_contexts,
            ratio_tolerance=args.ratio_tolerance,
            repair_steps=args.repair_steps,
            force=args.force,
            dry_run=args.dry_run,
        )
        position_count = _generate_variants(
            datasets=DEFAULT_SHAPE_DATASETS,
            terms=terms,
            ratio=args.ratio,
            block_length=args.block_length,
            patterns=["multi_block"],
            positions=DEFAULT_POSITIONS,
            data_path=data_path,
            output_dir=output_dir,
            seed=args.seed,
            max_context=args.max_context,
            repair_steps=args.repair_steps,
            force=args.force,
            dry_run=args.dry_run,
        )
        pattern_count = _generate_variants(
            datasets=DEFAULT_SHAPE_DATASETS,
            terms=terms,
            ratio=args.ratio,
            block_length=args.block_length,
            patterns=DEFAULT_PATTERNS,
            positions=["any"],
            data_path=data_path,
            output_dir=output_dir,
            seed=args.seed,
            max_context=args.max_context,
            repair_steps=args.repair_steps,
            force=args.force,
            dry_run=args.dry_run,
        )
        generated = ratio_count + length_count + position_count + pattern_count

    label = "Planned files" if args.dry_run else "Generated files"
    print(f"{label}: {generated}")


if __name__ == "__main__":
    main()
