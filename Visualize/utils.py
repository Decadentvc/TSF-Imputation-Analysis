from __future__ import annotations


def normalize_ratio(raw_ratio: str) -> str:
    text = raw_ratio.strip()
    if not text:
        raise ValueError("missing_ratio cannot be empty")

    if text.isdigit():
        if len(text) == 3:
            return text
        if len(text) <= 2:
            return text.zfill(3)
        raise ValueError(f"Invalid missing_ratio: {raw_ratio}")

    value = float(text)
    if value <= 1:
        value *= 100
    ratio_int = int(round(value))
    if ratio_int < 0 or ratio_int > 100:
        raise ValueError(f"missing_ratio out of range: {raw_ratio}")
    return f"{ratio_int:03d}"
