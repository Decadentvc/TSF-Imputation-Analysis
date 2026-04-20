from __future__ import annotations

from typing import List


METRIC_COLUMNS: List[str] = [
    "trend_strength",
    "trend_linearity",
    "seasonal_strength",
    "seasonal_correlation",
    "residual_autocorr_lag1",
    "spectral_entropy",
]

TERM_CHOICES: List[str] = ["short", "medium", "long"]
