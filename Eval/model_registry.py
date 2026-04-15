"""
模型注册与构建。
"""

from __future__ import annotations

from typing import Any

try:
    # Package-style import: python -m Eval.run_eval
    from .model_adapters import Chronos2Adapter, SundialAdapter
except ImportError:
    # Script-style import: python Eval/run_eval.py
    from model_adapters import Chronos2Adapter, SundialAdapter


def build_model_adapter(
    model: str,
    prediction_length: int,
    batch_size: int,
    device: str,
    num_samples: int = 100,
    model_name: str | None = None,
    **kwargs: Any,
):
    model_key = model.lower()

    if model_key == "sundial":
        return SundialAdapter(
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            model_name=model_name or "thuml/sundial-base-128m",
        )

    if model_key == "chronos2":
        return Chronos2Adapter(
            prediction_length=prediction_length,
            batch_size=batch_size,
            device=device,
            model_name=model_name or "amazon/chronos-2",
            predict_batches_jointly=bool(kwargs.get("predict_batches_jointly", False)),
            torch_dtype=kwargs.get("torch_dtype"),
        )

    raise ValueError(f"Unsupported model: {model}. Available: sundial, chronos2")
