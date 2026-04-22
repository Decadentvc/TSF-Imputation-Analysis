"""
模型注册与构建。
"""

from __future__ import annotations

from typing import Any

try:
    # Package-style import: python -m Eval.run_eval
    from .model_adapters import (
        Chronos2Adapter,
        Kairos23mAdapter,
        Kairos50mAdapter,
        SundialAdapter,
        TimesFM2p0Adapter,
        TimesFM2p5Adapter,
        VisionTSppAdapter,
    )
except ImportError:
    # Script-style import: python Eval/run_eval.py
    from model_adapters import (
        Chronos2Adapter,
        Kairos23mAdapter,
        Kairos50mAdapter,
        SundialAdapter,
        TimesFM2p0Adapter,
        TimesFM2p5Adapter,
        VisionTSppAdapter,
    )


def _opt_str(raw: object, default: str) -> str:
    return default if raw is None else str(raw)


def _opt_int(raw: object, default: int) -> int:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, (str, bytes)):
        return int(raw)
    raise TypeError(f"Cannot convert {type(raw).__name__} to int")


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

    if model_key == "timesfm2p5":
        return TimesFM2p5Adapter(
            prediction_length=prediction_length,
            batch_size=batch_size,
            device=device,
            model_name=model_name or "google/timesfm-2.5-200m-pytorch",
        )

    if model_key in {"kairos23m", "kairos_23m"}:
        return Kairos23mAdapter(
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            model_name=model_name or "mldi-lab/Kairos_23m",
        )

    if model_key in {"kairos50m", "kairos_50m"}:
        return Kairos50mAdapter(
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            model_name=model_name or "mldi-lab/Kairos_50m",
        )

    if model_key in {"timesfm2p0", "timesfm_2p0_500m", "timesfm2p0_500m"}:
        return TimesFM2p0Adapter(
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            model_name=model_name or "google/timesfm-2.0-500m-pytorch",
        )

    if model_key in {"visiontspp", "visionts++"}:
        return VisionTSppAdapter(
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
<<<<<<< HEAD
            model_name=model_name or "visiontspp-local",
            model_size=_opt_str(kwargs.get("model_size"), "base").lower(),
            context_length=_opt_int(kwargs.get("context_length"), 4000),
            ckpt_dir=_opt_str(kwargs.get("ckpt_dir"), "./hf_models/VisionTSpp"),
            num_patch_input=_opt_int(kwargs.get("num_patch_input"), 7),
            padding_mode=_opt_str(kwargs.get("padding_mode"), "constant"),
            max_vars_per_pass=_opt_int(kwargs.get("max_vars_per_pass"), 16),
=======
            model_name=model_name or "Lefei/VisionTSpp",
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f
        )

    raise ValueError(
        f"Unsupported model: {model}. Available: sundial, chronos2, timesfm2p5, "
        f"kairos23m, kairos50m, timesfm2p0, visiontspp"
    )
