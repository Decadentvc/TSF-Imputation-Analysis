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
        KairosAdapter,
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
        KairosAdapter,
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


def _opt_bool(raw: object, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


KAIROS_SIZE_MAP = {
    "10m": "mldi-lab/Kairos_10m",
    "23m": "mldi-lab/Kairos_23m",
    "50m": "mldi-lab/Kairos_50m",
    "small": "mldi-lab/Kairos_10m",
    "base": "mldi-lab/Kairos_23m",
    "large": "mldi-lab/Kairos_50m",
}


def _kairos_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """从 build_model_adapter 的 **kwargs 中抽取 Kairos 共用可选参数。"""
    kairos_opt: dict[str, Any] = {
        "context_length": _opt_int(kwargs.get("context_length"), 2048),
        "preserve_positivity": _opt_bool(kwargs.get("preserve_positivity"), True),
        "average_with_flipped_input": _opt_bool(
            kwargs.get("average_with_flipped_input"), True
        ),
    }
    if kwargs.get("kairos_dir") is not None:
        kairos_opt["kairos_dir"] = _opt_str(kwargs.get("kairos_dir"), "")
    return kairos_opt


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

    if model_key in {"kairos", "kairos_auto"}:
        size_raw = _opt_str(kwargs.get("model_size"), "").lower()
        resolved = model_name or KAIROS_SIZE_MAP.get(size_raw)
        if resolved is None:
            resolved = "mldi-lab/Kairos_50m"
        return KairosAdapter(
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            model_name=resolved,
            **_kairos_kwargs(kwargs),
        )

    if model_key in {"kairos23m", "kairos_23m"}:
        return Kairos23mAdapter(
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            model_name=model_name or "mldi-lab/Kairos_23m",
            **_kairos_kwargs(kwargs),
        )

    if model_key in {"kairos50m", "kairos_50m"}:
        return Kairos50mAdapter(
            prediction_length=prediction_length,
            num_samples=num_samples,
            batch_size=batch_size,
            device=device,
            model_name=model_name or "mldi-lab/Kairos_50m",
            **_kairos_kwargs(kwargs),
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
            model_name=model_name or "visiontspp-local",
            model_size=_opt_str(kwargs.get("model_size"), "base").lower(),
            context_length=_opt_int(kwargs.get("context_length"), 4000),
            ckpt_dir=_opt_str(kwargs.get("ckpt_dir"), "./hf_models/VisionTSpp"),
            num_patch_input=_opt_int(kwargs.get("num_patch_input"), 7),
            padding_mode=_opt_str(kwargs.get("padding_mode"), "constant"),
            max_vars_per_pass=_opt_int(kwargs.get("max_vars_per_pass"), 16),
        )

    raise ValueError(
        f"Unsupported model: {model}. Available: sundial, chronos2, timesfm2p5, "
        f"kairos, kairos23m, kairos50m, timesfm2p0, visiontspp"
    )
