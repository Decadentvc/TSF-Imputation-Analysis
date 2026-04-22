from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np
import pandas as pd
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast, SampleForecast
from gluonts.transform import LastValueImputation
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM


logger = logging.getLogger(__name__)
DEFAULT_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class ForecastAdapter(Protocol):
    def predict(self, test_data_input) -> List[Any]: ...


def _extract_input_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, tuple):
        return entry[0]
    return entry


def _build_forecastor_input(item: Dict[str, Any]) -> tuple[pd.DataFrame, Optional[str]]:
    target = np.asarray(item["target"], dtype=np.float64).reshape(-1)
    if target.shape[0] < 2:
        raise ValueError("Forecastor adapters require at least 2 history points.")

    start = item["start"]
    freq = getattr(start, "freqstr", None)

    start_ts = None
    if hasattr(start, "to_timestamp"):
        start_ts = start.to_timestamp()
    elif isinstance(start, pd.Timestamp):
        start_ts = start

    if start_ts is not None and freq:
        time_index = pd.date_range(start=start_ts, periods=target.shape[0], freq=freq)
    else:
        time_index = pd.RangeIndex(start=0, stop=target.shape[0], step=1)

    return pd.DataFrame({"date": time_index, "target": target}), freq


def _to_deterministic_quantile_forecast(
    point_prediction: np.ndarray,
    start_date: Any,
    quantile_levels: Optional[List[float]],
) -> QuantileForecast:
    levels = quantile_levels or DEFAULT_QUANTILE_LEVELS
    pred = np.asarray(point_prediction, dtype=np.float64).reshape(-1)
    forecast_arrays = np.tile(pred, (len(levels), 1))
    return QuantileForecast(
        forecast_arrays=forecast_arrays,
        forecast_keys=list(map(str, levels)),
        start_date=start_date,
    )


@dataclass
class SundialAdapter:
    prediction_length: int
    num_samples: int = 100
    batch_size: int = 32
    device: str = "cpu"
    model_name: str = "thuml/sundial-base-128m"

    def __post_init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = self.model.to(self.device)

    @staticmethod
    def _left_pad_and_stack_1d(tensors: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            padding = torch.full(
                size=(max_len - len(c),),
                fill_value=torch.nan,
                device=c.device,
            )
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def _prepare_context(self, context: Iterable[torch.Tensor]) -> torch.Tensor:
        context = list(context)
        batch_x = self._left_pad_and_stack_1d(context)
        if batch_x.ndim == 1:
            batch_x = batch_x.unsqueeze(0)
        return batch_x

    @staticmethod
    def _extract_input_entry(entry: Any) -> Dict[str, Any]:
        # TestData 的元素可能是 (input, label) 元组
        if isinstance(entry, tuple):
            return entry[0]
        return entry

    def predict(
        self, test_data_input, batch_x_shape: int = 2880
    ) -> List[SampleForecast]:
        forecast_outputs = []
        input_metadata = []

        for batch in tqdm(batcher(test_data_input, batch_size=self.batch_size)):
            contexts = [
                torch.tensor(self._extract_input_entry(entry)["target"])
                for entry in batch
            ]
            batch_x = self._prepare_context(contexts)

            if batch_x.shape[-1] > batch_x_shape:
                batch_x = batch_x[..., -batch_x_shape:]

            if torch.isnan(batch_x).any():
                arr = np.array(batch_x)
                imputed_rows = [LastValueImputation()(row) for row in arr]
                batch_x = torch.tensor(np.vstack(imputed_rows))

            batch_x = batch_x.to(self.device)
            if self.device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        batch_x,
                        max_new_tokens=self.prediction_length,
                        revin=True,
                        num_samples=self.num_samples,
                    )
            else:
                outputs = self.model.generate(
                    batch_x,
                    max_new_tokens=self.prediction_length,
                    revin=True,
                    num_samples=self.num_samples,
                )

            forecast_outputs.append(outputs.detach().cpu().numpy())

            for entry in batch:
                input_entry = self._extract_input_entry(entry)
                input_metadata.append(
                    {
                        "start": input_entry["start"],
                        "target_length": len(input_entry["target"]),
                    }
                )

        forecast_outputs = np.concatenate(forecast_outputs)
        forecasts: List[SampleForecast] = []
        for item, meta in zip(forecast_outputs, input_metadata):
            forecast_start_date = meta["start"] + meta["target_length"]
            forecasts.append(
                SampleForecast(samples=item, start_date=forecast_start_date)
            )
        return forecasts


@dataclass
class Chronos2Adapter:
    prediction_length: int
    batch_size: int = 32
    model_name: str = "amazon/chronos-2"
    device: str = "cpu"
    quantile_levels: Optional[List[float]] = None
    predict_batches_jointly: bool = False
    torch_dtype: Optional[str] = None
    max_context: int = 8192

    def __post_init__(self):
        try:
            from chronos import BaseChronosPipeline, Chronos2Pipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chronos-forecasting 未安装。请先安装: pip install chronos-forecasting>=2.1"
            ) from exc

        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS

        pipeline_kwargs: Dict[str, Any] = {}
        if self.device:
            pipeline_kwargs["device_map"] = self.device

        if self.torch_dtype:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            if self.torch_dtype not in dtype_map:
                raise ValueError("torch_dtype 必须是 bfloat16/float16/float32")
            pipeline_kwargs["torch_dtype"] = dtype_map[self.torch_dtype]

        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name,
            **pipeline_kwargs,
        )
        if not isinstance(self.pipeline, Chronos2Pipeline):
            raise TypeError("当前适配器仅支持 Chronos-2 管线")

    @staticmethod
    def _extract_input_entry(entry: Any) -> Dict[str, Any]:
        if isinstance(entry, tuple):
            return entry[0]
        return entry

    def _pack_model_items(self, items: Iterable[Any]) -> List[Dict[str, Any]]:
        packed = []
        for entry in items:
            item = self._extract_input_entry(entry)
            target = np.asarray(item["target"])
            if self.max_context > 0 and target.shape[0] > self.max_context:
                target = target[-self.max_context :]
            packed.append({"target": target})
        return packed

    def predict(self, test_data_input) -> List[QuantileForecast]:
        input_entries = [self._extract_input_entry(x) for x in list(test_data_input)]
        input_data = self._pack_model_items(input_entries)
        model_batch_size = self.batch_size

        if self.predict_batches_jointly:
            logger.info(
                "Chronos-2 正在使用 predict_batches_jointly=True；请确保没有滚动窗口泄露。"
            )

        while True:
            try:
                quantiles, _ = self.pipeline.predict_quantiles(
                    inputs=input_data,
                    prediction_length=self.prediction_length,
                    batch_size=model_batch_size,
                    quantile_levels=self.quantile_levels,
                    predict_batches_jointly=self.predict_batches_jointly,
                )
                quantiles = torch.stack(quantiles)
                quantiles = quantiles.permute(0, 3, 2, 1).cpu().numpy()
                if input_data and input_data[0]["target"].ndim == 1:
                    quantiles = quantiles.squeeze(-1)
                break
            except torch.cuda.OutOfMemoryError:  # pragma: no cover
                if model_batch_size <= 1:
                    raise
                logger.warning(
                    "Chronos-2 OOM at batch_size=%s, fallback to %s",
                    model_batch_size,
                    model_batch_size // 2,
                )
                model_batch_size //= 2

        forecasts: List[QuantileForecast] = []
        q_levels = self.quantile_levels or DEFAULT_QUANTILE_LEVELS
        for item, ts in zip(quantiles, input_entries):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, q_levels)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts


@dataclass
class TimesFM2p5Adapter:
    prediction_length: int
    batch_size: int = 128
    model_name: str = "google/timesfm-2.5-200m-pytorch"
    device: str = "cpu"
    max_context: int = 4096
    per_core_batch_size: int = 128

    def __post_init__(self):
        try:
            import timesfm as timesfm_pkg
            from timesfm import configs
            from timesfm.timesfm_2p5 import timesfm_2p5_torch
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "timesfm 未安装。请先安装并确保包含 TimesFM-2.5: pip install -e /path/to/timesfm"
            ) from exc

        self.configs = configs
        try:
            self.tfm = timesfm_pkg.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_name,
                torch_compile=True,
            )
        except Exception as pretrained_exc:
            logger.warning(
                "TimesFM from_pretrained failed, fallback to load_checkpoint: %s",
                pretrained_exc,
            )
            self.tfm = timesfm_2p5_torch.TimesFM_2p5_200M_torch()
            try:
                self.tfm.load_checkpoint(repo_id=self.model_name)
            except TypeError:
                self.tfm.load_checkpoint()
                if self.model_name != "google/timesfm-2.5-200m-pytorch":
                    logger.warning(
                        "Current timesfm version ignores model_name=%s in load_checkpoint().",
                        self.model_name,
                    )
        self.quantiles = list(np.arange(1, 10) / 10.0)

    @staticmethod
    def _extract_input_entry(entry: Any) -> Dict[str, Any]:
        if isinstance(entry, tuple):
            return entry[0]
        return entry

    def predict(self, test_data_input) -> List[QuantileForecast]:
        input_entries = [self._extract_input_entry(x) for x in list(test_data_input)]
        if not input_entries:
            return []

        contexts: List[np.ndarray] = []
        global_max_context = 0
        for entry in input_entries:
            arr = np.asarray(entry["target"], dtype=np.float32)
            if self.max_context > 0 and arr.shape[0] > self.max_context:
                arr = arr[-self.max_context :]
            contexts.append(arr)
            if arr.shape[0] > global_max_context:
                global_max_context = arr.shape[0]

        patch_size = getattr(getattr(self.tfm, "model", None), "p", None)
        if isinstance(patch_size, int) and patch_size > 0:
            global_max_context = (
                (global_max_context + patch_size - 1) // patch_size
            ) * patch_size

        per_core_batch_size = max(1, min(self.per_core_batch_size, self.batch_size))
        self.tfm.compile(
            forecast_config=self.configs.ForecastConfig(
                max_context=min(self.max_context, global_max_context),
                max_horizon=self.prediction_length,
                infer_is_positive=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
                force_flip_invariance=True,
                return_backcast=False,
                normalize_inputs=True,
                per_core_batch_size=per_core_batch_size,
            )
        )

        forecast_outputs = []

        for batch_context in tqdm(batcher(contexts, batch_size=self.batch_size)):
            context = list(batch_context)

            _, full_preds = self.tfm.forecast(
                horizon=self.prediction_length,
                inputs=context,
            )
            full_preds = full_preds[:, 0 : self.prediction_length, 1:]
            forecast_outputs.append(full_preds.transpose((0, 2, 1)))

        forecast_outputs = np.concatenate(forecast_outputs)
        forecasts: List[QuantileForecast] = []
        for item, ts in zip(forecast_outputs, input_entries):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.quantiles)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts


@dataclass
class Kairos23mAdapter:
    prediction_length: int
    num_samples: int = 100
    batch_size: int = 32
    device: str = "cpu"
    model_name: str = "mldi-lab/Kairos_23m"
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
<<<<<<< HEAD
        try:
            from .model.kairos_23m_forecastor import kairos_23m_forecastor
        except ImportError:
            from model.kairos_23m_forecastor import kairos_23m_forecastor
=======
        try:
            from tsfm.model.kairos import AutoModel as KairosAutoModel
        except ImportError as exc:
            raise ImportError(
                "Kairos23mAdapter requires tsfm with kairos support. "
                "Please install the official Kairos dependency."
            ) from exc

        try:
            self._kairos_model = KairosAutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Kairos_23m model load failed. Ensure Kairos/tsfm dependencies are installed "
                "and provide a valid --model_name HF repo id."
            ) from exc
        self._kairos_model = self._kairos_model.to(self.device)
        self._kairos_model.eval()
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f
            from .model.kairos_23m_forecastor import kairos_23m_forecastor
        except ImportError:
            from model.kairos_23m_forecastor import kairos_23m_forecastor

        self._forecastor = kairos_23m_forecastor
        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS
<<<<<<< HEAD
        if self.model_name != "mldi-lab/Kairos_23m":
            logger.warning(
                "Kairos23mAdapter currently ignores model_name=%s; using forecastor defaults.",
                self.model_name,
            )
=======
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f
        if self.model_name != "mldi-lab/Kairos_23m":
            logger.warning(
                "Kairos23mAdapter currently ignores model_name=%s; using forecastor defaults.",
                self.model_name,
            )

    @staticmethod
    def _left_pad_and_stack_1d(tensors: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            padding = torch.full(
                size=(max_len - len(c),),
                fill_value=torch.nan,
                device=c.device,
            )
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def _prepare_context(self, context: Iterable[torch.Tensor]) -> torch.Tensor:
        context = list(context)
        batch_x = self._left_pad_and_stack_1d(context)
        if batch_x.ndim == 1:
            batch_x = batch_x.unsqueeze(0)
        return batch_x

    def _predict_with_kairos_model(self, batch_x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self._kairos_model(
                past_target=batch_x,
                prediction_length=self.prediction_length,
                generation=True,
                preserve_positivity=True,
                average_with_flipped_input=True,
            )

        if isinstance(outputs, dict):
            preds = outputs.get("prediction_outputs")
        else:
            preds = getattr(outputs, "prediction_outputs", None)

        if preds is None:
<<<<<<< HEAD
=======
            raise RuntimeError("Kairos_23m output missing prediction_outputs")

        out_arr = preds.detach().cpu().float().numpy()
        if out_arr.ndim != 3:
            raise RuntimeError(
                f"Kairos_23m prediction_outputs ndim must be 3, got {out_arr.ndim}"
            )

        if out_arr.shape[1] == self.prediction_length:
            pass
        elif out_arr.shape[2] == self.prediction_length:
            out_arr = out_arr.transpose((0, 2, 1))
        elif out_arr.shape[1] > self.prediction_length:
            out_arr = out_arr[:, -self.prediction_length :, :]
        elif out_arr.shape[2] > self.prediction_length:
            out_arr = out_arr[:, :, -self.prediction_length :].transpose((0, 2, 1))
        else:
            raise RuntimeError(
                "Kairos_23m generated horizon is shorter than prediction_length. "
                f"output shape={out_arr.shape}, prediction_length={self.prediction_length}"
            )
        return out_arr

    def predict(self, test_data_input) -> List[QuantileForecast]:
        forecasts: List[QuantileForecast] = []
        input_entries = [_extract_input_entry(x) for x in list(test_data_input)]

        forecast_outputs: List[np.ndarray] = []
        input_metadata = []
        for batch in tqdm(batcher(input_entries, batch_size=self.batch_size)):
            contexts = [torch.tensor(item["target"]) for item in batch]
            batch_x = self._prepare_context(contexts)

            if torch.isnan(batch_x).any():
                arr = np.array(batch_x)
                imputed_rows = [LastValueImputation()(row) for row in arr]
                batch_x = torch.tensor(np.vstack(imputed_rows))

            batch_x = batch_x.to(self.device)
            out_arr = self._predict_with_kairos_model(batch_x)
            forecast_outputs.append(out_arr)

            for item in batch:
                input_metadata.append(
                    {
                        "start": item["start"],
                        "target_length": len(item["target"]),
                    }
                )

        all_samples = np.concatenate(forecast_outputs, axis=0)
        for item, meta in zip(all_samples, input_metadata):
            forecast_start_date = meta["start"] + meta["target_length"]
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item.transpose((1, 0)),
                    forecast_keys=list(map(str, self.quantile_levels)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts


@dataclass
class Kairos50mAdapter:
    prediction_length: int
    num_samples: int = 100
    batch_size: int = 32
    device: str = "cpu"
    model_name: str = "mldi-lab/Kairos_50m"
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
        try:
            from tsfm.model.kairos import AutoModel as KairosAutoModel
        except ImportError as exc:
            raise ImportError(
                "Kairos50mAdapter requires tsfm with kairos support. "
                "Please install the official Kairos dependency."
            ) from exc

        try:
            self._kairos_model = KairosAutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Kairos_50m model load failed. Ensure Kairos/tsfm dependencies are installed "
                "and provide a valid --model_name HF repo id."
            ) from exc
        self._kairos_model = self._kairos_model.to(self.device)
        self._kairos_model.eval()

        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS

    @staticmethod
    def _left_pad_and_stack_1d(tensors: List[torch.Tensor]) -> torch.Tensor:
        max_len = max(len(c) for c in tensors)
        padded = []
        for c in tensors:
            padding = torch.full(
                size=(max_len - len(c),),
                fill_value=torch.nan,
                device=c.device,
            )
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def _prepare_context(self, context: Iterable[torch.Tensor]) -> torch.Tensor:
        context = list(context)
        batch_x = self._left_pad_and_stack_1d(context)
        if batch_x.ndim == 1:
            batch_x = batch_x.unsqueeze(0)
        return batch_x

    def _predict_with_kairos_model(self, batch_x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outputs = self._kairos_model(
                past_target=batch_x,
                prediction_length=self.prediction_length,
                generation=True,
                preserve_positivity=True,
                average_with_flipped_input=True,
            )

        if isinstance(outputs, dict):
            preds = outputs.get("prediction_outputs")
        else:
            preds = getattr(outputs, "prediction_outputs", None)

        if preds is None:
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f
            raise RuntimeError("Kairos_50m output missing prediction_outputs")

        out_arr = preds.detach().cpu().float().numpy()
        if out_arr.ndim != 3:
            raise RuntimeError(
                f"Kairos_50m prediction_outputs ndim must be 3, got {out_arr.ndim}"
            )

        if out_arr.shape[1] == self.prediction_length:
            pass
        elif out_arr.shape[2] == self.prediction_length:
            out_arr = out_arr.transpose((0, 2, 1))
        elif out_arr.shape[1] > self.prediction_length:
            out_arr = out_arr[:, -self.prediction_length :, :]
        elif out_arr.shape[2] > self.prediction_length:
            out_arr = out_arr[:, :, -self.prediction_length :].transpose((0, 2, 1))
        else:
            raise RuntimeError(
                "Kairos_50m generated horizon is shorter than prediction_length. "
                f"output shape={out_arr.shape}, prediction_length={self.prediction_length}"
            )
        return out_arr

    def predict(self, test_data_input) -> List[QuantileForecast]:
        forecasts: List[QuantileForecast] = []
        input_entries = [_extract_input_entry(x) for x in list(test_data_input)]

<<<<<<< HEAD
        for batch in tqdm(batcher(input_entries, batch_size=self.batch_size)):
            for item in batch:
                df_input, freq = _build_forecastor_input(item)
                output_df = self._forecastor(
                    dataframe=df_input,
                    forecast_length=self.prediction_length,
                    num_samples=self.num_samples,
                    freq=freq,
                    device=self.device,
                )
                point_prediction = pd.to_numeric(
                    output_df.iloc[:, -1], errors="coerce"
                ).to_numpy(dtype=np.float64)
                if point_prediction.shape[0] < self.prediction_length:
                    raise RuntimeError(
                        "Kairos_23m returned fewer points than prediction_length."
                    )
                forecasts.append(
                    _to_deterministic_quantile_forecast(
                        point_prediction[: self.prediction_length],
                        item["start"] + len(item["target"]),
                        self.quantile_levels,
                    )
                )
        return forecasts


@dataclass
class Kairos50mAdapter:
    prediction_length: int
    num_samples: int = 100
    batch_size: int = 32
    device: str = "cpu"
    model_name: str = "mldi-lab/Kairos_50m"
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
        try:
            from .model.kairos_50m_forecastor import kairos_50m_forecastor
        except ImportError:
            from model.kairos_50m_forecastor import kairos_50m_forecastor

        self._forecastor = kairos_50m_forecastor
        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS
        if self.model_name != "mldi-lab/Kairos_50m":
            logger.warning(
                "Kairos50mAdapter currently ignores model_name=%s; using forecastor defaults.",
                self.model_name,
            )

    def predict(self, test_data_input) -> List[QuantileForecast]:
        forecasts: List[QuantileForecast] = []
        input_entries = [_extract_input_entry(x) for x in list(test_data_input)]
=======
        forecast_outputs: List[np.ndarray] = []
        input_metadata = []
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f

        for batch in tqdm(batcher(input_entries, batch_size=self.batch_size)):
            for item in batch:
                df_input, freq = _build_forecastor_input(item)
                output_df = self._forecastor(
                    dataframe=df_input,
                    forecast_length=self.prediction_length,
                    num_samples=self.num_samples,
                    freq=freq,
                    device=self.device,
                )
                point_prediction = pd.to_numeric(
                    output_df.iloc[:, -1], errors="coerce"
                ).to_numpy(dtype=np.float64)
                if point_prediction.shape[0] < self.prediction_length:
                    raise RuntimeError(
                        "Kairos_23m returned fewer points than prediction_length."
                    )
                forecasts.append(
                    _to_deterministic_quantile_forecast(
                        point_prediction[: self.prediction_length],
                        item["start"] + len(item["target"]),
                        self.quantile_levels,
                    )
                )
        return forecasts


@dataclass
class Kairos50mAdapter:
    prediction_length: int
    num_samples: int = 100
    batch_size: int = 32
    device: str = "cpu"
    model_name: str = "mldi-lab/Kairos_50m"
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
        try:
            from .model.kairos_50m_forecastor import kairos_50m_forecastor
        except ImportError:
            from model.kairos_50m_forecastor import kairos_50m_forecastor

        self._forecastor = kairos_50m_forecastor
        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS

    @staticmethod
    def _extract_freq(item: Dict[str, Any]) -> str:
        start = item.get("start")
        freq = getattr(start, "freqstr", None)
        if freq:
            return freq
        freq_obj = getattr(start, "freq", None)
        return getattr(freq_obj, "freqstr", "H") or "H"

    @staticmethod
    def _freq_to_category(freq: str) -> int:
        try:
            base = pd.tseries.frequencies.to_offset(freq).name.upper()
        except Exception:
            base = str(freq).upper()

        if base in {"W", "M"}:
            return 1
        if base in {"Q", "Y", "A"}:
            return 2
        return 0

    @staticmethod
    def _linear_impute_nan(values: np.ndarray) -> np.ndarray:
        s = pd.Series(values, dtype=np.float32)
        s = s.interpolate(method="linear", limit_direction="both").ffill().bfill()
        return s.to_numpy(dtype=np.float32)

    def _compose_forecast_arrays(self, full_preds_item: np.ndarray) -> np.ndarray:
        levels = self.quantile_levels or DEFAULT_QUANTILE_LEVELS
        mean_arr = full_preds_item[:, 0]

        arrays: List[np.ndarray] = []
        for level in levels:
            col_idx = None
            for idx, q in enumerate(self._model_quantiles):
                if abs(q - float(level)) < 1e-6:
                    col_idx = idx + 1
                    break
            if col_idx is None or col_idx >= full_preds_item.shape[1]:
                arrays.append(mean_arr[: self.prediction_length])
            else:
                arrays.append(full_preds_item[: self.prediction_length, col_idx])
        return np.stack(arrays, axis=0)

    def predict(self, test_data_input) -> List[QuantileForecast]:
        forecasts: List[QuantileForecast] = []
        input_entries = [_extract_input_entry(x) for x in list(test_data_input)]

        for batch in tqdm(batcher(input_entries, batch_size=self.batch_size)):
            for item in batch:
                target = np.asarray(item["target"], dtype=np.float32).reshape(-1)
                if target.shape[0] < 2:
                    raise ValueError("TimesFM2p0Adapter requires at least 2 history points.")
                if np.isnan(target).any():
                    target = self._linear_impute_nan(target)
                if self.max_context > 0 and target.shape[0] > self.max_context:
                    target = target[-self.max_context :]

                contexts.append(
                    torch.tensor(target, dtype=torch.float32, device=self.device)
                )
                freq_inputs.append(self._freq_to_category(self._extract_freq(item)))
                batch_meta.append(item)

            with torch.no_grad():
                outputs = self._timesfm_model(
                    past_values=contexts,
                    freq=freq_inputs,
                    return_dict=True,
                )

            full_predictions = outputs.full_predictions.detach().cpu().float().numpy()
            if full_predictions.shape[1] < self.prediction_length:
                raise RuntimeError(
                    "TimesFM_2p0_500m returned fewer points than prediction_length."
                )

            for full_preds_item, item in zip(full_predictions, batch_meta):
                forecast_arrays = self._compose_forecast_arrays(full_preds_item)
                forecasts.append(
                    _to_deterministic_quantile_forecast(
                        point_prediction[: self.prediction_length],
                        item["start"] + len(item["target"]),
                        self.quantile_levels,
                    )
                )
        return forecasts


@dataclass
class VisionTSppAdapter:
    """VisionTS++ 适配器：直接调用 `visionts.VisionTSpp` 模型，输出 9 分位数预测。

    参照 `Eval/visiontspp.py` 的推理流程：
      1. 按 ckpt_dir / model_size 解析权重文件路径，缺失时通过 HuggingFace 下载
      2. 构造 VisionTSpp 模型（quantile=True、color=True），在预测前调用
         update_config(context_len, pred_len, periodicity, ...)
      3. 输出 [median, quantile_list] 后合并为 9 分位数的 QuantileForecast
    """

    prediction_length: int
    num_samples: int = 100  # 为了统一接口保留（VisionTS++ 不使用采样）
    batch_size: int = 32
    device: str = "cpu"
    model_name: str = "visiontspp-local"
    quantile_levels: Optional[List[float]] = None
    model_size: str = "base"
    context_length: int = 4000
    ckpt_dir: str = "./hf_models/VisionTSpp"
    num_patch_input: int = 7
    padding_mode: str = "constant"
    max_vars_per_pass: int = 16

    def __post_init__(self):
        try:
<<<<<<< HEAD
            from visionts import VisionTSpp
            import visionts.util as visionts_util
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "visionts 未安装。请先安装 VisionTSpp 包（参考 src/visionts）"
            ) from exc

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "huggingface_hub 未安装。请先安装: pip install huggingface_hub"
            ) from exc
=======
            from huggingface_hub import hf_hub_download
            from visionts import VisionTSpp, freq_to_seasonality_list
        except ImportError as exc:
            raise ImportError(
                "VisionTSppAdapter requires visionts and huggingface_hub. "
                "Please install: pip install visionts huggingface_hub"
            ) from exc

        self._freq_to_seasonality_list = freq_to_seasonality_list

        model_spec = (self.model_name or "").strip() or "Lefei/VisionTSpp"
        model_spec = model_spec.replace("hf://", "")
        if "::" in model_spec:
            repo_id, ckpt_name = model_spec.split("::", 1)
        else:
            repo_id, ckpt_name = model_spec, "visiontspp_model.ckpt"
        ckpt_name = ckpt_name.strip() or "visiontspp_model.ckpt"

        try:
            ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
        except Exception as exc:
            raise RuntimeError(
                f"VisionTS++ checkpoint download failed from repo '{repo_id}', file '{ckpt_name}'."
            ) from exc

        arch = "mae_large" if "large" in ckpt_name.lower() else "mae_base"
        self._vision_model = VisionTSpp(
            arch=arch,
            ckpt_path=ckpt_path,
            quantile=True,
            clip_input=True,
            complete_no_clip=False,
            color=False,
        ).to(self.device)
        self._vision_model.eval()
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f

        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS

<<<<<<< HEAD
        size_key = str(self.model_size).lower()
        if size_key == "base":
            arch = "mae_base"
            ckpt_filename = "visiontspp_base.ckpt"
        elif size_key == "large":
            arch = "mae_large"
            ckpt_filename = "visiontspp_large.ckpt"
        else:
            raise ValueError(
                f"model_size 必须是 'base' 或 'large'，当前值: {self.model_size}"
            )

        ckpt_path = os.path.join(self.ckpt_dir, ckpt_filename)
        if not os.path.exists(ckpt_path):
            logger.info("Downloading VisionTSpp checkpoint to %s", self.ckpt_dir)
            snapshot_download(
                repo_id="Lefei/VisionTSpp",
                local_dir=self.ckpt_dir,
                local_dir_use_symlinks=False,
            )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"VisionTSpp checkpoint missing after download: {ckpt_path}"
                )

        if self.model_name != "visiontspp-local":
            logger.warning(
                "VisionTSppAdapter currently ignores model_name=%s; using local checkpoint.",
                self.model_name,
            )
=======
    @staticmethod
    def _extract_freq(item: Dict[str, Any]) -> str:
        start = item.get("start")
        freq = getattr(start, "freqstr", None)
        if freq:
            return freq
        freq_obj = getattr(start, "freq", None)
        return getattr(freq_obj, "freqstr", "H") or "H"

    def _resolve_periodicity(self, freq: str) -> int:
        try:
            seasonalities = self._freq_to_seasonality_list(freq)
        except Exception:
            seasonalities = [1]
        if not seasonalities:
            return 1
        return max(1, int(seasonalities[0]))

    def _compose_quantile_forecast_arrays(self, outputs: Any) -> np.ndarray:
        levels = self.quantile_levels or DEFAULT_QUANTILE_LEVELS

        if isinstance(outputs, (list, tuple)):
            median_tensor = outputs[0]
            quantile_tensors = outputs[1] if len(outputs) > 1 else []
        else:
            median_tensor = outputs
            quantile_tensors = []

        median_arr = (
            median_tensor.detach().cpu().float().numpy()[0, : self.prediction_length, 0]
        )
        quantile_map: Dict[str, np.ndarray] = {"0.5": median_arr}

        if quantile_tensors:
            q_arrays = [
                q.detach().cpu().float().numpy()[0, : self.prediction_length, 0]
                for q in quantile_tensors
            ]
            non_median_levels = [q for q in levels if float(q) != 0.5]
            if len(q_arrays) == len(non_median_levels):
                for q, arr in zip(non_median_levels, q_arrays):
                    quantile_map[str(q)] = arr
            elif len(q_arrays) == len(levels):
                for q, arr in zip(levels, q_arrays):
                    quantile_map[str(q)] = arr
            else:
                logger.warning(
                    "VisionTS++ quantile output count (%s) does not match requested quantiles (%s). "
                    "Falling back to median for missing levels.",
                    len(q_arrays),
                    len(levels),
                )
                for idx, q in enumerate(non_median_levels):
                    if idx < len(q_arrays):
                        quantile_map[str(q)] = q_arrays[idx]

        arrays = [
            quantile_map.get(str(q), median_arr)[: self.prediction_length] for q in levels
        ]
        return np.stack(arrays, axis=0)
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f

        self._visionts_util = visionts_util
        self.model = VisionTSpp(
            arch,
            ckpt_path=ckpt_path,
            quantile=True,
            clip_input=True,
            complete_no_clip=False,
            color=True,
        ).to(self.device)
        self.model.eval()

<<<<<<< HEAD
    # ---------------------------- helpers ---------------------------- #

    @staticmethod
    def _normalize_offset_name(name: str) -> str:
        base = name.split("-")[0]
        base_lower = base.lower()
        if base_lower == "min":
            return "T"
        if base_lower == "h":
            return "H"
        if base_lower == "s":
            return "S"
        if base_lower == "d":
            return "D"
        if base_lower == "w":
            return "W"
        if base_lower in ("me", "ms", "m"):
            return "M"
        if base_lower.startswith("q"):
            return "Q"
        if base_lower.startswith(("y", "a")):
            return "A"
        if base_lower == "b":
            return "B"
        return base.upper()

    def _resolve_periodicity(self, freq: Optional[str]) -> int:
        """将频率映射为周期。优先用 visionts.util.POSSIBLE_SEASONALITIES。"""
        if not freq:
            return 1

        try:
            offset = pd.tseries.frequencies.to_offset(freq)
            base = self._normalize_offset_name(offset.name)
            base_seasonality_list = self._visionts_util.POSSIBLE_SEASONALITIES.get(
                base, []
            )
            candidates: List[int] = []
            for base_season in base_seasonality_list:
                seasonality, remainder = divmod(base_season, max(offset.n, 1))
                if not remainder:
                    candidates.append(seasonality)
            candidates.append(1)
            return candidates[0]
        except Exception:
            try:
                from visionts import freq_to_seasonality_list

                return freq_to_seasonality_list(freq)[0]
            except Exception:
                return 1

    @staticmethod
    def _clean_nan_target(target: np.ndarray) -> np.ndarray:
        arr = np.asarray(target, dtype=np.float64)
        if not np.any(np.isnan(arr)):
            return arr
        if arr.ndim == 1:
            return np.asarray(LastValueImputation()(arr), dtype=np.float64)
        imputed_rows = [LastValueImputation()(row) for row in arr]
        return np.asarray(np.vstack(imputed_rows), dtype=np.float64)

    def _run_inference(
        self, input_tensor: torch.Tensor, periodicity: int
    ) -> List[Any]:
        """与 visiontspp.py 中的 run_inference 等价，支持大变量数下分段。"""
        curr_ctx_len = input_tensor.shape[1]
        nvars_input = input_tensor.shape[2]
        max_vars = max(1, int(self.max_vars_per_pass))

        def _run_single_pass(tensor_chunk: torch.Tensor):
            chunk_vars = tensor_chunk.shape[2]
            self.model.update_config(
                context_len=curr_ctx_len,
                pred_len=self.prediction_length,
                periodicity=periodicity,
                num_patch_input=self.num_patch_input,
                padding_mode=self.padding_mode,
            )
            color_list = [i % 3 for i in range(chunk_vars)]
            with torch.no_grad():
                return self.model(
                    tensor_chunk, export_image=False, color_list=color_list
                )

        if nvars_input <= max_vars:
            return _run_single_pass(input_tensor)

        median_chunks: List[torch.Tensor] = []
        quantile_chunks: Optional[List[List[torch.Tensor]]] = None
        for start in range(0, nvars_input, max_vars):
            end = min(start + max_vars, nvars_input)
            chunk_output = _run_single_pass(input_tensor[:, :, start:end])
            preds_data = (
                chunk_output[0] if isinstance(chunk_output, tuple) else chunk_output
            )
            med_chunk = preds_data[0]
            q_chunk_list = preds_data[1]

            median_chunks.append(med_chunk)
            if quantile_chunks is None:
                quantile_chunks = [[] for _ in range(len(q_chunk_list))]
            for qi, q_tensor in enumerate(q_chunk_list):
                quantile_chunks[qi].append(q_tensor)

        med_full = torch.cat(median_chunks, dim=2)
        q_full = [torch.cat(parts, dim=2) for parts in (quantile_chunks or [])]
        return [med_full, q_full]

    @staticmethod
    def _build_forecast_array(
        medians: np.ndarray, q_np_list: List[np.ndarray], idx: int
    ) -> np.ndarray:
        """将 VisionTS++ 的 (median, 8 quantiles) 输出拼接为 [9, nvars, pred_len]。"""
        qs_sample = [q[idx] for q in q_np_list]
        qs_stacked = np.stack(qs_sample, axis=0)  # [8, pred_len, nvars]
        med_expanded = np.expand_dims(medians[idx], axis=0)  # [1, pred_len, nvars]
        full_quantiles = np.concatenate(
            [qs_stacked[:4], med_expanded, qs_stacked[4:]], axis=0
        )  # [9, pred_len, nvars]
        return full_quantiles.transpose(0, 2, 1)  # [9, nvars, pred_len]

    def _process_tensor(
        self, tensor: torch.Tensor, periodicity: int
    ) -> List[np.ndarray]:
        output = self._run_inference(tensor, periodicity)
        preds_data = output[0] if isinstance(output, tuple) else output
        medians = preds_data[0].detach().cpu().numpy()
        q_np_list = [q.detach().cpu().numpy() for q in preds_data[1]]
        return [
            self._build_forecast_array(medians, q_np_list, i)
            for i in range(medians.shape[0])
        ]

    # ---------------------------- predict ---------------------------- #

    def predict(self, test_data_input) -> List[QuantileForecast]:
        input_entries = [_extract_input_entry(x) for x in list(test_data_input)]
        if not input_entries:
            return []

        # 推断 freq & periodicity（eval_pipeline 保证所有 entry 同频率）
        freq: Optional[str] = None
        for item in input_entries:
            start = item.get("start")
            if start is not None:
                freq = getattr(start, "freqstr", None)
                if freq:
                    break
        periodicity = self._resolve_periodicity(freq)
        logger.info(
            "VisionTSpp inference: freq=%s, periodicity=%s, pred_len=%s",
            freq,
            periodicity,
            self.prediction_length,
        )

        # 每个 adapter entry 都是单变量 1D target → 转为 [seq_len, 1] 张量
        context_list: List[torch.Tensor] = []
        for item in input_entries:
            target = np.asarray(item["target"])
            if target.shape[-1] > self.context_length:
                target = target[..., -self.context_length :]
            target = self._clean_nan_target(target)
            if target.ndim == 1:
                target = target[np.newaxis, :]
            context_list.append(torch.tensor(target.T).float())

        fc_quantiles: List[np.ndarray] = []
        total_items = len(context_list)

        for start in tqdm(
            range(0, total_items, self.batch_size), desc="VisionTSpp"
        ):
            end = min(start + self.batch_size, total_items)
            batch_list = context_list[start:end]

            try:
                batch_input = torch.stack(batch_list).to(self.device)
                fc_quantiles.extend(self._process_tensor(batch_input, periodicity))
            except RuntimeError as exc:
                # 序列长度不一致或显存不足时，回退为逐样本推理
                if "stack" not in str(exc) and "out of memory" not in str(exc).lower():
                    raise
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache() if self.device.startswith("cuda") else None
                for item_tensor in batch_list:
                    single_input = item_tensor.unsqueeze(0).to(self.device)
                    fc_quantiles.extend(
                        self._process_tensor(single_input, periodicity)
=======
        for batch in tqdm(batcher(input_entries, batch_size=self.batch_size)):
            for item in batch:
                target = np.asarray(item["target"], dtype=np.float32).reshape(-1)
                if target.shape[0] < 2:
                    raise ValueError("VisionTSppAdapter requires at least 2 history points.")
                if np.isnan(target).any():
                    target = LastValueImputation()(target)

                context = torch.tensor(target, dtype=torch.float32).reshape(1, -1, 1)
                periodicity = self._resolve_periodicity(self._extract_freq(item))
                self._vision_model.update_config(
                    context_len=context.shape[1],
                    pred_len=self.prediction_length,
                    periodicity=periodicity,
                )

                with torch.no_grad():
                    outputs = self._vision_model(context.to(self.device))
                forecast_arrays = self._compose_quantile_forecast_arrays(outputs)
                if forecast_arrays.shape[1] < self.prediction_length:
                    raise RuntimeError(
                        "VisionTSpp returned fewer points than prediction_length."
                    )
                forecasts.append(
                    QuantileForecast(
                        forecast_arrays=forecast_arrays,
                        forecast_keys=list(map(str, self.quantile_levels)),
                        start_date=item["start"] + len(item["target"]),
>>>>>>> 4b363ae983185f1ce151e3cc419529a001f6d12f
                    )

        forecasts: List[QuantileForecast] = []
        q_levels = self.quantile_levels or DEFAULT_QUANTILE_LEVELS
        for raw_array, entry in zip(fc_quantiles, input_entries):
            # raw_array shape: [9, nvars, pred_len]；单变量 nvars=1
            forecast_array = raw_array[:, 0, :] if raw_array.ndim == 3 else raw_array
            forecast_array = forecast_array[:, : self.prediction_length].astype(
                np.float64
            )
            forecast_start_date = entry["start"] + len(entry["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=forecast_array,
                    forecast_keys=list(map(str, q_levels)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts
