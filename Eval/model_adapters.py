from __future__ import annotations

import logging
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
        try:
            from .model.kairos_23m_forecastor import kairos_23m_forecastor
        except ImportError:
            from model.kairos_23m_forecastor import kairos_23m_forecastor

        self._forecastor = kairos_23m_forecastor
        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS
        if self.model_name != "mldi-lab/Kairos_23m":
            logger.warning(
                "Kairos23mAdapter currently ignores model_name=%s; using forecastor defaults.",
                self.model_name,
            )

    def predict(self, test_data_input) -> List[QuantileForecast]:
        forecasts: List[QuantileForecast] = []
        input_entries = [_extract_input_entry(x) for x in list(test_data_input)]

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
                        "Kairos_50m returned fewer points than prediction_length."
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
class TimesFM2p0Adapter:
    prediction_length: int
    num_samples: int = 100
    batch_size: int = 32
    device: str = "cpu"
    model_name: str = "google/timesfm-2.0-500m-pytorch"
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
        try:
            from .model.timesfm_2p0_500m_forecastor import timesfm_2p0_500m_forecastor
        except ImportError:
            from model.timesfm_2p0_500m_forecastor import timesfm_2p0_500m_forecastor

        self._forecastor = timesfm_2p0_500m_forecastor
        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS
        if self.model_name != "google/timesfm-2.0-500m-pytorch":
            logger.warning(
                "TimesFM2p0Adapter currently ignores model_name=%s; using forecastor defaults.",
                self.model_name,
            )

    def predict(self, test_data_input) -> List[QuantileForecast]:
        forecasts: List[QuantileForecast] = []
        input_entries = [_extract_input_entry(x) for x in list(test_data_input)]

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
                        "TimesFM_2p0_500m returned fewer points than prediction_length."
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
class VisionTSppAdapter:
    prediction_length: int
    num_samples: int = 100
    batch_size: int = 32
    device: str = "cpu"
    model_name: str = "visiontspp-local"
    quantile_levels: Optional[List[float]] = None

    def __post_init__(self):
        try:
            from .model.visiontspp_forecastor import visiontspp_forecastor
        except ImportError:
            from model.visiontspp_forecastor import visiontspp_forecastor

        self._forecastor = visiontspp_forecastor
        if self.quantile_levels is None:
            self.quantile_levels = DEFAULT_QUANTILE_LEVELS
        if self.model_name != "visiontspp-local":
            logger.warning(
                "VisionTSppAdapter currently ignores model_name=%s; using forecastor defaults.",
                self.model_name,
            )

    def predict(self, test_data_input) -> List[QuantileForecast]:
        forecasts: List[QuantileForecast] = []
        input_entries = [_extract_input_entry(x) for x in list(test_data_input)]

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
                        "VisionTSpp returned fewer points than prediction_length."
                    )
                forecasts.append(
                    _to_deterministic_quantile_forecast(
                        point_prediction[: self.prediction_length],
                        item["start"] + len(item["target"]),
                        self.quantile_levels,
                    )
                )
        return forecasts
