"""
模型适配层：为不同模型提供统一的预测接口。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Protocol

import numpy as np
import torch
from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast, SampleForecast
from gluonts.transform import LastValueImputation
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM


logger = logging.getLogger(__name__)


class ForecastAdapter(Protocol):
    """统一预测器接口。"""

    def predict(self, test_data_input) -> List[Any]: ...


@dataclass
class SundialAdapter:
    """Sundial 预测器适配器。"""

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
    """Chronos-2 预测器适配器。"""

    prediction_length: int
    batch_size: int = 32
    model_name: str = "amazon/chronos-2"
    device: str = "cpu"
    quantile_levels: Optional[List[float]] = None
    predict_batches_jointly: bool = False
    torch_dtype: Optional[str] = None

    def __post_init__(self):
        try:
            from chronos import BaseChronosPipeline, Chronos2Pipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "chronos-forecasting 未安装。请先安装: pip install chronos-forecasting>=2.1"
            ) from exc

        if self.quantile_levels is None:
            self.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
            packed.append({"target": item["target"]})
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
                # [batch, variates, seq_len, quantiles] -> [batch, quantiles, seq_len, variates]
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
        for item, ts in zip(quantiles, input_entries):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecasts.append(
                QuantileForecast(
                    forecast_arrays=item,
                    forecast_keys=list(map(str, self.quantile_levels)),
                    start_date=forecast_start_date,
                )
            )
        return forecasts
