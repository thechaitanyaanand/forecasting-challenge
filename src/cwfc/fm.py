from __future__ import annotations

from typing import Literal

import numpy as np

from cwfc.data import Panel
from cwfc.models import LGB_TARGETS, _filled

MEAN_LEVELS = [round(0.1 * i, 1) for i in range(1, 10)]  # decile grid approximates the mean


class FMUnavailableError(RuntimeError):
    pass


INSTALL_HINT = 'Foundation models need extra dependencies: pip install -e ".[fm]"'

# model_id -> (flagged, note with source); see reports/fm_contamination.md
CONTAMINATION: dict[str, tuple[bool, str]] = {
    "amazon/chronos-bolt-base": (False, "corpus undocumented, no cloud traces named: https://huggingface.co/amazon/chronos-bolt-base"),
    "amazon/chronos-2": (True, "pretrained on Azure VM 2017, Alibaba 2018, Borg 2011: https://arxiv.org/html/2510.15821"),
    "google/timesfm-2.5-200m-pytorch": (True, "GiftEvalPretrain includes CloudOps TSF (Azure/Borg/Alibaba): https://arxiv.org/html/2410.10393"),
}


def fm_available() -> bool:
    try:
        import chronos  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _ffill_rows(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    idx = np.maximum.accumulate(np.where(ok, np.arange(x.shape[1]), 0), axis=1)
    out = np.take_along_axis(x, idx, axis=1)
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), 0)
    lead = np.arange(x.shape[1]) < first[:, None]
    return np.where(lead, np.take_along_axis(out, first[:, None], axis=1), out)


class Chronos:
    complexity = 4

    def __init__(
        self, model_id: str = "amazon/chronos-bolt-base",
        mode: Literal["aggregate", "bottom_up"] = "aggregate",
        device: str = "auto", batch_size: int = 512,
    ):
        if mode not in ("aggregate", "bottom_up"):
            raise ValueError(f"mode must be 'aggregate' or 'bottom_up', got {mode!r}")
        if not fm_available():
            raise FMUnavailableError(INSTALL_HINT)
        self.model_id, self.mode, self.device, self.batch_size = model_id, mode, device, batch_size
        self.name = model_id.split("/")[-1].replace("-", "_") + ("_bottom_up" if mode == "bottom_up" else "")
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from chronos import BaseChronosPipeline

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            self._pipe = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device, torch_dtype=dtype)
        return self._pipe

    def _point(self, rows: np.ndarray, horizon: int, stat: Literal["median", "mean"] = "median") -> np.ndarray:
        import torch

        levels = [0.5] if stat == "median" else MEAN_LEVELS

        pipe = self._pipeline()
        out = []
        for i in range(0, len(rows), self.batch_size):
            ctx = torch.as_tensor(rows[i:i + self.batch_size], dtype=torch.float32)
            q, _ = pipe.predict_quantiles(list(ctx), prediction_length=horizon, quantile_levels=levels)
            q = torch.cat(q) if isinstance(q, list) else q
            out.append(q.reshape(len(ctx), horizon, len(levels)).float().mean(-1).cpu().numpy())
        return np.concatenate(out)

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray:
        if self.mode == "aggregate":
            return self._point(_filled(panel, target)[None, :], horizon)[0]
        if target not in LGB_TARGETS:
            raise ValueError(f"bottom_up supports {list(LGB_TARGETS)}, got {target!r}")
        active = np.isfinite(panel.avg[:, -1])
        if not active.any():
            return np.zeros(horizon)
        pred = self._point(_ffill_rows(panel.avg[active]), horizon, stat="mean")
        w = (panel.static["cores"].to_numpy(dtype=float)[active] if target == "core_weighted_cpu"
             else np.ones(len(pred)))
        return (pred * w[:, None]).sum(axis=0) / w.sum()


class TimesFM:
    complexity = 4

    def __init__(
        self, model_id: str = "google/timesfm-2.5-200m-pytorch", device: str = "auto",
        batch_size: int = 64, max_context: int = 1024,
    ):
        if not fm_available():
            raise FMUnavailableError(INSTALL_HINT)
        try:
            import timesfm  # noqa: F401
        except ImportError as e:
            raise FMUnavailableError('TimesFM needs: pip install -e ".[timesfm]"') from e
        self.model_id, self.device, self.batch_size, self.max_context = model_id, device, batch_size, max_context
        self.name = "timesfm_2_5"
        self._model, self._horizon = None, 0

    def _compiled(self, horizon: int):
        import timesfm
        import torch

        if self._model is None:
            self._model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.model_id, torch_compile=False)
            dev = self.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = dev
            self._model.model.device = torch.device(dev)
            self._model.model.to(dev)
        if horizon > self._horizon:
            self._model.compile(timesfm.ForecastConfig(
                max_context=self.max_context, max_horizon=horizon, normalize_inputs=True,
                per_core_batch_size=self.batch_size, use_continuous_quantile_head=True,
                fix_quantile_crossing=True))
            self._horizon = horizon
        return self._model

    def forecast_rows(self, rows: np.ndarray, horizon: int) -> np.ndarray:
        _, q = self._compiled(horizon).forecast(horizon=horizon, inputs=list(rows))
        return np.asarray(q)[:len(rows), :horizon, 5]

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray:
        return self.forecast_rows(_filled(panel, target)[None, :], horizon)[0]


def fm_candidates(target: str, include_flagged: bool = False) -> list[Chronos]:
    models = [Chronos("amazon/chronos-bolt-base"), Chronos("amazon/chronos-2")]
    if target in LGB_TARGETS:
        models.append(Chronos("amazon/chronos-bolt-base", mode="bottom_up"))
    return [m for m in models if include_flagged or not CONTAMINATION.get(m.model_id, (True, "unknown model: not vetted"))[0]]
