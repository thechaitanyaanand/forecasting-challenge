from __future__ import annotations

import warnings
from typing import Literal

import lightgbm as lgb
import numpy as np

from cwfc.backtest import Forecaster
from cwfc.data import Panel
from cwfc.features import panel_inference_frame, panel_training_frame
from cwfc.targets import series

MIN_TRAIN_ROWS = 200
LGB_TARGETS = ("aggregate_cpu_utilization", "core_weighted_cpu")


def _filled(panel: Panel, target: str) -> np.ndarray:
    y = series(panel, target)
    ok = np.isfinite(y)
    if not ok.any():
        return np.zeros_like(y)
    y = y[np.maximum.accumulate(np.where(ok, np.arange(len(y)), 0))]
    first = np.flatnonzero(ok)[0]
    y[:first] = y[first]
    return y


class Persistence:
    name, complexity = "persistence", 0

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray:
        return np.full(horizon, _filled(panel, target)[-1])


class ExpandingMean:
    name, complexity = "expanding_mean", 1

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray:
        return np.full(horizon, _filled(panel, target).mean())


class RollingMean:
    complexity = 1

    def __init__(self, window: int):
        self.window = window
        self.name = f"rolling_mean_{window}"

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray:
        y = series(panel, target)
        y = y[np.isfinite(y)][-self.window:]
        return np.full(horizon, y.mean() if len(y) else 0.0)


class Drift:
    name, complexity = "drift", 1

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray:
        y = _filled(panel, target)
        slope = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.0
        return y[-1] + slope * np.arange(1, horizon + 1)


class StatsForecast:
    complexity = 2

    def __init__(self, kind: Literal["ets", "arima"]):
        if kind not in ("ets", "arima"):
            raise ValueError(f"kind must be 'ets' or 'arima', got {kind!r}")
        self.kind, self.name = kind, kind
        self.fallback_used = False

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray:
        from statsforecast.models import AutoARIMA, AutoETS

        y = _filled(panel, target)
        self.fallback_used = False
        if len(y) >= 8:
            try:
                model = (AutoETS if self.kind == "ets" else AutoARIMA)(season_length=1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = np.asarray(model.forecast(y=y, h=horizon)["mean"], dtype=float)
                if out.shape == (horizon,) and np.isfinite(out).all():
                    return out
            except Exception:
                pass
        self.fallback_used = True
        return np.full(horizon, y[-1])


class LightGBMPanel:
    """Per-VM direct multi-horizon LightGBM, aggregated to the fleet target.

    Assumes VMs with a finite avg in the last observed column stay active over the horizon.
    """

    name, complexity = "lightgbm_panel", 3

    def __init__(
        self, lags: int = 12, origin_stride: int = 6, params: dict | None = None,
        seed: int = 0, max_train_rows: int = 2_000_000,
    ):
        self.lags, self.origin_stride, self.seed = lags, origin_stride, seed
        self.max_train_rows = max_train_rows
        self.params = dict(
            objective="l2", learning_rate=0.05, num_leaves=63, n_estimators=400,
            min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            seed=seed, verbose=-1, n_jobs=-1, deterministic=True,
        ) | (params or {})
        self.fallback_used = False
        self.model_: lgb.Booster | None = None
        self.feature_names_: list[str] = []

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray:
        if target not in LGB_TARGETS:
            raise ValueError(f"LightGBMPanel supports {list(LGB_TARGETS)}, got {target!r}")
        self.fallback_used = False
        X, y, _ = panel_training_frame(
            panel, horizon, self.lags, self.origin_stride, self.max_train_rows, self.seed)
        active = np.isfinite(panel.avg[:, -1])
        if len(y) < MIN_TRAIN_ROWS or not active.any():
            self.fallback_used = True
            return Persistence().predict(panel, target, horizon)
        self.model_ = lgb.train(dict(self.params), lgb.Dataset(X, y))
        self.feature_names_ = list(X.columns)
        n = len(active)
        pred = self.model_.predict(panel_inference_frame(panel, horizon, self.lags))
        pred = pred.reshape(horizon, n)
        w = (panel.static["cores"].to_numpy(dtype=float) if target == "core_weighted_cpu"
             else np.ones(n))[active]
        return (pred[:, active] * w).sum(axis=1) / w.sum()


def default_candidates(
    target: str, with_fm: bool = False, include_lightgbm: bool = True,
) -> list[Forecaster]:
    models: list[Forecaster] = [
        Persistence(), ExpandingMean(), RollingMean(6), RollingMean(12), RollingMean(24), Drift(),
        StatsForecast("ets"), StatsForecast("arima"),
    ]
    if include_lightgbm and target in LGB_TARGETS:
        models.append(LightGBMPanel())
    if with_fm:
        from cwfc.fm import fm_candidates

        models.extend(fm_candidates(target))
    return models
