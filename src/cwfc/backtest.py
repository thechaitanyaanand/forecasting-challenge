from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from cwfc.data import Panel
from cwfc.metrics import diebold_mariano
from cwfc.targets import MIN_TRAIN_STEPS, series


class Forecaster(Protocol):
    name: str
    complexity: int

    def predict(self, panel: Panel, target: str, horizon: int) -> np.ndarray: ...


def default_origins(
    n_train: int, horizon: int, n_folds: int = 20, step: int = 2,
    min_history: int = MIN_TRAIN_STEPS,
) -> list[int]:
    origins = list(range(n_train - horizon, min_history - 1, -step))[:n_folds]
    if not origins:
        raise ValueError(
            f"no origins: n_train must be at least horizon + min_history = {horizon + min_history}, "
            f"got {n_train}"
        )
    return origins[::-1]


@dataclass
class BacktestResult:
    name: str
    target: str
    origins: list[int]
    preds: np.ndarray
    actuals: np.ndarray
    seconds: float
    complexity: int = 0

    @property
    def errors(self) -> np.ndarray:
        return self.actuals - self.preds

    @property
    def fold_mae(self) -> np.ndarray:
        e = np.abs(self.errors)
        fin = np.isfinite(e)
        n = fin.sum(axis=1)
        return np.where(n > 0, np.where(fin, e, 0.0).sum(axis=1) / np.maximum(n, 1), np.nan)

    @property
    def mae(self) -> float:
        e = self.errors[np.isfinite(self.errors)]
        return float(np.abs(e).mean()) if len(e) else float("nan")

    @property
    def rmse(self) -> float:
        e = self.errors[np.isfinite(self.errors)]
        return float(np.sqrt((e ** 2).mean())) if len(e) else float("nan")


def backtest(
    model: Forecaster, panel: Panel, target: str, horizon: int, origins: list[int],
    min_history: int = MIN_TRAIN_STEPS,
) -> BacktestResult:
    y = series(panel, target)
    preds, actuals, seconds = [], [], 0.0
    for o in origins:
        if o < min_history or o + horizon > panel.T:
            raise ValueError(
                f"origin {o} invalid: need {min_history} <= origin and origin + {horizon} <= {panel.T}"
            )
        t0 = time.perf_counter()
        p = np.asarray(model.predict(panel.truncate(o), target, horizon), dtype=float)
        seconds += time.perf_counter() - t0
        if p.shape != (horizon,) or not np.isfinite(p).all():
            raise ValueError(
                f"model {model.name!r} returned invalid forecast at origin {o}: "
                f"shape {p.shape}, expected ({horizon},) and all finite"
            )
        preds.append(p)
        actuals.append(y[o:o + horizon])
    return BacktestResult(
        model.name, target, list(origins), np.array(preds), np.array(actuals), seconds,
        model.complexity,
    )


@dataclass
class Selection:
    name: str
    reason: str
    table: pd.DataFrame


def select(results: list[BacktestResult], alpha: float = 0.05) -> Selection:
    if not results:
        raise ValueError("no results to select from")
    ref = results[0]
    if any(r.origins != ref.origins or r.target != ref.target for r in results):
        raise ValueError("all results must share target and origins")
    n_folds, horizon = ref.preds.shape
    mae_ = {r.name: (r.mae if np.isfinite(r.mae) else math.inf) for r in results}
    best = min(results, key=lambda r: (mae_[r.name], r.complexity, r.name))
    step = max(1, int(np.diff(ref.origins).min())) if n_folds > 1 else 1

    pvals: dict[str, float] = {}
    single = n_folds == 1
    if not single:
        bf = best.fold_mae
        for r in results:
            if r is best:
                continue
            ok = np.isfinite(r.fold_mae) & np.isfinite(bf)
            if ok.sum() < 2:
                single = True
                break
            h = min(int(ok.sum()) - 1, math.ceil(horizon / step))
            pvals[r.name] = diebold_mariano(r.fold_mae[ok], bf[ok], h=h, power=1)[1]

    if single:
        near = [r for r in results if mae_[r.name] <= mae_[best.name] * 1.01]
        chosen = min(near, key=lambda r: (r.complexity, mae_[r.name], r.name))
        reason = (
            f"single fold, no significance test; simplest within 1% of best MAE ({best.name})"
        )
    else:
        chosen, reason = best, ""
        simpler = sorted(
            (r for r in results if r.complexity < best.complexity),
            key=lambda r: (r.complexity, mae_[r.name], r.name),
        )
        for r in simpler:
            if pvals[r.name] >= alpha:
                chosen = r
                reason = f"not significantly worse than {best.name} (DM p={pvals[r.name]:.3f})"
                break
        if chosen is best:
            ps = ", ".join(f"{r.name}: {pvals[r.name]:.3g}" for r in simpler)
            reason = (
                f"lowest MAE; simpler models significantly worse (DM p: {ps})" if ps
                else "lowest MAE; no simpler candidates"
            )

    table = pd.DataFrame({
        "name": [r.name for r in results],
        "complexity": [r.complexity for r in results],
        "mae": [r.mae for r in results],
        "rmse": [r.rmse for r in results],
        "dm_p_vs_best": [pvals.get(r.name, np.nan) for r in results],
        "selected": [r is chosen for r in results],
    }).sort_values("mae", kind="stable").reset_index(drop=True)
    return Selection(chosen.name, reason, table)
