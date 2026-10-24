from __future__ import annotations

import numpy as np
import pandas as pd

from cwfc.data import Panel

TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
EXTRA_TARGETS = ("core_weighted_cpu", "peak_core_demand")
MIN_TRAIN_STEPS = 12


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def series(panel: Panel, target: str) -> np.ndarray:
    """Target series; a step with no finite readings is a telemetry gap, so every target is NaN there."""
    ok = np.isfinite(panel.avg)
    gap = ~ok.any(axis=0)
    cores = panel.static["cores"].to_numpy(dtype=float)[:, None]
    if target == "active_vm_count":
        return np.where(gap, np.nan, ok.sum(axis=0).astype(float))
    if target == "aggregate_cpu_utilization":
        return _ratio(np.where(ok, panel.avg, 0.0).sum(axis=0), ok.sum(axis=0).astype(float))
    if target == "core_weighted_cpu":
        return _ratio(
            np.where(ok, panel.avg * cores, 0.0).sum(axis=0), (ok * cores).sum(axis=0)
        )
    if target == "peak_core_demand":
        fin = np.isfinite(panel.max)
        demand = np.where(fin, panel.max * cores, 0.0).sum(axis=0) / 100.0
        return np.where(fin.any(axis=0), demand, np.nan)
    raise ValueError(f"unknown target {target!r}; valid: {list(TARGETS + EXTRA_TARGETS)}")


def build_targets(panel: Panel, include_extra: bool = False) -> pd.DataFrame:
    names = TARGETS + EXTRA_TARGETS if include_extra else TARGETS
    df = pd.DataFrame({"timestamp": panel.timestamps})
    for name in names:
        df[name] = series(panel, name)
    return df


def holdout_split(panel: Panel, horizon: int) -> tuple[Panel, np.ndarray]:
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if panel.T <= horizon + MIN_TRAIN_STEPS:
        raise ValueError(
            f"panel has {panel.T} steps; need at least {horizon + MIN_TRAIN_STEPS + 1} "
            f"for horizon {horizon}"
        )
    return panel.truncate(panel.T - horizon), panel.timestamps[-horizon:]
