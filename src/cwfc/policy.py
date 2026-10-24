from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def conformal_upper(errors: np.ndarray, level: float, pool: int = 2) -> np.ndarray:
    """Per-horizon upper offset: split-conformal quantile of signed residuals (actual - pred).

    Residuals are pooled over horizons h +/- pool (clipped at the ends); the quantile is the
    ceil((n + 1) * level)-th smallest, capped at the maximum.
    """
    e = np.asarray(errors, dtype=float)
    if e.ndim != 2 or e.size == 0:
        raise ValueError(f"errors must be a non-empty (folds, horizon) array, got shape {e.shape}")
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")
    H = e.shape[1]
    out = np.empty(H)
    for h in range(H):
        r = e[:, max(0, h - pool):min(H, h + pool + 1)].ravel()
        r = np.sort(r[np.isfinite(r)])
        if not len(r):
            raise ValueError(f"no finite residuals at horizon {h + 1}")
        k = min(int(np.ceil((len(r) + 1) * level)), len(r))
        out[h] = r[k - 1]
    return out


def conformal_interval(
    errors: np.ndarray, level: float, pool: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided offsets (lower, upper) to add to the point forecast."""
    q = (1 + level) / 2
    return -conformal_upper(-np.asarray(errors, dtype=float), q, pool), conformal_upper(errors, q, pool)


def tradeoff(
    errors: np.ndarray, levels: Sequence[float], under_cost: float = 10.0, over_cost: float = 1.0,
    pool: int = 2,
) -> pd.DataFrame:
    """Breach / headroom / cost per provisioning level, relative to the point forecast.

    Calibration view: offsets are fitted on the same fold errors they are scored on, so
    this is in-sample, not an out-of-sample estimate.
    """
    e = np.asarray(errors, dtype=float)
    rows = []
    for level in levels:
        gap = conformal_upper(e, level, pool)[None, :] - e  # provision - actual
        gap = gap[np.isfinite(gap)]
        breach = gap < 0
        rows.append({
            "level": level,
            "breach_rate": float(breach.mean()),
            "mean_headroom": float(gap[~breach].mean()) if (~breach).any() else 0.0,
            "expected_cost": float(
                under_cost * np.maximum(-gap, 0).mean() + over_cost * np.maximum(gap, 0).mean()
            ),
        })
    return pd.DataFrame(rows)
