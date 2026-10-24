from __future__ import annotations

import numpy as np
from scipy import stats


def mae(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denom = np.maximum(np.abs(a) + np.abs(b), eps)
    return float(np.mean(2.0 * np.abs(a - b) / denom) * 100.0)


def normalized_mae(y_true, y_pred, baseline_pred) -> float:
    model_mae = mae(y_true, y_pred)
    baseline_mae = mae(y_true, baseline_pred)
    if baseline_mae == 0:
        return 0.0 if model_mae == 0 else float("inf")
    return float(model_mae / baseline_mae)


def mase(y_true, y_pred, insample, m: int = 1) -> float:
    ins = np.asarray(insample, dtype=float)
    if not 0 < m < len(ins):
        raise ValueError("insample must be longer than the seasonal lag m >= 1")
    scale = float(np.mean(np.abs(ins[m:] - ins[:-m])))
    err = mae(y_true, y_pred)
    if scale == 0:
        return 0.0 if err == 0 else float("inf")
    return err / scale


def pinball(y_true, q_pred, q: float) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(q_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError("y_true and q_pred must have the same shape")
    d = a - b
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def coverage(y, lo, hi) -> float:
    a = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(lo, dtype=float) <= a) & (a <= np.asarray(hi, dtype=float))))


def diebold_mariano(e1, e2, h: int = 1, power: int = 2) -> tuple[float, float]:
    """Diebold-Mariano test; positive statistic means e1 is worse.

    Newey-West (Bartlett-kernel) long-run variance with the Harvey-Leybourne-Newbold
    small-sample correction and Student t(n-1) p-value.
    """
    a = np.asarray(e1, dtype=float)
    b = np.asarray(e2, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("e1 and e2 must be 1-D arrays of the same length")
    n = len(a)
    if n < 2 or h < 1 or h > n - 1:
        raise ValueError(f"need n >= 2 and 1 <= h <= n-1, got n={n}, h={h}")
    d = np.abs(a) ** power - np.abs(b) ** power
    mean = float(d.mean())
    c = d - mean
    g0 = float(c @ c) / n
    if g0 <= 0:
        return (0.0, 1.0) if mean == 0 else (float(np.sign(mean) * np.inf), 0.0)
    lrv = g0 + 2 * sum((1 - k / h) * float(c[k:] @ c[:-k]) / n for k in range(1, h))
    var = (lrv if lrv > 0 else g0) / n
    stat = mean / np.sqrt(var) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))
