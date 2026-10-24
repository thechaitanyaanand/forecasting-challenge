from __future__ import annotations

import numpy as np
import pandas as pd

from cwfc.data import Panel

WINDOWS = (6, 12, 24)
EWM_SPAN = 12
FLEET_LAGS = 6
CAT_COLUMNS = ("vm_category", "memory_gb_bucket")


def _rolling(a: np.ndarray, o: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    win = a[:, max(0, o - w):o]
    ok = np.isfinite(win)
    cnt = ok.sum(axis=1)
    safe = np.maximum(cnt, 1)
    mean = np.where(ok, win, 0.0).sum(axis=1) / safe
    var = np.where(ok, (win - mean[:, None]) ** 2, 0.0).sum(axis=1) / safe
    none = cnt == 0
    return np.where(none, np.nan, mean), np.where(none, np.nan, np.sqrt(var))


def _ewm(avg: np.ndarray) -> np.ndarray:
    alpha = 2.0 / (EWM_SPAN + 1)
    out = np.full(avg.shape, np.nan)
    num = np.zeros(avg.shape[0])
    den = np.zeros(avg.shape[0])
    for t in range(avg.shape[1]):
        x = avg[:, t]
        ok = np.isfinite(x)
        num = np.where(ok, (1 - alpha) * num + np.where(ok, x, 0.0), num)
        den = np.where(ok, (1 - alpha) * den + 1.0, den)
        out[:, t] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return out


class _Features:
    def __init__(self, panel: Panel, lags: int):
        self.panel, self.lags = panel, lags
        self.ewm = _ewm(panel.avg)
        ok = np.isfinite(panel.avg)
        self.fleet = np.where(
            ok.any(axis=0),
            np.where(ok, panel.avg, 0.0).sum(axis=0) / np.maximum(ok.sum(axis=0), 1),
            np.nan,
        )
        self.names = (
            [f"lag_avg_{k}" for k in range(1, lags + 1)]
            + [f"lag_min_{k}" for k in range(1, lags + 1)]
            + [f"lag_max_{k}" for k in range(1, lags + 1)]
            + [f"roll_mean_{w}" for w in WINDOWS] + [f"roll_std_{w}" for w in WINDOWS]
            + ["ewm_12"] + [f"fleet_lag_{k}" for k in range(1, FLEET_LAGS + 1)]
        )
        static = panel.static
        self.cats = {c: pd.Categorical(static[c]).categories for c in CAT_COLUMNS}
        self.codes = {
            c: pd.Categorical(static[c], categories=self.cats[c]).codes for c in CAT_COLUMNS
        }
        self.cores = static["cores"].to_numpy(dtype=np.float32)

    def block(self, o: int) -> np.ndarray:
        p, n = self.panel, self.panel.avg.shape[0]
        nan = np.full(n, np.nan)
        cols = []
        for a in (p.avg, p.min, p.max):
            cols += [a[:, o - k] if o - k >= 0 else nan for k in range(1, self.lags + 1)]
        stats = [_rolling(p.avg, o, w) for w in WINDOWS]
        cols += [s[0] for s in stats] + [s[1] for s in stats]
        cols.append(self.ewm[:, o - 1] if o >= 1 else nan)
        cols += [
            np.full(n, self.fleet[o - k] if o - k >= 0 else np.nan)
            for k in range(1, FLEET_LAGS + 1)
        ]
        return np.stack(cols, axis=1).astype(np.float32)

    def frame(self, blocks: list[np.ndarray], hs: list[np.ndarray], vms: list[np.ndarray]) -> pd.DataFrame:
        idx = np.concatenate(vms).astype(int)
        df = pd.DataFrame(np.concatenate(blocks), columns=self.names)
        df["h"] = np.concatenate(hs).astype(np.float32)
        df["vm_category"] = pd.Categorical.from_codes(
            self.codes["vm_category"][idx], self.cats["vm_category"])
        df["cores"] = self.cores[idx]
        df["memory_gb_bucket"] = pd.Categorical.from_codes(
            self.codes["memory_gb_bucket"][idx], self.cats["memory_gb_bucket"])
        return df


def panel_training_frame(
    panel: Panel, horizon: int, lags: int, stride: int,
    max_rows: int = 2_000_000, seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rows (vm, origin o', h): features use columns < o' only, label is avg[vm, o'+h-1]."""
    T, n = panel.T, panel.avg.shape[0]
    fx = _Features(panel, lags)
    origins = [o for o in range(T - 1, lags - 1, -stride)]
    pairs = [(o, h) for o in origins for h in range(1, min(horizon, T - o) + 1)]
    total = sum(int(np.isfinite(panel.avg[:, o + h - 1]).sum()) for o, h in pairs)
    keep = np.ones(n, dtype=bool)
    if total > max_rows:
        m = max(1, int(n * max_rows / total))
        keep[:] = False
        keep[np.random.default_rng(seed).choice(n, size=m, replace=False)] = True
    blocks, hs, vms, ys, lbl = [], [], [], [], []
    for o in origins:
        b = fx.block(o)
        for h in range(1, min(horizon, T - o) + 1):
            y = panel.avg[:, o + h - 1]
            rows = np.flatnonzero(keep & np.isfinite(y))
            blocks.append(b[rows])
            hs.append(np.full(len(rows), h))
            vms.append(rows)
            ys.append(y[rows])
            lbl.append(np.full(len(rows), o + h - 1))
    if not blocks:
        empty = fx.frame([np.empty((0, len(fx.names)), np.float32)], [np.empty(0)], [np.empty(0)])
        return empty, np.empty(0), np.empty(0, dtype=int)
    return fx.frame(blocks, hs, vms), np.concatenate(ys), np.concatenate(lbl).astype(int)


def panel_inference_frame(panel: Panel, horizon: int, lags: int) -> pd.DataFrame:
    """Rows ordered h-major: for each h in 1..horizon, all VMs, at origin = panel.T."""
    fx = _Features(panel, lags)
    n = panel.avg.shape[0]
    b = fx.block(panel.T)
    return fx.frame(
        [b] * horizon, [np.full(n, h) for h in range(1, horizon + 1)], [np.arange(n)] * horizon)
