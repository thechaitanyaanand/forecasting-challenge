"""Full model comparison on the TRAIN panel only (never the holdout); feeds the analysis notebook."""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.backtest import BacktestResult, backtest, default_origins
from cwfc.data import DataContractError, build_panel, load_metadata, load_readings
from cwfc.features import panel_inference_frame
from cwfc.fm import (
    CONTAMINATION, INSTALL_HINT, FMUnavailableError, _ffill_rows, fm_available, fm_candidates,
)
from cwfc.metrics import diebold_mariano, mase
from cwfc.models import LightGBMPanel, default_candidates
from cwfc.predict import _long
from cwfc.targets import holdout_split, series

CPU, COUNT = "aggregate_cpu_utilization", "active_vm_count"
LATENCY_SERIES, LATENCY_REPEATS = 1000, 3


def _flagged(model) -> bool:
    return CONTAMINATION.get(getattr(model, "model_id", ""), (False, ""))[0]


def _timesfm() -> list:
    try:
        from cwfc.fm import TimesFM

        return [TimesFM()]
    except FMUnavailableError:
        print("timesfm not installed: skipped")
        return []


def _candidates(target: str, with_fm: bool, lightgbm: bool) -> list:
    cands = default_candidates(target, include_lightgbm=lightgbm)
    if with_fm:
        cands += fm_candidates(target, include_flagged=True) + _timesfm()
    return cands


def _summary(results: list[BacktestResult], models: list, insample: np.ndarray) -> pd.DataFrame:
    by = {m.name: m for m in models}
    rows = []
    for r in results:
        e = r.errors[np.isfinite(r.errors)]
        flagged = _flagged(by[r.name])
        rows.append({
            "model": r.name, "complexity": r.complexity, "mae": r.mae, "rmse": r.rmse,
            "mase": mase(np.zeros(len(e)), -e, insample), "mean_error": float(e.mean()),
            "seconds": r.seconds, "flagged": flagged, "eligible": not flagged,
        })
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def _dm_matrix(results: list[BacktestResult]) -> pd.DataFrame:
    names = [r.name for r in results]
    step = max(1, int(np.diff(results[0].origins).min())) if len(results[0].origins) > 1 else 1
    horizon = results[0].preds.shape[1]
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for a in results:
        for b in results:
            if a is b:
                continue
            ok = np.isfinite(a.fold_mae) & np.isfinite(b.fold_mae)
            if ok.sum() >= 2:
                h = min(int(ok.sum()) - 1, math.ceil(horizon / step))
                out.loc[a.name, b.name] = diebold_mariano(a.fold_mae[ok], b.fold_mae[ok], h=h, power=1)[1]
    return out


def _median_seconds(fn) -> float:
    ts = []
    for _ in range(LATENCY_REPEATS):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _latency(train, horizon: int, with_fm: bool, lightgbm: bool) -> pd.DataFrame:
    """Inference time for 1,000 series (weights loaded / models fitted outside the timing)."""
    sub = train.subset(np.arange(len(train.vm_ids)) < LATENCY_SERIES)
    rows = _ffill_rows(sub.avg)
    entries = [("persistence", "cpu", lambda: np.repeat(rows[:, -1:], horizon, axis=1))]
    if lightgbm:
        lgbm = LightGBMPanel()
        lgbm.predict(train, CPU, horizon)
        frame = panel_inference_frame(sub, horizon, lgbm.lags)
        entries.append(("lightgbm_panel", "cpu", lambda: lgbm.model_.predict(frame)))
    if with_fm:
        import torch

        for dev in (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]):
            for m in fm_candidates(CPU, include_flagged=True) + _timesfm():
                m.device = dev
                f = (lambda m=m: m._point(rows, horizon)) if hasattr(m, "_point") else (
                    lambda m=m: m.forecast_rows(rows, horizon))
                f()  # warm-up outside the timing
                entries.append((m.name, dev, f))
    out = []
    for name, dev, fn in entries:
        s = _median_seconds(fn)
        out.append({"model": name, "device": dev, "n_series": len(rows), "seconds": s,
                    "ms_per_series": 1000 * s / len(rows)})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cwfc.benchmark", description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, default=Path("reports/artifacts"))
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--with-fm", action="store_true")
    ap.add_argument("--no-lightgbm", action="store_true", help="skip the slow LightGBM model (tests)")
    args = ap.parse_args(argv)
    try:
        if args.with_fm and not fm_available():
            raise FMUnavailableError(INSTALL_HINT)
        panel = build_panel(load_readings(args.data), load_metadata(args.data))
        train, _ = holdout_split(panel, args.horizon)
    except (DataContractError, FileNotFoundError, FMUnavailableError) as e:
        print(f"error: {e}".splitlines()[0], file=sys.stderr)
        return 2
    H, art, lgb_on = args.horizon, args.artifacts, not args.no_lightgbm
    art.mkdir(parents=True, exist_ok=True)
    origins = default_origins(train.T, H)

    for target, tag, lgb in ((CPU, "cpu", lgb_on), (COUNT, "count", False)):
        models = _candidates(target, args.with_fm, lgb)
        results = []
        for m in models:
            results.append(backtest(m, train, target, H, origins))
            print(f"{tag}: {m.name} mae={results[-1].mae:.4f} ({results[-1].seconds:.1f}s)", flush=True)
        pd.concat([_long(r) for r in results]).to_csv(art / f"benchmark_{tag}.csv", index=False, lineterminator="\n")
        ins = series(train, target)
        summary = _summary(results, models, ins[np.isfinite(ins)])
        summary.to_csv(art / ("benchmark_summary.csv" if tag == "cpu" else "benchmark_count_summary.csv"),
                       index=False, lineterminator="\n")
        if tag == "cpu":
            eligible = [r for r, m in zip(results, models) if not _flagged(m)]
            _dm_matrix(eligible).to_csv(art / "dm_matrix.csv", lineterminator="\n")
    _latency(train, H, args.with_fm, lgb_on).to_csv(art / "latency.csv", index=False, lineterminator="\n")
    print(f"wrote benchmark artifacts in {art}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
