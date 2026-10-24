from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.backtest import BacktestResult, backtest, default_origins, select
from cwfc.data import DataContractError, build_panel, load_metadata, load_readings, validate_readings
from cwfc.fm import INSTALL_HINT, FMUnavailableError, fm_available
from cwfc.models import Persistence, default_candidates
from cwfc.policy import conformal_interval, conformal_upper, tradeoff
from cwfc.targets import TARGETS, holdout_split

COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
CAPACITY_TARGET = "core_weighted_cpu"
LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)
SEED = 0


class SubmissionError(ValueError):
    pass


def check_submission(df: pd.DataFrame, holdout_timestamps: np.ndarray, horizon: int) -> None:
    if list(df.columns) != COLUMNS:
        raise SubmissionError(f"columns must be exactly {COLUMNS} in order, got {list(df.columns)}")
    if len(df) != horizon:
        raise SubmissionError(f"expected {horizon} rows, got {len(df)}")
    ts = df["timestamp"].to_numpy()
    if not np.issubdtype(ts.dtype, np.integer):
        raise SubmissionError("timestamps must be integer seconds")
    if df["timestamp"].duplicated().any():
        raise SubmissionError("duplicate timestamps")
    expected = np.asarray(holdout_timestamps)
    if set(ts.tolist()) != set(expected.tolist()):
        raise SubmissionError("timestamps do not match the holdout timestamps")
    if ts.tolist() != expected.tolist():
        raise SubmissionError("timestamps are not in chronological order")
    vals = df[COLUMNS[1:]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise SubmissionError("predictions must be finite numeric values")
    if (vals[:, 0] % 1 != 0).any():
        raise SubmissionError("active_vm_count must be integers")
    if (df["active_vm_count"] < 0).any():
        raise SubmissionError("active_vm_count must be >= 0")
    if ((df["aggregate_cpu_utilization"] < 0) | (df["aggregate_cpu_utilization"] > 100)).any():
        raise SubmissionError("aggregate_cpu_utilization must be in [0, 100]")


def _clip(target: str, x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 100) if target != "active_vm_count" else np.maximum(x, 0)


def _long(r: BacktestResult) -> pd.DataFrame:
    F, H = r.preds.shape
    return pd.DataFrame({
        "model": r.name, "origin": np.repeat(r.origins, H), "h": np.tile(np.arange(1, H + 1), F),
        "pred": r.preds.ravel(), "actual": r.actuals.ravel(),
    })


def _run(candidates, panel, target, horizon, origins):
    results = [backtest(m, panel, target, horizon, origins) for m in candidates]
    sel = select(results)
    i = [m.name for m in candidates].index(sel.name)
    return sel, results[i], candidates[i], results


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="cwfc.predict", description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--with-fm", action="store_true", help="add eligible foundation models")
    ap.add_argument("--include-lightgbm", action="store_true", help="add the slow LightGBM panel model")
    ap.add_argument("--artifacts", type=Path, default=Path("reports/artifacts"))
    return ap


def _fit_predict(args) -> None:
    readings, metadata = load_readings(args.data), load_metadata(args.data)
    _, report = validate_readings(readings)
    print("data report:", json.dumps(report.to_dict()))
    panel = build_panel(readings, metadata)
    H = args.horizon
    try:
        train, holdout_ts = holdout_split(panel, H)
    except ValueError as e:
        raise DataContractError(f"series too short: {e}") from None
    try:
        origins = default_origins(train.T, H)
    except ValueError:
        origins = None  # too short to backtest: persistence only, no intervals/capacity
    art = args.artifacts
    art.mkdir(parents=True, exist_ok=True)

    out = {"timestamp": holdout_ts.astype(np.int64)}
    selection: dict = {"horizon": H, "seed": SEED, "data_report": report.to_dict(), "targets": {}}
    intervals = []
    short_reason = "history too short for backtesting (needs ≥ horizon + 12 training steps)"
    for target in TARGETS:
        if origins is None:
            point = _clip(target, np.asarray(Persistence().predict(train, target, H), dtype=float))
            out[target] = np.maximum(np.rint(point), 0) if target == "active_vm_count" else point
            selection["targets"][target] = {"selected": "persistence", "reason": short_reason}
            print(f"{target}: persistence ({short_reason})".replace("≥", ">="))
            continue
        cands = default_candidates(target, with_fm=args.with_fm, include_lightgbm=args.include_lightgbm)
        sel, res, model, results = _run(cands, train, target, H, origins)
        point = _clip(target, np.asarray(model.predict(train, target, H), dtype=float))
        if target == "active_vm_count":
            point = np.maximum(np.rint(point), 0)
        out[target] = point
        pd.concat([_long(r) for r in results]).to_csv(art / f"backtest_{target}.csv", index=False, lineterminator="\n")
        selection["targets"][target] = {
            "selected": sel.name, "reason": sel.reason, "table": sel.table.to_dict("records"),
            "candidates": [m.name for m in cands], "origins": origins,
        }
        rec = {"timestamp": holdout_ts, "target": target, "point": point}
        for lv in (80, 95):
            lo, hi = conformal_interval(res.errors, lv / 100)
            rec[f"lo{lv}"], rec[f"hi{lv}"] = _clip(target, point + lo), _clip(target, point + hi)
        intervals.append(pd.DataFrame(rec))
        print(f"{target}: {sel.name} ({sel.reason})")

    out["active_vm_count"] = out["active_vm_count"].astype(np.int64)
    sub = pd.DataFrame(out, columns=COLUMNS)
    check_submission(sub, holdout_ts, H)
    sub.to_csv(args.output, index=False, lineterminator="\n")
    if origins is None:
        selection["intervals_capacity_skipped"] = short_reason
        for f in ("intervals.csv", "tradeoff.csv", "capacity_plan.csv"):
            (art / f).unlink(missing_ok=True)
        (art / "selection.json").write_text(json.dumps(selection, indent=2, default=float))
        print(f"wrote {args.output}; {short_reason}: intervals and capacity plan skipped".replace("≥", ">="))
        return
    pd.concat(intervals).to_csv(art / "intervals.csv", index=False, lineterminator="\n")

    trade, plan, skipped = [], [], []
    cats = train.static["vm_category"].to_numpy()
    segments = {"fleet": train} | {str(c): train.subset(cats == c) for c in np.unique(cats)}
    for seg, p in segments.items():
        cands = default_candidates(CAPACITY_TARGET, include_lightgbm=False)
        try:  # ponytail: a too-short or empty segment is skipped, not fatal
            sel, res, model, _ = _run(cands, p, CAPACITY_TARGET, H, origins)
            t = tradeoff(res.errors, LEVELS)
            point = np.asarray(model.predict(p, CAPACITY_TARGET, H), dtype=float)
            uppers = {f"upper_{lv}": _clip(CAPACITY_TARGET, point + conformal_upper(res.errors, lv / 100))
                      for lv in (90, 95)}
        except ValueError:
            skipped.append(seg)
            continue
        trade.append(t.assign(segment=seg)[["segment", "level", "breach_rate", "mean_headroom", "expected_cost"]])
        plan.append(pd.DataFrame({
            "segment": seg, "timestamp": holdout_ts, "point": _clip(CAPACITY_TARGET, point), **uppers,
        }))
        selection.setdefault("capacity", {})[seg] = {"selected": sel.name, "reason": sel.reason}
    selection["capacity_skipped"] = skipped
    for name, frames in (("tradeoff.csv", trade), ("capacity_plan.csv", plan)):
        if frames:
            pd.concat(frames).to_csv(art / name, index=False, lineterminator="\n")
        else:
            (art / name).unlink(missing_ok=True)
    (art / "selection.json").write_text(json.dumps(selection, indent=2, default=float))
    print(f"wrote {args.output} and artifacts in {art}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.with_fm:
            if not fm_available():
                raise FMUnavailableError(INSTALL_HINT)
        _fit_predict(args)
    except (DataContractError, FileNotFoundError, FMUnavailableError, SubmissionError) as e:
        print(f"error: {e}".splitlines()[0], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
