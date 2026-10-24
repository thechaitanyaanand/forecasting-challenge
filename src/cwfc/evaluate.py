"""One-shot holdout scoring. Run only after the final model is frozen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.data import DataContractError, build_panel, load_metadata, load_readings
from cwfc.metrics import coverage, mae, mase, rmse
from cwfc.models import Persistence
from cwfc.predict import COLUMNS
from cwfc.targets import TARGETS, holdout_split, series


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cwfc.evaluate", description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, default=Path("predictions.csv"))
    ap.add_argument("--artifacts", type=Path, default=Path("reports/artifacts"))
    ap.add_argument("--horizon", type=int, default=24)
    args = ap.parse_args(argv)
    try:
        panel = build_panel(load_readings(args.data), load_metadata(args.data))
        train, ts = holdout_split(panel, args.horizon)
        pred = pd.read_csv(args.predictions)
        if list(pred.columns) != COLUMNS or pred["timestamp"].tolist() != ts.tolist():
            raise ValueError("predictions do not match the submission contract for this data")
        iv_path = args.artifacts / "intervals.csv"
        iv = pd.read_csv(iv_path) if iv_path.is_file() else None
    except (DataContractError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}".splitlines()[0], file=sys.stderr)
        return 2

    result = {}
    for target in TARGETS:
        full = series(panel, target)
        actual, insample = full[-args.horizon:], series(train, target)
        insample = insample[np.isfinite(insample)]
        p = pred[target].to_numpy(dtype=float)
        base = Persistence().predict(train, target, args.horizon)
        fin = np.isfinite(actual)
        peak = fin & (actual >= np.percentile(actual[fin], 75)) if fin.any() else fin
        row = {
            "mae": mae(actual, p), "rmse": rmse(actual, p), "mase": mase(actual, p, insample),
            "persistence_mae": mae(actual, base), "rel_mae_vs_persistence": mae(actual, p) / mae(actual, base) if mae(actual, base) > 0 else float("nan"),
            "peak_quartile_mae": mae(actual[peak], p[peak]) if peak.any() else float("nan"),
        }
        if iv is not None:
            d = iv[iv["target"] == target].sort_values("timestamp")
            for lv in (80, 95):
                row[f"coverage{lv}"] = coverage(actual, d[f"lo{lv}"], d[f"hi{lv}"])
        result[target] = row
    table = pd.DataFrame(result).T
    print(table.to_string(float_format=lambda x: f"{x:.4g}"))
    args.artifacts.mkdir(parents=True, exist_ok=True)
    (args.artifacts / "holdout_eval.json").write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
