from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.data import Panel
from cwfc.targets import EXTRA_TARGETS, TARGETS, build_targets, holdout_split, series


def make_panel(avg, cores, mx=None) -> Panel:
    avg = np.asarray(avg, dtype=float)
    n, t = avg.shape
    mx = avg if mx is None else np.asarray(mx, dtype=float)
    ids = np.array([f"vm{i}" for i in range(n)])
    static = pd.DataFrame(
        {"vm_category": "Delay-insensitive", "cores": cores, "memory_gb_bucket": "4"}, index=ids
    )
    return Panel(np.arange(t, dtype=np.int64) * 300, ids, avg, avg.copy(), mx, static)


def test_count_and_mean():
    p = make_panel([[10, 20, 30], [20, np.nan, 30], [30, 40, 30]], [1, 1, 1])
    np.testing.assert_array_equal(series(p, "active_vm_count"), [3, 2, 3])
    np.testing.assert_allclose(series(p, "aggregate_cpu_utilization"), [20, 30, 30])


def test_gap_timestamp_is_nan_for_every_target():
    p = make_panel([[10, np.nan, 30], [20, np.nan, 30]], [1, 1])
    for t in ("active_vm_count", "aggregate_cpu_utilization", "core_weighted_cpu", "peak_core_demand"):
        s = series(p, t)
        assert np.isnan(s[1]) and np.isfinite(s[[0, 2]]).all()


def test_core_weighted():
    p = make_panel([[10], [50]], [2, 8])
    assert series(p, "core_weighted_cpu")[0] == pytest.approx(42.0)


def test_peak_core_demand():
    p = make_panel([[10], [50]], [2, 8], mx=[[20], [50]])
    assert series(p, "peak_core_demand")[0] == pytest.approx((20 * 2 + 50 * 8) / 100)


def test_unknown_target_raises():
    p = make_panel([[1.0]], [1])
    with pytest.raises(ValueError, match="active_vm_count"):
        series(p, "nope")


def test_build_targets_columns():
    p = make_panel([[10, 20], [20, 30]], [1, 1])
    df = build_targets(p)
    assert list(df.columns) == ["timestamp", *TARGETS]
    assert list(build_targets(p, include_extra=True).columns) == ["timestamp", *TARGETS, *EXTRA_TARGETS]
    assert list(df.index) == [0, 1]


def test_holdout_split_sizes():
    p = make_panel(np.ones((2, 40)), [1, 1])
    train, hold = holdout_split(p, 24)
    assert train.T == 16
    assert len(hold) == 24
    np.testing.assert_array_equal(hold, p.timestamps[-24:])


def test_holdout_too_short_raises():
    p = make_panel(np.ones((2, 36)), [1, 1])
    with pytest.raises(ValueError, match="37"):
        holdout_split(p, 24)
    with pytest.raises(ValueError):
        holdout_split(p, 0)
