import numpy as np
import pandas as pd
import pytest

from cwfc.data import Panel, build_panel
from cwfc.features import panel_inference_frame, panel_training_frame
from cwfc.models import Drift, ExpandingMean, LightGBMPanel, Persistence, RollingMean, StatsForecast, default_candidates

T_ = "aggregate_cpu_utilization"
FAST = {"n_estimators": 5, "min_child_samples": 5, "num_leaves": 7, "n_jobs": 1}


def agg_panel(values) -> Panel:
    v = np.asarray(values, dtype=float)[None, :]
    return Panel(np.arange(v.shape[1]) * 300, np.array(["a"]), v, v - 1, v + 1,
                 pd.DataFrame({"vm_category": ["x"], "cores": [2], "memory_gb_bucket": ["4"]}))


@pytest.fixture
def panel(make_readings):
    return build_panel(*make_readings(30, 60, churn=True))


def test_persistence_last_value():
    p = agg_panel([1, 2, np.nan])
    assert Persistence().predict(p, T_, 3).tolist() == [2.0, 2.0, 2.0]


def test_expanding_mean():
    p = agg_panel([1, 2, np.nan, 5])
    m = ExpandingMean()
    assert m.predict(p, T_, 2).tolist() == [2.5, 2.5] and m.complexity == 1


def test_rolling_mean():
    p = agg_panel([1, 2, 3, 4, np.nan])
    assert RollingMean(2).predict(p, T_, 2).tolist() == [3.5, 3.5]
    assert RollingMean(10).name == "rolling_mean_10"
    assert RollingMean(10).predict(p, T_, 1)[0] == 2.5


def test_drift_linear_series_exact():
    p = agg_panel(np.arange(10) * 2.0)
    np.testing.assert_allclose(Drift().predict(p, T_, 3), [20, 22, 24])
    assert Drift().predict(agg_panel([5.0]), T_, 2).tolist() == [5.0, 5.0]


@pytest.mark.parametrize("kind", ["ets", "arima"])
def test_statsforecast_returns_finite_horizon(kind):
    m = StatsForecast(kind)
    out = m.predict(agg_panel(50 + np.sin(np.arange(40))), T_, 6)
    assert out.shape == (6,) and np.isfinite(out).all() and not m.fallback_used
    short = m.predict(agg_panel([1, 2, 3]), T_, 4)
    assert m.fallback_used and short.tolist() == [3.0] * 4


def test_training_frame_labels_observed(panel):
    X, y, idx = panel_training_frame(panel, 6, 4, 3)
    assert len(X) == len(y) == len(idx) > 0
    assert idx.max() < panel.T and np.isfinite(y).all()
    assert (X["h"] >= 1).all() and X["vm_category"].dtype == "category"


def test_training_frame_alignment_is_exact():
    # avg[vm, t] = 100 * vm + t, so every feature/label offset is checkable in closed form
    T, H, L = 40, 6, 4
    vm = np.arange(3)
    avg = 100.0 * vm[:, None] + np.arange(T)[None, :]
    p = Panel(np.arange(T) * 300, np.array(list("abc")), avg, avg - 1, avg + 1,
              pd.DataFrame({"vm_category": ["x"] * 3, "cores": [2] * 3, "memory_gb_bucket": ["4"] * 3}))
    X, y, idx = panel_training_frame(p, H, L, 3)
    h = X["h"].to_numpy()
    assert idx.max() < T and len(y) > 0
    for k in range(1, L + 1):
        d = y - X[f"lag_avg_{k}"].to_numpy()
        ok = np.isfinite(d)
        assert ok.any()
        np.testing.assert_allclose(d[ok], (h - 1 + k)[ok])
    o = idx - h + 1  # forecast origin of each row
    d = (y - X["roll_mean_6"].to_numpy())[o >= 6]
    np.testing.assert_allclose(d, (h - 1 + 3.5)[o >= 6])
    inf = panel_inference_frame(p, H, L)
    np.testing.assert_allclose(inf["lag_avg_1"].to_numpy(), np.tile(100.0 * vm + T - 1, H))


def test_training_frame_excludes_leaky_metadata(panel):
    X, _, _ = panel_training_frame(panel, 6, 4, 3)
    bad = {"avg_cpu", "max_cpu", "p95_max_cpu", "timestamp_vm_deleted", "timestamp_vm_created"}
    assert not bad & set(X.columns)
    assert {"h", "vm_category", "cores", "memory_gb_bucket", "ewm_12"} <= set(X.columns)


def test_training_frame_row_cap(panel):
    X, y, _ = panel_training_frame(panel, 6, 4, 3, max_rows=100)
    assert 0 < len(y) <= 130


def test_inference_frame_shape(panel):
    X = panel_inference_frame(panel, 6, 4)
    assert len(X) == 6 * len(panel.vm_ids)
    assert list(X.columns) == list(panel_training_frame(panel, 6, 4, 3)[0].columns)


@pytest.mark.parametrize("target", ["aggregate_cpu_utilization", "core_weighted_cpu"])
def test_lightgbm_shape_and_finite(panel, target):
    m = LightGBMPanel(lags=4, origin_stride=3, params=FAST)
    out = m.predict(panel, target, 6)
    assert out.shape == (6,) and np.isfinite(out).all() and not m.fallback_used
    assert m.model_ is not None and "h" in m.feature_names_


def test_lightgbm_rejects_unsupported(panel):
    with pytest.raises(ValueError):
        LightGBMPanel(params=FAST).predict(panel, "peak_core_demand", 6)


def test_lightgbm_tiny_panel_falls_back():
    p = agg_panel(np.arange(30.0))
    m = LightGBMPanel(lags=4, params=FAST)
    assert m.predict(p, T_, 3).tolist() == [29.0] * 3 and m.fallback_used


def test_default_candidates():
    names = [m.name for m in default_candidates(T_)]
    assert names == ["persistence", "expanding_mean", "rolling_mean_6", "rolling_mean_12",
                     "rolling_mean_24", "drift", "ets", "arima", "lightgbm_panel"]
    assert "lightgbm_panel" not in [m.name for m in default_candidates(T_, include_lightgbm=False)]
    assert "lightgbm_panel" not in [m.name for m in default_candidates("active_vm_count")]
