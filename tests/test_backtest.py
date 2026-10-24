import numpy as np
import pandas as pd
import pytest

from cwfc.backtest import BacktestResult, backtest, default_origins, select
from cwfc.data import Panel
from cwfc.targets import series


def make_panel(T: int = 60, n: int = 3) -> Panel:
    rng = np.random.default_rng(0)
    avg = rng.uniform(10, 50, (n, T))
    return Panel(
        np.arange(T) * 300, np.array([f"v{i}" for i in range(n)]), avg, avg - 1, avg + 1,
        pd.DataFrame({"vm_category": ["x"] * n, "cores": [2] * n, "memory_gb_bucket": ["4"] * n}),
    )


class Spy:
    name, complexity = "spy", 1

    def __init__(self):
        self.seen = []

    def predict(self, panel, target, horizon):
        self.seen.append(panel.T)
        return np.zeros(horizon)


class Spy2(Spy):
    name, complexity = "spy2", 2

    def predict(self, panel, target, horizon):
        return np.ones(horizon)


def test_default_origins_real_shape():
    o = default_origins(110, 24)
    assert o == list(range(48, 87, 2)) and len(o) == 20


def test_default_origins_small():
    assert default_origins(36, 24) == [12]


def test_default_origins_empty_raises():
    with pytest.raises(ValueError, match="n_train"):
        default_origins(30, 24)


def test_backtest_model_never_sees_future():
    panel, spy = make_panel(), Spy()
    r = backtest(spy, panel, "active_vm_count", 6, [12, 20, 30])
    assert spy.seen == [12, 20, 30]
    assert r.preds.shape == r.actuals.shape == (3, 6)
    assert np.array_equal(r.actuals[1], series(panel, "active_vm_count")[20:26])
    assert r.fold_mae.shape == (3,) and r.complexity == 1 and r.seconds >= 0


def test_backtest_validation_errors():
    panel = make_panel()
    with pytest.raises(ValueError):
        backtest(Spy(), panel, "active_vm_count", 6, [5])
    with pytest.raises(ValueError):
        backtest(Spy(), panel, "active_vm_count", 6, [58])

    class Bad(Spy):
        name = "bad"

        def predict(self, panel, target, horizon):
            return np.full(horizon, np.nan)

    with pytest.raises(ValueError, match="bad"):
        backtest(Bad(), panel, "active_vm_count", 6, [20])

    class Short(Bad):
        def predict(self, panel, target, horizon):
            return np.zeros(horizon + 1)

    with pytest.raises(ValueError, match="bad"):
        backtest(Short(), panel, "active_vm_count", 6, [20])


def _result(name, complexity, errors):
    errors = np.asarray(errors, dtype=float)
    return BacktestResult(
        name, "t", list(range(12, 12 + 2 * len(errors), 2)), -errors, np.zeros_like(errors),
        0.0, complexity,
    )


def _errors(scale):
    rng = np.random.default_rng(3)
    return rng.normal(1.0, 0.3, (20, 4)) * scale


def test_select_prefers_simpler_when_not_significant():
    e = _errors(1.0)
    noise = np.random.default_rng(9).normal(0, 0.3, e.shape)
    sel = select([_result("simple", 1, e), _result("complex", 5, e * 0.999 + noise * 0.1)])
    assert sel.name == "simple" and "not significantly worse" in sel.reason
    assert sel.table["selected"].sum() == 1
    assert list(sel.table.columns) == ["name", "complexity", "mae", "rmse", "dm_p_vs_best", "selected"]
    assert sel.table.iloc[0]["name"] == "complex"


def test_select_takes_complex_when_significant():
    e = _errors(1.0)
    sel = select([_result("simple", 1, e), _result("complex", 5, e * 0.5)])
    assert sel.name == "complex"


def test_select_single_fold():
    e = _errors(1.0)[:1]
    sel = select([_result("simple", 1, e * 1.005), _result("complex", 5, e)])
    assert sel.name == "simple" and "single fold" in sel.reason


def test_select_real_config_prefers_simpler_under_null():
    rng = np.random.default_rng(1)
    origins = list(range(48, 87, 2))
    wins = 0
    for _ in range(40):
        e1, e2 = rng.normal(size=(20, 24)), rng.normal(size=(20, 24))
        res = []
        for nm, c, e in (("simple", 1, e1), ("complex", 5, e2)):
            res.append(BacktestResult(nm, "t", origins, -e, np.zeros_like(e), 0.0, c))
        wins += select(res).name == "simple"
    assert wins >= 36


def test_backtest_nan_actual_keeps_mae_finite_and_select_works():
    p = make_panel()
    p.avg[:, 40:42] = np.nan  # telemetry gap inside a fold window
    origins = [30, 32, 34, 36]
    res = [backtest(m, p, "aggregate_cpu_utilization", 6, origins) for m in (Spy(), Spy2())]
    assert np.isnan(res[0].actuals).any()
    assert all(np.isfinite(r.mae) and np.isfinite(r.rmse) for r in res)
    assert np.isfinite(res[0].fold_mae).all()
    assert select(res).name in {"spy", "spy2"}


def test_select_drops_all_nan_fold():
    p = make_panel()
    p.avg[:, 36:44] = np.nan
    res = [backtest(m, p, "active_vm_count", 6, [30, 36, 37, 38]) for m in (Spy(), Spy2())]
    assert np.isnan(res[0].fold_mae[1:]).all()
    assert select(res).name in {"spy", "spy2"}
