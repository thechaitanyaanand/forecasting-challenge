import numpy as np
import pytest
from cwfc.metrics import mae, normalized_mae, rmse, smape


def test_metrics_basic():
    y = [1, 2, 3]
    p = [1, 3, 1]
    assert mae(y, p) == 1.0
    assert round(rmse(y, p), 8) == round((5 / 3) ** 0.5, 8)
    assert smape(y, p) > 0
    assert normalized_mae(y, p, [1, 1, 1]) > 0


def test_mase_naive_is_one():
    from cwfc.metrics import mase
    insample = [1.0, 2.0, 4.0, 7.0]
    assert mase([10.0], [8.0], insample) == pytest.approx(2.0 / 2.0)
    assert mase([1, 1], [1, 1], [3, 3, 3]) == 0.0
    assert mase([1, 2], [2, 2], [3, 3, 3]) == float("inf")


def test_pinball_known_value():
    from cwfc.metrics import pinball
    assert pinball([10.0, 10.0], [8.0, 12.0], 0.9) == pytest.approx((0.9 * 2 + 0.1 * 2) / 2)


def test_coverage():
    from cwfc.metrics import coverage
    assert coverage([1, 5, 9, 3], [0, 0, 0, 4], [2, 4, 10, 5]) == 0.5


def test_dm_identical_errors_p_one():
    from cwfc.metrics import diebold_mariano
    e = np.random.default_rng(0).normal(size=30)
    assert diebold_mariano(e, e) == (0.0, 1.0)


def test_dm_detects_difference():
    from cwfc.metrics import diebold_mariano
    e = np.random.default_rng(0).normal(size=60)
    stat, p = diebold_mariano(e, e * 0.5 + np.random.default_rng(1).normal(0, 0.05, 60))
    assert stat > 0 and p < 0.01
    assert diebold_mariano(e * 0.5, e)[0] < 0
    assert diebold_mariano(e, e + 1.0, h=3)[0] != 0


def test_dm_matches_direct_bartlett_computation():
    from scipy import stats
    from cwfc.metrics import diebold_mariano
    rng = np.random.default_rng(5)
    e1, e2 = rng.normal(size=25), rng.normal(size=25) * 0.8
    n, h = 25, 4
    d = np.abs(e1) - np.abs(e2)
    c = d - d.mean()
    gam = [sum(c[t] * c[t - k] for t in range(k, n)) / n for k in range(h)]
    lrv = gam[0] + 2 * sum((1 - k / h) * gam[k] for k in range(1, h))
    stat = d.mean() / np.sqrt(lrv / n) * np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    got = diebold_mariano(e1, e2, h=h, power=1)
    assert got[0] == pytest.approx(stat)
    assert got[1] == pytest.approx(2 * stats.t.sf(abs(stat), n - 1))


def test_dm_null_size_real_config():
    from cwfc.metrics import diebold_mariano
    rng = np.random.default_rng(0)
    rej = 0
    for _ in range(500):
        a, b = rng.gamma(2.0, size=20), rng.gamma(2.0, size=20)
        stat, p = diebold_mariano(a, b, h=12, power=1)
        assert p > 0
        rej += p < 0.05
    assert rej / 500 <= 0.10
