import numpy as np

from cwfc.policy import conformal_interval, conformal_upper, tradeoff


def test_conformal_finite_sample():
    errors = np.arange(1, 20, dtype=float)[:, None]
    assert conformal_upper(errors, 0.9).tolist() == [18.0]
    assert conformal_upper(errors, 0.999).tolist() == [19.0]


def test_conformal_pools_neighbouring_horizons():
    errors = np.array([[1.0, 100.0, 5.0]] * 4)
    assert conformal_upper(errors, 0.5, pool=0).tolist() == [1.0, 100.0, 5.0]
    assert conformal_upper(errors, 0.99, pool=1)[0] == 100.0


def test_conformal_coverage_gaussian():
    rng = np.random.default_rng(0)
    cal, test = rng.normal(size=(2000, 4)), rng.normal(size=(20000, 4))
    for level in (0.8, 0.95):
        lo, hi = conformal_interval(cal, level)
        assert lo.shape == hi.shape == (4,) and (lo < 0).all() and (hi > 0).all()
        cover = ((test >= lo) & (test <= hi)).mean()
        assert abs(cover - level) < 0.03


def test_tradeoff_monotone():
    rng = np.random.default_rng(1)
    df = tradeoff(rng.normal(size=(200, 6)), [0.5, 0.8, 0.9, 0.95, 0.99])
    assert list(df.columns) == ["level", "breach_rate", "mean_headroom", "expected_cost"]
    assert df["breach_rate"].is_monotonic_decreasing
    assert df["breach_rate"].iloc[-1] < df["breach_rate"].iloc[0]
    assert df["mean_headroom"].is_monotonic_increasing


def test_conformal_and_tradeoff_ignore_nan():
    e = np.arange(1, 20, dtype=float)[:, None]
    e_nan = np.vstack([e, [[np.nan]]])
    assert conformal_upper(e_nan, 0.9).tolist() == [18.0]
    assert np.isfinite(tradeoff(e_nan, [0.9]).to_numpy()).all()
