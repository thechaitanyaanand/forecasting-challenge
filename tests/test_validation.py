import pandas as pd
from cwfc.validation import rolling_origin_splits


def test_rolling_origin_splits_are_temporal():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC")
    })
    splits = rolling_origin_splits(frame, horizon=4, min_train_size=8, step=4)
    assert len(splits) == 3
    assert splits[0].train_end < splits[0].validation_start
    assert splits[0].validation_end < splits[1].validation_start


def test_rolling_origin_integer_seconds():
    frame = pd.DataFrame({"timestamp": [i * 300 for i in range(10)]})
    s = rolling_origin_splits(frame, horizon=2, min_train_size=4, step=2)[0]
    assert s.train_end == pd.Timestamp("1970-01-01 00:15:00", tz="UTC")
    assert s.validation_start == pd.Timestamp("1970-01-01 00:20:00", tz="UTC")


def test_rolling_origin_no_splits_raises():
    import pytest
    frame = pd.DataFrame({"timestamp": list(range(5))})
    with pytest.raises(ValueError, match="no splits"):
        rolling_origin_splits(frame, horizon=4, min_train_size=4)
