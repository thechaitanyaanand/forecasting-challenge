from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cwfc.data import (
    DataContractError,
    build_panel,
    core_count,
    load_metadata,
    load_readings,
    validate_readings,
)


def test_missing_column_raises(make_readings, write_dataset, tmp_path):
    readings, meta = make_readings(3, 10, 0)
    d = write_dataset(tmp_path, readings.drop(columns="avg_cpu"), meta)
    with pytest.raises(DataContractError, match="avg_cpu"):
        load_readings(d)


def test_missing_file_raises(tmp_path):
    with pytest.raises(DataContractError, match="vm_metadata"):
        load_metadata(tmp_path)


def test_report_counts_dirty_rows(make_readings):
    readings, _ = make_readings(5, 20, 1)
    gone = int(readings.timestamp.iloc[7 * 5])
    df = readings[readings.timestamp != gone].reset_index(drop=True)
    df.loc[0, "avg_cpu"] = np.nan
    df.loc[1, "avg_cpu"] = 120.0
    df.loc[2, "min_cpu"] = df.loc[2, "avg_cpu"] + 3
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    clean, rep = validate_readings(df)
    assert rep.duplicate_rows == 1
    assert rep.invalid_readings == 2
    assert rep.order_violations == 1
    assert rep.missing_timestamps == [gone]
    assert len(clean) == len(df) - 3
    assert rep.to_dict()["n_vms"] == 5


def test_unsorted_input_same_panel(make_readings):
    readings, meta = make_readings(4, 12, 2, churn=True)
    shuffled = readings.sample(frac=1, random_state=0)
    a = build_panel(readings, meta)
    b = build_panel(shuffled, meta)
    assert np.array_equal(a.avg, b.avg, equal_nan=True)
    assert np.array_equal(a.vm_ids, b.vm_ids)


def test_truncate_hides_future(make_readings):
    readings, meta = make_readings(3, 12, 3)
    full = build_panel(readings, meta)
    p = full.truncate(5)
    assert p.avg.shape[1] == 5 and p.T == 5
    assert np.array_equal(p.avg, full.avg[:, :5], equal_nan=True)
    assert np.array_equal(p.timestamps, full.timestamps[:5])
    p.avg[:] = -1
    assert full.avg.max() > 0


def test_panel_static_excludes_leaky_columns(make_readings):
    readings, meta = make_readings(5, 8, 4)
    p = build_panel(readings, meta)
    assert list(p.static.columns) == ["vm_category", "cores", "memory_gb_bucket"]
    assert p.static["cores"].dtype.kind == "i"
    assert p.subset(np.array([True, False, True, False, False])).avg.shape == (2, 8)


def test_vm_missing_from_metadata_raises(make_readings):
    readings, meta = make_readings(3, 5, 5)
    with pytest.raises(DataContractError, match="^1 VMs in readings are missing from metadata$"):
        build_panel(readings, meta.iloc[1:])


def test_roundtrip_load(make_readings, write_dataset, tmp_path):
    readings, meta = make_readings(3, 6, 6)
    d = write_dataset(tmp_path, readings, meta)
    r = load_readings(d)
    assert r.timestamp.is_monotonic_increasing
    assert load_metadata(d).virtual_core_bucket.map(type).eq(str).all()


def test_core_count():
    assert core_count("2") == 2
    assert core_count(">24") == 32
    assert core_count(">24", 48) == 48


def _dirty(make_readings):
    readings, meta = make_readings(5, 20, 7)
    df = readings.copy()
    df.loc[0, "avg_cpu"] = np.nan
    df.loc[1, "avg_cpu"] = 120.0
    conflict = df.iloc[[10]].copy()
    conflict[["min_cpu", "avg_cpu", "max_cpu"]] = [1.0, 2.0, 3.0]
    nan_dup = df.iloc[[20]].copy()
    nan_dup["avg_cpu"] = np.nan
    return pd.concat([df, conflict, nan_dup], ignore_index=True), meta


def test_shuffled_dirty_input_same_panel_and_report(make_readings):
    df, meta = _dirty(make_readings)
    shuffled = df.sample(frac=1, random_state=3)
    c1, r1 = validate_readings(df)
    c2, r2 = validate_readings(shuffled)
    assert r1 == r2
    assert r1.duplicate_rows == 1 and r1.invalid_readings == 3
    pd.testing.assert_frame_equal(c1, c2)
    a, b = build_panel(df, meta), build_panel(shuffled, meta)
    assert np.array_equal(a.avg, b.avg, equal_nan=True)


def test_build_panel_dirty_equals_build_panel_clean(make_readings):
    df, meta = _dirty(make_readings)
    clean, _ = validate_readings(df)
    a, b = build_panel(df, meta), build_panel(clean, meta)
    assert np.array_equal(a.avg, b.avg, equal_nan=True)
    assert np.array_equal(a.min, b.min, equal_nan=True)
    assert np.array_equal(a.vm_ids, b.vm_ids)


def test_off_grid_timestamp_raises(make_readings):
    readings, meta = make_readings(3, 10, 8)
    readings.loc[5, "timestamp"] += 7
    bad = int(readings.loc[5, "timestamp"])
    for fn in (validate_readings, lambda r: build_panel(r, meta)):
        with pytest.raises(DataContractError, match=rf"1 timestamps .*first: {bad}"):
            fn(readings)
