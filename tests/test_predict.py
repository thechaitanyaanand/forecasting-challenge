import json

import numpy as np
import pandas as pd
import pytest

from cwfc.predict import check_submission, main

COLS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
TS = np.arange(36, 60) * 300


@pytest.fixture
def smoke(make_readings, write_dataset, tmp_path):
    readings, meta = make_readings(7, 60, seed=3, churn=True)
    readings = readings[readings["timestamp"] != 30 * 300]
    (tmp_path / "data").mkdir()
    return readings, meta, write_dataset(tmp_path / "data", readings, meta)


def run(data, out, art):
    return main(["--data", str(data), "--output", str(out), "--artifacts", str(art)])


def good() -> pd.DataFrame:
    return pd.DataFrame({COLS[0]: TS, COLS[1]: 5.0, COLS[2]: 50.0})


def test_cli_smoke_dataset(smoke, tmp_path):
    out, art = tmp_path / "predictions.csv", tmp_path / "art"
    assert run(smoke[2], out, art) == 0
    df = pd.read_csv(out)
    assert list(df.columns) == COLS and len(df) == 24
    assert df["timestamp"].tolist() == TS.tolist()
    assert (df["active_vm_count"] % 1 == 0).all()
    check_submission(df, TS, 24)
    sel = json.loads((art / "selection.json").read_text())
    assert sel["targets"]["active_vm_count"]["origins"] == [12]
    for f in ["backtest_active_vm_count.csv", "intervals.csv", "tradeoff.csv", "capacity_plan.csv"]:
        assert (art / f).is_file()
    iv = pd.read_csv(art / "intervals.csv")
    assert (iv["lo95"] <= iv["lo80"]).all() and (iv["hi80"] <= iv["hi95"]).all()
    assert set(pd.read_csv(art / "tradeoff.csv")["segment"]) >= {"fleet"}


def test_cli_holdout_perturbation_invariant(smoke, make_readings, write_dataset, tmp_path):
    readings, meta, data = smoke
    assert run(data, tmp_path / "a.csv", tmp_path / "art_a") == 0
    r2 = readings.copy()
    last = np.sort(r2["timestamp"].unique())[-24:]
    r2.loc[r2["timestamp"].isin(last), "avg_cpu"] *= 3
    (tmp_path / "d2").mkdir()
    assert run(write_dataset(tmp_path / "d2", r2, meta), tmp_path / "b.csv", tmp_path / "art_b") == 0
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    for f in ["intervals.csv", "capacity_plan.csv", "tradeoff.csv"]:
        assert (tmp_path / "art_a" / f).read_bytes() == (tmp_path / "art_b" / f).read_bytes()


def test_cli_missing_file_exit_2(tmp_path, capsys):
    assert run(tmp_path, tmp_path / "p.csv", tmp_path / "art") == 2
    err = capsys.readouterr().err
    assert "missing file" in err and "Traceback" not in err and not (tmp_path / "p.csv").exists()


def test_cli_missing_column_exit_2(smoke, tmp_path, capsys):
    readings, meta, data = smoke
    readings.drop(columns=["avg_cpu"]).to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    assert run(data, tmp_path / "p.csv", tmp_path / "art") == 2
    assert "avg_cpu" in capsys.readouterr().err


def test_cli_with_fm_unavailable_exit_2(smoke, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("cwfc.fm.fm_available", lambda: False)
    assert main(["--data", str(smoke[2]), "--output", str(tmp_path / "p.csv"),
                 "--artifacts", str(tmp_path / "art"), "--with-fm"]) == 2
    assert "pip install -e" in capsys.readouterr().err


def test_check_submission_accepts_good():
    check_submission(good(), TS, 24)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d[COLS[::-1]], "columns"),
    (lambda d: d.iloc[:23], "rows"),
    (lambda d: d.assign(timestamp=d["timestamp"] + 300), "timestamps"),
    (lambda d: d.iloc[::-1].reset_index(drop=True), "order"),
    (lambda d: d.assign(timestamp=[TS[0]] * 24), "duplicate"),
    (lambda d: d.assign(active_vm_count=np.nan), "finite"),
    (lambda d: d.assign(active_vm_count=-1.0), "active_vm_count"),
    (lambda d: d.assign(active_vm_count=5.5), "integers"),
    (lambda d: d.assign(timestamp=d["timestamp"].astype(float)), "integer"),
    (lambda d: d.assign(aggregate_cpu_utilization=100.5), "aggregate_cpu_utilization"),
])
def test_check_submission_rejects(mutate, msg):
    with pytest.raises(ValueError, match=msg):
        check_submission(mutate(good()), TS, 24)


def test_evaluate_smoke(smoke, tmp_path, capsys):
    from cwfc.evaluate import main as evaluate

    run(smoke[2], tmp_path / "p.csv", tmp_path / "art")
    assert evaluate(["--data", str(smoke[2]), "--predictions", str(tmp_path / "p.csv"),
                     "--artifacts", str(tmp_path / "art")]) == 0
    res = json.loads((tmp_path / "art" / "holdout_eval.json").read_text())
    assert set(res) == {"active_vm_count", "aggregate_cpu_utilization"}
    assert "coverage95" in res["active_vm_count"]


def test_cli_gap_not_imputed_and_no_skips(smoke, tmp_path):
    assert run(smoke[2], tmp_path / "p.csv", tmp_path / "art") == 0
    sel = json.loads((tmp_path / "art" / "selection.json").read_text())
    assert sel["capacity_skipped"] == []
    bt = pd.read_csv(tmp_path / "art" / "backtest_aggregate_cpu_utilization.csv")
    assert bt["actual"].isna().sum() == bt["model"].nunique()  # the dropped timestamp stays a missing actual
    assert (tmp_path / "p.csv").read_text().splitlines()[1].count(".") == 1  # integer count


def test_cli_corrupt_gz_exit_2(smoke, tmp_path, capsys):
    (smoke[2] / "vm_cpu_readings.csv.gz").write_bytes(b"not gzip")
    assert run(smoke[2], tmp_path / "p.csv", tmp_path / "art") == 2
    assert "vm_cpu_readings" in capsys.readouterr().err


def _short(make_readings, write_dataset, tmp_path, n):
    readings, meta = make_readings(5, n, seed=3)
    (tmp_path / "data").mkdir()
    return write_dataset(tmp_path / "data", readings, meta)


def test_cli_short_history_falls_back_to_persistence(make_readings, write_dataset, tmp_path):
    data = _short(make_readings, write_dataset, tmp_path, 50)
    out, art = tmp_path / "p.csv", tmp_path / "art"
    art.mkdir()
    (art / "intervals.csv").write_text("stale")
    assert run(data, out, art) == 0
    df = pd.read_csv(out)
    check_submission(df, np.arange(26, 50) * 300, 24)
    sel = json.loads((art / "selection.json").read_text())
    assert sel["targets"]["aggregate_cpu_utilization"]["selected"] == "persistence"
    assert "too short" in sel["intervals_capacity_skipped"]
    assert not (art / "intervals.csv").exists()
    assert b"\r" not in out.read_bytes()


def test_cli_too_short_for_holdout_exit_2(make_readings, write_dataset, tmp_path, capsys):
    data = _short(make_readings, write_dataset, tmp_path, 30)
    assert run(data, tmp_path / "p.csv", tmp_path / "art") == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "37" in err and len(err.strip().splitlines()) == 1
