import pandas as pd

from cwfc.benchmark import main


def test_benchmark_smoke(make_readings, write_dataset, tmp_path):
    readings, meta = make_readings(7, 60, seed=3, churn=True)
    data = write_dataset(tmp_path, readings, meta)
    art = tmp_path / "art"
    assert main(["--data", str(data), "--artifacts", str(art), "--no-lightgbm"]) == 0
    s = pd.read_csv(art / "benchmark_summary.csv")
    cols = {"model", "complexity", "mae", "rmse", "mase", "mean_error", "seconds", "flagged", "eligible"}
    assert cols <= set(s) and "persistence" in set(s["model"]) and s["eligible"].all()
    assert set(pd.read_csv(art / "benchmark_cpu.csv").columns) == {"model", "origin", "h", "pred", "actual"}
    dm = pd.read_csv(art / "dm_matrix.csv", index_col=0)
    assert dm.shape[0] == dm.shape[1] == len(s)
    assert (art / "benchmark_count.csv").is_file() and (art / "latency.csv").is_file()


def test_benchmark_bad_data_exit_2(tmp_path):
    assert main(["--data", str(tmp_path), "--artifacts", str(tmp_path / "a")]) == 2
