from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

READINGS_FILE = "vm_cpu_readings.csv.gz"
METADATA_FILE = "vm_metadata.csv.gz"
READING_COLUMNS = ["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"]
METADATA_COLUMNS = [
    "vm_id", "subscription_id", "deployment_id", "timestamp_vm_created",
    "timestamp_vm_deleted", "max_cpu", "avg_cpu", "p95_max_cpu", "vm_category",
    "virtual_core_bucket", "memory_gb_bucket",
]
STATIC_COLUMNS = ["vm_category", "cores", "memory_gb_bucket"]
ORDER_TOL = 1e-9


class DataContractError(ValueError):
    pass


def _read(data_dir: Path, name: str, required: list[str], **kwargs) -> pd.DataFrame:
    path = Path(data_dir) / name
    if not path.is_file():
        raise DataContractError(f"missing file: {name} (looked in {data_dir})")
    try:
        df = pd.read_csv(path, **kwargs)
    except (OSError, EOFError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        raise DataContractError(f"cannot read {name}: {e}") from e
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataContractError(f"{name} is missing required columns: {missing}")
    return df


def load_readings(data_dir: Path) -> pd.DataFrame:
    df = _read(data_dir, READINGS_FILE, READING_COLUMNS, dtype={"vm_id": str})
    return df.sort_values(["timestamp", "vm_id"], kind="stable").reset_index(drop=True)


def load_metadata(data_dir: Path) -> pd.DataFrame:
    return _read(
        data_dir, METADATA_FILE, METADATA_COLUMNS,
        dtype={"vm_id": str, "virtual_core_bucket": str, "memory_gb_bucket": str},
    )


@dataclass
class DataReport:
    n_rows: int
    n_vms: int
    n_timestamps: int
    step_seconds: int
    missing_timestamps: list[int]
    duplicate_rows: int
    invalid_readings: int
    order_violations: int

    def to_dict(self) -> dict:
        return asdict(self)


def _infer_step(timestamps: np.ndarray) -> int:
    diffs = np.diff(np.unique(timestamps))
    if len(diffs) == 0:
        raise DataContractError("cannot infer step from fewer than two timestamps")
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[np.argmax(counts)])


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["timestamp", "min_cpu", "max_cpu", "avg_cpu"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _clean(df: pd.DataFrame, step_seconds: int | None = None) -> tuple[pd.DataFrame, dict]:
    n_rows = len(df)
    df = _numeric(df)
    bad_key = df["timestamp"].isna() | df["vm_id"].isna() | (df["vm_id"].astype(str).str.strip() == "")
    df = df[~bad_key].astype({"timestamp": "int64"})
    step = step_seconds or _infer_step(df["timestamp"].to_numpy())
    t0 = df["timestamp"].min()
    off = df.loc[(df["timestamp"] - t0) % step != 0, "timestamp"]
    if len(off):
        raise DataContractError(
            f"{len(off)} timestamps are off the {step}s grid; first: {int(off.iloc[0])}"
        )
    grid = np.arange(t0, df["timestamp"].max() + 1, step, dtype=np.int64)
    missing = np.setdiff1d(grid, df["timestamp"].unique())

    bad = df["avg_cpu"].isna() | (df["avg_cpu"] < 0) | (df["avg_cpu"] > 100)
    valid = df[~bad]
    clean = (
        valid.groupby(["timestamp", "vm_id"], sort=True, as_index=False)[
            ["min_cpu", "avg_cpu", "max_cpu"]
        ].mean()
        .reset_index(drop=True)
    )
    order = (clean["min_cpu"] > clean["avg_cpu"] + ORDER_TOL) | (
        clean["avg_cpu"] > clean["max_cpu"] + ORDER_TOL
    )
    info = {
        "n_rows": n_rows,
        "step": step,
        "grid": grid,
        "missing": [int(t) for t in missing],
        "duplicate_rows": len(valid) - len(clean),
        "invalid_readings": int(bad_key.sum() + bad.sum()),
        "order_violations": int(order.sum()),
    }
    return clean[READING_COLUMNS], info


def validate_readings(df: pd.DataFrame) -> tuple[pd.DataFrame, DataReport]:
    clean, info = _clean(df)
    report = DataReport(
        n_rows=info["n_rows"],
        n_vms=int(clean["vm_id"].nunique()),
        n_timestamps=len(info["grid"]),
        step_seconds=info["step"],
        missing_timestamps=info["missing"],
        duplicate_rows=info["duplicate_rows"],
        invalid_readings=info["invalid_readings"],
        order_violations=info["order_violations"],
    )
    return clean, report


def core_count(bucket, gt24_cores: int = 32) -> int:
    s = str(bucket).strip()
    return gt24_cores if s.startswith(">") else int(float(s))


@dataclass(frozen=True)
class Panel:
    timestamps: np.ndarray
    vm_ids: np.ndarray
    avg: np.ndarray
    min: np.ndarray
    max: np.ndarray
    static: pd.DataFrame

    @property
    def T(self) -> int:
        return self.avg.shape[1]

    def truncate(self, end: int) -> Panel:
        return Panel(
            self.timestamps[:end].copy(), self.vm_ids.copy(), self.avg[:, :end].copy(),
            self.min[:, :end].copy(), self.max[:, :end].copy(), self.static.copy(),
        )

    def subset(self, mask: np.ndarray) -> Panel:
        mask = np.asarray(mask, dtype=bool)
        return Panel(
            self.timestamps.copy(), self.vm_ids[mask], self.avg[mask].copy(),
            self.min[mask].copy(), self.max[mask].copy(), self.static[mask].copy(),
        )


def build_panel(
    readings: pd.DataFrame,
    metadata: pd.DataFrame,
    step_seconds: int | None = None,
    gt24_cores: int = 32,
) -> Panel:
    df, info = _clean(readings, step_seconds)
    step, timestamps = info["step"], info["grid"]
    vm_ids = np.sort(df["vm_id"].unique())

    meta = metadata.drop_duplicates("vm_id").set_index("vm_id")
    absent = np.setdiff1d(vm_ids, meta.index.to_numpy())
    if len(absent):
        raise DataContractError(f"{len(absent)} VMs in readings are missing from metadata")
    meta = meta.loc[vm_ids]
    static = pd.DataFrame({
        "vm_category": meta["vm_category"].to_numpy(),
        "cores": meta["virtual_core_bucket"].map(lambda b: core_count(b, gt24_cores)).astype(int).to_numpy(),
        "memory_gb_bucket": meta["memory_gb_bucket"].to_numpy(),
    }, index=vm_ids)

    row = np.searchsorted(vm_ids, df["vm_id"].to_numpy())
    col = (df["timestamp"].to_numpy() - timestamps[0]) // step
    arrays = {}
    for name in ["avg", "min", "max"]:
        a = np.full((len(vm_ids), len(timestamps)), np.nan)
        a[row, col] = df[f"{name}_cpu"].to_numpy(dtype=float)
        arrays[name] = a
    return Panel(timestamps, vm_ids, arrays["avg"], arrays["min"], arrays["max"], static)
