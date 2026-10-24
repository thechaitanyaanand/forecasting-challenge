from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STEP = 300
CATEGORIES = ["Delay-insensitive", "Interactive", "Unknown"]
CORES = ["2", "4", "8", "24", ">24"]
MEMORY = ["2", "4", "8", "32", "64", ">64"]


@pytest.fixture
def make_readings():
    def factory(n_vms: int, n_steps: int, seed: int = 0, churn: bool = False):
        rng = np.random.default_rng(seed)
        rows = []
        meta = []
        for i in range(n_vms):
            start, end = 0, n_steps
            if churn and i % 3 == 1:
                start = int(rng.integers(1, max(2, n_steps // 3)))
            if churn and i % 3 == 2:
                end = int(rng.integers(n_steps // 2, n_steps))
            n = end - start
            avg = np.clip(rng.uniform(5, 80) + rng.normal(0, 5, n), 0, 100)
            lo = np.clip(avg - rng.uniform(0, 5, n), 0, 100)
            hi = np.clip(avg + rng.uniform(0, 5, n), 0, 100)
            vm = f"vm{i:03d}"
            rows.append(pd.DataFrame({
                "timestamp": np.arange(start, end, dtype=np.int64) * STEP,
                "vm_id": vm, "min_cpu": lo, "max_cpu": hi, "avg_cpu": avg,
            }))
            meta.append({
                "vm_id": vm, "subscription_id": f"sub{i % 3}", "deployment_id": f"dep{i}",
                "timestamp_vm_created": start * STEP, "timestamp_vm_deleted": end * STEP,
                "max_cpu": float(hi.max()), "avg_cpu": float(avg.mean()),
                "p95_max_cpu": float(np.percentile(hi, 95)),
                "vm_category": CATEGORIES[i % 3],
                "virtual_core_bucket": CORES[i % len(CORES)],
                "memory_gb_bucket": MEMORY[i % len(MEMORY)],
            })
        return pd.concat(rows, ignore_index=True), pd.DataFrame(meta)

    return factory


@pytest.fixture
def write_dataset():
    def write(tmp_path: Path, readings: pd.DataFrame, metadata: pd.DataFrame) -> Path:
        readings.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
        metadata.to_csv(tmp_path / "vm_metadata.csv.gz", index=False)
        return tmp_path

    return write
