import builtins

import numpy as np
import pytest

from cwfc import fm
from cwfc.data import build_panel
from cwfc.fm import Chronos, fm_available, fm_candidates

T_ = "aggregate_cpu_utilization"


def test_fm_missing_gives_clear_error(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.split(".")[0] in ("torch", "chronos"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    assert not fm_available()
    with pytest.raises(RuntimeError, match=r'pip install -e "\.\[fm\]"'):
        Chronos()


def test_candidates_and_contamination_filter(monkeypatch):
    monkeypatch.setattr(fm, "fm_available", lambda: True)  # constructors only; no model is loaded
    names = [m.name for m in fm_candidates(T_, include_flagged=True)]
    assert names == ["chronos_bolt_base", "chronos_2", "chronos_bolt_base_bottom_up"]
    assert [m.name for m in fm_candidates("active_vm_count", include_flagged=True)] == names[:2]
    assert "chronos_2" not in [m.name for m in fm_candidates(T_)]
    monkeypatch.setitem(fm.CONTAMINATION, "amazon/chronos-bolt-base", (True, "x"))
    assert [m.name for m in fm_candidates(T_)] == []
    assert all(m.complexity == 4 for m in fm_candidates(T_, include_flagged=True))


def test_unknown_model_id_is_excluded_unless_flagged_included(monkeypatch):
    monkeypatch.setattr(fm, "fm_available", lambda: True)
    monkeypatch.delitem(fm.CONTAMINATION, "amazon/chronos-bolt-base")
    assert "chronos_bolt_base" not in [m.name for m in fm_candidates(T_)]
    assert "chronos_bolt_base" in [m.name for m in fm_candidates(T_, include_flagged=True)]


@pytest.mark.skipif(not fm_available(), reason="fm extras not installed")
@pytest.mark.parametrize("mode", ["aggregate", "bottom_up"])
def test_chronos_aggregate_shape(make_readings, mode):
    panel = build_panel(*make_readings(8, 40, churn=True))
    model = Chronos("amazon/chronos-bolt-tiny", mode=mode, device="cpu")
    try:
        out = model.predict(panel, T_, 6)
    except OSError:
        pytest.skip("model not available offline")
    assert out.shape == (6,) and np.isfinite(out).all()


def test_bottom_up_aggregates_per_vm_mean(monkeypatch):
    from cwfc.data import Panel
    import pandas as pd

    rng = np.random.default_rng(0)
    avg = rng.lognormal(1.0, 1.0, (50, 20))
    panel = Panel(np.arange(20) * 300, np.array([f"v{i}" for i in range(50)]), avg, avg, avg,
                  pd.DataFrame({"vm_category": "x", "cores": 2, "memory_gb_bucket": "4"},
                               index=range(50)))
    seen = {}

    def fake(self, rows, horizon, stat="median"):
        seen["stat"] = stat
        return np.repeat(rows.mean(axis=1, keepdims=True), horizon, axis=1)

    monkeypatch.setattr(Chronos, "_point", fake)
    monkeypatch.setattr(fm, "fm_available", lambda: True)
    out = Chronos(mode="bottom_up").predict(panel, T_, 3)
    assert seen["stat"] == "mean"
    assert np.allclose(out, avg.mean())
