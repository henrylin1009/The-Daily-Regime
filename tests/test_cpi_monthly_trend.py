"""CPI YoY uses monthly MoM, not daily ffill slope."""

import pandas as pd

from src.macro_quant_engine import _cpi_yoy_monthly_trend, get_momentum_profile


def test_cpi_yoy_from_synthetic_fred(monkeypatch):
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    level = pd.Series(range(100, 124), index=idx, dtype=float)
    monkeypatch.setattr(
        "src.macro_quant_engine._fred_series",
        lambda _id: level,
    )
    out = _cpi_yoy_monthly_trend()
    assert out["trend"] in ("Accelerating", "Decelerating", "Unchanged", "Stale")
    assert out["latest_yoy"] is not None


def test_ffill_flat_series_has_no_momentum_signal():
    daily = pd.Series(3.0, index=pd.date_range("2024-01-01", periods=30, freq="B"))
    prof = get_momentum_profile(daily, good_when_z_rises=True, days=20)
    assert prof["label"] in ("Stable", "Unknown")
    assert prof.get("z_change_20d") == 0 or prof["label"] == "Stable"
