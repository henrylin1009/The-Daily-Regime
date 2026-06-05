"""Tests for Taiwan sector rotation (FinMind indices + RRG)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.sector_rotation import TW_SECTORS, build_rrg_plotly, build_tw_rrg_plotly


def _synthetic_tw_cache(n: int = 900, freq: str = "B") -> dict[str, pd.DataFrame]:
    """Build long daily series for TAIEX + five industry indices."""
    dates = pd.bdate_range("2018-01-01", periods=n, freq=freq)
    rng = np.random.default_rng(42)
    bench = 10000 + np.cumsum(rng.normal(0.05, 8, size=n))
    data: dict[str, pd.DataFrame] = {
        "tw_bench": pd.DataFrame({"date": dates, "value": bench}),
    }
    drift = {"tw_semi": 0.08, "tw_comp": 0.03, "tw_fin": 0.01, "tw_ship": -0.02, "tw_mach": 0.04}
    for key, d in drift.items():
        series = bench * (1 + d) + np.cumsum(rng.normal(0, 12, size=n))
        data[key] = pd.DataFrame({"date": dates, "value": series})
    return data


def test_tw_rrg_plotly_returns_traces():
    result = build_tw_rrg_plotly(_synthetic_tw_cache())
    assert result.get("traces"), "expected Plotly traces"
    assert result.get("layout")
    assert result.get("summary", {}).get("en")
    assert len(result.get("sector_stats", [])) >= 3
    assert len(result.get("sector_performance", [])) >= 2


def test_tw_universe_parameter_matches_wrapper():
    direct = build_rrg_plotly(_synthetic_tw_cache(), universe=TW_SECTORS)
    wrapped = build_tw_rrg_plotly(_synthetic_tw_cache())
    assert len(direct.get("traces", [])) == len(wrapped.get("traces", []))
    assert direct.get("sector_performance")[0]["label"] == "TAIEX"
    labels = {r["label"]: r.get("label_zh") for r in direct["sector_performance"]}
    assert labels["Semiconductor ETF (00892)"] == "半導體 ETF (00892)"
    assert labels["High Dividend ETF (00919)"] == "高息 ETF (00919)"


def test_fetch_finmind_v4_parses_close():
    from src.collect import _fetch_finmind_v4

    payload = {
        "status": 200,
        "data": [
            {"date": "2024-01-02", "close": 178.5},
            {"date": "2024-01-03", "close": 179.1},
        ],
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = payload

    with patch("src.collect.requests.get", return_value=mock_resp):
        df = _fetch_finmind_v4("TaiwanStockPrice", "00892", "2024-01-01")

    assert len(df) == 2
    assert df["value"].iloc[-1] == pytest.approx(179.1)


def test_fetch_finmind_v4_empty_on_bad_status():
    from src.collect import _fetch_finmind_v4

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"status": 400, "data": []}

    with patch("src.collect.requests.get", return_value=mock_resp):
        df = _fetch_finmind_v4("TaiwanStockTotalReturnIndex", "TAIEX", "2024-01-01", value_col="price")

    assert df.empty
