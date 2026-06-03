"""Tests for S&P 500 contributors module.

We intentionally avoid network downloads during CI by only running when
local caches exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.collect import load_all_from_cache
from src.index_contrib import (
    SPX_PRICE_CACHE,
    SPX_TICKERS_CACHE,
    SPY_HOLDINGS_CACHE,
    compute_spx_contributions,
    spx_contrib_to_factor_panels,
)


def _raw_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "raw"


def test_spx_contrib_structure_or_skip():
    raw_dir = _raw_dir()
    assert raw_dir.exists()

    # Cache prerequisites:
    # - If holdings weights cache exists, tickers cache is optional.
    # - If holdings cache doesn't exist, we need tickers cache to avoid Wikipedia fetch.
    need_prices = SPX_PRICE_CACHE.exists()
    have_holdings = SPY_HOLDINGS_CACHE.exists()
    have_tickers = SPX_TICKERS_CACHE.exists()

    if not need_prices:
        pytest.skip("No price cache found; run the pipeline once to generate cache.")
    if not have_holdings and not have_tickers:
        pytest.skip("No holdings/tickers cache found; would require network fetch.")

    data = load_all_from_cache()
    if "spy" not in data or data["spy"].empty:
        pytest.skip("No cached spy series found.")

    result = compute_spx_contributions(
        data,
        allow_download=False,
        force_refresh=False,
        top_n=3,
    )

    assert "available" in result
    assert result["available"] in {True, False}
    assert "windows" in result
    assert "1d" in result["windows"]
    assert "21d" in result["windows"]

    if result["available"]:
        for win_id in ("1d", "21d"):
            w = result["windows"][win_id]
            for key in ("top_positive", "top_negative"):
                for row in w.get(key, []):
                    assert "ticker" in row
                    assert "weight_pct" in row
                    assert "return_pct" in row
                    assert "contribution_pct" in row
            assert "spy_return_pct" in w
            assert "explained_pct" in w

        panels = spx_contrib_to_factor_panels(result)
        assert len(panels) == 2
        available = [p for p in panels if p.get("available")]
        if available:
            p = available[0]
            assert p.get("decomposition") is True
            assert p["factors"][0]["share_pct"] is not None


def test_spx_factor_panel_shape():
    spx = {
        "available": True,
        "as_of": "2026-05-26",
        "note": "test",
        "windows": {
            "1d": {
                "spy_return_pct": 0.5,
                "explained_pct": 0.48,
                "residual_pct": 0.02,
                "constituents_used": 500,
                "top_positive": [
                    {
                        "ticker": "MU",
                        "weight_pct": 1.3,
                        "return_pct": 19.0,
                        "contribution_pct": 0.25,
                    }
                ],
                "top_negative": [
                    {
                        "ticker": "XOM",
                        "weight_pct": 1.0,
                        "return_pct": -3.0,
                        "contribution_pct": -0.03,
                    }
                ],
            },
            "21d": {
                "spy_return_pct": 2.0,
                "explained_pct": 1.9,
                "residual_pct": 0.1,
                "constituents_used": 500,
                "top_positive": [],
                "top_negative": [],
            },
        },
    }
    panels = spx_contrib_to_factor_panels(spx)
    assert panels[0]["available"]
    assert panels[0]["factors"][0]["code"] == "MU"
    assert panels[0]["factors"][-1]["code"] == "XOM"
    assert panels[0]["factors"][0]["share_pct"] == 50.0
    contribs = [f["contribution_pct"] for f in panels[0]["factors"]]
    assert contribs == sorted(contribs, reverse=True)

