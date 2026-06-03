"""Penalized divergence scoring (no network)."""

import pandas as pd

from src.macro_quant_engine import compute_penalized_divergence


def test_hedging_penalty():
    df = pd.DataFrame(
        {
            "US_Growth_Z": [-1.0],
            "ERP_Z": [-1.5],
            "Spread_2y10y_Z": [0.5],
            "SPY_Z": [2.0],
            "HYG_Z": [1.0],
            "VIX_Z": [-0.5],
            "SKEW": [135.0],
        }
    )
    mom = {"SPY": {"label": "Improving", "z_latest": 2.0}, "HYG": {"label": "Stable"}}
    out = compute_penalized_divergence(df, mom)
    assert out["divergence_score"] >= 20
    assert "Hedging Cost Anomaly" in out["penalty_reasons"]


def test_crowded_penalty():
    n = 25
    z_rsp = list(range(n, 0, -1))
    df = pd.DataFrame(
        {
            "US_Growth_Z": [0.0] * n,
            "SPY_Z": [2.0] * n,
            "HYG_Z": [0.5] * n,
            "VIX_Z": [0.0] * n,
            "RSP_SPY_Ratio_Z": z_rsp,
        }
    )
    mom = {"SPY": {"label": "Improving", "z_latest": 2.0}}
    out = compute_penalized_divergence(df, mom)
    assert "Extreme Crowdedness/Leverage" in out["penalty_reasons"] or out["divergence_score"] <= 100


def test_score_capped_at_100():
    df = pd.DataFrame(
        {
            "US_Growth_Z": [-3.0],
            "ERP_Z": [-3.0],
            "SPY_Z": [3.0],
            "HYG_Z": [3.0],
            "VIX_Z": [3.0],
            "SKEW": [150.0],
        }
    )
    mom = {"SPY": {"label": "Improving", "z_latest": 3.0}}
    out = compute_penalized_divergence(df, mom)
    assert out["divergence_score"] <= 100
