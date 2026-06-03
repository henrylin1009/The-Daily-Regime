"""Appendix anomaly scan extraction tests."""

from src.appendix_exceptions import build_exception_dashboard


def _l1_minimal():
    return {
        "cot_rows": [
            {
                "contract": "JPY",
                "net_fmt": "100",
                "chg4w_fmt": "-5",
                "zscore_fmt": "2.35",
                "extreme": True,
                "net_position": -100,
                "chg4w": -5,
            },
            {
                "contract": "EUR",
                "net_fmt": "50",
                "chg4w_fmt": "1",
                "zscore_fmt": "0.80",
                "extreme": True,
                "net_position": 50,
                "chg4w": 1,
            },
        ],
        "liq_rows": [
            {"indicator": "TEDRATE", "value": "0.6", "change": "0.1", "signal": "stress"},
        ],
    }


def test_anomaly_scan_high_z_cftc():
    rows = build_exception_dashboard(_l1_minimal(), {}, {"l3_available": False}, [])
    labels = [r["label"] for r in rows]
    assert "JPY" in labels
    assert "EUR" not in labels


def test_anomaly_scan_ignores_liquidity_text_without_red_or_z():
    rows = build_exception_dashboard(_l1_minimal(), {}, {"l3_available": False}, [])
    assert not any(r["label"] == "TEDRATE" for r in rows)


def test_anomaly_scan_l2_red_country():
    country_rows = [{"country": "Europe", "signal": "RED", "fx_1m": "-1.2%", "spread_10y": "+40bp"}]
    rows = build_exception_dashboard(_l1_minimal(), {}, {"l3_available": False}, country_rows)
    assert any(r["label"] == "Europe" and r["signal"] == "RED" for r in rows)


def test_anomaly_scan_surfaced_red_and_high_z():
    l3 = {
        "l3_available": True,
        "event_of_day": {"name": "MOVE", "latest": "120", "change": "+5", "z": "2.5", "signal": "YELLOW"},
        "surfaced_rows": [
            {"name": "HYG flow", "latest": "1.2", "change": "-0.3", "z": "2.8", "signal": "RED"},
            {"name": "DXY", "latest": "104", "change": "+0.1", "z": "0.5", "signal": "GREEN"},
        ],
        "validation_panel": [],
        "quant_momentum": {"VIX": {"label": "Deteriorating", "z_latest": "1.2"}},
        "vol_skew_rows": [],
    }
    rows = build_exception_dashboard({}, {}, l3, [])
    labels = [r["label"] for r in rows]
    assert "HYG flow" in labels
    assert "MOVE" in labels
    assert "DXY" not in labels
    assert "VIX" not in labels
