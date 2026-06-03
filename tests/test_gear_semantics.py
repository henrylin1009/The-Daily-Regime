"""Gear matrix semantic translation helpers (no network)."""

from src.macra_ui import (
    apply_gear_semantics_to_gears,
    apply_gear_semantics_to_l3_panel,
    apply_gear_semantics_to_l3_view_model,
    build_l2_gear_columns,
)


def test_apply_gear_semantics_to_gears_replaces_machine_speak():
    raw = build_l2_gear_columns("US", {"signals": {"US": {"signal": "RED", "reasons": ["FX drag"]}}}, {})
    out = apply_gear_semantics_to_gears(
        raw,
        {
            "expectation": "Markets are fighting gravity; tactical shorts have edge.",
            "plumbing": "Capital is leaking into USD on spread compression.",
            "hedging": "Complacency dominates — buy downside protection.",
        },
    )
    assert "Signal RED" not in out["expectation"]
    assert "Capital is leaking" in out["plumbing"]
    assert "downside protection" in out["hedging"]


def test_apply_gear_semantics_to_l3_panel_keeps_recent_and_cio():
    panel = {
        "country": "Japan",
        "gears": [
            {
                "title": "Expectation Arbitrage",
                "l1": "L1 raw",
                "l2": "Signal YELLOW. Drivers: FX.",
                "recent": "YELLOW: FX drag",
                "cio": "CIO note stays",
            },
            {
                "title": "Crowdedness & Leverage",
                "l1": "L1 plumbing",
                "l2": "FX 1m vs USD: +1.00%",
                "recent": "YELLOW: FX drag",
                "cio": "Crowded CIO",
            },
            {
                "title": "Hedging Cost Anomaly",
                "l1": "L1 premium",
                "l2": "SKEW 140",
                "recent": "SKEW 140",
                "cio": "Hedge CIO",
            },
        ],
    }
    apply_gear_semantics_to_l3_panel(
        panel,
        {
            "expectation": "JPY carry is the pressure valve — fade the bounce.",
            "plumbing": "Flows favor USD; local liquidity is draining.",
            "hedging": "Tail hedges are cheap — load protection.",
            "expectation_label": "Carry Unwind Risk",
            "plumbing_label": "USD Magnet Flows",
            "hedging_label": "Cheap Tail Insurance",
        },
    )
    assert panel["gears"][0]["l2"].startswith("JPY carry")
    assert panel["gears"][0]["title"] == "Carry Unwind Risk"
    assert panel["gears"][0]["recent"] == "YELLOW: FX drag"
    assert panel["gears"][0]["cio"] == "CIO note stays"
    assert panel["gears"][1]["l1"] == "L1 plumbing"


def test_apply_gear_semantics_to_l3_view_model_all_countries():
    l3_vm = {
        "country_matrix": [
            {"country": "US", "gears": [{"title": "A", "l2": "raw", "recent": "r", "cio": "c"}]},
            {"country": "Japan", "gears": [{"title": "A", "l2": "raw", "recent": "r", "cio": "c"}]},
        ]
    }
    apply_gear_semantics_to_l3_view_model(
        l3_vm,
        {
            "US": {"expectation": "US translated", "expectation_label": "Gravity Fight"},
            "Japan": {"expectation": "JP translated", "expectation_label": "Carry Stress"},
        },
    )
    assert l3_vm["country_matrix"][0]["gears"][0]["l2"] == "US translated"
    assert l3_vm["country_matrix"][1]["gears"][0]["title"] == "Carry Stress"
