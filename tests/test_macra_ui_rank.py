"""Tests for MACRA 3.0 Phase 2 ranking heuristics and formatting."""

from src.macra_ui import (
    build_country_rank_rows,
    build_l1_gear_columns,
    format_accordion_summary,
    spread_trade_line,
)


def test_l2_red_ranks_before_green():
    rows = build_country_rank_rows(
        [
            {"country": "Japan", "signal": "GREEN", "spread_10y": "10.0 bp", "spread_bp": 10.0},
            {"country": "China", "signal": "RED", "spread_10y": "5.0 bp", "spread_bp": 5.0},
        ],
        layer="L2",
    )
    assert rows[0]["country"] == "China"
    assert rows[0]["rank"] == 1
    assert rows[-1]["country"] == "Japan"


def test_l2_tie_break_spread_then_alpha():
    rows = build_country_rank_rows(
        [
            {"country": "Japan", "signal": "YELLOW", "spread_bp": 5.0},
            {"country": "China", "signal": "YELLOW", "spread_bp": 20.0},
        ],
        layer="L2",
    )
    assert rows[0]["country"] == "China"
    assert rows[1]["country"] == "Japan"


def test_us_core_flag():
    rows = build_country_rank_rows(
        [{"country": "US", "signal": "GREEN"}, {"country": "Taiwan", "signal": "RED"}],
        layer="L2",
    )
    us = next(r for r in rows if r["country"] == "US")
    assert us["is_us_core"] is True


def test_spread_trade_uses_rank_1_and_6():
    ranked = build_country_rank_rows(
        [
            {"country": "US", "signal": "GREEN"},
            {"country": "Japan", "signal": "YELLOW"},
            {"country": "China", "signal": "RED"},
        ],
        layer="L2",
    )
    line = spread_trade_line(ranked)
    assert "[LONG]" in line
    assert "[SHORT]" in line
    assert "China" in line
    assert "US" in line


def test_format_accordion_summary():
    s = format_accordion_summary({"country": "US", "is_us_core": True, "signal": "YELLOW"}, 1, 43)
    assert s["rank_label"] == "[ Rank 1 ]"
    assert s["is_core"] is True
    assert "43" in s["divergence_text"]


def test_l1_none_policy_rate_formats_cleanly():
    payload = {
        "module_a": {
            "policy_rates": {
                "Taiwan": {"latest": None, "direction": "on hold", "mom_change": None},
            }
        },
        "module_c": {},
        "module_d": {},
        "module_e": {"t3mff": {}, "tedrate": {}},
        "llm": {},
    }
    cols = build_l1_gear_columns("Taiwan", payload)
    assert "None" not in cols["gravity"]
    assert "—" in cols["gravity"]
