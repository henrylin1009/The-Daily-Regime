"""Conservative slim: dedupe indicator blocks, compact JSON, full quant structure kept."""

from __future__ import annotations

from unittest.mock import patch

import synthesis as syn


def _minimal_prompt_inputs():
    quant = {
        "macro_regime_label": "Goldilocks",
        "Layer_2_Tactical": {"growth": "improving"},
        "momentum_profile": {"SPY": {"label": "Improving"}},
        "Risk_Analytics": {
            "divergence_score": 42,
            "penalty_reason": "L1/L2 gap",
            "penalty_reasons": ["gap"],
            "layer2_risk": {},
        },
        "Fed_Liquidity": {"walcl_bn": 7000, "summary": "prose to strip"},
        "Fed_Rate_Path": {"cuts_2026": 2, "summary": "prose to strip"},
    }
    l1 = {"llm": {"structural": "ok"}, "module_a": {"x": 1}}
    l2a = {"signals": {}, "executive_summary": "", "narrative": "", "matches": []}
    l2b = {"flow_payload": {"signals": {}}, "data_coverage": {}, "llm": {}}
    return l1, l2a, l2b, quant


def test_build_synthesis_prompt_conservative_slim():
    l1, l2a, l2b, quant = _minimal_prompt_inputs()
    with (
        patch.object(syn, "_market_snapshot_for_prompt", return_value={"spy": 1}),
        patch.object(syn, "_build_carry_relative", return_value={}),
        patch.object(syn, "_build_indicator_pulse_block", return_value=[{"name": "VIX"}]),
        patch.object(syn, "_build_flagged_signals", return_value="VIX elevated"),
        patch.object(syn, "_format_historical_analogues_for_llm", return_value="(none)"),
    ):
        prompt = syn._build_synthesis_prompt(
            l1, l2a, l2b, quant, {"headline": "evt"}, [], "2026-06-07"
        )

    assert "indicator_pulse" not in prompt.lower()
    assert "us_financial_conditions" not in prompt.lower()
    assert "PRE-FLAGGED SIGNALS" in prompt
    assert "VIX elevated" in prompt
    assert "Layer_2_Tactical" in prompt
    assert '"walcl_bn":7000' in prompt or '"walcl_bn": 7000' in prompt
    assert "prose to strip" not in prompt
    assert "\n  " not in prompt.split("Quant Engine")[1][:500]


def test_slim_divergence_context_keeps_core_fields():
    full = {
        "source": "quant_engine",
        "divergence_score": 55,
        "likely_status": "DIVERGENCE",
        "penalty_reason": "gap",
        "drivers": ["a", "b", "c", "d"],
        "macro_regime_label": "Goldilocks",
    }
    slim = syn._slim_divergence_context(full)
    assert slim["divergence_score"] == 55
    assert slim["likely_status"] == "DIVERGENCE"
    assert slim["penalty_reason"] == "gap"
    assert slim["drivers"] == ["a", "b", "c"]
    assert "macro_regime_label" not in slim
