"""War Room 3.0 synthesis normalization (no network)."""

from synthesis import (
    FALLBACK,
    _coerce_narrative,
    _migrate_legacy_synthesis,
    _normalize_synthesis,
    is_synthesis_placeholder,
)


def test_legacy_migration_populates_zones():
    legacy = {
        "pre_analysis": {"cross_layer_conflict": "x", "momentum_verification": "y", "event_pulse_analysis": "z"},
        "macro_pulse": {"event_title": "TW flow", "event_metrics": "big inflow", "cio_verdict": "Stay cautious"},
        "cio_macro_thesis": "Flows risk-on but structure heavy.",
        "structural_vs_tactical_dynamics": "L1 cautious.\n\nL2 flows green.",
        "core_risks_actionable": ["If vol spikes, cut beta."],
        "headline": "Tactical over structural",
        "convergence_divergence": {"status": "DIVERGENCE", "reason": "mixed"},
    }
    out = _normalize_synthesis(_migrate_legacy_synthesis(legacy))
    assert out["zone1_pulse"]["flash_bullets"]["capital_flows"]
    assert out["cio_directive"]["the_stance"] == "Stay cautious"
    pa = out["relationship_analysis"]["pure_alpha"]
    assert pa["expectation_arbitrage"]
    assert pa["crowdedness_leverage"]
    assert len(_coerce_narrative(out["cio_directive"]["the_narrative"])) == 2
    assert out["relationship_analysis"]["status"] == "DIVERGENCE"


def test_coerce_narrative_splits_paragraphs():
    assert len(_coerce_narrative("First.\n\nSecond.")) == 2


def test_placeholder_detects_war_room_fields():
    assert is_synthesis_placeholder(
        {
            "cio_directive": {"the_stance": FALLBACK},
            "relationship_analysis": {"pure_alpha": {"expectation_arbitrage": FALLBACK}},
        }
    )
    assert not is_synthesis_placeholder(
        {
            "cio_directive": {"the_stance": "Hold risk assets", "the_narrative": ["a", "b"]},
            "relationship_analysis": {
                "pure_alpha": {
                    "expectation_arbitrage": "L2 prices cuts into Goldilocks.",
                    "crowdedness_leverage": "SPY/HYG improving; RSP lagging.",
                    "hedging_cost_anomaly": "SKEW elevated vs rally.",
                }
            },
        }
    )
