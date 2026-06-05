"""Tests for gear-row info (i) tooltips in synthesis_lite global tab."""

from __future__ import annotations

from jinja2 import Template

import synthesis as syn


def _sample_panel():
    return [
        {
            "country": "US",
            "gears": [
                {"title": "A", "l2": "raw exp", "cio": "cio exp"},
                {"title": "B", "l2": "raw plumb", "cio": "cio plumb"},
                {"title": "C", "l2": "raw hedge", "cio": "cio hedge"},
            ],
        }
    ]


def _sample_l2b():
    return {
        "signals": {
            "US": {"signal": "GREEN", "reasons": ["Equity inflows", "Credit tight"]},
        },
        "flow_payload": {
            "fx_1m_local_vs_usd": {"US": 0.6},
            "bond_spreads_bp": {"US": 0},
        },
    }


def _sample_quant():
    return {
        "Risk_Analytics": {
            "layer2_risk": {
                "SKEW": {"latest_level": 135.2, "signal": "elevated"},
                "VVIX": {"signal": "normal"},
            }
        }
    }


def test_normalize_gear_reasoning_bilingual():
    block = syn._normalize_gear_block_bilingual(
        {
            "expectation_reasoning_en": "Signal GREEN because inflows.",
            "expectation_reasoning_zh": "綠燈因資金流入。",
        }
    )
    assert block["expectation_reasoning_en"] == "Signal GREEN because inflows."
    assert block["expectation_reasoning_zh"] == "綠燈因資金流入。"


def test_gear_reasoning_llm_wins_over_quant():
    panels = syn._lite_country_matrix(
        _sample_panel(),
        {
            "US": {
                "expectation_en": "Stable growth",
                "expectation_zh": "穩定增長",
                "expectation_reasoning_en": "LLM explains GREEN signal.",
                "expectation_reasoning_zh": "LLM 解讀綠燈。",
                "plumbing_en": "Flows",
                "plumbing_zh": "資金流",
                "hedging_en": "Caution",
                "hedging_zh": "謹慎",
            }
        },
        "zh",
        l2b=_sample_l2b(),
        quant_context_json=_sample_quant(),
    )
    assert panels[0]["gears"][0]["reasoning"] == "LLM 解讀綠燈。"


def test_gear_reasoning_quant_fallback_when_no_llm():
    panels = syn._lite_country_matrix(
        _sample_panel(),
        {
            "US": {
                "expectation_en": "Stable growth",
                "expectation_zh": "穩定增長",
                "plumbing_en": "Flows",
                "plumbing_zh": "資金流",
                "hedging_en": "Caution",
                "hedging_zh": "謹慎",
            }
        },
        "zh",
        l2b=_sample_l2b(),
        quant_context_json=_sample_quant(),
    )
    assert "綠燈" in panels[0]["gears"][0]["reasoning"]
    assert "0.60%" in panels[0]["gears"][1]["reasoning"]
    assert "SKEW" in panels[0]["gears"][2]["reasoning"]


def test_lite_html_cty_row_has_info():
    gear_sem = {
        "US": {
            "expectation_zh": "穩定增長",
            "plumbing_zh": "資金流",
            "hedging_zh": "謹慎",
        }
    }
    blocks = syn._build_lite_lang_blocks(
        report_date="2026-06-03",
        macro_regime_label="Goldilocks",
        regime_gauge_tone="stable",
        divergence_gauge={"score": 40},
        daily_themes=[],
        directive={
            "the_stance_en": "S",
            "the_stance_zh": "立",
            "the_stance": "S",
            "the_narrative_en": ["A"],
            "the_narrative_zh": ["甲"],
            "the_narrative": ["A"],
            "the_watchout_en": "W",
            "the_watchout_zh": "觀",
            "the_watchout": "W",
        },
        watch_bilingual=[],
        regime_asset_tilt_label="Goldilocks",
        country_matrix=_sample_panel(),
        gear_semantics=gear_sem,
        vs_us_alignment={},
        historical_matches_en=[],
        divergence_matches_en=[],
        historical_descriptions_zh=[],
        divergence_descriptions_zh=[],
        history_translation_fallback=False,
        l2b=_sample_l2b(),
        quant_context_json=_sample_quant(),
    )
    html = Template(syn.HTML_TMPL_LITE).render(
        report_date="2026-06-03",
        lite_lang_blocks=blocks,
        llm_meta={"provider": "deepseek", "status": "ok", "model": "deepseek-chat", "error": None},
        llm_placeholder=False,
        rrg_data="{}",
        rrg_summary={},
        rrg_sector_stats=[],
        sector_performance=[],
        rrg_rank_trans={},
        tw_rrg_data="{}",
        tw_rrg_summary={},
        tw_rrg_sector_stats=[],
        tw_sector_performance=[],
        tw_rrg_rank_trans={},
        regime_quadrant_data=None,
        zscore_data_zh=None,
        zscore_data_en=None,
        regime_prob_ts_data="{}",
    )
    assert "cty-row-head" in html
    assert 'class="info' in html
    assert "綠燈" in html or "SKEW" in html
    assert "positionInfoTip" in html
