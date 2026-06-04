"""Tests for synthesis_lite bilingual UI and normalization."""

from __future__ import annotations

from jinja2 import Template

import synthesis as syn


def test_bilingual_pair_legacy_fallback():
    en, zh = syn._bilingual_pair({"title": "Hello"}, "title")
    assert en == "Hello"
    assert zh == syn.FALLBACK


def test_normalize_daily_themes_bilingual():
    themes = syn._normalize_daily_themes_bilingual(
        [
            {
                "title_en": "Yen extreme",
                "title_zh": "日圓極端",
                "body_en": "Body EN",
                "body_zh": "內文",
                "implication_en": "Imp EN",
                "implication_zh": "意義",
                "supporting_countries": ["Japan"],
            }
        ]
    )
    assert themes[0]["title"] == "Yen extreme"
    assert themes[0]["title_zh"] == "日圓極端"
    zh_themes = syn._lite_daily_themes(themes, "zh")
    assert zh_themes[0]["title"] == "日圓極端"


def test_regime_label_display_zh():
    assert syn._regime_label_display("Goldilocks", "zh") == "金絲雀"
    assert syn._regime_label_display("Goldilocks", "en") == "Goldilocks"


def test_regime_asset_tilt_for_lang():
    rows = syn._regime_asset_tilt_for_lang("Goldilocks", "zh")
    assets = {r["asset"] for r in rows}
    assert "股票" in assets
    assert "Equities" not in assets


def test_lite_template_has_lang_switch_and_blocks():
    blocks = syn._build_lite_lang_blocks(
        report_date="2026-05-31",
        macro_regime_label="Goldilocks",
        regime_gauge_tone="stable",
        divergence_gauge={"score": 43},
        daily_themes=[],
        directive={
            "the_stance_en": "Stance",
            "the_stance_zh": "立場",
            "the_stance": "Stance",
            "the_narrative_en": ["A"],
            "the_narrative_zh": ["甲"],
            "the_narrative": ["A"],
            "the_watchout_en": "Watch",
            "the_watchout_zh": "觀察",
            "the_watchout": "Watch",
            "positioning_6_12m_en": "",
            "positioning_6_12m_zh": "",
            "positioning_6_12m": "",
        },
        watch_bilingual=[{"en": "Item", "zh": "項目"}],
        regime_asset_tilt_label="Goldilocks",
        country_matrix=[],
        gear_semantics={},
        vs_us_alignment={},
        historical_matches_en=[],
        divergence_matches_en=[],
        historical_descriptions_zh=[],
        divergence_descriptions_zh=[],
        history_translation_fallback=False,
    )
    html = Template(syn.HTML_TMPL_LITE).render(
        report_date="2026-05-31",
        lite_lang_blocks=blocks,
        llm_meta={"provider": "deepseek", "status": "ok", "model": "deepseek-chat", "error": None},
        llm_placeholder=False,
    )
    assert "llm-banner--ok" in html
    assert 'class="lang-switch"' in html
    assert 'data-lang="zh-Hant"' in html
    assert 'data-lang="en"' in html
    assert "macraLiteLang" in html
    assert "lang-block" in html


def test_normalize_watch_list_bilingual_from_dicts():
    en_list, bi = syn._normalize_watch_list_bilingual(
        [
            {
                "en": "US CPI next release",
                "zh": "美國下次CPI發布",
            }
        ]
    )
    assert en_list == ["US CPI next release"]
    assert bi[0]["zh"] == "美國下次CPI發布"
    zh_items = syn._lite_critical_watchout(bi, "zh")
    assert zh_items == [{"text": "美國下次CPI發布", "reasoning": ""}]
    en_items = syn._lite_critical_watchout(bi, "en")
    assert en_items == [{"text": "US CPI next release", "reasoning": ""}]


def test_lite_theme_kicker_zh():
    assert syn._lite_theme_kicker(["US", "Japan"], "zh") == "美國 · 日本"


def test_lite_positioning_dedup():
    themes = [{"implication": "Same text about equities and bonds for twelve months."}]
    directive = {"positioning_6_12m_en": "Same text about equities and bonds for twelve months."}
    view = syn._lite_directive_view(directive, "en")
    assert syn._lite_positioning_text(view, themes) is None


def test_alignment_label_for_lang():
    align = {"label": "與美國一致", "label_en": "Aligned with US", "label_zh": "與美國一致"}
    assert syn._alignment_label_for_lang(align, "en") == "Aligned with US"
    assert syn._alignment_label_for_lang(align, "zh") == "與美國一致"
