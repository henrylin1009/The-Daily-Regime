"""Every divergence analogue EN lookup key must have a ZH counterpart."""

from __future__ import annotations

from src.divergence_match import (
    DIVERGENCE_PERIOD_DESCRIPTIONS,
    DIVERGENCE_PERIOD_DESCRIPTIONS_ZH,
    _period_desc,
)


def test_divergence_period_descriptions_zh_keys_match_en():
    assert set(DIVERGENCE_PERIOD_DESCRIPTIONS_ZH.keys()) == set(DIVERGENCE_PERIOD_DESCRIPTIONS.keys())


def test_period_desc_returns_both_languages():
    period = "2014-03"
    en = _period_desc(period, "en")
    zh = _period_desc(period, "zh")
    assert en == DIVERGENCE_PERIOD_DESCRIPTIONS["2014"]
    assert zh == DIVERGENCE_PERIOD_DESCRIPTIONS_ZH["2014"]
    assert en != zh
