"""Every historical analogue EN lookup key must have a ZH counterpart."""

from __future__ import annotations

from src.history_match import PERIOD_DESCRIPTIONS, PERIOD_DESCRIPTIONS_ZH, _period_description


def test_period_descriptions_zh_keys_match_en():
    assert set(PERIOD_DESCRIPTIONS_ZH.keys()) == set(PERIOD_DESCRIPTIONS.keys())


def test_period_description_returns_both_languages():
    period = "2022-06"
    en = _period_description(period, "en")
    zh = _period_description(period, "zh")
    assert en == PERIOD_DESCRIPTIONS["2022"]
    assert zh == PERIOD_DESCRIPTIONS_ZH["2022"]
    assert en != zh


def test_find_similar_periods_includes_description_zh(monkeypatch):
    import numpy as np
    import pandas as pd

    from src import history_match as hm

    dates = pd.date_range("2018-01-31", periods=40, freq="ME")
    cols = hm.REGIME_FEATURES
    matrix = pd.DataFrame(
        np.random.default_rng(0).standard_normal((len(dates), len(cols))),
        index=dates,
        columns=cols,
    )
    spy_dates = pd.date_range("2018-02-01", periods=500, freq="D")
    spy = pd.DataFrame({"date": spy_dates, "value": np.linspace(100, 150, len(spy_dates))})
    data = {"spy": spy}

    matches = hm.find_similar_periods(matrix, data, n_matches=1, exclude_recent_months=6, min_separation_months=1)
    assert matches
    m = matches[0]
    assert m.get("description")
    assert m.get("description_zh")
    assert m["description"] != m["description_zh"]
