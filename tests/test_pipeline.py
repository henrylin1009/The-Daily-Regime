"""pytest — lock key pipeline invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.collect import INDICATOR_FETCHERS, load_all_from_cache, validate_all_caches
from src.factor_attrib import compute_factor_attribution
from src.history_match import REGIME_FEATURES, build_feature_matrix, find_similar_periods
from src.regime_stats import compute_regime_stats


@pytest.fixture(scope="module")
def cached_data():
    data = load_all_from_cache()
    if len(data) < len(REGIME_FEATURES) + 1:
        pytest.skip("Cached data not present — run python -m src.collect first")
    return data


def test_validate_caches():
    raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    if not raw_dir.is_dir() or not any(raw_dir.glob("*.csv")):
        pytest.skip("No cached data — run python -m src.collect locally or use CI fixtures")
    errs = validate_all_caches()
    missing = [e for e in errs if "missing" in e]
    assert missing == [], f"Missing cache files: {missing}"


def test_feature_matrix_length(cached_data):
    matrix = build_feature_matrix(cached_data)
    assert len(matrix) >= 300, f"Expected 300+ months, got {len(matrix)}"


def test_feature_matrix_columns(cached_data):
    matrix = build_feature_matrix(cached_data)
    for feat in REGIME_FEATURES:
        assert feat in matrix.columns


def test_history_match_rules(cached_data):
    matrix = build_feature_matrix(cached_data)
    matches = find_similar_periods(matrix, cached_data)
    assert len(matches) == 3

    decades = {int(m["date"][:4]) // 10 * 10 for m in matches}
    assert len(decades) >= 2, "Matches should span at least two decades"

    dates = sorted(pd.Timestamp(m["date"]) for m in matches)
    for i in range(len(dates) - 1):
        gap = (dates[i + 1] - dates[i]).days
        assert gap >= 730 or len(dates) == 1, "Matches should be separated by ~24 months"


def test_factor_attribution_shape(cached_data):
    result = compute_factor_attribution(cached_data)
    assert result.get("available"), result.get("message", "unknown")
    assert len(result["panels"]) >= 3
    available = [p for p in result["panels"] if p.get("available")]
    assert len(available) >= 2
    by_id = {p["id"]: p for p in result["panels"]}
    if by_id.get("tradeable", {}).get("available"):
        assert len(by_id["tradeable"]["factors"]) == 6
        codes = {f["code"] for f in by_id["tradeable"]["factors"]}
        assert "tech" not in codes
    if by_id.get("sectors", {}).get("available"):
        assert len(by_id["sectors"]["factors"]) == 6
    if by_id.get("fama_french", {}).get("available"):
        assert len(by_id["fama_french"]["factors"]) >= 5
    if by_id.get("crr", {}).get("available"):
        assert len(by_id["crr"]["factors"]) >= 4


def test_regime_stats(cached_data):
    matrix = build_feature_matrix(cached_data)
    stats = compute_regime_stats(matrix)
    assert stats.get("available")
    assert stats["current_regime_months"] >= 1


def test_signals_json_exists():
    processed = Path(__file__).resolve().parent.parent / "data" / "processed"
    files = list(processed.glob("signals_*.json"))
    if not files:
        pytest.skip("No signals JSON — run python run.py first")
    latest = max(files, key=lambda p: p.stat().st_mtime)
    payload = json.loads(latest.read_text())
    assert "headline" in payload
    assert len(payload["headline"]) == 3
