"""Macro quadrant label matrix (no network)."""

from src.macro_quant_engine import compute_macro_regime_label


def test_regime_matrix():
    assert compute_macro_regime_label(True, False) == "Goldilocks"
    assert compute_macro_regime_label(True, True) == "Overheat"
    assert compute_macro_regime_label(False, True) == "Stagflation"
    assert compute_macro_regime_label(False, False) == "Deflationary Bust"
