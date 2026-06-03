"""Unit tests for macro quant momentum labels (no network)."""

import numpy as np
import pandas as pd

from src.macro_quant_engine import MOMENTUM_STREAK_MIN, get_momentum_profile


def test_improving_when_z_rises_three_days_and_good_when_up():
    z = pd.Series([0.0, 0.1, 0.2, 0.35, 0.5])
    out = get_momentum_profile(z, good_when_z_rises=True, days=20)
    assert out["label"] == "Improving"
    assert out["streak_days"] >= MOMENTUM_STREAK_MIN


def test_deteriorating_when_z_rises_and_bad_when_up():
    z = pd.Series([0.0, 0.2, 0.4, 0.55, 0.7])
    out = get_momentum_profile(z, good_when_z_rises=False, days=20)
    assert out["label"] == "Deteriorating"


def test_stable_when_mixed_moves():
    z = pd.Series([0.0, 0.3, 0.1, 0.35, 0.05, 0.4])
    out = get_momentum_profile(z, good_when_z_rises=True, days=20)
    assert out["label"] == "Stable"
