"""Landing scenario scoring — soft vs hard landing probability.

Independent of regime classification. Only meaningful in G− environments
(Stagflation / Deflationary Bust). G+ environments return an expansion
momentum block instead.

Five indicators, each voting +1 (soft) / 0 (neutral) / −1 (hard):
  HY credit spread momentum  30%
  Unemployment 3m change     25%
  NFP 3m average             20%
  Yield curve slope trend    15%
  JOLTS 3m change            10%
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_DIR

# ── weights ──────────────────────────────────────────────────────────────────
WEIGHTS: dict[str, float] = {
    "hy_spread":    0.30,
    "unemployment": 0.25,
    "nfp":          0.20,
    "yield_curve":  0.15,
    "jolts":        0.10,
}


def _read_csv_series(filename: str) -> pd.Series:
    """Read a two-column date/value CSV from RAW_DIR, return Series sorted by date."""
    path = RAW_DIR / filename
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["value"]).sort_values("date").set_index("date")
    return df["value"].astype(float)


# ── individual signal voters ─────────────────────────────────────────────────

def _vote_hy_spread() -> tuple[int, str]:
    """HY credit spread momentum: z-score of last 20 daily observations.
    Soft: z < +0.8 and recent direction is stable/tightening.
    Hard: z > +1.5 or rapidly widening.
    """
    s = _read_csv_series("credit_spread_hy.csv")
    if len(s) < 25:
        return 0, "HY spread: insufficient data"
    z = (s.iloc[-1] - s.iloc[-60:].mean()) / (s.iloc[-60:].std() + 1e-9)
    delta_20d = s.iloc[-1] - s.iloc[-20]  # positive = widening = bad
    if z < 0.8 and delta_20d <= 0.10:
        return 1, f"HY spread z={z:.2f}, 20d Δ={delta_20d:+.2f} — contained"
    if z > 1.5 or delta_20d > 0.30:
        return -1, f"HY spread z={z:.2f}, 20d Δ={delta_20d:+.2f} — widening"
    return 0, f"HY spread z={z:.2f}, 20d Δ={delta_20d:+.2f} — neutral"


def _vote_unemployment() -> tuple[int, str]:
    """Unemployment rate 3m change.
    Soft: Δ < +0.3pp   Hard: Δ > +0.5pp
    """
    s = _read_csv_series("unemployment_rate.csv")
    if len(s) < 4:
        return 0, "Unemployment: insufficient data"
    delta = float(s.iloc[-1] - s.iloc[-4])   # last vs 3 months ago
    latest = float(s.iloc[-1])
    if delta < 0.30:
        return 1, f"Unemployment {latest:.1f}%, 3m Δ={delta:+.2f}pp — stable"
    if delta > 0.50:
        return -1, f"Unemployment {latest:.1f}%, 3m Δ={delta:+.2f}pp — rising"
    return 0, f"Unemployment {latest:.1f}%, 3m Δ={delta:+.2f}pp — neutral"


def _vote_nfp() -> tuple[int, str]:
    """NFP 3-month average.
    Soft: avg > +75k   Hard: avg < 0
    """
    s = _read_csv_series("nfp_change.csv")
    if len(s) < 3:
        return 0, "NFP: insufficient data"
    avg3 = float(s.iloc[-3:].mean())
    if avg3 > 75:
        return 1, f"NFP 3m avg +{avg3:.0f}k — positive"
    if avg3 < 0:
        return -1, f"NFP 3m avg {avg3:.0f}k — negative"
    return 0, f"NFP 3m avg +{avg3:.0f}k — neutral"


def _vote_yield_curve() -> tuple[int, str]:
    """Yield curve 2y10y slope trend.
    Soft: slope positive OR improving from negative (going less inverted)
    Hard: slope becoming more negative over last 20 obs
    """
    s = _read_csv_series("spread_2y10y.csv")
    if len(s) < 25:
        return 0, "Yield curve: insufficient data"
    current = float(s.iloc[-1])
    prev_20 = float(s.iloc[-20])
    if current >= 0:
        return 1, f"Yield curve +{current:.2f}% — uninverted"
    if current > prev_20:  # still negative but improving
        return 1, f"Yield curve {current:.2f}% — uninverting (was {prev_20:.2f}%)"
    if current < prev_20:  # more inverted
        return -1, f"Yield curve {current:.2f}% — deepening inversion"
    return 0, f"Yield curve {current:.2f}% — flat"


def _vote_jolts() -> tuple[int, str]:
    """JOLTS openings 3m change (thousands).
    Soft: 3m Δ > −300k   Hard: 3m Δ < −500k
    """
    s = _read_csv_series("jolts_openings.csv")
    if len(s) < 4:
        return 0, "JOLTS: insufficient data"
    delta = float(s.iloc[-1] - s.iloc[-4])  # last vs 3 months ago (thousands)
    latest = float(s.iloc[-1])
    if delta > -300:
        return 1, f"JOLTS {latest:.0f}k, 3m Δ={delta:+.0f}k — gradual"
    if delta < -500:
        return -1, f"JOLTS {latest:.0f}k, 3m Δ={delta:+.0f}k — sharp decline"
    return 0, f"JOLTS {latest:.0f}k, 3m Δ={delta:+.0f}k — neutral"


# ── expansion momentum (for G+ regimes) ─────────────────────────────────────

def _expansion_momentum() -> dict:
    """Return a simple expansion strength block for Goldilocks/Overheat."""
    nfp_s = _read_csv_series("nfp_change.csv")
    unemp_s = _read_csv_series("unemployment_rate.csv")
    hy_s = _read_csv_series("credit_spread_hy.csv")

    signals: list[str] = []
    score_pts = 0

    if len(nfp_s) >= 3:
        avg3 = float(nfp_s.iloc[-3:].mean())
        if avg3 > 150:
            score_pts += 2
            signals.append(f"NFP +{avg3:.0f}k avg — strong")
        elif avg3 > 75:
            score_pts += 1
            signals.append(f"NFP +{avg3:.0f}k avg — solid")
        else:
            signals.append(f"NFP +{avg3:.0f}k avg — slowing")

    if len(unemp_s) >= 2:
        delta = float(unemp_s.iloc[-1] - unemp_s.iloc[-2])
        u = float(unemp_s.iloc[-1])
        if delta <= 0:
            score_pts += 1
            signals.append(f"Unemployment {u:.1f}% — stable/declining")
        else:
            signals.append(f"Unemployment {u:.1f}% — ticking up")

    if len(hy_s) >= 20:
        z = (hy_s.iloc[-1] - hy_s.iloc[-60:].mean()) / (hy_s.iloc[-60:].std() + 1e-9)
        if z < 0:
            score_pts += 1
            signals.append(f"HY spread z={z:.2f} — credit supportive")
        else:
            signals.append(f"HY spread z={z:.2f} — slight caution")

    strength = "Strong" if score_pts >= 3 else ("Moderate" if score_pts >= 1 else "Fading")
    strength_zh = {"Strong": "強勁", "Moderate": "穩健", "Fading": "放緩"}[strength]
    bar_pct = min(100, max(10, score_pts * 25))

    return {
        "mode": "expansion",
        "strength": strength,
        "strength_zh": strength_zh,
        "bar_pct": bar_pct,
        "signals": signals,
    }


# ── main entry point ─────────────────────────────────────────────────────────

def compute_landing_scenario(macro_regime_label: str) -> dict:
    """Compute landing scenario scoring.

    Returns a dict with mode='landing' or mode='expansion'.
    Always safe to call — returns a minimal dict on any error.
    """
    try:
        return _compute(macro_regime_label)
    except Exception as exc:
        return {"mode": "error", "error": str(exc)}


def _compute(macro_regime_label: str) -> dict:
    growth_positive = macro_regime_label in ("Goldilocks", "Overheat")
    if growth_positive:
        result = _expansion_momentum()
        return result

    # G− regime: compute soft vs hard landing score
    voters = [
        ("hy_spread",    _vote_hy_spread()),
        ("unemployment", _vote_unemployment()),
        ("nfp",          _vote_nfp()),
        ("yield_curve",  _vote_yield_curve()),
        ("jolts",        _vote_jolts()),
    ]

    weighted_score = 0.0  # range −1 to +1
    drivers_soft: list[str] = []
    drivers_hard: list[str] = []
    drivers_neutral: list[str] = []

    for key, (vote, desc) in voters:
        w = WEIGHTS[key]
        weighted_score += vote * w
        if vote == 1:
            drivers_soft.append(desc)
        elif vote == -1:
            drivers_hard.append(desc)
        else:
            drivers_neutral.append(desc)

    # Map −1..+1 → 0..1 soft probability
    soft_prob = (weighted_score + 1) / 2
    hard_prob = 1 - soft_prob

    # Label
    if soft_prob >= 0.65:
        label = "Soft Landing"
        label_zh = "軟著陸"
    elif hard_prob >= 0.65:
        label = "Hard Landing"
        label_zh = "硬著陸"
    else:
        label = "Uncertain"
        label_zh = "前景不明"

    # Confidence: how unanimous the votes are
    n_decisive = sum(1 for _, (v, _) in voters if v != 0)
    confidence = "High" if n_decisive >= 4 else ("Medium" if n_decisive >= 2 else "Low")
    confidence_zh = {"High": "高", "Medium": "中", "Low": "低"}[confidence]

    all_drivers = drivers_soft + drivers_hard + drivers_neutral

    return {
        "mode": "landing",
        "label": label,
        "label_zh": label_zh,
        "soft_pct": round(soft_prob * 100),
        "hard_pct": round(hard_prob * 100),
        "confidence": confidence,
        "confidence_zh": confidence_zh,
        "drivers": all_drivers[:5],
        "drivers_soft": drivers_soft,
        "drivers_hard": drivers_hard,
    }
