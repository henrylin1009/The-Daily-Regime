"""
Per-country quantitative signal layer — pure computation, no LLM.

Outputs a structured dict per country with four dimensions:
  growth      → accelerating / stable / slowing
  inflation   → accelerating / stable / cooling
  policy      → tightening / on_hold / easing  (+gap vs market expectations)
  capital     → inflow / neutral / outflow

These signals feed synthesis.py as structured context for LLM interpretation.
Countries: US, Japan, Europe, China, Taiwan, EM
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_series(df: pd.DataFrame, col: str = "value") -> pd.Series:
    return df[col].dropna() if col in df.columns else pd.Series(dtype=float)


def _last(series: pd.DataFrame | None, col: str = "value") -> float | None:
    """Return most-recent non-null value from a single-column DataFrame."""
    if series is None or not isinstance(series, pd.DataFrame) or series.empty:
        return None
    s = _get_series(series, col)
    return float(s.iloc[-1]) if not s.empty else None


def _slope(series: pd.DataFrame | None, col: str = "value", window: int = 3) -> float | None:
    """Linear slope of last `window` values (normalised by mean abs value)."""
    if series is None or not isinstance(series, pd.DataFrame) or series.empty:
        return None
    s = _get_series(series, col)
    if len(s) < window:
        return None
    y = s.iloc[-window:].values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    scale = float(np.mean(np.abs(y))) or 1.0
    return slope / scale  # dimensionless relative slope


def _zscore(series: pd.DataFrame | None, col: str = "value", lookback: int = 252) -> float | None:
    if series is None or not isinstance(series, pd.DataFrame) or series.empty:
        return None
    s = _get_series(series, col)
    if len(s) < 20:
        return None
    hist = s.iloc[-lookback:].astype(float)
    mu, sigma = float(hist.mean()), float(hist.std())
    if sigma < 1e-9:
        return 0.0
    return float((hist.iloc[-1] - mu) / sigma)


def _dir(slope: float | None, hi: float = 0.05, lo: float = -0.05) -> str:
    """Convert normalised slope to direction label."""
    if slope is None:
        return "unclear"
    if slope > hi:
        return "accelerating"
    if slope < lo:
        return "slowing"
    return "stable"


def _policy_dir(rate_now: float | None, rate_prev: float | None) -> str:
    if rate_now is None or rate_prev is None:
        return "unclear"
    diff = rate_now - rate_prev
    if diff > 0.05:
        return "tightening"
    if diff < -0.05:
        return "easing"
    return "on_hold"


def _policy_dir_from_series(df: pd.DataFrame | None, lookback: int = 12) -> str:
    """Compare latest rate to ~12 months ago to capture easing/tightening cycles."""
    if df is None or not isinstance(df, pd.DataFrame) or "value" not in df.columns:
        return "unclear"
    s = df["value"].dropna()
    if len(s) < 4:
        return "unclear"
    now = float(s.iloc[-1])
    prev = float(s.iloc[-min(lookback, len(s)-1)])
    return _policy_dir(now, prev)


def _capital_dir(fx_1m: float | None, etf_z: float | None) -> str:
    """
    Combine FX momentum and ETF z-score.
    Convention: positive fx_1m = local currency STRENGTHENING vs USD.
    """
    score = 0
    if fx_1m is not None:
        if fx_1m > 0.5:
            score += 1
        elif fx_1m < -0.5:
            score -= 1
    if etf_z is not None:
        if etf_z > 0.5:
            score += 1
        elif etf_z < -0.5:
            score -= 1
    if score >= 1:
        return "inflow"
    if score <= -1:
        return "outflow"
    return "neutral"


def _gap_label(actual: float | None, market_implied: float | None) -> str | None:
    """Label how far actual policy rate is from what market expects (implied by futures)."""
    if actual is None or market_implied is None:
        return None
    diff = actual - market_implied
    if diff > 0.25:
        return "market_expects_cuts"
    if diff < -0.25:
        return "market_expects_hikes"
    return "aligned"


def _round(v: float | None, n: int = 2) -> float | None:
    return round(v, n) if v is not None else None


# ---------------------------------------------------------------------------
# Per-country builders
# ---------------------------------------------------------------------------

def _us_signals(data: dict[str, Any]) -> dict:
    gdp = data.get("gdp_growth")
    indpro = data.get("indpro_growth")
    cpi = data.get("cpi_yoy")
    pce = data.get("pce_yoy")
    fed = data.get("fed_funds_rate")
    zq = data.get("zq_futures")
    dxy = data.get("dxy")
    spy = data.get("spy")

    gdp_slope = _slope(gdp, window=4)
    indpro_slope = _slope(indpro, window=3)
    growth_slope = gdp_slope if gdp_slope is not None else indpro_slope

    cpi_slope = _slope(cpi, window=3)
    pce_slope = _slope(pce, window=3)
    inflation_slope = cpi_slope if cpi_slope is not None else pce_slope

    fed_now = _last(fed)
    fed_prev = None
    if fed is not None and isinstance(fed, pd.DataFrame) and "value" in fed.columns:
        s = fed["value"].dropna()
        fed_prev = float(s.iloc[-4]) if len(s) >= 4 else None

    # ZQ futures imply fed funds ~3 months forward
    zq_implied = _last(zq)
    zq_implied = (100.0 - zq_implied) if zq_implied is not None else None

    dxy_z = _zscore(dxy)
    spy_z = _zscore(spy)
    # DXY up = USD strength = capital inflow to US
    dxy_1m = None
    if dxy is not None and isinstance(dxy, pd.DataFrame) and "value" in dxy.columns:
        s = dxy["value"].dropna()
        if len(s) >= 22:
            dxy_1m = float((s.iloc[-1] - s.iloc[-22]) / s.iloc[-22] * 100)

    return {
        "growth": {
            "direction": _dir(growth_slope),
            "slope": _round(growth_slope),
            "gdp_yoy": _round(_last(gdp)),
            "indpro_yoy": _round(_last(indpro)),
        },
        "inflation": {
            "direction": _dir(inflation_slope, hi=0.04, lo=-0.04),
            "slope": _round(inflation_slope),
            "cpi_yoy": _round(_last(cpi)),
            "pce_yoy": _round(_last(pce)),
        },
        "policy": {
            "direction": _policy_dir(fed_now, fed_prev),
            "rate": _round(fed_now),
            "market_implied": _round(zq_implied),
            "gap": _gap_label(fed_now, zq_implied),
        },
        "capital": {
            "direction": _capital_dir(dxy_1m, spy_z),
            "dxy_z": _round(dxy_z),
            "equity_z": _round(spy_z),
        },
    }


def _japan_signals(data: dict[str, Any], flow_payload: dict | None) -> dict:
    boj = data.get("boj_policy_rate")
    boj_now = _last(boj)
    boj_prev = None
    if boj is not None and isinstance(boj, pd.DataFrame) and "value" in boj.columns:
        s = boj["value"].dropna()
        boj_prev = float(s.iloc[-12]) if len(s) >= 12 else (float(s.iloc[-4]) if len(s) >= 4 else None)

    fp = (flow_payload or {}).get("flow_payload") or {}
    fx = (fp.get("fx_1m_local_vs_usd") or {}).get("Japan")
    spread = (fp.get("bond_spreads_bp") or {}).get("Japan")

    # Proxy Japan growth/inflation from FX and spread (direct FRED data sparse)
    return {
        "growth": {
            "direction": "unclear",
            "note": "Limited direct data; proxied via export FX and BoJ statements",
        },
        "inflation": {
            "direction": "unclear",
            "note": "Japan CPI not in current collect scope",
        },
        "policy": {
            "direction": _policy_dir(boj_now, boj_prev),
            "rate": _round(boj_now),
            "market_implied": None,
            "gap": None,
        },
        "capital": {
            "direction": _capital_dir(fx, None),
            "fx_1m_vs_usd": _round(fx),
            "us_spread_bp": _round(spread),
        },
    }


def _europe_signals(data: dict[str, Any], flow_payload: dict | None) -> dict:
    ecb = data.get("ecb_deposit_rate")
    ecb_now = _last(ecb)
    ecb_prev = None
    if ecb is not None and isinstance(ecb, pd.DataFrame) and "value" in ecb.columns:
        s = ecb["value"].dropna()
        ecb_prev = float(s.iloc[-12]) if len(s) >= 12 else (float(s.iloc[-4]) if len(s) >= 4 else None)

    fp = (flow_payload or {}).get("flow_payload") or {}
    fx = (fp.get("fx_1m_local_vs_usd") or {}).get("Europe")
    spread = (fp.get("bond_spreads_bp") or {}).get("Europe")

    return {
        "growth": {
            "direction": "unclear",
            "note": "Europe GDP not in current collect scope",
        },
        "inflation": {
            "direction": "unclear",
            "note": "Europe CPI not in current collect scope",
        },
        "policy": {
            "direction": _policy_dir(ecb_now, ecb_prev),
            "rate": _round(ecb_now),
            "market_implied": None,
            "gap": None,
        },
        "capital": {
            "direction": _capital_dir(fx, None),
            "fx_1m_vs_usd": _round(fx),
            "us_spread_bp": _round(spread),
        },
    }


def _china_signals(data: dict[str, Any], flow_payload: dict | None) -> dict:
    fp = (flow_payload or {}).get("flow_payload") or {}
    fx = (fp.get("fx_1m_local_vs_usd") or {}).get("China")
    spread = (fp.get("bond_spreads_bp") or {}).get("China")

    # Copper as China demand proxy
    copper = data.get("copper")
    copper_z = _zscore(copper)
    copper_slope = _slope(copper, window=5)

    return {
        "growth": {
            "direction": _dir(copper_slope, hi=0.04, lo=-0.04),
            "proxy": "copper_momentum",
            "copper_z": _round(copper_z),
            "note": "Copper used as China demand proxy; official GDP not used",
        },
        "inflation": {
            "direction": "unclear",
            "note": "China CPI not in current collect scope",
        },
        "policy": {
            "direction": "unclear",
            "note": "PBOC policy rate not in current collect scope",
        },
        "capital": {
            "direction": _capital_dir(fx, copper_z),
            "fx_1m_vs_usd": _round(fx),
            "us_spread_bp": _round(spread),
        },
    }


def _taiwan_signals(data: dict[str, Any], flow_payload: dict | None) -> dict:
    fp = (flow_payload or {}).get("flow_payload") or {}
    fx = (fp.get("fx_1m_local_vs_usd") or {}).get("Taiwan")
    spread = (fp.get("bond_spreads_bp") or {}).get("Taiwan")

    # Taiwan foreign flows from TIC/flow system
    foreign_flow = fp.get("taiwan_foreign_flow_5d")
    flow_dir = "neutral"
    if foreign_flow is not None:
        try:
            v = float(foreign_flow)
            flow_dir = "inflow" if v > 0 else ("outflow" if v < 0 else "neutral")
        except (TypeError, ValueError):
            pass

    return {
        "growth": {
            "direction": "unclear",
            "note": "Taiwan GDP not in current collect scope",
        },
        "inflation": {
            "direction": "unclear",
            "note": "Taiwan CPI not in current collect scope",
        },
        "policy": {
            "direction": "unclear",
            "note": "CBC policy rate not in current collect scope",
        },
        "capital": {
            "direction": _capital_dir(fx, None) if flow_dir == "neutral" else flow_dir,
            "fx_1m_vs_usd": _round(fx),
            "us_spread_bp": _round(spread),
            "foreign_flow_5d_bn": _round(foreign_flow),
        },
    }


def _em_signals(data: dict[str, Any]) -> dict:
    emb = data.get("emb")
    eem = data.get("eem")
    copper = data.get("copper")

    emb_z = _zscore(emb)
    eem_z = _zscore(eem)
    copper_z = _zscore(copper)

    emb_1m = None
    if emb is not None and isinstance(emb, pd.DataFrame) and "value" in emb.columns:
        s = emb["value"].dropna()
        if len(s) >= 22:
            emb_1m = float((s.iloc[-1] - s.iloc[-22]) / s.iloc[-22] * 100)

    eem_1m = None
    if eem is not None and isinstance(eem, pd.DataFrame) and "value" in eem.columns:
        s = eem["value"].dropna()
        if len(s) >= 22:
            eem_1m = float((s.iloc[-1] - s.iloc[-22]) / s.iloc[-22] * 100)

    # Combine EM bond + equity + copper for overall EM capital flow direction
    score = 0
    for z in [emb_z, eem_z, copper_z]:
        if z is not None:
            if z > 0.3:
                score += 1
            elif z < -0.3:
                score -= 1

    capital_dir = "inflow" if score >= 2 else ("outflow" if score <= -2 else "neutral")

    return {
        "growth": {
            "direction": _dir(_slope(copper, window=5), hi=0.04, lo=-0.04),
            "proxy": "copper_momentum",
            "copper_z": _round(copper_z),
        },
        "inflation": {
            "direction": "unclear",
            "note": "EM aggregate CPI not in current collect scope",
        },
        "policy": {
            "direction": "unclear",
            "note": "EM aggregate policy not tracked",
        },
        "capital": {
            "direction": capital_dir,
            "emb_z": _round(emb_z),
            "eem_z": _round(eem_z),
            "copper_z": _round(copper_z),
            "emb_1m_pct": _round(emb_1m),
            "eem_1m_pct": _round(eem_1m),
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_country_signals(
    data: dict[str, Any],
    flow_payload: dict | None = None,
) -> dict[str, dict]:
    """
    Compute four-grid quantitative signals for all countries.
    Returns a dict keyed by country name.
    Pure computation — no LLM, no side effects.
    """
    return {
        "US": _us_signals(data),
        "Japan": _japan_signals(data, flow_payload),
        "Europe": _europe_signals(data, flow_payload),
        "China": _china_signals(data, flow_payload),
        "Taiwan": _taiwan_signals(data, flow_payload),
        "EM": _em_signals(data),
    }


def compute_vs_us_alignment(signals: dict[str, dict]) -> dict[str, dict]:
    """
    Compare each country's signals to the US baseline.
    Returns a per-country alignment summary for display.

    alignment values: "aligned" | "diverging" | "mixed" | "limited"
    """
    us = signals.get("US", {})
    us_policy = us.get("policy", {}).get("direction", "unclear")
    us_capital = us.get("capital", {}).get("direction", "unclear")
    us_growth = us.get("growth", {}).get("direction", "unclear")

    # Human-readable policy labels
    _policy_label = {
        "tightening": "升息",
        "on_hold": "按兵不動",
        "easing": "降息",
        "unclear": "不明",
    }
    _capital_label = {
        "inflow": "資金流入",
        "outflow": "資金流出",
        "neutral": "中性",
        "unclear": "不明",
    }

    result: dict[str, dict] = {}

    for country, sig in signals.items():
        if country == "US":
            continue

        divergences: list[str] = []
        alignments: list[str] = []

        c_policy = sig.get("policy", {}).get("direction", "unclear")
        c_capital = sig.get("capital", {}).get("direction", "unclear")
        c_growth = sig.get("growth", {}).get("direction", "unclear")

        # Policy alignment
        if c_policy != "unclear" and us_policy != "unclear":
            if c_policy != us_policy:
                divergences.append(
                    f"Policy: {_policy_label.get(c_policy, c_policy)} "
                    f"vs US {_policy_label.get(us_policy, us_policy)}"
                )
            else:
                alignments.append("Policy aligned")

        # Capital alignment
        if c_capital != "unclear" and us_capital != "unclear":
            if c_capital == "outflow" and us_capital == "inflow":
                divergences.append(f"Capital: {_capital_label.get(c_capital, c_capital)}")
            elif c_capital == us_capital:
                alignments.append("Capital aligned")

        # Growth alignment (only when both known)
        if c_growth != "unclear" and us_growth != "unclear":
            if c_growth != us_growth:
                divergences.append(f"Growth: {c_growth} vs US {us_growth}")

        # Determine overall status
        has_data = bool(divergences or alignments)
        if not has_data:
            alignment = "limited"
            label = "資料有限"
            label_en = "Limited data"
            color = "gray"
            key_divergence = None
        elif divergences and not alignments:
            alignment = "diverging"
            label = "與美國分歧"
            label_en = "Diverging from US"
            color = "orange"
            key_divergence = divergences[0]
        elif divergences and alignments:
            alignment = "mixed"
            label = "部分分歧"
            label_en = "Partially diverging"
            color = "yellow"
            key_divergence = divergences[0]
        else:
            alignment = "aligned"
            label = "與美國一致"
            label_en = "Aligned with US"
            color = "green"
            key_divergence = None

        result[country] = {
            "alignment": alignment,
            "label": label,
            "label_zh": label,
            "label_en": label_en,
            "color": color,
            "key_divergence": key_divergence,
            "divergences": divergences,
            "alignments": alignments,
        }

    return result
