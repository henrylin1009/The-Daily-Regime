"""Stage 2: Compute signals, percentiles, trends, and traffic lights."""

from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

from src.collect import load_all_from_cache
from src.config import PROCESSED_DIR

# Display names for the 19-indicator detail table
DETAIL_INDICATORS: list[tuple[str, str, bool]] = [
    # (key, display_name, higher_is_worse)
    ("cpi_yoy", "CPI YoY", True),
    ("core_cpi_yoy", "Core CPI YoY", True),
    ("pce_yoy", "PCE YoY", True),
    ("nfp_change", "NFP Monthly Change", False),
    ("unemployment_rate", "Unemployment Rate", True),
    ("jolts_openings", "JOLTS Job Openings", False),
    ("gdp_growth", "GDP Growth YoY", False),
    ("spread_2y10y", "2y10y Spread", False),
    ("credit_spread_hy", "Credit Spread (Baa, regime)", True),
    ("credit_spread_hy_oas", "HY OAS Spread", True),
    ("fed_funds_rate", "Fed Funds Rate", True),
    ("vix", "VIX", True),
    ("dxy", "DXY", True),
    ("gld", "Gold (GLD ETF)", False),
    ("uso", "Oil proxy (DBO ETF)", True),
    ("xlk", "Tech (XLK)", False),
    ("xlf", "Financials (XLF)", False),
    ("xlu", "Utilities (XLU)", False),
    ("xle", "Energy (XLE)", False),
    ("xlv", "Healthcare (XLV)", False),
    ("xli", "Industrials (XLI)", False),
    ("zq_futures", "FF Futures (ZQ=F)", False),
]

HEADLINE_KEYS = {
    "recession_risk": "spread_2y10y",
    "inflation": "cpi_yoy",
    "financial_stress": "vix",
}

# Trending price series: percentile on 12-month return, not absolute price level
RETURN_PERCENTILE_KEYS = frozenset({
    "gld", "xlk", "xlf", "xlu", "xle", "xlv", "xli", "dxy", "zq_futures",
})

INDICATOR_NOTES: dict[str, str] = {
    "credit_spread_hy": "Used for regime match (long history)",
    "credit_spread_hy_oas": "FRED HY OAS; data from 2023",
    "gld": "ETF share price, not spot gold",
    "uso": "Oil proxy: DBO ETF (key kept as `uso` for backward compatibility)",
}


def historical_percentile(series: pd.Series, current_value: float, key: str = "") -> float:
    """Returns 0-100 percentile of current_value within series (or return series)."""
    if key in RETURN_PERCENTILE_KEYS:
        full = series.dropna().sort_index()
        periods = 252 if len(full) > 300 else max(12, len(full) // 4)
        # Use log-returns for stability across long horizons.
        # 12m ≈ 252 trading days when we have enough history; otherwise scale with history length.
        logp = np.log(full.clip(lower=1e-12))
        ret_series = (logp.diff(periods) * 100).dropna()
        if ret_series.empty:
            return 50.0
        current_ret = float(ret_series.iloc[-1]) if len(full) > periods else 0.0
        return float(stats.percentileofscore(ret_series, current_ret))

    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float(stats.percentileofscore(clean, current_value))


def trend_direction(
    series: pd.Series,
    months: int = 3,
    higher_is_worse: bool = True,
) -> str:
    """Return improving, stable, or worsening based on 3-month change."""
    s = series.dropna().sort_index()
    if len(s) < 2:
        return "stable"

    if isinstance(s.index, pd.DatetimeIndex):
        end = s.index[-1]
        start = end - pd.DateOffset(months=months)
        past = s[s.index <= start]
        if past.empty:
            past_val = s.iloc[0]
        else:
            past_val = past.iloc[-1]
    else:
        lookback = min(len(s) - 1, months * 21)
        past_val = s.iloc[-1 - lookback]

    current = s.iloc[-1]
    delta = current - past_val
    if abs(delta) < 1e-9:
        return "stable"

    rising = delta > 0
    if higher_is_worse:
        return "worsening" if rising else "improving"
    return "improving" if rising else "worsening"


def _latest_value(df: pd.DataFrame) -> tuple[float, date]:
    row = df.sort_values("date").iloc[-1]
    return float(row["value"]), pd.Timestamp(row["date"]).date()


def _spread_bps(value: float) -> float:
    """FRED T10Y2Y is in %; convert to basis points."""
    return value * 100


def _traffic_light_recession(spread_bps: float) -> str:
    if spread_bps < -50:
        return "RED"
    if spread_bps < 0:
        return "YELLOW"
    return "GREEN"


def _traffic_light_inflation(cpi_yoy: float) -> str:
    if cpi_yoy > 4.0:
        return "RED"
    if cpi_yoy > 2.5:
        return "YELLOW"
    return "GREEN"


def _traffic_light_stress(vix: float) -> str:
    if vix > 30:
        return "RED"
    if vix > 20:
        return "YELLOW"
    return "GREEN"


def _build_indicator_row(
    key: str,
    display_name: str,
    df: pd.DataFrame,
    higher_is_worse: bool,
    signal: str | None = None,
    display_value: float | None = None,
    unit: str = "",
) -> dict:
    value, as_of = _latest_value(df)
    disp = display_value if display_value is not None else value
    series = df.set_index("date")["value"]
    pct = historical_percentile(series, value, key=key)
    trend = trend_direction(series, higher_is_worse=higher_is_worse)

    if key == "spread_2y10y":
        label = f"2y10y spread: {disp:.0f}bp ({pct:.0f}th pct, {trend})"
    elif key in RETURN_PERCENTILE_KEYS:
        label = f"{display_name}: {disp:.2f} ({pct:.0f}th pct 12m return, {trend})"
    elif unit == "%":
        label = f"{display_name}: {disp:.2f}% ({pct:.0f}th pct, {trend})"
    else:
        label = f"{display_name}: {disp:.2f} ({pct:.0f}th pct, {trend})"

    return {
        "key": key,
        "name": display_name,
        "value": disp,
        "raw_value": value,
        "as_of": str(as_of),
        "percentile": round(pct, 1),
        "trend": trend,
        "signal": signal,
        "label": label,
        "notes": INDICATOR_NOTES.get(key, ""),
    }


def compute_signals(data: dict[str, pd.DataFrame]) -> dict:
    """Compute headline traffic lights and 19-row detail table."""
    details: list[dict] = []
    for key, name, higher_is_worse in DETAIL_INDICATORS:
        if key not in data:
            continue
        df = data[key]
        display_value = None
        unit = ""
        if key == "spread_2y10y":
            raw, _ = _latest_value(df)
            display_value = _spread_bps(raw)
            unit = "bp"

        details.append(
            _build_indicator_row(key, name, df, higher_is_worse, signal=None, display_value=display_value, unit=unit)
        )

    by_key = {d["key"]: d for d in details}

    spread_bps = by_key["spread_2y10y"]["value"]
    cpi = by_key["cpi_yoy"]["value"]
    vix = by_key["vix"]["value"]

    headline = {
        "recession_risk": {
            **by_key["spread_2y10y"],
            "signal": _traffic_light_recession(spread_bps),
        },
        "inflation": {
            **by_key["cpi_yoy"],
            "signal": _traffic_light_inflation(cpi),
        },
        "financial_stress": {
            **by_key["vix"],
            "signal": _traffic_light_stress(vix),
        },
    }

    as_of_dates = [d["as_of"] for d in details]
    result = {
        "as_of": max(as_of_dates) if as_of_dates else str(date.today()),
        "headline": headline,
        "details": details,
    }

    out_path = PROCESSED_DIR / f"signals_{result['as_of']}.json"
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    data = load_all_from_cache()
    if not data:
        print("No cached data. Run: python -m src.collect", file=sys.stderr)
        sys.exit(1)

    signals = compute_signals(data)
    print(f"As of: {signals['as_of']}\n")
    print("HEADLINE TRAFFIC LIGHTS:")
    for name, info in signals["headline"].items():
        print(f"  [{info['signal']}] {name}: {info['label']}")

    print("\nSAMPLE DETAILS (first 5):")
    for row in signals["details"][:5]:
        print(f"  {row['name']}: value={row['value']}, pct={row['percentile']}, trend={row['trend']}")

    # Debug: print date ranges for key return-percentile ETFs (helps catch short history issues).
    print("\nRETURN-PERCENTILE INPUT RANGES:")
    for k in sorted(RETURN_PERCENTILE_KEYS):
        if k not in data or data[k].empty:
            continue
        s = data[k].copy()
        s["date"] = pd.to_datetime(s["date"])
        idx = s["date"].sort_values()
        print(f"  {k}: {idx.iloc[0].date()} -> {idx.iloc[-1].date()} (n={len(s)})")


if __name__ == "__main__":
    main()
