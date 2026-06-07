"""Layer 1 & 2 macro quant engine: panel fetch, rolling Z-scores, regime tags, LLM JSON."""

from __future__ import annotations

import json
import sys
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

# Trading-day windows (align with flow_run 1m ≈ 21 sessions)
ROLLING_Z_WINDOW = 252
ROLLING_Z_MIN_PERIODS = 60
DELTA_LOOKBACK = 21
CFTC_Z_EXTREME = 2.0

# Daily yfinance panel (3y) — US 2Y primary: 2YY=F futures proxy
YF_SPECS: list[tuple[str, str]] = [
    ("MOVE", "^MOVE"),
    ("US_10Y", "^TNX"),
    ("US_2Y", "2YY=F"),
    ("DXY", "DX-Y.NYB"),
    ("SPY", "SPY"),
    ("VIX", "^VIX"),
    ("USDJPY", "USDJPY=X"),
    ("USDTWD", "TWD=X"),
    ("HYG", "HYG"),
    ("CN_10Y", "CN10YT=RR"),
    ("TW_10Y", "TW10YT=RR"),
    ("SKEW", "^SKEW"),
    ("VVIX", "^VVIX"),
    ("SHY", "SHY"),
    ("TLT", "TLT"),
    ("RSP", "RSP"),
    ("HG", "HG=F"),
    ("GC", "GC=F"),
    ("TIP", "TIP"),
    ("IEF", "IEF"),
    ("Oil", "CL=F"),
    ("ZQ", "ZQ=F"),       # Fed funds futures — market Fed-path pricing
    ("OVX", "^OVX"),      # Oil volatility — geopolitical/supply shock pricing
    ("GVZ", "^GVZ"),      # Gold volatility — safe-haven demand pricing
    ("XLE", "XLE"),       # Energy sector — oil/geopolitical transmission
    ("XLF", "XLF"),       # Financials — credit/rate transmission
    ("XLK", "XLK"),       # Tech — risk-on / growth
    ("XLU", "XLU"),       # Utilities — defensive / rate-sensitive
    ("XLI", "XLI"),       # Industrials — growth cycle
]

CPI_FRED = "CPILFESL"
CPI_STALE_DAYS = 60
SKEW_HEDGING_THRESHOLD = 130.0
CROWDED_SPY_Z_THRESHOLD = 1.5
DIVERGENCE_BASE_CAP = 60
PENALTY_HEDGING = 20
PENALTY_CROWDED = 15
PENALTY_VOL_SPIKE = 10   # OVX or GVZ momentum Deteriorating while risk-on
PENALTY_FED_SHOCK = 10   # ZQ momentum Deteriorating (market pricing more hikes)
PENALTY_DEFENSIVE_ROTATION = 8  # sector rotation flips defensive while risk-on

# Layer 2 retail-alpha risk series (display key, column, good_when_z_rises)
LAYER2_RISK_SPECS: list[tuple[str, str, bool]] = [
    ("SKEW", "SKEW_Z", False),
    ("VVIX", "VVIX_Z", False),
    ("SHY_TLT", "SHY_TLT_Ratio_Z", True),
    ("RSP_SPY", "RSP_SPY_Ratio_Z", True),
]

MOMENTUM_DAYS = 20
MOMENTUM_STREAK_MIN = 3
MOMENTUM_DZ_EPS = 1e-6

# (display_key, column in enriched df, good_when_z_rises, column is already *_Z)
MOMENTUM_SPECS: list[tuple[str, str, bool, bool]] = [
    ("MOVE", "MOVE_Z", False, True),
    ("VIX", "VIX_Z", False, True),
    ("DXY", "DXY_Z", False, True),
    ("JPY", "USDJPY_Z", False, True),
    ("TWD", "USDTWD_Z", False, True),
    ("SPY", "SPY_Z", True, True),
    ("HYG", "HYG_Z", True, True),
    # Batch A additions
    ("Gold", "GC_Z", True, True),           # safe-haven / geopolitical fear
    ("ZQ", "ZQ_Z", True, True),             # Fed-path pricing (ZQ rises = mkt prices fewer hikes)
    ("Gold_Oil_Ratio", "Gold_Oil_Ratio_Z", True, True),  # geopolitical vs supply driver
]

TW_10Y_FALLBACK_TICKERS = ["TWD10Y=RR", "TAIEX10Y"]
TW_10Y_FRED_FALLBACKS = ["IRLTLT01TWM156N", "IRLTLT01TWN156N"]
CN_10Y_FALLBACK_TICKERS = ["CNGGB10Y=R"]
CN_10Y_FRED_FALLBACKS = ["DLTNTNO10Y"]
US_2Y_FALLBACK_TICKERS = ["^US2Y"]
US_2Y_FRED = "DGS2"
MOVE_FRED = "MOVEINDEX"
US_GROWTH_FRED_PRIMARY = "UMCSENT"
US_GROWTH_FRED_FALLBACK = "INDPRO"
STATIC_PE = 25.0

REGIME_MATCH_FEATURES = ["Spread_2y10y_Z", "VIX_Z", "MOVE_Z"]
REGIME_MATCH_EXCLUDE_TAIL = 30
REGIME_MATCH_KEY_FEATURE_DIFF = (
    "Calculated via Euclidean distance on Curve, VIX, and MOVE Z-scores"
)

# Rolling 252d Z-score inputs (ERP_Z computed separately from ERP_Proxy)
Z_SCORE_COLUMNS = [
    "US_2Y",
    "US_10Y",
    "MOVE",
    "VIX",
    "DXY",
    "US_Growth",
    "Spread_TW_US",
    "Spread_CN_US",
    "JPY_1M_Return",
]


def _fred_series(series_id: str) -> pd.Series:
    # Official FRED JSON API first (reliable from CI IPs), CSV as fallback.
    import os
    api_key = os.getenv("FRED_API_KEY") or os.getenv("fred_api_key")
    if api_key:
        import time
        import requests
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
        )
        for attempt in range(3):
            resp = requests.get(url, timeout=60)
            if resp.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            obs = resp.json().get("observations", [])
            if obs:
                df2 = pd.DataFrame(obs)[["date", "value"]]
                df2["value"] = pd.to_numeric(df2["value"], errors="coerce")
                out = pd.Series(df2["value"].values, index=pd.to_datetime(df2["date"])).dropna()
                out.index = out.index.tz_localize(None) if hasattr(out.index, "tz") and out.index.tz is not None else out.index
                return out.sort_index()
            break
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    if date_col not in df.columns or series_id not in df.columns:
        raise RuntimeError(f"Unexpected FRED format for {series_id}")
    s = pd.to_numeric(df[series_id], errors="coerce")
    out = pd.Series(s.values, index=pd.to_datetime(df[date_col])).dropna()
    out.index = out.index.tz_localize(None) if hasattr(out.index, "tz") and out.index.tz is not None else out.index
    return out.sort_index()


def _yf_close_series(ticker: str, period: str = "3y") -> pd.Series:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No yfinance data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = pd.Series(close, dtype=float).dropna()
    s.index = pd.to_datetime(s.index)
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _try_yf_column(name: str, ticker: str, period: str, frames: dict[str, pd.Series]) -> bool:
    try:
        frames[name] = _yf_close_series(ticker, period=period)
        print(f"  macro panel: {name} <- yfinance:{ticker}", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"  macro panel: skip yfinance:{ticker} ({exc})", file=sys.stderr)
        return False


def _try_fred_column(name: str, series_id: str, frames: dict[str, pd.Series]) -> bool:
    try:
        frames[name] = _fred_series(series_id)
        print(f"  macro panel: {name} <- fred:{series_id}", file=sys.stderr)
        return True
    except Exception as exc:
        print(f"  macro panel: skip fred:{series_id} ({exc})", file=sys.stderr)
        return False


def _first_available_series(
    name: str,
    yf_tickers: list[str],
    fred_ids: list[str],
    period: str,
    frames: dict[str, pd.Series],
) -> None:
    if name in frames and not frames[name].dropna().empty:
        return
    for ticker in yf_tickers:
        if _try_yf_column(name, ticker, period, frames):
            return
    for series_id in fred_ids:
        if _try_fred_column(name, series_id, frames):
            return
    if name not in frames:
        frames[name] = pd.Series(dtype=float)
        print(f"  macro panel: {name} unavailable (all sources failed)", file=sys.stderr)


def _fetch_us_growth(frames: dict[str, pd.Series]) -> None:
    """US growth proxy: Michigan Sentiment, else Industrial Production (monthly FRED)."""
    if "US_Growth" in frames and not frames["US_Growth"].dropna().empty:
        return
    if _try_fred_column("US_Growth", US_GROWTH_FRED_PRIMARY, frames):
        if not frames["US_Growth"].dropna().empty:
            return
    if _try_fred_column("US_Growth", US_GROWTH_FRED_FALLBACK, frames):
        if not frames["US_Growth"].dropna().empty:
            return
    frames["US_Growth"] = pd.Series(dtype=float)
    print("  macro panel: US_Growth unavailable (UMCSENT and INDPRO failed)", file=sys.stderr)


def _attach_cftc_jpy_z(df: pd.DataFrame) -> pd.DataFrame:
    """Weekly CFTC JPY net positioning Z-score, forward-filled onto the daily index."""
    out = df.copy()
    out["CFTC_Z_Score"] = np.nan
    try:
        from global_regime import fetch_module_b

        cot = fetch_module_b(force_refresh=False)
        jpy = cot.get("contracts", {}).get("JPY", {})
        z_val = jpy.get("zscore_52w")
        if z_val is not None and not (isinstance(z_val, float) and np.isnan(z_val)):
            out["CFTC_Z_Score"] = float(z_val)
            print(f"  macro panel: CFTC JPY Z-score = {z_val:+.2f} (weekly, broadcast to daily)", file=sys.stderr)
    except Exception as exc:
        print(f"  macro panel: CFTC JPY Z-score unavailable ({exc})", file=sys.stderr)
    return out


def fetch_macro_panel(period: str = "max") -> pd.DataFrame:
    """
    Build an aligned daily macro panel (full history) for rolling quant logic.
    Missing series are left as NaN; the pipeline does not raise on fetch failure.
    """
    frames: dict[str, pd.Series] = {}

    for name, ticker in YF_SPECS:
        if name == "TW_10Y":
            _first_available_series(name, [ticker] + TW_10Y_FALLBACK_TICKERS, TW_10Y_FRED_FALLBACKS, period, frames)
        elif name == "CN_10Y":
            _first_available_series(name, [ticker] + CN_10Y_FALLBACK_TICKERS, CN_10Y_FRED_FALLBACKS, period, frames)
        elif name == "MOVE":
            if not _try_yf_column(name, ticker, period, frames):
                _try_fred_column(name, MOVE_FRED, frames)
        elif name == "US_2Y":
            if not _try_yf_column(name, ticker, period, frames):
                _first_available_series(name, US_2Y_FALLBACK_TICKERS, [US_2Y_FRED], period, frames)
        else:
            _try_yf_column(name, ticker, period, frames)

    _fetch_us_growth(frames)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1, join="outer")
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Holiday / timezone misalignment across US, CN, TW
    df = df.ffill()

    if "USDJPY" in df.columns:
        jpy = df["USDJPY"]
        prev = jpy.shift(DELTA_LOOKBACK)
        df["JPY_1M_Return"] = np.where(prev != 0, (jpy / prev - 1.0) * 100.0, np.nan)

    if "TW_10Y" in df.columns and "US_10Y" in df.columns:
        df["Spread_TW_US"] = df["TW_10Y"] - df["US_10Y"]
    if "CN_10Y" in df.columns and "US_10Y" in df.columns:
        df["Spread_CN_US"] = df["CN_10Y"] - df["US_10Y"]

    if "SHY" in df.columns and "TLT" in df.columns:
        denom = df["TLT"].replace(0, np.nan)
        df["SHY_TLT_Ratio"] = df["SHY"] / denom
    if "RSP" in df.columns and "SPY" in df.columns:
        denom = df["SPY"].replace(0, np.nan)
        df["RSP_SPY_Ratio"] = df["RSP"] / denom
    if "HG" in df.columns and "GC" in df.columns:
        denom = df["GC"].replace(0, np.nan)
        df["Copper_Gold_Ratio"] = df["HG"] / denom
    if "GC" in df.columns and "Oil" in df.columns:
        denom = df["Oil"].replace(0, np.nan)
        df["Gold_Oil_Ratio"] = df["GC"] / denom  # rises = geopolitical fear > supply shock
    if "TIP" in df.columns and "IEF" in df.columns:
        denom = df["IEF"].replace(0, np.nan)
        df["TIP_IEF_Ratio"] = df["TIP"] / denom

    df = _attach_cftc_jpy_z(df)
    return df


def _max_directional_streak(dz: pd.Series, positive: bool) -> int:
    """Longest trailing run of same-sign daily changes in Z."""
    if dz.empty:
        return 0
    signs = dz > MOMENTUM_DZ_EPS if positive else dz < -MOMENTUM_DZ_EPS
    streak = 0
    best = 0
    for val in signs.iloc[::-1]:
        if bool(val):
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def get_momentum_profile(
    z_series: pd.Series,
    *,
    good_when_z_rises: bool,
    days: int = MOMENTUM_DAYS,
) -> dict[str, Any]:
    """
  20-day Z-score momentum: Improving / Deteriorating / Stable from consecutive daily Z moves.
    """
    z = pd.to_numeric(z_series, errors="coerce").dropna()
    empty: dict[str, Any] = {
        "label": "Unknown",
        "z_latest": None,
        "z_change_20d": None,
        "streak_days": 0,
        "direction": "flat",
    }
    if len(z) < MOMENTUM_STREAK_MIN + 1:
        return empty

    window = z.tail(days + 1)
    dz = window.diff().dropna()
    if dz.empty:
        return empty

    up_streak = _max_directional_streak(dz, positive=True)
    down_streak = _max_directional_streak(dz, positive=False)
    z_latest = float(window.iloc[-1])
    z_change = float(window.iloc[-1] - window.iloc[0]) if len(window) > 1 else 0.0

    label = "Stable"
    direction = "flat"
    streak_days = 0

    if up_streak >= MOMENTUM_STREAK_MIN and up_streak >= down_streak:
        streak_days = up_streak
        direction = "up"
        label = "Improving" if good_when_z_rises else "Deteriorating"
    elif down_streak >= MOMENTUM_STREAK_MIN:
        streak_days = down_streak
        direction = "down"
        label = "Deteriorating" if good_when_z_rises else "Improving"

    return {
        "label": label,
        "z_latest": _safe_float(z_latest),
        "z_change_20d": _safe_float(z_change),
        "streak_days": int(streak_days),
        "direction": direction,
    }


def _momentum_is_accelerating(z_series: pd.Series, *, good_when_z_rises: bool = True) -> bool | None:
    """True=Accelerating, False=Decelerating, None=unavailable."""
    prof = get_momentum_profile(z_series, good_when_z_rises=good_when_z_rises)
    label = prof.get("label")
    if label == "Unknown":
        z = pd.to_numeric(z_series, errors="coerce").dropna()
        if len(z) < 2:
            return None
        chg = float(z.iloc[-1] - z.iloc[max(0, len(z) - MOMENTUM_DAYS - 1)])
        if chg == 0:
            return None
        rising = chg > 0
        return rising if good_when_z_rises else not rising
    if label == "Stable":
        return None
    return label == "Improving"


def _cpi_yoy_monthly_trend() -> dict[str, Any]:
    """
    Monthly CPI YoY MoM comparison — never use daily ffill slope on CPILFESL.
    Context-only for macro_regime_detail / LLM; does not vote in the quadrant.
    """
    empty: dict[str, Any] = {
        "latest_yoy": None,
        "prior_yoy": None,
        "trend": "Stale",
        "as_of_month": None,
    }
    try:
        level = _fred_series(CPI_FRED)
        monthly = level.resample("ME").last().dropna()
        if len(monthly) < 14:
            return empty
        yoy = (monthly / monthly.shift(12) - 1.0) * 100.0
        yoy = yoy.dropna()
        if len(yoy) < 2:
            return empty
        latest_ts = yoy.index[-1]
        as_of_month = pd.Timestamp(latest_ts).strftime("%Y-%m")
        age_days = (pd.Timestamp.now().normalize() - pd.Timestamp(latest_ts).normalize()).days
        latest_v = float(yoy.iloc[-1])
        prior_v = float(yoy.iloc[-2])
        if age_days > CPI_STALE_DAYS:
            return {
                "latest_yoy": _safe_float(latest_v, 2),
                "prior_yoy": _safe_float(prior_v, 2),
                "trend": "Stale",
                "as_of_month": as_of_month,
            }
        if latest_v > prior_v:
            trend = "Accelerating"
        elif latest_v < prior_v:
            trend = "Decelerating"
        else:
            trend = "Unchanged"
        return {
            "latest_yoy": _safe_float(latest_v, 2),
            "prior_yoy": _safe_float(prior_v, 2),
            "trend": trend,
            "as_of_month": as_of_month,
        }
    except Exception as exc:
        print(f"  macro panel: CPI YoY context unavailable ({exc})", file=sys.stderr)
        return empty


def compute_macro_regime_label(growth_accel: bool, inflation_accel: bool) -> str:
    if growth_accel and not inflation_accel:
        return "Goldilocks"
    if growth_accel and inflation_accel:
        return "Overheat"
    if not growth_accel and inflation_accel:
        return "Stagflation"
    return "Deflationary Bust"


def compute_macro_regime_detail(df: pd.DataFrame) -> dict[str, Any]:
    """Growth 2-of-3 daily votes; inflation 2-of-2 TIP/IEF + Oil (CPI context only)."""
    drivers: list[str] = []
    growth_votes: list[bool | None] = []
    if "US_Growth_Z" in df.columns:
        growth_votes.append(_momentum_is_accelerating(df["US_Growth_Z"], good_when_z_rises=True))
    if "SPY_Z" in df.columns:
        growth_votes.append(_momentum_is_accelerating(df["SPY_Z"], good_when_z_rises=True))
    if "Copper_Gold_Ratio_Z" in df.columns:
        growth_votes.append(_momentum_is_accelerating(df["Copper_Gold_Ratio_Z"], good_when_z_rises=True))
    elif "Copper_Gold_Ratio" in df.columns:
        growth_votes.append(
            _momentum_is_accelerating(rolling_zscore(df["Copper_Gold_Ratio"]), good_when_z_rises=True)
        )

    g_valid = [v for v in growth_votes if v is not None]
    growth_accel = sum(1 for v in g_valid if v) >= 2 if len(g_valid) >= 2 else (
        g_valid[0] if len(g_valid) == 1 else False
    )
    if "US_Growth_Z" in df.columns:
        drivers.append(f"US_Growth momentum {'up' if growth_votes[0] else 'down'}" if growth_votes[0] is not None else "US_Growth n/a")
    if len(growth_votes) > 1 and growth_votes[1] is not None:
        drivers.append(f"SPY Z momentum {'accelerating' if growth_votes[1] else 'decelerating'}")

    inflation_votes: list[bool | None] = []
    if "TIP_IEF_Ratio_Z" in df.columns:
        inflation_votes.append(_momentum_is_accelerating(df["TIP_IEF_Ratio_Z"], good_when_z_rises=True))
    elif "TIP_IEF_Ratio" in df.columns:
        inflation_votes.append(
            _momentum_is_accelerating(rolling_zscore(df["TIP_IEF_Ratio"]), good_when_z_rises=True)
        )
    if "Oil_Z" in df.columns:
        inflation_votes.append(_momentum_is_accelerating(df["Oil_Z"], good_when_z_rises=True))
    elif "Oil" in df.columns:
        inflation_votes.append(_momentum_is_accelerating(rolling_zscore(df["Oil"]), good_when_z_rises=True))

    # Hard CPI YoY momentum is the PRIMARY inflation vote (Investment-Clock style);
    # market proxies (TIP/IEF breakeven, Oil) are fallback only when CPI is stale.
    cpi_ctx = _cpi_yoy_monthly_trend()
    cpi_trend = cpi_ctx.get("trend")
    inflation_vote_sources: list[str] = []
    if cpi_trend == "Accelerating":
        inflation_accel = True
        inflation_vote_sources = ["CPI_YoY"]
        drivers.append(
            f"CPI YoY {cpi_ctx.get('prior_yoy')}% → {cpi_ctx.get('latest_yoy')}% (Accelerating) — hard-data vote"
        )
    elif cpi_trend == "Decelerating":
        inflation_accel = False
        inflation_vote_sources = ["CPI_YoY"]
        drivers.append(
            f"CPI YoY {cpi_ctx.get('prior_yoy')}% → {cpi_ctx.get('latest_yoy')}% (Decelerating) — hard-data vote"
        )
    else:
        # CPI stale/unavailable — fall back to daily market proxies
        i_valid = [v for v in inflation_votes if v is not None]
        if len(i_valid) == 0:
            inflation_accel = False
            drivers.append("inflation_vote_insufficient")
        elif len(i_valid) == 1:
            inflation_accel = bool(i_valid[0])
        else:
            inflation_accel = all(i_valid)
        inflation_vote_sources = ["TIP_IEF_Ratio", "Oil"]
        if inflation_votes and inflation_votes[0] is not None:
            drivers.append(f"TIP/IEF inflation expectation {'rising' if inflation_votes[0] else 'falling'} (CPI stale)")
        if len(inflation_votes) > 1 and inflation_votes[1] is not None:
            drivers.append(f"Oil momentum {'up' if inflation_votes[1] else 'down'} (CPI stale)")

    label = compute_macro_regime_label(growth_accel, inflation_accel)
    print(
        f"  macro regime: {label} (growth={'Accelerating' if growth_accel else 'Decelerating'}, "
        f"inflation={'Accelerating' if inflation_accel else 'Decelerating'})",
        file=sys.stderr,
    )
    return {
        "macro_regime_label": label,
        "growth": "Accelerating" if growth_accel else "Decelerating",
        "inflation": "Accelerating" if inflation_accel else "Decelerating",
        "inflation_vote_sources": inflation_vote_sources,
        "cpi_yoy_context": cpi_ctx,
        "drivers": drivers,
    }


def build_layer2_risk_profile(df: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    if df is None or df.empty:
        return profile
    row = _last_valid_row(df)
    for key, col, good_up in LAYER2_RISK_SPECS:
        if col in df.columns:
            profile[key] = get_momentum_profile(df[col], good_when_z_rises=good_up)
            if row is not None and col.replace("_Z", "") in df.columns:
                raw_col = col.replace("_Z", "")
                profile[key]["latest"] = _safe_float(row.get(raw_col))
        elif col.replace("_Z", "") in df.columns:
            z = rolling_zscore(df[col.replace("_Z", "")])
            profile[key] = get_momentum_profile(z, good_when_z_rises=good_up)
        else:
            profile[key] = {"label": "Unknown", "z_latest": None}
    if row is not None and "SKEW" in df.columns:
        profile.setdefault("SKEW", {})
        if isinstance(profile["SKEW"], dict):
            profile["SKEW"]["latest_level"] = _safe_float(row.get("SKEW"))
            # Momentum of SKEW itself: rising SKEW = market pricing more tail risk
            if "SKEW_Z" in df.columns:
                skew_mom = get_momentum_profile(df["SKEW_Z"], good_when_z_rises=False)
                profile["SKEW"]["momentum"] = skew_mom.get("label", "Unknown")
                profile["SKEW"]["momentum_streak"] = skew_mom.get("streak_days", 0)
    return profile


def compute_penalized_divergence(
    df: pd.DataFrame,
    momentum_profile: dict[str, Any],
) -> dict[str, Any]:
    """Base L1 vs L2 Z clash + hedging/crowdedness penalties (cap 100)."""
    row = _last_valid_row(df)
    empty: dict[str, Any] = {
        "divergence_score": 0,
        "base_score": 0,
        "penalty_reasons": [],
        "penalty_reason": "",
    }
    if row is None:
        return empty

    l1_cols = [c for c in ("US_Growth_Z", "ERP_Z", "Spread_2y10y_Z") if c in df.columns]
    l2_cols = [c for c in ("SPY_Z", "HYG_Z", "VIX_Z") if c in df.columns]
    gaps: list[float] = []
    for a in l1_cols:
        for b in l2_cols:
            av, bv = row.get(a), row.get(b)
            if av is not None and bv is not None and pd.notna(av) and pd.notna(bv):
                gaps.append(abs(float(av) - float(bv)))
    base = min(DIVERGENCE_BASE_CAP, int(np.mean(gaps) * 12)) if gaps else 30

    penalties: list[str] = []
    skew_level = _safe_float(row.get("SKEW")) if "SKEW" in df.columns else None
    spy = momentum_profile.get("SPY") or {}
    hyg = momentum_profile.get("HYG") or {}
    risk_on = spy.get("label") == "Improving" or hyg.get("label") == "Improving"
    # SKEW triggers on level >130 OR on accelerating momentum (market buying tail protection)
    skew_z_momentum = None
    if "SKEW_Z" in df.columns:
        skew_mom = get_momentum_profile(df["SKEW_Z"], good_when_z_rises=False)
        skew_z_momentum = skew_mom.get("label")
    skew_elevated = (skew_level is not None and skew_level > SKEW_HEDGING_THRESHOLD)
    skew_accelerating = (skew_z_momentum == "Deteriorating")  # rising SKEW_Z = bad (more tail priced)
    if (skew_elevated or skew_accelerating) and risk_on:
        base += PENALTY_HEDGING
        reason = "Hedging Cost Anomaly"
        if skew_accelerating and not skew_elevated:
            reason = "Hedging Cost Anomaly (SKEW momentum)"
        penalties.append(reason)

    spy_z = spy.get("z_latest")
    rsp_prof = momentum_profile.get("RSP_SPY") or {}
    if "RSP_SPY_Ratio_Z" in df.columns:
        rsp_prof = get_momentum_profile(df["RSP_SPY_Ratio_Z"], good_when_z_rises=True)
    elif rsp_prof.get("label") == "Unknown" and "RSP_SPY_Ratio_Z" in df.columns:
        rsp_prof = get_momentum_profile(df["RSP_SPY_Ratio_Z"], good_when_z_rises=True)
    try:
        spy_z_f = float(spy_z) if spy_z is not None else None
    except (TypeError, ValueError):
        spy_z_f = None
    if (
        spy_z_f is not None
        and spy_z_f > CROWDED_SPY_Z_THRESHOLD
        and rsp_prof.get("label") == "Deteriorating"
    ):
        base += PENALTY_CROWDED
        penalties.append("Extreme Crowdedness/Leverage")

    # OVX/GVZ momentum penalty: geopolitical vol spike while positioned risk-on
    if risk_on:
        for vol_key, z_col in (("OVX", "OVX_Z"), ("GVZ", "GVZ_Z")):
            if z_col in df.columns:
                vol_mom = get_momentum_profile(df[z_col], good_when_z_rises=False)
                if vol_mom.get("label") == "Deteriorating":
                    base += PENALTY_VOL_SPIKE
                    penalties.append(f"Vol Spike ({vol_key} momentum)")
                    break  # one penalty even if both fire

    # ZQ futures momentum penalty: market abruptly repricing Fed path (more hikes)
    # ZQ rises = fewer hikes priced (good); ZQ falls = more hikes priced (bad for risk)
    if "ZQ_Z" in df.columns:
        zq_mom = get_momentum_profile(df["ZQ_Z"], good_when_z_rises=True)
        if zq_mom.get("label") == "Deteriorating" and risk_on:
            base += PENALTY_FED_SHOCK
            penalties.append("Fed Path Shock (ZQ momentum)")

    # Sector rotation penalty: capital rotating defensive while positioned risk-on
    if risk_on:
        xle_col, xlu_col = "XLE_Z", "XLU_Z"
        if xle_col in df.columns and xlu_col in df.columns:
            xle_z = _safe_float(row.get(xle_col))
            xlu_z = _safe_float(row.get(xlu_col))
            if xle_z is not None and xlu_z is not None:
                # Defensive rotation: Utilities outperforming Energy significantly
                if xlu_z > xle_z + 0.8:
                    base += PENALTY_DEFENSIVE_ROTATION
                    penalties.append("Defensive Sector Rotation (XLU > XLE)")

    score = max(0, min(100, int(base)))
    reason = "; ".join(penalties)
    if penalties:
        print(f"  divergence: {score}% penalties={reason}", file=sys.stderr)
    return {
        "divergence_score": score,
        "base_score": min(DIVERGENCE_BASE_CAP, int(np.mean(gaps) * 12)) if gaps else 30,
        "penalty_reasons": penalties,
        "penalty_reason": reason,
    }


def build_momentum_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Momentum labels for critical macro indicators (uses enriched Z columns)."""
    profile: dict[str, Any] = {}
    if df is None or df.empty:
        return profile

    for key, col, good_up, is_z_col in MOMENTUM_SPECS:
        series: pd.Series | None = None
        if col in df.columns:
            series = df[col]
        elif not is_z_col:
            raw = col.replace("_Z", "")
            if raw in df.columns:
                series = rolling_zscore(df[raw])
        if series is None or series.dropna().empty:
            profile[key] = {
                "label": "Unknown",
                "z_latest": None,
                "z_change_20d": None,
                "streak_days": 0,
                "direction": "unavailable",
            }
            continue
        profile[key] = get_momentum_profile(series, good_when_z_rises=good_up)

    return profile


def rolling_zscore(
    series: pd.Series,
    window: int = ROLLING_Z_WINDOW,
    min_periods: int = ROLLING_Z_MIN_PERIODS,
) -> pd.Series:
    """
    Rolling Z-score: (X_t - mean_252) / std_252.
    Zero rolling std is replaced with NaN to avoid divide-by-zero blowups.
    """
    s = pd.to_numeric(series, errors="coerce")
    mu = s.rolling(window, min_periods=min_periods).mean()
    sigma = s.rolling(window, min_periods=min_periods).std(ddof=0).replace(0, np.nan)
    return (s - mu) / sigma


def _classify_yield_curve_vectorized(
    delta_spread: pd.Series,
    delta_2y: pd.Series,
) -> pd.Series:
    """L1: Bull/Bear Steepener/Flattener from 21d changes in 2y10y spread and 2Y."""
    ds = delta_spread
    d2 = delta_2y
    valid = ds.notna() & d2.notna() & ((ds != 0) | (d2 != 0))
    tags = pd.Series("Unknown", index=ds.index, dtype=object)
    tags.loc[valid & (ds > 0) & (d2 < 0)] = "Bull Steepener"
    tags.loc[valid & (ds > 0) & (d2 > 0)] = "Bear Steepener"
    tags.loc[valid & (ds < 0) & (d2 < 0)] = "Bull Flattener"
    tags.loc[valid & (ds < 0) & (d2 > 0)] = "Bear Flattener"
    return tags


def _classify_squeeze_vectorized(cftc_z: pd.Series, ret_1m: pd.Series) -> pd.Series:
    """L2: CFTC positioning extreme + 1M price reversal (JPY mock)."""
    z = cftc_z
    r = ret_1m
    tags = pd.Series("Neutral/Trend", index=z.index, dtype=object)
    valid = z.notna() & r.notna()
    tags.loc[valid & (z < -CFTC_Z_EXTREME) & (r > 0)] = "Short Squeeze Risk"
    tags.loc[valid & (z > CFTC_Z_EXTREME) & (r < 0)] = "Long Liquidation Risk"
    missing = z.isna() | r.isna()
    tags.loc[missing] = "Unknown"
    return tags


def compute_euclidean_regime_match(df: pd.DataFrame) -> dict[str, Any]:
    """
    Find the historical trading day most similar to today on global macro Z-scores.
    Uses Spread_2y10y_Z (curve), VIX_Z, and MOVE_Z; excludes the last 30 sessions.
    """
    empty: dict[str, Any] = {
        "matched_date": None,
        "distance": None,
        "feature_diffs": None,
    }
    missing = [c for c in REGIME_MATCH_FEATURES if c not in df.columns]
    if missing:
        print(f"  regime match: skip (missing columns {missing})", file=sys.stderr)
        return empty

    matrix = df[REGIME_MATCH_FEATURES].dropna(how="any")
    min_rows = REGIME_MATCH_EXCLUDE_TAIL + 2
    if len(matrix) < min_rows:
        print(
            f"  regime match: skip (only {len(matrix)} rows with full features, need {min_rows})",
            file=sys.stderr,
        )
        return empty

    target = matrix.iloc[-1].to_numpy(dtype=float)
    hist = matrix.iloc[:-REGIME_MATCH_EXCLUDE_TAIL]
    if hist.empty:
        return empty

    dists = np.linalg.norm(hist.to_numpy(dtype=float) - target, axis=1)
    best_idx = int(np.argmin(dists))
    matched_ts = hist.index[best_idx]
    matched_date = pd.Timestamp(matched_ts).date().isoformat()
    distance = float(dists[best_idx])

    best_row = hist.iloc[best_idx]
    feature_diffs = {
        feat: round(float(best_row[feat] - target[i]), 4)
        for i, feat in enumerate(REGIME_MATCH_FEATURES)
    }

    print(
        f"  regime match: {matched_date} distance={distance:.4f} "
        f"(excluded last {REGIME_MATCH_EXCLUDE_TAIL} sessions)",
        file=sys.stderr,
    )
    return {
        "matched_date": matched_date,
        "distance": round(distance, 4),
        "feature_diffs": feature_diffs,
    }


def _compute_regime_duration(df: pd.DataFrame) -> dict[str, Any]:
    """
    Count how many consecutive trading days the current regime label has held,
    and summarize historical duration stats for that regime.
    Requires a 'macro_regime_label' column computed per-row (uses compute_macro_regime_detail logic).
    Falls back to using the enriched df.attrs if per-row labels are not available.
    """
    empty: dict[str, Any] = {
        "current_regime": None,
        "duration_days": None,
        "avg_duration_days": None,
        "max_duration_days": None,
        "pct_of_history": None,
    }
    try:
        # Build per-row regime labels using growth/inflation Z-scores
        required = {"US_Growth_Z", "SPY_Z", "TIP_IEF_Ratio_Z"}
        available = required.intersection(set(df.columns))
        if len(available) < 2:
            return empty

        labels: list[str] = []
        dates: list[Any] = []
        for ts, row in df.iterrows():
            try:
                g_votes = []
                if "US_Growth_Z" in df.columns:
                    v = row.get("US_Growth_Z")
                    if pd.notna(v):
                        g_votes.append(float(v) > 0)
                if "SPY_Z" in df.columns:
                    v = row.get("SPY_Z")
                    if pd.notna(v):
                        g_votes.append(float(v) > 0)
                if "Copper_Gold_Ratio_Z" in df.columns:
                    v = row.get("Copper_Gold_Ratio_Z")
                    if pd.notna(v):
                        g_votes.append(float(v) > 0)

                i_votes = []
                if "TIP_IEF_Ratio_Z" in df.columns:
                    v = row.get("TIP_IEF_Ratio_Z")
                    if pd.notna(v):
                        i_votes.append(float(v) > 0)

                if len(g_votes) < 1 or len(i_votes) < 1:
                    continue

                g_valid = g_votes
                growth_accel = sum(g_valid) >= 2 if len(g_valid) >= 2 else g_valid[0]
                inflation_accel = i_votes[0]
                label = compute_macro_regime_label(growth_accel, inflation_accel)
                labels.append(label)
                dates.append(ts)
            except Exception:
                continue

        if len(labels) < 30:
            return empty

        label_series = pd.Series(labels, index=dates)
        current_label = label_series.iloc[-1]

        # Count consecutive days of current regime from the end
        duration = 0
        for lbl in reversed(labels):
            if lbl == current_label:
                duration += 1
            else:
                break

        # Compute historical episode stats for this regime
        episodes: list[int] = []
        count = 0
        prev = None
        for lbl in labels:
            if lbl == current_label:
                count += 1
            else:
                if prev == current_label and count > 0:
                    episodes.append(count)
                count = 0
            prev = lbl
        if count > 0:
            episodes.append(count)  # include current episode

        avg_dur = round(float(np.mean(episodes)), 1) if episodes else None
        max_dur = int(max(episodes)) if episodes else None
        total_days = len(labels)
        regime_days = sum(1 for l in labels if l == current_label)
        pct = round(regime_days / total_days * 100, 1) if total_days > 0 else None

        print(
            f"  regime duration: {current_label} for {duration} days "
            f"(avg={avg_dur}, max={max_dur}, {pct}% of history)",
            file=sys.stderr,
        )
        return {
            "current_regime": current_label,
            "duration_days": duration,
            "avg_duration_days": avg_dur,
            "max_duration_days": max_dur,
            "pct_of_history": pct,
        }
    except Exception as exc:
        print(f"  regime duration: error — {exc}", file=sys.stderr)
        return empty


def _safe_float(value: Any, decimals: int = 2) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return round(f, decimals)


class MacroQuantEngine:
    """Enrich a macro panel with rolling Z-scores and L1/L2 regime tags."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.regime_match: dict[str, Any] = {}
        self.momentum_profile: dict[str, Any] = {}
        self.macro_regime: dict[str, Any] = {}
        self.risk_analytics: dict[str, Any] = {}

    def enrich(self) -> pd.DataFrame:
        out = self.df.copy()
        if out.empty:
            return out

        z_cols = [c for c in Z_SCORE_COLUMNS if c in out.columns]
        for col in z_cols:
            out[f"{col}_Z"] = rolling_zscore(out[col])

        for raw in ("SPY", "HYG", "USDJPY", "USDTWD", "SKEW", "VVIX", "Oil", "GC", "ZQ", "OVX", "GVZ",
                    "XLE", "XLF", "XLK", "XLU", "XLI"):
            if raw in out.columns and f"{raw}_Z" not in out.columns:
                out[f"{raw}_Z"] = rolling_zscore(out[raw])

        for ratio in ("SHY_TLT_Ratio", "RSP_SPY_Ratio", "Copper_Gold_Ratio", "TIP_IEF_Ratio", "Gold_Oil_Ratio"):
            if ratio in out.columns and f"{ratio}_Z" not in out.columns:
                out[f"{ratio}_Z"] = rolling_zscore(out[ratio])

        if "US_2Y" in out.columns and "US_10Y" in out.columns:
            out["Spread_2y10y"] = out["US_10Y"] - out["US_2Y"]
            out["Spread_2y10y_Z"] = rolling_zscore(out["Spread_2y10y"])
            out["Delta_Spread"] = out["Spread_2y10y"].diff(DELTA_LOOKBACK)
            out["Delta_2Y"] = out["US_2Y"].diff(DELTA_LOOKBACK)
            out["L1_Yield_Curve_Tag"] = _classify_yield_curve_vectorized(out["Delta_Spread"], out["Delta_2Y"])
        else:
            out["L1_Yield_Curve_Tag"] = "Unknown"

        ret_col = "JPY_1M_Return" if "JPY_1M_Return" in out.columns else None
        if ret_col and "CFTC_Z_Score" in out.columns:
            out["L2_Squeeze_Tag"] = _classify_squeeze_vectorized(out["CFTC_Z_Score"], out[ret_col])
        else:
            out["L2_Squeeze_Tag"] = "Unknown"

        # ERP structural proxy: static P/E 25 earnings yield minus US 10Y (decimal)
        if "US_10Y" in out.columns:
            out["ERP_Proxy"] = (1.0 / STATIC_PE) - (out["US_10Y"] / 100.0)
            out["ERP_Z"] = rolling_zscore(out["ERP_Proxy"])

        match = compute_euclidean_regime_match(out)
        self.regime_match = match
        out.attrs["regime_match"] = match
        momentum = build_momentum_profile(out)
        out.attrs["momentum_profile"] = momentum
        self.momentum_profile = momentum

        regime_detail = compute_macro_regime_detail(out)
        out.attrs["macro_regime"] = regime_detail
        self.macro_regime = regime_detail

        layer2_risk = build_layer2_risk_profile(out)
        out.attrs["layer2_risk"] = layer2_risk

        risk_analytics = compute_penalized_divergence(out, momentum)
        risk_analytics["layer2_risk"] = layer2_risk
        out.attrs["risk_analytics"] = risk_analytics
        self.risk_analytics = risk_analytics

        self.df = out
        return out

    def build_momentum_profile(self) -> dict[str, Any]:
        """Return latest momentum labels (call after enrich)."""
        if hasattr(self, "momentum_profile") and self.momentum_profile:
            return self.momentum_profile
        return build_momentum_profile(self.df)

    def run(self) -> pd.DataFrame:
        return self.enrich()


def _last_valid_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    subset = df.dropna(how="all")
    if subset.empty:
        return df.iloc[-1]
    return subset.iloc[-1]


def _build_sector_rotation_block(df: pd.DataFrame) -> dict[str, Any]:
    """XLE/XLF/XLK/XLU/XLI relative to SPY — where capital is rotating."""
    sectors = {"XLE": "Energy", "XLF": "Financials", "XLK": "Technology",
               "XLU": "Utilities", "XLI": "Industrials"}
    out: dict[str, Any] = {}
    for col, name in sectors.items():
        z_col = f"{col}_Z"
        if z_col in df.columns:
            mom = get_momentum_profile(df[z_col], good_when_z_rises=True)
            out[name] = {"label": mom.get("label"), "z_latest": mom.get("z_latest")}
    # Regime signal: XLE vs XLU direction tells us risk-on/defensive rotation
    xle = out.get("Energy", {}).get("z_latest")
    xlu = out.get("Utilities", {}).get("z_latest")
    if xle is not None and xlu is not None:
        out["rotation_signal"] = "risk_on" if xle > xlu else "defensive"
    return out


def _build_geopolitical_vol_block(df: pd.DataFrame, row: pd.Series | None) -> dict[str, Any]:
    """OVX + GVZ momentum — market's real-time geopolitical/supply shock pricing."""
    out: dict[str, Any] = {}
    for key, z_col, raw_col in [("OVX", "OVX_Z", "OVX"), ("GVZ", "GVZ_Z", "GVZ")]:
        if z_col in df.columns:
            mom = get_momentum_profile(df[z_col], good_when_z_rises=False)
            out[key] = {
                "label": mom.get("label"),
                "z_latest": mom.get("z_latest"),
                "streak_days": mom.get("streak_days"),
                "latest_level": _safe_float(row.get(raw_col)) if row is not None else None,
            }
    if "OVX_Z" in df.columns and "GVZ_Z" in df.columns and row is not None:
        ovx_z = _safe_float(row.get("OVX_Z"))
        gvz_z = _safe_float(row.get("GVZ_Z"))
        if ovx_z is not None and gvz_z is not None:
            # Both elevated = systemic risk; GVZ > OVX = safe-haven driven (geopolitical)
            out["interpretation"] = (
                "geopolitical_fear" if gvz_z > ovx_z + 0.5
                else "supply_shock" if ovx_z > gvz_z + 0.5
                else "mixed"
            )
    return out


def _build_labour_inflation_context() -> dict[str, Any]:
    """JOLTS + NFP + PCE from disk — labour market and Fed-preferred inflation context."""
    from pathlib import Path
    RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
    out: dict[str, Any] = {}

    def _load(fname: str) -> pd.Series | None:
        p = RAW / f"{fname}.csv"
        if not p.exists():
            return None
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            s = df.iloc[:, 0].dropna().sort_index()
            return s if not s.empty else None
        except Exception:
            return None

    def _mom_label(s: pd.Series) -> str:
        if len(s) < 3:
            return "Insufficient data"
        recent = s.iloc[-3:].values
        if recent[-1] > recent[-2] > recent[-3]:
            return "Accelerating"
        if recent[-1] < recent[-2] < recent[-3]:
            return "Decelerating"
        return "Mixed"

    for key, fname in [("JOLTS_Openings", "jolts_openings"), ("NFP_Change", "nfp_change"), ("PCE_YoY", "pce_yoy")]:
        s = _load(fname)
        if s is not None:
            out[key] = {
                "latest": round(float(s.iloc[-1]), 2),
                "prior": round(float(s.iloc[-2]), 2) if len(s) >= 2 else None,
                "trend_3m": _mom_label(s),
                "as_of": s.index[-1].date().isoformat(),
            }
    return out


def _build_ff_factor_block() -> dict[str, Any]:
    """Fama-French factor momentum (20-day z-score of cumulative returns).

    Uses ff_factors_5mom_daily.csv (Mkt-RF, SMB, HML, RMW, CMA, Mom).
    Tells the LLM about current factor rotation regime:
      SMB+ → small-cap leadership (risk-on breadth)
      HML+ → value outperforming growth (late-cycle / cheap)
      Mom+ → momentum factor intact (trend continuation)
    """
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "raw" / "ff_factors_5mom_daily.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df = df.sort_index().astype(float)
        if len(df) < 30:
            return {}
        window = df.iloc[-60:]  # ~3 months context
        recent = df.iloc[-20:]  # ~1 month for z-score
        out: dict[str, Any] = {}
        factor_map = {
            "Mkt-RF": ("Market Excess Return", True),
            "SMB": ("Small-Cap Premium", True),
            "HML": ("Value Premium", True),
            "Mom": ("Momentum Factor", True),
            "RMW": ("Profitability Premium", True),
            "CMA": ("Investment Premium", True),
        }
        for col, (label, good_up) in factor_map.items():
            if col not in df.columns:
                continue
            cum_60 = window[col].cumsum()
            if cum_60.std() < 1e-9:
                continue
            z = (cum_60.iloc[-1] - cum_60.mean()) / cum_60.std()
            mom20 = recent[col].sum()  # total return last 20 days
            trend = ("Positive" if mom20 > 1.0 else "Negative" if mom20 < -1.0 else "Flat")
            out[label] = {
                "z_60d": round(float(z), 2),
                "return_20d": round(float(mom20), 2),
                "trend": trend,
            }
        # Summarise factor regime
        smb_z = out.get("Small-Cap Premium", {}).get("z_60d", 0)
        hml_z = out.get("Value Premium", {}).get("z_60d", 0)
        mom_z = out.get("Momentum Factor", {}).get("z_60d", 0)
        if smb_z > 0.5 and mom_z > 0:
            factor_regime = "risk_on_breadth"
        elif hml_z > 0.5 and smb_z < 0:
            factor_regime = "value_rotation"
        elif mom_z < -0.5:
            factor_regime = "momentum_breakdown"
        else:
            factor_regime = "mixed"
        out["factor_regime"] = factor_regime
        out["as_of"] = df.index[-1].date().isoformat()
        return out
    except Exception:
        return {}


def _build_fed_rate_path_block() -> dict[str, Any]:
    """Fetch ZQ monthly futures term structure → implied Fed rate path + cuts/hikes priced in.

    Uses ZQ{M}{YY}.CBT contracts on CME (yfinance fast_info).
    Implied rate = 100 - price.  change_vs_now_bp < 0 = cuts priced in.
    Called from generate_quant_payload_dict() so the result is inside quant_engine JSON.
    """
    from datetime import datetime as _dt
    _MONTH_CODES = "FGHJKMNQUVXZ"
    today = _dt.today()
    contracts: list[tuple[str, int, int]] = []
    y, m = today.year, today.month
    for _ in range(12):
        m += 1
        if m > 12:
            m, y = 1, y + 1
        contracts.append((f"ZQ{_MONTH_CODES[m - 1]}{str(y)[-2:]}.CBT", y, m))

    term_structure: list[dict] = []
    for ticker, yr, mo in contracts:
        try:
            price = yf.Ticker(ticker).fast_info.last_price
            if price is None or price != price:
                continue
            term_structure.append({
                "ticker": ticker,
                "year": yr,
                "month": mo,
                "implied_rate_pct": round(100.0 - float(price), 4),
            })
            if len(term_structure) >= 8:
                break
        except Exception:
            continue

    if not term_structure:
        return {"error": "no ZQ contracts available", "summary": "unavailable"}

    # Current rate: nearest contract as proxy (FEDFUNDS monthly lags by ~5 weeks)
    current_rate = term_structure[0]["implied_rate_pct"]

    for row in term_structure:
        row["change_vs_now_bp"] = round((row["implied_rate_pct"] - current_rate) * 100, 1)

    def _describe(bp: float) -> str:
        n25 = round(abs(bp) / 25)
        if bp < -10:
            return f"{n25} cut{'s' if n25 != 1 else ''} priced in ({-bp:.0f}bp)"
        if bp > 10:
            return f"{n25} hike{'s' if n25 != 1 else ''} priced in (+{bp:.0f}bp)"
        return "on hold"

    last = term_structure[-1]
    eoy = next((r for r in term_structure if r["year"] == today.year and r["month"] == 12), last)
    summary = (
        f"Nearest implied rate: {current_rate:.2f}%. "
        f"Year-end {eoy['year']}-{eoy['month']:02d}: {_describe(eoy['change_vs_now_bp'])}. "
        f"Horizon {last['year']}-{last['month']:02d}: {_describe(last['change_vs_now_bp'])}."
    )
    print(f"  fed rate path: {summary}", file=sys.stderr)
    return {
        "nearest_implied_rate_pct": current_rate,
        "term_structure": term_structure,
        "summary": summary,
        "as_of": today.date().isoformat(),
    }


def _build_fed_liquidity_block() -> dict[str, Any]:
    """Comprehensive Fed liquidity & money supply block.

    Fetches (all weekly FRED unless noted):
      WALCL   – Fed total assets ($M)
      WRESBAL – Bank reserves ($M)
      WTREGEN – Treasury General Account / TGA ($M)
      RRPTTLD – Overnight reverse repo ($B, daily)
      M2SL    – M2 money supply ($B, monthly)

    Fed Net Liquidity = WALCL − TGA − RRP  (the money that actually hits markets)
    """
    def _fetch(sid: str) -> pd.Series | None:
        try:
            s = _fred_series(sid).dropna().sort_index()
            return s if len(s) >= 4 else None
        except Exception as exc:
            print(f"  liquidity: {sid} unavailable ({exc})", file=sys.stderr)
            return None

    def _latest_bn(s: pd.Series | None) -> float | None:
        if s is None:
            return None
        return round(float(s.iloc[-1]) / 1_000, 1)

    def _pct_chg(s: pd.Series | None, periods: int) -> float | None:
        if s is None or len(s) <= periods:
            return None
        old = float(s.iloc[-periods])
        return round((float(s.iloc[-1]) - old) / old * 100, 2) if old else None

    def _fmt(v: float | None) -> str:
        if v is None:
            return "n/a"
        return f"{'+' if v >= 0 else ''}{v:.1f}%"

    walcl = _fetch("WALCL")     # $M weekly
    reserves = _fetch("WRESBAL")  # $M weekly
    tga = _fetch("WTREGEN")      # $M weekly
    rrp = _fetch("RRPTTLD")      # $B daily (already in $B)
    m2 = _fetch("M2SL")          # $B monthly

    walcl_bn = _latest_bn(walcl)
    reserves_bn = _latest_bn(reserves)
    tga_bn = _latest_bn(tga)
    # RRP is already in $B
    rrp_bn = round(float(rrp.iloc[-1]), 1) if rrp is not None else None

    # Fed Net Liquidity = WALCL − TGA − RRP
    net_liquidity_bn: float | None = None
    if walcl_bn is not None and tga_bn is not None:
        net_liquidity_bn = round(walcl_bn - tga_bn - (rrp_bn or 0), 1)

    # 13w changes
    walcl_13w = _pct_chg(walcl, 13)
    walcl_52w = _pct_chg(walcl, 52)
    reserves_13w = _pct_chg(reserves, 13)

    # M2 YoY (monthly — 12 observations back)
    m2_yoy: float | None = None
    m2_latest_bn: float | None = None
    if m2 is not None and len(m2) >= 13:
        m2_latest_bn = round(float(m2.iloc[-1]), 1)
        old_m2 = float(m2.iloc[-13])
        m2_yoy = round((float(m2.iloc[-1]) - old_m2) / old_m2 * 100, 2) if old_m2 else None

    # QT/QE signal from WALCL 13w change
    if walcl_13w is not None:
        bs_signal = "QE/expanding" if walcl_13w > 1.0 else "QT/shrinking" if walcl_13w < -1.0 else "stable"
    else:
        bs_signal = "unknown"

    # 8-week slope trend
    bs_trend = "unknown"
    if walcl is not None and len(walcl) >= 8:
        recent = walcl.iloc[-8:]
        cov = sum((i - 3.5) * (float(v) - float(recent.mean())) for i, v in enumerate(recent))
        bs_trend = "expanding" if cov > 0 else "shrinking"

    summary_parts = [f"Fed assets: ${walcl_bn:.0f}B ({_fmt(walcl_13w)} 13w, {_fmt(walcl_52w)} 52w) → {bs_signal}."]
    if net_liquidity_bn is not None:
        summary_parts.append(f"Net liquidity (assets−TGA−RRP): ${net_liquidity_bn:.0f}B.")
    if reserves_bn is not None:
        summary_parts.append(f"Bank reserves: ${reserves_bn:.0f}B ({_fmt(reserves_13w)} 13w).")
    if m2_yoy is not None:
        m2_label = "expanding" if m2_yoy > 0 else "contracting"
        summary_parts.append(f"M2: ${m2_latest_bn:.0f}B, YoY {_fmt(m2_yoy)} ({m2_label}).")
    summary = " ".join(summary_parts)

    print(f"  fed liquidity: {summary}", file=sys.stderr)
    return {
        "as_of": walcl.index[-1].date().isoformat() if walcl is not None else None,
        "fed_total_assets_bn": walcl_bn,
        "fed_total_assets_13w_pct": walcl_13w,
        "fed_total_assets_52w_pct": walcl_52w,
        "balance_sheet_signal": bs_signal,
        "balance_sheet_trend": bs_trend,
        "bank_reserves_bn": reserves_bn,
        "bank_reserves_13w_pct": reserves_13w,
        "tga_bn": tga_bn,
        "rrp_bn": rrp_bn,
        "net_liquidity_bn": net_liquidity_bn,
        "m2_bn": m2_latest_bn,
        "m2_yoy_pct": m2_yoy,
        "summary": summary,
    }


def generate_quant_payload_dict(df: pd.DataFrame) -> dict[str, Any]:
    """
    Structured quant metadata for Layer 3 LLM synthesis (no raw price series).
    """
    row = _last_valid_row(df)
    as_of = None
    if row is not None and hasattr(row, "name"):
        try:
            as_of = pd.Timestamp(row.name).date().isoformat()
        except Exception:
            as_of = str(row.name)

    spread_z = None
    spread_val = None
    l1_state = "Unknown"
    if row is not None:
        spread_z = _safe_float(row.get("Spread_2y10y_Z"))
        spread_val = _safe_float(row.get("Spread_2y10y"))
        tag = row.get("L1_Yield_Curve_Tag", "Unknown")
        l1_state = str(tag) if pd.notna(tag) else "Unknown"

    l2_tag = "Unknown"
    if row is not None:
        t2 = row.get("L2_Squeeze_Tag", "Unknown")
        l2_tag = str(t2) if pd.notna(t2) else "Unknown"

    regime_match = df.attrs.get("regime_match", {}) if hasattr(df, "attrs") else {}
    if not regime_match:
        regime_match = {}

    momentum_profile = df.attrs.get("momentum_profile", {}) if hasattr(df, "attrs") else {}
    if not momentum_profile:
        momentum_profile = build_momentum_profile(df)

    macro_regime = df.attrs.get("macro_regime", {}) if hasattr(df, "attrs") else {}
    if not macro_regime:
        macro_regime = compute_macro_regime_detail(df)

    risk_analytics = df.attrs.get("risk_analytics", {}) if hasattr(df, "attrs") else {}
    if not risk_analytics:
        risk_analytics = compute_penalized_divergence(df, momentum_profile)
        risk_analytics["layer2_risk"] = build_layer2_risk_profile(df)

    payload: dict[str, Any] = {
        "as_of": as_of,
        "macro_regime_label": macro_regime.get("macro_regime_label", "Unknown"),
        "macro_regime_detail": macro_regime,
        "momentum_profile": momentum_profile,
        "Risk_Analytics": {
            "divergence_score": risk_analytics.get("divergence_score", 0),
            "base_score": risk_analytics.get("base_score", 0),
            "penalty_reasons": risk_analytics.get("penalty_reasons", []),
            "penalty_reason": risk_analytics.get("penalty_reason", ""),
            "layer2_risk": risk_analytics.get("layer2_risk", {}),
            "geopolitical_vol": _build_geopolitical_vol_block(df, row),
        },
        "Layer_1_Structural": {
            "Yield_Curve": {
                "State": l1_state,
                "Z_Score": spread_z,
                "Spread_2y10y": spread_val,
            },
            "Cross_Market_Yields": {
                "Spread_TW_US": _safe_float(row.get("Spread_TW_US")) if row is not None else None,
                "Spread_CN_US": _safe_float(row.get("Spread_CN_US")) if row is not None else None,
                "VIX_Z": _safe_float(row.get("VIX_Z")) if row is not None else None,
                "MOVE_Z": _safe_float(row.get("MOVE_Z")) if row is not None else None,
            },
            "Valuation_Gravity": {
                "ERP_Z_Score": _safe_float(row.get("ERP_Z")) if row is not None else None,
            },
            "Global_Liquidity_and_Growth": {
                "DXY_Z_Score": _safe_float(row.get("DXY_Z")) if row is not None else None,
                "US_Growth_Z_Score": _safe_float(row.get("US_Growth_Z")) if row is not None else None,
                "Fed_Liquidity": _build_fed_liquidity_block(),
            },
            "Fed_Rate_Path": _build_fed_rate_path_block(),
            "Labour_and_Inflation_Context": _build_labour_inflation_context(),
        },
        "Layer_2_Tactical": {
            "Asset_Squeeze_Risk": l2_tag,
            "Asset": "JPY",
            "Sector_Rotation": _build_sector_rotation_block(df),
            "FF_Factors": _build_ff_factor_block(),
        },
        "Regime_Match": {
            "Global_Macro": {
                "Matched_Date": regime_match.get("matched_date"),
                "Distance": _safe_float(regime_match.get("distance")),
                "Key_Feature_Diff": REGIME_MATCH_KEY_FEATURE_DIFF,
            },
        },
        "Regime_Duration": _compute_regime_duration(df),
    }
    return payload


def generate_quant_payload(df: pd.DataFrame) -> str:
    """JSON string wrapper for flow_run / file export."""
    return json.dumps(generate_quant_payload_dict(df), ensure_ascii=False, indent=2)


# Legacy aliases — kept for any external scripts that may reference old names.
generate_llm_payload_dict = generate_quant_payload_dict
generate_llm_payload = generate_quant_payload
