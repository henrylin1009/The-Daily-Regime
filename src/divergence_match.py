"""
Global Divergence Historical Matching — Part B of Historical Analogues.

Measures the cross-country policy / inflation / FX divergence pattern and finds
the 3 most similar historical episodes. Forward returns use EEM, GLD and copper
(not SPY) because those assets are most sensitive to global divergence dynamics.

Features (all z-scored, monthly):
  1. fed_ecb_spread      — Fed minus ECB rate (policy divergence US vs Europe)
  2. fed_boj_spread      — Fed minus BOJ rate (carry-trade pressure / JPY risk)
  3. fed_kr_spread       — Fed minus Korea rate (US vs EM policy divergence)
  4. us_eu_cpi_spread    — US CPI YoY minus Euro Area CPI YoY (inflation divergence)
  5. eem_vs_spy_12m      — EEM 12-month return minus SPY 12-month return (EM appeal)
  6. cn_fx_reserves_12m  — China FX reserves 12-month change (capital pressure signal)
  7. usdjpy_12m          — USD/JPY 12-month trend (yen pressure / carry unwind risk)
  8. eurusd_12m          — EUR/USD 12-month trend (dollar strength vs Europe)
  9. credit_spread_hy    — US HY credit spread (global risk appetite anchor)

NOTE: China CPI YoY (FRED CHNCPIALLMINMEI) was dropped from the feature set —
FRED stopped updating it (frozen ~2025-04), which previously truncated the whole
matrix via dropna() and froze the "current" match vector ~14 months in the past.
It is still collected; re-add here if/when FRED resumes timely updates.

Common history starts ~1999 (EUR/USD, ECB rate, Euro CPI). Effective candidate
pool after excluding recent 24 months: ~1999–2023 ≈ 288 months.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.collect import load_all_from_cache
from src.config import PROCESSED_DIR

DIVERGENCE_FEATURES = [
    "fed_ecb_spread",
    "fed_boj_spread",
    "fed_kr_spread",
    "us_eu_cpi_spread",
    "eem_vs_spy_12m",
    "cn_fx_reserves_12m",
    "usdjpy_12m",
    "eurusd_12m",
    "credit_spread_hy",
]

DIVERGENCE_PERIOD_DESCRIPTIONS: dict[str, str] = {
    "1999": "Euro launch era — dollar strength, EM recovery post-Asia crisis",
    "2000": "Dot-com peak — Fed tightening, strong dollar, EM under pressure",
    "2001": "Post dot-com — Fed easing cycle begins, dollar weakens",
    "2004": "Mid-2000s expansion — synchronized global growth, EM outperformance",
    "2006": "Pre-GFC liquidity peak — carry trades rampant, low vol globally",
    "2007": "Pre-GFC — dollar weakness, commodity boom, EM near peak",
    "2008": "GFC — synchronized global crash, dollar surge, carry unwind",
    "2009": "Post-GFC recovery — Fed near zero, EM rebound, dollar weak",
    "2011": "Euro debt crisis — ECB hiking then cutting, divergence spike",
    "2013": "Taper tantrum — Fed signals exit, EM capital flight, dollar up",
    "2014": "Divergence begins — Fed ends QE, ECB cuts, BOJ expands, DXY surges",
    "2015": "Peak divergence — dollar dominance, China devaluation, EM stress",
    "2018": "Fed hikes alone — dollar rally, EM under pressure, carry unwind",
    "2019": "Fed pivots dovish — ECB/BOJ still loose, dollar softens",
    "2020": "COVID shock — synchronized easing, EM rebound late year",
    "2021": "Reflation — synchronized recovery, EM outperforms briefly",
    "2022": "Inflation surge — Fed aggressive, ECB follows, EM squeezed by dollar",
    "2023": "Disinflation — Fed holds high, ECB catches up, BOJ still outlier",
}


def _period_desc(period: str) -> str:
    year = period[:4]
    for prefix, desc in sorted(DIVERGENCE_PERIOD_DESCRIPTIONS.items(), key=lambda x: -len(x[0])):
        if year.startswith(prefix):
            return desc
    return f"Global divergence regime circa {period}"


def _monthly_return_series(df: pd.DataFrame, months: int = 12) -> pd.Series:
    """Convert price DataFrame to monthly resample then compute N-month pct change."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    prices = d.set_index("date")["value"].sort_index()
    monthly = prices.resample("ME").last().ffill()
    return monthly.pct_change(months) * 100


def _monthly_level(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    s = d.set_index("date")["value"].sort_index()
    return s.resample("ME").last().ffill()


def _monthly_12m_change(df: pd.DataFrame) -> pd.Series:
    """Monthly series, 12-month absolute change (for rates / CPI)."""
    s = _monthly_level(df)
    return s.diff(12)


def build_divergence_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build and z-score the global divergence feature matrix.
    Returns monthly DataFrame indexed by month-end date.
    """
    required = [
        "fed_funds_rate", "ecb_deposit_rate", "boj_policy_rate", "kr_policy_rate",
        "cpi_yoy", "eu_cpi_yoy",
        "eem", "spy",
        "cn_fx_reserves",
        "fx_usdjpy", "fx_eurusd",
        "credit_spread_hy",
    ]
    missing = [k for k in required if k not in data or data[k].empty]
    if missing:
        raise ValueError(f"divergence_match: missing data keys: {missing}")

    fed = _monthly_level(data["fed_funds_rate"])
    ecb = _monthly_level(data["ecb_deposit_rate"])
    boj = _monthly_level(data["boj_policy_rate"])
    kr  = _monthly_level(data["kr_policy_rate"])

    us_cpi  = _monthly_level(data["cpi_yoy"])
    eu_cpi  = _monthly_level(data["eu_cpi_yoy"])

    eem_12m = _monthly_return_series(data["eem"], 12)
    spy_12m = _monthly_return_series(data["spy"], 12)

    cn_fx   = _monthly_12m_change(data["cn_fx_reserves"])   # bn USD change
    usdjpy  = _monthly_return_series(data["fx_usdjpy"], 12)  # higher = yen weakness
    eurusd  = _monthly_return_series(data["fx_eurusd"], 12)  # higher = dollar weakness
    hy      = _monthly_level(data["credit_spread_hy"])

    features = pd.DataFrame({
        "fed_ecb_spread":     fed - ecb,
        "fed_boj_spread":     fed - boj,
        "fed_kr_spread":      fed - kr,
        "us_eu_cpi_spread":   us_cpi - eu_cpi,
        "eem_vs_spy_12m":     eem_12m - spy_12m,
        "cn_fx_reserves_12m": cn_fx,
        "usdjpy_12m":         usdjpy,
        "eurusd_12m":         eurusd,
        "credit_spread_hy":   hy,
    }).dropna(how="any")

    # Z-score on full history
    z = (features - features.mean()) / features.std()

    z.to_csv(PROCESSED_DIR / "divergence_matrix_zscored.csv")
    features.to_csv(PROCESSED_DIR / "divergence_matrix_raw.csv")

    return z


def _forward_stats(
    price_df: pd.DataFrame,
    match_date: pd.Timestamp,
    label: str,
) -> dict[str, float | None]:
    """Compute 3m / 12m return and max 12m drawdown after match_date."""
    d = price_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    prices = d.set_index("date")["value"].sort_index()
    prices = prices[prices.index > match_date]
    if prices.empty:
        return {f"{label}_3m": None, f"{label}_12m": None, f"{label}_maxdd_12m": None}

    start = prices.iloc[0]

    def _ret(months: int) -> float | None:
        end = match_date + pd.DateOffset(months=months)
        w = prices[prices.index <= end]
        return None if w.empty else (w.iloc[-1] / start - 1) * 100

    r3  = _ret(3)
    r12 = _ret(12)

    w12 = prices[prices.index <= match_date + pd.DateOffset(months=12)]
    maxdd = None
    if len(w12) > 1:
        cum = w12 / start
        maxdd = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)

    return {f"{label}_3m": r3, f"{label}_12m": r12, f"{label}_maxdd_12m": maxdd}


def find_divergence_periods(
    div_matrix: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    n_matches: int = 3,
    exclude_recent_months: int = 24,
    min_separation_months: int = 24,
) -> list[dict]:
    """Find top-N historical periods with similar global divergence pattern."""
    if div_matrix.empty or len(div_matrix) < 4:
        return []

    today_vec = div_matrix.iloc[-1].values
    exclude_recent_months = max(int(exclude_recent_months), 24)
    cutoff = div_matrix.index[-1] - pd.DateOffset(months=exclude_recent_months)
    candidates_df = div_matrix[div_matrix.index <= cutoff]

    window = 3
    candidates: list[tuple[float, str, pd.Timestamp]] = []
    for i in range(window - 1, len(candidates_df)):
        end_date = candidates_df.index[i]
        vec = candidates_df.iloc[i - window + 1: i + 1].mean().values
        dist = float(np.linalg.norm(vec - today_vec))
        candidates.append((dist, end_date.strftime("%Y-%m"), end_date))

    candidates.sort(key=lambda x: x[0])

    matches: list[dict] = []
    selected: list[pd.Timestamp] = []

    for dist, period, end_date in candidates:
        if any(abs((end_date - s).days) < min_separation_months * 30 for s in selected):
            continue
        # Decade diversity: max 2 per decade
        bucket = (end_date.year // 10) * 10
        if [( s.year // 10) * 10 for s in selected].count(bucket) >= 2:
            continue

        selected.append(end_date)

        # Forward returns: EEM, GLD, copper
        stats: dict = {}
        for key, label in [("eem", "eem"), ("gld", "gld"), ("copper", "copper")]:
            if key in data and not data[key].empty:
                stats.update(_forward_stats(data[key], end_date, label))
            else:
                stats.update({f"{label}_3m": None, f"{label}_12m": None, f"{label}_maxdd_12m": None})

        matches.append({
            "date": period,
            "distance": round(dist, 2),
            "description": _period_desc(period),
            **stats,
        })
        if len(matches) >= n_matches:
            break

    return matches


def save_divergence_matches(matches: list[dict]) -> None:
    df = pd.DataFrame(matches)
    df.to_csv(PROCESSED_DIR / "divergence_matches.csv", index=False)


def main() -> None:
    data = load_all_from_cache()
    if not data:
        print("No cached data. Run: python -m src.collect", file=sys.stderr)
        sys.exit(1)

    matrix = build_divergence_matrix(data)
    matches = find_divergence_periods(matrix, data)
    save_divergence_matches(matches)

    print(f"Divergence matrix: {matrix.index.min().date()} → {matrix.index.max().date()} ({len(matrix)} months)")
    print(f"\nTop {len(matches)} global divergence analogues:\n")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m['date']} (distance={m['distance']:.2f})")
        print(f"     {m['description']}")
        for asset in ["eem", "gld", "copper"]:
            r12 = m.get(f"{asset}_12m")
            dd  = m.get(f"{asset}_maxdd_12m")
            print(f"     {asset.upper():6s} 12m: {f'{r12:+.1f}%' if r12 is not None else 'N/A':>8}  Max DD: {f'{dd:.1f}%' if dd is not None else 'N/A'}")
        print()


if __name__ == "__main__":
    main()
