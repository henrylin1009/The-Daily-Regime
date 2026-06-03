"""Stage 3: Historical regime comparison via feature-vector similarity."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.collect import load_all_from_cache
from src.config import PROCESSED_DIR

REGIME_FEATURES = [
    "cpi_yoy",
    "unemployment_rate",
    "spread_2y10y",
    "vix",
    "credit_spread_hy",
    "fed_funds_rate",
]

# Included in feature matrix when cached (multi-central-bank context)
OPTIONAL_REGIME_FEATURES = [
    "ecb_deposit_rate",
    "boj_policy_rate",
]

PERIOD_DESCRIPTIONS: dict[str, str] = {
    "1999": "Late 1990s boom — low unemployment, strong growth",
    "2000": "Dot-com peak — tight labour, elevated valuations",
    "2005": "Mid-2000s expansion — rising rates, solid growth",
    "2007": "Pre-GFC — flat curve, low vol, credit complacency",
    "2008": "Global Financial Crisis — credit stress, policy easing",
    "2009": "Post-crisis recovery — stimulus, risk rebound",
    "2011": "Euro debt crisis — risk-off, slow growth",
    "2014": "Mid-cycle slowdown — low vol, dollar strength, oil decline begins",
    "2015": "China slowdown fears — commodity weakness",
    "2018": "Late 2018 — Fed tightening, curve inversion beginning",
    "2020": "COVID shock — unprecedented stimulus, volatility spike",
    "2022": "Inflation surge — aggressive Fed hiking cycle",
    "2023": "Disinflation glide — higher rates, resilient labour",
}


def _decade_bucket(ts: pd.Timestamp) -> int:
    """Bucket year into decade (2000, 2010, ...) for diversity constraint."""
    return (ts.year // 10) * 10


def _period_description(period: str) -> str:
    year = period[:4]
    for prefix, desc in sorted(PERIOD_DESCRIPTIONS.items(), key=lambda x: -len(x[0])):
        if year.startswith(prefix):
            return desc
    return f"Macro regime circa {period}"


def build_feature_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge regime features into monthly DataFrame, forward-fill daily data,
    z-score each column. Returns DataFrame indexed by month-end date.
    """
    frames: list[pd.Series] = []

    features = list(REGIME_FEATURES)
    for feat in OPTIONAL_REGIME_FEATURES:
        if feat in data and not data[feat].empty:
            features.append(feat)

    for feat in features:
        if feat not in data:
            print(f"  history_match: skipping missing feature {feat}", file=sys.stderr)
            continue
        df = data[feat].copy()
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")["value"].sort_index()
        monthly = s.resample("ME").last().ffill()
        frames.append(monthly.rename(feat))

    raw = pd.concat(frames, axis=1, sort=True).dropna(how="any")

    # Z-score normalization on full history
    z = (raw - raw.mean()) / raw.std()

    # Persist for downstream regime models (HMM).
    # These are monthly, month-end indexed.
    (PROCESSED_DIR / "feature_matrix_raw.csv").write_text(raw.to_csv())
    (PROCESSED_DIR / "feature_matrix_zscored.csv").write_text(z.to_csv())

    return z


def _spy_forward_stats(
    spy: pd.DataFrame,
    match_date: pd.Timestamp,
) -> dict[str, float | None]:
    """Compute SPY 3m/12m returns and max 12m drawdown after match_date."""
    df = spy.copy()
    df["date"] = pd.to_datetime(df["date"])
    prices = df.set_index("date")["value"].sort_index()
    prices = prices[prices.index > match_date]
    if prices.empty:
        return {"spy_return_3m": None, "spy_return_12m": None, "max_drawdown_12m": None}

    start_price = prices.iloc[0]

    def _return(months: int) -> float | None:
        end = match_date + pd.DateOffset(months=months)
        window = prices[prices.index <= end]
        if window.empty:
            return None
        return (window.iloc[-1] / start_price - 1) * 100

    ret_3m = _return(3)
    ret_12m = _return(12)

    window_12m = prices[prices.index <= match_date + pd.DateOffset(months=12)]
    max_dd = None
    if len(window_12m) > 1:
        cumulative = window_12m / start_price
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = float(drawdown.min() * 100)

    return {
        "spy_return_3m": ret_3m,
        "spy_return_12m": ret_12m,
        "max_drawdown_12m": max_dd,
    }


def find_similar_periods(
    feature_matrix: pd.DataFrame,
    data: dict[str, pd.DataFrame],
    n_matches: int = 3,
    exclude_recent_months: int = 24,
    min_separation_months: int = 24,
    current_episode_start: pd.Timestamp | None = None,
) -> list[dict]:
    """
    Find top-N historical 3-month windows most similar to today.

    Excludes recent months and enforces minimum separation between matches
    so results span different regimes (not three adjacent 2025 months).
    """
    if feature_matrix.empty or len(feature_matrix) < 4:
        return []

    today_vec = feature_matrix.iloc[-1].values
    # Always exclude the most recent 24 months so all matches have full 12m forward data
    # and don't reflect conditions too close to the present.
    exclude_recent_months = max(int(exclude_recent_months), 24)
    cutoff = feature_matrix.index[-1] - pd.DateOffset(months=exclude_recent_months)
    candidate_matrix = feature_matrix[feature_matrix.index <= cutoff]
    if current_episode_start is not None:
        current_episode_start = pd.Timestamp(current_episode_start)
        # Keep only candidates whose full 12m forward window is fully before current episode.
        latest_valid_match = current_episode_start - pd.DateOffset(months=13)
        candidate_matrix = candidate_matrix[candidate_matrix.index <= latest_valid_match]

    spy = data.get("spy")
    if spy is None:
        raise ValueError("SPY data required for forward return calculation")

    candidates: list[tuple[float, str, pd.Timestamp]] = []

    window = 3
    for i in range(window - 1, len(candidate_matrix)):
        end_date = candidate_matrix.index[i]
        window_data = candidate_matrix.iloc[i - window + 1 : i + 1]
        vec = window_data.mean().values
        dist = float(np.linalg.norm(vec - today_vec))
        period = end_date.strftime("%Y-%m")
        candidates.append((dist, period, end_date))

    candidates.sort(key=lambda x: x[0])

    matches: list[dict] = []
    selected_dates: list[pd.Timestamp] = []

    for dist, period, end_date in candidates:
        too_close = any(
            abs((end_date - sd).days) < min_separation_months * 30
            for sd in selected_dates
        )
        if too_close:
            continue

        bucket = _decade_bucket(end_date)
        buckets_selected = [_decade_bucket(sd) for sd in selected_dates]
        if buckets_selected.count(bucket) >= 2:
            continue
        if len(matches) == n_matches - 1 and len(set(buckets_selected)) < 2 and bucket in buckets_selected:
            continue

        selected_dates.append(end_date)
        spy_stats = _spy_forward_stats(spy, end_date)
        matches.append(
            {
                "date": period,
                "distance": round(dist, 2),
                "description": _period_description(period),
                "what_happened": "",
                **spy_stats,
            }
        )
        if len(matches) >= n_matches:
            break

    return matches


def save_matches(matches: list[dict]) -> None:
    """Persist matches to data/processed/history_matches.csv."""
    df = pd.DataFrame(matches)
    path = PROCESSED_DIR / "history_matches.csv"
    df.to_csv(path, index=False)


def main() -> None:
    data = load_all_from_cache()
    if not data:
        print("No cached data. Run: python -m src.collect", file=sys.stderr)
        sys.exit(1)

    matrix = build_feature_matrix(data)
    matches = find_similar_periods(matrix, data)
    save_matches(matches)

    print(f"Feature matrix: {matrix.index.min().date()} -> {matrix.index.max().date()} ({len(matrix)} months)")
    print(f"\nTop {len(matches)} similar periods:\n")
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m['date']} (distance={m['distance']:.2f})")
        print(f"     {m['description']}")
        if m["spy_return_3m"] is not None:
            print(
                f"     SPY 3m: {m['spy_return_3m']:.1f}%  |  "
                f"12m: {m['spy_return_12m']:.1f}%  |  "
                f"Max DD 12m: {m['max_drawdown_12m']:.1f}%"
            )
        else:
            print("     SPY forward stats: N/A")
        print()


if __name__ == "__main__":
    main()
