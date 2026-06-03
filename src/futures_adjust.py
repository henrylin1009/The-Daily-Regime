"""Panama back-adjustment for continuous futures (ZQ=F roll artifacts)."""

from __future__ import annotations

import pandas as pd


def panama_back_adjust(df: pd.DataFrame, price_col: str = "value") -> pd.DataFrame:
    """
    Apply Panama back-adjustment when price jumps exceed a roll threshold.

    Detects large overnight gaps (typical of futures rolls) and shifts
    all prior prices by the gap so the series is continuous.
    """
    if df.empty or len(df) < 2:
        return df

    out = df.sort_values("date").reset_index(drop=True).copy()
    prices = out[price_col].astype(float).values.copy()
    roll_threshold = 0.03

    # Panama back-adjust: walk backwards, shift prior prices at each roll
    for i in range(len(prices) - 1, 0, -1):
        if prices[i - 1] == 0:
            continue
        pct = abs((prices[i] - prices[i - 1]) / prices[i - 1])
        if pct > roll_threshold:
            gap = prices[i] - prices[i - 1]
            prices[:i] -= gap

    out[price_col] = prices
    return out
