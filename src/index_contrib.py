"""S&P 500 (approx) constituent contributors for SPY returns.

This module aims to answer: "Which stocks contributed most to SPY move?"

Implementation notes:
- True "S&P 500 membership weights" is hard without paid datasets.
- We try OpenBB `obb.etf.holdings(symbol='SPY')` for tradeable weights.
  If credentials are missing, we fall back to equal weights using a cached
  S&P 500 ticker list from Wikipedia.
- Prices for constituents are pulled with `yfinance` and cached in a rolling
  long-format CSV to keep daily runtime reasonable.
"""

from __future__ import annotations

import io
import json
import re
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from src.config import RAW_DIR, configure_openbb

SPY_HOLDINGS_CACHE = RAW_DIR / "spy_holdings.csv"
SPX_TICKERS_CACHE = RAW_DIR / "spx500_tickers.csv"
SPX_PRICE_CACHE = RAW_DIR / "spy_constituent_prices_cache.csv"  # long format

# Official daily SPY holdings (no API key; same weights as S&P 500 index methodology).
SSGA_SPY_HOLDINGS_URL = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx"
)
_HTTP_HEADERS = {
    "User-Agent": "MacroIntelligencePlatform/1.0 (local research; +https://github.com/)",
}

# Rolling cache window. Needs >= 22 trading days for the 21d return window.
PRICE_LOOKBACK_CAL_DAYS = 120


def _as_date(d) -> pd.Timestamp:
    return pd.to_datetime(d).tz_localize(None).normalize()


def _normalize_yfinance_ticker(t: str) -> str:
    """Convert holdings/Wikipedia tickers to yfinance symbols."""
    t = str(t).strip().upper()
    if t in {"BRK.B", "BRK-B"}:
        return "BRK-B"
    if t in {"BF.B", "BF-B"}:
        return "BF-B"
    # Wikipedia uses dots for share classes; yfinance prefers hyphens.
    if "." in t and "-" not in t:
        return t.replace(".", "-")
    return t


def _download_spx500_tickers_from_wikipedia() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(url, timeout=30, headers=_HTTP_HEADERS)
    resp.raise_for_status()

    # The table is large; pandas read_html is the simplest.
    tables = pd.read_html(io.StringIO(resp.text))
    # Heuristic: the constituents table contains a column named "Symbol".
    target = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any(c.lower() == "symbol" for c in cols):
            target = t
            break
    if target is None:
        raise RuntimeError("Could not parse S&P 500 tickers table from Wikipedia")

    # Standardize columns.
    if "Symbol" not in target.columns:
        # Try case-insensitive match
        symbol_col = next(c for c in target.columns if str(c).lower() == "symbol")
        target = target.rename(columns={symbol_col: "Symbol"})

    out = target[["Symbol"]].copy()
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    out = out[out["Symbol"] != ""]
    return out.rename(columns={"Symbol": "ticker"}).reset_index(drop=True)


def get_spx500_tickers(force_refresh: bool = False) -> pd.DataFrame:
    """Return S&P 500 tickers (cached)."""
    if SPX_TICKERS_CACHE.exists() and not force_refresh:
        return pd.read_csv(SPX_TICKERS_CACHE)
    df = _download_spx500_tickers_from_wikipedia()
    SPX_TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPX_TICKERS_CACHE, index=False)
    return df


def _find_weight_and_ticker_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    cols_lower = {c: str(c).lower() for c in df.columns}
    weight_candidates = [
        c for c, lc in cols_lower.items() if "weight" in lc and lc != "weight_price"
    ]
    ticker_candidates = [
        c for c, lc in cols_lower.items() if lc in {"symbol", "ticker"} or "symbol" in lc
    ]
    weight_col = weight_candidates[0] if weight_candidates else None
    ticker_col = ticker_candidates[0] if ticker_candidates else None
    return weight_col, ticker_col


def _finalize_holdings_df(out: pd.DataFrame) -> pd.DataFrame:
    """Normalize tickers/weights and write cache."""
    out["ticker"] = out["ticker"].map(_normalize_yfinance_ticker)
    out = out[out["weight"] > 0].copy()
    if out.empty:
        raise RuntimeError("No positive weights extracted from SPY holdings")
    out["weight"] = out["weight"] / out["weight"].sum()
    out = out[["ticker", "weight"]].reset_index(drop=True)
    SPY_HOLDINGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SPY_HOLDINGS_CACHE, index=False)
    return out


def _download_spy_holdings_from_ssga() -> pd.DataFrame:
    resp = requests.get(SSGA_SPY_HOLDINGS_URL, timeout=60, headers=_HTTP_HEADERS)
    resp.raise_for_status()
    raw = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=4)
    if "Ticker" not in raw.columns or "Weight" not in raw.columns:
        raise RuntimeError(
            f"Unexpected SSGA holdings columns: {raw.columns.tolist()}"
        )
    out = raw[["Ticker", "Weight"]].copy()
    out.columns = ["ticker", "weight_raw"]
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["weight_raw"] = pd.to_numeric(out["weight_raw"], errors="coerce")
    out = out.dropna(subset=["ticker", "weight_raw"])
    total = float(out["weight_raw"].sum())
    out["weight"] = out["weight_raw"] / 100.0 if total > 1.5 else out["weight_raw"]
    return _finalize_holdings_df(out)


def _download_spy_holdings_from_fmp() -> pd.DataFrame:
    from openbb import obb  # local import to avoid hard dependency in test mode

    configure_openbb()
    result = obb.etf.holdings(symbol="SPY", provider="fmp")
    df = result.to_df()
    if df.empty:
        raise RuntimeError("OpenBB returned empty SPY holdings")

    weight_col, ticker_col = _find_weight_and_ticker_columns(df)
    if not ticker_col or not weight_col:
        raise RuntimeError(
            f"Could not extract ticker/weight columns from holdings df. Columns: {df.columns.tolist()}"
        )

    out = df[[ticker_col, weight_col]].copy()
    out.columns = ["ticker", "weight_raw"]
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["weight_raw"] = pd.to_numeric(out["weight_raw"], errors="coerce")
    out = out.dropna(subset=["ticker", "weight_raw"])
    total = float(out["weight_raw"].sum())
    out["weight"] = out["weight_raw"] / 100.0 if total > 1.5 else out["weight_raw"]
    return _finalize_holdings_df(out)


def get_spy_holdings_weights(force_refresh: bool = False) -> pd.DataFrame:
    """SPY holdings weights (decimal 0–1). SSGA daily file first; FMP optional fallback.

    Returns columns: ticker (str), weight (decimal 0-1)
    """
    if SPY_HOLDINGS_CACHE.exists() and not force_refresh:
        df = pd.read_csv(SPY_HOLDINGS_CACHE)
        if "ticker" in df.columns and "weight" in df.columns:
            return df

    errors: list[str] = []
    for fetch in (_download_spy_holdings_from_ssga, _download_spy_holdings_from_fmp):
        try:
            return fetch()
        except Exception as exc:
            errors.append(f"{fetch.__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _chunked(it: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(it), n):
        yield it[i : i + n]


def _download_yfinance_closes(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_size: int = 50,
) -> pd.DataFrame:
    """Download close prices; return long format: date,ticker,close."""
    import yfinance as yf

    # yfinance's end is exclusive; add 1 day to include end.
    y_end = (end + pd.Timedelta(days=1)).date()
    y_start = start.date()

    rows: list[pd.DataFrame] = []
    for chunk in _chunked(tickers, chunk_size):
        # Join tickers with space: yfinance accepts list too, but space string is stable.
        chunk_tickers = [t for t in chunk if t]
        if not chunk_tickers:
            continue

        data = yf.download(
            chunk_tickers,
            start=y_start,
            end=y_end,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
        if data is None or data.empty:
            continue

        # When downloading multiple tickers, columns are MultiIndex: (ticker, field)
        if isinstance(data.columns, pd.MultiIndex):
            # Expect Close under field level.
            for t in chunk_tickers:
                if (t, "Close") not in data.columns:
                    continue
                s = data[(t, "Close")].dropna()
                if s.empty:
                    continue
                df_s = s.reset_index()
                df_s.columns = ["date", "close"]
                df_s["ticker"] = t
                rows.append(df_s)
        else:
            # Single ticker case.
            # If Close column exists, use it.
            if "Close" not in data.columns:
                continue
            s = data["Close"].dropna()
            if s.empty:
                continue
            df_s = s.reset_index()
            df_s.columns = ["date", "close"]
            df_s["ticker"] = chunk_tickers[0]
            rows.append(df_s)

    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "close"])

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out["ticker"] = out["ticker"].astype(str)
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "close"])
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def _ensure_price_cache(
    tickers: list[str],
    required_end: pd.Timestamp,
    required_start: pd.Timestamp,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Ensure the global rolling price cache covers required range.

    Performance goal:
    - On first run, download `required_start..required_end` for all tickers.
    - On subsequent runs, only download the missing tail (and/or missing head)
      by appending to the existing cached long-format CSV.
    """
    SPX_PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if SPX_PRICE_CACHE.exists() and not force_refresh:
        cache = pd.read_csv(SPX_PRICE_CACHE)
        if cache.empty:
            SPX_PRICE_CACHE.unlink(missing_ok=True)  # type: ignore[attr-defined]
        else:
            cache["date"] = pd.to_datetime(cache["date"])
            cache["ticker"] = cache["ticker"].astype(str)
            # Coverage check
            if cache["date"].min() <= required_start and cache["date"].max() >= required_end:
                return cache

            # Partial coverage: download only missing segments.
            downloads: list[pd.DataFrame] = []
            cached_min = cache["date"].min()
            cached_max = cache["date"].max()

            # Missing head (rare, only if lookback window changes drastically)
            if cached_min > required_start:
                new_start = required_start
                new_end = cached_min - pd.Timedelta(days=1)
                if new_end >= new_start:
                    downloads.append(_download_yfinance_closes(tickers, new_start, new_end))

            # Missing tail (daily use-case)
            if cached_max < required_end:
                new_start = cached_max + pd.Timedelta(days=1)
                new_end = required_end
                if new_end >= new_start:
                    downloads.append(_download_yfinance_closes(tickers, new_start, new_end))

            if downloads:
                new_data = pd.concat([cache] + downloads, ignore_index=True)
                new_data["date"] = pd.to_datetime(new_data["date"])
                new_data["ticker"] = new_data["ticker"].astype(str)
                new_data["close"] = pd.to_numeric(new_data["close"], errors="coerce")
                new_data = new_data.dropna(subset=["date", "ticker", "close"])
                new_data = new_data.drop_duplicates(subset=["date", "ticker"], keep="last")
                new_data = new_data.sort_values(["ticker", "date"]).reset_index(drop=True)
                new_data.to_csv(SPX_PRICE_CACHE, index=False)
                return new_data

            # Cache exists but doesn't cover and we failed to download.
            return cache

    # Missing cache (or force refresh): download full required range.
    downloaded = _download_yfinance_closes(tickers, required_start, required_end)
    if downloaded.empty:
        return downloaded
    downloaded.to_csv(SPX_PRICE_CACHE, index=False)
    return downloaded


def _spy_window_return_pct(
    spy_df: pd.DataFrame, end_date: pd.Timestamp, lag_trading_days: int
) -> float | None:
    """SPY total return % over lag_trading_days (same calendar alignment as constituents)."""
    g = spy_df.copy()
    g["date"] = pd.to_datetime(g["date"])
    price_col = "close" if "close" in g.columns else "value"
    g = g[g["date"] <= end_date].sort_values("date")
    if len(g) < lag_trading_days + 1:
        return None
    close_t = float(g[price_col].iloc[-1])
    close_prev = float(g[price_col].iloc[-(lag_trading_days + 1)])
    return (close_t / close_prev - 1.0) * 100.0


def _compute_contrib_from_prices(
    weights: pd.DataFrame,
    prices_long: pd.DataFrame,
    *,
    end_date: pd.Timestamp,
    window_specs: dict[str, int],
    top_n: int,
    spy_df: pd.DataFrame | None = None,
) -> dict:
    """Compute top positive/negative contributions."""
    # weights: ticker, weight (0-1)
    w = weights.set_index("ticker")["weight"].to_dict()
    close_by_ticker = {
        t: g.sort_values("date") for t, g in prices_long.groupby("ticker")
    }

    windows_out: dict[str, dict] = {}
    for win_name, lag_days in window_specs.items():
        # lag_days is number of trading days back; 1d => lag 1 trading day prior.
        recs: list[dict] = []
        for ticker, weight in w.items():
            g = close_by_ticker.get(ticker)
            if g is None or g.empty:
                continue
            g = g[g["date"] <= end_date]
            if g.empty:
                continue

            # We need end plus `lag_days` prior bars => length >= lag_days+1
            if len(g) < lag_days + 1:
                continue
            g = g.reset_index(drop=True)
            close_t = float(g["close"].iloc[-1])
            close_prev = float(g["close"].iloc[-(lag_days + 1)])

            # Percent returns.
            ret_pct = (close_t / close_prev - 1.0) * 100.0
            contrib_pct = weight * ret_pct

            recs.append(
                {
                    "ticker": ticker,
                    "weight_pct": round(weight * 100.0, 4),
                    "return_pct": round(ret_pct, 4),
                    "contribution_pct": round(contrib_pct, 4),
                }
            )

        if not recs:
            windows_out[win_name] = {
                "top_positive": [],
                "top_negative": [],
                "note": "No tickers with enough price history for this window.",
            }
            continue

        recs_sorted = sorted(recs, key=lambda r: r["contribution_pct"])
        top_negative = recs_sorted[:top_n]
        top_positive = list(reversed(recs_sorted[-top_n:]))

        explained = float(sum(r["contribution_pct"] for r in recs))
        spy_return = (
            _spy_window_return_pct(spy_df, end_date, lag_days)
            if spy_df is not None
            else explained
        )
        residual = (
            round(float(spy_return) - explained, 4)
            if spy_return is not None
            else None
        )
        windows_out[win_name] = {
            "top_positive": top_positive,
            "top_negative": top_negative,
            "spy_return_pct": round(float(spy_return), 4) if spy_return is not None else None,
            "explained_pct": round(explained, 4),
            "residual_pct": residual,
            "constituents_used": len(recs),
        }

    return windows_out


def spx_contrib_to_factor_panels(spx_contrib: dict) -> list[dict]:
    """Convert top stock contributors into factor-attribution panel rows."""
    if not spx_contrib.get("available"):
        return []

    panel_specs = (
        ("1d", "spx_contrib_1d", "E. Top S&P 500 holdings (1 day)", "Last trading day"),
        (
            "21d",
            "spx_contrib_21d",
            "F. Top S&P 500 holdings (~21 trading days)",
            "Last ~21 trading days",
        ),
    )
    panels: list[dict] = []
    for win_key, panel_id, title, window_label in panel_specs:
        w = spx_contrib.get("windows", {}).get(win_key, {})
        spy_ret = w.get("spy_return_pct")
        explained = w.get("explained_pct")
        residual = w.get("residual_pct")

        movers = list(w.get("top_negative", [])) + list(w.get("top_positive", []))
        factors = [_contrib_row_to_factor(r, spy_ret) for r in movers]
        factors.sort(key=lambda row: row["contribution_pct"], reverse=True)

        if not factors:
            panels.append(
                {
                    "available": False,
                    "id": panel_id,
                    "title": title,
                    "message": w.get("note", "No constituent data for this window."),
                }
            )
            continue

        panels.append(
            {
                "available": True,
                "id": panel_id,
                "title": title,
                "model_label": "Holdings-weight decomposition (weight × stock return)",
                "window_label": window_label,
                "decomposition": True,
                "weight_beta": True,
                "regression_days": None,
                "spy_return_pct": spy_ret,
                "spy_excess_return_pct": spy_ret,
                "explained_pct": explained,
                "alpha_contribution_pct": None,
                "unexplained_pct": residual,
                "r_squared": None,
                "coverage_pct": (
                    round(100.0 * explained / spy_ret, 1)
                    if spy_ret not in (None, 0)
                    else None
                ),
                "constituents_used": w.get("constituents_used"),
                "factors": factors,
                "as_of": spx_contrib.get("as_of"),
                "note": spx_contrib.get("note"),
            }
        )
    return panels


def _contrib_row_to_factor(row: dict, spy_return_pct: float | None) -> dict:
    contrib = float(row["contribution_pct"])
    share = (
        round(100.0 * contrib / float(spy_return_pct), 1)
        if spy_return_pct not in (None, 0)
        else 0.0
    )
    return {
        "code": row["ticker"],
        "name": row["ticker"],
        "factor_return_pct": row["return_pct"],
        "beta": round(float(row["weight_pct"]) / 100.0, 4),
        "weight_pct": row["weight_pct"],
        "contribution_pct": contrib,
        "share_pct": share,
    }


def compute_spx_contributions(
    data: dict[str, pd.DataFrame],
    *,
    end_date: pd.Timestamp | None = None,
    top_n: int = 5,
    force_refresh: bool = False,
    allow_download: bool = True,
) -> dict:
    """Return top contributors based on weight * return (approx)."""
    if "spy" not in data or data["spy"].empty:
        return {
            "available": False,
            "message": "SPY required for end_date alignment",
            "windows": {"1d": {"top_positive": [], "top_negative": []}, "21d": {"top_positive": [], "top_negative": []}},
        }

    if end_date is None:
        end_date = pd.to_datetime(data["spy"]["date"].iloc[-1])
    end_date = _as_date(end_date)
    required_start = end_date - pd.Timedelta(days=PRICE_LOOKBACK_CAL_DAYS)

    # 1) weights (try holdings first)
    weight_source = "equal_weights"
    weights: pd.DataFrame
    try:
        if SPY_HOLDINGS_CACHE.exists() and not force_refresh:
            weights = get_spy_holdings_weights(force_refresh=False)
            weight_source = "spy_holdings_cached"
        else:
            weights = get_spy_holdings_weights(force_refresh=force_refresh)
            weight_source = "spy_holdings_ssga"
    except Exception as exc:
        # Fallback to equal weights using cached tickers.
        try:
            tickers = get_spx500_tickers(force_refresh=False)["ticker"].tolist()
        except Exception:
            return {
                "available": False,
                "message": f"Cannot get holdings weights and cannot load SPX tickers: {exc}",
                "windows": {
                    "1d": {"top_positive": [], "top_negative": []},
                    "21d": {"top_positive": [], "top_negative": []},
                },
            }
        if not tickers:
            return {
                "available": False,
                "message": "No SPX tickers available for fallback equal-weight model.",
                "windows": {
                    "1d": {"top_positive": [], "top_negative": []},
                    "21d": {"top_positive": [], "top_negative": []},
                },
            }
        w = 1.0 / len(tickers)
        weights = pd.DataFrame({"ticker": tickers, "weight": [w] * len(tickers)})
        weight_source = "spx_equal_weight_fallback"

    tickers = sorted(set(weights["ticker"].astype(str).tolist()))

    # 2) prices (yfinance, cached)
    if not allow_download and not SPX_PRICE_CACHE.exists():
        return {
            "available": False,
            "message": "Price cache not found and downloads disabled.",
            "windows": {
                "1d": {"top_positive": [], "top_negative": []},
                "21d": {"top_positive": [], "top_negative": []},
            },
        }

    if allow_download:
        prices_long = _ensure_price_cache(
            tickers,
            required_end=end_date,
            required_start=required_start,
            force_refresh=force_refresh,
        )
    else:
        # Load cache only if present.
        prices_long = pd.DataFrame(columns=["date", "ticker", "close"])
        if SPX_PRICE_CACHE.exists():
            prices_long = pd.read_csv(SPX_PRICE_CACHE)
            if not prices_long.empty:
                prices_long["date"] = pd.to_datetime(prices_long["date"])

    if prices_long.empty:
        return {
            "available": False,
            "message": "No constituent prices available (cache miss).",
            "windows": {
                "1d": {"top_positive": [], "top_negative": []},
                "21d": {"top_positive": [], "top_negative": []},
            },
        }

    windows = _compute_contrib_from_prices(
        weights,
        prices_long,
        end_date=end_date,
        window_specs={"1d": 1, "21d": 21},
        top_n=top_n,
        spy_df=data["spy"],
    )

    available = any(w["top_positive"] or w["top_negative"] for w in windows.values())
    note = (
        "Contributions are approximate: weight × constituent return using "
        "SPY (or equal-weight) as a proxy. Not identical to index calculation "
        "due to dividends, rebalancing, and trading-day alignment."
    )
    if weight_source.startswith("spy_holdings"):
        note = (
            "Using SPY holdings weights as proxy. Contributions are weight × return "
            "(approx). " + note
        )
    else:
        note = (
            "SPY holdings weights unavailable; using equal-weight S&P 500 tickers as fallback. "
            + note
        )

    return {
        "available": available,
        "as_of": str(end_date.date()),
        "weight_source": weight_source,
        "note": note,
        "windows": windows,
    }


def main() -> None:
    from src.collect import load_all_from_cache

    data = load_all_from_cache()
    result = compute_spx_contributions(data, allow_download=True, force_refresh=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

