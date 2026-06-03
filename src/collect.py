"""Stage 1: Pull macro indicators via OpenBB and cache to CSV."""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from src.config import RAW_DIR, configure_openbb, fred_api_key_valid
from src.futures_adjust import panama_back_adjust

DEFAULT_START = "1993-01-01"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={symbol}"

def _latest_value(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or "value" not in df.columns:
        return None
    s = pd.to_numeric(df["value"], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _sanity_check_price(
    *,
    name: str,
    df: pd.DataFrame,
    min_ok: float,
    max_ok: float,
    hint: str,
    hard_fail: bool = True,
) -> None:
    v = _latest_value(df)
    if v is None:
        return
    if not (min_ok <= v <= max_ok):
        msg = f"{name} price {v:.4g} out of expected range [{min_ok}, {max_ok}]. {hint}"
        if hard_fail:
            raise RuntimeError(msg)
        print(f"[WARN] {msg}", file=sys.stderr)


def _normalize_df(df: pd.DataFrame, value_col: str | None = None) -> pd.DataFrame:
    """Normalize OpenBB response to date, value columns."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "value"])

    out = df.copy()
    if "date" not in out.columns:
        out = out.reset_index()
    date_col = "date" if "date" in out.columns else out.columns[0]
    out["date"] = pd.to_datetime(out[date_col]).dt.normalize()

    if value_col and value_col in out.columns:
        out["value"] = pd.to_numeric(out[value_col], errors="coerce")
    elif "value" in out.columns:
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
    elif "close" in out.columns:
        out["value"] = pd.to_numeric(out["close"], errors="coerce")
    else:
        numeric_cols = out.select_dtypes(include="number").columns
        out["value"] = pd.to_numeric(out[numeric_cols[0]], errors="coerce")

    return out[["date", "value"]].dropna(subset=["value"]).sort_values("date")


def _fetch_fred_csv(symbol: str, start_date: str) -> pd.DataFrame:
    """FRED public CSV export — no API key required."""
    url = FRED_CSV_URL.format(symbol=symbol)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw = pd.read_csv(io.StringIO(resp.text))
    date_col, val_col = raw.columns[0], raw.columns[1]
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col]),
            "value": pd.to_numeric(raw[val_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["value"]).sort_values("date")
    out = out[out["date"] >= pd.Timestamp(start_date)]
    return out.reset_index(drop=True)


def _fetch_fred_api(symbol: str, start_date: str) -> pd.DataFrame:
    """FRED official JSON API with api_key — avoids fredgraph.csv rate limits."""
    import os
    api_key = os.getenv("FRED_API_KEY") or os.getenv("fred_api_key")
    if not api_key:
        print("  _fetch_fred_api: FRED_API_KEY not set, skipping", file=sys.stderr)
        raise RuntimeError("FRED_API_KEY not set")
    url = (
        f"https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={symbol}&api_key={api_key}&file_type=json"
        f"&observation_start={start_date}&vintage_dates="
    )
    resp = requests.get(url, timeout=60, headers={"Accept": "application/json"})
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    if not obs:
        raise RuntimeError(f"No observations returned for {symbol}")
    out = pd.DataFrame(obs)[["date", "value"]]
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


def _fetch_fred_openbb(
    symbol: str,
    start_date: str,
    transform: str | None = None,
) -> pd.DataFrame:
    from openbb import obb

    configure_openbb()
    kwargs: dict = {"symbol": symbol, "start_date": start_date, "provider": "fred"}
    if transform:
        kwargs["transform"] = transform
    result = obb.economy.fred_series(**kwargs)
    if not result.results:
        return pd.DataFrame(columns=["date", "value"])
    return _normalize_df(result.to_df())


def _fetch_fred(
    symbol: str,
    start_date: str,
    transform: str | None = None,
) -> pd.DataFrame:
    """FRED: official JSON API first (most reliable), then CSV fallback."""
    try:
        df = _fetch_fred_api(symbol, start_date)
        if not df.empty:
            return df
    except Exception:
        pass
    return _fetch_fred_csv(symbol, start_date)


def _fetch_fred_level_diff(symbol: str, start_date: str) -> pd.DataFrame:
    """Fetch FRED level series and return 1-period change."""
    df = _fetch_fred(symbol, start_date)
    if df.empty:
        return df
    df = df.sort_values("date")
    df["value"] = df["value"].diff()
    return df.dropna(subset=["value"])


def _fetch_fred_yoy(symbol: str, start_date: str) -> pd.DataFrame:
    """Fetch FRED series and compute 12-month YoY % change."""
    df = _fetch_fred(symbol, start_date)
    if df.empty:
        return df
    df = df.sort_values("date").set_index("date")
    yoy = (df["value"].pct_change(12) * 100).dropna()
    out = yoy.reset_index()
    out.columns = ["date", "value"]
    return out


def _fetch_fred_gdp_yoy(symbol: str, start_date: str) -> pd.DataFrame:
    """GDP real level -> YoY % growth (quarterly)."""
    df = _fetch_fred(symbol, start_date)
    if df.empty:
        return df
    df = df.sort_values("date").set_index("date")
    yoy = (df["value"].pct_change(4) * 100).dropna()
    out = yoy.reset_index()
    out.columns = ["date", "value"]
    return out


def _fetch_equity(symbol: str, start_date: str) -> pd.DataFrame:
    from openbb import obb

    configure_openbb()
    result = obb.equity.price.historical(symbol=symbol, start_date=start_date)
    return _normalize_df(result.to_df(), value_col="close")


def _fetch_etf(symbol: str, start_date: str) -> pd.DataFrame:
    from openbb import obb

    configure_openbb()
    result = obb.etf.historical(symbol=symbol, start_date=start_date)
    return _normalize_df(result.to_df(), value_col="close")


def _fetch_etf_yfinance(symbol: str, start_date: str) -> pd.DataFrame:
    """Fetch ETF historical closes from yfinance."""
    import yfinance as yf

    df = yf.download(
        symbol,
        start=start_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "value"])
    # yfinance may return either single-level columns or a MultiIndex (Price, Ticker).
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", symbol) in df.columns:
            close = df[("Close", symbol)]
        elif (symbol, "Close") in df.columns:
            close = df[(symbol, "Close")]
        else:
            # Best effort: find any "Close" field.
            close_cols = [c for c in df.columns if str(c[0]).lower() == "close" or str(c[1]).lower() == "close"]
            if not close_cols:
                return pd.DataFrame(columns=["date", "value"])
            close = df[close_cols[0]]
    else:
        if "Close" not in df.columns:
            return pd.DataFrame(columns=["date", "value"])
        close = df["Close"]

    out = close.dropna().reset_index()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)


def _fetch_futures(symbol: str, start_date: str) -> pd.DataFrame:
    from openbb import obb

    configure_openbb()
    result = obb.derivatives.futures.historical(symbol=symbol, start_date=start_date)
    df = _normalize_df(result.to_df(), value_col="close")
    return panama_back_adjust(df)


def _fetch_dxy(start_date: str) -> pd.DataFrame:
    """Try multiple DXY symbols; return first that works."""
    symbols = ["DX-Y.NYB", "UUP", "^DXY"]
    for sym in symbols:
        try:
            df = _fetch_equity(sym, start_date)
            if not df.empty:
                return df
        except Exception:
            continue
    raise RuntimeError(f"Could not fetch DXY using symbols: {symbols}")


# Registry: name -> fetch function
INDICATOR_FETCHERS: dict[str, Callable[[str], pd.DataFrame]] = {
    # FRED — inflation (YoY %)
    "cpi_yoy": lambda s: _fetch_fred_yoy("CPIAUCSL", s),
    "core_cpi_yoy": lambda s: _fetch_fred_yoy("CPILFESL", s),
    "pce_yoy": lambda s: _fetch_fred_yoy("PCEPI", s),
    # FRED — employment
    "nfp_change": lambda s: _fetch_fred_level_diff("PAYEMS", s),
    "unemployment_rate": lambda s: _fetch_fred("UNRATE", s),
    # FRED — growth & rates
    "gdp_growth": lambda s: _fetch_fred_gdp_yoy("GDPC1", s),
    "indpro_growth": lambda s: _fetch_fred_yoy("INDPRO", s),
    "oecd_cli": lambda s: _fetch_fred("USALOLITOAASTSAM", s),
    "initial_claims": lambda s: _fetch_fred("ICSA", s),
    "capacity_utilization": lambda s: _fetch_fred("TCU", s),
    "spread_2y10y": lambda s: _fetch_fred("T10Y2Y", s),
    "credit_spread_hy": lambda s: _fetch_fred("BAA10Y", s),
    "credit_spread_hy_oas": lambda s: _fetch_fred("BAMLH0A0HYM2", s),
    "fed_funds_rate": lambda s: _fetch_fred("FEDFUNDS", s),
    # FRED — other central banks (optional regime features)
    "ecb_deposit_rate": lambda s: _fetch_fred("ECBDFR", s),
    "boj_policy_rate": lambda s: _fetch_fred("IRSTCI01JPM156N", s),
    # Market — daily
    "vix": lambda s: _fetch_equity("^VIX", s),
    "dxy": _fetch_dxy,
    # GLD is an ETF (share price). If this looks like spot gold / futures,
    # we'd rather fail early than propagate nonsense into percentiles/regimes.
    "gld": lambda s: _fetch_etf_yfinance("GLD", s),
    # Oil proxy: use DBO instead of USO / WTI-spot.
    # Reason: USO has complex roll/split history and data vendors often disagree on adjustments.
    # DBO tends to have a cleaner long history for regime analytics.
    #
    # NOTE: We keep the key name `uso` for backward compatibility across the pipeline/output.
    "uso": lambda s: _fetch_etf_yfinance("DBO", s),
    "xlk": lambda s: _fetch_etf("XLK", s),
    "xlf": lambda s: _fetch_etf("XLF", s),
    "xlu": lambda s: _fetch_etf("XLU", s),
    "xle": lambda s: _fetch_etf("XLE", s),
    "xlv": lambda s: _fetch_etf("XLV", s),
    "xli": lambda s: _fetch_etf("XLI", s),
    "zq_futures": lambda s: _fetch_futures("ZQ=F", s),
    # Benchmark (internal — forward returns)
    "spy": lambda s: _fetch_etf("SPY", s),
    # EM aggregate signals
    "emb": lambda s: _fetch_etf_yfinance("EMB", s),
    "eem": lambda s: _fetch_etf_yfinance("EEM", s),
    "copper": lambda s: _fetch_etf_yfinance("HG=F", s),
    # International macro — for global divergence matching (Part B)
    "eu_cpi_yoy": lambda s: _fetch_fred_yoy("CP0000EZ19M086NEST", s),
    "cn_cpi_yoy": lambda s: _fetch_fred_yoy("CHNCPIALLMINMEI", s),
    "kr_policy_rate": lambda s: _fetch_fred("IRSTCI01KRM156N", s),
    "cn_fx_reserves": lambda s: _fetch_fred("TRESEGCNM052N", s),
    "fx_usdjpy": lambda s: _fetch_fred("DEXJPUS", s),
    "fx_eurusd": lambda s: _fetch_fred("DEXUSEU", s),
    "fx_usdcny": lambda s: _fetch_fred("DEXCHUS", s),
}


def _cache_path(name: str) -> Path:
    return RAW_DIR / f"{name}.csv"


def _load_cache(name: str) -> pd.DataFrame:
    path = _cache_path(name)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _save_cache(name: str, df: pd.DataFrame) -> None:
    path = _cache_path(name)
    df.to_csv(path, index=False)


def _update_meta(name: str) -> None:
    meta_path = RAW_DIR / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    meta[name] = datetime.utcnow().isoformat() + "Z"
    meta_path.write_text(json.dumps(meta, indent=2))


def collect_one(
    name: str,
    start_date: str = DEFAULT_START,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch or load a single indicator."""
    if not force_refresh and _cache_path(name).exists():
        return _load_cache(name)

    fetcher = INDICATOR_FETCHERS.get(name)
    if fetcher is None:
        raise ValueError(f"Unknown indicator: {name}")

    df = fetcher(start_date)
    if df.empty:
        raise RuntimeError(f"No data returned for {name}")

    # Spot-check critical prices to catch symbol/unit mismatches early.
    if name == "gld":
        gld_latest = _latest_value(df)
        assert gld_latest is not None, "GLD latest is missing"
        # NOTE: GLD has traded >400 in 2026; keep sanity bounds wide enough to avoid false positives.
        assert 200 < gld_latest < 600, f"GLD {gld_latest} looks wrong"
    if name == "uso":
        dbo_latest = _latest_value(df)
        assert dbo_latest is not None, "DBO latest is missing"
        assert 10 < dbo_latest < 50, f"DBO {dbo_latest} looks wrong"

    _save_cache(name, df)
    _update_meta(name)
    return df


def collect_all(
    start_date: str = DEFAULT_START,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Returns dict of DataFrames keyed by indicator name.
    Caches to data/raw/*.csv. Uses cache if exists and not force_refresh.
    """
    if not force_refresh and all(_cache_path(n).exists() for n in INDICATOR_FETCHERS):
        return load_all_from_cache()

    data: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for name in INDICATOR_FETCHERS:
        try:
            data[name] = collect_one(name, start_date, force_refresh)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            cache = _cache_path(name)
            if cache.exists():
                try:
                    data[name] = pd.read_csv(cache, index_col=0, parse_dates=True)
                    print(f"  collect: {name} fetch failed, using cache ({exc})", file=sys.stderr)
                except Exception:
                    print(f"  collect: {name} failed and no cache ({exc})", file=sys.stderr)
            else:
                print(f"  collect: {name} failed and no cache ({exc})", file=sys.stderr)

    return data


def load_all_from_cache() -> dict[str, pd.DataFrame]:
    """Load all cached indicators without fetching."""
    data: dict[str, pd.DataFrame] = {}
    for name in INDICATOR_FETCHERS:
        path = _cache_path(name)
        if path.exists():
            data[name] = _load_cache(name)
    return data


# Series that may start after DEFAULT_START (ETF inception, FRED availability)
LATE_START_OK: dict[str, str] = {
    "cpi_yoy": "1994-01-01",
    "core_cpi_yoy": "1994-01-01",
    "pce_yoy": "1994-01-01",
    "gdp_growth": "1994-01-01",
    "indpro_growth": "1994-01-01",
    "oecd_cli": "1960-01-01",
    "initial_claims": "1967-01-01",
    "capacity_utilization": "1967-01-01",
    "emb": "2007-12-01",
    "eem": "2003-04-01",
    "credit_spread_hy_oas": "2023-06-01",
    "ecb_deposit_rate": "1999-01-01",
    "boj_policy_rate": "1994-01-01",
    "eu_cpi_yoy": "1997-01-01",
    "cn_cpi_yoy": "1994-01-01",
    "kr_policy_rate": "1992-01-01",
    "cn_fx_reserves": "1978-01-01",
    "fx_usdjpy": "1971-01-01",
    "fx_eurusd": "1999-01-01",
    "fx_usdcny": "1981-01-01",
    "zq_futures": "2000-09-01",
    "gld": "2004-11-01",
    # Key kept as 'uso' but fetcher is DBO (see INDICATOR_FETCHERS).
    "uso": "2007-01-01",
    "xlk": "1999-01-01",
    "xlf": "1999-01-01",
    "xlu": "1999-01-01",
    "xle": "1999-01-01",
    "xlv": "1999-01-01",
    "xli": "1999-01-01",
}


def validate_all_caches() -> list[str]:
    """Validate cached CSVs; return list of error messages (empty if OK)."""
    errors: list[str] = []
    min_default = pd.Timestamp(DEFAULT_START)

    for name in INDICATOR_FETCHERS:
        path = _cache_path(name)
        if not path.exists():
            errors.append(f"{name}: missing {path.name}")
            continue

        df = _load_cache(name)
        if df.empty:
            errors.append(f"{name}: empty file")
            continue

        if not df["date"].is_monotonic_increasing:
            errors.append(f"{name}: dates not monotonic")

        if df["value"].tail(5).isna().all():
            errors.append(f"{name}: last 5 rows all NaN")

        start = pd.Timestamp(df["date"].iloc[0])
        min_expected = pd.Timestamp(LATE_START_OK.get(name, DEFAULT_START))
        if start > min_expected + pd.DateOffset(months=3):
            errors.append(
                f"{name}: starts {start.date()} (expected on or before {min_expected.date()})"
            )

        if name == "spy" and len(df) < 500:
            errors.append(f"{name}: only {len(df)} rows (expected 500+)")

    return errors


def main() -> None:
    if "--validate" in sys.argv:
        errs = validate_all_caches()
        if errs:
            print("Validation FAILED:\n", file=sys.stderr)
            for e in errs:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)
        print(f"Validation OK: {len(INDICATOR_FETCHERS)} series")
        for name in sorted(INDICATOR_FETCHERS):
            df = _load_cache(name)
            print(f"  {name}: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()} ({len(df)} rows)")
        sys.exit(0)

    force = "--force-refresh" in sys.argv
    try:
        data = collect_all(force_refresh=force)
    except EnvironmentError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Collection error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Collected {len(data)} indicators:\n")
    for name, df in data.items():
        print(f"  {name}: {len(df)} rows, last={df['date'].iloc[-1].date()}, value={df['value'].iloc[-1]:.4f}")
        print(df.tail(3).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
