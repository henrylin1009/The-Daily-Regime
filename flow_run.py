#!/usr/bin/env python3
"""Standalone cross-border capital flow monitor."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from jinja2 import Template

from src.config import OUTPUT_DIR
from src.macro_quant_engine import (
    MacroQuantEngine,
    fetch_macro_panel,
    generate_quant_payload,
)

FLOW_COUNTRIES = ["US", "Japan", "China", "Taiwan", "Europe", "UK"]

FX_SPECS = {
    "Japan": {"code": "JPY", "ticker": "USDJPY=X", "convention": "USD/JPY", "invert": False},
    "Europe": {"code": "EUR", "ticker": "EURUSD=X", "convention": "EUR/USD", "invert": True},
    "UK": {"code": "GBP", "ticker": "GBPUSD=X", "convention": "GBP/USD", "invert": True},
    "Taiwan": {"code": "TWD", "ticker": "USDTWD=X", "convention": "USD/TWD", "invert": False},
    "China": {"code": "CNH", "ticker": "USDCNH=X", "convention": "USD/CNH", "invert": False},
}

BOND_SPECS = {
    "US": {"ticker": "^TNX", "fallback_fred": "DGS10"},
    "Japan": {"tickers": ["^JP10YT=RR"], "fallback_fred": ["IRLTLT01JPM156N"]},
    "Europe": {"tickers": ["^DE10YT=RR"], "fallback_fred": ["IRLTLT01DEM156N"]},
    "UK": {"tickers": ["^GB10YT=RR"], "fallback_fred": ["IRLTLT01GBM156N"]},
    "China": {"tickers": ["CN10YT=RR", "CNGGB10Y=R"], "fallback_fred": ["DLTNTNO10Y"]},
    "Taiwan": {"tickers": ["TWD10Y=RR", "TAIEX10Y"], "fallback_fred": ["IRLTLT01TWM156N", "IRLTLT01TWN156N"]},
}

EQUITY_SPECS = {
    "US": "SPY",
    "Japan": "EWJ",
    "Europe": "EZU",
    "UK": "EWU",
    "Taiwan": "EWT",
    "China": "MCHI",
}

VALIDATION_FX = {
    "AUD/JPY": "AUDJPY=X",
    "NZD/JPY": "NZDJPY=X",
    "USD/CHF": "USDCHF=X",
}

VALIDATION_ETF = {
    "USD Bull ETF (UUP)": "UUP",
    "JPY ETF (FXY)": "FXY",
    "EM Bond ETF (EMB)": "EMB",
    "HY Credit ETF (HYG)": "HYG",
}

FLOW_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Global Flow Monitor — {{ report_date }}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-primary: #ffffff; --bg-secondary: #f8f9fb; --bg-card: #ffffff;
      --border: #e8ecf0; --text-primary: #0f1923; --text-muted: #8a96a3;
      --text-number: #374151; --green: #16a34a; --yellow: #b45309;
      --red: #dc2626; --blue: #2563eb; --accent: #4f46e5;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--text-primary); background: #f8f9fb; padding: 12px 14px; }
    .mono { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--text-number); }
    .container { max-width: 1400px; margin: 0 auto; padding: 0 0 2rem; }
    .sticky-header { position: sticky; top: 0; z-index: 100; background: #fff; border-bottom: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-radius: 12px; margin-bottom: 12px; }
    .header-inner { display: grid; grid-template-columns: 1.2fr 2fr; gap: 1rem; align-items: center; max-width: 1400px; margin: 0 auto; padding: 0.85rem 1rem; }
    .brand-title { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
    .brand-sub { font-size: 0.78rem; color: var(--text-muted); margin-top: 0.18rem; }
    .traffic-wrap { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; }
    .pill { border: 1px solid #e2e8f0; border-radius: 999px; padding: 0.38rem 0.68rem; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.4rem; background: #f1f5f9; white-space: nowrap; }
    .pill.GREEN { background: #f0fdf4; border-color: #bbf7d0; color: #15803d; }
    .pill.YELLOW { background: #fffbeb; border-color: #fde68a; color: #92400e; }
    .pill.RED { background: #fef2f2; border-color: #fecaca; color: #b91c1c; }
    .dot { font-size: 0.8rem; line-height: 1; }
    .section { background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; margin-top: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    .section-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-bottom: 0.8rem; font-weight: 600; }
    table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    thead th { color: var(--text-muted); text-transform: uppercase; font-size: 0.67rem; letter-spacing: 0.06em; text-align: left; padding: 0.5rem 0.52rem; border-bottom: 1px solid var(--border); }
    tbody td { padding: 0.48rem 0.52rem; border-bottom: 1px solid #eef1f4; vertical-align: middle; }
    tbody tr:nth-child(odd) { background: #fafbfc; }
    .pos { color: var(--green); } .neg { color: var(--red); } .muted { color: var(--text-muted); }
    .bar-wrap { width: 220px; height: 10px; border: 1px solid #e2e8f0; border-radius: 999px; background: #f1f5f9; position: relative; overflow: hidden; }
    .bar-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #cbd5e1; }
    .bar-fill { position: absolute; top: 0; bottom: 0; }
    .bar-fill.blue { background: linear-gradient(90deg, #93c5fd, #2563eb); }
    .bar-fill.red { background: linear-gradient(90deg, #fca5a5, #dc2626); }
    .signal-pill { border-radius: 999px; font-size: 0.68rem; font-weight: 700; border: 1px solid transparent; padding: 0.14rem 0.38rem; }
    .signal-pill.GREEN { color: #15803d; border-color: #bbf7d0; background: #f0fdf4; }
    .signal-pill.YELLOW { color: #92400e; border-color: #fde68a; background: #fffbeb; }
    .signal-pill.RED { color: #b91c1c; border-color: #fecaca; background: #fef2f2; }
    .fx-strong { color: #15803d; font-weight: 600; } .fx-weak { color: #b91c1c; font-weight: 600; }
    .bars { display: flex; gap: 3px; align-items: flex-end; height: 92px; }
    .bar { width: 13px; border-radius: 2px 2px 0 0; }
    .bar.p { background: #22c55e; } .bar.n { background: #ef4444; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
    .overview-split { display: grid; grid-template-columns: 1fr 2.5fr; gap: 1.5rem; align-items: start; }
    @media (max-width: 768px) { .overview-split { grid-template-columns: 1fr; gap: 1rem; } }
    .narrative { line-height: 1.65; color: #1f2937; }
    .footer { border-top: 1px solid var(--border); margin-top: 1rem; padding-top: 0.65rem; font-size: 0.7rem; color: var(--text-muted); display: flex; justify-content: space-between; }
  </style>
</head>
<body>
  <header class="sticky-header">
    <div class="header-inner">
      <div>
        <div class="brand-title">Global Flow Monitor</div>
        <div class="brand-sub mono">{{ report_date }} | snapshot {{ snapshot_date }}</div>
      </div>
    </div>
  </header>
  <main class="container">
    <section class="section">
      <div class="section-title">Overview</div>
      <div class="overview-split">
        <!-- Left: Status Pills -->
        <div class="traffic-wrap" style="display: flex; flex-direction: column; gap: 0.5rem; justify-content: flex-start;">
          {% for c in countries %}
            <span class="pill {{ signals[c].signal }}" style="justify-content: flex-start; width: 100%;"><span class="dot">●</span>{{ c }}: {{ signals[c].signal }}</span>
          {% endfor %}
        </div>
        <!-- Right: LLM narrative -->
        <div>
          <p class="narrative" style="margin-top: 0; line-height: 1.6; font-size: 0.88rem; color: #1f2937;">{{ surfaced_summary }}</p>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-title">What Rose to Surface Today (Auto-ranked)</div>
      <table style="margin-top:0.5rem;">
        <thead><tr><th>Indicator</th><th>Latest</th><th>Change</th><th>Z-Score</th><th>Attention</th><th>Signal</th></tr></thead>
        <tbody>
          {% for r in surfaced_rows %}
          <tr>
            <td>{{ r.name }}</td>
            <td class="mono">{{ r.latest }}</td>
            <td class="mono {% if r.chg_val is not none and r.chg_val > 0 %}pos{% elif r.chg_val is not none and r.chg_val < 0 %}neg{% else %}muted{% endif %}">{{ r.change }}</td>
            <td class="mono">{{ r.z }}</td>
            <td class="mono">{{ r.attention }}</td>
            <td><span class="signal-pill {{ r.signal }}">{{ r.signal }}</span></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-title">Section 1: Yield Spread Matrix</div>
      <table>
        <thead><tr><th>Country</th><th>10Y Yield</th><th>US - Local (bp)</th><th>Spread Visual</th></tr></thead>
        <tbody>
          {% for r in spread_rows %}
          <tr>
            <td>{{ r.country }}</td>
            <td class="mono">{{ r.yield_pct }}</td>
            <td class="mono {% if r.spread_bp is none %}muted{% elif r.spread_bp >= 0 %}pos{% else %}neg{% endif %}">{{ r.spread_bp_fmt }}</td>
            <td>
              {% if r.spread_bp is not none %}
                <div class="bar-wrap">
                  <span class="bar-mid"></span>
                  <span class="bar-fill {{ r.bar_color }}" style="{{ r.bar_style }}"></span>
                </div>
              {% else %}
                <span class="muted">—</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-title">Section 2: FX Momentum</div>
      <table>
        <thead><tr><th>Currency</th><th>Current Rate</th><th>1m (local vs USD)</th><th>3m (local vs USD)</th><th>Signal</th></tr></thead>
        <tbody>
          {% for r in fx_rows %}
          <tr>
            <td>{{ r.label }}</td>
            <td class="mono">{{ r.rate }}</td>
            <td class="{{ r.m1_css }}">{{ r.m1_arrow }} {{ r.m1_fmt }}</td>
            <td class="{{ r.m3_css }}">{{ r.m3_arrow }} {{ r.m3_fmt }}</td>
            <td><span class="signal-pill {{ r.signal }}">{{ r.signal }}</span></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <section class="section">
      <div class="section-title">Section 3: Full Validation Universe</div>
      <table>
        <thead><tr><th>Indicator</th><th>Latest</th><th>1m Change</th><th>Z-Score(1y)</th><th>Signal</th></tr></thead>
        <tbody>
          {% for r in validation_rows %}
          <tr>
            <td>{{ r.name }}</td>
            <td class="mono">{{ r.latest }}</td>
            <td class="mono {% if r.chg_1m_val is not none and r.chg_1m_val > 0 %}pos{% elif r.chg_1m_val is not none and r.chg_1m_val < 0 %}neg{% else %}muted{% endif %}">{{ r.chg_1m }}</td>
            <td class="mono {% if r.z_val is not none and r.z_val >= 1.0 %}neg{% elif r.z_val is not none and r.z_val <= -1.0 %}pos{% else %}muted{% endif %}">{{ r.z }}</td>
            <td><span class="signal-pill {{ r.signal }}">{{ r.signal }}</span></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </section>

    <footer class="footer">
      <span>All signals are directional heuristics; not investment advice.</span>
      <span class="mono">Generated: {{ report_date }}</span>
    </footer>
  </main>
</body>
</html>
"""

FALLBACK_LLM = {
    "flow_narrative": (
        "Global flows remain mixed with USD still attracting rate-sensitive capital while selective equity risk appetite rotates regionally. "
        "Japan and China indicators are watched closely for policy-defense versus private outflow pressures. "
        "Watch relative yield spreads and FX momentum for confirmation of whether this is a temporary pause or a broader regime shift."
    ),
    "japan_carry_interpretation": "Japan carry dynamics look mixed, with FX and holdings data not yet showing a decisive unwind or rebuild.",
    "china_capital_interpretation": "China capital signals are mixed, with CNH and reserves suggesting intermittent pressure rather than a one-way outflow.",
}


def clean_json_response(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


def _yf_close_series(ticker: str, period: str = "1y") -> pd.Series:
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No yfinance data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = pd.Series(close).dropna()
    s.index = pd.to_datetime(s.index)
    return s


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
                return pd.Series(df2["value"].values, index=pd.to_datetime(df2["date"])).dropna()
            break
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    if date_col not in df.columns or series_id not in df.columns:
        raise RuntimeError(f"Unexpected FRED format for {series_id}")
    s = pd.to_numeric(df[series_id], errors="coerce")
    out = pd.Series(s.values, index=pd.to_datetime(df[date_col])).dropna()
    return out


def _latest_and_pct(series: pd.Series, periods: int) -> tuple[float | None, float | None]:
    s = series.dropna()
    if s.empty:
        return None, None
    latest = float(s.iloc[-1])
    if len(s) <= periods:
        return latest, None
    prev = float(s.iloc[-1 - periods])
    if prev == 0:
        return latest, None
    return latest, (latest / prev - 1.0) * 100.0


def _zscore_1y(series: pd.Series) -> float | None:
    s = pd.Series(series).dropna().tail(252)
    if len(s) < 20:
        return None
    std = float(s.std(ddof=0))
    if std == 0:
        return None
    return float((s.iloc[-1] - s.mean()) / std)


def _fetch_bond_yields() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for country, spec in BOND_SPECS.items():
        source = "unavailable"
        s = pd.Series(dtype=float)
        yf_tickers = [spec.get("ticker")] if spec.get("ticker") else spec.get("tickers", [])
        fred_series_ids = [spec.get("fallback_fred")] if isinstance(spec.get("fallback_fred"), str) else spec.get("fallback_fred", [])

        for ticker in yf_tickers:
            try:
                s = _yf_close_series(ticker, period="1y")
                source = f"yfinance:{ticker}"
                break
            except Exception:
                continue

        if s.empty:
            for series_id in fred_series_ids:
                try:
                    s = _fred_series(series_id)
                    source = f"fred:{series_id}"
                    break
                except Exception:
                    continue

        latest, chg_1m = _latest_and_pct(s, periods=21)
        out[country] = {
            "series": s,
            "latest": latest,
            "chg_1m_pct": chg_1m,
            "source": source,
            "as_of": s.index[-1].date().isoformat() if not s.empty else None,
        }
    return out


def _fx_local_changes() -> dict[str, dict]:
    out: dict[str, dict] = {}
    dxy = _yf_close_series("DX-Y.NYB", period="1y")
    dxy_latest, dxy_1m = _latest_and_pct(dxy, periods=21)
    _, dxy_3m = _latest_and_pct(dxy, periods=63)
    out["DXY"] = {"latest": dxy_latest, "local_1m": dxy_1m, "local_3m": dxy_3m, "label": "DXY"}

    for country, spec in FX_SPECS.items():
        candidate_series: list[tuple[str, pd.Series]] = []
        tickers = [spec["ticker"]]
        if country == "China":
            tickers.append("CNY=X")
        for ticker in tickers:
            try:
                candidate_series.append((f"yfinance:{ticker}", _yf_close_series(ticker, period="1y")))
            except Exception:
                continue
        if country == "China":
            try:
                candidate_series.append(("fred:DEXCHUS", _fred_series("DEXCHUS")))
            except Exception:
                pass
        if not candidate_series:
            candidate_series = [("unavailable", pd.Series(dtype=float))]

        # Pick source with best momentum availability (prefer having both 1m and 3m).
        best_source, best_series = candidate_series[0]
        best_score = -1
        best_pair_1m, best_pair_3m = None, None
        for src, cand_s in candidate_series:
            _, cand_1m = _latest_and_pct(cand_s, periods=21)
            _, cand_3m = _latest_and_pct(cand_s, periods=63)
            score = int(cand_1m is not None) + int(cand_3m is not None)
            if score > best_score:
                best_source, best_series = src, cand_s
                best_score = score
                best_pair_1m, best_pair_3m = cand_1m, cand_3m

        s = best_series
        source = best_source
        usd_base_pair = str(spec.get("convention", "")).startswith("USD/")
        pair_series = s
        # Normalize ambiguous fallback feeds to the declared pair convention.
        if usd_base_pair and not s.empty and float(s.iloc[-1]) < 1.0:
            # Convert USD/CNY-like inverse quote back to USD/local.
            pair_series = (1.0 / s.replace(0, pd.NA)).dropna()

        latest, pair_1m = _latest_and_pct(pair_series, periods=21)
        _, pair_3m = _latest_and_pct(pair_series, periods=63)
        if spec["invert"]:
            # EURUSD/GBPUSD up means local strength vs USD.
            local_1m = pair_1m
            local_3m = pair_3m
        else:
            # USDJPY/USDCNH/USDTWD up means local weakness vs USD.
            local_1m = None if pair_1m is None else -pair_1m
            local_3m = None if pair_3m is None else -pair_3m
        out[country] = {
            "label": f"{spec['code']} ({spec['convention']})",
            "ticker": spec["ticker"],
            "source": source,
            "pair_1m": pair_1m,
            "pair_3m": pair_3m,
            "usd_base": usd_base_pair,
            "rate": latest,
            "local_1m": local_1m,
            "local_3m": local_3m,
            "series": pair_series,
        }
    return out


def _fetch_twse_last_20_days() -> pd.DataFrame:
    rows: list[dict] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    today = date.today()

    for i in range(60):
        d = today - timedelta(days=i)
        ds = d.strftime("%Y%m%d")
        url = f"https://www.twse.com.tw/rwd/en/fund/BFI82U?response=json&dayDate={ds}&type=day"
        try:
            resp = session.get(url, timeout=12)
            if resp.status_code != 200:
                continue
            payload = resp.json()
            data = payload.get("data") or []
            if not data:
                continue
            hints = str(payload.get("hints") or "")
            foreign_row = None
            for r in data:
                if not r:
                    continue
                label = str(r[0]).lower()
                if "foreign investors" in label and "excluded" in label:
                    foreign_row = r
                    break
            if foreign_row is None:
                continue

            # BFI82U hints are (NT$); convert to TWD billions.
            diff = _to_num(foreign_row[3]) if len(foreign_row) > 3 else 0.0
            unit_divisor = 1e9
            if "thousand" in hints.lower():
                unit_divisor = 1e6
            net = diff / unit_divisor
            rows.append({"date": d.isoformat(), "net_bn_twd": net})
            if len(rows) >= 20:
                break
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["date", "net_bn_twd"])
    df = pd.DataFrame(rows).sort_values("date")
    return df.tail(20)


def _to_num(value) -> float:
    if value is None:
        return 0.0
    s = str(value).replace(",", "").replace("+", "").strip()
    if not s or s in {"--", "N/A", "nan"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _parse_japan_tic() -> dict:
    url = "https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/mfh.txt"
    text = requests.get(url, timeout=20).text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    japan_line = next((ln for ln in lines if ln.strip().startswith("Japan")), "")
    if not japan_line:
        raise RuntimeError("Japan line not found in TIC file")
    nums = [float(x) for x in re.findall(r"\d+\.\d", japan_line)]
    if len(nums) < 2:
        raise RuntimeError("Not enough TIC datapoints")
    latest = nums[0]
    prev = nums[1]
    return {"latest_bn": latest, "mom_change_bn": latest - prev}


def _fetch_china_reserves() -> dict:
    try:
        s = _fred_series("CHFRESNS")
        # CHFRESNS is in millions USD; convert to billions for display and MoM.
        s = s / 1000.0
    except Exception:
        s = _fred_series("TRESEGCNM052N")
        # Fallback series can also be in millions; normalize to billions.
        if not s.empty and float(s.dropna().iloc[-1]) > 100000:
            s = s / 1000.0
    latest = float(s.iloc[-1]) if len(s) else None
    mom = (latest - float(s.iloc[-2])) if len(s) > 1 else None
    return {"series": s, "latest_bn": latest, "mom_change_bn": mom}


def _fetch_equity_returns() -> dict[str, dict]:
    out = {}
    for country, ticker in EQUITY_SPECS.items():
        s = _yf_close_series(ticker, period="1y")
        _, r1m = _latest_and_pct(s, periods=21)
        out[country] = {"ticker": ticker, "ret_1m": r1m}
    return out


def _fetch_fed_rate_expectations() -> dict:
    """Fetch Fed funds futures term structure and compute implied cuts/hikes.

    Uses ZQ{M}{YY}.CBT monthly contracts (30-day Fed funds futures on CME via yfinance).
    Implied rate = 100 - price. Negative change_vs_now_bp = cuts priced in.
    """
    from datetime import datetime

    # CME month codes: F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
    _MONTH_CODES = "FGHJKMNQUVXZ"
    today = datetime.today()
    contracts = []
    y, m = today.year, today.month
    for _ in range(12):
        m += 1
        if m > 12:
            m = 1
            y += 1
        code = f"ZQ{_MONTH_CODES[m - 1]}{str(y)[-2:]}.CBT"
        contracts.append((code, y, m))

    term_structure: list[dict] = []
    for ticker, yr, mo in contracts:
        try:
            t = yf.Ticker(ticker)
            price = t.fast_info.last_price
            if price is None or price != price:  # NaN check
                continue
            implied_rate = round(100.0 - float(price), 4)
            term_structure.append({
                "ticker": ticker,
                "year": yr,
                "month": mo,
                "price": round(float(price), 4),
                "implied_rate_pct": implied_rate,
            })
            if len(term_structure) >= 8:
                break
        except Exception:
            continue

    if not term_structure:
        return {"error": "no contracts fetched", "term_structure": [], "summary": "unavailable"}

    try:
        from pathlib import Path as _Path
        import pandas as _pd
        _raw = _Path(__file__).parent / "data" / "raw" / "fed_funds_rate.csv"
        _ff = _pd.read_csv(_raw, index_col=0, parse_dates=True)["value"].dropna()
        current_rate = float(_ff.iloc[-1])
    except Exception:
        current_rate = term_structure[0]["implied_rate_pct"]

    for row in term_structure:
        row["change_vs_now_bp"] = round((row["implied_rate_pct"] - current_rate) * 100, 1)

    def _describe(bp: float) -> str:
        cuts = round(-bp / 25)
        hikes = round(bp / 25)
        if bp < -10:
            return f"{cuts} cut{'s' if cuts != 1 else ''} priced in ({-bp:.0f}bp)"
        elif bp > 10:
            return f"{hikes} hike{'s' if hikes != 1 else ''} priced in (+{bp:.0f}bp)"
        else:
            return "on hold (no move priced in)"

    last = term_structure[-1]
    eoy_candidates = [r for r in term_structure if r["year"] == today.year and r["month"] == 12]
    eoy = eoy_candidates[0] if eoy_candidates else last

    summary = (
        f"Current rate: {current_rate:.2f}%. "
        f"Year-end ({eoy['year']}-{eoy['month']:02d}): {_describe(eoy['change_vs_now_bp'])}. "
        f"Horizon ({last['year']}-{last['month']:02d}): {_describe(last['change_vs_now_bp'])}."
    )

    return {
        "current_fed_funds_pct": current_rate,
        "term_structure": term_structure,
        "summary": summary,
        "as_of": today.date().isoformat(),
    }


def _compute_fed_balance_sheet_signal() -> dict:
    """Load WALCL from cache, compute 13w/52w change and QT/QE trend signal."""
    from pathlib import Path as _Path
    import pandas as _pd
    import math as _math

    _raw = _Path(__file__).parent / "data" / "raw" / "walcl.csv"
    if not _raw.exists():
        return {"error": "walcl.csv not found — run collect with --force-refresh", "summary": "unavailable"}

    df = _pd.read_csv(_raw, index_col=0, parse_dates=True)["value"].dropna().sort_index()
    if len(df) < 10:
        return {"error": "insufficient WALCL data", "summary": "unavailable"}

    latest_bn = round(df.iloc[-1] / 1_000, 1)        # convert $M → $B
    as_of = df.index[-1].date().isoformat()

    # 13-week (~1 quarter) and 52-week change
    def _pct_chg(periods: int) -> float | None:
        if len(df) <= periods:
            return None
        old = df.iloc[-periods]
        if old == 0:
            return None
        return round((df.iloc[-1] - old) / old * 100, 2)

    chg_13w = _pct_chg(13)
    chg_52w = _pct_chg(52)

    # Trend: rolling 4-week slope (positive = expanding/QE, negative = shrinking/QT)
    recent = df.iloc[-8:]
    slope_sign = None
    if len(recent) >= 4:
        x = list(range(len(recent)))
        xm = sum(x) / len(x)
        ym = recent.mean()
        cov = sum((xi - xm) * (yi - ym) for xi, yi in zip(x, recent))
        slope_sign = "expanding" if cov > 0 else "shrinking"

    # Signal: label the liquidity regime
    if chg_13w is not None:
        if chg_13w > 1.0:
            signal = "QE / expanding (liquidity injection)"
        elif chg_13w < -1.0:
            signal = "QT / shrinking (liquidity drain)"
        else:
            signal = "roughly stable"
    else:
        signal = "unknown"

    summary = (
        f"Fed balance sheet: ${latest_bn:.1f}B (as of {as_of}). "
        f"13w change: {'+' if (chg_13w or 0) >= 0 else ''}{chg_13w:.1f}%. "
        f"52w change: {'+' if (chg_52w or 0) >= 0 else ''}{chg_52w:.1f}%. "
        f"Trend: {slope_sign or 'n/a'} → {signal}."
    )

    return {
        "total_assets_bn": latest_bn,
        "as_of": as_of,
        "chg_13w_pct": chg_13w,
        "chg_52w_pct": chg_52w,
        "trend": slope_sign,
        "signal": signal,
        "summary": summary,
    }


def _fetch_validation_panel() -> list[dict]:
    rows: list[dict] = []
    # Carry crosses and USD funding proxy
    for name, ticker in VALIDATION_FX.items():
        try:
            s = _yf_close_series(ticker, period="1y")
            latest, chg_1m = _latest_and_pct(s, periods=21)
            z = _zscore_1y(s)
            # For AUD/JPY and NZD/JPY, higher generally indicates carry-friendly.
            if name in {"AUD/JPY", "NZD/JPY"}:
                signal = "GREEN" if (chg_1m or 0.0) > 0 else ("RED" if (chg_1m or 0.0) < 0 else "YELLOW")
            else:
                # USD/CHF up tends to align with USD strength / defensive tone.
                signal = "RED" if (chg_1m or 0.0) > 0 else ("GREEN" if (chg_1m or 0.0) < 0 else "YELLOW")
            rows.append(
                {
                    "name": name,
                    "latest": "—" if latest is None else f"{latest:.4f}",
                    "chg_1m": "—" if chg_1m is None else f"{chg_1m:+.2f}%",
                    "chg_1m_val": chg_1m,
                    "z": "—" if z is None else f"{z:+.2f}",
                    "z_val": z,
                    "signal": signal,
                }
            )
        except Exception:
            rows.append({"name": name, "latest": "—", "chg_1m": "—", "chg_1m_val": None, "z": "—", "z_val": None, "signal": "YELLOW"})

    # ETF risk/funding proxies
    for name, ticker in VALIDATION_ETF.items():
        try:
            s = _yf_close_series(ticker, period="1y")
            latest, chg_1m = _latest_and_pct(s, periods=21)
            z = _zscore_1y(s)
            if ticker in {"EMB", "HYG"}:
                signal = "GREEN" if (chg_1m or 0.0) > 0 else ("RED" if (chg_1m or 0.0) < 0 else "YELLOW")
            elif ticker == "UUP":
                signal = "RED" if (chg_1m or 0.0) > 0 else ("GREEN" if (chg_1m or 0.0) < 0 else "YELLOW")
            else:
                signal = "GREEN" if (chg_1m or 0.0) > 0 else ("RED" if (chg_1m or 0.0) < 0 else "YELLOW")
            rows.append(
                {
                    "name": name,
                    "latest": "—" if latest is None else f"{latest:.2f}",
                    "chg_1m": "—" if chg_1m is None else f"{chg_1m:+.2f}%",
                    "chg_1m_val": chg_1m,
                    "z": "—" if z is None else f"{z:+.2f}",
                    "z_val": z,
                    "signal": signal,
                }
            )
        except Exception:
            rows.append({"name": name, "latest": "—", "chg_1m": "—", "chg_1m_val": None, "z": "—", "z_val": None, "signal": "YELLOW"})

    # MOVE and front-end policy differential proxy (US-EU)
    move = None
    try:
        move = _yf_close_series("^MOVE", period="1y")
    except Exception:
        try:
            move = _fred_series("MOVEINDEX")
        except Exception:
            move = None
    if move is not None and not move.empty:
        latest, chg_1m = _latest_and_pct(move, periods=21)
        z = _zscore_1y(move)
        rows.append(
            {
                "name": "MOVE (UST vol)",
                "latest": "—" if latest is None else f"{latest:.2f}",
                "chg_1m": "—" if chg_1m is None else f"{chg_1m:+.2f}%",
                "chg_1m_val": chg_1m,
                "z": "—" if z is None else f"{z:+.2f}",
                "z_val": z,
                "signal": "RED" if (chg_1m or 0.0) > 0 else ("GREEN" if (chg_1m or 0.0) < 0 else "YELLOW"),
            }
        )
    else:
        rows.append({"name": "MOVE (UST vol)", "latest": "—", "chg_1m": "—", "chg_1m_val": None, "z": "—", "z_val": None, "signal": "YELLOW"})

    try:
        fed = _fred_series("FEDFUNDS")
        ecb = _fred_series("ECBDFR")
        common = pd.concat([fed, ecb], axis=1, join="inner").dropna()
        spread = common.iloc[:, 0] - common.iloc[:, 1]
        latest, chg_1m = _latest_and_pct(spread, periods=1)
        z = _zscore_1y(spread)
        rows.append(
            {
                "name": "US-EU policy spread",
                "latest": "—" if latest is None else f"{latest:.2f}%",
                "chg_1m": "—" if chg_1m is None else f"{chg_1m:+.2f}%",
                "chg_1m_val": chg_1m,
                "z": "—" if z is None else f"{z:+.2f}",
                "z_val": z,
                "signal": "RED" if (chg_1m or 0.0) > 0 else ("GREEN" if (chg_1m or 0.0) < 0 else "YELLOW"),
            }
        )
    except Exception:
        rows.append({"name": "US-EU policy spread", "latest": "—", "chg_1m": "—", "chg_1m_val": None, "z": "—", "z_val": None, "signal": "YELLOW"})

    return rows


def _build_surfaced_rows(
    bonds: dict,
    fx: dict,
    tw_flow: pd.DataFrame,
    japan_tic: dict,
    china_res: dict,
    validation_rows: list[dict],
) -> list[dict]:
    candidates: list[dict] = []

    def add(name: str, latest: str, chg_val: float | None, chg_text: str, z_val: float | None):
        abs_z = abs(z_val) if z_val is not None else 0.0
        abs_chg = abs(chg_val) if chg_val is not None else 0.0
        attention = abs_z * 0.7 + min(abs_chg / 2.0, 3.0) * 0.3
        if attention >= 1.2:
            signal = "RED"
        elif attention <= 0.5:
            signal = "GREEN"
        else:
            signal = "YELLOW"
        candidates.append(
            {
                "name": name,
                "latest": latest,
                "chg_val": chg_val,
                "change": chg_text,
                "z": "—" if z_val is None else f"{z_val:+.2f}",
                "attention_val": attention,
                "attention": f"{attention:.2f}",
                "signal": signal,
            }
        )

    # Taiwan flow momentum
    tw_vals = tw_flow["net_bn_twd"] if not tw_flow.empty else pd.Series(dtype=float)
    tw_5d = float(tw_vals.tail(5).sum()) if not tw_vals.empty else None
    tw_20d = float(tw_vals.tail(20).sum()) if not tw_vals.empty else None
    tw_z = None
    if len(tw_vals) >= 20:
        std = float(tw_vals.std(ddof=0))
        tw_z = None if std == 0 else float((tw_vals.iloc[-1] - tw_vals.mean()) / std)
    add("Taiwan foreign flow (5d)", "—" if tw_5d is None else f"{tw_5d:.2f} bn", tw_5d, "—" if tw_5d is None else f"{tw_5d:+.2f} bn", tw_z)
    add("Taiwan foreign flow (20d)", "—" if tw_20d is None else f"{tw_20d:.2f} bn", tw_20d, "—" if tw_20d is None else f"{tw_20d:+.2f} bn", tw_z)

    # Japan carry proxies
    jpy_s = fx["Japan"]["series"]
    jpy_z = _zscore_1y(jpy_s) if isinstance(jpy_s, pd.Series) else None
    add("USD/JPY", "—" if fx["Japan"]["rate"] is None else f"{fx['Japan']['rate']:.3f}", fx["Japan"]["pair_1m"], "—" if fx["Japan"]["pair_1m"] is None else f"{fx['Japan']['pair_1m']:+.2f}%", jpy_z)
    jp_spread = None
    jp_spread_series = None
    if not bonds["US"]["series"].empty and not bonds["Japan"]["series"].empty:
        common = pd.concat([bonds["US"]["series"], bonds["Japan"]["series"]], axis=1, join="inner").dropna()
        if not common.empty:
            jp_spread_series = common.iloc[:, 0] - common.iloc[:, 1]
            jp_spread = float(jp_spread_series.iloc[-1] * 100.0)
    add("US-JP 10Y spread", "—" if jp_spread is None else f"{jp_spread:+.1f} bp", None, "—", _zscore_1y(jp_spread_series) if jp_spread_series is not None else None)
    add("Japan UST holdings MoM", "—" if japan_tic.get("latest_bn") is None else f"{japan_tic['latest_bn']:.1f} bn", japan_tic.get("mom_change_bn"), "—" if japan_tic.get("mom_change_bn") is None else f"{japan_tic['mom_change_bn']:+.1f} bn", None)

    # China capital proxies
    cnh_s = fx["China"]["series"]
    cnh_z = _zscore_1y(cnh_s) if isinstance(cnh_s, pd.Series) else None
    add("USD/CNH", "—" if fx["China"]["rate"] is None else f"{fx['China']['rate']:.4f}", fx["China"]["pair_1m"], "—" if fx["China"]["pair_1m"] is None else f"{fx['China']['pair_1m']:+.2f}%", cnh_z)
    add("China FX reserves MoM", "—" if china_res.get("latest_bn") is None else f"{china_res['latest_bn']:,.1f} bn", china_res.get("mom_change_bn"), "—" if china_res.get("mom_change_bn") is None else f"{china_res['mom_change_bn']:+,.1f} bn", None)

    # Include validation panel universe
    for r in validation_rows:
        add(r["name"], r["latest"], r.get("chg_1m_val"), r["chg_1m"], r.get("z_val"))

    top = sorted(candidates, key=lambda x: x["attention_val"], reverse=True)[:8]
    return top


def _build_signals(
    bonds: dict,
    fx: dict,
    equities: dict,
    tw_flow: pd.DataFrame,
    japan_tic: dict,
    china_res: dict,
) -> dict[str, dict]:
    us_series = bonds["US"]["series"]
    signals: dict[str, dict] = {}

    for country in FLOW_COUNTRIES:
        points: list[int] = []
        reasons: list[str] = []
        is_us = country == "US"

        if not is_us:
            f = fx[country]
            if f["local_1m"] is not None:
                points.append(1 if f["local_1m"] > 0 else 0)
                reasons.append("FX 1m")
            if f["local_3m"] is not None:
                points.append(1 if f["local_3m"] > 0 else 0)
                reasons.append("FX 3m")

            local_series = bonds[country]["series"].dropna()
            common = pd.concat([us_series, local_series], axis=1, join="inner").dropna()
            if len(common) > 21:
                spread_now = float(common.iloc[-1, 0] - common.iloc[-1, 1])
                spread_then = float(common.iloc[-22, 0] - common.iloc[-22, 1])
                # Narrowing spread favors local market.
                points.append(1 if spread_now < spread_then else 0)
                reasons.append("Spread narrowing")
        else:
            dxy = fx["DXY"]
            if dxy["local_1m"] is not None:
                points.append(1 if dxy["local_1m"] > 0 else 0)
                reasons.append("USD broad 1m")
            if dxy["local_3m"] is not None:
                points.append(1 if dxy["local_3m"] > 0 else 0)
                reasons.append("USD broad 3m")

            spread_values = []
            spread_values_old = []
            for peer in ["Japan", "China", "Taiwan", "Europe", "UK"]:
                peer_series = bonds[peer]["series"].dropna()
                common = pd.concat([us_series, peer_series], axis=1, join="inner").dropna()
                if len(common) > 21:
                    spread_values.append(float(common.iloc[-1, 0] - common.iloc[-1, 1]))
                    spread_values_old.append(float(common.iloc[-22, 0] - common.iloc[-22, 1]))
            if spread_values and spread_values_old:
                points.append(1 if (sum(spread_values) / len(spread_values)) > (sum(spread_values_old) / len(spread_values_old)) else 0)
                reasons.append("Avg spread widening")

        eq = equities[country]["ret_1m"]
        if eq is not None:
            points.append(1 if eq > 0 else 0)
            reasons.append("Equity 1m")

        if country == "Taiwan" and not tw_flow.empty:
            flow_5d = float(tw_flow["net_bn_twd"].tail(5).sum())
            points.append(1 if flow_5d > 0 else 0)
            reasons.append("TW foreign 5d")
        if country == "Japan":
            j = japan_tic.get("mom_change_bn")
            if j is not None:
                points.append(1 if j > 0 else 0)
                reasons.append("Japan TIC MoM")
        if country == "China":
            c = china_res.get("mom_change_bn")
            if c is not None:
                points.append(1 if c > 0 else 0)
                reasons.append("FX reserves MoM")

        score = (sum(points) / len(points)) if points else 0.5
        signal = "GREEN" if score > 0.60 else ("RED" if score < 0.40 else "YELLOW")
        signals[country] = {"score": score, "signal": signal, "points": points, "reasons": reasons}
    return signals


def _build_quant_flow(flow_payload: dict, signals: dict) -> dict:
    """Build structured flow summary from quantitative data — no LLM."""
    fx1m = flow_payload.get("fx_1m_local_vs_usd") or {}
    spreads = flow_payload.get("bond_spreads_bp") or {}

    # Flow narrative from signals
    green = [c for c, s in signals.items() if isinstance(s, dict) and s.get("signal") == "GREEN"]
    red = [c for c, s in signals.items() if isinstance(s, dict) and s.get("signal") == "RED"]
    flow_narrative = (
        f"Capital flow signals are positive for: {', '.join(green) or 'none'}. "
        f"Caution signals for: {', '.join(red) or 'none'}. "
        "Full thematic interpretation available in Strategic Synthesis (L3)."
    )

    # Japan carry
    jpy_fx = fx1m.get("Japan")
    if jpy_fx is not None:
        jpy_dir = "strengthening" if float(jpy_fx) > 0 else "weakening"
        japan_carry = f"The yen is {jpy_dir} against the dollar on a one-month basis, which affects carry trade positioning."
    else:
        japan_carry = "Yen carry trade data not available."

    # China capital
    cnh_fx = fx1m.get("China")
    if cnh_fx is not None:
        cnh_dir = "appreciating" if float(cnh_fx) > 0 else "depreciating"
        china_capital = f"The yuan is {cnh_dir} versus the dollar, suggesting capital {('inflow' if float(cnh_fx) > 0 else 'outflow')} pressure."
    else:
        china_capital = "China capital flow data not available."

    return {
        "flow_narrative": flow_narrative,
        "japan_carry_interpretation": japan_carry,
        "china_capital_interpretation": china_capital,
    }


def _render_html(
    output_path: Path,
    report_date: date,
    snapshot_date: str,
    signals: dict[str, dict],
    spread_rows: list[dict],
    fx_rows: list[dict],
    tw_flow: pd.DataFrame,
    llm_bundle: dict,
    llm_failure_reason: str | None,
    data_coverage: dict,
    surfaced_rows: list[dict],
    validation_rows: list[dict],
) -> None:
    tw_vals = tw_flow["net_bn_twd"].tolist() if not tw_flow.empty else [0.0] * 20
    max_abs = max([abs(x) for x in tw_vals] + [1.0])
    tw_bar_scale = 82.0 / max_abs
    tw_5d = float(sum(tw_vals[-5:])) if tw_vals else 0.0
    tw_20d = float(sum(tw_vals)) if tw_vals else 0.0
    surfaced_summary = llm_bundle.get("flow_narrative", "")

    html = Template(FLOW_HTML_TEMPLATE).render(
        report_date=report_date.isoformat(),
        snapshot_date=snapshot_date,
        countries=FLOW_COUNTRIES,
        signals=signals,
        spread_rows=spread_rows,
        fx_rows=fx_rows,
        tw_bars=tw_vals[-20:],
        tw_bar_scale=tw_bar_scale,
        tw_5d=tw_5d,
        tw_20d=tw_20d,
        surfaced_summary=surfaced_summary,
        surfaced_rows=surfaced_rows,
        validation_rows=validation_rows,
        data_coverage=data_coverage,
    )
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Global cross-border flow monitor")
    parser.add_argument("--date", type=str, default=None, help="Report date YYYY-MM-DD (default: today)")
    parser.add_argument("--output", type=str, default=None, help="Output html path")
    args = parser.parse_args()

    report_date = date.fromisoformat(args.date) if args.date else date.today()
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "flow_brief.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Stage 1: Fetching bond yields...")
    bonds = _fetch_bond_yields()
    snapshot_candidates = [v["as_of"] for v in bonds.values() if v.get("as_of")]
    snapshot_date = max(snapshot_candidates) if snapshot_candidates else report_date.isoformat()

    print("Stage 2: Fetching FX...")
    fx = _fx_local_changes()

    print("Stage 3: Fetching Taiwan foreign flow...")
    tw_flow = _fetch_twse_last_20_days()

    print("Stage 4: Fetching Japan TIC and China reserves...")
    japan_tic = _parse_japan_tic()
    china_res = _fetch_china_reserves()

    print("Stage 5: Fetching equities...")
    equities = _fetch_equity_returns()
    validation_rows = _fetch_validation_panel()

    print("Stage 6: Building flow signals...")
    signals = _build_signals(bonds, fx, equities, tw_flow, japan_tic, china_res)

    print("Stage 6b: Macro quant engine (Layer 1 & 2)...")
    quant_payload: dict = {}
    quant_panel_meta: dict = {"rows": 0, "columns": [], "last_date": None}
    try:
        macro_df = fetch_macro_panel(period="3y")
        enriched = MacroQuantEngine(macro_df).enrich()
        quant_json_str = generate_quant_payload(enriched)
        quant_payload = json.loads(quant_json_str)
        quant_panel_meta = {
            "rows": len(enriched),
            "columns": list(enriched.columns),
            "last_date": enriched.index[-1].isoformat() if len(enriched) else None,
        }
        quant_path = OUTPUT_DIR / f"quant_engine_{report_date.isoformat()}.json"
        quant_path.write_text(quant_json_str, encoding="utf-8")
        print(f"  Quant engine JSON: {quant_path}", file=sys.stderr)
    except Exception as exc:
        print(f"  Macro quant engine failed (non-fatal): {exc}", file=sys.stderr)
        quant_payload = {"error": str(exc)}

    print("Stage 6c: Fed rate expectations curve...")
    try:
        quant_payload["fed_rate_expectations"] = _fetch_fed_rate_expectations()
        print(f"  Rate expectations: {quant_payload['fed_rate_expectations'].get('summary')}", file=sys.stderr)
    except Exception as exc:
        print(f"  Rate expectations failed (non-fatal): {exc}", file=sys.stderr)

    print("Stage 6d: Fed balance sheet (WALCL) liquidity signal...")
    try:
        quant_payload["fed_balance_sheet"] = _compute_fed_balance_sheet_signal()
        print(f"  Balance sheet: {quant_payload['fed_balance_sheet'].get('summary')}", file=sys.stderr)
    except Exception as exc:
        print(f"  Balance sheet signal failed (non-fatal): {exc}", file=sys.stderr)

    # Table prep: spread rows
    spread_vals = []
    for country in FLOW_COUNTRIES:
        y = bonds[country]["latest"]
        spread_bp = None
        if country == "US":
            spread_bp = 0.0
        elif bonds["US"]["latest"] is not None and y is not None:
            spread_bp = (bonds["US"]["latest"] - y) * 100.0
        if spread_bp is not None:
            spread_vals.append(abs(spread_bp))
    max_spread = max(spread_vals + [1.0])
    spread_rows = []
    for country in FLOW_COUNTRIES:
        y = bonds[country]["latest"]
        spread_bp = None
        if country == "US":
            spread_bp = 0.0
        elif bonds["US"]["latest"] is not None and y is not None:
            spread_bp = (bonds["US"]["latest"] - y) * 100.0
        if spread_bp is None:
            style = ""
            color = "blue"
        else:
            w = min(abs(spread_bp) / max_spread * 50.0, 50.0)
            if spread_bp >= 0:
                style = f"left:50%; width:{w:.1f}%;"
                color = "blue"
            else:
                style = f"left:{50.0 - w:.1f}%; width:{w:.1f}%;"
                color = "red"
        spread_rows.append(
            {
                "country": country,
                "yield_pct": "—" if y is None else f"{y:.2f}%",
                "spread_bp": spread_bp,
                "spread_bp_fmt": "—" if spread_bp is None else f"{spread_bp:+.1f}",
                "bar_style": style,
                "bar_color": color,
            }
        )

    fx_rows = []
    for country in ["Japan", "Europe", "UK", "Taiwan", "China"]:
        row = fx[country]
        m1 = row["local_1m"]
        m3 = row["local_3m"]
        m1v = m1 if m1 is not None else 0.0
        m3v = m3 if m3 is not None else 0.0
        fx_point_score = (int(m1v > 0) + int(m3v > 0)) / 2.0
        fx_signal = "GREEN" if fx_point_score > 0.6 else ("RED" if fx_point_score < 0.4 else "YELLOW")
        if row.get("usd_base"):
            pair_1m = row.get("pair_1m")
            pair_3m = row.get("pair_3m")
            # Display local-currency change vs USD (negated from USD/base pair change)
            disp_1m = None if pair_1m is None else -pair_1m
            disp_3m = None if pair_3m is None else -pair_3m
            m1_arrow = "—" if disp_1m is None else ("↑" if disp_1m > 0 else "↓")
            m3_arrow = "—" if disp_3m is None else ("↑" if disp_3m > 0 else "↓")
            m1_css = "muted" if disp_1m is None else ("fx-strong" if disp_1m > 0 else "fx-weak")
            m3_css = "muted" if disp_3m is None else ("fx-strong" if disp_3m > 0 else "fx-weak")
            m1_fmt = "—" if disp_1m is None else f"{disp_1m:+.2f}%"
            m3_fmt = "—" if disp_3m is None else f"{disp_3m:+.2f}%"
        else:
            m1_arrow = "—" if m1 is None else ("↑" if m1 >= 0 else "↓")
            m3_arrow = "—" if m3 is None else ("↑" if m3 >= 0 else "↓")
            m1_css = "muted" if m1 is None else ("fx-strong" if m1 >= 0 else "fx-weak")
            m3_css = "muted" if m3 is None else ("fx-strong" if m3 >= 0 else "fx-weak")
            m1_fmt = "—" if m1 is None else f"{m1:+.2f}%"
            m3_fmt = "—" if m3 is None else f"{m3:+.2f}%"

        fx_rows.append(
            {
                "label": row["label"],
                "rate": f"{row['rate']:.4f}" if row["rate"] is not None else "—",
                "m1": m1,
                "m3": m3,
                "m1_arrow": m1_arrow,
                "m3_arrow": m3_arrow,
                "m1_fmt": m1_fmt,
                "m3_fmt": m3_fmt,
                "m1_css": m1_css,
                "m3_css": m3_css,
                "signal": fx_signal,
            }
        )

    flow_payload = {
        "signals": {k: {"signal": v["signal"], "score": round(v["score"], 3)} for k, v in signals.items()},
        "bond_spreads_bp": {r["country"]: (round(r["spread_bp"], 1) if r["spread_bp"] is not None else None) for r in spread_rows if r["country"] != "US"},
        "fx_1m_local_vs_usd": {c: round(fx[c]["local_1m"], 2) if fx[c]["local_1m"] is not None else None for c in ["Japan", "Europe", "UK", "Taiwan", "China"]},
        "fx_3m_local_vs_usd": {c: round(fx[c]["local_3m"], 2) if fx[c]["local_3m"] is not None else None for c in ["Japan", "Europe", "UK", "Taiwan", "China"]},
        "equity_1m_returns": {c: round(equities[c]["ret_1m"], 2) if equities[c]["ret_1m"] is not None else None for c in ["US", "Japan", "Europe", "UK", "Taiwan", "China"]},
        "taiwan_foreign_flow": {
            "days": len(tw_flow),
            "cum_5d_bn_twd": round(float(tw_flow["net_bn_twd"].tail(5).sum()), 2) if not tw_flow.empty else 0.0,
            "cum_20d_bn_twd": round(float(tw_flow["net_bn_twd"].tail(20).sum()), 2) if not tw_flow.empty else 0.0,
        },
        "japan_tic_mom_change_bn": round(japan_tic["mom_change_bn"], 2),
        "china_fx_reserves_mom_change_bn": round(china_res["mom_change_bn"], 2) if china_res["mom_change_bn"] is not None else None,
    }

    core_checks = [
        ("US 10Y yield", bonds["US"]["latest"] is not None),
        ("Japan 10Y yield", bonds["Japan"]["latest"] is not None),
        ("Europe 10Y yield", bonds["Europe"]["latest"] is not None),
        ("UK 10Y yield", bonds["UK"]["latest"] is not None),
        ("China FX 1m", fx["China"]["local_1m"] is not None),
        ("China FX 3m", fx["China"]["local_3m"] is not None),
        ("Taiwan 5d flow", not tw_flow.empty),
        ("Japan TIC MoM", japan_tic.get("mom_change_bn") is not None),
        ("China reserves MoM", china_res.get("mom_change_bn") is not None),
    ]
    total_core = len(core_checks)
    available_core = sum(1 for _, ok in core_checks if ok)
    data_coverage = {
        "available_core": available_core,
        "total_core": total_core,
        "ratio_pct": round((available_core / total_core) * 100.0, 1) if total_core else 0.0,
        "missing_critical_fields": [name for name, ok in core_checks if not ok][:3],
    }

    llm_bundle = _build_quant_flow(flow_payload, signals)
    llm_failure_reason: str | None = None

    surfaced_rows = _build_surfaced_rows(bonds, fx, tw_flow, japan_tic, china_res, validation_rows)

    print("Stage 7: Rendering HTML...")
    _render_html(
        output_path=output_path,
        report_date=report_date,
        snapshot_date=snapshot_date,
        signals=signals,
        spread_rows=spread_rows,
        fx_rows=fx_rows,
        tw_flow=tw_flow,
        llm_bundle=llm_bundle,
        llm_failure_reason=llm_failure_reason,
        data_coverage=data_coverage,
        surfaced_rows=surfaced_rows,
        validation_rows=validation_rows,
    )

    flow_data_path = OUTPUT_DIR / f"flow_data_{report_date.isoformat()}.json"
    flow_data_payload = {
        "date": report_date.isoformat(),
        "snapshot_date": snapshot_date,
        "module_c": {
            "taiwan_flow": {
                "days": len(tw_flow),
                "cum_5d_bn_twd": round(float(tw_flow["net_bn_twd"].tail(5).sum()), 2) if not tw_flow.empty else 0.0,
                "cum_20d_bn_twd": round(float(tw_flow["net_bn_twd"].sum()), 2) if not tw_flow.empty else 0.0,
            },
            "china_fx_reserves": {
                "latest_bn": china_res.get("latest_bn"),
                "mom_change_bn": china_res.get("mom_change_bn"),
            },
        },
        "signals": signals,
        "flow_payload": flow_payload,
        "validation_panel": validation_rows,
        "llm": {
            "flow_narrative": llm_bundle.get("flow_narrative"),
            "japan_carry_interpretation": llm_bundle.get("japan_carry_interpretation"),
            "china_capital_interpretation": llm_bundle.get("china_capital_interpretation"),
        },
        "llm_failure_reason": llm_failure_reason,
        "data_coverage": data_coverage,
        "surfaced_rows": surfaced_rows,
        "event_of_day": surfaced_rows[0] if surfaced_rows else None,
        "quant_engine": {
            "payload": quant_payload,
            "panel_meta": quant_panel_meta,
        },
    }
    flow_data_path.write_text(json.dumps(flow_data_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Output: {output_path}")
    print(f"Data JSON: {flow_data_path}")


if __name__ == "__main__":
    main()
