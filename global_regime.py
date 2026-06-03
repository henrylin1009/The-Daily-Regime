#!/usr/bin/env python3
"""Layer 1 structural macro regime monitor."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from jinja2 import Template

from flow_run import _fetch_china_reserves
from src.config import OUTPUT_DIR, PROJECT_ROOT
from src.macra_assets import macra_style_block
from src.macra_ui import build_l1_accordion_rows, global_divergence_from_quant

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "global_regime.json"
COT_CACHE_PATH = CACHE_DIR / "cot_financial_cache.parquet"
COT_CACHE_META_PATH = CACHE_DIR / "cot_financial_cache_meta.json"
COT_CACHE_CSV_PATH = CACHE_DIR / "cot_financial_cache.csv.gz"

POLICY_RATE_SERIES = {
    "US": ["FEDFUNDS"],
    "Japan": ["IRSTCB01JPM156N"],
    "Europe": ["IRSTCB01EZM156N", "ECBDFR"],
    "China": ["IRSTCB01CNM156N"],
    "Taiwan": ["IRSTCB01TWM156N"],
}

BALANCE_SHEET_SERIES = {
    "Fed": "WALCL",
    "ECB": "ECBASSETSW",
    "BOJ": "JPNASSETS",
    "PBOC": "CHNASSETS",
}

CURRENT_ACCOUNT_SERIES = {
    "US": {"series": "BOPGSTB", "freq": "monthly"},
    "Japan": {"series": "XTIMVA01JPM667S", "freq": "monthly"},
    "Germany": {"series": "DEUB6BLTT02STSAQ", "freq": "quarterly"},
    "China": {"series": "CHNB6BLTT02STSAQ", "freq": "quarterly"},
}

COT_TARGETS = {
    "JPY": "097741",
    "EUR": "099741",
    "GBP": "096742",
    "AUD": "232741",
    "10Y UST": "043602",
    "Gold": "088691",
}

HTML_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>宏觀重力 (Macro Gravity) — {{ report_date }}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  {{ macra_style_block | safe }}
  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
  <style>
    :root {
      --bg-primary: #ffffff; --bg-secondary: #f8f9fb; --border: #e8ecf0;
      --text-primary: #0f1923; --text-muted: #8a96a3; --green: #16a34a;
      --yellow: #b45309; --red: #dc2626; --blue: #2563eb; --accent: #4f46e5;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: var(--macra-font, "Inter", sans-serif); color: var(--text-primary); background: #f8f9fb; padding: 12px 14px; }
    .mono { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: #374151; }
    .container { max-width: 1400px; margin: 0 auto; padding: 0 0 2rem; }
    .traffic-wrap { display: flex; justify-content: center; gap: 0.5rem; flex-wrap: wrap; }
    .pill { border: 1px solid #e2e8f0; border-radius: 999px; padding: 0.38rem 0.68rem; font-size: 0.75rem; font-weight: 600; display: inline-flex; align-items: center; gap: 0.4rem; background: #f1f5f9; }
    .pill.GREEN { background: #f0fdf4; border-color: #bbf7d0; color: #15803d; }
    .pill.YELLOW { background: #fffbeb; border-color: #fde68a; color: #92400e; }
    .pill.RED { background: #fef2f2; border-color: #fecaca; color: #b91c1c; }
    .dot { font-size: 0.8rem; line-height: 1; }
    .section { background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; margin-top: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
    .section-lead { }
    .section-title { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; margin-bottom: 0.8rem; font-weight: 600; }
    .narrative { line-height: 1.65; color: #1f2937; }
    .theme-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.8rem; margin-top: 0.8rem; }
    .theme-card { background: #fff; border: 1px solid var(--border); border-radius: 10px; padding: 0.85rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }
    .badge { display: inline-block; border-radius: 999px; border: 1px solid #e2e8f0; padding: 0.1rem 0.4rem; font-size: 0.68rem; color: #6b7280; }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.9rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.79rem; }
    thead th { color: var(--text-muted); text-transform: uppercase; font-size: 0.67rem; letter-spacing: 0.06em; text-align: left; padding: 0.5rem 0.52rem; border-bottom: 1px solid var(--border); }
    tbody td { padding: 0.48rem 0.52rem; border-bottom: 1px solid #eef1f4; vertical-align: middle; }
    tbody tr:nth-child(odd) { background: #fafbfc; }
    .pos { color: var(--green); } .neg { color: var(--red); } .warn { color: var(--yellow); } .muted { color: var(--text-muted); }
    .bar-wrap { width: 180px; height: 10px; border: 1px solid #e2e8f0; border-radius: 999px; background: #f1f5f9; position: relative; overflow: hidden; }
    .bar-mid { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: #cbd5e1; }
    .bar-fill { position: absolute; top: 0; bottom: 0; }
    .bar-fill.blue { background: linear-gradient(90deg, #93c5fd, #2563eb); }
    .bar-fill.red { background: linear-gradient(90deg, #fca5a5, #dc2626); }
    .row-extreme { background: #fffbeb !important; }
    .footer { border-top: 1px solid var(--border); margin-top: 1rem; padding-top: 0.65rem; font-size: 0.7rem; color: var(--text-muted); display: flex; justify-content: space-between; }
    .page-title { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.25rem; }
    .page-sub { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 1rem; }
    @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  {%- macro ph_cls(text) -%}
  {%- set t = (text or '')|trim|lower -%}
  {%- if not t or t == '—' or 'not available' in t or 'not available' in t -%} macra-placeholder{%- endif -%}
  {%- endmacro -%}
  {%- macro fmt_val(v) -%}
  {%- if v is none or (v|string)|trim|lower in ['none','null','nan',''] -%}—{%- else -%}{{ v }}{%- endif -%}
  {%- endmacro -%}
  <main class="container">
    <h1 class="page-title">Macra Times · 宏觀重力 (Macro Gravity)</h1>
    <p class="page-sub mono">{{ report_date }} | Ranked structural matrix · Rank 1–6</p>

    <section class="macra-card macra-card-pad">
      <div class="macra-recon-head">
        <h2 class="macra-recon-title">AI MACRO RECON</h2>
        <span class="macra-badge">{{ ai_recon_badge }}</span>
      </div>
      <p class="macra-recon-body {{ ph_cls(ai_recon_text) }}">{{ ai_recon_text }}</p>
    </section>

    <section class="macra-matrix-section">
      <h2 class="macra-matrix-heading">MARKET MATRIX</h2>
      <p class="macra-matrix-subtitle">Macro Gravity / 宏觀重力</p>
    {% for acc in accordion_rows %}
    <details class="macra-accordion"{% if acc.row.is_us_core %} open{% endif %}>
      <summary class="{% if acc.row.is_us_core %}core-row{% endif %}">
        <span>{{ acc.summary.rank_label }}</span>
        <span>{{ acc.summary.country_label }}</span>
        {% if acc.summary.is_core %}<span class="core-badge">{{ acc.summary.core_badge_html }}</span>{% endif %}
        <span class="mono muted">{{ acc.summary.divergence_text }}</span>
        <span class="pill {{ acc.summary.signal }}">{{ acc.summary.signal }}</span>
        <span class="macra-chevron" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></span>
      </summary>
      <div class="macra-acc-body">
        <div class="macra-gear-grid">
          <div class="macra-gear-cell">
            <h4 class="macra-gear-title">[1] 結構重力</h4>
            <p class="macra-col-body {{ ph_cls(acc.gears.gravity) }}">{{ acc.gears.gravity }}</p>
          </div>
          <div class="macra-gear-cell">
            <h4 class="macra-gear-title">[2] 結構水管</h4>
            <p class="macra-col-body {{ ph_cls(acc.gears.plumbing) }}">{{ acc.gears.plumbing }}</p>
          </div>
          <div class="macra-gear-cell">
            <h4 class="macra-gear-title">[3] 結構溢酬</h4>
            <p class="macra-col-body {{ ph_cls(acc.gears.premium) }}">{{ acc.gears.premium }}</p>
          </div>
        </div>
      </div>
    </details>
    {% endfor %}
    </section>

    <footer class="footer">
      <span>Structural data refreshes weekly. Not investment advice.</span>
      <span class="mono">Generated {{ report_date }} | data as of {{ snapshot_date }} | cache age {{ cache_age_days }}d</span>
    </footer>
  </main>
</body>
</html>"""

FALLBACK_LLM = {
    "regime_summary": "LLM analysis not available",
    "dominant_themes": [
        {
            "theme": "Structural scan unavailable",
            "evidence": "LLM analysis not available",
            "stage": "developing",
            "key_indicator": "LLM analysis not available",
        }
    ],
    "country_cycle": {k: "unclear" for k in ["US", "Japan", "Europe", "China", "Taiwan"]},
    "regime_risks": "LLM analysis not available",
}

_MODULE_KEYS = ("module_a", "module_b", "module_c", "module_d", "module_e")


def _llm_is_placeholder(llm: object) -> bool:
    """True when cached/returned LLM block is the error template, not a real scan."""
    if not isinstance(llm, dict):
        return True
    summary = str(llm.get("regime_summary") or "").strip()
    if not summary or summary == FALLBACK_LLM["regime_summary"]:
        return True
    themes = llm.get("dominant_themes")
    if not isinstance(themes, list) or not themes:
        return True
    first = themes[0] if isinstance(themes[0], dict) else {}
    if str(first.get("theme") or "").strip() == "Structural scan unavailable":
        return True
    return False


def clean_json_response(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


def _fred_series(series_id: str, fred_api_key: str | None = None) -> pd.Series:
    import requests, io
    api_key = fred_api_key or os.getenv("FRED_API_KEY") or os.getenv("fred_api_key")
    if api_key:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key}&file_type=json"
        )
        import time
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


def _latest_change(series: pd.Series, periods: int) -> tuple[float | None, float | None]:
    s = series.dropna()
    if s.empty:
        return None, None
    latest = float(s.iloc[-1])
    if len(s) <= periods:
        return latest, None
    prev = float(s.iloc[-1 - periods])
    return latest, latest - prev


def _latest_pct_change(series: pd.Series, periods: int) -> tuple[float | None, float | None]:
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


def _trend_label(delta: float | None, flat_threshold: float = 0.1) -> str:
    if delta is None:
        return "flat"
    if delta > flat_threshold:
        return "rising"
    if delta < -flat_threshold:
        return "falling"
    return "flat"


def _policy_direction(mom: float | None) -> str:
    if mom is None:
        return "on hold"
    if mom > 0.05:
        return "hiking"
    if mom < -0.05:
        return "cutting"
    return "on hold"


def _bs_direction(chg3m: float | None) -> str:
    if chg3m is None:
        return "flat"
    if chg3m > 0.5:
        return "expanding"
    if chg3m < -0.5:
        return "contracting"
    return "flat"


def fetch_module_a(fred_key: str | None) -> dict:
    policy: dict[str, dict] = {}
    for country, sid_list in POLICY_RATE_SERIES.items():
        done = False
        for sid in sid_list:
            try:
                s = _fred_series(sid, fred_key).tail(24)
                latest, mom = _latest_change(s, 1)
                _, chg3 = _latest_change(s, 3)
                policy[country] = {
                    "latest": latest,
                    "mom_change": mom,
                    "trend_3m": _trend_label(chg3, 0.1),
                    "direction": _policy_direction(mom),
                    "as_of": s.index[-1].date().isoformat() if not s.empty else None,
                    "source": sid,
                }
                done = True
                break
            except Exception:
                continue
        if not done:
            policy[country] = {"latest": None, "mom_change": None, "trend_3m": "flat", "direction": "on hold", "as_of": None, "source": None}

    bs: dict[str, dict] = {}
    for cb, sid in BALANCE_SHEET_SERIES.items():
        try:
            s = _fred_series(sid, fred_key).tail(24)
            latest, mom_pct = _latest_pct_change(s, 1)
            _, chg3_pct = _latest_pct_change(s, 3)
            bs[cb] = {
                "latest": latest,
                "mom_pct": mom_pct,
                "chg3m_pct": chg3_pct,
                "direction": _bs_direction(chg3_pct),
                "as_of": s.index[-1].date().isoformat() if not s.empty else None,
            }
        except Exception:
            bs[cb] = {"latest": None, "mom_pct": None, "chg3m_pct": None, "direction": "flat", "as_of": None}

    try:
        zq = yf.download("ZQ=F", period="6mo", interval="1d", progress=False, auto_adjust=False)
        close = zq["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        last = float(pd.Series(close).dropna().iloc[-1])
        implied = 100.0 - last
        fed_latest = policy.get("US", {}).get("latest")
        implied_change_bps = None if fed_latest is None else (implied - float(fed_latest)) * 100.0
        expectations = {"implied_rate": implied, "implied_change_bps": implied_change_bps, "months_to_next_meeting": 1}
    except Exception:
        expectations = {"implied_rate": None, "implied_change_bps": None, "months_to_next_meeting": None}

    return {"policy_rates": policy, "balance_sheets": bs, "rate_expectations": expectations}


def _download_cot_dataframe() -> pd.DataFrame:
    urls = ["https://www.cftc.gov/files/dea/newcot/financial_lcom.zip"]
    yr = date.today().year
    urls.extend(
        [
            f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{yr}.zip",
            f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{yr - 1}.zip",
        ]
    )
    frames: list[pd.DataFrame] = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                data_name = next(
                    (
                        n
                        for n in zf.namelist()
                        if n.lower().endswith(".csv") or n.lower().endswith(".txt")
                    ),
                    None,
                )
                if not data_name:
                    continue
                with zf.open(data_name) as fp:
                    frames.append(pd.read_csv(fp))
            if "history/fut_fin_txt_" not in url and frames:
                return frames[0]
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True)
    raise RuntimeError("Unable to download COT dataset")


def _load_or_refresh_cot_cache(force_refresh: bool) -> pd.DataFrame:
    if not force_refresh and COT_CACHE_PATH.exists() and COT_CACHE_META_PATH.exists():
        try:
            meta = json.loads(COT_CACHE_META_PATH.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(meta.get("fetched_at"))
            if datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc) < timedelta(days=7):
                try:
                    return pd.read_parquet(COT_CACHE_PATH)
                except Exception:
                    if COT_CACHE_CSV_PATH.exists():
                        return pd.read_csv(COT_CACHE_CSV_PATH)
        except Exception:
            pass
    df = _download_cot_dataframe()
    try:
        df.to_parquet(COT_CACHE_PATH, index=False)
    except Exception:
        pass
    df.to_csv(COT_CACHE_CSV_PATH, index=False, compression="gzip")
    COT_CACHE_META_PATH.write_text(json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
    return df


def fetch_module_b(force_refresh: bool) -> dict:
    try:
        df = _load_or_refresh_cot_cache(force_refresh)
    except Exception:
        return {
            "contracts": {
                k: {
                    "latest_net": None,
                    "chg_4w": None,
                    "direction": "net short",
                    "zscore_52w": None,
                    "extreme": False,
                }
                for k in COT_TARGETS
            }
        }
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("report_date_as_yyyy-mm-dd") or cols.get("as_of_date_in_form_yyyymmdd")
    code_col = cols.get("cftc_contract_market_code") or cols.get("cftc_market_code")
    long_col = (
        cols.get("noncomm_positions_long_all")
        or cols.get("leveraged_funds_positions_long_all")
        or cols.get("asset_mgr_positions_long_all")
    )
    short_col = (
        cols.get("noncomm_positions_short_all")
        or cols.get("leveraged_funds_positions_short_all")
        or cols.get("asset_mgr_positions_short_all")
    )
    if not all([date_col, code_col, long_col, short_col]):
        raise RuntimeError("COT columns missing")

    df = df[[date_col, code_col, long_col, short_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df["net"] = pd.to_numeric(df[long_col], errors="coerce") - pd.to_numeric(df[short_col], errors="coerce")
    results: dict[str, dict] = {}

    for name, code in COT_TARGETS.items():
        code_series = df[code_col].astype(str).str.replace(r"\D", "", regex=True).str[-6:]
        d = df[code_series == code].sort_values(date_col).dropna(subset=["net"])
        if d.empty:
            results[name] = {"latest_net": None, "chg_4w": None, "direction": "net short", "zscore_52w": None, "extreme": False}
            continue
        net = d["net"].reset_index(drop=True)
        latest = float(net.iloc[-1])
        chg4w = float(latest - net.iloc[-5]) if len(net) > 4 else None
        hist = net.tail(52)
        if len(hist) > 8 and float(hist.std(ddof=0)) > 0:
            z = (latest - float(hist.mean())) / float(hist.std(ddof=0))
        else:
            z = None
        results[name] = {
            "latest_net": latest,
            "chg_4w": chg4w,
            "direction": "net long" if latest >= 0 else "net short",
            "zscore_52w": z,
            "extreme": bool(z is not None and abs(z) > 1.5),
        }
    return {"contracts": results}


def _parse_tic_holdings() -> dict:
    url = "https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/mfh.txt"
    text = requests.get(url, timeout=20).text
    lines = [ln for ln in text.splitlines() if ln.strip()]
    targets = {
        "Japan": "Japan",
        "China": "China, Mainland",
        "Taiwan": "Taiwan",
        "All foreign": "Total",
    }
    out = {}
    for k, prefix in targets.items():
        if k == "All foreign":
            line = next((ln for ln in lines if ln.strip().startswith("Grand Total") or ln.strip().startswith("Total")), None)
        else:
            line = next((ln for ln in lines if ln.strip().startswith(prefix)), None)
        if not line:
            out[k] = {"latest_bn": None, "mom_change_bn": None, "trend_3m": "flat"}
            continue
        nums = [float(x) for x in re.findall(r"\d+\.\d", line)]
        if len(nums) < 2:
            out[k] = {"latest_bn": None, "mom_change_bn": None, "trend_3m": "flat"}
            continue
        latest, prev = nums[0], nums[1]
        trend = "flat"
        if len(nums) >= 4:
            d1, d2 = nums[0] - nums[1], nums[1] - nums[2]
            trend = "increasing" if d1 > 0 and d2 > 0 else ("decreasing" if d1 < 0 and d2 < 0 else "mixed")
        out[k] = {"latest_bn": latest, "mom_change_bn": latest - prev, "trend_3m": trend}
    return out


def fetch_module_c(run_date: date) -> dict:
    flow_json = OUTPUT_DIR / f"flow_data_{run_date.isoformat()}.json"
    cn = None
    if flow_json.exists():
        try:
            p = json.loads(flow_json.read_text(encoding="utf-8"))
            cn = p.get("module_c", {}).get("china_fx_reserves")
        except Exception:
            cn = None
    if cn is None:
        cn_data = _fetch_china_reserves()
        cn = {"latest_bn": cn_data.get("latest_bn"), "mom_change_bn": cn_data.get("mom_change_bn")}
    return {"tic_holdings": _parse_tic_holdings(), "china_fx_reserves": cn}


def _ca_trend(latest: float | None, change: float | None) -> str:
    if latest is None or change is None:
        return "surplus narrowing"
    if latest >= 0:
        return "surplus widening" if change > 0 else "surplus narrowing"
    return "deficit widening" if change < 0 else "deficit narrowing"


def fetch_module_d(fred_key: str | None) -> dict:
    out = {}
    for country, meta in CURRENT_ACCOUNT_SERIES.items():
        try:
            s = _fred_series(meta["series"], fred_key)
            s = s.tail(8 if meta["freq"] == "quarterly" else 24)
            p = 1 if meta["freq"] == "monthly" else 1
            latest, chg = _latest_change(s, p)
            out[country] = {"latest": latest, "change": chg, "trend": _ca_trend(latest, chg), "as_of": s.index[-1].date().isoformat() if not s.empty else None}
        except Exception:
            out[country] = {"latest": None, "change": None, "trend": "surplus narrowing", "as_of": None}
    return out


def fetch_module_e(fred_key: str | None) -> dict:
    t3 = _fred_series("T3MFF", fred_key)
    t3_latest, t3_chg = _latest_change(t3, 1)
    try:
        ted = _fred_series("TEDRATE", fred_key)
    except Exception:
        ted = t3
    ted_latest, ted_chg = _latest_change(ted, 1)
    emb = yf.download("EMB", period="6mo", interval="1d", progress=False, auto_adjust=False)["Close"]
    ief = yf.download("IEF", period="6mo", interval="1d", progress=False, auto_adjust=False)["Close"]
    if isinstance(emb, pd.DataFrame):
        emb = emb.iloc[:, 0]
    if isinstance(ief, pd.DataFrame):
        ief = ief.iloc[:, 0]
    emb = pd.Series(emb).dropna()
    ief = pd.Series(ief).dropna()
    emb_1m = (float(emb.iloc[-1]) / float(emb.iloc[-22]) - 1.0) * 100.0 if len(emb) > 21 else None
    ief_1m = (float(ief.iloc[-1]) / float(ief.iloc[-22]) - 1.0) * 100.0 if len(ief) > 21 else None
    em_spread = None if emb_1m is None or ief_1m is None else emb_1m - ief_1m
    return {
        "t3mff": {"latest": t3_latest, "change": t3_chg},
        "tedrate": {"latest": ted_latest, "change": ted_chg},
        "em_risk_proxy": {"emb_1m_minus_ief_1m": em_spread},
    }


_GLOBAL_REGIME_SYSTEM = (
    "You are a macro analyst reviewing structural global data. "
    "Return only valid JSON. No markdown fences, no preamble."
)


def _build_quant_regime(data: dict) -> dict:
    """Build structured regime summary from quantitative modules — no LLM."""
    module_a = data.get("module_a", {})
    module_e = data.get("module_e", {})

    rates = module_a.get("policy_rates", {})
    cycle_map: dict[str, str] = {}
    for country in ["US", "Japan", "Europe", "China", "Taiwan"]:
        r = rates.get(country, {})
        trend = r.get("trend_3m", "flat")
        if trend == "rising":
            cycle_map[country] = "tightening"
        elif trend == "falling":
            cycle_map[country] = "easing"
        else:
            cycle_map[country] = "on_hold"

    ted = (module_e.get("tedrate") or {}).get("latest")
    liq_label = "elevated stress" if ted and float(ted) > 0.5 else "calm"

    countries_summary = ", ".join(f"{c}: {v}" for c, v in cycle_map.items())
    regime_summary = (
        f"Global policy cycle — {countries_summary}. "
        f"Liquidity conditions: {liq_label}. "
        "Full thematic interpretation available in Strategic Synthesis (L3)."
    )

    dominant_themes = [
        {
            "theme": "Policy divergence",
            "evidence": countries_summary,
            "stage": "developing",
            "key_indicator": "Rate differentials and FX momentum",
        }
    ]

    return {
        "regime_summary": regime_summary,
        "dominant_themes": dominant_themes,
        "country_cycle": cycle_map,
        "regime_risks": "See L3 Strategic Synthesis for risk scenarios.",
    }


def _cache_fresh(cache_payload: dict, now: datetime) -> tuple[bool, int]:
    fetched_at = cache_payload.get("fetched_at")
    if not fetched_at:
        return False, 999
    try:
        ts = datetime.fromisoformat(fetched_at)
        age = now - ts.replace(tzinfo=timezone.utc)
        return age < timedelta(days=7), max(0, int(age.total_seconds() // 86400))
    except Exception:
        return False, 999


def _read_cache() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_or_na(v: float | None, ndigits: int = 2, suffix: str = "") -> str:
    if v is None:
        return "—"
    return f"{v:,.{ndigits}f}{suffix}"


def build_l1_html_context(payload: dict, report_date: date, cache_age_days: int) -> dict:
    """Build Jinja context for Layer 1 HTML (main page + appendix)."""
    llm = payload.get("llm", FALLBACK_LLM)
    module_a = payload["module_a"]
    module_b = payload["module_b"]
    module_c = payload["module_c"]
    module_d = payload["module_d"]
    module_e = payload["module_e"]

    policy_rows = []
    for c in ["US", "Japan", "Europe", "China", "Taiwan"]:
        r = module_a["policy_rates"].get(c, {})
        policy_rows.append(
            {
                "country": c,
                "latest": _format_or_na(r.get("latest"), 2, "%"),
                "mom": _format_or_na(r.get("mom_change"), 2),
                "trend": r.get("trend_3m", "flat"),
                "direction": r.get("direction", "on hold"),
            }
        )

    bs_rows = []
    for cb in ["Fed", "ECB", "BOJ", "PBOC"]:
        r = module_a["balance_sheets"].get(cb, {})
        bs_rows.append(
            {
                "cb": cb,
                "latest": _format_or_na(r.get("latest"), 1),
                "mom": _format_or_na(r.get("mom_pct"), 2, "%"),
                "chg3m": _format_or_na(r.get("chg3m_pct"), 2, "%"),
                "direction": r.get("direction", "flat"),
            }
        )

    cot_rows = []
    max_abs = 1.0
    for v in module_b["contracts"].values():
        n = v.get("latest_net")
        if n is not None:
            max_abs = max(max_abs, abs(float(n)))
    for name, r in module_b["contracts"].items():
        net = r.get("latest_net") or 0.0
        w = min(abs(net) / max_abs * 50.0, 50.0)
        style = f"left:50%; width:{w:.1f}%;" if net >= 0 else f"left:{50.0 - w:.1f}%; width:{w:.1f}%;"
        cot_rows.append(
            {
                "contract": name,
                "net_position": net,
                "net_fmt": _format_or_na(r.get("latest_net"), 0),
                "chg4w": r.get("chg_4w") or 0.0,
                "chg4w_fmt": _format_or_na(r.get("chg_4w"), 0),
                "zscore_fmt": _format_or_na(r.get("zscore_52w"), 2),
                "extreme": bool(r.get("extreme")),
                "bar_style": style,
                "bar_color": "blue" if net >= 0 else "red",
            }
        )

    tic_rows = []
    for c in ["Japan", "China", "Taiwan", "All foreign"]:
        r = module_c["tic_holdings"].get(c, {})
        tic_rows.append(
            {
                "country": c,
                "latest": _format_or_na(r.get("latest_bn"), 1),
                "mom": _format_or_na(r.get("mom_change_bn"), 1),
                "mom_val": r.get("mom_change_bn") or 0.0,
                "trend": r.get("trend_3m", "flat"),
            }
        )

    ca_rows = []
    for c in ["US", "Japan", "Germany", "China"]:
        r = module_d.get(c, {})
        ca_rows.append({"country": c, "latest": _format_or_na(r.get("latest"), 2), "change": _format_or_na(r.get("change"), 2), "trend": r.get("trend", "surplus narrowing")})

    liq_rows = [
        {
            "indicator": "T3MFF",
            "value": _format_or_na(module_e["t3mff"].get("latest"), 2),
            "change": _format_or_na(module_e["t3mff"].get("change"), 2),
            "signal": "tightening" if (module_e["t3mff"].get("latest") or 0.0) > 0 else "easy",
        },
        {
            "indicator": "TEDRATE",
            "value": _format_or_na(module_e["tedrate"].get("latest"), 2),
            "change": _format_or_na(module_e["tedrate"].get("change"), 2),
            "signal": "stress" if (module_e["tedrate"].get("latest") or 0.0) > 0.5 else "calm",
        },
        {
            "indicator": "EMB-IEF 1m spread",
            "value": _format_or_na(module_e["em_risk_proxy"].get("emb_1m_minus_ief_1m"), 2, "%"),
            "change": "—",
            "signal": "risk-on" if (module_e["em_risk_proxy"].get("emb_1m_minus_ief_1m") or 0.0) > 0 else "risk-off",
        },
    ]

    snapshot_candidates = []
    for r in module_a["policy_rates"].values():
        if r.get("as_of"):
            snapshot_candidates.append(r["as_of"])
    snapshot_date = max(snapshot_candidates) if snapshot_candidates else report_date.isoformat()
    cn = module_c["china_fx_reserves"]
    core_checks = [
        ("US policy rate", module_a["policy_rates"]["US"]["latest"] is not None),
        ("Europe policy rate", module_a["policy_rates"]["Europe"]["latest"] is not None),
        ("COT JPY", module_b["contracts"]["JPY"]["latest_net"] is not None),
        ("COT EUR", module_b["contracts"]["EUR"]["latest_net"] is not None),
        ("TIC All foreign", module_c["tic_holdings"]["All foreign"]["latest_bn"] is not None),
        ("China reserves", cn.get("latest_bn") is not None),
    ]
    total_core = len(core_checks)
    available_core = sum(1 for _, ok in core_checks if ok)
    data_coverage = {
        "available_core": available_core,
        "total_core": total_core,
        "ratio_pct": round((available_core / total_core) * 100.0, 1) if total_core else 0.0,
        "missing_critical_fields": [name for name, ok in core_checks if not ok][:3],
    }
    quant_payload: dict = {}
    flow_path = OUTPUT_DIR / f"flow_data_{report_date.isoformat()}.json"
    if flow_path.exists():
        try:
            flow_doc = json.loads(flow_path.read_text(encoding="utf-8"))
            quant_payload = (flow_doc.get("quant_engine") or {}).get("payload") or {}
        except Exception:
            quant_payload = {}
    global_div = global_divergence_from_quant(quant_payload)
    accordion_rows = build_l1_accordion_rows(payload, global_div)
    ai_recon_text = str(llm.get("regime_summary") or FALLBACK_LLM["regime_summary"]).strip()
    ai_recon_badge = f"Quant | {report_date.isoformat()}"

    return {
        "report_date": report_date.isoformat(),
        "snapshot_date": snapshot_date,
        "cache_age_days": cache_age_days,
        "accordion_rows": accordion_rows,
        "ai_recon_text": ai_recon_text,
        "ai_recon_badge": ai_recon_badge,
        "regime_summary": llm.get("regime_summary", FALLBACK_LLM["regime_summary"]),
        "dominant_themes": (llm.get("dominant_themes") or FALLBACK_LLM["dominant_themes"])[:3],
        "country_cycle": llm.get("country_cycle", FALLBACK_LLM["country_cycle"]),
        "regime_risks": llm.get("regime_risks", FALLBACK_LLM["regime_risks"]),
        "policy_rows": policy_rows,
        "bs_rows": bs_rows,
        "cot_rows": cot_rows,
        "tic_rows": tic_rows,
        "data_coverage": data_coverage,
        "cn_res_latest": _format_or_na(cn.get("latest_bn"), 1),
        "cn_res_mom": _format_or_na(cn.get("mom_change_bn"), 1),
        "cn_res_mom_val": cn.get("mom_change_bn") or 0.0,
        "ca_rows": ca_rows,
        "liq_rows": liq_rows,
    }


def render_html(output_path: Path, payload: dict, report_date: date, cache_age_days: int) -> None:
    ctx = build_l1_html_context(payload, report_date, cache_age_days)
    html = Template(HTML_TMPL).render(
        **ctx,
        macra_style_block=macra_style_block(),
    )
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Global structural regime monitor")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and fetch all modules")
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    report_date = date.today()
    html_path = Path(args.output) if args.output else OUTPUT_DIR / "global_regime.html"

    cache_payload = _read_cache()
    fresh = False
    cache_age_days = 999
    if cache_payload:
        fresh, cache_age_days = _cache_fresh(cache_payload, now)

    if cache_payload and fresh and not args.force_refresh:
        print("Cache fresh (<7d), loading cache...")
        payload = cache_payload
        if _llm_is_placeholder(payload.get("llm")):
            modules = {k: payload[k] for k in _MODULE_KEYS if k in payload}
            if len(modules) == len(_MODULE_KEYS):
                llm_payload = _build_quant_regime(modules)
                payload = {**payload, "llm": llm_payload}
                _write_json(CACHE_PATH, payload)
                cache_age_days = 0
            else:
                cache_payload = None
                fresh = False

    if not (cache_payload and fresh and not args.force_refresh):
        print("Fetching structural modules A-E...")
        fred_key = os.getenv("FRED_API_KEY")
        module_a = fetch_module_a(fred_key)
        module_b = fetch_module_b(force_refresh=args.force_refresh)
        module_c = fetch_module_c(report_date)
        module_d = fetch_module_d(fred_key)
        module_e = fetch_module_e(fred_key)
        data_payload = {
            "module_a": module_a,
            "module_b": module_b,
            "module_c": module_c,
            "module_d": module_d,
            "module_e": module_e,
        }
        llm_payload = _build_quant_regime(data_payload)

        payload = {
            **data_payload,
            "llm": llm_payload,
            "fetched_at": now.isoformat(),
        }
        _write_json(CACHE_PATH, payload)
        cache_age_days = 0

    out_json = OUTPUT_DIR / f"global_regime_data_{report_date.isoformat()}.json"
    _write_json(out_json, payload)
    render_html(html_path, payload, report_date, cache_age_days)
    print(f"Done. Output: {html_path}")
    print(f"Data JSON: {out_json}")


if __name__ == "__main__":
    main()
