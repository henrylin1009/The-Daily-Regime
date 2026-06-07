#!/usr/bin/env python3
"""Entry point — runs the full macro intelligence pipeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from src.analyst import (
    PLACEHOLDER_EXECUTIVE_SUMMARY,
    PLACEHOLDER_NARRATIVE,
    enrich_matches_what_happened,
)
from src.collect import collect_all
from src.config import OUTPUT_DIR
from src.history_match import build_feature_matrix, find_similar_periods, save_matches
from src.divergence_match import build_divergence_matrix, find_divergence_periods, save_divergence_matches
from src.indicators import compute_signals
from src.factor_attrib import compute_factor_attribution
from src.macra_assets import macra_style_block
from src.macra_ui import (
    apply_gear_semantics_to_gears,
    build_l2_accordion_rows,
    build_l2_country_rows_from_sources,
    global_divergence_from_quant,
    spread_trade_line,
)
from src.render import render_html


def clean_json_response(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start : end + 1]
    return text


def _build_country_rows(output_date: date, signals: dict) -> list[dict]:
    return build_l2_country_rows_from_sources(output_date, signals, None, None)


def _placeholder_class(text: object) -> str:
    s = str(text or "").strip().lower()
    if not s or s in {"—", "-", "n/a", "na"} or "not available" in s or "not available" in s:
        return " macra-placeholder"
    return ""


def _write_international_brief_pages(
    output_date: date,
    us_brief_filename: str,
    country_rows: list[dict],
    us_as_of: str | None,
    us_headline: dict | None,
    us_regime: dict | None,
    gear_semantics: dict | None = None,
) -> None:
    d = output_date.isoformat()
    cb = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    by_country = {r["country"]: r for r in country_rows}
    countries = ["US", "Japan", "Europe", "China", "Taiwan"]

    def _country_page(country: str) -> str:
        r = by_country.get(
            country,
            {
                "signal": "YELLOW",
                "policy_rate": "—",
                "fx_1m": "—",
                "spread_10y": "—",
                "notes": "Partial data",
            },
        )
        tabs = "".join(
            [
                f'<a class="tab {"active" if c == country else ""}" href="brief_{c.lower()}.html">{c}</a>'
                for c in countries
            ]
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{country} Macro View</title>
  <style>
    :root {{ --border:#e8ecf0; --muted:#8a96a3; --accent:#4f46e5; }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family: "IBM Plex Sans", sans-serif; color:#0f1923; background:#f8f9fb; padding: 12px 14px; }}
    .container {{ width:100%; max-width:1400px; margin:0 auto; padding:0 0 1.5rem; }}
    .sticky-header {{ position: sticky; top: 0; z-index: 20; background:#fff; border-bottom:1px solid var(--border); border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); margin-bottom: 12px; }}
    .header-inner {{ display:grid; grid-template-columns:1fr auto; align-items:center; gap:0.8rem; padding:0.85rem 1rem; max-width:1400px; margin:0 auto; }}
    .brand-title {{ font-size:1.05rem; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; }}
    .brand-sub {{ font-size:0.78rem; color:var(--muted); margin-top:0.18rem; }}
    .traffic-wrap {{ display:flex; justify-content:flex-end; gap:0.4rem; flex-wrap:nowrap; }}
    .pill {{ border:1px solid #e2e8f0; border-radius:999px; padding:0.38rem 0.68rem; font-size:0.75rem; font-weight:600; background:#f1f5f9; }}
    .pill.GREEN {{ color:#15803d; border-color:#bbf7d0; background:#f0fdf4; }}
    .pill.YELLOW {{ color:#92400e; border-color:#fde68a; background:#fffbeb; }}
    .pill.RED {{ color:#b91c1c; border-color:#fecaca; background:#fef2f2; }}
    .regime-badge {{ display:none; }}
    .regime-label {{ font-size:0.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; }}
    .regime-value {{ margin-top:0.2rem; font-size:0.86rem; font-weight:600; }}
    .section {{ background:#fff; border:1px solid var(--border); border-radius:12px; padding:1.2rem; margin-top:12px; box-shadow:0 1px 4px rgba(0,0,0,0.04); }}
    .section-title {{ font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; color:#6b7280; margin-bottom:0.8rem; font-weight:600; }}
    .mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; }}
    .muted {{ color:var(--muted); }}
    table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
    th, td {{ border-bottom:1px solid #eef1f4; padding:0.6rem 0.52rem; text-align:left; }}
    th {{ color:#6b7280; text-transform:uppercase; font-size:0.67rem; letter-spacing:0.06em; }}
  </style>
</head>
<body>
  <header class="sticky-header">
    <div class="header-inner">
      <div>
        <div class="brand-title">Macro Intelligence — {country}</div>
        <div class="brand-sub mono">{d} | country view</div>
      </div>
      <div class="traffic-wrap">
        <span class="pill {r["signal"]}">{country} Risk: {r["signal"]}</span>
      </div>
      <div class="regime-badge">
        <div class="regime-label">Country Focus</div>
        <div class="regime-value">{country}</div>
      </div>
    </div>
  </header>
  <main class="container">
    <section class="section">
      <div class="section-title">A. Country Risk Thermometer</div>
      <p>{country} signal is <span class="mono">{r["signal"]}</span>. Missing fields are shown as <span class="mono">—</span>.</p>
    </section>
    <section class="section">
      <div class="section-title">B. Country Feature Snapshot</div>
      <table>
        <thead><tr><th>Feature</th><th>Value</th><th>Note</th></tr></thead>
        <tbody>
          <tr><td>Policy Rate</td><td class="mono">{r["policy_rate"]}</td><td>From global regime module</td></tr>
          <tr><td>FX 1m (local vs USD)</td><td class="mono">{r["fx_1m"]}</td><td>From flow daily module</td></tr>
          <tr><td>US-local 10Y spread</td><td class="mono">{r["spread_10y"]}</td><td>From flow daily module</td></tr>
        </tbody>
      </table>
    </section>
    <section class="section">
      <div class="section-title">C. Country Attribution</div>
      <p class="muted">Country-level attribution is available in Layer 3 Country Index Attribution Explorer.</p>
    </section>
    <section class="section">
      <div class="section-title">D. Country Indicator Coverage</div>
      <p class="muted">Detailed country indicator expansion can be added when more complete country time series are connected.</p>
    </section>
  </main>
</body>
</html>
"""

    # country pages
    for c in countries:
        p = OUTPUT_DIR / f"brief_{c.lower()}.html"
        if c == "US":
            # US stays as original full brief
            p.write_text((OUTPUT_DIR / us_brief_filename).read_text(encoding="utf-8"), encoding="utf-8")
        else:
            p.write_text(_country_page(c), encoding="utf-8")

    # single-page global wrapper: AI Recon -> Matrix accordion -> Appendix
    brief_data_path = OUTPUT_DIR / f"brief_data_{d}.json"
    recon_parts: list[str] = []
    if brief_data_path.exists():
        try:
            brief_data = json.loads(brief_data_path.read_text(encoding="utf-8"))
            exec_sum = str(brief_data.get("executive_summary") or "").strip()
            if exec_sum and exec_sum not in {PLACEHOLDER_EXECUTIVE_SUMMARY, "LLM analysis not available"}:
                recon_parts.append(exec_sum)
        except Exception:
            pass
    flow_summary = "Flow brief focuses on FX, cross-border capital movement, and rate-spread signals."
    flow_micro = {
        "fx": "FX momentum: mixed",
        "rates": "Rate spread tone: mixed",
        "flows": "Cross-border flow tone: mixed",
    }
    flow_top_rows_html = ""
    flow_spread_rows_html = ""
    flow_data_path = OUTPUT_DIR / f"flow_data_{d}.json"
    if flow_data_path.exists():
        try:
            flow_payload = json.loads(flow_data_path.read_text(encoding="utf-8"))
            llm_block = flow_payload.get("llm", {}) if isinstance(flow_payload, dict) else {}
            candidate = str(llm_block.get("flow_narrative") or "").strip()
            if candidate:
                flow_summary = candidate
            flow_sig = flow_payload.get("flow_signals", {}) if isinstance(flow_payload, dict) else {}
            fx_local = flow_payload.get("fx_1m_local_vs_usd", {}) if isinstance(flow_payload, dict) else {}
            spreads = flow_payload.get("bond_spreads_bp", {}) if isinstance(flow_payload, dict) else {}
            tw_flow = flow_payload.get("taiwan_foreign_flow", {}) if isinstance(flow_payload, dict) else {}

            def _fmt_pct(v: object) -> str:
                try:
                    return f"{float(v):+.2f}%"
                except Exception:
                    return "—"

            def _fmt_bp(v: object) -> str:
                try:
                    return f"{float(v):+.0f}bp"
                except Exception:
                    return "—"

            fx_jp = _fmt_pct(fx_local.get("Japan"))
            fx_eu = _fmt_pct(fx_local.get("Europe"))
            flow_micro["fx"] = f"FX momentum: JPY {fx_jp}, EUR {fx_eu}"

            sp_jp = _fmt_bp(spreads.get("Japan"))
            sp_eu = _fmt_bp(spreads.get("Europe"))
            flow_micro["rates"] = f"Rate spread tone: JP {sp_jp}, EU {sp_eu}"

            tw_5d = _fmt_pct(tw_flow.get("flow_5d"))
            tw_20d = _fmt_pct(tw_flow.get("flow_20d"))
            us_sig = str((flow_sig.get("US") or {}).get("signal", "—"))
            flow_micro["flows"] = f"Cross-border flow tone: TW 5d {tw_5d}, TW 20d {tw_20d}, US {us_sig}"

            validation_panel = flow_payload.get("validation_panel", []) if isinstance(flow_payload, dict) else []
            if isinstance(validation_panel, list) and validation_panel:
                ranked = sorted(
                    validation_panel,
                    key=lambda r: abs(float(r.get("z_val") or 0.0)),
                    reverse=True,
                )[:8]
                flow_top_rows_html = "".join(
                    [
                        f"""
                        <tr>
                          <td>{r.get("name", "—")}</td>
                          <td class="mono">{r.get("latest", "—")}</td>
                          <td class="mono">{r.get("chg_1m", "—")}</td>
                          <td class="mono">{r.get("z", "—")}</td>
                          <td class="mono">—</td>
                          <td><span class="signal-chip {r.get("signal", "YELLOW")}">{r.get("signal", "YELLOW")}</span></td>
                        </tr>
                        """
                        for r in ranked
                    ]
                )

            spread_map = flow_payload.get("flow_payload", {}).get("bond_spreads_bp", {}) if isinstance(flow_payload, dict) else {}
            if isinstance(spread_map, dict):
                spread_order = ["US", "Japan", "China", "Taiwan", "Europe"]
                rows = []
                for c in spread_order:
                    v = spread_map.get(c)
                    spread_txt = "—" if v is None else f"{float(v):+.1f}"
                    rows.append(
                        f"""
                        <tr>
                          <td>{c}</td>
                          <td class="mono">—</td>
                          <td class="mono">{spread_txt}</td>
                        </tr>
                        """
                    )
                flow_spread_rows_html = "".join(rows)
        except Exception:
            pass
    flow_default = "Flow brief focuses on FX, cross-border capital movement, and rate-spread signals."
    if flow_summary and flow_summary != flow_default and flow_summary not in recon_parts:
        recon_parts.append(flow_summary)
    recon_text = "\n\n".join(recon_parts) if recon_parts else "完整分析見戰略綜合（L3）。"
    ai_recon_badge = f"Quant | {d}"

    def _pill(label: str, signal: str) -> str:
        cls = signal if signal in {"GREEN", "YELLOW", "RED"} else "YELLOW"
        txt = "LOW" if cls == "GREEN" else ("ELEVATED" if cls == "YELLOW" else "HIGH")
        return f'<span class="pill {cls}">{label}:{txt}</span>'

    country_cards = "".join(
        [
            f"""
            <button class="card card-btn" type="button" data-src="brief_{r['country'].lower()}.html?cb={cb}">
              <div class="topline"><span class="ctry">{r['country']}</span><span class="sig {r['signal']}">{r['signal']}</span></div>
              <div class="meta">{d} | {us_as_of or d}</div>
              <div class="mini-pills">
                {_pill("R", r.get("signal", "YELLOW"))}
                {_pill("I", us_headline.get("inflation", {}).get("signal", "YELLOW") if r["country"] == "US" and us_headline else "YELLOW")}
                {_pill("S", us_headline.get("financial_stress", {}).get("signal", "YELLOW") if r["country"] == "US" and us_headline else "YELLOW")}
              </div>
              <div class="regime-mini">{((us_regime or {}).get("label", "—") + " · " + str(round((us_regime or {}).get("probability", 0) * 100, 0)) + "%") if r["country"] == "US" and us_regime else "—"}</div>
              <div class="row"><span>P</span><span class="mono">{r['policy_rate']}</span><span>FX1m</span><span class="mono">{r['fx_1m']}</span></div>
            </button>
            """
            for r in country_rows
        ]
    )
    country_signal_chips = "".join(
        [
            f'<span class="signal-chip {r.get("signal", "YELLOW")}">● {r["country"]}: {r.get("signal", "YELLOW")}</span>'
            for r in country_rows
        ]
    )

    rec_sig = us_headline.get("recession_risk", {}).get("signal", "YELLOW") if us_headline else "YELLOW"
    inf_sig = us_headline.get("inflation", {}).get("signal", "YELLOW") if us_headline else "YELLOW"
    stress_sig = us_headline.get("financial_stress", {}).get("signal", "YELLOW") if us_headline else "YELLOW"

    def _header_pill(label: str, sig: str) -> str:
        txt = "LOW" if sig == "GREEN" else ("ELEVATED" if sig == "YELLOW" else "HIGH")
        return f'<span class="pill {sig}"><span class="dot">●</span>{label}: {txt}</span>'

    us_pills_html = "".join([
        _header_pill("Recession", rec_sig),
        _header_pill("Inflation", inf_sig),
        _header_pill("Stress", stress_sig)
    ])

    flow_doc: dict = {}
    if flow_data_path.exists():
        try:
            flow_doc = json.loads(flow_data_path.read_text(encoding="utf-8"))
        except Exception:
            flow_doc = {}
    quant_payload = (flow_doc.get("quant_engine") or {}).get("payload") or {}
    global_div = global_divergence_from_quant(quant_payload)
    l2_accordion = build_l2_accordion_rows(country_rows, flow_doc, quant_payload, global_div)
    spread_line = spread_trade_line(l2_accordion and [a["row"] for a in l2_accordion] or country_rows)

    CHEVRON = '<span class="macra-chevron" aria-hidden="true"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></span>'
    accordion_parts: list[str] = []
    for acc in l2_accordion:
        row = acc["row"]
        sm = acc["summary"]
        country = str(row.get("country") or "")
        sem = (gear_semantics or {}).get(country, {}) if isinstance(gear_semantics, dict) else {}
        g = apply_gear_semantics_to_gears(acc["gears"], sem if isinstance(sem, dict) else None)
        exp_title = str((sem or {}).get("expectation_label") or "").strip() or "戰術預期"
        plumb_title = str((sem or {}).get("plumbing_label") or "").strip() or "底層水管"
        hedge_title = str((sem or {}).get("hedging_label") or "").strip() or "避險異常"
        open_attr = " open" if row.get("is_us_core") else ""
        sum_cls = "core-row" if row.get("is_us_core") else ""
        core_badge = (
            f'<span class="core-badge">{sm["core_badge_html"]}</span>' if sm.get("is_core") else ""
        )
        sig = sm.get("signal", "YELLOW")
        accordion_parts.append(
            f"""
    <details class="macra-accordion"{open_attr}>
      <summary class="{sum_cls}">
        <span>{sm['rank_label']}</span>
        <span>{sm['country_label']}</span>
        {core_badge}
        <span class="mono meta">{sm['divergence_text']}</span>
        <span class="pill {sig}">{sig}</span>
        {CHEVRON}
      </summary>
      <div class="macra-acc-body">
        <div class="macra-gear-grid">
          <div class="macra-gear-cell"><h4 class="macra-gear-title">[1] {exp_title}</h4><p class="macra-col-body{_placeholder_class(g['expectation'])}">{g['expectation']}</p></div>
          <div class="macra-gear-cell"><h4 class="macra-gear-title">[2] {plumb_title}</h4><p class="macra-col-body{_placeholder_class(g['plumbing'])}">{g['plumbing']}</p></div>
          <div class="macra-gear-cell"><h4 class="macra-gear-title">[3] {hedge_title}</h4><p class="macra-col-body{_placeholder_class(g['hedging'])}">{g['hedging']}</p></div>
        </div>
      </div>
    </details>"""
        )
    accordion_html = "".join(accordion_parts)

    wrapper = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8" /><meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>市場動能 (Market Momentum)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
{macra_style_block()}
<script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
<style>
:root {{ --border:#e8ecf0; --muted:#8a96a3; --accent:#4f46e5; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family: "Inter", "IBM Plex Sans", sans-serif; color:#0f172a; background:#f8f9fb; padding: 12px 14px; }}
.wrap {{ max-width:1400px; margin:0 auto; }}
.sticky-header {{ position: sticky; top: 0; z-index: 100; background: #fff; border-bottom: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.06); border-radius: 12px; margin-bottom: 12px; }}
.header-inner {{ display: grid; grid-template-columns: 1.2fr 2fr; gap: 1rem; align-items: center; padding: 0.85rem 1rem; }}
.brand-title {{ font-size: 1.05rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }}
.brand-sub {{ font-size: 0.78rem; color: var(--muted); margin-top: 0.18rem; }}
.mono {{ font-family: "JetBrains Mono", ui-monospace, monospace; }}
.meta {{ font-size:0.68rem; color:#8a96a3; }}
.country-list {{ display:grid; grid-template-columns:1fr; gap:12px; }}
.card {{ border:1px solid var(--border); border-radius:12px; padding:14px; background:#fff; box-shadow: 0 1px 3px rgba(0,0,0,0.04); transition: all 0.2s ease; }}
.card-btn {{ width:100%; text-align:left; cursor:pointer; outline:none; }}
.card-btn:hover {{ transform: translateY(-2px); box-shadow: 0 6px 18px rgba(0,0,0,0.06); border-color: var(--accent); }}
.topline {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; }}
.ctry {{ font-weight:700; margin-bottom:3px; font-size:0.95rem; }}
.sig {{ font-size:0.7rem; font-weight:700; border-radius:999px; padding:2px 7px; border:1px solid #e2e8f0; }}
.sig.GREEN {{ color:#15803d; border-color:#bbf7d0; background:#f0fdf4; }}
.sig.YELLOW {{ color:#92400e; border-color:#fde68a; background:#fffbeb; }}
.sig.RED {{ color:#b91c1c; border-color:#fecaca; background:#fef2f2; }}
.mini-pills {{ display:flex; gap:4px; flex-wrap:nowrap; margin-bottom:6px; }}
.mini-pills .pill {{ font-size:0.64rem; padding:2px 6px; }}
.regime-mini {{ font-size:0.68rem; color:#374151; margin-bottom:6px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.row {{ display:flex; justify-content:space-between; gap:6px; font-size:0.76rem; border-top:1px solid #edf1f5; padding-top:6px; margin-top:6px; }}
.signal-row {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }}
.signal-chip {{ border:1px solid #e2e8f0; border-radius:999px; padding:0.25rem 0.55rem; font-size:0.74rem; font-weight:700; background:#f8f9fb; }}
.signal-chip.GREEN {{ color:#15803d; border-color:#bbf7d0; background:#f0fdf4; }}
.signal-chip.YELLOW {{ color:#92400e; border-color:#fde68a; background:#fffbeb; }}
.signal-chip.RED {{ color:#b91c1c; border-color:#fecaca; background:#fef2f2; }}
.flow-subtitle {{ font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:#6b7280; margin:10px 0 6px; font-weight:700; }}
.deep-iframe {{ width:100%; border:1px solid var(--border); border-radius:10px; min-height:720px; margin-top:12px; }}
.modal {{ position:fixed; inset:0; background:rgba(15,25,35,0.5); display:none; align-items:center; justify-content:center; z-index:999; }}
.modal.open {{ display:flex; }}
.modal-box {{ width:min(1200px,94vw); height:min(86vh,900px); background:#fff; border-radius:12px; border:1px solid var(--border); overflow:hidden; box-shadow:0 20px 50px rgba(0,0,0,.25); }}
.modal-head {{ height:44px; border-bottom:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; padding:0 12px; }}
.modal-title {{ font-size:0.82rem; font-weight:700; color:#374151; }}
.modal-close {{ border:1px solid var(--border); border-radius:8px; background:#fff; padding:4px 8px; font-size:0.78rem; cursor:pointer; }}
.modal-frame {{ width:100%; height:calc(100% - 44px); border:0; }}
</style></head>
<body>
  <main class="wrap">
    <header class="sticky-header">
      <div class="header-inner" style="display:block; padding:0.85rem 1rem;">
        <div class="brand-title">Macra Times · 市場動能 (Market Momentum)</div>
        <div class="brand-sub mono">{d} | 宏觀資料最新：{us_as_of or d}（FRED 月頻資料，非即時）</div>
      </div>
    </header>
    <section class="macra-card macra-card-pad" style="margin-top:12px;">
      <div class="macra-recon-head">
        <h2 class="macra-recon-title">AI MACRO RECON</h2>
        <span class="macra-badge">{ai_recon_badge}</span>
      </div>
      <p class="macra-recon-body{_placeholder_class(recon_text)}">{recon_text}</p>
    </section>
    <section class="macra-matrix-section">
      <h2 class="macra-matrix-heading">MARKET MATRIX</h2>
      <p class="macra-matrix-subtitle">Tactical Dynamics / 市場動能</p>
      <p style="font-size:0.75rem;color:#6b7280;margin:0.25rem 0 0.75rem;">排名依風險信號強度排列，Rank 1 = 最需關注，最後一名 = 當前動能最強。點開可查看各國詳細分析。</p>
      <p class="macra-matrix-spread" style="margin-top:0;">{spread_line}</p>
    {accordion_html}
    </section>
  </main>
</body></html>"""
    (OUTPUT_DIR / "brief_global.html").write_text(wrapper, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro Intelligence Platform")
    parser.add_argument("--force-refresh", action="store_true", help="Re-fetch all data from OpenBB")
    parser.add_argument("--date", type=str, default=None, help="Output date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    output_date = date.fromisoformat(args.date) if args.date else date.today()

    try:
        print("Stage 1: Collecting data...")
        data = collect_all(force_refresh=args.force_refresh)

        print("Stage 2: Computing signals...")
        signals = compute_signals(data)

        print("Stage 2b: Factor attribution...")
        factor_attrib = compute_factor_attribution(
            data, force_refresh=args.force_refresh
        )

        print("Stage 3: Finding historical matches...")
        feature_matrix = build_feature_matrix(data)

        # Exclude last 12 months from history matching to avoid self-similar recent periods
        current_episode_start = pd.Timestamp.today() - pd.DateOffset(months=12)

        matches = find_similar_periods(
            feature_matrix,
            data,
            current_episode_start=current_episode_start,
        )
        print("Enriching historical match context...")
        matches = enrich_matches_what_happened(matches, signals, use_api=False)
        save_matches(matches)

        print("Stage 3b: Global divergence matching...")
        try:
            div_matrix = build_divergence_matrix(data)
            divergence_matches = find_divergence_periods(div_matrix, data)
            save_divergence_matches(divergence_matches)
            print(f"  Divergence matches: {[m['date'] for m in divergence_matches]}")
        except Exception as _div_exc:
            print(f"  Warning: divergence matching failed: {_div_exc}", file=__import__("sys").stderr)
            divergence_matches = []

        regime_stats = {}

        narrative = ""
        executive_summary = ""

        print("Rendering output...")
        country_rows = _build_country_rows(output_date, signals)
        out_path = render_html(
            signals,
            matches,
            narrative,
            executive_summary=executive_summary,
            output_date=output_date,
            factor_attrib=factor_attrib,
            regime_stats=regime_stats,
            regime_summary=regime_summary,
            country_rows=country_rows,
            show_country_matrix=False,
        )
        print(f"Done. Output: {out_path}")
        _write_international_brief_pages(
            output_date,
            out_path.name,
            country_rows,
            signals.get("as_of"),
            signals.get("headline"),
            regime_summary,
        )
        brief_data_path = out_path.parent / f"brief_data_{output_date.isoformat()}.json"
        brief_data_payload = {
            "date": output_date.isoformat(),
            "signals": signals,
            "matches": matches,
            "divergence_matches": divergence_matches,
            "regime_summary": regime_summary,
            "regime_stats": regime_stats,
            "narrative": narrative,
            "executive_summary": executive_summary,
            "factor_attrib": factor_attrib,
        }
        brief_data_path.write_text(json.dumps(brief_data_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Data JSON: {brief_data_path}")
    except EnvironmentError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
