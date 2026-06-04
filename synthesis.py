#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import date
from functools import lru_cache
from pathlib import Path

import yfinance as yf

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore[assignment]
    types = None  # type: ignore[assignment]
from jinja2 import Template
from openai import OpenAI

import math

import numpy as np
import pandas as pd

from src.config import (
    OUTPUT_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    get_deepseek_base_url,
    get_deepseek_model,
    get_gemini_model,
    require_deepseek_key,
    require_gemini_key,
)
from src.macra_ui import (
    apply_gear_semantics_to_l3_view_model,
    build_l2_country_rows_from_sources,
    build_l2_gear_raw_context,
    build_l3_view_model,
)
from src.macra_assets import macra_style_block
from src.country_signals import compute_country_signals, compute_vs_us_alignment
from src.collect import load_all_from_cache

FALLBACK = "LLM analysis not available"

CIO_SYSTEM_PROMPT = """You are a macro analyst writing a daily brief for sophisticated retail investors. Your readers understand macro concepts when explained clearly, but do not know financial jargon or quant metrics. Your thinking is rigorous; your language is plain.

【No-Jargon Rule — Absolute】
Never write ticker symbols, Z-scores, basis points, or technical metric names in the output. Translate everything:
- "SKEW rising while VVIX falls" → "institutions are quietly buying downside protection even as surface volatility looks calm"
- "SHY/TLT deteriorating" → "the bond market is pricing out rate cuts"
- "RSP/SPY improving" → "the rally is broadening — more stocks are participating, not just the mega-caps"
- "DXY deteriorating" → "the US dollar is weakening against major currencies"
- "Z-score = -2.5" → "positioning is at an extreme rarely seen in recent history"
- "JPY net short extreme" → "almost everyone is betting against the yen — a crowded trade that could unwind violently"

【Absolute Anchor】
The quant engine provides macro_regime_label (Goldilocks, Overheat, Stagflation, Deflationary Bust). Treat it as structural truth. Your job: explain whether market behaviour is consistent with this regime, and where the tensions are.

【Step 1 — Internal Reasoning (not shown in output)】
pre_analysis fields (cross_layer_conflict, momentum_verification, event_pulse_analysis): think through the data, identify the dominant story and any conflicts. This is your scratchpad — you may use metric names here since it is internal.

【Step 2 — Daily Themes (2-3 themes, MOST IMPORTANT)】
Extract 2-3 dominant themes from ALL the country data combined. Each theme:
- title: REQUIRED for every theme. A short punchy headline, max 8 words, NOT a full sentence and NOT the first sentence of the body. (e.g. "Goldilocks holds, but cracks are forming", "Yen positioning at a dangerous extreme", "Taiwan inflows running hot"). Never leave this blank.
- body: 2-3 sentences in plain language. Cite 1-2 specific countries as evidence. Do NOT repeat the title verbatim. No jargon.
- implication: One sentence on what this means for a long-term investor's portfolio over the next 6–12 months — not "watch X", but "this favours / pressures / is neutral for [asset class or region]".
- reasoning: 3-4 sentences — the "show your work" long version shown only on hover. UNLIKE every other field, HERE you SHOULD cite specific figures so the reader can see how you reached this theme: name the actual numbers (e.g. "CPI at 3.95% and still rising", "foreign inflows in the top fifth of the past year", "the 2-10y curve flattened to +0.49%", "yen net shorts near a record"). Still readable prose, not a ticker dump, but quantitatively concrete. This is where the data justification lives.
- supporting_countries: list of country names used as evidence

【Step 3 — Daily Overview (zone1_pulse.flash_bullets)】
CRITICAL: This is shown DIRECTLY to retail readers. Do NOT copy from pre_analysis — pre_analysis is your internal scratchpad and is full of jargon. You must REWRITE everything in plain language here.
ABSOLUTELY FORBIDDEN in these three fields: ticker symbols (SPY, HYG, DXY, NZD/JPY, USD/CNH, SHY/TLT, etc.), Z-scores (z=2.56), the word "divergence score", numbers like "(43)", "RED signal", "bp", "$165bn", "SKEW", "VVIX", "Layer 2". Translate every one of these into a plain concept.
- capital_flows: Where is money moving globally and why? (plain language, one sentence)
- volatility: Is the market calm or nervous, and does it match the underlying risk? (plain language, one sentence)
- macro_sentiment: What is the market's overall mood and is it justified? (plain language, one sentence)
Example of WRONG: "SPY momentum is accelerating (z=2.56), HYG improving, DXY deteriorating."
Example of RIGHT: "The stock rally is broadening out, the dollar is softening, and risk appetite is healthy."

【Step 4 — Main Narrative (cio_directive)】
- the_stance: One sentence on the overall situation (plain language, ≤20 words, no jargon)
- the_narrative: Two paragraphs. First explains the current regime and why it holds. Second explains the key tension or divergence to watch. Accessible to a reader who does not follow markets daily. NEVER mention ratio/ticker names like "TIP/IEF", "SHY/TLT", "RSP/SPY" — describe what they mean instead (e.g. "the bond market's inflation expectations").
- the_watchout: "If [plain language condition] then [plain language consequence for investors]"
- positioning_6_12m: One sentence on how a long-term investor should tilt their portfolio over the next 6–12 months given this regime (e.g. "This environment historically favours equities and real assets over long-duration bonds"). Plain language, no tickers, no jargon.
- positioning_6_12m_reasoning: 2-3 sentences — hover-only long version explaining WHY this tilt, citing the concrete drivers and figures (regime, historical conditional returns, key data points). Quantitatively concrete is encouraged here.

【Step 4b — Watch List】
Each watch_list item is a short alert (en/zh) PLUS a reasoning_en/reasoning_zh long version (shown only on hover): 1-2 sentences explaining why this matters, citing the concrete trigger and figures (e.g. "yen net shorts are near a record and the carry trade is crowded — an unwind could be violent"). Numbers are welcome in the reasoning fields.

【Step 5 — Gear Matrix (all six countries)】
For each country, translate the raw signals into plain language:
- expectation_en / expectation_zh: What is this market pricing in? (≤28 words each, no jargon)
- plumbing_en / plumbing_zh: How is money flowing? (≤28 words each)
- hedging_en / hedging_zh: What risk to watch? (≤28 words each)
- expectation_label_en/zh, plumbing_label_en/zh, hedging_label_en/zh: Short titles (≤6 words each language)

【Bilingual Output — Required】
Every reader-facing string MUST include BOTH English (_en) and Traditional Chinese (_zh, 繁體白話, same meaning, equally plain).
Also set legacy unsuffixed keys (title, body, the_stance, expectation, etc.) to the English (_en) value for backward compatibility.

【Output — valid JSON only】
{
  "daily_themes": [
    {
      "title_en": "...",
      "title_zh": "...",
      "title": "...",
      "body_en": "...",
      "body_zh": "...",
      "body": "...",
      "implication_en": "...",
      "implication_zh": "...",
      "implication": "...",
      "reasoning_en": "... (numbers welcome — the data justification)",
      "reasoning_zh": "...（可放具體數字 — 數據依據）",
      "supporting_countries": ["US", "Japan"]
    }
  ],
  "pre_analysis": { "cross_layer_conflict": "...", "momentum_verification": "...", "event_pulse_analysis": "..." },
  "zone1_pulse": { "flash_bullets": { "capital_flows": "...", "volatility": "...", "macro_sentiment": "..." } },
  "cio_directive": {
    "the_stance_en": "...",
    "the_stance_zh": "...",
    "the_stance": "...",
    "the_narrative_en": ["...", "..."],
    "the_narrative_zh": ["...", "..."],
    "the_narrative": ["...", "..."],
    "the_watchout_en": "If ... then ...",
    "the_watchout_zh": "若……則……",
    "the_watchout": "If ... then ...",
    "positioning_6_12m_en": "...",
    "positioning_6_12m_zh": "...",
    "positioning_6_12m": "...",
    "positioning_6_12m_reasoning_en": "... (numbers welcome)",
    "positioning_6_12m_reasoning_zh": "...（可放數字）"
  },
  "relationship_analysis": {
    "status": "CONVERGENCE or DIVERGENCE",
    "pure_alpha": {
      "expectation_arbitrage": "...",
      "crowdedness_leverage": "...",
      "hedging_cost_anomaly": "..."
    },
    "divergence_explanation": "...",
    "body": "..."
  },
  "convergence_divergence": {"status": "...", "reason": "..."},
  "watch_list": [{"en": "...", "zh": "...", "reasoning_en": "... (numbers welcome)", "reasoning_zh": "...（可放數字）"}],
  "country_attribution_notes": { "US": "...", "Japan": "...", "Europe": "...", "China": "...", "Taiwan": "..." },
  "gear_matrix_semantics": {
    "US": {
      "expectation_en": "...", "expectation_zh": "...", "expectation": "...",
      "plumbing_en": "...", "plumbing_zh": "...", "plumbing": "...",
      "hedging_en": "...", "hedging_zh": "...", "hedging": "...",
      "expectation_label_en": "...", "expectation_label_zh": "...", "expectation_label": "...",
      "plumbing_label_en": "...", "plumbing_label_zh": "...", "plumbing_label": "...",
      "hedging_label_en": "...", "hedging_label_zh": "...", "hedging_label": "..."
    },
    "Japan": { "...": "same shape as US" },
    "Europe": { "...": "same shape as US" },
    "China": { "...": "same shape as US" },
    "Taiwan": { "...": "same shape as US" }
  }
}
"""

DEEPSEEK_MODEL_FALLBACKS = ("deepseek-chat",)

_CIO_REQUIRED_KEYS = frozenset(
    {
        "pre_analysis",
        "zone1_pulse",
        "cio_directive",
        "relationship_analysis",
    }
)

_LEGACY_CIO_KEYS = frozenset(
    {
        "pre_analysis",
        "macro_pulse",
        "cio_macro_thesis",
        "headline",
        "structural_vs_tactical_dynamics",
    }
)

_DIRECTIVE_NUMBER_RE = re.compile(
    r"(\d+\.?\d*\s*%|\bZ[\s=-]|\bz-score\b|\b\d+\.\d+\b|\b\d+\s*bn\b|=\s*\d)",
    re.IGNORECASE,
)

COUNTRY_CYCLE_KEYS = ("US", "Japan", "Europe", "UK", "China", "Taiwan")


def _us_growth_z_score(quant_context: dict) -> float | None:
    """Read US_Growth_Z_Score from quant_engine payload (macro_quant_engine schema)."""
    if not isinstance(quant_context, dict):
        return None
    raw = (
        quant_context.get("Layer_1_Structural", {})
        .get("Global_Liquidity_and_Growth", {})
        .get("US_Growth_Z_Score")
    )
    if raw is None:
        return None
    try:
        z = float(raw)
    except (TypeError, ValueError):
        return None
    if not (z == z):  # NaN
        return None
    return z


def _us_cycle_from_growth_z(z: float) -> str:
    """Deterministic US cycle label from Michigan/INDPRO growth Z-score."""
    if z > 1.0:
        return "Expansion"
    if z < -1.0:
        return "Contraction" if z <= -2.0 else "Slowdown"
    return "Neutral"


def _apply_quant_country_cycle(
    synthesis: dict,
    l1: dict,
    quant_context: dict,
) -> tuple[dict, dict[str, str]]:
    """
    Merge Layer 1 LLM country_cycle with quant overrides (US from US_Growth_Z).
    Returns updated synthesis dict and the merged country_cycle map.
    """
    llm_block = l1.get("llm") if isinstance(l1.get("llm"), dict) else {}
    base: dict[str, str] = {}
    for c in COUNTRY_CYCLE_KEYS:
        val = (llm_block.get("country_cycle") or {}).get(c) if isinstance(llm_block.get("country_cycle"), dict) else None
        base[c] = str(val).strip() if val else "unclear"

    z = _us_growth_z_score(quant_context)
    if z is not None:
        base["US"] = _us_cycle_from_growth_z(z)
        print(
            f"  country cycle: US={base['US']} (US_Growth_Z_Score={z:+.2f})",
            file=sys.stderr,
        )

    merged_synthesis = {**synthesis, "country_cycle": base}
    return merged_synthesis, base


HTML_TMPL = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>戰略綜合 (Strategic Synthesis) — {{ report_date }}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  {{ macra_style_block | safe }}
  <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
  <style>
    :root { --bg:#f8f9fb; --card:#fff; --ink:#111827; --muted:#6b7280; --line:#e8ecf0; --green:#10b981; --red:#ef4444; }
    body { margin:0; background:#f8f9fb; color:var(--ink); font-family:'Inter', 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; -webkit-font-smoothing:antialiased; }
    .wrap { max-width:1100px; margin:0 auto; padding:1.2rem 1.4rem; }
    .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:1rem 1.15rem; margin-bottom:0.9rem; box-shadow:0 1px 3px rgba(0,0,0,0.06); }
    .header-row { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:1.25rem; }
    .title { font-size:1.35rem; font-weight:700; margin:0 0 0.25rem; text-transform:uppercase; letter-spacing:0.04em; }
    .meta { color:var(--muted); font-family:'JetBrains Mono', ui-monospace, monospace; font-size:0.82rem; line-height:1.45; }
    .divergence-badge { display:inline-flex; flex-direction:column; align-items:center; justify-content:center; padding:0.65rem 1.1rem; border-radius:12px; border:2px solid var(--line); min-width:108px; box-shadow:0 2px 6px rgba(0,0,0,0.06); }
    .db-label { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.1em; font-weight:700; opacity:0.85; }
    .db-score { font-size:1.75rem; font-weight:800; font-family:'JetBrains Mono', ui-monospace, monospace; line-height:1.1; margin:0.15rem 0; }
    .db-level { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; }
    .gauge-risk { border-color:#fca5a5; background:#fef2f2; }
    .gauge-risk .db-score, .gauge-risk .db-level { color:#b91c1c; }
    .gauge-warning { border-color:#fde68a; background:#fffbeb; }
    .gauge-warning .db-score, .gauge-warning .db-level { color:#b45309; }
    .gauge-stable { border-color:#86efac; background:#f0fdf4; }
    .gauge-stable .db-score, .gauge-stable .db-level { color:#047857; }
    .section-title { font-size:0.95rem; font-weight:700; margin:0 0 0.5rem; letter-spacing:-0.01em; }
    .mono { font-family:'JetBrains Mono', ui-monospace, monospace; font-size:0.82rem; }
    .muted { color:var(--muted); }
    .zone { margin-bottom:1.35rem; padding:1.2rem 1.35rem; }
    .zone-label { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); font-weight:700; margin-bottom:0.55rem; }
    .zone-prose { font-size:0.92rem; line-height:1.72; color:#374151; margin:0 0 0.85rem; }
    .zone-prose.narrative { font-size:0.93rem; line-height:1.75; }
    .zone-prose:last-child { margin-bottom:0; }
    .zone-1 { border-left:3px solid #6b7280; }
    .zone-2 { border-left:3px solid #111827; }
    .zone-3 { border-left:3px solid #4b5563; }
    .flash-list { margin:0; padding:0; list-style:none; }
    .flash-list li { font-size:0.9rem; line-height:1.55; padding:0.35rem 0; border-bottom:1px solid #f3f4f6; color:#1f2937; }
    .flash-list li:last-child { border-bottom:none; }
    .flash-list strong { color:#6b7280; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em; margin-right:0.35rem; }
    .stance-line { font-size:1.18rem; font-weight:700; line-height:1.5; color:#111827; margin:0 0 1rem; }
    .watchout-box { margin-top:1rem; padding:0.75rem 0.95rem; background:#f9fafb; border:1px solid var(--line); border-radius:8px; font-size:0.9rem; line-height:1.6; color:#1f2937; }
    .positioning-box { margin-top:0.6rem; padding:0.75rem 0.95rem; background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; font-size:0.9rem; line-height:1.6; color:#14532d; }
    .dual-col { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }
    @media (max-width:768px) { .dual-col, .conflict-matrix { grid-template-columns:1fr; } }
    .skeptic-label { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); font-weight:700; margin:0 0 0.35rem; }
    .skeptic-block p { margin:0; font-size:0.9rem; line-height:1.65; color:#374151; }
    .conflict-matrix { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1rem 0; }
    .matrix-col { padding:0.75rem 0.85rem; border-radius:8px; border:1px solid var(--line); }
    .matrix-col.bull { background:#f0fdf4; }
    .matrix-col.bear { background:#fef2f2; }
    .matrix-col h4 { margin:0 0 0.45rem; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em; }
    .matrix-col ul { margin:0; padding-left:1.1rem; font-size:0.86rem; line-height:1.55; }
    .trap-box { margin:1rem 0; padding:0.85rem 1rem; background:#fffbeb; border:1px solid #fde68a; border-radius:8px; }
    .trap-box h4 { margin:0 0 0.4rem; font-size:0.72rem; text-transform:uppercase; color:#92400e; letter-spacing:0.05em; }
    .trap-box p { margin:0; font-size:0.9rem; line-height:1.65; color:#78350f; }
    .divergence-explainer { margin:0.85rem 0 1rem; padding:0.65rem 0.85rem; background:#f9fafb; border-left:3px solid #9ca3af; font-size:0.88rem; line-height:1.6; color:#374151; }
    .watchout-label { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); font-weight:700; margin-bottom:0.25rem; }
    .logic-head { display:flex; flex-wrap:wrap; align-items:center; gap:0.5rem; margin-bottom:0.5rem; }
    .status-badge { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; padding:3px 8px; border-radius:6px; }
    .status-badge.diverge { background:#fee2e2; color:#b91c1c; }
    .status-badge.converge { background:#dcfce7; color:#15803d; }
    @media (max-width:768px) { .header-row { flex-direction:column; align-items:flex-start; } }
    .traffic-wrap { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.25rem; }
    .pill { display:inline-flex; align-items:center; gap:0.3rem; border-radius:999px; padding:0.18rem 0.5rem; font-size:0.74rem; font-weight:600; border:1px solid var(--line); }
    .pill.GREEN { background:#ecfdf5; color:#166534; border-color:#bbf7d0; }
    .pill.RED { background:#fef2f2; color:#991b1b; border-color:#fecaca; }
    .pill.YELLOW { background:#fffbeb; color:#92400e; border-color:#fde68a; }
    .dot { font-size:0.5rem; }
    .momentum-row { display:flex; flex-wrap:wrap; gap:0.4rem; margin-top:0.45rem; }
    .mom-tag { font-size:0.72rem; font-weight:600; padding:0.2rem 0.5rem; border-radius:6px; border:1px solid transparent; }
    .mom-improving { background:#ecfdf5; color:#047857; border-color:#a7f3d0; }
    .mom-deteriorating { background:#fef2f2; color:#b91c1c; border-color:#fecaca; }
    .mom-stable { background:#f3f4f6; color:#4b5563; border-color:#e5e7eb; }
    .banner-converge { background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; border-radius:12px; padding:0.75rem 1rem; margin-bottom:0.85rem; font-size:0.88rem; display:flex; align-items:center; line-height:1.45; }
    .banner-diverge { background:#fef2f2; border:1px solid #fecaca; color:#991b1b; border-radius:12px; padding:0.75rem 1rem; margin-bottom:0.85rem; font-size:0.88rem; display:flex; align-items:center; line-height:1.45; }
    .banner-badge { font-weight:700; text-transform:uppercase; font-size:0.72rem; letter-spacing:0.05em; padding:2px 7px; border-radius:6px; margin-right:8px; flex-shrink:0; }
    .banner-converge .banner-badge { background:#dcfce7; color:#15803d; }
    .banner-diverge .banner-badge { background:#fee2e2; color:#b91c1c; }
    .banner-llm-warn { background:#fffbeb; border:1px solid #fde68a; color:#92400e; border-radius:12px; padding:0.85rem 1rem; margin-bottom:0.85rem; font-size:0.88rem; line-height:1.5; }
    .banner-llm-warn strong { display:block; margin-bottom:0.25rem; }
    details.appendix { margin-top:0.25rem; border:1px solid var(--line); border-radius:12px; background:var(--card); padding:0 1rem 0.75rem; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
    details.appendix summary { cursor:pointer; font-weight:700; font-size:0.92rem; color:#374151; padding:0.85rem 0; list-style:none; }
    details.appendix summary::-webkit-details-marker { display:none; }
    .appendix-inner { padding-bottom:0.5rem; }
    .appendix-inner .card { margin-top:0.75rem; }
    .data-table th { text-align:left; border-bottom:1px solid #e5e7eb; padding:6px 8px; color:var(--muted); font-weight:600; font-size:0.72rem; text-transform:uppercase; }
    .data-table td { padding:6px 8px; border-bottom:1px solid #eef1f4; }
    .sig-RED { color:#b91c1c; font-weight:600; }
    .sig-YELLOW { color:#92400e; font-weight:600; }
    .sig-GREEN { color:#047857; font-weight:600; }
    .deep-iframe { width:100%; border:1px solid var(--line); border-radius:10px; min-height:420px; background:#fff; margin-top:0.5rem; }
    .iframe-caption { font-size:0.78rem; color:var(--muted); margin:0.5rem 0 0.25rem; font-weight:600; }
    .tab-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }
    .tab-btn { border:1px solid var(--line); background:#fff; border-radius:8px; padding:6px 10px; font-size:0.78rem; cursor:pointer; }
    .tab-btn.active { border-color:#4f46e5; box-shadow:inset 0 0 0 1px #4f46e5; font-weight:700; }
    .country-panel { display:none; }
    .country-panel.active { display:block; }
    .sub-tab-row { display:flex; gap:8px; flex-wrap:wrap; margin:8px 0; }
    .sub-tab-btn { border:1px solid var(--line); background:#fff; border-radius:8px; padding:5px 9px; font-size:0.76rem; cursor:pointer; }
    .sub-tab-btn.active { border-color:#4f46e5; box-shadow:inset 0 0 0 1px #4f46e5; font-weight:700; }
    .sub-panel { display:none; }
    .sub-panel.active { display:block; }
    .zone-overview { font-size:0.93rem; line-height:1.75; color:#1f2937; }
    .zone-watchout { background:#fffbeb; border:1px solid #fde68a; border-radius:12px; padding:1rem 1.15rem; margin-bottom:0.9rem; }
    .zone-watchout ul { margin:0.5rem 0 0 1.1rem; font-size:0.9rem; line-height:1.6; }
    .spread-trade-line { font-size:0.88rem; font-weight:600; color:#374151; margin:0.75rem 0; }
    /* Theme cards */
    .theme-hero { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px,1fr)); gap:1rem; margin-bottom:1rem; }
    .theme-card { background:#fff; border:1px solid #e0e7ff; border-left:4px solid #4f46e5; border-radius:12px; padding:1.1rem 1.2rem; }
    .theme-card-title { font-size:1rem; font-weight:700; color:#1e1b4b; margin:0 0 0.55rem; }
    .theme-card-body { font-size:0.88rem; line-height:1.7; color:#374151; margin:0 0 0.6rem; }
    .theme-card-implication { font-size:0.82rem; color:#4f46e5; font-weight:600; margin:0 0 0.6rem; }
    .theme-country-badges { display:flex; gap:5px; flex-wrap:wrap; }
    .theme-country-badge { font-size:0.72rem; background:#ede9fe; color:#4338ca; border-radius:5px; padding:2px 7px; font-weight:600; }
    /* Evidence layer label */
    .evidence-label { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:#9ca3af; font-weight:700; margin-bottom:0.5rem; }
    /* VS US alignment badge */
    .vs-us-bar { display:flex; align-items:center; gap:0.5rem; margin-bottom:0.9rem; padding:0.5rem 0.75rem; border-radius:8px; font-size:0.82rem; }
    .vs-us-bar.aligned { background:#f0fdf4; border:1px solid #bbf7d0; }
    .vs-us-bar.diverging { background:#fff7ed; border:1px solid #fed7aa; }
    .vs-us-bar.mixed { background:#fefce8; border:1px solid #fde68a; }
    .vs-us-bar.limited { background:#f8fafc; border:1px solid #e2e8f0; }
    .vs-us-tag { font-weight:700; }
    .vs-us-bar.aligned .vs-us-tag { color:#15803d; }
    .vs-us-bar.diverging .vs-us-tag { color:#c2410c; }
    .vs-us-bar.mixed .vs-us-tag { color:#92400e; }
    .vs-us-bar.limited .vs-us-tag { color:#94a3b8; }
    .vs-us-detail { color:#64748b; }
    @media (max-width:900px) { .dual-col, .conflict-matrix { grid-template-columns:1fr; } }
    /* Historical analogues */
    .analogue-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px,1fr)); gap:0.85rem; margin-top:0.75rem; }
    .analogue-card { border:1px solid var(--line); border-radius:10px; padding:0.9rem 1rem; background:#fafbfc; }
    .analogue-period { font-size:0.85rem; font-weight:700; color:#111827; margin:0 0 0.2rem; font-family:'JetBrains Mono', ui-monospace, monospace; }
    .analogue-desc { font-size:0.8rem; color:#6b7280; margin:0 0 0.75rem; line-height:1.45; }
    .analogue-returns { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.4rem; }
    .analogue-stat { text-align:center; padding:0.4rem 0.3rem; border-radius:7px; }
    .analogue-stat-label { font-size:0.65rem; text-transform:uppercase; letter-spacing:0.06em; color:#9ca3af; font-weight:700; display:block; margin-bottom:0.2rem; }
    .analogue-stat-val { font-size:0.95rem; font-weight:800; font-family:'JetBrains Mono', ui-monospace, monospace; display:block; }
    .analogue-stat.pos { background:#f0fdf4; }
    .analogue-stat.pos .analogue-stat-val { color:#047857; }
    .analogue-stat.neg { background:#fef2f2; }
    .analogue-stat.neg .analogue-stat-val { color:#b91c1c; }
    .analogue-stat.na { background:#f9fafb; }
    .analogue-stat.na .analogue-stat-val { color:#9ca3af; }
    /* Regime asset tilt grid */
    .tilt-grid { display:flex; flex-wrap:wrap; gap:0.5rem; }
    .tilt-item { display:flex; align-items:center; gap:0.35rem; padding:0.3rem 0.65rem; border-radius:20px; font-size:0.78rem; border:1px solid transparent; }
    .tilt-item.tilt-overweight  { background:#f0fdf4; border-color:#bbf7d0; color:#14532d; }
    .tilt-item.tilt-neutral     { background:#f9fafb; border-color:#e5e7eb; color:#4b5563; }
    .tilt-item.tilt-underweight { background:#fef2f2; border-color:#fecaca; color:#7f1d1d; }
    .tilt-icon { font-size:0.75rem; }
    .tilt-asset { font-weight:600; }
    .tilt-label { font-size:0.68rem; opacity:0.75; text-transform:uppercase; letter-spacing:0.04em; }
    .analogue-sub-label { font-size:0.78rem; font-weight:700; color:#374151; margin:0.5rem 0 0.3rem; }
    .analogue-sub-desc { font-size:0.8rem; color:#6b7280; margin:0 0 0.6rem; line-height:1.5; }
    .analogue-disclaimer { font-size:0.73rem; color:#9ca3af; margin-top:0.65rem; font-style:italic; }
  </style>
</head>
<body>
  {%- macro muted_if_placeholder(text) -%}
  {%- set t = (text or '')|trim|lower -%}
  {%- if not t or t == '—' or t == fallback_text|lower or 'not available' in t or 'not available' in t -%} macra-placeholder{%- endif -%}
  {%- endmacro -%}
  {%- macro fmt_val(v) -%}
  {%- if v is none or (v|string)|trim|lower in ['none','null','nan',''] -%}—{%- else -%}{{ v }}{%- endif -%}
  {%- endmacro -%}
  <div class="wrap">
    <section class="card">
      <div class="header-row">
        <div>
          <h1 class="title">The Ephemerist · 戰略綜合 (Strategic Synthesis)</h1>
          <div class="meta">Report {{ report_date }} · LLM {{ llm_meta.provider }} ({{ llm_meta.status }})</div>
        </div>
        <div style="display:flex; flex-wrap:wrap; align-items:flex-start; gap:1rem;">
          <div class="divergence-badge gauge-{{ regime_gauge_tone }}">
            <span class="db-label">Macro Regime <span style="font-weight:400;opacity:0.7;">(US-led)</span></span>
            <span class="db-score" style="font-size:1.05rem;">{{ macro_regime_label }}</span>
          </div>
          <div>
            <div class="divergence-badge gauge-{{ divergence_gauge.tone }}">
              <span class="db-label">訊號一致性 · Signal Coherence</span>
              <span class="db-score">{{ divergence_gauge.score }}%</span>
              <span class="db-level">{{ divergence_gauge.level }}</span>
            </div>
            <div class="meta" style="margin-top:0.35rem; text-align:center; max-width:160px; line-height:1.3;">
              {% if divergence_gauge.score >= 50 %}各市場訊號分歧，表面與底層不一致
              {% elif divergence_gauge.score >= 30 %}部分訊號出現矛盾，需留意
              {% else %}各市場訊號一致，方向清晰
              {% endif %}{% if penalty_reason %} · {{ penalty_reason }}{% endif %}
            </div>
          </div>
          </div>
        </div>
      </div>
    {% if regime_asset_tilt %}
    <section class="card" style="padding:0.75rem 1.15rem;">
      <div class="zone-label" style="margin-bottom:0.5rem;">資產配置傾向 · Regime Asset Tilt <span style="font-weight:400;opacity:0.6;font-size:0.65rem;">(Bridgewater framework · {{ macro_regime_label }})</span></div>
      <div class="tilt-grid">
        {% for item in regime_asset_tilt %}
        <div class="tilt-item tilt-{{ item.tilt }}">
          <span class="tilt-icon">{{ item.icon }}</span>
          <span class="tilt-asset">{{ item.asset }}</span>
          <span class="tilt-label">{{ item.tilt }}</span>
        </div>
        {% endfor %}
      </div>
    </section>
    {% endif %}

    {% if landing_scenario and landing_scenario.mode %}
    <section class="card landing-card" style="padding:0.75rem 1.15rem;">
      {% if landing_scenario.mode == 'expansion' %}
      <div class="zone-label" style="margin-bottom:0.5rem;">擴張動能 · Expansion Momentum
        <span style="font-weight:400;opacity:0.6;font-size:0.65rem;">G+ Regime</span>
      </div>
      <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;">
        <div style="flex:1;background:#e5e7eb;border-radius:4px;height:8px;overflow:hidden;">
          <div style="width:{{ landing_scenario.bar_pct }}%;height:100%;background:#22c55e;border-radius:4px;"></div>
        </div>
        <span style="font-size:0.85rem;font-weight:700;color:#15803d;white-space:nowrap;">{{ landing_scenario.strength_zh }} · {{ landing_scenario.strength }}</span>
      </div>
      {% elif landing_scenario.mode == 'landing' %}
      <div class="zone-label" style="margin-bottom:0.5rem;">著陸情境 · Landing Scenario
        <span style="font-weight:400;opacity:0.6;font-size:0.65rem;">信心 {{ landing_scenario.confidence_zh }} · {{ landing_scenario.confidence }}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-bottom:0.6rem;">
        <div>
          <div style="font-size:0.7rem;opacity:0.6;margin-bottom:2px;">軟著陸 · Soft Landing</div>
          <div style="display:flex;align-items:center;gap:0.4rem;">
            <div style="flex:1;background:#e5e7eb;border-radius:4px;height:7px;overflow:hidden;">
              <div style="width:{{ landing_scenario.soft_pct }}%;height:100%;background:#22c55e;border-radius:4px;"></div>
            </div>
            <span style="font-size:0.85rem;font-weight:700;color:#15803d;">{{ landing_scenario.soft_pct }}%</span>
          </div>
        </div>
        <div>
          <div style="font-size:0.7rem;opacity:0.6;margin-bottom:2px;">硬著陸 · Hard Landing</div>
          <div style="display:flex;align-items:center;gap:0.4rem;">
            <div style="flex:1;background:#e5e7eb;border-radius:4px;height:7px;overflow:hidden;">
              <div style="width:{{ landing_scenario.hard_pct }}%;height:100%;background:#ef4444;border-radius:4px;"></div>
            </div>
            <span style="font-size:0.85rem;font-weight:700;color:#dc2626;">{{ landing_scenario.hard_pct }}%</span>
          </div>
        </div>
      </div>
      <div style="font-size:0.75rem;font-weight:700;margin-bottom:0.25rem;color:{% if landing_scenario.label == 'Soft Landing' %}#15803d{% elif landing_scenario.label == 'Hard Landing' %}#dc2626{% else %}#92400e{% endif %};">
        {{ landing_scenario.label_zh }} · {{ landing_scenario.label }}
      </div>
      {% endif %}
      {% if landing_scenario.drivers %}
      <ul style="margin:0;padding-left:1.1rem;font-size:0.72rem;opacity:0.75;line-height:1.6;">
        {% for d in landing_scenario.drivers %}
        <li>{{ d }}</li>
        {% endfor %}
      </ul>
      {% endif %}
    </section>
    {% endif %}

    {% if llm_placeholder %}
    <div class="banner-llm-warn">
      <strong>CIO narrative unavailable</strong>
      Run synthesis without <span class="mono">--skip-llm</span>. {% if llm_meta.error %}Detail: {{ llm_meta.error }}{% endif %}
    </div>
    {% endif %}

    {% if daily_themes %}
    <section class="card" style="padding-bottom:0.5rem;">
      <div class="zone-label" style="color:#4f46e5; margin-bottom:0.75rem;">今日主題 · Today's Themes</div>
      <div class="theme-hero">
        {% for theme in daily_themes %}
        <div class="theme-card">
          <div class="theme-card-title">{{ theme.title }}</div>
          <p class="theme-card-body">{{ theme.body }}</p>
          <p class="theme-card-implication">長線意義：{{ theme.implication }}</p>
          {% if theme.supporting_countries %}
          <div class="theme-country-badges">
            {% for c in theme.supporting_countries %}
            <span class="theme-country-badge">{{ c }}</span>
            {% endfor %}
          </div>
          {% endif %}
        </div>
        {% endfor %}
      </div>
    </section>
    {% endif %}

    <section class="card zone zone-watchout">
      <div class="zone-label">Critical Watchout</div>
      <ul>
        {% for item in critical_watchout %}
        <li class="{{ muted_if_placeholder(item) }}">{{ item }}</li>
        {% endfor %}
      </ul>
    </section>

    <section class="card zone zone-2">
      <div class="zone-label">CIO Executive Directive &amp; Top Spread Trade</div>
      <p class="spread-trade-line">{{ spread_trade_line }}</p>
      <p class="stance-line">{{ directive.the_stance }}</p>
      {% for para in directive.the_narrative %}
      <p class="zone-prose narrative{{ muted_if_placeholder(para) }}">{{ para }}</p>
      {% endfor %}
      <div class="watchout-box">
        <div class="watchout-label">The Watchout</div>
        {{ directive.the_watchout }}
      </div>
      {% if directive.positioning_6_12m %}
      <div class="positioning-box">
        <div class="watchout-label">6–12 個月配置傾向 · Portfolio Tilt</div>
        {{ directive.positioning_6_12m }}
      </div>
      {% endif %}
    </section>

    {% if historical_matches or divergence_matches %}
    <section class="card">
      <div class="zone-label">歷史借鏡 · Historical Analogues</div>

      {% if historical_matches %}
      <div class="analogue-sub-label">🇺🇸 美國體制借鏡 — US Macro Regime</div>
      <p class="analogue-sub-desc">當前美國宏觀格局（通膨、就業、利率、信用、波動度）最相似的歷史時期，以及那之後美國股市的表現。</p>
      <div class="analogue-grid">
        {% for m in historical_matches %}
        {% set r3 = m.spy_return_3m %}
        {% set r12 = m.spy_return_12m %}
        {% set dd = m.max_drawdown_12m %}
        <div class="analogue-card">
          <div class="analogue-period">{{ m.date }}</div>
          <div class="analogue-desc">{{ m.description }}</div>
          <div class="analogue-returns">
            <div class="analogue-stat {% if r3 is none %}na{% elif r3 >= 0 %}pos{% else %}neg{% endif %}">
              <span class="analogue-stat-label">US stocks 3m</span>
              <span class="analogue-stat-val">{% if r3 is none %}—{% else %}{{ "%+.1f"|format(r3) }}%{% endif %}</span>
            </div>
            <div class="analogue-stat {% if r12 is none %}na{% elif r12 >= 0 %}pos{% else %}neg{% endif %}">
              <span class="analogue-stat-label">US stocks 12m</span>
              <span class="analogue-stat-val">{% if r12 is none %}—{% else %}{{ "%+.1f"|format(r12) }}%{% endif %}</span>
            </div>
            <div class="analogue-stat {% if dd is none %}na{% else %}neg{% endif %}">
              <span class="analogue-stat-label">Max DD</span>
              <span class="analogue-stat-val">{% if dd is none %}—{% else %}{{ "%.1f"|format(dd) }}%{% endif %}</span>
            </div>
          </div>
        </div>
        {% endfor %}
      </div>
      {% endif %}

      {% if divergence_matches %}
      <div class="analogue-sub-label" style="margin-top:1.1rem;">🌐 全球分歧借鏡 — Cross-Country Divergence</div>
      <p class="analogue-sub-desc">當前多國政策利率差距、通膨分歧、FX 走勢、資金流向格局，最相似的歷史時期。這種分歧格局下，新興市場、黃金、銅的後續表現如何。</p>
      <div class="analogue-grid">
        {% for m in divergence_matches %}
        <div class="analogue-card">
          <div class="analogue-period">{{ m.date }}</div>
          <div class="analogue-desc">{{ m.description }}</div>
          <div class="analogue-returns" style="grid-template-columns:1fr 1fr 1fr;">
            {% for asset, label in [('eem','EM stocks'),('gld','Gold'),('copper','Copper')] %}
            {% set r12 = m.get(asset ~ '_12m') if m.get is defined else m[asset ~ '_12m'] %}
            <div class="analogue-stat {% if r12 is none %}na{% elif r12 >= 0 %}pos{% else %}neg{% endif %}">
              <span class="analogue-stat-label">{{ label }} 12m</span>
              <span class="analogue-stat-val">{% if r12 is none %}—{% else %}{{ "%+.1f"|format(r12) }}%{% endif %}</span>
            </div>
            {% endfor %}
          </div>
        </div>
        {% endfor %}
      </div>
      {% endif %}

      <div class="analogue-disclaimer">* 歷史數據僅供脈絡參考，不代表未來表現。美國體制借鏡以通膨、就業、利率、信用、波動度計算；全球分歧借鏡以多國政策利差、通膨差距、FX 走勢、EM 資金流向計算。</div>
    </section>
    {% endif %}

    <section class="macra-card macra-matrix-shell zone zone-3">
      <div class="macra-matrix-shell-head logic-head">
        <div>
          <div class="evidence-label">國家佐證層 · Country Evidence</div>
          <div class="zone-label" style="margin:0;">3-Gear Analysis per Country</div>
        </div>
        {% set rel_status = relationship.status or '' %}
        <span class="status-badge {% if 'DIVERG' in rel_status|upper %}diverge{% else %}converge{% endif %}">{{ rel_status }}</span>
      </div>
      <div class="macra-tablist" role="tablist">
        {% for tab in matrix_tabs %}
        <button type="button" class="macra-tab {% if tab.active %}macra-tab-active{% endif %}" data-country="{{ tab.country }}">{{ tab.label }}</button>
        {% endfor %}
      </div>
      <div class="macra-tab-panel">
      {% for panel in country_matrix %}
      <div class="macra-country-panel country-matrix-panel" data-country="{{ panel.country }}"{% if not panel.active %} style="display:none;"{% endif %}>
        {% set align = vs_us_alignment.get(panel.country) if vs_us_alignment else none %}
        {% if align %}
        <div class="vs-us-bar {{ align.alignment }}">
          <span class="vs-us-tag">{{ align.label }}</span>
          {% if align.key_divergence %}
          <span class="vs-us-detail">· {{ align.key_divergence }}</span>
          {% elif align.alignment == 'aligned' %}
          <span class="vs-us-detail">· 與美國 Goldilocks 同向</span>
          {% endif %}
        </div>
        {% endif %}
        {% for gear in panel.gears %}
        <div class="macra-gear-cell">
          <h4 class="macra-gear-title">{{ gear.title }}</h4>
          <p class="macra-col-label">結構面 · Structural backdrop</p>
          <p class="macra-col-body{{ muted_if_placeholder(gear.l1) }}">{{ gear.l1 }}</p>
          <p class="macra-col-label">市場面 · Market action</p>
          <p class="macra-col-body{{ muted_if_placeholder(gear.l2) }}">{{ gear.l2 }}</p>
          <p class="macra-col-label">近期動態 · Recent move</p>
          <p class="macra-col-body{{ muted_if_placeholder(gear.recent) }}">{{ gear.recent }}</p>
          <p class="macra-col-label">綜合研判 · What it means</p>
          <p class="macra-col-body{{ muted_if_placeholder(gear.cio) }}">{{ gear.cio }}</p>
        </div>
        {% endfor %}
      </div>
      {% endfor %}
      </div>
    </section>
  </div>
  <script>
    (function () {
      const matrixTabs = Array.from(document.querySelectorAll('.macra-tab'));
      const matrixPanels = Array.from(document.querySelectorAll('.country-matrix-panel'));
      matrixTabs.forEach((btn) => {
        btn.addEventListener('click', () => {
          const key = btn.getAttribute('data-country');
          matrixTabs.forEach((b) => b.classList.remove('macra-tab-active'));
          matrixPanels.forEach((p) => { p.style.display = 'none'; });
          btn.classList.add('macra-tab-active');
          const panel = document.querySelector('.country-matrix-panel[data-country="' + key + '"]');
          if (panel) panel.style.display = 'grid';
        });
      });
    })();
  </script>
</body>
</html>
"""


REGIME_LABEL_ZH: dict[str, str] = {
    "Goldilocks": "金絲雀",
    "Overheat": "過熱",
    "Stagflation": "停滯通膨",
    "Deflationary Bust": "通縮崩潰",
}

ASSET_LABEL_ZH: dict[str, str] = {
    "Equities": "股票",
    "Credit": "信用債",
    "Real Assets": "實質資產",
    "Long Bonds": "長債",
    "Cash": "現金",
    "Gold": "黃金",
    "Commodities": "大宗商品",
    "EM Assets": "新興市場資產",
}

COUNTRY_NAME_ZH: dict[str, str] = {
    "US": "美國",
    "Japan": "日本",
    "Europe": "歐洲",
    "UK": "英國",
    "China": "中國",
    "Taiwan": "台灣",
}

DEFAULT_L3_GEAR_TITLES_ZH = (
    "預期差套利",
    "擁擠與槓桿",
    "避險成本異常",
)

LITE_UI: dict[str, dict[str, str]] = {
    "zh": {
        "subtitle": "戰略綜合",
        "today_line_kicker": "今日一句",
        "zone_us_anchor": "美國基準",
        "zone_us_anchor_sub": "全球定價的錨與開關 — 美國週期、金融條件、資產配置",
        "zone_global": "國際承接面",
        "zone_global_sub": "各國相對美國 — 利差、資本流向、利差交易（美國為原點）",
        "zone_synthesis": "今日主題",
        "zone_synthesis_sub": "今天最重要的主題與風險",
        "regime_k": "市場體制（美國主導）",
        "coherence_k": "訊號一致性",
        "regime_stable": "成長與通膨環境穩定",
        "regime_warning": "環境轉折，留意風險",
        "regime_risk": "高風險體制",
        "regime_neutral": "體制中性",
        "coherence_high": "表面與底層不一致",
        "coherence_mid": "部分訊號矛盾",
        "coherence_low": "方向清晰一致",
        "asset_tilt_head": "資產配置傾向",
        "tilt_legend": "↑↑ 強加碼　↑ 加碼　— 中性　↓ 減碼　↓↓ 強減碼",
        "tilt_legend_suffix": "排名為信號 · 百分比為該體制歷史月報酬年化參考（Greetham 2004 投資時鐘 · 你的資料估計）",
        "themes_head": "今日主題",
        "implication_label": "長線意義",
        "bottom_line_head": "研判底線",
        "positioning_lab": "6–12 個月配置傾向",
        "watch_head": "關鍵觀察",
        "cio_summary": "完整研判分析",
        "watchout_label": "觸發風險",
        "zone_history": "歷史借鏡",
        "zone_history_sub": "最相似歷史環境與其後資產表現",
        "history_summary": "歷史借鏡",
        "history_us_sub": "美國體制借鏡",
        "history_us_sub2": "最相似歷史環境與其後美股表現",
        "history_div_sub": "全球分歧借鏡",
        "history_div_sub2": "多國分歧下 EM／黃金／銅表現",
        "history_disc": "歷史數據僅供脈絡參考，不代表未來表現。",
        "history_disc_fallback": "部分描述未能翻譯，仍顯示英文原文。",
        "spy_12m": "美股 12m",
        "country_summary": "各國佐證",
        "country_suffix": "國",
        "full_link": "→ 看完整版（含 3-gear 細節）",
        "footer": "Translating data into macro narratives",
        "details_open": "展開 +",
        "details_close": "收起 −",
        "lang_zh": "中文",
        "lang_en": "EN",
        "asset_em": "EM",
        "asset_gold": "金",
        "asset_copper": "銅",
    },
    "en": {
        "subtitle": "Strategic Synthesis",
        "today_line_kicker": "In one line",
        "zone_us_anchor": "US Anchor",
        "zone_us_anchor_sub": "The global pricing anchor & switch — US cycle, financial conditions, allocation",
        "zone_global": "Global Relative",
        "zone_global_sub": "Each market vs the US — rate differentials, capital flows, carry (US is the origin)",
        "zone_synthesis": "Today's Themes",
        "zone_synthesis_sub": "The themes and risks that matter most today",
        "regime_k": "Market Regime (US-led)",
        "coherence_k": "Signal Coherence",
        "regime_stable": "Growth and inflation environment stable",
        "regime_warning": "Regime shifting — watch risks",
        "regime_risk": "High-risk regime",
        "regime_neutral": "Neutral regime",
        "coherence_high": "Surface and underlying signals diverge",
        "coherence_mid": "Some signals in conflict",
        "coherence_low": "Signals aligned and clear",
        "asset_tilt_head": "Asset Allocation Tilt",
        "tilt_legend": "↑↑ Strong OW　↑ OW　— Neutral　↓ UW　↓↓ Strong UW",
        "tilt_legend_suffix": "Ranking is the signal · % = annualised mean monthly return in this regime, for reference (Greetham 2004 Investment Clock · estimated from your data)",
        "themes_head": "Today's Briefing",
        "implication_label": "Long-term takeaway",
        "bottom_line_head": "The Bottom Line",
        "positioning_lab": "6–12 month positioning",
        "watch_head": "What to Watch",
        "cio_summary": "Narrative",
        "watchout_label": "Trigger risk",
        "zone_history": "Historical Analogues",
        "zone_history_sub": "Closest historical setups and subsequent asset returns",
        "history_summary": "Historical Analogues",
        "history_us_sub": "US regime analogues",
        "history_us_sub2": "closest historical setups and subsequent S&P returns",
        "history_div_sub": "Global divergence analogues",
        "history_div_sub2": "EM / gold / copper after multi-country splits",
        "history_disc": "Historical data for context only; not a forecast.",
        "history_disc_fallback": "Some descriptions could not be translated and remain in English.",
        "spy_12m": "S&P 12m",
        "country_summary": "Country Evidence",
        "country_suffix": " markets",
        "full_link": "→ Full version (3-gear detail)",
        "footer": "Translating data into macro narratives",
        "details_open": "Expand +",
        "details_close": "Collapse −",
        "lang_zh": "中文",
        "lang_en": "EN",
        "asset_em": "EM",
        "asset_gold": "Gold",
        "asset_copper": "Copper",
    },
}


def _bilingual_pair(record: dict, base_key: str, *, zh_fallback_en: bool = False) -> tuple[str, str]:
    """Return (en, zh) from _en/_zh suffixed or legacy single field."""
    en = str(record.get(f"{base_key}_en") or record.get(base_key) or "").strip()
    zh = str(record.get(f"{base_key}_zh") or "").strip()
    if not en or en.lower() in {"none", "n/a"}:
        en = FALLBACK
    if not zh:
        zh = en if zh_fallback_en and en != FALLBACK else FALLBACK
    return en, zh


def _bilingual_narrative(record: dict, base_key: str = "the_narrative") -> tuple[list[str], list[str]]:
    en_raw = record.get(f"{base_key}_en")
    if en_raw is None:
        en_raw = record.get(base_key)
    zh_raw = record.get(f"{base_key}_zh")
    en = [_sanitize_jargon(p) for p in _coerce_narrative(en_raw)]
    zh = [_sanitize_jargon(p) for p in _coerce_narrative(zh_raw)] if zh_raw is not None else []
    if not zh or all(p == FALLBACK for p in zh):
        zh = list(en)
    if not en:
        en = [FALLBACK]
    if not zh:
        zh = [FALLBACK]
    return en, zh


def _regime_label_display(label: str, lang: str) -> str:
    key = str(label or "").strip()
    if lang == "zh":
        return REGIME_LABEL_ZH.get(key, key)
    return key


def _regime_subtext_ui(tone: str, ui: dict[str, str]) -> str:
    if tone == "stable":
        return ui["regime_stable"]
    if tone == "warning":
        return ui["regime_warning"]
    if tone == "risk":
        return ui["regime_risk"]
    return ui["regime_neutral"]


def _regime_asset_tilt_for_lang(label: str, lang: str) -> list[dict]:
    rows = _regime_asset_tilt(label)
    out: list[dict] = []
    for row in rows:
        asset_en = str(row.get("asset") or "")
        item = dict(row)
        item["asset"] = ASSET_LABEL_ZH.get(asset_en, asset_en) if lang == "zh" else asset_en
        out.append(item)
    return out


def _country_display_name(country: str, lang: str) -> str:
    if lang == "zh":
        return COUNTRY_NAME_ZH.get(country, country)
    return country


def _alignment_label_for_lang(align: dict | None, lang: str) -> str:
    if not align:
        return ""
    if lang == "zh":
        return str(align.get("label_zh") or align.get("label") or "")
    return str(align.get("label_en") or align.get("label") or "")


def _normalize_daily_themes_bilingual(themes: list) -> list[dict]:
    out: list[dict] = []
    for raw in themes or []:
        if not isinstance(raw, dict):
            continue
        title_en, title_zh = _bilingual_pair(raw, "title")
        body_en, body_zh = _bilingual_pair(raw, "body")
        imp_en, imp_zh = _bilingual_pair(raw, "implication")
        reason_en, reason_zh = _bilingual_pair(raw, "reasoning")
        title_en, body_en = _derive_theme_title(title_en, body_en)
        if not title_zh or title_zh == FALLBACK:
            title_zh = title_en
        theme = {
            "title_en": _sanitize_jargon(title_en),
            "title_zh": _sanitize_jargon(title_zh),
            "title": _sanitize_jargon(title_en),
            "body_en": _sanitize_jargon(body_en),
            "body_zh": _sanitize_jargon(body_zh),
            "body": _sanitize_jargon(body_en),
            "implication_en": _sanitize_jargon(imp_en),
            "implication_zh": _sanitize_jargon(imp_zh),
            "implication": _sanitize_jargon(imp_en),
            "reasoning_en": _sanitize_jargon(reason_en) if reason_en and reason_en != FALLBACK else "",
            "reasoning_zh": _sanitize_jargon(reason_zh) if reason_zh and reason_zh != FALLBACK else "",
            "supporting_countries": raw.get("supporting_countries") or [],
        }
        out.append(theme)
    return out


def _normalize_cio_directive_bilingual(directive: dict) -> dict:
    stance_en, stance_zh = _bilingual_pair(directive, "the_stance")
    watch_en, watch_zh = _bilingual_pair(directive, "the_watchout")
    pos_en, pos_zh = _bilingual_pair(directive, "positioning_6_12m")
    pos_r_en, pos_r_zh = _bilingual_pair(directive, "positioning_6_12m_reasoning")
    narr_en, narr_zh = _bilingual_narrative(directive)
    out = dict(directive)
    out.update(
        {
            "the_stance_en": _sanitize_jargon(stance_en),
            "the_stance_zh": _sanitize_jargon(stance_zh),
            "the_stance": _sanitize_jargon(stance_en),
            "the_watchout_en": _sanitize_jargon(watch_en),
            "the_watchout_zh": _sanitize_jargon(watch_zh),
            "the_watchout": _sanitize_jargon(watch_en),
            "positioning_6_12m_en": _sanitize_jargon(pos_en),
            "positioning_6_12m_zh": _sanitize_jargon(pos_zh),
            "positioning_6_12m": _sanitize_jargon(pos_en),
            "positioning_6_12m_reasoning_en": _sanitize_jargon(pos_r_en) if pos_r_en and pos_r_en != FALLBACK else "",
            "positioning_6_12m_reasoning_zh": _sanitize_jargon(pos_r_zh) if pos_r_zh and pos_r_zh != FALLBACK else "",
            "the_narrative_en": narr_en,
            "the_narrative_zh": narr_zh,
            "the_narrative": narr_en,
        }
    )
    return out


def _normalize_watch_list_bilingual(wl: list) -> tuple[list[str], list[dict]]:
    """Return English list (legacy) and bilingual dict items."""
    bilingual: list[dict] = []
    for item in wl or []:
        if isinstance(item, dict):
            if "en" in item or "zh" in item:
                en = str(item.get("en") or "").strip()
                zh = str(item.get("zh") or "").strip()
            else:
                en, zh = _bilingual_pair(item, "text")
        else:
            en = str(item or "").strip()
            zh = FALLBACK
        en = _sanitize_jargon(en) if en else FALLBACK
        zh = _sanitize_jargon(zh) if zh and zh != FALLBACK else (FALLBACK if en == FALLBACK else en)
        r_en = r_zh = ""
        if isinstance(item, dict):
            r_en = _sanitize_jargon(str(item.get("reasoning_en") or "").strip())
            r_zh = _sanitize_jargon(str(item.get("reasoning_zh") or "").strip())
        bilingual.append({"en": en, "zh": zh, "reasoning_en": r_en, "reasoning_zh": r_zh})
    if not bilingual:
        bilingual = [{"en": FALLBACK, "zh": FALLBACK, "reasoning_en": "", "reasoning_zh": ""}]
    return [x["en"] for x in bilingual], bilingual


def _normalize_gear_block_bilingual(block: dict) -> dict:
    out = dict(block)
    for base in ("expectation", "plumbing", "hedging"):
        en, zh = _bilingual_pair(out, base)
        out[f"{base}_en"] = _sanitize_jargon(en)
        out[f"{base}_zh"] = _sanitize_jargon(zh)
        out[base] = out[f"{base}_en"]
    for base in ("expectation_label", "plumbing_label", "hedging_label"):
        en, zh = _bilingual_pair(out, base)
        out[f"{base}_en"] = en
        out[f"{base}_zh"] = zh
        out[base] = en
    return out


def _gear_semantic_text(semantics: dict, field: str, lang: str) -> str | None:
    suffix = "_zh" if lang == "zh" else "_en"
    val = semantics.get(f"{field}{suffix}") or semantics.get(field)
    text = str(val or "").strip()
    if not text or text == FALLBACK or text.lower().startswith("llm analysis"):
        return None
    return text


def _lite_country_matrix(
    country_matrix: list[dict],
    gear_semantics: dict,
    lang: str,
    vs_us_alignment: dict | None = None,
) -> list[dict]:
    semantic_keys = ("expectation", "plumbing", "hedging")
    label_keys = ("expectation_label", "plumbing_label", "hedging_label")
    default_titles = DEFAULT_L3_GEAR_TITLES_ZH if lang == "zh" else (
        "Expectation Arbitrage",
        "Crowdedness & Leverage",
        "Hedging Cost Anomaly",
    )
    panels: list[dict] = []
    for panel in country_matrix or []:
        if not isinstance(panel, dict):
            continue
        country = str(panel.get("country") or "")
        semantics = gear_semantics.get(country) if isinstance(gear_semantics, dict) else {}
        if not isinstance(semantics, dict):
            semantics = {}
        gears_out: list[dict] = []
        for idx, gear in enumerate(panel.get("gears") or []):
            if not isinstance(gear, dict):
                continue
            field = semantic_keys[idx] if idx < len(semantic_keys) else semantic_keys[0]
            label_field = label_keys[idx] if idx < len(label_keys) else label_keys[0]
            title = _gear_semantic_text(semantics, label_field, lang)
            if not title:
                title = default_titles[idx] if idx < len(default_titles) else gear.get("title", "")
            body = _gear_semantic_text(semantics, field, lang)
            if not body:
                body = str(gear.get("cio") or gear.get("l2") or FALLBACK)
            gears_out.append({"title": title, "body": body})
        align = (vs_us_alignment or {}).get(country) if vs_us_alignment else None
        panels.append(
            {
                "country": _country_display_name(country, lang),
                "country_code": country,
                "gears": gears_out,
                "align_label": _alignment_label_for_lang(align, lang),
                "align_class": str(align.get("alignment") or "") if isinstance(align, dict) else "",
            }
        )
    return panels


def _lite_directive_view(directive: dict, lang: str) -> dict:
    suffix = "_zh" if lang == "zh" else "_en"
    return {
        "the_stance": directive.get(f"the_stance{suffix}") or directive.get("the_stance", FALLBACK),
        "the_narrative": directive.get(f"the_narrative{suffix}") or directive.get("the_narrative", [FALLBACK]),
        "the_watchout": directive.get(f"the_watchout{suffix}") or directive.get("the_watchout", FALLBACK),
        "positioning_6_12m": directive.get(f"positioning_6_12m{suffix}") or directive.get("positioning_6_12m", ""),
        "positioning_6_12m_reasoning": directive.get(f"positioning_6_12m_reasoning{suffix}") or "",
    }


def _lite_compare_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", str(text or "").lower(), flags=re.UNICODE)


def _lite_theme_kicker(countries: list, lang: str) -> str:
    if not countries:
        return "市場" if lang == "zh" else "MARKETS"
    if lang == "zh":
        return " · ".join(_country_display_name(str(c), "zh") for c in countries)
    return " · ".join(str(c) for c in countries)


def _lite_positioning_text(directive_view: dict, themes: list[dict]) -> str | None:
    """Return 6–12m positioning only when it adds information beyond the first theme."""
    pos = str(directive_view.get("positioning_6_12m") or "").strip()
    if not pos or pos == FALLBACK:
        return None
    if not themes:
        return pos
    first_imp = str(themes[0].get("implication") or "").strip()
    if not first_imp:
        return pos
    pos_n, imp_n = _lite_compare_text(pos), _lite_compare_text(first_imp)
    if pos_n == imp_n:
        return None
    if len(pos_n) >= 48 and (pos_n in imp_n or imp_n in pos_n):
        return None
    return pos


def _lite_daily_themes(themes: list[dict], lang: str) -> list[dict]:
    suffix = "_zh" if lang == "zh" else "_en"
    out: list[dict] = []
    for t in themes or []:
        if not isinstance(t, dict):
            continue
        countries = t.get("supporting_countries") or []
        out.append(
            {
                "title": t.get(f"title{suffix}") or t.get("title", FALLBACK),
                "body": t.get(f"body{suffix}") or t.get("body", FALLBACK),
                "implication": t.get(f"implication{suffix}") or t.get("implication", ""),
                "reasoning": t.get(f"reasoning{suffix}") or "",
                "supporting_countries": countries,
                "kicker": _lite_theme_kicker(countries, lang),
            }
        )
    return out


def _watch_text_for_lang(item: object, lang: str) -> str:
    """Resolve one watch-list entry to a single-language string."""
    key = "zh" if lang == "zh" else "en"
    if isinstance(item, dict):
        return str(item.get(key) or item.get("en") or "").strip()
    text = str(item or "").strip()
    if text.startswith("{") and f"'{key}'" in text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return str(parsed.get(key) or parsed.get("en") or "").strip()
        except (SyntaxError, ValueError):
            pass
    return text


def _lite_critical_watchout(watch_bilingual: list[dict], lang: str) -> list[dict]:
    """Watch-list bullets (text + hover reasoning); trigger risk stays in the CIO narrative."""
    r_key = "reasoning_zh" if lang == "zh" else "reasoning_en"
    out: list[dict] = []
    for x in watch_bilingual:
        text = _watch_text_for_lang(x, lang)
        text = str(text or "").strip()
        if not text or text == FALLBACK:
            continue
        reasoning = ""
        if isinstance(x, dict):
            reasoning = str(x.get(r_key) or "").strip()
        out.append({"text": text, "reasoning": reasoning})
    return out


def _localize_historical_matches(matches: list[dict], descriptions_zh: list[str]) -> list[dict]:
    out: list[dict] = []
    for i, m in enumerate(matches or []):
        if not isinstance(m, dict):
            continue
        row = dict(m)
        desc_en = str(row.get("description") or "")
        row["description_en"] = desc_en
        row["description"] = descriptions_zh[i] if i < len(descriptions_zh) and descriptions_zh[i] else desc_en
        out.append(row)
    return out


def _translate_descriptions_to_zh(descriptions: list[str]) -> tuple[list[str], bool]:
    """Batch-translate short historical descriptions; returns (zh_list, used_fallback)."""
    clean = [str(d).strip() for d in descriptions if str(d).strip()]
    if not clean:
        return [], False
    prompt = (
        "Translate each string to Traditional Chinese (繁體中文) for retail investors. "
        "Keep proper nouns where natural. Return ONLY a JSON array of strings, same length and order.\n"
        + json.dumps(clean, ensure_ascii=False)
    )
    for caller in (_call_deepseek_translate, _call_gemini_translate):
        try:
            result = caller(prompt)
            if isinstance(result, list) and len(result) == len(clean):
                return [str(x) for x in result], False
        except Exception as exc:
            print(f"  Historical description translation failed: {exc}", file=sys.stderr)
    return list(clean), True


def _call_deepseek_translate(prompt: str) -> list[str] | None:
    client = OpenAI(api_key=require_deepseek_key(), base_url=get_deepseek_base_url())
    for model in _deepseek_models_to_try():
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = json.loads(_clean_json(raw))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            continue
    return None


def _call_gemini_translate(prompt: str) -> list[str] | None:
    client = genai.Client(api_key=require_gemini_key())
    for model in _gemini_models_to_try():
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.2),
            )
            raw = response.text or ""
            parsed = json.loads(_clean_json(raw))
            if isinstance(parsed, list):
                return parsed
        except Exception:
            continue
    return None


def _build_regime_quadrant_plotly() -> dict:
    """Plotly JSON data for interactive 2x2 Growth vs Inflation quadrant chart.

    Uses the same composite growth/inflation signals as the regime badge and
    asset-tilt engine (regime_tilt.compute_momentum_state) so the chart and the
    headline regime are always consistent.
    """
    from src.collect import load_all_from_cache
    from src.regime_tilt import compute_momentum_state

    mom = compute_momentum_state(load_all_from_cache())
    g = mom["growth_mom"].rename("growth_z")
    i = mom["infl_mom"].rename("inf_z")
    combined = pd.concat([g, i], axis=1).dropna().tail(13)

    dates = [d.strftime("%Y-%m") for d in combined.index]
    gx = combined["growth_z"].tolist()
    iy = combined["inf_z"].tolist()
    n = len(gx)

    # Trajectory (all but last) — colour fades from light grey to dark
    traj_colors = [f"rgba(120,120,120,{0.25 + 0.55 * i / max(n - 2, 1):.2f})" for i in range(n - 1)]

    traces = [
        # Trajectory line
        {
            "type": "scatter", "x": gx[:-1], "y": iy[:-1],
            "mode": "lines+markers",
            "line": {"color": "#AAAAAA", "width": 1.8, "dash": "dot"},
            "marker": {"size": 7, "color": traj_colors, "line": {"width": 0}},
            "text": dates[:-1],
            "hovertemplate": "<b>%{text}</b><br>Growth: %{x:.2f}σ<br>Inflation: %{y:.2f}σ<extra></extra>",
            "name": "12m Trajectory", "showlegend": True,
        },
        # Current point
        {
            "type": "scatter", "x": [gx[-1]], "y": [iy[-1]],
            "mode": "markers+text",
            "marker": {"size": 16, "color": "#E3120B", "line": {"color": "white", "width": 2}},
            "text": ["Now"], "textposition": "top right",
            "textfont": {"color": "#E3120B", "size": 12, "family": "Inter, sans-serif"},
            "hovertemplate": "<b>Now (%{customdata})</b><br>Growth: %{x:.2f}σ<br>Inflation: %{y:.2f}σ<extra></extra>",
            "customdata": [dates[-1]],
            "name": "Current", "showlegend": True,
        },
    ]

    layout = {
        "height": 340,
        "margin": {"l": 55, "r": 20, "t": 28, "b": 50},
        "paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff",
        "font": {"family": "Inter, sans-serif", "size": 11},
        "dragmode": False,
        "xaxis": {
            "title": "Growth (z-score)", "range": [-2.8, 2.8], "fixedrange": True,
            "zeroline": True, "zerolinecolor": "#333", "zerolinewidth": 1.5,
            "gridcolor": "#F0F0F0", "tickvals": [-2, -1, 0, 1, 2],
        },
        "yaxis": {
            "title": "Inflation (z-score)", "range": [-2.8, 2.8], "fixedrange": True,
            "zeroline": True, "zerolinecolor": "#333", "zerolinewidth": 1.5,
            "gridcolor": "#F0F0F0", "tickvals": [-2, -1, 0, 1, 2],
        },
        "legend": {"x": 0.99, "y": 0.01, "xanchor": "right", "yanchor": "bottom", "bgcolor": "rgba(255,255,255,0.8)", "borderwidth": 0},
        "shapes": [
            {"type": "rect", "x0": 0,   "x1": 2.8,  "y0": 0,    "y1": 2.8,  "fillcolor": "#dedad4", "opacity": 0.7, "line": {"width": 0}},
            {"type": "rect", "x0": -2.8,"x1": 0,    "y0": -2.8, "y1": 0,    "fillcolor": "#dedad4", "opacity": 0.7, "line": {"width": 0}},
            {"type": "rect", "x0": -2.8,"x1": 0,    "y0": 0,    "y1": 2.8,  "fillcolor": "#f4f3f0", "opacity": 0.5, "line": {"width": 0}},
            {"type": "rect", "x0": 0,   "x1": 2.8,  "y0": -2.8, "y1": 0,    "fillcolor": "#f4f3f0", "opacity": 0.5, "line": {"width": 0}},
        ],
        "annotations": [
            {"x": -1.8, "y": 2.5, "text": "Stagflation",      "showarrow": False, "font": {"size": 11, "color": "#9b958a", "family": "Inter"}},
            {"x": 1.8,  "y": 2.5, "text": "Overheat",        "showarrow": False, "font": {"size": 11, "color": "#9b958a", "family": "Inter"}},
            {"x": 0.85, "y": -2.5,"text": "Goldilocks",       "showarrow": False, "font": {"size": 11, "color": "#9b958a", "family": "Inter"}},
            {"x": -1.8, "y": -2.5,"text": "Deflationary Bust","showarrow": False, "font": {"size": 11, "color": "#9b958a", "family": "Inter"}},
        ],
    }
    return {"traces": traces, "layout": layout}


def _build_zscore_plotly(lang: str = "zh-Hant") -> dict:
    """
    Plotly JSON: 5-panel financial conditions monitor using actual historical KDE.

    Replaces the old theoretical-normal bell curves with real KDE so thick-tailed
    indicators (VIX, credit spreads) display their true right-skewed shape.
    Indicators are chosen to cover the cycle framework's blind spots — credit
    stress, policy tightness, currency, and market fear — NOT cycle inputs
    (CPI/unemployment are already inside the composite regime engine).

    Indicators:
      Credit Spreads HY  — credit stress / liquidity
      Yield Curve 2-10y  — recession signal / policy transmission
      Real Policy Rate   — fed funds − CPI (policy tightness)
      USD Index DXY      — global dollar liquidity
      VIX                — market fear / risk appetite
    """
    from scipy.stats import gaussian_kde  # noqa: PLC0415

    def _load_monthly_raw(fname: str) -> pd.Series:
        s = pd.read_csv(RAW_DIR / fname, index_col=0, parse_dates=True).squeeze()
        s.index = pd.to_datetime(s.index)
        return s.resample("ME").last().astype(float).dropna()

    ff        = _load_monthly_raw("fed_funds_rate.csv")
    cpi       = _load_monthly_raw("cpi_yoy.csv")
    real_rate = (ff - cpi).dropna()

    credit = _load_monthly_raw("credit_spread_hy.csv")
    curve  = _load_monthly_raw("spread_2y10y.csv")
    dxy    = _load_monthly_raw("dxy.csv")
    vix    = _load_monthly_raw("vix.csv")

    # (series, zh_label, en_label, stress_dir)
    # stress_dir: +1 = high is stress, -1 = low is stress, 0 = neutral
    _indicators = [
        (credit,    "信用利差 BAA",      "Credit Spreads BAA", +1),
        (curve,     "殖利率曲線 2-10y", "Yield Curve 2-10y",  -1),
        (real_rate, "實質政策利率",      "Real Policy Rate",    0),
        (dxy,       "美元指數 DXY",     "USD Index DXY",       0),
        (vix,       "VIX 恐慌指數",     "VIX",                +1),
    ]
    indicators = [(s, en if lang == "en" else zh, en, d) for s, zh, en, d in _indicators]

    traces: list[dict] = []
    annotations: list[dict] = []

    for idx, (series, zh_lbl, en_lbl, stress_dir) in enumerate(indicators):
        ax_suffix = "" if idx == 0 else str(idx + 1)
        xref = f"x{ax_suffix}"
        yref = f"y{ax_suffix}"

        vals = series.dropna().values
        cur  = float(vals[-1])

        # Actual historical percentile rank (exclude last point from history)
        hist_vals = vals[:-1] if len(vals) > 1 else vals
        pct = float((hist_vals < cur).mean() * 100)

        # Color by percentile + stress direction
        if stress_dir == +1:   # high = stress
            if pct > 85:   color = "#E3120B"
            elif pct > 70: color = "#6f6f6f"
            else:          color = "#4a7c59"
        elif stress_dir == -1: # low = stress (yield curve)
            if pct < 15:   color = "#E3120B"
            elif pct < 30: color = "#6f6f6f"
            else:          color = "#4a7c59"
        else:                  # neutral: extreme either direction
            if pct > 85 or pct < 15:   color = "#E3120B"
            elif pct > 75 or pct < 25: color = "#6f6f6f"
            else:                      color = "#4a7c59"

        r_c, g_c, b_c = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        # KDE over actual history
        kde = gaussian_kde(vals, bw_method="scott")
        pad = max((vals.max() - vals.min()) * 0.12, 1.0)
        x0, x1 = vals.min() - pad, vals.max() + pad
        xs_kde = [x0 + i * (x1 - x0) / 200 for i in range(201)]
        ys_kde = [float(kde([x])[0]) for x in xs_kde]
        y_peak = max(ys_kde) if ys_kde else 0.4

        # Percentile fill (shaded area left of current value)
        xs_left = [x for x in xs_kde if x <= cur]
        ys_left = [float(kde([x])[0]) for x in xs_left]
        if xs_left:
            xs_fill = [xs_left[0]] + xs_left + [xs_left[-1]]
            ys_fill = [0.0] + ys_left + [0.0]
        else:
            xs_fill, ys_fill = [], []

        # Full KDE curve (grey background)
        traces.append({
            "type": "scatter", "x": xs_kde, "y": ys_kde,
            "mode": "lines",
            "line": {"color": "#CCCCCC", "width": 1.5},
            "fill": "tozeroy", "fillcolor": "rgba(200,200,200,0.15)",
            "xaxis": xref, "yaxis": yref,
            "showlegend": False, "hoverinfo": "skip",
        })

        # Coloured percentile fill
        if xs_fill:
            traces.append({
                "type": "scatter", "x": xs_fill, "y": ys_fill,
                "mode": "lines", "line": {"width": 0},
                "fill": "tozeroy",
                "fillcolor": f"rgba({r_c},{g_c},{b_c},0.22)",
                "xaxis": xref, "yaxis": yref,
                "showlegend": False, "hoverinfo": "skip",
            })

        # Vertical marker line at current value
        marker_y = float(kde([cur])[0])
        # Format current value label
        cur_fmt = f"{cur:.1f}" if abs(cur) > 20 else f"{cur:+.2f}"
        if lang == "en":
            hover = f"<b>{en_lbl}</b><br>Current {cur_fmt}<br>{pct:.0f}th percentile<extra></extra>"
        else:
            hover = f"<b>{en_lbl}</b><br>現值 {cur_fmt}<br>歷史第 {pct:.0f} 百分位<extra></extra>"

        traces.append({
            "type": "scatter",
            "x": [cur, cur], "y": [0, marker_y],
            "mode": "lines",
            "line": {"color": color, "width": 2.5},
            "xaxis": xref, "yaxis": yref,
            "showlegend": False,
            "hovertemplate": hover,
        })
        traces.append({
            "type": "scatter",
            "x": [cur], "y": [marker_y],
            "mode": "markers",
            "marker": {"size": 8, "color": color},
            "xaxis": xref, "yaxis": yref,
            "showlegend": False,
            "hovertemplate": hover,
        })

        # Panel title
        annotations.append({
            "text": f"<b>{zh_lbl}</b>",
            "xref": f"{xref} domain", "yref": f"{yref} domain",
            "x": 0.5, "y": 1.2, "showarrow": False,
            "font": {"size": 10, "family": "Inter, sans-serif", "color": "#333"},
            "xanchor": "center",
        })
        # Value + percentile
        pct_label = f"{pct:.0f}th pct" if lang == "en" else f"第 {pct:.0f} 百分位"
        annotations.append({
            "text": (
                f"<span style='color:{color}'><b>{cur_fmt}</b></span>"
                f" · {pct_label}"
            ),
            "xref": f"{xref} domain", "yref": f"{yref} domain",
            "x": 0.5, "y": 1.05, "showarrow": False,
            "font": {"size": 8.5, "family": "Inter, sans-serif"},
            "xanchor": "center",
        })

    layout: dict = {
        "height": 200,
        "margin": {"l": 10, "r": 10, "t": 50, "b": 16},
        "paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff",
        "font": {"family": "Inter, sans-serif"},
        "showlegend": False,
        "dragmode": False,
        "annotations": annotations,
        "grid": {"rows": 1, "columns": 5, "pattern": "independent", "xgap": 0.07},
    }
    for i in range(5):
        sfx = "" if i == 0 else str(i + 1)
        layout[f"xaxis{sfx}"] = {
            "showticklabels": True,
            "tickfont": {"size": 7.5, "color": "#AAAAAA"},
            "zeroline": True, "zerolinecolor": "#E5E5E5", "zerolinewidth": 1,
            "showgrid": False, "nticks": 4,
            "tickformat": ".0f",
            "fixedrange": True,
        }
        layout[f"yaxis{sfx}"] = {
            "showticklabels": False, "showgrid": False, "zeroline": False,
            "fixedrange": True,
        }

    return {"traces": traces, "layout": layout}


_CARRY_COUNTRY_ZH = {"Japan": "日本", "Europe": "歐洲", "China": "中國", "Taiwan": "台灣"}


def _us_financial_conditions_summary() -> list[dict]:
    """
    Structured US financial conditions for the LLM prompt (圈層1 · US ANCHOR).

    Same 5 indicators as the visual monitor (`_build_zscore_plotly`), reduced to
    {indicator, value, percentile, read} so the model can reference them WITHOUT
    re-deriving. United-States-only — must never be applied to global/per-country.
    """
    def _pct(fname: str) -> tuple[float | None, int | None]:
        try:
            s = pd.read_csv(RAW_DIR / fname, index_col=0, parse_dates=True).squeeze()
            s.index = pd.to_datetime(s.index)
            m = s.resample("ME").last().astype(float).dropna()
            cur = float(m.iloc[-1])
            pct = int(round(float((m.iloc[:-1] < cur).mean()) * 100))
            return cur, pct
        except Exception:
            return None, None

    out: list[dict] = []
    # (file, label, stress_dir) — stress_dir: +1 high=stress, -1 low=stress, 0 neutral
    specs = [
        ("credit_spread_hy.csv", "信用利差 BAA", +1),
        ("spread_2y10y.csv", "殖利率曲線 2-10y", -1),
        ("dxy.csv", "美元指數 DXY", 0),
        ("vix.csv", "VIX", +1),
    ]
    for fname, label, sd in specs:
        cur, pct = _pct(fname)
        if cur is None:
            continue
        if sd == +1:
            read = "異常寬鬆/低" if pct < 20 else ("偏緊/高" if pct > 80 else "中性")
        elif sd == -1:
            read = "異常平坦/倒掛" if pct < 20 else ("陡峭" if pct > 80 else "中性")
        else:
            read = "偏強" if pct > 70 else ("偏弱" if pct < 30 else "中性")
        out.append({"indicator": label, "value": round(cur, 2), "percentile": pct, "read": read})
    # Real policy rate = fed funds − CPI
    try:
        ff = pd.read_csv(RAW_DIR / "fed_funds_rate.csv", index_col=0, parse_dates=True).squeeze()
        cpi = pd.read_csv(RAW_DIR / "cpi_yoy.csv", index_col=0, parse_dates=True).squeeze()
        for x in (ff, cpi):
            x.index = pd.to_datetime(x.index)
        rr = (ff.resample("ME").last() - cpi.resample("ME").last()).dropna()
        cur = float(rr.iloc[-1])
        pct = int(round(float((rr.iloc[:-1] < cur).mean()) * 100))
        read = "限制性" if cur > 1 else ("寬鬆" if cur < 0 else "中性")
        out.append({"indicator": "實質政策利率", "value": round(cur, 2), "percentile": pct, "read": read})
    except Exception:
        pass
    return out


def _build_carry_relative(l2b: dict) -> dict:
    """
    圈層2 · 國際承接面（Global Relative）.

    Each market RELATIVE TO the US anchor — rate differential, FX, capital flow,
    and the global carry switch (VIX). US is the origin, not a row here.

    Principle: ATTACH the raw pieces (spread + FX + flow + VIX switch); do NOT
    bake a single "carry attractiveness score" (that would repeat the coherence
    black-box mistake). Reader sees the parts and judges.

    All data is reused from l2b.flow_payload (no new sources). China has a capital
    account that is largely closed, so it is framed as policy-divergence / FX, not
    carry flow.
    """
    fp = (l2b or {}).get("flow_payload") or l2b or {}
    spreads = fp.get("bond_spreads_bp") or {}
    fx1 = fp.get("fx_1m_local_vs_usd") or {}

    # ── Global carry switch: VIX historical percentile (low = carry-on) ───────
    try:
        vix = pd.read_csv(RAW_DIR / "vix.csv", index_col=0, parse_dates=True).squeeze()
        vix.index = pd.to_datetime(vix.index)
        vix_m = vix.resample("ME").last().astype(float).dropna()
        vix_cur = float(vix_m.iloc[-1])
        vix_pct = int(round(float((vix_m.iloc[:-1] < vix_cur).mean()) * 100))
    except Exception:
        vix_cur, vix_pct = None, None

    if vix_pct is None:
        switch_zh, switch_en, switch_tone = "未知", "unknown", "neutral"
    elif vix_pct < 35:
        switch_zh, switch_en, switch_tone = "carry-on（低波動 · 套息活躍）", "carry-on (low vol)", "stable"
    elif vix_pct > 70:
        switch_zh, switch_en, switch_tone = "carry-off（高波動 · 平倉風險）", "carry-off (high vol)", "risk"
    else:
        switch_zh, switch_en, switch_tone = "中性（波動居中）", "neutral", "warning"

    # ── US anchor row: DXY 22d change + US 10Y yield ────────────────────────
    try:
        dxy = pd.read_csv(RAW_DIR / "dxy.csv", skiprows=1, header=None,
                          names=["date", "value"]).dropna()
        dxy_vals = dxy["value"].astype(float)
        dxy_cur = float(dxy_vals.iloc[-1])
        dxy_1m = float((dxy_cur / dxy_vals.iloc[-22] - 1) * 100) if len(dxy_vals) >= 22 else None
    except Exception:
        dxy_1m, dxy_cur = None, None
    try:
        dgs10 = pd.read_csv(RAW_DIR / "dgs10.csv", skiprows=1, header=None,
                            names=["date", "value"]).dropna()
        us10y = float(dgs10["value"].iloc[-1])
    except Exception:
        us10y = None

    us_flow = f"DXY {dxy_cur:.1f} · {dxy_1m:+.2f}%" if dxy_cur and dxy_1m is not None else None
    us_note_zh = f"全球定價錨 · 美債10Y {us10y:.2f}%" if us10y else "全球定價錨"
    us_note_en = f"Global pricing anchor · US 10Y {us10y:.2f}%" if us10y else "Global pricing anchor"
    rows: list[dict] = [
        {
            "country": "US",
            "country_zh": "美國",
            "spread_bp": None,
            "fx_1m": None,
            "flow_label": us_flow,
            "flow_dir": "supportive" if (dxy_1m or 0) < 0 else "drag",
            "role": "anchor",
            "note_zh": us_note_zh,
            "note_en": us_note_en,
        }
    ]
    for c in ("Japan", "Europe", "China", "Taiwan"):
        sp = spreads.get(c)
        fx = fx1.get(c)

        # Capital flow direction (reused fields) — bilingual labels
        if c == "Japan":
            v = fp.get("japan_tic_mom_change_bn")
            flow = ((f"持有美債 {v:+.0f}bn", f"UST holdings {v:+.0f}bn",
                     "supportive" if (v or 0) > 0 else "drag")) if v is not None else None
        elif c == "China":
            v = fp.get("china_fx_reserves_mom_change_bn")
            flow = ((f"外匯儲備 {v:+.0f}bn", f"FX reserves {v:+.0f}bn",
                     "supportive" if (v or 0) > 0 else "drag")) if v is not None else None
        elif c == "Taiwan":
            tw = fp.get("taiwan_foreign_flow") or {}
            v = tw.get("cum_20d_bn_twd")
            flow = ((f"外資20日 {v:+.0f}bn TWD", f"Foreign 20d {v:+.0f}bn TWD",
                     "supportive" if (v or 0) > 0 else "drag")) if v is not None else None
        else:
            flow = None

        # carry role + note (China = capital-controlled → policy/FX, not carry;
        # Taiwan = small open economy → foreign-flow driven, not a rate-carry leg)
        if c == "China":
            role = "managed"
            note_zh = "資本管制 · 看政策分化/人民幣，非 carry"
            note_en = "Capital-controlled · watch policy/FX, not carry"
        elif c == "Taiwan":
            role = "flow"
            note_zh = "小型開放經濟 · 看外資進出與匯率，非利差 carry"
            note_en = "Small open economy · watch foreign flows & FX, not rate carry"
        elif sp is not None:
            if sp > 50:
                role = "funding"
                note_zh = f"美債高 {sp:+.0f}bp · 當地為 funding 端（借當地→投美元）"
                note_en = f"UST higher {sp:+.0f}bp · local is funding leg (borrow local → buy USD)"
            elif sp < -50:
                role = "target"
                note_zh = f"當地利率高 {sp:+.0f}bp · carry 目的地"
                note_en = f"Local higher {sp:+.0f}bp · carry destination"
            else:
                role = "neutral"
                note_zh = f"利差 {sp:+.0f}bp · 接近平價"
                note_en = f"Spread {sp:+.0f}bp · near parity"
        else:
            role = "na"
            note_zh = "利差資料不足"
            note_en = "Insufficient spread data"

        rows.append({
            "country": c,
            "country_zh": _CARRY_COUNTRY_ZH.get(c, c),
            "spread_bp": None if sp is None else round(float(sp), 0),
            "fx_1m": None if fx is None else round(float(fx), 2),
            "flow_label": flow[0] if flow else None,
            "flow_label_en": flow[1] if flow else None,
            "flow_dir": flow[2] if flow else None,
            "role": role,
            "note_zh": note_zh,
            "note_en": note_en,
        })

    return {
        "vix_cur": None if vix_cur is None else round(vix_cur, 1),
        "vix_pct": vix_pct,
        "switch_zh": switch_zh,
        "switch_en": switch_en,
        "switch_tone": switch_tone,
        "rows": rows,
    }


def _tilt_rows_for_lang(tilt_result: dict, lang: str) -> list[dict]:
    """Localized, sorted asset-tilt rows from the data-driven engine."""
    rows: list[dict] = []
    for t in tilt_result.get("tilts", []):
        hist = t.get("hist_return")
        uncond = t.get("uncond_return")
        effect = t.get("effect")
        hit = t.get("hit_rate")
        # Cash is positive almost every month (rate/12), so its hit-rate carries
        # no information — suppress it to avoid a puzzling "100%".
        if t.get("asset") == "Cash":
            hit = None
        vol = t.get("volatility")
        dv = t.get("downside_vol")
        rows.append({
            "asset": t["asset_zh"] if lang == "zh" else t["asset"],
            "icon": t["icon"],
            "tilt": t["tilt"],
            "level": t["level"],
            "score": t["score"],
            "hist_return": hist,
            "uncond_return": uncond,
            "effect": effect,
            "effect_pos": (effect >= 0) if effect is not None else True,
            "hit_rate": int(hit) if hit is not None else None,
            "volatility": int(vol) if vol is not None else None,
            "downside_vol": int(dv) if dv is not None else None,
        })
    rows.sort(key=lambda r: (-r["level"], -(r["score"] or 0)))
    return rows


def _regime_stat_sub_line(tilt_result: dict | None, lang: str, fallback: str) -> str:
    """MARKET REGIME subtitle: G/I direction · months in regime · historical avg episode."""
    if not tilt_result or tilt_result.get("in_transition"):
        return fallback
    cur_dur = int(tilt_result.get("cur_duration") or 0)
    avg_dur = tilt_result.get("avg_duration")
    g_r = bool(tilt_result.get("growth_rising", True))
    i_r = bool(tilt_result.get("infl_rising", True))
    if lang == "zh":
        g_dir = "成長↑" if g_r else "成長↓"
        i_dir = "通膨↑" if i_r else "通膨↓"
        base = f"{g_dir}{i_dir}"
        if cur_dur and avg_dur is not None:
            return f"{base} · 已持續 {cur_dur} 月 · 歷史平均 {avg_dur:g} 月"
        if cur_dur:
            return f"{base} · 已持續 {cur_dur} 月"
        return fallback
    g_dir = "G↑" if g_r else "G↓"
    i_dir = "I↑" if i_r else "I↓"
    base = f"{g_dir} {i_dir}"
    if cur_dur and avg_dur is not None:
        return f"{base} · in regime {cur_dur}m · avg {avg_dur:g}m"
    if cur_dur:
        return f"{base} · in regime {cur_dur}m"
    return fallback


def _tilt_context_line(tilt_result: dict, lang: str) -> str:
    """One-line provenance: conviction (duration lives on MARKET REGIME stat-sub)."""
    conv = int(round(float(tilt_result.get("conviction") or 0) * 100))
    n = int(tilt_result.get("sample_n") or 0)
    in_transition = tilt_result.get("in_transition", False)
    nearest = tilt_result.get("environment_nearest", "")
    if lang == "zh":
        if in_transition:
            nearest_zh = {"Goldilocks": "金髮女孩", "Overheat": "過熱",
                          "Stagflation": "停滯通膨", "Deflationary Bust": "通縮崩潰"}.get(nearest, nearest)
            return f"動能尚未確認 · 預備朝{nearest_zh}輕倉布局 · 排名與報酬均為{nearest_zh}象限歷史估計（Fidelity Investment Clock 過渡期做法）"
        return f"信心 {conv}% · 基於 {n} 個同類歷史月份"
    if in_transition:
        return f"momentum not yet confirmed · pre-position lightly toward {nearest} (Fidelity Investment Clock transition approach)"
    return f"conviction {conv}% · {n} analogous months"


_PENALTY_LABEL_ZH = {
    "Hedging Cost Anomaly": "避險成本異常（買尾部保護）",
    "Hedging Cost Anomaly (SKEW momentum)": "避險成本異常（尾部保護動能升）",
    "Extreme Crowdedness/Leverage": "極端擁擠/槓桿",
    "Vol Spike (OVX momentum)": "油價波動飆升",
    "Vol Spike (GVZ momentum)": "黃金波動飆升",
    "Fed Path Shock (ZQ momentum)": "Fed 路徑衝擊（市場改 price-in 更多升息）",
    "Defensive Sector Rotation (XLU > XLE)": "防禦型類股輪動（公用 > 能源）",
}


def _build_methodology(
    lang: str,
    tilt_result: dict | None,
    risk_analytics: dict | None,
    divergence_score: int,
) -> dict:
    """Formula + current-values tooltip text for the four quant blocks (no LLM)."""
    zh = lang == "zh"
    tr = tilt_result or {}
    ra = risk_analytics or {}
    out: dict[str, str] = {}

    # ── Regime ────────────────────────────────────────────────────────────
    g = tr.get("growth_mom_now")
    i = tr.get("infl_mom_now")
    g_pct = tr.get("g_pct")
    i_pct = tr.get("i_pct")
    env = tr.get("environment_zh") if zh else tr.get("environment")
    if g is not None and i is not None:
        if zh:
            pct_str = f"成長 z={g:+.2f}（歷史第 {g_pct} 百分位）、通膨 z={i:+.2f}（歷史第 {i_pct} 百分位）。" if g_pct is not None else f"成長 {g:+.2f}、通膨 {i:+.2f}。"
            out["regime"] = (
                f"四象限投資時鐘。{pct_str}"
                f"成長 = OECD CLI 50%＋工業生產 20%＋初領失業金 15%＋失業率 15%；"
                f"通膨 = CPI對2%目標 40%＋CPI動能 30%＋產能利用率 30%。"
                f"皆用 24 月滾動 z 分數（無前視）。"
                f"判定：成長↑通膨↓→金髮女孩；↑↑→過熱；↓↑→停滯通膨；↓↓→通縮崩潰，"
                f"並用 0.2σ 滯後帶防止頻繁切換。目前落點：{env}。"
            )
        else:
            pct_str = f"Growth z={g:+.2f} ({g_pct}th pct), inflation z={i:+.2f} ({i_pct}th pct). " if g_pct is not None else f"Growth {g:+.2f}, inflation {i:+.2f}. "
            out["regime"] = (
                f"Four-quadrant Investment Clock. {pct_str}"
                f"Growth = OECD CLI 50% + industrial production 20% "
                f"+ jobless claims 15% + unemployment 15%; inflation = CPI vs 2% target 40% "
                f"+ CPI momentum 30% + capacity utilisation 30%. Both use trailing 24-month "
                f"rolling z-scores (no look-ahead). G↑I↓→Goldilocks; ↑↑→Overheat; ↓↑→Stagflation; "
                f"↓↓→Deflationary Bust, with a 0.2σ hysteresis band to avoid whipsaw. Current: {env}."
            )

    # ── Asset tilt ────────────────────────────────────────────────────────
    n = int(tr.get("sample_n") or 0)
    conv = int(round(float(tr.get("conviction") or 0) * 100))
    if n:
        if zh:
            out["asset_tilt"] = (
                f"排名（↑↑～↓↓）為 Greetham (2004) 投資時鐘對「{env}」象限的理論加減碼，"
                f"不隨資料變動。百分比為該象限歷史 {n} 個月的月報酬均值年化（FactSet／美林慣例："
                f"匯總所有月份算月均報酬再年化，不對短期片段各自年化）。"
                f"「體制效果」＝該象限條件報酬 −全樣本無條件報酬，隔離體制貢獻與背景趨勢；"
                f"「命中」為正報酬月份占比、「波動」年化標準差、「下行」僅負月份年化標準差。"
                f"僅供脈絡參考。信心 {conv}%。"
            )
        else:
            out["asset_tilt"] = (
                f"Ranks (↑↑–↓↓) are Greetham (2004) Investment Clock's theoretical over/under-weights "
                f"for the {env} quadrant — fixed, not data-derived. The % is the annualised mean monthly "
                f"return across the {n} historical months in this regime (FactSet/Merrill convention: pool "
                f"all months, take the mean monthly return, annualise — no per-episode annualising). "
                f"'Regime effect' = conditional return − all-months unconditional return, isolating the regime's "
                f"contribution from the background trend; 'hit' = share of positive months, 'vol' = annualised "
                f"std, 'downside' = annualised std of negative months only. "
                f"Context only. Conviction {conv}%."
            )
    return out


# ── Investment Clock regime descriptions (Greetham 2004) ─────────────────────
_CLOCK_TIP_ZH = {
    "Goldilocks":       "林奇投資時鐘：金髮女孩象限（G↑ I↓）。成長加速、通膨受控。歷史上股票領漲，信用利差收窄，央行傾向寬鬆或維持中性。",
    "Overheat":         "林奇投資時鐘：過熱象限（G↑ I↑）。成長強勁但通膨升溫，央行趨向緊縮。歷史上實質資產（商品、黃金）跑贏，股票波動加大。",
    "Stagflation":      "林奇投資時鐘：停滯通膨象限（G↓ I↑）。成長放緩但通膨仍高，企業獲利受壓、央行兩難。歷史上商品和黃金跑贏，股票和信用債承壓。",
    "Deflationary Bust":"林奇投資時鐘：通縮崩潰象限（G↓ I↓）。成長與通膨雙雙下滑，衰退風險最高。歷史上長期公債和現金跑贏，風險資產普跌。",
}
_CLOCK_TIP_EN = {
    "Goldilocks":       "Investment Clock · Goldilocks (G↑ I↓): growth accelerating, inflation contained. Historically equities lead, credit spreads tighten, central banks stay neutral or ease.",
    "Overheat":         "Investment Clock · Overheat (G↑ I↑): strong growth but rising inflation, central banks lean hawkish. Historically real assets (commodities, gold) outperform; equity volatility rises.",
    "Stagflation":      "Investment Clock · Stagflation (G↓ I↑): growth slowing but inflation still elevated — the hardest environment for central banks. Historically commodities and gold outperform; equities and credit under pressure.",
    "Deflationary Bust":"Investment Clock · Deflationary Bust (G↓ I↓): both growth and inflation falling, recession risk highest. Historically long bonds and cash outperform; risk assets broadly decline.",
}

def _regime_clock_tip(macro_regime_label: str, lang: str) -> str:
    """Return the Investment Clock description for the current regime quadrant."""
    if lang == "zh":
        return _CLOCK_TIP_ZH.get(macro_regime_label, "")
    return _CLOCK_TIP_EN.get(macro_regime_label, "")


def _regime_buffer_line(tilt_result: dict, lang: str) -> str:
    """One-line: how far each axis is from its hysteresis flip threshold."""
    g_buf = tilt_result.get("g_buffer")
    i_buf = tilt_result.get("i_buffer")
    g_rising = tilt_result.get("growth_rising", True)
    i_rising = tilt_result.get("infl_rising", True)
    if g_buf is None or i_buf is None:
        return ""
    # Direction arrows
    if lang == "zh":
        g_dir = "成長↑" if g_rising else "成長↓"
        i_dir = "通膨↑" if i_rising else "通膨↓"
        g_stable = "穩固" if g_buf > 0.5 else ("臨界" if g_buf > 0.2 else "脆弱")
        i_stable = "穩固" if i_buf > 0.5 else ("臨界" if i_buf > 0.2 else "脆弱")
        return f"判定穩定性：{g_dir} 緩衝 {g_buf:+.2f}σ（{g_stable}）· {i_dir} 緩衝 {i_buf:+.2f}σ（{i_stable}）"
    else:
        g_dir = "G↑" if g_rising else "G↓"
        i_dir = "I↑" if i_rising else "I↓"
        g_stable = "firm" if g_buf > 0.5 else ("marginal" if g_buf > 0.2 else "fragile")
        i_stable = "firm" if i_buf > 0.5 else ("marginal" if i_buf > 0.2 else "fragile")
        return f"Regime stability: {g_dir} buffer {g_buf:+.2f}σ ({g_stable}) · {i_dir} buffer {i_buf:+.2f}σ ({i_stable})"


_ENV_ZH = {"Goldilocks": "金髮女孩", "Overheat": "過熱",
            "Stagflation": "停滯通膨", "Deflationary Bust": "通縮崩潰"}

def _transition_matrix_rows(tilt_result: dict, lang: str) -> dict:
    """Regime transitions split into stickiness + conditional-on-leaving direction.

    Monthly transition matrices have a very high diagonal (this month's regime
    usually persists), so raw off-diagonal probabilities look uniformly tiny and
    convey no direction. We surface two things instead:
      * stay_pct  — how 'sticky' the current quadrant is (P stay next month)
      * exits     — where it goes *given it leaves* (conditional probabilities)
    """
    matrix = tilt_result.get("transition_matrix", {})
    cur_env = tilt_result.get("environment", "")
    row = matrix.get(cur_env, {})
    from src.regime_tilt import ENV_META
    stay_pct = int(round(row.get(cur_env, 0)))
    leave_total = sum(pct for ne, pct in row.items() if ne != cur_env)
    exits = []
    if leave_total > 0:
        others = [(ne, pct) for ne, pct in row.items() if ne != cur_env and pct > 0]
        others.sort(key=lambda x: -x[1])
        for ne, pct in others:
            cond = int(round(pct / leave_total * 100))
            if cond <= 0:
                continue
            exits.append({
                "env": _ENV_ZH.get(ne, ne) if lang == "zh" else ne,
                "pct": cond,
                "tone": ENV_META.get(ne, {}).get("tone", ""),
            })
    return {"stay_pct": stay_pct, "exits": exits}


_CARRY_ROLE_ZH = {
    "anchor": "全球錨", "funding": "funding 端", "target": "carry 目的地",
    "managed": "管制/政策", "neutral": "平價", "na": "資料不足",
    "flow": "外資流向主導",
}
_CARRY_ROLE_EN = {
    "anchor": "global anchor", "funding": "funding leg", "target": "carry target",
    "managed": "managed/policy", "neutral": "near parity", "na": "n/a",
    "flow": "flow-driven",
}


def _carry_rows_for_lang(carry: dict, lang: str) -> dict:
    """Localize the圈層2 carry/relative block for display."""
    if not carry:
        return None
    is_zh = lang == "zh-Hant" or lang == "zh"
    rows = []
    for r in carry.get("rows", []):
        rows.append({
            "country": r["country_zh"] if is_zh else r["country"],
            "country_code": r["country"],
            "spread_bp": r.get("spread_bp"),
            "fx_1m": r.get("fx_1m"),
            "flow_label": r.get("flow_label") if is_zh else (r.get("flow_label_en") or r.get("flow_label")),
            "flow_dir": r.get("flow_dir"),
            "role": r.get("role"),
            "role_label": (_CARRY_ROLE_ZH if is_zh else _CARRY_ROLE_EN).get(r.get("role"), ""),
            "note": r.get("note_en") if not is_zh and r.get("note_en") else r.get("note_zh"),
        })
    return {
        "vix_cur": carry.get("vix_cur"),
        "vix_pct": carry.get("vix_pct"),
        "switch": carry.get("switch_zh") if is_zh else carry.get("switch_en"),
        "switch_tone": carry.get("switch_tone"),
        "rows": rows,
    }


def _build_lite_lang_blocks(
    *,
    report_date: str,
    macro_regime_label: str,
    regime_gauge_tone: str,
    divergence_gauge: dict,
    daily_themes: list[dict],
    directive: dict,
    watch_bilingual: list[dict],
    regime_asset_tilt_label: str,
    country_matrix: list[dict],
    gear_semantics: dict,
    vs_us_alignment: dict,
    historical_matches_en: list[dict],
    divergence_matches_en: list[dict],
    historical_descriptions_zh: list[str],
    divergence_descriptions_zh: list[str],
    history_translation_fallback: bool,
    tilt_result: dict | None = None,
    carry_relative: dict | None = None,
    risk_analytics: dict | None = None,
) -> list[dict]:
    hist_zh = _localize_historical_matches(historical_matches_en, historical_descriptions_zh)
    div_zh = _localize_historical_matches(divergence_matches_en, divergence_descriptions_zh)
    blocks: list[dict] = []
    # When the data-driven tilt engine is available, it is the source of truth for
    # both the regime badge and the asset tilt (hard-data Investment-Clock).
    use_tilt = isinstance(tilt_result, dict) and tilt_result.get("tilts")
    for lang_key, html_lang in (("zh", "zh-Hant"), ("en", "en")):
        ui = LITE_UI[lang_key]
        score = int(divergence_gauge.get("score") or 0)
        directive_view = _lite_directive_view(directive, lang_key)
        themes_local = _lite_daily_themes(daily_themes, lang_key)
        if lang_key == "zh":
            hist = hist_zh
            divm = div_zh
        else:
            hist = historical_matches_en
            divm = divergence_matches_en
        # Regime badge: 5-state (Transition + 4 confirmed quadrants).
        # In Transition, show "過渡期 · 偏向 X" with nearest confirmed quadrant.
        regime_alt_label = None
        if use_tilt:
            in_transition = tilt_result.get("in_transition", False)
            nearest = tilt_result.get("environment_nearest", "")
            if in_transition:
                if lang_key == "zh":
                    nearest_zh = {"Goldilocks": "金髮女孩", "Overheat": "過熱",
                                  "Stagflation": "停滯通膨", "Deflationary Bust": "通縮崩潰"}.get(nearest, nearest)
                    regime_disp = f"過渡期"
                    regime_alt_label = f"轉向{nearest_zh} · 輕倉預備"
                else:
                    regime_disp = "Transition"
                    regime_alt_label = f"rotating toward {nearest} · light positioning"
            else:
                regime_disp = tilt_result["environment_zh"] if lang_key == "zh" else tilt_result["environment"]
                quant_disp = _regime_label_display(macro_regime_label, lang_key)
                if quant_disp and quant_disp != regime_disp:
                    regime_alt_label = quant_disp
            regime_tone = tilt_result.get("environment_tone") or regime_gauge_tone
            asset_tilt = _tilt_rows_for_lang(tilt_result, lang_key)
            tilt_context = _tilt_context_line(tilt_result, lang_key)
        else:
            in_transition = False
            regime_disp = _regime_label_display(macro_regime_label, lang_key)
            regime_tone = regime_gauge_tone
            asset_tilt = _regime_asset_tilt_for_lang(regime_asset_tilt_label, lang_key)
            tilt_context = ""
        blocks.append(
            {
                "lang": html_lang,
                "ui": ui,
                "report_date": report_date,
                "macro_regime_label": regime_disp,
                "regime_alt_label": regime_alt_label,
                "regime_gauge_tone": regime_tone,
                "in_transition": in_transition,
                "regime_subtext": _regime_stat_sub_line(
                    tilt_result if use_tilt else None,
                    lang_key,
                    "" if in_transition else _regime_subtext_ui(regime_tone, ui),
                ),
                "regime_asset_tilt": asset_tilt,
                "tilt_context": tilt_context,
                "regime_buffer": _regime_buffer_line(tilt_result, lang_key) if use_tilt and tilt_result else "",
                "transition_matrix": _transition_matrix_rows(tilt_result, lang_key) if use_tilt and tilt_result else [],
                "carry_relative": (_carry_for_lang := _carry_rows_for_lang(carry_relative, lang_key) if carry_relative else None),
                "carry_by_code": {r["country_code"]: r for r in (_carry_for_lang or {}).get("rows", [])},
                "tilt_legend_regime": (nearest_zh if lang_key == "zh" else nearest) if in_transition else regime_disp,
                "daily_themes": themes_local,
                "directive": directive_view,
                "positioning_text": _lite_positioning_text(directive_view, themes_local),
                "positioning_reasoning": directive_view.get("positioning_6_12m_reasoning") or "",
                "critical_watchout": _lite_critical_watchout(watch_bilingual, lang_key),
                "historical_matches": hist,
                "divergence_matches": divm,
                "country_matrix": _lite_country_matrix(
                    country_matrix, gear_semantics, lang_key, vs_us_alignment
                ),
                "vs_us_alignment": vs_us_alignment,
                "history_translation_fallback": history_translation_fallback and lang_key == "zh",
                "regime_clock_tip": _regime_clock_tip(
                    (tilt_result.get("environment_nearest") or tilt_result.get("environment"))
                    if use_tilt and tilt_result else macro_regime_label,
                    lang_key
                ),
                "methodology": _build_methodology(
                    lang_key, tilt_result, risk_analytics, score
                ),
            }
        )
    return blocks


# ──────────────────────────────────────────────────────────────────────────
# LITE template — a 3-minute scannable version of the same synthesis data.
# Renders to synthesis_lite.html. Reuses the existing synthesis JSON (no extra
# LLM call). Core content is visible; depth (full narrative, historical
# analogues, per-country matrix) is tucked into collapsible <details>.
# ──────────────────────────────────────────────────────────────────────────
HTML_TMPL_LITE = """<!doctype html>
<html lang="en" data-lite-lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>The Daily Regime — Strategic Synthesis · {{ report_date }}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Spectral:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,500&family=Noto+Serif+TC:wght@400;500;600;700;900&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <style>
    :root {
      --red:#E3120B; --ink:#121212; --paper:#ffffff;
      --block:#f4f3f0;                 /* Notion-light warm grey — page surface (outer) */
      --rule:#d9d6cc; --rule-strong:#121212; --muted:#6f6f6f; --cream:#f3f1ea;
      --serif:'Spectral','Noto Serif TC',Georgia,'Songti TC',serif;
      --sans:'Inter','PingFang TC','Microsoft JhengHei',-apple-system,sans-serif;
      --mono:'IBM Plex Mono',ui-monospace,'SF Mono',Menlo,monospace;
      --green:#4a7c59;                 /* restored — down-move (ZH) / up-move (EN) + gauge benign */
    }
    * { box-sizing:border-box; }
    html { -webkit-overflow-scrolling:touch; }
    body { margin:0; background:var(--block); color:var(--ink); font-family:var(--serif); -webkit-font-smoothing:antialiased; }
    /* scroll-margin always applies, regardless of motion preference */
    .snap-anchor { scroll-margin-top: 1.5rem; }
    /* scroll-snap disabled — natural mobile scroll */
    @media (prefers-reduced-motion: no-preference) {
      html { scroll-snap-type: none; }
    }
    /* Section break divider before US Anchor */
    /* Full-screen section break — the editorial/quant boundary */
    .section-break { height:20vh; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:0.75rem; margin:0; border-top:1px solid var(--rule); }
    .section-break-label { font-family:var(--sans); font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.14em; color:var(--muted); }
    .section-break-arrow { color:var(--muted); font-size:1.4rem; line-height:1; animation:bob 1.8s ease-in-out infinite; }
    @keyframes bob { 0%,100%{transform:translateY(0)} 50%{transform:translateY(6px)} }
    @media (prefers-reduced-motion: reduce) { .section-break-arrow { animation:none; } }
    /* Sticky masthead + tab nav */
    .sticky-masthead { margin:0 -2rem; padding:0 2rem; }
    @media (max-width:560px) { .sticky-masthead { margin:0 -1.1rem; padding:0 1.1rem; } }
    .sticky-tabs { position:sticky; top:0; z-index:100; background:var(--block); border-bottom:1px solid var(--rule); margin:0 -2rem; padding:0 2rem; }
    @media (max-width:560px) { .sticky-tabs { margin:0 -1.1rem; padding:0 1.1rem; } }
    /* Sheet title (white header with red rule) */
    .sheet-title { display:flex; align-items:center; justify-content:space-between; gap:0.75rem; flex-wrap:wrap; padding:1.3rem 0 0.6rem; margin-bottom:0; }
    .tab-nav { display:flex; overflow-x:auto; scrollbar-width:none; gap:0; margin:0; }
    .tab-nav::-webkit-scrollbar { display:none; }
    .tab-btn { font-family:var(--sans); font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; padding:0.85rem 1.1rem; border:none; background:none; color:var(--muted); cursor:pointer; border-bottom:2px solid transparent; white-space:nowrap; flex-shrink:0; transition:color 0.15s,border-color 0.15s; }
    .tab-btn.active { color:var(--red); border-bottom-color:var(--red); }
    .tab-btn:hover:not(.active) { color:var(--ink); }
    .tab-pane { display:none; }
    .tab-pane.active { display:block; }
    /* Hero — today's stance as a white card, integrated with the block system */
    .hero { background:var(--paper); border-radius:18px; padding:1.8rem 2rem; margin:1.4rem 0 1.3rem; }
    @media (max-width:560px) { .hero { padding:1.35rem 1.3rem; } }
    .hero-meta { font-family:var(--sans); font-size:0.66rem; text-transform:uppercase; letter-spacing:0.11em; color:var(--muted); font-weight:600; margin-bottom:1rem; display:flex; gap:0.55rem; align-items:center; flex-wrap:wrap; }
    .hero-meta .ok { color:var(--green); font-weight:700; }
    .hero-kicker { font-family:var(--sans); font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:var(--red); margin-bottom:0.55rem; display:flex; align-items:center; }
    .hero-stance { font-family:var(--serif); font-size:2.05rem; font-weight:700; line-height:1.2; letter-spacing:-0.02em; margin:0; color:var(--ink); }
    @media (max-width:560px) { .hero-stance { font-size:1.55rem; } }
    .logo { font-family:'Playfair Display',Georgia,serif; font-size:2rem; line-height:1; color:var(--ink); }
    .tagline { font-family:var(--sans); font-size:0.78rem; color:var(--muted); margin-top:0.35rem; letter-spacing:0.01em; }
    .logo .logo-the { font-weight:700; }
    .logo .logo-regime { font-weight:700; }
    .logo .logo-dot { color:var(--red); }
    .sheet-meta { font-family:var(--mono); font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); font-weight:500; }
    .sheet { max-width:1040px; margin:0 auto; background:var(--block); padding:1.4rem 2rem 3rem; }
    @media (max-width:560px) { .sheet { padding:1.1rem 1.1rem 2.5rem; } }
    /* Reading measure — keep prose at a comfortable width even on a wide sheet */
    .measure { max-width:680px; }
    /* Kicker / section heads — Economist furniture */
    .kicker { font-family:var(--sans); font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:var(--red); margin-bottom:0.35rem; }
    .section-head { font-family:var(--sans); font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:var(--red); margin:1.6rem 0 0.9rem; }
    /* No divider lines — blocks define the sections. Zone header is just a big title. */
    .zone-divider { margin:2.4rem 0 1rem; padding:0; border:none; }
    .zone-title { font-family:var(--serif); font-size:1.85rem; font-weight:700; color:var(--ink); letter-spacing:-0.015em; line-height:1.08; }
    .zone-sub { font-family:var(--sans); font-size:0.82rem; color:var(--muted); margin-top:0.4rem; letter-spacing:0; line-height:1.45; }
    /* Big colour-block module (Notion structure + Economist palette, no rules) */
    .modcard { background:var(--paper); border-radius:16px; padding:1.7rem 1.8rem; margin:1.1rem 0; }
    @media (max-width:560px) { .modcard { padding:1.2rem 1.15rem; border-radius:12px; } }
    .mod-label { font-family:var(--sans); font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:var(--red); margin-bottom:0.45rem; }
    .mod-head { font-family:var(--serif); font-size:1.55rem; font-weight:700; line-height:1.12; letter-spacing:-0.01em; color:var(--ink); margin:0 0 0.9rem; }
    .carry-switch { font-family:var(--sans); font-size:0.62rem; font-weight:600; padding:0.1rem 0.4rem; border-radius:3px; margin-left:0.5rem; text-transform:none; letter-spacing:0; }
    .carry-switch.stable { background:var(--cream); color:var(--ink); }
    .carry-switch.warning { background:var(--cream); color:var(--muted); }
    .carry-switch.risk { background:#fdeee4; color:var(--red); }
    .carry-switch.neutral { background:var(--cream); color:var(--muted); }
    .carry-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:0.9rem; }
    @media (max-width:860px) { .carry-grid { grid-template-columns:repeat(2,1fr); } }
    @media (max-width:480px) { .carry-grid { grid-template-columns:1fr; } }
    .carry-cell { background:var(--paper); border-radius:16px; padding:1.1rem 1.15rem; }
    .carry-top { display:flex; align-items:baseline; gap:0.5rem; }
    .carry-country { font-weight:700; font-size:0.95rem; }
    .carry-role { font-family:var(--sans); font-size:0.58rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; padding:0.08rem 0.4rem; border-radius:3px; background:transparent; color:var(--muted); border:1px solid var(--rule); }
    .carry-stats { font-family:var(--sans); font-size:0.7rem; color:var(--muted); margin-top:0.18rem; display:flex; flex-wrap:wrap; gap:0 0.4rem; }
    .carry-stats .ep { font-weight:600; } .carry-stats .en { font-weight:600; }
    .carry-stats .sp { color:#c2bdb4; }
    .carry-note { font-family:var(--sans); font-size:0.66rem; color:var(--muted); margin-top:0.18rem; }
    .section-head:first-of-type { margin-top:0.6rem; }
    /* Lead: regime stat strip */
    .lead { display:flex; align-items:stretch; gap:0; padding:0.4rem 0 0.9rem; margin-top:0.2rem; }
    .stat { flex:1; min-width:0; padding:0 1rem; }
    .stat:first-child { padding-left:0; }
    .stat-div { width:1px; flex-shrink:0; background:var(--rule); }
    .stat-k { font-family:var(--sans); font-size:0.6rem; font-weight:700; text-transform:uppercase; letter-spacing:0.07em; color:var(--muted); white-space:nowrap; overflow:visible; display:flex; align-items:center; gap:0; }
    .stat-val { font-family:var(--serif); font-size:clamp(1.4rem,5.5vw,1.95rem); font-weight:700; line-height:1.12; margin-top:0.15rem; letter-spacing:-0.01em; white-space:nowrap; }
    .stat-sub { font-size:0.75rem; color:var(--muted); margin-top:0.2rem; line-height:1.4; display:flex; align-items:center; gap:0.3rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .regime-lens { font-size:0.85rem; color:var(--muted); margin-top:0.25rem; line-height:1.3; opacity:0.9; white-space:normal; }
    .regime-buffer { font-family:var(--sans); font-size:0.7rem; color:var(--muted); margin-top:0.3rem; }
    .buf-firm { color:var(--ink); font-weight:700; }
    .buf-marginal { color:var(--muted); font-weight:600; }
    .buf-fragile { color:var(--red); font-weight:600; }
    /* Transition matrix — big numbers (lives in the Market Regime card) */
    .trans-block { margin-top:1.3rem; }
    .trans-head { font-family:var(--sans); font-size:0.7rem; font-weight:600; color:var(--muted); margin-bottom:0.75rem; letter-spacing:0.01em; }
    .trans-stats { display:flex; flex-wrap:nowrap; gap:1.3rem 2.4rem; align-items:flex-start; overflow-x:auto; padding-bottom:0.25rem; }
    .trans-stat { display:flex; flex-direction:column; gap:0.25rem; }
    .ts-num { font-family:var(--mono); font-size:1.7rem; font-weight:600; line-height:0.95; letter-spacing:-0.03em; color:var(--ink); }
    .trans-stat.ts-stay .ts-num { font-size:2.3rem; }
    .ts-lab { font-family:var(--sans); font-size:0.72rem; color:var(--muted); font-weight:600; }
    .trans-stat.ts-stay .ts-lab { color:var(--ink); }
    .trans-div { width:1px; align-self:stretch; background:var(--rule); margin:0.1rem 0; }
    .dot { width:9px; height:9px; border-radius:50%; display:inline-block; flex-shrink:0; }
    .dot.stable { background:var(--ink); } .dot.warning { background:var(--muted); } .dot.risk { background:var(--red); } .dot.neutral { background:var(--muted); }
    /* Asset tilt — data-driven Investment-Clock grid */
    .tilt-context { font-family:var(--sans); font-size:0.7rem; color:var(--muted); margin:-0.3rem 0 0.7rem; line-height:1.5; }
    .tilt-grid { display:flex; flex-direction:column; gap:0.1rem; }
    .tilt-cell { display:flex; align-items:flex-start; gap:0.6rem; padding:0.5rem 0; }
    .tilt-rank { font-family:var(--sans); font-weight:700; font-size:0.78rem; min-width:1.4em; color:var(--muted); padding-top:0.12rem; }
    .tilt-mk { font-family:var(--mono); font-weight:600; font-size:0.9rem; min-width:1.9em; letter-spacing:-0.04em; padding-top:0.05rem; }
    .tilt-overweight .tilt-mk { color:var(--red); }
    .tilt-underweight .tilt-mk { color:var(--muted); }
    .tilt-neutral .tilt-mk { color:#bdb8af; }
    .tilt-body { flex:1; display:flex; flex-direction:column; gap:0.15rem; }
    .tilt-main-row { display:flex; align-items:baseline; gap:0.5rem; }
    .tilt-asset { font-weight:600; flex:1; font-size:0.95rem; }
    .tilt-ret { font-family:var(--mono); font-size:0.74rem; font-weight:600; letter-spacing:-0.02em; }
    /* gain/loss colour is localized — see lang-scoped block below */
    .tilt-stats { font-family:var(--sans); font-size:0.68rem; color:var(--muted); display:flex; flex-wrap:wrap; gap:0 0.55rem; }
    .tilt-stats .sp { color:#c2bdb4; }
    .tilt-stats .ep, .tilt-stats .en { font-weight:600; }
    .tilt-legend { font-family:var(--sans); font-size:0.66rem; color:var(--muted); margin-top:0.7rem; }
    /* Stories (themes) */
    .story-grid { display:flex; flex-direction:column; gap:1.1rem; margin-top:0.4rem; }
    .story { background:var(--paper); border-radius:16px; padding:1.5rem 1.7rem; overflow:visible; }
    .headline-row { display:flex; align-items:flex-start; gap:0.4rem; margin:0.15rem 0 0.5rem; }
    .headline { font-size:1.4rem; font-weight:700; line-height:1.24; letter-spacing:-0.01em; margin:0; flex:1; min-width:0; }
    .headline-row .info { flex-shrink:0; margin-top:0.3rem; margin-left:0; }
    .story-body { font-size:1rem; line-height:1.62; color:#1d1d1b; margin:0 0 0.55rem; }
    .implication { font-size:0.95rem; font-style:italic; line-height:1.55; color:#3a3a38; margin:0; padding-left:0.85rem; border-left:3px solid var(--ink); }
    .implication b { font-style:normal; }
    /* Bottom line */
    .stance { font-size:1.28rem; font-weight:700; line-height:1.34; margin:0.2rem 0 0.85rem; letter-spacing:-0.01em; }
    .posbox { border:1px solid var(--rule); border-left:4px solid var(--red); background:var(--cream); padding:0.75rem 0.95rem; margin-bottom:0.95rem; }
    .posbox-lab { font-family:var(--sans); font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--red); display:block; margin-bottom:0.25rem; }
    .posbox-txt { font-size:0.96rem; line-height:1.55; color:#1d1d1b; }
    .sect-rule { border:none; border-top:1px solid var(--rule); margin:1.2rem 0 1rem; }
    .watch-k { font-family:var(--sans); font-size:0.66rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:var(--ink); margin-bottom:0.4rem; }
    .watch-list { margin:0; padding:0; list-style:none; }
    .watch-list li { font-size:0.95rem; line-height:1.55; padding:0.4rem 0 0.4rem 1.1rem; position:relative; display:flex; align-items:baseline; gap:0.3rem; }
    .watch-list li:last-child { border-bottom:none; }
    .watch-list li::before { content:"—"; position:absolute; left:0; color:var(--ink); font-weight:700; }
    /* Expanded detail sections — now plain white cards (no toggle, no rules) */
    .det-inner { padding:0; }
    .narr-p { font-size:0.98rem; line-height:1.72; color:#1d1d1b; margin:0 0 0.8rem; }
    /* Analogues */
    .an-sub { font-family:var(--sans); font-size:0.82rem; font-weight:700; text-transform:uppercase; color:var(--red); margin:0.3rem 0 0.1rem; letter-spacing:0.09em; }
    .an-sub2 { font-family:var(--sans); font-size:0.68rem; color:var(--muted); margin:0 0 0.6rem; letter-spacing:0.01em; }
    .an-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(195px,1fr)); gap:0.7rem; margin-bottom:0.5rem; }
    .an-card { border:none; border-radius:12px; padding:0.85rem 0.95rem; background:var(--block); }
    .an-per { font-family:var(--mono); font-size:0.8rem; font-weight:600; letter-spacing:-0.02em; }
    .an-desc { font-size:0.82rem; color:var(--muted); margin:0.15rem 0 0.55rem; line-height:1.45; }
    .an-ret { display:grid; grid-template-columns:1fr 1fr 1fr; gap:0.35rem; }
    .an-stat { text-align:center; padding:0.4rem 0.2rem; border:none; border-radius:8px; background:var(--paper); }
    /* an-stat gain/loss colour is localized — see lang-scoped block below */
    .an-l { font-family:var(--sans); font-size:0.58rem; text-transform:uppercase; color:var(--muted); font-weight:700; display:block; letter-spacing:0.03em; }
    .an-v { font-family:var(--mono); font-size:0.86rem; font-weight:600; display:block; margin-top:0.1rem; letter-spacing:-0.02em; }
    .an-disc { font-size:0.74rem; color:var(--muted); font-style:italic; margin-top:0.5rem; }
    /* Country mini */
    .cty-wrap { display:grid; grid-template-columns:1fr 1fr; gap:0.8rem; }
    @media (max-width:680px) { .cty-wrap { grid-template-columns:1fr; } }
    .cty { background:var(--block); border-radius:12px; padding:0.95rem 1.05rem; }
    .cty-h { font-size:1.02rem; font-weight:700; margin-bottom:0.4rem; }
    .cty-vs { font-family:var(--sans); display:inline-block; font-size:0.64rem; padding:0.12rem 0.5rem; margin-left:0.5rem; font-weight:700; text-transform:uppercase; letter-spacing:0.04em; vertical-align:middle; }
    .cty-vs.aligned { background:var(--cream); color:var(--ink); } .cty-vs.diverging { background:#fdeee4; color:var(--red); }
    .cty-vs.mixed { background:var(--cream); color:var(--muted); } .cty-vs.limited { background:var(--cream); color:var(--muted); }
    .cty-row { font-size:0.92rem; line-height:1.55; color:#1d1d1b; margin:0.55rem 0 0; display:flex; flex-direction:column; gap:0.1rem; }
    .cty-row b { font-family:var(--sans); color:var(--red); font-weight:700; font-size:0.65rem; text-transform:uppercase; letter-spacing:0.07em; }
    /* Merged country+carry cards */
    .cty-carry-grid { display:grid; grid-template-columns:1fr 1fr; gap:0.9rem; }
    @media (max-width:680px) { .cty-carry-grid { grid-template-columns:1fr; } }
    .cty-carry-card { background:var(--block); border-radius:12px; padding:0.95rem 1.05rem; }
    .cty-carry-head { display:flex; align-items:baseline; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.25rem; }
    .cty-carry-head .carry-country { font-size:1.02rem; font-weight:700; }
    .cty-carry-top { background:var(--paper); border-radius:10px; padding:0.6rem 0.75rem; margin-bottom:0.5rem; }
    .cty-carry-sep { display:none; }
    .cty-gears .cty-row { margin:0.15rem 0; }
    .full-link { font-family:var(--sans); display:inline-block; margin-top:0.5rem; font-size:0.78rem; color:var(--ink); text-decoration:underline; font-weight:700; }
    .full-link:hover { text-decoration:underline; }
    .foot { font-family:var(--sans); text-align:center; font-size:0.66rem; color:var(--muted); margin-top:1.8rem; letter-spacing:0.04em; }
    .lang-switch { display:flex; gap:0; border:1px solid var(--rule); border-radius:4px; overflow:hidden; flex-shrink:0; }
    .lang-switch button { font-family:var(--sans); font-size:0.68rem; font-weight:700; letter-spacing:0.06em; text-transform:uppercase; padding:0.35rem 0.65rem; border:none; background:#fff; color:var(--muted); cursor:pointer; }
    .lang-switch button.is-active { background:var(--cream); color:var(--red); box-shadow:inset 0 0 0 1px var(--rule); }
    .sheet-title-right { display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap; }
    .llm-banner { font-family:var(--sans); font-size:0.72rem; line-height:1.45; padding:0.55rem 0.75rem; margin:0 0 0.85rem; border-radius:4px; border:1px solid var(--rule); }
    .llm-banner--ok { background:#eef3ef; border-color:#c4d2c8; color:#2f4537; }
    .llm-banner--warn { background:#fdeee4; border-color:#f5c4a8; color:#9a3412; }
    .llm-banner b { font-weight:700; }
    .viz-section { margin:1.1rem 0; }
    .viz-card { border:none; border-radius:16px; padding:1.1rem 1rem 0.7rem; background:var(--paper); overflow-x:auto; }
    @media (max-width:600px) { .viz-card [id^="chart-zscore"] { min-width:560px; } }
    .viz-label { font-family:var(--sans); font-size:0.82rem; font-weight:700; text-transform:uppercase; letter-spacing:0.09em; color:var(--red); margin-bottom:0.5rem; }
    .sector-perf-scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; padding:0 0.2rem 0.15rem 0; }
    .sector-perf-table { width:max-content; min-width:100%; margin-top:0.4rem; }
    .sector-perf-table th, .sector-perf-table td { white-space:nowrap; }
    .sector-perf-table th:last-child, .sector-perf-table td:last-child { padding-right:12px; }
    .sector-perf-label { display:inline-flex; align-items:center; gap:6px; white-space:nowrap; }
    .sector-perf-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
    .sector-perf-excess { font-family:var(--mono); font-size:0.78rem; color:var(--muted); }
    .edition-kicker { font-family:var(--sans); font-size:0.7rem; text-transform:uppercase; letter-spacing:0.14em; color:var(--muted); font-weight:600; margin-bottom:0.35rem; }
    .today-line-wrap { padding-bottom:0.6rem; margin-bottom:0.15rem; }
    .today-line-k { font-family:var(--sans); font-size:0.62rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:var(--red); margin-bottom:0.35rem; }
    .today-line { font-size:1.5rem; font-weight:700; line-height:1.34; letter-spacing:-0.015em; margin:0; }
    html[data-lite-lang="en"] .lang-block[data-lang="zh-Hant"] { display:none; }
    html[data-lite-lang="zh-Hant"] .lang-block[data-lang="en"] { display:none; }
    html[data-lite-lang="en"] .ai-zh { display:none; }
    html[data-lite-lang="zh-Hant"] .ai-en { display:none; }
    /* Localized gain/loss colour — Western(EN): up=green, down=red · 紅漲綠跌(ZH): 漲=紅, 跌=綠 */
    .lang-block[data-lang="en"] .ep,
    .lang-block[data-lang="en"] .tilt-ret.pos,
    .lang-block[data-lang="en"] .an-stat.pos .an-v { color:var(--green); font-weight:700; }
    .lang-block[data-lang="en"] .en,
    .lang-block[data-lang="en"] .tilt-ret.neg,
    .lang-block[data-lang="en"] .an-stat.neg .an-v { color:var(--red); font-weight:700; }
    .lang-block[data-lang="zh-Hant"] .ep,
    .lang-block[data-lang="zh-Hant"] .tilt-ret.pos,
    .lang-block[data-lang="zh-Hant"] .an-stat.pos .an-v { color:var(--red); font-weight:700; }
    .lang-block[data-lang="zh-Hant"] .en,
    .lang-block[data-lang="zh-Hant"] .tilt-ret.neg,
    .lang-block[data-lang="zh-Hant"] .an-stat.neg .an-v { color:var(--green); font-weight:700; }
    .lang-block[data-lang="zh-Hant"] details summary::after { content:"展開 +"; }
    .lang-block[data-lang="zh-Hant"] details[open] summary::after { content:"收起 −"; }
    .lang-block[data-lang="en"] details summary::after { content:"Expand +"; }
    .lang-block[data-lang="en"] details[open] summary::after { content:"Collapse −"; }
    details summary::after { font-family:var(--sans); font-size:0.7rem; color:var(--red); font-weight:700; letter-spacing:0.04em; }
    /* Methodology info tooltips (hover on desktop, tap/focus on mobile) */
    .info { position:relative; display:inline-flex; align-items:center; justify-content:center; width:15px; height:15px; margin-left:0.3rem; border-radius:50%; border:1px solid var(--muted); color:var(--muted); font-size:0.62rem; font-weight:700; font-family:var(--sans); cursor:help; vertical-align:middle; flex-shrink:0; line-height:1; user-select:none; transition:background 0.15s,color 0.15s; }
    .info:hover, .info:focus { background:var(--ink); color:#fff; border-color:var(--ink); outline:none; }
    .info .tip { display:none; position:absolute; z-index:50; top:calc(100% + 8px); left:50%; transform:translateX(-50%); width:min(280px,78vw); padding:0.6rem 0.75rem; background:#1f2430; color:#f3f4f6; font-size:0.72rem; font-weight:400; line-height:1.55; letter-spacing:0; text-align:left; text-transform:none; border-radius:7px; box-shadow:0 8px 24px rgba(0,0,0,0.28); white-space:normal; font-family:var(--sans); }
    .info .tip::before { content:""; position:absolute; bottom:100%; left:50%; transform:translateX(-50%); border:6px solid transparent; border-bottom-color:#1f2430; }
    .info:hover .tip, .info:focus .tip, .info.tip-open .tip { display:block; }
    .info.tip-open { background:var(--ink); color:#fff; border-color:var(--ink); }
    /* Flip tooltip to the right edge when icon sits near the right side */
    .info.tip-left .tip { left:auto; right:0; transform:none; }
    .info.tip-left .tip::before { left:auto; right:5px; transform:none; }
    /* Open rightward when icon sits near the left edge */
    .info.tip-right .tip { left:0; right:auto; transform:none; }
    .info.tip-right .tip::before { left:5px; right:auto; transform:none; }
    /* ── Interactions (Tier 1) ─────────────────────────────────────── */
    /* Top scroll-progress hairline (brand red) */
    .scroll-progress { position:fixed; top:0; left:0; height:3px; width:0; background:var(--red); z-index:200; transition:width 0.08s linear; }
    /* Card tactility — hover lift on pointer devices, press on touch */
    .modcard, .viz-card, .story, .carry-cell { transition:box-shadow 0.22s ease, transform 0.22s ease; }
    @media (hover:hover) and (prefers-reduced-motion:no-preference) {
      .modcard:hover, .viz-card:hover, .story:hover, .carry-cell:hover { box-shadow:0 6px 22px rgba(18,18,18,0.08); transform:translateY(-2px); }
    }
    .story:active, .carry-cell:active { transform:scale(0.992); }
    /* Scroll-reveal — modules fade-up as they enter view (disabled if reduced-motion) */
    @media (prefers-reduced-motion:no-preference) {
      .reveal { opacity:0; transform:translateY(12px); transition:opacity 0.5s ease, transform 0.5s ease; }
      .reveal.in { opacity:1; transform:none; }
    }
  </style>
  <script>
    (function () {
      var KEY = 'macraLiteLang';
      var root = document.documentElement;
      function apply(lang) {
        lang = lang === 'en' ? 'en' : 'zh-Hant';
        root.setAttribute('data-lite-lang', lang);
        root.setAttribute('lang', lang);
        document.querySelectorAll('.lang-switch button').forEach(function (btn) {
          btn.classList.toggle('is-active', btn.getAttribute('data-lang') === lang);
        });
        try { localStorage.setItem(KEY, lang); } catch (e) {}
        if (window.Plotly) {
          setTimeout(function () {
            document.querySelectorAll('.lang-block[data-lang="' + lang + '"] .js-plotly-plot').forEach(function (g) {
              try { Plotly.Plots.resize(g); } catch (e) {}
            });
          }, 60);
        }
      }
      var saved = null;
      try { saved = localStorage.getItem(KEY); } catch (e) {}
      apply(saved || 'en');
      document.addEventListener('click', function (ev) {
        var btn = ev.target.closest('.lang-switch button');
        if (btn) { apply(btn.getAttribute('data-lang')); return; }
        // Toggle info tooltip on tap/click
        var info = ev.target.closest('.info');
        if (info) {
          ev.stopPropagation();
          var isOpen = info.classList.contains('tip-open');
          document.querySelectorAll('.info.tip-open').forEach(function(el){ el.classList.remove('tip-open'); });
          if (!isOpen) info.classList.add('tip-open');
          return;
        }
        // Close all open tooltips when clicking elsewhere
        document.querySelectorAll('.info.tip-open').forEach(function(el){ el.classList.remove('tip-open'); });
      });
    })();
  </script>
</head>
<body>
  <div class="snap-top" aria-hidden="true"></div>
  <div class="scroll-progress" aria-hidden="true"></div>
  {% macro info(t, side='') %}{% if t %}<span class="info {{ side }}" tabindex="0" role="button" aria-label="方法說明">i<span class="tip">{{ t }}</span></span>{% endif %}{% endmacro %}
  <div class="sheet">
    <div class="sticky-masthead">
      <div class="sheet-title">
        <div>
          <div class="logo"><span class="logo-the">The Daily </span><span class="logo-regime">Regime</span><span class="logo-dot">.</span></div>
          <div class="tagline">Translating data into macro narratives</div>
        </div>
        <div class="sheet-title-right">
          <div class="sheet-meta">{{ report_date }}</div>
          <div class="lang-switch" role="group" aria-label="Language">
            <button type="button" data-lang="zh-Hant">中文</button>
            <button type="button" data-lang="en" class="is-active">EN</button>
          </div>
        </div>
      </div>
    </div>
    <div class="sticky-tabs">
      <nav class="tab-nav" role="tablist">
        <button class="tab-btn active" data-pane="today" role="tab"><span class="ai-zh">今日主題</span><span class="ai-en">Today</span></button>
        <button class="tab-btn" data-pane="us" role="tab"><span class="ai-zh">美國基準</span><span class="ai-en">US Anchor</span></button>
        <button class="tab-btn" data-pane="global" role="tab"><span class="ai-zh">國際承接面</span><span class="ai-en">Global</span></button>
        <button class="tab-btn" data-pane="history" role="tab"><span class="ai-zh">歷史借鏡</span><span class="ai-en">History</span></button>
        <button class="tab-btn" data-pane="sectors" role="tab"><span class="ai-zh">板塊輪動</span><span class="ai-en">Sectors</span></button>
      </nav>
    </div>

    {% if llm_placeholder %}
    <div class="llm-banner llm-banner--warn" role="status">
      <b>Narrative offline.</b> This edition was not generated by the CIO model
      ({% if llm_meta.error %}{{ llm_meta.error }}{% else %}{{ llm_meta.status }}{% endif %}).
      Run <span style="font-family:monospace;font-size:0.68rem">synthesis.py --date {{ report_date }}</span> without <span style="font-family:monospace;font-size:0.68rem">--skip-llm</span>.
    </div>
    {% endif %}

    {% for block in lite_lang_blocks %}
    <div class="lang-block" data-lang="{{ block.lang }}">
      <div class="tab-pane active" data-pane="today">
      <div class="hero">
        <div class="hero-kicker">{{ block.ui.today_line_kicker }}{{ info(block.directive.the_narrative | join(' '), 'tip-right') }}</div>
        <h1 class="hero-stance">{{ block.directive.the_stance }}</h1>
      </div>

      {% if block.daily_themes %}
      <div class="zone-divider zone-3">
        <div class="zone-title">{{ block.ui.zone_synthesis }}</div>
        <div class="zone-sub">{{ block.ui.zone_synthesis_sub }}</div>
      </div>
      <div class="story-grid">
      {% for t in block.daily_themes %}
      <article class="story">
        <div class="kicker">{{ t.kicker }}</div>
        <div class="headline-row"><h2 class="headline">{{ t.title }}</h2>{{ info(t.reasoning, 'tip-left') }}</div>
        <p class="story-body">{{ t.body }}</p>
        {% if t.implication %}<p class="implication"><b>{{ block.ui.implication_label }} ·</b> {{ t.implication }}</p>{% endif %}
      </article>
      {% endfor %}
      </div>
      {% endif %}

      {% if block.critical_watchout %}
      <div class="modcard">
        <div class="mod-label">{{ block.ui.watch_head }}</div>
        <ul class="watch-list">
          {% for w in block.critical_watchout %}<li>{{ w.text }}{{ info(w.reasoning, 'tip-left') }}</li>{% endfor %}
        </ul>
      </div>
      {% endif %}

      {% if block.directive.the_narrative %}
      <div class="modcard">
        <div class="mod-label">{{ block.ui.cio_summary }}</div>
        <div class="det-inner">
          {% for para in block.directive.the_narrative %}<p class="narr-p">{{ para }}</p>{% endfor %}
          {% if block.directive.the_watchout %}<p class="narr-p"><b>{{ block.ui.watchout_label }} ·</b> {{ block.directive.the_watchout }}</p>{% endif %}
        </div>
      </div>
      {% endif %}
      </div><!-- /tab-pane today -->

      <div class="tab-pane" data-pane="us">
      <div class="zone-divider zone-1 snap-anchor">
        <div class="zone-title">{{ block.ui.zone_us_anchor }}</div>
        <div class="zone-sub">{{ block.ui.zone_us_anchor_sub }}</div>
      </div>

      <div class="modcard">
        <div class="mod-label">{{ block.ui.regime_k }}{{ info(block.methodology.regime) }}</div>
        <div class="lead">
          <div class="stat">
            <div class="stat-val">{{ block.macro_regime_label }}{% if block.regime_clock_tip %}{{ info(block.regime_clock_tip, side='tip-left') }}{% endif %}</div>
            <div class="stat-sub"><span class="dot {{ block.regime_gauge_tone }}"></span>{{ block.regime_subtext }}</div>
            {% if block.regime_alt_label %}<div class="regime-lens">{{ block.regime_alt_label }}</div>{% endif %}
            {% if block.regime_buffer %}<div class="regime-buffer">{{ block.regime_buffer }}</div>{% endif %}
          </div>
        </div>
        {% if block.transition_matrix and block.transition_matrix.exits %}
        <div class="trans-block">
          <div class="trans-head">{% if block.lang == 'zh-Hant' %}體制轉移機率 · 一旦離開最常轉向{% else %}Regime transition odds · once it leaves, most often to{% endif %}</div>
          <div class="trans-stats">
            <div class="trans-stat ts-stay">
              <div class="ts-num">{{ block.transition_matrix.stay_pct }}%</div>
              <div class="ts-lab">{% if block.lang == 'zh-Hant' %}留在本象限{% else %}stays in regime{% endif %}</div>
            </div>
            <div class="trans-div"></div>
            {% for t in block.transition_matrix.exits %}
            <div class="trans-stat">
              <div class="ts-num">{{ t.pct }}%</div>
              <div class="ts-lab">→ {{ t.env }}</div>
            </div>
            {% endfor %}
          </div>
        </div>
        {% endif %}
      </div>

      {% if regime_prob_ts_data and regime_prob_ts_data != '{}' %}
      <div class="viz-card" style="margin-top:0.4rem;">
        <div class="viz-label">{% if block.lang == 'zh-Hant' %}Regime 歷史週期 · HMM 月度分類（1999–今）{% else %}Regime History · HMM monthly classification (1999–present){% endif %}</div>
        <div id="chart-regime-prob-{{ block.lang }}" style="width:100%;"></div>
        <div style="display:flex;flex-wrap:wrap;gap:0.55rem 1.1rem;margin-top:0.55rem;">
          <span style="display:flex;align-items:center;gap:0.35rem;font-size:0.72rem;color:#5a5550;"><span style="width:11px;height:11px;background:#d4ede4;border-radius:2px;flex-shrink:0;display:inline-block;"></span>{% if block.lang == 'zh-Hant' %}擴張{% else %}Expansionary{% endif %}</span>
          <span style="display:flex;align-items:center;gap:0.35rem;font-size:0.72rem;color:#5a5550;"><span style="width:11px;height:11px;background:#f5dada;border-radius:2px;flex-shrink:0;display:inline-block;"></span>{% if block.lang == 'zh-Hant' %}壓力/收縮{% else %}Stress / Contraction{% endif %}</span>
          <span style="display:flex;align-items:center;gap:0.35rem;font-size:0.72rem;color:#5a5550;"><span style="width:11px;height:11px;background:#ebe7de;border-radius:2px;flex-shrink:0;display:inline-block;"></span>{% if block.lang == 'zh-Hant' %}中性/過渡{% else %}Neutral / Transitional{% endif %}</span>
          <span style="display:flex;align-items:center;gap:0.35rem;font-size:0.72rem;color:#5a5550;"><span style="width:11px;height:11px;background:#f5e9cf;border-radius:2px;flex-shrink:0;display:inline-block;"></span>{% if block.lang == 'zh-Hant' %}緊縮{% else %}Restrictive{% endif %}</span>
        </div>
        <script>
        (function(){
          var d = {{ regime_prob_ts_data | safe }};
          if(d.traces) Plotly.newPlot('chart-regime-prob-{{ block.lang }}', d.traces, d.layout, {responsive:true,displayModeBar:false,scrollZoom:false,doubleClick:false,editable:false});
        })();
        </script>
      </div>
      {% endif %}

      {% if regime_quadrant_data %}
      <div class="viz-card" style="margin-top:0.4rem;">
        <div class="viz-label">{% if block.lang == 'zh-Hant' %}體制軌跡 · 成長 × 通膨（近 12 個月）{% else %}Regime trajectory · Growth × Inflation (12m){% endif %}</div>
        <div id="chart-regime-{{ block.lang }}" style="width:100%;"></div>
        <script>
        (function(){
          var d = {{ regime_quadrant_data | safe }};
          if(d.traces) Plotly.newPlot('chart-regime-{{ block.lang }}', d.traces, d.layout, {responsive:true,displayModeBar:false,scrollZoom:false,doubleClick:false,editable:false});
        })();
        </script>
      </div>
      {% endif %}

      {% if zscore_data_zh or zscore_data_en %}
      <div class="viz-card" style="margin-top:0.6rem;">
        <div class="viz-label">{% if block.lang == 'zh-Hant' %}美國金融條件溫度計 · 實際歷史分布（非理論常態）{% else %}US Financial Conditions · Empirical distribution{% endif %}</div>
        <div id="chart-zscore-{{ block.lang }}" style="width:100%;"></div>
        <script>
        (function(){
          var d = {{ (zscore_data_zh if block.lang == 'zh-Hant' else zscore_data_en) | safe }};
          if(d.traces) Plotly.newPlot('chart-zscore-{{ block.lang }}', d.traces, d.layout, {responsive:true,displayModeBar:false,scrollZoom:false,doubleClick:false,editable:false});
        })();
        </script>
      </div>
      {% endif %}

      {% if block.regime_asset_tilt %}
      <div class="modcard">
      <div class="mod-label">{% if block.lang == 'zh-Hant' %}資產配置{% else %}ASSET ALLOCATION{% endif %}</div>
      <div class="mod-head">{{ block.tilt_legend_regime }}{% if block.lang == 'zh-Hant' %} · 配置傾向{% else %} tilt{% endif %}{{ info(block.methodology.asset_tilt) }}</div>
      {% if block.tilt_context %}<div class="tilt-context">{% if not block.in_transition %}{{ block.tilt_legend_regime }} · {% endif %}{{ block.tilt_context }}</div>{% endif %}
      <div class="tilt-grid">
        {% for item in block.regime_asset_tilt %}
        <div class="tilt-cell tilt-{{ item.tilt }}">
          <span class="tilt-rank">{{ loop.index }}</span>
          <span class="tilt-mk lv{{ item.level }}">{{ item.icon }}</span>
          <div class="tilt-body">
            <div class="tilt-main-row">
              <span class="tilt-asset">{{ item.asset }}</span>
              {% if item.hist_return is defined and item.hist_return is not none %}<span class="tilt-ret {% if item.hist_return >= 0 %}pos{% else %}neg{% endif %}">{{ "%+.1f"|format(item.hist_return) }}%</span>{% endif %}
            </div>
            {% if item.effect is defined and item.effect is not none %}
            <div class="tilt-stats">
              <span>{% if block.lang == 'zh-Hant' %}體制效果{% else %}regime effect{% endif %} <span class="{% if item.effect_pos %}ep{% else %}en{% endif %}">{{ "%+.1f"|format(item.effect) }}%</span></span>
              <span class="sp">·</span>
              {% if item.hit_rate is defined and item.hit_rate is not none %}<span>{% if block.lang == 'zh-Hant' %}命中{% else %}hit{% endif %} {{ item.hit_rate }}%</span><span class="sp">·</span>{% endif %}
              <span>{% if block.lang == 'zh-Hant' %}波動{% else %}vol{% endif %} {{ item.volatility }}%</span>
              {% if item.downside_vol is defined and item.downside_vol is not none %}<span class="sp">·</span><span>{% if block.lang == 'zh-Hant' %}下行{% else %}dside{% endif %} {{ item.downside_vol }}%</span>{% endif %}
            </div>
            {% endif %}
          </div>
        </div>
        {% endfor %}
      </div>
      <div class="tilt-legend">{{ block.ui.tilt_legend }} · {{ block.ui.tilt_legend_suffix }}</div>
      </div>
      {% endif %}

      </div><!-- /tab-pane us -->

      <div class="tab-pane" data-pane="global">
      <div class="zone-divider zone-2">
        <div class="zone-title">{{ block.ui.zone_global }}</div>
        <div class="zone-sub">{{ block.ui.zone_global_sub }}</div>
      </div>

      {% if block.country_matrix %}
      <div class="modcard">
      {% if block.carry_relative %}
      <div class="section-head" style="margin-top:0;">{% if block.lang == 'zh-Hant' %}利差承接面{% else %}Rate-Differential Map{% endif %}
        {% if block.carry_relative.vix_pct is not none %}<span class="carry-switch {{ block.carry_relative.switch_tone }}">VIX {{ block.carry_relative.vix_pct }}{% if block.lang == 'zh-Hant' %} 百分位{% else %}th{% endif %} · {{ block.carry_relative.switch }}</span>{% endif %}
      </div>
      {% endif %}
      <div class="cty-carry-grid">
        {% for panel in block.country_matrix %}
        {% set cr = block.carry_by_code[panel.country_code] if panel.country_code in block.carry_by_code else none %}
        <div class="cty-carry-card">
          {% if cr or panel.align_label %}
          <div class="cty-carry-top">
            <div class="cty-carry-head">
              <span class="carry-country">{{ panel.country }}</span>
              {% if panel.align_label %}<span class="cty-vs {{ panel.align_class }}">{{ panel.align_label }}</span>{% endif %}
              {% if cr %}<span class="carry-role role-{{ cr.role }}">{{ cr.role_label }}</span>{% endif %}
            </div>
            {% if cr %}
            <div class="carry-stats">
              {% if cr.spread_bp is not none %}<span>{% if block.lang == 'zh-Hant' %}美債利差{% else %}UST spread{% endif %} {{ "%+.0f"|format(cr.spread_bp) }}bp</span>{% endif %}
              {% if cr.fx_1m is not none %}<span class="sp">·</span><span>{% if block.lang == 'zh-Hant' %}匯率1m{% else %}FX 1m{% endif %} {{ "%+.2f"|format(cr.fx_1m) }}%</span>{% endif %}
              {% if cr.flow_label %}<span class="sp">·</span><span class="{% if cr.flow_dir == 'supportive' %}ep{% else %}en{% endif %}">{{ cr.flow_label }}</span>{% endif %}
            </div>
            {% if cr.note %}<div class="carry-note">{{ cr.note }}</div>{% endif %}
            {% endif %}
          </div>
          {% endif %}
          <div class="cty-gears">
            {% for gear in panel.gears %}
            <div class="cty-row"><b>{{ gear.title }}</b>{{ gear.body }}</div>
            {% endfor %}
          </div>
        </div>
        {% endfor %}
      </div>
      {% if block.carry_relative %}
      <div class="tilt-legend">{% if block.lang == 'zh-Hant' %}利差 = 美債 − 當地 10Y · carry 開關看 VIX · 攤開不評分（美國為原點，中國資本管制走政策分化）{% else %}Spread = UST − local 10Y · carry switch = VIX · parts shown, not scored (US is origin; China is capital-controlled → policy/FX){% endif %}</div>
      {% endif %}
      </div>
      {% endif %}

      </div><!-- /tab-pane global -->

      <div class="tab-pane" data-pane="history">
      {% if block.historical_matches or block.divergence_matches %}
      <div class="zone-divider zone-hist">
        <div class="zone-title">{{ block.ui.zone_history }}</div>
        <div class="zone-sub">{{ block.ui.zone_history_sub }}</div>
      </div>
      <div class="modcard">
        <div class="det-inner">
          {% if block.historical_matches %}
          <div class="an-sub">{{ block.ui.history_us_sub }}</div>
          <div class="an-sub2">{{ block.ui.history_us_sub2 }}</div>
          <div class="an-grid">
            {% for m in block.historical_matches %}
            {% set r12 = m.spy_return_12m %}{% set dd = m.max_drawdown_12m %}
            <div class="an-card">
              <div class="an-per">{{ m.date }}</div>
              <div class="an-desc">{{ m.description }}</div>
              <div class="an-ret">
                <div class="an-stat {% if r12 is none %}{% elif r12 >= 0 %}pos{% else %}neg{% endif %}"><span class="an-l">{{ block.ui.spy_12m }}</span><span class="an-v">{% if r12 is none %}—{% else %}{{ "%+.0f"|format(r12) }}%{% endif %}</span></div>
                <div class="an-stat {% if m.spy_return_3m is none %}{% elif m.spy_return_3m >= 0 %}pos{% else %}neg{% endif %}"><span class="an-l">3m</span><span class="an-v">{% if m.spy_return_3m is none %}—{% else %}{{ "%+.0f"|format(m.spy_return_3m) }}%{% endif %}</span></div>
                <div class="an-stat {% if dd is none %}{% else %}neg{% endif %}"><span class="an-l">Max DD</span><span class="an-v">{% if dd is none %}—{% else %}{{ "%.0f"|format(dd) }}%{% endif %}</span></div>
              </div>
            </div>
            {% endfor %}
          </div>
          {% endif %}
          {% if block.divergence_matches %}
          <div class="an-sub" style="margin-top:0.9rem;">{{ block.ui.history_div_sub }}</div>
          <div class="an-sub2">{{ block.ui.history_div_sub2 }}</div>
          <div class="an-grid">
            {% for m in block.divergence_matches %}
            <div class="an-card">
              <div class="an-per">{{ m.date }}</div>
              <div class="an-desc">{{ m.description }}</div>
              <div class="an-ret">
                {% for asset, label in [('eem', block.ui.asset_em), ('gld', block.ui.asset_gold), ('copper', block.ui.asset_copper)] %}
                {% set r12 = m[asset ~ '_12m'] %}
                <div class="an-stat {% if r12 is none %}{% elif r12 >= 0 %}pos{% else %}neg{% endif %}"><span class="an-l">{{ label }} 12m</span><span class="an-v">{% if r12 is none %}—{% else %}{{ "%+.0f"|format(r12) }}%{% endif %}</span></div>
                {% endfor %}
              </div>
            </div>
            {% endfor %}
          </div>
          {% endif %}
          <div class="an-disc">* {{ block.ui.history_disc }}{% if block.history_translation_fallback %} {{ block.ui.history_disc_fallback }}{% endif %}</div>
        </div>
      </div>
      {% endif %}

      </div><!-- /tab-pane history -->

      <div class="tab-pane" data-pane="sectors">
      <div class="zone-divider zone-1 snap-anchor">
        <div class="zone-title"><span class="ai-zh">板塊輪動</span><span class="ai-en">Sector Rotation</span></div>
        <div class="zone-sub"><span class="ai-zh">各板塊相對 SPY 的強弱與動能方向（近 12 週軌跡）</span><span class="ai-en">Relative strength &amp; momentum vs SPY — 12-week trail</span></div>
      </div>

      {% if rrg_data and rrg_data != '{}' %}
      {% if rrg_summary %}
      <div class="hero">
        <div class="hero-kicker"><span class="ai-zh">一句話</span><span class="ai-en">In one line</span></div>
        <h1 class="hero-stance">
          <span class="ai-zh">{{ rrg_summary.zh }}</span>
          <span class="ai-en">{{ rrg_summary.en }}</span>
        </h1>
      </div>
      {% endif %}
      {% if rrg_sector_stats %}
      {% set sorted_stats = rrg_sector_stats | sort(attribute='rank_score', reverse=True) %}
      {% set top = sorted_stats[0] %}
      {% if rrg_rank_trans and rrg_rank_trans.stay_pct %}
      <div class="modcard" style="margin-top:0.4rem;">
        <div class="mod-label"><span class="ai-zh">現在領先板塊</span><span class="ai-en">CURRENT LEADER</span></div>
        <div class="lead">
          <div class="stat">
            <div class="stat-val" style="color:{{ top.color }};">
              <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{{ top.color }};margin-right:6px;vertical-align:middle;"></span>{{ top.label }}
            </div>
            <div class="stat-sub">
              <span class="ai-zh">{{ top.quadrant_zh }} · 已領先 {{ top.weeks_current }} 週{% if top.avg_weeks_as_rank1 %} · 當第一時歷史平均 {{ top.avg_weeks_as_rank1 }} 週{% endif %}</span>
              <span class="ai-en">{{ top.quadrant_en }} · leading for {{ top.weeks_current }}w{% if top.avg_weeks_as_rank1 %} · avg {{ top.avg_weeks_as_rank1 }}w as #1{% endif %}</span>
            </div>
          </div>
        </div>
        <div class="trans-block">
          <div class="trans-head"><span class="ai-zh">領先板塊轉移機率 · 一旦換人最常是誰接班</span><span class="ai-en">Leadership transition odds · once it changes, most often replaced by</span></div>
          <div class="trans-stats">
            <div class="trans-stat ts-stay">
              <div class="ts-num">{{ rrg_rank_trans.stay_pct }}%</div>
              <div class="ts-lab"><span class="ai-zh">繼續領先</span><span class="ai-en">stays leader</span></div>
            </div>
            <div class="trans-div"></div>
            {% for e in rrg_rank_trans.exits %}
            <div class="trans-stat">
              <div class="ts-num">{{ e.pct }}%</div>
              <div class="ts-lab">→ {{ e.label }}</div>
            </div>
            {% endfor %}
          </div>
        </div>
      </div>
      {% endif %}
      {% endif %}

      {% if sector_performance %}
      <div class="modcard" style="margin-top:0.4rem;">
        <div class="mod-label"><span class="ai-zh">板塊表現 · 對 SPY</span><span class="ai-en">Sector performance · vs SPY</span></div>
        <div class="sector-perf-scroll">
        <table class="data-table sector-perf-table">
          <thead>
            <tr>
              <th><span class="ai-zh">板塊</span><span class="ai-en">Sector</span></th>
              <th style="text-align:right"><span class="ai-zh">1 週</span><span class="ai-en">1W</span></th>
              <th style="text-align:right"><span class="ai-zh">1 月</span><span class="ai-en">1M</span></th>
              <th style="text-align:right"><span class="ai-zh">1 年</span><span class="ai-en">1Y</span></th>
            </tr>
          </thead>
          <tbody>
          {% for r in sector_performance %}
          <tr{% if r.is_benchmark %} style="background:#f8f9fb;"{% endif %}>
            <td>
              <span class="sector-perf-label">
                <span class="sector-perf-dot" style="background:{{ r.color }};"></span>
                <strong{% if r.is_benchmark %} style="color:var(--ink);"{% endif %}>{{ r.label }}</strong>
              </span>
            </td>
            <td style="text-align:right;font-family:var(--mono);font-size:0.88rem;">
              {% if r.ret_1w is not none %}<span class="tilt-ret {% if r.ret_1w >= 0 %}pos{% else %}neg{% endif %}">{{ "%+.1f"|format(r.ret_1w) }}%</span>{% if r.excess_1w is not none %} <span class="sector-perf-excess">({{ "%+.1f"|format(r.excess_1w) }}%)</span>{% endif %}{% else %}—{% endif %}
            </td>
            <td style="text-align:right;font-family:var(--mono);font-size:0.88rem;">
              {% if r.ret_1m is not none %}<span class="tilt-ret {% if r.ret_1m >= 0 %}pos{% else %}neg{% endif %}">{{ "%+.1f"|format(r.ret_1m) }}%</span>{% if r.excess_1m is not none %} <span class="sector-perf-excess">({{ "%+.1f"|format(r.excess_1m) }}%)</span>{% endif %}{% else %}—{% endif %}
            </td>
            <td style="text-align:right;font-family:var(--mono);font-size:0.88rem;">
              {% if r.ret_1y is not none %}<span class="tilt-ret {% if r.ret_1y >= 0 %}pos{% else %}neg{% endif %}">{{ "%+.1f"|format(r.ret_1y) }}%</span>{% if r.excess_1y is not none %} <span class="sector-perf-excess">({{ "%+.1f"|format(r.excess_1y) }}%)</span>{% endif %}{% else %}—{% endif %}
            </td>
          </tr>
          {% endfor %}
          </tbody>
        </table>
        </div>
        <div style="margin-top:0.5rem;font-size:0.7rem;color:var(--muted);line-height:1.45;">
          <span class="ai-zh">SPY 為基準；板塊括號內為相對 SPY 超額。板塊依 1 週超額排序。</span>
          <span class="ai-en">SPY is the benchmark; sector parentheses show excess vs SPY. Sectors sorted by 1W excess.</span>
        </div>
      </div>
      {% endif %}

      <div class="viz-card" style="margin-top:0.4rem;">
        <div class="viz-label"><span class="ai-zh">相對輪動圖 (RRG) · 以 SPY 為基準</span><span class="ai-en">Relative Rotation Graph (RRG) · Benchmark: SPY</span></div>
        <div id="chart-rrg-{{ block.lang }}" style="width:100%;"></div>
        <div style="display:flex;flex-wrap:wrap;gap:0.55rem 1.4rem;margin-top:0.55rem;">
          <span style="display:flex;align-items:center;gap:0.35rem;font-size:0.72rem;color:#5a5550;"><span style="width:11px;height:11px;background:#d4ede4;border-radius:2px;flex-shrink:0;display:inline-block;"></span><span class="ai-zh">Leading — 強且加速</span><span class="ai-en">Leading — strong &amp; accelerating</span></span>
          <span style="display:flex;align-items:center;gap:0.35rem;font-size:0.72rem;color:#5a5550;"><span style="width:11px;height:11px;background:#f5e9cf;border-radius:2px;flex-shrink:0;display:inline-block;"></span><span class="ai-zh">Weakening — 強但放緩</span><span class="ai-en">Weakening — strong but slowing</span></span>
          <span style="display:flex;align-items:center;gap:0.35rem;font-size:0.72rem;color:#5a5550;"><span style="width:11px;height:11px;background:#dbeafe;border-radius:2px;flex-shrink:0;display:inline-block;"></span><span class="ai-zh">Improving — 弱但轉強</span><span class="ai-en">Improving — weak but turning</span></span>
          <span style="display:flex;align-items:center;gap:0.35rem;font-size:0.72rem;color:#5a5550;"><span style="width:11px;height:11px;background:#f5dada;border-radius:2px;flex-shrink:0;display:inline-block;"></span><span class="ai-zh">Lagging — 弱且減速</span><span class="ai-en">Lagging — weak &amp; decelerating</span></span>
        </div>
        <script>
        (function(){
          var d = {{ rrg_data | safe }};
          if(d.traces) Plotly.newPlot('chart-rrg-{{ block.lang }}', d.traces, d.layout, {responsive:true,displayModeBar:false,scrollZoom:false,doubleClick:false,editable:false});
        })();
        </script>
      </div>

      {% else %}
      <div class="modcard" style="margin-top:0.4rem;color:var(--muted);font-size:0.82rem;">
        <span class="ai-zh">板塊資料載入中，請重新整理。</span><span class="ai-en">Sector data unavailable — refresh to retry.</span>
      </div>
      {% endif %}

      </div><!-- /tab-pane sectors -->

      <div class="foot">THE DAILY REGIME · {{ block.ui.footer }} · {{ block.report_date }}</div>
    </div>
    {% endfor %}
  </div>
  <script>
    (function () {
      // Tab navigation
      (function () {
        var tabBtns = Array.prototype.slice.call(document.querySelectorAll('.tab-btn'));
        function switchTab(pane) {
          tabBtns.forEach(function (b) { b.classList.toggle('active', b.getAttribute('data-pane') === pane); });
          document.querySelectorAll('.tab-pane').forEach(function (p) { p.classList.toggle('active', p.getAttribute('data-pane') === pane); });
          window.scrollTo({ top: 0, behavior: 'smooth' });
          // Trigger Plotly resize for charts in newly visible pane
          [50, 200].forEach(function (ms) {
            setTimeout(function () {
              if (window.Plotly) {
                document.querySelectorAll('.tab-pane.active .js-plotly-plot').forEach(function (g) { Plotly.Plots.resize(g); });
              }
            }, ms);
          });
        }
        tabBtns.forEach(function (btn) {
          btn.addEventListener('click', function () { switchTab(btn.getAttribute('data-pane')); });
        });
      })();
      var sel = '.hero, .zone-divider, .modcard, .viz-card, .story, .carry-cell';
      var targets = Array.prototype.slice.call(document.querySelectorAll(sel));
      targets.forEach(function (el) { el.classList.add('reveal'); });
      if ('IntersectionObserver' in window) {
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (e) {
            if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
          });
        }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
        targets.forEach(function (el) { io.observe(el); });
      } else {
        targets.forEach(function (el) { el.classList.add('in'); });
      }
      // On language switch, reveal everything in the now-visible block instantly
      document.addEventListener('click', function (ev) {
        if (ev.target.closest && ev.target.closest('.lang-switch button')) {
          setTimeout(function () {
            document.querySelectorAll('.reveal').forEach(function (el) { el.classList.add('in'); });
          }, 30);
        }
      });
      // Top scroll-progress bar — tracks window scroll
      var bar = document.querySelector('.scroll-progress');
      if (bar) {
        var upd = function () {
          var max = document.documentElement.scrollHeight - window.innerHeight;
          bar.style.width = (max > 0 ? (window.scrollY / max * 100) : 0) + '%';
        };
        window.addEventListener('scroll', upd, { passive: true });
        window.addEventListener('resize', upd);
        upd();
      }
    })();
  </script>
</body>
</html>
"""


def _clean_json(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[-1]
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0]
    start = s.find("{")
    end = s.rfind("}")
    return s[start : end + 1] if start >= 0 and end >= 0 else s


@lru_cache(maxsize=1)
def _deepseek_client() -> OpenAI:
    return OpenAI(
        api_key=require_deepseek_key(),
        base_url=get_deepseek_base_url(),
    )


def _load_quant_context_json(report_date_iso: str, l2b: dict) -> dict:
    """Quant payload from flow snapshot, flow_data files, or quant_engine JSON (nearest date)."""
    payload = (l2b.get("quant_engine") or {}).get("payload") or {}
    if isinstance(payload, dict) and payload.get("macro_regime_label"):
        return payload

    def _read_quant_file(path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  Warning: could not read {path.name}: {exc}", file=sys.stderr)
            return {}
        if not isinstance(data, dict):
            return {}
        if data.get("macro_regime_label"):
            return data
        nested = (data.get("quant_engine") or {}).get("payload") or {}
        return nested if isinstance(nested, dict) and nested.get("macro_regime_label") else {}

    exact = OUTPUT_DIR / f"quant_engine_{report_date_iso}.json"
    if exact.exists():
        data = _read_quant_file(exact)
        if data:
            print(
                f"  Quant context from {exact.name} (flow_data missing quant_engine.payload).",
                file=sys.stderr,
            )
            return data

    target = date.fromisoformat(report_date_iso)

    flow_candidates: list[tuple[date, Path]] = []
    for path in OUTPUT_DIR.glob("flow_data_20*.json"):
        stem = path.stem.replace("flow_data_", "")
        try:
            flow_candidates.append((date.fromisoformat(stem), path))
        except ValueError:
            continue
    if flow_candidates:
        on_or_before = [c for c in flow_candidates if c[0] <= target]
        pick = (
            on_or_before[-1]
            if on_or_before
            else min(flow_candidates, key=lambda c: abs((c[0] - target).days))
        )
        data = _read_quant_file(pick[1])
        if data:
            print(
                f"  Quant context from {pick[1].name} (nearest flow snapshot for {report_date_iso}).",
                file=sys.stderr,
            )
            return data

    q_candidates: list[tuple[date, Path]] = []
    for path in OUTPUT_DIR.glob("quant_engine_20*.json"):
        stem = path.stem.replace("quant_engine_", "")
        try:
            q_candidates.append((date.fromisoformat(stem), path))
        except ValueError:
            continue
    if not q_candidates:
        return {}

    with_regime = [(dt, p) for dt, p in q_candidates if _read_quant_file(p)]
    pool = with_regime or q_candidates
    on_or_before = [c for c in pool if c[0] <= target]
    pick = on_or_before[-1] if on_or_before else min(pool, key=lambda c: abs((c[0] - target).days))
    data = _read_quant_file(pick[1])
    if data:
        print(
            f"  Quant context from {pick[1].name} (no exact payload for {report_date_iso}).",
            file=sys.stderr,
        )
        return data
    return {}


def _guard_skip_llm_overwrite(cache_path: Path, skip_llm: bool) -> None:
    """Prevent accidental --skip-llm from replacing a successful CIO run."""
    if not skip_llm or not cache_path.exists():
        return
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if cached.get("llm_placeholder"):
        return
    meta = cached.get("llm_meta") if isinstance(cached.get("llm_meta"), dict) else {}
    if meta.get("status") == "ok":
        raise RuntimeError(
            f"Refusing --skip-llm: {cache_path.name} already has LLM status=ok "
            f"({meta.get('provider')}/{meta.get('model')}). "
            "Run without --skip-llm to refresh, or delete the JSON first."
        )


def is_synthesis_placeholder(synthesis: dict | None) -> bool:
    """True when War Room CIO fields are still the offline placeholder."""
    if not isinstance(synthesis, dict):
        return True
    directive = synthesis.get("cio_directive") if isinstance(synthesis.get("cio_directive"), dict) else {}
    rel = synthesis.get("relationship_analysis") if isinstance(synthesis.get("relationship_analysis"), dict) else {}
    pa = rel.get("pure_alpha") if isinstance(rel.get("pure_alpha"), dict) else {}
    stance = str(directive.get("the_stance") or "").strip()
    exp = str(pa.get("expectation_arbitrage") or "").strip()
    crowd = str(pa.get("crowdedness_leverage") or "").strip()
    hedge = str(pa.get("hedging_cost_anomaly") or "").strip()
    if (
        stance
        and exp
        and crowd
        and hedge
        and stance != FALLBACK
        and exp != FALLBACK
        and crowd != FALLBACK
        and hedge != FALLBACK
    ):
        return False
    if stance == FALLBACK or exp == FALLBACK:
        return True
    headline = str(synthesis.get("headline") or "").strip()
    thesis = str(synthesis.get("cio_macro_thesis") or "").strip()
    return headline == FALLBACK or thesis == FALLBACK


def _parse_cio_json(raw: str) -> dict | None:
    if not raw.strip():
        return None
    try:
        payload = json.loads(_clean_json(raw))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if _CIO_REQUIRED_KEYS.issubset(payload.keys()):
        return payload
    if _LEGACY_CIO_KEYS.issubset(payload.keys()):
        return payload
    return None


def _coerce_narrative(value: object) -> list[str]:
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        if len(parts) >= 2:
            return parts[:2]
        if len(parts) == 1:
            return _coerce_narrative(parts[0])
    if isinstance(value, str) and value.strip():
        chunks = [p.strip() for p in value.split("\n\n") if p.strip()]
        if len(chunks) >= 2:
            return chunks[:2]
        return [value.strip(), value.strip()]
    return [FALLBACK, FALLBACK]


def _migrate_legacy_synthesis(out: dict) -> dict:
    """Map pre-3.0 CIO JSON fields into War Room 3.0 zones when new keys are absent."""
    z1 = out.get("zone1_pulse")
    if not isinstance(z1, dict) or not z1.get("market_overview"):
        mp = out.get("macro_pulse") if isinstance(out.get("macro_pulse"), dict) else {}
        thesis = str(out.get("cio_macro_thesis") or "").strip()
        headline = str(out.get("headline") or "").strip()
        out["zone1_pulse"] = {
            "market_overview": mp.get("event_metrics") or thesis or headline or FALLBACK,
            "news_macro_context": mp.get("event_title") or mp.get("cio_verdict") or FALLBACK,
        }
    directive = out.get("cio_directive")
    if not isinstance(directive, dict) or not directive.get("the_stance"):
        mp = out.get("macro_pulse") if isinstance(out.get("macro_pulse"), dict) else {}
        dyn = str(out.get("structural_vs_tactical_dynamics") or "").strip()
        thesis = str(out.get("cio_macro_thesis") or "").strip()
        risks = out.get("core_risks_actionable")
        watch = risks[0] if isinstance(risks, list) and risks else FALLBACK
        out["cio_directive"] = {
            "the_stance": mp.get("cio_verdict") or str(out.get("headline") or FALLBACK),
            "the_narrative": _coerce_narrative(dyn or thesis),
            "the_watchout": watch,
        }
    rel = out.get("relationship_analysis")
    if not isinstance(rel, dict) or not rel.get("pure_alpha"):
        cd = out.get("convergence_divergence") if isinstance(out.get("convergence_divergence"), dict) else {}
        body = str(out.get("cio_macro_thesis") or out.get("structural_vs_tactical_dynamics") or FALLBACK)
        trap = FALLBACK
        if isinstance(rel, dict):
            trap = str(rel.get("the_trap") or FALLBACK)
            body = str(rel.get("body") or body)
        out["relationship_analysis"] = {
            "status": cd.get("status") or "CONVERGENCE",
            "pure_alpha": {
                "expectation_arbitrage": (
                    (rel or {}).get("consensus_view") or (rel or {}).get("data_reality") or body
                ),
                "crowdedness_leverage": trap,
                "hedging_cost_anomaly": (rel or {}).get("data_reality") or body,
            },
            "divergence_explanation": cd.get("reason") or FALLBACK,
            "body": body,
        }
    z1 = out.get("zone1_pulse")
    if isinstance(z1, dict) and not z1.get("flash_bullets"):
        z1["flash_bullets"] = {
            "capital_flows": z1.get("market_overview") or FALLBACK,
            "volatility": FALLBACK,
            "macro_sentiment": z1.get("news_macro_context") or FALLBACK,
        }
    return out


def _coerce_bullet_list(value: object) -> list[str]:
    if isinstance(value, list):
        items = [str(x).strip() for x in value if str(x).strip()]
        return items or [FALLBACK]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return [FALLBACK]


def _fill_zone1_from_pre_analysis(out: dict) -> None:
    """Derive Bloomberg Flash bullets from CoT when the LLM skipped zone1_pulse."""
    z1 = out.get("zone1_pulse")
    if not isinstance(z1, dict):
        return
    bullets = z1.get("flash_bullets")
    if not isinstance(bullets, dict):
        return
    if bullets.get("capital_flows") != FALLBACK:
        return
    pa = out.get("pre_analysis")
    if not isinstance(pa, dict):
        return
    ev = str(pa.get("event_pulse_analysis") or "").strip()
    mom = str(pa.get("momentum_verification") or "").strip()
    xlayer = str(pa.get("cross_layer_conflict") or "").strip()
    if ev == FALLBACK and mom == FALLBACK:
        return
    bullets["capital_flows"] = ev or FALLBACK
    bullets["volatility"] = mom or FALLBACK
    bullets["macro_sentiment"] = xlayer or FALLBACK
    z1["market_overview"] = bullets["capital_flows"]
    z1["news_macro_context"] = bullets["macro_sentiment"]


_JARGON_PATTERNS = re.compile(
    r"\bz\s*=\s*-?\d|\bSKEW\b|\bVVIX\b|\bDXY\b|\bHYG\b|\bSPY\b|SHY/TLT|RSP/SPY|"
    r"[A-Z]{3}/[A-Z]{3}|USD/[A-Z]{3}|[A-Z]{3}/USD|divergence score|"
    r"\bbp\b|\$\d+\s*bn|\bLayer\s*[12]\b|RED signal|YELLOW signal|\(\d{2}\)",
    re.IGNORECASE,
)


def _warn_jargon_in_field(label: str, text: str) -> None:
    """Print a warning if user-facing text still contains quant jargon."""
    if not text or text == FALLBACK:
        return
    hits = _JARGON_PATTERNS.findall(text)
    if hits:
        print(
            f"  ⚠ JARGON LEAK in {label}: {sorted(set(h for h in hits if h))[:6]} "
            f"— LLM ignored No-Jargon Rule.",
            file=sys.stderr,
        )


# Final-line defense: replace any ticker/jargon that leaked into user-facing text.
# Order matters — longer phrases first so "SKEW vs VVIX" resolves before single tokens.
_JARGON_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"SKEW\s*(?:vs\.?|/|and)\s*VVIX", re.I), "downside-hedging demand vs surface volatility"),
    (re.compile(r"\bSHY\s*/\s*TLT\b", re.I), "the bond market's rate-cut expectations"),
    (re.compile(r"\bRSP\s*/\s*SPY\b", re.I), "how broadly the rally is participating"),
    (re.compile(r"\bTIP\s*/\s*IEF\b", re.I), "inflation-breakeven signals"),
    (re.compile(r"\bSKEW\b", re.I), "demand for downside protection"),
    (re.compile(r"\bVVIX\b", re.I), "the volatility of volatility"),
    (re.compile(r"\bDXY\b", re.I), "the US dollar"),
    (re.compile(r"\bHYG\b", re.I), "high-yield credit"),
    (re.compile(r"\bSPY\b", re.I), "the US stock market"),
    (re.compile(r"\bNZD/JPY\b", re.I), "the New Zealand dollar carry trade"),
    (re.compile(r"\bAUD/JPY\b", re.I), "the Australian dollar carry trade"),
    (re.compile(r"\bUSD/CNH\b", re.I), "the Chinese yuan"),
    (re.compile(r"\bUSD/JPY\b", re.I), "the Japanese yen"),
    (re.compile(r"\bsignal is (?:RED|YELLOW)\b", re.I), "signals caution"),
    (re.compile(r"\bsignal is GREEN\b", re.I), "signals healthy flows"),
    (re.compile(r"\b(?:RED|YELLOW) signal\b", re.I), "caution signal"),
    (re.compile(r"\bGREEN signal\b", re.I), "healthy-flow signal"),
    (re.compile(r"\bT3MFF\b", re.I), "financial conditions"),
    (re.compile(r"\bVIX\b", re.I), "market volatility"),
    (re.compile(r"\bnet short(?:\s+position(?:ing|s)?)?\b", re.I), "bearish bets"),
    (re.compile(r"\bnet long(?:\s+position(?:ing|s)?)?\b", re.I), "bullish bets"),
    (re.compile(r"\bdivergence score\b", re.I), "the gap between markets"),
    (re.compile(r"\bLayer\s*[12]\b", re.I), "the underlying data"),
    (re.compile(r"\s*\(?\bz\s*=\s*-?\d+\.?\d*\)?", re.I), ""),
    (re.compile(r"\s*\(\d{2}\)"), ""),
    # Prevent LLM from leaking raw historical return figures (e.g. "-38.9%", "+41.2%")
    # when it copies numbers from the historical analogues context block.
    (re.compile(r"\b([+-]?\d{1,3}\.\d+)%\s*(?:over\s+12m|12.month|twelve.month)", re.I), ""),
    # Remove redundant parenthetical when sanitizer creates "X (X ratio)" double phrases.
    (re.compile(r"\b(inflation[\w\s-]{0,20}expectations)\s*\([^)]{0,60}inflation[\w\s-]{0,20}expectations[^)]{0,30}\)", re.I), r"\1"),
    (re.compile(r"\b(rate.cut expectations)\s*\([^)]{0,60}rate.cut expectations[^)]{0,30}\)", re.I), r"\1"),
]


def _sanitize_jargon(text: object) -> str:
    """Strip leaked tickers/jargon from a user-facing string."""
    s = str(text or "")
    if not s.strip():
        return s
    for pat, repl in _JARGON_REPLACEMENTS:
        s = pat.sub(repl, s)
    cleaned = re.sub(r"\s{2,}", " ", s).strip()
    # Never let sanitization blank out a previously non-empty field.
    return cleaned or str(text).strip()


def _derive_theme_title(raw_title: object, raw_body: object) -> tuple[str, str]:
    """
    Return (title, body) for a theme card.

    DeepSeek frequently omits the title, or returns it as a verbatim copy of the
    body's first sentence. This produces a clean short title without mid-word
    truncation, and strips any duplicate opening from the body.
    """
    title = _sanitize_jargon(raw_title).strip()
    body = _sanitize_jargon(raw_body).strip()

    def _shorten(text: str, max_words: int = 8) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text.rstrip(" ,.;:")
        return " ".join(words[:max_words]).rstrip(" ,.;:") + "…"

    # Case 1: usable title that is NOT just the body's opening — keep it.
    if title and not (body and body.lower().startswith(title.lower().rstrip("…").rstrip())):
        return _shorten(title, 10), body

    # Case 2: no title, or title duplicates body. Derive from first sentence.
    if not body:
        return (title or "市場主題"), body

    sentences = re.split(r"(?<=[.!?])\s+", body)
    first = sentences[0].strip()
    rest = " ".join(sentences[1:]).strip()

    derived = _shorten(first, 8)

    # If removing the first sentence leaves enough body, do so to avoid duplication.
    if rest and len(rest) >= 30:
        return derived, rest
    # Otherwise keep the full body and use a shortened lead as the title.
    return derived, body


def _normalize_zone1_pulse(z1: dict) -> dict:
    fb = _synthesis_fallback()["zone1_pulse"]
    bullets = z1.get("flash_bullets")
    if not isinstance(bullets, dict):
        bullets = {}
    z1["flash_bullets"] = {
        "capital_flows": bullets.get("capital_flows") or z1.get("market_overview") or FALLBACK,
        "volatility": bullets.get("volatility") or FALLBACK,
        "macro_sentiment": bullets.get("macro_sentiment") or z1.get("news_macro_context") or FALLBACK,
    }
    z1.setdefault("market_overview", z1["flash_bullets"]["capital_flows"])
    z1.setdefault("news_macro_context", z1["flash_bullets"]["macro_sentiment"])
    return z1


def _normalize_relationship_analysis(rel: dict) -> dict:
    rel.setdefault("status", "CONVERGENCE")
    rel.setdefault("divergence_explanation", FALLBACK)
    rel.setdefault("body", FALLBACK)
    pa = rel.get("pure_alpha")
    if not isinstance(pa, dict):
        pa = {}
    if not str(pa.get("expectation_arbitrage") or "").strip():
        pa["expectation_arbitrage"] = rel.get("consensus_view") or rel.get("data_reality") or FALLBACK
    if not str(pa.get("crowdedness_leverage") or "").strip():
        pa["crowdedness_leverage"] = rel.get("the_trap") or FALLBACK
    if not str(pa.get("hedging_cost_anomaly") or "").strip():
        pa["hedging_cost_anomaly"] = rel.get("data_reality") or FALLBACK
    rel["pure_alpha"] = {
        "expectation_arbitrage": str(pa.get("expectation_arbitrage") or FALLBACK).strip() or FALLBACK,
        "crowdedness_leverage": str(pa.get("crowdedness_leverage") or FALLBACK).strip() or FALLBACK,
        "hedging_cost_anomaly": str(pa.get("hedging_cost_anomaly") or FALLBACK).strip() or FALLBACK,
    }
    return rel


def _regime_gauge_tone(label: str) -> str:
    mapping = {
        "Goldilocks": "stable",
        "Overheat": "warning",
        "Stagflation": "risk",
        "Deflationary Bust": "warning",
    }
    return mapping.get(str(label or "").strip(), "neutral")


_REGIME_ASSET_TILT: dict[str, list[dict]] = {
    "Goldilocks": [
        {"asset": "Equities",       "tilt": "overweight",  "icon": "↑↑", "level": 2},
        {"asset": "Credit",         "tilt": "overweight",  "icon": "↑",  "level": 1},
        {"asset": "Real Assets",    "tilt": "neutral",     "icon": "—",  "level": 0},
        {"asset": "Long Bonds",     "tilt": "neutral",     "icon": "—",  "level": 0},
        {"asset": "Cash",           "tilt": "underweight", "icon": "↓",  "level": -1},
        {"asset": "Gold",           "tilt": "neutral",     "icon": "—",  "level": 0},
    ],
    "Overheat": [
        {"asset": "Equities",       "tilt": "overweight",  "icon": "↑↑", "level": 2},
        {"asset": "Commodities",    "tilt": "overweight",  "icon": "↑↑", "level": 2},
        {"asset": "Real Assets",    "tilt": "overweight",  "icon": "↑",  "level": 1},
        {"asset": "EM Assets",      "tilt": "overweight",  "icon": "↑",  "level": 1},
        {"asset": "Long Bonds",     "tilt": "underweight", "icon": "↓↓", "level": -2},
        {"asset": "Cash",           "tilt": "underweight", "icon": "↓↓", "level": -2},
    ],
    "Stagflation": [
        {"asset": "Gold",           "tilt": "overweight",  "icon": "↑↑", "level": 2},
        {"asset": "Commodities",    "tilt": "overweight",  "icon": "↑↑", "level": 2},
        {"asset": "Real Assets",    "tilt": "overweight",  "icon": "↑",  "level": 1},
        {"asset": "Equities",       "tilt": "underweight", "icon": "↓",  "level": -1},
        {"asset": "Long Bonds",     "tilt": "underweight", "icon": "↓↓", "level": -2},
        {"asset": "Credit",         "tilt": "underweight", "icon": "↓",  "level": -1},
    ],
    "Deflationary Bust": [
        {"asset": "Long Bonds",     "tilt": "overweight",  "icon": "↑↑", "level": 2},
        {"asset": "Cash",           "tilt": "overweight",  "icon": "↑",  "level": 1},
        {"asset": "Gold",           "tilt": "neutral",     "icon": "—",  "level": 0},
        {"asset": "Equities",       "tilt": "underweight", "icon": "↓↓", "level": -2},
        {"asset": "Credit",         "tilt": "underweight", "icon": "↓",  "level": -1},
        {"asset": "Commodities",    "tilt": "underweight", "icon": "↓↓", "level": -2},
    ],
}


def _regime_asset_tilt(label: str) -> list[dict]:
    """Return Bridgewater-style asset tilt for the current macro regime."""
    return _REGIME_ASSET_TILT.get(str(label or "").strip(), [])


def _divergence_prompt_context(
    quant_momentum: dict | None,
    l1: dict,
    l2a: dict,
    l2b: dict,
    quant_context: dict | None = None,
) -> dict:
    """Divergence drivers for Zone 3 — prefer quant Risk_Analytics when present."""
    qc = quant_context if isinstance(quant_context, dict) else {}
    ra = qc.get("Risk_Analytics") if isinstance(qc.get("Risk_Analytics"), dict) else {}
    if ra.get("divergence_score") is not None:
        score = max(0, min(100, int(ra["divergence_score"])))
        if score >= 70:
            level = "High"
        elif score >= 40:
            level = "Elevated"
        else:
            level = "Low"
        return {
            "source": "quant_engine",
            "macro_regime_label": qc.get("macro_regime_label"),
            "macro_regime_detail": qc.get("macro_regime_detail"),
            "divergence_score": score,
            "estimated_score_pct": score,
            "estimated_level": level,
            "penalty_reason": ra.get("penalty_reason", ""),
            "penalty_reasons": ra.get("penalty_reasons", []),
            "layer2_risk": ra.get("layer2_risk"),
            "cpi_yoy_context": qc.get("cpi_yoy_context"),
            "likely_status": "DIVERGENCE" if score >= 55 else "CONVERGENCE",
            "drivers": [str(ra.get("penalty_reason") or "Quant L1/L2 gap with penalty rules")],
        }

    momentum = quant_momentum if isinstance(quant_momentum, dict) else {}
    stress_keys = ("MOVE", "VIX", "DXY", "JPY", "TWD")
    risk_on_keys = ("SPY", "HYG")
    stress_bad = sum(
        1 for k in stress_keys if (momentum.get(k) or {}).get("label") == "Deteriorating"
    )
    risk_good = sum(
        1 for k in risk_on_keys if (momentum.get(k) or {}).get("label") == "Improving"
    )
    vol_improving = sum(
        1 for k in ("MOVE", "VIX") if (momentum.get(k) or {}).get("label") == "Improving"
    )
    score = 48
    level = "Moderate"
    likely_status = "DIVERGENCE"
    drivers: list[str] = []

    l1_llm = l1.get("llm") if isinstance(l1.get("llm"), dict) else {}
    flow_sig = (l2b.get("flow_payload") or {}).get("signals", {}) if isinstance(l2b, dict) else {}
    green_flows = sum(
        1
        for c in COUNTRY_CYCLE_KEYS
        if isinstance(flow_sig.get(c), dict) and str(flow_sig[c].get("signal", "")).upper() == "GREEN"
    )

    if stress_bad >= 2 and risk_good >= 1:
        score = 85
        level = "High"
        drivers.append(
            "L2 risk-on (SPY/HYG momentum Improving) vs stress/liquidity indicators (MOVE/VIX/DXY/JPY/TWD) deteriorating"
        )
    elif stress_bad >= 1 and risk_good >= 1:
        score = 62
        level = "Elevated"
        drivers.append("Mixed momentum: risk assets improving while USD/JPY or liquidity metrics deteriorate")
    elif vol_improving >= 2 and risk_good >= 2:
        score = 22
        level = "Low"
        likely_status = "CONVERGENCE"
        drivers.append("Vol compression and risk-on momentum aligned across SPY, HYG, VIX, MOVE")
    else:
        drivers.append("Partial alignment between Layer 1 structural themes and Layer 2 flow signals")

    if green_flows >= 4 and l1_llm:
        drivers.append(
            f"Tactical flows broadly green ({green_flows}/6 regions) against Layer 1 structural caution"
        )
        if score < 70:
            score = min(78, score + 12)

    mom_notes = [
        f"{k} {momentum[k].get('label')}"
        for k in ("MOVE", "VIX", "DXY", "JPY", "TWD", "SPY", "HYG")
        if k in momentum and isinstance(momentum[k], dict) and momentum[k].get("label")
    ]
    return {
        "estimated_score_pct": max(0, min(100, int(score))),
        "estimated_level": level,
        "likely_status": likely_status,
        "drivers": drivers,
        "momentum_snapshot": mom_notes,
        "flow_green_count": green_flows,
    }


def _directive_has_forbidden_numbers(text: str) -> bool:
    return bool(_DIRECTIVE_NUMBER_RE.search(text or ""))


def _warn_directive_numbers(directive: dict) -> None:
    parts = [
        str(directive.get("the_stance") or ""),
        " ".join(_coerce_narrative(directive.get("the_narrative"))),
        str(directive.get("the_watchout") or ""),
    ]
    if any(_directive_has_forbidden_numbers(p) for p in parts):
        print(
            "  Warning: Zone 2 (cio_directive) may contain numeric literals; prompt requires number-free prose.",
            file=sys.stderr,
        )


def _fetch_close_and_change(ticker: str) -> tuple[float | None, float | None, str | None]:
    try:
        hist = yf.download(ticker, period="7d", interval="1d", progress=False, auto_adjust=False)
        if hist is None or hist.empty:
            return None, None, None
        close = hist["Close"]
        series = close.iloc[:, 0] if getattr(close, "ndim", 1) > 1 else close
        series = series.dropna()
        if series.empty:
            return None, None, None
        last = float(series.iloc[-1])
        prev = float(series.iloc[-2]) if len(series) >= 2 else None
        chg_pct = ((last / prev) - 1.0) * 100.0 if prev not in (None, 0.0) else None
        return last, chg_pct, str(series.index[-1].date())
    except Exception:
        return None, None, None


def _market_snapshot_for_prompt(report_date: str) -> list[dict]:
    labels = [
        ("S&P 500", "^GSPC"),
        ("Dow Jones", "^DJI"),
        ("Nikkei 225", "^N225"),
        ("Taiwan Index", "^TWII"),
        ("Oil (WTI)", "CL=F"),
        ("Gold", "GC=F"),
        ("Bitcoin", "BTC-USD"),
    ]
    rows: list[dict] = []
    for name, ticker in labels:
        px, chg_pct, asof = _fetch_close_and_change(ticker)
        rows.append(
            {
                "name": name,
                "ticker": ticker,
                "last": px,
                "chg_1d_pct": round(chg_pct, 2) if chg_pct is not None else None,
                "as_of": asof,
            }
        )
    return rows


def _call_deepseek(prompt: str, model: str) -> tuple[dict | None, str | None]:
    try:
        resp = _deepseek_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CIO_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": prompt.strip()},
            ],
            response_format={"type": "json_object"},
            temperature=0.35,
            max_tokens=8192,
        )
        raw = (resp.choices[0].message.content or "").strip()
        payload = _parse_cio_json(raw)
        if payload is None:
            finish = getattr(resp.choices[0], "finish_reason", None)
            if finish == "length":
                return None, "invalid_json_truncated"
            return None, "invalid_json"
        return payload, None
    except Exception as exc:
        print(f"  DeepSeek call failed ({model}): {exc}", file=sys.stderr)
        return None, str(exc)


def _gemini_models_to_try() -> list[str]:
    primary = get_gemini_model()
    seen: set[str] = set()
    ordered: list[str] = []
    for m in (primary, "gemini-2.5-flash-lite", "gemini-2.0-flash-lite", "gemini-2.5-flash"):
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _call_gemini_cio(prompt: str, model: str) -> tuple[dict | None, str | None]:
    try:
        client = genai.Client(api_key=require_gemini_key())
        response = client.models.generate_content(
            model=model,
            contents=prompt.strip(),
            config=types.GenerateContentConfig(
                system_instruction=(
                    CIO_SYSTEM_PROMPT.strip()
                    + "\n\nReturn strict JSON only. No markdown fences."
                ),
                max_output_tokens=8192,
                temperature=0.35,
            ),
        )
        raw = response.text or ""
        if not raw:
            parts: list[str] = []
            for cand in response.candidates or []:
                for part in cand.content.parts or []:
                    if getattr(part, "text", None):
                        parts.append(part.text)
            raw = "\n".join(parts)
        payload = _parse_cio_json(raw)
        if payload is None:
            return None, "invalid_json"
        return payload, None
    except Exception as exc:
        print(f"  Gemini CIO fallback failed ({model}): {exc}", file=sys.stderr)
        return None, str(exc)


def _synthesis_fallback() -> dict:
    fb = {
        "pre_analysis": {
            "cross_layer_conflict": FALLBACK,
            "momentum_verification": FALLBACK,
            "event_pulse_analysis": FALLBACK,
        },
        "zone1_pulse": {
            "flash_bullets": {
                "capital_flows": FALLBACK,
                "volatility": FALLBACK,
                "macro_sentiment": FALLBACK,
            },
            "market_overview": FALLBACK,
            "news_macro_context": FALLBACK,
        },
        "cio_directive": {
            "the_stance": FALLBACK,
            "the_narrative": [FALLBACK, FALLBACK],
            "the_watchout": FALLBACK,
        },
        "relationship_analysis": {
            "status": "CONVERGENCE",
            "pure_alpha": {
                "expectation_arbitrage": FALLBACK,
                "crowdedness_leverage": FALLBACK,
                "hedging_cost_anomaly": FALLBACK,
            },
            "divergence_explanation": FALLBACK,
            "body": FALLBACK,
        },
        "convergence_divergence": {
            "status": "CONVERGENCE",
            "reason": FALLBACK,
        },
        "watch_list": [FALLBACK],
        "country_attribution_notes": {
            "US": FALLBACK,
            "Japan": FALLBACK,
            "Europe": FALLBACK,
            "UK": FALLBACK,
            "China": FALLBACK,
            "Taiwan": FALLBACK,
        },
        "gear_matrix_semantics": _empty_gear_matrix_semantics(),
    }
    return fb


def _empty_gear_matrix_semantics() -> dict[str, dict[str, str]]:
    block = {
        "expectation": FALLBACK,
        "expectation_en": FALLBACK,
        "expectation_zh": FALLBACK,
        "plumbing": FALLBACK,
        "plumbing_en": FALLBACK,
        "plumbing_zh": FALLBACK,
        "hedging": FALLBACK,
        "hedging_en": FALLBACK,
        "hedging_zh": FALLBACK,
        "expectation_label": "",
        "expectation_label_en": "",
        "expectation_label_zh": "",
        "plumbing_label": "",
        "plumbing_label_en": "",
        "plumbing_label_zh": "",
        "hedging_label": "",
        "hedging_label_en": "",
        "hedging_label_zh": "",
    }
    return {country: dict(block) for country in COUNTRY_CYCLE_KEYS}


def _normalize_gear_matrix_semantics(raw: dict | None) -> dict[str, dict[str, str]]:
    out = _empty_gear_matrix_semantics()
    if not isinstance(raw, dict):
        return out
    for country in COUNTRY_CYCLE_KEYS:
        block = raw.get(country)
        if not isinstance(block, dict):
            continue
        out[country] = _normalize_gear_block_bilingual({**out[country], **block})
    return out


def _normalize_synthesis(out: dict) -> dict:
    """Coerce War Room 3.0 LLM fields for stable HTML rendering."""
    out = _migrate_legacy_synthesis(out)
    fb = _synthesis_fallback()

    pa = out.get("pre_analysis")
    if not isinstance(pa, dict):
        out["pre_analysis"] = fb["pre_analysis"]
    else:
        for k in ("cross_layer_conflict", "momentum_verification", "event_pulse_analysis"):
            pa.setdefault(k, FALLBACK)

    z1 = out.get("zone1_pulse")
    if not isinstance(z1, dict):
        out["zone1_pulse"] = dict(fb["zone1_pulse"])
    else:
        out["zone1_pulse"] = _normalize_zone1_pulse(z1)

    directive = out.get("cio_directive")
    if not isinstance(directive, dict):
        out["cio_directive"] = fb["cio_directive"]
    else:
        directive.setdefault("the_stance", FALLBACK)
        directive["the_narrative"] = _coerce_narrative(directive.get("the_narrative"))
        directive.setdefault("the_watchout", FALLBACK)

    rel = out.get("relationship_analysis")
    if not isinstance(rel, dict):
        out["relationship_analysis"] = dict(fb["relationship_analysis"])
        rel = out["relationship_analysis"]
    else:
        out["relationship_analysis"] = _normalize_relationship_analysis(rel)
        rel = out["relationship_analysis"]

    cd = out.get("convergence_divergence")
    if not isinstance(cd, dict):
        out["convergence_divergence"] = {"status": rel.get("status", "CONVERGENCE"), "reason": FALLBACK}
    else:
        cd.setdefault("reason", FALLBACK)
        if not cd.get("status"):
            cd["status"] = rel.get("status", "CONVERGENCE")
    if rel.get("status") and not str(cd.get("status") or "").strip():
        cd["status"] = rel["status"]
    if cd.get("status") and not rel.get("status"):
        rel["status"] = cd["status"]

    notes = out.get("country_attribution_notes")
    if not isinstance(notes, dict):
        out["country_attribution_notes"] = fb["country_attribution_notes"]

    out["gear_matrix_semantics"] = _normalize_gear_matrix_semantics(out.get("gear_matrix_semantics"))

    _fill_zone1_from_pre_analysis(out)
    _warn_directive_numbers(out["cio_directive"])

    # Bilingual reader-facing fields (lite + legacy English keys).
    out["daily_themes"] = _normalize_daily_themes_bilingual(out.get("daily_themes") or [])
    out["cio_directive"] = _normalize_cio_directive_bilingual(
        out["cio_directive"] if isinstance(out.get("cio_directive"), dict) else {}
    )
    raw_wl = out.get("watch_list_bilingual")
    if not isinstance(raw_wl, list) or not raw_wl:
        raw_wl = out.get("watch_list")
    if not isinstance(raw_wl, list):
        raw_wl = []
    wl_en, wl_bi = _normalize_watch_list_bilingual(raw_wl)
    out["watch_list"] = wl_en or [FALLBACK]
    out["watch_list_bilingual"] = wl_bi

    directive = out["cio_directive"]
    # Per-country CIO notes (shown in gear matrix CIO column for non-US countries)
    notes = out.get("country_attribution_notes")
    if isinstance(notes, dict):
        for c in list(notes.keys()):
            notes[c] = _sanitize_jargon(notes[c])
    # pure_alpha drives the US CIO column + relationship analysis prose
    rel = out.get("relationship_analysis")
    if isinstance(rel, dict):
        rel["divergence_explanation"] = _sanitize_jargon(rel.get("divergence_explanation"))
        rel["body"] = _sanitize_jargon(rel.get("body"))
        pa = rel.get("pure_alpha")
        if isinstance(pa, dict):
            for fld in ("expectation_arbitrage", "crowdedness_leverage", "hedging_cost_anomaly"):
                if fld in pa:
                    pa[fld] = _sanitize_jargon(pa[fld])
    return out


def _deepseek_models_to_try() -> list[str]:
    primary = get_deepseek_model()
    seen: set[str] = set()
    ordered: list[str] = []
    for m in (primary, *DEEPSEEK_MODEL_FALLBACKS):
        if m and m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _get_synthesis(prompt: str, skip_llm: bool) -> tuple[dict, dict]:
    fallback = _synthesis_fallback()
    llm_meta: dict = {
        "provider": "deepseek",
        "status": "failed",
        "model": None,
        "error": None,
    }

    if skip_llm:
        llm_meta["status"] = "skipped"
        llm_meta["error"] = "skip-llm flag set"
        print("  Layer 3 LLM skipped (--skip-llm); using placeholder CIO text.", file=sys.stderr)
        return fallback, llm_meta

    last_error: str | None = None
    for model in _deepseek_models_to_try():
        out, err = _call_deepseek(prompt, model)
        if out:
            print(f"  DeepSeek synthesis succeeded ({model})", file=sys.stderr)
            llm_meta.update({"provider": "deepseek", "status": "ok", "model": model, "error": None})
            return _normalize_synthesis({**fallback, **out}), llm_meta
        last_error = err or "unknown"
        if err == "invalid_json_truncated":
            print(
                "  DeepSeek JSON truncated (raise max_tokens or shorten prompt); trying fallback...",
                file=sys.stderr,
            )

    print("  DeepSeek failed; trying Gemini CIO fallback...", file=sys.stderr)
    for model in _gemini_models_to_try():
        out, err = _call_gemini_cio(prompt, model)
        if out:
            print(f"  Gemini CIO fallback succeeded ({model})", file=sys.stderr)
            llm_meta.update(
                {
                    "provider": "gemini_fallback",
                    "status": "ok",
                    "model": model,
                    "error": None,
                }
            )
            return _normalize_synthesis({**fallback, **out}), llm_meta
        last_error = err or last_error

    llm_meta["error"] = last_error or "all providers failed"
    print("  Layer 3 LLM failed for all models; using placeholder CIO text.", file=sys.stderr)
    return fallback, llm_meta


def _resolve_event_of_day(l2b: dict) -> dict:
    """Top surfaced flow anomaly for Event of the Day (deterministic)."""
    empty = {
        "name": "No surfaced event",
        "latest": "—",
        "change": "—",
        "z": "—",
        "signal": "YELLOW",
        "attention": "—",
    }
    if not isinstance(l2b, dict):
        return empty
    ev = l2b.get("event_of_day")
    if isinstance(ev, dict) and ev.get("name"):
        return {
            "name": str(ev.get("name", "—")),
            "latest": str(ev.get("latest", "—")),
            "change": str(ev.get("change", "—")),
            "z": str(ev.get("z", "—")),
            "signal": str(ev.get("signal", "YELLOW")),
            "attention": str(ev.get("attention", ev.get("attention_val", "—"))),
        }
    rows = l2b.get("surfaced_rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        r = rows[0]
        return {
            "name": str(r.get("name", "—")),
            "latest": str(r.get("latest", "—")),
            "change": str(r.get("change", "—")),
            "z": str(r.get("z", "—")),
            "signal": str(r.get("signal", "YELLOW")),
            "attention": str(r.get("attention", r.get("attention_val", "—"))),
        }
    panel = l2b.get("validation_panel")
    if isinstance(panel, list) and panel:
        best = max(
            panel,
            key=lambda x: float(x.get("attention_val", x.get("z_val") or 0) or 0)
            if isinstance(x, dict)
            else 0,
        )
        if isinstance(best, dict):
            return {
                "name": str(best.get("name", "—")),
                "latest": str(best.get("latest", "—")),
                "change": str(best.get("chg_1m", best.get("change", "—"))),
                "z": str(best.get("z", "—")),
                "signal": str(best.get("signal", "YELLOW")),
                "attention": str(best.get("attention", "—")),
            }
    return empty


def _divergence_gauge_from_quant(
    quant_context: dict | None,
    synthesis: dict,
    quant_momentum: dict | None = None,
) -> dict:
    """Prefer quant Risk_Analytics.divergence_score; fall back to heuristic gauge."""
    qc = quant_context if isinstance(quant_context, dict) else {}
    ra = qc.get("Risk_Analytics") if isinstance(qc.get("Risk_Analytics"), dict) else {}
    score_raw = ra.get("divergence_score")
    if score_raw is not None:
        score = max(0, min(100, int(score_raw)))
        if score >= 70:
            level, tone = "High", "risk"
        elif score >= 40:
            level, tone = "Elevated", "warning"
        else:
            level, tone = "Low", "stable"
        rel = synthesis.get("relationship_analysis") if isinstance(synthesis, dict) else {}
        cd = synthesis.get("convergence_divergence") if isinstance(synthesis, dict) else {}
        status = str((rel or {}).get("status") or (cd or {}).get("status") or "").strip().upper()
        return {
            "score": score,
            "level": level,
            "tone": tone,
            "penalty_reason": str(ra.get("penalty_reason") or "").strip(),
            "status_label": status or "UNSET",
        }
    return _compute_divergence_gauge(synthesis, quant_momentum)


def _compute_divergence_gauge(
    synthesis: dict,
    quant_momentum: dict | None = None,
) -> dict:
    """
    Heuristic 0–100% divergence score when quant Risk_Analytics is unavailable.
    """
    rel = synthesis.get("relationship_analysis") if isinstance(synthesis, dict) else {}
    cd = synthesis.get("convergence_divergence") if isinstance(synthesis, dict) else {}
    status = str((rel or {}).get("status") or (cd or {}).get("status") or "").strip().upper()
    score = 50
    level = "Moderate"
    tone = "neutral"

    if "DIVERG" in status:
        score = 85
        level = "High"
        tone = "risk"
    elif "CONV" in status:
        score = 24
        level = "Low"
        tone = "stable"
    else:
        score = 48
        level = "Moderate"
        tone = "warning"

    if isinstance(quant_momentum, dict) and quant_momentum:
        stress_keys = ("MOVE", "VIX", "DXY", "JPY", "TWD")
        risk_on_keys = ("SPY", "HYG")
        stress_bad = sum(
            1 for k in stress_keys if (quant_momentum.get(k) or {}).get("label") == "Deteriorating"
        )
        risk_good = sum(
            1 for k in risk_on_keys if (quant_momentum.get(k) or {}).get("label") == "Improving"
        )
        if stress_bad >= 2 and risk_good >= 1:
            score = max(score, 85)
            level = "High"
            tone = "risk"
        elif stress_bad >= 1 and risk_good >= 1 and tone != "risk":
            score = max(score, 55)
            level = "Elevated"
            tone = "warning"
        elif stress_bad == 0 and risk_good >= 2 and "DIVERG" not in status:
            score = min(score, 28)
            level = "Low"
            tone = "stable"

    score = max(0, min(100, int(score)))
    if tone == "neutral" or (tone == "warning" and score < 40):
        tone = "stable"
    elif tone != "risk" and 40 <= score < 70:
        tone = "warning"
        if level == "Moderate":
            level = "Elevated"
    elif score >= 70:
        tone = "risk"
        level = "High"
    return {
        "score": score,
        "level": level,
        "tone": tone,
        "penalty_reason": "",
        "status_label": status or "UNSET",
    }


def _format_historical_analogues_for_llm(l2a: dict) -> str:
    """
    Format both sets of historical matches into a compact LLM-readable block.
    Omits raw numbers — LLM should reference period names and qualitative direction only.
    """
    def _direction(val: float | None) -> str:
        if val is None:
            return "no data"
        if val >= 15:
            return "strongly positive"
        if val >= 5:
            return "moderately positive"
        if val >= -5:
            return "roughly flat"
        if val >= -20:
            return "moderately negative"
        return "sharply negative"

    lines: list[str] = []

    regime_matches = (l2a.get("matches") or [])[:3]
    if regime_matches:
        lines.append("US REGIME ANALOGUES (most similar US macro environments):")
        for m in regime_matches:
            r12 = m.get("spy_return_12m")
            lines.append(
                f"  • {m['date']} — {m['description']}"
                f" → US stocks 12m after: {_direction(r12)}"
            )

    div_matches = (l2a.get("divergence_matches") or [])[:3]
    if div_matches:
        lines.append("")
        lines.append("GLOBAL DIVERGENCE ANALOGUES (most similar cross-country policy/FX/inflation divergence):")
        for m in div_matches:
            eem = m.get("eem_12m")
            gld = m.get("gld_12m")
            cop = m.get("copper_12m")
            lines.append(
                f"  • {m['date']} — {m['description']}"
                f" → EM stocks: {_direction(eem)}, Gold: {_direction(gld)}, Copper: {_direction(cop)}"
            )

    if not lines:
        return ""

    lines.append("")
    lines.append(
        "USAGE: Reference these analogues to add historical depth to daily_themes and cio_directive."
        " Write 'This pattern echoes [year]' or 'Periods like [year] saw...' — never cite exact % figures."
        " Do NOT force a match if the current situation differs materially from the analogue."
    )
    return "\n".join(lines)


def _build_synthesis_prompt(
    l1: dict,
    l2a: dict,
    l2b: dict,
    quant_context_json: dict,
    event_of_day: dict,
    surfaced_rows: list[dict],
    report_date: str,
    country_signals: dict | None = None,
) -> str:
    flow_block = {
        "signals": (l2b.get("flow_payload") or {}).get("signals", {}),
        "coverage": l2b.get("data_coverage", {}),
        "llm": l2b.get("llm", {}),
    }
    brief_block = {
        "signals": l2a.get("signals", {}),
        "executive_summary": l2a.get("executive_summary", ""),
        "narrative": l2a.get("narrative", ""),
        "matches": (l2a.get("matches") or [])[:3],
    }
    market_snapshot = _market_snapshot_for_prompt(report_date)
    momentum = quant_context_json.get("momentum_profile") if isinstance(quant_context_json, dict) else {}
    div_ctx = _divergence_prompt_context(momentum, l1, l2a, l2b, quant_context_json)
    regime_label = quant_context_json.get("macro_regime_label") or "Unknown"
    gear_matrix_raw = build_l2_gear_raw_context(l2b, quant_context_json)
    try:
        us_financial_conditions = _us_financial_conditions_summary()
    except Exception:
        us_financial_conditions = []
    try:
        carry_relative = _build_carry_relative(l2b)
    except Exception:
        carry_relative = {}
    return f"""
You are synthesizing today's The Macro Pulse War Room executive brief ({report_date}).

This product is a **US-anchored Global Financial Cycle** map (Hélène Rey framework):
the US (Fed + financial conditions + VIX) is the global pricing ANCHOR and switch;
other markets are the RELATIVE transmission surface. Inputs below are split into
two zones. **NEVER apply a [US ANCHOR] number to global or per-country claims, and
never describe a [GLOBAL RELATIVE] number as if it were the US.**

=== ABSOLUTE ANCHOR: macro_regime_label = {regime_label} (US regime; do not contradict) ===

╔══════════════ [US ANCHOR] — United States ONLY ══════════════╗

=== [US ANCHOR] Quant Engine (US growth/inflation momentum + regime match) ===
{json.dumps(quant_context_json, ensure_ascii=False, indent=2)}

=== [US ANCHOR] US Financial Conditions (the global switch; US-only — do NOT generalise to other markets) ===
{json.dumps(us_financial_conditions, ensure_ascii=False, indent=2)}

=== [US ANCHOR] Market snapshot (US equities/rates; Zone 1 flash; numbers allowed here only) ===
{json.dumps(market_snapshot, ensure_ascii=False, indent=2)}

╔══════════════ [GLOBAL RELATIVE] — each market vs the US ══════════════╗

=== [GLOBAL RELATIVE] Carry / Rate-Differential Surface (each market RELATIVE to US; US is origin; China = capital-controlled, read as policy/FX not carry) ===
{json.dumps(carry_relative, ensure_ascii=False, indent=2)}

=== [GLOBAL RELATIVE] Layer 1 structural snapshot (MUST reference in Zone 3) ===
{json.dumps(l1.get("llm", {}), ensure_ascii=False, indent=2)}

=== [GLOBAL RELATIVE] Layer 1 modules summary ===
{json.dumps({k: l1.get(k) for k in ("module_a", "module_b", "module_c", "module_d", "module_e") if l1.get(k)}, ensure_ascii=False, indent=2)}

=== [GLOBAL RELATIVE] Layer 2 flow (MUST reference in Zone 3) ===
{json.dumps(flow_block, ensure_ascii=False, indent=2)}

=== [GLOBAL RELATIVE] Gear Matrix raw inputs (translate in gear_matrix_semantics; do not echo verbatim) ===
{json.dumps(gear_matrix_raw, ensure_ascii=False, indent=2)}

=== [GLOBAL RELATIVE] Per-Country Quantitative Signals (four-grid: growth / inflation / policy / capital) ===
{json.dumps(country_signals or {}, ensure_ascii=False, indent=2)}

╔══════════════ SHARED CONTEXT ══════════════╗

=== Divergence / Risk_Analytics (explain in relationship_analysis.divergence_explanation; cite penalty_reason) ===
{json.dumps(div_ctx, ensure_ascii=False, indent=2)}

=== Event of the Day (PRIMARY tactical pulse) ===
{json.dumps(event_of_day, ensure_ascii=False, indent=2)}

=== What Rose to Surface Today (top 5 auto-ranked) ===
{json.dumps(surfaced_rows[:5], ensure_ascii=False, indent=2)}

=== Layer 2 macro brief (MUST reference in Zone 3) ===
{json.dumps(brief_block, ensure_ascii=False, indent=2)}

=== Historical Analogues (use for qualitative depth in themes and CIO directive; never cite raw %) ===
{_format_historical_analogues_for_llm(l2a)}

Instructions:
- Complete pre_analysis before all executive zones.
- ZONE/COUNTRY DISCIPLINE: [US ANCHOR] data describes the US only; use it for the US regime/anchor narrative. Each non-US market must be described from [GLOBAL RELATIVE] (its rate differential, FX, flow vs the US). Do NOT claim US financial conditions = global conditions; do NOT attribute a country's carry/flow to the US.
- Zone 1: flash_bullets (capital_flows from the carry/relative surface, volatility from US VIX, macro_sentiment) aligned with macro_regime_label.
- Zone 2: decisive, cold, contrarian; NO digits, %, Z-scores, tickers, bn.
- Zone 3: pure_alpha three sections (expectation_arbitrage, crowdedness_leverage, hedging_cost_anomaly); cite macro_regime_label, divergence_score, penalty_reason, SKEW, VVIX, SHY/TLT, MOVE/VIX/DXY/JPY/TWD/SPY/HYG.
- gear_matrix_semantics: all six countries; expectation/plumbing/hedging sentences plus threat/opportunity labels (see system Step 3).
- relationship_analysis.status must match convergence_divergence.status (likely {div_ctx.get("likely_status")}).
- Failure mode: generic prose with no named indicators from the payloads above.

Return exactly the Macro Pulse 3.0 JSON schema in your system instructions.
"""


def _driver_candidates(country: str, flow_payload: dict, sig: dict, eq: dict) -> list[dict]:
    """Build richer country attribution drivers from flow payload fields."""
    drivers: list[dict] = []
    reasons = (sig.get(country, {}) or {}).get("reasons", []) or []
    points = (sig.get(country, {}) or {}).get("points", []) or []
    point_map = {reasons[i]: points[i] for i in range(min(len(reasons), len(points)))}

    def add(name: str, value: float | None, positive_is_supportive: bool = True) -> None:
        if value is None:
            return
        v = float(value)
        supportive = (v >= 0) if positive_is_supportive else (v <= 0)
        if name in point_map:
            supportive = bool(point_map[name] == 1)
        drivers.append(
            {
                "name": name,
                "direction": "supportive" if supportive else "drag",
                "weight": abs(v),
            }
        )

    fx1 = flow_payload.get("fx_1m_local_vs_usd", {})
    fx3 = flow_payload.get("fx_3m_local_vs_usd", {})
    eq1 = eq
    spread = flow_payload.get("bond_spreads_bp", {})

    if country == "US":
        add("USD broad 1m", fx1.get("China"))
        add("USD broad 3m", fx3.get("China"))
        add("Equity 1m", eq1.get("US"))
    else:
        add("FX 1m", fx1.get(country))
        add("FX 3m", fx3.get(country))
        add("Equity 1m", eq1.get(country))
        add("US-local 10Y spread", spread.get(country), positive_is_supportive=False)
        if country == "Japan":
            add("Japan TIC MoM", flow_payload.get("japan_tic_mom_change_bn"))
        if country == "China":
            add("FX reserves MoM", flow_payload.get("china_fx_reserves_mom_change_bn"))
        if country == "Taiwan":
            tw = flow_payload.get("taiwan_foreign_flow", {})
            add("TW foreign 5d", tw.get("cum_5d_bn_twd"))

    # Keep signal reasons visible even if a numeric series is missing today.
    existing = {d["name"] for d in drivers}
    for reason in reasons:
        if reason not in existing:
            drivers.append(
                {
                    "name": reason,
                    "direction": "supportive" if point_map.get(reason, 0) == 1 else "drag",
                    "weight": 1.0,
                }
            )
    return drivers


def build_l3_appendix_context(
    report_date: date,
    l1: dict,
    l2a: dict,
    l2b: dict,
    synthesis: dict,
    quant_context_json: dict,
    synth_doc: dict | None = None,
) -> dict:
    """Context for templates/appendix.html (Layer 3 raw tables & deep dives)."""
    d = report_date.isoformat()
    synth_doc = synth_doc if isinstance(synth_doc, dict) else {}
    event_of_day = synth_doc.get("event_of_day") if isinstance(synth_doc.get("event_of_day"), dict) else _resolve_event_of_day(l2b)
    surfaced_rows = (
        synth_doc.get("surfaced_rows")
        if isinstance(synth_doc.get("surfaced_rows"), list)
        else (l2b.get("surfaced_rows") if isinstance(l2b.get("surfaced_rows"), list) else [])
    )
    quant_momentum = quant_context_json.get("momentum_profile") or {}
    country_cycle = synth_doc.get("country_cycle") if isinstance(synth_doc.get("country_cycle"), dict) else synthesis.get("country_cycle") or {}

    index_map = {"US": "SPY", "Japan": "EWJ", "Europe": "EZU", "UK": "EWU", "China": "MCHI", "Taiwan": "EWT"}
    flow_payload = l2b.get("flow_payload", {})
    sig = l2b.get("signals", {})
    eq = flow_payload.get("equity_1m_returns", {}) if isinstance(flow_payload, dict) else {}
    notes = synthesis.get("country_attribution_notes", {}) if isinstance(synthesis, dict) else {}
    country_attrib: list[dict] = []
    for c in COUNTRY_CYCLE_KEYS:
        raw_drivers = _driver_candidates(c, flow_payload, sig, eq)
        total_weight = sum(item.get("weight", 0.0) for item in raw_drivers) or 1.0
        drivers = [
            {
                "name": item["name"],
                "direction": item["direction"],
                "share": f"{(item.get('weight', 0.0) / total_weight) * 100.0:.1f}%",
            }
            for item in raw_drivers
        ]
        drivers = sorted(drivers, key=lambda x: float(x["share"].rstrip("%")), reverse=True)
        country_attrib.append(
            {
                "country": c,
                "index_proxy": index_map[c],
                "index_1m": "—" if eq.get(c) is None else f"{float(eq.get(c)):+.2f}%",
                "drivers": drivers[:6],
                "llm_note": notes.get(c, FALLBACK),
            }
        )

    us_panels_raw = ((l2a.get("factor_attrib") or {}).get("panels") or [])
    wanted = {"tradeable", "sectors", "fama_french", "spx_contrib_1d", "spx_contrib_21d"}
    us_panels = [p for p in us_panels_raw if p.get("id") in wanted and p.get("available")]

    country_deep_dive: list[dict] = []
    for c in country_attrib:
        if c["country"] == "US" and us_panels:
            panels = []
            for p in us_panels:
                rows = [
                    {
                        "factor": f.get("name", "—"),
                        "move": f"{f.get('factor_return_pct', '—')}%",
                        "contribution": f"{f.get('contribution_pct', '—')}%",
                        "share": f"{f.get('share_pct', '—')}%",
                    }
                    for f in (p.get("factors") or [])
                ]
                panels.append(
                    {
                        "id": p.get("id", "panel"),
                        "title": p.get("title", p.get("id", "Panel")),
                        "model_label": p.get("model_label", "US attribution"),
                        "window_label": p.get("window_label", "Latest window"),
                        "rows": rows,
                    }
                )
        else:
            panels = [
                {
                    "id": "country_flow",
                    "title": "Flow & Macro Drivers",
                    "model_label": "Country composite driver mix",
                    "window_label": "Latest 1m signal window",
                    "rows": [
                        {
                            "factor": item["name"],
                            "move": item["direction"],
                            "contribution": "—",
                            "share": item["share"],
                        }
                        for item in c.get("drivers", [])
                    ],
                }
            ]
        country_deep_dive.append(
            {
                "country": c["country"],
                "index_proxy": c["index_proxy"],
                "index_1m": c["index_1m"],
                "llm_note": c["llm_note"],
                "panels": panels,
            }
        )

    z1 = _normalize_zone1_pulse(dict(synthesis.get("zone1_pulse") or {}))
    relationship = _normalize_relationship_analysis(dict(synthesis.get("relationship_analysis") or {}))
    divergence_gauge = synth_doc.get("divergence_gauge") if isinstance(synth_doc.get("divergence_gauge"), dict) else _divergence_gauge_from_quant(
        quant_context_json, synthesis, quant_momentum
    )
    cb = d.replace("-", "")
    layer_embeds: dict[str, str] = {}
    if (OUTPUT_DIR / "brief.html").exists():
        layer_embeds["country_risk"] = f"brief.html?cb={cb}"
    if (OUTPUT_DIR / "flow_brief.html").exists():
        layer_embeds["flow_monitor"] = f"flow_brief.html?cb={cb}"

    pre = synthesis.get("pre_analysis") if isinstance(synthesis.get("pre_analysis"), dict) else {}
    return {
        "l3_available": True,
        "zone1": z1,
        "relationship": relationship,
        "pure_alpha": relationship.get("pure_alpha", {}),
        "synthesis": {
            **synthesis,
            "pre_analysis": {
                "cross_layer_conflict": pre.get("cross_layer_conflict", FALLBACK),
                "momentum_verification": pre.get("momentum_verification", FALLBACK),
                "event_pulse_analysis": pre.get("event_pulse_analysis", FALLBACK),
            },
            "watch_list": synthesis.get("watch_list") if isinstance(synthesis.get("watch_list"), list) else [FALLBACK],
        },
        "event_of_day": event_of_day,
        "surfaced_rows": surfaced_rows,
        "validation_panel": l2b.get("validation_panel") if isinstance(l2b.get("validation_panel"), list) else [],
        "quant_momentum": quant_momentum,
        "country_cycle": country_cycle,
        "country_cycle_countries": list(COUNTRY_CYCLE_KEYS),
        "divergence_gauge": divergence_gauge,
        "l2_narrative": {
            "executive_summary": l2a.get("executive_summary", FALLBACK),
            "narrative": l2a.get("narrative", FALLBACK),
        },
        "historical_matches": (l2a.get("matches") or [])[:3],
        "divergence_matches": (l2a.get("divergence_matches") or [])[:3],
        "flow_narrative": {
            "flow_narrative": (l2b.get("llm") or {}).get("flow_narrative", FALLBACK),
            "japan_carry_interpretation": (l2b.get("llm") or {}).get("japan_carry_interpretation", FALLBACK),
            "china_capital_interpretation": (l2b.get("llm") or {}).get("china_capital_interpretation", FALLBACK),
        },
        "country_deep_dive": country_deep_dive,
        "layer_embeds": layer_embeds,
    }


def _gear_semantics_usable(gear_semantics: dict | None) -> bool:
    if not isinstance(gear_semantics, dict):
        return False
    for country in COUNTRY_CYCLE_KEYS:
        block = gear_semantics.get(country)
        if not isinstance(block, dict):
            continue
        for key in ("expectation", "plumbing", "hedging"):
            text = str(block.get(key) or "").strip()
            if text and text != FALLBACK and not text.lower().startswith("llm analysis not"):
                return True
    return False


def _refresh_brief_global_gear_narratives(
    report_date: date,
    l2a: dict,
    l1: dict,
    l2b: dict,
    gear_semantics: dict | None,
) -> None:
    """Re-render brief_global.html with CIO gear narratives after Layer 3 LLM."""
    if not _gear_semantics_usable(gear_semantics):
        return
    brief_path = OUTPUT_DIR / "brief.html"
    if not brief_path.exists():
        return
    try:
        from run import _write_international_brief_pages

        country_rows = build_l2_country_rows_from_sources(report_date, l2a, l2b, l1)
        signals = l2a.get("signals") if isinstance(l2a.get("signals"), dict) else {}
        _write_international_brief_pages(
            report_date,
            brief_path.name,
            country_rows,
            signals.get("as_of"),
            signals.get("headline"),
            l2a.get("regime_summary"),
            gear_semantics=gear_semantics,
        )
        print("  Refreshed brief_global.html with gear semantic translations.", file=sys.stderr)
    except Exception as exc:
        print(f"  Warning: could not refresh brief_global.html ({exc})", file=sys.stderr)


def _write_synthesis_lite_html(
    *,
    report_date: date,
    d: str,
    synthesis: dict,
    l1: dict,
    l2a: dict,
    l2b: dict,
    quant_context_json: dict,
    quant_momentum: dict,
    vs_us_alignment: dict,
    llm_meta: dict | None = None,
    llm_placeholder: bool = False,
) -> Path:
    """Render synthesis_lite.html from an existing synthesis dict."""
    directive = synthesis.get("cio_directive") if isinstance(synthesis.get("cio_directive"), dict) else {}
    directive = _normalize_cio_directive_bilingual(directive)
    synthesis["cio_directive"] = directive
    macro_regime_label = str(quant_context_json.get("macro_regime_label") or "Unknown")
    regime_gauge_tone = _regime_gauge_tone(macro_regime_label)
    divergence_gauge = _divergence_gauge_from_quant(quant_context_json, synthesis, quant_momentum)
    country_rows = build_l2_country_rows_from_sources(report_date, l2a, l2b, l1)
    l3_vm = build_l3_view_model(
        synthesis, l1, l2a, l2b, quant_context_json,
        country_rows=country_rows,
        report_date=report_date,
    )
    gear_semantics = synthesis.get("gear_matrix_semantics") if isinstance(synthesis, dict) else {}
    l3_vm = apply_gear_semantics_to_l3_view_model(l3_vm, gear_semantics)
    daily_themes_bi = _normalize_daily_themes_bilingual(
        synthesis.get("daily_themes") if isinstance(synthesis, dict) else []
    )
    synthesis["daily_themes"] = daily_themes_bi
    watch_bi = synthesis.get("watch_list_bilingual")
    if not isinstance(watch_bi, list):
        _, watch_bi = _normalize_watch_list_bilingual(synthesis.get("watch_list") or [])
    hist_en = (l2a.get("matches") or [])[:3]
    div_en = (l2a.get("divergence_matches") or [])[:3]
    hist_desc = [str(m.get("description") or "") for m in hist_en if isinstance(m, dict)]
    div_desc = [str(m.get("description") or "") for m in div_en if isinstance(m, dict)]
    hist_zh, hist_fb = _translate_descriptions_to_zh(hist_desc)
    div_zh, div_fb = _translate_descriptions_to_zh(div_desc)
    # Data-driven Investment-Clock asset tilt (hard-data momentum + severity axis).
    try:
        from src.regime_tilt import compute_asset_tilts
        tilt_result = compute_asset_tilts(load_all_from_cache())
    except Exception as exc:
        print(f"  Warning: asset tilt engine failed ({exc})", file=sys.stderr)
        tilt_result = None
    try:
        carry_relative = _build_carry_relative(l2b)
    except Exception as exc:
        print(f"  Warning: carry/relative engine failed ({exc})", file=sys.stderr)
        carry_relative = None
    lite_lang_blocks = _build_lite_lang_blocks(
        report_date=d,
        macro_regime_label=macro_regime_label,
        regime_gauge_tone=regime_gauge_tone,
        divergence_gauge=divergence_gauge,
        daily_themes=daily_themes_bi,
        directive=directive,
        watch_bilingual=watch_bi,
        regime_asset_tilt_label=macro_regime_label,
        country_matrix=l3_vm["country_matrix"],
        gear_semantics=gear_semantics if isinstance(gear_semantics, dict) else {},
        vs_us_alignment=vs_us_alignment,
        historical_matches_en=hist_en,
        divergence_matches_en=div_en,
        historical_descriptions_zh=hist_zh,
        divergence_descriptions_zh=div_zh,
        history_translation_fallback=hist_fb or div_fb,
        tilt_result=tilt_result,
        carry_relative=carry_relative,
        risk_analytics=(quant_context_json.get("Risk_Analytics") if isinstance(quant_context_json, dict) else None),
    )
    try:
        from src.sector_rotation import build_rrg_plotly as _build_rrg
        from src.collect import load_all_from_cache as _load_cache
        _rrg = _build_rrg(_load_cache())
        rrg_data = {k: v for k, v in _rrg.items() if k not in ("summary", "sector_stats", "sector_performance", "rank_trans")}
        rrg_summary = _rrg.get("summary", {})
        rrg_sector_stats = _rrg.get("sector_stats", [])
        sector_performance = _rrg.get("sector_performance", [])
        rrg_rank_trans = _rrg.get("rank_trans", {})
    except Exception as exc:
        print(f"  Warning: RRG chart failed ({exc})", file=sys.stderr)
        rrg_data = {}
        rrg_summary = {}
        rrg_sector_stats = []
        sector_performance = []
        rrg_rank_trans = {}
    try:
        regime_quadrant_data = _build_regime_quadrant_plotly()
    except Exception as exc:
        print(f"  Warning: quadrant chart failed ({exc})", file=sys.stderr)
        regime_quadrant_data = {}
    try:
        zscore_data_zh = _build_zscore_plotly(lang="zh-Hant")
        zscore_data_en = _build_zscore_plotly(lang="en")
    except Exception as exc:
        print(f"  Warning: z-score chart failed ({exc})", file=sys.stderr)
        zscore_data_zh = zscore_data_en = {}
    try:
        from src.config import PROCESSED_DIR as _PDIR
        _hmm_path = _PDIR / "regime_hmm_summary.json"
        _hmm_summary = json.loads(_hmm_path.read_text()) if _hmm_path.exists() else {}
        regime_prob_ts_data = _hmm_summary.get("prob_timeseries", {})
    except Exception as exc:
        print(f"  Warning: regime prob timeseries failed ({exc})", file=sys.stderr)
        regime_prob_ts_data = {}
    out_html_lite = OUTPUT_DIR / "synthesis_lite.html"
    meta = llm_meta if isinstance(llm_meta, dict) else {}
    out_html_lite.write_text(
        Template(HTML_TMPL_LITE).render(
            report_date=d,
            lite_lang_blocks=lite_lang_blocks,
            llm_meta=meta,
            llm_placeholder=bool(llm_placeholder),
            regime_quadrant_data=json.dumps(regime_quadrant_data),
            zscore_data_zh=json.dumps(zscore_data_zh),
            zscore_data_en=json.dumps(zscore_data_en),
            regime_prob_ts_data=json.dumps(regime_prob_ts_data),
            rrg_data=json.dumps(rrg_data),
            rrg_summary=rrg_summary,
            rrg_sector_stats=rrg_sector_stats,
            sector_performance=sector_performance,
            rrg_rank_trans=rrg_rank_trans,
        ),
        encoding="utf-8",
    )
    print(f"Done. Lite output: {out_html_lite}")
    return out_html_lite


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 3 synthesis from Layer 1/2 snapshots")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--rerender-lite",
        action="store_true",
        help="Rebuild synthesis_lite.html from existing synthesis_data JSON (no LLM)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    report_date = date.fromisoformat(args.date) if args.date else date.today()
    d = report_date.isoformat()

    l1_path = OUTPUT_DIR / f"global_regime_data_{d}.json"
    l2a_path = OUTPUT_DIR / f"brief_data_{d}.json"
    l2b_path = OUTPUT_DIR / f"flow_data_{d}.json"
    for p in [l1_path, l2a_path, l2b_path]:
        if not p.exists():
            raise RuntimeError(f"Missing input snapshot: {p}")

    l1 = json.loads(l1_path.read_text(encoding="utf-8"))
    l2a = json.loads(l2a_path.read_text(encoding="utf-8"))
    l2b = json.loads(l2b_path.read_text(encoding="utf-8"))

    out_data = OUTPUT_DIR / f"synthesis_data_{d}.json"
    quant_context_json = _load_quant_context_json(d, l2b)
    if not quant_context_json:
        print(
            f"Warning: no quant context for {d} (flow payload and quant_engine_*.json). "
            "Run flow_run.py first. Synthesis will proceed without regime tags.",
            file=sys.stderr,
        )

    event_of_day = _resolve_event_of_day(l2b)
    surfaced_rows = l2b.get("surfaced_rows") if isinstance(l2b.get("surfaced_rows"), list) else []
    quant_momentum = quant_context_json.get("momentum_profile") or {}

    flow_payload = l2b.get("flow_payload") if isinstance(l2b, dict) else None
    try:
        raw_data = load_all_from_cache()
        country_signals = compute_country_signals(raw_data, {"flow_payload": flow_payload} if flow_payload else None)
        vs_us_alignment = compute_vs_us_alignment(country_signals)
    except Exception as exc:
        print(f"Warning: country_signals computation failed: {exc}", file=sys.stderr)
        country_signals = {}
        vs_us_alignment = {}

    if args.rerender_lite:
        cache_path = out_data
        if not cache_path.exists():
            raise RuntimeError(f"No cached synthesis data: {cache_path}. Run synthesis.py without --rerender-lite first.")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        synthesis = _normalize_synthesis({**_synthesis_fallback(), **(cached.get("synthesis") or {})})
        llm_meta = cached.get("llm_meta") if isinstance(cached.get("llm_meta"), dict) else {}
        llm_placeholder = bool(cached.get("llm_placeholder"))
        country_cycle = cached.get("country_cycle") if isinstance(cached.get("country_cycle"), dict) else {}
        us_growth_z = cached.get("us_growth_z_score")
        if cached.get("quant_context"):
            quant_context_json = cached["quant_context"]
        if llm_placeholder:
            print(
                f"  WARNING: {cache_path.name} is placeholder-only — lite HTML will show offline banner. "
                "Run full synthesis without --skip-llm first.",
                file=sys.stderr,
            )
        print(f"  Rerender lite from {cache_path.name} (LLM skipped).", file=sys.stderr)
    else:
        _guard_skip_llm_overwrite(out_data, args.skip_llm)
        prompt = _build_synthesis_prompt(
            l1, l2a, l2b, quant_context_json, event_of_day, surfaced_rows, d,
            country_signals=country_signals,
        )
        synthesis, llm_meta = _get_synthesis(prompt, args.skip_llm)
        synthesis, country_cycle = _apply_quant_country_cycle(synthesis, l1, quant_context_json)
        llm_placeholder = is_synthesis_placeholder(synthesis)
        us_growth_z = _us_growth_z_score(quant_context_json)

    # Persist quant-derived US cycle into Layer 1 snapshot + HTML (legacy country_cycle keys)
    if isinstance(l1.get("llm"), dict):
        l1["llm"]["country_cycle"] = country_cycle
    else:
        l1["llm"] = {"country_cycle": country_cycle}
    l1_path.write_text(json.dumps(l1, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        from global_regime import render_html as render_global_regime_html

        cache_age = 0
        fetched_at = l1.get("fetched_at")
        if fetched_at:
            try:
                from datetime import datetime, timezone

                ts = datetime.fromisoformat(str(fetched_at))
                cache_age = max(
                    0,
                    int((datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)).total_seconds() // 86400),
                )
            except Exception:
                cache_age = 0
        render_global_regime_html(
            OUTPUT_DIR / "global_regime.html",
            l1,
            report_date,
            cache_age,
        )
    except Exception as exc:
        print(f"  Warning: could not refresh global_regime.html ({exc})", file=sys.stderr)

    if args.rerender_lite:
        _write_synthesis_lite_html(
            report_date=report_date,
            d=d,
            synthesis=synthesis,
            l1=l1,
            l2a=l2a,
            l2b=l2b,
            quant_context_json=quant_context_json,
            quant_momentum=quant_momentum,
            vs_us_alignment=vs_us_alignment,
        )
        return

    index_map = {"US": "SPY", "Japan": "EWJ", "Europe": "EZU", "UK": "EWU", "China": "MCHI", "Taiwan": "EWT"}
    flow_payload = l2b.get("flow_payload", {})
    sig = l2b.get("signals", {})
    eq = flow_payload.get("equity_1m_returns", {})
    notes = synthesis.get("country_attribution_notes", {}) if isinstance(synthesis, dict) else {}
    country_attrib = []
    for c in ["US", "Japan", "Europe", "UK", "China", "Taiwan"]:
        raw_drivers = _driver_candidates(c, flow_payload, sig, eq)
        total_weight = sum(d.get("weight", 0.0) for d in raw_drivers) or 1.0
        drivers = [
            {
                "name": d["name"],
                "direction": d["direction"],
                "share": f"{(d.get('weight', 0.0) / total_weight) * 100.0:.1f}%",
            }
            for d in raw_drivers
        ]
        drivers = sorted(drivers, key=lambda x: float(x["share"].rstrip("%")), reverse=True)
        country_attrib.append(
            {
                "country": c,
                "index_proxy": index_map[c],
                "index_1m": "—" if eq.get(c) is None else f"{float(eq.get(c)):+.2f}%",
                "drivers": drivers[:6],
                "llm_note": notes.get(c, FALLBACK),
            }
        )

    us_panels_raw = ((l2a.get("factor_attrib") or {}).get("panels") or [])
    wanted = {"tradeable", "sectors", "fama_french", "spx_contrib_1d", "spx_contrib_21d"}
    us_panels = [p for p in us_panels_raw if p.get("id") in wanted and p.get("available")]

    country_deep_dive = []
    for c in country_attrib:
        if c["country"] == "US" and us_panels:
            panels = []
            for p in us_panels:
                rows = [
                    {
                        "factor": f.get("name", "—"),
                        "move": f"{f.get('factor_return_pct', '—')}%",
                        "contribution": f"{f.get('contribution_pct', '—')}%",
                        "share": f"{f.get('share_pct', '—')}%",
                    }
                    for f in (p.get("factors") or [])
                ]
                panels.append(
                    {
                        "id": p.get("id", "panel"),
                        "title": p.get("title", p.get("id", "Panel")),
                        "model_label": p.get("model_label", "US attribution"),
                        "window_label": p.get("window_label", "Latest window"),
                        "rows": rows,
                    }
                )
        else:
            panels = [
                {
                    "id": "country_flow",
                    "title": "Flow & Macro Drivers",
                    "model_label": "Country composite driver mix",
                    "window_label": "Latest 1m signal window",
                    "rows": [
                        {
                            "factor": d["name"],
                            "move": d["direction"],
                            "contribution": "—",
                            "share": d["share"],
                        }
                        for d in c.get("drivers", [])
                    ],
                }
            ]
        country_deep_dive.append(
            {
                "country": c["country"],
                "index_proxy": c["index_proxy"],
                "index_1m": c["index_1m"],
                "llm_note": c["llm_note"],
                "panels": panels,
            }
        )

    checks = [
        ("layer1 llm", bool(l1.get("llm"))),
        ("layer2 brief narrative", bool(l2a.get("narrative"))),
        ("layer2 flow payload", bool(l2b.get("flow_payload"))),
        ("layer2 flow narrative", bool((l2b.get("llm") or {}).get("flow_narrative"))),
        ("quant engine payload", bool(quant_context_json)),
    ]
    total = len(checks)
    available = sum(1 for _, ok in checks if ok)
    missing = [name for name, ok in checks if not ok][:3]
    coverage = {
        "available": available,
        "total": total,
        "ratio_pct": round((available / total) * 100.0, 1) if total else 0.0,
        "missing": ", ".join(missing) if missing else "none",
    }

    macro_regime_label = str(quant_context_json.get("macro_regime_label") or "Unknown")
    regime_gauge_tone = _regime_gauge_tone(macro_regime_label)
    divergence_gauge = _divergence_gauge_from_quant(quant_context_json, synthesis, quant_momentum)
    penalty_reason = str(divergence_gauge.get("penalty_reason") or "").strip()
    validation_panel = (
        l2b.get("validation_panel") if isinstance(l2b.get("validation_panel"), list) else []
    )
    cb = d.replace("-", "")
    layer_embeds: dict[str, str] = {}
    brief_html = OUTPUT_DIR / "brief.html"
    flow_html = OUTPUT_DIR / "flow_brief.html"
    if brief_html.exists():
        layer_embeds["country_risk"] = f"brief.html?cb={cb}"
    if flow_html.exists():
        layer_embeds["flow_monitor"] = f"flow_brief.html?cb={cb}"

    out_html = Path(args.output) if args.output else OUTPUT_DIR / "synthesis.html"
    z1_raw = synthesis.get("zone1_pulse") if isinstance(synthesis.get("zone1_pulse"), dict) else {}
    z1 = _normalize_zone1_pulse(dict(z1_raw))
    directive = synthesis.get("cio_directive") if isinstance(synthesis.get("cio_directive"), dict) else {}
    directive = _normalize_cio_directive_bilingual(directive)
    synthesis["cio_directive"] = directive
    relationship = _normalize_relationship_analysis(
        dict(synthesis.get("relationship_analysis") if isinstance(synthesis.get("relationship_analysis"), dict) else {})
    )
    directive_view = {
        "the_stance": directive.get("the_stance", FALLBACK),
        "the_narrative": _coerce_narrative(directive.get("the_narrative")),
        "the_watchout": directive.get("the_watchout", FALLBACK),
        "positioning_6_12m": _sanitize_jargon(directive.get("positioning_6_12m", "")),
    }
    country_rows = build_l2_country_rows_from_sources(report_date, l2a, l2b, l1)
    l3_vm = build_l3_view_model(
        synthesis, l1, l2a, l2b, quant_context_json,
        country_rows=country_rows,
        report_date=report_date,
    )
    gear_semantics = synthesis.get("gear_matrix_semantics") if isinstance(synthesis, dict) else {}
    l3_vm = apply_gear_semantics_to_l3_view_model(l3_vm, gear_semantics)
    _refresh_brief_global_gear_narratives(report_date, l2a, l1, l2b, gear_semantics)
    out_html.write_text(
        Template(HTML_TMPL).render(
            report_date=d,
            synthesis=synthesis,
            zone1=z1,
            directive=directive_view,
            relationship=relationship,
            pure_alpha=relationship.get("pure_alpha", {}),
            daily_themes=synthesis.get("daily_themes") if isinstance(synthesis, dict) else [],
            daily_overview=l3_vm["daily_overview"],
            critical_watchout=l3_vm["critical_watchout"],
            spread_trade_line=l3_vm["spread_trade_line"],
            matrix_tabs=l3_vm["matrix_tabs"],
            country_matrix=l3_vm["country_matrix"],
            vs_us_alignment=vs_us_alignment,
            macro_regime_label=macro_regime_label,
            regime_gauge_tone=regime_gauge_tone,
            regime_asset_tilt=_regime_asset_tilt(macro_regime_label),
            penalty_reason=penalty_reason,
            fallback_text=FALLBACK,
            divergence_gauge=divergence_gauge,
            llm_meta=llm_meta,
            macra_style_block=macra_style_block(),
            llm_placeholder=llm_placeholder,
            event_of_day=event_of_day,
            surfaced_rows=surfaced_rows,
            validation_panel=validation_panel,
            layer_embeds=layer_embeds,
            quant_momentum=quant_momentum,
            country_cycle=country_cycle,
            country_cycle_countries=list(COUNTRY_CYCLE_KEYS),
            us_growth_z=us_growth_z,
            coverage=coverage,
            l2_narrative={
                "executive_summary": l2a.get("executive_summary", FALLBACK),
                "narrative": l2a.get("narrative", FALLBACK),
            },
            historical_matches=(l2a.get("matches") or [])[:3],
            divergence_matches=(l2a.get("divergence_matches") or [])[:3],
            flow_narrative={
                "flow_narrative": (l2b.get("llm") or {}).get("flow_narrative", FALLBACK),
                "japan_carry_interpretation": (l2b.get("llm") or {}).get("japan_carry_interpretation", FALLBACK),
                "china_capital_interpretation": (l2b.get("llm") or {}).get("china_capital_interpretation", FALLBACK),
            },
            country_deep_dive=country_deep_dive,
        ),
        encoding="utf-8",
    )
    _write_synthesis_lite_html(
        report_date=report_date,
        d=d,
        synthesis=synthesis,
        l1=l1,
        l2a=l2a,
        l2b=l2b,
        quant_context_json=quant_context_json,
        quant_momentum=quant_momentum,
        vs_us_alignment=vs_us_alignment,
        llm_meta=llm_meta,
        llm_placeholder=llm_placeholder,
    )
    out_data.write_text(
        json.dumps(
            {
                "date": d,
                "inputs": {"layer1": str(l1_path), "layer2_brief": str(l2a_path), "layer2_flow": str(l2b_path)},
                "quant_context": quant_context_json,
                "event_of_day": event_of_day,
                "surfaced_rows": surfaced_rows,
                "country_cycle": country_cycle,
                "us_growth_z_score": us_growth_z,
                "llm_meta": llm_meta,
                "llm_placeholder": llm_placeholder,
                "divergence_gauge": divergence_gauge,
                "synthesis": synthesis,
                "data_coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done. Output: {out_html}")
    print(f"Data JSON: {out_data}")
    try:
        from src.appendix_page import render_appendix_html

        render_appendix_html(d)
        print(f"Done. Output: {OUTPUT_DIR / 'appendix.html'}")
    except Exception as exc:
        print(f"  Warning: could not generate appendix.html ({exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
