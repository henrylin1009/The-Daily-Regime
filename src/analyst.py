"""Stage 4: Gemini API narrative generation."""

from __future__ import annotations

import argparse
import json
import re
import sys

from google import genai
from google.genai import types

from src.config import get_gemini_model, require_gemini_key
from src.indicators import RETURN_PERCENTILE_KEYS

SYSTEM_PROMPT = """
You are a macro analyst writing a daily brief.
Return only valid JSON with no markdown fences, no preamble, no explanation.
Follow the schema exactly.
"""

MATCH_CONTEXT_PROMPT = """
You are a macro historian. For each period below, write exactly 1-2 sentences describing
what was happening in the economy and markets (Fed policy, inflation, shocks).
Be specific. Do not predict or give advice.
Return ONLY a JSON array of objects with keys "date" and "what_happened".
"""


def _format_headline(signals: dict) -> str:
    lines = []
    for name, info in signals["headline"].items():
        lines.append(f"  - {name.replace('_', ' ').title()}: [{info['signal']}] {info['label']}")
    return "\n".join(lines)


def _format_details(signals: dict) -> str:
    lines = []
    for row in signals["details"]:
        pct_note = " (12m return pct)" if row["key"] in RETURN_PERCENTILE_KEYS else ""
        lines.append(
            f"  - {row['name']}: {row['value']:.2f} | {row['percentile']:.0f}th percentile{pct_note} | "
            f"trend: {row['trend']} | as of {row['as_of']}"
        )
    return "\n".join(lines)


def _format_matches(matches: list[dict]) -> str:
    if not matches:
        return "  (no historical matches available)"
    lines = []
    for i, m in enumerate(matches, 1):
        spy3 = m.get("spy_return_3m")
        spy12 = m.get("spy_return_12m")
        dd12 = m.get("max_drawdown_12m")
        spy3s = f"{spy3:.1f}%" if spy3 is not None else "N/A"
        spy12s = f"{spy12:.1f}%" if spy12 is not None else "N/A"
        dd12s = f"{dd12:.1f}%" if dd12 is not None else "N/A"
        lines.append(
            f"  {i}. {m['date']} — {m['description']}\n"
            f"     Distance: {m['distance']:.2f} | SPY +3m: {spy3s} | SPY +12m: {spy12s} | "
            f"Max DD 12m: {dd12s}"
        )
    return "\n".join(lines)


def _format_extremes(signals: dict) -> str:
    extremes: list[str] = []
    for row in signals.get("details", []):
        try:
            p = float(row.get("percentile", 50))
        except Exception:
            continue
        if p > 90 or p < 10:
            pct_note = " (12m return pct)" if row.get("key") in RETURN_PERCENTILE_KEYS else ""
            extremes.append(
                f"  - {row.get('name')}: {row.get('value')} ({p:.0f}th percentile{pct_note}, {row.get('trend')})"
            )
    return "\n".join(extremes) if extremes else "  (none)"


def _format_regime_summary(regime_summary: dict | None) -> str:
    if not regime_summary:
        return "  (no regime model output)"
    probs = regime_summary.get("transition_probs") or {}
    probs_s = ", ".join([f"{k}: {v*100:.0f}%" for k, v in probs.items()]) if probs else "n/a"
    return "\n".join(
        [
            f"  - Current regime: {regime_summary.get('label', 'N/A')} (p={regime_summary.get('probability', 0):.2f})",
            f"  - Months in regime: {regime_summary.get('months_in_regime', 'N/A')} | avg duration: {regime_summary.get('avg_duration_months', 'N/A')}",
            f"  - SPY avg 12m forward (regime-conditional): {regime_summary.get('avg_12m_forward_return', 'N/A')}%",
            f"  - Worst drawdown (regime-conditional): {regime_summary.get('worst_drawdown', 'N/A')}%",
            f"  - Transition probs (next month): {probs_s}",
        ]
    )


def build_user_prompt(signals: dict, matches: list[dict], regime_summary: dict | None = None) -> str:
    """Build a rich user prompt with all signals and historical context."""
    return f"""
=== DATE ===
As of: {signals.get('as_of', 'N/A')}

=== TRAFFIC LIGHTS (headline signals) ===
{_format_headline(signals)}

=== INDICATORS AT EXTREMES (>90th or <10th percentile) ===
{_format_extremes(signals)}

=== ALL INDICATORS ===
{_format_details(signals)}

=== HISTORICAL REGIME MATCHES (statistical similarity, not forecast) ===
{_format_matches(matches)}

=== REGIME MODEL (HMM, if available) ===
{_format_regime_summary(regime_summary)}

=== REQUIRED JSON SCHEMA ===
{{
  "executive_summary": "string",
  "narrative": "string",
  "match_descriptions": {{
    "YYYY-MM": "string",
    "YYYY-MM": "string",
    "YYYY-MM": "string"
  }}
}}

=== FIELD INSTRUCTIONS ===
- executive_summary: exactly 3 sentences, no raw numbers or percentile references, written for an educated
  non-specialist. Sentence 1 = cycle position and dominant tension. Sentence 2 = what history suggests.
  Sentence 3 = single most important forward indicator to watch.
- narrative: detailed macro narrative in current style, 4 paragraphs, can include indicator values and
  percentiles, no investment advice.
- match_descriptions: one entry per historical match date from the list above. Each value is 2-3 sentences
  describing macro/policy context in past tense. Do NOT include forward return figures in these descriptions.
"""


def _extract_response_text(response) -> str:
    """Collect full text from all response parts (avoids truncated .text)."""
    if response.text:
        return response.text.strip()
    chunks: list[str] = []
    for cand in response.candidates or []:
        for part in cand.content.parts or []:
            if getattr(part, "text", None):
                chunks.append(part.text)
    return "\n".join(chunks).strip()


def generate_template_narrative(signals: dict, matches: list[dict]) -> str:
    """Data-rich fallback narrative when the API is unavailable or truncated."""
    h = signals["headline"]
    top = matches[0] if matches else {}
    spy12 = top.get("spy_return_12m")
    spy12s = f"{spy12:+.1f}%" if spy12 is not None else "N/A"

    by_key = {d["key"]: d for d in signals["details"]}
    unrate = by_key.get("unemployment_rate", {})
    jolts = by_key.get("jolts_openings", {})
    credit = by_key.get("credit_spread_hy", {})
    oil = by_key.get("uso", {})
    gold = by_key.get("gld", {})

    match_lines = []
    for m in matches[:3]:
        s12 = m.get("spy_return_12m")
        s12t = f"{s12:+.1f}%" if s12 is not None else "N/A"
        match_lines.append(f"{m['date']} ({m['description']}; SPY +12m: {s12t})")

    return f"""The macro regime as of {signals.get('as_of', 'today')} sits between late-cycle caution and
still-resilient growth. Recession risk reads {h['recession_risk']['signal']}: the 2y10y curve is
{h['recession_risk']['value']:.0f}bp ({h['recession_risk']['percentile']:.0f}th percentile, {h['recession_risk']['trend']}).
Inflation is {h['inflation']['signal']} at {h['inflation']['value']:.2f}% CPI YoY ({h['inflation']['percentile']:.0f}th pct,
{h['inflation']['trend']}), above the Fed's 2% target but off peak levels. Financial stress is
{h['financial_stress']['signal']} with VIX at {h['financial_stress']['value']:.1f} ({h['financial_stress']['percentile']:.0f}th pct).

Labour markets remain tight by historical standards: unemployment {unrate.get('value', 0):.1f}%
({unrate.get('percentile', 0):.0f}th pct) with JOLTS openings at {jolts.get('value', 0):.0f}k
({jolts.get('trend', 'stable')}). Investment-grade credit spreads (Baa) are {credit.get('value', 0):.2f}%
({credit.get('percentile', 0):.0f}th pct). WTI crude is ${oil.get('value', 0):.2f}/bbl and gold (GLD)
${gold.get('value', 0):.0f} — both reflect {oil.get('trend', 'stable')} energy and {gold.get('trend', 'stable')}
safe-haven dynamics over the past quarter.

Statistically, today's feature vector most resembles {top.get('date', 'N/A')} ({top.get('description', '')}),
with similarity distance {top.get('distance', 0):.2f}. Other close eras: {'; '.join(match_lines[1:]) or 'n/a'}.
In that top match, SPY returned {spy12s} over the following 12 months — a historical analogue only, not a forecast.

Watch (i) whether inflation re-accelerates while the curve stays flat/inverted, (ii) credit spread widening
if growth slows, and (iii) sector rotation — tech (XLK {by_key.get('xlk', {}).get('percentile', 0):.0f}th pct
12m return) vs defensives (XLU {by_key.get('xlu', {}).get('percentile', 0):.0f}th pct). Not investment advice."""


def _strip_markdown_fences(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()


def _parse_brief_payload(text: str, match_dates: list[str]) -> dict | None:
    """Parse and validate combined JSON payload."""
    cleaned = _strip_markdown_fences(text)
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start:end])
    except Exception:
        return None
    if not all(k in payload for k in ("executive_summary", "narrative", "match_descriptions")):
        return None
    narrative = str(payload.get("narrative", "")).strip()
    executive_summary = str(payload.get("executive_summary", "")).strip()
    md = payload.get("match_descriptions")
    if not isinstance(md, dict):
        return None
    if any(d not in md for d in match_dates):
        return None
    sentences = re.findall(r"[^.!?]+[.!?]", executive_summary)
    if len(sentences) != 3:
        return None
    if not narrative:
        return None
    match_descriptions = {d: str(md.get(d, "")).strip() for d in match_dates}
    return {
        "executive_summary": executive_summary,
        "narrative": narrative,
        "match_descriptions": match_descriptions,
    }


def generate_brief_bundle(
    prompt: str,
    signals: dict | None = None,
    matches: list | None = None,
    model: str | None = None,
) -> dict | None:
    """Single Gemini call per run; returns all generated fields."""

    if signals is None or matches is None:
        raise ValueError("signals and matches required for template fallback")

    client = genai.Client(api_key=require_gemini_key())
    match_dates = [str(m.get("date")) for m in matches]
    response = client.models.generate_content(
        model=model or get_gemini_model(),
        contents=prompt.strip(),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.strip(),
            max_output_tokens=1400,
            temperature=0.7,
        ),
    )
    text = _extract_response_text(response)
    return _parse_brief_payload(text, match_dates)


def generate_narrative(prompt: str, signals: dict | None = None, matches: list | None = None) -> str | None:
    """Backward-compatible wrapper returning only the long narrative."""
    bundle = generate_brief_bundle(prompt, signals=signals, matches=matches)
    if not bundle:
        return None
    return bundle["narrative"]


def _template_what_happened(match: dict) -> str:
    """Fallback text for a historical match (label only)."""
    return match.get("description", f"Period {match.get('date', '')}")


def enrich_matches_what_happened(
    matches: list[dict],
    signals: dict,
    use_api: bool = True,
) -> list[dict]:
    """Fill what_happened using deterministic fallback text only."""
    if not matches:
        return matches

    for m in matches:
        m["what_happened"] = _template_what_happened(m)

    for m in matches:
        if not (m.get("what_happened") or "").strip():
            m["what_happened"] = _template_what_happened(m)
    return matches


PLACEHOLDER_NARRATIVE = (
    "Macro narrative unavailable (offline mode). "
    "Run without --skip-llm and ensure GEMINI_API_KEY is set."
)
PLACEHOLDER_EXECUTIVE_SUMMARY = "Narrative unavailable — run without --skip-llm to generate summary."


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate macro narrative via Gemini")
    parser.add_argument("--live", action="store_true", help="Call Gemini API")
    args = parser.parse_args()

    from src.collect import load_all_from_cache
    from src.history_match import build_feature_matrix, find_similar_periods
    from src.indicators import compute_signals

    data = load_all_from_cache()
    if not data:
        print("No cached data. Run: python -m src.collect", file=sys.stderr)
        sys.exit(1)

    signals = compute_signals(data)
    matrix = build_feature_matrix(data)
    matches = find_similar_periods(matrix, data)
    prompt = build_user_prompt(signals, matches)

    print("=== USER PROMPT ===")
    print(prompt)

    if args.live:
        try:
            bundle = generate_brief_bundle(prompt, signals, matches)
            print("\n=== NARRATIVE ===")
            print(bundle["narrative"] if bundle else "(none)")
            print("\n=== EXECUTIVE SUMMARY ===")
            print(bundle["executive_summary"] if bundle else "(none)")
            print("\n=== MATCH DESCRIPTIONS ===")
            print((bundle or {}).get("match_descriptions", {}))
        except EnvironmentError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n(Dry run — pass --live to call Gemini API)")


if __name__ == "__main__":
    main()
