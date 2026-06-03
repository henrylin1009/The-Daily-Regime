"""Macro news retrieval via Claude Haiku 4.5 + web_search.

Independent step — does NOT touch synthesis.py / DeepSeek. Produces a
structured, cited, regime-tagged news JSON cached one-per-day.

Design (agreed):
  * Only events that map to one of 6 regime variables are kept.
  * Every event MUST carry a real source_url (anti-hallucination hard rule).
  * Every event tagged confirms / contradicts / beyond_quant, judged against
    a compact quant-conclusion summary fed into the prompt.
  * Output forced through a `record_events` tool (structured, not free text).
  * Failures degrade gracefully -> {"events": [], "errors": [...]}; never raise
    into a caller's pipeline.

CLI:  .venv/bin/python -m src.news_retrieval --date 2026-05-31 [--force]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

from src.config import OUTPUT_DIR, PROCESSED_DIR, get_anthropic_model, require_anthropic_key

REGIME_VARS = ("growth", "inflation", "rates", "fiscal", "risk_liquidity", "flows")
TAGS = ("confirms", "contradicts", "beyond_quant")

_MAX_SEARCHES = 4  # caps cost; web_search results dominate input tokens

_RECORD_EVENTS_TOOL = {
    "name": "record_events",
    "description": (
        "Record the macro/geopolitical events you found. Call exactly once with "
        "all events. Only include events that map to a regime variable AND have a "
        "real source URL from your web search."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "headline": {"type": "string", "description": "One concise line."},
                        "regime_var": {"type": "string", "enum": list(REGIME_VARS)},
                        "direction": {
                            "type": "string",
                            "description": "Effect on that variable, e.g. 'up', 'down', "
                            "'easing', 'tightening', 'disinflationary'.",
                        },
                        "tag": {"type": "string", "enum": list(TAGS)},
                        "rationale": {
                            "type": "string",
                            "description": "Short transmission chain to the regime variable.",
                        },
                        "source_url": {"type": "string", "description": "Real URL from search."},
                        "source_name": {"type": "string"},
                        "event_date": {"type": "string", "description": "YYYY-MM-DD if known."},
                    },
                    "required": [
                        "headline",
                        "regime_var",
                        "direction",
                        "tag",
                        "rationale",
                        "source_url",
                    ],
                },
            }
        },
        "required": ["events"],
    },
}

_SYSTEM = (
    "You are a macro news scout for a quantitative investing platform. Your job is "
    "to surface ONLY large macro / geopolitical / policy events from the past ~7 days "
    "that plausibly transmit to one of these 6 regime variables: "
    "growth, inflation, rates (monetary policy), fiscal, risk_liquidity, flows (capital/FX).\n\n"
    "HARD RULES:\n"
    "1. Use web_search to find real, recent events. Do NOT rely on memory.\n"
    "2. Every event MUST have a real source_url returned by your search. If you cannot "
    "cite it, drop it.\n"
    "3. Ignore single-stock news, earnings, sports, celebrity, local crime — anything "
    "that does not transmit to a regime variable.\n"
    "4. Tag each event against the QUANT CONCLUSION provided:\n"
    "   - 'confirms'   = event agrees with / explains the quant conclusion\n"
    "   - 'contradicts' = event points the opposite way to the quant conclusion\n"
    "   - 'beyond_quant' = a forward catalyst or risk the quant cannot see yet\n"
    "   Report contradicting events honestly; do not cherry-pick only confirming ones.\n"
    "5. Aim for 3-6 events. Quality over quantity.\n"
    "Finish by calling record_events exactly once."
)


def _build_quant_summary(date: str) -> str:
    """Compact quant-conclusion summary from the synthesis snapshot (best effort)."""
    snap = OUTPUT_DIR / f"synthesis_data_{date}.json"
    if not snap.exists():
        return "Quant conclusion unavailable for this date."
    try:
        d = json.loads(snap.read_text())
    except Exception:
        return "Quant conclusion unavailable (snapshot unreadable)."
    qc = d.get("quant_context") or {}
    label = qc.get("macro_regime_label", "?")
    detail = qc.get("macro_regime_detail") or {}
    growth = detail.get("growth", "?")
    inflation = detail.get("inflation", "?")
    gz = d.get("us_growth_z_score")
    l1 = qc.get("Layer_1_Structural") or {}
    yc = (l1.get("Yield_Curve") or {}).get("State", "?")
    parts = [
        f"Regime label: {label}.",
        f"Growth momentum: {growth} (US growth z={gz}).",
        f"Inflation momentum: {inflation}.",
        f"Yield curve: {yc}.",
    ]
    return " ".join(parts)


def _cache_path(date: str) -> Path:
    return PROCESSED_DIR / f"news_{date}.json"


def _validate(events: list[dict]) -> list[dict]:
    """Keep only well-formed, cited, in-taxonomy events."""
    clean = []
    for e in events:
        if not isinstance(e, dict):
            continue
        url = (e.get("source_url") or "").strip()
        if not url.startswith("http"):
            continue  # anti-hallucination hard rule
        if e.get("regime_var") not in REGIME_VARS:
            continue
        if e.get("tag") not in TAGS:
            e["tag"] = "beyond_quant"
        clean.append(e)
    return clean


def retrieve_news(date: str, force: bool = False) -> dict[str, Any]:
    """Retrieve & cache regime-tagged macro news for `date`. Never raises."""
    cache = _cache_path(date)
    if cache.exists() and not force:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass  # corrupt cache -> re-fetch

    out: dict[str, Any] = {
        "retrieved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "date": date,
        "model": get_anthropic_model(),
        "quant_summary": _build_quant_summary(date),
        "events": [],
        "errors": [],
    }

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=require_anthropic_key())
        resp = client.messages.create(
            model=out["model"],
            max_tokens=2048,
            system=_SYSTEM,
            tools=[
                {"type": "web_search_20250305", "name": "web_search", "max_uses": _MAX_SEARCHES},
                _RECORD_EVENTS_TOOL,
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"QUANT CONCLUSION (as of {date}):\n{out['quant_summary']}\n\n"
                        "Find the biggest macro/geopolitical events of the past week that "
                        "transmit to a regime variable, then call record_events."
                    ),
                }
            ],
        )
        out["usage"] = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "web_search_requests": resp.usage.server_tool_use.web_search_requests
            if resp.usage.server_tool_use
            else 0,
        }
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == "record_events":
                raw = (getattr(block, "input", {}) or {}).get("events", [])
                out["events"] = _validate(raw)
        if not out["events"]:
            out["errors"].append("model returned no valid (cited) events")
    except Exception as exc:  # degrade gracefully — never break a caller
        out["errors"].append(f"{type(exc).__name__}: {exc}")

    try:
        cache.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception:
        pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Retrieve regime-tagged macro news via Claude web_search.")
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    ap.add_argument("--force", action="store_true", help="ignore cache, re-fetch")
    args = ap.parse_args()

    res = retrieve_news(args.date, force=args.force)
    print(f"date={res['date']}  model={res['model']}")
    print(f"quant_summary: {res['quant_summary']}")
    print(f"usage: {res.get('usage')}")
    print(f"events: {len(res['events'])}   errors: {res['errors']}")
    for e in res["events"]:
        print(f"\n  [{e['tag']:11}] ({e['regime_var']}/{e['direction']}) {e['headline']}")
        print(f"      → {e['rationale']}")
        print(f"      {e.get('source_name','')} {e['source_url']}")
    print(f"\ncached → {_cache_path(res['date'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
