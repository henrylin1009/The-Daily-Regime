#!/usr/bin/env python3
"""Post the daily synthesis report to Threads as a thread (chain of posts).

Reads output/synthesis_data_{date}.json (produced by synthesis.py) and posts a
short English summary thread via the Threads Graph API.

Env vars required:
  THREADS_ACCESS_TOKEN  long-lived user access token (from the Meta Developer
                        Console's "User token generator" for a Threads tester)
  THREADS_USER_ID       numeric Threads user id (GET /v1.0/me?fields=id with
                        the access token above)

Usage:
  python scripts/post_to_threads.py --date 2026-06-16
  python scripts/post_to_threads.py --date 2026-06-16 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

GRAPH_BASE = "https://graph.threads.net/v1.0"
MAX_LEN = 500  # Threads per-post character limit
HASHTAG = "#macro"
ROOT = Path(__file__).resolve().parent.parent

# today_pulse.signal -> short header tag shown after the date
SIGNAL_TAGS = {
    "risk_on": "RISK-ON",
    "risk_off": "RISK-OFF",
    "rates_tightening": "RATES TIGHTENING",
    "rates_easing": "RATES EASING",
    "liquidity_draining": "LIQUIDITY DRAINING",
    "liquidity_expanding": "LIQUIDITY EXPANDING",
}

# Strip leaked tickers/jargon from reader-facing text (subset of synthesis.py's table).
# Case-sensitive on the bare ticker (all-caps) to avoid clobbering common words.
JARGON_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bDXY\b"), "the US dollar"),
    (re.compile(r"\bHYG\b"), "high-yield credit"),
    (re.compile(r"\bSPY\b"), "the US stock market"),
    (re.compile(r"\bVVIX\b"), "the volatility of volatility"),
    (re.compile(r"\bVIX\b"), "volatility index"),
    (re.compile(r"\bTLT\b"), "long-term bonds"),
    (re.compile(r"\s*\(?\bz\s*=\s*-?\d+\.?\d*\)?", re.I), ""),
]


def _sanitize(text: str) -> str:
    s = text or ""
    for pat, repl in JARGON_REPLACEMENTS:
        s = pat.sub(repl, s)
    # Collapse "The the volatility index" -> "The volatility index" etc.
    s = re.sub(r"\b([Tt]he)\s+the\b", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _truncate(text: str, limit: int = MAX_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _truncate_sentences(text: str, limit: int = MAX_LEN) -> str:
    """Trim to the last full sentence that fits, so it never ends mid-thought."""
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    cut = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "))
    if cut > limit // 2:
        return clipped[: cut + 1]
    return _truncate(text, limit)


def generate_hook(headline: str, body: str) -> str:
    """Generate a two-line contrast hook via DeepSeek.

    Returns "" on any failure so the daily post never breaks.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or not headline:
        return ""

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        )
        prompt = (
            "You write the opening hook for a macro markets post on Threads.\n"
            "Style: cold, precise, no hype, no emoji, no hashtags, no jargon.\n\n"
            f"Today headline: {headline}\n"
            f"Today detail: {body}\n\n"
            "Write a two-line opener in the style "
            '"Everyone sees X.\\nAlmost no one is watching Y." — the crowd view vs. '
            "the tension building underneath. Two short lines separated by \\n, "
            "max ~20 words total.\n"
            'Return only JSON: {"hook": "...""}'
        )
        resp = client.chat.completions.create(
            model=os.environ.get("DEEPSEEK_GEAR_MODEL", "deepseek-v4-flash"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,  # v4-flash is a reasoning model; leave room for thinking + JSON
            temperature=0.6,
            response_format={"type": "json_object"},
        )
        out = json.loads(resp.choices[0].message.content)
        return out.get("hook", "").strip()
    except Exception as exc:  # noqa: BLE001 — never let hook generation break posting
        print(f"WARN: hook generation failed, posting without hook: {exc}", file=sys.stderr)
        return ""


def build_thread_posts(data: dict, hook: str = "") -> list[str]:
    """Build the three-post thread: lead post, market summary, watch list."""
    synthesis = data.get("synthesis", {})
    pulse = synthesis.get("today_pulse", {})
    body = pulse.get("body_en", "").strip()
    date = data.get("date", "")
    tag = SIGNAL_TAGS.get(pulse.get("signal", ""), "")

    header = f"// THE DAILY REGIME · {date}"
    if tag:
        header += f" · {tag}"

    # Post 1 — header on top, hook, then the plain-language pulse. Headline is
    # dropped: the hook already names the move, so repeating it reads redundant.
    lead_parts = [header]
    if hook:
        lead_parts.append(hook)
    if body:
        lead_parts.append(body)
    lead_parts.append(HASHTAG)
    posts = [_truncate("\n\n".join(lead_parts))]

    # Post 2 — fuller market summary, jargon-sanitised.
    summary = _sanitize(pulse.get("market_summary_en", "").strip())
    if summary:
        summary = _truncate_sentences(summary, MAX_LEN - len("// MARKET SUMMARY\n\n"))
        posts.append(f"// MARKET SUMMARY\n\n{summary}")

    # Post 3 — watch list.
    watch = synthesis.get("watch_list", [])
    if watch:
        watch_txt = "\n".join(f"▸ {w}" for w in watch)
        posts.append(_truncate(f"// WATCH LIST\n\n{watch_txt}"))

    return [p for p in posts if p]


def create_and_publish(token: str, user_id: str, text: str, reply_to_id: str | None) -> str:
    """Create a Threads media container for `text`, then publish it. Returns post id."""
    params = {
        "media_type": "TEXT",
        "text": text,
        "access_token": token,
    }
    if reply_to_id:
        params["reply_to_id"] = reply_to_id

    resp = requests.post(f"{GRAPH_BASE}/{user_id}/threads", params=params, timeout=30)
    resp.raise_for_status()
    creation_id = resp.json()["id"]

    # Give the container a moment to finish processing before publishing.
    time.sleep(2)

    pub = requests.post(
        f"{GRAPH_BASE}/{user_id}/threads_publish",
        params={"creation_id": creation_id, "access_token": token},
        timeout=30,
    )
    pub.raise_for_status()
    return pub.json()["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Post the daily report as a Threads thread")
    parser.add_argument("--date", required=True, help="Report date, e.g. 2026-06-16")
    parser.add_argument("--dry-run", action="store_true", help="Print posts instead of publishing")
    args = parser.parse_args()

    data_path = ROOT / "output" / f"synthesis_data_{args.date}.json"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found — run synthesis.py for this date first.", file=sys.stderr)
        sys.exit(1)

    data = json.loads(data_path.read_text())
    pulse = data.get("synthesis", {}).get("today_pulse", {})
    hook = generate_hook(
        pulse.get("headline_en", "").strip(), pulse.get("body_en", "").strip()
    )
    posts = build_thread_posts(data, hook=hook)
    if not posts:
        print("ERROR: no content extracted from synthesis data.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        for i, p in enumerate(posts, 1):
            print(f"--- post {i}/{len(posts)} ({len(p)} chars) ---")
            print(p)
            print()
        return

    token = os.environ.get("THREADS_ACCESS_TOKEN")
    user_id = os.environ.get("THREADS_USER_ID")
    if not token or not user_id:
        print("ERROR: set THREADS_ACCESS_TOKEN and THREADS_USER_ID in .env", file=sys.stderr)
        sys.exit(1)

    reply_to_id: str | None = None
    for i, text in enumerate(posts, 1):
        post_id = create_and_publish(token, user_id, text, reply_to_id)
        print(f"[{i}/{len(posts)}] posted id={post_id}")
        reply_to_id = post_id
        time.sleep(3)  # be gentle on rate limits between chained posts


if __name__ == "__main__":
    main()
