"""Smoke test: Claude Haiku 4.5 + web_search tool.

Standalone — does NOT touch the main pipeline. Verifies three things:
  1. ANTHROPIC_API_KEY connects
  2. Haiku 4.5 responds
  3. web_search tool actually runs a live search and returns citations

Run: .venv/bin/python scripts/smoke_web_search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

from src.config import get_anthropic_model, require_anthropic_key  # noqa: E402


def main() -> int:
    client = anthropic.Anthropic(api_key=require_anthropic_key())
    model = get_anthropic_model()
    print(f"Model: {model}")

    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        messages=[
            {
                "role": "user",
                "content": (
                    "List the 3 biggest macro / geopolitical events of the past week "
                    "that could move global inflation, growth, interest rates, or oil. "
                    "For each: one line, with the source. Be concise."
                ),
            }
        ],
    )

    print("\n=== stop_reason:", resp.stop_reason)
    searched = False
    citations = []
    for block in resp.content:
        btype = getattr(block, "type", "?")
        if btype == "server_tool_use" and getattr(block, "name", "") == "web_search":
            searched = True
            q = getattr(block, "input", {}).get("query", "")
            print(f"[web_search] query: {q}")
        elif btype == "web_search_tool_result":
            results = getattr(block, "content", [])
            n = len(results) if isinstance(results, list) else "?"
            print(f"[web_search_result] {n} results")
        elif btype == "text":
            print("\n--- TEXT ---")
            print(getattr(block, "text", ""))
            for c in (getattr(block, "citations", None) or []):
                url = getattr(c, "url", None)
                if url:
                    citations.append(url)

    print("\n=== usage:", resp.usage)
    print("=== web_search ran:", searched)
    print("=== citations captured:", len(citations))
    for u in citations[:8]:
        print("   ", u)

    return 0 if searched else 2


if __name__ == "__main__":
    raise SystemExit(main())
