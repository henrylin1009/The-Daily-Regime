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
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

GRAPH_BASE = "https://graph.threads.net/v1.0"
MAX_LEN = 500  # Threads per-post character limit
ROOT = Path(__file__).resolve().parent.parent


def _truncate(text: str, limit: int = MAX_LEN) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def build_thread_posts(data: dict) -> list[str]:
    """Turn the synthesis JSON into a single post: today's headline + watch list."""
    pulse = data.get("synthesis", {}).get("today_pulse", {})
    headline = pulse.get("headline_en", "").strip()
    body = pulse.get("body_en", "").strip()
    date = data.get("date", "")

    main_parts = [f"THE DAILY REGIME / {date}"]
    if headline:
        main_parts.append(headline)
    if body:
        main_parts.append(body)
    main_post = _truncate("\n\n".join(main_parts))

    posts = [main_post] if main_post else []

    watch = data.get("synthesis", {}).get("watch_list", [])
    if watch:
        watch_txt = "\n".join(f"▸ {w}" for w in watch)
        posts.append(_truncate(f"WATCH LIST\n\n{watch_txt}"))

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
    posts = build_thread_posts(data)
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
