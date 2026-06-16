#!/usr/bin/env python3
"""Refresh the Threads long-lived access token and write it back to GitHub Secrets.

Threads long-lived tokens last 60 days. This script:
  1. Calls Threads' refresh endpoint to get a new 60-day token.
  2. Encrypts it with the repo's Actions public key and updates the
     THREADS_ACCESS_TOKEN secret via the GitHub REST API.

Env vars required:
  THREADS_ACCESS_TOKEN   current (not-yet-expired) long-lived token
  GH_PAT                 GitHub PAT with permission to write repo Actions secrets
  GITHUB_REPOSITORY      "owner/repo" (auto-set by GitHub Actions)

Usage:
  python scripts/refresh_threads_token.py
"""

from __future__ import annotations

import base64
import os
import sys

import requests
from nacl import encoding, public

GRAPH_BASE = "https://graph.threads.net/v1.0"
GITHUB_API = "https://api.github.com"


def refresh_token(current_token: str) -> str:
    resp = requests.get(
        f"{GRAPH_BASE}/refresh_access_token",
        params={"grant_type": "th_refresh_token", "access_token": current_token},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["access_token"]


def encrypt_secret(public_key_b64: str, secret_value: str) -> str:
    """Encrypt a secret value for the GitHub Actions secrets API (libsodium sealed box)."""
    public_key = public.PublicKey(public_key_b64.encode("utf-8"), encoding.Base64Encoder())
    sealed_box = public.SealedBox(public_key)
    encrypted = sealed_box.encrypt(secret_value.encode("utf-8"))
    return base64.b64encode(encrypted).decode("utf-8")


def update_github_secret(repo: str, pat: str, secret_name: str, secret_value: str) -> None:
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
    }
    key_resp = requests.get(
        f"{GITHUB_API}/repos/{repo}/actions/secrets/public-key", headers=headers, timeout=30
    )
    key_resp.raise_for_status()
    key_data = key_resp.json()

    encrypted_value = encrypt_secret(key_data["key"], secret_value)

    put_resp = requests.put(
        f"{GITHUB_API}/repos/{repo}/actions/secrets/{secret_name}",
        headers=headers,
        json={"encrypted_value": encrypted_value, "key_id": key_data["key_id"]},
        timeout=30,
    )
    put_resp.raise_for_status()


def main() -> None:
    current_token = os.environ.get("THREADS_ACCESS_TOKEN")
    pat = os.environ.get("GH_PAT")
    repo = os.environ.get("GITHUB_REPOSITORY")

    if not current_token or not pat or not repo:
        print(
            "ERROR: THREADS_ACCESS_TOKEN, GH_PAT, and GITHUB_REPOSITORY must be set",
            file=sys.stderr,
        )
        sys.exit(1)

    new_token = refresh_token(current_token)
    update_github_secret(repo, pat, "THREADS_ACCESS_TOKEN", new_token)
    print("THREADS_ACCESS_TOKEN refreshed and updated in GitHub Secrets.")


if __name__ == "__main__":
    main()
