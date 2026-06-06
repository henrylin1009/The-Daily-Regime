"""Shared paths and environment configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

load_dotenv(PROJECT_ROOT / ".env", override=True)

# Accept common .env typos (env var names are case-sensitive on macOS/Linux)
if not os.getenv("GEMINI_API_KEY"):
    for _alt in ("gemini_api_KEY", "GEMINI_api_KEY", "gemini_API_KEY"):
        if os.getenv(_alt):
            os.environ["GEMINI_API_KEY"] = os.environ[_alt]
            break

if not os.getenv("FMP_API_KEY"):
    for _alt in ("fmp_api_KEY", "FMP_api_KEY", "fmp_API_KEY", "fmp_api_key"):
        if os.getenv(_alt):
            os.environ["FMP_API_KEY"] = os.environ[_alt]
            break

if not os.getenv("DEEPSEEK_API_KEY"):
    for _alt in (
        "DEEP_SEEK_KEY",
        "DEEPSEEK_KEY",
        "deepseek_api_KEY",
        "DEEPSEEK_api_KEY",
        "deepseek_API_KEY",
    ):
        if os.getenv(_alt):
            os.environ["DEEPSEEK_API_KEY"] = os.environ[_alt]
            break

for _dir in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


def _api_key_valid(name: str) -> bool:
    key = (os.getenv(name) or "").strip()
    return bool(key) and key not in ("...", "your_key_here", "REPLACE") and len(key) > 8


def fred_api_key_valid() -> bool:
    """True if a real FRED API key is configured (optional — CSV fallback works without it)."""
    return _api_key_valid("FRED_API_KEY")


def fmp_api_key_valid() -> bool:
    """True if FMP key is set (needed for SPY holdings / ETF weights via OpenBB)."""
    return _api_key_valid("FMP_API_KEY") or _api_key_valid("fmp_api_key")


def configure_openbb() -> None:
    """Configure OpenBB; inject FRED/FMP keys when present."""
    from openbb import obb

    if fred_api_key_valid():
        obb.user.credentials.fred_api_key = os.environ["FRED_API_KEY"]
    if fmp_api_key_valid():
        obb.user.credentials.fmp_api_key = (
            os.getenv("fmp_api_key") or os.getenv("FMP_API_KEY") or ""
        )


def require_gemini_key() -> str:
    """Return Gemini API key or raise."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to .env (get a key at "
            "https://aistudio.google.com/apikey). GOOGLE_API_KEY is also accepted."
        )
    return key


def require_anthropic_key() -> str:
    """Return Anthropic (Claude) API key or raise."""
    if not _api_key_valid("ANTHROPIC_API_KEY"):
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to .env "
            "(get a key at https://console.anthropic.com/)."
        )
    return os.environ["ANTHROPIC_API_KEY"].strip()


def get_anthropic_model() -> str:
    """Claude model id for news retrieval (default: lowest-cost Haiku 4.5)."""
    return os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def get_gemini_model() -> str:
    """Model id for narrative generation (default: lowest-cost flash-lite)."""
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


def require_deepseek_key() -> str:
    """Return DeepSeek API key or raise."""
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key or not _api_key_valid("DEEPSEEK_API_KEY"):
        raise EnvironmentError(
            "DEEPSEEK_API_KEY is not set. Add it to .env (https://platform.deepseek.com/api_keys)."
        )
    return key.strip()


def get_deepseek_model() -> str:
    """DeepSeek chat model id (OpenAI-compatible API)."""
    return os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")


def get_deepseek_base_url() -> str:
    """DeepSeek OpenAI-compatible API base URL."""
    return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
