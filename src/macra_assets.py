"""Static UI asset helpers for MACRA institutional CSS."""

from __future__ import annotations

from functools import lru_cache

from src.config import TEMPLATES_DIR


@lru_cache(maxsize=1)
def read_macra_institutional_css() -> str:
    src = TEMPLATES_DIR / "macra_institutional.css"
    if not src.exists():
        return ""
    return src.read_text(encoding="utf-8")


def macra_style_block() -> str:
    """Inline <style> block for embedding in generated HTML heads."""
    css = read_macra_institutional_css()
    if not css.strip():
        return ""
    return f"<style>\n{css}\n</style>"
