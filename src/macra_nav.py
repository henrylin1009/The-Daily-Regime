"""Shared top-level layer navigation for MACRA TIMES shell pages."""

from __future__ import annotations

LAYER1_LABEL = "宏觀結構（佐證）"
LAYER2_LABEL = "市場動能（佐證）"
LAYER3_LABEL = "戰略綜合 · 主題分析"
APPENDIX_LABEL = "APPENDIX (驗證數據)"

LAYER_NAV_CSS = """
    .top { border-bottom: 1px solid var(--border); display:flex; align-items:center; justify-content:space-between; padding:0 16px; background: #fff; }
    .logo-area { display: flex; flex-direction: column; }
    .logo { font-size: 1.15rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; color: #111827; }
    .date-sub { font-size: 0.76rem; color: var(--muted); margin-top: 1px; font-family: "JetBrains Mono", monospace; }
    .right-area { display: flex; align-items: center; gap: 16px; }
    .note { font-size: 0.74rem; color: #b45309; margin-left: 8px; }
    .nav { display:flex; gap:8px; align-items:center; flex-wrap:wrap; justify-content:flex-end; }
    .tab {
      border:1px solid var(--border); border-radius:8px; padding:6px 10px;
      text-decoration:none; color:#64748b; font-size:0.78rem; background:#fff;
      transition: color 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .tab:hover { color:#0f172a; }
    .tab.active {
      border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent);
      font-weight: 600; color:#0f172a; background:#fff;
    }
"""


def layer_nav_hrefs(report_date: str) -> dict[str, str]:
    cb = report_date.replace("-", "")
    return {
        "layer3": "layer3.html",
        "layer2": "layer2.html",
        "layer1": "layer1.html",
        "appendix": f"appendix.html?cb={cb}",
    }


def build_layer_nav_html(active_layer: str, hrefs: dict[str, str]) -> str:
    """Return the four-tab nav block (L3 → L2 → L1 → Appendix)."""
    tabs = [
        ("layer3", LAYER3_LABEL, hrefs.get("layer3", "layer3.html")),
        ("layer2", LAYER2_LABEL, hrefs.get("layer2", "layer2.html")),
        ("layer1", LAYER1_LABEL, hrefs.get("layer1", "layer1.html")),
        ("appendix", APPENDIX_LABEL, hrefs.get("appendix", "appendix.html")),
    ]
    parts: list[str] = []
    for key, label, href in tabs:
        cls = "tab active" if active_layer == key else "tab"
        parts.append(f'<a class="{cls}" href="{href}">{label}</a>')
    return '<div class="nav">' + "".join(parts) + "</div>"
