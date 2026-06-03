#!/usr/bin/env python3
"""Generate 3 standalone layer pages in order: 3 -> 2 -> 1."""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import yfinance as yf

from src.config import OUTPUT_DIR
from src.macra_assets import macra_style_block
from src.macra_nav import (
    APPENDIX_LABEL,
    LAYER1_LABEL,
    LAYER2_LABEL,
    LAYER3_LABEL,
    LAYER_NAV_CSS,
    build_layer_nav_html,
    layer_nav_hrefs,
)
from src.appendix_page import render_appendix_html


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _page_template(
    title: str,
    report_date: str,
    frame_src: str,
    note: str,
    active_layer: str,
    layer3_href: str,
    layer2_href: str,
    layer1_href: str,
    appendix_href: str,
    market_snapshot_html: str,
    sub_tabs: list[dict] | None = None,
    macra_style: str = "",
) -> str:
    note_html = f'<span class="note">{note}</span>' if note else ""
    nav_html = build_layer_nav_html(
        active_layer,
        {
            "layer3": layer3_href,
            "layer2": layer2_href,
            "layer1": layer1_href,
            "appendix": appendix_href,
        },
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  {macra_style}
  <title>{title}</title>
  <style>
    :root {{ --border:#e8ecf0; --muted:#8a96a3; --accent:#4f46e5; }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ width: 100%; height: 100%; font-family: "Inter", "IBM Plex Sans", sans-serif; background: #f8f9fb; }}
    .wrap {{ height: 100%; display: grid; grid-template-rows: 56px auto{' 32px' if sub_tabs else ''} 1fr; }}
    .top {{ grid-row: 1; }}
    {LAYER_NAV_CSS}
    .sub-nav {{ grid-row: 3; display:flex; gap:6px; align-items:center; padding:6px 12px; border-bottom:1px solid var(--border); background:#fafbff; }}
    .sub-tab {{
      border:1px solid var(--border); border-radius:6px; padding:4px 9px;
      color:#374151; font-size:0.75rem; background:#fff; cursor:pointer;
      transition: border-color .15s;
    }}
    .sub-tab.active {{ border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); font-weight: 700; color: var(--accent); }}
    .snapshot {{
      grid-row: 2;
      border-bottom: 1px solid var(--border);
      padding: 8px 0;
      background: #fafbff;
      overflow: hidden;
      white-space: nowrap;
      position: relative;
    }}
    .ticker-track {{
      display: inline-flex;
      gap: 8px;
      padding: 0 12px;
      animation: ticker-scroll 28s linear infinite;
      will-change: transform;
    }}
    .snapshot:active .ticker-track {{
      animation-play-state: paused;
    }}
    .chip {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 8px;
      background: #fff;
      min-height: 50px;
      min-width: 140px;
      white-space: normal;
    }}
    .k {{ font-size: 0.72rem; color: var(--muted); }}
    .v {{ font-size: 0.9rem; font-weight: 600; margin-top: 2px; }}
    .chg {{ font-size: 0.75rem; margin-top: 2px; font-family: "JetBrains Mono", ui-monospace, monospace; }}
    .pos {{ color: #b91c1c; }}
    .neg {{ color: #047857; }}
    .flat {{ color: var(--muted); }}
    .asof {{ font-size: 0.68rem; color: var(--muted); margin-top: 2px; }}
    @keyframes ticker-scroll {{
      0% {{ transform: translateX(0); }}
      100% {{ transform: translateX(-50%); }}
    }}
    iframe {{ grid-row: {'4' if sub_tabs else '3'}; width: 100%; height: 100%; border: 0; min-height: 0; }}
  </style>
</head>
<body>
  <main class="wrap">
    <header class="top">
      <div class="logo-area">
        <div class="logo">Macra Times</div>
        <div class="date-sub">{report_date}</div>
      </div>
      <div class="right-area">
        {nav_html}
        {note_html}
      </div>
    </header>
    {market_snapshot_html}
    {'<div class="sub-nav">' + ''.join(f'<button class="sub-tab{" active" if i==0 else ""}" data-src="{t["src"]}">{t["label"]}</button>' for i, t in enumerate(sub_tabs)) + '</div>' if sub_tabs else ''}
    <iframe id="layer-frame" src="{frame_src}"></iframe>
  </main>
  {'<script>(function(){const frame=document.getElementById("layer-frame");const tabs=Array.from(document.querySelectorAll(".sub-tab"));tabs.forEach(function(btn){btn.addEventListener("click",function(){tabs.forEach(function(b){b.classList.remove("active");});btn.classList.add("active");if(frame)frame.src=btn.getAttribute("data-src");});});})()</script>' if sub_tabs else ''}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate standalone pages for Layer 3/2/1.")
    parser.add_argument("--date", type=str, default=date.today().isoformat(), help="Date YYYY-MM-DD")
    args = parser.parse_args()

    d = args.date
    cb = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    macra_style = macra_style_block()
    layer3_src = OUTPUT_DIR / "synthesis.html"
    layer2a_src = OUTPUT_DIR / "brief_global.html"
    if not _exists(layer2a_src):
        layer2a_src = OUTPUT_DIR / "brief.html"
    layer2b_src = OUTPUT_DIR / "flow_brief.html"
    layer1_src = OUTPUT_DIR / "global_regime.html"

    l3_page = OUTPUT_DIR / "layer3.html"
    l2_page = OUTPUT_DIR / "layer2.html"
    l1_page = OUTPUT_DIR / "layer1.html"
    appendix_page = OUTPUT_DIR / "appendix.html"
    nav_hrefs = layer_nav_hrefs(d)
    appendix_href = nav_hrefs["appendix"]

    l3_note = "" if _exists(layer3_src) else f"Missing: {layer3_src.name}"
    l2_note_parts = []
    if not _exists(layer2a_src):
        l2_note_parts.append(layer2a_src.name)
    if not _exists(layer2b_src):
        l2_note_parts.append(layer2b_src.name)
    l2_note = "" if not l2_note_parts else "Missing: " + ", ".join(l2_note_parts)
    l1_note = "" if _exists(layer1_src) else f"Missing: {layer1_src.name}"
    market_snapshot_html = _build_market_snapshot(d)

    l3_page.write_text(
        _page_template(
            title=LAYER3_LABEL,
            report_date=d,
            frame_src=(layer3_src.name + f"?cb={cb}") if _exists(layer3_src) else "about:blank",
            note=l3_note,
            active_layer="layer3",
            layer3_href=l3_page.name,
            layer2_href=l2_page.name,
            layer1_href=l1_page.name,
            appendix_href=appendix_href,
            market_snapshot_html=market_snapshot_html,
            macra_style=macra_style,
        ),
        encoding="utf-8",
    )
    l2_frame_src = (layer2a_src.name + f"?cb={cb}") if _exists(layer2a_src) else "about:blank"
    l2_page.write_text(
        _page_template(
            title=LAYER2_LABEL,
            report_date=d,
            frame_src=l2_frame_src,
            note=l2_note,
            active_layer="layer2",
            layer3_href=l3_page.name,
            layer2_href=l2_page.name,
            layer1_href=l1_page.name,
            appendix_href=appendix_href,
            market_snapshot_html="",
            macra_style=macra_style,
        ),
        encoding="utf-8",
    )
    l1_page.write_text(
        _page_template(
            title=LAYER1_LABEL,
            report_date=d,
            frame_src=(layer1_src.name + f"?cb={cb}") if _exists(layer1_src) else "about:blank",
            note=l1_note,
            active_layer="layer1",
            layer3_href=l3_page.name,
            layer2_href=l2_page.name,
            layer1_href=l1_page.name,
            appendix_href=appendix_href,
            market_snapshot_html="",
            macra_style=macra_style,
        ),
        encoding="utf-8",
    )

    print(f"Done. Output: {l3_page}")
    print(f"Done. Output: {l2_page}")
    print(f"Done. Output: {l1_page}")

    try:
        appendix_path = render_appendix_html(d)
        print(f"Done. Output: {appendix_path}")
    except Exception as exc:
        print(f"Warning: could not generate appendix.html ({exc})")


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.2f}"


def _fetch_close_and_change(ticker: str) -> tuple[float | None, float | None, str | None]:
    try:
        hist = yf.download(ticker, period="7d", interval="1d", progress=False, auto_adjust=False)
        if hist is None or hist.empty:
            return None, None, None
        close = hist["Close"]
        if hasattr(close, "iloc") and hasattr(close, "dropna"):
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
    return None, None, None


def _build_market_snapshot(d: str) -> str:
    labels = [
        ("S&P 500", "^GSPC"),
        ("Dow Jones", "^DJI"),
        ("Nikkei 225", "^N225"),
        ("Taiwan Index", "^TWII"),
        ("Oil (WTI)", "CL=F"),
        ("Gold", "GC=F"),
        ("Bitcoin", "BTC-USD"),
    ]

    chips = []
    fallback_detail_map: dict[str, float] = {}
    brief_data_path = OUTPUT_DIR / f"brief_data_{d}.json"
    if brief_data_path.exists():
        try:
            payload = json.loads(brief_data_path.read_text(encoding="utf-8"))
            details = payload.get("signals", {}).get("details", [])
            for row in details:
                key = str(row.get("key", ""))
                val = row.get("value")
                if isinstance(val, (int, float)):
                    fallback_detail_map[key] = float(val)
        except Exception:
            fallback_detail_map = {}

    for name, ticker in labels:
        px, chg_pct, asof = _fetch_close_and_change(ticker)
        if px is None:
            if ticker == "GC=F":
                px = fallback_detail_map.get("gld")
            elif ticker == "CL=F":
                px = fallback_detail_map.get("uso")
        if chg_pct is None:
            chg_text = "—"
            chg_cls = "flat"
        elif chg_pct > 0:
            chg_text = f"▲ {chg_pct:+.2f}%"
            chg_cls = "pos"
        elif chg_pct < 0:
            chg_text = f"▼ {chg_pct:+.2f}%"
            chg_cls = "neg"
        else:
            chg_text = f"{chg_pct:+.2f}%"
            chg_cls = "flat"
        chips.append(
            f'<div class="chip"><div class="k">{name}</div><div class="v">{_fmt(px)}</div><div class="chg {chg_cls}">{chg_text}</div><div class="asof">{asof or "as of —"}</div></div>'
        )
    chips_html = "".join(chips)
    return '<section class="snapshot"><div class="ticker-track">' + chips_html + chips_html + "</div></section>"


if __name__ == "__main__":
    main()

