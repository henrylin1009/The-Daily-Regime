#!/usr/bin/env python3
"""Build a single split-page HTML for Layer 1/2/3."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.config import OUTPUT_DIR


def _exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def build_html(report_date: str) -> str:
    layer1 = OUTPUT_DIR / "global_regime.html"
    layer2a = OUTPUT_DIR / "brief.html"
    layer2b = OUTPUT_DIR / "flow_brief.html"
    layer3 = OUTPUT_DIR / "synthesis.html"

    def src_or_blank(path: Path) -> tuple[str, str]:
        if _exists(path):
            return path.name, ""
        return "about:blank", f"Missing: {path.name}"

    l1_src, l1_note = src_or_blank(layer1)
    l2a_src, l2a_note = src_or_blank(layer2a)
    l2b_src, l2b_note = src_or_blank(layer2b)
    l3_src, l3_note = src_or_blank(layer3)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Macro Intelligence Unified View — {report_date}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #ffffff; --panel: #f8f9fb; --border: #e8ecf0; --text: #0f1923; --muted: #8a96a3; --accent:#4f46e5;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{ width: 100%; height: 100%; font-family: "IBM Plex Sans", sans-serif; color: var(--text); background: var(--bg); }}
    .mono {{ font-family: "JetBrains Mono", monospace; color: #374151; }}
    .topbar {{
      height: 56px; display: flex; align-items: center; justify-content: space-between;
      padding: 0 14px; border-bottom: 1px solid var(--border); background: #fff; position: sticky; top: 0; z-index: 10;
    }}
    .title {{ font-size: 0.96rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; }}
    .date {{ font-size: 0.78rem; color: var(--muted); }}
    .layout {{
      height: calc(100% - 56px); display: grid; grid-template-rows: 36% 40% 24%; gap: 8px; padding: 8px;
    }}
    .section {{
      border: 1px solid var(--border); border-radius: 10px; overflow: hidden; background: var(--panel); display: grid; grid-template-rows: 34px 1fr;
    }}
    .section-header {{
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 10px; background: #fff; border-bottom: 1px solid var(--border); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
    }}
    .split-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 8px; background: var(--panel); }}
    .panel {{
      border: 1px solid var(--border); border-radius: 8px; overflow: hidden; background: #fff; display: grid; grid-template-rows: 28px 1fr;
    }}
    .panel-header {{
      padding: 0 8px; display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid var(--border); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em;
    }}
    .note {{ color: #b45309; font-size: 0.72rem; text-transform: none; letter-spacing: 0; }}
    iframe {{ width: 100%; height: 100%; border: 0; background: #fff; }}
  </style>
</head>
<body>
  <header class="topbar" style="display: flex; flex-direction: column; justify-content: center; align-items: flex-start; padding: 0 16px; height: 56px;">
    <div class="title" style="font-size: 1.15rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; color: #111827;">Macra Times</div>
    <div class="date mono" style="font-size: 0.76rem; color: var(--muted); margin-top: 1px;">{report_date}</div>
  </header>

  <main class="layout">
    <section class="section">
      <div class="section-header">
        <span>宏觀重力 (Macro Gravity)</span>
      </div>
      <iframe src="{l1_src}" title="Layer 1"></iframe>
    </section>

    <section class="section">
      <div class="section-header">
        <span>市場動能 (Market Momentum)</span>
      </div>
      <div class="split-2">
        <article class="panel">
          <div class="panel-header">
            <span>brief.html</span>
            <span class="note">{l2a_note}</span>
          </div>
          <iframe src="{l2a_src}" title="Layer 2 Macro Brief"></iframe>
        </article>
        <article class="panel">
          <div class="panel-header">
            <span>flow_brief.html</span>
            <span class="note">{l2b_note}</span>
          </div>
          <iframe src="{l2b_src}" title="Layer 2 Flow Brief"></iframe>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="section-header">
        <span>戰略綜合 (Strategic Synthesis)</span>
      </div>
      <article class="panel" style="margin:8px;">
        <div class="panel-header">
          <span>synthesis.html</span>
          <span class="note">{l3_note}</span>
        </div>
        <iframe src="{l3_src}" title="Layer 3 Synthesis"></iframe>
      </article>
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified Layer 1/2/3 split view HTML")
    parser.add_argument("--date", type=str, default=date.today().isoformat(), help="Date string YYYY-MM-DD")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else OUTPUT_DIR / "unified.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(args.date), encoding="utf-8")
    print(f"Done. Output: {out_path}")


if __name__ == "__main__":
    main()

