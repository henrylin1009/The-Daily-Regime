# Macra Times UI Specification & Design Guidelines

This document defines the layout, styling, and structural rules for the **Macra Times** 3-Layer Macro Intelligence dashboard. Future developers or AI assistants (e.g. Cursor, Claude, etc.) **must** adhere strictly to these rules to maintain design consistency and prevent layout disruption.

---

## 1. Global Rebranding & Typography

* **Website Name**: **Macra Times** (must be uppercase `Macra Times` in branding).
  * Positioned at the leftmost side of the outer page header.
* **Report Date**: Format `YYYY-MM-DD` (must be positioned directly below the `Macra Times` logo on the left).
* **Typography**:
  * Primary font: `"IBM Plex Sans"`, sans-serif.
  * Monospace font (for numbers, rates, and z-scores): `"JetBrains Mono"`, monospace.
* **No Script/Debug Names**: 
  * Do not show python script names or file paths (e.g. `run.py`, `flow_run.py`, `synthesis.py → synthesis.html`, `global_regime.py → global_regime.html`) in user-facing page layouts, headers, or section subtitles. Keep all debugging logs and pipeline details completely hidden from the UI.

---

## 2. Color Palette & Card Design

* **Page Viewport Background**: Unified dashboard grey (`#f8f9fb` or `#f4f6fb`). All layer pages, sub-pages, and briefs must use this background color.
* **Card Containers**: 
  * Background: Pure white (`#ffffff`).
  * Border: `1px solid #e8ecf0`.
  * Border Radius: `12px` (or `10px` for minor nested panels).
  * Box Shadow: Soft, subtle shadow (`box-shadow: 0 1px 4px rgba(0, 0, 0, 0.04)`).
  * Transitions: Soft hover translation (`transform: translateY(-2px)`) only on clickable cards (e.g., country cards with class `.card-btn`), not on static content panels.
* **No Accent Blue/Indigo Highlights in Content**:
  * Do not add thick blue or indigo left-borders or blue backgrounds to text blocks (specifically, do not use `border-left: 4px solid var(--accent)` or `background: #fafbff` for `.lead` or `.section-lead` elements). Keep overview and summary cards styled with uniform borders and clean white backgrounds.
  * Keep the accent color (`#4f46e5`) restricted only to active navigation elements (such as selected navigation tabs).

---

## 3. Outer Navigation Shell (`layer_pages.py`)

* **Navigation Tabs**: Placed on the right side of the main header, labeled exactly as:
  1. `LAYER 3 — 整合層（日頻，LLM）`
  2. `LAYER 2 — 動能層（日頻）`
  3. `LAYER 1 — 結構層（週/月頻）`
* **Active State**: The active tab uses a border highlight (`border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent)`), maintaining dark text color `#111827`.
* **Note Messages**: If there are missing files, the alert note `{note_html}` should be placed right next to the tab group, not as a main page title.

---

## 4. Layer 2 Tactical Matrix (`run.py` → `brief_global.html`)

Phase 1 **full-page accordion** (no main-view flow iframe):

* **Header**: `Macra Times · Tactical Matrix` + date.
* **Overview** + deterministic **spread trade line** (Long Rank 6 / Short Rank 1).
* **Main scroll**: 6-country `<details>` accordion, sorted Rank 1→6 (L2 signal stress heuristic). **US only** `<details open>` + `core-row` highlight + `[Global Core]` badge.
* **Expanded body**: `grid-3-gear` — [1] 戰術預期, [2] 底層水管, [3] 避險異常 (Phase 1 mapped from flow signals + quant `layer2_risk`).
* **Appendix** (`margin-top: 4rem`): legacy country cards, modal deep-dive, **iframe** `flow_brief.html`, optional tables.

[`flow_run.py`](flow_run.py) still generates `flow_brief.html` for appendix embed.

---

## 5. Layer 3 CIO War Room (`synthesis.py`)

Vertical executive flow. Background `#f8f9fb`. White cards; zone left-border accents only.

* **Header**: Macro Regime + Core Divergence badges (`divergence-badge` + `gauge-*`); penalty `(Penalty: …)` via `.meta`.
* **24H Macro Overview**: `daily_overview` (Phase 1: Zone 1 flash bullets or L2 executive summary).
* **Critical Watchout**: bulleted `critical_watchout` on `#fffbeb` tint.
* **CIO Executive Directive & Top Spread Trade**: stance, narrative, watchout + spread line.
* **3-Gear Conflict Matrix**: tab switcher (US default); per-country `grid-3-gear` with stacked L1 / L2 / Recent / CIO blocks (vanilla JS tab toggle).
* **Appendix**: legacy flash, global pure_alpha, momentum pills, raw tables, CoT, iframes, attribution.
* **Main scroll must not show**: raw Z tables, global momentum row, or legacy single-column pure_alpha (moved to appendix).

---

## 7. MACRA TIMES 3.0 Blueprint (Phase 1 vs Phase 2)

**Phase 1 (current):** Thin view-model in [`src/macra_ui.py`](../src/macra_ui.py) — ranking heuristics, accordion/matrix layouts, FALLBACK mappings. No quant engine or LLM schema changes.

**Phase 2 (planned):** Per-country divergence scores, structured `gear_matrix` JSON, LLM `daily_overview` / `critical_watchout` fields.

**Ranking philosophy:** Visual Rank 1–6 ordering establishes **ranking capability** before absolute point precision.

| Layer | Generator | Main UI |
|-------|-----------|---------|
| L1 | `global_regime.py` | Structural accordion — 結構重力 / 水管 / 溢酬 |
| L2 | `run.py` | Tactical accordion — 戰術預期 / 水管 / 避險 |
| L3 | `synthesis.py` | Vertical CIO War Room + tabbed 3-Gear matrix |

---

## 6. Layer 1 Structural Matrix (`global_regime.py`)

Phase 1 **accordion-first** main scroll; legacy Sections 1–7 in appendix.

* **Main scroll**: Ranked 6-country accordion (L1 policy-direction heuristic). US auto-expand + Global Core badge.
* **Columns**: [1] 結構重力, [2] 結構水管, [3] 結構溢酬.
* **Appendix**: Country cycle pills, regime summary, CB/CFTC/TIC/CA/liquidity tables.

---

## 8. Detail / Country Summary Pages

* **Style Uniformity**: All detail templates (generated via `templates/daily_brief.html`) must use a unified body background `#f8f9fb` and clean white container sections (`.section`), with no left-border accent lines or blue highlights on the summary blocks.
