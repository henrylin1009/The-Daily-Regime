# Three-Layer Gap Matrix (2026-05-28)

## Layer 1 (`global_regime_YYYY-MM-DD.html`)

- **Now:** `global_regime.py` renders all 7 required structural sections and exports `output/global_regime_data_YYYY-MM-DD.json`; cache file is `cache/global_regime.json` with 7-day freshness logic.
- **Should:** Match target as-is.
- **Gap:** No major functional gap.
- **Action:** Keep current Layer 1 data and layout logic; only ensure it remains content-only when embedded by `layer1_*.html`.

## Layer 2 (`brief_global_YYYY-MM-DD.html`)

- **Now:** Split view is implemented in `layer2_inner_YYYY-MM-DD.html`; `brief_global_YYYY-MM-DD.html` currently contains only macro summary + country explorer.
- **Should:** `brief_global_YYYY-MM-DD.html` itself is the split-view page (left macro brief iframe, right `flow_brief_YYYY-MM-DD.html` iframe) and keeps country explorer drill-down.
- **Gap:** Split responsibility is split across `run.py` and `layer_pages.py`; target says split should live in `brief_global`.
- **Action:** Move split-view back into `brief_global` generation in `run.py`; stop generating/using `layer2_inner` for main navigation.

## Layer 3 (`synthesis_YYYY-MM-DD.html`)

- **Now:** Includes Headline, Structural Context, Tactical Read, Interpretation, Watch List, Historical Matches, and a single `Country Attribution Deep Dive` section.
- **Should:** Same structure with no overlap between old `Country Index Attribution Explorer` and deep-dive block.
- **Gap:** No major structural gap found in current template.
- **Action:** Preserve single deep-dive section; verify regenerated output contains no duplicate attribution block.

## Three-Page Navigation

- **Now:** `layer1_*.html`, `layer2_*.html`, `layer3_*.html` wrappers cross-link correctly; default web entry goes to latest `layer1_*.html`.
- **Should:** Three pages mutually navigable and not isolated.
- **Gap:** `layer2_*.html` currently points to `layer2_inner_*.html` instead of `brief_global_*.html` as target Layer 2 page identity.
- **Action:** Point `layer2_*.html` directly to `brief_global_*.html`; keep wrappers as the canonical inter-layer navigation surface.

## Overlap/Regression Risks

- **Now:** Prior duplicate split issue was reduced by removing split from `brief_global`, but this diverges from target ownership.
- **Should:** No duplicated split blocks, no duplicated header/nav in embedded pages, no duplicated attribution sections.
- **Gap:** Ownership mismatch can reintroduce duplication or drift on reruns.
- **Action:** Enforce ownership boundaries in generator code and regenerate all pages from source.
