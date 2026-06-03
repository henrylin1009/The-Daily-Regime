# Macro Intelligence Session Handoff

## Goal We're Working Toward

Restore a stable and understandable 3-layer browsing experience with minimal confusion:
- one clear entry point for users
- layer navigation from header tabs
- no duplicated split-view/header blocks
- preserve existing Layer 2/3 data behavior while reducing UI drift

## Current State of the Code

- Local web server is running and reachable at `http://127.0.0.1:8080`.
- Root route now prefers the latest `layer1_YYYY-MM-DD.html` as the default landing page; it falls back to latest `brief_YYYY-MM-DD.html` if no layer page exists.
- Layer 2 duplicate split-view issue is resolved:
  - `layer2_2026-05-28.html` provides the split shell (left/right iframe container)
  - `brief_global_2026-05-28.html` no longer re-renders an additional "B. Layer 2 Split View" section
- Global Regime inner page duplicate top header/navigation was removed from `global_regime` output to avoid double headers inside `layer1` iframe shell.
- Current "what to open" UX is simplified: open only `http://127.0.0.1:8080`, then switch layers from tab header.

## Files You're Actively Editing

- `src/web.py`
  - Added layer-first default routing (`/` now redirects to latest `layer1_*.html` when present)
  - Added `/file/{filename}` endpoint to serve generated HTML entry files directly
- `run.py`
  - Removed duplicated Layer 2 split-view section from generated `brief_global_*`
  - Kept country explorer section as the next section (`B`)
- `global_regime.py`
  - Removed inner sticky top header and layer nav links from the generated global regime content
  - Kept content body and country-cycle info as an in-page section
- `output/brief_global_2026-05-28.html` (hotfix output sync)
- `output/global_regime_2026-05-28.html` (hotfix output sync)

## Everything You've Tried That Failed

- Earlier server process (`Start local FastAPI brief server`) ended with `exit_code=137`; that process was replaced by a fresh server instance and is now healthy.
- Previous UX required users to manually pick among many HTML files, causing repeated "which file should I open" confusion.
- Layer 2 structure had nested split-view rendering (split shell + split inside `brief_global`) which visually duplicated content.
- Global Regime page had duplicated layer header/nav when embedded inside `layer1` shell iframe.

## Next Step I'd Take

1. Add a minimal "Entry Hub" page (or streamline `/archive`) that only lists core pages:
   - `layer1_YYYY-MM-DD.html`
   - `layer2_YYYY-MM-DD.html`
   - `layer3_YYYY-MM-DD.html`
   and hides legacy/noisy outputs by default.
2. Regenerate fresh dated outputs (instead of hotfixing only current output files) so generator source and output are fully aligned for next run.
3. Run a quick integrity check of layer navigation:
   - tabs switch correctly
   - no repeated headers
   - Layer 2 split appears once
   - country detail modal still works.

