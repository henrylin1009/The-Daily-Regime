# The Daily Regime

A personal daily macro intelligence report that runs automatically on GitHub Actions every night at midnight (Taipei time). Pulls market and economic data from public APIs, runs a quantitative regime engine, finds historical analogues, and calls a large language model to synthesise everything into a plain-language brief.

**Problem it solves:** A long-term investor needs to know whether the macro backdrop changed — without reading 20 news sources or staring at Bloomberg terminals. This pipeline delivers that in one HTML page, every morning.

---

## What it produces

A single self-contained HTML report (`output/synthesis_lite.html`) committed to the repo each night, covering:

| Section | What it shows |
|---|---|
| **In One Line** | Single-sentence regime stance |
| **What to Watch** | 3 forward-looking alerts from the LLM |
| **Today's Pulse** | Biggest market mover of the day, with exact numbers |
| **Trend Tracking** | 2–3 medium-term macro themes (days to weeks) |
| **Narrative** | CIO-style stance + the key tension to watch |
| **Structural Backdrop** | Where we are in the Fed/liquidity/rate cycle — multi-year view |
| **Historical Analogues** | Closest past periods to today's macro fingerprint |
| **Cross-Border Flows** | Capital flows across US, Japan, China, Taiwan, Europe, UK |
| **Global Regime Map** | Investment Clock positioning for each major economy |

---

## Architecture

```
GitHub Actions (cron 0 16 * * *  →  00:00 Taipei)
│
├── flow_run.py          Cross-border capital flows + FX + bond yields
│                        → flow_data_{date}.json
│
├── run.py               FRED/yfinance data collection, Z-score signals,
│                        historical regime matching (30yr window),
│                        factor attribution, divergence matching
│                        → brief_data_{date}.json
│
├── global_regime.py     Investment Clock per country (A–E structural modules)
│                        → global_regime_data_{date}.json
│
└── synthesis.py         Two-call LLM strategy → synthesis_lite.html (committed)
    │
    ├── Call 1 (DeepSeek V4 Pro)    Main report — all narrative fields
    └── Call 2 (DeepSeek V4 Flash)  Gear matrix — asset class positioning
```

Only `output/synthesis_lite.html` is committed to git. All raw data stays local (gitignored).

---

## Key technical decisions

### Quantitative regime engine (`src/macro_quant_engine.py`)
- Pulls market data via yfinance with `period="max"` (20+ years)
- Computes rolling 252-day Z-scores across growth, inflation, liquidity, and risk dimensions
- Daily regime label (Goldilocks / Overheat / Stagflation / Deflationary Bust) via 2-of-3 vote on growth momentum + 2-of-2 on inflation momentum
- Investment Clock (monthly, `src/regime_tilt.py`) is the authoritative source for regime + duration displayed in the UI — the daily label feeds only the LLM narrative

### Historical analogue matching (`src/history_match.py`)
- Feature matrix built from 20+ years of monthly macro data (FRED + yfinance)
- Fixed 12-month exclusion window to prevent self-similar recent matches
- Returns top 3 closest historical periods with what-happened context
- Separate divergence matching (`src/divergence_match.py`) for cross-country signal splits

### Two-call LLM strategy (`synthesis.py`)
Split into two API calls to stay within token limits and control cost:
- **Call 1** — DeepSeek V4 Pro, `max_tokens=16384`: all reader-facing narrative fields (today_pulse, daily_themes, cio_directive, structural_context, historical_analogue_commentary)
- **Call 2** — DeepSeek V4 Flash, `max_tokens=6144`: gear matrix (asset class positioning table)
- Fallback chain: Flash → Pro
- Temperature 0.35 for consistent, non-hallucinated output

### Prompt engineering
- No-jargon rule enforced in system prompt: LLM is explicitly forbidden from using ticker symbols, Z-scores, or abbreviations (DXY, TLT, HYG, etc.) in reader-facing fields
- Each field has a concrete plain-language translation example in the prompt
- Labour market rule: if NFP trend is accelerating or decelerating, the LLM must mention it
- `structural_context` prompt explicitly told not to mention regime duration (regime_tilt card already shows it — avoids duplication)

---

## Data sources

| Data | Source | Used for |
|---|---|---|
| Equities, FX, bonds, commodities | yfinance | Z-scores, regime signals, flows |
| CPI, NFP, JOLTS, PCE, WALCL, M2 | FRED API | Inflation/labour context, Fed liquidity |
| ZQ futures (Fed funds path) | yfinance | Rate cycle positioning |
| Taiwan foreign equity flows | TWSE (scraped) | Cross-border flow monitor |
| Japan TIC data | US Treasury (scraped) | Cross-border flow monitor |
| China FX reserves | PBC (scraped) | Cross-border flow monitor |
| Fama-French 5+Mom factors | Ken French Data Library | Factor attribution |

---

## Project structure

```
flow_run.py             Cross-border capital flow pipeline
run.py                  Main data + quant pipeline
global_regime.py        Investment Clock per country
synthesis.py            LLM synthesis + HTML rendering

src/
  collect.py            FRED + yfinance data collection (30yr CSVs)
  macro_quant_engine.py Z-scores, regime label, rate path, Fed liquidity
  history_match.py      Historical analogue matching
  divergence_match.py   Cross-country divergence matching
  regime_tilt.py        Investment Clock (monthly regime + duration)
  factor_attrib.py      Fama-French + sector factor attribution
  indicators.py         Signal computation
  country_signals.py    Per-country macro signal builder
  sector_rotation.py    Sector momentum
  config.py             Model config, path constants

.github/workflows/
  daily.yml             Nightly pipeline (cron)
  ci.yml                Import + pytest on push
```

---

## Running locally

```bash
pip install -r requirements.txt

# Full pipeline (requires API keys in environment)
python flow_run.py --date 2026-06-09
python run.py --date 2026-06-09 --force-refresh
python global_regime.py --force-refresh --date 2026-06-09
python synthesis.py --date 2026-06-09

# Skip LLM call (uses placeholder text, useful for layout testing)
python synthesis.py --date 2026-06-09 --skip-llm
```

### Required environment variables

| Variable | Source |
|---|---|
| `DEEPSEEK_API_KEY` | [platform.deepseek.com](https://platform.deepseek.com) |
| `FRED_API_KEY` | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| `FMP_API_KEY` | [financialmodelingprep.com](https://financialmodelingprep.com) |

---

## CI

GitHub Actions runs `pytest tests/ -v` and an import check on every push to `main`. The daily pipeline itself runs on cron and commits only the final HTML — no credentials or raw data are committed.
