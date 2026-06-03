# Macro Intelligence Platform

A daily macro intelligence tool for long-term index investors. Pulls public data via OpenBB, computes quantitative signals, and uses Gemini to generate plain-language macro summaries.

**Core value**: ~30 seconds a day to know if the big picture changed.

## Setup

```bash
cd /Users/henrylin/Desktop/analysis
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
```

### API keys

| Key | Source |
|-----|--------|
| `FRED_API_KEY` | Optional — [free at FRED](https://fred.stlouisfed.org/docs/api/api_key.html). Without it, macro series use FRED's public CSV export (no key needed). |
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey |

## Usage

```bash
# Full pipeline
python run.py

# Force refresh cached data
python run.py --force-refresh

# Offline dev (no Gemini API call; uses cached data if present)
python run.py --skip-llm

# If all data/raw/*.csv files exist, Stage 1 loads from cache without API keys

# Validate cached data (exit 1 on failure)
python -m src.collect --validate

# Test individual stages
python -m src.collect --force-refresh
python -m src.indicators
python -m src.history_match
python -m src.analyst --live
```

Output: `output/brief_YYYY-MM-DD.html`

**Progress report:** [docs/PROGRESS.md](docs/PROGRESS.md) (繁中，專案現況與限制)

## MVP complete — what you get

1. **20 market/macro series** cached under `data/raw/` (+ SPY for forward returns)
2. **3 traffic lights** (recession / inflation / financial stress) + detail table with per-series **As of** dates
3. **Historical regime match** (1994+ feature history) with decade diversity and `what_happened` blurbs
4. **Gemini macro summary** (or labeled template fallback)
5. Single command: `python run.py`

## Data sources (display vs regime)

| Key | Source | Notes |
|-----|--------|-------|
| Macro (CPI, NFP, …) | FRED via public CSV | Optional `FRED_API_KEY` for OpenBB |
| `credit_spread_hy` | FRED `BAA10Y` | Used in **regime matching** (long history) |
| `credit_spread_hy_oas` | FRED `BAMLH0A0HYM2` | **Display only**; FRED series from ~2023 |
| `uso` | DBO ETF | Oil proxy (key kept as `uso` for backward compatibility) |
| `gld` | GLD ETF | Share price; percentiles use 12m return |
| Sectors / VIX / DXY | OpenBB / yfinance | |
| `spy` | SPY ETF | Internal benchmark for match forward returns |

## Project structure

```
run.py              # Entry point
src/
  collect.py        # Stage 1: data collection
  indicators.py     # Stage 2: signals + traffic lights
  history_match.py  # Stage 3: historical regime comparison
  analyst.py        # Stage 4: Gemini narrative
  factor_attrib.py  # Stage 2b: 4-panel SPY attribution (macro / sectors / CRR / FF5+Mom)
  index_contrib.py  # Stage 2c: Top S&P 500 contributors (SPY proxy)
  regime_stats.py   # Stage 3b: regime persistence stats
  notify.py         # Email / Telegram delivery
  web.py            # FastAPI brief browser
  render.py         # Stage 4b: HTML output
  futures_adjust.py # Panama back-adjustment for ZQ=F
templates/
  daily_brief.html
data/raw/           # Cached CSVs (gitignored)
data/processed/     # Computed outputs (gitignored)
output/             # Daily HTML briefs (gitignored)
```

## Known limitations

1. **OpenBB FRED access**: Some FRED series require a free API key. Get one at https://fred.stlouisfed.org/docs/api/api_key.html

2. **ZQ=F roll adjustment**: CME continuous futures contract has roll artifacts at month boundaries. Panama back-adjustment is applied in `src/futures_adjust.py` before computing daily changes. (The `fed_surprise` reference project was not available; logic is self-contained here.)

3. **Monthly vs daily frequency**: Some indicators update monthly (CPI, NFP). Forward-fill to daily/monthly for the feature matrix. Staleness is noted per series in the output.

4. **No real-time data**: Output is as of previous market close. Not suitable for intraday decisions.

5. **Correlation ≠ causation**: Historical matches are statistical similarity, not causal prediction. LLM output reflects this.

6. **HMM regime model caveats**:
   - State labels are heuristic (assigned after training from feature means).
   - EM training can shift state boundaries over time; `random_state=42` reduces but does not eliminate this.
   - Model only trains on months where all regime features exist (drops rows with missing values).
   - Transition probabilities are descriptive (historical frequencies), not forecasts.
   - 3 states is a starting point; increase `N_STATES` in `src/regime_hmm.py` if regimes look too coarse.

## Phase 2 features

| Feature | Module | Usage |
|---------|--------|-------|
| Four-panel factor attribution | `src/factor_attrib.py` | Macro/assets + 6 sector SPDRs + CRR + FF5/Mom; Stage 2b |
| Top S&P 500 contributors (Panels E–F) | `src/index_contrib.py` | SSGA daily SPY holdings + yfinance prices; Stage 2c |
| HMM regime classification | `src/regime_hmm.py` | Stage 3c (optional); adds Regime Analysis block |
| Regime persistence / drawdown | `src/regime_stats.py` | Auto in `run.py` Stage 3b |
| Email / Telegram delivery | `src/notify.py` | `python run.py --notify` or set `NOTIFY_CHANNELS` |
| Web UI | `src/web.py` | `python -m src.web` → http://127.0.0.1:8080 |
| Daily schedule | `scripts/daily_run.sh` | Cron or `scripts/com.macro.daily.plist.example` |
| ECB / BOJ rates | `collect.py` | Optional columns in regime feature matrix |
| Tests + CI | `tests/`, `.github/workflows/ci.yml` | `pytest tests/ -v` |

```bash
# Web UI
python -m src.web

# Factor attribution (four panels: macro, XLK/XLF/XLV/XLE/XLU/XLI, CRR, FF5+Mom)
python -m src.factor_attrib

# S&P 500 contributors (SPY proxy)
python -m src.index_contrib

# HMM regime classification (requires feature matrix outputs)
python -m src.history_match
python -m src.regime_hmm

# Daily cron (after configuring .env notify vars)
chmod +x scripts/daily_run.sh
./scripts/daily_run.sh
```

## Future layers

- Institutional 13F positioning (L5)
- Mom factor / 5-factor model extension
