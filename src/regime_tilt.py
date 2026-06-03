"""
Regime-conditional asset tilt engine — quantitative Investment Clock replication.

Built on the Merrill Lynch Investment Clock (Greetham 2004), implemented in the
style of mainstream quantitative replications:

  * Growth & inflation are each a *weighted composite* of standardised indicators
    (OECD CLI, INDPRO, jobless claims, unemployment for growth; CPI-vs-target,
    CPI momentum, capacity utilisation for inflation).
  * Standardisation uses a trailing EWMA z-score (span≈24m) — no look-ahead bias.
  * The economy is always in one of the 4 quadrants; a hysteresis band (0.2σ)
    on each axis prevents whipsaw switching on minor data noise (there is no
    separate "transition" state).
  * Asset ranking is the hardcoded Greetham theoretical tilt for the current
    quadrant; historical episode returns are shown as context only.

Everything is computed from cached series — no new data sources at call time.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.config import RAW_DIR

# Environments — Merrill Lynch Investment Clock (always one of 4 quadrants)
ENV_GOLDILOCKS = "Goldilocks"
ENV_OVERHEAT = "Overheat"
ENV_STAGFLATION = "Stagflation"
ENV_DEFLATIONARY_BUST = "Deflationary Bust"

ENV_ORDER = [
    ENV_GOLDILOCKS,
    ENV_OVERHEAT,
    ENV_STAGFLATION,
    ENV_DEFLATIONARY_BUST,
]

# Trailing EWMA window for standardisation (no look-ahead).
EWM_SPAN = 24
# Hysteresis band (in composite-σ): an axis only flips sign once the signal
# crosses past the opposite side by this much — prevents whipsaw switching.
HYSTERESIS_BAND = 0.2

# Display metadata per environment
ENV_META = {
    ENV_GOLDILOCKS:       {"zh": "金髮女孩",   "stance": "risk-on",    "tone": "stable"},
    ENV_OVERHEAT:         {"zh": "過熱",       "stance": "real-assets","tone": "warning"},
    ENV_STAGFLATION:      {"zh": "停滯通膨",   "stance": "defensive",  "tone": "warning"},
    ENV_DEFLATIONARY_BUST:{"zh": "通縮崩潰",   "stance": "defensive",  "tone": "risk"},
}

# ── Greetham (2004) theoretical asset ranking per confirmed regime ─────────
# Scores: +2=strong OW, +1=OW, 0=neutral, -1=UW, -2=strong UW
# These drive the ↑↑/↑/—/↓/↓↓ display; historical % is shown as context only.
GREETHAM_TILT: dict[str, dict[str, int]] = {
    ENV_GOLDILOCKS: {
        "Equities": 2, "Credit": 1, "EM Assets": 1,
        "Long Bonds": 0, "Gold": 0, "Commodities": -1, "Cash": -1,
    },
    ENV_OVERHEAT: {
        "Commodities": 2, "Equities": 1, "Gold": 1,
        "EM Assets": 0, "Credit": 0, "Cash": -1, "Long Bonds": -2,
    },
    ENV_STAGFLATION: {
        "Commodities": 2, "Gold": 1, "Cash": 0,
        "Long Bonds": -1, "Credit": -1, "EM Assets": -1, "Equities": -2,
    },
    ENV_DEFLATIONARY_BUST: {
        "Long Bonds": 2, "Gold": 1, "Cash": 1,
        "Credit": -1, "Equities": -1, "EM Assets": -2, "Commodities": -2,
    },
}


# ── helpers ────────────────────────────────────────────────────────────────
def _to_monthly(df: pd.DataFrame | pd.Series) -> pd.Series:
    """Coerce a cached {date,value} frame (or series) to a month-end Series."""
    if isinstance(df, pd.Series):
        s = df.copy()
    else:
        s = df.set_index("date")["value"]
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last().astype(float)


def _monthly_return(df: pd.DataFrame | pd.Series) -> pd.Series:
    return _to_monthly(df).pct_change() * 100.0


def _zscore_ewm(s: pd.Series, span: int = EWM_SPAN) -> pd.Series:
    """Trailing EWMA z-score — standardise each point against an exponentially
    weighted *past* window only (no look-ahead). Used for all regime signals."""
    mu = s.ewm(span=span, min_periods=12).mean()
    sigma = s.ewm(span=span, min_periods=12).std()
    return (s - mu) / sigma.replace(0.0, np.nan)


def _hysteresis_sign(z: pd.Series, band: float = HYSTERESIS_BAND) -> pd.Series:
    """Path-dependent +1/-1 sign for one axis. The sign flips only when the
    signal crosses past the opposite side by more than `band` — this is the
    whipsaw guard mainstream quant replications use instead of a neutral zone."""
    out: list[int] = []
    state = 0
    for v in z:
        if pd.isna(v):
            out.append(state if state else 1)
            continue
        if state == 0:
            state = 1 if v >= 0 else -1
        elif state > 0 and v < -band:
            state = -1
        elif state < 0 and v > band:
            state = 1
        out.append(state)
    return pd.Series(out, index=z.index)


# ── Composite regime signals (quantitative Investment Clock replication) ─────
def compute_momentum_state(data: dict) -> dict:
    """
    Composite Investment Clock signals based on quantitative replication research.

    Growth composite (weighted z-score):
      OECD CLI 3m change     50%  — leading indicator, 6-9m ahead of GDP
      INDPRO YoY 3m change   20%  — coincident, industrial production
      Initial claims (inv)   15%  — leading, labour market
      Unemployment (inv)     15%  — lagging, confirms regime

    Inflation composite (weighted z-score):
      CPI YoY vs 2% target   40%  — Greetham level definition
      CPI YoY 3m change      30%  — momentum direction
      Capacity utilisation   30%  — demand-pull inflation pressure
    """
    # ── Growth composite (all directional, trailing-standardised) ─────────────
    cli   = _to_monthly(data["oecd_cli"])
    g_cli = _zscore_ewm(cli.diff(3))                      # CLI 3m change

    indpro = _to_monthly(data["indpro_growth"])
    g_ind  = _zscore_ewm(indpro.diff(3))                  # INDPRO YoY momentum

    claims = _to_monthly(data["initial_claims"])
    g_claims = _zscore_ewm(-claims.diff(3))               # inverted: fewer claims = G+

    unemp  = _to_monthly(data["unemployment_rate"])
    g_unemp = _zscore_ewm(-unemp.diff(3))                 # inverted: falling unemp = G+

    g_composite = (
        0.50 * g_cli
        + 0.20 * g_ind
        + 0.15 * g_claims
        + 0.15 * g_unemp
    )

    # ── Inflation composite ───────────────────────────────────────────────────
    cpi    = _to_monthly(data["cpi_yoy"])
    # Level vs the *fixed* 2% target — scaled (not re-centred) so the 2% anchor
    # survives. Expanding std keeps the scale stable without look-ahead.
    cpi_std = cpi.expanding(min_periods=24).std()
    i_gap  = (cpi - 2.0) / cpi_std
    i_mom  = _zscore_ewm(cpi.diff(3))                     # 3m momentum

    tcu    = _to_monthly(data["capacity_utilization"])
    i_tcu  = _zscore_ewm(tcu)                             # vs trailing-normal utilisation

    i_composite = (
        0.40 * i_gap
        + 0.30 * i_mom
        + 0.30 * i_tcu
    )

    g_now = float(g_composite.dropna().iloc[-1])
    i_now = float(i_composite.dropna().iloc[-1])
    i_now_z = i_now  # composite is already on a standardised scale

    return {
        "growth_mom": g_composite,
        "infl_mom": i_composite,
        "growth_mom_now": g_now,
        "infl_mom_now": i_now,
        "infl_mom_now_z": i_now_z,
        "growth_rising": g_now > 0,
        "infl_rising": i_now > 0,
    }


def _classify_env(g_rising: bool, i_rising: bool) -> str:
    """Pure 4-quadrant Investment Clock classification."""
    if g_rising and not i_rising:
        return ENV_GOLDILOCKS
    if g_rising and i_rising:
        return ENV_OVERHEAT
    if not g_rising and i_rising:
        return ENV_STAGFLATION
    return ENV_DEFLATIONARY_BUST


def _classify_env_series(g_composite: pd.Series, i_composite: pd.Series) -> pd.Series:
    """Hysteresis-classified 4-quadrant regime for every month (no neutral zone)."""
    idx = g_composite.dropna().index.intersection(i_composite.dropna().index)
    g = g_composite.reindex(idx)
    i = i_composite.reindex(idx)
    g_sign = _hysteresis_sign(g)
    i_sign = _hysteresis_sign(i)
    return pd.Series(
        [_classify_env(gs > 0, is_ > 0) for gs, is_ in zip(g_sign, i_sign)],
        index=idx,
    )


# ── asset universe (all data-backed proxies) ────────────────────────────────
def _build_asset_returns(data: dict) -> pd.DataFrame:
    """Monthly total-return proxies for each asset class."""
    cols: dict[str, pd.Series] = {}
    cols["Equities"] = _monthly_return(data["spy"])
    cols["EM Assets"] = _monthly_return(data["eem"])
    cols["Gold"] = _monthly_return(data["gld"])

    copper = _monthly_return(data["copper"])
    uso = _monthly_return(data["uso"])
    cols["Commodities"] = pd.concat([copper, uso], axis=1).mean(axis=1)

    cols["Credit"] = _monthly_return(data["emb"])  # EM hard-currency bonds proxy

    # Long bonds: synthesise total return from 10y yield level (duration ~8.5)
    dgs10 = data.get("dgs10")
    if dgs10 is None:
        raw = pd.read_csv(RAW_DIR / "dgs10.csv", index_col=0, parse_dates=True).squeeze()
        y10 = _to_monthly(raw)  # percent
    else:
        y10 = _to_monthly(dgs10)
    dy = y10.diff()
    cols["Long Bonds"] = (-8.5 * dy / 100.0 * 100.0) + (y10.shift(1) / 1200.0 * 100.0)

    # Cash: short rate / 12 (monthly), always small positive
    ff = _to_monthly(data["fed_funds_rate"])
    cols["Cash"] = ff.shift(1) / 12.0

    return pd.DataFrame(cols)


ASSET_ORDER = ["Equities", "Credit", "EM Assets", "Commodities", "Gold", "Long Bonds", "Cash"]
ASSET_ZH = {
    "Equities": "股票",
    "Credit": "信用債",
    "EM Assets": "新興市場",
    "Commodities": "商品",
    "Gold": "黃金",
    "Long Bonds": "長債",
    "Cash": "現金",
}


# ── conditional tilt estimation ─────────────────────────────────────────────
def _regime_episodes(env_series: pd.Series) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Identify contiguous regime episodes: list of (env, start, end) month-end dates."""
    episodes: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if env_series.empty:
        return episodes
    prev_env = env_series.iloc[0]
    ep_start = env_series.index[0]
    for date, env in env_series.items():
        if env != prev_env:
            episodes.append((prev_env, ep_start, date))
            prev_env = env
            ep_start = date
    episodes.append((prev_env, ep_start, env_series.index[-1]))
    return episodes


def compute_asset_tilts(data: dict) -> dict:
    """
    Quantitative Investment Clock asset tilt engine — always one of 4 quadrants.

    Regime:   hysteresis-classified quadrant from the growth/inflation composites
              (no neutral "transition" state; 0.2σ band guards against whipsaw).
    Ranking:  hardcoded Greetham theoretical tilt for the current quadrant.
    Context:  historical annualised episode returns shown as reference %.
    """
    mom = compute_momentum_state(data)
    rets = _build_asset_returns(data)

    g_mom = mom["growth_mom"]
    i_mom = mom["infl_mom"]

    # Composites are already standardised; use them directly as the axes.
    g_z_now = mom["growth_mom_now"]
    i_z_now = mom["infl_mom_now"]

    # ── Current quadrant (hysteresis, path-dependent) ─────────────────────────
    env_series = _classify_env_series(g_mom, i_mom)
    cur_env = str(env_series.iloc[-1])
    tilt_env = cur_env

    # ── Historical regime returns (context %) ─────────────────────────────────
    # Pool *all* months in each regime and annualise the mean monthly return —
    # the FactSet / Merrill Lynch convention. (Annualising each short episode
    # separately and averaging blows up on 2-4 month spikes, so we don't.)
    episodes = _regime_episodes(env_series)
    hist_returns: dict = {}
    hit_rates: dict = {}     # % months with positive return in regime
    volatilities: dict = {}  # annualised return std in regime
    counts: dict = {}        # number of contiguous episodes (for reference)
    month_counts: dict = {}  # number of months pooled (the actual sample size)
    for env in ENV_ORDER:
        counts[env] = len([ep for ep in episodes if ep[0] == env])
        env_months = env_series.index[env_series == env]
        env_rets = rets.loc[rets.index.isin(env_months)]
        month_counts[env] = int(env_rets.dropna(how="all").shape[0])
        for a in ASSET_ORDER:
            col = env_rets[a].dropna() / 100.0
            if len(col) < 6:  # need a meaningful month sample
                hist_returns[f"{env}:{a}"] = None
                hit_rates[f"{env}:{a}"] = None
                volatilities[f"{env}:{a}"] = None
                continue
            mean_monthly = float(col.mean())
            hist_returns[f"{env}:{a}"] = ((1 + mean_monthly) ** 12 - 1) * 100.0
            hit_rates[f"{env}:{a}"] = float((col > 0).mean()) * 100.0
            volatilities[f"{env}:{a}"] = float(col.std() * np.sqrt(12)) * 100.0

    # ── Unconditional (all months) returns — the regime-agnostic baseline ──────
    uncond_returns: dict = {}
    for a in ASSET_ORDER:
        col_all = rets[a].dropna() / 100.0
        if len(col_all) < 12:
            uncond_returns[a] = None
            continue
        mean_m = float(col_all.mean())
        uncond_returns[a] = ((1 + mean_m) ** 12 - 1) * 100.0

    # ── Greetham theoretical ranking → tilt levels ────────────────────────────
    theory_scores = GREETHAM_TILT[tilt_env]
    strength = min(1.0, (abs(g_z_now) + abs(i_z_now)) / 2.0)
    conviction = round(strength, 2)

    tilts: list[dict] = []
    for a in ASSET_ORDER:
        score = float(theory_scores.get(a, 0))
        level, tilt_word, icon = _tilt_level(score)
        hist = hist_returns.get(f"{tilt_env}:{a}")
        uncond = uncond_returns.get(a)
        hit = hit_rates.get(f"{tilt_env}:{a}")
        vol = volatilities.get(f"{tilt_env}:{a}")
        effect = (round(hist - uncond, 1) if hist is not None and uncond is not None else None)
        # Downside volatility: std of negative monthly returns × √12 × 100 (annualised)
        env_months = env_series.index[env_series == tilt_env]
        env_rets_a = rets.loc[rets.index.isin(env_months), a].dropna() / 100.0
        neg_col = env_rets_a[env_rets_a < 0]
        dv = float(neg_col.std() * np.sqrt(12)) * 100.0 if len(neg_col) >= 3 else None
        tilts.append({
            "asset": a,
            "asset_zh": ASSET_ZH[a],
            "tilt": tilt_word,
            "level": level,
            "icon": icon,
            "score": round(score, 2),
            "hist_return": None if hist is None else round(hist, 1),
            "uncond_return": None if uncond is None else round(uncond, 1),
            "effect": effect,
            "hit_rate": None if hit is None else round(hit, 0),
            "volatility": None if vol is None else round(vol, 0),
            "downside_vol": None if dv is None else round(dv, 0),
        })

    # ── 1. Regime transition matrix (Markov) ─────────────────────────────────
    trans_count: dict = defaultdict(lambda: defaultdict(int))
    env_vals = env_series.values
    for idx_i in range(len(env_vals) - 1):
        trans_count[env_vals[idx_i]][env_vals[idx_i + 1]] += 1
    transition_matrix: dict = {}
    for e in ENV_ORDER:
        row_total = sum(trans_count[e].values())
        transition_matrix[e] = {
            ne: (round(trans_count[e].get(ne, 0) / row_total * 100) if row_total > 0 else 0)
            for ne in ENV_ORDER
        }

    # ── 2. Current episode duration vs historical average ─────────────────────
    cur_duration = 0
    for v in env_vals[::-1]:
        if v == cur_env:
            cur_duration += 1
        else:
            break
    # Per-episode lengths for cur_env
    ep_lengths: list[int] = []
    run = 0
    for v in env_vals:
        if v == cur_env:
            run += 1
        elif run > 0:
            ep_lengths.append(run)
            run = 0
    if run > 0:
        ep_lengths.append(run)
    avg_duration = round(sum(ep_lengths) / len(ep_lengths), 1) if ep_lengths else None

    # ── 3. Distance to axis flip (hysteresis buffer) ──────────────────────────
    # The current axis state flips when it crosses ±HYSTERESIS_BAND in the opposite direction.
    # Buffer = how far the current signal is from the flip threshold (σ units).
    g_state = 1 if mom["growth_rising"] else -1
    i_state = 1 if mom["infl_rising"] else -1
    g_buffer = round(
        (g_z_now - (-HYSTERESIS_BAND)) if g_state > 0 else (HYSTERESIS_BAND - g_z_now), 2
    )
    i_buffer = round(
        (i_z_now - (-HYSTERESIS_BAND)) if i_state > 0 else (HYSTERESIS_BAND - i_z_now), 2
    )

    # ── 4. Historical percentile of current signals ────────────────────────────
    g_hist = mom["growth_mom"].dropna()
    i_hist = mom["infl_mom"].dropna()
    g_pct = int(round((g_hist < g_z_now).mean() * 100))
    i_pct = int(round((i_hist < i_z_now).mean() * 100))

    return {
        "environment": cur_env,
        "environment_nearest": cur_env,
        "environment_zh": ENV_META[cur_env]["zh"],
        "environment_tone": ENV_META[cur_env]["tone"],
        "environment_stance": ENV_META[cur_env]["stance"],
        "in_transition": False,
        "growth_rising": mom["growth_rising"],
        "infl_rising": mom["infl_rising"],
        "growth_mom_now": round(mom["growth_mom_now"], 3),
        "infl_mom_now": round(mom["infl_mom_now"], 3),
        "g_z_now": round(g_z_now, 3),
        "i_z_now": round(i_z_now, 3),
        "conviction": conviction,
        "sample_n": month_counts.get(cur_env, 0),
        "episode_n": counts.get(cur_env, 0),
        "tilts": tilts,
        "env_counts": counts,
        "conditional_table": {
            e: {a: hist_returns.get(f"{e}:{a}") for a in ASSET_ORDER}
            for e in ENV_ORDER
        },
        # ── New quantitative analytics ────────────────────────────────────────
        "transition_matrix": transition_matrix,    # {env: {next_env: pct}}
        "cur_duration": cur_duration,              # months in current episode
        "avg_duration": avg_duration,              # historical avg episode length (months)
        "g_buffer": g_buffer,                      # σ to flip growth axis
        "i_buffer": i_buffer,                      # σ to flip inflation axis
        "g_pct": g_pct,                            # g composite historical percentile
        "i_pct": i_pct,                            # i composite historical percentile
    }

def _tilt_level(z: float) -> tuple[int, str, str]:
    """Map a cross-sectional z-score to a 5-level tilt."""
    if z >= 0.85:
        return 2, "overweight", "↑↑"
    if z >= 0.25:
        return 1, "overweight", "↑"
    if z <= -0.85:
        return -2, "underweight", "↓↓"
    if z <= -0.25:
        return -1, "underweight", "↓"
    return 0, "neutral", "—"
