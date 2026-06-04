"""
Hidden Markov Model (HMM) regime classification.

Trains on monthly macro feature matrix (z-scored) and classifies the current regime.

Standalone:
  python -m src.regime_hmm

Dependencies:
  - hmmlearn (added to requirements.txt in Phase 2)
  - Feature matrices are written by `src.history_match.build_feature_matrix()` to:
      data/processed/feature_matrix_raw.csv
      data/processed/feature_matrix_zscored.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn import hmm

from src.config import PROCESSED_DIR, RAW_DIR
from src.history_match import REGIME_FEATURES

N_STATES = 4


def load_feature_matrix() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load monthly feature matrices produced by history_match.py."""
    raw_path = PROCESSED_DIR / "feature_matrix_raw.csv"
    z_path = PROCESSED_DIR / "feature_matrix_zscored.csv"
    if not raw_path.exists() or not z_path.exists():
        raise FileNotFoundError(
            "Feature matrix not found. Run history_match first:\n"
            "  python -m src.history_match"
        )
    raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    z = pd.read_csv(z_path, index_col=0, parse_dates=True)
    return raw, z


def train_hmm(z_matrix: pd.DataFrame, n_states: int = N_STATES) -> hmm.GaussianHMM:
    """Fit Gaussian HMM on z-scored features."""
    X_df = z_matrix[REGIME_FEATURES].dropna()
    if X_df.empty:
        raise RuntimeError("No rows available for HMM training (all NaN after dropping).")
    X = X_df.values
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
        verbose=False,
    )
    model.fit(X)
    return model


def label_states(model: hmm.GaussianHMM, feature_names: list[str]) -> dict[int, str]:
    """
    Auto-label states based on feature means after training.

    Heuristic:
      - highest fed_funds_rate mean -> Restrictive
      - highest vix mean -> Stress / Contraction
      - remaining -> Expansionary
    """
    means_df = pd.DataFrame(model.means_, columns=feature_names)
    restrictive = int(means_df["fed_funds_rate"].idxmax()) if "fed_funds_rate" in means_df.columns else 0
    stress = int(means_df["vix"].idxmax()) if "vix" in means_df.columns else (restrictive + 1) % model.n_components

    # Ensure unique assignment.
    if restrictive == stress:
        # Pick stress by vix; pick restrictive as best remaining by fed_funds_rate (or 0).
        stress = int(means_df["vix"].idxmax()) if "vix" in means_df.columns else restrictive
        remaining = [i for i in range(model.n_components) if i != stress]
        if remaining:
            if "fed_funds_rate" in means_df.columns:
                restrictive = int(means_df.loc[remaining, "fed_funds_rate"].idxmax())
            else:
                restrictive = remaining[0]

    remaining = [i for i in range(model.n_components) if i not in {restrictive, stress}]

    # Pick an "Expansionary" state among the remaining ones:
    # prefer low vix and low policy rate (both z-scored features).
    if remaining:
        vix_col = "vix" if "vix" in means_df.columns else None
        r_col = "fed_funds_rate" if "fed_funds_rate" in means_df.columns else None
        if vix_col and r_col:
            score = (means_df.loc[remaining, vix_col] + means_df.loc[remaining, r_col]).astype(float)
            expansionary = int(score.idxmin())
        elif vix_col:
            expansionary = int(means_df.loc[remaining, vix_col].idxmin())
        elif r_col:
            expansionary = int(means_df.loc[remaining, r_col].idxmin())
        else:
            expansionary = int(remaining[0])
    else:
        expansionary = int((restrictive + 1) % model.n_components)

    labels: dict[int, str] = {
        restrictive: "Restrictive",
        stress: "Stress / Contraction",
        expansionary: "Expansionary",
    }
    # Label any leftover states as Neutral / Transitional.
    for s in range(model.n_components):
        if s not in labels:
            labels[int(s)] = "Neutral / Transitional"
    return labels


def decode_regimes(model: hmm.GaussianHMM, z_matrix: pd.DataFrame) -> pd.Series:
    """Decode most likely hidden state sequence for all available months."""
    X_df = z_matrix[REGIME_FEATURES].dropna()
    states = model.predict(X_df.values)
    return pd.Series(states, index=X_df.index, name="regime")


def regime_durations(regime_series: pd.Series) -> pd.DataFrame:
    """Run-length encode regime sequence; returns episodes and durations in months."""
    if regime_series.empty:
        return pd.DataFrame(columns=["state", "start", "end", "duration_months"])
    records: list[dict] = []
    current = int(regime_series.iloc[0])
    start = regime_series.index[0]
    for dt, state in regime_series.items():
        state_i = int(state)
        if state_i != current:
            records.append(
                {
                    "state": current,
                    "start": start,
                    "end": dt,
                    "duration_months": int((regime_series.loc[start:dt]).shape[0]),
                }
            )
            current = state_i
            start = dt
    records.append(
        {
            "state": current,
            "start": start,
            "end": regime_series.index[-1],
            "duration_months": int((regime_series.loc[start:]).shape[0]),
        }
    )
    return pd.DataFrame(records)


def _max_drawdown(cum: pd.Series) -> float:
    """Max peak-to-trough drawdown as negative percent."""
    peak = cum.cummax()
    dd = (cum - peak) / peak * 100
    return float(dd.min())


def regime_spy_stats(
    regime_series: pd.Series,
    spy_monthly_return_pct: pd.Series,
    state_labels: dict[int, str],
) -> dict[int, dict]:
    """Compute regime-conditional SPY stats per state."""
    durations = regime_durations(regime_series)
    stats_out: dict[int, dict] = {}

    for state in sorted(state_labels.keys()):
        in_regime_idx = regime_series[regime_series == state].index
        monthly_rets = spy_monthly_return_pct.reindex(in_regime_idx).dropna()

        episodes = durations[durations["state"] == state]
        fwd_12m: list[float] = []
        dds: list[float] = []
        for _, ep in episodes.iterrows():
            start = pd.Timestamp(ep["start"])
            end = pd.Timestamp(ep["end"])
            fwd_slice = spy_monthly_return_pct.loc[start:].iloc[:12].dropna()
            if len(fwd_slice) >= 6:
                cum = (1.0 + fwd_slice / 100.0).cumprod()
                fwd_12m.append(float((cum.iloc[-1] - 1.0) * 100.0))
            episode_idx = regime_series.loc[start:end].index
            ep_rets = spy_monthly_return_pct.reindex(episode_idx).dropna()
            if len(ep_rets) >= 2:
                ep_cum = (1.0 + ep_rets / 100.0).cumprod()
                dds.append(_max_drawdown(ep_cum))

        stats_out[int(state)] = {
            "label": state_labels[int(state)],
            "avg_monthly_return": float(monthly_rets.mean()) if len(monthly_rets) else 0.0,
            "avg_12m_forward_return": float(np.mean(fwd_12m)) if fwd_12m else 0.0,
            "worst_drawdown": float(min(dds)) if dds else 0.0,
            "avg_duration_months": float(episodes["duration_months"].mean()) if len(episodes) else 0.0,
            "n_episodes": int(len(episodes)),
        }

    return stats_out


def current_regime_summary(
    model: hmm.GaussianHMM,
    z_matrix: pd.DataFrame,
    regime_series: pd.Series,
    spy_stats: dict[int, dict],
    state_labels: dict[int, str],
) -> dict:
    """Summarize current regime and key stats for rendering and prompt context."""
    X_df = z_matrix[REGIME_FEATURES].dropna()
    if X_df.empty:
        raise RuntimeError("No observations available for regime decoding.")

    probs = model.predict_proba(X_df.values)
    current_state = int(regime_series.iloc[-1])
    prob = float(probs[-1, current_state])

    durations = regime_durations(regime_series)
    last_ep = durations[durations["state"] == current_state].iloc[-1]
    months_in_regime = int(last_ep["duration_months"])

    trans_probs = {
        state_labels[s]: float(model.transmat_[current_state, s])
        for s in range(model.n_components)
    }

    st = spy_stats.get(current_state, {})
    return {
        "state": current_state,
        "label": state_labels.get(current_state, str(current_state)),
        "probability": prob,
        "months_in_regime": months_in_regime,
        "avg_duration_months": st.get("avg_duration_months", 0.0),
        "avg_monthly_return": st.get("avg_monthly_return", 0.0),
        "avg_12m_forward_return": st.get("avg_12m_forward_return", 0.0),
        "worst_drawdown": st.get("worst_drawdown", 0.0),
        "n_episodes": st.get("n_episodes", 0),
        "transition_probs": trans_probs,
        "as_of": str(X_df.index[-1].date()),
    }


def regime_prob_timeseries(
    model: hmm.GaussianHMM,
    z_matrix: pd.DataFrame,
    state_labels: dict[int, str],
) -> dict:
    """Return Plotly-ready JSON for regime color-band timeline (full history)."""
    X_df = z_matrix[REGIME_FEATURES].dropna()
    if X_df.empty:
        return {}

    states = model.predict(X_df.values)
    regime_series = pd.Series(states, index=X_df.index)

    # Map state int → label and color (matches page palette: --green #4a7c59, --red #E3120B, --cream #f3f1ea)
    palette = {
        0: "#d4ede4",  # Expansionary — soft green
        1: "#f5dada",  # Stress / Contraction — soft red
        2: "#ebe7de",  # Neutral / Transitional — warm cream
        3: "#f5e9cf",  # Restrictive — soft amber
    }
    # Build episodes (consecutive runs of same state)
    episodes = []
    cur_state = int(regime_series.iloc[0])
    cur_start = regime_series.index[0]
    for dt, st in regime_series.items():
        st_i = int(st)
        if st_i != cur_state:
            episodes.append({"state": cur_state, "start": cur_start, "end": dt})
            cur_state = st_i
            cur_start = dt
    episodes.append({"state": cur_state, "start": cur_start, "end": regime_series.index[-1]})

    # Build shapes + text annotations (label inside each band, no separate legend)
    shapes = []
    annotations = []
    total_months = len(X_df)
    MIN_BAND_MONTHS = 18  # only label bands wide enough to fit text

    for ep in episodes:
        x0, x1 = ep["start"], ep["end"]
        duration = len(regime_series.loc[x0:x1])
        shapes.append({
            "type": "rect",
            "xref": "x", "yref": "paper",
            "x0": str(x0.date()), "x1": str(x1.date()),
            "y0": 0, "y1": 1,
            "fillcolor": palette.get(ep["state"], "#cccccc"),
            "opacity": 0.9,
            "line": {"width": 0},
            "layer": "below",
        })
        if duration >= MIN_BAND_MONTHS:
            mid = x0 + (x1 - x0) / 2
            annotations.append({
                "xref": "x", "yref": "paper",
                "x": str(mid.date()), "y": 0.5,
                "text": state_labels.get(ep["state"], ""),
                "showarrow": False,
                "font": {"size": 9, "color": "#5a5550", "family": "Inter, sans-serif"},
                "xanchor": "center", "yanchor": "middle",
            })

    # Hover trace only
    hover_labels = [state_labels.get(int(s), str(s)) for s in states]
    traces = [{
        "type": "scatter",
        "mode": "markers",
        "x": [str(d.date()) for d in X_df.index],
        "y": [0.5] * len(X_df),
        "marker": {"opacity": 0, "size": 8},
        "text": hover_labels,
        "hovertemplate": "%{x}  %{text}<extra></extra>",
        "showlegend": False,
    }]

    layout = {
        "margin": {"t": 8, "b": 24, "l": 10, "r": 10},
        "height": 72,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "shapes": shapes,
        "annotations": annotations,
        "showlegend": False,
        "xaxis": {"showgrid": False, "tickfont": {"size": 10}, "type": "date"},
        "yaxis": {"visible": False, "range": [0, 1]},
        "hovermode": "closest",
    }
    return {"traces": traces, "layout": layout}


def run_regime_analysis(spy_monthly_return_pct: pd.Series) -> dict:
    """End-to-end pipeline: load → train → decode → stats → summarize → persist."""
    _raw, z = load_feature_matrix()
    model = train_hmm(z, n_states=N_STATES)
    state_labels = label_states(model, REGIME_FEATURES)
    regime_series = decode_regimes(model, z)
    spy_stats = regime_spy_stats(regime_series, spy_monthly_return_pct, state_labels)
    summary = current_regime_summary(model, z, regime_series, spy_stats, state_labels)
    summary["prob_timeseries"] = regime_prob_timeseries(model, z, state_labels)

    out_path = PROCESSED_DIR / "regime_hmm_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _load_spy_monthly_return_pct_from_cache() -> pd.Series:
    spy_path = RAW_DIR / "spy.csv"
    if not spy_path.exists():
        raise FileNotFoundError("Missing SPY cache (data/raw/spy.csv). Run: python run.py")
    df = pd.read_csv(spy_path, parse_dates=["date"])
    s = df.set_index("date")["value"].sort_index()
    spy_m = s.resample("ME").last().pct_change() * 100.0
    return spy_m.dropna()


def main() -> None:
    spy_m = _load_spy_monthly_return_pct_from_cache()
    summary = run_regime_analysis(spy_m)
    print("\n=== Current Regime Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

