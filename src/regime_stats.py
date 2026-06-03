"""Stage 3b: HMM state-based regime persistence statistics."""

from __future__ import annotations

import sys

import pandas as pd

from src.collect import load_all_from_cache
from src.config import PROCESSED_DIR
from src.history_match import build_feature_matrix
from src.regime_hmm import (
    decode_regimes,
    regime_durations,
    train_hmm,
)


def compute_regime_stats(
    feature_matrix: pd.DataFrame,
    regime_summary: dict | None = None,
) -> dict:
    """
    Summarize regime persistence from HMM state sequence.

    Uses:
    - current state's month count and average duration from regime_hmm_summary
    - historical episodes from decoded regime series (same state as current)
    """
    if feature_matrix.empty or len(feature_matrix) < 12:
        return {"available": False, "message": "Feature matrix too short."}

    if regime_summary is None:
        path = PROCESSED_DIR / "regime_hmm_summary.json"
        if path.exists():
            import json

            regime_summary = json.loads(path.read_text())

    if not regime_summary:
        return {"available": False, "message": "Missing HMM regime summary."}

    current_state = int(regime_summary["state"])
    model = train_hmm(feature_matrix, n_states=4)
    reg_series = decode_regimes(model, feature_matrix)
    durations = regime_durations(reg_series)

    state_eps = durations[durations["state"] == current_state].copy()
    if state_eps.empty:
        return {"available": False, "message": "No historical episodes for current state."}

    latest_dt = reg_series.index[-1]
    completed_eps = state_eps[pd.to_datetime(state_eps["end"]) < latest_dt].copy()
    longest = None
    longest_note = None
    if len(completed_eps) >= 2:
        longest_idx = int(completed_eps["duration_months"].astype(int).idxmax())
        longest = completed_eps.loc[longest_idx]
    elif len(completed_eps) == 1:
        longest_note = f"single prior completed episode: {int(completed_eps.iloc[0]['duration_months'])}m"
    else:
        longest_note = "N/A"

    # Optional: keep current-state drawdown metric from summary as descriptive risk context.
    worst_dd = regime_summary.get("worst_drawdown")
    prior_completed_episodes = int(len(completed_eps))
    if prior_completed_episodes >= 2:
        avg_prior_duration = float(completed_eps["duration_months"].mean())
    else:
        avg_prior_duration = None

    prior_single_duration = (
        int(completed_eps.iloc[0]["duration_months"])
        if prior_completed_episodes == 1
        else None
    )

    return {
        "available": True,
        "regime_label": regime_summary.get("label", "N/A"),
        "current_state": current_state,
        "current_regime_months": int(regime_summary.get("months_in_regime", 0)),
        "avg_duration_months": float(regime_summary.get("avg_duration_months", state_eps["duration_months"].mean())),
        "total_episodes": int(len(state_eps)),
        "prior_completed_episodes": prior_completed_episodes,
        "avg_prior_duration_months": avg_prior_duration,
        "prior_single_duration_months": prior_single_duration,
        "worst_drawdown_pct": worst_dd,
        "longest_similar_episode": (
            {
                "start": str(pd.Timestamp(longest["start"]).date())[:7],
                "end": str(pd.Timestamp(longest["end"]).date())[:7],
                "months": int(longest["duration_months"]),
            }
            if longest is not None
            else None
        ),
        "longest_similar_episode_note": longest_note,
    }


def save_regime_stats(stats: dict) -> None:
    path = PROCESSED_DIR / "regime_stats.json"
    import json

    path.write_text(json.dumps(stats, indent=2))


def main() -> None:
    data = load_all_from_cache()
    matrix = build_feature_matrix(data)
    stats = compute_regime_stats(matrix)
    save_regime_stats(stats)
    if not stats.get("available"):
        print(stats.get("message"), file=sys.stderr)
        sys.exit(1)
    print(f"Current regime ({stats['regime_label']}): {stats['current_regime_months']} months")
    print(f"Avg duration: {stats['avg_duration_months']:.1f} months | episodes: {stats['total_episodes']}")


if __name__ == "__main__":
    main()
