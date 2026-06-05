"""Sector Rotation: RRG computation with trajectory arrows and historical stats."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

SECTORS: dict[str, str] = {
    "xlk": "Tech",
    "xlf": "Financials",
    "xlv": "Healthcare",
    "xle": "Energy",
    "xlu": "Utilities",
    "xli": "Industrials",
}

_COLORS: dict[str, str] = {
    "xlk": "#4f46e5",
    "xlf": "#0891b2",
    "xlv": "#059669",
    "xle": "#d97706",
    "xlu": "#9333ea",
    "xli": "#dc2626",
}

_BENCH_COLOR = "#6b7280"

_Q_LABELS_EN = {
    "leading": "Leading", "weakening": "Weakening",
    "improving": "Improving", "lagging": "Lagging",
}
_Q_LABELS_ZH = {
    "leading": "Leading 領漲", "weakening": "Weakening 轉弱",
    "improving": "Improving 轉強", "lagging": "Lagging 落後",
}

_SMOOTH = 10   # weeks for RS smoothing
_NORM = 52     # weeks for z-score normalisation window
_TRAIL = 12    # weeks of tail to draw
_ARROW_WEEKS = 3  # weeks back to compute trajectory vector
_RRG_R_MIN = 2.8   # minimum half-axis span (σ)
_RRG_R_PAD = 0.18  # padding fraction on data extent
_RRG_LABEL_PAD = 0.45  # extra σ for marker + text above points
_RET_1W = 5
_RET_1M = 21
_RET_1Y = 252


@dataclass(frozen=True)
class SectorUniverse:
    benchmark_key: str
    benchmark_label: str
    sectors: dict[str, str]
    colors: dict[str, str]
    benchmark_color: str = _BENCH_COLOR
    sectors_zh: dict[str, str] | None = None


US_SECTORS = SectorUniverse(
    benchmark_key="spy",
    benchmark_label="SPY",
    sectors=SECTORS,
    colors=_COLORS,
)

TW_SECTORS = SectorUniverse(
    benchmark_key="tw_bench",
    benchmark_label="TAIEX",
    sectors={
        "tw_semi": "Semiconductor ETF (00892)",
        "tw_comp": "Electronics ETF (0053)",
        "tw_fin": "Financial ETF (0055)",
        "tw_ship": "High Dividend ETF (00919)",
        "tw_mach": "Blue Chip 30 (00690)",
    },
    sectors_zh={
        "tw_semi": "半導體 ETF (00892)",
        "tw_comp": "電子 ETF (0053)",
        "tw_fin": "金融 ETF (0055)",
        "tw_ship": "高息 ETF (00919)",
        "tw_mach": "藍籌 30 (00690)",
    },
    colors={
        "tw_semi": "#4f46e5",
        "tw_comp": "#0891b2",
        "tw_fin": "#059669",
        "tw_ship": "#d97706",
        "tw_mach": "#dc2626",
    },
)


def _rrg_axis_limit(coords: list[tuple[float, float]]) -> float:
    """Symmetric axis half-range from trail + arrow tips, with padding."""
    if not coords:
        return _RRG_R_MIN
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    extent = max(max(abs(v) for v in xs), max(abs(v) for v in ys))
    padded = extent * (1.0 + _RRG_R_PAD) + _RRG_LABEL_PAD
    if padded <= _RRG_R_MIN:
        return _RRG_R_MIN
    return math.ceil(padded * 2) / 2


def _rrg_axis_ticks(r: float) -> list[int]:
    n = max(1, int(math.floor(r)))
    return list(range(-n, n + 1))


def _pct_return(series: pd.Series, periods: int) -> float | None:
    s = series.dropna()
    if len(s) <= periods:
        return None
    prev = float(s.iloc[-1 - periods])
    if prev == 0:
        return None
    return (float(s.iloc[-1]) / prev - 1.0) * 100.0


def _build_sector_performance(data: dict, *, universe: SectorUniverse = US_SECTORS) -> list[dict]:
    """Benchmark row + sector rows sorted by 1W excess vs benchmark."""
    bench_s = _series(data, universe.benchmark_key)
    if bench_s.empty:
        return []

    bench_rets = {
        "1w": _pct_return(bench_s, _RET_1W),
        "1m": _pct_return(bench_s, _RET_1M),
        "1y": _pct_return(bench_s, _RET_1Y),
    }

    def _rnd(v: float | None) -> float | None:
        return None if v is None else round(v, 1)

    rows: list[dict] = [
        {
            "label": universe.benchmark_label,
            "color": universe.benchmark_color,
            "is_benchmark": True,
            "ret_1w": _rnd(bench_rets["1w"]),
            "ret_1m": _rnd(bench_rets["1m"]),
            "ret_1y": _rnd(bench_rets["1y"]),
            "excess_1w": None,
            "excess_1m": None,
            "excess_1y": None,
            "sort_key": 10_000.0,
        }
    ]

    for ticker, label in universe.sectors.items():
        s = _series(data, ticker)
        if s.empty:
            continue
        r1w = _pct_return(s, _RET_1W)
        r1m = _pct_return(s, _RET_1M)
        r1y = _pct_return(s, _RET_1Y)
        ex1w = None if r1w is None or bench_rets["1w"] is None else round(r1w - bench_rets["1w"], 1)
        ex1m = None if r1m is None or bench_rets["1m"] is None else round(r1m - bench_rets["1m"], 1)
        ex1y = None if r1y is None or bench_rets["1y"] is None else round(r1y - bench_rets["1y"], 1)
        row = {
            "label": label,
            "color": universe.colors[ticker],
            "is_benchmark": False,
            "ret_1w": None if r1w is None else round(r1w, 1),
            "ret_1m": None if r1m is None else round(r1m, 1),
            "ret_1y": None if r1y is None else round(r1y, 1),
            "excess_1w": ex1w,
            "excess_1m": ex1m,
            "excess_1y": ex1y,
            "sort_key": ex1w if ex1w is not None else -999.0,
        }
        if universe.sectors_zh and ticker in universe.sectors_zh:
            row["label_zh"] = universe.sectors_zh[ticker]
        rows.append(row)

    bench = rows[0]
    sectors = sorted(rows[1:], key=lambda r: r["sort_key"], reverse=True)
    return [bench] + sectors


def _series(data: dict, key: str) -> pd.Series:
    df = data.get(key)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].sort_index()


def _quadrant(rx: float, ry: float) -> str:
    if rx >= 0 and ry >= 0:
        return "leading"
    if rx < 0 and ry >= 0:
        return "weakening"
    if rx >= 0 and ry < 0:
        return "improving"
    return "lagging"


def _compute_rrg_series(bench_w: pd.Series, sec_w: pd.Series) -> pd.DataFrame | None:
    """Return full weekly RS-Ratio / RS-Momentum history, or None if insufficient data."""
    base = pd.concat([bench_w.rename("bench"), sec_w.rename("sec")], axis=1).dropna()
    if len(base) < _NORM + _SMOOTH + 5:
        return None
    rs = np.log(base["sec"] / base["bench"])
    rs_sm = rs.rolling(_SMOOTH).mean()
    rs_ratio = (rs_sm - rs_sm.rolling(_NORM).mean()) / rs_sm.rolling(_NORM).std()
    mom_raw = rs_ratio.diff(1).rolling(_SMOOTH).mean()
    rs_mom = (mom_raw - mom_raw.rolling(_NORM).mean()) / mom_raw.rolling(_NORM).std()
    return pd.concat([rs_ratio.rename("r"), rs_mom.rename("m")], axis=1).dropna()


def _historical_stats(full: pd.DataFrame) -> dict:
    """
    From full RS-Ratio/RS-Momentum history compute week-by-week transition stats:
    - stay_pct: % of weeks that stay in same quadrant (like macro regime block)
    - exits: distribution of transitions given it leaves (sums to 100%)
    - avg stay per quadrant (weeks, from run-length)
    - current streak length
    """
    qs = full.apply(lambda row: _quadrant(row["r"], row["m"]), axis=1)

    ww_stay: dict[str, int] = defaultdict(int)
    ww_exit: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    q_list = qs.tolist()
    for i in range(len(q_list) - 1):
        cur, nxt = q_list[i], q_list[i + 1]
        if cur == nxt:
            ww_stay[cur] += 1
        else:
            ww_exit[cur][nxt] += 1

    stay_pct: dict[str, int] = {}
    exits: dict[str, list[dict]] = {}
    for q in set(q_list):
        stay_n = ww_stay.get(q, 0)
        exit_n = sum(ww_exit[q].values())
        total = stay_n + exit_n
        if total == 0:
            continue
        stay_pct[q] = round(stay_n / total * 100)
        if exit_n > 0:
            exits[q] = sorted(
                [{"q": nq, "pct": round(cnt / exit_n * 100)} for nq, cnt in ww_exit[q].items()],
                key=lambda x: -x["pct"],
            )[:3]

    runs: list[dict] = []
    i = 0
    while i < len(q_list):
        q = q_list[i]
        j = i
        while j < len(q_list) and q_list[j] == q:
            j += 1
        runs.append({"q": q, "len": j - i})
        i = j

    last_q = q_list[-1]
    streak = 1
    for q in reversed(q_list[:-1]):
        if q == last_q:
            streak += 1
        else:
            break

    q_durations: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        q_durations[run["q"]].append(run["len"])

    avg_stay: dict[str, float] = {
        q: round(sum(v) / len(v), 1) for q, v in q_durations.items() if v
    }

    trans_pct = exits

    return {
        "streak": streak,
        "avg_stay": avg_stay,
        "trans_pct": trans_pct,
        "stay_pct": stay_pct,
        "exits": exits,
        "current_q": last_q,
    }


def _build_summary(positions: dict) -> dict:
    """One-line headline of current RRG state."""
    leading   = sorted([p for p in positions.values() if p["quadrant"] == "leading"],   key=lambda p: -p["rs_ratio"])
    weakening = [p for p in positions.values() if p["quadrant"] == "weakening"]
    improving = [p for p in positions.values() if p["quadrant"] == "improving"]
    lagging   = [p for p in positions.values() if p["quadrant"] == "lagging"]

    def _join(lst: list[str]) -> str:
        if not lst:
            return ""
        if len(lst) == 1:
            return lst[0]
        return ", ".join(lst[:-1]) + " & " + lst[-1]

    parts_en, parts_zh = [], []
    if leading:
        parts_en.append(f"{_join([p['label'] for p in leading])} leads")
        parts_zh.append(f"{_join([p['label'] for p in leading])} 領漲")
    if weakening:
        parts_en.append(f"{_join([p['label'] for p in weakening])} fading")
        parts_zh.append(f"{_join([p['label'] for p in weakening])} 動能轉弱")
    if improving:
        parts_en.append(f"{_join([p['label'] for p in improving])} rotating in")
        parts_zh.append(f"{_join([p['label'] for p in improving])} 資金開始流入")
    if lagging:
        parts_en.append(f"{_join([p['label'] for p in lagging])} lags")
        parts_zh.append(f"{_join([p['label'] for p in lagging])} 落後")

    return {
        "en": ", ".join(parts_en) + "." if parts_en else "",
        "zh": "，".join(parts_zh) + "。" if parts_zh else "",
    }


_Q_RANK_MULT = {"leading": 3, "improving": 2, "weakening": 1, "lagging": 0}


def _rank_score(rx: float, ry: float, q: str) -> float:
    return _Q_RANK_MULT[q] * 10 + rx + ry


def _weekly_rank_score_leaders(all_full: dict[str, pd.DataFrame]) -> pd.Series:
    """Per week: ticker with highest rank_score (same rule as CURRENT LEADER headline)."""
    cols: dict[str, pd.Series] = {}
    for ticker, full in all_full.items():
        qs = full.apply(lambda row: _quadrant(row["r"], row["m"]), axis=1)
        cols[ticker] = qs.map(_Q_RANK_MULT) * 10 + full["r"] + full["m"]
    mat = pd.concat(cols, axis=1).dropna()
    if len(mat) < 2:
        return pd.Series(dtype=object)
    return mat.idxmax(axis=1)


def _avg_weeks_when_rank1(leader_series: pd.Series, ticker: str) -> float | None:
    """Mean run length (weeks) when *ticker* was rank_score #1."""
    runs: list[int] = []
    vals = leader_series.tolist()
    i = 0
    while i < len(vals):
        if vals[i] == ticker:
            j = i
            while j < len(vals) and vals[j] == ticker:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    if not runs:
        return None
    return round(sum(runs) / len(runs), 1)


def _attach_avg_weeks_as_rank1(
    sector_stats: list[dict],
    all_full: dict[str, pd.DataFrame],
    universe: SectorUniverse,
) -> None:
    leader_series = _weekly_rank_score_leaders(all_full)
    if leader_series.empty:
        return
    label_to_ticker = {label: t for t, label in universe.sectors.items()}
    for row in sector_stats:
        ticker = label_to_ticker.get(row.get("label", ""))
        if not ticker:
            continue
        avg = _avg_weeks_when_rank1(leader_series, ticker)
        if avg is not None:
            row["avg_weeks_as_rank1"] = avg


def _rank_transitions(all_full: dict[str, pd.DataFrame], universe: SectorUniverse) -> dict:
    """
    Compute historical #1 rank transitions across all sectors.
    Returns for the current #1: stay_pct + who takes over (with %).
    """
    score_frames = {
        ticker: (df["r"] + df["m"]).rename(ticker)
        for ticker, df in all_full.items()
    }
    scores = pd.concat(score_frames.values(), axis=1).dropna()
    if len(scores) < 20:
        return {}

    leader_series = scores.idxmax(axis=1)
    current_leader = leader_series.iloc[-1]

    stay = 0
    successors: dict[str, int] = defaultdict(int)
    leader_list = leader_series.tolist()
    for i in range(len(leader_list) - 1):
        if leader_list[i] == current_leader:
            if leader_list[i + 1] == current_leader:
                stay += 1
            else:
                successors[leader_list[i + 1]] += 1

    total = stay + sum(successors.values())
    if total == 0:
        return {}

    stay_pct = round(stay / total * 100)
    exits = []
    for t, cnt in successors.items():
        if t not in universe.sectors:
            continue
        exit_row: dict = {
            "ticker": t,
            "label": universe.sectors[t],
            "pct": round(cnt / (total - stay) * 100),
        }
        if universe.sectors_zh and t in universe.sectors_zh:
            exit_row["label_zh"] = universe.sectors_zh[t]
        exits.append(exit_row)
    exits = sorted(exits, key=lambda x: -x["pct"])[:3]

    return {
        "current_leader": current_leader,
        "stay_pct": stay_pct,
        "exits": exits,
    }


def build_rrg_plotly(data: dict, *, universe: SectorUniverse = US_SECTORS) -> dict:
    """Return Plotly JSON + summary + historical stats table, or {} if unavailable."""
    bench = _series(data, universe.benchmark_key)
    if bench.empty:
        return {}

    bench_w = bench.resample("W-FRI").last().dropna()
    traces: list[dict] = []
    annotations: list[dict] = []
    extent_coords: list[tuple[float, float]] = []
    positions: dict[str, dict] = {}
    sector_stats: list[dict] = []
    all_full: dict[str, pd.DataFrame] = {}

    for ticker, label in universe.sectors.items():
        s = _series(data, ticker)
        if s.empty:
            continue
        s_w = s.resample("W-FRI").last().dropna()

        full = _compute_rrg_series(bench_w, s_w)
        if full is None:
            continue

        all_full[ticker] = full
        trail = full.tail(_TRAIL)
        x = trail["r"].tolist()
        y = trail["m"].tolist()
        dates = [d.strftime("%Y-%m-%d") for d in trail.index]
        col = universe.colors[ticker]
        extent_coords.extend(zip(x, y))

        traces.append({
            "type": "scatter", "x": x, "y": y, "mode": "lines",
            "line": {"color": col, "width": 1.5, "dash": "dot"},
            "hoverinfo": "skip", "showlegend": False, "legendgroup": ticker,
        })
        traces.append({
            "type": "scatter", "x": x[:-1], "y": y[:-1], "mode": "markers",
            "marker": {"size": 4, "color": col, "opacity": 0.35},
            "text": dates[:-1],
            "hovertemplate": (
                f"<b>{label}</b><br>%{{text}}<br>"
                "RS-Ratio: %{x:.2f}σ<br>RS-Mom: %{y:.2f}σ<extra></extra>"
            ),
            "showlegend": False, "legendgroup": ticker,
        })
        traces.append({
            "type": "scatter", "x": [x[-1]], "y": [y[-1]],
            "mode": "markers+text",
            "marker": {"size": 12, "color": col, "line": {"color": "white", "width": 1.5}},
            "text": [label], "textposition": "top center",
            "textfont": {"size": 10, "color": col, "family": "Inter, sans-serif"},
            "hovertemplate": (
                f"<b>{label}</b><br>{dates[-1]}<br>"
                "RS-Ratio: %{x:.2f}σ<br>RS-Mom: %{y:.2f}σ<extra></extra>"
            ),
            "name": label, "showlegend": True, "legendgroup": ticker,
        })

        if len(x) >= _ARROW_WEEKS + 1:
            dx = x[-1] - x[-(_ARROW_WEEKS + 1)]
            dy = y[-1] - y[-(_ARROW_WEEKS + 1)]
            mag = (dx ** 2 + dy ** 2) ** 0.5
            if mag > 0.01:
                scale = 0.45 / mag
                tip_x = x[-1] + dx * scale
                tip_y = y[-1] + dy * scale
                extent_coords.append((tip_x, tip_y))
                annotations.append({
                    "x": tip_x,
                    "y": tip_y,
                    "ax": x[-1], "ay": y[-1],
                    "xref": "x", "yref": "y",
                    "axref": "x", "ayref": "y",
                    "showarrow": True, "arrowhead": 2,
                    "arrowsize": 1.2, "arrowwidth": 2,
                    "arrowcolor": col, "text": "",
                })

        rx, ry = x[-1], y[-1]
        q = _quadrant(rx, ry)
        positions[ticker] = {"label": label, "quadrant": q, "rs_ratio": rx, "rs_mom": ry, "color": col}

        stats = _historical_stats(full)
        avg = stats["avg_stay"].get(q)
        stay = stats["stay_pct"].get(q)
        exits = stats["exits"].get(q, [])
        rank_score = _rank_score(rx, ry, q)
        stat_row = {
            "label": label,
            "color": col,
            "quadrant_en": _Q_LABELS_EN[q],
            "quadrant_zh": _Q_LABELS_ZH[q],
            "weeks_current": stats["streak"],
            "avg_stay": avg,
            "stay_pct": stay,
            "exits_en": [{"label": _Q_LABELS_EN[t["q"]], "pct": t["pct"]} for t in exits],
            "exits_zh": [{"label": _Q_LABELS_ZH[t["q"]], "pct": t["pct"]} for t in exits],
            "rank_score": rank_score,
        }
        if universe.sectors_zh and ticker in universe.sectors_zh:
            stat_row["label_zh"] = universe.sectors_zh[ticker]
        sector_stats.append(stat_row)

    if not traces:
        return {}

    _attach_avg_weeks_as_rank1(sector_stats, all_full, universe)
    rank_trans = _rank_transitions(all_full, universe)
    summary = _build_summary(positions)

    R = _rrg_axis_limit(extent_coords)
    ticks = _rrg_axis_ticks(R)
    layout = {
        "height": 440,
        "autosize": True,
        "margin": {"l": 55, "r": 48, "t": 36, "b": 50},
        "paper_bgcolor": "#ffffff", "plot_bgcolor": "#ffffff",
        "font": {"family": "Inter, sans-serif", "size": 11},
        "dragmode": False,
        "xaxis": {
            "title": "RS-Ratio (σ)", "range": [-R, R], "fixedrange": True,
            "zeroline": True, "zerolinecolor": "#333", "zerolinewidth": 1.5,
            "gridcolor": "#F0F0F0", "tickvals": ticks,
        },
        "yaxis": {
            "title": "RS-Momentum (σ)", "range": [-R, R], "fixedrange": True,
            "zeroline": True, "zerolinecolor": "#333", "zerolinewidth": 1.5,
            "gridcolor": "#F0F0F0", "tickvals": ticks,
        },
        "showlegend": False,
        "shapes": [
            {"type": "rect", "x0": 0,  "x1": R,  "y0": 0,  "y1": R,
             "fillcolor": "#d4ede4", "opacity": 0.5, "line": {"width": 0}},
            {"type": "rect", "x0": -R, "x1": 0,  "y0": 0,  "y1": R,
             "fillcolor": "#f5e9cf", "opacity": 0.5, "line": {"width": 0}},
            {"type": "rect", "x0": 0,  "x1": R,  "y0": -R, "y1": 0,
             "fillcolor": "#dbeafe", "opacity": 0.4, "line": {"width": 0}},
            {"type": "rect", "x0": -R, "x1": 0,  "y0": -R, "y1": 0,
             "fillcolor": "#f5dada", "opacity": 0.4, "line": {"width": 0}},
        ],
        "annotations": [
            {"x":  R*0.65, "y":  R*0.92, "text": "Leading",   "showarrow": False,
             "font": {"size": 11, "color": "#047857", "family": "Inter"}},
            {"x": -R*0.65, "y":  R*0.92, "text": "Weakening", "showarrow": False,
             "font": {"size": 11, "color": "#92400e", "family": "Inter"}},
            {"x":  R*0.65, "y": -R*0.92, "text": "Improving", "showarrow": False,
             "font": {"size": 11, "color": "#1d4ed8", "family": "Inter"}},
            {"x": -R*0.65, "y": -R*0.92, "text": "Lagging",   "showarrow": False,
             "font": {"size": 11, "color": "#991b1b", "family": "Inter"}},
        ] + annotations,
    }
    return {
        "traces": traces,
        "layout": layout,
        "summary": summary,
        "sector_stats": sector_stats,
        "sector_performance": _build_sector_performance(data, universe=universe),
        "rank_trans": rank_trans,
    }


def build_tw_rrg_plotly(data: dict) -> dict:
    """Taiwan listed-ETF RRG vs TAIEX total return."""
    return build_rrg_plotly(data, universe=TW_SECTORS)
