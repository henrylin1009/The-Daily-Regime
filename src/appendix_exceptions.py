"""Extract anomaly scan rows (RED signal or |Z|>2 only) for the appendix."""

from __future__ import annotations

import re
from typing import Any

COUNTRY_CYCLE_KEYS = ["US", "Japan", "Europe", "China", "Taiwan"]


def _parse_z(value: object) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace("—", "").replace("N/A", "")
    if not s:
        return None
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _is_red_signal(signal: object) -> bool:
    return str(signal or "").strip().upper() == "RED"


def _qualifies(signal: object, z: float | None) -> bool:
    return _is_red_signal(signal) or (z is not None and abs(z) > 2.0)


def _append(
    out: list[dict[str, str]],
    *,
    drawer: str,
    source: str,
    label: str,
    detail: str,
    signal: str,
    z: float | None = None,
) -> None:
    sig = str(signal or "—").strip().upper() or "—"
    if not _qualifies(sig if sig != "—" else "", z):
        return
    reasons: list[str] = []
    if _is_red_signal(sig):
        reasons.append("RED")
    if z is not None and abs(z) > 2.0:
        reasons.append(f"|Z|={abs(z):.2f}")
    out.append(
        {
            "drawer": drawer,
            "section": drawer,
            "source": source,
            "label": label,
            "detail": detail,
            "signal": sig if sig != "—" else "—",
            "z_display": f"{z:+.2f}" if z is not None else "—",
            "reason": " · ".join(reasons),
        }
    )


def build_exception_dashboard(
    l1: dict[str, Any],
    l2: dict[str, Any],
    l3: dict[str, Any],
    country_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Return sorted anomaly rows: RED signal or |Z| > 2.0 only."""
    out: list[dict[str, str]] = []
    country_rows = country_rows or []

    for r in l1.get("cot_rows") or []:
        if not isinstance(r, dict):
            continue
        z = _parse_z(r.get("zscore_fmt"))
        _append(
            out,
            drawer="Sentiment & Stress",
            source="CFTC",
            label=str(r.get("contract") or "—"),
            detail=f"Net {r.get('net_fmt', '—')} · 4w {r.get('chg4w_fmt', '—')}",
            signal="—",
            z=z,
        )

    for r in country_rows:
        if not isinstance(r, dict):
            continue
        if _is_red_signal(r.get("signal")):
            _append(
                out,
                drawer="Tactical Tape",
                source="FX Momentum",
                label=str(r.get("country") or "—"),
                detail=f"FX 1m {r.get('fx_1m', '—')} · spread {r.get('spread_10y', '—')}",
                signal="RED",
            )

    if not l3.get("l3_available"):
        return _sort_exceptions(out)

    eod = l3.get("event_of_day") if isinstance(l3.get("event_of_day"), dict) else {}
    _append(
        out,
        drawer="Tactical Tape",
        source="Event of Day",
        label=str(eod.get("name") or "—"),
        detail=f"Latest {eod.get('latest', '—')} · Δ {eod.get('change', '—')}",
        signal=str(eod.get("signal") or "—"),
        z=_parse_z(eod.get("z")),
    )

    for panel_name, rows in (
        ("Surfaced", l3.get("surfaced_rows") or []),
        ("Flow Validation", l3.get("validation_panel") or []),
    ):
        for r in rows:
            if not isinstance(r, dict):
                continue
            _append(
                out,
                drawer="Tactical Tape",
                source=panel_name,
                label=str(r.get("name") or "—"),
                detail=f"Latest {r.get('latest', '—')} · Δ {r.get('change', r.get('chg_1m', '—'))}",
                signal=str(r.get("signal") or "—"),
                z=_parse_z(r.get("z")),
            )

    momentum = l3.get("quant_momentum") if isinstance(l3.get("quant_momentum"), dict) else {}
    for k, v in momentum.items():
        if not isinstance(v, dict):
            continue
        _append(
            out,
            drawer="Sentiment & Stress",
            source="Vol / Momentum",
            label=str(k),
            detail=str(v.get("label") or "—"),
            signal="—",
            z=_parse_z(v.get("z_latest")),
        )

    for r in l3.get("vol_skew_rows") or []:
        if not isinstance(r, dict):
            continue
        _append(
            out,
            drawer="Sentiment & Stress",
            source="Vol / Skew",
            label=str(r.get("name") or "—"),
            detail=f"Level {r.get('level', '—')} · {r.get('label', '—')}",
            signal=str(r.get("signal") or "—"),
            z=_parse_z(r.get("z")),
        )

    return _sort_exceptions(out)


def _sort_exceptions(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def _key(row: dict[str, str]) -> tuple[int, float]:
        sig_rank = 0 if row.get("signal") == "RED" else 1
        z = _parse_z(row.get("z_display")) or 0.0
        return (sig_rank, -abs(z))

    return sorted(rows, key=_key)


def anomaly_label_set(rows: list[dict[str, str]]) -> frozenset[str]:
    return frozenset(str(r.get("label") or "") for r in rows if r.get("label"))


def mark_anomaly_rows(rows: list[dict], label_key: str, labels: frozenset[str]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        item = dict(r)
        item["anomaly"] = str(item.get(label_key) or "") in labels
        out.append(item)
    return out
