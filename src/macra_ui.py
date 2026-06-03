"""MACRA TIMES 3.0 — view-model helpers for ranked accordion UI and L3 matrix."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from src.config import OUTPUT_DIR

FALLBACK = "—"
NA = "N/A"
COUNTRY_ORDER = ["US", "Japan", "Europe", "China", "Taiwan"]
COUNTRY_IDS = {
    "US": "US",
    "Japan": "JP",
    "Europe": "EU",
    "China": "CN",
    "Taiwan": "TW",
}
TAB_LABELS = {c: COUNTRY_IDS[c] for c in COUNTRY_ORDER}

_SIGNAL_SCORE = {"RED": 3, "YELLOW": 2, "GREEN": 1}
_POLICY_SCORE = {"hiking": 3, "on hold": 2, "hold": 2, "cutting": 1, "easing": 1}
_CYCLE_BOOST = {"Slowdown", "Contraction"}


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"none", "nan", "null", "n/a"}


def _fmt_num(value: object, ndigits: int = 2, suffix: str = "") -> str:
    if _is_missing(value):
        return FALLBACK
    try:
        return f"{float(value):.{ndigits}f}{suffix}"
    except (TypeError, ValueError):
        s = str(value).strip()
        return s if s else FALLBACK


def _fmt_rate_pct(value: object) -> str:
    if _is_missing(value):
        return FALLBACK
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return FALLBACK


def _fmt_mom(value: object) -> str:
    return _fmt_num(value, ndigits=2, suffix="")


def _signal_score(signal: str) -> int:
    return _SIGNAL_SCORE.get(str(signal or "").strip().upper(), 2)


def _policy_score(direction: str) -> int:
    d = str(direction or "").strip().lower()
    for key, val in _POLICY_SCORE.items():
        if key in d:
            return val
    return 2


def _spread_abs_bp(spread_10y: str | float | None) -> float:
    if isinstance(spread_10y, (int, float)):
        return abs(float(spread_10y))
    s = str(spread_10y or "").replace("bp", "").replace("—", "").replace(NA, "").strip()
    try:
        return abs(float(s))
    except ValueError:
        return 0.0


def build_country_rank_rows(countries_data: list[dict], layer: str = "L2") -> list[dict]:
    """
    Sort countries Rank 1 (most stress) → Rank 6 (least stress).
    L2 tie-break: abs(spread_10y) desc, then country name asc.
    """
    rows: list[dict] = []
    for raw in countries_data:
        country = str(raw.get("country") or raw.get("name") or raw.get("id") or "").strip()
        if not country:
            continue
        row = dict(raw)
        row["country"] = country
        row["name"] = country
        row["id"] = COUNTRY_IDS.get(country, country[:2].upper())
        row["is_us_core"] = country == "US"
        row["signal_color"] = str(row.get("signal") or "YELLOW").upper()

        spread_abs = row.get("spread_bp")
        if spread_abs is None:
            spread_abs = _spread_abs_bp(row.get("spread_10y"))
        row["spread_bp"] = spread_abs

        if layer.upper() == "L1":
            direction = raw.get("direction") or raw.get("policy_direction") or "on hold"
            cycle = str(raw.get("country_cycle") or "")
            stress = float(_policy_score(direction))
            if cycle in _CYCLE_BOOST:
                stress += 0.5
            try:
                mom = abs(float(raw.get("mom_val") if not _is_missing(raw.get("mom_val")) else 0))
            except (TypeError, ValueError):
                mom = 0.0
            row["_stress"] = stress + mom * 0.01
            row["_spread_abs"] = 0.0
        else:
            row["_stress"] = float(_signal_score(str(raw.get("signal", "YELLOW"))))
            row["_spread_abs"] = float(spread_abs)

        row["score"] = row["_stress"] + row["_spread_abs"] * 0.01
        rows.append(row)

    rows.sort(key=lambda r: (-r["_stress"], -r["_spread_abs"], r["country"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def format_accordion_summary(row: dict, rank_index: int, global_divergence: int | float) -> dict:
    country = str(row.get("country") or "")
    is_core = bool(row.get("is_us_core") or country == "US")
    signal = str(row.get("signal") or "YELLOW")

    # Per-country context hint shown next to country name
    spread_bp = row.get("spread_bp")
    fx = row.get("fx_1m")
    _spread_val = None
    if spread_bp is not None and not _is_missing(spread_bp):
        try:
            _spread_val = float(spread_bp)
        except (TypeError, ValueError):
            pass
    if _spread_val is not None and _spread_val != 0.0:
        hint = f"spread {_spread_val:+.0f}bp"
    elif fx is not None and not _is_missing(fx):
        try:
            hint = f"FX {float(fx):+.2f}%"
        except (TypeError, ValueError):
            hint = ""
    else:
        hint = {"RED": "高風險", "YELLOW": "留意觀察", "GREEN": "穩定"}.get(signal, "")

    return {
        "rank_label": f"[ {rank_index} ]",
        "country_label": country,
        "signal": signal,
        "is_core": is_core,
        "divergence_text": f"({hint})" if hint else "",
        "core_badge_html": "[Global Core]" if is_core else "",
    }


def spread_trade_line(ranked_rows: list[dict]) -> str:
    if len(ranked_rows) < 2:
        return "目前國家覆蓋不足，無法產生動能排名。"
    # Rank 1 = highest stress/risk (most attention needed), last = strongest momentum
    weakest = ranked_rows[0].get("country", FALLBACK)
    strongest = ranked_rows[-1].get("country", FALLBACK)
    return f"動能最強：{strongest}　　動能最弱（最需注意）：{weakest}"


def global_divergence_from_quant(quant_payload: dict | None) -> int:
    if not isinstance(quant_payload, dict):
        return 0
    ra = quant_payload.get("Risk_Analytics")
    if isinstance(ra, dict) and ra.get("divergence_score") is not None:
        return max(0, min(100, int(ra["divergence_score"])))
    return 0


def build_l1_gear_columns(country: str, gr_payload: dict | None) -> dict[str, str]:
    gr = gr_payload if isinstance(gr_payload, dict) else {}
    llm = gr.get("llm") if isinstance(gr.get("llm"), dict) else {}
    mod_a = gr.get("module_a") if isinstance(gr.get("module_a"), dict) else {}
    mod_c = gr.get("module_c") if isinstance(gr.get("module_c"), dict) else {}
    mod_d = gr.get("module_d") if isinstance(gr.get("module_d"), dict) else {}
    mod_e = gr.get("module_e") if isinstance(gr.get("module_e"), dict) else {}

    pr = (mod_a.get("policy_rates") or {}).get(country, {})
    rate_txt = _fmt_rate_pct(pr.get("latest"))
    direction_txt = FALLBACK if _is_missing(pr.get("direction")) else str(pr.get("direction"))
    gravity = f"Policy rate {rate_txt}, {direction_txt}."

    tic = (mod_c.get("tic_holdings") or {}).get(country, {})
    if country == "China":
        cn = mod_c.get("china_fx_reserves") or {}
        plumbing = (
            f"FX reserves ${_fmt_num(cn.get('latest_bn'), 0)} billion, "
            f"{_fmt_num(cn.get('mom_change_bn'), 1)} billion vs last month."
        )
    elif tic:
        plumbing = (
            f"Holds ${_fmt_num(tic.get('latest_bn'), 0)} billion of US Treasuries, "
            f"{_fmt_num(tic.get('mom_change_bn'), 1)} billion vs last month "
            f"(trend: {tic.get('trend_3m') or FALLBACK})."
        )
    else:
        ca = mod_d.get(country) or mod_d.get("Germany" if country == "Europe" else "")
        if ca:
            latest = ca.get("latest")
            change = ca.get("change")
            if _is_missing(latest):
                plumbing = FALLBACK
            else:
                balance = "deficit" if float(latest) < 0 else "surplus"
                if _is_missing(change):
                    trend = ""
                elif float(change) < 0:
                    trend = ", widening" if balance == "deficit" else ", narrowing"
                else:
                    trend = ", narrowing" if balance == "deficit" else ", widening"
                plumbing = f"Running a current account {balance}{trend}."
        else:
            plumbing = FALLBACK

    ted = mod_e.get("tedrate") or {}
    ted_val = ted.get("latest")
    ted_stress = ted_val is not None and float(ted_val) > 0.5
    regime_snip = str(llm.get("regime_summary") or "").strip()[:200]
    if ted_stress:
        liq_txt = "⚠ Funding stress is rising in the financial system."
    else:
        liq_txt = "Financial system liquidity looks healthy."
    premium = f"{liq_txt} {regime_snip}".strip()
    return {"gravity": gravity, "plumbing": plumbing, "premium": premium}


def build_l2_gear_columns(
    country: str,
    flow_payload: dict | None,
    quant_payload: dict | None,
) -> dict[str, str]:
    flow = flow_payload if isinstance(flow_payload, dict) else {}
    sig = flow.get("signals") or {}
    country_sig = sig.get(country, {}) if isinstance(sig, dict) else {}
    reasons = country_sig.get("reasons") if isinstance(country_sig.get("reasons"), list) else []
    fp = flow.get("flow_payload") if isinstance(flow.get("flow_payload"), dict) else {}
    fx1 = (fp.get("fx_1m_local_vs_usd") or {}).get(country)
    spread = (fp.get("bond_spreads_bp") or {}).get(country)

    signal_txt = country_sig.get("signal") or FALLBACK
    expectation = (
        f"Signal {signal_txt}. "
        f"Drivers: {', '.join(str(r) for r in reasons[:4]) or FALLBACK}."
    )
    fx_txt = FALLBACK if _is_missing(fx1) else f"{float(fx1):+.2f}%"
    spread_txt = FALLBACK if _is_missing(spread) else f"{float(spread):+.1f} bp"
    plumbing = f"FX 1m vs USD: {fx_txt} · US-local 10Y spread: {spread_txt}."

    ra = (quant_payload or {}).get("Risk_Analytics") if isinstance(quant_payload, dict) else {}
    l2r = (ra or {}).get("layer2_risk") if isinstance(ra, dict) else {}
    skew = l2r.get("SKEW") if isinstance(l2r, dict) else {}
    vvix = l2r.get("VVIX") if isinstance(l2r, dict) else {}
    skew_level = skew.get("latest_level", skew.get("level")) if isinstance(skew, dict) else None
    skew_label = (skew.get("signal") or skew.get("label") or "").lower() if isinstance(skew, dict) else ""
    vvix_label = (vvix.get("signal") or vvix.get("label") or "").lower() if isinstance(vvix, dict) else ""
    skew_extreme = (skew_level is not None and float(skew_level) > 130) or skew_label in ("elevated", "high", "extreme")
    vvix_extreme = vvix_label in ("elevated", "high", "extreme")
    if skew_extreme or vvix_extreme:
        hedging = (
            "⚠ Tail-risk protection is getting expensive — investors are quietly buying "
            "downside insurance even as the surface looks calm. A sign of hidden caution."
        )
    else:
        hedging = "Tail-risk hedging costs look normal — no hidden caution signal."
    return {"expectation": expectation, "plumbing": plumbing, "hedging": hedging}


def _recent_status_for_country(country: str, flow_payload: dict | None) -> str:
    flow = flow_payload if isinstance(flow_payload, dict) else {}
    sig = flow.get("signals") or {}
    country_sig = sig.get(country, {}) if isinstance(sig, dict) else {}
    signal = str(country_sig.get("signal") or FALLBACK)
    signal_plain = {
        "GREEN": "資金流入 · Inflows",
        "RED": "需留意 · Caution",
        "YELLOW": "中性 · Neutral",
    }.get(signal, signal)
    reasons = country_sig.get("reasons") if isinstance(country_sig.get("reasons"), list) else []
    if reasons:
        return f"{signal_plain}（{reasons[0]}）"
    surfaced = flow.get("surfaced_rows")
    if isinstance(surfaced, list):
        for row in surfaced:
            if isinstance(row, dict) and country.lower() in str(row.get("name", "")).lower():
                return str(row.get("name") or FALLBACK)
        if surfaced and isinstance(surfaced[0], dict):
            return str(surfaced[0].get("name") or FALLBACK)
    return signal


def build_gear_matrix_struct(
    country: str,
    synthesis: dict | None,
    gr_payload: dict | None,
    flow_payload: dict | None,
    quant_payload: dict | None,
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Return (gear_matrix dict, gears list for Jinja loop)."""
    syn = synthesis if isinstance(synthesis, dict) else {}
    rel = syn.get("relationship_analysis") if isinstance(syn.get("relationship_analysis"), dict) else {}
    pa = rel.get("pure_alpha") if isinstance(rel.get("pure_alpha"), dict) else {}
    notes = syn.get("country_attribution_notes") if isinstance(syn.get("country_attribution_notes"), dict) else {}
    l1_cols = build_l1_gear_columns(country, gr_payload)
    l2_cols = build_l2_gear_columns(country, flow_payload, quant_payload)
    recent = _recent_status_for_country(country, flow_payload)
    cio_note = str(notes.get(country) or "").strip() or FALLBACK
    is_us = country == "US"

    def _cio(key: str) -> str:
        if is_us:
            v = pa.get(key)
            if v and str(v).strip() and str(v).strip() != FALLBACK:
                return str(v).strip()
        return cio_note

    gear_matrix = {
        "expectation": {
            "l1": l1_cols["gravity"],
            "l2": l2_cols["expectation"],
            "recent_status": recent,
            "cio_synthesis": _cio("expectation_arbitrage"),
        },
        "crowdedness": {
            "l1": l1_cols["plumbing"],
            "l2": l2_cols["plumbing"],
            "recent_status": recent,
            "cio_synthesis": _cio("crowdedness_leverage"),
        },
        "hedging": {
            "l1": l1_cols["premium"],
            "l2": l2_cols["hedging"],
            "recent_status": recent,
            "cio_synthesis": _cio("hedging_cost_anomaly"),
        },
    }
    for block in gear_matrix.values():
        for k, v in block.items():
            if not str(v or "").strip() or str(v).strip().lower() == "none":
                block[k] = FALLBACK

    gears = [
        {
            "title": "Expectation Arbitrage",
            "l1": gear_matrix["expectation"]["l1"],
            "l2": gear_matrix["expectation"]["l2"],
            "recent": gear_matrix["expectation"]["recent_status"],
            "cio": gear_matrix["expectation"]["cio_synthesis"],
        },
        {
            "title": "Crowdedness & Leverage",
            "l1": gear_matrix["crowdedness"]["l1"],
            "l2": gear_matrix["crowdedness"]["l2"],
            "recent": gear_matrix["crowdedness"]["recent_status"],
            "cio": gear_matrix["crowdedness"]["cio_synthesis"],
        },
        {
            "title": "Hedging Cost Anomaly",
            "l1": gear_matrix["hedging"]["l1"],
            "l2": gear_matrix["hedging"]["l2"],
            "recent": gear_matrix["hedging"]["recent_status"],
            "cio": gear_matrix["hedging"]["cio_synthesis"],
        },
    ]
    return gear_matrix, gears


GEAR_SEMANTIC_KEYS = ("expectation", "plumbing", "hedging")
GEAR_LABEL_KEYS = ("expectation_label", "plumbing_label", "hedging_label")
DEFAULT_L3_GEAR_TITLES = (
    "Expectation Arbitrage",
    "Crowdedness & Leverage",
    "Hedging Cost Anomaly",
)


def _valid_semantic_text(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or text == FALLBACK or text.lower() in {"none", "n/a"}:
        return None
    if text.lower().startswith("llm analysis not"):
        return None
    return text


def apply_gear_semantics_to_gears(gears: dict[str, str], semantics: dict | None) -> dict[str, str]:
    """Overlay L2 accordion gear text with CIO narrative sentences."""
    if not isinstance(gears, dict):
        return gears
    if not isinstance(semantics, dict):
        return dict(gears)
    out = dict(gears)
    for key in GEAR_SEMANTIC_KEYS:
        translated = _valid_semantic_text(semantics.get(key))
        if translated:
            out[key] = translated
    return out


def apply_gear_semantics_to_l3_panel(panel: dict[str, Any], semantics: dict | None) -> None:
    """Replace L2 gear prose and titles in an L3 country panel; leave recent/CIO untouched."""
    gears = panel.get("gears")
    if not isinstance(gears, list) or not isinstance(semantics, dict):
        return
    semantic_keys = GEAR_SEMANTIC_KEYS
    label_keys = GEAR_LABEL_KEYS
    for idx, gear in enumerate(gears):
        if not isinstance(gear, dict) or idx >= len(semantic_keys):
            continue
        translated = _valid_semantic_text(semantics.get(semantic_keys[idx]))
        if translated:
            gear["l2"] = translated
        label = _valid_semantic_text(semantics.get(label_keys[idx]))
        if label:
            gear["title"] = label
        elif not str(gear.get("title") or "").strip():
            gear["title"] = DEFAULT_L3_GEAR_TITLES[idx]


def apply_gear_semantics_to_l3_view_model(
    l3_vm: dict[str, Any],
    gear_matrix_semantics: dict | None,
) -> dict[str, Any]:
    """Apply per-country gear translations to the L3 3-Gear Conflict Matrix."""
    if not isinstance(l3_vm, dict) or not isinstance(gear_matrix_semantics, dict):
        return l3_vm
    for panel in l3_vm.get("country_matrix") or []:
        if not isinstance(panel, dict):
            continue
        country = str(panel.get("country") or "")
        semantics = gear_matrix_semantics.get(country)
        if isinstance(semantics, dict):
            apply_gear_semantics_to_l3_panel(panel, semantics)
    return l3_vm


def build_l2_gear_raw_context(
    flow_payload: dict | None,
    quant_payload: dict | None,
) -> dict[str, dict[str, Any]]:
    """Structured raw inputs for LLM gear semantic translation."""
    flow = flow_payload if isinstance(flow_payload, dict) else {}
    sig = flow.get("signals") or {}
    fp = flow.get("flow_payload") if isinstance(flow.get("flow_payload"), dict) else {}
    fx1m = fp.get("fx_1m_local_vs_usd") if isinstance(fp.get("fx_1m_local_vs_usd"), dict) else {}
    spreads = fp.get("bond_spreads_bp") if isinstance(fp.get("bond_spreads_bp"), dict) else {}
    ra = (quant_payload or {}).get("Risk_Analytics") if isinstance(quant_payload, dict) else {}
    l2r = (ra or {}).get("layer2_risk") if isinstance(ra, dict) else {}

    countries: dict[str, dict[str, Any]] = {}
    for country in COUNTRY_ORDER:
        country_sig = sig.get(country, {}) if isinstance(sig, dict) else {}
        reasons = country_sig.get("reasons") if isinstance(country_sig.get("reasons"), list) else []
        countries[country] = {
            "signal": country_sig.get("signal"),
            "drivers": [str(r) for r in reasons[:4]],
            "fx_1m_local_vs_usd": fx1m.get(country),
            "bond_spread_bp": spreads.get(country),
            "raw_gears": build_l2_gear_columns(country, flow, quant_payload),
        }

    countries["_global_risk"] = {
        "SKEW": l2r.get("SKEW") if isinstance(l2r, dict) else {},
        "VVIX": l2r.get("VVIX") if isinstance(l2r, dict) else {},
        "penalty_reason": (ra or {}).get("penalty_reason") if isinstance(ra, dict) else None,
    }
    return countries


def build_l3_country_panel(
    country: str,
    rank_row: dict,
    synthesis: dict | None,
    gr_payload: dict | None,
    flow_payload: dict | None,
    quant_payload: dict | None,
    global_divergence: int,
) -> dict[str, Any]:
    gear_matrix, gears = build_gear_matrix_struct(
        country, synthesis, gr_payload, flow_payload, quant_payload
    )
    return {
        "country": country,
        "name": country,
        "id": COUNTRY_IDS.get(country, country[:2].upper()),
        "rank": int(rank_row.get("rank", 0)),
        "is_us_core": bool(rank_row.get("is_us_core") or country == "US"),
        "divergence_score": global_divergence,
        "signal_color": str(rank_row.get("signal_color") or rank_row.get("signal") or "YELLOW"),
        "active": country == "US",
        "gear_matrix": gear_matrix,
        "gears": gears,
    }


def build_l1_accordion_rows(gr_payload: dict | None, global_divergence: int = 0) -> list[dict]:
    ranked = build_country_rank_rows(build_l1_country_input(gr_payload), layer="L1")
    out: list[dict] = []
    for row in ranked:
        out.append(
            {
                "row": row,
                "summary": format_accordion_summary(row, int(row["rank"]), global_divergence),
                "gears": build_l1_gear_columns(row["country"], gr_payload),
            }
        )
    return out


def build_l2_accordion_rows(
    country_rows: list[dict],
    flow_payload: dict | None,
    quant_payload: dict | None,
    global_divergence: int = 0,
) -> list[dict]:
    ranked = build_country_rank_rows(country_rows, layer="L2")
    out: list[dict] = []
    for row in ranked:
        out.append(
            {
                "row": row,
                "summary": format_accordion_summary(row, int(row["rank"]), global_divergence),
                "gears": build_l2_gear_columns(row["country"], flow_payload, quant_payload),
            }
        )
    return out


def build_l2_country_rows_from_sources(
    output_date: date,
    signals: dict | None,
    flow_doc: dict | None,
    gr_payload: dict | None,
) -> list[dict]:
    """Full L2 country rows (same logic as run._build_country_rows)."""
    signals = signals if isinstance(signals, dict) else {}
    rows: dict[str, dict] = {
        c: {
            "country": c,
            "signal": "YELLOW",
            "policy_rate": FALLBACK,
            "fx_1m": FALLBACK,
            "spread_10y": FALLBACK,
            "notes": "Partial data",
        }
        for c in COUNTRY_ORDER
    }

    rec = signals.get("headline", {}).get("recession_risk", {}).get("signal", "YELLOW")
    inf = signals.get("headline", {}).get("inflation", {}).get("signal", "YELLOW")
    stress = signals.get("headline", {}).get("financial_stress", {}).get("signal", "YELLOW")
    score = [rec, inf, stress].count("GREEN") - [rec, inf, stress].count("RED")
    rows["US"]["signal"] = "GREEN" if score > 0 else ("RED" if score < 0 else "YELLOW")
    fed_row = next((r for r in signals.get("details", []) if r.get("key") == "fed_funds_rate"), None)
    if fed_row and not _is_missing(fed_row.get("value")):
        rows["US"]["policy_rate"] = _fmt_rate_pct(fed_row.get("value"))

    flow = flow_doc if isinstance(flow_doc, dict) else {}
    if not flow:
        flow_path = OUTPUT_DIR / f"flow_data_{output_date.isoformat()}.json"
        if flow_path.exists():
            try:
                flow = json.loads(flow_path.read_text(encoding="utf-8"))
            except Exception:
                flow = {}

    sig = flow.get("signals", {}) if isinstance(flow.get("signals"), dict) else {}
    payload = flow.get("flow_payload", {}) if isinstance(flow.get("flow_payload"), dict) else {}
    fx1m = payload.get("fx_1m_local_vs_usd", {}) if isinstance(payload, dict) else {}
    spreads = payload.get("bond_spreads_bp", {}) if isinstance(payload, dict) else {}

    for c in COUNTRY_ORDER:
        if c == "US":
            continue
        cs = sig.get(c, {}) if isinstance(sig.get(c), dict) else {}
        rows[c]["signal"] = str(cs.get("signal") or rows[c]["signal"])
        vfx = fx1m.get(c) if isinstance(fx1m, dict) else None
        vsp = spreads.get(c) if isinstance(spreads, dict) else None
        rows[c]["fx_1m"] = FALLBACK if _is_missing(vfx) else f"{float(vfx):+.2f}%"
        if _is_missing(vsp):
            rows[c]["spread_10y"] = FALLBACK
            rows[c]["spread_bp"] = 0.0
        else:
            rows[c]["spread_10y"] = f"{float(vsp):+.1f} bp"
            rows[c]["spread_bp"] = abs(float(vsp))
        rows[c]["notes"] = "From flow_run.py daily flow data"

    gr = gr_payload if isinstance(gr_payload, dict) else {}
    if not gr:
        gr_path = OUTPUT_DIR / f"global_regime_data_{output_date.isoformat()}.json"
        if gr_path.exists():
            try:
                gr = json.loads(gr_path.read_text(encoding="utf-8"))
            except Exception:
                gr = {}
    pr = gr.get("module_a", {}).get("policy_rates", {}) if isinstance(gr.get("module_a"), dict) else {}
    if isinstance(pr, dict):
        for c in COUNTRY_ORDER:
            p = (pr.get(c) or {}).get("latest")
            if not _is_missing(p):
                rows[c]["policy_rate"] = _fmt_rate_pct(p)

    return [rows[c] for c in COUNTRY_ORDER]


def build_l1_country_input(gr_payload: dict | None) -> list[dict]:
    gr = gr_payload if isinstance(gr_payload, dict) else {}
    llm = gr.get("llm") if isinstance(gr.get("llm"), dict) else {}
    cycle = llm.get("country_cycle") if isinstance(llm.get("country_cycle"), dict) else {}
    mod_a = gr.get("module_a") if isinstance(gr.get("module_a"), dict) else {}
    rates = mod_a.get("policy_rates") if isinstance(mod_a.get("policy_rates"), dict) else {}
    rows = []
    for c in COUNTRY_ORDER:
        r = rates.get(c, {}) if isinstance(rates.get(c), dict) else {}
        cyc = str(cycle.get(c, "") or "")
        sig = "YELLOW"
        if cyc in ("Slowdown", "Contraction", "hiking"):
            sig = "RED"
        elif cyc in ("Expansion", "cutting"):
            sig = "GREEN"
        rows.append(
            {
                "country": c,
                "signal": sig,
                "direction": r.get("direction") or "on hold",
                "mom_val": r.get("mom_change"),
                "country_cycle": cyc,
                "policy_rate": r.get("latest"),
            }
        )
    return rows


def build_l3_view_model(
    synthesis: dict | None,
    l1: dict | None,
    l2a: dict | None,
    l2b: dict | None,
    quant_payload: dict | None,
    country_rows: list[dict] | None = None,
    report_date: date | None = None,
) -> dict[str, Any]:
    """Phase 2 L3 view-model: overview, watchout, spread trade, country matrix."""
    syn = synthesis if isinstance(synthesis, dict) else {}
    l2a = l2a if isinstance(l2a, dict) else {}
    l1 = l1 if isinstance(l1, dict) else {}
    l2b = l2b if isinstance(l2b, dict) else {}
    quant_payload = quant_payload if isinstance(quant_payload, dict) else {}

    z1 = syn.get("zone1_pulse") if isinstance(syn.get("zone1_pulse"), dict) else {}
    bullets = z1.get("flash_bullets") if isinstance(z1.get("flash_bullets"), dict) else {}
    flash_parts = [
        str(bullets.get("capital_flows") or "").strip(),
        str(bullets.get("volatility") or "").strip(),
        str(bullets.get("macro_sentiment") or "").strip(),
    ]
    flash_parts = [p for p in flash_parts if p and p != FALLBACK and p.lower() != "none"]
    daily_overview = " ".join(flash_parts)
    if not daily_overview:
        daily_overview = str(l2a.get("executive_summary") or "").strip()
    if not daily_overview or daily_overview == FALLBACK:
        daily_overview = FALLBACK

    critical_watchout: list[str] = []
    watch_list = syn.get("watch_list")
    if isinstance(watch_list, list):
        for item in watch_list:
            s = str(item or "").strip()
            if s and s != FALLBACK and s.lower() != "none":
                critical_watchout.append(s)
    directive = syn.get("cio_directive") if isinstance(syn.get("cio_directive"), dict) else {}
    wo = str(directive.get("the_watchout") or "").strip()
    if wo and wo != FALLBACK and wo not in critical_watchout:
        critical_watchout.append(wo)
    if not critical_watchout:
        critical_watchout = [FALLBACK]

    d = report_date or date.today()
    rows = country_rows or build_l2_country_rows_from_sources(d, l2a, l2b, l1)
    ranked = build_country_rank_rows(rows, layer="L2")
    rank_by_country = {r["country"]: r for r in ranked}
    global_div = global_divergence_from_quant(quant_payload)
    spread_line = spread_trade_line(ranked) if ranked else FALLBACK

    matrix_tabs = [
        {
            "country": c,
            "label": TAB_LABELS.get(c, c[:2]),
            "active": c == "US",
            "rank": int(rank_by_country.get(c, {}).get("rank", 0)),
        }
        for c in COUNTRY_ORDER
    ]

    country_matrix: list[dict[str, Any]] = []
    for c in COUNTRY_ORDER:
        rank_row = rank_by_country.get(c, {"country": c, "rank": 0, "signal": "YELLOW", "is_us_core": c == "US"})
        country_matrix.append(
            build_l3_country_panel(c, rank_row, syn, l1, l2b, quant_payload, global_div)
        )

    return {
        "daily_overview": daily_overview,
        "critical_watchout": critical_watchout,
        "spread_trade_line": spread_line,
        "matrix_tabs": matrix_tabs,
        "country_matrix": country_matrix,
        "ranked_countries": ranked,
    }


# Backward-compatible alias
build_l3_gear_matrix = build_gear_matrix_struct
