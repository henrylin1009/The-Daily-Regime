"""Generate the unified MACRA DATA APPENDIX page from layer snapshots."""

from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.appendix_exceptions import (
    anomaly_label_set,
    build_exception_dashboard,
    mark_anomaly_rows,
)
from src.config import OUTPUT_DIR, TEMPLATES_DIR
from src.macra_assets import macra_style_block
from src.macra_nav import LAYER_NAV_CSS, build_layer_nav_html, layer_nav_hrefs
from src.macra_ui import build_l2_country_rows_from_sources

_PLACEHOLDER_RE = re.compile(
    r"^(—|n/a|na|none|null|nan|llm analysis not avaliable|llm analysis not available)$",
    re.I,
)


def _norm_name(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _is_taiwan_flow(name: object) -> bool:
    return "taiwan foreign flow" in _norm_name(name)


def _validation_name_set(validation_panel: list[dict] | None) -> frozenset[str]:
    return frozenset(_norm_name(r.get("name")) for r in validation_panel or [] if isinstance(r, dict))


def _build_surfaced_rows(
    surfaced_rows: list[dict] | None,
    validation_panel: list[dict] | None,
) -> list[dict[str, str]]:
    val_names = _validation_name_set(validation_panel)
    rows: list[dict[str, str]] = []
    for r in surfaced_rows or []:
        if not isinstance(r, dict):
            continue
        name = _dash(r.get("name"))
        key = _norm_name(name)
        if _is_taiwan_flow(key) or key in val_names:
            continue
        rows.append(
            {
                "name": name,
                "latest": _dash(r.get("latest")),
                "change": _dash(r.get("change")),
                "z": _dash(r.get("z")),
                "signal": _dash(r.get("signal")),
            }
        )
    return rows


def _dash(value: object) -> str:
    if value is None:
        return "—"
    s = str(value).strip()
    if not s or _PLACEHOLDER_RE.match(s):
        return "—"
    return s


def _fmt_pct(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return _dash(value)


def _build_fx_momentum_rows(
    country_rows: list[dict[str, Any]],
    flow_payload: dict[str, Any] | None,
) -> list[dict[str, str]]:
    payload = flow_payload if isinstance(flow_payload, dict) else {}
    fx3 = payload.get("fx_3m_local_vs_usd") if isinstance(payload.get("fx_3m_local_vs_usd"), dict) else {}
    rows: list[dict[str, str]] = []
    for r in country_rows:
        if not isinstance(r, dict):
            continue
        country = str(r.get("country") or "—")
        fx3_val = fx3.get(country) if country != "US" else None
        rows.append(
            {
                "country": country,
                "signal": _dash(r.get("signal")),
                "fx_1m": _dash(r.get("fx_1m")),
                "fx_3m": _fmt_pct(fx3_val) if fx3_val is not None else "—",
                "spread_10y": _dash(r.get("spread_10y")),
            }
        )
    return rows


def _build_flow_validation_rows(
    validation_panel: list[dict] | None,
    flow_payload: dict[str, Any] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for r in validation_panel or []:
        if not isinstance(r, dict):
            continue
        if _is_taiwan_flow(r.get("name")):
            continue
        rows.append(
            {
                "name": _dash(r.get("name")),
                "latest": _dash(r.get("latest")),
                "chg_1m": _dash(r.get("chg_1m", r.get("change"))),
                "z": _dash(r.get("z")),
                "signal": _dash(r.get("signal")),
            }
        )

    payload = flow_payload if isinstance(flow_payload, dict) else {}
    tw = payload.get("taiwan_foreign_flow") if isinstance(payload.get("taiwan_foreign_flow"), dict) else {}
    if tw:
        rows.append(
            {
                "name": "Taiwan foreign flow",
                "latest": f"5d {_dash(tw.get('cum_5d_bn_twd'))} · 20d {_dash(tw.get('cum_20d_bn_twd'))} TWD bn",
                "chg_1m": f"{tw.get('days', 0)} obs",
                "z": "—",
                "signal": "—",
            }
        )
    return rows


def _build_vol_skew_rows(
    quant_momentum: dict[str, Any] | None,
    quant_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    momentum = quant_momentum if isinstance(quant_momentum, dict) else {}
    for name, payload in momentum.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "name": str(name),
                "level": "—",
                "label": _dash(payload.get("label")),
                "z": _dash(payload.get("z_latest")),
                "signal": "—",
            }
        )

    ra = (quant_context or {}).get("risk_analytics") if isinstance(quant_context, dict) else {}
    l2r = ra.get("layer2_risk") if isinstance(ra, dict) else {}
    if isinstance(l2r, dict):
        for name in ("SKEW", "VVIX"):
            obj = l2r.get(name)
            if not isinstance(obj, dict):
                continue
            level = obj.get("latest_level", obj.get("level"))
            rows.append(
                {
                    "name": name,
                    "level": _dash(level),
                    "label": _dash(obj.get("signal", obj.get("label"))),
                    "z": _dash(obj.get("z_latest", obj.get("z"))),
                    "signal": _dash(obj.get("signal")),
                }
            )
    return rows


def _build_historical_rows(matches: list[dict] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for m in matches or []:
        if not isinstance(m, dict):
            continue
        rows.append(
            {
                "date": _dash(m.get("date")),
                "distance": _dash(m.get("distance")),
                "description": _dash(m.get("description")),
            }
        )
    return rows


def _build_macro_drawer(l1_ctx: dict[str, Any], anomaly_labels: frozenset[str]) -> dict[str, Any]:
    return {
        "policy_rows": mark_anomaly_rows(l1_ctx.get("policy_rows") or [], "country", anomaly_labels),
        "bs_rows": mark_anomaly_rows(l1_ctx.get("bs_rows") or [], "cb", anomaly_labels),
        "ca_rows": mark_anomaly_rows(l1_ctx.get("ca_rows") or [], "country", anomaly_labels),
        "tic_rows": mark_anomaly_rows(l1_ctx.get("tic_rows") or [], "country", anomaly_labels),
        "cn_res_latest": _dash(l1_ctx.get("cn_res_latest")),
        "cn_res_mom": _dash(l1_ctx.get("cn_res_mom")),
        "cn_res_mom_val": l1_ctx.get("cn_res_mom_val") or 0.0,
    }


def _build_tactical_drawer(
    l3_ctx: dict[str, Any],
    fx_rows: list[dict[str, str]],
    flow_rows: list[dict[str, str]],
    surfaced_rows: list[dict[str, str]],
    anomaly_labels: frozenset[str],
) -> dict[str, Any]:
    surfaced = []
    for r in surfaced_rows:
        item = dict(r)
        item["anomaly"] = str(item.get("name") or "") in anomaly_labels
        surfaced.append(item)

    flow_marked = mark_anomaly_rows(flow_rows, "name", anomaly_labels)
    for item in flow_marked:
        if item.get("name") == "Taiwan foreign flow":
            item["anomaly"] = any(_is_taiwan_flow(lb) for lb in anomaly_labels)

    eod = l3_ctx.get("event_of_day") if isinstance(l3_ctx.get("event_of_day"), dict) else {}
    eod_name = _dash(eod.get("name"))
    show_eod = eod_name != "—" and not _is_taiwan_flow(eod_name)
    return {
        "l3_available": bool(l3_ctx.get("l3_available")),
        "surfaced_rows": surfaced,
        "fx_rows": mark_anomaly_rows(fx_rows, "country", anomaly_labels),
        "flow_rows": flow_marked,
        "event_of_day": {
            "name": eod_name,
            "latest": _dash(eod.get("latest")),
            "change": _dash(eod.get("change")),
            "z": _dash(eod.get("z")),
            "signal": _dash(eod.get("signal")),
            "anomaly": eod_name in anomaly_labels,
            "show": show_eod,
        },
    }


def _build_sentiment_drawer(
    l1_ctx: dict[str, Any],
    l3_ctx: dict[str, Any],
    vol_skew_rows: list[dict[str, str]],
    historical_rows: list[dict[str, str]],
    anomaly_labels: frozenset[str],
) -> dict[str, Any]:
    cot_rows = []
    for r in l1_ctx.get("cot_rows") or []:
        if not isinstance(r, dict):
            continue
        contract = _dash(r.get("contract"))
        cot_rows.append(
            {
                "contract": contract,
                "net_fmt": _dash(r.get("net_fmt")),
                "chg4w_fmt": _dash(r.get("chg4w_fmt")),
                "zscore_fmt": _dash(r.get("zscore_fmt")),
                "extreme": bool(r.get("extreme")),
                "anomaly": contract in anomaly_labels,
            }
        )

    liq_rows = []
    for r in l1_ctx.get("liq_rows") or []:
        if not isinstance(r, dict):
            continue
        indicator = _dash(r.get("indicator"))
        liq_rows.append(
            {
                "indicator": indicator,
                "value": _dash(r.get("value")),
                "change": _dash(r.get("change")),
                "signal": _dash(r.get("signal")),
                "anomaly": indicator in anomaly_labels,
            }
        )

    return {
        "cot_rows": cot_rows,
        "liq_rows": liq_rows,
        "vol_skew_rows": mark_anomaly_rows(vol_skew_rows, "name", anomaly_labels),
        "historical_rows": historical_rows,
        "l3_available": bool(l3_ctx.get("l3_available")),
    }


def render_appendix_html(report_date: str | date) -> Path:
    """Build output/appendix.html from the same JSON snapshots as L1/L2/L3."""
    if isinstance(report_date, str):
        rd = date.fromisoformat(report_date)
        d = report_date
    else:
        rd = report_date
        d = report_date.isoformat()

    l1_path = OUTPUT_DIR / f"global_regime_data_{d}.json"
    l2a_path = OUTPUT_DIR / f"brief_data_{d}.json"
    l2b_path = OUTPUT_DIR / f"flow_data_{d}.json"
    synth_path = OUTPUT_DIR / f"synthesis_data_{d}.json"

    missing = [p.name for p in (l1_path, l2a_path, l2b_path) if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing appendix inputs: {', '.join(missing)}")

    from global_regime import build_l1_html_context

    l1 = json.loads(l1_path.read_text(encoding="utf-8"))
    l2a = json.loads(l2a_path.read_text(encoding="utf-8"))
    l2b = json.loads(l2b_path.read_text(encoding="utf-8"))

    cache_age = 0
    fetched_at = l1.get("fetched_at")
    if fetched_at:
        try:
            from datetime import timezone

            ts = datetime.fromisoformat(str(fetched_at))
            cache_age = max(
                0,
                int((datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)).total_seconds() // 86400),
            )
        except Exception:
            cache_age = 0

    l1_ctx = build_l1_html_context(l1, rd, cache_age)
    country_rows = build_l2_country_rows_from_sources(rd, l2a, l2b, l1)
    flow_payload = l2b.get("flow_payload") if isinstance(l2b.get("flow_payload"), dict) else {}

    l3_ctx: dict[str, Any] = {"l3_available": False}
    quant_context: dict[str, Any] = {}
    if synth_path.exists():
        try:
            from synthesis import build_l3_appendix_context

            synth_doc = json.loads(synth_path.read_text(encoding="utf-8"))
            synthesis = synth_doc.get("synthesis") if isinstance(synth_doc.get("synthesis"), dict) else {}
            quant_context = synth_doc.get("quant_context") if isinstance(synth_doc.get("quant_context"), dict) else {}
            l3_ctx = build_l3_appendix_context(
                rd,
                l1,
                l2a,
                l2b,
                synthesis,
                quant_context,
                synth_doc,
            )
        except Exception as exc:
            print(f"  Warning: L3 appendix context incomplete ({exc})", file=sys.stderr)
            l3_ctx = {"l3_available": False}

    vol_skew_rows = _build_vol_skew_rows(l3_ctx.get("quant_momentum"), quant_context)
    l3_ctx["vol_skew_rows"] = vol_skew_rows

    fx_rows = _build_fx_momentum_rows(country_rows, flow_payload)
    validation_panel = l3_ctx.get("validation_panel") if isinstance(l3_ctx.get("validation_panel"), list) else []
    surfaced_rows = _build_surfaced_rows(l3_ctx.get("surfaced_rows"), validation_panel)
    flow_rows = _build_flow_validation_rows(validation_panel, flow_payload)
    historical_rows = _build_historical_rows(l2a.get("matches") if isinstance(l2a.get("matches"), list) else [])

    exceptions = build_exception_dashboard(l1_ctx, {}, l3_ctx, country_rows)
    anomaly_labels = anomaly_label_set(exceptions)

    macro = _build_macro_drawer(l1_ctx, anomaly_labels)
    tactical = _build_tactical_drawer(l3_ctx, fx_rows, flow_rows, surfaced_rows, anomaly_labels)
    sentiment = _build_sentiment_drawer(l1_ctx, l3_ctx, vol_skew_rows, historical_rows, anomaly_labels)

    cb = d.replace("-", "")
    nav_hrefs = layer_nav_hrefs(d)
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("appendix.html")
    html = template.render(
        report_date=d,
        cache_buster=cb,
        macra_style_block=macra_style_block(),
        layer_nav_css=LAYER_NAV_CSS,
        layer_nav_html=build_layer_nav_html("appendix", nav_hrefs),
        exceptions=exceptions,
        macro=macro,
        tactical=tactical,
        sentiment=sentiment,
    )
    out_path = OUTPUT_DIR / "appendix.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
