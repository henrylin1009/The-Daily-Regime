"""Stage 4b: Render daily brief as HTML via Jinja2."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.config import OUTPUT_DIR, TEMPLATES_DIR


def _fmt_pct(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _fmt_value(value, key: str = "") -> str:
    if value is None:
        return "N/A"
    if key == "spread_2y10y":
        return f"{value:.0f} bp"
    if key in (
        "cpi_yoy", "core_cpi_yoy", "pce_yoy", "gdp_growth",
        "unemployment_rate", "fed_funds_rate",
        "credit_spread_hy", "credit_spread_hy_oas",
    ):
        return f"{value:.2f}%"
    if key == "uso":
        return f"${value:.2f}"
    if key == "gld":
        return f"${value:.2f}"
    return f"{value:.2f}"


def render_html(
    signals: dict,
    matches: list[dict],
    narrative: str,
    executive_summary: str | None = None,
    output_date: date | None = None,
    factor_attrib: dict | None = None,
    spx_contrib: dict | None = None,
    regime_stats: dict | None = None,
    regime_summary: dict | None = None,
    country_rows: list[dict] | None = None,
    show_country_matrix: bool = True,
) -> Path:
    """Render daily brief HTML and return output path."""
    report_date = output_date or date.today()
    filename = "brief.html"
    out_path = OUTPUT_DIR / filename

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["fmt_pct"] = _fmt_pct
    env.filters["fmt_value"] = _fmt_value

    template = env.get_template("daily_brief.html")

    # Merge headline signals into detail rows for display
    headline_keys = {"spread_2y10y": "recession_risk", "cpi_yoy": "inflation", "vix": "financial_stress"}
    details = []
    for row in signals["details"]:
        r = dict(row)
        for key, headline_name in headline_keys.items():
            if r["key"] == key:
                r["signal"] = signals["headline"][headline_name]["signal"]
        details.append(r)

    top_match = matches[0] if matches else {
        "date": "N/A",
        "description": "No match found",
        "distance": 0,
        "what_happened": "",
        "spy_return_3m": None,
        "spy_return_12m": None,
        "max_drawdown_12m": None,
    }
    other_matches = matches[1:] if len(matches) > 1 else []

    stripped = narrative.strip()
    narrative_is_template = stripped.startswith("[Template narrative")
    narrative_is_offline = stripped.startswith("Macro narrative unavailable (offline mode)")
    executive_summary = (executive_summary or "").strip()
    executive_summary_is_fallback = executive_summary.startswith("Narrative unavailable")

    core_keys = {
        "cpi_yoy",
        "unemployment_rate",
        "spread_2y10y",
        "vix",
        "credit_spread_hy",
        "fed_funds_rate",
    }
    core_rows = [r for r in details if r.get("key") in core_keys]
    total_core = len(core_rows)
    available_core = sum(1 for r in core_rows if r.get("value") is not None)
    ratio = (available_core / total_core) if total_core else 0.0
    missing_names = [str(r.get("name", r.get("key", ""))) for r in core_rows if r.get("value") is None][:3]
    rec = signals["headline"]["recession_risk"]["signal"]
    inf = signals["headline"]["inflation"]["signal"]
    stress = signals["headline"]["financial_stress"]["signal"]
    risk_conclusion = (
        f"Recession risk {rec.lower()}, inflation {inf.lower()}, financial stress {stress.lower()}."
    )

    html = template.render(
        report_date=report_date.isoformat(),
        as_of=signals.get("as_of", report_date.isoformat()),
        headline=signals["headline"],
        regime=regime_summary,
        narrative=narrative,
        executive_summary=executive_summary,
        executive_summary_is_fallback=executive_summary_is_fallback,
        narrative_is_template=narrative_is_template,
        narrative_is_offline=narrative_is_offline,
        top_match=top_match,
        other_matches=other_matches,
        details=details,
        risk_conclusion=risk_conclusion,
        country_rows=country_rows or [],
        show_country_matrix=show_country_matrix,
        data_coverage={
            "available_core": available_core,
            "total_core": total_core,
            "ratio_pct": round(ratio * 100.0, 1),
            "missing_critical_fields": missing_names,
        },
        factor_attrib=factor_attrib or {"available": False},
        spx_contrib=spx_contrib or {"available": False},
        regime_stats=regime_stats or {"available": False},
    )

    out_path.write_text(html, encoding="utf-8")
    return out_path


def main() -> None:
    from src.analyst import PLACEHOLDER_NARRATIVE
    from src.collect import load_all_from_cache
    from src.history_match import build_feature_matrix, find_similar_periods, save_matches
    from src.indicators import compute_signals

    data = load_all_from_cache()
    signals = compute_signals(data)
    matrix = build_feature_matrix(data)
    matches = find_similar_periods(matrix, data)
    save_matches(matches)

    path = render_html(signals, matches, PLACEHOLDER_NARRATIVE)
    print(f"Rendered: {path}")


if __name__ == "__main__":
    main()
