#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from src.config import OUTPUT_DIR


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _coverage_from_map(name: str, checks: list[tuple[str, bool]]) -> dict:
    total = len(checks)
    available = sum(1 for _, ok in checks if ok)
    ratio = (available / total) if total else 0.0
    return {
        "layer": name,
        "available_core": available,
        "total_core": total,
        "ratio_pct": round(ratio * 100.0, 1),
        "degraded": ratio < 0.8,
        "missing_critical_fields": [k for k, ok in checks if not ok][:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily data completeness report")
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    args = parser.parse_args()
    d = (date.fromisoformat(args.date) if args.date else date.today()).isoformat()

    l1 = _read(OUTPUT_DIR / f"global_regime_data_{d}.json")
    l2a = _read(OUTPUT_DIR / f"brief_data_{d}.json")
    l2b = _read(OUTPUT_DIR / f"flow_data_{d}.json")
    l3 = _read(OUTPUT_DIR / f"synthesis_data_{d}.json")

    l1_checks = [
        ("US policy rate", l1.get("module_a", {}).get("policy_rates", {}).get("US", {}).get("latest") is not None),
        ("Europe policy rate", l1.get("module_a", {}).get("policy_rates", {}).get("Europe", {}).get("latest") is not None),
        ("UK policy rate", l1.get("module_a", {}).get("policy_rates", {}).get("UK", {}).get("latest") is not None),
        ("COT JPY", l1.get("module_b", {}).get("contracts", {}).get("JPY", {}).get("latest_net") is not None),
        ("TIC All foreign", l1.get("module_c", {}).get("tic_holdings", {}).get("All foreign", {}).get("latest_bn") is not None),
        ("China reserves", l1.get("module_c", {}).get("china_fx_reserves", {}).get("latest_bn") is not None),
    ]
    l2_checks = [
        ("signals", bool(l2a.get("signals"))),
        ("executive_summary", bool(l2a.get("executive_summary"))),
        ("flow signals", bool(l2b.get("flow_payload", {}).get("signals"))),
        ("TW flow 5d", l2b.get("module_c", {}).get("taiwan_flow", {}).get("cum_5d_bn_twd") is not None),
        ("CN reserves MoM", l2b.get("module_c", {}).get("china_fx_reserves", {}).get("mom_change_bn") is not None),
    ]
    syn = l3.get("synthesis", {}) if isinstance(l3.get("synthesis"), dict) else {}
    llm_meta = l3.get("llm_meta", {}) if isinstance(l3.get("llm_meta"), dict) else {}
    placeholder = "LLM analysis not available"
    directive = syn.get("cio_directive") if isinstance(syn.get("cio_directive"), dict) else {}
    rel = syn.get("relationship_analysis") if isinstance(syn.get("relationship_analysis"), dict) else {}
    pa = rel.get("pure_alpha") if isinstance(rel.get("pure_alpha"), dict) else {}
    quant = l3.get("quant_context") if isinstance(l3.get("quant_context"), dict) else {}
    stance_ok = bool(directive.get("the_stance")) and str(directive.get("the_stance")).strip() != placeholder
    rel_ok = bool(pa.get("expectation_arbitrage")) and str(pa.get("expectation_arbitrage")).strip() != placeholder
    regime_ok = bool(quant.get("macro_regime_label"))
    z1 = syn.get("zone1_pulse") if isinstance(syn.get("zone1_pulse"), dict) else {}
    zone1_ok = bool(z1.get("market_overview")) and str(z1.get("market_overview")).strip() != placeholder
    llm_ok = llm_meta.get("status") == "ok" and not l3.get("llm_placeholder", False)
    l3_checks = [
        ("cio_directive stance", stance_ok),
        ("pure_alpha expectation_arbitrage", rel_ok),
        ("quant macro_regime_label", regime_ok),
        ("zone1_pulse overview", zone1_ok),
        ("watch_list", bool(syn.get("watch_list")) and syn.get("watch_list") != [placeholder]),
        ("llm_meta ok", llm_ok),
    ]

    report = {
        "date": d,
        "layers": [
            _coverage_from_map("Layer 1", l1_checks),
            _coverage_from_map("Layer 2", l2_checks),
            _coverage_from_map("Layer 3", l3_checks),
        ],
    }

    out = OUTPUT_DIR / f"daily_data_quality_{d}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Output: {out}")
    for layer in report["layers"]:
        print(
            f'{layer["layer"]}: {layer["available_core"]}/{layer["total_core"]} ({layer["ratio_pct"]}%) '
            f'| degraded={layer["degraded"]} | missing={", ".join(layer["missing_critical_fields"]) or "none"}'
        )


if __name__ == "__main__":
    main()
