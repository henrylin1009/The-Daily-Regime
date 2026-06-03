"""SPY return attribution: tradeable macro, CRR-style macro, and Fama-French factors."""

from __future__ import annotations

import io
import sys
import zipfile
from typing import Literal

import numpy as np
import pandas as pd
import requests

from src.config import RAW_DIR

# --- Kenneth French (Panel C) ---
FF5_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)
MOM_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)
FF_CACHE = RAW_DIR / "ff_factors_5mom_daily.csv"

FF_LABELS = {
    "Mkt-RF": "Market (Mkt-RF)",
    "SMB": "Size (SMB)",
    "HML": "Value (HML)",
    "RMW": "Profitability (RMW)",
    "CMA": "Investment (CMA)",
    "Mom": "Momentum (Mom)",
}

# --- Panel A: tradeable macro/asset ---
TRADEABLE_SPECS: list[tuple[str, str, Literal["return_pct", "diff"]]] = [
    ("gold", "gld", "return_pct"),
    ("rates", "spread_2y10y", "diff"),
    ("credit", "credit_spread_hy", "diff"),
    ("oil", "uso", "return_pct"),
    ("usd", "dxy", "return_pct"),
    ("vol", "vix", "diff"),
]

TRADEABLE_LABELS = {
    "gold": "Gold (GLD)",
    "rates": "Rates (2y10y Δ)",
    "credit": "Credit (Baa spread Δ)",
    "oil": "Oil (WTI)",
    "usd": "Dollar (DXY)",
    "vol": "Volatility (VIX Δ)",
}

# --- Panel D: US sector SPDRs ---
SECTOR_SPECS: list[tuple[str, str, Literal["return_pct"]]] = [
    ("tech", "xlk", "return_pct"),
    ("financials", "xlf", "return_pct"),
    ("healthcare", "xlv", "return_pct"),
    ("energy", "xle", "return_pct"),
    ("utilities", "xlu", "return_pct"),
    ("industrials", "xli", "return_pct"),
]

SECTOR_LABELS = {
    "tech": "Tech (XLK)",
    "financials": "Financials (XLF)",
    "healthcare": "Healthcare (XLV)",
    "energy": "Energy (XLE)",
    "utilities": "Utilities (XLU)",
    "industrials": "Industrials (XLI)",
}

# --- Panel B: CRR-style (monthly) ---
CRR_SPECS: list[tuple[str, str, Literal["monthly_diff", "monthly_yoy_diff"]]] = [
    ("output", "indpro_growth", "monthly_diff"),
    ("inflation", "cpi_yoy", "monthly_diff"),
    ("term_structure", "spread_2y10y", "monthly_diff"),
    ("default_risk", "credit_spread_hy", "monthly_diff"),
    ("labor", "unemployment_rate", "monthly_diff"),
]

CRR_LABELS = {
    "output": "Industrial production (YoY Δ)",
    "inflation": "Inflation (CPI YoY Δ)",
    "term_structure": "Term structure (2y10y Δ)",
    "default_risk": "Default risk (Baa spread Δ)",
    "labor": "Labor market (unemployment Δ)",
}


def _parse_french_daily_csv(text: str, required_cols: list[str] | None = None) -> pd.DataFrame:
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line or (required_cols and all(c in line for c in required_cols if c != "Mom")):
            if "SMB" in line or "Mom" in line:
                header_idx = i
                break
        if required_cols == ["Mom"] and "Mom" in line:
            header_idx = i
            break
    if header_idx is None:
        for i, line in enumerate(lines):
            if "Mom" in line and line.strip().startswith(","):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("Could not find Fama-French header row")

    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            break
        data_lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)))
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    date_col = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    for col in df.columns:
        if col not in ("date", date_col):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    extra = required_cols or ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    keep = ["date"] + [c for c in extra if c in df.columns]
    return df[keep].dropna(subset=["date"])


def fetch_ff5_momentum(force_refresh: bool = False) -> pd.DataFrame:
    if FF_CACHE.exists() and not force_refresh:
        return pd.read_csv(FF_CACHE, parse_dates=["date"])

    resp5 = requests.get(FF5_DAILY_URL, timeout=120)
    resp5.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp5.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        ff5 = _parse_french_daily_csv(zf.read(name).decode("utf-8", errors="ignore"))

    resp_m = requests.get(MOM_DAILY_URL, timeout=120)
    resp_m.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp_m.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        mom = _parse_french_daily_csv(
            zf.read(name).decode("utf-8", errors="ignore"),
            required_cols=["Mom"],
        )
        mom_cols = [c for c in mom.columns if c != "date"]
        mom = mom.rename(columns={mom_cols[0]: "Mom"}) if "Mom" not in mom.columns else mom

    merged = pd.merge(ff5, mom[["date", "Mom"]], on="date", how="inner")
    FF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(FF_CACHE, index=False)
    return merged


def _series_from_cache(data: dict, key: str) -> pd.Series:
    if key not in data or data[key].empty:
        return pd.Series(dtype=float)
    df = data[key].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].sort_index()


def _spy_daily_returns(spy: pd.DataFrame) -> pd.Series:
    prices = _series_from_cache({"spy": spy}, "spy")
    return prices.pct_change().dropna() * 100


def _factor_return(series: pd.Series, transform: Literal["return_pct", "diff"]) -> pd.Series:
    if series.empty:
        return series
    if transform == "return_pct":
        return series.pct_change() * 100
    return series.diff()


def build_tradeable_factors(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for code, key, transform in TRADEABLE_SPECS:
        s = _series_from_cache(data, key)
        if s.empty:
            continue
        frames[code] = _factor_return(s, transform)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).dropna(how="all")


def build_sector_factors(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for code, key, transform in SECTOR_SPECS:
        s = _series_from_cache(data, key)
        if s.empty:
            continue
        frames[code] = _factor_return(s, transform)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).dropna(how="all")


def _monthly_factor_changes(data: dict, key: str, mode: str) -> pd.Series:
    s = _series_from_cache(data, key)
    if s.empty:
        return s
    monthly = s.resample("ME").last()
    if mode == "monthly_yoy_diff":
        return monthly.diff(12)
    return monthly.diff()


def build_crr_factors(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    for code, key, mode in CRR_SPECS:
        s = _monthly_factor_changes(data, key, mode)
        if s.empty:
            continue
        frames[code] = s
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).dropna(how="all")


def _run_attribution_panel(
    y: pd.Series,
    factors: pd.DataFrame,
    factor_cols: list[str],
    labels: dict[str, str],
    *,
    model_label: str,
    window_label: str,
    regression_obs: int,
    attribution_obs: int,
    use_rf: bool = False,
    rf: pd.Series | None = None,
) -> dict:
    merged = pd.DataFrame({"y": y})
    for col in factor_cols:
        merged[col] = factors[col]
    if use_rf and rf is not None:
        merged["rf"] = rf
    merged = merged.dropna().sort_index()

    need = regression_obs + attribution_obs
    if len(merged) < need or len(factor_cols) < 2:
        return {
            "available": False,
            "message": f"Insufficient data for {model_label}.",
            "model_label": model_label,
        }

    reg = merged.iloc[-need:-attribution_obs]
    attrib = merged.iloc[-attribution_obs:]

    if use_rf and "rf" in reg.columns:
        y_reg = (reg["y"] - reg["rf"]).values
        y_attrib_total = attrib["y"].sum()
        y_attrib_excess = y_attrib_total - attrib["rf"].sum()
    else:
        y_reg = reg["y"].values
        y_attrib_total = attrib["y"].sum()
        y_attrib_excess = y_attrib_total

    x = reg[factor_cols].values
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    x_design = np.column_stack([np.ones(len(x)), x])
    betas, _, rank, _ = np.linalg.lstsq(x_design, y_reg, rcond=None)
    if rank < x_design.shape[1]:
        return {
            "available": False,
            "message": f"Singular regression for {model_label}.",
            "model_label": model_label,
        }

    alpha, *beta_factors = betas
    explained = float(alpha * len(attrib))
    rows: list[dict] = []
    for col, beta in zip(factor_cols, beta_factors):
        fsum = float(attrib[col].sum())
        contrib = float(beta * fsum)
        explained += contrib
        rows.append(
            {
                "code": col,
                "name": labels.get(col, col),
                "factor_return_pct": round(fsum, 2),
                "beta": round(float(beta), 3),
                "contribution_pct": round(contrib, 2),
            }
        )

    for row in rows:
        row["share_pct"] = (
            round(100 * row["contribution_pct"] / explained, 1) if explained else 0.0
        )

    y_hat = x_design @ betas
    ss_res = float(np.sum((y_reg - y_hat) ** 2))
    ss_tot = float(np.sum((y_reg - np.mean(y_reg)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "available": True,
        "model_label": model_label,
        "window_label": window_label,
        "regression_days": regression_obs,
        "spy_return_pct": round(y_attrib_total, 2),
        "spy_excess_return_pct": round(y_attrib_excess, 2),
        "explained_pct": round(explained, 2),
        "alpha_contribution_pct": round(float(alpha * len(attrib)), 2),
        "unexplained_pct": round(y_attrib_excess - explained, 2),
        "r_squared": round(r_squared, 3),
        "factors": rows,
        "as_of": str(merged.index[-1].date())[:10],
        "spy_return_note": (
            "SPY return is computed on this panel's aligned attribution sample "
            "(rows where all required factors are available), so it may differ across panels."
        ),
    }


def compute_tradeable_attribution(
    data: dict[str, pd.DataFrame],
    regression_days: int = 63,
    attribution_days: int = 21,
) -> dict:
    if "spy" not in data:
        return {"available": False, "message": "SPY required", "id": "tradeable"}

    spy_ret = _spy_daily_returns(data["spy"])
    factors = build_tradeable_factors(data)
    cols = [c for c in factors.columns if c in TRADEABLE_LABELS]
    panel = _run_attribution_panel(
        spy_ret,
        factors,
        cols,
        TRADEABLE_LABELS,
        model_label="Tradeable macro & asset risk premia",
        window_label=f"Last {attribution_days} trading days (~1 month)",
        regression_obs=regression_days,
        attribution_obs=attribution_days,
    )
    panel["id"] = "tradeable"
    panel["title"] = "A. Tradeable Macro & Assets"
    return panel


def compute_sector_attribution(
    data: dict[str, pd.DataFrame],
    regression_days: int = 63,
    attribution_days: int = 21,
) -> dict:
    if "spy" not in data:
        return {"available": False, "message": "SPY required", "id": "sectors"}

    spy_ret = _spy_daily_returns(data["spy"])
    factors = build_sector_factors(data)
    cols = [c for c in factors.columns if c in SECTOR_LABELS]
    panel = _run_attribution_panel(
        spy_ret,
        factors,
        cols,
        SECTOR_LABELS,
        model_label="US sector ETF risk premia",
        window_label=f"Last {attribution_days} trading days (~1 month)",
        regression_obs=regression_days,
        attribution_obs=attribution_days,
    )
    panel["id"] = "sectors"
    panel["title"] = "D. US Sector ETFs"
    panel["note"] = "Healthcare uses XLV (broad medical), not XBI biotech-only."
    return panel


def compute_crr_attribution(
    data: dict[str, pd.DataFrame],
    regression_months: int = 24,
    attribution_months: int = 1,
) -> dict:
    if "spy" not in data:
        return {"available": False, "message": "SPY required", "id": "crr"}

    spy = _series_from_cache(data, "spy")
    spy_m = (spy.resample("ME").last().pct_change() * 100).dropna()
    factors = build_crr_factors(data)
    cols = [c for c in factors.columns if c in CRR_LABELS]
    panel = _run_attribution_panel(
        spy_m,
        factors,
        cols,
        CRR_LABELS,
        model_label="Macro APT proxies (Chen–Roll–Ross style, monthly)",
        window_label=f"Last {attribution_months} month(s); betas from prior {regression_months} months",
        regression_obs=regression_months,
        attribution_obs=attribution_months,
    )
    panel["id"] = "crr"
    panel["title"] = "B. Macro APT (CRR-style)"
    return panel


def compute_fama_french_attribution(
    data: dict[str, pd.DataFrame],
    regression_days: int = 63,
    attribution_days: int = 21,
    force_refresh: bool = False,
) -> dict:
    if "spy" not in data:
        return {"available": False, "message": "SPY required", "id": "fama_french"}

    try:
        ff = fetch_ff5_momentum(force_refresh=force_refresh)
    except Exception as exc:
        return {
            "available": False,
            "message": str(exc),
            "id": "fama_french",
            "model_label": "Fama–French 5-factor + Momentum",
        }

    spy_ret = _spy_daily_returns(data["spy"])
    ff_idx = ff.set_index("date")
    rf = ff_idx["RF"] if "RF" in ff_idx.columns else pd.Series(0.0, index=ff_idx.index)
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"] if c in ff_idx.columns]

    factors = ff_idx[factor_cols]
    panel = _run_attribution_panel(
        spy_ret,
        factors,
        factor_cols,
        FF_LABELS,
        model_label="Fama–French 5-factor + Momentum",
        window_label=f"Last {attribution_days} trading days (~1 month)",
        regression_obs=regression_days,
        attribution_obs=attribution_days,
        use_rf=True,
        rf=rf,
    )
    panel["id"] = "fama_french"
    panel["title"] = "C. Fama–French 5 + Momentum"
    return panel


def compute_factor_attribution(
    data: dict[str, pd.DataFrame],
    regression_days: int = 63,
    attribution_days: int = 21,
    force_refresh: bool = False,
) -> dict:
    """Run all four attribution panels; backward-compatible top-level keys from first available panel."""
    panels = [
        compute_tradeable_attribution(data, regression_days, attribution_days),
        compute_sector_attribution(data, regression_days, attribution_days),
        compute_fama_french_attribution(
            data, regression_days, attribution_days, force_refresh=force_refresh
        ),
    ]
    available = [p for p in panels if p.get("available")]
    result: dict = {
        "available": len(available) > 0,
        "panels": panels,
        "footnote": (
            "Four separate models; R² is not comparable across panels. "
            "Sector and macro factors are marginal associations, not causal. "
            "Not investment advice."
        ),
    }
    if not available:
        result["message"] = "No attribution panel could be computed."
        return result

    primary = available[0]
    for key in (
        "window_label",
        "spy_return_pct",
        "spy_excess_return_pct",
        "r_squared",
        "factors",
        "model_label",
    ):
        if key in primary:
            result[key] = primary[key]
    return result


def main() -> None:
    from src.collect import load_all_from_cache

    data = load_all_from_cache()
    result = compute_factor_attribution(data)
    if not result.get("available"):
        print(result.get("message", "N/A"), file=sys.stderr)
        sys.exit(1)
    for panel in result["panels"]:
        print(f"\n=== {panel.get('title', panel.get('id'))} ===")
        if not panel.get("available"):
            print(f"  Skipped: {panel.get('message')}")
            continue
        print(
            f"  {panel['window_label']} | SPY {panel['spy_return_pct']}% | R² {panel['r_squared']}"
        )
        for row in panel["factors"]:
            print(
                f"    {row['name']}: {row['factor_return_pct']:+.2f}% → {row['contribution_pct']:+.2f}%"
            )


if __name__ == "__main__":
    main()
