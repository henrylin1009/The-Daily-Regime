# Macro Data Boundary (MECE)

This document defines clear ownership across Layer 1 / Layer 2 / Layer 3.

## Layer 1 — Structural Regime

- **Purpose**: Define slow-moving macro environment.
- **Primary Script**: `global_regime.py`
- **Primary Output**: `output/global_regime_YYYY-MM-DD.html`, `output/global_regime_data_YYYY-MM-DD.json`

### Core Indicators (Required)

- **Central Bank Cycle (monthly/weekly)**
  - Policy rates: US, Japan, Europe, UK, China, Taiwan
  - Balance sheets: Fed, ECB, BOJ, PBOC (PBOC can be missing and flagged)
- **Positioning Structure (weekly)**
  - CFTC financial COT: JPY, EUR, GBP, AUD, 10Y UST, Gold
- **External / Flow Structure (monthly)**
  - TIC holdings: Japan, China, UK, Taiwan, All foreign
  - China FX reserves (monthly)
- **Current Account Structure (monthly/quarterly)**
  - US, Japan, Germany, China
- **Liquidity & Risk Premia (weekly/daily aggregated)**
  - T3MFF / TEDRATE proxy
  - EM risk proxy (EMB vs IEF)

### Optional Indicators

- Fed funds futures implied rate shift (ZQ=F derived)
- Additional policy proxy series fallback values

### Explicit Exclusions

- Daily Taiwan 5d/20d flow momentum values (owned by Layer 2)
- Daily FX momentum tables (owned by Layer 2)

## Layer 2 — Tactical State

- **Purpose**: Describe what is happening now in market momentum and pressure.
- **Primary Scripts**: `run.py`, `flow_run.py`
- **Primary Outputs**:
  - `output/brief_YYYY-MM-DD.html`
  - `output/flow_brief_YYYY-MM-DD.html`
  - `output/brief_data_YYYY-MM-DD.json`
  - `output/flow_data_YYYY-MM-DD.json`

### Core Indicators (Required)

- **Macro Brief (`run.py`)**
  - Headline risk signals and key indicator table
  - Historical match context
- **Flow Brief (`flow_run.py`)**
  - Cross-country yield spread matrix
  - FX 1m/3m momentum
  - Taiwan 5d/20d foreign flow (fast flow)
  - Japan carry watch (FX + TIC MoM)
  - China capital watch (CNH + reserves MoM)

### Optional Indicators

- Extended attribution sub-panels
- Additional cross-asset tactical proxies

## Layer 3 — Synthesis

- **Purpose**: Interpret Layer 2 signals under Layer 1 regime context.
- **Primary Script**: `synthesis.py`
- **Primary Outputs**:
  - `output/synthesis_YYYY-MM-DD.html`
  - `output/synthesis_data_YYYY-MM-DD.json`

### Core Inputs

- `output/global_regime_data_YYYY-MM-DD.json`
- `output/brief_data_YYYY-MM-DD.json`
- `output/flow_data_YYYY-MM-DD.json`

### Core Outputs

- Headline
- Structural context
- Signal interpretation
- Convergence/divergence
- Watch list

## Ownership Summary

- `global_regime.py` owns **slow structural** fields only.
- `run.py` + `flow_run.py` own **daily tactical** fields.
- `synthesis.py` owns **interpretation only**, not new raw indicator generation.
