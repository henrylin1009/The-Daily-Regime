# Data Quality & Completeness Rules

## Missing Value Rule

- Render missing values as `—`.
- Never replace missing values with `0` unless `0` is a true observed value.
- All percent and bp fields must preserve missing states and avoid synthetic defaults.

## Core Completeness Threshold

Each layer reports:

- `core_coverage_ratio = available_core_fields / total_core_fields`
- `missing_critical_fields` list (max 3 shown in UI)

### Degraded Mode

- Enter degraded mode when:
  - `core_coverage_ratio < 0.80` (more than 20% missing), or
  - any required source group is fully unavailable (e.g., all COT contracts missing).

### Layer Severity Guidance

- **Layer 1 degraded**: show warning in structural report; synthesis should still run but mark confidence reduced.
- **Layer 2 degraded**: keep tactical page output, but flag missing tactical inputs.
- **Layer 3 degraded**: allow synthesis fallback text when critical inputs are missing.

## LLM Fallback Standard

For any Gemini-dependent narrative field (API outage, quota limit, parse failure, skip mode):

- Use exactly: `LLM analysis not avaliable`

No alternative fallback text should be emitted in rendered content.

## Reliability Expectations by Source

- **FRED**: must include fallback series where possible and preserve as-of timestamps.
- **COT**: parsing should tolerate minor column naming differences and code format differences.
- **TIC**: parser must handle line-label variation for the all-foreign aggregate.
- **yfinance**: when source fails, avoid silent substitution that changes semantic sign or units.

## Daily Quality Report (Target)

A daily script should output:

- per-layer `core_coverage_ratio`
- degraded mode flags
- top missing critical fields (max 3 each layer)
- execution timestamp and source snapshot dates
