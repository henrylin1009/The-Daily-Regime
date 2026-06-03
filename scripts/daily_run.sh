#!/usr/bin/env bash
# Daily macro brief — use with cron or launchd.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="$ROOT"
python run.py --notify 2>&1 | tee -a "$ROOT/output/cron.log"
