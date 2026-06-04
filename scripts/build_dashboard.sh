#!/usr/bin/env bash
# Full dashboard build — synthesis always calls DeepSeek (no --skip-llm).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
DATE="${1:-$(TZ=Asia/Taipei date +%Y-%m-%d)}"

echo "=== Build dashboard for ${DATE} (Asia/Taipei) ==="
"$PY" flow_run.py --date "$DATE"
"$PY" run.py --date "$DATE" --skip-hmm
"$PY" global_regime.py --force-refresh --date "$DATE"
"$PY" synthesis.py --date "$DATE"
"$PY" layer_pages.py --date "$DATE"
echo "=== Done. Open http://127.0.0.1:8080/file/layer3.html ==="
