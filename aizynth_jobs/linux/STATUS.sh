#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${AIZYNTH_PYTHON:-/workspace/aizynth-revision-py310/bin/python}"
OUTPUT_ROOT="${AIZYNTH_OUTPUT_ROOT:-outputs/jcheminform_revision/candidate_pools/aizynth_onepass}"

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
"$PYTHON" -m rerank.data.generate_aizynth_candidate_pools status --output-root "$OUTPUT_ROOT"

if pgrep -f '[g]enerate_aizynth_candidate_pools.py generate' >/dev/null; then
  echo "process_alive=true"
else
  echo "process_alive=false"
fi
