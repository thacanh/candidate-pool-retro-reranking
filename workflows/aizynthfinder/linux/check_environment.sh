#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${AIZYNTH_PYTHON:-/workspace/aizynthfinder-py310/bin/python}"

cd "$ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: $PYTHON" >&2
  exit 1
fi

"$PYTHON" workflows/aizynthfinder/verify_machine.py \
  --bundle-root "$ROOT" \
  --output workflows/aizynthfinder/runtime_lock/machine_check.json

echo "MACHINE CHECK PASSED"
