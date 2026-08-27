#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${AIZYNTH_ENV_ROOT:-/workspace/aizynth-revision-py310}"
PYTHON="$ENV_ROOT/bin/python"

cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required by this portable Vast.ai setup." >&2
  exit 1
fi

echo "Creating pinned Python 3.10 AiZynthFinder environment at $ENV_ROOT"
uv python install 3.10.14
uv venv --clear --seed --python 3.10.14 "$ENV_ROOT"
uv pip install --python "$PYTHON" --require-hashes \
  -r aizynth_jobs/requirements-aizynth-linux-py310.lock

"$PYTHON" -m pip check
mkdir -p aizynth_jobs/runtime_lock
uv pip freeze --python "$PYTHON" > aizynth_jobs/runtime_lock/pip-resolved-linux.txt

export AIZYNTH_PYTHON="$PYTHON"
bash aizynth_jobs/linux/CHECK_MACHINE.sh

echo
echo "AIZYNTH ENVIRONMENT READY: $PYTHON"
