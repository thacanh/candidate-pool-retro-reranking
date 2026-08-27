#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_ROOT="${CHEMFORMER_ENV_ROOT:-/workspace/chemformer-f1-py310}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is required; use a Vast.ai PyTorch template with uv."
  exit 2
fi

uv python install 3.10.14
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  uv venv --python 3.10.14 "$ENV_ROOT"
fi
uv pip install --python "$ENV_ROOT/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  --index-strategy unsafe-best-match \
  --require-hashes \
  -r "$ROOT/chemformer_jobs/linux/requirements-chemformer-linux-py310.lock"
uv pip check --python "$ENV_ROOT/bin/python"

mkdir -p "$ROOT/logs/chemformer"
uv pip freeze --python "$ENV_ROOT/bin/python" \
  > "$ROOT/logs/chemformer/environment-resolved.txt"

export CHEMFORMER_PYTHON="$ENV_ROOT/bin/python"
bash "$ROOT/chemformer_jobs/linux/CHECK_MACHINE.sh"
echo "CHEMFORMER ENVIRONMENT READY: $CHEMFORMER_PYTHON"
