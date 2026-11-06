#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${S4_PYTHON:-/workspace/retro-reranking-py310/bin/python}
case "${1:-status}" in
  setup)
    uv python install 3.10.14
    if [ ! -x "$PY" ]; then uv venv --python 3.10.14 /workspace/retro-reranking-py310; fi
    uv pip install --python "$PY" 'torch==2.2.2+cpu' --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python "$PY" -r workflows/transfer/requirements-cpu.txt
    "$PY" workflows/transfer/search_hyperparameters.py verify
    "$PY" -m pytest -q tests/test_hyperparameter_search.py
    ;;
  start)
    [ -x "$PY" ] || { echo 'Run setup first'; exit 1; }
    "$PY" workflows/transfer/search_hyperparameters.py verify
    workers=${2:-6}
    [[ "$workers" =~ ^[0-9]+$ ]] && ((workers>=1 && workers<=32)) || { echo 'workers must be 1..32'; exit 1; }
    if tmux has-session -t dd_s4 2>/dev/null; then echo 'dd_s4 is already running'; exit 1; fi
    [ ! -f S4_RUNNING.lock ] || { echo 'Lock exists: inspect PID/host before restarting'; cat S4_RUNNING.lock; exit 1; }
    mkdir -p logs
    printf -v cmd 'cd %q && export S4_PYTHON=%q && bash workflows/transfer/search_hyperparameters.sh foreground %q' "$ROOT" "$PY" "$workers"
    tmux new-session -d -s dd_s4 "$cmd"
    echo 'Started in tmux dd_s4. Follow with: bash workflows/transfer/search_hyperparameters.sh status'
    ;;
  foreground)
    mkdir -p logs
    log="logs/search_$(date -u +%Y%m%d_%H%M%S).log"
    set +e
    "$PY" workflows/transfer/search_hyperparameters.py run --workers "${2:-6}" > "$log" 2>&1
    rc=$?
    echo "EXIT_CODE=$rc" >> "$log"
    exit "$rc"
    ;;
  status)
    "$PY" workflows/transfer/search_hyperparameters.py status
    if tmux has-session -t dd_s4 2>/dev/null; then echo 'tmux: running'; else echo 'tmux: not running'; fi
    find archives -maxdepth 1 -name '*.receipt.json' -print 2>/dev/null || true
    ;;
  *) echo 'Usage: setup | start [workers] | foreground [workers] | status'; exit 2 ;;
esac
