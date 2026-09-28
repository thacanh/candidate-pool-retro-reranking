#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${S4_PYTHON:-/workspace/dd-s4-py310/bin/python}
case "${1:-status}" in
  setup)
    uv python install 3.10.14
    if [ ! -x "$PY" ]; then uv venv --python 3.10.14 /workspace/dd-s4-py310; fi
    uv pip install --python "$PY" 'torch==2.2.2+cpu' --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python "$PY" -r docs/dd_remote_requirements.txt
    "$PY" scripts/dd_n3_s4.py verify
    "$PY" -m pytest -q tests/test_dd_n3_s4.py
    ;;
  start)
    [ -x "$PY" ] || { echo 'Run setup first'; exit 1; }
    "$PY" scripts/dd_n3_s4.py verify
    workers=${2:-6}
    [[ "$workers" =~ ^[0-9]+$ ]] && ((workers>=1 && workers<=32)) || { echo 'workers must be 1..32'; exit 1; }
    if tmux has-session -t dd_s4 2>/dev/null; then echo 'dd_s4 is already running'; exit 1; fi
    [ ! -f S4_RUNNING.lock ] || { echo 'Lock exists: inspect PID/host before restarting'; cat S4_RUNNING.lock; exit 1; }
    mkdir -p logs
    printf -v cmd 'cd %q && export S4_PYTHON=%q && bash scripts/dd_n3_s4.sh foreground %q' "$ROOT" "$PY" "$workers"
    tmux new-session -d -s dd_s4 "$cmd"
    echo 'Started in tmux dd_s4. Follow with: bash scripts/dd_n3_s4.sh status'
    ;;
  foreground)
    mkdir -p logs
    log="logs/search_$(date -u +%Y%m%d_%H%M%S).log"
    set +e
    "$PY" scripts/dd_n3_s4.py run --workers "${2:-6}" > "$log" 2>&1
    rc=$?
    echo "EXIT_CODE=$rc" >> "$log"
    exit "$rc"
    ;;
  status)
    "$PY" scripts/dd_n3_s4.py status
    if tmux has-session -t dd_s4 2>/dev/null; then echo 'tmux: running'; else echo 'tmux: not running'; fi
    find archives -maxdepth 1 -name '*.receipt.json' -print 2>/dev/null || true
    ;;
  *) echo 'Usage: setup | start [workers] | foreground [workers] | status'; exit 2 ;;
esac
