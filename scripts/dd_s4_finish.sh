#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=${S4_PYTHON:-/workspace/dd-s4-py310/bin/python}
case "${1:-status}" in
  check)
    "$PY" scripts/dd_s4_finish.py verify
    "$PY" -m pytest -q tests/test_dd_s4_finish.py
    ;;
  start)
    workers=${2:-8}
    [[ "$workers" =~ ^[0-9]+$ ]] && ((workers>=1 && workers<=16)) || { echo 'Use 1--16 workers'; exit 1; }
    "$PY" scripts/dd_s4_finish.py verify
    if tmux has-session -t dd_s4_finish 2>/dev/null; then echo 'Already running'; exit 1; fi
    [ ! -f FINISH_RUNNING.lock ] || { echo 'Lock exists; inspect PID before resuming'; cat FINISH_RUNNING.lock; exit 1; }
    printf -v cmd 'cd %q && export S4_PYTHON=%q && bash scripts/dd_s4_finish.sh foreground %q' "$ROOT" "$PY" "$workers"
    tmux new-session -d -s dd_s4_finish "$cmd"
    echo 'Started in tmux dd_s4_finish; logs are saved, no streaming required.'
    ;;
  foreground)
    mkdir -p logs
    log="logs/finish_$(date -u +%Y%m%d_%H%M%S).log"
    set +e
    "$PY" scripts/dd_s4_finish.py run --workers "${2:-8}" > "$log" 2>&1
    rc=$?
    echo "EXIT_CODE=$rc" >> "$log"
    exit "$rc"
    ;;
  status)
    "$PY" scripts/dd_s4_finish.py status
    if tmux has-session -t dd_s4_finish 2>/dev/null; then echo 'tmux: running'; else echo 'tmux: not running'; fi
    find archives -maxdepth 1 -name '*.receipt.json' -print 2>/dev/null || true
    ;;
  *) echo 'Usage: check | start [workers] | status'; exit 2 ;;
esac
