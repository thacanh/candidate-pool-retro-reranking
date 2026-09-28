#!/usr/bin/env bash
set -euo pipefail
ROOT=/workspace/dd_n3
SELF="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
cd "$ROOT"
if [[ "${1:-}" == _run ]]; then
  export DD_PYTHON=/workspace/dd-final-py310/bin/python
  export DD_FIT_WORKERS=12 PYTHONUNBUFFERED=1
  export PYTHONPATH="$ROOT/src"
  set -o noclobber
  exec > logs/dd_final_round/N3_master.log 2>&1
  trap 'rc=$?; echo "$rc" > logs/dd_final_round/N3.exit; echo "JOB_EXIT_CODE=$rc"' EXIT
  echo 'Resuming N3-S1 with audited identical-input tie handling; existing Python environment reused.'
  bash scripts/run_dd_n3_remote.sh
else
  [[ -f .patch_backups/n3_s1_identical_input_ties_20260923/patch_audit.json ]] || { echo 'Apply patch first'; exit 1; }
  [[ ! -e logs/dd_final_round/N3_master.log ]] || { echo 'Existing run log; inspect status'; exit 1; }
  printf -v COMMAND '%q ' bash "$SELF" _run
  tmux new-session -d -s dd-n3 "$COMMAND"
  echo 'N3 resumed in tmux dd-n3. Use the same status command.'
fi
