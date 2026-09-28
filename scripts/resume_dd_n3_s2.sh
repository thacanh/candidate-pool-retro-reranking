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
  echo 'N3 resumes at S2; completed S1 archive will be verified and retained.'
  bash .patch_backups/n3_merged_path_20260923/resume_s2_pipeline.sh
else
  [[ -f .patch_backups/n3_merged_path_20260923/audit.json ]] || { echo 'Apply repair first'; exit 1; }
  [[ ! -e logs/dd_final_round/N3_master.log ]] || { echo 'Existing run log; inspect status'; exit 1; }
  printf -v COMMAND '%q ' bash "$SELF" _run
  tmux new-session -d -s dd-n3 "$COMMAND"
  echo 'N3 S2/S3 running in tmux dd-n3; use the existing status command.'
fi
