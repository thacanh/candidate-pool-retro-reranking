#!/usr/bin/env bash
set -euo pipefail
cd /workspace/dd_n4
if [[ "${1:-}" != _run ]]; then
  [[ -f .patch_backups/n4_frozen_reuse_20260924/patch_audit.json ]] || { echo 'Apply patch first'; exit 1; }
  [[ ! -e logs/dd_final_round/N4_resume_master.log ]] || { echo 'Existing continuation log: inspect it first'; exit 1; }
  tmux new-session -d -s dd-n4 'bash /workspace/dd_n4/scripts/resume_dd_n4_after_patch.sh _run'
  echo 'Evaluation resumed in tmux dd-n4. No feature generation or training.'
  exit 0
fi
export PYTHONPATH="$PWD/src" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/workspace/dd-final-py310/bin/python
set -o noclobber
exec > logs/dd_final_round/N4_resume_master.log 2>&1
trap 'rc=$?; echo "$rc" >| logs/dd_final_round/N4.exit; echo "JOB_EXIT_CODE=$rc"' EXIT
"$PY" scripts/dd_pack_remote.py verify-extracted --root "$PWD" --manifest BUNDLE_INPUTS.json
"$PY" -m pytest -q --import-mode=importlib tests/test_dd_n4_reuse.py
"$PY" scripts/resume_dd_n4_evaluation.py
"$PY" scripts/dd_archive_stage.py archive --stage N4 --protocol-id dd-n4-morgan-atom-transfer-v1 \
  --result-dir outputs/dd_final_round/N4 --archive-dir /workspace/dd_n4_archives \
  --approval-record docs/dd_remote_approval.json \
  --required transfer_effects.csv --required historical/test_results/manifest.json \
  --required aizynthfinder_only/test_results/manifest.json \
  --required localretro_only/test_results/manifest.json --required merged/test_results/manifest.json \
  --required replay_audit/continuation.json --log-file logs/dd_final_round/N4_run.log
"$PY" scripts/dd_archive_stage.py verify --receipt /workspace/dd_n4_archives/N4.receipt.json
echo 'N4 COMPLETE: /workspace/dd_n4_archives/N4.zip and N4.receipt.json'
