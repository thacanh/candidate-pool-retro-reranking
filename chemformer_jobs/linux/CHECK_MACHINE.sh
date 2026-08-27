#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_EXE="${CHEMFORMER_PYTHON:-/workspace/chemformer-f1-py310/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/chemformer_jobs:$ROOT/assets/aizynthmodels_official:$ROOT/assets/chemformer_checkpoint_source"

"$PYTHON_EXE" chemformer_jobs/verify_machine.py \
  --bundle-root "$ROOT" --require-cuda

mkdir -p outputs/jcheminform_revision/f1_roundtrip/runtime logs/chemformer
"$PYTHON_EXE" chemformer_jobs/migrate_checkpoint.py \
  --source assets/chemformer_forward/last.ckpt \
  --output outputs/jcheminform_revision/f1_roundtrip/runtime/last_aizynthmodels.ckpt \
  --manifest outputs/jcheminform_revision/f1_roundtrip/runtime/checkpoint_migration.json \
  --legacy-source assets/chemformer_checkpoint_source \
  > logs/chemformer/checkpoint_migration.log

"$PYTHON_EXE" chemformer_jobs/verify_model.py \
  --checkpoint outputs/jcheminform_revision/f1_roundtrip/runtime/last_aizynthmodels.ckpt \
  --vocabulary assets/chemformer_official/bart_vocab_downstream.json \
  --device cuda \
  | tee logs/chemformer/model_smoke.json

echo "MACHINE, MIGRATION AND MODEL-LOAD CHECKS PASSED"
