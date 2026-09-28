#!/usr/bin/env bash
set -euo pipefail
cd "${DD_BUNDLE_ROOT:-$PWD}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
PY="${DD_PYTHON:-/workspace/dd-final-py310/bin/python}"
mkdir -p logs/dd_final_round
"$PY" scripts/dd_pack_remote.py verify-extracted --root "$PWD" --manifest BUNDLE_INPUTS.json
"$PY" scripts/dd_n3_preflight.py > logs/dd_final_round/N3_preflight.log 2>&1
"$PY" -m pytest -q --import-mode=importlib tests/test_dd_n3_release.py \
  > logs/dd_final_round/N3_synthetic_smoke.log 2>&1
"$PY" scripts/dd_n3_pipeline.py --workers "${DD_FIT_WORKERS:-6}" \
  --archives "${DD_ARCHIVE_DIR:-/workspace/dd_n3_archives}" \
  --approval "${DD_APPROVAL:-docs/dd_remote_approval.json}"
