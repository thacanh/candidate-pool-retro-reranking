#!/usr/bin/env bash
set -euo pipefail

cd "${DD_BUNDLE_ROOT:-$PWD}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${DD_PYTHON:-/workspace/retro-reranking-py310/bin/python}"
APPROVAL="${DD_APPROVAL:-docs/dd_remote_approval.json}"
PRIMARY="outputs/jcheminform_revision/tuned_primary/conformer_seed_42"
MORGAN="outputs/jcheminform_revision/encoder_controls/morgan_atom"
POOL="outputs/jcheminform_revision/candidate_pools/ws_e_three_pools_filtered_v2"
INDEX="outputs/jcheminform_revision/ws_e_streaming/index"
FEATURES="outputs/jcheminform_revision/ws_e_streaming/pool_features_canonical_identity"
COMMON="outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_reactions.csv"
OUT="outputs/dd_final_round/N4"
ARCHIVES="${DD_ARCHIVE_DIR:-/workspace/dd_n4_archives}"
mkdir -p logs/dd_final_round "$ARCHIVES"

"$PY" workflows/transfer/package_inputs.py verify-extracted --root "$PWD" --manifest BUNDLE_INPUTS.json \
  > logs/dd_final_round/N4_preflight.log 2>&1
"$PY" workflows/transfer/check_morgan_inputs.py >> logs/dd_final_round/N4_preflight.log 2>&1
"$PY" -m pytest -q --import-mode=importlib tests/test_morgan_workflow.py tests/test_transfer_preparation.py \
  > logs/dd_final_round/N4_synthetic_smoke.log 2>&1
"$PY" workflows/transfer/run_morgan_comparison.py \
  --approval-record "$APPROVAL" --output-root "$OUT" \
  --pool-root "$POOL" --index-root "$INDEX" --feature-root "$FEATURES" \
  --products-jsonl "$POOL/products.jsonl" \
  --source-csv data/uspto_smiles.csv --metadata-csv data/uspto_reaction_metadata.csv \
  --primary-freeze "$PRIMARY/freeze/model_selection.json" \
  --g1-freeze "$PRIMARY/g1_20seed/freeze.json" \
  --historical-prediction-root "$PRIMARY/g1_20seed/test_results/predictions" \
  --morgan-control-freeze "$MORGAN/ranking/freeze.json" \
  --morgan-control-root "$MORGAN/ranking/artifacts" \
  --morgan-selection-bundle "$MORGAN/ranking/selection.pkl" \
  --morgan-test-cache "$MORGAN/cache/train_test.pkl" \
  --morgan-old-predictions "$MORGAN/ranking/test_results/predictions" \
  --common-reactions-csv "$COMMON" --device cpu \
  --feature-workers "${DD_FEATURE_WORKERS:-8}" --fit-workers "${DD_FIT_WORKERS:-8}" \
  > logs/dd_final_round/N4_run.log 2>&1
"$PY" workflows/transfer/archive_results.py archive \
  --stage N4 --protocol-id dd-n4-morgan-atom-transfer-v1 \
  --result-dir "$OUT" --archive-dir "$ARCHIVES" --approval-record "$APPROVAL" \
  --required transfer_effects.csv --required historical/test_results/manifest.json \
  --required aizynthfinder_only/test_results/manifest.json \
  --required localretro_only/test_results/manifest.json \
  --required merged/test_results/manifest.json \
  --log-file logs/dd_final_round/N4_run.log \
  > logs/dd_final_round/N4_archive.log 2>&1
"$PY" workflows/transfer/archive_results.py verify --receipt "$ARCHIVES/N4.receipt.json" \
  >> logs/dd_final_round/N4_archive.log 2>&1
echo "N4 COMPLETE: $ARCHIVES/N4.zip"
