#!/usr/bin/env bash
set -euo pipefail

cd "${DD_BUNDLE_ROOT:-$PWD}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY="${DD_PYTHON:-/workspace/retro-reranking-py310/bin/python}"
APPROVAL="${DD_APPROVAL:-docs/dd_remote_approval.json}"
ROOT="outputs/jcheminform_revision/tuned_primary/conformer_seed_42"
G1="$ROOT/g1_20seed"
K1="outputs/digital_discovery_round_jk/k1/aizynthfinder_only"
FEATURE="outputs/jcheminform_revision/ws_e_streaming/pool_features_canonical_identity/aizynthfinder_only/manifest.json"
POOL="outputs/jcheminform_revision/candidate_pools/ws_e_three_pools_filtered_v2"
COMMON="outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_reactions.csv"
OUT="outputs/dd_final_round/N2"
ARCHIVES="${DD_ARCHIVE_DIR:-/workspace/dd_n2_archives}"
mkdir -p logs/dd_final_round "$ARCHIVES"

"$PY" workflows/transfer/package_inputs.py verify-extracted --root "$PWD" --manifest BUNDLE_INPUTS.json \
  > logs/dd_final_round/N2_preflight.log 2>&1
"$PY" -m rerank.experiments.run_cross_inference \
  --approval-record "$APPROVAL" --analysis-plan docs/analysis_plan.md \
  --historical-freeze "$G1/freeze.json" --historical-model-root "$G1" \
  --historical-cache outputs/jcheminform_revision/conformers/seed_42/features/official_3d_prior_schema1.pkl \
  --historical-prediction-root "$G1/test_results/predictions" \
  --historical-test-manifest "$G1/test_results/manifest.json" \
  --aiz-freeze "$K1/models/freeze.json" --aiz-model-root "$K1/models" \
  --aiz-feature-freeze "$FEATURE" \
  --aiz-prediction-jsonl "$K1/test_results/reaction_ranks.jsonl" \
  --aiz-test-manifest "$K1/test_results/manifest.json" \
  --aiz-selection-manifest "$K1/selection/manifest.json" \
  --common-reactions-csv "$COMMON" \
  --source-csv data/uspto_smiles.csv --metadata-csv data/uspto_reaction_metadata.csv \
  --products-jsonl "$POOL/products.jsonl" --device cpu --output-dir "$OUT" \
  > logs/dd_final_round/N2_run.log 2>&1
"$PY" workflows/transfer/archive_results.py archive \
  --stage N2 --protocol-id dd-n2-cross-inference-v1 \
  --result-dir "$OUT" --archive-dir "$ARCHIVES" --approval-record "$APPROVAL" \
  --required cell_ranks.npz --required cross_inference_effects.csv --required clipping.csv \
  --log-file logs/dd_final_round/N2_run.log \
  > logs/dd_final_round/N2_archive.log 2>&1
"$PY" workflows/transfer/archive_results.py verify --receipt "$ARCHIVES/N2.receipt.json" \
  >> logs/dd_final_round/N2_archive.log 2>&1
echo "N2 COMPLETE: $ARCHIVES/N2.zip"
