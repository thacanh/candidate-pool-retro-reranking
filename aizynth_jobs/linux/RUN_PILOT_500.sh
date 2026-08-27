#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${AIZYNTH_PYTHON:-/workspace/aizynth-revision-py310/bin/python}"
OUTPUT_ROOT="outputs/jcheminform_revision/candidate_pools/aizynth_pilot_500"

cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export CUDA_VISIBLE_DEVICES=""

bash aizynth_jobs/linux/CHECK_MACHINE.sh

WORKERS="${AIZYNTH_WORKERS:-$($PYTHON aizynth_jobs/recommend_workers.py)}"
echo "TIMING PILOT ONLY | products=500 | workers=$WORKERS | no scientific gate"

"$PYTHON" -m rerank.data.generate_aizynth_candidate_pools generate \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --chunk-size 50 \
  --max-products 500

cat "$OUTPUT_ROOT/pilot_summary.json"
