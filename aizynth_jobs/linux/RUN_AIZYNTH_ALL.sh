#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${AIZYNTH_PYTHON:-/workspace/aizynth-revision-py310/bin/python}"
OUTPUT_ROOT="${AIZYNTH_OUTPUT_ROOT:-outputs/jcheminform_revision/candidate_pools/aizynth_onepass}"

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
echo "ONE-PASS A-CAP10-REPRO + A-CAP50 | workers=$WORKERS"
echo "The same raw Top-50 stream supplies both caps; cap-50 remains blocked unless cap-10 matches."

set +e
"$PYTHON" -m rerank.data.generate_aizynth_candidate_pools generate \
  --output-root "$OUTPUT_ROOT" \
  --workers "$WORKERS" \
  --chunk-size 128
RUN_RC=$?
set -e

STAMP="$(date -u +%Y%m%d_%H%M%S)"
ARCHIVE="/workspace/aizynth_cap10_cap50_results_${STAMP}.tar.gz"

if [[ -f "$OUTPUT_ROOT/generation_manifest.json" ]]; then
  tar -czf "$ARCHIVE" \
    --exclude='./chunks' \
    -C "$OUTPUT_ROOT" .
  sha256sum "$ARCHIVE"
  du -h "$ARCHIVE"
  echo "RESULT ARCHIVE: $ARCHIVE"
else
  echo "No final manifest exists yet; resume the same command after fixing the reported error." >&2
fi

if [[ "$RUN_RC" -eq 2 ]]; then
  echo "CAP-10 REPRODUCTION MISMATCH: downstream cap-50 use is blocked." >&2
  echo "Download the result archive for discrepancy review; do not run embeddings." >&2
fi
exit "$RUN_RC"
