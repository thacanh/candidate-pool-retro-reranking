#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_EXE="${CHEMFORMER_PYTHON:-/workspace/chemformer-f1-py310/bin/python}"
export PYTHONPATH="$ROOT/src:$ROOT/chemformer_jobs:$ROOT/assets/aizynthmodels_official:$ROOT/assets/chemformer_checkpoint_source"

bash chemformer_jobs/linux/CHECK_MACHINE.sh

INPUT="outputs/jcheminform_revision/f1_roundtrip/prepared/chemformer_input.tsv"
RUNTIME="outputs/jcheminform_revision/f1_roundtrip/runtime"
MODEL_OUTPUT="$RUNTIME/chemformer_beam5.tsv"
RESULTS="outputs/jcheminform_revision/f1_roundtrip/results"

echo "PHASE 1/2: Chemformer beam-5 forward prediction (4,762 unique inputs)"
if [[ ! -e "$MODEL_OUTPUT" && ! -e "$RUNTIME/inference_manifest.json" ]]; then
  "$PYTHON_EXE" chemformer_jobs/run_inference.py \
    --input "$INPUT" \
    --checkpoint "$RUNTIME/last_aizynthmodels.ckpt" \
    --vocabulary assets/chemformer_official/bart_vocab_downstream.json \
    --output "$MODEL_OUTPUT" \
    --manifest "$RUNTIME/inference_manifest.json" \
    --device cuda --batch-size "${CHEMFORMER_BATCH_SIZE:-32}" \
    --workers "${CHEMFORMER_WORKERS:-4}"
else
  echo "Existing inference artifacts found; refusing a second scientific inference."
fi

echo "PHASE 2/2: frozen round-trip metrics and paired inference"
if [[ ! -e "$RESULTS/manifest.json" ]]; then
  "$PYTHON_EXE" -m rerank.experiments.run_forward_roundtrip evaluate \
    --prepare-dir outputs/jcheminform_revision/f1_roundtrip/prepared \
    --chemformer-output "$MODEL_OUTPUT" \
    --result-dir "$RESULTS"
else
  echo "Existing F1 result manifest found; refusing a second evaluation."
fi

timestamp=$(date -u +%Y%m%d_%H%M%S)
archive="/workspace/chemformer_f1_results_${timestamp}.tar.gz"
tar -czf "$archive" "$RUNTIME/inference_manifest.json" "$MODEL_OUTPUT" "$RESULTS" logs/chemformer
sha256sum "$archive"
du -h "$archive"
echo "RESULT ARCHIVE: $archive"
