#!/usr/bin/env bash
set -euo pipefail

cd "${DD_BUNDLE_ROOT:-$PWD}"
uv python install 3.10.14
uv venv --python 3.10.14 /workspace/retro-reranking-py310
uv pip install --python /workspace/retro-reranking-py310/bin/python \
  'torch==2.2.2+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /workspace/retro-reranking-py310/bin/python \
  -r workflows/transfer/requirements-cpu.txt
export PYTHONPATH="$PWD/src"
/workspace/retro-reranking-py310/bin/python -c \
  'import numpy,pandas,scipy,rdkit,torch; print("DD CPU runtime ready",torch.__version__)'
/workspace/retro-reranking-py310/bin/python workflows/transfer/package_inputs.py verify-extracted \
  --root "$PWD" --manifest BUNDLE_INPUTS.json
