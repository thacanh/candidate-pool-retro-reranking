#!/usr/bin/env bash
set -euo pipefail

cd "${DD_BUNDLE_ROOT:-$PWD}"
uv python install 3.10.14
uv venv --python 3.10.14 /workspace/dd-final-py310
uv pip install --python /workspace/dd-final-py310/bin/python \
  'torch==2.2.2+cpu' --index-url https://download.pytorch.org/whl/cpu
uv pip install --python /workspace/dd-final-py310/bin/python \
  -r docs/dd_remote_requirements.txt
export PYTHONPATH="$PWD/src"
/workspace/dd-final-py310/bin/python -c \
  'import numpy,pandas,scipy,rdkit,torch; print("DD CPU runtime ready",torch.__version__)'
/workspace/dd-final-py310/bin/python scripts/dd_pack_remote.py verify-extracted \
  --root "$PWD" --manifest BUNDLE_INPUTS.json
