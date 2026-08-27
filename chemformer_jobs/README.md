# F1 Chemformer forward round-trip job

This job runs the prespecified `f1-chemformer-forward-roundtrip-v1` control on
4,762 deduplicated precursor/product inputs derived from the frozen 20-seed G1
official-test predictions. It never trains or selects a model.

The Figshare checkpoint is verified byte-for-byte. The runtime applies only the
official Chemformer README migration from `vocab_size` to `vocabulary_size` and
records a logical hash proving that all 188 state tensors are unchanged. The
runtime implementation is the official successor, AiZynthModels, at the pinned
commit in `data/revision_external_assets.json`.

On a fresh Linux CUDA instance with `uv`:

```bash
cd /workspace/<extracted-bundle>
bash chemformer_jobs/linux/SETUP_CHEMFORMER_ENV.sh
export CHEMFORMER_PYTHON=/workspace/chemformer-f1-py310/bin/python
tmux new -s chemformer
bash chemformer_jobs/linux/RUN_F1_ALL.sh
```

Detach from tmux with `Ctrl+B`, release, then `D`. The retained archive is
written to `/workspace/chemformer_f1_results_<UTC timestamp>.tar.gz`.

The `.txt` file is the reviewed top-level specification. The Linux/Python 3.10
`.lock` resolves all 65 packages, pins CUDA 12.1 PyTorch and requires artifact
SHA-256 hashes. Setup also writes the actual installed snapshot to
`logs/chemformer/environment-resolved.txt`.
