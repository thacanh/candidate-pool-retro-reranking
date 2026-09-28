# Frozen numerical outputs

This directory contains only compact, model-free numerical artifacts used by
the manuscript and figure code.

- `historical_anchor/` is a read-only numerical anchor from the same study.
- `transfer_analysis/` contains the candidate-pool transfer analyses.
- `transfer_analysis/final_round/` includes N1--N5, corrected headroom,
  full-grid adaptation and prior-gap summaries. These are secondary analyses;
  they do not replace the frozen primary comparison.

CSV payloads retain original bytes. Final-round JSON manifests are public
copies with local user-path prefixes removed, not byte-identical originals.
`data/provenance/final_round_artifact_map.json` records both hashes and the
path-only transformation. Original scientific artifacts remain unchanged.

Raw reactions, candidate-level prediction archives, checkpoints, and embedding
caches are intentionally excluded from Git. See `docs/public_release.md`.
