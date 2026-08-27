# Candidate-pool dependence in retrosynthesis reranking

This repository contains the code and frozen numerical summaries for the
*Digital Discovery* study **Candidate-pool dependence of
molecular-representation reranking in single-step retrosynthesis**.

The study separates three questions that are often conflated in reranking
benchmarks:

1. whether a candidate generator covers the recorded reference reaction;
2. how well a ranker orders candidates within a covered pool; and
3. whether a representation-assisted ranking gain transfers when the candidate
   generator or pool construction changes.

The active manuscript is maintained privately in Overleaf and is intentionally
excluded from this public code repository while editorial revisions continue.

## Main frozen result

On the historical cap-10 pool, 3,985 of 5,004 official-test reactions are
covered. Across paired training seeds 42--61, adding the Uni-Mol-derived scalar
features improves conditional Top-1 by 0.00670 (95% seed-marginal interval
0.00112--0.01228) and MRR by 0.00485 (0.00173--0.00790).

Alternative AiZynthFinder-only, LocalRetro-only, and merged pools increase
coverage, but their matched representation-assisted effects are compatible
with zero. Direct paired inference on 3,814 commonly covered reactions finds
negative transfer-loss intervals for Top-1 and MRR in all three alternative
pools.

## Public release layout

The clean release produced by
`python -m rerank.data.build_public_release` has this structure:

```text
src/rerank/                 Installable analysis and experiment package
tests/                      Unit, protocol, pairing, and release checks
docs/                       Prespecification, provenance, and release guide
data/provenance/            Dataset, model, and generator identity records
outputs/historical_anchor/  Small read-only historical numerical summaries
outputs/transfer_analysis/  Small transfer-analysis CSV/JSON artifacts
release_manifest.json       SHA256 and size of every released payload file
checksums.sha256             Transport checksums for the complete release
```

The research worktree retains larger caches and historical storage labels for
provenance. Those local paths are not separate publications and are not copied
into the clean Git repository.

## Installation

Python 3.10 is the validated version for the core environment.

```bash
conda env create -f environment-revision.yml
conda activate retrosynthesis-revision-benchmark-py310
python -m pip install -e .
```

The core environment is deliberately separate from AiZynthFinder, LocalRetro,
GROVER, Chemformer, and RXN-EBM. Their pinned identities, checkpoints, and
environment decisions are recorded in `docs/analysis_plan.md`, the provenance
JSON files, and the corresponding job directories.

## Rebuild the numerical figures

From either the research worktree or the clean public release:

```bash
python -m rerank.figures.plot_digital_discovery_figures \
  --repo-root . \
  --output-dir figures_rebuilt
```

The plotting code prefers the neutral public artifact paths and falls back to
the read-only local provenance paths when run in the research worktree.

## Validation

```bash
python -m pytest -q
```

Some integration tests require non-redistributable datasets, third-party model
weights, or heavyweight local caches. The release guide identifies the compact
test gate that runs using only public payload files.

## Data and heavyweight artifacts

The clean Git repository intentionally excludes raw USPTO reaction files,
third-party checkpoints, candidate-level prediction archives, model states,
SQLite embedding stores, and PKL/NPZ caches. These files are either too large
for Git, subject to third-party redistribution terms, or both. Exact source
identities and checksums are supplied so an archival data deposit can be linked
without silently changing the scientific protocol.

See `docs/dataset_source_audit.md` and `docs/public_release.md` for details.

## Manuscript status

The manuscript and Supporting Information are maintained in a separate private
Overleaf project. At submission, that project will cite this repository and its
versioned archival DOI; manuscript source and journal template assets are not
part of this Git release.

## License

Original software under `src/`, `tests/`, and the runnable job-script
directories is released under the MIT License; see `LICENSE`. This license does
not extend to third-party datasets, mapped reaction files, model repositories,
model weights, journal template assets, the manuscript, figures, or frozen
numerical artifacts. Those materials retain their original terms or are all
rights reserved unless a separate notice states otherwise.
