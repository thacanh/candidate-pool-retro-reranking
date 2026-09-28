# Public release guide

This document defines the clean Git release for the active *Digital Discovery*
paper. It does not alter a scientific protocol or recompute a scientific
result.

## Current local organization and release hold (2026-09-24)

### Code update authorized 2026-09-28

The user authorized preparation and incremental upload of code and compact
results. GitHub Release/tag creation and archival DOI deposition remain on
hold pending author metadata and the user's final instruction. No manuscript,
private PowerPoint/PDF figure, author placeholder or DOI field is part of this
update. Preserve the existing code-only Git history; do not force-push again.
The new staging payload is `release/public_repository_staging_20260928/`.

Returned ZIPs and their original receipts are stored together under
`archive/imports/dd_final_round/{N3,N3_S4,N4}/`; the original manuscript ZIP is
under `archive/imports/manuscripts/`. The organization ledger records before/
after SHA256 checks at `archive/imports/organization_20260924.json`.
Local import/audit helpers resolve these paths and still support newly
downloaded root files, but reject ambiguous duplicate copies.

No public release or upload was performed during organization. Existing
`release/public_repository*` directories and TAR exports are historical,
potentially stale snapshots (some contain paper material), not the latest
approved code-only payload. Preserve their Git histories. Build a new empty
staging directory with the allowlist below for the authorized code update;
verify its current N2/N3/N4/S4 coverage and exclude all paper
material before publishing. DOI and private manuscript metadata remain
untouched. The current private Overleaf ZIP is indexed in `release/README.md`.

## Build

From the research worktree, with `PYTHONPATH=src` when the package is not
installed:

```powershell
$env:PYTHONPATH = "src"
python -m rerank.data.build_public_release `
  --output release\public_repository
```

The builder is allowlist-based and refuses to overwrite an existing non-empty
directory. It copies source files, maps frozen numerical
artifacts to neutral paths, derives one compact plotting table from four frozen
test manifests, scans public text for machine-local paths and credential-like
material, and writes both `release_manifest.json` and `checksums.sha256`.

## Included

- installable Python source under `src/rerank/`;
- tests and the three retained environment/job descriptions;
- the signed analysis plan and provenance records;
- model-free CSV/JSON numerical summaries used by the paper; and
- file sizes and SHA256 values for every release payload.

Final-round summaries include N1, N2 and its prior diagnostics, N3-S1--S4,
N4, N5, corrected headroom and the exploratory prior-gap supplement. Selected
CSV files are copied byte-for-byte. Public copies of final-round JSON files
remove machine-local user-path prefixes and are reserialized; numerical
values, protocol IDs and scientific settings are not changed. Both original
and public SHA256 values are recorded in
`data/provenance/final_round_artifact_map.json`. Do not treat those redacted
JSON files as the original byte-identical manifests. Original manifest
references to excluded heavy inputs remain provenance, not download links.

The compact `figure_inputs/adaptation.csv` is assembled from existing N5,
N3-S3 and N3-S4 statistics, with no new fit or resampling. Its source artifacts
are included under `outputs/transfer_analysis/final_round/`.

## Excluded

- `.git/` and the dirty research history;
- manuscript and Supporting Information sources, rendered PDFs, paper figures,
  bibliography, and RSC template assets, which remain in the private Overleaf
  workflow until submission;
- raw USPTO reaction tables and third-party mapped copies;
- third-party repositories and model checkpoints;
- SQLite, PKL, PT/PTH, and candidate-level cache files;
- validation checkpoints and full prediction archives;
- Vast.ai bundles, imported TAR/ZIP files, logs, scratch directories,
  `__pycache__`, and LaTeX build intermediates; and
- abandoned manuscript drafts and internal cleanup records.

The exclusions are deliberate. Git is the code and compact numerical release;
large redistributable artifacts should be placed in a separately versioned
archival deposit and linked by DOI.

## Compact release QA

Run inside the generated directory:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests\test_public_release.py
python -m rerank.figures.plot_digital_discovery_figures `
  --repo-root . `
  --output-dir figures_rebuilt
```

Verify the transport ledger independently:

```powershell
python -m rerank.data.build_public_release --verify .
```

## Clean Git history and upload

Do not push the current 30-GiB research worktree. Its Git object database also
contains old heavyweight blobs. The commands below apply to an initial
publication only. For the existing public repository, synchronize a verified
payload into its clean checkout and make a normal fast-forward commit instead:

```powershell
cd release\public_repository
git init -b main
git add .
git commit -m "Public research release"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

The public code is MIT-licensed; the scope statement in `README.md` excludes
third-party material and frozen numerical artifacts. The manuscript stores the
repository URL and archival DOI in its private `release_metadata.tex` file.
Creating or publishing a remote repository remains a separate external action
requiring explicit authorization.
