# Public release guide

This document defines the clean Git release for the active *Digital Discovery*
paper. It does not alter a scientific protocol or recompute a scientific
result.

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
contains old heavyweight blobs. Initialize a new history inside the generated
release only:

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
