# Reproducing the results

Run commands from the repository root with Python 3.10 and the package
installed. The README gives the installation commands.

## Included data and results

- `data/common_test_3814/` contains the covered test cohort, four candidate
  lists per reaction and reference ranks across seeds 42--61.
- `outputs/historical_anchor/` contains historical numerical summaries.
- `outputs/transfer_analysis/` contains transfer analyses and figure inputs.
- `data/provenance/` records source identities and original artifact hashes.

The processed benchmark is the 50,037-reaction Chemformer distribution of
USPTO-50K: 40,029 training, 5,004 validation and 5,004 test reactions.
See [the source audit](dataset_source_audit.md) for its lineage.

The common cohort has 3,814 reactions, not 3,814 individual candidates.
It was selected by reference coverage and is an evaluation set. Do not train
or select settings on it. Recorded-reference agreement does not establish
the chemical feasibility of an alternative reaction.

## Inspect a reaction

```bash
python scripts/inspect_candidates.py --reaction-id 45033 --seed 42
```

This prints the product, recorded reactants, all four candidate lists and
the stored reference ranks. It needs only the Python standard library.

## Rebuild plots

```bash
python -m rerank.figures.plot_figures --repo-root . --output-dir figures_rebuilt
```

The notebook `notebooks/rebuild_data_figures.ipynb` uses the same plotting
module and includes the data checks before plotting. It needs Jupyter:

```bash
python -m pip install jupyter
jupyter notebook notebooks/rebuild_data_figures.ipynb
```

## Training and inference

The following commands show the required arguments without running a model:

```bash
python -m rerank.experiments.run_tuned --help
python -m rerank.experiments.run_cross_inference --help
python -m rerank.experiments.run_prior_sensitivity --help
python -m rerank.experiments.run_morgan_comparison --help
python -m rerank.experiments.run_pool_ranking --help
```

Reusing a frozen experiment requires its full candidate pools, official
train/validation inputs, feature caches, selected settings and any model
weights used by that workflow. These heavy files are not included in Git.
The compact summaries cannot substitute for training examples or checkpoints.

AiZynthFinder, Chemformer and conformer workflows live under `workflows/`.
Each requires additional inputs; read its README before launching it.
The transfer workflows use prepared frozen-feature bundles and their input
contracts. Historical protocol IDs and source paths are retained inside
manifests so the original results remain traceable.

## Tests

```bash
python -m pytest -q tests/test_public_release.py tests/test_common_test_data.py
```

Other tests cover synthetic training fixtures and optional workflows.
Tests that require separately supplied scientific assets are skipped when
those assets are absent.

## Optional integrity check

```bash
python -m rerank.data.verify_release --repo-root .
```

`checksums.sha256` is the single transport ledger for the shared files.
It detects missing or changed files; it is not needed to inspect candidates.
Model and result manifests also retain their own input hashes because these
identify the exact inputs used in a scientific run.

If you intentionally change code, the original checksum check will report
those changes. That is expected. Record new experiments in a separate output
directory with their own input and configuration manifest.
