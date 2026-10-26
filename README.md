# Candidate-pool dependence in retrosynthesis reranking

Code for comparing reaction-candidate rerankers on fixed candidate pools.
The models combine candidate priors, 2D features and optional
representation-derived features. Evaluation separates reference coverage,
ranking within covered pools and end-to-end accuracy.

The experiments use the 50,037-reaction Chemformer distribution of USPTO-50K:
40,029 training, 5,004 validation and 5,004 test reactions. See
[dataset provenance](docs/dataset_source_audit.md) for the source and checksums.

## Install

Python 3.10 is the validated version.

```bash
git clone https://github.com/thacanh/candidate-pool-retro-reranking.git
cd candidate-pool-retro-reranking
conda env create -f environment.yml
conda activate candidate-pool-retro-reranking
python -m pip install -e .
```

AiZynthFinder, LocalRetro, GROVER, Chemformer and RXN-EBM use separate
environments. Their source and model identities are recorded in
`data/provenance/` and the relevant experiment modules.

## Contents

| Path | Contents |
| --- | --- |
| `src/rerank/` | Features, ranker training, evaluation and analysis |
| `tests/` | Unit and protocol checks |
| `outputs/historical_anchor/` | Frozen historical numerical summaries |
| `outputs/transfer_analysis/` | Transfer analyses and final-round summaries |
| `notebooks/rebuild_data_figures.ipynb` | Editable plots using saved summaries |
| `data/provenance/` | Dataset, generator and model source records |
| [`data/common_test_3814/`](data/common_test_3814/) | 3,814 common test reactions, four candidate pools and reference ranks |
| `docs/analysis_plan.md` | Experiment definitions and decision rules |

The repository currently includes code and compact results. Full training
inputs, third-party weights, trained model checkpoints and embedding caches
are distributed separately. The included summary tables suffice to rebuild
the numerical plots, but do not replace the inputs needed to train a model.

The 3,814-reaction comparison cohort is the intersection of reference-covered
official-test reactions across four cap-10 pools: historical, AiZynthFinder,
LocalRetro and merged. Each reaction has a separate candidate list in each
pool. [The data folder](data/common_test_3814/README.md) contains these lists,
recorded reactants and reference ranks across 20 paired seeds. It is about
32 MiB uncompressed and needs no archive extraction. This is an evaluation
subset, not a training set; train/validation inputs are supplied separately.

## Rebuild plots

```bash
python -m rerank.figures.plot_digital_discovery_figures \
  --repo-root . --output-dir figures_rebuilt
```

Open `notebooks/rebuild_data_figures.ipynb` to change plot styles.
These commands use saved results; they do not retrain models.

## Check the downloaded files

```bash
python -m rerank.data.build_public_release --verify .
python -m pytest -q tests/test_public_release.py tests/test_common_test_data.py
```

The full research test suite also exercises workflows requiring the separately
supplied data and models.

## Run a new embedding comparison

Start from the same candidate lists and official splits. Fit preprocessing on
training data, choose settings on validation data and evaluate paired models
with the same seeds and metrics. Report candidate coverage separately from
ranking performance.

Training and evaluation modules are under `src/rerank/experiments/`.
Use their `--help` options for required inputs. To reuse the frozen experiments,
request the corresponding candidate pools and train/validation inputs;
cloning this repository alone is not sufficient.

## License

Original software is covered by the [MIT License](LICENSE). Third-party data,
repositories and model weights retain their own terms. The software license
does not grant a license to the frozen numerical artifacts or manuscript.
