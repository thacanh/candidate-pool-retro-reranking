# Candidate-pool dependence in retrosynthesis reranking

Code and data for comparing reaction-candidate rerankers across candidate pools.
The experiments separate reference coverage, ranking within covered pools,
and end-to-end accuracy.

## Install

Use Python 3.10. The core environment is pinned in `environment.yml`.

```bash
git clone https://github.com/thacanh/candidate-pool-retro-reranking.git
cd candidate-pool-retro-reranking
conda env create -f environment.yml
conda activate candidate-pool-retro-reranking
python -m pip install -e .
```

## Inspect the candidate pools

[`data/common_test_3814/`](data/common_test_3814/) contains 3,814 official-test
reactions covered by all four cap-10 pools: historical, AiZynthFinder,
LocalRetro and merged. It includes ordered candidates and reference ranks
for 20 paired seeds.

```bash
python scripts/inspect_candidates.py
python scripts/inspect_candidates.py --reaction-id 45033 --seed 42
```

The files are unpacked and readable with standard Python. This is a
coverage-selected evaluation subset; it must not be used to train a ranker.

## Rebuild the figures

```bash
python -m rerank.figures.plot_figures --repo-root . --output-dir figures_rebuilt
```

For style changes, open `notebooks/rebuild_data_figures.ipynb`.
These plots use the included summaries and do not rerun models.

## Repository layout

| Folder | Contents |
| --- | --- |
| `src/rerank/` | Features, models, training, evaluation and analysis |
| `data/` | Shared test cohort and source provenance |
| `outputs/` | Frozen numerical summaries |
| `scripts/` | Candidate inspection and reporting tools |
| `workflows/` | Optional candidate-generation and training workflows |
| `tests/` | Unit, protocol and data checks |
| `docs/` | Reproduction guide, dataset audit and original analysis plan |

## Run a new comparison

Use the same candidate lists, official splits and paired seeds. Fit
preprocessing on training data and choose settings on validation data.
Full train/validation inputs, embedding caches and model weights are supplied
separately; cloning this repository is not enough to rerun every experiment.

See [reproduction instructions](docs/reproducibility.md) for entry points,
required inputs and optional file-integrity checks.

## License

Original software is covered by the [MIT License](LICENSE). Third-party data,
repositories and model weights retain their own terms. The software license
does not grant a license to the frozen numerical artifacts or manuscript.
