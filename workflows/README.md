# Optional workflows

These workflows require more than the files in a fresh clone.

| Folder | Task | Additional inputs |
| --- | --- | --- |
| `aizynthfinder/` | Generate candidate pools | Official reaction tables, expansion-policy assets and historical pool |
| `conformers/` | Generate conformer features and fit paired rankers | Reaction tables, candidate pool and Uni-Mol weights |
| `chemformer/` | Score forward round trips | Prepared precursor inputs, official source checkouts and checkpoint |
| `transfer/` | Compare priors, encoders and selected settings | Frozen feature bundles, train/validation arrays, selected models and input contracts |

The included `data/common_test_3814/` folder is an inspection/evaluation
subset and cannot replace the full inputs in this table.

Transfer entry points:

```bash
python workflows/transfer/run_prior_selection.py --help
python workflows/transfer/run_morgan_comparison.py --help
python workflows/transfer/search_hyperparameters.py --help
python workflows/transfer/evaluate_selected_models.py --help
```

`package_*.py` assembles supplied inputs, `check_*_inputs.py` validates them,
and `import_*_results.py` checks returned artifacts. The setup and job-manager
shell scripts are for prepared Linux bundles; they do not download the data
or grant an execution contract automatically.

Old protocol labels such as N3/N4 remain in frozen manifests and job modes.
They identify the recorded comparisons, not separate public package versions.
