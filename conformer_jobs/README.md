# One-click conformer jobs

There are ten independent Windows launchers, one for every prespecified
conformer seed 42--51.  Double-click exactly one `run_seed_N.cmd` per borrowed
machine.  Do not run the same seed on two machines.

Run `RUN_SHARED_2D.cmd` exactly once on any machine.  This is the trained
four-input 2D comparator shared by all ten conformer runs; it does not generate
a conformer or call Uni-Mol.  Each seed folder already contains the prior-only
reference plus the trained seven-feature arm, so the aggregate needs this one
additional shared arm to form the correct 2D-versus-2D+Uni-Mol comparison.

Run `CHECK_MACHINE.cmd` first on every borrowed machine.  It verifies inputs,
checkpoint assets, Python discovery and free disk without generating a
conformer.

To make a portable input bundle from the prepared working tree, double-click
`MAKE_BORROWED_MACHINE_BUNDLE.cmd` (it invokes
`PACK_FOR_BORROWED_MACHINES.ps1`).  It copies the exact current `src/`, minimal
dataset/candidate inputs, launchers, environment specs and pinned Uni-Mol
weights, records SHA-256 for every bundled file, and creates a timestamped ZIP
without overwriting an older bundle.  On the borrowed machine, extract it and
run `SETUP_ENVIRONMENT.cmd`, then `CHECK_MACHINE.cmd`.

The setup script detects an NVIDIA GPU and installs the pinned CUDA 11.8
PyTorch build when possible.  Every seed launcher uses `--device auto`: Uni-Mol
inference and neural-ranker training use CUDA when PyTorch confirms that CUDA
is available, and otherwise use CPU.  RDKit conformer generation remains a CPU
task.  Batch size is selected conservatively from GPU VRAM (or for CPU), CPU
threads are selected from physical cores, and a CUDA out-of-memory error causes
only that inference batch to be halved and retried.  `manifest.json` records the
actual hardware, device, VRAM, batch size and thread count.

Run `CHECK_MACHINE.cmd` after setup and read `runtime_selection` in its JSON.
If a machine has an NVIDIA card but reports `resolved_device: cpu`, its driver
or CUDA-enabled PyTorch installation is not usable; the scientific job can
still run on CPU.  To force a diagnostic run, invoke the Python runner with
`--device cpu` or `--device cuda` (the latter fails closed when unavailable).

CPU and GPU floating-point execution can differ at very small rounding levels.
For the cleanest confirmatory comparison, put seeds 42--46 on one device class
(all CUDA or all CPU) when possible.  The exact backend is always recorded, so
mixed borrowed-machine runs remain auditable rather than silently pooled.

## What each launcher does

1. Verifies the frozen USPTO-50K source, reaction metadata, candidate pool,
   Uni-Mol checkpoint and dictionary by SHA-256.
2. Builds or resumes one seed-local SQLite atom-embedding cache.
3. Builds the compact official-split seven-feature cache.
4. Runs training seeds 42--46 under the labeled
   `legacy-cap10-fixed50-v1` sensitivity protocol.
5. Saves reaction-level predictions, Top-1/3/5/10, MRR, models, normalizers,
   environment provenance, manifests and checksums.
6. Validates expected train/test counts, all feature arrays, all five metric
   records and all five prediction CSVs.
7. Deletes only that seed's temporary SQLite cache after validation succeeds.
   If anything fails, the cache stays in place and another double-click resumes.

## Per-machine prerequisites

- Copy or clone the same repository revision to the machine.
- Copy `data/uspto_smiles.csv`, `data/uspto_reaction_metadata.csv`, and
  `outputs/rerank_dataset.jsonl` without changing their bytes.
- Create the pinned Python environment from `environment-revision.yml` and the
  checked environment records.  The launcher looks first for the Conda env
  `retrosynthesis-revision-benchmark-py310` under Anaconda or Miniconda.
- The official `mol_pre_no_h_220816.pt` and `mol.dict.txt` must be in the
  installed `unimol_tools/weights` directory with the pinned hashes.
- Have at least 20 GiB free on the output drive.  The retained seed folder is
  much smaller; the large cache exists only while the job is incomplete.

If Python is elsewhere, set `CONFORMER_PYTHON` to its full `python.exe` path
before launching.

## Bringing results back

Copy the complete seed folder from each machine into the same directory on the
main machine:

```text
outputs/jcheminform_revision/conformers/
  seed_42/
  seed_43/
  ...
  seed_51/
```

A valid folder contains `COMPLETED.json`, `manifest.json`,
`result_summary.json`, `checksums.sha256`, `features/`, and
`ranking_legacy_fixed50/`.  Never merge the contents of two seed folders.

After all ten folders and the shared 2D result are present, double-click
`COLLECT_RESULTS.cmd`.  It
first re-hashes every retained artifact and writes `conformer_run_index.csv/json`,
then runs the prespecified B1 stability analysis, B2 5-by-5 crossed analysis,
and B3 ten-conformer scalar average under
`outputs/jcheminform_revision/conformer_aggregate`.  It fails closed if even
one seed folder or artifact is missing or changed.  It never regenerates an
embedding cache.

The aggregate step retains only CSV/JSON reports and two compact seven-feature
PKLs for the B3 averaged-scalar arm.  It does not retain or reconstruct any
atom embedding.  Existing aggregate outputs are not overwritten automatically;
review or move them before intentionally starting a replacement analysis.

Seeds 42--46 are the prespecified B1/B2 conformer replicates.  Seeds 47--51
are additional B3 inputs for the ten-conformer average; their individual
ranking summaries are explicitly labeled exploratory.  The later aggregate
analysis consumes the compact feature files, so no conformer needs to be
generated again.
