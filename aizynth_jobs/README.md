# AiZynthFinder cap-10/cap-50 one-instance workflow

This bundle runs `A-CAP10-REPRO` and `A-CAP50` from one expansion-policy
stream. It is resumable by immutable chunks and fail-closed: the cap-50 output
cannot be used downstream unless the derived cap-10 pool matches the frozen
historical pool under `rerank.analysis.compare_candidate_pools`.

## Build the upload archive on Windows

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File aizynth_jobs\PACK_FOR_VAST.ps1
```

Upload the resulting `vast_bundles/aizynth_onepass_bundle_*.tar.gz` to
`/workspace/` on the rented instance. A practical minimum is 16 logical CPUs,
16 GiB RAM and 30 GiB disk. GPU is not used during candidate generation.

## Set up and benchmark on Vast.ai

```bash
cd /workspace
tar -xzf aizynth_onepass_bundle_YYYYMMDD_HHMMSS.tar.gz
cd aizynth_onepass_bundle_YYYYMMDD_HHMMSS

bash aizynth_jobs/linux/SETUP_AIZYNTH_ENV.sh
export AIZYNTH_PYTHON=/workspace/aizynth-revision-py310/bin/python
bash aizynth_jobs/linux/RUN_PILOT_500.sh
```

The pilot is explicitly non-scientific. Inspect
`outputs/jcheminform_revision/candidate_pools/aizynth_pilot_500/pilot_summary.json`
before starting the full run.

## Run both caps once

Use tmux, then:

```bash
cd /workspace/aizynth_onepass_bundle_YYYYMMDD_HHMMSS
export AIZYNTH_PYTHON=/workspace/aizynth-revision-py310/bin/python
bash aizynth_jobs/linux/RUN_AIZYNTH_ALL.sh
```

The default worker count is conservative for 16 GiB RAM. Override only after
the pilot, for example `export AIZYNTH_WORKERS=6`. Re-running the same command
validates and resumes completed chunks; it never overwrites a partial or
tampered chunk.

Status from a second terminal:

```bash
bash aizynth_jobs/linux/STATUS.sh
```

At completion the script prints a result archive path and SHA-256. Exit code 2
means the cap-10 reproduction gate failed. In that case download the archive
for discrepancy review and do not start cap-50 embeddings.
