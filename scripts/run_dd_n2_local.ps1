param(
    [string]$ApprovalRecord = "docs/dd_remote_approval.json",
    [string]$ArchiveDir = "release/dd_remote_results"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$python = if ($env:DD_PYTHON) {
    $env:DD_PYTHON
} else {
    Join-Path $env:USERPROFILE "anaconda3\envs\candidate-pool-retro-reranking\python.exe"
}
$protocol = "dd-n2-cross-inference-v1"
$output = "outputs/dd_final_round/N2"
$runLog = "logs/dd_final_round/N2_run.log"
$archiveLog = "logs/dd_final_round/N2_archive.log"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Pinned Python not found: $python"
}
if (-not (Test-Path -LiteralPath $ApprovalRecord -PathType Leaf)) {
    throw "N2 is blocked: an authorization record is missing: $ApprovalRecord"
}
if ((Test-Path -LiteralPath $output) -or (Test-Path -LiteralPath $runLog) -or
    (Test-Path -LiteralPath $archiveLog) -or
    (Test-Path -LiteralPath (Join-Path $ArchiveDir "N2.zip")) -or
    (Test-Path -LiteralPath (Join-Path $ArchiveDir "N2.receipt.json"))) {
    throw "N2 output, log, or archive already exists; inspect it before any rerun."
}

$env:PYTHONPATH = (Join-Path (Get-Location).Path "src")
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

# Read-only authorization gate, before creating a log or touching scientific data.
& $python -c "import sys; from rerank.experiments.dd_remote_gate import require_approval; require_approval(*sys.argv[1:]); print('N2 approval verified')" `
    $ApprovalRecord "docs/analysis_plan.md" "src/rerank/experiments/run_dd_n2_cross_inference.py" $protocol
if ($LASTEXITCODE -ne 0) { throw "N2 approval verification failed." }

New-Item -ItemType Directory -Path (Split-Path $runLog -Parent) -Force | Out-Null
$root = "outputs/jcheminform_revision/tuned_primary/conformer_seed_42"
$g1 = "$root/g1_20seed"
$k1 = "outputs/digital_discovery_round_jk/k1/aizynthfinder_only"
$feature = "outputs/jcheminform_revision/ws_e_streaming/pool_features_canonical_identity/aizynthfinder_only/manifest.json"
$pool = "outputs/jcheminform_revision/candidate_pools/ws_e_three_pools_filtered_v2"
$common = "outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_reactions.csv"

$runArgs = @(
    "-m", "rerank.experiments.run_dd_n2_cross_inference",
    "--approval-record", $ApprovalRecord, "--analysis-plan", "docs/analysis_plan.md",
    "--historical-freeze", "$g1/freeze.json", "--historical-model-root", $g1,
    "--historical-cache", "outputs/jcheminform_revision/conformers/seed_42/features/official_3d_prior_schema1.pkl",
    "--historical-prediction-root", "$g1/test_results/predictions",
    "--historical-test-manifest", "$g1/test_results/manifest.json",
    "--aiz-freeze", "$k1/models/freeze.json", "--aiz-model-root", "$k1/models",
    "--aiz-feature-freeze", $feature,
    "--aiz-prediction-jsonl", "$k1/test_results/reaction_ranks.jsonl",
    "--aiz-test-manifest", "$k1/test_results/manifest.json",
    "--aiz-selection-manifest", "$k1/selection/manifest.json",
    "--common-reactions-csv", $common,
    "--source-csv", "data/uspto_smiles.csv",
    "--metadata-csv", "data/uspto_reaction_metadata.csv",
    "--products-jsonl", "$pool/products.jsonl",
    "--device", "cpu", "--output-dir", $output
)
& $python @runArgs *> $runLog
if ($LASTEXITCODE -ne 0) { throw "N2 failed; inspect $runLog. Partial output was preserved." }

$archiveArgs = @(
    "scripts/dd_archive_stage.py", "archive", "--stage", "N2",
    "--protocol-id", $protocol, "--result-dir", $output,
    "--archive-dir", $ArchiveDir, "--approval-record", $ApprovalRecord,
    "--required", "cell_ranks.npz", "--required", "cross_inference_effects.csv",
    "--required", "clipping.csv", "--log-file", $runLog
)
& $python @archiveArgs *> $archiveLog
if ($LASTEXITCODE -ne 0) { throw "N2 archive failed; inspect $archiveLog. Results were preserved." }
& $python scripts/dd_archive_stage.py verify --receipt (Join-Path $ArchiveDir "N2.receipt.json") *>> $archiveLog
if ($LASTEXITCODE -ne 0) { throw "N2 archive verification failed; inspect $archiveLog." }
Write-Output "N2 complete and verified: $(Join-Path $ArchiveDir 'N2.zip')"
