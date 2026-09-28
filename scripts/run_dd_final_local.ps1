param(
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot
$env:PYTHONPATH = "src"

$logDirectory = Join-Path $RepositoryRoot "logs\dd_final_round"
$logPath = Join-Path $logDirectory "run.log"
$outputDirectory = Join-Path $RepositoryRoot "outputs\digital_discovery_final_round\n5_crossed_transfer"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

"[$([DateTime]::UtcNow.ToString('o'))] START dd-n5-crossed-transfer-inference-v1" |
    Set-Content -LiteralPath $logPath -Encoding utf8
"Repository: $RepositoryRoot" | Add-Content -LiteralPath $logPath -Encoding utf8
"PID: $PID" | Add-Content -LiteralPath $logPath -Encoding utf8

try {
    & python -m rerank.analysis.analyze_dd_submission_checks n5-crossed-transfer `
        --approval-record docs/dd_final_round_approval.json `
        --analysis-plan docs/analysis_plan.md `
        --m2-transfer-csv outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_reactions.csv `
        --anchor-prediction-root outputs/jcheminform_revision/tuned_primary/conformer_seed_42/g1_20seed/test_results/predictions `
        --k1-manifest outputs/digital_discovery_round_jk/k1/aizynthfinder_only/test_results/manifest.json `
        --k1-manifest outputs/digital_discovery_round_jk/k1/localretro_only/test_results/manifest.json `
        --k1-manifest outputs/digital_discovery_round_jk/k1/merged/test_results/manifest.json `
        --bootstrap-samples 10000 `
        --rng-seed 2070 `
        --batch-size 100 `
        --output-dir $outputDirectory 2>&1 |
        Tee-Object -FilePath $logPath -Append | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "N5 runner exited with code $LASTEXITCODE"
    }
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE exit=0" |
        Add-Content -LiteralPath $logPath -Encoding utf8
}
catch {
    "[$([DateTime]::UtcNow.ToString('o'))] FAILED $($_.Exception.Message)" |
        Add-Content -LiteralPath $logPath -Encoding utf8
    exit 1
}
