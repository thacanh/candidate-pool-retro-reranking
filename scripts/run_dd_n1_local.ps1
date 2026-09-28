param(
    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepositoryRoot
$env:PYTHONPATH = "src"

$logDirectory = Join-Path $RepositoryRoot "logs\dd_final_round"
$logPath = Join-Path $logDirectory "run.log"
$outputDirectory = Join-Path $RepositoryRoot "outputs\digital_discovery_final_round\n1_matched_list_joint_sensitivity_v2"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null

"[$([DateTime]::UtcNow.ToString('o'))] START dd-n1-matched-list-joint-sensitivity-v2" |
    Add-Content -LiteralPath $logPath -Encoding utf8
"Repository: $RepositoryRoot" | Add-Content -LiteralPath $logPath -Encoding utf8
"PID: $PID" | Add-Content -LiteralPath $logPath -Encoding utf8

try {
    # Windows PowerShell represents native stderr records as PowerShell errors.
    # Preserve them in the log without allowing a harmless warning to trip the
    # wrapper; the native process exit code remains the authoritative gate.
    $ErrorActionPreference = "Continue"
    & python -m rerank.analysis.analyze_dd_submission_checks n1-matched-list `
        --approval-record docs/dd_final_round_approval.json `
        --analysis-plan docs/analysis_plan.md `
        --m2-descriptor-csv outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_reaction_pool_descriptors.csv `
        --anchor-prediction-root outputs/jcheminform_revision/tuned_primary/conformer_seed_42/g1_20seed/test_results/predictions `
        --k1-aiz-manifest outputs/digital_discovery_round_jk/k1/aizynthfinder_only/test_results/manifest.json `
        --historical-feature-cache outputs/jcheminform_revision/conformers/seed_42/features/official_3d_prior_schema1.pkl `
        --aizynth-pool-jsonl outputs/jcheminform_revision/candidate_pools/ws_e_three_pools_filtered_v2/aizynthfinder_only.jsonl `
        --reference-effect-csv outputs/digital_discovery_final_round/n1_matched_list_joint_sensitivity/n1_within_pipeline_effects.csv `
        --reference-contrast-csv outputs/digital_discovery_final_round/n1_matched_list_joint_sensitivity/n1_joint_pipeline_contrasts.csv `
        --bootstrap-samples 10000 `
        --sign-flip-samples 100000 `
        --rng-seed 2080 `
        --output-dir $outputDirectory 2>&1 |
        Tee-Object -FilePath $logPath -Append | Out-Null
    $pythonExitCode = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    if ($pythonExitCode -ne 0) {
        throw "N1 runner exited with code $pythonExitCode"
    }
    "[$([DateTime]::UtcNow.ToString('o'))] COMPLETE exit=0" |
        Add-Content -LiteralPath $logPath -Encoding utf8
}
catch {
    "[$([DateTime]::UtcNow.ToString('o'))] FAILED $($_.Exception.Message)" |
        Add-Content -LiteralPath $logPath -Encoding utf8
    exit 1
}
