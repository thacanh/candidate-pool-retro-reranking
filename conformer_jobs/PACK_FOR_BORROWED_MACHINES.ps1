[CmdletBinding()]
param(
    [string]$DestinationRoot = "machine_bundles"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$destinationBase = Join-Path $repoRoot $DestinationRoot
$bundleRoot = Join-Path $destinationBase "conformer_bundle_$timestamp"
$zipPath = "$bundleRoot.zip"

if ((Test-Path -LiteralPath $bundleRoot) -or (Test-Path -LiteralPath $zipPath)) {
    throw "Refusing to overwrite an existing bundle target: $bundleRoot"
}

New-Item -ItemType Directory -Path $bundleRoot -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundleRoot "data") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundleRoot "outputs") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $bundleRoot "assets\unimol_weights") -Force | Out-Null

$requiredFiles = @(
    "docs\analysis_plan.md",
    "AGENTS.md",
    "environment-revision.yml",
    "requirements-revision.txt",
    "constraints-revision-py310.txt",
    "data\uspto_smiles.csv",
    "data\uspto_reaction_metadata.csv",
    "outputs\rerank_dataset.jsonl"
)

foreach ($relative in $requiredFiles) {
    $source = Join-Path $repoRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing required bundle input: $source"
    }
    $target = Join-Path $bundleRoot $relative
    $targetParent = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $targetParent -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $target
}

Copy-Item -LiteralPath (Join-Path $repoRoot "src") -Destination (Join-Path $bundleRoot "src") -Recurse
Copy-Item -LiteralPath (Join-Path $repoRoot "conformer_jobs") -Destination (Join-Path $bundleRoot "conformer_jobs") -Recurse

$weightCandidates = @()
if ($env:UNIMOL_WEIGHT_DIR) {
    $weightCandidates += $env:UNIMOL_WEIGHT_DIR
}
$weightCandidates += (Join-Path $env:USERPROFILE "anaconda3\envs\retrosynthesis-revision-benchmark-py310\Lib\site-packages\unimol_tools\weights")
$weightCandidates += (Join-Path $env:USERPROFILE "miniconda3\envs\retrosynthesis-revision-benchmark-py310\Lib\site-packages\unimol_tools\weights")
$weightDir = $weightCandidates | Where-Object {
    Test-Path -LiteralPath (Join-Path $_ "mol_pre_no_h_220816.pt") -PathType Leaf
} | Select-Object -First 1
if (-not $weightDir) {
    throw "Could not locate the pinned Uni-Mol weight directory."
}
Copy-Item -LiteralPath (Join-Path $weightDir "mol_pre_no_h_220816.pt") -Destination (Join-Path $bundleRoot "assets\unimol_weights\mol_pre_no_h_220816.pt")
Copy-Item -LiteralPath (Join-Path $weightDir "mol.dict.txt") -Destination (Join-Path $bundleRoot "assets\unimol_weights\mol.dict.txt")

$manifest = [ordered]@{
    created_at = (Get-Date).ToUniversalTime().ToString("o")
    source_repository = $repoRoot
    bundle_root = $bundleRoot
    files = @()
}
Get-ChildItem -LiteralPath $bundleRoot -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($bundleRoot.Length + 1).Replace("\", "/")
    $manifest.files += [ordered]@{
        path = $relative
        size_bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $bundleRoot "BUNDLE_MANIFEST.json") -Encoding UTF8

Compress-Archive -LiteralPath $bundleRoot -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host "Created borrowed-machine bundle:"
Write-Host "  $zipPath"
Write-Host "Copy the ZIP to a machine, extract it, run conformer_jobs\SETUP_ENVIRONMENT.cmd,"
Write-Host "then conformer_jobs\CHECK_MACHINE.cmd and exactly one run_seed_N.cmd."
