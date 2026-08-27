param(
    [string]$Destination = "vast_bundles",
    [switch]$KeepStaging
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$destinationRoot = Join-Path $repoRoot $Destination
$bundleName = "aizynth_onepass_bundle_$stamp"
$staging = Join-Path $destinationRoot $bundleName
$archive = Join-Path $destinationRoot "$bundleName.tar.gz"

New-Item -ItemType Directory -Force -Path $staging | Out-Null

$rootFiles = @("AGENTS.md", "pytest.ini")
foreach ($relative in $rootFiles) {
    $source = Join-Path $repoRoot $relative
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $staging $relative)
    }
}

$directories = @("aizynth_jobs", "modelchem")
foreach ($relative in $directories) {
    Copy-Item -LiteralPath (Join-Path $repoRoot $relative) -Destination (Join-Path $staging $relative) -Recurse
}

$packageSource = Join-Path $repoRoot "src\rerank"
$packageTarget = Join-Path $staging "src\rerank"
New-Item -ItemType Directory -Force -Path (Split-Path $packageTarget -Parent) | Out-Null
Copy-Item -LiteralPath $packageSource -Destination $packageTarget -Recurse

$planSource = Join-Path $repoRoot "docs\analysis_plan.md"
$planTarget = Join-Path $staging "docs\analysis_plan.md"
New-Item -ItemType Directory -Force -Path (Split-Path $planTarget -Parent) | Out-Null
Copy-Item -LiteralPath $planSource -Destination $planTarget

$files = @(
    "tests\test_aizynth_candidate_generation.py",
    "tests\test_candidate_pool_comparator.py",
    "data\uspto_smiles.csv",
    "data\uspto_reaction_metadata.csv",
    "data\candidate_generator_provenance.json",
    "outputs\rerank_dataset.jsonl"
)
foreach ($relative in $files) {
    $source = Join-Path $repoRoot $relative
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required bundle file missing: $relative"
    }
    $target = Join-Path $staging $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target
}

$runtimeLock = Join-Path $staging "aizynth_jobs\runtime_lock"
if (Test-Path -LiteralPath $runtimeLock) {
    Remove-Item -LiteralPath $runtimeLock -Recurse -Force
}
Get-ChildItem -LiteralPath $staging -Directory -Recurse -Filter "__pycache__" | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Recurse -Force
}

$manifestFiles = Get-ChildItem -LiteralPath $staging -File -Recurse | Sort-Object FullName | ForEach-Object {
    [ordered]@{
        path = $_.FullName.Substring($staging.Length + 1).Replace("\", "/")
        size_bytes = $_.Length
        sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$manifest = [ordered]@{
    schema_version = 1
    bundle_name = $bundleName
    purpose = "A-CAP10-REPRO and A-CAP50 one-pass generation on one Vast.ai instance"
    created_at_utc = [DateTime]::UtcNow.ToString("o")
    source_git_commit = (git -C $repoRoot rev-parse HEAD 2>$null)
    files = @($manifestFiles)
}
$manifestPath = Join-Path $staging "BUNDLE_MANIFEST.json"
$manifestJson = $manifest | ConvertTo-Json -Depth 6
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($manifestPath, $manifestJson + [Environment]::NewLine, $utf8NoBom)

New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
if (Test-Path -LiteralPath $archive) {
    throw "Refusing to overwrite existing archive: $archive"
}
Push-Location $destinationRoot
try {
    & tar.exe -czf $archive $bundleName
    if ($LASTEXITCODE -ne 0) { throw "tar.exe failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}

$archiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "BUNDLE: $archive"
Write-Host "SHA256: $archiveHash"
Write-Host "SIZE: $((Get-Item -LiteralPath $archive).Length) bytes"

if (-not $KeepStaging) {
    $resolvedStaging = (Resolve-Path -LiteralPath $staging).Path
    $resolvedDestination = (Resolve-Path -LiteralPath $destinationRoot).Path
    if (-not $resolvedStaging.StartsWith($resolvedDestination + [IO.Path]::DirectorySeparatorChar)) {
        throw "Refusing to remove staging outside destination root: $resolvedStaging"
    }
    Remove-Item -LiteralPath $resolvedStaging -Recurse -Force
}
