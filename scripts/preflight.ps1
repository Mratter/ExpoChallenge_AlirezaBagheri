param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu', [switch]$Full)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')

if ($Profile -ne 'cpu') { throw 'The frozen Feature Complete policy supports the CPU profile only.' }
foreach ($Path in @(
    'uv.lock',
    'frontend/package-lock.json',
    'frontend/dist/index.html',
    'artifacts/manifest.lock.json',
    'artifacts/frozen_policy.v1.json',
    'artifacts/city_recovery_ppo.v1.zip',
    'artifacts/city_recovery_ppo.v1.onnx',
    'artifacts/city_recovery_ppo.v1.metadata.json',
    'evaluation/policy_parity.v1.json',
    'evaluation/protocol.v1.json',
    'evaluation/feature_complete_report.v1.json'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $Path))) { throw "Required dependency is missing: $Path" }
}
$Occupied = Get-NetTCPConnection -State Listen -LocalPort 4117 -ErrorAction SilentlyContinue
if ($Occupied) { throw 'Fixed port 4117 is already occupied.' }

$ProjectEnvironment = $null
$LocationPushed = $false
try {
    $ProjectEnvironment = Enter-Ai17ProjectEnvironment -Root $Root
    Push-Location $Root
    $LocationPushed = $true
    & uv run --frozen python scripts/preflight_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Artifact or smoke inference preflight failed' }
    if ($Full) {
        & uv run --frozen python -m pytest
        if ($LASTEXITCODE -ne 0) { throw 'backend tests failed' }
        & uv run --frozen python -m ruff check backend scripts
        if ($LASTEXITCODE -ne 0) { throw 'backend lint failed' }
        & npm test --prefix frontend
        if ($LASTEXITCODE -ne 0) { throw 'frontend tests failed' }
        & npm run build --prefix frontend
        if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' }
    }
    Write-Host '[preflight] PASS'
}
finally {
    if ($LocationPushed) { Pop-Location }
    if ($null -ne $ProjectEnvironment) { Exit-Ai17ProjectEnvironment -Context $ProjectEnvironment }
}
