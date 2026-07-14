param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu', [switch]$Full)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot

if ($Profile -ne 'cpu') { throw 'The Gate 2 GPU profile is not implemented; use -Profile cpu.' }
foreach ($Path in @('uv.lock', 'frontend/package-lock.json', 'frontend/dist/index.html', 'artifacts/manifest.lock.json', 'artifacts/frozen_policy.v1.json')) {
    if (-not (Test-Path -LiteralPath (Join-Path $Root $Path))) { throw "Required dependency is missing: $Path" }
}
$Occupied = Get-NetTCPConnection -State Listen -LocalPort 4117 -ErrorAction SilentlyContinue
if ($Occupied) { throw 'Fixed port 4117 is already occupied.' }

Push-Location $Root
try {
    & uv run --frozen python scripts/preflight_check.py
    if ($LASTEXITCODE -ne 0) { throw 'Artifact or smoke inference preflight failed' }
    if ($Full) {
        & uv run --frozen pytest
        if ($LASTEXITCODE -ne 0) { throw 'backend tests failed' }
        & uv run --frozen ruff check backend scripts
        if ($LASTEXITCODE -ne 0) { throw 'backend lint failed' }
        & npm test --prefix frontend
        if ($LASTEXITCODE -ne 0) { throw 'frontend tests failed' }
        & npm run build --prefix frontend
        if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' }
    }
    Write-Host '[preflight] PASS'
}
finally { Pop-Location }
