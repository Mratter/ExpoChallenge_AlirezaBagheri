param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot

if ($Profile -ne 'cpu') { throw 'The Gate 2 GPU profile is not implemented; use -Profile cpu.' }
foreach ($Command in @('uv', 'node', 'npm')) {
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required setup command is missing: $Command"
    }
}

Push-Location $Root
try {
    Write-Host '[setup] Syncing pinned Python 3.12 environment'
    & uv sync --frozen --python 3.12
    if ($LASTEXITCODE -ne 0) { throw 'uv sync failed' }
    Write-Host '[setup] Installing pinned frontend dependencies'
    & npm ci --prefix frontend
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed' }
    Write-Host '[setup] Building the production frontend'
    & npm run build --prefix frontend
    if ($LASTEXITCODE -ne 0) { throw 'frontend build failed' }
    Write-Host '[setup] Complete'
}
finally { Pop-Location }
