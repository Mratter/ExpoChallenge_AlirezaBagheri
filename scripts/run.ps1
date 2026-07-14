param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot 'preflight.ps1') -Profile $Profile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location $Root
try {
    $env:INNOVERSE_PROFILE = $Profile
    $env:INNOVERSE_RUNTIME = '1'
    $env:INNOVERSE_COMMIT = (& git rev-parse --short=12 HEAD).Trim()
    Write-Host 'Civic Relay is running at http://127.0.0.1:4117'
    $Python = Join-Path $Root '.venv\Scripts\python.exe'
    & $Python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 4117 --no-access-log
    exit $LASTEXITCODE
}
finally { Pop-Location }
