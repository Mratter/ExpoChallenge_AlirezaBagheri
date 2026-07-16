param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'project_environment.ps1')
& (Join-Path $PSScriptRoot 'preflight.ps1') -Profile $Profile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$ProjectEnvironment = $null
$LocationPushed = $false
$RuntimeExitCode = 1
try {
    $ProjectEnvironment = Enter-Ai17ProjectEnvironment -Root $Root
    Push-Location $Root
    $LocationPushed = $true
    $env:INNOVERSE_PROFILE = $Profile
    $env:INNOVERSE_RUNTIME = '1'
    $env:INNOVERSE_COMMIT = (& git rev-parse --short=12 HEAD).Trim()
    Write-Host 'Civic Relay is running at http://127.0.0.1:4117'
    & $ProjectEnvironment.PythonPath -m uvicorn backend.app.main:app --host 127.0.0.1 --port 4117 --no-access-log
    $RuntimeExitCode = $LASTEXITCODE
}
finally {
    if ($LocationPushed) { Pop-Location }
    if ($null -ne $ProjectEnvironment) { Exit-Ai17ProjectEnvironment -Context $ProjectEnvironment }
}
exit $RuntimeExitCode
