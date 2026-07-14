param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'preflight.ps1') -Profile $Profile
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Error 'Runtime contract violation: Gate marker exists but no project runner has replaced the foundation script.'
exit 3
