param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'preflight.ps1') -Profile $Profile -Full
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Error 'Verification blocked: project-specific tests have not been implemented.'
exit 3
