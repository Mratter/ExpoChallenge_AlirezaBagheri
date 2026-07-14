param([ValidateSet('cpu','gpu')][string]$Profile = 'cpu')
$ErrorActionPreference = 'Stop'
Write-Host 'Foundation only: dependency installation is not available until the architecture manifest passes Gate 1.'
exit 2
